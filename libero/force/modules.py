# modules.py
from __future__ import annotations
import numpy as np
import mujoco


class MujocoSensorReader:
    """
    Read MuJoCo sensors directly from sim.model / sim.data.
    Works even if robosuite get_sensor_measurement() is not exposed.
    """
    def __init__(self, sim, sensor_names: list[str]):
        self.sim = sim
        self.sensor_names = sensor_names
        self._name2slice = self._build_name2slice()

    def _build_name2slice(self) -> dict[str, slice]:
        m = self.sim.model
        name2slice = {}

        # Preferred: mujoco>=2.3 style m.sensor(i).name
        for i in range(m.nsensor):
            try:
                name = m.sensor(i).name
            except Exception:
                # If name API not available, skip; user can supply mapping manually.
                name = None

            dim = int(m.sensor_dim[i])
            adr = int(m.sensor_adr[i])
            if name is not None and name in self.sensor_names:
                name2slice[name] = slice(adr, adr + dim)

        missing = [n for n in self.sensor_names if n not in name2slice]
        if missing:
            # Give a clear error with debugging hint
            available = []
            for i in range(m.nsensor):
                try:
                    available.append(m.sensor(i).name)
                except Exception:
                    available.append(f"sensor_{i}")
            raise KeyError(
                f"Missing sensors: {missing}. Available sensors: {available}"
            )

        return name2slice

    def read(self, name: str) -> np.ndarray:
        sl = self._name2slice[name]
        return np.asarray(self.sim.data.sensordata[sl], dtype=np.float32).copy()

    def read_wrench(self, force_name: str, torque_name: str) -> np.ndarray:
        f = self.read(force_name).reshape(3)
        t = self.read(torque_name).reshape(3)
        return np.concatenate([f, t], axis=0)



class WrenchObsWrapper:
    """
    Wrap OffScreenRenderEnv (or similar) and inject wrench into obs dict.
    """
    def __init__(
        self,
        env,
        force_sensor="gripper0_force_ee",
        torque_sensor="gripper0_torque_ee",
    ):
        self.env = env
        self.force_sensor = force_sensor
        self.torque_sensor = torque_sensor
        self.reader = None

    def seed(self, seed):
        return self.env.seed(seed)

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        # Build reader AFTER reset (sim/model ready)
        sim = self.env.env.sim
        self.reader = MujocoSensorReader(sim, [self.force_sensor, self.torque_sensor])
        return self._inject(obs)

    def set_init_state(self, *args, **kwargs):
        return self.env.set_init_state(*args, **kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._inject(obs)
        return obs, reward, done, info

    def close(self):
        return self.env.close()

    def _inject(self, obs):
        assert self.reader is not None, "Call reset() before step()"
        f = self.reader.read(self.force_sensor).reshape(3)
        t = self.reader.read(self.torque_sensor).reshape(3)
        w = np.concatenate([f, t], axis=0)

        if isinstance(obs, dict):
            obs["force_ee"] = f
            obs["torque_ee"] = t
            obs["wrench_ee"] = w
            return obs
        return {"obs": obs, "force_ee": f, "torque_ee": t, "wrench_ee": w}

    def __getattr__(self, name):
        return getattr(self.env, name)

    def build_reader_if_needed(self):
        if self.reader is None:
            sim = self.env.env.sim
            self.reader = MujocoSensorReader(sim, [self.force_sensor, self.torque_sensor])

    def read_wrench_from_sim(self):
        """
        Read wrench from CURRENT sim state (assumes mj_forward already called if you changed state).
        """
        self.build_reader_if_needed()
        f = self.reader.read(self.force_sensor).reshape(3)
        t = self.reader.read(self.torque_sensor).reshape(3)
        return np.concatenate([f, t], axis=0)

    def set_flattened_state_and_forward(self, flat_state):
        """
        Helper for state-replay.
        """
        sim = self.env.env.sim
        sim.set_state_from_flattened(flat_state)
        # ``mj_step1`` followed by ``mj_step2`` is a complete integration step.
        # It advances time and changes qpos / qvel, which makes a dataset state
        # replay start one MuJoCo step away from the recorded row.  Forward
        # dynamics updates kinematics and sensors without changing the state.
        mujoco.mj_forward(sim.model._model, sim.data._data)

