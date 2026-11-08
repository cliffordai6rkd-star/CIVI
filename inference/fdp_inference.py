"""Streaming FDP inference with a causal CAD-WM future-wrench provider."""

from collections import deque
import contextlib
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

from .cad_wm_provider import CADWMWrenchProvider
from .config import load_config
from .inference_base import InferenceBase, InferenceObservation, InferenceResult, InferenceSpec


def load_dp(config, device):
    """Restore policy plus saved normalizer; optimizer states are never applied."""
    import dill
    import hydra
    from omegaconf import OmegaConf, open_dict

    root = Path(config["repo"]).resolve()
    if not (root / "diffusion_policy").is_dir():
        raise FileNotFoundError(f"DP repo not found: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import diffusion_policy
    if Path(diffusion_policy.__file__).resolve().parent != root / "diffusion_policy":
        raise RuntimeError("A different diffusion_policy package was already imported; use a fresh process")
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    checkpoint = Path(config["checkpoint"])
    # The DP checkpoint contains the resolved config and optimizer-era Python
    # objects in addition to tensors.  PyTorch 2.6 changed the implicit
    # default to ``weights_only=True``; opt out explicitly for this trusted
    # project checkpoint and keep compatibility with older torch releases.
    try:
        payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill,
                             weights_only=False)
    except TypeError:  # torch versions before the weights_only argument
        payload = torch.load(checkpoint, map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]
    if not OmegaConf.is_config(cfg):
        cfg = OmegaConf.create(cfg)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    with open_dict(cfg):
        backbone = config.get("backbone") or cfg.policy.dino_model_name_or_path
        if not Path(backbone).is_absolute():
            backbone = str(root / backbone)
        cfg.policy.dino_model_name_or_path = str(backbone)
        if config.get("diffusion_steps") is not None:
            cfg.policy.num_inference_steps = int(config["diffusion_steps"])
    key = config.get("state_key", "auto")
    if key == "auto":
        key = "ema_model" if "ema_model" in payload["state_dicts"] else "model"
    if key not in payload["state_dicts"]:
        raise ValueError(f"DP checkpoint has no state_dicts[{key!r}]")
    policy = hydra.utils.instantiate(cfg.policy)
    policy.load_state_dict(payload["state_dicts"][key], strict=True)
    policy.to(device).eval().requires_grad_(False)
    del payload
    return policy, cfg, key


def _vector(value, size, name):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite physical-unit vector [{size}]")
    return array.copy()


def _image(value, shape, name):
    value = np.asarray(value)
    channels, height, width = shape
    if value.dtype == np.uint8 and value.ndim == 3 and value.shape[-1] == channels:
        if value.shape[:2] != (height, width):
            value = np.asarray(Image.fromarray(value).resize((width, height), Image.Resampling.BILINEAR))
        return np.ascontiguousarray(value.transpose(2, 0, 1), dtype=np.float32) / 255.0
    if value.shape == shape and np.issubdtype(value.dtype, np.floating):
        if not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
            raise ValueError(f"{name} float CHW RGB must be finite and in [0,1]")
        return np.array(value, dtype=np.float32, copy=True)
    raise ValueError(f"{name} must be uint8 HWC RGB or float CHW {shape} in [0,1]")


class FDPInference(InferenceBase):
    """Emit absolute xyz+xyzw chunks, with a gripper command for 8D policies.

    The phase source is selected by the runtime config. With ``cad_wm`` the
    previous t+1 forecast is the current phase wrench; with ``measured`` the
    current sensor wrench drives the phase history. Future network inputs always
    come from CAD-WM. The padded WM history makes a future forecast available on
    the first frame; image context still needs two observations before action.
    """

    @classmethod
    def from_config(cls, path):
        config = load_config(path)
        return cls.from_loaded_config(config)

    @classmethod
    def from_loaded_config(cls, config):
        """Build an engine from an already resolved config mapping.

        The HTTP FDP service uses this hook to override the teacher checkpoint
        from the command line without writing a temporary YAML file.
        """
        runtime = config["runtime"]
        device = torch.device(runtime["device"])
        # Fail on the missing WM path before allocating a large DP policy.
        wm = CADWMWrenchProvider(
            repo=config["cad_wm"]["repo"], checkpoint=config["cad_wm"].get("checkpoint"),
            device=config["cad_wm"].get("device") or str(device),
            steps=config["cad_wm"].get("steps"), solver=config["cad_wm"].get("solver"),
        )
        policy, dp_config, state_key = load_dp(config["dp"], device)
        instance = cls(policy=policy, dp_config=dp_config, wrench_provider=wm,
                       device=device, phase_wrench_source=runtime["phase_wrench_source"],
                       future_contact_mode=runtime.get("future_contact_mode"),
                       seed=runtime["seed"], timing_tolerance_s=runtime["timing_tolerance_s"],
                       amp_enabled=runtime["amp_enabled"], amp_dtype=runtime["amp_dtype"])
        instance.checkpoint_info = {"dp": config["dp"]["checkpoint"], "dp_state_key": state_key,
                                    "cad_wm": config["cad_wm"]["checkpoint"],
                                    "future_contact_mode": instance.future_contact_mode}
        return instance

    def __init__(self, *, policy, dp_config, wrench_provider, device="cpu", seed=42,
                 phase_wrench_source="cad_wm", timing_tolerance_s=0.001,
                 future_contact_mode=None,
                 amp_enabled=True, amp_dtype="bfloat16"):
        self.device = torch.device(device)
        self.policy = policy.to(self.device).eval().requires_grad_(False)
        self.wrench_provider = wrench_provider
        self.seed = int(seed)
        self.phase_wrench_source = phase_wrench_source
        if phase_wrench_source not in {"cad_wm", "measured"}:
            raise ValueError("phase_wrench_source must be cad_wm or measured")
        checkpoint_contact_mode = str(getattr(policy, "future_contact_mode", "legacy_rollout"))
        contact_mode = checkpoint_contact_mode if future_contact_mode is None else str(future_contact_mode)
        if contact_mode not in {"legacy_rollout", "token_threshold"}:
            raise ValueError(
                "future_contact_mode must be legacy_rollout or token_threshold"
            )
        # The teacher checkpoint owns the default preprocessing contract.  A
        # runtime override is explicit because a stochastic CAD-WM forecast
        # can otherwise toggle token_threshold at the contact boundary every
        # frame.  This does not alter diffusion sampling or action noise.
        self.future_contact_mode = contact_mode
        if hasattr(self.policy, "future_contact_mode"):
            self.policy.future_contact_mode = contact_mode
        self.timing_tolerance_s = float(timing_tolerance_s)
        self.dt = float(dp_config.task.dataset.timestamp_step_sec)
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("DP timestamp_step_sec must be positive")
        if not np.isfinite(self.timing_tolerance_s) or not 0 <= self.timing_tolerance_s < self.dt / 2:
            raise ValueError("timing_tolerance_s must be finite, nonnegative and below half a frame")
        self.action_dim = int(dp_config.shape_meta.action.shape[0])
        fields = list(dp_config.task.dataset.action_feature_fields)
        columns = dp_config.task.dataset.get("action_feature_columns", {})
        pose_target = self.action_dim == 7 and fields == ["observation.ee_pose"]
        gripper_target = (self.action_dim == 8 and fields == ["observation.ee_pose", "action"]
                          and list(columns.get("action", [])) == [6])
        if bool(dp_config.policy.relative_pose_actions) or not (pose_target or gripper_target):
            raise ValueError("This FDP adapter expects absolute observation.ee_pose, optionally followed by action column 6 (gripper)")
        self.wrench_key = str(policy.wrench_key)
        self.phase_key = str(policy.phase_wrench_key)
        if self.wrench_key == self.phase_key:
            raise ValueError("This adapter requires separate future and history wrench keys")
        shape_meta = dp_config.shape_meta
        self.image_shapes = {key: tuple(shape_meta.obs[key].shape) for key in policy.image_keys}
        self.future_steps, wrench_dim = tuple(shape_meta.obs[self.wrench_key].shape)
        self.phase_steps, phase_dim = tuple(shape_meta.obs[self.phase_key].shape)
        if wrench_dim != 6 or phase_dim != 6 or wrench_provider.joint_dim != 7:
            raise ValueError("Expected seven robot joints and six wrench channels")
        windows = dp_config.task.dataset.windowed_lowdim_feature_groups
        offsets = np.asarray(windows[self.wrench_key].offsets_sec, dtype=np.float64)
        history = windows[self.phase_key]
        if int(history.start_offset) != -(self.phase_steps - 1) or int(history.stride) != 1:
            raise ValueError("Phase wrench history must be consecutive native rows ending at the current frame")
        if offsets.shape != (self.future_steps,) or not np.allclose(offsets, np.arange(1, self.future_steps + 1) * self.dt, atol=1e-7, rtol=0):
            raise ValueError("FDP future wrench offsets must be consecutive t+1 through t+K")
        if not np.isclose(wrench_provider.sample_rate_hz, 1 / self.dt) or (
            len(wrench_provider.future_offsets_s) != len(offsets)
            or not np.allclose(wrench_provider.future_offsets_s, offsets, atol=1e-7, rtol=0)
        ):
            raise ValueError("CAD-WM and FDP sample rates / future wrench timestamps do not match")
        self.future_offsets_s = offsets
        self.observation_steps = int(policy.n_obs_steps)
        if tuple(shape_meta.action.shape) != (self.action_dim,) or self.action_dim not in (7, 8):
            raise ValueError("FDP output must be xyz+xyzw, optionally followed by a gripper command")
        # This checkpoint's returned indices start at the current anchor, t,
        # because action_start_index == n_obs_steps-1. Preserve that convention.
        latency = int(dp_config.task.dataset.get("n_latency_steps", 0))
        if latency < 0:
            raise ValueError("n_latency_steps must be nonnegative")
        first_action_offset = int(policy.action_start_index) + latency - (self.observation_steps - 1)
        if first_action_offset < 0:
            raise ValueError(
                "FDP action chunk starts before the latest observation; "
                "check action_start_index, n_obs_steps and n_latency_steps"
            )
        self.action_offsets_s = (np.arange(policy.n_action_steps) + policy.action_start_index
                                 - (self.observation_steps - 1) + latency) * self.dt
        self.amp_enabled = bool(amp_enabled) and self.device.type == "cuda"
        if amp_dtype not in {"bfloat16", "float16"}:
            raise ValueError("amp_dtype must be bfloat16 or float16")
        self.amp_dtype = getattr(torch, amp_dtype)
        self.checkpoint_info = {}
        self.reset()

    @property
    def spec(self):
        required = ("images", "q", "dq", "tau")
        if self.phase_wrench_source == "measured":
            required += ("wrench",)
        action_format = "absolute_ee_pose_xyz_xyzw" + ("_gripper_command" if self.action_dim == 8 else "")
        return InferenceSpec(1 / self.dt, self.observation_steps, self.image_shapes,
                             (int(self.policy.n_action_steps), self.action_dim), action_format, required)

    def reset(self, *, seed=None):
        if seed is not None:
            self.seed = int(seed)
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self.wrench_provider.reset(seed=self.seed + 1)
        self._state_history = deque(maxlen=self.wrench_provider.history_steps)
        self._phase_history = deque(maxlen=self.phase_steps)
        self._contexts = deque(maxlen=self.observation_steps)
        self._last_timestamp = None
        self._previous_future = None
        self._latest_future = None
        self._frames_seen = 0
        self._warmup_reason = "waiting_for_observation"
        self.policy.reset()

    @torch.inference_mode()
    def observe(self, observation: InferenceObservation):
        timestamp = float(observation.timestamp_s)
        if not np.isfinite(timestamp):
            raise ValueError("timestamp_s must be finite seconds")
        if self._last_timestamp is not None:
            delta = timestamp - self._last_timestamp
            if delta <= 0 or abs(delta - self.dt) > self.timing_tolerance_s:
                raise ValueError(f"Expected a new frame every {self.dt}s, got {delta}s. "
                                 "Call observe for every frame, or reset at an episode boundary.")
        state = {key: _vector(getattr(observation, key), 7, key) for key in ("q", "dq", "tau")}
        images = {key: _image(observation.images[key], shape, key) for key, shape in self.image_shapes.items()}
        phase = None
        if self.phase_wrench_source == "measured":
            phase = _vector(observation.wrench, 6, "wrench")
        elif self._previous_future is not None:
            phase = self._previous_future[0].copy()
        states = [*self._state_history, state][-self.wrench_provider.history_steps:]
        future = None
        if self.wrench_provider.pad_history or len(states) == self.wrench_provider.history_steps:
            history = {key: np.stack([row[key] for row in states])[None] for key in state}
            future = np.asarray(self.wrench_provider.predict(history), dtype=np.float32)
            if future.shape != (self.future_steps, 6) or not np.isfinite(future).all():
                raise ValueError(f"CAD-WM must return finite physical wrench [{self.future_steps},6]")
            future = future.copy()
        # Commit the frame only after validating all sensor/model outputs.
        self._state_history.append(state)
        self._last_timestamp = timestamp
        self._frames_seen += 1
        self._previous_future = future
        self._latest_future = future
        if phase is not None:
            self._phase_history.append(phase)
        if phase is not None and future is not None:
            rows = list(self._phase_history)
            rows = [rows[0]] * (self.phase_steps - len(rows)) + rows
            self._contexts.append({"images": images, "phase": np.stack(rows),
                                   "future": future, "timestamp": timestamp})
        self._warmup_reason = ("waiting_for_cad_wm_history" if future is None else
                               "waiting_for_current_wrench_forecast" if phase is None else
                               "waiting_for_image_history" if len(self._contexts) < self.observation_steps else None)
        return self._warmup_reason is None

    def _policy_observations(self):
        obs = {key: np.stack([row["images"][key] for row in self._contexts])[None] for key in self.image_shapes}
        obs[self.wrench_key] = np.stack([row["future"] for row in self._contexts])[None]
        obs[self.phase_key] = np.stack([row["phase"] for row in self._contexts])[None]
        if self.policy.future_phase_wrench_key is not None:
            obs[self.policy.future_phase_wrench_key] = obs[self.wrench_key]
        return {key: torch.as_tensor(value, dtype=torch.float32, device=self.device) for key, value in obs.items()}

    @torch.inference_mode()
    def predict(self, observation=None):
        if observation is not None:
            self.observe(observation)
        if self._last_timestamp is None:
            raise RuntimeError("Call observe() or pass an observation before predicting")
        info = {"wrench_source": "cad_wm", "phase_wrench_source": self.phase_wrench_source,
                "future_contact_mode": self.future_contact_mode,
                "frames_seen": self._frames_seen, "warmup_reason": self._warmup_reason,
                "future_wrench": None if self._latest_future is None else self._latest_future.copy(),
                "future_wrench_offsets_s": self.future_offsets_s.copy(), **self.checkpoint_info}
        if self._warmup_reason is not None:
            return InferenceResult(False, self._last_timestamp, None, None, self.spec.action_format, info)
        autocast = (torch.autocast("cuda", dtype=self.amp_dtype) if self.amp_enabled else contextlib.nullcontext())
        with autocast:
            result = self.policy.predict_action(self._policy_observations(), generator=self.generator)
        actions = result["action"][0].detach().float().cpu().numpy().copy()
        if actions.shape != self.spec.action_shape or not np.isfinite(actions).all():
            raise RuntimeError("FDP returned an invalid action chunk")
        norms = np.linalg.norm(actions[:, 3:7], axis=-1)
        if not np.allclose(norms, 1, atol=1e-3, rtol=0):
            raise RuntimeError("FDP returned a non-unit xyzw quaternion")
        info["context_timestamps_s"] = np.array([row["timestamp"] for row in self._contexts])
        return InferenceResult(True, self._last_timestamp, actions,
                               self._last_timestamp + self.action_offsets_s, self.spec.action_format, info)
