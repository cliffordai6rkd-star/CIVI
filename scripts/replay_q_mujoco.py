#!/usr/bin/env python3
"""Replay joint positions in MuJoCo and record dq / tau.

The common LIBERO dataset layout is supported out of the box::

    python scripts/replay_q_mujoco.py \
        --dataset /path/to/*_demo.hdf5 --demo demo_0

    python scripts/replay_q_mujoco.py \
        --dataset dataset/libero_90/task_demo.hdf5 --demo-range 10 19

By default the script reads ``data/<demo>/obs/joint_states`` (the seven Panda
joint positions), reconstructs the scene from the HDF5 BDDL metadata, and
writes a copy of the original HDF5 with two new datasets,
``data/<demo>/obs/dq`` and ``data/<demo>/obs/tau``. Visualization is optional
and disabled by default. Pass ``--viewer`` to open the interactive MuJoCo
viewer and watch the q trajectory in real time; ``--viewer-hold`` keeps the
last frame open after a demo finishes.

Without ``--viewer`` the script runs as an offline calculation and does not
sleep between samples. Use ``--realtime`` when the process itself must follow
the q command period.

``dq`` is obtained from the replayed q trajectory (or from ``--dq-key`` when
one is present).  The default ``tau`` is MuJoCo inverse dynamics
(``data.qfrc_inverse``), which is well-defined for a position-only trajectory.
Use ``--tau-source actuator`` to record ``data.qfrc_actuator`` instead.  With
no controls that quantity is zero, so an actuator/control dataset or a PD
controller must be supplied when actuator torque is desired.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _ensure_libero_config(libero_root: Path) -> None:
    """Create a non-interactive LIBERO config when a fresh checkout is used.

    The upstream LIBERO package asks an interactive question on its first
    import.  A replay script should be usable from a batch job, so we create
    the same config file with paths relative to this checkout when necessary.
    """

    config_dir = Path(os.environ.get("LIBERO_CONFIG_PATH", "~/.libero")).expanduser()
    config_file = config_dir / "config.yaml"
    if config_file.exists():
        return
    config_dir.mkdir(parents=True, exist_ok=True)
    import yaml

    root = (libero_root / "libero").resolve()
    dataset_root = (libero_root / "dataset").resolve()
    if not dataset_root.exists():
        # Keep LIBERO's upstream default for checkouts that do not carry a
        # local dataset directory yet.
        dataset_root = root.parent / "datasets"
    config = {
        "benchmark_root": str(root),
        "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"),
        "datasets": str(dataset_root),
        "assets": str(root / "assets"),
    }
    config_file.write_text(yaml.safe_dump(config), encoding="utf-8")


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise RuntimeError("HDF5 input/output requires h5py (pip install h5py).") from exc
    return h5py


def _decode_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _decode_attr(value.item())
    return value


def _demo_name(value: str | int) -> str:
    value = str(value)
    return value if value.startswith("demo_") else f"demo_{value}"


def _natural_demo_keys(keys: Iterable[str]) -> List[str]:
    def sort_key(x: str):
        match = re.search(r"(\d+)$", x)
        return int(match.group(1)) if match else x

    return sorted(keys, key=sort_key)


def _get_nested(container: Any, key: str) -> Any:
    """Read a slash-separated key from an h5py group or a mapping."""

    obj = container
    for part in key.strip("/").split("/"):
        if not part:
            continue
        if isinstance(obj, dict):
            if part not in obj:
                raise KeyError(key)
            obj = obj[part]
        else:
            obj = obj[part]
    return obj


def _candidate_q_keys(group: Any) -> List[str]:
    candidates = [
        "q",
        "qpos",
        "joint_positions",
        "joint_states",
        "robot0_joint_pos",
        "obs/joint_states",
        "obs/robot0_joint_pos",
        "obs/q",
        "obs/qpos",
        "states",
    ]
    return [key for key in candidates if _key_exists(group, key)]


def _key_exists(container: Any, key: str) -> bool:
    try:
        _get_nested(container, key)
        return True
    except (KeyError, IndexError, TypeError):
        return False


def _read_array(container: Any, key: str) -> np.ndarray:
    obj = _get_nested(container, key)
    return np.asarray(obj[()] if hasattr(obj, "__getitem__") and hasattr(obj, "shape") else obj)


@dataclass
class DatasetDemo:
    q: np.ndarray
    dq: Optional[np.ndarray]
    ctrl: Optional[np.ndarray]
    q_key: str
    dq_key: Optional[str]
    ctrl_key: Optional[str]
    attrs: Dict[str, Any]
    bddl_file_name: Optional[str]
    env_kwargs: Dict[str, Any]
    model_file: Optional[str] = None


class ViewerClosed(RuntimeError):
    """Raised when the user closes the MuJoCo viewer during replay."""


def _require_mujoco():
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - depends on user environment
        raise RuntimeError(
            "MuJoCo is required for q replay. Install it in the active environment with "
            '`python -m pip install "mujoco>=2.3.0"`.'
        ) from exc
    return mujoco


def _require_libero_runtime() -> None:
    """Check imports needed to construct a LIBERO BDDL environment."""

    required = {
        "yaml": "pyyaml",
        "robosuite": "robosuite==1.4.0",
        "bddl": "bddl==1.0.1",
        "future": "future",
        "easydict": "easydict",
        "cloudpickle": "cloudpickle",
        "gym": "gym==0.25.2",
    }
    missing = []
    for module_name, package_name in required.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(package_name)
    if missing:
        raise RuntimeError(
            "LIBERO environment dependencies are missing: " + ", ".join(missing) + ". "
            "Run `python -m pip install -r requirements.txt` in the active environment."
        )


def _load_hdf5_demo(path: Path, demo: str, q_key: Optional[str], dq_key: Optional[str],
                    ctrl_key: Optional[str]) -> DatasetDemo:
    h5py = _require_h5py()
    with h5py.File(path, "r") as file:
        if "data" in file:
            data_group = file["data"]
            demo_key = _demo_name(demo)
            if demo_key not in data_group:
                raise KeyError(f"{demo_key} not found. Available demos: {_natural_demo_keys(data_group.keys())}")
            group = data_group[demo_key]
            attrs = {key: _decode_attr(value) for key, value in data_group.attrs.items()}
            demo_attrs = {key: _decode_attr(value) for key, value in group.attrs.items()}
        else:
            group = file
            attrs = {key: _decode_attr(value) for key, value in file.attrs.items()}
            demo_attrs = {key: _decode_attr(value) for key, value in group.attrs.items()}
            demo_key = str(demo)

        chosen_q_key = q_key
        if chosen_q_key is None:
            candidates = _candidate_q_keys(group)
            if not candidates:
                raise KeyError(
                    f"No q trajectory found in {path}:{demo_key}. Tried: "
                    "q, qpos, joint_positions, joint_states, obs/joint_states, states."
                )
            chosen_q_key = candidates[0]
        q = np.asarray(_read_array(group, chosen_q_key), dtype=np.float64)
        if q.ndim == 1:
            q = q[:, None]
        if q.ndim != 2:
            raise ValueError(f"q at {chosen_q_key} must have shape (T,D), got {q.shape}")

        def optional_array(key: Optional[str], aliases: Sequence[str]) -> Tuple[Optional[np.ndarray], Optional[str]]:
            selected = key
            if selected is None:
                for alias in aliases:
                    if _key_exists(group, alias):
                        selected = alias
                        break
            if selected is None:
                return None, None
            value = np.asarray(_read_array(group, selected), dtype=np.float64)
            if value.ndim == 1:
                value = value[:, None]
            if value.ndim != 2 or value.shape[0] != q.shape[0]:
                raise ValueError(f"{selected} must have shape (T,D) with T={q.shape[0]}, got {value.shape}")
            return value, selected

        dq, selected_dq_key = optional_array(
            dq_key,
            ("dq", "qvel", "joint_vel", "robot0_joint_vel", "obs/joint_vel", "obs/robot0_joint_vel"),
        )
        ctrl, selected_ctrl_key = optional_array(ctrl_key, ("ctrl", "controls", "torques"))

        env_args = attrs.get("env_args", "")
        if isinstance(env_args, str):
            try:
                env_args = json.loads(env_args)
            except json.JSONDecodeError:
                env_args = {}
        if not isinstance(env_args, dict):
            env_args = {}
        env_kwargs = dict(env_args.get("env_kwargs", {}))
        bddl = attrs.get("bddl_file_name")
        if bddl is None:
            bddl = env_kwargs.get("bddl_file_name")
        return DatasetDemo(
            q=q,
            dq=dq,
            ctrl=ctrl,
            q_key=chosen_q_key,
            dq_key=selected_dq_key,
            ctrl_key=selected_ctrl_key,
            attrs=attrs,
            bddl_file_name=str(bddl) if bddl is not None else None,
            env_kwargs=env_kwargs,
            model_file=str(demo_attrs["model_file"]) if demo_attrs.get("model_file") is not None else None,
        )


def _load_array_dataset(path: Path, q_key: Optional[str], dq_key: Optional[str],
                        ctrl_key: Optional[str]) -> DatasetDemo:
    if path.suffix.lower() == ".npy":
        q = np.asarray(np.load(path), dtype=np.float64)
        arrays: Dict[str, np.ndarray] = {}
    elif path.suffix.lower() == ".npz":
        loaded = np.load(path)
        arrays = {key: np.asarray(loaded[key]) for key in loaded.files}
        selected = q_key or next((key for key in ("q", "qpos", "joint_states", "robot0_joint_pos", "states") if key in arrays), None)
        if selected is None:
            raise KeyError(f"No q/qpos/joint_states/states array found in {path}; pass --q-key.")
        q = np.asarray(arrays[selected], dtype=np.float64)
        q_key = selected
    else:
        raise ValueError(f"Unsupported dataset format {path.suffix}; use .hdf5, .h5, .npz or .npy.")
    if q.ndim == 1:
        q = q[:, None]
    if q.ndim != 2:
        raise ValueError(f"q must have shape (T,D), got {q.shape}")

    def pick(key: Optional[str], names: Sequence[str]) -> Tuple[Optional[np.ndarray], Optional[str]]:
        selected = key or next((name for name in names if name in arrays), None)
        if selected is None:
            return None, None
        value = np.asarray(arrays[selected], dtype=np.float64)
        if value.ndim == 1:
            value = value[:, None]
        if value.ndim != 2 or value.shape[0] != q.shape[0]:
            raise ValueError(f"{selected} must have shape (T,D) with T={q.shape[0]}, got {value.shape}")
        return value, selected

    dq, dq_key = pick(dq_key, ("dq", "qvel", "joint_vel", "robot0_joint_vel"))
    ctrl, ctrl_key = pick(ctrl_key, ("ctrl", "controls", "torques"))
    return DatasetDemo(q, dq, ctrl, q_key or "array", dq_key, ctrl_key, {}, None, {})


def load_demos(path: Path, demo: str, q_key: Optional[str], dq_key: Optional[str],
               ctrl_key: Optional[str], demo_range: Optional[Tuple[int, int]] = None
               ) -> List[Tuple[str, DatasetDemo]]:
    if path.suffix.lower() in (".hdf5", ".h5"):
        if demo_range is None and demo.lower() != "all":
            return [(_demo_name(demo), _load_hdf5_demo(path, demo, q_key, dq_key, ctrl_key))]
        h5py = _require_h5py()
        with h5py.File(path, "r") as file:
            if "data" not in file:
                raise ValueError("Demo selection requires an HDF5 file with a data/ group.")
            names = _natural_demo_keys(file["data"].keys())
        all_names = list(names)
        if demo_range is not None:
            start, end = demo_range
            if start < 0 or end < start:
                raise ValueError(f"Invalid demo range {demo_range}; expected 0 <= START <= END.")
            selected_names = []
            for position, name in enumerate(names):
                match = re.search(r"(\d+)$", name)
                demo_index = int(match.group(1)) if match else position
                if start <= demo_index <= end:
                    selected_names.append(name)
            names = selected_names
            if not names:
                raise ValueError(
                    f"Demo range [{start}, {end}] selected no demos. Available demos: {all_names}"
                )
        return [(name, _load_hdf5_demo(path, name, q_key, dq_key, ctrl_key)) for name in names]
    if demo_range is not None:
        raise ValueError("--demo-range is supported for HDF5 datasets only.")
    return [("demo_0", _load_array_dataset(path, q_key, dq_key, ctrl_key))]


def _resolve_bddl(raw: Optional[str], dataset: Path, repo_root: Path) -> Optional[Path]:
    if not raw:
        return None
    raw_path = Path(os.path.expanduser(str(raw)))
    candidates = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend([
            Path.cwd() / raw_path,
            dataset.parent / raw_path,
            repo_root / raw_path,
            repo_root / "libero" / raw_path,
            repo_root / "libero" / "libero" / "bddl_files" / raw_path,
        ])
        text = str(raw_path).replace("\\", "/")
        marker = "libero/libero/"
        if marker in text:
            candidates.append(repo_root / text[text.index(marker) :])
        if text.startswith("chiliocosm/bddl_files/"):
            suffix = text.split("chiliocosm/bddl_files/", 1)[1]
            candidates.extend([
                repo_root / "libero" / "libero" / "bddl_files" / suffix.replace("libero_100/", "", 1),
                repo_root / "libero" / "libero" / "bddl_files" / suffix,
            ])
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    return None


@dataclass
class SimHandle:
    model: Any
    data: Any
    env: Any = None
    sim: Any = None
    renderer: Any = None
    viewer: Any = None

    def close(self) -> None:
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
        if self.env is not None:
            self.env.close()


def _make_sim(demo: DatasetDemo, dataset: Path, args: argparse.Namespace, repo_root: Path) -> SimHandle:
    mujoco = _require_mujoco()

    if args.xml:
        model = mujoco.MjModel.from_xml_path(str(Path(args.xml).expanduser().resolve()))
        return SimHandle(model=model, data=mujoco.MjData(model))

    if args.scene_source == "dataset" and demo.model_file:
        xml = _normalize_dataset_model_xml(demo.model_file, repo_root)
        try:
            model = mujoco.MjModel.from_xml_string(xml)
        except Exception as exc:
            raise RuntimeError(
                "The model_file stored in the HDF5 could not be loaded after asset-path "
                "normalization. Use --scene-source bddl to rebuild the scene from BDDL."
            ) from exc
        return SimHandle(model=model, data=mujoco.MjData(model))

    bddl = _resolve_bddl(args.bddl or demo.bddl_file_name, dataset, repo_root)
    if bddl is None:
        raise FileNotFoundError(
            "Could not locate a BDDL file. Pass --bddl /path/to/task.bddl, or use --xml. "
            f"Dataset metadata value was {demo.bddl_file_name!r}."
        )

    _ensure_libero_config(repo_root)
    from libero.libero.envs import OffScreenRenderEnv

    kwargs = dict(demo.env_kwargs)
    kwargs.pop("controller_configs", None)
    kwargs["bddl_file_name"] = str(bddl)
    kwargs["has_renderer"] = False
    kwargs["has_offscreen_renderer"] = True
    kwargs["use_camera_obs"] = False
    kwargs["camera_names"] = [args.camera]
    kwargs["camera_heights"] = args.render_height
    kwargs["camera_widths"] = args.render_width
    kwargs.pop("camera_depths", None)
    env = OffScreenRenderEnv(**kwargs)
    env.reset()
    sim = env.env.sim
    return SimHandle(model=sim.model._model, data=sim.data._data, env=env, sim=sim)


def _launch_viewer(handle: SimHandle, show_collision: bool = False) -> None:
    """Open the interactive MuJoCo viewer for the exact model/data being replayed."""

    mujoco = _require_mujoco()
    try:
        import mujoco.viewer
        handle.viewer = mujoco.viewer.launch_passive(handle.model, handle.data)
        if not show_collision:
            # LIBERO / robosuite put collision geoms in group 0 and visual
            # meshes in group 1. MuJoCo's viewer shows group 0 by default,
            # which makes the arm look green / blue and can look like a mesh
            # interference. Hide that debug geometry while keeping collision
            # calculations active in the model and data.
            with handle.viewer.lock():
                handle.viewer.opt.geomgroup[:] = 0
                if handle.viewer.opt.geomgroup.shape[0] > 1:
                    handle.viewer.opt.geomgroup[1] = 1
        handle.viewer.sync()
    except Exception as exc:  # pragma: no cover - display/GL dependent
        raise RuntimeError(
            "Could not open the MuJoCo viewer. Check that a graphical display is available "
            "(or use an X11/remote desktop session); headless servers need Xvfb for the viewer."
        ) from exc


def _normalize_dataset_model_xml(xml: str, repo_root: Path) -> str:
    """Relocate absolute asset paths embedded in old LIBERO model XML."""

    import importlib.util

    local_assets = (repo_root / "libero" / "libero" / "assets").resolve()
    xml = xml.replace(
        "/home/yifengz/workspace/libero-dev/chiliocosm/assets",
        str(local_assets),
    )
    spec = importlib.util.find_spec("robosuite")
    if spec is not None and spec.origin is not None:
        robosuite_assets = Path(spec.origin).resolve().parent / "models" / "assets"
        xml = xml.replace(
            "/home/yifengz/workspace/robosuite-master/robosuite/models/assets",
            str(robosuite_assets),
        )
    return xml


def _joint_names(model: Any) -> List[str]:
    return [model.joint(i).name or f"joint_{i}" for i in range(model.njnt)]


def _joint_widths(model: Any, joint_id: int) -> Tuple[int, int]:
    import mujoco

    kind = int(model.jnt_type[joint_id])
    if kind == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7, 6
    if kind == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4, 3
    return 1, 1


@dataclass
class JointMap:
    names: List[str]
    qpos_indices: List[int]
    dof_indices: List[int]
    full_qpos: bool = False


def _make_joint_map(model: Any, q_dim: int, explicit_names: Optional[str]) -> JointMap:
    import mujoco

    if q_dim == int(model.nq):
        return JointMap([f"qpos_{i}" for i in range(model.nq)], list(range(model.nq)), list(range(model.nv)), True)

    names = _joint_names(model)
    if explicit_names:
        requested = [name.strip() for name in explicit_names.split(",") if name.strip()]
        missing = [name for name in requested if name not in names]
        if missing:
            raise KeyError(f"Unknown joint names {missing}. Available joints: {names}")
        joint_ids = [names.index(name) for name in requested]
    else:
        actuated = []
        for i in range(int(model.nu)):
            jid = int(model.actuator_trnid[i, 0])
            if jid >= 0 and jid not in actuated:
                actuated.append(jid)
        robot = [jid for jid in actuated if "robot0_" in names[jid].lower() and "gripper" not in names[jid].lower()]
        gripper = [jid for jid in actuated if "gripper" in names[jid].lower()]
        ordered = robot + gripper + [jid for jid in actuated if jid not in robot and jid not in gripper]
        if not ordered:
            ordered = [jid for jid in range(int(model.njnt)) if int(model.jnt_type[jid]) in (
                int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))]
        joint_ids = []
        width = 0
        for jid in ordered:
            q_width, _ = _joint_widths(model, jid)
            if width + q_width > q_dim:
                break
            joint_ids.append(jid)
            width += q_width
            if width == q_dim:
                break

    qpos_indices: List[int] = []
    dof_indices: List[int] = []
    selected_names: List[str] = []
    for jid in joint_ids:
        q_width, dof_width = _joint_widths(model, jid)
        qpos_start = int(model.jnt_qposadr[jid])
        dof_start = int(model.jnt_dofadr[jid])
        qpos_indices.extend(range(qpos_start, qpos_start + q_width))
        dof_indices.extend(range(dof_start, dof_start + dof_width))
        selected_names.extend([names[jid]] * q_width)

    if len(qpos_indices) != q_dim:
        raise ValueError(
            f"q has D={q_dim}, but could not map it to this model (nq={model.nq}, nv={model.nv}). "
            f"Selected joints={selected_names}. Pass --joint-names as a comma-separated list. "
            f"Available joints: {names}"
        )
    return JointMap(selected_names, qpos_indices, dof_indices, False)


def _expand_q_to_full(model: Any, q: np.ndarray, mapping: JointMap, initial_qpos: np.ndarray) -> np.ndarray:
    full = np.repeat(np.asarray(initial_qpos, dtype=np.float64)[None, :], q.shape[0], axis=0)
    if mapping.full_qpos:
        full[:] = q
    else:
        full[:, mapping.qpos_indices] = q
    return full


def _differentiate_qpos(model: Any, qpos: np.ndarray, dt: float) -> np.ndarray:
    import mujoco

    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    dq = np.zeros((qpos.shape[0], int(model.nv)), dtype=np.float64)
    if qpos.shape[0] == 1:
        return dq
    for t in range(qpos.shape[0]):
        if t == 0:
            q0, q1 = qpos[0], qpos[1]
        elif t == qpos.shape[0] - 1:
            q0, q1 = qpos[-2], qpos[-1]
        else:
            q0, q1 = qpos[t - 1], qpos[t + 1]
        step = dt if t in (0, qpos.shape[0] - 1) else 2.0 * dt
        mujoco.mj_differentiatePos(model, dq[t], step, q0, q1)
    return dq


def _unpack_flattened_state(
    model: Any, states: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Extract qpos and qvel from robosuite's flattened simulator state.

    ``MjSimState.flatten()`` is ``[time, qpos, qvel, ...]``.  The time value
    is easy to mistake for the first joint position, which shifts every robot
    joint and produces an apparently incorrect posture in the viewer.
    """

    nq, nv = int(model.nq), int(model.nv)
    if states.shape[1] >= 1 + nq + nv:
        offset = 1  # robosuite / MjSimState layout
    elif states.shape[1] >= nq + nv:
        offset = 0  # plain [qpos, qvel] layout
    else:
        raise ValueError(
            f"Flattened state has D={states.shape[1]}, but model needs at least "
            f"nq+nv={nq + nv} values (or 1+nq+nv with time)."
        )
    qpos = np.asarray(states[:, offset : offset + nq], dtype=np.float64)
    qvel = np.asarray(states[:, offset + nq : offset + nq + nv], dtype=np.float64)
    state_time = np.asarray(states[:, 0], dtype=np.float64) if offset == 1 else None
    return qpos, qvel, state_time


def _map_optional_dq(model: Any, value: np.ndarray, mapping: JointMap, dt: float) -> np.ndarray:
    if value.shape[1] == int(model.nv):
        return value.copy()
    if value.shape[1] == len(mapping.dof_indices):
        full = np.zeros((value.shape[0], int(model.nv)), dtype=np.float64)
        full[:, mapping.dof_indices] = value
        return full
    raise ValueError(
        f"dq has D={value.shape[1]}, expected model.nv={model.nv} or mapped D={len(mapping.dof_indices)}."
    )


def _finite_difference(value: np.ndarray, dt: float) -> np.ndarray:
    if value.shape[0] < 2:
        return np.zeros_like(value)
    return np.gradient(value, dt, axis=0, edge_order=1)


def _clip_ctrl(model: Any, ctrl: np.ndarray) -> np.ndarray:
    result = np.asarray(ctrl, dtype=np.float64).copy()
    for i in range(int(model.nu)):
        if int(model.actuator_ctrllimited[i]):
            result[i] = np.clip(result[i], model.actuator_ctrlrange[i, 0], model.actuator_ctrlrange[i, 1])
    return result


def _camera_argument(model: Any, requested: str) -> Optional[str | int]:
    """Return a valid camera argument, falling back to MuJoCo's free camera."""

    if requested is None or str(requested).lower() in ("none", "free", "-1"):
        return None if requested is None or str(requested).lower() == "none" else -1
    try:
        available = {model.camera(i).name for i in range(int(model.ncam))}
    except Exception:
        available = set()
    return requested if requested in available else -1


def replay_demo(handle: SimHandle, demo: DatasetDemo, args: argparse.Namespace) -> Dict[str, np.ndarray | List[str]]:
    import mujoco

    model, data = handle.model, handle.data
    q = np.asarray(demo.q, dtype=np.float64)
    state_dq: Optional[np.ndarray] = None
    state_time: Optional[np.ndarray] = None
    # A robosuite ``states`` array is [time, qpos, qvel, actuator-state, ...].
    # Accept it as a convenient full-scene q source when --q-key states is
    # selected, while preserving the recorded qvel for dynamics calculations.
    is_flat_state = args.q_format == "state" or (
        args.q_format == "auto" and demo.q_key.rsplit("/", 1)[-1] == "states" and q.shape[1] > model.nq
    )
    if is_flat_state:
        q, state_dq, state_time = _unpack_flattened_state(model, q)
    mapping = _make_joint_map(model, q.shape[1], args.joint_names)
    initial_qpos = np.asarray(data.qpos, dtype=np.float64).copy()
    q_full = _expand_q_to_full(model, q, mapping, initial_qpos)
    dq_full = _differentiate_qpos(model, q_full, args.dt)
    if state_dq is not None:
        dq_full = state_dq
    elif demo.dq is not None:
        dq_full = _map_optional_dq(model, demo.dq, mapping, args.dt)
    qacc_full = _finite_difference(dq_full, args.dt)

    tau_inverse = np.zeros((q_full.shape[0], int(model.nv)), dtype=np.float64)
    tau_actuator = np.zeros_like(tau_inverse)
    qvel_out = np.zeros_like(dq_full)
    qacc_out = np.zeros_like(qacc_full)
    frames: List[np.ndarray] = []
    render_camera = _camera_argument(model, args.camera)
    viewer = handle.viewer

    for t in range(q_full.shape[0]):
        if viewer is not None and not viewer.is_running():
            raise ViewerClosed("The MuJoCo viewer was closed by the user.")
        lock = viewer.lock() if viewer is not None else nullcontext()
        with lock:
            data.qpos[:] = q_full[t]
            data.qvel[:] = dq_full[t]
            data.qacc[:] = qacc_full[t]
            if state_time is not None:
                data.time = state_time[t]
            data.qfrc_applied[:] = 0
            data.qfrc_actuator[:] = 0
            data.ctrl[:] = 0
            if demo.ctrl is not None:
                if demo.ctrl.shape[1] != int(model.nu):
                    raise ValueError(f"Control data has D={demo.ctrl.shape[1]}, but model.nu={model.nu}.")
                data.ctrl[:] = _clip_ctrl(model, demo.ctrl[t])
            mujoco.mj_forward(model, data)
            qvel_out[t] = data.qvel
            qacc_out[t] = qacc_full[t]
            tau_actuator[t] = data.qfrc_actuator
            # mj_forward computes the acceleration resulting from the current
            # controls and overwrites data.qacc. Restore the trajectory
            # acceleration before inverse dynamics so qfrc_inverse corresponds
            # to the requested q/dq path rather than a zero-control step.
            data.qacc[:] = qacc_full[t]
            mujoco.mj_inverse(model, data)
            tau_inverse[t] = data.qfrc_inverse

        if viewer is not None:
            viewer.sync()
        if (viewer is not None or getattr(args, "realtime", False)) and t < q_full.shape[0] - 1:
            # Wall-clock pacing is independent from the numerical sample
            # period used for dq/tau. Viewer mode is paced automatically;
            # --realtime opts into the same pacing for headless runs.
            time.sleep(args.dt / args.viewer_speed)

        should_render = args.video_path is not None and (t % args.render_every == 0 or t == q_full.shape[0] - 1)
        if should_render:
            render_lock = viewer.lock() if viewer is not None else nullcontext()
            with render_lock:
                if handle.sim is not None:
                    frame = handle.sim.render(width=args.render_width, height=args.render_height, camera_name=render_camera)
                    frame = np.flipud(np.asarray(frame))
                else:
                    if handle.renderer is None:
                        handle.renderer = mujoco.Renderer(model, height=args.render_height, width=args.render_width)
                    handle.renderer.update_scene(data, camera=render_camera)
                    frame = handle.renderer.render()
            frames.append(np.asarray(frame, dtype=np.uint8).copy())

    if mapping.full_qpos:
        q_out = q
        dq_out = dq_full
        tau_out = tau_inverse if args.tau_source == "inverse" else tau_actuator
    else:
        q_out = q
        dq_out = dq_full[:, mapping.dof_indices]
        tau_out = (tau_inverse if args.tau_source == "inverse" else tau_actuator)[:, mapping.dof_indices]
    return {
        "q": np.asarray(q_out),
        "dq": np.asarray(dq_out),
        "tau": np.asarray(tau_out),
        "time": np.arange(q_full.shape[0], dtype=np.float64) * args.dt,
        "qpos_full": q_full,
        "qvel_full": qvel_out,
        "qacc_full": qacc_out,
        "tau_inverse_full": tau_inverse,
        "tau_actuator_full": tau_actuator,
        "joint_names": mapping.names if not mapping.full_qpos else [f"qpos_{i}" for i in range(model.nq)],
        "frames": frames,
    }


def _write_video(frames: Sequence[np.ndarray], path: Path, fps: float, show: bool) -> None:
    if not frames:
        return
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    first = np.asarray(frames[0])
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    try:
        for frame in frames:
            bgr = np.asarray(frame)[:, :, ::-1]
            writer.write(bgr)
            if show:
                cv2.imshow("MuJoCo q replay", bgr)
                cv2.waitKey(max(1, int(1000.0 / fps)))
    finally:
        writer.release()
        if show:
            cv2.destroyWindow("MuJoCo q replay")


def _write_plot(result: Dict[str, Any], path: Path, args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    time = result["time"]
    labels = result["joint_names"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    for values, axis, title in ((result["q"], axes[0], "q (rad / m)"),
                                (result["dq"], axes[1], "dq (rad/s / m/s)"),
                                (result["tau"], axes[2], f"tau ({args.tau_source})")):
        values = np.asarray(values)
        for i in range(values.shape[1]):
            label = labels[i] if i < len(labels) else f"joint_{i}"
            axis.plot(time, values[:, i], label=label)
        axis.set_ylabel(title)
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("time (s)")
    axes[0].legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_result(path: Path, demo_name: str, result: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    if path.suffix.lower() in (".hdf5", ".h5"):
        h5py = _require_h5py()
        mode = "a" if path.exists() else "w"
        with h5py.File(path, mode) as file:
            root = file.require_group("data")
            if demo_name in root:
                del root[demo_name]
            group = root.create_group(demo_name)
            for key in ("q", "dq", "tau", "time", "qpos_full", "qvel_full", "qacc_full",
                        "tau_inverse_full", "tau_actuator_full"):
                group.create_dataset(key, data=np.asarray(result[key]), compression="gzip")
            group.attrs["joint_names"] = np.asarray(result["joint_names"], dtype="S")
            for key, value in metadata.items():
                if value is not None:
                    group.attrs[key] = str(value)
    elif path.suffix.lower() == ".npz":
        arrays = {key: np.asarray(result[key]) for key in (
            "q", "dq", "tau", "time", "qpos_full", "qvel_full", "qacc_full",
            "tau_inverse_full", "tau_actuator_full")}
        np.savez_compressed(path, **arrays)
    else:
        raise ValueError(f"Output must end in .hdf5, .h5 or .npz, got {path}")


def _copy_hdf5_dataset(source: Path, destination: Path) -> None:
    """Copy an input HDF5 byte-for-byte at the group/dataset level."""

    h5py = _require_h5py()
    with h5py.File(source, "r") as source_file, h5py.File(destination, "w") as destination_file:
        for key, value in source_file.attrs.items():
            destination_file.attrs[key] = value
        for key in source_file.keys():
            source_file.copy(key, destination_file, name=key)


def _add_q_fields_to_hdf5(path: Path, demo_name: str, result: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    """Add exactly the two requested fields while preserving all source data."""

    h5py = _require_h5py()
    with h5py.File(path, "a") as file:
        if "data" not in file or demo_name not in file["data"]:
            raise KeyError(f"Cannot add fields: data/{demo_name} is missing from {path}")
        group = file["data"][demo_name]
        obs = group.require_group("obs")
        for key in ("dq", "tau"):
            if key in obs:
                del obs[key]
        obs.create_dataset("dq", data=np.asarray(result["dq"], dtype=np.float64), compression="gzip")
        obs.create_dataset("tau", data=np.asarray(result["tau"], dtype=np.float64), compression="gzip")
        obs["dq"].attrs["description"] = "joint velocity from MuJoCo q trajectory"
        obs["tau"].attrs["description"] = f"joint torque from MuJoCo {metadata['tau_source']}"
        obs["tau"].attrs["dt"] = float(metadata["dt"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, type=Path, help="Input HDF5/NPZ/NPY q dataset.")
    parser.add_argument("--demo", default="demo_0", help="HDF5 demo name/index, or 'all'.")
    parser.add_argument("--demo-range", nargs=2, type=int, metavar=("START", "END"), default=None,
                        help="Inclusive demo index range, e.g. --demo-range 10 19.")
    parser.add_argument("--q-key", default=None, help="Dataset key containing q, e.g. obs/joint_states or qpos.")
    parser.add_argument("--dq-key", default=None, help="Optional key containing dq/qvel.")
    parser.add_argument("--ctrl-key", default=None, help="Optional HDF5/NPZ key containing MuJoCo ctrl (T,nu).")
    parser.add_argument("--q-format", choices=("auto", "qpos", "state"), default="auto",
                        help="Interpret q-key states as [time, qpos, qvel, ...] and extract qpos/qvel correctly.")
    parser.add_argument("--bddl", default=None, help="BDDL path; otherwise read it from HDF5 metadata.")
    parser.add_argument("--xml", default=None, help="MuJoCo XML path (for NPZ/NPY or standalone XML replay).")
    parser.add_argument("--scene-source", choices=("dataset", "bddl"), default="bddl",
                        help="Scene source: BDDL reconstruction (default) or stored HDF5 model_file.")
    parser.add_argument("--joint-names", default=None, help="Comma-separated MuJoCo joint names in q column order.")
    parser.add_argument("--dt", type=float, default=None,
                        help="Trajectory sample period in seconds (default: 1 / HDF5 control_freq, usually 0.05).")
    parser.add_argument("--tau-source", choices=("inverse", "actuator"), default="inverse")
    parser.add_argument("--viewer", action="store_true",
                        help="Open the interactive MuJoCo viewer and replay in real time.")
    parser.add_argument("--realtime", action="store_true",
                        help="Pace a headless replay at the q control period (default: offline, no sleep).")
    parser.add_argument("--viewer-speed", type=float, default=1.0,
                        help="Viewer playback speed multiplier; 2 means twice real time.")
    parser.add_argument("--viewer-hold", action="store_true",
                        help="Hold the final frame until the viewer window is closed.")
    parser.add_argument("--show-collision", action="store_true",
                        help="Show collision geoms in the viewer for debugging (hidden by default).")
    parser.add_argument("--camera", default="agentview", help="MuJoCo camera used for the replay video.")
    parser.add_argument("--render-width", type=int, default=512)
    parser.add_argument("--render-height", type=int, default=512)
    parser.add_argument("--render-every", type=int, default=1, help="Render every Nth q sample.")
    parser.add_argument("--output", type=Path, default=None, help="Output .hdf5/.npz path.")
    parser.add_argument("--wm-data-dir", type=Path, default=None,
                        help="Directory for generated data. HDF5 default: <dataset-root>/wm_data.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite an existing output file. Without this flag, existing output is an error.")
    parser.add_argument("--video", type=Path, default=None, help="Optional replay video path; disabled by default.")
    parser.add_argument("--plot", type=Path, default=None, help="Optional q/dq/tau plot path; disabled by default.")
    parser.add_argument("--no-video", action="store_true", help="Skip video rendering.")
    parser.add_argument("--no-plot", action="store_true", help="Skip q/dq/tau plot.")
    parser.add_argument("--show", action="store_true", help="Show the replay in an OpenCV window while saving it.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    # Fail before copying the source HDF5 when the active environment is not
    # the one where MuJoCo was installed.
    _require_mujoco()
    args.dataset = args.dataset.expanduser().resolve()
    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)
    if args.render_every < 1:
        raise ValueError("--render-every must be >= 1")
    if args.viewer_speed <= 0:
        raise ValueError("--viewer-speed must be positive")
    if args.output is None:
        if args.wm_data_dir is not None:
            output_dir = args.wm_data_dir.expanduser().resolve()
        else:
            # For dataset/libero_90/file.hdf5, this resolves to dataset/wm_data.
            output_dir = args.dataset.parent.parent / "wm_data"
        args.output = output_dir / (args.dataset.stem + "_with_dq_tau.hdf5")
    args.output = args.output.expanduser().resolve()
    if args.output == args.dataset:
        raise ValueError("--output must differ from --dataset")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {args.output}. Pass --overwrite to replace it."
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    args.video_path = None if args.no_video or args.video is None else args.video.expanduser().resolve()
    args.plot_path = None if args.no_plot or args.plot is None else args.plot.expanduser().resolve()

    if args.demo_range is not None and args.demo.lower() not in ("demo_0", "0", "all"):
        raise ValueError("Use either --demo or --demo-range, not both.")
    demos = load_demos(
        args.dataset,
        args.demo,
        args.q_key,
        args.dq_key,
        args.ctrl_key,
        demo_range=tuple(args.demo_range) if args.demo_range is not None else None,
    )
    if args.dt is None:
        control_freq = demos[0][1].env_kwargs.get("control_freq", 20.0) if demos else 20.0
        args.dt = 1.0 / float(control_freq)
    if args.dt <= 0:
        raise ValueError(f"--dt must be positive, got {args.dt}")
    is_hdf5 = args.dataset.suffix.lower() in (".hdf5", ".h5")
    if is_hdf5 and args.xml is None and args.scene_source == "bddl":
        _require_libero_runtime()
    if is_hdf5:
        _copy_hdf5_dataset(args.dataset, args.output)
        print(f"Copied original dataset to {args.output}")
    for index, (demo_name, demo) in enumerate(demos):
        print(f"[{demo_name}] q={demo.q.shape}, q_key={demo.q_key}, dq_key={demo.dq_key}")
        if args.tau_source == "actuator" and demo.ctrl is None:
            print("[Warning] tau-source=actuator but no ctrl dataset was found; "
                  "the recorded actuator torque comes from zero controls and model actuator biases.")
        handle = _make_sim(demo, args.dataset, args, REPO_ROOT)
        try:
            if args.viewer:
                _launch_viewer(handle, show_collision=args.show_collision)
            result = replay_demo(handle, demo, args)
            metadata = {
                "source_dataset": str(args.dataset),
                "source_demo": demo_name,
                "q_key": demo.q_key,
                "dq_key": demo.dq_key,
                "tau_source": args.tau_source,
                "dt": args.dt,
                "q_format": args.q_format,
            }
            output_path = args.output
            if is_hdf5:
                _add_q_fields_to_hdf5(output_path, demo_name, result, metadata)
            else:
                if len(demos) > 1 and index > 0 and output_path.suffix.lower() == ".npz":
                    output_path = output_path.with_name(f"{output_path.stem}_{demo_name}.npz")
                _save_result(output_path, demo_name, result, metadata)

            if args.plot_path is not None:
                plot_output = args.plot_path if len(demos) == 1 else args.plot_path.with_name(
                    f"{args.plot_path.stem}_{demo_name}{args.plot_path.suffix}")
                _write_plot(result, plot_output, args)
            if args.video_path is not None:
                video_output = args.video_path if len(demos) == 1 else args.video_path.with_name(
                    f"{args.video_path.stem}_{demo_name}{args.video_path.suffix}")
                _write_video(result["frames"], video_output, 1.0 / args.dt / args.render_every, args.show)
            if args.viewer and args.viewer_hold and handle.viewer is not None:
                print(f"[{demo_name}] replay finished; close the MuJoCo viewer to continue.")
                while handle.viewer.is_running():
                    time.sleep(0.1)
            print(f"[{demo_name}] wrote {output_path}")
        except ViewerClosed:
            print(f"[{demo_name}] viewer closed; stopping before writing this demo.")
            break
        finally:
            handle.close()


if __name__ == "__main__":
    main()
