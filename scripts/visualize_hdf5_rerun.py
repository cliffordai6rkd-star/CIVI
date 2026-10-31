#!/usr/bin/env python3
"""Visualize LIBERO-FT HDF5 demos in Rerun.

The default LIBERO-FT layout is:

    data/<demo>/obs/agentview_rgb
    data/<demo>/obs/eye_in_hand_rgb
    data/<demo>/obs/wrenches       # Fx, Fy, Fz, Mx, My, Mz

Examples::

    python scripts/visualize_hdf5_rerun.py \
        --dataset dataset/libero_90/task_demo_wrench.hdf5 \
        --demo 0

    python scripts/visualize_hdf5_rerun.py \
        --dataset dataset/libero_90/task_demo_wrench.hdf5 \
        --demo 0 --save /tmp/libero_demo_0.rrd

Rerun receives one image from each camera and six scalar time series for each
control tick. The HDF5 control frequency is used for the time axis, so this
script does not resample or modify the recorded wrench values.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


WRENCH_NAMES = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
WRENCH_COLORS = (
    (220, 60, 60),
    (60, 150, 60),
    (60, 100, 220),
    (220, 140, 40),
    (170, 70, 190),
    (40, 170, 170),
)


def _natural_demo_keys(keys: Iterable[str]) -> List[str]:
    def key(value: str):
        match = re.search(r"(\d+)$", value)
        return (0, int(match.group(1))) if match else (1, value)

    return sorted((str(item) for item in keys), key=key)


def _demo_name(value: str | int) -> str:
    value = str(value)
    return value if value.startswith("demo_") else f"demo_{value}"


def _read_nested(group: h5py.Group, key: str) -> np.ndarray:
    obj: Any = group
    for part in str(key).strip("/").split("/"):
        if part:
            obj = obj[part]
    return np.asarray(obj[()])


def _first_existing(group: h5py.Group, keys: Sequence[str]) -> Tuple[np.ndarray, str]:
    for key in keys:
        try:
            return _read_nested(group, key), key
        except (KeyError, ValueError):
            continue
    raise KeyError(f"None of these HDF5 keys exists: {', '.join(keys)}")


def _normalise_image(image: np.ndarray, flip_vertical: bool) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected an image with shape (H,W,C), got {image.shape}")
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating) and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    if flip_vertical:
        image = np.flipud(image)
    return np.ascontiguousarray(image)


def _control_frequency(data_group: h5py.Group, demo_group: h5py.Group, override: Optional[float]) -> float:
    if override is not None:
        frequency = float(override)
    else:
        frequency = None
        raw = data_group.attrs.get("env_args")
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if raw is not None:
            try:
                env_args = json.loads(str(raw))
                frequency = env_args.get("env_kwargs", {}).get("control_freq")
            except (TypeError, ValueError, json.JSONDecodeError):
                frequency = None
        if frequency is None:
            frequency = demo_group.attrs.get("control_freq", 20.0)
        frequency = float(frequency)
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError(f"Control frequency must be positive, got {frequency}")
    return frequency


def _state_times(demo_group: h5py.Group, length: int, frequency: float) -> np.ndarray:
    """Use recorded state times when available, otherwise control ticks."""

    if "states" in demo_group:
        states = np.asarray(demo_group["states"])
        if states.ndim == 2 and states.shape[0] >= length and states.shape[1] > 0:
            values = np.asarray(states[:length, 0], dtype=np.float64)
            if np.all(np.isfinite(values)) and np.all(np.diff(values) >= 0):
                return values - values[0]
    return np.arange(length, dtype=np.float64) / frequency


def _select_demos(file: h5py.File, demo: str, demo_range: Optional[Tuple[int, int]]) -> List[str]:
    if "data" not in file:
        raise ValueError("The HDF5 file must contain a data/ group.")
    names = _natural_demo_keys(file["data"].keys())
    if demo_range is not None:
        start, end = demo_range
        if start < 0 or end < start:
            raise ValueError("Expected 0 <= START <= END for --demo-range.")
        available = names
        names = [
            name for name in names
            if (match := re.search(r"(\d+)$", name)) is not None
            and start <= int(match.group(1)) <= end
        ]
        if not names:
            raise ValueError(f"No demos selected by range {demo_range}; available: {available}")
        return names
    if str(demo).lower() == "all":
        return names
    selected = _demo_name(demo)
    if selected not in file["data"]:
        raise KeyError(f"{selected} not found. Available demos: {names}")
    return [selected]


def _set_time(rr: Any, timeline: str, sequence: Optional[int] = None,
              seconds: Optional[float] = None) -> None:
    """Use the current Rerun time API, with compatibility for older SDKs."""

    if hasattr(rr, "set_time"):
        kwargs = {"sequence": sequence} if sequence is not None else {"duration": seconds}
        rr.set_time(timeline, **kwargs)
    elif sequence is not None:
        rr.set_time_sequence(timeline, sequence)
    else:
        rr.set_time_seconds(timeline, float(seconds))


def _log_scalar(rr: Any, path: str, value: float) -> None:
    """Log a scalar across Rerun SDK versions."""

    scalars = getattr(rr, "Scalars", None)
    if scalars is None:
        scalars = getattr(getattr(rr, "archetypes", None), "Scalars", None)
    if scalars is not None:
        rr.log(path, scalars(float(value)))
        return
    scalar = getattr(rr, "Scalar", None)
    if scalar is None:
        scalar = getattr(getattr(rr, "components", None), "Scalar", None)
    if scalar is None:
        raise RuntimeError("The installed rerun-sdk does not expose a scalar logging API.")
    rr.log(path, scalar(float(value)))


def _log_series_style(rr: Any, path: str, name: str, color: Sequence[int]) -> None:
    """Set line colors/names when supported by the installed Rerun SDK."""

    series_lines = getattr(rr, "SeriesLines", None)
    if series_lines is None:
        series_lines = getattr(getattr(rr, "archetypes", None), "SeriesLines", None)
    if series_lines is not None:
        try:
            rr.log(path, series_lines(colors=color, names=name), static=True)
            return
        except (AttributeError, TypeError, ValueError):
            pass
    series_line = getattr(rr, "SeriesLine", None)
    if series_line is not None:
        try:
            rr.log(path, series_line(name=name, color=color), static=True)
        except (AttributeError, TypeError, ValueError):
            pass


def _log_static_metadata(rr: Any, prefix: str, dataset: Path, demo_name: str,
                         frequency: float, length: int, camera_keys: Sequence[str],
                         wrench_key: str) -> None:
    rr.log(f"{prefix}/metadata", rr.TextDocument(
        "\n".join((
            f"dataset: {dataset}",
            f"demo: {demo_name}",
            f"samples: {length}",
            f"control frequency: {frequency:g} Hz",
            f"camera A: {camera_keys[0]}",
            f"camera B: {camera_keys[1]}",
            f"wrench: {wrench_key}",
        )),
        media_type="text/plain",
    ))
    for name, color in zip(WRENCH_NAMES, WRENCH_COLORS):
        _log_series_style(
            rr,
            f"{prefix}/wrench/{'force' if name[0] == 'F' else 'torque'}/{name}",
            name,
            color,
        )


def _log_demo(rr: Any, demo_group: h5py.Group, demo_name: str, dataset: Path,
              args: argparse.Namespace) -> int:
    camera_a, camera_a_key = _first_existing(demo_group, [args.camera_a_key])
    camera_b, camera_b_key = _first_existing(demo_group, [args.camera_b_key])
    wrench, wrench_key = _first_existing(
        demo_group, [args.wrench_key, "obs/wrenches", "obs/wrench_ee", "wrenches"]
    )
    camera_a = np.asarray(camera_a)
    camera_b = np.asarray(camera_b)
    wrench = np.asarray(wrench, dtype=np.float64)
    if camera_a.ndim != 4 or camera_b.ndim != 4:
        raise ValueError(f"Camera datasets must have shape (T,H,W,C), got {camera_a.shape} and {camera_b.shape}")
    if wrench.ndim != 2 or wrench.shape[1] != 6:
        raise ValueError(f"Wrench dataset must have shape (T,6), got {wrench.shape}")
    length = min(camera_a.shape[0], camera_b.shape[0], wrench.shape[0])
    if length == 0:
        raise ValueError(f"{demo_name} contains no synchronized samples")
    if len({camera_a.shape[0], camera_b.shape[0], wrench.shape[0]}) != 1:
        print(
            f"[Warning] {demo_name} lengths differ: camera A={camera_a.shape[0]}, "
            f"camera B={camera_b.shape[0]}, wrench={wrench.shape[0]}; truncating to {length}.",
            file=sys.stderr,
        )
    data_group = demo_group.parent
    frequency = _control_frequency(data_group, demo_group, args.frequency)
    times = _state_times(demo_group, length, frequency)
    prefix = f"demo/{demo_name}"
    _log_static_metadata(
        rr, prefix, dataset, demo_name, frequency, length,
        (camera_a_key, camera_b_key), wrench_key,
    )
    print(f"[{demo_name}] samples={length}, frequency={frequency:g} Hz, dt={1.0 / frequency:g} s")

    for index in range(length):
        seconds = float(times[index])
        _set_time(rr, "frame", sequence=index)
        _set_time(rr, "control_time", seconds=seconds)
        rr.log(f"{prefix}/camera/agentview", rr.Image(_normalise_image(camera_a[index], args.flip_vertical)))
        rr.log(f"{prefix}/camera/eye_in_hand", rr.Image(_normalise_image(camera_b[index], args.flip_vertical)))
        for column, (name, color) in enumerate(zip(WRENCH_NAMES, WRENCH_COLORS)):
            path = f"{prefix}/wrench/{'force' if column < 3 else 'torque'}/{name}"
            try:
                    _log_scalar(rr, path, float(wrench[index, column]))
            except (ValueError, TypeError):
                # Keep NaNs from breaking a whole recording. Rerun's scalar
                # view can still show all valid samples around the gap.
                if np.isfinite(wrench[index, column]):
                    raise
        if args.realtime and index + 1 < length:
            time.sleep((seconds if index == 0 else float(times[index + 1] - times[index])) / args.speed)
    return length


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="LIBERO-FT HDF5 file")
    parser.add_argument("--demo", default="demo_0", help="Demo name/index, or all")
    parser.add_argument("--demo-range", nargs=2, type=int, metavar=("START", "END"), default=None)
    parser.add_argument("--camera-a-key", default="obs/agentview_rgb")
    parser.add_argument("--camera-b-key", default="obs/eye_in_hand_rgb")
    parser.add_argument("--wrench-key", default="obs/wrenches")
    parser.add_argument("--frequency", type=float, default=None,
                        help="Override HDF5 control frequency in Hz")
    parser.add_argument("--no-flip-vertical", dest="flip_vertical", action="store_false",
                        help="Do not vertically flip camera images before logging")
    parser.set_defaults(flip_vertical=True)
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Realtime playback speed multiplier")
    parser.add_argument("--realtime", action="store_true",
                        help="Sleep between samples while logging")
    parser.add_argument("--save", type=Path, default=None,
                        help="Optional .rrd recording path")
    parser.add_argument("--no-spawn", action="store_true",
                        help="Do not launch the Rerun viewer; useful with --save")
    parser.add_argument("--app-id", default="libero_ft_hdf5")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.speed <= 0:
        raise ValueError("--speed must be positive")
    dataset = args.dataset.expanduser().resolve()
    if not dataset.exists():
        raise FileNotFoundError(dataset)
    if args.save is not None:
        args.save = args.save.expanduser().resolve()
        args.save.parent.mkdir(parents=True, exist_ok=True)

    try:
        import rerun as rr
    except ImportError as exc:
        raise RuntimeError(
            "This script requires rerun-sdk. Install it with: "
            "python -m pip install rerun-sdk"
        ) from exc

    rr.init(args.app_id, spawn=not args.no_spawn)
    if args.save is not None:
        rr.save(str(args.save))

    with h5py.File(dataset, "r") as file:
        demos = _select_demos(file, args.demo, tuple(args.demo_range) if args.demo_range else None)
        for demo_name in demos:
            _log_demo(rr, file["data"][demo_name], demo_name, dataset, args)
    if hasattr(rr, "flush"):
        rr.flush()
    if args.save is not None:
        print(f"Saved Rerun recording to {args.save}")
    if args.no_spawn:
        print("Viewer was not spawned; open the .rrd file with `rerun --open <file.rrd>`." if args.save else "Finished without spawning a viewer.")


if __name__ == "__main__":
    main()
