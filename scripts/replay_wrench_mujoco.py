#!/usr/bin/env python3
"""Inspect recorded MuJoCo states with a wrench-controlled time cursor.

Run from LIBERO-FT (or use an absolute script path)::

    python scripts/replay_wrench_mujoco.py --dataset dataset/wm_data/task.hdf5

Drag the plot's time slider or click/drag either curve panel to change the
MuJoCo frame. Time advances only in response to user input. By default
``states`` restores the whole recorded scene and ``obs/wrenches`` supplies
Fx,Fy,Fz,Tx,Ty,Tz. The HDF5 is opened read-only.
"""

from __future__ import annotations

import argparse
import math
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.replay_q_mujoco import (
    DatasetDemo,
    _expand_q_to_full,
    _load_hdf5_demo,
    _make_joint_map,
    _make_sim,
    _natural_demo_keys,
    _require_h5py,
    _require_mujoco,
    _unpack_flattened_state,
)
from scripts.wrench_replay_plot import WrenchReplayPlot


@dataclass
class ReplayEpisode:
    name: str
    demo: DatasetDemo
    times: np.ndarray
    wrench: np.ndarray
    fps: float


def load_episode(dataset: Path, name: str, *, q_key: str = "states",
                 wrench_key: str = "obs/wrenches", time_key: Optional[str] = None,
                 fps: Optional[float] = None) -> ReplayEpisode:
    """Load aligned low-dimensional arrays, without decoding recorded images."""
    demo = _load_hdf5_demo(dataset, name, q_key, None, None)
    h5py = _require_h5py()
    with h5py.File(dataset, "r") as source:
        group = source[f"data/{name}"]
        if wrench_key not in group:
            raise KeyError(f"data/{name}/{wrench_key} is missing; select it with --wrench-key")
        wrench = np.asarray(group[wrench_key], dtype=np.float64)
        if wrench.shape != (len(demo.q), 6):
            raise ValueError(
                f"data/{name}/{wrench_key} must be [T,6] aligned with {q_key}; "
                f"got {wrench.shape} for {len(demo.q)} states"
            )
        if fps is None:
            fps = demo.env_kwargs.get("control_freq")
            if fps is None:
                fps = source["data"].attrs.get("fps", source.attrs.get("fps", 20.0))
        fps = float(fps)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps must be positive and finite")
        if time_key is None:
            times = np.arange(len(wrench), dtype=np.float64) / fps
        else:
            times = np.asarray(group[time_key], dtype=np.float64).reshape(-1)
            if times.shape != (len(wrench),):
                raise ValueError(f"{time_key} must have one timestamp per state")
            times = times - times[0] if len(times) else times
    if not len(wrench):
        raise ValueError(f"{name} contains no frames")
    if not all(np.isfinite(values).all() for values in (demo.q, wrench, times)):
        raise ValueError(f"{name} contains non-finite state, timestamp or wrench values")
    if np.any(np.diff(times) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    return ReplayEpisode(name, demo, times, wrench, fps)


def frame_at_time(times, seconds):
    """Select the closest recorded frame, clamped to the episode bounds."""
    right = int(np.clip(np.searchsorted(times, seconds), 0, len(times) - 1))
    left = max(0, right - 1)
    return left if abs(seconds - times[left]) <= abs(times[right] - seconds) else right


def _select_demos(dataset, demo, demo_range):
    h5py = _require_h5py()
    with h5py.File(dataset, "r") as source:
        if "data" not in source:
            raise ValueError("expected an HDF5 data/ group containing demo_* episodes")
        names = _natural_demo_keys(name for name in source["data"]
                                   if name.startswith("demo_"))
    if demo_range is not None:
        start, stop = demo_range
        if not 0 <= start <= stop:
            raise ValueError("--demo-range requires 0 <= START <= END (inclusive)")
        names = [name for name in names if start <= int(name.rsplit("_", 1)[1]) <= stop]
    elif demo != "all":
        selected = demo if demo.startswith("demo_") else f"demo_{demo}"
        if selected not in names:
            raise KeyError(f"{selected} is missing; available demos: {names}")
        names = [selected]
    if not names:
        raise ValueError("no episodes selected")
    return names


def _scene_trajectory(handle, episode, args):
    q = episode.demo.q
    is_state = args.q_format == "state" or (
        args.q_format == "auto" and episode.demo.q_key.rsplit("/", 1)[-1] == "states"
    )
    if is_state:
        qpos, qvel, sim_times = _unpack_flattened_state(handle.model, q)
        return qpos, qvel, sim_times
    mapping = _make_joint_map(handle.model, q.shape[1], args.joint_names)
    qpos = _expand_q_to_full(handle.model, q, mapping, handle.data.qpos.copy())
    return qpos, np.zeros((len(q), handle.model.nv)), None


def _apply_frame(handle, qpos, qvel, sim_times, episode, index):
    mujoco = _require_mujoco()
    handle.data.qpos[:] = qpos[index]
    handle.data.qvel[:] = qvel[index]
    handle.data.time = episode.times[index] if sim_times is None else sim_times[index]
    mujoco.mj_forward(handle.model, handle.data)


def replay_episode(episode, dataset, args):
    """Return next/previous/quit after interactive cursor inspection."""
    handle = _make_sim(episode.demo, dataset, args, REPO_ROOT)
    plot = None
    viewer_threads = []
    try:
        qpos, qvel, sim_times = _scene_trajectory(handle, episode, args)
        if args.headless:
            count = len(qpos) if args.max_frames is None else min(len(qpos), args.max_frames)
            for index in range(count):
                _apply_frame(handle, qpos, qvel, sim_times, episode, index)
            return "next", count

        # MuJoCo invokes key callbacks on its viewer thread; queue commands so
        # all frame selection and plotting stay on the main thread.
        events = queue.SimpleQueue()

        def viewer_key(code):
            names = {262: "right", 263: "left", 268: "home", 269: "end",
                     256: "escape", 82: "r", 78: "n", 80: "p", 81: "q"}
            if code in names:
                events.put(("key", names[code]))

        plot = WrenchReplayPlot(
            episode.times, episode.wrench,
            title=f"Recorded wrench | {episode.name} | {dataset.name}",
            on_seek=lambda seconds: events.put(("seek", seconds)),
            on_key=lambda key: events.put(("key", key)),
            backend=args.plot_backend,
        )
        import mujoco.viewer

        existing_threads = set(threading.enumerate())
        handle.viewer = mujoco.viewer.launch_passive(handle.model, handle.data,
                                                    key_callback=viewer_key)
        viewer_threads = [thread for thread in threading.enumerate()
                          if thread not in existing_threads]
        if not args.show_collision:
            with handle.viewer.lock():
                handle.viewer.opt.geomgroup[0] = 0
                handle.viewer.opt.geomgroup[1] = 1
        index = 0
        last_index = -1
        displayed = 0
        while handle.viewer.is_running() and plot.is_open:
            while not events.empty():
                kind, value = events.get_nowait()
                if kind == "seek":
                    index = frame_at_time(episode.times, value)
                    continue
                key = str(value).lower()
                if key in ("q", "escape"):
                    return "quit", displayed
                if key == "n":
                    return "next", displayed
                if key == "p":
                    return "previous", displayed
                if key in ("r", "home"):
                    index = 0
                elif key == "end":
                    index = len(episode.times) - 1
                elif key in ("left", "right"):
                    index = int(np.clip(index + (-1 if key == "left" else 1),
                                        0, len(episode.times) - 1))

            if index != last_index:
                with handle.viewer.lock():
                    _apply_frame(handle, qpos, qvel, sim_times, episode, index)
                plot.update(index, paused=True)
                last_index = index
                displayed += 1
            handle.viewer.sync()
            plot.pump()
            if args.max_frames is not None and displayed >= args.max_frames:
                return "quit", displayed
            time.sleep(0.005)
        return "quit", displayed
    finally:
        if handle.viewer is not None:
            handle.viewer.close()
            # close() only signals MuJoCo's daemon render thread. Let it release
            # GL resources before Tk or interpreter shutdown tears down X11.
            for thread in viewer_threads:
                thread.join(timeout=5.0)
            handle.viewer = None
        if plot is not None:
            plot.close()
        handle.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True, help="Input LIBERO HDF5 file (read-only).")
    parser.add_argument("--demo", default="demo_0", help="Zero-based demo name/index, or all.")
    parser.add_argument("--demo-range", nargs=2, type=int, metavar=("START", "END"),
                        help="Inclusive, zero-based demo range.")
    parser.add_argument("--wrench-key", default="obs/wrenches", help="Recorded [T,6] force/torque field.")
    parser.add_argument("--q-key", default="states", help="Default states restores the full scene.")
    parser.add_argument("--q-format", choices=("auto", "state", "qpos"), default="auto")
    parser.add_argument("--time-key", help="Optional per-demo timestamp field in seconds.")
    parser.add_argument("--fps", type=float, help="Override dataset control_freq (fallback: 20 Hz).")
    parser.add_argument("--plot-backend", default="TkAgg", help="Interactive Matplotlib backend; default TkAgg.")
    parser.add_argument("--show-collision", action="store_true")
    parser.add_argument("--headless", action="store_true", help="Validate replay without windows or frame delays.")
    parser.add_argument("--max-frames", type=int, help="Stop after this many displayed frames per demo.")
    parser.add_argument("--xml", type=Path, help="Optional replacement MuJoCo XML scene.")
    parser.add_argument("--scene-source", choices=("dataset", "bddl"), default="dataset")
    parser.add_argument("--bddl", help="Optional LIBERO BDDL path for scene reconstruction.")
    parser.add_argument("--joint-names", help="Comma-separated names when replaying joint positions.")
    # Used only when the existing scene builder reconstructs a BDDL scene.
    parser.set_defaults(camera="agentview", render_height=480, render_width=640)
    args = parser.parse_args(argv)
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        parser.error("--fps must be positive and finite")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    dataset = args.dataset.expanduser().resolve()
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    names = _select_demos(dataset, args.demo, args.demo_range)
    if not args.headless:
        print("Manual cursor control: drag the time slider or click/drag the wrench plot. "
              "Left/Right: step; Home/End: first/last frame; N/P: episode; Q/Esc: quit.", flush=True)
    position = 0
    try:
        while position < len(names):
            episode = load_episode(dataset, names[position], q_key=args.q_key,
                                   wrench_key=args.wrench_key, time_key=args.time_key,
                                   fps=args.fps)
            print(f"{episode.name}: {len(episode.times)} frames, {episode.fps:g} Hz, "
                  f"{episode.times[-1]:.3f} s, wrench={args.wrench_key}", flush=True)
            action, frames = replay_episode(episode, dataset, args)
            print(f"{episode.name}: replayed {frames} frames", flush=True)
            if action == "quit":
                break
            position = max(0, position - 1) if action == "previous" else position + 1
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
