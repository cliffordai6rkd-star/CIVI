"""Replay recorded sensor frames without using future ground-truth wrench."""

import json
from pathlib import Path

import numpy as np
import yaml

from .inference_base import InferenceObservation


def sensor_frames(config, *, phase_source):
    import h5py

    replay = config["replay"]
    fields = replay["fields"]
    start = int(replay["start_frame"])
    count = replay.get("max_frames")
    if start < 0 or (count is not None and int(count) < 1):
        raise ValueError("replay.start_frame must be nonnegative; max_frames must be positive or null")
    columns = np.asarray(fields["joint_columns"], dtype=np.int64)
    if columns.shape != (7,) or len(set(columns)) != 7 or (columns < 0).any():
        raise ValueError("replay.fields.joint_columns must specify seven distinct joint columns")
    with h5py.File(replay["dataset"], "r") as file:
        demo = file[f"data/{replay['demo']}"]
        length = len(demo[fields["q"]])
        if start >= length:
            raise ValueError(f"replay.start_frame={start} is outside this {length}-frame demo")
        stop = length if count is None else min(length, start + int(count))
        required = [fields[key] for key in ("q", "dq", "tau", "timestamp")]
        required += list(fields["cameras"].values())
        if phase_source == "measured":
            required.append(fields["wrench"])
        for key in required:
            if len(demo[key]) != length:
                raise ValueError(f"Sensor field {key} does not share the native episode timeline")
        for row in range(start, stop):
            timestamp = np.asarray(demo[fields["timestamp"]][row]).reshape(-1)[0]
            # Explicit mapping excludes extra MuJoCo object/gripper DoFs.
            state = {key: np.asarray(demo[fields[key]][row])[columns] for key in ("q", "dq", "tau")}
            yield row, InferenceObservation(
                timestamp_s=float(timestamp),
                images={camera: demo[key][row] for camera, key in fields["cameras"].items()},
                wrench=demo[fields["wrench"]][row] if phase_source == "measured" else None,
                **state,
            )


def run_replay(engine, config):
    output = Path(config["replay"]["output"])
    if output.exists():
        raise FileExistsError(f"Inference output exists; choose a new replay.output: {output}")
    stride = int(config["replay"]["predict_every"])
    if stride < 1:
        raise ValueError("replay.predict_every must be positive")
    records = []
    frames_seen = 0
    engine.reset()
    for frame, observation in sensor_frames(config, phase_source=engine.phase_wrench_source):
        ready = engine.observe(observation) # WM sees every native-rate frame.
        frames_seen += 1
        if not ready or (frames_seen - 1) % stride:
            continue
        result = engine.predict()
        records.append((frame, result))
        if len(records) == 1 or len(records) % 20 == 0:
            print(f"frame={frame} t={observation.timestamp_s:.3f}s predictions={len(records)}", flush=True)
    if not records:
        raise RuntimeError("No pose predictions produced; include enough frames for WM and image warmup")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against a concurrent replay overwriting results.
    with output.open("xb") as stream:
        np.savez_compressed(
            stream, frame_index=np.array([frame for frame, _ in records]),
            observation_timestamp_s=np.array([result.observation_timestamp_s for _, result in records]),
            actions=np.stack([result.actions for _, result in records]),
            action_timestamps_s=np.stack([result.action_timestamps_s for _, result in records]),
            future_wrench=np.stack([result.info["future_wrench"] for _, result in records]),
            future_wrench_offsets_s=engine.future_offsets_s,
            action_format=engine.spec.action_format, wrench_source="cad_wm",
        )
    manifest = {"output": str(output), "sensor_frames": frames_seen, "prediction_chunks": len(records),
                "action_format": engine.spec.action_format, "action_shape": list(engine.spec.action_shape),
                "wrench_source": "cad_wm", "phase_wrench_source": engine.phase_wrench_source,
                "checkpoint_info": engine.checkpoint_info,
                "mode": "recorded-input replay; no simulator or robot actions executed"}
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    output.with_suffix(".config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return manifest
