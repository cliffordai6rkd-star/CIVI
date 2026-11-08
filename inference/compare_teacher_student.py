"""Offline teacher/student action comparison with a frozen FDP teacher.

RenderNet is an image renderer. The teacher fuses original RGB with wrench;
the student renders RGB with wrench, then uses the frozen image-only encoder
and DiT without running teacher wrench fusion. Both paths share the CAD-WM
forecast and diffusion noise. Recorded LIBERO-FT HDF5 replay measures offline
prediction agreement, not simulator or real-robot task success.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

from .config import DEFAULT_CONFIG, load_config
from .fdp_inference import FDPInference
from .replay import sensor_frames
from .render_student import predict_student_action


def _load_student(config, device):
    """Load RenderNet and its teacher-aligned wrench preprocessor."""
    checkpoint = Path(config["student"]["checkpoint"]).resolve()
    repo = Path(config["student"]["repo"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"RenderNet checkpoint does not exist: {checkpoint}")
    if not (repo / "diffusion_policy").is_dir():
        raise FileNotFoundError(f"DP repo not found for RenderNet: {repo}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import diffusion_policy
    if Path(diffusion_policy.__file__).resolve().parent != repo / "diffusion_policy":
        raise RuntimeError("A different diffusion_policy package was already imported")
    from diffusion_policy.scripts.visualize_render_net_episode import load_render_net

    model, preprocessor, cfg, teacher_cfg, render_device, info = load_render_net(
        checkpoint, device=str(device)
    )
    if render_device != torch.device(device):
        raise RuntimeError(f"RenderNet loaded on {render_device}, expected {device}")
    requested = config["student"].get("use_attention_mask")
    use_attention_mask = model.high_attention_enabled if requested is None else bool(requested)
    return model, preprocessor, cfg, teacher_cfg, use_attention_mask, info


def _action_array(result, label):
    actions = result["action"][0].detach().float().cpu().numpy().copy()
    if actions.ndim != 2 or actions.shape[-1] not in (7, 8) or not np.isfinite(actions).all():
        raise RuntimeError(f"{label} returned invalid action shape/values: {actions.shape}")
    if not np.allclose(np.linalg.norm(actions[:, 3:7], axis=-1), 1, atol=1e-3, rtol=0):
        raise RuntimeError(f"{label} returned a non-unit xyzw quaternion")
    return actions


@torch.inference_mode()
def predict_pair(policy, render_net, preprocessor, obs, generator, *,
                 amp_enabled=False, amp_dtype=torch.bfloat16, use_attention_mask=None):
    """Use one diffusion random stream step for a paired teacher/student call."""
    autocast = (torch.autocast("cuda", dtype=amp_dtype)
                if amp_enabled else contextlib.nullcontext())
    generator_state = generator.get_state()
    with autocast:
        teacher = policy.predict_action(obs, generator=generator)
        generator.set_state(generator_state)
        student = predict_student_action(
            policy, render_net, preprocessor, obs, generator=generator,
            use_attention_mask=use_attention_mask,
        )
    return _action_array(teacher, "Teacher"), _action_array(student, "Student")


@torch.inference_mode()
def run_compare(config):
    runtime = config["runtime"]
    device = torch.device(runtime["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable")

    # One policy instance is used for both passes. Restoring its generator
    # state makes paired action differences independent of diffusion noise.
    engine = FDPInference.from_config(config["_path"])
    model, preprocessor, render_cfg, teacher_cfg, use_attention_mask, render_info = _load_student(
        config, device
    )
    image_keys = tuple(engine.image_shapes)
    if set(image_keys) != set(getattr(model, "image_keys", image_keys)):
        raise ValueError(f"Teacher/RenderNet camera mismatch: {image_keys} vs {model.image_keys}")

    replay_cfg = config["replay"]
    output = Path(replay_cfg["output"])
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"Comparison output exists; choose a new replay.output: {output}")
    stride = int(replay_cfg["predict_every"])
    if stride < 1:
        raise ValueError("replay.predict_every must be positive")

    records = []
    frames_seen = 0
    engine.reset(seed=int(runtime["seed"]))
    amp_dtype = getattr(torch, runtime["amp_dtype"])
    amp_enabled = bool(runtime["amp_enabled"]) and device.type == "cuda"
    try:
        for frame, observation in sensor_frames(config, phase_source=engine.phase_wrench_source):
            ready = engine.observe(observation)
            frames_seen += 1
            if not ready or (frames_seen - 1) % stride:
                continue

            obs = engine._policy_observations()
            teacher_actions, student_actions = predict_pair(
                engine.policy, model, preprocessor, obs, engine.generator,
                amp_enabled=amp_enabled, amp_dtype=amp_dtype,
                use_attention_mask=use_attention_mask,
            )
            delta = student_actions - teacher_actions
            context = engine._contexts[-1]
            phase = np.asarray(context["phase"][-1], dtype=np.float32)
            future = np.asarray(context["future"], dtype=np.float32)
            records.append((frame, observation.timestamp_s, teacher_actions,
                            student_actions, delta, phase, future))
            if len(records) == 1 or len(records) % 20 == 0:
                print(f"frame={frame} t={observation.timestamp_s:.3f}s comparisons={len(records)}",
                      flush=True)
    finally:
        engine.close()

    if not records:
        raise RuntimeError("No paired predictions produced; include enough replay frames")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame_index = np.asarray([row[0] for row in records], dtype=np.int64)
    timestamp = np.asarray([row[1] for row in records], dtype=np.float64)
    teacher = np.stack([row[2] for row in records])
    student = np.stack([row[3] for row in records])
    delta = np.stack([row[4] for row in records])
    phase = np.stack([row[5] for row in records])
    future = np.stack([row[6] for row in records])
    delta_l2 = np.linalg.norm(delta, axis=-1)
    delta_pos = np.linalg.norm(delta[..., :3], axis=-1)
    delta_rot = np.linalg.norm(delta[..., 3:7], axis=-1)
    with output.open("xb") as stream:
        np.savez_compressed(
            stream, frame_index=frame_index, observation_timestamp_s=timestamp,
            teacher_actions=teacher, student_actions=student, action_delta=delta,
            action_delta_l2=delta_l2, action_delta_position_l2=delta_pos,
            action_delta_quaternion_l2=delta_rot, phase_wrench=phase,
            future_wrench=future, future_wrench_offsets_s=engine.future_offsets_s,
            action_format=engine.spec.action_format,
        )
    report = {
        "student_path_version": 2,
        "student_context_path": "rendered RGB -> encode_image_context -> DiT; no teacher wrench fusion",
        "output": str(output), "sensor_frames": frames_seen,
        "prediction_chunks": len(records), "action_shape": list(teacher.shape[1:]),
        "teacher_checkpoint": config["dp"]["checkpoint"],
        "teacher_state_key": engine.checkpoint_info.get("dp_state_key"),
        "student_checkpoint": config["student"]["checkpoint"],
        "cad_wm_checkpoint": config["cad_wm"]["checkpoint"],
        "student_rendering_mode": str(getattr(model, "rendering_mode", "unknown")),
        "student_high_attention_mask": bool(use_attention_mask),
        "action_delta_l2_mean": float(delta_l2.mean()),
        "action_delta_l2_median": float(np.median(delta_l2)),
        "action_delta_l2_p95": float(np.percentile(delta_l2, 95)),
        "action_delta_l2_max": float(delta_l2.max()),
        "action_delta_position_l2_mean": float(delta_pos.mean()),
        "action_delta_quaternion_l2_mean": float(delta_rot.mean()),
        "rendernet_global_step": render_info.get("global_step"),
        "rendernet_best_val_loss": render_info.get("best_val_loss"),
        "mode": "recorded-input offline teacher/student comparison; no simulator or robot actions executed",
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    output.with_suffix(".config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Comparison YAML")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config["_path"] = str(Path(args.config).resolve())
    report = run_compare(config)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
