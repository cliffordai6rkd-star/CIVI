"""Compare teacher fusion branches on the same recorded inputs and DDIM noise.

Run from LIBERO-FT with ``python -m inference.debug_teacher_contact``.
Only the observation encoder's contact gate changes between the three passes;
CAD-WM forecasts and the teacher's wrench preprocessing stay identical.
"""

import argparse
import contextlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from .config import DEFAULT_CONFIG, load_config
from .fdp_inference import FDPInference
from .replay import sensor_frames


def array(value):
    return value.detach().float().cpu().numpy().copy()


@torch.inference_mode()
def predict_branch(engine, obs, branch, action_index):
    captured = {}

    def select_gate(module, args, kwargs):
        gate = kwargs["contact_gate"]
        if branch != "normal":
            kwargs = dict(kwargs)
            kwargs["contact_gate"] = torch.full_like(gate, branch == "fused")
        captured["gate"] = kwargs["contact_gate"].detach().cpu().tolist()[0]
        captured["normalized_wrench"] = array(kwargs["wrench_history"])[0]
        return args, kwargs

    def save_context(module, args, output):
        captured["context"] = array(output)

    encoder = engine.policy.obs_encoder
    pre_hook = encoder.register_forward_pre_hook(select_gate, with_kwargs=True)
    post_hook = encoder.register_forward_hook(save_context)
    amp = (torch.autocast("cuda", dtype=engine.amp_dtype)
           if engine.amp_enabled else contextlib.nullcontext())
    try:
        # A fresh generator per pass also removes noise changes between frames.
        generator = torch.Generator(device=engine.device).manual_seed(engine.seed)
        with amp:
            result = engine.policy.predict_action(obs, generator=generator)
    finally:
        pre_hook.remove()
        post_hook.remove()
    actions = array(result["action"])[0]
    captured["actions"] = actions
    captured["position_m"] = actions[action_index, :3]
    return captured


@torch.inference_mode()
def diagnose(config, *, max_frames, sample_frames, output):
    config["replay"]["start_frame"] = 0
    config["replay"]["max_frames"] = max_frames
    config["runtime"]["print_wrench_chunks"] = False
    engine = FDPInference.from_loaded_config(config)
    policy = engine.policy
    if not policy.contact_fusion_gate:
        raise ValueError("This diagnostic requires the teacher's contact_fusion_gate")
    action_index = int(config["runtime"].get("execution_action_index", 1))
    rows, comparisons = [], []
    previous_raw = None
    try:
        for frame, observation in sensor_frames(config, phase_source=engine.phase_wrench_source):
            if not engine.observe(observation):
                continue
            obs = engine._policy_observations()
            _, normalized, current_contact, has_contact = policy.prepare_context_inputs(obs)
            raw = np.stack([row["future"] for row in engine._contexts])
            policy_future = array(obs[policy.wrench_key])[0]
            processed = array(policy._latest_processed_wrench)[0]
            normalized = array(normalized)[0]
            mask = None
            raw_mask = None
            if engine.future_contact_mode == "token_threshold":
                mask = policy.contact_detector.token_contact(
                    obs[policy.future_phase_wrench_key]
                ).detach().cpu().tolist()[0]
                raw_mask = policy.contact_detector.token_contact(
                    torch.as_tensor(raw[None], device=engine.device)
                ).detach().cpu().tolist()[0]
            row = {
                "frame_index": frame,
                "timestamp_s": float(observation.timestamp_s),
                "gate": has_contact.detach().cpu().tolist()[0],
                "current_contact": current_contact.detach().cpu().tolist()[0],
                "token_contact": mask,
                "raw_token_contact": raw_mask,
                "raw_force_norm_n": np.linalg.norm(raw[..., :3], axis=-1).tolist(),
                "processed_force_norm_n": np.linalg.norm(processed[..., :3], axis=-1).tolist(),
                "normalized_force_norm": np.linalg.norm(normalized[..., :3], axis=-1).tolist(),
                "raw_wrench": raw.tolist(),
                "policy_future_wrench": policy_future.tolist(),
                "contact_decisions": [row["contact_decision"] for row in engine._contexts],
                "processed_wrench": processed.tolist(),
                "normalized_wrench": normalized.tolist(),
                # These forecasts refer to the same absolute target times.
                "overlap_force_rmse_n": None if previous_raw is None else float(
                    np.sqrt(np.mean((raw[-1, :-1, :3] - previous_raw[1:, :3]) ** 2))
                ),
            }
            previous_raw = raw[-1].copy()
            rows.append(row)
            if frame not in sample_frames:
                continue
            passes = {branch: predict_branch(engine, obs, branch, action_index)
                      for branch in ("normal", "image", "fused")}
            normal = passes["normal"]
            positions = {name: result["position_m"].tolist() for name, result in passes.items()}
            distances = {name: float(1000 * np.linalg.norm(
                result["position_m"] - normal["position_m"]
            )) for name, result in passes.items() if name != "normal"}
            comparison = {
                "frame_index": frame, "gate": row["gate"],
                "effective_gate": {name: result["gate"] for name, result in passes.items()},
                "position_m": positions, "distance_from_normal_mm": distances,
                "fused_image_distance_mm": float(1000 * np.linalg.norm(
                    passes["fused"]["position_m"] - passes["image"]["position_m"]
                )),
                "context_rmse_from_normal": {name: float(np.sqrt(np.mean(
                    (result["context"] - normal["context"]) ** 2
                ))) for name, result in passes.items() if name != "normal"},
                "normalized_input_max_difference": max(float(np.max(np.abs(
                    result["normalized_wrench"] - normal["normalized_wrench"]
                ))) for result in passes.values()),
                "actions": {name: result["actions"].tolist() for name, result in passes.items()},
            }
            comparisons.append(comparison)
            print(f"frame={frame:3d} gate={row['gate']} "
                  f"normal-image={distances['image']:.3f} mm "
                  f"normal-fused={distances['fused']:.3f} mm "
                  f"image-fused={comparison['fused_image_distance_mm']:.3f} mm", flush=True)
    finally:
        engine.close()
    flips = [row["frame_index"] for previous, row in zip(rows, rows[1:])
             if previous["gate"][-1] != row["gate"][-1]]
    report = {
        "checkpoint_info": engine.checkpoint_info,
        "policy_source": inspect.getfile(type(policy)),
        "preprocessing_source": inspect.getfile(policy.prepare_context_inputs.__func__),
        "seed": engine.seed, "action_index": action_index,
        "phase_wrench_source": engine.phase_wrench_source,
        "future_contact_mode": engine.future_contact_mode,
        "future_contact_stabilization": engine.future_contact_stabilization,
        "contact_fusion_gate": policy.contact_fusion_gate,
        "contact_threshold_n": float(policy.contact_detector.contact_threshold),
        "fixed_noise": "Same DDIM generator seed for all frames and all branches; one WM forecast per sensor frame",
        "forced_branches": "Only contact_gate is overridden; both keep the normal preprocessed wrench",
        "latest_context_gate_flip_frames": flips,
        "frames": rows, "comparisons": comparisons,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(output), "sensor_frames": max_frames,
                      "compared_frames": len(comparisons), "gate_flip_frames": flips}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--sample-frames", default="20,38,40,41,42,43,44,45,46,47,48,49,50,51,52,60,70,76,77,78,79,80,81,90")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--disable-contact-stabilization", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.disable_contact_stabilization:
        config["runtime"]["future_contact_stabilization"] = {"enabled": False}
    diagnose(config, max_frames=args.max_frames,
             sample_frames={int(value) for value in args.sample_frames.split(",")},
             output=args.output)


if __name__ == "__main__":
    main()
