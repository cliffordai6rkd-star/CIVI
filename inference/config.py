"""Resolve model paths relative to an explicit, config-relative project root."""

from pathlib import Path

import numpy as np
import yaml

from .wrench_filter import resolve_wrench_filter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/inference/fdp_cad_wm.yaml"


def load_config(path=DEFAULT_CONFIG):
    path = Path(path).expanduser().resolve()
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Inference config must be a mapping")
    root = (path.parent / Path(config.get("project_root", ".")).expanduser()).resolve()
    for section, keys in {"dp": ("repo", "checkpoint", "backbone"),
                          "cad_wm": ("repo", "checkpoint", "train_config"),
                          "student": ("repo", "checkpoint"),
                          "replay": ("dataset", "output")}.items():
        section_config = config.get(section) or {}
        if not isinstance(section_config, dict):
            raise ValueError(f"Inference config section {section!r} must be a mapping")
        config[section] = section_config
        for key in keys:
            value = section_config.get(key)
            if value is not None:
                config[section][key] = str((root / Path(value).expanduser()).resolve())
    return config


def check_config(config):
    """Inspect training contracts without allocating models or needing WM weights."""
    from omegaconf import OmegaConf

    checkpoint = Path(config["dp"]["checkpoint"])
    saved = checkpoint.parent.parent / ".hydra/config.yaml"
    dp = OmegaConf.load(saved)
    # Resolve only the fields below. Training-only logging paths can contain
    # Hydra's ${now:...}, which is unavailable in this standalone CLI.
    wm = yaml.safe_load(Path(config["cad_wm"]["train_config"]).read_text())
    dataset = dp.task.dataset
    dt = float(dataset.timestamp_step_sec)
    offsets = list(dataset.windowed_lowdim_feature_groups[dp.policy.wrench_key].offsets_sec)
    wm_offsets = np.arange(1, int(wm["dataloader"]["prediction_horizon"]) + 1) / float(wm["dataloader"]["high_fps"])
    if not np.isclose(dt, 1 / float(wm["dataloader"]["high_fps"])) or (
        len(offsets) != len(wm_offsets) or not np.allclose(offsets, wm_offsets, atol=1e-7, rtol=0)
    ):
        raise ValueError("DP and CAD-WM native rate / future wrench offsets differ")
    if list(wm["model"]["inputs"]) != ["q", "dq", "tau"] or list(wm["model"]["outputs"]) != ["wrench"]:
        raise ValueError("CAD-WM must use q/dq/tau -> wrench")
    wm_checkpoint = config["cad_wm"].get("checkpoint")
    wrench_filter = resolve_wrench_filter(config["runtime"].get("wrench_filter"))
    training_filter = resolve_wrench_filter(dataset.get("wrench_filter"))
    training_filter.pop("apply_to_future") # Runtime switch, not a Dataset parameter.
    filter_matches = (wrench_filter["enabled"] == training_filter["enabled"] and
                      (not wrench_filter["enabled"] or
                       np.isclose(wrench_filter["cutoff_hz"], training_filter["cutoff_hz"])))
    return {"dp_checkpoint": str(checkpoint), "dp_checkpoint_exists": checkpoint.is_file(),
            "cad_wm_checkpoint": wm_checkpoint,
            "cad_wm_checkpoint_exists": bool(wm_checkpoint and Path(wm_checkpoint).is_file()),
            "ready_to_load": checkpoint.is_file() and bool(wm_checkpoint and Path(wm_checkpoint).is_file()),
            "sample_rate_hz": 1 / dt, "observation_steps": int(dp.n_obs_steps),
            "cad_wm_history_steps": int(wm["dataloader"]["state_history_horizon"]),
            "future_wrench_offsets_s": offsets,
            "action_format": "absolute_ee_pose_xyz_xyzw" + ("_gripper_command" if int(dp.shape_meta.action.shape[0]) == 8 else ""),
            "action_shape": [int(dp.n_action_steps), int(dp.shape_meta.action.shape[0])],
            "phase_wrench_source": config["runtime"]["phase_wrench_source"],
            "future_contact_mode": (config["runtime"].get("future_contact_mode") or
                                    str(dp.policy.get("future_contact_mode", "legacy_rollout"))),
            "wrench_filter": wrench_filter,
            "training_wrench_filter": training_filter,
            "wrench_filter_matches_training": bool(filter_matches),
            "future_contact_stabilization": config["runtime"].get("future_contact_stabilization", {"enabled": False}),
            "lstm_layers": int(dp.policy.force_encoder_layers),
            "cross_attention_query": str(dp.policy.cross_attn_query),
            "cross_attention_layers": int(dp.policy.cross_attn_layers),
            "note": "Contract checked against training configs; runtime also checks checkpoint contracts."}
