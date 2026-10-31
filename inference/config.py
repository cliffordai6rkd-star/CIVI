"""Resolve model paths relative to an explicit, config-relative project root."""

from pathlib import Path

import numpy as np
import yaml


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
                          "replay": ("dataset", "output")}.items():
        for key in keys:
            value = config.get(section, {}).get(key)
            if value is not None:
                config[section][key] = str((root / Path(value).expanduser()).resolve())
    return config


def check_config(config):
    """Inspect training contracts without allocating models or needing WM weights."""
    from omegaconf import OmegaConf

    checkpoint = Path(config["dp"]["checkpoint"])
    saved = checkpoint.parent.parent / ".hydra/config.yaml"
    dp = OmegaConf.load(saved)
    OmegaConf.resolve(dp)
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
    return {"dp_checkpoint": str(checkpoint), "dp_checkpoint_exists": checkpoint.is_file(),
            "cad_wm_checkpoint": wm_checkpoint,
            "cad_wm_checkpoint_exists": bool(wm_checkpoint and Path(wm_checkpoint).is_file()),
            "ready_to_load": checkpoint.is_file() and bool(wm_checkpoint and Path(wm_checkpoint).is_file()),
            "sample_rate_hz": 1 / dt, "observation_steps": int(dp.n_obs_steps),
            "cad_wm_history_steps": int(wm["dataloader"]["state_history_horizon"]),
            "future_wrench_offsets_s": offsets,
            "action_format": "absolute_ee_pose_xyz_xyzw", "action_shape": [int(dp.n_action_steps), 7],
            "phase_wrench_source": config["runtime"]["phase_wrench_source"],
            "lstm_layers": int(dp.policy.force_encoder_layers),
            "cross_attention_query": str(dp.policy.cross_attn_query),
            "cross_attention_layers": int(dp.policy.cross_attn_layers),
            "note": "Contract checked against training configs; runtime also checks checkpoint contracts."}
