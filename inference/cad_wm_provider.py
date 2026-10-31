"""Load CAD-WM in a private namespace to avoid the DP repo's train.py module."""

import hashlib
import importlib
from pathlib import Path
import sys
import types

import numpy as np
import torch


def predictor_class(repo):
    repo = Path(repo).expanduser().resolve()
    if not (repo / "model/wrench_predictor.py").is_file():
        raise FileNotFoundError(f"CAD-WM repo does not contain model/wrench_predictor.py: {repo}")
    name = "_cad_wm_runtime_" + hashlib.sha256(str(repo).encode()).hexdigest()[:12]
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(repo)]
        package.__package__ = name
        sys.modules[name] = package
    return importlib.import_module(f"{name}.model.wrench_predictor").WrenchPredictor


class CADWMWrenchProvider:
    """Physical state histories -> physical wrench forecasts; no target inputs."""

    def __init__(self, *, repo, checkpoint, device="cpu", steps=None, solver=None):
        if not checkpoint:
            raise ValueError("Set cad_wm.checkpoint in the inference config after CAD-WM training")
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"CAD-WM checkpoint does not exist: {checkpoint}")
        self.predictor = predictor_class(repo)(checkpoint, device=device)
        self.device = self.predictor.device
        self.history_steps = int(self.predictor.model.history_horizon)
        self.joint_dim = int(self.predictor.model.joint_dim)
        self.sample_rate_hz = float(self.predictor.config["dataloader"]["high_fps"])
        self.future_offsets_s = self.predictor.future_time.detach().cpu().numpy().astype(np.float64)
        self.pad_history = bool(self.predictor.config["dataloader"].get("pad_history", True))
        self.steps = steps
        self.solver = solver
        if steps is not None and int(steps) < 1:
            raise ValueError("cad_wm.steps must be positive or null")
        if solver not in (None, "heun", "euler"):
            raise ValueError("cad_wm.solver must be heun, euler or null")
        self.reset(seed=42)

    def reset(self, *, seed):
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))

    @torch.inference_mode()
    def predict(self, history):
        noise = torch.randn((1, len(self.future_offsets_s), 6), device=self.device, generator=self.generator)
        result = self.predictor(history, source_noise=noise, steps=self.steps, solver=self.solver)
        if not np.allclose(result["future_time"].detach().cpu().numpy(), self.future_offsets_s, atol=1e-7, rtol=0):
            raise ValueError("CAD-WM changed forecast timestamps")
        return result["future_wrench"][0].detach().float().cpu().numpy()
