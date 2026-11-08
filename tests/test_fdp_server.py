import numpy as np
import pytest
import threading
from pathlib import Path

import fdp_server
from fdp_server import FDPService, absolute_pose_to_libero_action


def test_service_checkpoint_overrides_replace_both_model_paths(monkeypatch, tmp_path):
    config = {
        "dp": {"checkpoint": "yaml-dp.ckpt", "backbone": "yaml-backbone"},
        "cad_wm": {"checkpoint": "yaml-wm.pt"},
    }

    class FakeEngine:
        def close(self):
            pass

    monkeypatch.setattr(fdp_server, "load_config", lambda _: config)
    monkeypatch.setattr(
        fdp_server.FDPInference,
        "from_loaded_config",
        lambda resolved: FakeEngine(),
    )
    dp = tmp_path / "teacher.ckpt"
    wm = tmp_path / "cad_wm.pt"
    service = FDPService(
        "ignored.yaml", checkpoint=dp, wm_checkpoint=wm, seed=7
    )

    assert Path(service.config["dp"]["checkpoint"]) == dp.resolve()
    assert Path(service.config["cad_wm"]["checkpoint"]) == wm.resolve()
    assert service.seed == 7


def test_absolute_pose_is_normalized_to_osc_input_range():
    action = absolute_pose_to_libero_action(
        [0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    )
    np.testing.assert_allclose(action, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_absolute_pose_uses_runtime_osc_bounds_and_clips():
    action = absolute_pose_to_libero_action(
        [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        osc_action_min=[-0.1, -0.1, -0.1, -1.0, -1.0, -1.0],
        osc_action_max=[0.1, 0.1, 0.1, 1.0, 1.0, 1.0],
    )
    np.testing.assert_allclose(action, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_absolute_pose_rejects_invalid_runtime_bounds():
    with pytest.raises(ValueError, match="bounds"):
        absolute_pose_to_libero_action(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], np.zeros(3), [0.0, 0.0, 0.0, 1.0],
            osc_action_min=np.zeros(6), osc_action_max=np.zeros(6),
        )


def test_service_requires_measured_wrench_when_configured(monkeypatch):
    class FakeEngine:
        phase_wrench_source = "measured"
        dt = 0.05

        def reset(self, *, seed):
            pass

        def predict(self, observation):
            raise AssertionError("missing wrench must be rejected before predict")

    service = object.__new__(FDPService)
    service.engine = FakeEngine()
    service.seed = 42
    service.lock = threading.RLock()
    service.frame_index = 0
    service.initialized = True
    monkeypatch.setattr(
        fdp_server, "_decode_video",
        lambda payload: [np.zeros((8, 8, 3), dtype=np.uint8)] * 2,
    )
    payload = {
        "gt_video": [["ignored", "ignored"]],
        "fdp_state": {
            "q": [0.0] * 7, "dq": [0.0] * 7, "tau": [0.0] * 7,
            "eef_pos": [0.0] * 3, "eef_quat": [0.0, 0.0, 0.0, 1.0],
        },
    }
    with pytest.raises(ValueError, match="wrench.*required"):
        service.step(payload)


def test_service_returns_cad_wm_future_wrench_for_live_visualization(monkeypatch):
    class FakeResult:
        ready = False
        info = {
            "warmup_reason": "waiting_for_image_history",
            "future_wrench": np.arange(48, dtype=np.float32).reshape(8, 6),
            "future_wrench_offsets_s": np.arange(1, 9, dtype=np.float64) / 20.0,
            "future_contact_mode": "legacy_rollout",
        }

    class FakeEngine:
        phase_wrench_source = "measured"
        dt = 0.05

        def reset(self, *, seed):
            pass

        def predict(self, observation):
            return FakeResult()

    service = object.__new__(FDPService)
    service.engine = FakeEngine()
    service.seed = 42
    service.lock = threading.RLock()
    service.frame_index = 0
    service.initialized = True
    monkeypatch.setattr(
        fdp_server,
        "_decode_video",
        lambda payload: [np.zeros((8, 8, 3), dtype=np.uint8)] * 2,
    )
    payload = {
        "gt_video": [["ignored", "ignored"]],
        "fdp_state": {
            "q": [0.0] * 7,
            "dq": [0.0] * 7,
            "tau": [0.0] * 7,
            "eef_pos": [0.0] * 3,
            "eef_quat": [0.0, 0.0, 0.0, 1.0],
            "wrench": [0.0] * 6,
        },
    }
    response = service.step(payload)
    np.testing.assert_allclose(response["future_wrench"], FakeResult.info["future_wrench"])
    np.testing.assert_allclose(
        response["future_wrench_offsets_s"], FakeResult.info["future_wrench_offsets_s"]
    )
    assert response["future_contact_mode"] == "legacy_rollout"
