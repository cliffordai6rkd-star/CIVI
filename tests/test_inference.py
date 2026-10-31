import copy
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
import yaml

from inference import InferenceBase, InferenceObservation
from inference.cad_wm_provider import CADWMWrenchProvider, predictor_class
from inference.config import DEFAULT_CONFIG, PROJECT_ROOT, check_config, load_config
from inference.fdp_inference import FDPInference
from inference.replay import run_replay, sensor_frames


class FakePolicy(torch.nn.Module):
    image_keys = ("side", "wrist")
    wrench_key = "future_wrench"
    phase_wrench_key = "wrench"
    future_phase_wrench_key = "future_wrench"
    n_obs_steps = 2
    n_action_steps = 8
    action_start_index = 1

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.last_obs = None

    def reset(self):
        self.last_obs = None

    def predict_action(self, obs, generator):
        assert not self.training and not torch.is_grad_enabled()
        self.last_obs = {key: value.clone() for key, value in obs.items()}
        action = torch.zeros((1, 8, 7))
        action[..., :3] = torch.randn((1, 8, 3), generator=generator)
        action[..., 6] = 1
        return {"action": action}


class FakeWM:
    history_steps = 4
    joint_dim = 7
    sample_rate_hz = 20
    future_offsets_s = np.arange(1, 9) / 20
    pad_history = True

    def reset(self, *, seed):
        self.calls = []

    def predict(self, history):
        self.calls.append(copy.deepcopy(history))
        latest = history["q"][0, -1, 0]
        return np.broadcast_to((latest + np.arange(1, 9))[:, None], (8, 6)).copy()


def dp_config():
    shapes = {camera: {"shape": [3, 16, 16], "type": "rgb"} for camera in ("side", "wrist")}
    shapes.update({key: {"shape": [8, 6], "type": "low_dim"} for key in ("wrench", "future_wrench")})
    return OmegaConf.create({
        "policy": {"relative_pose_actions": False},
        "shape_meta": {"obs": shapes, "action": {"shape": [7]}},
        "task": {"dataset": {"timestamp_step_sec": 0.05, "action_feature_fields": ["observation.ee_pose"],
                             "windowed_lowdim_feature_groups": {
                                 "wrench": {"start_offset": -7, "stride": 1},
                                 "future_wrench": {"offsets_sec": (np.arange(1, 9) / 20).tolist()},
                             }}},
    })


def engine(**kwargs):
    return FDPInference(policy=FakePolicy(), dp_config=dp_config(), wrench_provider=FakeWM(), **kwargs)


def observation(index, *, wrench=None):
    return InferenceObservation(
        timestamp_s=index / 20,
        images={camera: np.full((8, 8, 3), index % 256, dtype=np.uint8) for camera in ("side", "wrist")},
        q=np.full(7, index), dq=np.full(7, index + 0.1), tau=np.full(7, index + 0.2), wrench=wrench,
    )


def test_base_is_abstract_and_fdp_obeys_shared_api():
    with pytest.raises(TypeError):
        InferenceBase()
    model = engine()
    assert isinstance(model, InferenceBase)
    assert model.spec.action_shape == (8, 7) and model.spec.sample_rate_hz == 20
    assert "wrench" not in model.spec.required_fields
    with pytest.raises(RuntimeError, match="before predicting"):
        model.predict()


def test_causal_forecasts_are_cached_per_image_anchor_and_phase_uses_previous_prediction():
    model = engine()
    first = model(observation(0))
    assert not first.ready and first.actions is None
    assert first.info["warmup_reason"] == "waiting_for_current_wrench_forecast"
    assert not model(observation(1)).ready
    result = model(observation(2, wrench=np.full(6, np.nan)))
    assert result.ready and result.actions.shape == (8, 7)
    obs = model.policy.last_obs
    assert obs["side"].shape == (1, 2, 3, 16, 16)
    np.testing.assert_array_equal(obs["future_wrench"][0, 0, :, 0], np.arange(2, 10))
    np.testing.assert_array_equal(obs["future_wrench"][0, 1, :, 0], np.arange(3, 11))
    np.testing.assert_array_equal(obs["wrench"][0, 0, :, 0], np.ones(8))
    np.testing.assert_array_equal(obs["wrench"][0, 1, :, 0], [1] * 7 + [2])
    np.testing.assert_allclose(result.action_timestamps_s, 0.1 + np.arange(8) / 20)
    np.testing.assert_allclose(result.info["context_timestamps_s"], [0.05, 0.1])
    for frame, history in enumerate(model.wrench_provider.calls):
        assert set(history) == {"q", "dq", "tau"}
        assert history["q"][0, -1, 0] == frame
        assert history["q"].max() <= frame # no later state or action inputs
    assert all(not parameter.requires_grad for parameter in model.policy.parameters())


def test_reset_clears_both_model_streams_and_reproduces_sampling():
    model = engine(seed=123)
    for i in range(3):
        result = model(observation(i))
    expected = result.actions.copy()
    # Returned diagnostic arrays do not alias internal future predictions.
    result.info["future_wrench"].fill(999)
    assert model.predict().info["future_wrench"].max() < 999
    model.reset(seed=123)
    for i in range(3):
        actual = model(observation(i))
    np.testing.assert_array_equal(actual.actions, expected)
    assert actual.info["frames_seen"] == 3 and len(model.wrench_provider.calls) == 3


def test_optional_measured_phase_does_not_replace_cad_future():
    model = engine(phase_wrench_source="measured")
    assert "wrench" in model.spec.required_fields
    assert not model(observation(0, wrench=np.full(6, 100))).ready
    assert model(observation(1, wrench=np.full(6, 101))).ready
    obs = model.policy.last_obs
    np.testing.assert_array_equal(obs["wrench"][0, 0], np.full((8, 6), 100))
    np.testing.assert_array_equal(obs["future_wrench"][0, 0, :, 0], np.arange(1, 9))


def test_timestamp_gaps_duplicates_and_episode_reuse_are_rejected_without_consuming_state():
    model = engine()
    model.observe(observation(0))
    for invalid in (observation(0), observation(2), observation(-1)):
        with pytest.raises(ValueError, match="every frame"):
            model.observe(invalid)
    assert len(model.wrench_provider.calls) == 1
    model.observe(observation(1))
    assert model(observation(2)).ready


@pytest.mark.parametrize("key", ["q", "dq", "tau"])
def test_invalid_state_is_rejected_before_wm_call(key):
    model = engine()
    values = dict(observation(0).__dict__)
    values[key] = np.zeros(24)
    with pytest.raises(ValueError, match=key):
        model.observe(InferenceObservation(**values))
    assert not model.wrench_provider.calls


def test_unpadded_world_model_waits_for_complete_history_then_two_phase_ready_frames():
    wm = FakeWM()
    wm.pad_history = False
    model = FDPInference(policy=FakePolicy(), dp_config=dp_config(), wrench_provider=wm)
    for index in range(5):
        assert not model(observation(index)).ready
    assert model(observation(5)).ready
    assert wm.calls[0]["q"].shape == (1, 4, 7)
    np.testing.assert_array_equal(wm.calls[0]["q"][0, :, 0], np.arange(4))


@pytest.mark.parametrize("mismatch", ["rate", "offsets", "target", "history"])
def test_incompatible_training_contract_is_rejected(mismatch):
    wm, cfg = FakeWM(), dp_config()
    if mismatch == "rate":
        wm.sample_rate_hz = 50
    elif mismatch == "offsets":
        wm.future_offsets_s = np.arange(2, 10) / 20
    elif mismatch == "target":
        cfg.task.dataset.action_feature_fields = ["action"]
    else:
        cfg.task.dataset.windowed_lowdim_feature_groups.wrench.stride = 2
    with pytest.raises(ValueError):
        FDPInference(policy=FakePolicy(), dp_config=cfg, wrench_provider=wm)


def test_observe_between_replans_keeps_each_native_wm_frame():
    model = engine()
    for i in range(10):
        model.observe(observation(i))
    result = model.predict()
    assert result.ready and len(model.wrench_provider.calls) == 10
    np.testing.assert_allclose(result.info["context_timestamps_s"], [0.4, 0.45])
    assert model.wrench_provider.calls[-1]["q"].shape == (1, 4, 7)
    np.testing.assert_array_equal(model.wrench_provider.calls[-1]["q"][0, :, 0], [6, 7, 8, 9])


@pytest.fixture
def replay_config(tmp_path):
    cfg = load_config()
    source = tmp_path / "sensors.hdf5"
    with h5py.File(source, "w") as file:
        demo = file.create_group("data/demo_0")
        demo.create_dataset("states", data=np.arange(8)[:, None] / 20)
        for key, width in (("joint_states", 7), ("dq", 24), ("tau", 24)):
            demo.create_dataset(f"obs/{key}", data=np.broadcast_to(np.arange(8)[:, None], (8, width)))
        for camera in ("eye_in_hand_rgb", "agentview_rgb"):
            demo.create_dataset(f"obs/{camera}", data=np.zeros((8, 8, 8, 3), dtype=np.uint8))
        # Deliberately NO wrenches, actions, rewards or future ground truth.
    cfg["replay"].update(dataset=str(source), output=str(tmp_path / "out.npz"), max_frames=None, predict_every=2)
    return cfg


def test_replay_has_no_ground_truth_wrench_dependency_and_preserves_observe_rate(replay_config):
    model = engine()
    summary = run_replay(model, replay_config)
    assert summary["sensor_frames"] == 8 and summary["prediction_chunks"] == 3
    assert len(model.wrench_provider.calls) == 8
    output = Path(summary["output"])
    with np.load(output, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["frame_index"], [2, 4, 6])
        assert data["actions"].shape == (3, 8, 7)
        assert data["future_wrench"].shape == (3, 8, 6)
        assert data["action_format"].item() == "absolute_ee_pose_xyz_xyzw"
    assert json.loads(output.with_suffix(".json").read_text())["wrench_source"] == "cad_wm"
    with pytest.raises(FileExistsError):
        run_replay(model, replay_config)


def test_config_paths_are_anchored_to_config_file_not_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = load_config(DEFAULT_CONFIG)
    assert Path(cfg["dp"]["repo"]) == PROJECT_ROOT.parent / "dp_hirol_main"
    assert cfg["cad_wm"]["checkpoint"] is None
    with pytest.raises(ValueError, match="cad_wm.checkpoint"):
        FDPInference.from_config(DEFAULT_CONFIG)


def test_config_check_can_compare_training_contracts_before_wm_training(tmp_path):
    cfg = load_config()
    teacher_cfg = dp_config()
    teacher_cfg.n_obs_steps = 2
    teacher_cfg.n_action_steps = 8
    teacher_cfg.policy.update(wrench_key="future_wrench", force_encoder_layers=2,
                              cross_attn_query="image", cross_attn_layers=1)
    teacher_run = tmp_path / "teacher"
    (teacher_run / ".hydra").mkdir(parents=True)
    OmegaConf.save(teacher_cfg, teacher_run / ".hydra/config.yaml")
    cfg["dp"]["checkpoint"] = str(teacher_run / "checkpoints/not_yet_written.ckpt")
    report = check_config(cfg)
    assert report["ready_to_load"] is False
    assert report["cad_wm_checkpoint"] is None
    assert report["sample_rate_hz"] == 20 and report["action_shape"] == [8, 7]
    wm_config = yaml.safe_load(Path(cfg["cad_wm"]["train_config"]).read_text())
    wm_config["dataloader"]["high_fps"] = 50
    mismatched = tmp_path / "wm.yaml"
    mismatched.write_text(yaml.safe_dump(wm_config))
    cfg["cad_wm"]["train_config"] = str(mismatched)
    with pytest.raises(ValueError, match="offsets differ"):
        check_config(cfg)


@pytest.fixture
def tiny_wm_checkpoint(tmp_path):
    predictor = predictor_class(PROJECT_ROOT / "cad-wm")
    module = sys.modules[predictor.__module__]
    config = yaml.safe_load((PROJECT_ROOT / "cad-wm/config/bench_train_config/libero_drawer_wrench_h5.yaml").read_text())
    config["dataloader"]["state_history_horizon"] = 4
    config["model"].update(hidden_dim=8, attention_heads=2, flow_layers=1, lstm_layers=2,
                           free_dynamics_hidden_dim=8, flow_inference_steps=2, dropout=0.0)
    model = module.CADWorldModel(config)
    # Test fixture only: no model is trained and no production weights are written.
    stats = {key: {"mean": torch.zeros(size), "std": torch.ones(size)}
             for key, size in (("q", 7), ("dq", 7), ("tau", 7), ("wrench", 6))}
    stats["wrench"]["mean"].fill_(12)
    checkpoint = tmp_path / "test_only_wm.pt"
    torch.save({"model_version": "cad-wm", "config": config, "model": model.state_dict(),
                "cad_wm_contract": model.checkpoint_contract(), "normalizer": {"stats": stats, "eps": 1e-6}}, checkpoint)
    return checkpoint


def test_real_cad_predictor_private_import_contract_and_deterministic_stream(tiny_wm_checkpoint):
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        wm = CADWMWrenchProvider(repo=PROJECT_ROOT / "cad-wm", checkpoint=tiny_wm_checkpoint)
        assert wm.history_steps == 4 and wm.pad_history
        history = {key: np.zeros((1, 2, 7), dtype=np.float32) for key in ("q", "dq", "tau")}
        wm.reset(seed=7)
        expected = wm.predict(history)
        wm.reset(seed=7)
        np.testing.assert_array_equal(wm.predict(history), expected)
        assert expected.shape == (8, 6) and np.isfinite(expected).all()
        assert expected.mean() > 5 # Output is denormalized into physical units.
        assert wm.predictor.__class__.__module__.startswith("_cad_wm_runtime_")
        model = FDPInference(policy=FakePolicy(), dp_config=dp_config(), wrench_provider=wm)
        for frame in range(3):
            result = model(observation(frame))
        assert result.ready and np.isfinite(result.actions).all()
    finally:
        torch.set_num_threads(threads)


def test_cad_import_does_not_import_dp_train_script_or_pollute_generic_names():
    script = f'''
import sys
from inference.cad_wm_provider import predictor_class
sys.path.insert(0, {str(PROJECT_ROOT.parent / 'dp_hirol_main')!r})
predictor_class({str(PROJECT_ROOT / 'cad-wm')!r})
assert 'train' not in sys.modules
assert 'model' not in sys.modules
assert 'data_process' not in sys.modules
print('namespace isolated')
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=PROJECT_ROOT, text=True, capture_output=True, check=True)
    assert "namespace isolated" in result.stdout
