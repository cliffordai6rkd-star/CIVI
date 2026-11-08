"""Verify recorded data alignment, cursor selection and scene restoration."""

import json

import h5py
import numpy as np
import pytest

from scripts import replay_wrench_mujoco as replay


@pytest.fixture
def recording(tmp_path):
    path = tmp_path / "demo.hdf5"
    with h5py.File(path, "w") as source:
        root = source.create_group("data")
        root.attrs["env_args"] = json.dumps({"env_kwargs": {"control_freq": 10}})
        demo = root.create_group("demo_0")
        # Flattened state = [time, qpos, qvel] for a one-joint test model.
        demo.create_dataset("states", data=[[3.0, 0.1, 1.0],
                                             [3.1, 0.2, 1.0],
                                             [3.2, 0.3, 1.0]])
        demo.create_dataset("obs/wrenches", data=np.arange(18).reshape(3, 6))
        demo.create_dataset("timestamp", data=[10.0, 10.12, 10.25])
    return path


def test_load_uses_recorded_wrench_and_metadata_fps(recording):
    episode = replay.load_episode(recording, "demo_0")
    np.testing.assert_allclose(episode.times, [0, 0.1, 0.2])
    np.testing.assert_array_equal(episode.wrench, np.arange(18).reshape(3, 6))
    episode = replay.load_episode(recording, "demo_0", time_key="timestamp")
    np.testing.assert_allclose(episode.times, [0, 0.12, 0.25])


def test_misaligned_wrench_is_rejected(recording):
    with h5py.File(recording, "a") as source:
        del source["data/demo_0/obs/wrenches"]
        source.create_dataset("data/demo_0/obs/wrenches", data=np.zeros((2, 6)))
    with pytest.raises(ValueError, match="aligned"):
        replay.load_episode(recording, "demo_0")


def test_cursor_selects_nearest_frame_and_clamps_to_bounds():
    times = np.array([0, 0.1, 0.2, 0.3, 0.4])
    assert replay.frame_at_time(times, 0.08) == 1
    assert replay.frame_at_time(times, 0.21) == 2
    assert replay.frame_at_time(times, -10) == 0
    assert replay.frame_at_time(times, 10) == 4


def test_headless_replay_restores_recorded_state(recording, tmp_path, monkeypatch):
    mujoco = pytest.importorskip("mujoco")
    xml = tmp_path / "scene.xml"
    xml.write_text('<mujoco><worldbody><body><joint name="joint" type="slide"/>'
                   '<geom type="sphere" size="0.1"/></body></worldbody></mujoco>')
    args = replay.parse_args(["--dataset", str(recording), "--xml", str(xml), "--headless"])
    episode = replay.load_episode(recording, "demo_0")
    handle = replay._make_sim(episode.demo, recording, args, replay.REPO_ROOT)
    monkeypatch.setattr(replay, "_make_sim", lambda *unused: handle)
    result, frames = replay.replay_episode(episode, recording, args)
    assert result == "next"
    assert frames == 3
    np.testing.assert_allclose(handle.data.qpos, [0.3])
    np.testing.assert_allclose(handle.data.qvel, [1.0])
    assert handle.data.time == pytest.approx(3.2)
    # Verify replay has not replaced the source wrench or written derived fields.
    with h5py.File(recording, "r") as source:
        np.testing.assert_array_equal(source["data/demo_0/obs/wrenches"],
                                      np.arange(18).reshape(3, 6))
        assert "dq" not in source["data/demo_0/obs"]


def test_demo_selection_is_zero_based_and_inclusive(recording):
    with h5py.File(recording, "a") as source:
        source["data"].create_group("demo_2")
        source["data"].create_group("demo_10")
    assert replay._select_demos(recording, "all", None) == ["demo_0", "demo_2", "demo_10"]
    assert replay._select_demos(recording, "all", (0, 2)) == ["demo_0", "demo_2"]
    with pytest.raises(KeyError):
        replay._select_demos(recording, "3", None)
