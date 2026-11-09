import numpy as np
import pytest

from inference.contact_stabilizer import FutureContactStabilizer


def forecast(*contact_indices, force=20.0):
    future = np.zeros((8, 6), dtype=np.float32)
    future[:, 0] = 5.0
    future[list(contact_indices), 0] = force
    return future, future[:, 0] > 8.0


def test_isolated_spike_never_opens_contact():
    stabilizer = FutureContactStabilizer()
    for values in (forecast(), forecast(7), forecast(), forecast()):
        filtered, decision = stabilizer.update(*values)
        assert not decision["has_contact"]
        np.testing.assert_array_equal(filtered, np.zeros((8, 6)))


def test_consecutive_forecasts_must_agree_on_target_time():
    stabilizer = FutureContactStabilizer()
    for _ in range(3):
        _, decision = stabilizer.update(*forecast(7))
        assert not decision["has_contact"]  # Each spike is a new absolute time.
    future, mask = forecast(6)
    filtered, decision = stabilizer.update(future, mask)
    assert decision == {"raw_has_contact": True, "has_contact": True, "holding": False}
    np.testing.assert_array_equal(filtered, future)


def test_exit_confirmation_holds_a_real_contact_forecast_then_zeros_both_paths():
    stabilizer = FutureContactStabilizer(exit_frames=2)
    stabilizer.update(*forecast(7))
    accepted, mask = forecast(6, 7)
    stabilizer.update(accepted, mask)
    filtered, decision = stabilizer.update(*forecast())
    assert decision == {"raw_has_contact": False, "has_contact": True, "holding": True}
    np.testing.assert_array_equal(filtered, accepted)
    # Returned values cannot mutate the cached contact condition.
    filtered.fill(99)
    filtered, decision = stabilizer.update(*forecast())
    assert not decision["has_contact"] and not decision["holding"]
    np.testing.assert_array_equal(filtered, np.zeros_like(accepted))


def test_recovered_contact_resets_exit_confirmation():
    stabilizer = FutureContactStabilizer(exit_frames=2)
    stabilizer.update(*forecast(7))
    stabilizer.update(*forecast(6, 7))
    stabilizer.update(*forecast())
    new_future, mask = forecast(3, 4, force=30)
    stabilizer.update(new_future, mask)
    filtered, decision = stabilizer.update(*forecast())
    assert decision["has_contact"] and decision["holding"]
    np.testing.assert_array_equal(filtered, new_future)
    stabilizer.reset()
    _, decision = stabilizer.update(*forecast(6))
    assert not decision["has_contact"]


def test_confirmation_counts_are_configurable():
    stabilizer = FutureContactStabilizer(enter_frames=3, exit_frames=1)
    for index, expected in ((7, False), (6, False), (5, True)):
        _, decision = stabilizer.update(*forecast(index))
        assert decision["has_contact"] == expected
    _, decision = stabilizer.update(*forecast())
    assert not decision["has_contact"]
    with pytest.raises(ValueError, match="positive"):
        FutureContactStabilizer(enter_frames=0)


def test_three_frames_must_confirm_one_target_in_all_three_forecasts():
    stabilizer = FutureContactStabilizer(enter_frames=3)
    # Pairs agree, but the first and third forecast refer to different targets.
    for values in (forecast(7), forecast(6, 3), forecast(2)):
        _, decision = stabilizer.update(*values)
        assert not decision["has_contact"]
    _, decision = stabilizer.update(*forecast(1))
    assert decision["has_contact"]


def test_default_exits_on_first_free_forecast():
    stabilizer = FutureContactStabilizer()
    stabilizer.update(*forecast(7))
    stabilizer.update(*forecast(6))
    filtered, decision = stabilizer.update(*forecast())
    assert decision == {"raw_has_contact": False, "has_contact": False, "holding": False}
    np.testing.assert_array_equal(filtered, np.zeros((8, 6)))
