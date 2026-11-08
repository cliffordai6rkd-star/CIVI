"""Check timeline synchronization and input routing without a GUI display."""

from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pytest

from scripts.wrench_replay_plot import WrenchLivePlot, WrenchReplayPlot


def test_import_does_not_import_pyplot_or_select_backend():
    code = (
        "import sys; import scripts.wrench_replay_plot; "
        "assert 'matplotlib.pyplot' not in sys.modules; "
        "assert 'matplotlib' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_noninteractive_backend_is_rejected():
    with pytest.raises(RuntimeError, match="cannot open an interactive wrench window"):
        WrenchReplayPlot(
            np.arange(3), np.zeros((3, 6)), "Replay", lambda _: None, lambda _: None,
            backend="Agg",
        )


def test_cursor_moves_without_rescaling_and_input_is_forwarded():
    times = np.array([0.0, 0.1, 0.2])
    wrench = np.arange(18, dtype=float).reshape(3, 6)
    seeks, keys = [], []
    plot = WrenchReplayPlot(
        times, wrench, "Replay", seeks.append, keys.append,
        backend="Agg", allow_headless=True,
    )
    try:
        limits = [(axis.get_xlim(), axis.get_ylim()) for axis in plot.axes]
        plot.update(2, paused=True)
        plot.pump()
        for axis, cursor, expected_limits in zip(plot.axes, plot._cursors, limits):
            np.testing.assert_allclose(cursor.get_xdata(), [0.2, 0.2])
            assert (axis.get_xlim(), axis.get_ylim()) == expected_limits
        for column, point in enumerate(plot._points):
            np.testing.assert_allclose(point.get_ydata(), [wrench[2, column]])
        assert "Cursor control" in plot._status.get_text()
        assert "0.200 s" in plot._status.get_text()
        assert plot.slider.val == 0.2
        assert seeks == []
        assert plot.axes[0].get_ylabel() == "Force [N]"
        assert plot.axes[1].get_ylabel() == "Torque [N.m]"
        plot._handle_click(SimpleNamespace(button=1, inaxes=plot.axes[1], xdata=0.12))
        plot._handle_click(SimpleNamespace(button=1, inaxes=None, xdata=None))
        plot._handle_motion(SimpleNamespace(inaxes=plot.axes[0], xdata=0.15))
        plot._handle_release(SimpleNamespace(button=1))
        plot._handle_motion(SimpleNamespace(inaxes=plot.axes[0], xdata=0.18))
        plot.slider.set_val(0.05)
        plot._handle_key(SimpleNamespace(key=" "))
        assert seeks == [0.12, 0.15, 0.05]
        assert keys == [" "]
        assert plot.is_open
    finally:
        plot.close()
    assert not plot.is_open
    plot.close()


def test_slider_seek_can_update_cursor_without_callback_recursion():
    times = np.array([0.0, 0.1, 0.2])
    seeks = []

    def seek(value):
        seeks.append(value)
        plot.update(int(np.abs(times - value).argmin()))

    plot = WrenchReplayPlot(
        times, np.zeros((3, 6)), "Cursor", seek, lambda _: None,
        backend="Agg", allow_headless=True,
    )
    try:
        assert seeks == []
        plot.slider.set_val(0.12)
        assert seeks == [0.12]
        assert plot.slider.val == 0.1
        assert plot.slider.eventson is True
        assert plot.slider.drawon is True
        plot.update(2, paused=False)
        assert seeks == [0.12]
        assert "Playing" not in plot._status.get_text()
        assert "Paused" not in plot._status.get_text()
    finally:
        plot.close()


def test_single_sample_slider_has_a_valid_range():
    plot = WrenchReplayPlot(
        np.array([0.5]), np.zeros((1, 6)), "Cursor", lambda _: None, lambda _: None,
        backend="Agg", allow_headless=True,
    )
    try:
        assert plot.slider.valmin < plot.slider.valmax
        assert plot.slider.val == 0.5
        assert not plot.slider.active
    finally:
        plot.close()


@pytest.mark.parametrize(
    "times,wrench",
    [
        ([], np.empty((0, 6))),
        ([0, 1], np.zeros((2, 3))),
        ([1, 0], np.zeros((2, 6))),
        ([0, np.nan], np.zeros((2, 6))),
    ],
)
def test_invalid_recordings_fail_before_opening_a_window(times, wrench):
    with pytest.raises(ValueError):
        WrenchReplayPlot(times, wrench, "Replay", lambda _: None, lambda _: None)


def test_live_plot_appends_samples_and_updates_curves():
    plot = WrenchLivePlot(
        title="Live", max_points=4, backend="Agg", allow_headless=True
    )
    try:
        plot.update(0.0, np.arange(6, dtype=float))
        plot.update(0.1, np.arange(6, dtype=float) + 1.0)
        plot.pump()
        assert plot.sample_count == 2
        np.testing.assert_allclose(plot.times, [0.0, 0.1])
        np.testing.assert_allclose(plot.wrench[-1], np.arange(6, dtype=float) + 1.0)
        np.testing.assert_allclose(plot._lines[0].get_xdata(), [0.0, 0.1])
        np.testing.assert_allclose(plot._lines[0].get_ydata(), [0.0, 1.0])
        np.testing.assert_allclose(plot._points[-1].get_ydata(), [6.0])
        assert "|F|" in plot._status.get_text()
        assert plot.is_open
    finally:
        plot.close()


def test_live_plot_rolls_history_and_resets():
    plot = WrenchLivePlot(
        title="Live", max_points=2, backend="Agg", allow_headless=True
    )
    try:
        for timestamp in (0.0, 0.1, 0.2):
            plot.update(timestamp, np.full(6, timestamp))
        np.testing.assert_allclose(plot.times, [0.1, 0.2])
        np.testing.assert_allclose(plot.wrench[:, 0], [0.1, 0.2])
        plot.reset()
        assert plot.sample_count == 0
        assert plot.wrench.shape == (0, 6)
        assert "waiting for samples" in plot._status.get_text()
        for line, point in zip(plot._lines, plot._points):
            assert len(line.get_xdata()) == 0
            assert len(point.get_xdata()) == 0
        for line in plot._prediction_lines:
            assert len(line.get_xdata()) == 0
    finally:
        plot.close()


def test_live_plot_overlays_cad_wm_forecast_without_changing_history():
    plot = WrenchLivePlot(
        title="Live", max_points=4, backend="Agg", allow_headless=True
    )
    try:
        plot.update(0.0, np.zeros(6))
        forecast = np.arange(48, dtype=float).reshape(8, 6)
        plot.update_prediction(0.0, np.arange(1, 9) / 20.0, forecast)
        plot.update(0.05, np.ones(6))
        np.testing.assert_allclose(plot.times, [0.0, 0.05])
        np.testing.assert_allclose(plot._prediction_times, np.arange(1, 9) / 20.0)
        np.testing.assert_allclose(plot._prediction_wrench, forecast)
        np.testing.assert_allclose(plot._prediction_lines[0].get_ydata(), forecast[:, 0])
        assert plot.axes[-1].get_xlim()[1] >= 0.4
        assert "CAD-WM forecast" in plot._status.get_text()
    finally:
        plot.close()


@pytest.mark.parametrize(
    "timestamp,wrench,match",
    [
        (0.0, np.zeros(3), "shape"),
        (0.0, np.full(6, np.nan), "finite"),
        (np.nan, np.zeros(6), "finite"),
        ("bad", np.zeros(6), "finite"),
    ],
)
def test_live_plot_rejects_invalid_samples(timestamp, wrench, match):
    plot = WrenchLivePlot(backend="Agg", allow_headless=True)
    try:
        with pytest.raises(ValueError, match=match):
            plot.update(timestamp, wrench)
    finally:
        plot.close()


def test_live_plot_rejects_backwards_time():
    plot = WrenchLivePlot(backend="Agg", allow_headless=True)
    try:
        plot.update(1.0, np.zeros(6))
        with pytest.raises(ValueError, match="nondecreasing"):
            plot.update(0.5, np.zeros(6))
    finally:
        plot.close()


@pytest.mark.parametrize(
    "offsets,wrench,match",
    [
        (np.arange(8) / 20.0, np.zeros((7, 6)), "shape"),
        ([0.05, 0.05], np.zeros((2, 6)), "strictly increasing"),
        ([-0.05], np.zeros((1, 6)), "nonnegative"),
    ],
)
def test_live_plot_rejects_invalid_forecasts(offsets, wrench, match):
    plot = WrenchLivePlot(backend="Agg", allow_headless=True)
    try:
        with pytest.raises(ValueError, match=match):
            plot.update_prediction(0.0, offsets, wrench)
    finally:
        plot.close()
