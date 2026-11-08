"""Interactive force/torque timeline used by the MuJoCo wrench replayer.

Importing this module does not import pyplot or select a Matplotlib backend.
Seeking and keyboard controls are delegated to the caller through callbacks.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


class WrenchReplayPlot:
    """Show the full wrench trajectory and a manually controlled time cursor.

    ``times`` contains seconds and ``wrench`` contains Fx, Fy, Fz, Tx, Ty, Tz
    in N and N.m. An explicit ``backend`` selects a Matplotlib backend; otherwise
    Matplotlib's current backend is used. A GUI backend is required in production.
    ``allow_headless`` only enables noninteractive rendering in automated tests.
    """

    def __init__(
        self,
        times: np.ndarray,
        wrench: np.ndarray,
        title: str,
        on_seek: Callable[[float], None],
        on_key: Callable[[str], None],
        backend: str | None = None,
        *,
        allow_headless: bool = False,
    ) -> None:
        self.times = np.asarray(times, dtype=float)
        self.wrench = np.asarray(wrench, dtype=float)
        if self.times.ndim != 1 or not self.times.size:
            raise ValueError("times must be a nonempty one-dimensional array")
        if self.wrench.shape != (self.times.size, 6):
            raise ValueError("wrench must have shape [len(times), 6]")
        if not np.isfinite(self.times).all() or not np.isfinite(self.wrench).all():
            raise ValueError("times and wrench must contain only finite values")
        if np.any(np.diff(self.times) < 0):
            raise ValueError("times must be sorted in ascending order")
        if not callable(on_seek) or not callable(on_key):
            raise TypeError("on_seek and on_key must be callable")

        self._on_seek = on_seek
        self._on_key = on_key
        self._closed = False
        self._dragging = False
        self.figure = None

        # Backend selection must happen before pyplot is imported, and only when
        # explicitly requested. In particular, do not silently accept Agg.
        import matplotlib

        try:
            if backend is not None:
                matplotlib.use(backend)
            from matplotlib import pyplot as plt
            from matplotlib.widgets import Slider

            self._plt = plt
            selected_backend = str(matplotlib.get_backend())
            if not allow_headless and not self._is_interactive_backend(selected_backend):
                raise RuntimeError(
                    f"Matplotlib backend {selected_backend!r} cannot open an interactive "
                    "wrench window. Select a GUI backend such as TkAgg and run with "
                    "an available desktop display."
                )

            self.figure, self.axes = plt.subplots(2, 1, sharex=True, figsize=(11, 6))
            self.figure.suptitle(title)
            manager = self.figure.canvas.manager
            if manager is not None:
                manager.set_window_title(title)
                # The caller owns keyboard behavior, including q.
                handler_id = getattr(manager, "key_press_handler_id", None)
                if handler_id is not None:
                    self.figure.canvas.mpl_disconnect(handler_id)

            self._points = []
            self._cursors = []
            colors = ("tab:red", "tab:green", "tab:blue")
            for row, (axis, prefix, units) in enumerate(
                zip(self.axes, ("F", "T"), ("N", "N.m"))
            ):
                for offset, (component, color) in enumerate(zip("xyz", colors)):
                    column = row * 3 + offset
                    axis.plot(
                        self.times,
                        self.wrench[:, column],
                        color=color,
                        alpha=0.3,
                        linewidth=1.25,
                        label=f"{prefix}{component}",
                    )
                    point, = axis.plot([], [], "o", color=color, markersize=6)
                    self._points.append(point)
                self._cursors.append(
                    axis.axvline(self.times[0], color="0.2", linewidth=1.1)
                )
                axis.set_ylabel(f"{'Force' if row == 0 else 'Torque'} [{units}]")
                axis.grid(True, alpha=0.2)
                axis.legend(loc="upper right", ncol=3)
                # Full-episode limits remain fixed as the time cursor moves.
                axis.relim()
                axis.autoscale_view()
                axis.set_autoscaley_on(False)

            first, last = float(self.times[0]), float(self.times[-1])
            if first == last:
                first, last = first - 0.5, last + 0.5
            self.axes[-1].set_xlim(first, last)
            self.axes[-1].set_xlabel("Time [s] (click or drag to position)")
            for axis in self.axes:
                axis.set_autoscalex_on(False)

            self._status = self.figure.text(
                0.99, 0.94, "", ha="right", va="top", fontsize=10
            )
            self.figure.tight_layout(rect=(0, 0.15, 1, 0.92))
            slider_axis = self.figure.add_axes([0.13, 0.055, 0.76, 0.035])
            self.slider = Slider(
                slider_axis, "Time [s]", first, last,
                valinit=float(self.times[0]), valfmt="%.3f s",
            )
            self.slider.set_active(self.times[-1] > self.times[0])
            self.slider.on_changed(self._handle_slider)
            self.figure.canvas.mpl_connect("button_press_event", self._handle_click)
            self.figure.canvas.mpl_connect("motion_notify_event", self._handle_motion)
            self.figure.canvas.mpl_connect("button_release_event", self._handle_release)
            self.figure.canvas.mpl_connect("key_press_event", self._handle_key)
            self.figure.canvas.mpl_connect("close_event", self._handle_close)
            self.update(0)
            if not allow_headless:
                plt.show(block=False)
                self.pump()
        except Exception as exc:
            if self.figure is not None:
                self._plt.close(self.figure)
            self._closed = True
            raise RuntimeError(
                "Could not open the interactive wrench plot. "
                "Use a GUI backend such as TkAgg with a working desktop display. "
                f"Details: {exc}"
            ) from exc

    @staticmethod
    def _is_interactive_backend(backend: str) -> bool:
        try:
            from matplotlib.backends.registry import backend_registry

            _, gui = backend_registry.resolve_backend(backend)
            return gui is not None
        except ImportError:  # Matplotlib < 3.9
            from matplotlib import rcsetup

            return backend.lower() in {name.lower() for name in rcsetup.interactive_bk}

    @property
    def is_open(self) -> bool:
        """Whether the figure is still available for cursor control."""
        return (
            not self._closed
            and self.figure is not None
            and self._plt.fignum_exists(self.figure.number)
        )

    def update(self, index: int, paused: bool = True) -> None:
        """Move to a sample without seeking again; ``paused`` is kept for compatibility."""
        if not self.is_open:
            return
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self.times):
            raise IndexError("wrench cursor index is outside the recorded trajectory")
        current_time = float(self.times[index])
        for cursor in self._cursors:
            cursor.set_xdata([current_time, current_time])
        for column, point in enumerate(self._points):
            point.set_data([current_time], [self.wrench[index, column]])
        self._status.set_text(
            f"Cursor control  |  t = {current_time:.3f} s  |  "
            f"sample {index + 1}/{len(self.times)}"
        )
        # Slider callbacks may call the controller, which calls update() again.
        # Synchronize silently to avoid a callback feedback loop.
        eventson, drawon = self.slider.eventson, self.slider.drawon
        self.slider.eventson = False
        self.slider.drawon = False
        try:
            self.slider.set_val(current_time)
        finally:
            self.slider.eventson = eventson
            self.slider.drawon = drawon
        self.figure.canvas.draw_idle()

    def pump(self) -> None:
        """Process pending GUI input without owning the replayer's main loop."""
        if self.is_open:
            self.figure.canvas.flush_events()

    def close(self) -> None:
        """Close the figure; repeated calls are harmless."""
        if self.figure is not None and not self._closed:
            self._closed = True
            self._dragging = False
            self._plt.close(self.figure)

    def _handle_click(self, event) -> None:
        if (
            self.is_open
            and event.button == 1
            and event.inaxes in self.axes
            and event.xdata is not None
        ):
            self._dragging = True
            self._on_seek(float(event.xdata))

    def _handle_motion(self, event) -> None:
        if (
            self.is_open
            and self._dragging
            and event.inaxes in self.axes
            and event.xdata is not None
        ):
            self._on_seek(float(event.xdata))

    def _handle_release(self, event) -> None:
        self._dragging = False

    def _handle_slider(self, value: float) -> None:
        if self.is_open:
            self._on_seek(float(value))

    def _handle_key(self, event) -> None:
        if self.is_open and event.key is not None:
            self._on_key(event.key)

    def _handle_close(self, event) -> None:
        self._closed = True
        self._dragging = False


class WrenchLivePlot:
    """Display the wrench measured by a running MuJoCo episode.

    The class deliberately keeps Matplotlib imports inside ``__init__`` so
    importing the simulator or its tests does not select a GUI backend.  A
    caller owns the rollout loop: :meth:`update` appends one sample and
    :meth:`pump` processes pending window events without blocking the loop.
    ``max_points`` bounds the in-memory history and keeps the window usable
    during long evaluations.
    """

    def __init__(
        self,
        title: str = "Live wrench",
        max_points: int = 400,
        backend: str | None = None,
        *,
        allow_headless: bool = False,
    ) -> None:
        if isinstance(max_points, (bool, np.bool_)) or not isinstance(
            max_points, (int, np.integer)
        ) or int(max_points) < 1:
            raise ValueError("max_points must be a positive integer")

        self.max_points = int(max_points)
        self._times: list[float] = []
        self._wrench: list[np.ndarray] = []
        self._closed = False
        self.figure = None
        self._plt = None
        self._lines = []
        self._prediction_lines = []
        self._points = []
        self._status = None
        self._prediction_times = np.empty((0,), dtype=float)
        self._prediction_wrench = np.empty((0, 6), dtype=float)

        # Backend selection must happen before pyplot is imported.  In
        # production a GUI backend is required; ``allow_headless`` is only for
        # tests that use the Agg renderer.
        import matplotlib

        try:
            if backend is not None:
                matplotlib.use(backend)
            from matplotlib import pyplot as plt

            self._plt = plt
            selected_backend = str(matplotlib.get_backend())
            if not allow_headless and not WrenchReplayPlot._is_interactive_backend(
                selected_backend
            ):
                raise RuntimeError(
                    f"Matplotlib backend {selected_backend!r} cannot open an interactive "
                    "wrench window. Select a GUI backend such as TkAgg and run with "
                    "an available desktop display."
                )

            self.figure, self.axes = plt.subplots(2, 1, sharex=True, figsize=(11, 6))
            self.figure.suptitle(title)
            manager = self.figure.canvas.manager
            if manager is not None:
                manager.set_window_title(title)

            colors = ("tab:red", "tab:green", "tab:blue")
            for row, (axis, prefix, units) in enumerate(
                zip(self.axes, ("F", "T"), ("N", "N.m"))
            ):
                for component, color in zip("xyz", colors):
                    line, = axis.plot(
                        [], [], color=color, linewidth=1.35, label=f"{prefix}{component} measured"
                    )
                    prediction_line, = axis.plot(
                        [], [], color=color, linewidth=1.15, linestyle="--",
                        alpha=0.75, label=f"{prefix}{component} forecast",
                    )
                    point, = axis.plot([], [], "o", color=color, markersize=6)
                    self._lines.append(line)
                    self._prediction_lines.append(prediction_line)
                    self._points.append(point)
                axis.set_ylabel(f"{'Force' if row == 0 else 'Torque'} [{units}]")
                axis.grid(True, alpha=0.2)
                axis.legend(loc="upper right", ncol=3)
                axis.set_autoscalex_on(False)

            self.axes[-1].set_xlabel("Time [s]")
            self.axes[-1].set_xlim(0.0, 1.0)
            self._status = self.figure.text(
                0.99, 0.94, "Live wrench | waiting for samples", ha="right", va="top", fontsize=10
            )
            self.figure.tight_layout(rect=(0, 0.03, 1, 0.92))
            self.figure.canvas.mpl_connect("close_event", self._handle_close)
            if not allow_headless:
                plt.show(block=False)
                self.pump()
        except Exception as exc:
            if self.figure is not None and self._plt is not None:
                self._plt.close(self.figure)
            self._closed = True
            raise RuntimeError(
                "Could not open the interactive wrench plot. "
                "Use a GUI backend such as TkAgg with a working desktop display. "
                f"Details: {exc}"
            ) from exc

    @property
    def is_open(self) -> bool:
        """Whether the live window is still available for updates."""
        return bool(
            not self._closed
            and self.figure is not None
            and self._plt is not None
            and self._plt.fignum_exists(self.figure.number)
        )

    @property
    def sample_count(self) -> int:
        return len(self._times)

    @property
    def times(self) -> np.ndarray:
        return np.asarray(self._times, dtype=float)

    @property
    def wrench(self) -> np.ndarray:
        if not self._wrench:
            return np.empty((0, 6), dtype=float)
        return np.asarray(self._wrench, dtype=float)

    def update(self, timestamp: float, wrench: np.ndarray) -> None:
        """Append one finite ``[Fx,Fy,Fz,Tx,Ty,Tz]`` sample and redraw."""
        timestamp_array = np.asarray(timestamp)
        if timestamp_array.ndim != 0:
            raise ValueError("timestamp must be a finite scalar")
        try:
            timestamp = float(timestamp)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("timestamp must be a finite scalar") from exc
        values = np.asarray(wrench, dtype=float)
        if values.shape != (6,):
            raise ValueError(f"wrench must have shape (6,), got {values.shape}")
        if not np.isfinite(timestamp) or not np.isfinite(values).all():
            raise ValueError("timestamp and wrench must contain only finite values")
        if self._times and timestamp < self._times[-1]:
            raise ValueError("timestamps must be nondecreasing")

        self._times.append(timestamp)
        self._wrench.append(values.copy())
        if len(self._times) > self.max_points:
            del self._times[:-self.max_points]
            del self._wrench[:-self.max_points]

        if not self.is_open:
            return
        times = np.asarray(self._times, dtype=float)
        values = self.wrench
        for column, line in enumerate(self._lines):
            line.set_data(times, values[:, column])
        current_time = float(times[-1])
        current_values = values[-1]
        for column, point in enumerate(self._points):
            point.set_data([current_time], [current_values[column]])

        all_times = times
        if self._prediction_times.size:
            all_times = np.concatenate((all_times, self._prediction_times))
        if all_times.size == 1:
            left, right = current_time - 0.5, current_time + 0.5
        else:
            left, right = float(np.min(all_times)), float(np.max(all_times))
            if right <= left:
                left, right = left - 0.5, right + 0.5
        self.axes[-1].set_xlim(left, right)
        for axis in self.axes:
            axis.relim()
            axis.autoscale_view(scalex=False, scaley=True)
            axis.set_autoscalex_on(False)
        force_norm = float(np.linalg.norm(current_values[:3]))
        if self._prediction_times.size:
            self._status.set_text(
                f"Live wrench + CAD-WM forecast  |  t = {current_time:.3f} s  |  "
                f"|F| = {force_norm:.3f} N  |  forecast to {self._prediction_times[-1]:.3f} s"
            )
        else:
            self._status.set_text(
                f"Live wrench  |  t = {current_time:.3f} s  |  "
                f"|F| = {force_norm:.3f} N  |  sample {len(times)}"
            )
        self.figure.canvas.draw_idle()

    def update_prediction(
        self,
        anchor_timestamp: float,
        future_offsets: np.ndarray,
        future_wrench: np.ndarray,
    ) -> None:
        """Show one CAD-WM forecast horizon anchored at the current frame.

        ``future_wrench`` is physical ``[Fx,Fy,Fz,Tx,Ty,Tz]`` with one row per
        positive ``future_offsets`` entry.  The forecast is drawn as dashed
        lines over the measured history and is never fed back into MuJoCo.
        """
        anchor_array = np.asarray(anchor_timestamp)
        if anchor_array.ndim != 0:
            raise ValueError("anchor_timestamp must be a finite scalar")
        try:
            anchor_timestamp = float(anchor_timestamp)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("anchor_timestamp must be a finite scalar") from exc
        offsets = np.asarray(future_offsets, dtype=float).reshape(-1)
        values = np.asarray(future_wrench, dtype=float)
        if offsets.ndim != 1 or offsets.size < 1:
            raise ValueError("future_offsets must be a nonempty one-dimensional array")
        if values.shape != (offsets.size, 6):
            raise ValueError(
                f"future_wrench must have shape ({offsets.size}, 6), got {values.shape}"
            )
        if (not np.isfinite(anchor_timestamp) or not np.isfinite(offsets).all()
                or not np.isfinite(values).all()):
            raise ValueError("forecast timestamp and wrench must contain only finite values")
        if np.any(offsets < 0) or np.any(np.diff(offsets) <= 0):
            raise ValueError("future_offsets must be strictly increasing and nonnegative")

        self._prediction_times = anchor_timestamp + offsets
        self._prediction_wrench = values.copy()
        if not self.is_open:
            return
        for column, line in enumerate(self._prediction_lines):
            line.set_data(self._prediction_times, self._prediction_wrench[:, column])
        all_times = self._prediction_times
        if self._times:
            all_times = np.concatenate((np.asarray(self._times, dtype=float), all_times))
        left, right = float(np.min(all_times)), float(np.max(all_times))
        if right <= left:
            left, right = left - 0.5, right + 0.5
        self.axes[-1].set_xlim(left, right)
        for axis in self.axes:
            axis.relim()
            axis.autoscale_view(scalex=False, scaley=True)
            axis.set_autoscalex_on(False)
        self._status.set_text(
            f"Live wrench + CAD-WM forecast  |  t = {anchor_timestamp:.3f} s  |  "
            f"forecast to {self._prediction_times[-1]:.3f} s"
        )
        self.figure.canvas.draw_idle()

    def reset(self) -> None:
        """Clear the current episode while keeping the window open."""
        self._times.clear()
        self._wrench.clear()
        self._prediction_times = np.empty((0,), dtype=float)
        self._prediction_wrench = np.empty((0, 6), dtype=float)
        if not self.is_open:
            return
        for line, prediction_line, point in zip(
            self._lines, self._prediction_lines, self._points
        ):
            line.set_data([], [])
            prediction_line.set_data([], [])
            point.set_data([], [])
        for axis in self.axes:
            axis.relim()
            axis.autoscale_view(scalex=False, scaley=True)
            axis.set_autoscalex_on(False)
        self.axes[-1].set_xlim(0.0, 1.0)
        self._status.set_text("Live wrench | waiting for samples")
        self.figure.canvas.draw_idle()

    def pump(self) -> None:
        """Process pending GUI events without blocking the rollout loop."""
        if self.is_open:
            self.figure.canvas.flush_events()

    def close(self) -> None:
        """Close the window; repeated calls are harmless."""
        if self.figure is not None and self._plt is not None and not self._closed:
            self._closed = True
            self._plt.close(self.figure)

    def _handle_close(self, event) -> None:
        self._closed = True
