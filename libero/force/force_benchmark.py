import os
import re
import select
import sys
import threading
import xml.etree.ElementTree as ET
import inspect
from pathlib import Path
import numpy as np
import h5py
from typing import Tuple, Optional, List, Dict, Any, Union, Sequence
from tqdm import tqdm

import torch
import mujoco

try:
    import requests
except ImportError:  # keep the simulator evaluator usable in the lean env
    import json as _json
    from urllib import request as _urlrequest

    class _URLResponse:
        def __init__(self, response):
            self._response = response
            self.status_code = int(getattr(response, "status", 200))
            self.ok = 200 <= self.status_code < 400
            self._body = None

        def _read_body(self):
            if self._body is None:
                self._body = self._response.read()
            return self._body

        def raise_for_status(self):
            if not self.ok:
                raise RuntimeError(f"HTTP request failed ({self.status_code}): {self.text[:2000]}")

        @property
        def text(self):
            return self._read_body().decode("utf-8", errors="replace")

        def json(self):
            return _json.loads(self.text)

    class _URLSession:
        @staticmethod
        def get(url):
            return _URLResponse(_urlrequest.urlopen(url))

        @staticmethod
        def post(url, json=None):
            body = _json.dumps(json).encode("utf-8")
            request = _urlrequest.Request(
                url, data=body, headers={"Content-Type": "application/json"}, method="POST"
            )
            return _URLResponse(_urlrequest.urlopen(request))

    class _RequestsCompat:
        Session = _URLSession

    requests = _RequestsCompat()

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from libero.force.modules import WrenchObsWrapper
from libero.force.physics_helper import PhysicsHelper
from libero.force.visual_helper import VisualHelper
from libero.force.hdf5_replayer import HDF5Replayer
from libero.force.hdf5_to_task import get_task_from_hdf5
from libero.force.utils import dict_apply

from robokit.connects.protocols import StepRequestFromPolicy, StepRequestFromEvaluator
from robokit.data_manager.utils_multiview import cat_multiview_video_with_another
from robokit.debug_utils.images import concatenate_rgb_images, save_frames_as_video, plot_force_sensor_wrt_time
from robokit.debug_utils.printer import print_batch
from scripts.wrench_replay_plot import CameraLiveView, WrenchLivePlot


class SingleForceEnv:
    def __init__(
            self,
            hdf5_read_path: str = None,
            camera_hw: tuple = (128, 128),
            env_seed: int = 0,
            libero_code_root: str = None,
            bddl_path: str = None,
            benchmark_task_suite_name: str = None,  # in `libero_{10/90/object/goal/spatial}`
            show_viewer: bool = False,
            terminal_interactive: bool = False,
            show_wrench_plot: bool = False,
            wrench_plot_backend: str = "TkAgg",
            show_camera_view: bool = False,
            camera_view_backend: str = "TkAgg",
            wrench_plot_max_points: int = 400,
            # PhysicsHelper params
            phy_keywords: tuple = ("cabinet", "drawer", "door", "basket", "object", "tool"),
    ):
        self.hdf5_read_path = hdf5_read_path
        self.camera_hw = camera_hw
        self.env_seed = env_seed
        # Keep all files used to reconstruct an HDF5 scene anchored to the
        # checkout supplied by the caller.  LIBERO's ``get_libero_path``
        # reads a process-global ~/.libero/config.yaml, which may point to a
        # different checkout and is not needed when the demo stores its full
        # flattened state.
        checkout_root = (
            Path(libero_code_root).expanduser()
            if libero_code_root is not None
            else Path(__file__).resolve().parents[2]
        )
        self.libero_code_root = str(checkout_root.resolve())
        self.show_viewer = bool(show_viewer)
        self.terminal_interactive = bool(terminal_interactive)
        self.show_wrench_plot = bool(show_wrench_plot)
        self.wrench_plot_backend = wrench_plot_backend
        self.wrench_plot_max_points = wrench_plot_max_points
        self.show_camera_view = bool(show_camera_view)
        self.camera_view_backend = camera_view_backend
        self.wrench_plot = None
        self.camera_view = None
        self._wrench_plot_disabled = False
        self._wrench_plot_error_reported = False
        self._camera_view_disabled = False
        self._camera_view_error_reported = False
        self.interactive = self.show_viewer or self.terminal_interactive
        # Viewer state is attached after the off-screen env is initialized.
        self.mujoco_viewer = None
        self._viewer_sim = None
        # Native MuJoCo viewer callbacks run on the GUI thread.  Use events so
        # the evaluator thread can safely pause, reset, and resume rollouts.
        self._viewer_started = threading.Event()
        self._viewer_reset_requested = threading.Event()
        self._terminal_stop_requested = threading.Event()
        self._terminal_input_closed = threading.Event()
        self._terminal_thread = None
        if not self.interactive:
            self._viewer_started.set()

        self.step_cnt = 0
        self._fdp_previous_qvel = None

        # 0) Create env
        assert (hdf5_read_path is not None or
                bddl_path is not None or
                benchmark_task_suite_name is not None), \
            "hdf5_read_path or bddl_path must be provided."
        if hdf5_read_path is not None:
            task_info = get_task_from_hdf5(hdf5_read_path)
            '''
            suite_name: libero_90
            task_id: 23
            task_name: KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer
            task_language: close the bottom drawer of the cabinet and open the top drawer
            bddl_file_from_h5: libero/libero/bddl_files/libero_90/KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer.bddl
            '''
            task_suite_name = task_info["suite_name"]
            task_id = task_info["task_id"]
            task_name = task_info["task_name"]
            task_language = task_info["task_language"]
            task_bddl_file_path = task_info["bddl_file_from_h5"]
            task_env_kwargs = task_info["env_kwargs"]

            self.bddl_path = self._resolve_checkout_file(task_bddl_file_path)
            self.task_name = task_name
            self.task_language = task_language
            self.hdf5_env_kwargs = task_env_kwargs
            self.hdf5_env_kwargs.update({
                "bddl_file_name": self.bddl_path,
                "camera_heights": self.camera_hw[0],
                "camera_widths": self.camera_hw[1],
            })
            if "controller_configs" in self.hdf5_env_kwargs:
                del self.hdf5_env_kwargs["controller_configs"]

            task_suite = benchmark.get_benchmark_dict()[task_suite_name]()
            task = task_suite.get_task(task_id)
            self.task_suite_name = task_suite_name
            self.task_id = int(task_id)
            self.task_suite = task_suite
            # HDF5 evaluation restores states[0] directly.  Delay loading the
            # benchmark init file until a caller explicitly requests the
            # benchmark fallback; this avoids depending on ~/.libero for the
            # normal data-driven path.
            self.init_states = None
            print("[DEBUG] benchmark init states will be loaded lazily")

        elif bddl_path is not None:
            self.bddl_path = self._resolve_checkout_file(bddl_path)
            # Keep the standalone BDDL constructor usable by socket callers
            # that only need an instruction string.  A direct BDDL path has
            # no benchmark task metadata, so derive a stable human-readable
            # fallback from its filename.
            self.task_name = Path(self.bddl_path).stem
            self.task_language = self.task_name.replace("_", " ")
            self.init_states = None
            self.hdf5_env_kwargs = {
                "bddl_file_name": self.bddl_path,
                "camera_heights": self.camera_hw[0],
                "camera_widths": self.camera_hw[1],
            }
        elif benchmark_task_suite_name is not None:
            print("[Warning] Using benchmark_task_suite_name to get BDDL path.")
            task_suite = benchmark.get_benchmark_dict()[benchmark_task_suite_name]()
            task = task_suite.get_task(0)  # just to get bddl path
            self.task_suite_name = benchmark_task_suite_name
            self.task_id = 0
            self.task_suite = task_suite
            self.task_name = task.name
            self.task_language = task.language
            self.init_states = None
            self.bddl_path = self._resolve_checkout_file(
                os.path.join("libero", "libero", "bddl_files", task.problem_folder, task.bddl_file)
            )
            self.hdf5_env_kwargs = {
                "bddl_file_name": self.bddl_path,
                "camera_heights": self.camera_hw[0],
                "camera_widths": self.camera_hw[1],
            }

        self.init_env()
        if getattr(self, "show_viewer", False):
            self.start_viewer()
        if self.show_wrench_plot:
            self.wrench_plot = WrenchLivePlot(
                title=f"Live wrench | {self.task_name}",
                max_points=self.wrench_plot_max_points,
                backend=self.wrench_plot_backend,
            )
        if self.show_camera_view:
            self.camera_view = CameraLiveView(
                title=f"FDP cameras | {self.task_name}",
                backend=self.camera_view_backend,
            )

        # 1) PhysicsHelper for dynamics randomization
        self.physics_helper = PhysicsHelper(
            self.env.env.sim.model,
            object_keywords=phy_keywords,
            random_seed=self.env_seed,
        )
        self.visual_helper = VisualHelper(
            self.env.env.sim.model,
            random_seed=self.env_seed,
        )
        # Note: usage: apply_dynamics_shift; restore_original_params;

        print("[Info] SingleForceEnv initialized with BDDL:", self.bddl_path)

    def _resolve_checkout_file(self, value: str) -> str:
        """Resolve a LIBERO file against the requested checkout.

        HDF5 metadata may contain either ``libero/libero/...`` paths from this
        repository or the shorter ``libero/...`` form used by older exports.
        Absolute paths remain valid when they exist; otherwise the checkout
        candidates are tried before returning the first deterministic path so
        MuJoCo reports a useful missing-file error.
        """
        raw = os.path.expanduser(str(value))
        path = os.path.abspath(raw) if os.path.isabs(raw) else None
        if path is not None and os.path.isfile(path):
            return path
        root = self.libero_code_root
        candidates = []
        if path is not None:
            candidates.append(path)
        else:
            candidates.extend([
                os.path.join(root, raw),
                os.path.join(root, "libero", raw),
            ])
        for candidate in candidates:
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
        # Return the most natural checkout-relative path for a clear error
        # from the downstream environment if none of the candidates exists.
        return os.path.abspath(candidates[0] if candidates else os.path.join(root, raw))

    def _load_benchmark_init_states(self):
        """Load benchmark initial states only when the fallback is used."""
        if self.init_states is not None:
            return self.init_states
        suite = getattr(self, "task_suite", None)
        task_id = getattr(self, "task_id", None)
        suite_name = getattr(self, "task_suite_name", None)
        if suite is None or task_id is None or suite_name is None:
            raise RuntimeError("benchmark initial states are unavailable for this environment")

        task = suite.get_task(int(task_id))
        local_path = os.path.join(
            self.libero_code_root,
            "libero", "libero", "init_files", suite_name,
            task.init_states_file,
        )
        if not os.path.isfile(local_path):
            # Support callers that pass the package directory itself as the
            # checkout root, and finally preserve LIBERO's configured-path
            # fallback for installations outside this repository.
            local_path = os.path.join(
                self.libero_code_root, "libero", "init_files", suite_name,
                task.init_states_file,
            )
        if os.path.isfile(local_path):
            try:
                self.init_states = torch.load(local_path, weights_only=False)
            except TypeError:  # PyTorch < 2.6
                self.init_states = torch.load(local_path)
        else:
            self.init_states = suite.get_task_init_states(int(task_id))
        if len(self.init_states) == 0:
            raise RuntimeError(f"No benchmark initial states found in {local_path}")
        print("[DEBUG] loaded init_states cnt:", len(self.init_states))
        return self.init_states

    def init_env(self):
        base_env = OffScreenRenderEnv(
            **self.hdf5_env_kwargs
        )
        self.env = WrenchObsWrapper(
            base_env,
            force_sensor="gripper0_force_ee",
            torque_sensor="gripper0_torque_ee",
        )
        self.reset_env()

    def _disable_wrench_plot(self, error: Exception) -> None:
        """Disable only plotting after a GUI error so rollout errors propagate."""
        if not self._wrench_plot_error_reported:
            print(f"[wrench-plot] disabled after update error: {error}", flush=True)
            self._wrench_plot_error_reported = True
        self._wrench_plot_disabled = True
        if self.wrench_plot is not None:
            try:
                self.wrench_plot.close()
            except Exception:
                pass

    def _reset_wrench_plot(self) -> None:
        if self.wrench_plot is None or self._wrench_plot_disabled:
            return
        try:
            self.wrench_plot.reset()
        except Exception as exc:
            self._disable_wrench_plot(exc)

    def _record_wrench(self, wrench: np.ndarray, timestamp_s: float) -> None:
        """Feed one measured sample to the optional GUI without altering control."""
        if self.wrench_plot is None or self._wrench_plot_disabled:
            return
        try:
            self.wrench_plot.update(timestamp_s, wrench)
            self.wrench_plot.pump()
        except Exception as exc:
            self._disable_wrench_plot(exc)

    def _disable_camera_view(self, error: Exception) -> None:
        """Disable only camera plotting after a GUI error."""
        if not self._camera_view_error_reported:
            print(f"[camera-view] disabled after update error: {error}", flush=True)
            self._camera_view_error_reported = True
        self._camera_view_disabled = True
        if self.camera_view is not None:
            try:
                self.camera_view.close()
            except Exception:
                pass

    def _reset_camera_view(self) -> None:
        if self.camera_view is None or self._camera_view_disabled:
            return
        try:
            self.camera_view.reset()
        except Exception as exc:
            self._disable_camera_view(exc)

    def record_camera_view(
        self, timestamp_s: float, side_image: np.ndarray, wrist_image: np.ndarray
    ) -> None:
        """Display the exact unmodified frames sent to FDP."""
        if self.camera_view is None or self._camera_view_disabled:
            return
        try:
            self.camera_view.update(timestamp_s, side_image, wrist_image)
        except Exception as exc:
            self._disable_camera_view(exc)

    def record_wrench_prediction(
        self,
        anchor_timestamp_s: float,
        future_offsets_s: np.ndarray,
        future_wrench: np.ndarray,
    ) -> None:
        """Feed the current CAD-WM horizon to the optional GUI."""
        if self.wrench_plot is None or self._wrench_plot_disabled:
            return
        try:
            self.wrench_plot.update_prediction(
                anchor_timestamp_s, future_offsets_s, future_wrench
            )
            self.wrench_plot.pump()
        except Exception as exc:
            self._disable_wrench_plot(exc)

    def close_env(self):
        self.stop_terminal_listener()
        if self.wrench_plot is not None:
            try:
                self.wrench_plot.close()
            except Exception:
                pass
            self.wrench_plot = None
        if self.camera_view is not None:
            try:
                self.camera_view.close()
            except Exception:
                pass
            self.camera_view = None
        if self.mujoco_viewer is not None:
            self.mujoco_viewer.close()
            self.mujoco_viewer = None
            self._viewer_sim = None
        self.env.close()

    def start_viewer(self):
        """Open MuJoCo's native passive viewer for the live simulator state."""
        sim = self.env.env.sim
        if self.mujoco_viewer is not None and self._viewer_sim is sim:
            return
        import mujoco.viewer
        # robosuite may rebuild the simulator during reset().  A passive viewer
        # keeps the model/data objects supplied at launch, so rebind it when
        # the active sim changes or the window would display a frozen episode.
        if self.mujoco_viewer is not None:
            self.mujoco_viewer.close()
            self.mujoco_viewer = None
        model = getattr(sim.model, "_model", sim.model)
        data = getattr(sim.data, "_data", sim.data)
        self.mujoco_viewer = mujoco.viewer.launch_passive(
            model, data, key_callback=self._viewer_key_callback
        )
        self._viewer_sim = sim
        # ``OffScreenRenderEnv`` is configured with ``render_collision_mesh=False``
        # for the existing LIBERO scene.  MuJoCo's native viewer does not inherit
        # that robosuite flag and otherwise draws the Panda's group-0 collision
        # meshes on top of the visual meshes.  The overlapping collision meshes can
        # look like self-interference (especially around the wrist) and obscure the
        # inference rollout.  Keep physics collisions enabled, but show the same
        # visual geometry as the project's camera renderer.
        viewer_opt = getattr(self.mujoco_viewer, "opt", None)
        if viewer_opt is not None and hasattr(viewer_opt, "geomgroup"):
            viewer_opt.geomgroup[:] = 0
            # LIBERO/robosuite visual meshes use geom group 1.
            if viewer_opt.geomgroup.shape[0] > 1:
                viewer_opt.geomgroup[1] = 1
        self.mujoco_viewer.sync()
        print("[viewer] paused; press 's' to start inference, 'i' to reset")

    def _viewer_key_callback(self, key: int):
        """Handle the small interactive control surface for the passive viewer."""
        key = int(key)
        if key in (ord("s"), ord("S")):
            self.request_inference_start(source="viewer")
        elif key in (ord("i"), ord("I")):
            # Reset is consumed by the evaluator thread, which owns the env and
            # policy HTTP session.  Pausing here avoids stepping while reset is
            # being applied from the callback thread.
            self.request_reset(source="viewer")

    def request_inference_start(self, source: str = "control"):
        self._viewer_started.set()
        print(f"[{source}] inference started", flush=True)

    def request_reset(self, source: str = "control"):
        self._viewer_reset_requested.set()
        self._viewer_started.clear()
        print(f"[{source}] reset requested; press 's' to resume", flush=True)

    def _handle_terminal_key(self, key: str):
        key = key.strip()[:1].lower()
        if key == "s":
            self.request_inference_start(source="terminal")
        elif key == "i":
            self.request_reset(source="terminal")

    def _terminal_input_loop(self):
        """Read one-character s/i commands without blocking the evaluator."""
        stream = sys.stdin
        is_tty = bool(getattr(stream, "isatty", lambda: False)())
        tty_stream = None
        old_termios = None
        fd = None
        try:
            # ``conda run`` can detach stdin from the controlling terminal.
            # Reopen /dev/tty so s/i still work in the user's terminal.
            if not is_tty:
                try:
                    tty_stream = open("/dev/tty", "rb", buffering=0)
                    stream = tty_stream
                    is_tty = True
                except OSError:
                    tty_stream = None
            if is_tty:
                import termios
                import tty

                fd = stream.fileno()
                old_termios = termios.tcgetattr(fd)
                tty.setcbreak(fd)
                while not self._terminal_stop_requested.is_set():
                    readable, _, _ = select.select([fd], [], [], 0.2)
                    if not readable:
                        continue
                    key = os.read(fd, 1).decode("utf-8", errors="ignore")
                    self._handle_terminal_key(key)
            else:
                # This keeps --interactive usable in a pipe or a test harness.
                while not self._terminal_stop_requested.is_set():
                    line = stream.readline()
                    if line == "":
                        break
                    self._handle_terminal_key(line)
        except (EOFError, OSError, select.error):
            pass
        finally:
            if old_termios is not None and fd is not None:
                try:
                    import termios
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_termios)
                except (OSError, termios.error):
                    pass
            if tty_stream is not None:
                try:
                    tty_stream.close()
                except OSError:
                    pass
            self._terminal_input_closed.set()

    def start_terminal_listener(self):
        if not self.terminal_interactive:
            return
        if self._terminal_thread is not None and self._terminal_thread.is_alive():
            return
        self._terminal_stop_requested.clear()
        self._terminal_input_closed.clear()
        self._terminal_thread = threading.Thread(
            target=self._terminal_input_loop,
            name="libero-terminal-control",
            daemon=True,
        )
        self._terminal_thread.start()
        print("[terminal] paused; press 's' to start inference, 'i' to reset", flush=True)

    def stop_terminal_listener(self):
        self._terminal_stop_requested.set()
        thread = self._terminal_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.5)
        self._terminal_thread = None

    def wait_for_inference_start(self) -> bool:
        """Wait for s in terminal/viewer interactive mode."""
        if not self.interactive:
            return True
        while True:
            if self._viewer_started.wait(timeout=0.1):
                return True
            # A viewer remains a valid control surface even when conda run
            # detached stdin and no /dev/tty is available.
            if (
                self._terminal_input_closed.is_set()
                and self.terminal_interactive
                and not self.show_viewer
            ):
                return False
            if self.show_viewer and (
                self.mujoco_viewer is None or not self.mujoco_viewer.is_running()
            ):
                return False

    # Backwards-compatible name for callers that used the viewer-only API.
    wait_for_viewer_start = wait_for_inference_start

    def consume_reset_request(self) -> bool:
        """Return and clear a pending terminal/viewer reset request."""
        if not self._viewer_reset_requested.is_set():
            return False
        self._viewer_reset_requested.clear()
        return True

    consume_viewer_reset = consume_reset_request

    def clear_inference_start(self):
        """Pause before the next episode in interactive mode."""
        if self.interactive:
            self._viewer_started.clear()

    clear_viewer_start = clear_inference_start

    def reset_env(self, env_seed: Optional[int] = None):
        self.env_seed = env_seed if env_seed is not None else self.env_seed
        self.env.seed(self.env_seed)
        self.env.reset()
        self.step_cnt = 0
        self._fdp_previous_qvel = None
        self._reset_wrench_plot()
        self._reset_camera_view()
        if hasattr(self, "visual_helper"):
            self.visual_helper.clear_dynamic_light_base()
            self.visual_helper.clear_camera_shift_base()
        if self.mujoco_viewer is not None:
            self.start_viewer()

    def reset_to_benchmark_init(self, init_state_id=0, settle_steps=5, env_seed: Optional[int] = None):
        self.reset_env(env_seed=env_seed)
        init_states = self._load_benchmark_init_states()
        self.env.set_init_state(init_states[init_state_id % len(init_states)])
        assert settle_steps > 0, "settle_steps must be greater than 0."
        for _ in range(settle_steps):
            obs, _, _, _ = self.env.step(np.zeros(7, dtype=np.float32))
        return obs

    def reset_to_dataset_init(
            self,
            state,
            model_file: Optional[str] = None,
            env_seed: Optional[int] = None,
    ):
        """Reset the simulator and apply only a dataset episode's first state."""
        self.reset_env(env_seed=env_seed)
        return self.set_dataset_state_without_reset(state, model_file=model_file)

    def apply_dataset_model_file(self, model_file: Optional[str]):
        """Restore per-demo body transforms stored in a LIBERO model XML.

        The flattened MuJoCo state contains joint state, time and actuator
        state, but static fixture placement (for example the cabinet body) is
        stored in the demo's ``model_file`` attribute.  Reapply those body
        transforms to the current BDDL-built model before restoring the state.
        This avoids loading old absolute mesh paths while preserving the
        recorded scene placement.
        """
        if model_file is None:
            return
        if isinstance(model_file, bytes):
            model_file = model_file.decode("utf-8")
        try:
            root = ET.fromstring(str(model_file))
        except (ET.ParseError, TypeError, ValueError) as exc:
            raise ValueError("Dataset model_file is not valid MuJoCo XML") from exc

        model = self.env.env.sim.model
        matched = 0
        updated = 0
        for body in root.iter("body"):
            name = body.get("name")
            if not name:
                continue
            try:
                body_id = int(model.body_name2id(name))
            except Exception:
                # The current checkout may omit an auxiliary body that was
                # present in the recording.  Matching bodies are still
                # restored below.
                continue
            matched += 1
            if body.get("pos") is not None:
                position = np.fromstring(body.get("pos"), sep=" ", dtype=np.float64)
                if position.shape != (3,) or not np.isfinite(position).all():
                    raise ValueError(f"Dataset model_file body {name!r} has invalid pos")
                model.body_pos[body_id] = position
                updated += 1
            if body.get("quat") is not None:
                quaternion = np.fromstring(body.get("quat"), sep=" ", dtype=np.float64)
                if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
                    raise ValueError(f"Dataset model_file body {name!r} has invalid quat")
                model.body_quat[body_id] = quaternion
                updated += 1
        if matched == 0:
            raise ValueError("Dataset model_file contains no bodies matching the current MuJoCo model")
        self.env.env.sim.forward()

    def set_dataset_state_without_reset(self, state, model_file: Optional[str] = None):
        """Apply a dataset state after a visual XML reset rebuilt the simulator."""
        state = np.asarray(state, dtype=np.float64)
        current_state = np.asarray(self.env.env.get_sim_state())
        if state.ndim != 1 or state.shape != current_state.shape:
            raise ValueError(
                "Dataset initial state shape mismatch: "
                f"expected {current_state.shape}, got {state.shape}"
            )
        self._reset_wrench_plot()
        self._reset_camera_view()
        self.apply_dataset_model_file(model_file)
        self.env.set_flattened_state_and_forward(state)
        self.step_cnt = 0
        self._fdp_previous_qvel = None
        if self.mujoco_viewer is not None:
            self.start_viewer()
            self.mujoco_viewer.sync()
        return self.regenerate_obs_from_current_state()

    def set_benchmark_init_without_reset(self, init_state_id=0, settle_steps=5):
        self.step_cnt = 0
        self._fdp_previous_qvel = None
        self._reset_wrench_plot()
        self._reset_camera_view()
        self.visual_helper.clear_dynamic_light_base()
        self.visual_helper.clear_camera_shift_base()
        init_states = self._load_benchmark_init_states()
        self.env.set_init_state(init_states[init_state_id % len(init_states)])
        if self.mujoco_viewer is not None:
            self.start_viewer()
        if hasattr(self.env, "build_reader_if_needed"):
            self.env.build_reader_if_needed()
        assert settle_steps > 0, "settle_steps must be greater than 0."
        for _ in range(settle_steps):
            obs, _, _, _ = self.env.step(np.zeros(7, dtype=np.float32))
        return obs

    def step(self, action_D: np.ndarray):
        obs, reward, done, info = self.env.step(action_D)
        self.step_cnt += 1
        wrench = obs.get("wrench_ee") if isinstance(obs, dict) else None
        if wrench is not None:
            self._record_wrench(
                wrench,
                timestamp_s=float(self.step_cnt) * self.control_timestep,
            )
        if self.mujoco_viewer is not None and self.mujoco_viewer.is_running():
            self.start_viewer()
            self.mujoco_viewer.sync()
        success = self.check_success()
        return obs, reward, done, {**info, "success": success}

    @property
    def control_timestep(self) -> float:
        """Seconds between simulator observations used by the policy stream."""
        offscreen_env = self.env.env
        base_env = getattr(offscreen_env, "env", None)
        value = getattr(offscreen_env, "control_timestep", None)
        if value is None and base_env is not None:
            value = getattr(base_env, "control_timestep", None)
        if value is None:
            frequency = getattr(offscreen_env, "control_freq", None)
            if frequency is None and base_env is not None:
                frequency = getattr(base_env, "control_freq", None)
        else:
            frequency = None
        if value is None:
            if frequency is None:
                raise RuntimeError("LIBERO environment does not expose control frequency")
            value = 1.0 / float(frequency)
        value = float(value)
        if not np.isfinite(value) or value <= 0:
            raise RuntimeError(f"Invalid simulator control timestep: {value}")
        return value

    def check_success(self):
        return bool(self.env.env.check_success())

    def regenerate_obs_from_current_state(self):
        sim_state = self.env.env.get_sim_state()
        obs = self.env.env.regenerate_obs_from_state(sim_state)
        obs["wrench_ee"] = self.env.read_wrench_from_sim()
        self._record_wrench(
            obs["wrench_ee"],
            timestamp_s=float(self.step_cnt) * self.control_timestep,
        )
        return obs

    def fdp_state(self, obs, timestamp_s: float):
        """Return the causal state contract consumed by the FDP interface.

        The legacy socket protocol only carried TCP pose and wrench. FDP's
        CAD-WM requires the seven arm joint q/dq/tau vectors, so extract them
        from robosuite's reference indices and attach the current EEF pose.
        """
        base_env = self.env.env
        robots = getattr(base_env, "robots", None)
        if not robots:
            raise RuntimeError("LIBERO environment has no robot state for FDP")
        robot = robots[0]
        q_indices = np.asarray(robot._ref_joint_pos_indexes[:7], dtype=np.int64)
        dq_indices = np.asarray(robot._ref_joint_vel_indexes[:7], dtype=np.int64)
        sim = base_env.sim
        q = np.asarray(sim.data.qpos[q_indices], dtype=np.float32)
        dq = np.asarray(sim.data.qvel[dq_indices], dtype=np.float32)
        # The training HDF5 stores inverse-dynamics torque.  ``mj_forward``
        # does not guarantee that qacc describes the observed q/dq stream, so
        # never pass a stale qacc into mj_inverse.  Use the causal backward
        # acceleration estimate from the previous simulator frame instead.
        qvel_full = np.asarray(sim.data.qvel, dtype=np.float64).copy()
        control_dt = self.control_timestep
        if not np.isfinite(control_dt) or control_dt <= 0:
            raise RuntimeError(f"Invalid simulator control timestep: {control_dt}")
        if self._fdp_previous_qvel is None:
            qacc_full = np.zeros_like(qvel_full)
        else:
            qacc_full = (qvel_full - self._fdp_previous_qvel) / control_dt
        previous_qacc = np.asarray(sim.data.qacc, dtype=np.float64).copy()
        sim.data.qacc[:] = qacc_full
        try:
            mujoco.mj_inverse(sim.model._model, sim.data._data)
            tau_all = np.asarray(getattr(sim.data, "qfrc_inverse", []), dtype=np.float32)
            if tau_all.size:
                tau = tau_all[dq_indices]
            else:
                tau = np.asarray(sim.data.qfrc_actuator[dq_indices], dtype=np.float32)
            if not np.isfinite(tau).all():
                tau = np.asarray(sim.data.qfrc_actuator[dq_indices], dtype=np.float32)
        finally:
            sim.data.qacc[:] = previous_qacc
        self._fdp_previous_qvel = qvel_full
        state = {
            "timestamp_s": float(timestamp_s),
            "q": q.tolist(),
            "dq": dq.tolist(),
            "tau": tau.tolist(),
            "eef_pos": np.asarray(obs["robot0_eef_pos"], dtype=np.float32).tolist(),
            "eef_quat": np.asarray(obs["robot0_eef_quat"], dtype=np.float32).tolist(),
        }
        controller = getattr(robot, "controller", None)
        action_min = getattr(controller, "output_min", None)
        action_max = getattr(controller, "output_max", None)
        if action_min is not None and action_max is not None:
            action_min = np.asarray(action_min, dtype=np.float32).reshape(-1)
            action_max = np.asarray(action_max, dtype=np.float32).reshape(-1)
            if (action_min.shape == (6,) and action_max.shape == (6,)
                    and np.isfinite(action_min).all() and np.isfinite(action_max).all()
                    and np.all(action_max > action_min)):
                state["osc_action_min"] = action_min.tolist()
                state["osc_action_max"] = action_max.tolist()
        if "wrench_ee" in obs:
            state["wrench"] = np.asarray(obs["wrench_ee"], dtype=np.float32).tolist()
        for key, value in (("q", q), ("dq", dq), ("tau", tau)):
            if value.shape != (7,) or not np.isfinite(value).all():
                raise RuntimeError(f"FDP state {key} must be finite with shape [7]")
        return state

    def apply_physics_shift(
            self,
            frictionloss_scale: Tuple[float, float] = (0.5, 2.0),
            damping_scale: Tuple[float, float] = (0.5, 2.0),
            sliding_friction_scale: Tuple[float, float] = (0.7, 1.3),
            gravity_z_range: Tuple[float, float] = (-10.3, -9.3),
            tweak_solref_range: Tuple[float, float] = (0.8, 1.2),
            rng: np.random.RandomState = None,
    ):
        self.physics_helper.apply_dynamics_shift(
            model=self.env.env.sim.model,  # NOTE: don't know why, but we have to resend the sim.model here
            frictionloss_scale=frictionloss_scale,
            damping_scale=damping_scale,
            sliding_friction_scale=sliding_friction_scale,
            gravity_z_range=gravity_z_range,
            tweak_solref_range=tweak_solref_range,
            rng=rng,
        )
        self.env.env.sim.forward()
        print(f"[Info] Applied physics shift. gravity={self.env.env.sim.model.opt.gravity}, "
              f"friction/damping/sliding={damping_scale}, solref={tweak_solref_range}.")

    def restore_physics(self):
        self.physics_helper.restore_original_params()

    def apply_visual_shift(
            self,
            table_center_xyz: tuple = None,
            darken: bool = False,
            color_light: bool = False,
            texture: bool = False,
            texture_dir: str = "images/dtd",
            camera_shift: bool = False,
            rng: np.random.RandomState = None,
            verbose: bool = False,
    ) -> Dict[str, Any]:
        if (not darken) and (not color_light) and (not texture) and (not camera_shift):
            return {"applied": False, "reset_applied": False}
        self.visual_helper.clear_dynamic_light_base()
        self.visual_helper.clear_camera_shift_base()

        texture_info = None
        flashlight_info = None
        color_info = None
        camera_info = None

        if texture:
            texture_info = self.visual_helper.apply_table_texture_via_xml_wrapper(
                texture_dir=texture_dir,
                env=self.env,
                preferred_material_names=("desk", "table_texture", "table_mat"),
                apply_to_all_keyword_textures=True,
                reset_env=True,
                rng=rng,
                verbose=verbose,
            )
            # reset_from_xml_string rebuilds sim/model, refresh helper model refs.
            self.physics_helper.model = self.env.env.sim.model
            if getattr(self.env, "reader", None) is not None:
                self.env.reader = None

        if darken or color_light:
            light_info = self.visual_helper.apply_color_and_darken(
                model=self.env.env.sim.model,
                rng=rng,
                darken=darken,
                color_light=color_light,
                table_center_xyz=table_center_xyz if table_center_xyz is not None else (0.0, 0.0, 0.9),
                table_half_size_xy=(0.05, 0.10),
                preserve_luminance=True,
                verbose=verbose,
            )
            flashlight_info = light_info.get("flashlight_info")
            color_info = light_info.get("color_info")

        if camera_shift:
            camera_info = self.visual_helper.apply_camera_shift(
                model=self.env.env.sim.model,
                rng=rng,
                verbose=verbose,
            )

        self.env.env.sim.forward()
        return {
            "applied": True,
            "reset_applied": bool(texture_info.get("reset_applied", False)) if texture_info is not None else False,
            "texture_info": texture_info,
            "flashlight_info": flashlight_info,
            "color_info": color_info,
            "camera_info": camera_info,
        }

    def restore_visual(self):
        self.visual_helper.restore_original_params(model=self.env.env.sim.model)
        self.visual_helper.clear_dynamic_light_base()
        self.visual_helper.clear_camera_shift_base()


class LiberoForceSocketEvaluator:
    def __init__(
            self,
            # SingleForceEnv params
            hdf5_read_path: str = None,
            camera_hw: tuple = (128, 128),
            env_seed: int = 0,
            libero_code_root: str = None,
            bddl_path: str = None,
            benchmark_task_suite_name: str = None,  # in `libero_{10/90/object/goal/spatial}`
            # PhysicsHelper params
            phy_keywords: tuple = ("cabinet", "drawer", "door", "basket", "object", "tool", "microwave"),
            # Eval params
            test_cnt: int = 20,
            test_start_seed: int = 10000,
            test_max_steps: int = 400,
            # Socket related
            policy_url='http://localhost:6006',
            send_per_frames: int = 1,
            interface: str = "legacy",
            show_viewer: bool = False,
            terminal_interactive: bool = False,
            show_wrench_plot: bool = False,
            wrench_plot_backend: str = "TkAgg",
            show_camera_view: bool = False,
            camera_view_backend: str = "TkAgg",
    ):
        self.env = SingleForceEnv(
            hdf5_read_path=hdf5_read_path,
            camera_hw=camera_hw,
            env_seed=env_seed,
            libero_code_root=libero_code_root,
            bddl_path=bddl_path,
            benchmark_task_suite_name=benchmark_task_suite_name,
            show_viewer=show_viewer,
            terminal_interactive=terminal_interactive,
            show_wrench_plot=show_wrench_plot,
            wrench_plot_backend=wrench_plot_backend,
            show_camera_view=show_camera_view,
            camera_view_backend=camera_view_backend,
            phy_keywords=phy_keywords,
        )
        self.language_goal = self.env.task_name.replace("_", " ")
        self.terminal_interactive = bool(terminal_interactive)

        # The HDF5 demo states contain the scene placement used by LIBERO-FT.
        # We use only states[0] for episode initialization; policy actions still
        # drive every subsequent simulator step.
        self.dataset_initial_states = []
        self.dataset_initial_model_files = []
        if hdf5_read_path is not None:
            with h5py.File(hdf5_read_path, "r") as h5_file:
                demo_keys = sorted(
                    (key for key in h5_file["data"].keys() if key.startswith("demo_")),
                    key=lambda key: int(key.split("_", 1)[1]),
                )
                for demo_key in demo_keys:
                    demo_group = h5_file["data"][demo_key]
                    if "states" in demo_group and len(demo_group["states"]) > 0:
                        self.dataset_initial_states.append(
                            np.asarray(demo_group["states"][0], dtype=np.float64)
                        )
                        model_file = demo_group.attrs.get(
                            "model_file", h5_file["data"].attrs.get("model_file")
                        )
                        if isinstance(model_file, bytes):
                            model_file = model_file.decode("utf-8")
                        self.dataset_initial_model_files.append(
                            None if model_file is None else str(model_file)
                        )
        if self.dataset_initial_states:
            print(
                "[LiberoForceSocketEvaluator] loaded dataset initial states:",
                len(self.dataset_initial_states),
            )

        # Eval related
        self.test_cnt = test_cnt
        self.test_start_seed = test_start_seed
        self.current_test_id = 0
        self.test_max_steps = test_max_steps

        # Socket related
        policy_url = str(policy_url).strip()
        if "://" not in policy_url:
            policy_url = f"http://{policy_url}"
        self.policy_url = policy_url.rstrip("/")
        self.send_per_frames = send_per_frames
        if interface not in {"legacy", "fdp"}:
            raise ValueError("interface must be legacy or fdp")
        self.interface = interface
        self.send_cnt = 0
        self.cache_actions_B_T_D = None
        self.latest_future_wrench = None
        self.latest_future_offsets_s = None
        self.latest_prediction_timestamp_s = None
        self.latest_prediction_updated = False
        self.http_session = requests.Session()
        self.task_instruction = None
        self.cache_actions_B_T_D = None

        self.frame_buffer = {
            'primary_rgb': [],
            'gripper_rgb': [],
            'joint_state': [],
        }  # some policies require 2 or more observation frames

    def _clear_stream_state(self):
        self.send_cnt = 0
        self.cache_actions_B_T_D = None
        self.latest_future_wrench = None
        self.latest_future_offsets_s = None
        self.latest_prediction_timestamp_s = None
        self.latest_prediction_updated = False
        self.cache_frames_B_H_W_C = None
        self.frame_buffer = {
            "primary_rgb": [],
            "gripper_rgb": [],
            "joint_state": [],
        }

    def _set_cache_horizon(self, value):
        """Normalize cache metadata returned by different robokit versions."""
        if value is None:
            value = self.send_per_frames
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) != 1:
                raise ValueError(f"max_cache_action must be a scalar, got {value!r}")
            value = value[0]
        try:
            numeric = float(value)
            value = int(numeric)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid max_cache_action={value!r}") from exc
        if not np.isfinite(numeric) or numeric != value:
            raise ValueError(f"max_cache_action must be an integer, got {value!r}")
        if value < 1:
            raise ValueError(f"max_cache_action must be positive, got {value}")
        self.send_per_frames = value
        return value

    def _validate_policy_timing(self, response):
        """Reject an FDP service whose native rate differs from MuJoCo."""
        if self.interface != "fdp" or not isinstance(response, dict):
            return
        policy_dt = response.get("timestamp_step_sec")
        if policy_dt is None:
            return  # Older services do not advertise the contract.
        try:
            policy_dt = float(policy_dt)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid policy timestamp_step_sec={policy_dt!r}") from exc
        sim_dt = self.env.control_timestep
        if not np.isfinite(policy_dt) or policy_dt <= 0 or not np.isclose(
                policy_dt, sim_dt, atol=1e-6, rtol=1e-3
        ):
            raise RuntimeError(
                f"FDP and MuJoCo rates differ: policy dt={policy_dt}, simulator dt={sim_dt}"
            )

    def init_socket(self, task_instruction: str = None) -> int:
        resp = self.http_session.get(f"{self.policy_url}/init")
        resp.raise_for_status()
        resp = resp.json()
        self._validate_policy_timing(resp)

        max_cache_action = self._set_cache_horizon(resp.get("max_cache_action"))
        self._clear_stream_state()

        if task_instruction is not None:
            self.task_instruction = task_instruction
        else:
            self.task_instruction = self.language_goal.lower()
            self.task_instruction = re.sub(r'(\d)(\s)', r' \1 ', self.task_instruction)
            self.task_instruction = self.task_instruction.replace("_", " ")
        print("[LiberoForceSocketEvaluator] Connected to policy server:", self.policy_url,
              f"max_cache_action={max_cache_action}, send_per_frames={self.send_per_frames}",
              f"task_instruction={self.task_instruction}")
        return max_cache_action

    def send_reset(self, task_instruction: str = None) -> int:
        assert self.task_instruction is not None, "Please call init_socket first."
        self.task_instruction = task_instruction if task_instruction is not None else self.task_instruction

        resp = self.http_session.get(f"{self.policy_url}/reset")
        resp.raise_for_status()
        resp = resp.json()
        self._validate_policy_timing(resp)

        max_cache_action = self._set_cache_horizon(resp.get("max_cache_action"))
        self._clear_stream_state()

        print("[LiberoForceSocketEvaluator] Reset policy server:", self.policy_url,
              f"max_cache_action={max_cache_action}, send_per_frames={self.send_per_frames},"
              f" task_instruction={self.task_instruction[:50]}")
        return max_cache_action

    def send_obs_and_get_action(self,
                                image_B_T_H_W_C: np.ndarray,
                                stage_flag: int,
                                robot_states_B_T_D: np.ndarray = None,
                                fdp_state: dict = None,
                                ) -> np.ndarray:
        assert self.task_instruction is not None, "Please call init_socket first."
        self.latest_prediction_updated = False

        if self.send_cnt % self.send_per_frames == 0:
            video_B_v2_H_W_C = image_B_T_H_W_C  # (B,Ts,H,W,3) uint8
            encode_kwargs = {
                "instruction": self.task_instruction,
                "stage_flag": stage_flag,
                "gt_video": video_B_v2_H_W_C,  # (B,Ts,H,W,3) uint8
                "tcp_state": robot_states_B_T_D,  # (B,Ts,D) float32, not used
                "num_camera_views": 2,
            }
            # Older robokit releases exposed max_cache_action on the request;
            # newer releases moved this contract to the /init response.
            if "max_cache_action" in inspect.signature(
                    StepRequestFromEvaluator.encode_from_raw
            ).parameters:
                encode_kwargs["max_cache_action"] = self.send_per_frames
            request_to_policy = StepRequestFromEvaluator.encode_from_raw(**encode_kwargs)

            # print(f"[LiberoForceSocketEvaluator] sending frames, i={self.send_cnt}, per={self.send_per_frames}, "
            #       f"i%per={self.send_cnt % self.send_per_frames}, video_B_v2_H_W_C:{video_B_v2_H_W_C.shape}, ")
            sending_dict = request_to_policy.model_dump(mode="json")
            if self.interface == "fdp":
                if fdp_state is None:
                    raise ValueError("fdp interface requires fdp_state")
                # Keep the original robokit payload intact for compatibility;
                # the new fdp service consumes this explicit causal state.
                sending_dict["fdp_state"] = fdp_state
            response = self.http_session.post(
                f"{self.policy_url}/step",
                json=sending_dict
            )
            if not response.ok:
                # Include the policy-side diagnostic; a bare HTTP status makes
                # malformed state/image payloads unnecessarily hard to debug.
                detail = getattr(response, "text", "")
                raise RuntimeError(
                    f"policy request failed ({response.status_code}): {detail[:2000]}"
                )

            response = response.json()
            if not isinstance(response, dict) or "action" not in response:
                raise RuntimeError("policy response must contain an action field")
            raw_actions = StepRequestFromPolicy(action=response["action"]).decode_to_raw()["action"]  # (B,H,D)
            if (raw_actions.ndim != 3 or raw_actions.shape[0] != 1
                    or raw_actions.shape[1] < 1 or raw_actions.shape[2] < 1
                    or not np.isfinite(raw_actions).all()):
                raise RuntimeError(f"policy returned invalid action shape {raw_actions.shape}")
            if response.get("max_cache_action") is not None:
                self._set_cache_horizon(response["max_cache_action"])
            if self.interface == "fdp":
                future_wrench = response.get("future_wrench")
                future_offsets = response.get("future_wrench_offsets_s")
                if future_wrench is None or future_offsets is None:
                    self.latest_future_wrench = None
                    self.latest_future_offsets_s = None
                    self.latest_prediction_timestamp_s = None
                else:
                    future_wrench = np.asarray(future_wrench, dtype=np.float32)
                    future_offsets = np.asarray(future_offsets, dtype=np.float64).reshape(-1)
                    if (future_wrench.ndim != 2 or future_wrench.shape[1] != 6
                            or future_wrench.shape[0] != future_offsets.size
                            or future_wrench.shape[0] < 1
                            or not np.isfinite(future_wrench).all()
                            or not np.isfinite(future_offsets).all()
                            or np.any(future_offsets < 0)
                            or np.any(np.diff(future_offsets) <= 0)):
                        raise RuntimeError(
                            "FDP returned an invalid future_wrench forecast"
                        )
                    self.latest_future_wrench = future_wrench
                    self.latest_future_offsets_s = future_offsets
                    self.latest_prediction_timestamp_s = (
                        None if fdp_state is None else float(fdp_state["timestamp_s"])
                    )
                    self.latest_prediction_updated = True
            # A server may advertise a larger cache than the returned chunk;
            # never index beyond the actual response horizon.
            self.send_per_frames = min(self.send_per_frames, int(raw_actions.shape[1]))

            # assert raw_actions.shape[0] == self.send_per_frames
            self.cache_actions_B_T_D = raw_actions
            self.cache_frames_B_H_W_C = None
        else:  # don't send anything, to save network bandwidth
            pass

        ret_actions = self.cache_actions_B_T_D[:, self.send_cnt % self.send_per_frames]  # (B,D)
        ret_actions_B_1_D = ret_actions[:, None]  # [B,D] -> (B,1,D)

        self.send_cnt += 1
        return ret_actions_B_1_D

    def run_eval(self,
                 device: Union[torch.device, str] = "cuda",
                 vis_pred_video=False,
                 load_cameras: List[str] = None,
                 close_online: bool = True,
                 friction_scale: float = 1.,
                 gz_scale: float = 1.,
                 solref_scale: float = 1.,
                 table_center_xyz: tuple = None,
                 darken: bool = False,
                 color_light: bool = False,
                 texture: bool = False,
                 texture_dir: str = "images/dtd/",
                 camera_shift: bool = False,
                 vis_rollout_video: bool = False,
                 **kwargs
    ):
        if kwargs.get("selected_light_id", None) is not None:
            darken = True
        use_dynamic_color_light = bool(color_light)
        domain_shift_tags = []
        if color_light:
            domain_shift_tags.append("vc")
        if texture:
            domain_shift_tags.append("vt")
        if camera_shift:
            domain_shift_tags.append("vv")
        if darken:
            domain_shift_tags.append("vd")
        domain_shift_str = "+".join(domain_shift_tags) if domain_shift_tags else "none"

        env = self.env
        env.start_terminal_listener()
        env.reset_env()

        self.send_reset()
        stage_flag = 0  # 0:cold start, 1:hot start

        # print("env_runner: ", self.language_goal)
        eval_results = {
            "success_list": [],
        }
        test_seed = self.test_start_seed
        for test_idx in range(self.test_cnt):
            # a) init envs
            ## NOTE: some policies' reset requires a long time, so we only reset env once
            ## You need to make sure this is correct for your policy.
            # self.send_reset()
            self.current_test_id = test_idx
            stage_flag = 0  # each episode starts cold after a policy reset
            # FDPInference enforces a monotonic native-rate timestamp.  Each
            # benchmark initial state is a separate rollout, so reset the
            # stateful FDP service at every episode boundary before sending the
            # first frame of the new rollout.
            if self.interface == "fdp":
                self.send_reset()
            dataset_state = None
            if self.dataset_initial_states:
                dataset_index = test_idx % len(self.dataset_initial_states)
                dataset_state = self.dataset_initial_states[dataset_index]
                dataset_model_file = self.dataset_initial_model_files[dataset_index]
                _ = env.reset_to_dataset_init(
                    dataset_state, model_file=dataset_model_file, env_seed=test_seed
                )
                print(f"[DEBUG] initial scene source: HDF5 demo_{dataset_index}")
            else:
                _ = env.reset_to_benchmark_init(
                    init_state_id=test_idx,
                    env_seed=test_seed
                )

            # add visual randomization here if needed
            visual_info = None
            if darken or texture or camera_shift:
                visual_info = env.apply_visual_shift(
                    table_center_xyz=table_center_xyz,
                    darken=darken,
                    color_light=False,
                    texture=texture,
                    texture_dir=texture_dir,
                    camera_shift=camera_shift,
                    rng=np.random.RandomState(test_seed),
                    verbose=False,
                )
                # texture reset rebuilds env internals; restore benchmark init state
                # without calling env.reset() again (which may overwrite patched XML).
                if visual_info.get("reset_applied", False):
                    if dataset_state is not None:
                        env.set_dataset_state_without_reset(
                            dataset_state, model_file=dataset_model_file
                        )
                    else:
                        env.set_benchmark_init_without_reset(init_state_id=test_idx)

            # add physics randomization here if needed
            if friction_scale != 1.0 or gz_scale != 1.0 or solref_scale != 1.0:
                if gz_scale > 1.0:
                    possible_gz_range = (-9.81 * (1 + gz_scale), -9.81)
                elif gz_scale < 1.0:
                    possible_gz_range = (-9.81 * (gz_scale ** 2), -9.81 * gz_scale)
                else:
                    possible_gz_range = None

                if friction_scale > 1.0:
                    possible_friction_range = (friction_scale, friction_scale ** 2)
                elif friction_scale < 1.0:
                    possible_friction_range = (friction_scale ** 2, friction_scale)
                else:
                    possible_friction_range = None

                if solref_scale > 1.0:
                    possible_solref_range = (solref_scale, solref_scale ** 2)
                elif solref_scale < 1.0:
                    possible_solref_range = (solref_scale ** 2, solref_scale)
                else:
                    possible_solref_range = None

                env.apply_physics_shift(
                    frictionloss_scale=possible_friction_range,
                    damping_scale=possible_friction_range,
                    sliding_friction_scale=possible_friction_range,
                    gravity_z_range=possible_gz_range,
                    tweak_solref_range=possible_solref_range,
                    rng=np.random.RandomState(test_seed),
                )
            if use_dynamic_color_light:
                env.visual_helper.capture_dynamic_light_base(
                    model=env.env.sim.model,
                )
                env.visual_helper.apply_dynamic_rainbow_light(
                    model=env.env.sim.model,
                    # step_idx=env.step_cnt,
                    step_idx=1,
                )
                env.env.sim.forward()

            print(
                "[DEBUG] Visual (ep="
                f"{test_idx+1}/{self.test_cnt}, seed={test_seed}): "
                f"darken(vd)={bool(darken)} "
                f"texture(vt)={bool(texture)} "
                f"color(vc)={bool(use_dynamic_color_light)} "
                f"view(vv)={bool(camera_shift)}"
            )
            obs = env.regenerate_obs_from_current_state()

            if not env.wait_for_inference_start():
                print("[interactive] input/viewer closed; stopping evaluation", flush=True)
                break

            env_name = env.task_name
            cur_sr = float(np.array(eval_results['success_list']).mean()) * 100. if len(
                eval_results['success_list']) > 0 else 0.0
            pbar = tqdm(
                total=self.test_max_steps,
                desc=f"Eval {env_name[:20]} ({test_idx+1}/{self.test_cnt}) "
                     f"(SR={cur_sr:.2f}%)",
                leave=False,
                # mininterval=self.tqdm_interval_sec,  # e.g, 5.0 seconds
            )
            vis_frames = []
            vis_forces = []
            done = False
            success = False

            # b) start rollout
            while not done:
                if env.consume_reset_request():
                    # The evaluator thread owns both the simulator and policy
                    # session.  Reset to the same demo initial state, then wait
                    # for the next s command before sending another action.
                    pbar.close()
                    self.send_reset()
                    stage_flag = 0
                    if dataset_state is not None:
                        obs = env.reset_to_dataset_init(
                            dataset_state, model_file=dataset_model_file, env_seed=test_seed
                        )
                    else:
                        obs = env.reset_to_benchmark_init(
                            init_state_id=test_idx,
                            env_seed=test_seed,
                        )
                    if use_dynamic_color_light:
                        env.visual_helper.capture_dynamic_light_base(model=env.env.sim.model)
                        env.visual_helper.apply_dynamic_rainbow_light(
                            model=env.env.sim.model,
                            step_idx=1,
                        )
                        env.env.sim.forward()
                    vis_frames = []
                    vis_forces = []
                    done = False
                    success = False
                    pbar = tqdm(
                        total=self.test_max_steps,
                        desc=f"Eval {env_name[:20]} ({test_idx+1}/{self.test_cnt}) (reset)",
                        leave=False,
                    )
                    if not env.wait_for_inference_start():
                        done = True
                        break
                    continue
                np_obs_dict = dict(obs)
                '''
                [DEBUG] 1st obs::                                    
                --robot0_eef_pos, <class 'numpy.ndarray'>, shape=(3,), min=-0.2077, max=1.1671, dtype=float64
                --robot0_eef_quat, <class 'numpy.ndarray'>, shape=(4,), min=-0.0367, max=0.9993, dtype=float64
                --robot0_gripper_qpos, <class 'numpy.ndarray'>, shape=(2,), min=-0.0207, max=0.0207, dtype=float64
                --robot0_eye_in_hand_image, <class 'numpy.ndarray'>, shape=(128, 128, 3), min=0.0000, max=240.0000, dtype=uint8
                --agentview_image, <class 'numpy.ndarray'>, shape=(128, 128, 3), min=0.0000, max=255.0000, dtype=uint8                                                                                                                                                                         
                --wrench_ee, <class 'numpy.ndarray'>, shape=(6,), min=-5.0506, max=0.1079, dtype=float32
                '''
                vis_frames.append(
                    concatenate_rgb_images(
                        np.flipud(np_obs_dict["agentview_image"]),
                        np.flipud(np_obs_dict["robot0_eye_in_hand_image"]),
                        vertical=True,
                    )
                )
                vis_forces.append(obs["wrench_ee"])

                # device transfer
                obs_dict = dict_apply(
                    np_obs_dict, lambda x: torch.from_numpy(x).to(device=device)
                )
                # add B,T dims
                for key in obs_dict:
                    obs_dict[key] = obs_dict[key].unsqueeze(0).unsqueeze(0)  # (H,W,C) -> (1,1,H,W,C)
                # print_batch("[DEBUG] transferred obs:", np_obs_dict)

                # not used
                if load_cameras is None:
                    load_cameras = ["agentview_image", "robot0_eye_in_hand_image"]
                latest_obs_B_T_H_W_C_tensors = []
                for camera in load_cameras:
                    latest_obs_B_T_H_W_C_tensors.append(
                        obs_dict[camera]
                    )  # (B,Ts,H,W,C)
                latest_obs_B_T_H_W_C = (torch.cat(
                    latest_obs_B_T_H_W_C_tensors, dim=1
                ).cpu().numpy()).astype(np.uint8)
                # (B,V*Ts,H,W,C) uint8 [0,255] WARN: already uint8 in [0,255], so no need to *255.

                # latest_robot_states_B_T_D = torch.cat([
                #     obs_dict["robot0_joint_pos"],
                #     obs_dict["robot0_gripper_qpos"][:, :, :1],  # only use one finger state
                # ], dim=2).cpu().numpy()  # (B,T,8) float32
                ee_pos = obs_dict["robot0_eef_pos"].cpu().numpy()  # (B,T,3)
                ee_ori = HDF5Replayer.quat_xyzw_to_rotvec(obs_dict["robot0_eef_quat"].cpu().numpy().squeeze())  # (3)
                ee_ori = ee_ori[None, None, :]  # (B,Ts,3)
                latest_robot_states_B_T_D = np.concatenate((ee_pos, ee_ori), axis=2)  # (B,T,6)
                latest_force_B_T_D = obs_dict["wrench_ee"].cpu().numpy()  # (B,Ts,6) float32
                # NOTE: need to be concatenated with force data
                latest_robot_states_B_T_D = np.concatenate(
                    (latest_robot_states_B_T_D, latest_force_B_T_D), axis=2
                )  # (B,Ts,6+6)

                cur_primary_rgb = np_obs_dict["agentview_image"]  # (H,W,C) uint8
                cur_gripper_rgb = np_obs_dict["robot0_eye_in_hand_image"]  # (H,W,C) uint8
                env.record_camera_view(
                    timestamp_s=float(env.step_cnt) * env.control_timestep,
                    side_image=cur_primary_rgb,
                    wrist_image=cur_gripper_rgb,
                )
                cur_joint_state = latest_robot_states_B_T_D[0, 0]  # (6+6) with force
                fdp_state = None
                if self.interface == "fdp":
                    fdp_state = env.fdp_state(
                        obs,
                        timestamp_s=float(env.step_cnt) * env.control_timestep,
                    )

                # work like a queue, only keep the latest `buffer_size` frames
                buffer_size = self.send_per_frames
                for key, val in zip(
                        ['primary_rgb', 'gripper_rgb', 'joint_state'],
                        [cur_primary_rgb, cur_gripper_rgb, cur_joint_state]
                ):
                    if len(self.frame_buffer[key]) >= buffer_size:
                        self.frame_buffer[key].pop(0)
                    self.frame_buffer[key].append(val)

                # construct output, add pad before the data (len = buffer_size)
                def pad_to_buffer(data_list, pad_with):
                    """
                    :return: (buffer_size,D) np.ndarray or List
                    """
                    padded = [pad_with for _ in range(buffer_size - len(data_list))] + data_list
                    return np.stack(padded) if isinstance(pad_with, np.ndarray) else padded

                cur_primary_rgb = pad_to_buffer(
                    self.frame_buffer['primary_rgb'],
                    self.frame_buffer['primary_rgb'][0]
                )  # (T,H,W,C)
                cur_gripper_rgb = pad_to_buffer(
                    self.frame_buffer['gripper_rgb'],
                    self.frame_buffer['gripper_rgb'][0]
                )
                cur_joint_state = pad_to_buffer(
                    self.frame_buffer['joint_state'],
                    self.frame_buffer['joint_state'][0]
                )  # (T,D)
                gt_video = cat_multiview_video_with_another(
                    video_A_B_VT_H_W_C=cur_primary_rgb[None, :],  # (1,Ts,H,W,C)
                    video_B_B_VT_H_W_C=cur_gripper_rgb[None, :],  # (1,Ts,H,W,C)
                    sample_n_views=1,
                )  # (1,V*Ts,H,W,C)
                # print("[DEBUG] gt_video:", gt_video.shape, gt_video.dtype, gt_video.min(), gt_video.max())
                # print("[DEBUG] cur_joint_state[None, :]:", cur_joint_state[None, :].shape, cur_joint_state.dtype,)

                # run policy
                with torch.no_grad():
                    out_action_B_1_D = self.send_obs_and_get_action(
                        image_B_T_H_W_C=gt_video,  # (B,V*Ts,H,W,C) uint8
                        stage_flag=stage_flag,
                        robot_states_B_T_D=cur_joint_state[None, :],  # (B,Ts,6+6)
                        fdp_state=fdp_state,
                    )
                    if (
                        self.interface == "fdp"
                        and self.latest_prediction_updated
                        and self.latest_prediction_timestamp_s is not None
                        and self.latest_future_offsets_s is not None
                        and self.latest_future_wrench is not None
                    ):
                        env.record_wrench_prediction(
                            self.latest_prediction_timestamp_s,
                            self.latest_future_offsets_s,
                            self.latest_future_wrench,
                        )
                    # print("[DEBUG] out_action_B_1_D:", out_action_B_1_D.shape, out_action_B_1_D.dtype, out_action_B_1_D.min(), out_action_B_1_D.max())
                    action_dict = {
                        "action": torch.from_numpy(out_action_B_1_D).to(device=device),  # (B,1,2)
                    }

                    if close_online:
                        pass
                    else:
                        stage_flag = 1  # after first step, all are hot start

                # device_transfer
                np_action_dict = dict_apply(
                    action_dict, lambda x: x.detach().to("cpu").numpy()
                )

                action = np_action_dict["action"]  # (1, T, Da)
                # print("[DEBUG] action:", action, action.shape, action.dtype, action.min(), action.max())
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")

                # step env
                env_action: np.ndarray = action[0, 0]  # (Da,)
                # if self.abs_action:
                #     env_action = self.undo_transform_action(action)
                if use_dynamic_color_light:
                    env.visual_helper.apply_dynamic_rainbow_light(
                        model=env.env.sim.model,
                        step_idx=env.step_cnt,
                    )
                    env.env.sim.forward()

                obs, reward, done, info = env.step(env_action)

                # for i in range(len(reward)):
                #     if reward[i] == 1:
                #         done[i] = True
                #
                # done = np.all(done)
                if reward == 1:
                    done = True
                    success = True

                if info["success"]:
                    done = True
                    success = True

                if env.step_cnt >= self.test_max_steps:
                    done = True  # failure due to timeout
                    success = False

                # # past_action = action
                # past_action_list.append(action)
                # if len(past_action_list) > 2:
                #     past_action_list.pop(0)

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()
            vis_force_T_D = np.array(vis_forces)  # (T,6)
            # NOTE: add a setting here
            # vis_force_frames, *_ = plot_force_sensor_wrt_time(vis_force_T_D, only_last_frame=True)
            # vis_frames = [concatenate_rgb_images(f, vis_force_frames[-1], resize_ratio=2.) for f in vis_frames]
            if vis_rollout_video:
                save_frames_as_video(
                    vis_frames,
                    save_path=f"/home/geyuan/code/LIBERO-FT/debug/libero_rollout/"
                              f"shift={domain_shift_str}_merge_liberoft_eval{test_idx:02d}_suc={success:1d}.mp4"
                )
            eval_results["success_list"].append(float(success))
            eval_results[f"test/seed{test_seed}_mean_score"] = float(success)

            # # collect data for this round
            # all_video_paths[this_global_slice] = env.render()[this_local_slice]
            # all_rewards[this_global_slice] = env.call("get_attr", "reward")[
            #     this_local_slice
            # ]

            if friction_scale != 1.0 or gz_scale != 1.0 or solref_scale != 1.0:
                env.restore_physics()
            env.clear_inference_start()
            test_seed += 1

        # clear out video buffer
        _ = env.reset_env()
        env.close_env()

        return eval_results

        # # log
        # max_rewards = collections.defaultdict(list)
        # log_data = dict()
        # # results reported in the paper are generated using the commented out line below
        # # which will only report and average metrics from first n_envs initial condition and seeds
        # # fortunately this won't invalidate our conclusion since
        # # 1. This bug only affects the variance of metrics, not their mean
        # # 2. All baseline methods are evaluated using the same code
        # # to completely reproduce reported numbers, uncomment this line:
        # # for i in range(len(self.env_fns)):
        # # and comment out this line
        # for i in range(n_inits):
        #     seed = self.env_seeds[i]
        #     prefix = self.env_prefixs[i]
        #     max_reward = np.max(all_rewards[i])
        #     max_rewards[prefix].append(max_reward)
        #     log_data[prefix + f"sim_max_reward_{seed}"] = max_reward
        #
        #     # visualize sim
        #     video_path = all_video_paths[i]
        #     if video_path is not None:
        #         sim_video = wandb.Video(video_path)
        #         log_data[prefix + f"sim_video_{seed}"] = sim_video

        # # log aggregate metrics
        # for prefix, value in max_rewards.items():
        #     name = prefix + "mean_score"
        #     value = np.mean(value)
        #     log_data[name] = value

        # return log_data


def eval_one_episode(env, policy, horizon=400, record_images=False):
    """
    Return: success(bool), episode_data(dict)
    episode_data: includes wrench norm、reward、images...
    """
    images = []
    wrench_norms = []

    obs = env.reset()
    if record_images:
        images.append(obs["agentview_image"].copy())

    for t in range(horizon):
        action = policy(obs)  # (7,)
        obs, reward, done, info = env.step(action)

        if record_images:
            images.append(obs["agentview_image"].copy())

        if "wrench_ee" in obs:
            wrench_norms.append(float(np.linalg.norm(obs["wrench_ee"][:3])))

        if hasattr(env, "env") and hasattr(env.env, "check_success"):
            if env.env.check_success():
                return True, {"images": images, "wrench_norms": wrench_norms}

    # check again at the end of episode, in case some policies only set success=True at the end
    success = env.env.check_success() if (hasattr(env, "env") and hasattr(env.env, "check_success")) else False
    return bool(success), {"images": images, "wrench_norms": wrench_norms}


def evaluate_task_suite(
    task_suite_name="libero_10",
    task_id=0,
    num_episodes=50,
    seed=0,
    use_wrench=False,
    record_images=False,
    horizon=400,
):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()

    task = task_suite.get_task(task_id)
    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"),
        task.problem_folder,
        task.bddl_file,
    )
    print("[task]", task.name)
    print("[lang]", task.language)
    print("[bddl]", task_bddl_file)

    init_states = task_suite.get_task_init_states(task_id)  # i.e. 50 for LIBERO
    assert len(init_states) >= num_episodes, f"init_states={len(init_states)} < num_episodes={num_episodes}"

    env = make_env(task_bddl_file, use_wrench=use_wrench)
    env.seed(seed)

    successes = []
    for ep in tqdm(range(num_episodes), desc="eval"):
        obs = env.reset()

        # fix init_state like LIBERO benchmark
        env.set_init_state(init_states[ep])

        # stabilize the env for a few steps, in case some policies require stable initial observations
        for _ in range(5):
            obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))

        success, ep_data = eval_one_episode(
            env=env,
            policy=policy,
            horizon=horizon,
            record_images=record_images,
        )
        successes.append(int(success))

        # DEBUG: save video for the first episode if it fails, to check if the env and policy are working correctly
        # if record_images and (not success) and ep == 0:
        #     save_video(ep_data["images"], "debug.mp4", fps=10)

    env.close()

    sr = float(np.mean(successes))
    print(f"[result] success_rate={sr:.3f} ({sum(successes)}/{len(successes)})")
    return sr


# -------------------------
# a simple dummy policy for debugging
# -------------------------
class ZeroPolicy:
    def __call__(self, obs):
        return np.zeros(7, dtype=np.float32)


if __name__ == "__main__":
    policy = ZeroPolicy()
    evaluate_task_suite(
        task_suite_name="libero_10",
        task_id=0,
        num_episodes=50,
        seed=0,
        use_wrench=True,        # if add wrench into the observation
        record_images=False,    # if save videos
        horizon=400,
    )
