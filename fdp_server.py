"""HTTP ``fdp`` policy interface for LIBERO-FT.

The simulator evaluator sends two RGB views and the causal q/dq/tau state in
``fdp_state``.  The default measured-phase configuration also requires the
current six-channel ``fdp_state.wrench``.  The teacher itself emits absolute
xyz+xyzw poses; this service converts the selected predicted pose to LIBERO's 7-D
OSC delta action and returns one action per request.  A standard-library HTTP
server is used so the model environment does not need FastAPI or robokit
installed.

Start from the LIBERO-FT checkout, for example::

    conda activate fdp
    PYTHONPATH=$PWD python fdp_server.py \
      --config configs/inference/fdp_cad_wm.yaml \
      --checkpoint /abs/path/to/teacher/checkpoints/optimizer_step=00180000.ckpt \
      --wm-checkpoint /abs/path/to/cad_wm/checkpoints/step_00180000.pt \
      --port 7075
"""

from __future__ import annotations

import argparse
import base64
import io
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading

import numpy as np
from PIL import Image

from inference.config import load_config
from inference.fdp_inference import FDPInference
from inference.inference_base import InferenceObservation


def _unit_quaternion_xyzw(value, name):
    q = np.asarray(value, dtype=np.float32)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError(f"{name} must be a finite quaternion [4]")
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        raise ValueError(f"{name} has zero norm")
    return q / norm


def _quat_multiply_xyzw(left, right):
    x1, y1, z1, w1 = left
    x2, y2, z2, w2 = right
    return np.asarray((
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ), dtype=np.float32)


def _quat_conjugate_xyzw(q):
    return np.asarray((-q[0], -q[1], -q[2], q[3]), dtype=np.float32)


def _quat_to_rotvec_xyzw(q):
    q = _unit_quaternion_xyzw(q, "relative quaternion")
    # q and -q encode the same rotation; use the shortest axis-angle path.
    if q[3] < 0:
        q = -q
    w = float(np.clip(q[3], -1.0, 1.0))
    angle = 2.0 * float(np.arccos(w))
    sin_half = float(np.linalg.norm(q[:3]))
    if sin_half < 1e-7 or angle < 1e-7:
        return (2.0 * q[:3]).astype(np.float32)
    return (q[:3] * (angle / sin_half)).astype(np.float32)


_DEFAULT_OSC_ACTION_MIN = np.asarray(
    (-0.05, -0.05, -0.05, -0.5, -0.5, -0.5), dtype=np.float32
)
_DEFAULT_OSC_ACTION_MAX = np.asarray(
    (0.05, 0.05, 0.05, 0.5, 0.5, 0.5), dtype=np.float32
)
_DEFAULT_EXECUTION_ACTION_INDEX = 1


def _normalize_osc_delta(delta, action_min=None, action_max=None):
    """Map physical OSC deltas to robosuite's normalized [-1, 1] input."""
    low = _DEFAULT_OSC_ACTION_MIN if action_min is None else np.asarray(action_min, dtype=np.float32)
    high = _DEFAULT_OSC_ACTION_MAX if action_max is None else np.asarray(action_max, dtype=np.float32)
    if (low.shape != (6,) or high.shape != (6,) or not np.isfinite(low).all()
            or not np.isfinite(high).all() or not np.all(high > low)):
        raise ValueError("OSC action bounds must be finite six-vectors with max > min")
    delta = np.asarray(delta, dtype=np.float32)
    center = 0.5 * (high + low)
    half_range = 0.5 * (high - low)
    return np.clip((delta - center) / half_range, -1.0, 1.0).astype(np.float32)


def absolute_pose_to_libero_action(
        target, current_pos, current_quat, *, osc_action_min=None, osc_action_max=None
):
    """Convert FDP's absolute pose to normalized LIBERO OSC_POSE + gripper.

    The DP checkpoint predicts metres and an axis-angle rotation in physical
    units.  robosuite's controller API accepts normalized ``[-1, 1]`` values;
    its ``output_min`` / ``output_max`` bounds are supplied by the evaluator
    when available and the standard OSC_POSE limits are used otherwise.
    """
    target = np.asarray(target, dtype=np.float32)
    if target.shape not in ((7,), (8,)) or not np.isfinite(target).all():
        raise ValueError(f"FDP target must have shape [7] or [8], got {target.shape}")
    current_pos = np.asarray(current_pos, dtype=np.float32)
    if current_pos.shape != (3,) or not np.isfinite(current_pos).all():
        raise ValueError("current_pos must be a finite vector [3]")
    target_quat = _unit_quaternion_xyzw(target[3:7], "target quaternion")
    current_quat = _unit_quaternion_xyzw(current_quat, "current quaternion")
    relative_quat = _quat_multiply_xyzw(target_quat, _quat_conjugate_xyzw(current_quat))
    delta_rot = _quat_to_rotvec_xyzw(relative_quat)
    physical_delta = np.concatenate((target[:3] - current_pos, delta_rot)).astype(np.float32)
    arm_action = _normalize_osc_delta(
        physical_delta, action_min=osc_action_min, action_max=osc_action_max
    )
    gripper = float(target[7]) if target.shape == (8,) else 0.0
    if not np.isfinite(gripper):
        raise ValueError("gripper command must be finite")
    action = np.concatenate((arm_action, np.asarray((np.clip(gripper, -1.0, 1.0),), dtype=np.float32)))
    return action.astype(np.float32)


def _decode_image(encoded):
    try:
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception as exc:  # pragma: no cover - error is surfaced to client
        raise ValueError("gt_video contains an invalid base64 JPEG") from exc


def _decode_video(payload):
    videos = payload.get("gt_video")
    if not isinstance(videos, list) or len(videos) != 1 or not videos[0]:
        raise ValueError("gt_video must contain one non-empty batch")
    return [_decode_image(item) for item in videos[0]]


class FDPService:
    def __init__(self, config_path, checkpoint=None, backbone=None, seed=42,
                 wm_checkpoint=None):
        config = load_config(config_path)
        if checkpoint is not None:
            config["dp"]["checkpoint"] = str(Path(checkpoint).expanduser().resolve())
        if wm_checkpoint is not None:
            config["cad_wm"]["checkpoint"] = str(
                Path(wm_checkpoint).expanduser().resolve()
            )
        if backbone is not None:
            config["dp"]["backbone"] = str(Path(backbone).expanduser().resolve())
        self.config = config
        self.seed = int(seed)
        runtime = config.get("runtime", {})
        self.execution_action_index = int(
            runtime.get("execution_action_index", _DEFAULT_EXECUTION_ACTION_INDEX)
        )
        if self.execution_action_index < 0:
            raise ValueError("runtime.execution_action_index must be non-negative")
        self.engine = FDPInference.from_loaded_config(config)
        self.lock = threading.RLock()
        self.frame_index = 0
        self.initialized = False

    @property
    def dt(self):
        return float(self.engine.dt)

    @property
    def max_cache_action(self):
        # FDPInference must see every native-rate frame to keep CAD-WM causal.
        return 1

    def metadata(self):
        return {
            "max_cache_action": self.max_cache_action,
            "timestamp_step_sec": self.dt,
            "sample_rate_hz": 1.0 / self.dt,
            "execution_action_index": getattr(
                self, "execution_action_index", _DEFAULT_EXECUTION_ACTION_INDEX
            ),
        }

    def reset(self):
        with self.lock:
            self.engine.reset(seed=self.seed)
            self.frame_index = 0
            self.initialized = True

    def step(self, payload):
        with self.lock:
            if not self.initialized:
                self.reset()
            if not isinstance(payload, dict):
                raise ValueError("step payload must be a JSON object")
            state = payload.get("fdp_state")
            if not isinstance(state, dict):
                raise ValueError("fdp_state is required by the fdp interface")
            images = _decode_video(payload)
            if len(images) < 2:
                raise ValueError("fdp requires side and wrist camera images")
            def state_vector(name, size):
                value = np.asarray(state.get(name), dtype=np.float32)
                if value.shape != (size,) or not np.isfinite(value).all():
                    raise ValueError(f"fdp_state[{name!r}] must be finite with shape [{size}]")
                return value

            timestamp = float(state.get("timestamp_s", self.frame_index * self.dt))
            if not np.isfinite(timestamp):
                raise ValueError("fdp_state['timestamp_s'] must be finite")
            q = state_vector("q", 7)
            dq = state_vector("dq", 7)
            tau = state_vector("tau", 7)
            eef_pos = state_vector("eef_pos", 3)
            eef_quat = state_vector("eef_quat", 4)
            if self.engine.phase_wrench_source == "measured" and state.get("wrench") is None:
                raise ValueError(
                    "fdp_state['wrench'] is required when runtime.phase_wrench_source=measured"
                )
            wrench_value = None if state.get("wrench") is None else state_vector("wrench", 6)
            # The evaluator sends agentview first and eye-in-hand second.
            observation = InferenceObservation(
                timestamp_s=timestamp,
                images={"side": images[0], "wrist": images[1]},
                q=q, dq=dq, tau=tau, wrench=wrench_value,
            )
            result = self.engine.predict(observation)
            predicted_delta_pos = None
            if result.ready:
                if result.actions is None or result.actions.ndim != 2 or result.actions.shape[0] < 1:
                    raise RuntimeError("FDP returned an empty action chunk")
                # The checkpoint's action_start_index points at the current
                # observation anchor.  Executing that row makes the OSC
                # target equal to the current pose, so the arm settles at the
                # first predicted waypoint and never advances.  Use the next
                # native-rate target by default; short legacy chunks fall
                # back to their only available action.
                action_index = min(
                    self.execution_action_index, int(result.actions.shape[0]) - 1
                )
                predicted_delta_pos = (
                    np.asarray(result.actions[action_index][:3], dtype=np.float32) - eef_pos
                )
                print(
                    f"[FDP action] frame={self.frame_index} action_index={action_index} "
                    f"predicted_delta_pos [m]="
                    f"{np.array2string(predicted_delta_pos, precision=5, separator=', ')}",
                    flush=True,
                )
                action = absolute_pose_to_libero_action(
                    result.actions[action_index], eef_pos, eef_quat,
                    osc_action_min=state.get("osc_action_min"),
                    osc_action_max=state.get("osc_action_max"),
                )
            else:
                # Hold position and gripper while the causal model warms up.
                action = np.zeros((7,), dtype=np.float32)
            self.frame_index += 1
            if not np.isfinite(action).all():
                raise RuntimeError("FDP returned a non-finite action")
            future_wrench = result.info.get("future_wrench")
            future_offsets = result.info.get("future_wrench_offsets_s")
            if future_wrench is not None:
                future_wrench = np.asarray(future_wrench, dtype=np.float32)
                if (future_wrench.ndim != 2 or future_wrench.shape[1] != 6
                        or not np.isfinite(future_wrench).all()):
                    raise RuntimeError("FDP returned an invalid future wrench forecast")
                future_wrench = future_wrench.tolist()
            if future_offsets is not None:
                future_offsets = np.asarray(future_offsets, dtype=np.float64).reshape(-1)
                if (future_offsets.size < 1 or not np.isfinite(future_offsets).all()
                        or np.any(future_offsets < 0)
                        or np.any(np.diff(future_offsets) <= 0)):
                    raise RuntimeError("FDP returned invalid future wrench timestamps")
                future_offsets = future_offsets.tolist()
            if (future_wrench is None) != (future_offsets is None):
                raise RuntimeError("FDP returned an incomplete future wrench forecast")
            return {"action": [[action.tolist()]], **self.metadata(),
                    "ready": bool(result.ready),
                    "warmup_reason": result.info.get("warmup_reason"),
                    "future_wrench": future_wrench,
                    "future_wrench_offsets_s": future_offsets,
                    "raw_future_has_contact": result.info.get("raw_future_has_contact"),
                    "future_contact_gate": result.info.get("future_contact_gate"),
                    "contact_wrench_held": result.info.get("contact_wrench_held"),
                    "predicted_delta_pos": (
                        None if predicted_delta_pos is None else predicted_delta_pos.tolist()
                    ),
                    "future_contact_mode": result.info.get("future_contact_mode")}


class _Handler(BaseHTTPRequestHandler):
    service = None

    def _json(self, status, value):
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        try:
            if self.path in ("/", "/health"):
                self._json(HTTPStatus.OK, {"interface": "fdp", "status": "ok"})
            elif self.path == "/init":
                self.service.reset()
                self._json(HTTPStatus.OK, {"message": "Initialized.",
                                           **self.service.metadata()})
            elif self.path == "/reset":
                self.service.reset()
                self._json(HTTPStatus.OK, {"message": "Reset.",
                                           **self.service.metadata()})
            else:
                self._json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
        except Exception as exc:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})

    def do_POST(self):  # noqa: N802
        if self.path != "/step":
            self._json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(size))
            self._json(HTTPStatus.OK, self.service.step(payload))
        except Exception as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def log_message(self, format, *args):  # noqa: A003
        print("[fdp] " + (format % args), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/inference/fdp_cad_wm.yaml")
    parser.add_argument("--checkpoint", help="Override dp.checkpoint from the YAML")
    parser.add_argument(
        "--wm-checkpoint", "--cad-wm-checkpoint", dest="wm_checkpoint",
        help="Override cad_wm.checkpoint from the YAML",
    )
    parser.add_argument("--backbone", help="Override dp.backbone from the YAML")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7075)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    service = FDPService(args.config, checkpoint=args.checkpoint,
                         wm_checkpoint=args.wm_checkpoint,
                         backbone=args.backbone, seed=args.seed)
    _Handler.service = service
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"[fdp] serving on http://{args.host}:{args.port}; "
          f"dp_checkpoint={service.config['dp']['checkpoint']}; "
          f"wm_checkpoint={service.config['cad_wm']['checkpoint']}; "
          f"execution_action_index={service.execution_action_index}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.engine.close()


if __name__ == "__main__":
    main()
