"""Export external force magnitude with contact phases from the CAD-WM labeler.

Example (from the repository root, in the fdp environment)::

    python scripts/plot_demo_wrench.py --demo demo_0

Demo names are the literal HDF5 keys: demo_0 is the first demonstration.
External force magnitude is sqrt(Fx**2 + Fy**2 + Fz**2), in N; torque components
are not combined with force. The default plot preserves the raw magnitude.
Use --schematic for arrow axes, dashed event markers, and compressed peaks for
illustration. This display transform never changes the source or phase labels.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

import h5py
import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
from matplotlib import pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
CAD_WM = ROOT / "cad-wm"
DEFAULT_CONFIG = CAD_WM / "config/bench_train_config/libero_drawer_wrench_h5.yaml"
FORCE_COLOR = "#174A7E"


def cad_wm_phase_labeler():
    # A private package preserves CAD-WM's relative imports and avoids collisions
    # with other repositories' top-level train modules. Use its actual labeler,
    # rather than copying the contact state machine into this plotting script.
    name = "_cad_wm_plot_" + hashlib.sha256(str(CAD_WM).encode()).hexdigest()[:12]
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(CAD_WM)]
        package.__package__ = name
        sys.modules[name] = package
    return importlib.import_module(f"{name}.data_process.libero_wrench_dataset").wrench_phase_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data", type=Path, help="Override the HDF5 path from the config")
    parser.add_argument("--demo", default="demo_0", help="Literal HDF5 key, e.g. demo_0")
    parser.add_argument("--output-prefix", type=Path, help="File prefix without extension")
    parser.add_argument("--schematic", action="store_true", help="Draw a schematic with compressed peaks")
    parser.add_argument("--reference-times", type=float, nargs=3, default=(2.1, 2.35, 2.75),
                        metavar=("T1", "T2", "T3"), help="Three dashed markers in seconds (schematic only)")
    parser.add_argument("--peak-knee", type=float, default=80.0,
                        help="Preserve values up to this force, then compress (schematic only)")
    parser.add_argument("--peak-scale", type=float, default=20.0,
                        help="Scale of logarithmic peak compression (schematic only)")
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text())
    data_config = config["dataloader"]
    if args.data is None:
        source_path = Path(data_config["root"]).expanduser()
        if not source_path.is_absolute():
            source_path = CAD_WM / source_path
    else:
        source_path = args.data.expanduser()
    source_path = source_path.resolve()
    group = data_config.get("h5_episode_group", "data")
    field = data_config["h5_fields"]["wrench"]
    with h5py.File(source_path, "r") as source:
        env = json.loads(source[group].attrs.get("env_args", "{}"))
        fps = float(env.get("env_kwargs", {}).get("control_freq", data_config["high_fps"]))
        if not np.isclose(fps, float(data_config["high_fps"])):
            raise ValueError("HDF5 sample rate differs from the CAD-WM config")
        wrench = np.asarray(source[f"{group}/{args.demo}/{field['path']}"], dtype=np.float32)
        if field.get("columns") is not None:
            wrench = wrench[:, list(field["columns"])]
    if wrench.ndim != 2 or wrench.shape[1] != 6 or not len(wrench) or not np.isfinite(wrench).all():
        raise ValueError("Wrench must be a nonempty, finite [T, 6] array")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Sample rate must be positive and finite")

    phases = cad_wm_phase_labeler()(torch.from_numpy(wrench.copy()), config, fps).numpy()
    force_magnitude = np.linalg.norm(wrench[:, :3].astype(np.float64), axis=1)
    displayed_force = force_magnitude.copy()
    if args.schematic:
        if (not np.isfinite([args.peak_knee, args.peak_scale]).all()
                or args.peak_knee < 0 or args.peak_scale <= 0):
            raise ValueError("Peak knee must be nonnegative and peak scale must be positive")
        if (not np.isfinite(args.reference_times).all()
                or any(t < 0 or t > len(wrench) / fps for t in args.reference_times)):
            raise ValueError("Reference times must be finite and within the demonstration")
        high = displayed_force > args.peak_knee
        displayed_force[high] = args.peak_knee + args.peak_scale * np.log1p(
            (displayed_force[high] - args.peak_knee) / args.peak_scale
        )
    times = np.arange(len(wrench)) / fps
    # Each offline phase label applies from this sample until the next sample.
    changes = np.diff(np.r_[False, phases == 2, False].astype(np.int8))
    starts, stops = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    intervals = [(int(start) / fps, int(stop) / fps) for start, stop in zip(starts, stops)]

    suffix = "_schematic" if args.schematic else ""
    output = args.output_prefix or ROOT / "outputs/figures" / f"{args.demo}_external_force{suffix}"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 19,
        "axes.labelsize": 13,
        "axes.edgecolor": "#69717B",
        "axes.linewidth": 0.8,
        "xtick.color": "#424B57",
        "ytick.color": "#424B57",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    fig, force_axis = plt.subplots(figsize=(12.8, 3.9) if args.schematic else (12.8, 5.8))
    if args.schematic:
        fig.subplots_adjust(left=0.045, right=0.965, bottom=0.12, top=0.80)
    else:
        fig.subplots_adjust(left=0.085, right=0.915, bottom=0.18, top=0.89)
    for start, stop in intervals:
        force_axis.axvspan(start, stop, color="#E9DFBC", alpha=0.58, linewidth=0)
        if not args.schematic:
            for boundary in (start, stop):
                force_axis.axvline(boundary, color="#AC9B66", linestyle=(0, (3, 4)), linewidth=0.8, alpha=0.6)
            force_axis.text(
                (start + stop) / 2, 0.92, "Contact",
                transform=force_axis.get_xaxis_transform(), ha="center", va="top",
                color="#74602D", fontsize=11,
            )

    color = "#AE8AE2" if args.schematic else FORCE_COLOR
    force_axis.plot(times, displayed_force, color=color,
                    linewidth=3.2 if args.schematic else 1.8, label="External force",
                    solid_capstyle="round", solid_joinstyle="round", zorder=3)
    force_axis.set_xlim(0, len(wrench) / fps)
    force_axis.margins(y=0.18)
    force_axis.set_ylim(bottom=0)
    if args.schematic:
        # Keep vertical headroom so reducing a peak also reduces its height in
        # the exported illustration, rather than autoscaling it back to the top.
        force_axis.set_ylim(0, max(200.0, float(displayed_force.max()) * 1.18))
        for time in args.reference_times:
            force_axis.axvline(time, ymin=0, ymax=0.94, color="#53555A",
                              linewidth=1.55, linestyle=(0, (5, 5)), zorder=2)
        force_axis.set_axis_off()
        arrow = dict(arrowstyle="-|>", color="#3E4348", linewidth=2.7,
                     mutation_scale=19, shrinkA=0, shrinkB=0)
        for end in ((1.025, 0), (0, 1.035)):
            force_axis.annotate("", xy=end, xytext=(0, 0),
                                xycoords="axes fraction", textcoords="axes fraction",
                                arrowprops=arrow, annotation_clip=False, zorder=5)
        force_axis.text(0.006, 1.08, "External Force", transform=force_axis.transAxes,
                        ha="left", va="bottom", fontsize=23, fontweight="semibold",
                        color="#30353B", clip_on=False)
    else:
        force_axis.set_title("External Force", pad=14, fontweight="medium")
        force_axis.set_xlabel("Time [s]", labelpad=9)
        force_axis.set_ylabel("External force [N]", labelpad=10)
        force_axis.grid(True, color="#CBD2DA", linewidth=0.65, alpha=0.5)
        force_axis.set_axisbelow(True)
        force_axis.spines["top"].set_visible(False)
        force_axis.spines["right"].set_visible(False)
        force_axis.tick_params(direction="out", length=4, width=0.7)
        force_axis.legend(
            loc="upper right", borderaxespad=0.8,
            handlelength=2.3, handletextpad=0.55, framealpha=0.96,
            facecolor="white", edgecolor="#D5DAE1", fontsize=12,
        )
        fig.text(0.085, 0.045, "Shaded regions: contact phase (CAD-WM)",
                 color="#65707B", fontsize=10)
        fig.text(0.915, 0.045, f"{args.demo}  ·  {fps:g} Hz",
                 ha="right", color="#65707B", fontsize=10)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(f"{output}.{extension}", dpi=300, facecolor="white")
    plt.close(fig)

    metadata = {
        "source": str(source_path), "hdf5_demo": args.demo,
        "config": str(config_path), "fps": fps, "samples": len(wrench),
        "time_origin": "first wrench sample at t=0",
        "wrench_order": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
        "smoothing": None, "contact_gate": config["contact_gate"],
        "phase_labeler": "cad-wm/data_process/libero_wrench_dataset.py:wrench_phase_labels",
        "phase_transition_backfill": True,
        "contact_intervals_s_exclusive_end": intervals,
        "contact_frame_intervals_exclusive_end": list(zip(starts.tolist(), stops.tolist())),
        "phase_counts_free_precontact_contact": np.bincount(phases, minlength=3).tolist(),
        "plotted_quantity": "sqrt(Fx**2 + Fy**2 + Fz**2)",
        "label": "External force", "raw_unit": "N", "color": color,
        "peak_force_N": float(force_magnitude.max()),
        "style": "schematic" if args.schematic else "scientific",
        "display_transform": ({
            "purpose": "Illustration only; source data and contact labels are unchanged",
            "formula": "F if F <= knee else knee + scale * log1p((F - knee) / scale)",
            "knee_N": args.peak_knee, "scale_N": args.peak_scale,
        } if args.schematic else None),
        "display_peak": float(displayed_force.max()),
        "reference_times_s": args.reference_times if args.schematic else [],
    }
    Path(f"{output}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output_prefix": str(output), "demo": args.demo,
                      "contact_intervals_s": intervals}, indent=2))


if __name__ == "__main__":
    main()
