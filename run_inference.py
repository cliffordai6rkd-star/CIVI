"""Config-driven FDP + CAD-WM entry point (inspection or recorded-input replay)."""

import argparse
import json

from inference.config import DEFAULT_CONFIG, check_config, load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Inference YAML; all model/data paths live here")
    parser.add_argument("--check-config", action="store_true", help="Check training contracts without loading weights")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.check_config:
        report = check_config(config)
    else:
        if not config["cad_wm"].get("checkpoint"):
            parser.error("Set cad_wm.checkpoint in the config after CAD-WM is trained. No random-weight fallback is used.")
        from inference.fdp_inference import FDPInference
        from inference.replay import run_replay
        with FDPInference.from_config(args.config) as engine:
            report = run_replay(engine, config)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
