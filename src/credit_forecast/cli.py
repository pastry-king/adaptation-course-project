"""Run analyses and experiments from the repository root or any directory."""

import argparse
from pathlib import Path

from .experiments import analyze, configuration, run


def main() -> None:
    parser = argparse.ArgumentParser(description="Одношаговое прогнозирование объёма кредитования")
    parser.add_argument("command", choices=["analyze", "run", "all"])
    parser.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    args = parser.parse_args()
    cfg = configuration(args.config)
    if args.command in ("analyze", "all"):
        analyze(cfg)
    if args.command in ("run", "all"):
        run(cfg)


if __name__ == "__main__":
    main()
