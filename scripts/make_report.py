"""Generate docs/results.md from pipeline outputs.

Safe to run at any stage: sections without results are marked pending with the command that
produces them. Re-run after training or evaluating to refresh the report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from exoplanet.config import DEFAULT_CONFIG, REPO_ROOT, load_config
from exoplanet.data.lightcurves import load_views
from exoplanet.data.splits import KOI, load_splits
from exoplanet.evaluation.report import build_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument("--output", default=None, help="report path (default: docs/results.md)")
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else DEFAULT_CONFIG
    config = load_config(config_path)
    splits = load_splits(config)
    views_dir = config["paths"]["views_dir"]
    views_kois = (
        load_views(config, splits[KOI]).index if views_dir.exists() else pd.Index([], name=KOI)
    )
    failures_path = config["paths"]["outputs_dir"] / "views_failures.csv"
    failures = pd.read_csv(failures_path) if failures_path.exists() else None

    output = Path(args.output) if args.output else REPO_ROOT / "docs" / "results.md"
    report = build_report(
        config, config_path, splits, views_kois, failures, output.parent / "figures"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
