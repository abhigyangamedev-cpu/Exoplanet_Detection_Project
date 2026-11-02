"""Write a Markdown inspection report for the cached KOI catalogue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from exoplanet.config import load_config
from exoplanet.data.catalogue import catalogue_path, load_koi_catalogue, metadata_path
from exoplanet.data.inspection import render_report, summarise_catalogue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument(
        "--output", default=None, help="report path (default: outputs/catalogue_inspection.md)"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    df = load_koi_catalogue(config)
    meta_file = metadata_path(catalogue_path(config))
    metadata = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else None

    labels = config["labels"]
    summary = summarise_catalogue(df, labelled=(labels["positive"], labels["negative"]))
    output = (
        Path(args.output)
        if args.output
        else config["paths"]["outputs_dir"] / "catalogue_inspection.md"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(summary, metadata), encoding="utf-8")
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
