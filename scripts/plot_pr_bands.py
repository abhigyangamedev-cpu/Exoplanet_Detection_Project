"""Draw precision-recall curves with 95% host-star bootstrap bands from saved scores.

Reads a scores file written by ``scripts/evaluate.py`` (one row per KOI with ``label``,
``kepid`` and one column per model), so the test-set figure needs no re-scoring.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from exoplanet.config import load_config
from exoplanet.evaluation.plots import plot_pr_bands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", help="scores CSV from scripts/evaluate.py")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--output", required=True, help="PNG path")
    parser.add_argument("--config", default=None, help="path to a YAML config")
    args = parser.parse_args()
    config = load_config(args.config)
    scores = pd.read_csv(args.scores)
    path = plot_pr_bands(
        scores,
        args.models,
        args.title,
        Path(args.output),
        n_boot=config["evaluation"]["n_bootstrap"],
        seed=config["seed"],
    )
    print(f"Figure: {path}")


if __name__ == "__main__":
    main()
