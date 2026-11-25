"""A tiny fake project on disk for the model-checker website tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

OPERATING_POINTS = """# Operating Points in Follow-up Terms

## Development folds

|  | model | target | threshold | precision | recall |
|---|---|---|---|---|---|
| 0 | tabular_lc | precision 0.99 | 0.976 | 0.990 | 0.475 |
| 1 | tabular_lc | precision 0.95 | 0.840 | 0.950 | 0.818 |
| 2 | cnn_full | precision 0.95 | 0.831 | 0.950 | 0.694 |
| 3 | cnn_full | recall 0.90 | 0.439 | 0.871 | 0.900 |

## Held-out test set (dev-chosen thresholds applied unchanged)

|  | model | target | threshold | precision | recall |
|---|---|---|---|---|---|
| 0 | tabular_lc | precision 0.95 | 0.111 | 0.973 | 0.804 |
| 1 | cnn_full | precision 0.95 | 0.222 | 0.973 | 0.804 |
"""

RESULTS_MD = """# Results

Methodology: [evaluation.md](evaluation.md).

## Data

| partition | kois |
|---|---|
| development | 4 |
| test | 2 |

![Precision-recall curves](figures/a.png)
"""

EVALUATION_MD = """# Evaluation

## Protocol

Scored once on the test set.
"""

#       koi          kepid  label split  fold  period  depth   prad   snr   dur  nt ss co ec
ROWS = [
    ("K00001.01", 1001, 1, "test", -1, 2.47, 14230.9, 13.04, 4304.3, 1.74, 0, 0, 0, 0),
    ("K00002.01", 1002, 0, "test", -1, 5.10, 800.0, 30.0, 55.0, 3.1, 0, 1, 0, 0),
    ("K00003.01", 1003, 1, "dev", 0, 9.49, 616.0, 2.26, 35.8, 2.96, 0, 0, 0, 0),
    ("K00003.02", 1003, 1, "dev", 0, 54.4, 875.0, 2.83, 25.8, 4.5, 0, 0, 0, 0),
    ("K00004.01", 1004, 0, "dev", 1, 1.74, 8080.0, 33.5, 505.6, 2.4, 0, 1, 1, 0),
    ("K00005.01", 1005, 1, "dev", 2, 2.53, 603.0, 2.75, np.nan, 1.65, 0, 0, 0, 0),
]
COLUMNS = [
    "kepoi_name", "kepid", "label", "split", "fold", "koi_period", "koi_depth", "koi_prad",
    "koi_model_snr", "koi_duration", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co",
    "koi_fpflag_ec",
]  # fmt: skip


def _write_views(views_dir: Path, kepid: int, names: list[str]) -> None:
    n = len(names)
    rng = np.random.default_rng(kepid)
    local = rng.normal(0, 0.1, (n, 201))
    local[:, 5] = np.nan
    np.savez(
        views_dir / f"{kepid}.npz",
        kepoi_name=np.array(names),
        global_view=rng.normal(0, 0.1, (n, 2001)),
        local_view=local,
        odd_view=rng.normal(0, 0.1, (n, 201)),
        even_view=rng.normal(0, 0.1, (n, 201)),
        secondary_view=rng.normal(0, 0.1, (n, 201)),
        n_in_transit=np.full(n, 40),
    )


def make_project(tmp_path: Path) -> dict[str, Any]:
    paths = {
        name: tmp_path / name
        for name in ("raw_dir", "processed_dir", "outputs_dir", "views_dir", "docs_dir")
    }
    for p in paths.values():
        p.mkdir()
    table = pd.DataFrame(ROWS, columns=COLUMNS)

    catalogue = table.drop(columns=["label", "split", "fold"]).assign(
        koi_disposition=np.where(table["label"] == 1, "CONFIRMED", "FALSE POSITIVE")
    )
    catalogue.to_csv(paths["raw_dir"] / "koi.csv", index=False)
    table[["kepoi_name", "kepid", "label", "split", "fold"]].to_csv(
        paths["processed_dir"] / "splits.csv", index=False
    )

    out = paths["outputs_dir"]
    pd.DataFrame(
        {
            "kepoi_name": ["K00003.01", "K00003.02", "K00004.01"],
            "fold": [0, 0, 1],
            "label": [1, 1, 0],
            "score": [0.91, 0.40, 0.05],
        }
    ).to_csv(out / "cnn_full_oof.csv", index=False)
    pd.DataFrame(
        {
            "kepoi_name": ["K00003.01", "K00003.02", "K00004.01", "K00005.01"],
            "fold": [0, 0, 1, 2],
            "label": [1, 1, 0, 1],
            "score": [0.95, 0.88, 0.10, 0.60],
        }
    ).to_csv(out / "tabular_lc_oof.csv", index=False)
    test = table[table["split"] == "test"].drop(columns=["split", "fold", "koi_duration"])
    test = test[
        ["kepoi_name", "kepid", "label", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co",
         "koi_fpflag_ec", "koi_period", "koi_depth", "koi_prad", "koi_model_snr"]
    ].assign(tabular_lc=[0.77, 0.90], cnn=[0.10, 0.20])  # fmt: skip
    test.to_csv(out / "evaluation_test_scores.csv", index=False)
    (out / "operating_points.md").write_text(OPERATING_POINTS, encoding="utf-8")

    _write_views(paths["views_dir"], 1001, ["K00001.01"])
    _write_views(paths["views_dir"], 1002, ["K00002.01"])
    _write_views(paths["views_dir"], 1003, ["K00003.01", "K00003.02"])
    _write_views(paths["views_dir"], 1005, ["K00005.01"])

    docs = paths["docs_dir"]
    (docs / "results.md").write_text(RESULTS_MD, encoding="utf-8")
    (docs / "evaluation.md").write_text(EVALUATION_MD, encoding="utf-8")
    (docs / "figures").mkdir()
    (docs / "figures" / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    return {
        "paths": paths,
        "catalogue": {"filename": "koi.csv"},
        "splits": {"filename": "splits.csv", "n_folds": 5, "group_column": "kepid"},
    }
