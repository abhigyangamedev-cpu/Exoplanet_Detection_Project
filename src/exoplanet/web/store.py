"""Everything the model-checker website shows, loaded once from the pipeline's saved files.

No model is run here. A development KOI's score is its out-of-fold score (from the fold model
that never saw its star); a test KOI's score is the one saved when the held-out test set was
scored. The two are never mixed, and a model that was not scored on the test set has no score
there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from exoplanet.data.catalogue import load_koi_catalogue
from exoplanet.data.labels import LABEL
from exoplanet.data.splits import FOLD, KOI, SPLIT, splits_path
from exoplanet.evaluation.false_positives import FP_FLAGS

CONTEXT_COLUMNS = [
    "koi_period",
    "koi_depth",
    "koi_prad",
    "koi_model_snr",
    "koi_duration",
    *FP_FLAGS,
]
PREFERRED_MODELS = ["cnn_full", "tabular_lc", "tabular"]
DEFAULT_THRESHOLD = 0.5
THRESHOLD_TARGET = "precision 0.95"
TEST_SCORES = "evaluation_test_scores.csv"
OPERATING_POINTS = "operating_points.md"
DEV_SECTION = "## Development folds"
# The last non-score column of the test scores file; model columns follow it.
LAST_CONTEXT_COLUMN = "koi_model_snr"


@dataclass(frozen=True)
class Store:
    kois: pd.DataFrame
    models: list[str]
    test_models: frozenset[str]
    thresholds: dict[str, float]
    threshold_is_default: dict[str, bool]


def score_column(model: str) -> str:
    return f"score_{model}"


def parse_thresholds(text: str, target: str = THRESHOLD_TARGET) -> dict[str, float]:
    """Per-model thresholds for ``target`` from the development table of operating_points.md."""
    thresholds: dict[str, float] = {}
    in_dev = False
    for line in text.splitlines():
        if line.startswith("## "):
            in_dev = line.strip() == DEV_SECTION
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if in_dev and len(cells) >= 4 and cells[2] == target:
            try:
                thresholds[cells[1]] = float(cells[3])
            except ValueError:
                continue
    return thresholds


def _order(models: set[str]) -> list[str]:
    first = [m for m in PREFERRED_MODELS if m in models]
    return first + sorted(models - set(first))


def load_store(config: dict[str, Any]) -> Store:
    outputs = config["paths"]["outputs_dir"]
    kois = pd.read_csv(splits_path(config)).set_index(KOI)[["kepid", LABEL, SPLIT, FOLD]]

    catalogue = load_koi_catalogue(config).drop_duplicates(KOI).set_index(KOI)
    context = [c for c in CONTEXT_COLUMNS if c in catalogue.columns]
    kois = kois.join(catalogue[context].apply(pd.to_numeric, errors="coerce"))

    views_dir = config["paths"]["views_dir"]
    built = {p.stem for p in views_dir.glob("*.npz")} if views_dir.exists() else set()
    kois["has_views"] = kois["kepid"].astype(str).isin(built)

    oof = {
        p.name.removesuffix("_oof.csv"): pd.read_csv(p).drop_duplicates(KOI).set_index(KOI)["score"]
        for p in sorted(outputs.glob("*_oof.csv"))
    }
    test_path = outputs / TEST_SCORES
    test = pd.DataFrame()
    if test_path.exists():
        table = pd.read_csv(test_path).drop_duplicates(KOI).set_index(KOI)
        test = table.iloc[:, list(table.columns).index(LAST_CONTEXT_COLUMN) + 1 :]

    is_dev = kois[SPLIT] == "dev"
    models = _order(set(oof) | set(test.columns))
    for model in models:
        score = pd.Series(float("nan"), index=kois.index)
        if model in oof:
            score = score.where(~is_dev, oof[model].reindex(kois.index))
        if model in test.columns:
            score = score.where(is_dev, test[model].reindex(kois.index))
        kois[score_column(model)] = score.astype(float)

    points = outputs / OPERATING_POINTS
    chosen = parse_thresholds(points.read_text(encoding="utf-8")) if points.exists() else {}
    return Store(
        kois=kois,
        models=models,
        test_models=frozenset(test.columns),
        thresholds={m: chosen.get(m, DEFAULT_THRESHOLD) for m in models},
        threshold_is_default={m: m not in chosen for m in models},
    )
