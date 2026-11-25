"""Questions the website asks of the store: which KOIs match, was the model right, one KOI."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from exoplanet.data.labels import LABEL
from exoplanet.data.lightcurves import VIEW_ARRAYS, star_path
from exoplanet.data.splits import FOLD, KOI, SPLIT
from exoplanet.evaluation.false_positives import FP_FLAGS
from exoplanet.web.store import CONTEXT_COLUMNS, Store, score_column

SCORED = "scored"
NOT_ON_TEST = "not scored on test"
NO_SCORE = "no score"
SOURCES = {"dev": "out-of-fold", "test": "held-out test"}
SNR = "koi_model_snr"
PERIOD = "koi_period"


@dataclass(frozen=True)
class Filters:
    model: str
    threshold: float
    q: str = ""
    split: str | None = None
    label: int | None = None
    mistakes_only: bool = False
    snr_min: float | None = None
    snr_max: float | None = None
    period_min: float | None = None
    period_max: float | None = None


def _number(value: Any) -> float | None:
    """A JSON-safe float: missing and non-finite values become None."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def score_status(store: Store, model: str, split: str, score: float) -> str:
    if _number(score) is not None:
        return SCORED
    if split == "test" and model not in store.test_models:
        return NOT_ON_TEST
    return NO_SCORE


def _in_range(values: pd.Series, low: float | None, high: float | None) -> pd.Series:
    """True where the value lies in the range; a missing value fails any bound that is set."""
    keep = pd.Series(True, index=values.index)
    if low is not None:
        keep &= values >= low
    if high is not None:
        keep &= values <= high
    return keep


def _matches(kois: pd.DataFrame, q: str) -> pd.Series:
    q = q.strip().upper()
    if not q:
        return pd.Series(True, index=kois.index)
    by_name = kois.index.to_series().str.contains(q, regex=False)
    by_kic = kois["kepid"].astype(str).str.startswith(q)
    return by_name | by_kic


def list_kois(store: Store, f: Filters, page: int = 1, page_size: int = 50) -> dict[str, Any]:
    kois = store.kois
    keep = _matches(kois, f.q)
    if f.split is not None:
        keep &= kois[SPLIT] == f.split
    if f.label is not None:
        keep &= kois[LABEL] == f.label
    if SNR in kois:
        keep &= _in_range(kois[SNR], f.snr_min, f.snr_max)
    if PERIOD in kois:
        keep &= _in_range(kois[PERIOD], f.period_min, f.period_max)

    rows = kois[keep].sort_index()
    score = rows[score_column(f.model)]
    scored = score.notna()
    verdict = (score >= f.threshold).astype(int)
    correct = verdict == rows[LABEL]
    if f.mistakes_only:
        wrong = scored & ~correct
        rows, score, scored, verdict, correct = (
            x[wrong] for x in (rows, score, scored, verdict, correct)
        )

    total = len(rows)
    start = (page - 1) * page_size
    items = [
        {
            "kepoi_name": name,
            "kepid": int(rows.at[name, "kepid"]),
            "split": rows.at[name, SPLIT],
            "label": int(rows.at[name, LABEL]),
            "score": _number(score[name]),
            "verdict": int(verdict[name]) if scored[name] else None,
            "correct": bool(correct[name]) if scored[name] else None,
        }
        for name in rows.index[start : start + page_size]
    ]
    return {
        "total": total,
        "right": int((scored & correct).sum()),
        "wrong": int((scored & ~correct).sum()),
        "unscored": int((~scored).sum()),
        "page": page,
        "pages": max(1, math.ceil(total / page_size)),
        "items": items,
    }


def koi_detail(store: Store, name: str) -> dict[str, Any] | None:
    if name not in store.kois.index:
        return None
    row = store.kois.loc[name]
    split = row[SPLIT]
    scores = []
    for model in store.models:
        score = _number(row[score_column(model)])
        scores.append(
            {
                "model": model,
                "score": score,
                "status": score_status(store, model, split, row[score_column(model)]),
                "source": SOURCES[split] if score is not None else None,
            }
        )
    present = [c for c in CONTEXT_COLUMNS if c in row.index]
    return {
        KOI: name,
        "kepid": int(row["kepid"]),
        "split": split,
        "fold": int(row[FOLD]),
        "label": int(row[LABEL]),
        "catalogue": {c: _number(row[c]) for c in present if c not in FP_FLAGS},
        "fp_flags": [text for col, text in FP_FLAGS.items() if col in present and row[col] == 1],
        "scores": scores,
    }


def koi_views(store: Store, config: dict[str, Any], name: str) -> dict[str, list] | None:
    """The binned light-curve views for one KOI, or None if its star has none built."""
    if name not in store.kois.index:
        return None
    # The file name comes from the store's own KIC number, never from request text.
    path = star_path(int(store.kois.at[name, "kepid"]), config)
    if not path.exists():
        return None
    with np.load(path) as star:
        rows = np.flatnonzero(star[KOI] == name)
        if not len(rows):
            return None
        return {
            view: [_number(x) for x in star[view][rows[0]]]
            for view in VIEW_ARRAYS
            if view in star.files
        }
