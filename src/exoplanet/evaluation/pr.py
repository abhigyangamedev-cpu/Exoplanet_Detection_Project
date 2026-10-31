"""Precision-recall comparison of models scored on the same KOIs.

Uncertainty comes from a cluster bootstrap over host stars: KOIs on one star share stellar
parameters and systematics, so resampling KOIs independently would understate the spread.
Model differences use paired resamples (the same stars for every model).
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from exoplanet.data.labels import LABEL


def star_bootstrap_indices(groups: np.ndarray, n_boot: int, seed: int) -> list[np.ndarray]:
    """Row indices for ``n_boot`` resamples drawing whole groups (stars) with replacement."""
    rng = np.random.default_rng(seed)
    codes, uniques = pd.factorize(groups)
    members = [np.flatnonzero(codes == k) for k in range(len(uniques))]
    return [
        np.concatenate([members[k] for k in rng.integers(0, len(uniques), len(uniques))])
        for _ in range(n_boot)
    ]


def compare_pr_auc(
    scores: pd.DataFrame,
    models: Sequence[str],
    group_col: str = "kepid",
    n_boot: int = 1000,
    seed: int = 0,
    ci: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """PR-AUC per model and paired differences, with star-bootstrap confidence intervals.

    ``scores`` has one row per KOI with ``label``, ``group_col`` and one score column per
    model. Returns ``(per_model, differences)``; ``p_value`` is the two-sided bootstrap
    p-value for a zero difference.
    """
    y = scores[LABEL].to_numpy()
    s = {m: scores[m].to_numpy() for m in models}
    point = {m: average_precision_score(y, s[m]) for m in models}
    boots = {m: [] for m in models}
    for idx in star_bootstrap_indices(scores[group_col].to_numpy(), n_boot, seed):
        if y[idx].min() == y[idx].max():
            continue
        for m in models:
            boots[m].append(average_precision_score(y[idx], s[m][idx]))
    boots = {m: np.array(v) for m, v in boots.items()}
    lo, hi = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100

    per_model = pd.DataFrame(
        {
            "pr_auc": [point[m] for m in models],
            "ci_low": [np.percentile(boots[m], lo) for m in models],
            "ci_high": [np.percentile(boots[m], hi) for m in models],
        },
        index=pd.Index(models, name="model"),
    )
    rows = []
    for a, b in combinations(models, 2):
        diff = boots[a] - boots[b]
        p = 2 * min((diff <= 0).mean(), (diff >= 0).mean())
        rows.append(
            {
                "comparison": f"{a} - {b}",
                "difference": point[a] - point[b],
                "ci_low": np.percentile(diff, lo),
                "ci_high": np.percentile(diff, hi),
                "p_value": min(p, 1.0),
            }
        )
    return per_model, pd.DataFrame(rows).set_index("comparison")


def threshold_at_precision(label: np.ndarray, score: np.ndarray, target: float) -> float:
    """Lowest score threshold whose precision reaches ``target`` (maximising recall there).

    Choose it on development scores and only apply it to the test set.
    """
    precision, _, thresholds = precision_recall_curve(label, score)
    ok = np.flatnonzero(precision[:-1] >= target)
    if not len(ok):
        raise ValueError(f"precision never reaches {target}")
    return float(thresholds[ok[0]])


def max_precision_threshold(label: np.ndarray, score: np.ndarray) -> float:
    """Threshold with the highest precision (the lowest such threshold, maximising recall):
    a fallback for models whose precision never reaches the target.
    """
    precision, _, thresholds = precision_recall_curve(label, score)
    best = precision[:-1].max()
    return float(thresholds[np.flatnonzero(precision[:-1] == best)[0]])


def operating_point(label: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float]:
    """Confusion counts, precision and recall when predicting positive at ``score >= threshold``."""
    pred = score >= threshold
    tp = int((pred & (label == 1)).sum())
    fp = int((pred & (label == 0)).sum())
    fn = int((~pred & (label == 1)).sum())
    tn = int((~pred & (label == 0)).sum())
    return {
        "threshold": threshold,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
