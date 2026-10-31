"""Metrics shared by every model: out-of-fold scores share one format.

An out-of-fold frame is indexed by ``kepoi_name`` with columns ``fold``, ``label`` and
``score`` (predicted probability of the positive class).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from exoplanet.data.labels import LABEL
from exoplanet.data.splits import FOLD

SCORE = "score"


def fold_metrics(fold: int, label: np.ndarray, score: np.ndarray) -> dict[str, float]:
    """PR-AUC and ROC-AUC of one validation fold, with its size and positive fraction."""
    return {
        FOLD: fold,
        "n_val": len(label),
        "positive_fraction": float(np.mean(label)),
        "pr_auc": average_precision_score(label, score),
        "roc_auc": roc_auc_score(label, score),
    }


def pooled_pr_auc(oof: pd.DataFrame) -> float:
    return float(average_precision_score(oof[LABEL], oof[SCORE]))


def summarise_folds(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """Per-fold metrics plus mean and standard deviation rows."""
    stats = fold_metrics.agg(["mean", "std"])
    out = pd.concat([fold_metrics, stats])
    out.index = [f"fold {i}" if isinstance(i, int | np.integer) else i for i in out.index]
    return out.rename_axis("fold")
