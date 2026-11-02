"""Gradient-boosted tabular baseline on leakage-checked catalogue features.

Histogram gradient boosting handles missing values natively, so catalogue gaps need no
imputation (and imputation statistics cannot leak across folds). Training uses the shared
development folds only; the held-out test set is never touched here.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

from exoplanet.data.labels import LABEL
from exoplanet.data.splits import FOLD, KOI, SPLIT, TEST, cv_folds
from exoplanet.evaluation.metrics import SCORE, fold_metrics


def build_model(params: dict[str, Any], seed: int) -> HistGradientBoostingClassifier:
    # Internal early stopping would carve an ungrouped validation split out of the training
    # fold, so the number of boosting iterations is fixed by config instead.
    return HistGradientBoostingClassifier(**params, early_stopping=False, random_state=seed)


def _dev_matrix(features: pd.DataFrame, splits: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Features and labels for development KOIs, indexed by ``kepoi_name``."""
    if features.index.name != KOI:
        raise ValueError(f"features must be indexed by {KOI}")
    dev = splits[splits[SPLIT] != TEST].set_index(KOI)
    missing = dev.index.difference(features.index)
    if len(missing):
        raise KeyError(f"{len(missing)} split KOIs have no features, e.g. {list(missing[:5])}")
    return features.loc[dev.index], dev[LABEL]


def cross_validate(
    features: pd.DataFrame,
    splits: pd.DataFrame,
    params: dict[str, Any],
    seed: int,
    permutation_repeats: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit one model per development fold and score its validation fold.

    Returns ``(oof, fold_metrics, importance)``: out-of-fold scores (one row per dev KOI with
    ``fold``, ``label``, ``score``), per-fold PR-AUC/ROC-AUC, and permutation importance
    (mean PR-AUC drop on validation folds; empty if ``permutation_repeats`` is 0).
    """
    X, y = _dev_matrix(features, splits)
    oof, metrics, importances = [], [], []
    for fold, train, val in cv_folds(splits):
        model = build_model(params, seed).fit(X.loc[train], y.loc[train])
        score = model.predict_proba(X.loc[val])[:, 1]
        oof.append(pd.DataFrame({FOLD: fold, LABEL: y.loc[val], SCORE: score}, index=val))
        metrics.append(fold_metrics(fold, y.loc[val].to_numpy(), score))
        if permutation_repeats:
            result = permutation_importance(
                model,
                X.loc[val],
                y.loc[val],
                scoring="average_precision",
                n_repeats=permutation_repeats,
                random_state=seed,
            )
            importances.append(pd.Series(result.importances_mean, index=X.columns, name=fold))

    importance = (
        pd.concat(importances, axis=1)
        .agg(["mean", "std"], axis=1)
        .rename(columns={"mean": "pr_auc_drop", "std": "pr_auc_drop_std"})
        .sort_values("pr_auc_drop", ascending=False)
        .rename_axis("feature")
        if importances
        else pd.DataFrame()
    )
    return (
        pd.concat(oof).rename_axis(KOI),
        pd.DataFrame(metrics).set_index(FOLD),
        importance,
    )


def fit_final(
    features: pd.DataFrame, splits: pd.DataFrame, params: dict[str, Any], seed: int
) -> HistGradientBoostingClassifier:
    """Refit on all development KOIs for later evaluation on the held-out test set."""
    X, y = _dev_matrix(features, splits)
    return build_model(params, seed).fit(X, y)
