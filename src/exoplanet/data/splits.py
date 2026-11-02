"""Host-star grouped, label-stratified train/validation/test splits.

All KOIs on a star share stellar parameters and light-curve systematics, so a star must
never appear on both sides of a split. Splits are computed once, saved, and reused by every
model so that comparisons are made on identical data.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from exoplanet.data.labels import LABEL

KOI = "kepoi_name"
SPLIT = "split"
FOLD = "fold"
TEST, DEV = "test", "dev"
TEST_FOLD = -1


def make_splits(
    sample: pd.DataFrame,
    test_fraction: float,
    n_folds: int,
    seed: int,
    group_col: str = "kepid",
) -> pd.DataFrame:
    """Assign each KOI to the test set or one of ``n_folds`` development folds.

    Returns one row per KOI with columns ``kepoi_name``, ``group_col``, ``label``, ``split``
    (``"test"``/``"dev"``) and ``fold`` (``-1`` for test, ``0..n_folds-1`` for dev).
    """
    if not 0 < test_fraction < 1:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    # Sort so the assignment depends only on the data, not on input row order.
    df = sample[[KOI, group_col, LABEL]].sort_values(KOI).reset_index(drop=True)

    # The test set is one fold of a grouped, stratified K-fold with K ~ 1 / test_fraction.
    n_outer = max(2, round(1 / test_fraction))
    outer = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    _, test_idx = next(outer.split(df, df[LABEL], df[group_col]))

    df[SPLIT] = DEV
    df.loc[test_idx, SPLIT] = TEST
    df[FOLD] = TEST_FOLD

    dev = df.index[df[SPLIT] == DEV]
    inner = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(
        inner.split(dev, df.loc[dev, LABEL], df.loc[dev, group_col])
    ):
        df.loc[dev[val_idx], FOLD] = fold

    validate_splits(df, n_folds, group_col)
    return df


def validate_splits(splits: pd.DataFrame, n_folds: int, group_col: str = "kepid") -> None:
    """Raise ``ValueError`` if any star spans more than one fold/test set, or assignments are
    incomplete or inconsistent.
    """
    if splits[KOI].duplicated().any():
        raise ValueError("Duplicate KOIs in split assignments")

    expected = {TEST_FOLD, *range(n_folds)}
    if not set(splits[FOLD].unique()) == expected:
        raise ValueError(f"Folds {sorted(splits[FOLD].unique())} != expected {sorted(expected)}")
    if not ((splits[SPLIT] == TEST) == (splits[FOLD] == TEST_FOLD)).all():
        raise ValueError("split and fold columns disagree")

    folds_per_star = splits.groupby(group_col)[FOLD].nunique()
    leaking = folds_per_star[folds_per_star > 1]
    if len(leaking):
        raise ValueError(
            f"{len(leaking)} host stars span multiple folds, e.g. {list(leaking.index[:5])}"
        )


def summarise_splits(splits: pd.DataFrame, group_col: str = "kepid") -> pd.DataFrame:
    """Per-fold KOI count, star count, positive fraction and fraction of KOIs on multi-KOI
    stars (which are mostly confirmed, so should also be balanced).
    """
    multi = splits[group_col].map(splits[group_col].value_counts()) > 1
    summary = (
        splits.assign(_multi=multi)
        .groupby(FOLD)
        .agg(
            kois=(KOI, "size"),
            stars=(group_col, "nunique"),
            positive_fraction=(LABEL, "mean"),
            multi_koi_fraction=("_multi", "mean"),
        )
    )
    summary.index = ["test" if f == TEST_FOLD else f"dev fold {f}" for f in summary.index]
    return summary.rename_axis("partition")


def cv_folds(splits: pd.DataFrame) -> Iterator[tuple[int, np.ndarray, np.ndarray]]:
    """Yield ``(fold, train_kois, val_kois)`` over development folds; test KOIs never appear."""
    dev = splits[splits[SPLIT] == DEV]
    for fold in sorted(dev[FOLD].unique()):
        in_fold = dev[FOLD] == fold
        yield int(fold), dev.loc[~in_fold, KOI].to_numpy(), dev.loc[in_fold, KOI].to_numpy()


def held_out_kois(splits: pd.DataFrame) -> np.ndarray:
    """KOIs in the held-out test set: for final evaluation only."""
    return splits.loc[splits[SPLIT] == TEST, KOI].to_numpy()


def splits_path(config: dict[str, Any]) -> Path:
    return config["paths"]["processed_dir"] / config["splits"]["filename"]


def save_splits(splits: pd.DataFrame, config: dict[str, Any], provenance: dict[str, Any]) -> Path:
    """Write assignments CSV and a JSON sidecar recording how they were produced."""
    path = splits_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(path, index=False)
    path.with_suffix(".meta.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return path


def load_splits(config: dict[str, Any]) -> pd.DataFrame:
    path = splits_path(config)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run scripts/make_splits.py")
    splits = pd.read_csv(path)
    validate_splits(splits, config["splits"]["n_folds"], config["splits"]["group_column"])
    return splits
