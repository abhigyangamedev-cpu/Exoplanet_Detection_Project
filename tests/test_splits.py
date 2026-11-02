import numpy as np
import pandas as pd
import pytest

from exoplanet.config import load_config
from exoplanet.data.splits import (
    FOLD,
    KOI,
    SPLIT,
    TEST_FOLD,
    cv_folds,
    held_out_kois,
    load_splits,
    make_splits,
    save_splits,
    summarise_splits,
    validate_splits,
)


@pytest.fixture
def sample():
    """1,000 stars hosting 1-4 KOIs each; multi-KOI stars are mostly positive."""
    rng = np.random.default_rng(1)
    rows = []
    for star in range(1000):
        n = rng.choice([1, 2, 3, 4], p=[0.8, 0.12, 0.05, 0.03])
        p_pos = 0.25 if n == 1 else 0.85
        for k in range(n):
            rows.append(
                {
                    "kepoi_name": f"K{star:05d}.{k + 1:02d}",
                    "kepid": 10_000 + star,
                    "label": int(rng.random() < p_pos),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def splits(sample):
    return make_splits(sample, test_fraction=0.2, n_folds=5, seed=42)


def test_no_star_in_more_than_one_partition(splits):
    assert (splits.groupby("kepid")[FOLD].nunique() == 1).all()


def test_every_koi_assigned_once(sample, splits):
    assert sorted(splits[KOI]) == sorted(sample[KOI])
    assert set(splits[FOLD]) == {TEST_FOLD, 0, 1, 2, 3, 4}
    assert ((splits[SPLIT] == "test") == (splits[FOLD] == TEST_FOLD)).all()


def test_sizes_and_stratification(splits):
    summary = summarise_splits(splits)
    overall = splits["label"].mean()
    assert summary.loc["test", "kois"] / len(splits) == pytest.approx(0.2, abs=0.02)
    assert (summary["positive_fraction"] - overall).abs().max() < 0.03


def test_deterministic_and_row_order_invariant(sample, splits):
    shuffled = sample.sample(frac=1.0, random_state=7)
    again = make_splits(shuffled, test_fraction=0.2, n_folds=5, seed=42)
    pd.testing.assert_frame_equal(splits, again)


def test_seed_changes_assignment(sample, splits):
    other = make_splits(sample, test_fraction=0.2, n_folds=5, seed=0)
    assert not splits[FOLD].equals(other[FOLD])


def test_cv_folds_exclude_test_and_are_disjoint(splits):
    test = set(held_out_kois(splits))
    seen_val = set()
    folds = list(cv_folds(splits))
    assert [f for f, _, _ in folds] == [0, 1, 2, 3, 4]
    for _, train, val in folds:
        assert not (set(train) | set(val)) & test
        assert not set(train) & set(val)
        seen_val |= set(val)
    assert seen_val == set(splits[KOI]) - test


@pytest.mark.parametrize("test_fraction", [0.0, 1.0, -0.1])
def test_invalid_test_fraction(sample, test_fraction):
    with pytest.raises(ValueError):
        make_splits(sample, test_fraction=test_fraction, n_folds=5, seed=42)


def test_validate_rejects_star_spanning_folds(splits):
    bad = splits.copy()
    star = bad["kepid"].value_counts().index[0]  # a multi-KOI star
    rows = bad.index[bad["kepid"] == star]
    other = (bad.loc[rows[0], FOLD] + 1) % 5
    bad.loc[rows[1], [FOLD, SPLIT]] = [other, "dev"]
    bad.loc[rows[0], [FOLD, SPLIT]] = [(other + 1) % 5, "dev"]
    with pytest.raises(ValueError, match="span multiple folds"):
        validate_splits(bad, n_folds=5)


def test_validate_rejects_inconsistent_split_column(splits):
    bad = splits.copy()
    bad.loc[bad.index[bad[FOLD] == 0][0], SPLIT] = "test"
    with pytest.raises(ValueError, match="disagree"):
        validate_splits(bad, n_folds=5)


def test_save_and_load_roundtrip(tmp_path, splits):
    config = load_config()
    config["paths"]["processed_dir"] = tmp_path
    path = save_splits(splits, config, {"seed": 42})
    assert path.with_suffix(".meta.json").exists()
    pd.testing.assert_frame_equal(load_splits(config), splits)
