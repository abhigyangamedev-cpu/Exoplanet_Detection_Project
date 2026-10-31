import numpy as np
import pandas as pd
import pytest

from exoplanet.data.splits import FOLD, KOI, held_out_kois, make_splits
from exoplanet.evaluation.metrics import SCORE, pooled_pr_auc, summarise_folds
from exoplanet.models.tabular import cross_validate, fit_final

PARAMS = {"max_iter": 30, "max_leaf_nodes": 7}


@pytest.fixture
def data():
    """600 single-KOI stars; ``signal`` predicts the label, ``noise`` does not, and some
    ``signal`` values are missing.
    """
    rng = np.random.default_rng(0)
    n = 600
    label = rng.integers(0, 2, n)
    sample = pd.DataFrame(
        {KOI: [f"K{i:05d}.01" for i in range(n)], "kepid": np.arange(n), "label": label}
    )
    signal = label + rng.normal(0, 0.5, n)
    signal[rng.random(n) < 0.1] = np.nan
    features = pd.DataFrame(
        {"signal": signal, "noise": rng.normal(size=n)}, index=pd.Index(sample[KOI], name=KOI)
    )
    splits = make_splits(sample, test_fraction=0.2, n_folds=3, seed=0)
    return features, splits


def test_oof_covers_dev_only(data):
    features, splits = data
    oof, metrics, _ = cross_validate(features, splits, PARAMS, seed=0)
    test = set(held_out_kois(splits))
    assert not set(oof.index) & test
    assert set(oof.index) == set(splits[KOI]) - test
    assert list(metrics.index) == [0, 1, 2]
    assert oof[SCORE].between(0, 1).all()


def test_learns_signal_and_ranks_it_first(data):
    features, splits = data
    oof, metrics, importance = cross_validate(
        features, splits, PARAMS, seed=0, permutation_repeats=3
    )
    assert pooled_pr_auc(oof) > 0.85
    assert (metrics["pr_auc"] > 0.8).all()
    assert importance.index[0] == "signal"


def test_test_set_features_are_never_used(data):
    features, splits = data
    corrupted = features.copy()
    corrupted.loc[held_out_kois(splits)] = 1e9
    a, _, _ = cross_validate(features, splits, PARAMS, seed=0)
    b, _, _ = cross_validate(corrupted, splits, PARAMS, seed=0)
    pd.testing.assert_frame_equal(a, b)
    test_kois = held_out_kois(splits)
    dev = features.drop(index=test_kois)
    np.testing.assert_array_equal(
        fit_final(features, splits, PARAMS, 0).predict_proba(dev),
        fit_final(corrupted, splits, PARAMS, 0).predict_proba(dev),
    )


def test_deterministic(data):
    features, splits = data
    a, _, _ = cross_validate(features, splits, PARAMS, seed=0)
    b, _, _ = cross_validate(features, splits, PARAMS, seed=0)
    pd.testing.assert_frame_equal(a, b)


def test_missing_features_raise(data):
    features, splits = data
    with pytest.raises(KeyError, match="no features"):
        cross_validate(features.iloc[10:], splits, PARAMS, seed=0)


def test_requires_koi_index(data):
    features, splits = data
    with pytest.raises(ValueError, match="indexed by"):
        cross_validate(features.reset_index(), splits, PARAMS, seed=0)


def test_summarise_folds_adds_mean_and_std(data):
    features, splits = data
    _, metrics, _ = cross_validate(features, splits, PARAMS, seed=0)
    summary = summarise_folds(metrics)
    assert list(summary.index) == ["fold 0", "fold 1", "fold 2", "mean", "std"]
    assert summary.loc["mean", "pr_auc"] == pytest.approx(metrics["pr_auc"].mean())
    assert FOLD not in summary.columns
