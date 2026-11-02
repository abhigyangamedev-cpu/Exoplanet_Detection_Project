import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score

from exoplanet.evaluation.false_positives import fp_rejection_by_flag, recall_by_bin, worst_errors
from exoplanet.evaluation.pr import (
    compare_pr_auc,
    operating_point,
    star_bootstrap_indices,
    threshold_at_precision,
    threshold_at_recall,
)


@pytest.fixture
def scores():
    """Two models on 400 KOIs over 250 stars: ``good`` separates the classes, ``weak`` barely."""
    rng = np.random.default_rng(0)
    n = 400
    label = rng.integers(0, 2, n)
    return pd.DataFrame(
        {
            "kepid": rng.integers(0, 250, n),
            "label": label,
            "good": np.clip(label * 0.6 + rng.normal(0.2, 0.15, n), 0, 1),
            "weak": np.clip(label * 0.1 + rng.normal(0.45, 0.2, n), 0, 1),
        }
    )


def test_bootstrap_resamples_whole_stars():
    groups = np.array([7, 7, 7, 9, 3, 3])
    for idx in star_bootstrap_indices(groups, n_boot=50, seed=1):
        counts = pd.Series(groups[idx]).value_counts()
        expected = pd.Series(groups).value_counts()
        # Each drawn star contributes all its KOIs (a multiple of its size).
        assert all(counts[g] % expected[g] == 0 for g in counts.index)
        assert len(np.unique(groups[idx])) <= 3


def test_compare_pr_auc(scores):
    per_model, diffs = compare_pr_auc(scores, ["good", "weak"], n_boot=200, seed=0)
    assert per_model.loc["good", "pr_auc"] == pytest.approx(
        average_precision_score(scores["label"], scores["good"])
    )
    for m in ["good", "weak"]:
        row = per_model.loc[m]
        assert row.ci_low <= row.pr_auc <= row.ci_high
    d = diffs.loc["good - weak"]
    assert d.difference > 0 and d.ci_low > 0 and d.p_value < 0.05


def test_compare_identical_models_is_not_significant(scores):
    scores = scores.assign(copy=scores["good"])
    _, diffs = compare_pr_auc(scores, ["good", "copy"], n_boot=100, seed=0)
    assert diffs.loc["good - copy", "difference"] == 0
    assert diffs.loc["good - copy", "p_value"] == 1


def test_threshold_at_precision_and_operating_point(scores):
    y, s = scores["label"].to_numpy(), scores["good"].to_numpy()
    thr = threshold_at_precision(y, s, 0.95)
    op = operating_point(y, s, thr)
    assert op["precision"] >= 0.95
    # A slightly lower threshold falls below the target (we took the lowest passing one).
    lower = np.max(s[s < thr])
    assert operating_point(y, s, lower)["precision"] < 0.95 or lower == thr
    assert op["tp"] + op["fn"] == y.sum() and op["fp"] + op["tn"] == (y == 0).sum()


def test_max_precision_threshold_fallback():
    from exoplanet.evaluation.pr import max_precision_threshold

    y = np.array([0, 1, 0, 1, 1, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.1])
    thr = max_precision_threshold(y, s)
    best = max(operating_point(y, s, t)["precision"] for t in s)
    assert operating_point(y, s, thr)["precision"] == best


def test_threshold_unreachable_raises():
    with pytest.raises(ValueError, match="never reaches"):
        threshold_at_precision(np.array([0, 1, 0, 1]), np.array([0.9, 0.1, 0.8, 0.2]), 0.99)


def test_fp_rejection_by_flag():
    df = pd.DataFrame(
        {
            "label": [0, 0, 0, 0, 1],
            "koi_fpflag_nt": [1, 0, 0, 0, 0],
            "koi_fpflag_ss": [1, 1, 0, 0, 0],
            "koi_fpflag_co": [0, 0, 1, 0, 0],
            "koi_fpflag_ec": [0, 0, 0, 0, 0],
            "m": [0.1, 0.9, 0.2, 0.3, 0.9],
        }
    )
    out = fp_rejection_by_flag(df, {"m": 0.5})
    assert out.loc["all false positives", "n"] == 4
    assert out.loc["all false positives", "m"] == 0.75
    assert out.loc["stellar eclipse", "m"] == 0.5
    assert out.loc["centroid offset", "m"] == 1.0
    assert out.loc["ephemeris match", "n"] == 0 and np.isnan(out.loc["ephemeris match", "m"])


def test_recall_by_bin(scores):
    scores = scores.assign(snr=np.arange(len(scores), dtype=float))
    out = recall_by_bin(scores, {"good": 0.5}, "snr", n_bins=4)
    assert len(out) == 4 and out["n"].sum() == scores["label"].sum()
    assert out["good"].between(0, 1).all()


def test_worst_errors(scores):
    fp, planets = worst_errors(scores, "good", n=3, columns=["kepid"])
    assert len(fp) == 3 and (scores.loc[fp.index, "label"] == 0).all()
    assert fp["good"].is_monotonic_decreasing
    assert planets["good"].max() <= scores.loc[scores.label == 1, "good"].nsmallest(4).max()


def test_plot_pr_curves_writes_png(tmp_path, scores):
    from exoplanet.evaluation.plots import plot_pr_curves

    y = scores["label"].to_numpy()
    ops = {"good": operating_point(y, scores["good"].to_numpy(), 0.5)}
    path = plot_pr_curves(scores, ["good", "weak"], ops, "test", tmp_path / "pr.png")
    assert path.exists() and path.read_bytes()[:4] == b"\x89PNG"
    with pytest.raises(ValueError, match="at most"):
        plot_pr_curves(scores, ["a", "b", "c", "d", "e"], {}, "t", tmp_path / "x.png")


def test_threshold_at_recall_is_highest_threshold_reaching_target():
    y = np.array([1, 1, 0, 1, 0, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    # Recall 0.75 is first reached at score 0.6 (3 of 4 planets); a higher cut finds only 2.
    assert threshold_at_recall(y, s, 0.75) == 0.6
    assert threshold_at_recall(y, s, 1.0) == 0.4
    op = operating_point(y, s, threshold_at_recall(y, s, 0.75))
    assert op["recall"] == 0.75 and op["precision"] == 0.75
