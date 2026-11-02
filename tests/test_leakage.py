from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from exoplanet.features.leakage import (
    DENIED,
    DENY_CATEGORIES,
    LeakageError,
    assert_no_leakage,
    deny_category,
    flag_suspicious,
    univariate_audit,
)
from exoplanet.features.tabular import (
    FEATURE_GROUPS,
    classify_column,
    feature_columns,
    select_features,
    unclassified_columns,
)

CATALOGUE_COLUMNS = (Path(__file__).parent / "fixtures" / "koi_columns.txt").read_text().split()


def test_every_catalogue_column_is_classified():
    assert len(CATALOGUE_COLUMNS) == 153
    assert unclassified_columns(CATALOGUE_COLUMNS) == []


def test_allowed_groups_are_disjoint_and_not_denied():
    allowed = [c for cols in FEATURE_GROUPS.values() for c in cols]
    assert len(allowed) == len(set(allowed))
    assert [c for c in allowed if deny_category(c)] == []
    assert set(allowed) <= set(CATALOGUE_COLUMNS)


def test_deny_categories_are_documented():
    assert set(DENIED.values()) | {"uncertainty"} <= DENY_CATEGORIES.keys()


@pytest.mark.parametrize(
    "column",
    [
        "koi_disposition",
        "koi_pdisposition",
        "koi_score",
        "koi_fpflag_nt",
        "koi_fpflag_ss",
        "koi_fpflag_co",
        "koi_fpflag_ec",
        "kepler_name",
        "koi_comment",
        "koi_parm_prov",
        "koi_period_err1",
        "koi_steff_err2",
        "ra_err",
        "koi_time0bk",
    ],
)
def test_known_leaky_columns_are_denied(column):
    assert deny_category(column) is not None
    assert classify_column(column).startswith("denied:")


def test_assert_no_leakage():
    assert_no_leakage(["koi_period", "koi_depth"])
    with pytest.raises(LeakageError, match="koi_score"):
        assert_no_leakage(["koi_period", "koi_score"])


def test_no_feature_group_combination_leaks():
    assert_no_leakage(feature_columns(FEATURE_GROUPS))


def test_feature_columns_rejects_unknown_group():
    with pytest.raises(KeyError):
        feature_columns(["transit", "koi_score"])


def test_select_features_returns_numeric_group_columns():
    df = pd.DataFrame({c: ["1.5", "2"] for c in CATALOGUE_COLUMNS})
    X = select_features(df, ["transit", "stellar"])
    assert list(X.columns) == [*FEATURE_GROUPS["transit"], *FEATURE_GROUPS["stellar"]]
    assert all(np.issubdtype(t, np.number) for t in X.dtypes)


def test_univariate_audit_detects_value_and_missingness_signal():
    rng = np.random.default_rng(0)
    y = pd.Series(np.repeat([0, 1], 200))
    X = pd.DataFrame(
        {
            "noise": rng.normal(size=400),
            "leak": y + rng.normal(scale=0.01, size=400),
            "inverted": -y.astype(float),
            "missing_when_negative": np.where(y == 1, 1.0, np.nan),
        }
    )
    audit = univariate_audit(X, y)

    assert audit.loc["noise", "auc_value"] < 0.6
    assert audit.loc["leak", "auc_value"] == pytest.approx(1.0)
    assert audit.loc["inverted", "auc_value"] == pytest.approx(1.0)  # symmetrised
    assert audit.loc["missing_when_negative", "auc_missing"] == pytest.approx(1.0)
    assert audit.loc["missing_when_negative", "missing_fraction"] == pytest.approx(0.5)

    flagged = flag_suspicious(audit, max_value_auc=0.9, max_missing_auc=0.6)
    assert set(flagged.index) == {"leak", "inverted", "missing_when_negative"}
