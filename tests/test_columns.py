import pandas as pd
import pytest

from exoplanet.features.columns import column_table, describe
from exoplanet.features.leakage import DENIED
from exoplanet.features.tabular import FEATURE_GROUPS


def test_every_policy_column_is_described():
    columns = list(DENIED) + [c for cols in FEATURE_GROUPS.values() for c in cols]
    for col in columns:
        assert describe(col)


def test_uncertainty_columns_are_derived():
    assert describe("koi_period_err1") == "Upper uncertainty of: orbital period (days)"
    assert describe("koi_depth_err2").startswith("Lower uncertainty of:")
    with pytest.raises(KeyError):
        describe("not_a_column")


def test_column_table_decisions():
    audit = pd.DataFrame({"auc_value": [0.98], "auc_missing": [0.5]}, index=["koi_score"])
    table = column_table(
        ["koi_period", "koi_score", "koi_count", "koi_fwm_srao"], ["transit"], audit
    )
    assert table.loc["koi_period", "decision"] == "keep"
    assert table.loc["koi_score", "decision"] == "drop"
    assert table.loc["koi_count", "decision"] == "keep (optional, off)"
    assert table.loc["koi_fwm_srao", "decision"] == "keep (optional, off)"
    assert table.loc["koi_score", "auc_value"] == 0.98
    assert "Median 1.0" in table.loc["koi_score", "reason"]


def test_all_catalogue_columns_are_described():
    from pathlib import Path

    columns = (Path(__file__).parent / "fixtures" / "koi_columns.txt").read_text().split()
    table = column_table(columns, ["transit", "stellar", "centroid"])
    assert len(table) == 153
    assert table["meaning"].str.len().gt(0).all()
