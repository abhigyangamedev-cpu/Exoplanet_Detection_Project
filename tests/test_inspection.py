import pandas as pd
import pytest

from exoplanet.data.inspection import render_report, summarise_catalogue


@pytest.fixture
def catalogue():
    return pd.DataFrame(
        {
            "kepid": [1, 1, 1, 2, 3, 3],
            "kepoi_name": ["K1.01", "K1.02", "K1.03", "K2.01", "K3.01", "K3.02"],
            "koi_disposition": [
                "CONFIRMED",
                "CONFIRMED",
                "FALSE POSITIVE",
                "FALSE POSITIVE",
                "CANDIDATE",
                "CONFIRMED",
            ],
            "koi_pdisposition": [
                "CANDIDATE",
                "CANDIDATE",
                "FALSE POSITIVE",
                "FALSE POSITIVE",
                "CANDIDATE",
                "CANDIDATE",
            ],
            "koi_period": [1.0, 2.0, 3.0, 4.0, None, 6.0],
            "koi_empty": [None] * 6,
        }
    )


def test_summary_counts(catalogue):
    s = summarise_catalogue(catalogue)
    assert s["n_kois"] == 6
    assert s["n_stars"] == 3
    assert s["n_labelled"] == 5
    assert s["n_labelled_stars"] == 3
    assert s["positive_fraction"] == pytest.approx(3 / 5)
    assert s["kois_per_star"] == {1: 1, 2: 1, 3: 1}
    assert s["n_mixed_label_stars"] == 1  # star 1 has CONFIRMED and FALSE POSITIVE
    assert s["n_columns_all_missing"] == 1
    assert s["missing_fraction"]["koi_period"] == pytest.approx(1 / 6)


def test_multiplicity_confirmed_fraction(catalogue):
    table = summarise_catalogue(catalogue)["multiplicity_by_class"]
    # Star 1 (3 KOIs): 2 CONFIRMED, 1 FP.
    assert table.loc[3, "CONFIRMED fraction"] == pytest.approx(2 / 3)
    assert table.loc[1, "CONFIRMED fraction"] == 0.0


def test_render_report_contains_sections(catalogue):
    report = render_report(
        summarise_catalogue(catalogue),
        metadata={
            "source": "https://example.invalid",
            "query": "select * from cumulative",
            "retrieved_utc": "2026-10-31T00:00:00+00:00",
            "sha256": "0" * 64,
        },
    )
    for heading in (
        "## Overview",
        "## Dispositions",
        "## Host-star structure",
        "## Missing values",
    ):
        assert heading in report
    assert "select * from cumulative" in report
