"""Tests for video/extract_data.py: numbers and arrays pulled from project files."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from exoplanet.config import REPO_ROOT, load_config

sys.path.insert(0, str(REPO_ROOT / "video"))

from extract_data import (  # noqa: E402
    audit_columns,
    error_breakdown,
    extract_numbers,
    extract_star,
    markdown_tables,
    pick_examples,
    pr_curves,
    splits_sample,
)

CACHE = Path("C:/kepler_cache")
OUTPUTS = REPO_ROOT / "outputs"


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def numbers(config):
    return extract_numbers(config)


def test_numbers_match_project_files(numbers):
    n = numbers
    assert n["n_confirmed"] + n["n_false_positive"] == n["n_kois"]
    assert n["n_test"] + n["n_dev"] == n["n_kois"]
    assert n["n_audit_columns"] == 153
    assert 0 < n["n_audit_removed"] < n["n_audit_columns"]
    assert 0.9 < n["leak_lc_audited"] < n["leak_lc_plus_vetting"] <= 1.0
    # the light-curve-comparable model's inputs, before and after the six vetting columns
    assert isinstance(n["leak_lc_n_features"], int) and n["leak_lc_n_features"] > 0
    assert n["leak_lc_plus_vetting_n_features"] == n["leak_lc_n_features"] + 6
    assert n["sources"]["leak_lc_n_features"].endswith("leakage_inflation.md")
    assert n["sources"]["leak_lc_plus_vetting_n_features"].endswith("leakage_inflation.md")
    assert set(n["test_pr_auc"]) == {"tabular", "tabular_lc", "tabular_lc_sub", "cnn"}
    assert set(n["dev_full_pr_auc"]) == {"tabular_lc", "cnn_full"}
    assert 0 < n["n_views_built"] <= n["n_kois"]
    assert all(k in n["sources"] for k in ("leak_lc_audited", "test_pr_auc"))
    assert set(n["sources"]) >= set(n) - {"sources"}


def test_pr_auc_values_are_the_saved_evaluations(numbers):
    test = json.loads((OUTPUTS / "evaluation_test.json").read_text(encoding="utf-8"))
    for model, (value, lo, hi) in numbers["test_pr_auc"].items():
        saved = test["pr_auc"][model]
        assert (value, lo, hi) == (saved["pr_auc"], saved["ci_low"], saved["ci_high"])
        assert lo <= value <= hi


def test_text_fields(numbers):
    assert numbers["cnn_full_train_time_text"].endswith(" s")
    assert "min" in numbers["cnn_full_train_time_text"]
    assert len(numbers["next_steps"]) == 3
    assert all(0 < len(s) <= 70 and "`" not in s for s in numbers["next_steps"])
    assert numbers["sources"]["next_steps"] == "docs/evaluation.md"


def test_markdown_tables_parses_rows(tmp_path):
    md = tmp_path / "t.md"
    md.write_text("intro\n\n| a | b |\n|---|---|\n| x | 1 |\n| y | 2 |\n\ntext\n", encoding="utf-8")
    (table,) = markdown_tables(md)
    assert table == [{"a": "x", "b": "1"}, {"a": "y", "b": "2"}]


def test_audit_columns(config, numbers):
    cols = audit_columns(config)
    assert len(cols) == 153
    assert set(cols[0]) == {"column", "group", "kept"}
    assert sum(not c["kept"] for c in cols) == numbers["n_audit_removed"]
    by_name = {c["column"]: c for c in cols}
    assert not by_name["koi_score"]["kept"] and by_name["koi_period"]["kept"]


def test_error_breakdown_matches_saved_report():
    breakdown = error_breakdown()
    tables = markdown_tables(OUTPUTS / "evaluation_test.md")
    for key, column in (("by_snr", "koi_model_snr"), ("by_period", "koi_period")):
        (saved,) = [t for t in tables if next(iter(t[0])) == column]
        assert [b["bin"] for b in breakdown[key]] == [r[column] for r in saved]
        assert [round(b["recall"], 3) for b in breakdown[key]] == [float(r["cnn"]) for r in saved]
    assert breakdown["source"] == "outputs/evaluation_test_scores.csv"


def test_pr_curves_keys_and_ranges():
    curves = pr_curves(n_boot=20)
    for m in ("tabular", "tabular_lc", "tabular_lc_sub", "cnn", "dev_tabular_lc", "dev_cnn_full"):
        p, r = curves[f"precision_{m}"], curves[f"recall_{m}"]
        lo, hi = curves[f"band_lo_{m}"], curves[f"band_hi_{m}"]
        assert p.shape == r.shape == lo.shape == hi.shape
        assert ((0 <= p) & (p <= 1)).all() and np.all(np.diff(r) > 0)
        assert (lo <= hi + 1e-12).all()


def test_splits_sample(config):
    sample = splits_sample(config)
    assert len(sample) == 60 and len({s["kepid"] for s in sample}) == 60
    assert {s["partition"] for s in sample} <= {"test", *(f"dev fold {k}" for k in range(5))}
    assert any(s["partition"] == "test" for s in sample)


def test_extract_star_missing_raises_with_id(config):
    with pytest.raises(LookupError, match="K99999.99"):
        extract_star("K99999.99", config)


@pytest.mark.skipif(not CACHE.exists(), reason="needs cache")
def test_pick_examples(config):
    planet, binary = pick_examples(config)
    assert binary == "K06252.01"
    assert planet != binary and planet.startswith("K")


@pytest.mark.skipif(not CACHE.exists(), reason="needs cache")
def test_extract_star_shapes(config):
    planet, _ = pick_examples(config)
    s = extract_star(planet, config)
    assert s["global_view"].shape == (2001,) and s["local_view"].shape == (201,)
    assert len(s["time"]) == len(s["raw_flux"]) == len(s["trend"]) > 10_000
    assert len(s["stitched_flux"]) == len(s["quarter_index"]) == len(s["kept"]) == len(s["time"])
    assert len(s["detrended_flux"]) == len(s["phase"]) == s["kept"].sum()
    assert np.isfinite(s["detrended_flux"]).all()
    assert np.all(np.diff(s["time"]) > 0)
    assert str(s["koi_name"]) == planet and 3 <= float(s["period"]) <= 15
    # Re-running the pipeline step by step reproduces the saved CNN input exactly.
    with np.load(config["paths"]["views_dir"] / f"{int(s['kepid'])}.npz") as saved:
        i = list(saved["kepoi_name"]).index(planet)
        np.testing.assert_allclose(s["global_view"], saved["global_view"][i])
        np.testing.assert_allclose(s["local_view"], saved["local_view"][i])
