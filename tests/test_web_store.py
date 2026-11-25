import numpy as np
import pytest

from exoplanet.web.store import load_store, parse_thresholds
from web_fixtures import make_project


def test_dev_scores_come_from_oof_and_test_scores_from_test_file(tmp_path):
    store = load_store(make_project(tmp_path))
    assert store.kois.loc["K00003.01", "score_cnn_full"] == pytest.approx(0.91)  # OOF value
    assert store.kois.loc["K00001.01", "score_tabular_lc"] == pytest.approx(0.77)  # test file
    assert np.isnan(store.kois.loc["K00001.01", "score_cnn_full"])  # never scored on test
    assert np.isnan(store.kois.loc["K00003.01", "score_cnn"])  # no OOF file for cnn


def test_models_ordered_and_test_models_known(tmp_path):
    store = load_store(make_project(tmp_path))
    assert store.models == ["cnn_full", "tabular_lc", "cnn"]
    assert store.test_models == frozenset({"tabular_lc", "cnn"})


def test_thresholds_from_dev_table_else_default(tmp_path):
    store = load_store(make_project(tmp_path))
    assert store.thresholds == {"cnn_full": 0.831, "tabular_lc": 0.840, "cnn": 0.5}
    assert store.threshold_is_default == {"cnn_full": False, "tabular_lc": False, "cnn": True}


def test_parse_thresholds_reads_only_the_dev_table():
    from web_fixtures import OPERATING_POINTS

    assert parse_thresholds(OPERATING_POINTS) == {"tabular_lc": 0.840, "cnn_full": 0.831}


def test_has_views_and_missing_snr(tmp_path):
    store = load_store(make_project(tmp_path))
    assert not store.kois.loc["K00004.01", "has_views"]
    assert store.kois.loc["K00003.02", "has_views"]
    assert np.isnan(store.kois.loc["K00005.01", "koi_model_snr"])


def test_starts_without_test_scores_or_operating_points(tmp_path):
    config = make_project(tmp_path)
    (config["paths"]["outputs_dir"] / "evaluation_test_scores.csv").unlink()
    (config["paths"]["outputs_dir"] / "operating_points.md").unlink()
    store = load_store(config)
    assert store.test_models == frozenset()
    assert set(store.thresholds.values()) == {0.5}
