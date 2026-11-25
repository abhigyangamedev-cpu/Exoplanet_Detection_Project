import pytest

from exoplanet.web.query import Filters, koi_detail, koi_views, list_kois
from exoplanet.web.store import load_store
from web_fixtures import make_project


@pytest.fixture
def config(tmp_path):
    return make_project(tmp_path)


@pytest.fixture
def store(config):
    return load_store(config)


def base(store, **kw):
    return Filters(model="cnn_full", threshold=store.thresholds["cnn_full"], **kw)


def names(out):
    return [i["kepoi_name"] for i in out["items"]]


def test_verdict_and_tallies(store):
    out = list_kois(store, base(store, split="dev"))
    by = {i["kepoi_name"]: i for i in out["items"]}
    assert by["K00003.01"]["verdict"] == 1 and by["K00003.01"]["correct"] is True  # 0.91 >= 0.831
    assert by["K00003.02"]["verdict"] == 0 and by["K00003.02"]["correct"] is False  # planet, 0.40
    assert by["K00005.01"]["correct"] is None and by["K00005.01"]["score"] is None  # no score
    assert (out["total"], out["right"], out["wrong"], out["unscored"]) == (4, 2, 1, 1)


def test_threshold_moves_the_verdict(store):
    out = list_kois(store, Filters(model="cnn_full", threshold=0.3, split="dev"))
    assert out["wrong"] == 0


def test_score_exactly_at_threshold_is_planet(store):
    out = list_kois(store, Filters(model="cnn_full", threshold=0.91, q="K00003.01"))
    assert out["items"][0]["verdict"] == 1


def test_mistakes_only(store):
    assert names(list_kois(store, base(store, mistakes_only=True))) == ["K00003.02"]


@pytest.mark.parametrize("q", ["k00003.02", "K00003.02", " K00003.02 "])
def test_search_by_name_any_case(store, q):
    assert names(list_kois(store, base(store, q=q))) == ["K00003.02"]


def test_search_partial_name_and_kic(store):
    assert list_kois(store, base(store, q="K00003"))["total"] == 2
    kepid = str(store.kois.loc["K00003.01", "kepid"])
    assert list_kois(store, base(store, q=kepid))["total"] == 2


def test_range_filters_drop_missing_values(store):
    assert "K00005.01" not in names(list_kois(store, base(store, snr_min=0)))  # NaN SNR excluded


def test_range_filters_bound_both_sides(store):
    out = list_kois(store, base(store, period_min=5, period_max=10))
    assert names(out) == ["K00002.01", "K00003.01"]


def test_label_filter_and_paging(store):
    out = list_kois(store, base(store, label=1), page=2, page_size=2)
    assert out["total"] == 4 and out["pages"] == 2 and len(out["items"]) == 2


def test_page_past_the_end_is_empty_not_an_error(store):
    out = list_kois(store, base(store), page=9, page_size=2)
    assert out["items"] == [] and out["total"] == 6


def test_test_koi_of_dev_only_model_is_not_scored(store):
    d = koi_detail(store, "K00001.01")
    s = {x["model"]: x for x in d["scores"]}
    assert s["cnn_full"]["status"] == "not scored on test" and s["cnn_full"]["score"] is None
    assert s["cnn_full"]["source"] is None
    assert s["tabular_lc"]["source"] == "held-out test"
    assert s["tabular_lc"]["score"] == pytest.approx(0.77)


def test_dev_detail_sources_and_missing(store):
    d = koi_detail(store, "K00005.01")
    s = {x["model"]: x for x in d["scores"]}
    assert s["cnn_full"]["status"] == "no score"
    assert s["tabular_lc"]["source"] == "out-of-fold" and s["tabular_lc"]["status"] == "scored"
    assert d["catalogue"]["koi_model_snr"] is None
    assert d["catalogue"]["koi_period"] == pytest.approx(2.53)
    assert (d["split"], d["fold"], d["label"], d["kepid"]) == ("dev", 2, 1, 1005)
    assert koi_detail(store, "K99999.01") is None


def test_detail_names_the_false_positive_flags(store):
    assert koi_detail(store, "K00004.01")["fp_flags"] == ["stellar eclipse", "centroid offset"]
    assert koi_detail(store, "K00003.01")["fp_flags"] == []


def test_views(store, config):
    v = koi_views(store, config, "K00003.02")
    assert len(v["global_view"]) == 2001 and len(v["local_view"]) == 201
    assert v["local_view"][5] is None  # a gap, not NaN
    assert set(v) == {"global_view", "local_view", "odd_view", "even_view", "secondary_view"}
    assert koi_views(store, config, "K00004.01") is None  # star has no npz
    assert koi_views(store, config, "K99999.01") is None


def test_views_none_when_the_star_file_lacks_the_koi(store, config):
    import numpy as np

    path = config["paths"]["views_dir"] / "1003.npz"
    z = dict(np.load(path))
    np.savez(path, **{k: v[:1] for k, v in z.items()})  # keep only K00003.01
    assert koi_views(store, config, "K00003.02") is None
