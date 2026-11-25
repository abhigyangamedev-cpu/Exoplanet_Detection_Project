from pathlib import Path

import pytest

from exoplanet.config import REPO_ROOT
from exoplanet.web.app import create_app
from web_fixtures import make_project


@pytest.fixture
def client(tmp_path):
    return create_app(make_project(tmp_path)).test_client()


def test_meta(client):
    m = client.get("/api/meta").get_json()
    assert [x["name"] for x in m["models"]] == ["cnn_full", "tabular_lc", "cnn"]
    assert m["models"][0] == {
        "name": "cnn_full",
        "threshold": 0.831,
        "threshold_is_default": False,
        "scored_on_test": False,
    }
    assert m["models"][2]["threshold_is_default"] is True and m["models"][2]["scored_on_test"]
    assert m["counts"] == {"total": 6, "dev": 4, "test": 2}


def test_list_defaults_and_filters(client):
    assert client.get("/api/kois").get_json()["total"] == 6
    out = client.get("/api/kois?split=dev&mistakes=1").get_json()
    assert [i["kepoi_name"] for i in out["items"]] == ["K00003.02"]


def test_list_uses_the_picked_model_and_its_default_threshold(client):
    out = client.get("/api/kois?split=test&model=tabular_lc").get_json()
    # K00001.01 planet scored 0.77 < 0.840 -> wrong; K00002.01 false positive scored 0.90 -> wrong
    assert (out["right"], out["wrong"], out["unscored"]) == (0, 2, 0)
    out = client.get("/api/kois?split=test&model=tabular_lc&threshold=0.8").get_json()
    assert out["wrong"] == 2
    out = client.get("/api/kois?split=test&model=tabular_lc&threshold=0.5").get_json()
    assert (out["right"], out["wrong"]) == (1, 1)


def test_page_size_is_capped(client):
    out = client.get("/api/kois?page_size=100000").get_json()
    assert out["total"] == 6 and out["pages"] == 1


def test_detail_and_404(client):
    d = client.get("/api/koi/K00003.02").get_json()
    assert d["label"] == 1 and len(d["views"]["global_view"]) == 2001
    r = client.get("/api/koi/K99999.01")
    assert r.status_code == 404 and "error" in r.get_json()


def test_detail_without_views_is_not_an_error(client):
    r = client.get("/api/koi/K00004.01")
    assert r.status_code == 200 and r.get_json()["views"] is None


def test_json_never_contains_nan(client):
    for url in ["/api/kois?page_size=200", "/api/koi/K00005.01", "/api/koi/K00001.01"]:
        text = client.get(url).get_data(as_text=True)
        assert "NaN" not in text and "Infinity" not in text


@pytest.mark.parametrize(
    "qs",
    [
        "snr_min=abc",
        "snr_max=nan",
        "page=0",
        "page=x",
        "page_size=0",
        "model=nope",
        "threshold=2",
        "threshold=-0.1",
        "split=train",
        "label=5",
        "mistakes=maybe",
    ],
)
def test_bad_params_are_400(client, qs):
    r = client.get(f"/api/kois?{qs}")
    assert r.status_code == 400 and "error" in r.get_json()


def test_empty_params_mean_no_filter(client):
    r = client.get("/api/kois?q=&split=&label=&snr_min=&period_max=&mistakes=0")
    assert r.status_code == 200 and r.get_json()["total"] == 6


def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200 and b"<html" in r.data.lower()


def test_checker_page_wires_up_assets(client):
    html = client.get("/").get_data(as_text=True)
    assert "static/checker.js" in html and "static/charts.js" in html and "static/app.css" in html
    assert "http://" not in html and "https://" not in html  # offline: nothing external
    for path in ["/static/checker.js", "/static/charts.js", "/static/app.css"]:
        r = client.get(path)
        assert r.status_code == 200 and len(r.data) > 500


def test_web_package_runs_no_model():
    web = REPO_ROOT / "src" / "exoplanet" / "web"
    src = "".join(p.read_text(encoding="utf-8") for p in Path(web).glob("*.py"))
    assert "torch" not in src and "joblib" not in src and "predict" not in src
