import pytest

from exoplanet.web.app import create_app
from exoplanet.web.report import render_doc
from web_fixtures import make_project


@pytest.fixture
def config(tmp_path):
    return make_project(tmp_path)


@pytest.fixture
def docs_dir(config):
    return config["paths"]["docs_dir"]


@pytest.fixture
def client(config):
    return create_app(config).test_client()


def test_render_doc_tables_headings_toc(docs_dir):
    doc = render_doc(docs_dir, "results")
    assert "<table>" in doc["html"] and 'src="figures/a.png"' in doc["html"]
    assert doc["title"] == "Results"
    assert doc["toc"][0] == {"id": "results", "text": "Results", "level": 1}
    assert doc["toc"][1] == {"id": "data", "text": "Data", "level": 2}
    assert '<h2 id="data">' in doc["html"]


def test_repeated_headings_get_distinct_ids(docs_dir):
    (docs_dir / "twice.md").write_text("# T\n\n## Same\n\n## Same\n", encoding="utf-8")
    ids = [h["id"] for h in render_doc(docs_dir, "twice")["toc"]]
    assert len(set(ids)) == 3


def test_raw_html_in_a_doc_is_escaped(docs_dir):
    (docs_dir / "raw.md").write_text("# T\n\n<script>alert(1)</script>\n", encoding="utf-8")
    assert "<script>" not in render_doc(docs_dir, "raw")["html"]


@pytest.mark.parametrize("name", ["nope", "../README", "..\\secret", "figures/a", ""])
def test_render_doc_only_serves_docs(docs_dir, name):
    (docs_dir.parent / "README.md").write_text("# secret\n", encoding="utf-8")
    assert render_doc(docs_dir, name) is None


def test_report_routes(client):
    r = client.get("/report")
    assert r.status_code == 302 and r.headers["Location"].endswith("/report/results")
    r = client.get("/report/results")
    assert r.status_code == 200 and b"Evaluation" in r.data and b"<table>" in r.data
    assert client.get("/report/evaluation.md").status_code == 200
    assert client.get("/report/figures/a.png").status_code == 200
    assert client.get("/report/nope").status_code == 404
    assert client.get("/report/figures/../results.md").status_code == 404
    assert client.get("/report/figures/nope.png").status_code == 404


def test_report_page_survives_a_missing_document(client, docs_dir):
    (docs_dir / "results.md").unlink()
    assert client.get("/report/results").status_code == 404
    assert client.get("/report/evaluation").status_code == 200
