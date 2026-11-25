# Model Checker Website Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local website where the user picks any labelled KOI, sees each model's saved score and verdict marked RIGHT or WRONG against the true label, and reads the final report.

**Architecture:** A Flask app in `src/exoplanet/web/` reads the pipeline's saved score CSVs once at start into one table (`Store`), answers filter queries from it, and loads one star's `.npz` views on demand. One hand-built page (plain HTML/CSS/JS, charts drawn in the browser) talks to a small JSON API. No model is ever run.

**Tech Stack:** Python 3.12, Flask >= 3.0, mistune >= 3.0 (markdown), pandas, numpy, pytest; vanilla JS + SVG in the browser (no CDN, works offline).

**Spec:** `docs/superpowers/specs/2026-11-25-model-checker-website-design.md`

## Global Constraints

- The site runs no model: no `torch`, `joblib` or `predict` import anywhere under `src/exoplanet/web/`.
- Score source rule: a dev KOI's score comes only from `outputs/<model>_oof.csv`; a test KOI's score comes only from `outputs/evaluation_test_scores.csv`. Never mixed, never computed.
- A model absent from `evaluation_test_scores.csv` shows "not scored on test" on test KOIs. A model with no row for a dev KOI shows "no score".
- Every score carries a source tag: `"out-of-fold"` or `"held-out test"`.
- Verdict: planet if `score >= threshold`. Default threshold per model: the "precision 0.95" row of the "Development folds" table in `outputs/operating_points.md`; models with no such row default to `0.5`, labelled as a default.
- Server binds to `127.0.0.1` only. Works offline: no external scripts, fonts or stylesheets.
- Flask and mistune go in a new optional extra `web` in `pyproject.toml`; core dependencies unchanged.
- Run Python through the venv: `.venv/Scripts/python`. Ruff line length 100.
- Work on a new branch `feature/model-checker-web` cut from the current branch. Do not touch `outputs/` or `models/`.

## Review Focus

1. Catalogue values that are missing (NaN period, depth, SNR) and missing scores: JSON must carry `null`, never the bare token `NaN` (browsers reject it). Test in Task 3.
2. A KOI whose star has no `.npz`, or whose `.npz` lacks that KOI: detail returns scores with `"views": null`, not a 500. Test in Task 3.
3. Search typed the way a person types it: lowercase `k00752.01`, a partial `K00752`, a bare KIC number `10797460`: all match. Test in Task 2.
4. Bad filter values (`snr_min=abc`, `page=0`, `model=nope`, `threshold=2`): HTTP 400 with a message, not a 500. Test in Task 3.
5. A missing input file (no `evaluation_test_scores.csv`, no `operating_points.md`): the site still starts; test KOIs show "not scored on test", thresholds fall back to 0.5. Test in Task 1.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/exoplanet/web/__init__.py` | empty package marker |
| `src/exoplanet/web/store.py` | load KOIs, labels, catalogue numbers, scores and thresholds into a `Store` |
| `src/exoplanet/web/query.py` | filter the store, verdicts, right/wrong tallies, per-KOI detail and views |
| `src/exoplanet/web/report.py` | render `docs/*.md` to HTML |
| `src/exoplanet/web/app.py` | Flask app factory and routes |
| `src/exoplanet/web/templates/{base,checker,report}.html` | pages |
| `src/exoplanet/web/static/{app.css,checker.js,charts.js}` | look, behaviour, charts |
| `scripts/serve.py` | launcher |
| `tests/web_fixtures.py` | builds a tiny fake project in `tmp_path` |
| `tests/test_web_store.py`, `test_web_query.py`, `test_web_app.py`, `test_web_report.py` | tests |

---

### Task 1: Store (data loading and the score-source rule)

**Files:**
- Create: `src/exoplanet/web/__init__.py`, `src/exoplanet/web/store.py`, `tests/web_fixtures.py`, `tests/test_web_store.py`
- Modify: `pyproject.toml` (add extra `web = ["flask>=3.0", "mistune>=3.0"]`; add `[tool.setuptools.package-data] exoplanet = ["web/templates/*.html", "web/static/*"]`)

**Interfaces:**
- Consumes: `exoplanet.data.splits.load_splits(config)` (columns `kepoi_name, kepid, label, split, fold`), `exoplanet.data.catalogue.load_koi_catalogue(config)`, `exoplanet.evaluation.false_positives.FP_FLAGS` (dict: flag column -> name), `config["paths"]["outputs_dir" | "views_dir"]`.
- Produces:
  - `CONTEXT_COLUMNS = ["koi_period", "koi_depth", "koi_prad", "koi_model_snr", "koi_duration", *FP_FLAGS]`
  - `@dataclass(frozen=True) class Store` with fields
    `kois: pd.DataFrame` (index `kepoi_name`; columns `kepid, label, split, fold`, the context columns present in the catalogue, `has_views: bool`, and one float column `score_<model>` per model, NaN where no score),
    `models: list[str]` (ordered `cnn_full, tabular_lc, tabular` first, the rest alphabetical),
    `test_models: frozenset[str]` (models with a column in the test scores file),
    `thresholds: dict[str, float]`, `threshold_is_default: dict[str, bool]`.
  - `load_store(config: dict[str, Any]) -> Store`
  - `parse_thresholds(text: str, target: str = "precision 0.95") -> dict[str, float]`
  - `tests/web_fixtures.py`: `make_project(tmp_path: Path) -> dict[str, Any]` returning a config whose `paths` point into `tmp_path`, having written: a catalogue and `splits.csv` for 6 KOIs on 5 stars (`K00001.01` test planet, `K00002.01` test FP, `K00003.01`/`K00003.02` dev planets on one star, `K00004.01` dev FP, `K00005.01` dev planet with NaN `koi_model_snr`); `outputs/cnn_full_oof.csv` and `outputs/tabular_lc_oof.csv` for the dev KOIs except `K00005.01` missing from `cnn_full`; `outputs/evaluation_test_scores.csv` with columns `tabular_lc, cnn` only; pinned scores: `cnn_full` OOF `K00003.01` 0.91, `K00003.02` 0.40, `K00004.01` 0.05; `tabular_lc` test `K00001.01` 0.77; `outputs/operating_points.md` copied in shape from the real file with `cnn_full` 0.831 and `tabular_lc` 0.840 in the dev table and different numbers in the test table; `views/<kepid>.npz` for every star except `K00004.01`'s, in the real format (arrays `kepoi_name`, `global_view` (n, 2001), `local_view`/`odd_view`/`even_view`/`secondary_view` (n, 201), `n_in_transit`).

- [ ] **Step 1: Create the branch and commit the spec and plan**

```bash
git checkout -b feature/model-checker-web
git add docs/superpowers/specs/2026-11-25-model-checker-website-design.md docs/superpowers/plans/2026-11-25-model-checker-website.md
git commit -m "docs: model checker website design and plan"
```

- [ ] **Step 2: Install the new dependencies**

Run: `.venv/Scripts/python -m pip install -e ".[web]"`
Expected: Flask installed; `.venv/Scripts/python -c "import flask, mistune"` exits 0.

- [ ] **Step 3: Write `tests/web_fixtures.py` and the failing tests in `tests/test_web_store.py`**

```python
def test_dev_scores_come_from_oof_and_test_scores_from_test_file(tmp_path):
    store = load_store(make_project(tmp_path))
    assert store.kois.loc["K00003.01", "score_cnn_full"] == pytest.approx(0.91)   # OOF value
    assert store.kois.loc["K00001.01", "score_tabular_lc"] == pytest.approx(0.77)  # test file value
    assert np.isnan(store.kois.loc["K00001.01", "score_cnn_full"])  # never scored on test
    assert np.isnan(store.kois.loc["K00003.01", "score_cnn"])       # no OOF file for cnn

def test_models_ordered_and_test_models_known(tmp_path):
    store = load_store(make_project(tmp_path))
    assert store.models == ["cnn_full", "tabular_lc", "cnn"]
    assert store.test_models == frozenset({"tabular_lc", "cnn"})

def test_thresholds_from_dev_table_else_default(tmp_path):
    store = load_store(make_project(tmp_path))
    assert store.thresholds == {"cnn_full": 0.831, "tabular_lc": 0.840, "cnn": 0.5}
    assert store.threshold_is_default == {"cnn_full": False, "tabular_lc": False, "cnn": True}

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
```

- [ ] **Step 4: Run to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_web_store.py -v`
Expected: FAIL, `ModuleNotFoundError: exoplanet.web`.

- [ ] **Step 5: Implement `store.py`**

`load_store`: models = stems of `outputs/*_oof.csv` (strip `_oof`) plus score columns of the test file (every column after `koi_model_snr`). For each model, build one Series: OOF `score` for dev KOIs, test-file column for test KOIs; reindex to the splits index. `has_views` = `views_dir/<kepid>.npz` exists (list the directory once). `parse_thresholds` reads only the section between `## Development folds` and the next `## ` heading, taking rows whose `target` cell equals `target`.

- [ ] **Step 6: Run to verify they pass, lint, commit**

Run: `.venv/Scripts/python -m pytest tests/test_web_store.py -v && .venv/Scripts/python -m ruff check src tests`
Expected: 5 passed; no lint errors.

```bash
git add pyproject.toml src/exoplanet/web tests/web_fixtures.py tests/test_web_store.py
git commit -m "feat(web): score store with out-of-fold / held-out source rule"
```

---

### Task 2: Query (filters, verdicts, tallies, detail)

**Files:**
- Create: `src/exoplanet/web/query.py`, `tests/test_web_query.py`

**Interfaces:**
- Consumes: `Store`, `load_store`, `make_project` (Task 1); `exoplanet.data.lightcurves.star_path(kepid, config)`, `VIEW_ARRAYS`.
- Produces:
  - `@dataclass(frozen=True) class Filters`: `q: str = ""`, `split: str | None = None` (`"dev" | "test"`), `label: int | None = None`, `model: str` (required), `threshold: float` (required), `mistakes_only: bool = False`, `snr_min/snr_max/period_min/period_max: float | None = None`.
  - `score_status(store, model: str, split: str, score: float) -> str` returning `"scored"`, `"not scored on test"` or `"no score"`.
  - `list_kois(store: Store, f: Filters, page: int = 1, page_size: int = 50) -> dict` with keys `total, right, wrong, unscored, page, pages, items`; each item `{kepoi_name, kepid, split, label, score, verdict, correct}` where `verdict`/`correct` are `None` when unscored. Tallies cover the whole filtered set, not the page. Sorted by `kepoi_name`.
  - `koi_detail(store: Store, name: str) -> dict | None`: `{kepoi_name, kepid, split, fold, label, catalogue: {col: value|None}, fp_flags: [names of set flags], scores: [{model, score|None, status, source}]}`; `source` is `"out-of-fold"` for dev, `"held-out test"` for test, `None` when unscored. `None` for an unknown name.
  - `koi_views(store: Store, config, name: str) -> dict | None`: `{global_view, local_view, odd_view, even_view, secondary_view}` as lists of floats (non-finite -> `None`), or `None` when the file or the KOI's row is absent.

- [ ] **Step 1: Write the failing tests**

```python
def base(store, **kw):
    return Filters(model="cnn_full", threshold=store.thresholds["cnn_full"], **kw)

def test_verdict_and_tallies(store):          # fixture: store = load_store(make_project(tmp_path))
    out = list_kois(store, base(store, split="dev"))
    by = {i["kepoi_name"]: i for i in out["items"]}
    assert by["K00003.01"]["verdict"] == 1 and by["K00003.01"]["correct"] is True   # 0.91 >= 0.831
    assert by["K00003.02"]["verdict"] == 0 and by["K00003.02"]["correct"] is False  # planet scored 0.40
    assert by["K00005.01"]["correct"] is None                                        # no score
    assert (out["total"], out["right"], out["wrong"], out["unscored"]) == (4, 2, 1, 1)

def test_threshold_moves_the_verdict(store):
    out = list_kois(store, Filters(model="cnn_full", threshold=0.3, split="dev"))
    assert out["wrong"] == 0

def test_score_exactly_at_threshold_is_planet(store):
    out = list_kois(store, Filters(model="cnn_full", threshold=0.91, q="K00003.01"))
    assert out["items"][0]["verdict"] == 1

def test_mistakes_only(store):
    out = list_kois(store, base(store, mistakes_only=True))
    assert [i["kepoi_name"] for i in out["items"]] == ["K00003.02"]

@pytest.mark.parametrize("q", ["k00003.02", "K00003.02", " K00003.02 "])
def test_search_by_name_any_case(store, q):
    assert [i["kepoi_name"] for i in list_kois(store, base(store, q=q))["items"]] == ["K00003.02"]

def test_search_partial_name_and_kic(store):
    assert list_kois(store, base(store, q="K00003"))["total"] == 2
    kepid = str(store.kois.loc["K00003.01", "kepid"])
    assert list_kois(store, base(store, q=kepid))["total"] == 2

def test_range_filters_drop_missing_values(store):
    out = list_kois(store, base(store, snr_min=0))
    assert "K00005.01" not in [i["kepoi_name"] for i in out["items"]]   # NaN SNR excluded

def test_label_filter_and_paging(store):
    out = list_kois(store, base(store, label=1), page=2, page_size=2)
    assert out["total"] == 4 and out["pages"] == 2 and len(out["items"]) == 2

def test_test_koi_of_dev_only_model_is_not_scored(store):
    d = koi_detail(store, "K00001.01")
    s = {x["model"]: x for x in d["scores"]}
    assert s["cnn_full"]["status"] == "not scored on test" and s["cnn_full"]["score"] is None
    assert s["tabular_lc"]["source"] == "held-out test"

def test_dev_detail_sources_and_missing(store):
    d = koi_detail(store, "K00005.01")
    s = {x["model"]: x for x in d["scores"]}
    assert s["cnn_full"]["status"] == "no score"
    assert s["tabular_lc"]["source"] == "out-of-fold"
    assert d["catalogue"]["koi_model_snr"] is None
    assert koi_detail(store, "K99999.01") is None

def test_views(store, config):
    v = koi_views(store, config, "K00003.02")
    assert len(v["global_view"]) == 2001 and len(v["local_view"]) == 201
    assert koi_views(store, config, "K00004.01") is None   # star has no npz
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_web_query.py -v`
Expected: FAIL, `ModuleNotFoundError: exoplanet.web.query`.

- [ ] **Step 3: Implement `query.py`**

Search: strip and uppercase `q`; match when it is a substring of `kepoi_name` or equals/prefixes the `kepid` as text. `koi_views` opens `star_path(kepid, config)` with `np.load`, finds the row where `kepoi_name == name`, and returns the five `VIEW_ARRAYS` rows. The file path is built from the store's `kepid`, never from request text.

- [ ] **Step 4: Run to verify they pass, lint, commit**

Run: `.venv/Scripts/python -m pytest tests/test_web_query.py -v && .venv/Scripts/python -m ruff check src tests`
Expected: all passed.

```bash
git add src/exoplanet/web/query.py tests/test_web_query.py
git commit -m "feat(web): KOI filters, verdicts and right/wrong tallies"
```

---

### Task 3: Flask app, JSON API and launcher

**Files:**
- Create: `src/exoplanet/web/app.py`, `scripts/serve.py`, `tests/test_web_app.py`, placeholder `src/exoplanet/web/templates/checker.html` (replaced in Task 5)

**Interfaces:**
- Consumes: `load_store`, `Store` (Task 1); `Filters`, `list_kois`, `koi_detail`, `koi_views` (Task 2).
- Produces:
  - `create_app(config: dict[str, Any]) -> flask.Flask` (store loaded once, kept in `app.config["STORE"]`).
  - `GET /` -> checker page (HTML).
  - `GET /api/meta` -> `{"models": [{"name", "threshold", "threshold_is_default", "scored_on_test"}], "counts": {"total", "dev", "test"}}`.
  - `GET /api/kois` with query params `q, split, label, model, threshold, mistakes (0|1), snr_min, snr_max, period_min, period_max, page, page_size` -> `list_kois` result. `model` defaults to the first model, `threshold` to that model's default, `page_size` capped at 200.
  - `GET /api/koi/<name>` -> `koi_detail` result plus `"views": koi_views(...)` (`null` when absent); 404 `{"error": ...}` for an unknown name.
  - Errors: 400 `{"error": "<what was wrong>"}` for an unknown `model`/`split`/`label`, a non-numeric number, `page < 1`, or `threshold` outside [0, 1].
  - `scripts/serve.py`: args `--config`, `--port` (default 8000), `--no-browser`; prints `Model checker: http://127.0.0.1:<port>`, opens the browser, runs on `127.0.0.1`. If the port is taken, exit with a one-line message naming `--port`.

- [ ] **Step 1: Write the failing tests** (fixture `client` = `create_app(make_project(tmp_path)).test_client()`)

```python
def test_meta(client):
    m = client.get("/api/meta").get_json()
    assert [x["name"] for x in m["models"]] == ["cnn_full", "tabular_lc", "cnn"]
    assert m["models"][0] == {"name": "cnn_full", "threshold": 0.831,
                              "threshold_is_default": False, "scored_on_test": False}
    assert m["counts"] == {"total": 6, "dev": 4, "test": 2}

def test_list_defaults_and_filters(client):
    assert client.get("/api/kois").get_json()["total"] == 6
    out = client.get("/api/kois?split=dev&mistakes=1").get_json()
    assert [i["kepoi_name"] for i in out["items"]] == ["K00003.02"]

def test_detail_and_404(client):
    d = client.get("/api/koi/K00003.02").get_json()
    assert d["label"] == 1 and len(d["views"]["global_view"]) == 2001
    assert client.get("/api/koi/K99999.01").status_code == 404

def test_detail_without_views_is_not_an_error(client):
    r = client.get("/api/koi/K00004.01")
    assert r.status_code == 200 and r.get_json()["views"] is None

def test_json_never_contains_nan(client):
    for url in ["/api/kois?page_size=200", "/api/koi/K00005.01", "/api/koi/K00001.01"]:
        text = client.get(url).get_data(as_text=True)
        assert "NaN" not in text and "Infinity" not in text

@pytest.mark.parametrize("qs", ["snr_min=abc", "page=0", "model=nope", "threshold=2",
                                "split=train", "label=5"])
def test_bad_params_are_400(client, qs):
    r = client.get(f"/api/kois?{qs}")
    assert r.status_code == 400 and "error" in r.get_json()

def test_index_serves_html(client):
    r = client.get("/")
    assert r.status_code == 200 and b"<html" in r.data.lower()

def test_web_package_runs_no_model():
    src = "".join(p.read_text(encoding="utf-8") for p in Path("src/exoplanet/web").glob("*.py"))
    assert "torch" not in src and "joblib" not in src and "predict" not in src
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_web_app.py -v`
Expected: FAIL, `ModuleNotFoundError: exoplanet.web.app`.

- [ ] **Step 3: Implement `app.py` and `scripts/serve.py`**

- [ ] **Step 4: Run to verify they pass, then start it on the real data**

Run: `.venv/Scripts/python -m pytest tests/test_web_app.py -v && .venv/Scripts/python -m ruff check src tests scripts`
Expected: all passed.

Run: `.venv/Scripts/python scripts/serve.py --no-browser --port 8765` (background), then
`curl -s "http://127.0.0.1:8765/api/meta"` and `curl -s "http://127.0.0.1:8765/api/kois?split=test&model=tabular_lc&page_size=1"`
Expected: `counts.total` 6677, `counts.test` 1335; the test query has `total` 1335 and `unscored` 0. Stop the server.

- [ ] **Step 5: Commit**

```bash
git add src/exoplanet/web/app.py src/exoplanet/web/templates scripts/serve.py tests/test_web_app.py
git commit -m "feat(web): Flask app, JSON API and launcher"
```

---

### Task 4: Report page

**Files:**
- Create: `src/exoplanet/web/report.py`, `src/exoplanet/web/templates/base.html`, `src/exoplanet/web/templates/report.html`, `tests/test_web_report.py`
- Modify: `src/exoplanet/web/app.py` (routes), `tests/web_fixtures.py` (write `docs/results.md`, `docs/evaluation.md` with a table, a `![x](figures/a.png)` image, a link to `evaluation.md`, and `docs/figures/a.png`)

**Interfaces:**
- Consumes: `create_app` (Task 3). The docs directory is `config["paths"].get("docs_dir", REPO_ROOT / "docs")` so tests can point it at `tmp_path`.
- Produces:
  - `REPORT_TABS = [("results", "Results"), ("evaluation", "Evaluation")]`
  - `render_doc(docs_dir: Path, name: str) -> dict | None` -> `{"title": str, "html": str, "toc": [{"id", "text", "level"}]}` for `docs_dir/<name>.md`; `None` when `name` is not the stem of a `.md` file directly inside `docs_dir`. Headings get `id` attributes; tables render as `<table>`.
  - `GET /report` -> redirects to `/report/results`.
  - `GET /report/<name>` (a trailing `.md` is accepted and stripped) -> page with the two tabs, a contents list from `toc`, and the rendered document; 404 when `render_doc` returns `None`.
  - `GET /report/figures/<path:filename>` -> file from `docs_dir/figures` via `send_from_directory`. Relative `figures/x.png` and `evaluation.md` links in the markdown therefore resolve with no rewriting.

- [ ] **Step 1: Write the failing tests**

```python
def test_render_doc_tables_headings_toc(docs_dir):
    doc = render_doc(docs_dir, "results")
    assert "<table>" in doc["html"] and 'src="figures/a.png"' in doc["html"]
    assert doc["toc"][0]["text"] == "Results" and doc["toc"][0]["id"] in doc["html"]

@pytest.mark.parametrize("name", ["nope", "../README", "..\\secret", "figures/a"])
def test_render_doc_only_serves_docs(docs_dir, name):
    assert render_doc(docs_dir, name) is None

def test_report_routes(client):
    assert client.get("/report").status_code == 302
    r = client.get("/report/results")
    assert r.status_code == 200 and b"Evaluation" in r.data and b"<table>" in r.data
    assert client.get("/report/evaluation.md").status_code == 200
    assert client.get("/report/figures/a.png").status_code == 200
    assert client.get("/report/nope").status_code == 404
    assert client.get("/report/figures/../results.md").status_code == 404
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_web_report.py -v`
Expected: FAIL, `ModuleNotFoundError: exoplanet.web.report`.

- [ ] **Step 3: Implement `report.py`, the routes and the two templates**

Use `mistune.create_markdown(plugins=["table", "strikethrough"])` with a renderer subclass that slugs heading text into `id`s and collects the `toc`. `base.html` holds the top bar with the `Checker` / `Report` links and links `static/app.css`.

- [ ] **Step 4: Run to verify they pass, lint, commit**

Run: `.venv/Scripts/python -m pytest tests/test_web_report.py tests/test_web_app.py -v && .venv/Scripts/python -m ruff check src tests`
Expected: all passed.

```bash
git add src/exoplanet/web tests/web_fixtures.py tests/test_web_report.py
git commit -m "feat(web): report page rendering results.md and evaluation.md"
```

---

### Task 5: Checker page (the interface)

**Files:**
- Create: `src/exoplanet/web/static/app.css`, `src/exoplanet/web/static/checker.js`, `src/exoplanet/web/static/charts.js`
- Modify: `src/exoplanet/web/templates/checker.html` (replace placeholder; extends `base.html`), `src/exoplanet/web/templates/report.html` (shared styling), `tests/test_web_app.py`, `README.md` (a "Model checker website" section: install `.[web]`, `python scripts/serve.py`, what out-of-fold / held-out mean)

**Interfaces:**
- Consumes: `/api/meta`, `/api/kois`, `/api/koi/<name>` exactly as in Task 3.
- Produces: the page in the spec's sketch.

Before writing any markup, invoke the `frontend-design:frontend-design` skill and the `dataviz` skill and follow them for the look and the charts.

**Behaviour the page must have:**
- Left panel: search box (debounced 250 ms), partition (all / dev / test), label (all / planet / false positive), model picker, "only mistakes" checkbox, SNR min/max, period min/max, paged list (50 per page) where each row shows name, true label and a RIGHT / WRONG / no-score mark for the picked model, and a tally line `Shown N · right R · wrong W · no score U` with accuracy `R / (R + W)`.
- Right panel for the picked KOI: name, KIC, partition and fold, true label in words (`CONFIRMED PLANET` / `FALSE POSITIVE`); a table with one row per model: score (3 decimals), verdict, RIGHT / WRONG, and the source tag (`out-of-fold` / `held-out test`); unscored rows show their status text (`not scored on test` / `no score`) in place of a verdict.
- One threshold slider (0 to 1, step 0.001) for the picked model, starting at its default, with a "reset to default" control and a note when the default is the 0.5 fallback. Moving it updates that model's verdict in the right panel at once and re-queries the list and tallies (debounced). Other models keep their own defaults.
- RIGHT / WRONG is never shown by colour alone: always the word, plus colour.
- Four charts from `views`: global (2001 points), local (201), odd and even overlaid (201 each, two series with a legend), secondary (201). Flux on y, bin on x, gaps where a value is `null`. When `views` is `null`, show `No light curve built for this star` in place of the charts.
- Catalogue line: period (days), depth (ppm), planet radius (Earth radii), SNR, duration (hours); `—` for a missing value; the set false-positive flags by name, or `none`.
- The picked KOI and the filters live in the URL query string, so a reload or a shared link reopens the same view. Arrow keys move up and down the list.
- Usable at 1280 px wide and at phone width (panels stack).

- [ ] **Step 1: Add the failing test**

```python
def test_checker_page_wires_up_assets(client):
    html = client.get("/").get_data(as_text=True)
    assert "static/checker.js" in html and "static/charts.js" in html and "static/app.css" in html
    assert "http://" not in html and "https://" not in html      # offline: nothing external
    for path in ["/static/checker.js", "/static/charts.js", "/static/app.css"]:
        assert client.get(path).status_code == 200
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_web_app.py::test_checker_page_wires_up_assets -v`
Expected: FAIL (placeholder page has no assets).

- [ ] **Step 3: Build the page** (`checker.html`, `app.css`, `checker.js`, `charts.js`)

- [ ] **Step 4: Run the whole suite**

Run: `.venv/Scripts/python -m pytest -q && .venv/Scripts/python -m ruff check src tests scripts`
Expected: all passed (the existing tests too).

- [ ] **Step 5: Check it in a real browser on the real data**

Start `.venv/Scripts/python scripts/serve.py --no-browser`, open `http://127.0.0.1:8000` with the browser tools, and confirm each of these, with a screenshot of the first and the fourth:
1. Search `K00752.01`: scores show, each tagged `out-of-fold`, charts draw.
2. Open a test KOI (`K00001.01`): `cnn_full` reads `not scored on test`; `tabular_lc` is tagged `held-out test` with score 0.537.
3. Partition test + model `tabular_lc` + default threshold: tally reads right + wrong = 1335, and wrong is within one or two of the `tabular_lc` "precision 0.95" row of the test table in `outputs/operating_points.md` (planets_missed 107 + wasted_followups 12 = 119; that table covers only KOIs every model scored, and its threshold is rounded to 3 decimals).
4. "Only mistakes" on `cnn_full`, dev: every listed row reads WRONG; drag the slider and the tally changes.
5. Report tab: both documents render with tables and all four figures; the link from Results to `evaluation.md` works.
6. No errors in the browser console.

Fix anything that fails, then stop the server.

- [ ] **Step 6: Commit**

```bash
git add src/exoplanet/web README.md tests/test_web_app.py
git commit -m "feat(web): checker page with light-curve charts and threshold slider"
```
