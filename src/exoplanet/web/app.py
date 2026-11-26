"""The model-checker website: pick a KOI, see what each model said and whether it was right.

Read-only. Every score shown was saved by the pipeline; nothing is scored here.
"""

from __future__ import annotations

import math
from typing import Any

from flask import (
    Flask,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)

from exoplanet.config import REPO_ROOT
from exoplanet.data.splits import SPLIT
from exoplanet.web.query import Filters, field_points, koi_detail, koi_views, list_kois
from exoplanet.web.report import REPORT_TABS, render_doc
from exoplanet.web.store import Store, load_store

MAX_PAGE_SIZE = 200
SPLITS = ("dev", "test")


class BadRequest(ValueError):
    pass


def _text(name: str) -> str | None:
    value = request.args.get(name, "").strip()
    return value or None


def _float(name: str, low: float | None = None, high: float | None = None) -> float | None:
    raw = _text(name)
    if raw is None:
        return None
    try:
        value = float(raw)
    except ValueError:
        raise BadRequest(f"{name} must be a number, got {raw!r}") from None
    if not math.isfinite(value):
        raise BadRequest(f"{name} must be a finite number, got {raw!r}")
    if (low is not None and value < low) or (high is not None and value > high):
        raise BadRequest(f"{name} must be between {low} and {high}, got {raw!r}")
    return value


def _int(name: str, default: int, low: int = 1) -> int:
    raw = _text(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise BadRequest(f"{name} must be a whole number, got {raw!r}") from None
    if value < low:
        raise BadRequest(f"{name} must be at least {low}, got {raw!r}")
    return value


def _choice(name: str, allowed: tuple[str, ...] | list[str]) -> str | None:
    raw = _text(name)
    if raw is not None and raw not in allowed:
        raise BadRequest(f"{name} must be one of {', '.join(allowed)}; got {raw!r}")
    return raw


def _filters(store: Store) -> Filters:
    model = _choice("model", store.models) or store.models[0]
    threshold = _float("threshold", 0.0, 1.0)
    label = _choice("label", ("0", "1"))
    return Filters(
        model=model,
        threshold=store.thresholds[model] if threshold is None else threshold,
        q=request.args.get("q", ""),
        split=_choice("split", SPLITS),
        label=None if label is None else int(label),
        mistakes_only=_choice("mistakes", ("0", "1")) == "1",
        snr_min=_float("snr_min"),
        snr_max=_float("snr_max"),
        period_min=_float("period_min"),
        period_max=_float("period_max"),
    )


def create_app(config: dict[str, Any]) -> Flask:
    app = Flask(__name__)
    store = load_store(config)
    app.config["STORE"] = store
    docs_dir = config["paths"].get("docs_dir", REPO_ROOT / "docs")

    @app.errorhandler(BadRequest)
    def bad_request(error: BadRequest):
        return jsonify(error=str(error)), 400

    @app.get("/")
    def checker():
        return render_template("checker.html")

    @app.get("/api/meta")
    def meta():
        split = store.kois[SPLIT]
        return jsonify(
            models=[
                {
                    "name": m,
                    "threshold": store.thresholds[m],
                    "threshold_is_default": store.threshold_is_default[m],
                    "scored_on_test": m in store.test_models,
                }
                for m in store.models
            ],
            counts={
                "total": len(split),
                "dev": int((split == "dev").sum()),
                "test": int((split == "test").sum()),
            },
        )

    @app.get("/api/kois")
    def kois():
        if not store.models:
            raise BadRequest("no model scores found in the outputs directory")
        page_size = min(_int("page_size", 50), MAX_PAGE_SIZE)
        return jsonify(list_kois(store, _filters(store), _int("page", 1), page_size))

    @app.get("/api/field")
    def field():
        if not store.models:
            raise BadRequest("no model scores found in the outputs directory")
        return jsonify(field_points(store, _filters(store)))

    @app.get("/api/koi/<name>")
    def koi(name: str):
        name = name.strip().upper()
        detail = koi_detail(store, name)
        if detail is None:
            return jsonify(error=f"no labelled KOI named {name!r}"), 404
        return jsonify({**detail, "views": koi_views(store, config, name)})

    @app.get("/report")
    def report_home():
        return redirect(url_for("report", name=REPORT_TABS[0][0]))

    @app.get("/report/<name>")
    def report(name: str):
        name = name.removesuffix(".md")
        doc = render_doc(docs_dir, name)
        if doc is None:
            abort(404)
        return render_template("report.html", doc=doc, tabs=REPORT_TABS, current=name)

    @app.get("/report/figures/<path:filename>")
    def report_figure(filename: str):
        return send_from_directory(docs_dir / "figures", filename)

    return app
