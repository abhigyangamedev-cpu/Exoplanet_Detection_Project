"""Retrieval of the KOI cumulative table from the NASA Exoplanet Archive TAP service."""

from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

REQUIRED_COLUMNS = ("kepid", "kepoi_name", "koi_disposition")


def build_query(table: str, columns: list[str] | None = None) -> str:
    """ADQL query selecting ``columns`` (default: all) from ``table``."""
    selection = ", ".join(columns) if columns else "*"
    return f"select {selection} from {table}"


def make_session(retries: int) -> requests.Session:
    """HTTP session that retries transient failures (the archive returns 5xx intermittently)."""
    retry = Retry(
        total=retries,
        backoff_factor=2.0,
        status_forcelist=(429, 500, 502, 503, 504, 520, 522, 524, 525),
        allowed_methods=("GET",),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def catalogue_path(config: dict[str, Any]) -> Path:
    return config["paths"]["raw_dir"] / config["catalogue"]["filename"]


def metadata_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".meta.json")


def _parse_and_validate(text: str) -> pd.DataFrame:
    # TAP errors can arrive as a VOTable document with HTTP 200.
    if text.lstrip().startswith("<"):
        raise ValueError(f"TAP service returned an error document:\n{text[:500]}")
    df = pd.read_csv(io.StringIO(text), low_memory=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Catalogue is missing required columns: {missing}")
    if df.empty:
        raise ValueError("Catalogue query returned no rows")
    if df["kepoi_name"].duplicated().any():
        raise ValueError("Catalogue contains duplicate kepoi_name entries")
    return df


def fetch_koi_catalogue(
    config: dict[str, Any],
    force: bool = False,
    session: requests.Session | None = None,
) -> Path:
    """Download the KOI table to ``raw_dir`` with a provenance sidecar; reuse the cache unless
    ``force``.
    """
    csv_path = catalogue_path(config)
    if csv_path.exists() and not force:
        return csv_path

    cat = config["catalogue"]
    query = build_query(cat["table"])
    session = session or make_session(cat["retries"])
    response = session.get(
        cat["tap_url"],
        params={"query": query, "format": "csv"},
        timeout=cat["timeout_s"],
    )
    response.raise_for_status()
    df = _parse_and_validate(response.text)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = csv_path.with_suffix(".tmp")
    tmp_path.write_bytes(response.content)
    tmp_path.replace(csv_path)

    metadata = {
        "source": cat["tap_url"],
        "query": query,
        "retrieved_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "sha256": hashlib.sha256(response.content).hexdigest(),
    }
    metadata_path(csv_path).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return csv_path


def load_koi_catalogue(config: dict[str, Any]) -> pd.DataFrame:
    """Load the cached KOI table; run ``fetch_koi_catalogue`` first."""
    csv_path = catalogue_path(config)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found; run scripts/fetch_catalogue.py to download it"
        )
    return pd.read_csv(csv_path, low_memory=False)
