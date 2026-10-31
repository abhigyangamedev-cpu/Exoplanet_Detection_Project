"""Kepler light-curve retrieval from MAST and per-star view files.

Each host star is downloaded and detrended once; views for all of its modelling-sample KOIs
are written to ``<views_dir>/<kepid>.npz``. Existing view files are skipped, so interrupted
builds resume where they stopped.
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from exoplanet.data.splits import KOI, SPLIT, TEST
from exoplanet.features.lightcurve import Ephemeris, Views, process_star

VIEW_ARRAYS = ("global_view", "local_view")
VIEW_QC = ("global_empty", "local_empty", "n_in_transit")


def ephemerides(df: pd.DataFrame) -> dict[str, Ephemeris]:
    """KOI name -> ephemeris (catalogue durations are in hours), skipping incomplete rows."""
    cols = ["koi_period", "koi_time0bk", "koi_duration"]
    ok = df[cols].notna().all(axis=1) & (df["koi_period"] > 0) & (df["koi_duration"] > 0)
    return {
        row[KOI]: Ephemeris(row.koi_period, row.koi_time0bk, row.koi_duration / 24)
        for _, row in df[ok].iterrows()
    }


def star_cache_dirs(kepid: int, config: dict[str, Any]) -> list[Path]:
    """Cached MAST download folders for a star (one per set of quarters)."""
    root = config["paths"]["lightcurve_dir"] / "mastDownload" / "Kepler"
    return sorted(root.glob(f"kplr{kepid:09d}_lc_*"))


def download_star(kepid: int, config: dict[str, Any]) -> list[tuple[np.ndarray, np.ndarray]]:
    """Download (or read from cache) every long-cadence quarter of a star as (time, flux).

    A cached file truncated by an interrupted download is unreadable but never re-fetched,
    so on a read error the star's cache is cleared and the download retried once.
    """
    import lightkurve as lk

    lc = config["lightcurves"]
    search = lk.search_lightcurve(f"KIC {kepid}", mission="Kepler", author="Kepler", exptime="long")
    if len(search) == 0:
        return []

    def fetch() -> lk.LightCurveCollection:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # per-file units/quality chatter from astropy
            return search.download_all(
                download_dir=str(config["paths"]["lightcurve_dir"]),
                flux_column=lc["flux_column"],
                quality_bitmask=lc["quality_bitmask"],
            )

    try:
        collection = fetch()
    except lk.LightkurveError:
        for directory in star_cache_dirs(kepid, config):
            shutil.rmtree(directory)
        collection = fetch()
    return [(np.asarray(q.time.value, float), np.asarray(q.flux.value, float)) for q in collection]


def view_params(config: dict[str, Any]) -> dict[str, Any]:
    lc, views = config["lightcurves"], config["views"]
    keys = ("mask_width", "window_days", "window_durations", "gap_days", "clip_sigma")
    return {**{k: lc[k] for k in keys}, **views}


def select_stars(
    splits: pd.DataFrame, dev_stars: int | None, built: set[int], seed: int
) -> pd.DataFrame:
    """Host stars to build, test-set stars first so the test set completes before any cut-off.

    All test stars are kept. If ``dev_stars`` is set, development stars are limited to that
    many: those already built first (no download needed), then a seeded random sample of
    the rest. Returns ``kepid``, ``split`` and ``already_built`` in build order.
    """
    stars = splits.groupby("kepid")[SPLIT].first().rename("split").reset_index()
    stars["already_built"] = stars["kepid"].isin(built)
    test = stars[stars["split"] == TEST].sort_values("kepid")
    dev = stars[stars["split"] != TEST]
    if dev_stars is not None:
        have = dev[dev["already_built"]].sort_values("kepid").head(dev_stars)
        rest = dev[~dev["already_built"]]
        fill = rest.sample(n=min(dev_stars - len(have), len(rest)), random_state=seed)
        dev = pd.concat([have, fill.sort_values("kepid")])
    else:
        dev = dev.sort_values("kepid")
    return pd.concat([test, dev]).reset_index(drop=True)


def star_path(kepid: int, config: dict[str, Any]) -> Path:
    return config["paths"]["views_dir"] / f"{kepid}.npz"


def save_star_views(path: Path, views: dict[str, Views]) -> None:
    kois = sorted(views)
    arrays = {KOI: np.array(kois)}
    for name in (*VIEW_ARRAYS, *VIEW_QC):
        arrays[name] = np.stack([np.asarray(getattr(views[k], name)) for k in kois])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)  # atomic, so an interrupted write never looks complete


def build_star(
    kepid: int, targets: pd.DataFrame, star_kois: pd.DataFrame, config: dict[str, Any]
) -> int:
    """Download, detrend and save views for one star. Returns the number of KOIs written.

    ``targets`` are the modelling-sample KOIs on the star; ``star_kois`` are all catalogue
    KOIs on it (any disposition), whose transits are masked before detrending.
    """
    segments = download_star(kepid, config)
    if not segments:
        raise RuntimeError("no Kepler long-cadence light curves found")
    views = process_star(
        segments, ephemerides(targets), ephemerides(star_kois).values(), view_params(config)
    )
    if not views:
        raise RuntimeError("no usable data after detrending")
    save_star_views(star_path(kepid, config), views)
    return len(views)


def load_views(config: dict[str, Any], kois: pd.Series | None = None) -> pd.DataFrame:
    """Assemble saved views into one frame indexed by KOI.

    Columns: ``kepid``, ``global_view`` and ``local_view`` (one array per row) and the QC
    metrics. If ``kois`` is given, only those KOIs are returned (missing ones are absent).
    """
    rows = []
    for path in sorted(config["paths"]["views_dir"].glob("*.npz")):
        if path.name.endswith(".tmp.npz"):
            continue
        with np.load(path) as data:
            for i, koi in enumerate(data[KOI]):
                row = {KOI: str(koi), "kepid": int(path.stem)}
                row.update({name: data[name][i] for name in VIEW_ARRAYS})
                row.update({name: data[name][i].item() for name in VIEW_QC})
                rows.append(row)
    df = pd.DataFrame(rows, columns=[KOI, "kepid", *VIEW_ARRAYS, *VIEW_QC]).set_index(KOI)
    return df if kois is None else df[df.index.isin(kois)]
