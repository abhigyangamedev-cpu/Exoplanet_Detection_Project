"""Pull the numbers, light curves, PR curves and command output the video shows.

Runs in the project env. Everything goes to ``video/build/data/``; narration and scenes
read only those files, so every number on screen traces back to a project file named in
``numbers.json["sources"]``. Nothing is re-scored or retrained: PR curves and the CNN error
breakdown come from the saved ``*_scores.csv`` files.

    .venv\\Scripts\\python video/extract_data.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from exoplanet.config import REPO_ROOT, load_config
from exoplanet.data.catalogue import load_koi_catalogue
from exoplanet.data.labels import LABEL, label_catalogue
from exoplanet.data.lightcurves import (
    complete_cache_files,
    ephemerides,
    read_cached_star,
    star_path,
)
from exoplanet.data.splits import KOI, load_splits
from exoplanet.evaluation.false_positives import recall_by_bin
from exoplanet.evaluation.plots import RECALL_GRID, pr_bands
from exoplanet.features.columns import column_table
from exoplanet.features.lightcurve import (
    clip_upward_outliers,
    detrend,
    make_views,
    phase,
    stitch,
    transit_mask,
)

BUILD = REPO_ROOT / "video" / "build"
DATA = BUILD / "data"
OUTPUTS = "outputs"
TEST_SCORES = f"{OUTPUTS}/evaluation_test_scores.csv"
DEV_SCORES = f"{OUTPUTS}/evaluation_dev_full_scores.csv"
BINARY_EXAMPLE = "K06252.01"  # eclipsing binary with a visible secondary (lightcurve_pipeline.md)
N_SPLIT_STARS = 60


# --- parsing helpers ---------------------------------------------------------------------


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def markdown_tables(path: Path) -> list[list[dict[str, str]]]:
    """Every pipe table in a Markdown file, as lists of {header: cell} rows."""
    tables, block = [], []
    for line in [*path.read_text(encoding="utf-8").splitlines(), ""]:
        if line.startswith("|"):
            block.append([c.strip() for c in line.strip().strip("|").split("|")])
            continue
        if len(block) >= 2:
            header, rows = block[0], block[2:]
            tables.append([dict(zip(header, row, strict=True)) for row in rows])
        block = []
    return tables


def _table_with(rel: str, column: str) -> list[dict[str, str]]:
    for table in markdown_tables(REPO_ROOT / rel):
        if table and column in table[0]:
            return table
    raise LookupError(f"no table with column {column!r} in {rel}")


def _row(table: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    for row in table:
        if row[key] == value:
            return row
    raise LookupError(f"no row {key}={value!r}")


def _section(text: str, heading: str) -> str:
    start = text.index(heading) + len(heading)
    end = text.find("\n#", start)
    return text[start : end if end >= 0 else None]


def _short_step(bullet: str) -> str:
    """First clause of a next-steps bullet, without Markdown backticks."""
    clause = re.split(r"[(:;,]", bullet, maxsplit=1)[0]
    return clause.replace("`", "").strip()


# --- numbers -----------------------------------------------------------------------------


def extract_numbers(config: dict[str, Any]) -> dict[str, Any]:
    """Headline numbers, each from a saved project file recorded under ``sources``."""
    sources: dict[str, str] = {}
    n: dict[str, Any] = {}

    def put(key: str, value: Any, source: str) -> None:
        n[key], sources[key] = value, source

    views = f"{OUTPUTS}/views_summary.md"
    coverage = _table_with(views, "built_fraction")
    put("n_confirmed", int(_row(coverage, "label", "1")["kois"]), views)
    put("n_false_positive", int(_row(coverage, "label", "0")["kois"]), views)
    put("n_kois", n["n_confirmed"] + n["n_false_positive"], views)
    put("n_views_built", sum(int(r["built"]) for r in coverage), views)

    splits = f"{OUTPUTS}/splits_summary.md"
    partitions = _table_with(splits, "partition")
    put("n_test", int(_row(partitions, "partition", "test")["kois"]), splits)
    put(
        "n_dev",
        sum(int(r["kois"]) for r in partitions if r["partition"].startswith("dev")),
        splits,
    )

    audit = f"{OUTPUTS}/leakage_audit_results.md"
    policy = _table_with(audit, "n_columns")
    put("n_audit_columns", sum(int(r["n_columns"]) for r in policy), audit)
    put(
        "n_audit_removed",
        sum(int(r["n_columns"]) for r in policy if r["class"].startswith("denied")),
        audit,
    )

    leak = f"{OUTPUTS}/leakage_inflation.md"
    feature_sets = _table_with(leak, "feature_set")
    lc = _row(feature_sets, "feature_set", "audited_lc")
    lc_vet = _row(feature_sets, "feature_set", "audited_lc_plus_vetting")
    put("leak_lc_audited", float(lc["pr_auc"]), leak)
    put("leak_lc_plus_vetting", float(lc_vet["pr_auc"]), leak)
    put("leak_lc_n_features", int(lc["features"]), leak)
    put("leak_lc_plus_vetting_n_features", int(lc_vet["features"]), leak)

    for key, rel in (
        ("test_pr_auc", f"{OUTPUTS}/evaluation_test.json"),
        ("dev_full_pr_auc", f"{OUTPUTS}/evaluation_dev_full.json"),
    ):
        saved = json.loads(_read(rel))["pr_auc"]
        put(key, {m: [v["pr_auc"], v["ci_low"], v["ci_high"]] for m, v in saved.items()}, rel)

    evaluation = re.sub(r"\s+", " ", _read("docs/evaluation.md"))
    match = re.search(r"(\d+ min \d+ s) for five folds plus the final model", evaluation)
    if match is None:
        raise LookupError("cnn_full training time not found in docs/evaluation.md")
    put("cnn_full_train_time_text", match.group(1), "docs/evaluation.md")

    steps_text = _section(_read("docs/evaluation.md"), "### Most promising next steps")
    bullets = [b.strip() for b in re.split(r"\n- ", "\n" + steps_text.strip()) if b.strip()]
    if len(bullets) != 3:
        raise ValueError(f"expected 3 next steps in docs/evaluation.md, found {len(bullets)}")
    put("next_steps", [_short_step(b) for b in bullets], "docs/evaluation.md")

    n["sources"] = sources
    return n


def audit_columns(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Every catalogue column with its audit class and whether the audit keeps it."""
    table = column_table(load_koi_catalogue(config).columns, config["features"]["groups"])
    return [
        {"column": col, "group": row["class"], "kept": row["decision"] != "drop"}
        for col, row in table.iterrows()
    ]


def error_breakdown() -> dict[str, Any]:
    """CNN test recall on confirmed planets by SNR and period quartile.

    ``evaluation_test.json`` stores the threshold but not this breakdown, so it is recomputed
    from the saved test scores at that threshold, exactly as ``scripts/evaluate.py`` does.
    """
    evaluation = f"{OUTPUTS}/evaluation_test.json"
    threshold = json.loads(_read(evaluation))["operating_points"]["cnn"]["threshold"]
    scores = pd.read_csv(REPO_ROOT / TEST_SCORES)
    out: dict[str, Any] = {"model": "cnn", "threshold": threshold}
    for key, column in (("by_snr", "koi_model_snr"), ("by_period", "koi_period")):
        table = recall_by_bin(scores, {"cnn": threshold}, column)
        out[key] = [
            {"bin": str(b), "n": int(r["n"]), "recall": float(r["cnn"])}
            for b, r in table.iterrows()
        ]
    out["source"] = TEST_SCORES
    out["threshold_source"] = evaluation
    return out


def pr_curves(n_boot: int | None = None, seed: int | None = None) -> dict[str, np.ndarray]:
    """Interpolated PR curves with host-star bootstrap bands from the saved scores.

    Test models keep their names; development curves are prefixed ``dev_``.
    """
    config = load_config()
    n_boot = n_boot or config["evaluation"]["n_bootstrap"]
    seed = config["seed"] if seed is None else seed
    test_models = json.loads(_read(f"{OUTPUTS}/evaluation_test.json"))["models"]
    dev_models = json.loads(_read(f"{OUTPUTS}/evaluation_dev_full.json"))["models"]
    arrays: dict[str, np.ndarray] = {}
    for rel, models, prefix in ((TEST_SCORES, test_models, ""), (DEV_SCORES, dev_models, "dev_")):
        scores = pd.read_csv(REPO_ROOT / rel)
        for model in models:
            precision, lo, hi = pr_bands(scores, model, n_boot=n_boot, seed=seed)
            name = prefix + model
            arrays[f"precision_{name}"] = precision
            arrays[f"recall_{name}"] = RECALL_GRID.copy()
            arrays[f"band_lo_{name}"] = lo
            arrays[f"band_hi_{name}"] = hi
    return arrays


def splits_sample(config: dict[str, Any]) -> list[dict[str, Any]]:
    """A seeded sample of host stars with their partition, for the splits animation."""
    splits = load_splits(config)
    stars = splits.groupby("kepid").agg(
        split=("split", "first"), fold=("fold", "first"), n_kois=(KOI, "size")
    )
    sample = stars.sample(n=N_SPLIT_STARS, random_state=config["seed"]).sort_index()
    return [
        {
            "kepid": int(kepid),
            "partition": "test" if row["split"] == "test" else f"dev fold {int(row['fold'])}",
            "n_kois": int(row["n_kois"]),
        }
        for kepid, row in sample.iterrows()
    ]


# --- light curves ------------------------------------------------------------------------


def _koi_row(koi_name: str, catalogue: pd.DataFrame) -> pd.Series:
    rows = catalogue[catalogue[KOI] == koi_name]
    if rows.empty:
        raise LookupError(f"{koi_name} is not in the KOI catalogue")
    return rows.iloc[0]


def extract_star(koi_name: str, config: dict[str, Any]) -> dict[str, np.ndarray]:
    """Run one KOI's light curve through the view pipeline step by step, keeping each stage.

    Arrays ``time``, ``raw_flux``, ``quarter_index``, ``stitched_flux``, ``trend`` and
    ``kept`` share one time axis (stitch order). ``detrended_flux`` and ``phase`` hold the
    points that survive detrending and outlier clipping (``time[kept]``). ``trend`` is NaN on
    segments the detrender drops. Steps and parameters follow
    ``exoplanet.features.lightcurve.process_star``.
    """
    catalogue = load_koi_catalogue(config)
    row = _koi_row(koi_name, catalogue)
    kepid = int(row["kepid"])
    with warnings.catch_warnings():
        # lightkurve announces an optional submodule it cannot load on import.
        warnings.filterwarnings("ignore", message=".*tpfmodel submodule", category=UserWarning)
        segments = read_cached_star(kepid, config)
    if segments is None:
        raise LookupError(f"{koi_name}: no complete light-curve cache for KIC {kepid}")
    target = ephemerides(catalogue[catalogue[KOI] == koi_name])[koi_name]
    star_ephs = list(ephemerides(catalogue[catalogue["kepid"] == kepid]).values())
    lc, vw = config["lightcurves"], config["views"]

    # Raw quarters, kept in the order stitch() produces, so all stages share one time axis.
    raw_t, raw_f, quarter = [], [], []
    for q, (t, f) in enumerate(segments):
        t, f = np.asarray(t, float), np.asarray(f, float)
        ok = np.isfinite(t) & np.isfinite(f)
        if ok.any() and np.median(f[ok]) > 0:
            raw_t.append(t[ok])
            raw_f.append(f[ok])
            quarter.append(np.full(ok.sum(), q))
    order = np.argsort(np.concatenate(raw_t), kind="stable")
    raw_flux = np.concatenate(raw_f)[order]
    quarter_index = np.concatenate(quarter)[order]

    time, stitched = stitch(segments)
    mask = transit_mask(time, star_ephs, lc["mask_width"])
    longest = max(e.duration for e in star_ephs)
    window = max(lc["window_days"], lc["window_durations"] * longest)
    dt_time, dt_flux = detrend(time, stitched, mask, window, lc["gap_days"])

    retained = np.isin(time, dt_time)
    trend = np.full(len(time), np.nan)
    trend[retained] = stitched[retained] / dt_flux
    dt_mask = transit_mask(dt_time, star_ephs, lc["mask_width"])
    keep = clip_upward_outliers(dt_flux, dt_mask, lc["clip_sigma"])
    kept = np.zeros(len(time), bool)
    kept[np.flatnonzero(retained)[keep]] = True

    final_t, final_f = dt_time[keep], dt_flux[keep]
    views = make_views(
        final_t,
        final_f,
        target,
        vw["global_bins"],
        vw["local_bins"],
        vw["local_durations"],
        vw.get("local_bin_width"),
    )
    return {
        "time": time,
        "raw_flux": raw_flux,
        "quarter_index": quarter_index,
        "stitched_flux": stitched,
        "trend": trend,
        "kept": kept,
        "detrended_flux": final_f,
        "phase": phase(final_t, target.period, target.epoch),
        "global_view": views.global_view,
        "local_view": views.local_view,
        "secondary_view": views.secondary_view,
        "kepid": np.array(kepid),
        "koi_name": np.array(koi_name),
        "period": np.array(target.period),
        "epoch": np.array(target.epoch),
        "duration_hours": np.array(target.duration * 24),
    }


def _saved_view(kepid: int, koi: str, name: str, config: dict[str, Any]) -> np.ndarray | None:
    path = star_path(kepid, config)
    if not path.exists():
        return None
    with np.load(path) as data:
        names = list(data[KOI])
        return data[name][names.index(koi)] if koi in names else None


def pick_examples(config: dict[str, Any]) -> tuple[str, str]:
    """(confirmed planet, false positive) to animate, both with complete cache files.

    Planet: highest-SNR confirmed KOI alone on its star (no sibling transits) with SNR in the
    top decile of confirmed planets and a 3-15 d period. False positive: ``K06252.01``, else
    the highest-SNR stellar-eclipse false positive whose saved secondary view dips visibly.
    """
    sample = label_catalogue(load_koi_catalogue(config), config)
    planets = sample[sample[LABEL] == 1]
    snr_cut = planets["koi_model_snr"].quantile(0.9)
    planets = planets[
        (planets["koi_model_snr"] >= snr_cut)
        & planets["koi_period"].between(3, 15)
        & (planets["koi_count"] == 1)
    ].sort_values("koi_model_snr", ascending=False)
    fps = sample[(sample[LABEL] == 0) & (sample["koi_fpflag_ss"] == 1)]
    fps = fps.sort_values("koi_model_snr", ascending=False)
    fps = pd.concat([fps[fps[KOI] == BINARY_EXAMPLE], fps[fps[KOI] != BINARY_EXAMPLE]])

    def first(candidates: pd.DataFrame, ok=lambda kepid, koi: True) -> str:
        for koi, kepid in zip(candidates[KOI], candidates["kepid"], strict=True):
            if complete_cache_files(int(kepid), config) is not None and ok(int(kepid), koi):
                return koi
        raise LookupError("no example KOI with a complete light-curve cache")

    def visible_secondary(kepid: int, koi: str) -> bool:
        view = _saved_view(kepid, koi, "secondary_view", config)
        return view is not None and view.min() <= -0.1

    return first(planets), first(fps, visible_secondary)


# --- terminal captures -------------------------------------------------------------------


def _clean(text: str) -> str:
    """Show repository paths relative to the repo root (no local user folder on screen)."""
    root = str(REPO_ROOT)
    return text.replace(root + "\\", "").replace(root + "/", "").replace(root, ".")


def capture_terminal(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    build = BUILD.relative_to(REPO_ROOT).as_posix()
    commands = {
        "pytest": ["-m", "pytest", "-q", "-p", "no:cacheprovider"],
        "audit_leakage": [
            "scripts/audit_leakage.py",
            "--strict",
            "--output",
            f"{build}/leakage_audit_results.md",
            "--columns-output",
            f"{build}/leakage_columns.md",
        ],
        "inspect_catalogue": [
            "scripts/inspect_catalogue.py",
            "--output",
            f"{build}/catalogue_inspection.md",
        ],
        "make_report": ["scripts/make_report.py", "--output", f"{build}/report.md"],
    }
    for name, args in commands.items():
        print(f"  running {name} ...", flush=True)
        result = subprocess.run(
            [sys.executable, *args],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"{name} failed ({result.returncode}):\n{result.stderr}")
        (out_dir / f"{name}.txt").write_text(_clean(result.stdout), encoding="utf-8")

    outputs = REPO_ROOT / OUTPUTS
    train_log = (outputs / "train_cnn_full.log").read_text(encoding="utf-8")
    (out_dir / "train_cnn_full.txt").write_text(_clean(train_log), encoding="utf-8")
    build_log = (outputs / "build_views_full_dev_resume4.log").read_text(encoding="utf-8")
    tail = "\n".join(build_log.splitlines()[-20:]) + "\n"
    (out_dir / "build_views_tail.txt").write_text(_clean(tail), encoding="utf-8")


# --- main --------------------------------------------------------------------------------


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    config = load_config()
    DATA.mkdir(parents=True, exist_ok=True)

    print("numbers.json", flush=True)
    _write_json(DATA / "numbers.json", extract_numbers(config))
    print("audit_columns.json", flush=True)
    _write_json(DATA / "audit_columns.json", audit_columns(config))
    print("error_breakdown.json", flush=True)
    _write_json(DATA / "error_breakdown.json", error_breakdown())
    print("splits_sample.json", flush=True)
    _write_json(DATA / "splits_sample.json", splits_sample(config))

    planet, binary = pick_examples(config)
    for name, koi in (("star_planet", planet), ("star_binary", binary)):
        print(f"{name}.npz ({koi})", flush=True)
        np.savez_compressed(DATA / f"{name}.npz", **extract_star(koi, config))

    print("pr_curves.npz", flush=True)
    np.savez_compressed(DATA / "pr_curves.npz", **pr_curves())

    print("terminal/", flush=True)
    capture_terminal(DATA / "terminal")
    print(f"Done: {DATA}")


if __name__ == "__main__":
    main()
