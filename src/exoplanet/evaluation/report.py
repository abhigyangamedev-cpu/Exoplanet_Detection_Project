"""Assemble the results report from generated artefacts.

Every number in the report is read from pipeline outputs (splits, views, out-of-fold
scores, evaluation summaries); nothing is typed in by hand. Sections whose inputs do not
exist yet are marked pending with the command that produces them, so the report can be
regenerated at any stage and is complete once the pipeline has run end to end.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from exoplanet.config import REPO_ROOT
from exoplanet.data.labels import LABEL
from exoplanet.data.splits import FOLD, KOI, SPLIT, TEST
from exoplanet.evaluation.metrics import SCORE, fold_metrics, pooled_pr_auc
from exoplanet.reporting import markdown_table

TRAIN_COMMANDS = {
    "tabular": "python scripts/train_tabular.py",
    "tabular_lc": "python scripts/train_tabular.py --groups transit stellar --name tabular_lc",
    "cnn": "python scripts/train_cnn.py",
}
MODEL_DESCRIPTIONS = {
    "tabular": "gradient boosting, transit + stellar + centroid catalogue features",
    "tabular_lc": "gradient boosting, transit + stellar features (no centroid data, like the CNN)",
    "cnn": "two-tower 1D CNN on global and local light-curve views",
}


def pending(what: str, command: str) -> list[str]:
    return [f"_Pending: {what}. Run `{command}`._", ""]


def git_revision() -> str:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{rev} (uncommitted changes)" if dirty else rev


def provenance_section(config: dict[str, Any], config_path: Path) -> list[str]:
    raw = config["paths"]["raw_dir"] / "koi_cumulative.meta.json"
    catalogue = json.loads(raw.read_text(encoding="utf-8")) if raw.exists() else {}
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()[:12]
    return [
        "## Provenance",
        "",
        f"- Generated: {datetime.now(UTC).isoformat(timespec='seconds')} from commit "
        f"`{git_revision()}`; config `{config_path.name}` (SHA-256 `{config_hash}…`)",
        f"- KOI catalogue: NASA Exoplanet Archive cumulative table, retrieved "
        f"{catalogue.get('retrieved_utc', 'unknown')}, {catalogue.get('n_rows', '?')} rows, "
        f"SHA-256 `{str(catalogue.get('sha256', 'unknown'))[:12]}…`",
        "",
    ]


def data_section(splits: pd.DataFrame) -> list[str]:
    table = (
        splits.assign(partition=splits[SPLIT].map({TEST: "test", "dev": "development"}))
        .groupby("partition")
        .agg(kois=(KOI, "size"), stars=("kepid", "nunique"), positive_fraction=(LABEL, "mean"))
        .loc[["development", "test"]]
    )
    return [
        "## Data",
        "",
        f"Modelling sample: {len(splits)} labelled KOIs (CONFIRMED = 1, FALSE POSITIVE = 0) on "
        f"{splits['kepid'].nunique()} host stars, DR25 parameters only. The test set and the "
        f"{splits.loc[splits[FOLD] >= 0, FOLD].nunique()} development folds are grouped by host "
        "star and stratified by label.",
        "",
        markdown_table(table, floatfmt=".3f"),
        "",
    ]


def coverage_section(
    splits: pd.DataFrame, views_kois: pd.Index, failures: pd.DataFrame | None
) -> list[str]:
    lines = ["## Light-curve coverage", ""]
    if not len(views_kois):
        return lines + pending("no light-curve views built", "python scripts/build_views.py")
    has = splits[KOI].isin(views_kois)
    table = (
        splits.assign(
            partition=splits[SPLIT].map({TEST: "test", "dev": "development"}), with_views=has
        )
        .groupby(["partition", LABEL])
        .agg(kois=(KOI, "size"), with_views=("with_views", "mean"))
    )
    table.index = [f"{p}, label {lbl}" for p, lbl in table.index]
    lines += [
        f"{int(has.sum())} of {len(splits)} KOIs ({has.mean():.1%}) have light-curve views.",
        "",
        markdown_table(table.rename_axis("partition"), floatfmt=".3f"),
        "",
    ]
    if failures is not None and len(failures):
        lines += [f"{len(failures)} host stars failed in the last build run.", ""]
    return lines


def cv_section(outputs: Path, models: list[str]) -> list[str]:
    lines = ["## Cross-validation (development folds)", ""]
    rows = {}
    for m in models:
        path = outputs / f"{m}_oof.csv"
        if not path.exists():
            lines += pending(f"`{m}` not trained", TRAIN_COMMANDS.get(m, f"train {m}"))
            continue
        oof = pd.read_csv(path, index_col=KOI)
        folds = pd.DataFrame(
            [
                fold_metrics(f, g[LABEL].to_numpy(), g[SCORE].to_numpy())
                for f, g in oof.groupby(FOLD)
            ]
        )
        rows[m] = {
            "kois": len(oof),
            "pooled_pr_auc": pooled_pr_auc(oof),
            "fold_pr_auc_mean": folds["pr_auc"].mean(),
            "fold_pr_auc_std": folds["pr_auc"].std(),
            "fold_roc_auc_mean": folds["roc_auc"].mean(),
        }
    if rows:
        table = pd.DataFrame(rows).T.rename_axis("model")
        table["kois"] = table["kois"].astype(int)
        lines += [
            "Out-of-fold scores; each model is scored on the development KOIs it has inputs "
            "for (see the evaluation sections for a same-KOI comparison).",
            "",
            markdown_table(table, floatfmt=".3f"),
            "",
        ]
    return lines


def evaluation_section(
    outputs: Path, split: str, models: list[str], figures_dir: Path, figures_link: str
) -> list[str]:
    title = "development folds" if split == "dev" else "held-out test set"
    lines = [f"## Evaluation: {title}", ""]
    path = outputs / f"evaluation_{split}.json"
    command = f"python scripts/evaluate.py --split {split}"
    if not path.exists():
        return lines + pending(f"{title} not evaluated", command)
    s = json.loads(path.read_text(encoding="utf-8"))
    missing = [m for m in models if m not in s["models"]]
    if missing:
        lines += [
            f"_Incomplete: evaluated {s['models']}, not {missing}. Re-run `{command}` once "
            "they are trained._",
            "",
        ]

    figure = outputs / s["figure"]
    if figure.exists():
        figures_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(figure, figures_dir / figure.name)
        lines += [f"![Precision-recall curves]({figures_link}/{figure.name})", ""]

    pr = pd.DataFrame(s["pr_auc"]).T.rename_axis("model")
    ops = pd.DataFrame(s["operating_points"]).T[["threshold", "precision", "recall", "fp", "fn"]]
    for col in ["fp", "fn"]:
        ops[col] = ops[col].astype(int)
    pr = pr.join(ops.add_prefix("at_threshold_").drop(columns="at_threshold_threshold"))
    lines += [
        f"{s['n_kois']} KOIs scored by every model ({s['n_kois'] / s['n_total']:.1%} of "
        f"{s['n_total']}); positive fraction {s['positive_fraction']:.3f}. PR-AUC with 95% "
        f"host-star bootstrap intervals ({s['n_bootstrap']} resamples); operating threshold "
        f"set for precision {s['target_precision']} on development out-of-fold scores.",
        "",
        markdown_table(pr, floatfmt=".3f"),
        "",
    ]
    if s["differences"]:
        diffs = pd.DataFrame(s["differences"]).T.rename_axis("comparison")
        lines += ["Paired PR-AUC differences:", "", markdown_table(diffs, floatfmt=".3f"), ""]
    fp = pd.DataFrame(s["fp_rejection"]).T.rename_axis("flag")
    fp["n"] = fp["n"].astype(int)
    lines += [
        "False positives rejected at the operating threshold, by Robovetter flag:",
        "",
        markdown_table(fp, floatfmt=".3f"),
        "",
        f"Full report: `outputs/evaluation_{split}.md`.",
        "",
    ]
    return lines


def build_report(
    config: dict[str, Any],
    config_path: Path,
    splits: pd.DataFrame,
    views_kois: pd.Index,
    failures: pd.DataFrame | None,
    figures_dir: Path,
    figures_link: str = "figures",
) -> str:
    outputs = config["paths"]["outputs_dir"]
    models = config["evaluation"]["models"]
    model_list = [f"- `{m}`: {MODEL_DESCRIPTIONS.get(m, m)}" for m in models]
    lines = [
        "# Results",
        "",
        "Generated by `scripts/make_report.py` from pipeline outputs; do not edit by hand. "
        "Methodology: [evaluation.md](evaluation.md), "
        "[lightcurve_pipeline.md](lightcurve_pipeline.md), "
        "[tabular_baseline.md](tabular_baseline.md).",
        "",
        "Models:",
        "",
        *model_list,
        "",
    ]
    lines += data_section(splits)
    lines += coverage_section(splits, views_kois, failures)
    lines += cv_section(outputs, models)
    for split in ("dev", "test"):
        lines += evaluation_section(outputs, split, models, figures_dir, figures_link)
    lines += provenance_section(config, config_path)
    return "\n".join(lines)
