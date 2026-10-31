"""Precision-recall comparison and false-positive analysis across models.

``--split dev`` evaluates out-of-fold scores on the development folds and can be run at any
time. ``--split test`` scores the held-out test set with the final models; it is meant to
be run once, after all modelling decisions are made, and refuses to overwrite earlier test
results without --force. In both modes, operating thresholds come from the development
out-of-fold scores, and models are compared on the KOIs that all of them can score.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from typing import Any

import joblib
import pandas as pd
import torch

from exoplanet.config import load_config, set_seed
from exoplanet.data.catalogue import load_koi_catalogue
from exoplanet.data.labels import LABEL, label_catalogue
from exoplanet.data.lightcurves import load_views
from exoplanet.data.splits import KOI, held_out_kois, load_splits
from exoplanet.evaluation.false_positives import (
    FP_FLAGS,
    fp_rejection_by_flag,
    recall_by_bin,
    worst_errors,
)
from exoplanet.evaluation.metrics import SCORE
from exoplanet.evaluation.plots import plot_pr_curves
from exoplanet.evaluation.pr import (
    compare_pr_auc,
    max_precision_threshold,
    operating_point,
    threshold_at_precision,
)
from exoplanet.features.tabular import select_features
from exoplanet.models.cnn import TwoTowerCNN, attach_scalars, predict
from exoplanet.reporting import markdown_table

CONTEXT = ["kepid", LABEL, *FP_FLAGS, "koi_period", "koi_depth", "koi_prad", "koi_model_snr"]


def score_test(
    name: str, sample: pd.DataFrame, kois: pd.Index, config: dict[str, Any]
) -> pd.Series:
    """Final-model scores for the given test KOIs (those the model has inputs for)."""
    models_dir = config["paths"]["models_dir"]
    if (models_dir / f"{name}.joblib").exists():
        bundle = joblib.load(models_dir / f"{name}.joblib")
        prov = bundle["provenance"]
        features = select_features(sample.loc[kois], prov["feature_groups"])[prov["features"]]
        return pd.Series(bundle["model"].predict_proba(features)[:, 1], index=kois)
    if (models_dir / f"{name}.pt").exists():
        ckpt = torch.load(models_dir / f"{name}.pt", weights_only=True)
        views_cfg = ckpt["views"]
        scalars = ckpt["params"].get("scalars") or []
        model = TwoTowerCNN(
            views_cfg["global_bins"],
            views_cfg["local_bins"],
            ckpt["params"]["arch"],
            n_scalars=len(scalars),
        )
        # Checkpoints from before scalar inputs lack the (empty) scalar buffers.
        state = {**dict(model.state_dict()), **ckpt["state_dict"]}
        model.load_state_dict(state)
        views = load_views(config, kois.to_series())
        if scalars:
            views = attach_scalars(views, sample, scalars)
        return pd.Series(predict(model, views), index=views.index)
    raise FileNotFoundError(f"No final model for {name!r} in {models_dir}")


def common_index(series: dict[str, pd.Series]) -> pd.Index:
    return pd.Index(sorted(set.intersection(*(set(s.index) for s in series.values()))), name=KOI)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--models", nargs="+", default=None, help="default: evaluation.models")
    parser.add_argument("--force", action="store_true", help="overwrite existing test results")
    parser.add_argument("--tag", default=None, help="suffix for output files, for side experiments")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["seed"]
    set_seed(seed)
    ec = config["evaluation"]
    models = args.models or ec["models"]
    outputs = config["paths"]["outputs_dir"]
    stem = f"evaluation_{args.split}" + (f"_{args.tag}" if args.tag else "")
    scores_path = outputs / f"{stem}_scores.csv"
    if args.split == "test" and scores_path.exists() and not args.force:
        sys.exit(f"{scores_path} exists: the test set has already been scored (--force to redo)")

    sample = label_catalogue(load_koi_catalogue(config), config).set_index(KOI)
    splits = load_splits(config)

    # Development out-of-fold scores: thresholds always come from here.
    oof = {}
    for m in models:
        path = outputs / f"{m}_oof.csv"
        if not path.exists():
            sys.exit(f"{path} not found: train {m!r} first")
        oof[m] = pd.read_csv(path, index_col=KOI)[SCORE]
    dev_common = common_index(oof)
    dev_labels = sample.loc[dev_common, LABEL].to_numpy()
    thresholds, fallback = {}, []
    for m in models:
        dev_scores = oof[m].loc[dev_common].to_numpy()
        try:
            thresholds[m] = threshold_at_precision(dev_labels, dev_scores, ec["target_precision"])
        except ValueError:
            thresholds[m] = max_precision_threshold(dev_labels, dev_scores)
            fallback.append(m)
            print(
                f"warning: {m} never reaches precision {ec['target_precision']} on dev; "
                "using its highest-precision threshold"
            )

    if args.split == "dev":
        raw, n_total = oof, int((splits["split"] != "test").sum())
    else:
        test = pd.Index(held_out_kois(splits), name=KOI)
        raw, n_total = {m: score_test(m, sample, test, config) for m in models}, len(test)
    common = common_index(raw)
    scores = sample.loc[common, CONTEXT].assign(**{m: raw[m].loc[common] for m in models})

    per_model, diffs = compare_pr_auc(scores, models, n_boot=ec["n_bootstrap"], seed=seed)
    y = scores[LABEL].to_numpy()
    ops = {m: operating_point(y, scores[m].to_numpy(), thresholds[m]) for m in models}
    ops_table = pd.DataFrame(ops).T.rename_axis("model")
    for col in ["tp", "fp", "fn", "tn"]:
        ops_table[col] = ops_table[col].astype(int)

    split_name = "development out-of-fold" if args.split == "dev" else "held-out test"
    figure = plot_pr_curves(
        scores,
        models,
        ops,
        f"Precision-recall, {split_name} KOIs (n = {len(scores)})",
        outputs / f"{stem}_pr.png",
    )
    scores.to_csv(scores_path)

    report = [
        f"# Evaluation: {split_name} set",
        "",
        "Generated by `scripts/evaluate.py`. Label 1 = CONFIRMED. All models are scored on the "
        f"same {len(scores)} KOIs ({len(scores) / n_total:.1%} of the {n_total} {args.split} "
        "KOIs; the rest lack an input for at least one model).",
        "",
        f"![PR curves]({figure.name})",
        "",
        "## PR-AUC",
        "",
        f"95% intervals from {ec['n_bootstrap']} bootstrap resamples of host stars; differences "
        "are paired (the same resampled stars for both models).",
        "",
        markdown_table(per_model, floatfmt=".3f"),
        "",
        markdown_table(diffs, floatfmt=".3f"),
        "",
        f"## Operating point (precision >= {ec['target_precision']} on the development folds)",
        "",
        "Thresholds are chosen on development out-of-fold scores"
        + (" and applied unchanged to the test set." if args.split == "test" else ".")
        + (
            f" {fallback} never reach the target precision on dev, so their threshold is the "
            "one with the highest dev precision."
            if fallback
            else ""
        ),
        "",
        markdown_table(ops_table, floatfmt=".3f"),
        "",
        "## False positives rejected, by Robovetter flag",
        "",
        "Share of false positives scored below the threshold. Flags are vetting outputs, never "
        "model inputs; a KOI can carry several.",
        "",
        markdown_table(fp_rejection_by_flag(scores, thresholds), floatfmt=".3f"),
        "",
        "## Confirmed planets recovered, by signal-to-noise and period",
        "",
        markdown_table(recall_by_bin(scores, thresholds, "koi_model_snr"), floatfmt=".3f"),
        "",
        markdown_table(recall_by_bin(scores, thresholds, "koi_period"), floatfmt=".3f"),
        "",
        "## Most confident mistakes",
        "",
    ]
    context = ["koi_period", "koi_depth", "koi_prad", "koi_model_snr", *FP_FLAGS]
    for m in models:
        fp, missed = worst_errors(scores, m, 5, context)
        report += [
            f"### {m}",
            "",
            "Highest-scored false positives:",
            "",
            markdown_table(fp, floatfmt=".3g"),
            "",
            "Lowest-scored confirmed planets:",
            "",
            markdown_table(missed.drop(columns=list(FP_FLAGS)), floatfmt=".3g"),
            "",
        ]
    report_path = outputs / f"{stem}.md"
    report_path.write_text("\n".join(report), encoding="utf-8")

    summary = {
        "split": args.split,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "models": models,
        "n_kois": len(scores),
        "n_total": n_total,
        "positive_fraction": float(y.mean()),
        "target_precision": ec["target_precision"],
        "n_bootstrap": ec["n_bootstrap"],
        "pr_auc": per_model.to_dict(orient="index"),
        "differences": diffs.to_dict(orient="index"),
        "operating_points": ops,
        "threshold_fallback": fallback,
        "fp_rejection": fp_rejection_by_flag(scores, thresholds).to_dict(orient="index"),
        "figure": figure.name,
    }
    (outputs / f"{stem}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(per_model.to_string(float_format="%.3f"))
    print(diffs.to_string(float_format="%.3f"))
    print(f"Report: {report_path}\nFigure: {figure}\nScores: {scores_path}")


if __name__ == "__main__":
    main()
