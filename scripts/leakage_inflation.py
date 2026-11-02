"""Quantify how much leaked catalogue columns inflate performance (before/after the audit).

Trains the same gradient-boosting model, with the same parameters and the same host-star
grouped development folds, on four feature sets:

- ``audited``: the audited tabular features (config ``features.groups``), as in ``tabular``;
- ``audited_plus_vetting``: those plus the vetting outputs the audit removes
  (``koi_score``, the four ``koi_fpflag_*`` flags and ``koi_pdisposition``);
- ``audited_lc`` and ``audited_lc_plus_vetting``: the same pair for the transit + stellar
  features of ``tabular_lc`` (the model comparable with the light-curve CNN);
- ``vetting_only``: the vetting outputs alone;
- ``naive``: every catalogue column except the target and pure identifiers, as a team that
  skipped the audit would use it (text columns become "is present" flags, other categorical
  columns become category codes).

This script deliberately bypasses ``assert_no_leakage``: it exists to show what the audit
prevents. Its models are never saved or used elsewhere. Development folds only; the held-out
test set is not touched.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from exoplanet.config import load_config, set_seed
from exoplanet.data.catalogue import load_koi_catalogue
from exoplanet.data.labels import LABEL, label_catalogue
from exoplanet.data.splits import KOI, SPLIT, TEST, load_splits
from exoplanet.evaluation.pr import compare_pr_auc, operating_point, threshold_at_precision
from exoplanet.features.tabular import select_features
from exoplanet.models.tabular import cross_validate
from exoplanet.reporting import markdown_table

VETTING = [
    "koi_score",
    "koi_fpflag_nt",
    "koi_fpflag_ss",
    "koi_fpflag_co",
    "koi_fpflag_ec",
    "koi_pdisposition",
]
# Never features even for a naive team: the target itself and row identifiers.
TARGET_AND_IDS = ["koi_disposition", "kepid", "kepoi_name", "ra_str", "dec_str", LABEL]


def encode(frame: pd.DataFrame) -> pd.DataFrame:
    """Numeric matrix: numbers kept, high-cardinality text as presence flags, other text as
    category codes (missing stays NaN, which gradient boosting handles natively)."""
    out = {}
    for col in frame.columns:
        values = frame[col]
        if values.isna().all():
            continue  # entirely empty columns carry nothing
        if pd.api.types.is_numeric_dtype(values):
            out[col] = values.astype(float)
        elif values.nunique() > 50:
            out[col] = values.notna().astype(float)
        else:
            codes = values.astype("category").cat.codes.astype(float)
            out[col] = codes.where(codes >= 0)
    return pd.DataFrame(out, index=frame.index)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a YAML config")
    parser.add_argument(
        "--output", default=None, help="report path (default: outputs/leakage_inflation.md)"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["seed"]
    set_seed(seed)
    params = config["tabular"]["params"]
    groups = config["features"]["groups"]

    sample = label_catalogue(load_koi_catalogue(config), config).set_index(KOI)
    splits = load_splits(config)
    audited = select_features(sample, groups)
    audited_lc = select_features(sample, ["transit", "stellar"])
    vetting = encode(sample[VETTING])
    naive = encode(sample.drop(columns=[c for c in TARGET_AND_IDS if c in sample.columns]))
    variants = {
        "audited": audited,
        "audited_plus_vetting": pd.concat([audited, vetting], axis=1),
        "audited_lc": audited_lc,
        "audited_lc_plus_vetting": pd.concat([audited_lc, vetting], axis=1),
        "vetting_only": vetting,
        "naive": naive,
    }
    # Each leaky variant is compared with the audited set it extends.
    baselines = {
        "audited_plus_vetting": "audited",
        "audited_lc_plus_vetting": "audited_lc",
        "vetting_only": "audited",
        "naive": "audited",
    }

    dev = splits[splits[SPLIT] != TEST].set_index(KOI)
    scores = dev[[LABEL, "kepid"]].copy()
    n_features = {}
    for name, features in variants.items():
        oof, _, _ = cross_validate(features, splits, params, seed)
        scores[name] = oof.loc[scores.index, "score"]
        n_features[name] = features.shape[1]
        print(f"{name}: {features.shape[1]} features done")

    names = list(variants)
    per_model, diffs = compare_pr_auc(
        scores, names, n_boot=config["evaluation"]["n_bootstrap"], seed=seed
    )
    y = scores[LABEL].to_numpy()
    rows = []
    for name in names:
        s = scores[name].to_numpy()
        op = operating_point(y, s, threshold_at_precision(y, s, 0.95))
        order = np.argsort(-s)
        tp = np.cumsum(y[order])
        recall = tp / y.sum()
        precision = tp / np.arange(1, len(s) + 1)
        rows.append(
            {
                "features": n_features[name],
                "pr_auc": per_model.loc[name, "pr_auc"],
                "ci_low": per_model.loc[name, "ci_low"],
                "ci_high": per_model.loc[name, "ci_high"],
                "recall_at_p95": op["recall"],
                "precision_at_r90": precision[recall >= 0.90][0],
                "fp_at_r90": int((1 - y[order])[: np.flatnonzero(recall >= 0.90)[0] + 1].sum()),
            }
        )
    table = pd.DataFrame(rows, index=pd.Index(names, name="feature_set"))
    rows = []
    for leaky, base in baselines.items():
        key = f"{leaky} - {base}"
        if key in diffs.index:
            row = diffs.loc[key].copy()
        else:  # compare_pr_auc names each pair in listing order; flip the sign
            row = diffs.loc[f"{base} - {leaky}"].copy()
            row[["difference", "ci_low", "ci_high"]] = -row[
                ["difference", "ci_high", "ci_low"]
            ].to_numpy()
        row.name = key
        rows.append(row)
    inflation = pd.DataFrame(rows).rename_axis("comparison")

    report = [
        "# Leakage Inflation: Before and After the Audit",
        "",
        "Generated by `scripts/leakage_inflation.py`. Same gradient-boosting model and parameters "
        f"on the same {dev.shape[0]} development KOIs and host-star grouped folds; only the "
        "feature set changes. Out-of-fold scores; PR-AUC 95% intervals from "
        f"{config['evaluation']['n_bootstrap']} host-star bootstrap resamples. The held-out "
        "test set is not used.",
        "",
        f"- `audited`: the audited catalogue features used by `tabular` (groups `{groups}`)",
        "- `audited_plus_vetting`: plus the vetting outputs `" + "`, `".join(VETTING) + "`",
        "- `audited_lc`, `audited_lc_plus_vetting`: the same pair for the transit + stellar "
        "features of `tabular_lc`, the model comparable with the light-curve CNN",
        "- `vetting_only`: the vetting outputs alone",
        "- `naive`: every catalogue column except the target and row identifiers (no audit)",
        "",
        markdown_table(table, floatfmt=".3f"),
        "",
        "`fp_at_r90`: false positives accepted when the threshold is lowered until 90% of "
        "confirmed planets are found (thresholds set on these same scores, for comparison only).",
        "",
        "## Inflation relative to the audited model",
        "",
        "Paired differences on the same resampled stars.",
        "",
        markdown_table(inflation, floatfmt=".3f"),
        "",
    ]
    output = (
        Path(args.output)
        if args.output
        else config["paths"]["outputs_dir"] / "leakage_inflation.md"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(report), encoding="utf-8")
    print(table.to_string(float_format="%.3f"))
    print(inflation.to_string(float_format="%.3f"))
    print(f"Report: {output}")


if __name__ == "__main__":
    main()
