# ruff: noqa: E501  (notebook cell sources are kept readable as written)
"""Build and execute ``notebooks/walkthrough.ipynb`` (the report's notebook walkthrough).

The notebook reads cached artefacts only (catalogue, splits, views, out-of-fold and saved
test scores); it downloads nothing and retrains nothing. Requires ``pip install -e
".[notebook]"``. Run after the pipeline steps in the README.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]

CELLS: list[tuple[str, str]] = [
    (
        "md",
        """# Exoplanet Transit Detection (Kepler): Walkthrough

This notebook walks through the project end to end on **cached data**: it downloads nothing
and retrains nothing, so it runs in about a minute. Each section corresponds to an objective
in the project brief.

1. Data retrieval and provenance
2. Labels and modelling sample
3. Leakage audit: every column, and how much the leaked columns inflate results
4. Splits grouped by host star, with the disjointness assertion
5. Light-curve views and the period check
6. Models
7. Precision-recall evaluation, success criteria and the operating point
8. False-positive analysis
9. Class weighting

**What a score means.** The models reproduce a *vetting judgement*: a high score says the
signal resembles KOIs that were confirmed, and is worth follow-up. It is not a planet
detection; confirmation requires follow-up observation.""",
    ),
    (
        "code",
        """import json, os, sys
from pathlib import Path

ROOT = Path.cwd().resolve()
while not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent
os.chdir(ROOT)

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
from sklearn.metrics import average_precision_score, precision_recall_curve

from exoplanet.config import load_config
from exoplanet.data.catalogue import load_koi_catalogue
from exoplanet.data.labels import LABEL, PARAMETER_PROVENANCE, label_catalogue
from exoplanet.data.splits import FOLD, KOI, SPLIT, TEST, load_splits, validate_splits
from exoplanet.features.columns import column_table
from exoplanet.features.leakage import univariate_audit
from exoplanet.features.tabular import select_features

pd.set_option("display.max_colwidth", 120)
config = load_config(None)
print("Python", sys.version.split()[0], "| repository", ROOT.name)""",
    ),
    (
        "md",
        """## 1. Data retrieval and provenance

The KOI cumulative table was fetched once from the NASA Exoplanet Archive TAP service and
cached; the query, time and checksum were recorded next to it. Full details:
[docs/data_card.md](../docs/data_card.md).""",
    ),
    (
        "code",
        """meta = json.loads((config["paths"]["raw_dir"] / "koi_cumulative.meta.json").read_text())
display(pd.Series(meta, name="value").to_frame())
catalogue = load_koi_catalogue(config)
print(f"{catalogue.shape[0]:,} KOIs x {catalogue.shape[1]} columns")
catalogue["koi_disposition"].value_counts().to_frame("KOIs")""",
    ),
    (
        "md",
        """## 2. Labels and modelling sample

CONFIRMED = 1, FALSE POSITIVE = 0; CANDIDATE is excluded because it has no settled answer.
KOIs with parameters from older pipeline deliveries are dropped: they are almost all false
positives, so keeping them would teach the model delivery history instead of astrophysics.""",
    ),
    (
        "code",
        """all_labelled = label_catalogue(catalogue, {**config, "sample": {}})
sample = label_catalogue(catalogue, config)
prov = pd.crosstab(all_labelled[PARAMETER_PROVENANCE], all_labelled[LABEL])
prov["planet_fraction"] = (prov[1] / prov.sum(axis=1)).round(3)
display(prov)
print(f"Modelling sample: {len(sample):,} KOIs on {sample['kepid'].nunique():,} stars, "
      f"{sample[LABEL].mean():.1%} confirmed")""",
    ),
    (
        "md",
        """## 3. Leakage audit

### 3a. Every catalogue column

Each column has a meaning, a keep-or-drop decision and a reason. `auc_value` shows how well
the column alone separates the classes (0.5 = no signal); the dropped vetting columns are
near-perfect on their own, which is exactly why they must go. The same table is in
[docs/leakage_columns.md](../docs/leakage_columns.md).""",
    ),
    (
        "code",
        """numeric = [c for c in sample.select_dtypes("number").columns if c != LABEL]
audit = univariate_audit(sample[numeric], sample[LABEL])
columns = column_table(catalogue.columns, config["features"]["groups"], audit)
display(columns["decision"].value_counts().to_frame("columns"))
with pd.option_context("display.max_rows", 200):
    display(columns.round(3))""",
    ),
    (
        "md",
        """### 3b. Before and after: how much do the leaked columns inflate the result?

Same gradient-boosting model, same development folds, only the feature set changes
(`scripts/leakage_inflation.py`).""",
    ),
    (
        "code",
        """text = (config["paths"]["outputs_dir"] / "leakage_inflation.md").read_text(encoding="utf-8")
display(Markdown(text.split("\\n", 2)[2]))""",
    ),
    (
        "md",
        """## 4. Splits grouped by host star

One star can host several KOIs, and multi-KOI stars are mostly planets. Splitting by row would
put the same star on both sides. The splits are grouped by `kepid`; the cell below **asserts**
that no star contributes to more than one partition.""",
    ),
    (
        "code",
        """splits = load_splits(config)
test_stars = set(splits.loc[splits[SPLIT] == TEST, "kepid"])
dev_stars = set(splits.loc[splits[SPLIT] != TEST, "kepid"])
assert test_stars.isdisjoint(dev_stars), "a host star appears in both test and development"
folds_per_star = splits[splits[SPLIT] != TEST].groupby("kepid")[FOLD].nunique()
assert (folds_per_star == 1).all(), "a host star appears in more than one development fold"
validate_splits(splits, config["splits"]["n_folds"])  # the library check, which raises on any overlap
print(f"OK: {len(test_stars):,} test stars and {len(dev_stars):,} development stars are disjoint; "
      f"every development star sits in exactly one fold.")
splits.groupby([SPLIT, FOLD]).agg(kois=(KOI, "size"), stars=("kepid", "nunique"),
                                  planet_fraction=(LABEL, "mean")).round(3)""",
    ),
    (
        "md",
        """## 5. Light-curve views and the period check

Each star's long-cadence PDCSAP light curve is detrended with a moving median (every known
transit masked), folded on the catalogue period and binned into a **global view** (whole
orbit, 2,001 bins) and a **local view** (±2 transit durations, 201 bins). The secondary view
(phase 0.5, on the local view's scale) shows eclipsing binaries.""",
    ),
    (
        "code",
        """from exoplanet.data.lightcurves import load_views
from exoplanet.features.lightcurve import view_transit_snr

views = load_views(config, splits[KOI])
print(f"Views for {len(views):,} of {len(splits):,} KOIs ({len(views) / len(splits):.1%})")
examples = {"K00219.01": "confirmed planet", "K07132.01": "eclipsing binary (false positive)",
            "K01378.01": "centroid-offset false positive"}
fig, axes = plt.subplots(len(examples), 3, figsize=(13, 7.5))
for row, (koi, what) in zip(axes, examples.items()):
    v = views.loc[koi]
    row[0].plot(np.linspace(-0.5, 0.5, len(v["global_view"])), v["global_view"], lw=0.6, color="0.2")
    row[1].plot(np.linspace(-2, 2, len(v["local_view"])), v["local_view"], ".", ms=3, color="#2a78d6")
    row[2].plot(np.linspace(-2, 2, len(v["secondary_view"])), v["secondary_view"], ".", ms=3, color="#eb6834")
    row[2].set_ylim(row[1].get_ylim())
    row[0].set_ylabel(f"{koi}\\n{what}", fontsize=8)
for ax, title in zip(axes[0], ["global view (phase)", "local view (durations)", "secondary view, same scale"]):
    ax.set_title(title, fontsize=9)
fig.tight_layout()""",
    ),
    (
        "md",
        """**Period check.** Folding on a wrong period smears the transits, so the centre of the local
view loses its dip. `view_transit_snr` measures that dip against the out-of-transit scatter;
views below 3 are flagged.""",
    ),
    (
        "code",
        """snr = views["local_view"].map(lambda v: view_transit_snr(np.asarray(v), config["views"]["local_durations"]))
check = pd.DataFrame({"view_snr": snr, LABEL: splits.set_index(KOI).loc[views.index, LABEL]})
check["flagged"] = check["view_snr"] < 3
display(check.groupby(LABEL).agg(kois=("flagged", "size"), flagged=("flagged", "sum"),
                                 median_view_snr=("view_snr", "median")).round(1))
flagged_fp = sample.set_index(KOI).loc[check.index[check.flagged & (check[LABEL] == 0)]]
print(f"{len(flagged_fp)} flagged false positives, {int(flagged_fp['koi_fpflag_nt'].sum())} with the "
      "not-transit-like flag: signals that never were coherent transits.")""",
    ),
    (
        "md",
        """## 6. Models

| Model | Inputs | Notes |
|---|---|---|
| `tabular` | 41 audited catalogue features: transit, stellar, centroid | Histogram gradient boosting |
| `tabular_lc` | 30 transit + stellar features | No pixel (centroid) data, like the CNN |
| `tabular_lc_sub` | as `tabular_lc` | Trained only on the CNN's original 21% of dev KOIs |
| `cnn` | global + local views | Two-tower 1D CNN, trained on 21% of dev KOIs (test-scored) |
| `cnn_full` | global + local views | Same CNN, trained on all dev KOIs (dev only) |

All share the same host-star folds. Out-of-fold PR-AUC on the development folds:""",
    ),
    (
        "code",
        """rows = {}
for name in ["tabular", "tabular_lc", "tabular_lc_sub", "cnn", "cnn_full"]:
    oof = pd.read_csv(config["paths"]["outputs_dir"] / f"{name}_oof.csv")
    rows[name] = {"dev_kois": len(oof), "pooled_pr_auc": average_precision_score(oof[LABEL], oof["score"])}
pd.DataFrame(rows).T.round(3)""",
    ),
    (
        "md",
        """## 7. Evaluation

### 7a. Precision-recall curves with bootstrap confidence bands

Bands are 95% intervals from 1,000 resamples of host stars. The held-out test set was scored
once, after all modelling decisions.""",
    ),
    (
        "code",
        """display(Image(filename="docs/figures/pr_bands_test.png", width=900))
display(Image(filename="docs/figures/pr_bands_dev_full.png", width=900))""",
    ),
    (
        "md",
        """### 7b. Success criteria: PR-AUC >= 0.80 and recall >= 0.90 at a stated precision

The recall-0.90 threshold is chosen on development scores and applied unchanged to the saved
test scores, so test recall lands near 0.90 rather than exactly on it.""",
    ),
    (
        "code",
        """sys.path.insert(0, str(ROOT / "scripts"))
from operating_points import threshold_for  # same threshold rule as scripts/operating_points.py

out = config["paths"]["outputs_dir"]
oof = {m: pd.read_csv(out / f"{m}_oof.csv", index_col=KOI)
       for m in ["tabular", "tabular_lc", "tabular_lc_sub", "cnn", "cnn_full"]}
# Thresholds for the full-data models are set on the dev KOIs all three score (as in
# outputs/operating_points.md); the 21%-subset models use their own dev KOIs.
common = sorted(set(oof["tabular"].index) & set(oof["tabular_lc"].index) & set(oof["cnn_full"].index))
test = pd.read_csv(out / "evaluation_test_scores.csv")
rows = {}
for name in ["tabular", "tabular_lc", "tabular_lc_sub", "cnn"]:
    d = oof[name].loc[common] if name in ("tabular", "tabular_lc") else oof[name]
    thr = threshold_for(d[LABEL].to_numpy(), d["score"].to_numpy(), "recall", 0.90)
    y, s = test[LABEL].to_numpy(), test[name].to_numpy()
    pred = s >= thr
    rows[name] = {"test_pr_auc": average_precision_score(y, s),
                  "test_precision": (pred & (y == 1)).sum() / pred.sum(),
                  "test_recall": (pred & (y == 1)).sum() / y.sum()}
criteria = pd.DataFrame(rows).T
criteria["pr_auc_ok"] = criteria["test_pr_auc"] >= 0.80
criteria.round(3)""",
    ),
    (
        "md",
        """### 7c. Operating point in follow-up terms

Every false positive sent on wastes follow-up telescope time. The chosen operating point is
**precision 0.95** (about one wasted follow-up per 19 planets); the table shows what the
recall-0.90 alternative would cost.""",
    ),
    (
        "code",
        """text = (config["paths"]["outputs_dir"] / "operating_points.md").read_text(encoding="utf-8")
display(Markdown(text.split("## Development folds", 1)[1].split("## Held-out", 1)[0]))""",
    ),
    (
        "md",
        """## 8. False-positive analysis

Which false positives get through? Share of false positives *rejected* at each model's
precision-0.95 threshold, by the Robovetter's flag (flags are never model inputs), on the
5,331 development KOIs both models score.""",
    ),
    (
        "code",
        """dev = pd.read_csv(config["paths"]["outputs_dir"] / "evaluation_dev_full_scores.csv")
thr = {"tabular_lc": 0.840, "cnn_full": 0.831}
fps = dev[dev[LABEL] == 0]
flags = {"koi_fpflag_nt": "not transit-like", "koi_fpflag_ss": "stellar eclipse",
         "koi_fpflag_co": "centroid offset", "koi_fpflag_ec": "ephemeris match"}
table = {"all false positives": {m: (fps[m] < t).mean() for m, t in thr.items()} | {"n": len(fps)}}
for col, name in flags.items():
    sub = fps[fps[col] == 1]
    table[name] = {m: (sub[m] < t).mean() for m, t in thr.items()} | {"n": len(sub)}
display(pd.DataFrame(table).T.round(3))
accepted = fps[fps["cnn_full"] >= thr["cnn_full"]]
print(f"cnn_full lets {len(accepted)} false positives through; "
      f"{int(accepted['koi_fpflag_co'].sum())} carry the centroid-offset flag, "
      f"{int(accepted['koi_fpflag_ss'].sum())} the stellar-eclipse flag.")
accepted.sort_values("cnn_full", ascending=False).head(8)[
    [KOI, "cnn_full", "koi_period", "koi_prad", "koi_model_snr", *flags]]""",
    ),
    (
        "md",
        """**Dominant confusing population: centroid-offset false positives.** Their light curves look
like planets because the dip is real; it just comes from a nearby star blended into the same
pixels. Photometry alone cannot reveal that, which is why the full tabular model, with
centroid diagnostics, does best. Eclipsing binaries, by contrast, are rejected well: their
V shapes and secondary eclipses are visible in the views. Missed planets are mostly faint
(low SNR) and short-period.""",
    ),
    (
        "md",
        """## 9. Class weighting

False positives outnumber planets 59:41. Both model types were also trained with identical
balanced class weights (`--class-weight balanced`) and compared on the development folds.""",
    ),
    (
        "code",
        """rows = {}
for base in ["tabular", "tabular_lc", "cnn_full"]:
    r = {}
    for label, name in [("unweighted", base), ("balanced", f"{base}_w")]:
        oof = pd.read_csv(config["paths"]["outputs_dir"] / f"{name}_oof.csv")
        r[label] = average_precision_score(oof[LABEL], oof["score"])
    rows[base] = r
weights = pd.DataFrame(rows).T
weights["difference"] = weights["balanced"] - weights["unweighted"]
weights.round(3)""",
    ),
    (
        "md",
        """Weighting does not help (and slightly hurts the CNN), because the imbalance is mild and the
operating threshold is chosen separately on development scores. The reported models are
unweighted, identically for both model types.

## Conclusions

- After the leakage audit, every model clears PR-AUC 0.80; both tabular models reach recall
  0.90 at precision above 0.93 on the test set.
- The leaked vetting columns alone score 0.999: a near-perfect result on this dataset means
  the audit was skipped.
- Centroid diagnostics are the most valuable information; a light-curve-only CNN cannot see
  them, and centroid-offset false positives are what fools it.
- Training data mattered more for the CNN than any input change tried: on all development
  KOIs it draws level with the matched tabular model on the earlier comparison.
- Scores are vetting judgements, not detections; follow-up observation is still required.""",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "notebooks" / "walkthrough.ipynb"))
    parser.add_argument("--no-execute", action="store_true", help="write without running")
    args = parser.parse_args()
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "name": "python3",
        "display_name": "Python 3",
        "language": "python",
    }
    nb.cells = [
        nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src)
        for kind, src in CELLS
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_execute:
        NotebookClient(
            nb,
            timeout=900,
            kernel_name="python3",
            resources={"metadata": {"path": str(output.parent)}},
        ).execute()
    nbformat.write(nb, output)
    print(f"Notebook: {output}")


if __name__ == "__main__":
    main()
