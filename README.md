# Kepler Exoplanet Transit Detection

Classifying NASA Kepler Objects of Interest (KOIs) as confirmed planets or false positives,
using a 1D CNN on phase-folded light curves and a tabular baseline on catalogue features.

## Results

Held-out test set (1,334 KOIs), PR-AUC with 95% host-star bootstrap intervals:

| model | inputs | test PR-AUC |
|---|---|---|
| `tabular` | transit + stellar + centroid catalogue features | 0.993 (0.984-0.998) |
| `tabular_lc` | transit + stellar features | 0.966 (0.947-0.982) |
| `tabular_lc_sub` | as above, trained on the CNN's training KOIs only | 0.953 (0.935-0.969) |
| `cnn` | global and local light-curve views | 0.903 (0.870-0.934) |

- Catalogue centroid diagnostics are the strongest single source of information.
- The light-curve CNN trails a tabular model trained on the same KOIs by 0.051 PR-AUC (95%
  CI 0.023-0.079). It misses more low-SNR and short-period planets, and half the false
  positives it accepts are centroid-offset cases that photometry cannot reveal. Giving it
  transit depth as an input helped only marginally (+0.009 dev PR-AUC).
- The CNN was trained on 21% of the development KOIs, because light curves were built for all
  test stars but only 1,000 development stars. Retrained later on all development KOIs
  (`cnn_full`, development folds only), it reaches dev PR-AUC 0.946 against 0.969 for
  `tabular_lc`, and draws level with it on the earlier 1,123-KOI comparison.
- Leakage check: adding the six vetting columns the audit removes lifts the transit + stellar
  model from 0.969 to 0.999 dev PR-AUC; those six columns alone score 0.999
  ([before and after](docs/leakage_audit.md#before-and-after)).
- Success criteria (PR-AUC >= 0.80, recall >= 0.90 at a stated precision) are met by both
  tabular models on the test set and by `cnn_full` on dev
  ([details](docs/evaluation.md#success-criteria)).

**What the scores mean.** The models reproduce a vetting judgement: a high score says a
signal resembles KOIs that were confirmed, and that it is worth follow-up. It is not a planet
detection; confirmation requires follow-up observation.

Interpretation and next steps: [docs/evaluation.md](docs/evaluation.md). All generated
numbers: [docs/results.md](docs/results.md), produced by `scripts/make_report.py`. Background:

- **Walkthrough notebook: [notebooks/walkthrough.ipynb](notebooks/walkthrough.ipynb)**, which
  runs end to end on cached data
- [Data card](docs/data_card.md): catalogue release and query, label mapping, acknowledgements
- [Catalogue findings](docs/catalogue_findings.md): classes, provenance, leakage evidence
- [Leakage audit](docs/leakage_audit.md), [every column](docs/leakage_columns.md) and
  [results](docs/leakage_audit_results.md)
- [Splits](docs/splits.md): host-star grouped, label-stratified partitions
- [Tabular baseline](docs/tabular_baseline.md): why catalogue features score so high
- [Light-curve pipeline](docs/lightcurve_pipeline.md): detrending, folding, views
- [Evaluation protocol](docs/evaluation.md): PR-AUC, bootstrap, thresholds, error analysis

## Setup

Requires Python 3.11 or 3.12.

```bash
py -3.12 -m venv .venv          # Windows; use python3.12 elsewhere
.venv\Scripts\activate          # Windows; use source .venv/bin/activate elsewhere
pip install -e ".[dev]"
pytest
```

## Reproducing the pipeline

Run from the repository root, in order. Times are for a 12-core CPU without a GPU.

| Step | Command | Output | Time |
|---|---|---|---|
| Catalogue | `python scripts/fetch_catalogue.py` | `data/raw/koi_cumulative.csv` (cached; `--force` refreshes) | < 1 min |
| Inspection | `python scripts/inspect_catalogue.py` | `outputs/catalogue_inspection.md` | seconds |
| Leakage audit | `python scripts/audit_leakage.py --strict` | `outputs/leakage_audit_results.md` | seconds |
| Splits | `python scripts/make_splits.py` | `data/processed/splits.csv` (created once; `--force` to regenerate) | seconds |
| Tabular baselines | `python scripts/train_tabular.py` and `python scripts/train_tabular.py --groups transit stellar --name tabular_lc` | `outputs/<name>_oof.csv`, `outputs/<name>_cv.md`, `models/<name>.joblib` | ~1 min each |
| Light-curve views | `python scripts/build_views.py --workers 16 --dev-stars 1000` | `data/processed/views/<kepid>.npz` (resumable) | ~1.5 h (all stars: ~6-11 h) |
| Matched tabular control | `python scripts/train_tabular.py --views-only --groups transit stellar --name tabular_lc_sub` | as above | < 1 min |
| CNN | `python scripts/train_cnn.py` | `outputs/cnn_oof.csv`, `outputs/cnn_cv.md`, `models/cnn.pt` | ~20 min |
| Development evaluation | `python scripts/evaluate.py --split dev` | `outputs/evaluation_dev.{md,json,png}` | seconds |
| Test evaluation | `python scripts/evaluate.py --split test` | `outputs/evaluation_test.{md,json,png}`: run **once**, last | seconds |
| Report | `python scripts/make_report.py` | `docs/results.md`, `docs/figures/` | seconds |
| Leakage before/after | `python scripts/leakage_inflation.py` | `outputs/leakage_inflation.md` | ~1 min |
| Operating points | `python scripts/operating_points.py` | `outputs/operating_points.md` | seconds |
| PR bands | `python scripts/plot_pr_bands.py <scores.csv> --models ... --title ... --output ...` | PNG | ~1 min |
| Period check | `python scripts/check_periods.py` | `outputs/period_check.md` | ~1 min |
| Walkthrough | `jupyter lab notebooks/walkthrough.ipynb` (`pip install -e ".[notebook]"`) | executed notebook | ~1 min |

`train_tabular.py` and `train_cnn.py` take `--class-weight balanced` (tested, not used by the
reported models; see [evaluation.md](docs/evaluation.md#class-weighting)); `train_cnn.py`
takes `--device cuda` and uses a GPU automatically when one is available.

The view build downloads about 7 MB of Kepler FITS files per star from MAST into
`paths.lightcurve_dir` (`configs/default.yaml`): ~15 GB for the capped build used here, ~41 GB
for all stars. Point it at a local folder outside any
cloud-synced directory; the processed views are only a few MB. Interrupted builds resume
where they stopped, and stars that fail are listed in `outputs/views_failures.csv` and
retried on the next run.

Downloaded data (`data/`), trained models (`models/`) and run outputs (`outputs/`) are
generated locally and are not committed. Generated reports that are worth keeping are copied
into `docs/`.

## Layout

```
configs/default.yaml       All pipeline settings (paths, sample, features, models, evaluation)
src/exoplanet/
  config.py                Config loading, paths, seeding
  reporting.py             Markdown table helper
  data/
    catalogue.py           NASA Exoplanet Archive retrieval with provenance
    inspection.py          Catalogue inspection report
    labels.py              Label mapping and DR25 sample selection
    splits.py              Host-star grouped, label-stratified splits
    lightcurves.py         MAST light-curve retrieval and per-star view files
  features/
    leakage.py             Column deny policy and leakage checks
    columns.py             Meaning of every catalogue column; per-column audit table
    tabular.py             Allowed catalogue feature groups
    lightcurve.py          Stitching, detrending, phase folding, global/local views
  models/
    tabular.py             Gradient-boosted baseline
    cnn.py                 Two-tower 1D CNN
  evaluation/
    metrics.py             Shared out-of-fold metrics
    pr.py                  PR-AUC with host-star bootstrap, thresholds, operating points
    false_positives.py     Error analysis by Robovetter flag, SNR and period
    plots.py               Precision-recall figures, with bootstrap bands
    report.py              Results report assembly
scripts/                   Command-line entry points (see the table above)
notebooks/walkthrough.ipynb  End-to-end walkthrough on cached data (report companion)
tests/                     Unit tests (synthetic data; no network access needed)
docs/                      Findings, methodology notes and generated results
```

## Methodology principles

- **Labels:** `CONFIRMED` → 1, `FALSE POSITIVE` → 0. Unvetted `CANDIDATE` KOIs are excluded
  from training and evaluation.
- **Homogeneous sample:** only KOIs with DR25 parameters are modelled; older deliveries are
  ~98% false positives and would let models learn provenance.
- **No leakage:** every catalogue column is either in an allowed feature group or denied with a
  reason (vetting outputs, provenance, uncertainties, epoch, sky position); denied columns
  cannot reach a model. The transit epoch is used only to align light curves.
- **Grouped splits:** a held-out test set (20%) and 5 cross-validation folds on the rest, all
  grouped by host star (`kepid`) and stratified by label, so no star appears in more than one
  partition. Splits are generated once and shared by every model.
- **No tuning on validation or test data:** models train for fixed iterations or epochs; no
  early stopping or checkpoint selection on validation folds. Operating thresholds are set on
  development out-of-fold scores. The test set is scored once, at the end.
- **Fair comparison:** all models are evaluated on the same KOIs. The CNN, which sees no
  centroid data, is compared with both the full tabular baseline and a transit + stellar
  variant, and with that variant trained on the CNN's own training KOIs. PR-AUC intervals
  and model differences come from a paired bootstrap over host stars.

## Limitations

- **Labels come from the same diagnostics the tabular model sees.** Robovetter flags are
  computed from transit-shape and centroid statistics, so the tabular baseline partly
  re-learns the vetting rules; its high score is not evidence of independent vetting skill.
- **Vetted KOIs only.** Excluding candidates leaves the easier, already-resolved cases;
  performance on the unresolved candidates is likely lower and cannot be measured here.
- **Catalogue ephemerides.** Light curves are folded on catalogue periods, epochs and
  durations; an error in those propagates into the views. A period check flags views with no
  coherent transit (2.9%, almost all false positives; see
  [lightcurve_pipeline.md](docs/lightcurve_pipeline.md#period-check)).
- **Light-curve processing.** Long-cadence data only; moving-median detrending may distort
  very long transits. Odd/even and secondary-eclipse views were tried and did not help.
- **Modest CNN.** A small network with fixed hyperparameters and one seed; the test-scored
  version saw ~21% of the development KOIs (capped light-curve build); its views carry no
  absolute depth.

## Roadmap

1. [x] Project scaffold
2. [x] NASA data retrieval and catalogue inspection
3. [x] Label mapping and leakage audit
4. [x] Host-star grouped splitting
5. [x] Tabular baseline
6. [x] Light-curve pipeline: detrending, phase folding, global/local views
7. [x] 1D CNN
8. [x] Precision-recall evaluation and false-positive analysis
9. [x] Results report and documentation
