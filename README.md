# Kepler Exoplanet Transit Detection

Classifying NASA Kepler Objects of Interest (KOIs) as confirmed planets or false positives,
using a 1D CNN on phase-folded light curves and a tabular baseline on catalogue features.

> Status: all pipeline code is in place. The full light-curve view build is running; CNN
> training and the single held-out test evaluation follow it. See the roadmap below.

## Results

[docs/results.md](docs/results.md) is generated from pipeline outputs by
`scripts/make_report.py`; sections not yet run are marked pending. Background:

- [Catalogue findings](docs/catalogue_findings.md): classes, provenance, leakage evidence
- [Leakage audit](docs/leakage_audit.md) and [results](docs/leakage_audit_results.md)
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
| Light-curve views | `python scripts/build_views.py --workers 8` | `data/processed/views/<kepid>.npz` (resumable) | ~12 h |
| CNN | `python scripts/train_cnn.py` | `outputs/cnn_oof.csv`, `outputs/cnn_cv.md`, `models/cnn.pt` | ~20 min |
| Development evaluation | `python scripts/evaluate.py --split dev` | `outputs/evaluation_dev.{md,json,png}` | seconds |
| Test evaluation | `python scripts/evaluate.py --split test` | `outputs/evaluation_test.{md,json,png}`: run **once**, last | seconds |
| Report | `python scripts/make_report.py` | `docs/results.md`, `docs/figures/` | seconds |

The view build downloads about 41 GB of Kepler FITS files from MAST into
`paths.lightcurve_dir` (`configs/default.yaml`). Point it at a local folder outside any
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
    tabular.py             Allowed catalogue feature groups
    lightcurve.py          Stitching, detrending, phase folding, global/local views
  models/
    tabular.py             Gradient-boosted baseline
    cnn.py                 Two-tower 1D CNN
  evaluation/
    metrics.py             Shared out-of-fold metrics
    pr.py                  PR-AUC with host-star bootstrap, thresholds, operating points
    false_positives.py     Error analysis by Robovetter flag, SNR and period
    plots.py               Precision-recall figure
    report.py              Results report assembly
scripts/                   Command-line entry points (see the table above)
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
  variant. PR-AUC intervals and model differences come from a paired bootstrap over host
  stars.

## Limitations

- **Labels come from the same diagnostics the tabular model sees.** Robovetter flags are
  computed from transit-shape and centroid statistics, so the tabular baseline partly
  re-learns the vetting rules; its high score is not evidence of independent vetting skill.
- **Vetted KOIs only.** Excluding candidates leaves the easier, already-resolved cases;
  performance on the unresolved candidates is likely lower and cannot be measured here.
- **Catalogue ephemerides.** Light curves are folded on catalogue periods, epochs and
  durations; an error in those propagates into the views.
- **Light-curve processing.** Long-cadence data only; moving-median detrending may distort
  very long transits; no dedicated odd/even or secondary-eclipse views (extensions in later
  AstroNet-style work).
- **Modest CNN.** A small network trained on CPU with fixed hyperparameters and one seed;
  KOIs without usable light curves are excluded from CNN comparisons.

## Roadmap

1. [x] Project scaffold
2. [x] NASA data retrieval and catalogue inspection
3. [x] Label mapping and leakage audit
4. [x] Host-star grouped splitting
5. [x] Tabular baseline
6. [ ] Light-curve pipeline: detrending, phase folding, global/local views (code done; full
   build running)
7. [ ] 1D CNN (code done; training awaits the view build)
8. [ ] Precision-recall evaluation and false-positive analysis (code done; dev results for
   tabular models; CNN and test evaluation pending)
9. [ ] Results report and documentation (generator done; final interpretation pending)
