# Kepler Exoplanet Transit Detection

Classifying NASA Kepler Objects of Interest (KOIs) as confirmed planets or false positives,
using a 1D CNN on phase-folded light curves and a tabular baseline on catalogue features.

> Status: early development. See the roadmap below.

## Setup

Requires Python 3.11 or 3.12.

```bash
py -3.12 -m venv .venv          # Windows; use python3.12 elsewhere
.venv\Scripts\activate          # Windows; use source .venv/bin/activate elsewhere
pip install -e ".[dev]"
pytest
```

## Usage

```bash
python scripts/fetch_catalogue.py      # KOI table -> data/raw/ (cached; --force to refresh)
python scripts/inspect_catalogue.py    # report -> outputs/catalogue_inspection.md
python scripts/audit_leakage.py --strict  # report -> outputs/leakage_audit_results.md
```

Catalogue findings: [docs/catalogue_findings.md](docs/catalogue_findings.md). Leakage policy: [docs/leakage_audit.md](docs/leakage_audit.md).

## Layout

```
configs/                 Pipeline configuration (default.yaml)
src/exoplanet/
  config.py              Config loading, paths, seeding
  data/                  NASA retrieval, catalogue inspection, labels, grouped splits
  features/              Detrending, phase folding, global/local views, tabular features
  models/                1D CNN and tabular baseline
  evaluation/            Precision-recall evaluation, false-positive analysis
scripts/                 Command-line entry points
tests/                   Unit tests
docs/                    Data findings and methodology notes
```

Downloaded data (`data/`), trained models (`models/`) and run outputs (`outputs/`) are
generated locally and are not committed.

## Methodology principles

- **Labels:** `CONFIRMED` → 1, `FALSE POSITIVE` → 0. Unvetted `CANDIDATE` KOIs are excluded
  from training and evaluation.
- **Homogeneous sample:** only KOIs with DR25 parameters are modelled; older deliveries are
  ~98% false positives and would let models learn provenance.
- **No leakage:** every catalogue column is either in an allowed feature group or denied with a
  reason (vetting outputs, provenance, uncertainties, epoch, sky position); denied columns
  cannot reach a model.
- **Grouped splits:** cross-validation folds are grouped by host star (`kepid`) so that no
  star appears in both training and test data.
- **Evaluation:** precision-recall curves and PR-AUC, with both models evaluated on identical
  splits.

## Roadmap

1. [x] Project scaffold
2. [x] NASA data retrieval and catalogue inspection
3. [x] Label mapping and leakage audit
4. [ ] Host-star grouped splitting
5. [ ] Tabular baseline
6. [ ] Light-curve pipeline: detrending, phase folding, global/local views
7. [ ] 1D CNN
8. [ ] Precision-recall evaluation and false-positive analysis
9. [ ] Results report and documentation
