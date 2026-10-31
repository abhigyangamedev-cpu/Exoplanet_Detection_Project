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
```

Catalogue findings are summarised in [docs/catalogue_findings.md](docs/catalogue_findings.md).

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
- **No leakage:** vetting outputs (e.g. `koi_score`, `koi_fpflag_*`, `koi_pdisposition`) are
  never used as model features.
- **Grouped splits:** cross-validation folds are grouped by host star (`kepid`) so that no
  star appears in both training and test data.
- **Evaluation:** precision-recall curves and PR-AUC, with both models evaluated on identical
  splits.

## Roadmap

1. [x] Project scaffold
2. [x] NASA data retrieval and catalogue inspection
3. [ ] Label mapping and leakage audit
4. [ ] Host-star grouped splitting
5. [ ] Tabular baseline
6. [ ] Light-curve pipeline: detrending, phase folding, global/local views
7. [ ] 1D CNN
8. [ ] Precision-recall evaluation and false-positive analysis
9. [ ] Results report and documentation
