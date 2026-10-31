# KOI Catalogue: Key Findings

Snapshot retrieved 2026-10-31 (SHA-256 `a85348f2…`, 9,564 KOIs × 153 columns). Full generated
report: [catalogue_inspection.md](catalogue_inspection.md). Regenerate with
`python scripts/fetch_catalogue.py --force && python scripts/inspect_catalogue.py --output docs/catalogue_inspection.md`.

## Classes

| Disposition | KOIs |
|---|---|
| FALSE POSITIVE | 4,839 |
| CONFIRMED | 2,748 |
| CANDIDATE (excluded) | 1,977 |

The labelled set is 7,587 KOIs on 6,641 stars, 36.2% positive. That imbalance is moderate, so
evaluation will use precision-recall rather than accuracy.

## Leakage evidence (input to the step 3 audit)

Among labelled KOIs, these columns encode the label almost exactly and must be excluded from
features:

| Column | Evidence |
|---|---|
| `kepler_name` | Present for 2,746 of 2,748 CONFIRMED and 2 of 4,839 FALSE POSITIVE |
| `koi_pdisposition` | 4,838 of 4,839 FALSE POSITIVE are pipeline FALSE POSITIVE; every CONFIRMED but 9 is pipeline CANDIDATE |
| `koi_fpflag_nt/ss/co/ec` | Any flag set for 4,744 of 4,839 FALSE POSITIVE and 18 of 2,748 CONFIRMED |
| `koi_score` | Median 1.0 for CONFIRMED, 0.0 for FALSE POSITIVE |

The false-positive flags are still useful *after* modelling, to break errors down by
false-positive type (not-transit-like 31.5%, stellar eclipse 45.7%, centroid offset 39.0%,
ephemeris match 23.7% of FALSE POSITIVE; a KOI can have several flags).

## Host-star structure

- 8,214 host stars; 938 host more than one KOI (up to 7).
- The confirmed fraction rises steeply with multiplicity: 22% for single-KOI stars, 70% for two,
  around 93–95% for three or more. This is the known multiplicity effect: multi-transit systems
  are rarely false positives.
- `koi_count` equals the number of KOIs on the star for every row. It is a legitimate
  astrophysical signal but a very strong one, so step 3 must decide whether to include it, and
  the baseline should be reported with and without it.
- 47 stars host both a CONFIRMED and a FALSE POSITIVE KOI. Splitting by `kepid` keeps them
  on one side of each split.

## Data quality

- 24 columns are entirely empty (e.g. `koi_ingress`, `koi_sage`, `koi_longp`, `koi_model_chisq`)
  and can be dropped.
- About 12% of KOIs lack `koi_num_transits`, `koi_max_sngle_ev`, `koi_max_mult_ev` and
  `koi_quarters`, so the tabular baseline needs a missing-value strategy (e.g. a native-NaN
  gradient-boosting model).
