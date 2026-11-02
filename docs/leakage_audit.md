# Leakage Audit: Policy and Rationale

The goal is to classify KOIs from information available **before** vetting. Any column that
records the vetting decision, or that was updated because of it, would let a model look up the
answer instead of learning it.

The policy lives in code (`src/exoplanet/features/leakage.py`, `src/exoplanet/features/tabular.py`)
and is enforced in three ways:

1. **Complete classification.** Each of the 153 catalogue columns belongs to an allowed feature
   group or is denied with a category. The test suite and `scripts/audit_leakage.py` fail on any
   unclassified column, so new archive columns can't slip in silently.
2. **Hard check.** `select_features` calls `assert_no_leakage` on every feature matrix, which
   raises `LeakageError` if a denied column is present.
3. **Empirical audit.** `univariate_audit` scores each column's label separability, from its
   value and from its missingness alone. Any allowed feature above the thresholds in
   `configs/default.yaml` (value AUC > 0.9, missingness AUC > 0.6) is flagged, and `--strict`
   fails on it.

Current results: [leakage_audit_results.md](leakage_audit_results.md). Run with
`python scripts/audit_leakage.py --strict`.

## Labels

| `koi_disposition` | Label |
|---|---|
| CONFIRMED | 1 |
| FALSE POSITIVE | 0 |
| CANDIDATE | excluded: not yet vetted, so its true class is unknown |

Any other or missing disposition raises an error rather than being silently dropped.

## Modelling sample: DR25 parameters only

Labelled KOIs whose parameters come from older pipeline deliveries (`koi_parm_prov` =
`q1_q16_koi` or `q1_q17_dr24_koi`) are about 98% FALSE POSITIVE (893 of 910). These are KOIs
that were dropped before DR25 and never re-fitted, and they are also the rows with missing
stellar and transit fields. Keeping them would let a model learn "old delivery or missing
values ⇒ false positive". This is provenance, not astrophysics.

The default sample therefore keeps only `q1_q17_dr25_koi`: **6,677 KOIs on 5,830 stars,
40.9% CONFIRMED** (from 7,587 labelled KOIs). Within this sample no allowed feature's
missingness separates the classes by more than AUC 0.55. Set
`sample.parameter_provenance: null` to disable the filter.

## Denied columns

| Category | Examples | Why |
|---|---|---|
| label | `koi_disposition`, `koi_pdisposition`, `koi_score`, `koi_fpflag_*`, `koi_comment`, `kepler_name` | The target or vetting outputs. `kepler_name` exists only once a planet is confirmed (2,746 of 2,748 CONFIRMED vs 2 of 4,839 FALSE POSITIVE). `koi_score` alone gives AUC 0.98. |
| provenance | `koi_parm_prov`, `koi_tce_delivname`, `koi_sparprov`, `koi_fittype`, `koi_quarters` | Delivery and fit method encode the label (see above). |
| uncertainty | all `*_err`, `*_err1`, `*_err2` (68 columns) | Uncertainties separate the classes better than the values they belong to (`koi_steff_err2` AUC 0.78 vs `koi_steff` 0.63). The likely cause is follow-up characterisation of planet hosts, which is post-detection information. Excluded as a precaution. |
| epoch | `koi_time0bk`, `koi_time0` | An arbitrary reference time, yet AUC 0.66, through correlation with period and observing coverage. |
| sky position | `ra`, `dec`, `koi_fwm_sra`, `koi_fwm_sdec` | Not a property of the signal; risks memorising field or CCD systematics. The flux-weighted centroid RA/Dec are absolute positions; the centroid *offsets* are kept. |
| identifier | `kepid`, `kepoi_name`, `ra_str`, `dec_str`, data links | No physical information. `kepid` is still used, for grouping splits. |
| constant / empty | `koi_eccen`, `koi_ldm_coeff1–4`, `koi_model_chisq`, `koi_sage`, … | No information, or redundant with stellar parameters. |

## Allowed feature groups

| Group | Columns | Default | Notes |
|---|---|---|---|
| `transit` | 17 | on | Fit parameters and detection statistics (period, depth, duration, SNR, odd-even significance, derived radius and insolation) |
| `stellar` | 13 | on | Host temperature, gravity, metallicity, radius, mass and photometry |
| `centroid` | 11 | on | Centroid-offset diagnostics. Legitimate vetting *inputs*, and the strongest allowed signal (`koi_fwm_stat_sig`, AUC 0.79). The light-curve CNN does not see them. |
| `multiplicity` | 2 | **off** | `koi_count`, `koi_tce_plnt_num`. A real astrophysical effect (multi-KOI stars are rarely false positives), but it is system context the CNN cannot see. The tabular baseline will be reported with and without it. |

## Known residual risks

- **Allowed features are not free of vetting influence.** Stellar parameters in DR25 draw on
  heterogeneous sources, including spectroscopy, which was obtained preferentially for planet
  candidates. Uncertainties are excluded for this reason; the values themselves are kept, and
  the risk is accepted and documented.
- **Centroid diagnostics overlap with `koi_fpflag_co`,** which the Robovetter derived from them.
  Using them is legitimate, since they are its inputs, but the baseline partly re-learns the
  Robovetter's centroid test.
- **Labels are human and Robovetter decisions,** not ground truth. CONFIRMED also requires
  follow-up, which favours bright hosts and short periods.
