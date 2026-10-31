# KOI Catalogue Inspection

- Source: `https://exoplanetarchive.ipac.caltech.edu/TAP/sync`
- Query: `select * from cumulative`
- Retrieved (UTC): 2026-10-31T03:14:15+00:00
- SHA-256: `a85348f20082d24fb14d4a05a9a44c644b801998fad7cec36f8980eb1f0a69ab`

## Overview

- KOIs: 9,564 across 8,214 host stars (153 columns)
- Labelled (CONFIRMED + FALSE POSITIVE): 7,587 KOIs on 6,641 stars
- Positive (CONFIRMED) fraction among labelled: 36.2%
- Stars hosting both a CONFIRMED and a FALSE POSITIVE KOI: 47

## Dispositions

| koi_disposition | n |
|---|---|
| FALSE POSITIVE | 4839 |
| CONFIRMED | 2748 |
| CANDIDATE | 1977 |

Archive disposition (rows) vs. Kepler pipeline disposition (columns). `koi_pdisposition` is a vetting output and must not be used as a feature.

| koi_disposition | CANDIDATE | FALSE POSITIVE |
|---|---|---|
| CANDIDATE | 1977 | 0 |
| CONFIRMED | 2739 | 9 |
| FALSE POSITIVE | 1 | 4838 |

## Host-star structure

| kois_on_star | n_stars |
|---|---|
| 1 | 7276 |
| 2 | 661 |
| 3 | 179 |
| 4 | 68 |
| 5 | 25 |
| 6 | 3 |
| 7 | 2 |

KOIs by number of KOIs on the same host star, with the confirmed fraction among labelled KOIs at each multiplicity.

| kois_on_star | CANDIDATE | CONFIRMED | FALSE POSITIVE | CONFIRMED fraction |
|---|---|---|---|---|
| 1 | 1501 | 1282 | 4493 | 0.222 |
| 2 | 326 | 702 | 294 | 0.7048 |
| 3 | 91 | 423 | 23 | 0.9484 |
| 4+ | 59 | 341 | 29 | 0.9216 |

## Median key quantities by class (labelled KOIs)

| column | CONFIRMED | FALSE POSITIVE |
|---|---|---|
| koi_period | 11.36 | 5.244 |
| koi_duration | 3.492 | 4.057 |
| koi_depth | 448.8 | 576 |
| koi_prad | 2.16 | 8.97 |
| koi_model_snr | 28.6 | 34.25 |
| koi_steff | 5618 | 5867 |
| koi_srad | 0.9615 | 1.01 |
| koi_kepmag | 14.61 | 14.4 |

## Missing values

128 of 153 columns have missing values; 24 are entirely empty.

| column | missing_fraction |
|---|---|
| koi_gmag_err | 100.0% |
| koi_imag_err | 100.0% |
| koi_zmag_err | 100.0% |
| koi_rmag_err | 100.0% |
| koi_kepmag_err | 100.0% |
| koi_longp_err2 | 100.0% |
| koi_sage_err2 | 100.0% |
| koi_sage_err1 | 100.0% |
| koi_sage | 100.0% |
| koi_sma_err1 | 100.0% |
| koi_sma_err2 | 100.0% |
| koi_ingress_err1 | 100.0% |
| koi_ingress | 100.0% |
| koi_ingress_err2 | 100.0% |
| koi_incl_err2 | 100.0% |
| koi_teq_err1 | 100.0% |
| koi_teq_err2 | 100.0% |
| koi_incl_err1 | 100.0% |
| koi_model_dof | 100.0% |
| koi_eccen_err1 | 100.0% |
| koi_model_chisq | 100.0% |
| koi_longp | 100.0% |
| koi_longp_err1 | 100.0% |
| koi_eccen_err2 | 100.0% |
| kepler_name | 71.3% |
| koi_score | 15.8% |
| koi_bin_oedp_sig | 15.8% |
| koi_comment | 12.6% |
| koi_num_transits | 11.9% |
| koi_max_sngle_ev | 11.9% |
| koi_quarters | 11.9% |
| koi_max_mult_ev | 11.9% |
| koi_fwm_stat_sig | 11.3% |
| koi_fwm_prao_err | 8.7% |
| koi_fwm_prao | 8.7% |
| koi_fwm_pdeco_err | 8.5% |
| koi_fwm_pdeco | 8.5% |
| koi_zmag | 6.4% |
| koi_dicco_mdec | 6.3% |
| koi_dicco_msky | 6.3% |
| koi_dicco_mra_err | 6.3% |
| koi_dicco_mra | 6.3% |
| koi_dicco_mdec_err | 6.3% |
| koi_dicco_msky_err | 6.3% |
| koi_dikco_msky_err | 6.0% |
| koi_dikco_msky | 6.0% |
| koi_dikco_mra_err | 6.0% |
| koi_dikco_mdec_err | 6.0% |
| koi_dikco_mdec | 6.0% |
| koi_dikco_mra | 6.0% |
| koi_fwm_sra_err | 5.3% |
| koi_fwm_sdec | 5.3% |
| koi_fwm_sdec_err | 5.3% |
| koi_fwm_sra | 5.3% |
| koi_steff_err2 | 5.1% |
| koi_slogg_err1 | 4.9% |
| koi_slogg_err2 | 4.9% |
| koi_srad_err2 | 4.9% |
| koi_smass_err1 | 4.9% |
| koi_srad_err1 | 4.9% |
| koi_smass_err2 | 4.9% |
| koi_steff_err1 | 4.9% |
| koi_fwm_sdeco | 4.8% |
| koi_fwm_sdeco_err | 4.8% |
| koi_fwm_srao_err | 4.8% |
| koi_fwm_srao | 4.8% |
| koi_duration_err1 | 4.7% |
| koi_dor_err2 | 4.7% |
| koi_period_err1 | 4.7% |
| koi_impact_err2 | 4.7% |
| koi_impact_err1 | 4.7% |
| koi_duration_err2 | 4.7% |
| koi_depth_err2 | 4.7% |
| koi_period_err2 | 4.7% |
| koi_time0_err1 | 4.7% |
| koi_time0bk_err2 | 4.7% |
| koi_time0bk_err1 | 4.7% |
| koi_time0_err2 | 4.7% |
| koi_depth_err1 | 4.7% |
| koi_dor_err1 | 4.7% |
| koi_smet_err1 | 4.0% |
| koi_smet_err2 | 4.0% |
| koi_smet | 4.0% |
| koi_incl | 3.8% |
| koi_prad | 3.8% |
| koi_ldm_coeff3 | 3.8% |
| koi_impact | 3.8% |
| koi_steff | 3.8% |
| koi_teq | 3.8% |
| koi_depth | 3.8% |
| koi_slogg | 3.8% |
| koi_smass | 3.8% |
| koi_sparprov | 3.8% |
| koi_srad | 3.8% |
| koi_trans_mod | 3.8% |
| koi_ldm_coeff2 | 3.8% |
| koi_ldm_coeff1 | 3.8% |
| koi_sma | 3.8% |
| koi_prad_err1 | 3.8% |
| koi_ldm_coeff4 | 3.8% |
| koi_limbdark_mod | 3.8% |
| koi_ror_err2 | 3.8% |
| koi_ror_err1 | 3.8% |
| koi_ror | 3.8% |
| koi_dor | 3.8% |
| koi_prad_err2 | 3.8% |
| koi_eccen | 3.8% |
| koi_model_snr | 3.8% |
| koi_datalink_dvs | 3.6% |
| koi_datalink_dvr | 3.6% |
| koi_tce_delivname | 3.6% |
| koi_tce_plnt_num | 3.6% |
| koi_srho_err2 | 3.4% |
| koi_insol_err1 | 3.4% |
| koi_insol_err2 | 3.4% |
| koi_insol | 3.4% |
| koi_srho_err1 | 3.4% |
| koi_srho | 3.4% |
| koi_kmag_err | 2.0% |
| koi_imag | 1.6% |
| koi_hmag_err | 1.0% |
| koi_jmag_err | 0.8% |
| koi_gmag | 0.4% |
| koi_hmag | 0.3% |
| koi_kmag | 0.3% |
| koi_jmag | 0.3% |
| koi_rmag | 0.1% |
| koi_kepmag | 0.0% |

## Column types

float64: 127, object: 20, int64: 6
