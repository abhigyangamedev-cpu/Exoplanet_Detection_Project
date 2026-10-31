# Evaluation Protocol

`scripts/evaluate.py` compares models by precision-recall and analyses their errors.
Code: `exoplanet.evaluation` (`pr`, `false_positives`, `plots`, `metrics`).

## Protocol

- **Two modes.** `--split dev` evaluates out-of-fold scores from the development folds and
  can be run at any time. `--split test` scores the held-out test set with the final models
  (refit on all development KOIs). The test set is scored once, after all modelling
  decisions are made; the script refuses to overwrite test results without `--force`.
- **Same KOIs for every model.** Models are compared on the intersection of the KOIs they
  can score (the CNN lacks KOIs whose light curves could not be built); the report states
  the coverage.
- **Uncertainty.** PR-AUC 95% intervals come from 1,000 cluster-bootstrap resamples of host
  stars, because KOIs on one star are correlated. Model differences are paired: both models
  are scored on the same resampled stars.
- **Operating point.** Each model's threshold is the lowest score reaching precision 0.95 on
  the development out-of-fold scores, maximising recall there; it is applied unchanged to
  the test set.
- **False-positive analysis.** For each Robovetter flag (not transit-like, stellar eclipse,
  centroid offset, ephemeris match), the share of false positives scored below threshold.
  Flags are vetting outputs and never model inputs; KOIs can carry several. Recall on
  confirmed planets is broken down by SNR and period quartiles, and each model's most
  confident mistakes are listed.

Models compared (`evaluation.models` in the config): `tabular` (transit + stellar +
centroid features), `tabular_lc` (transit + stellar: the like-for-like comparison for the
light-curve CNN, which sees no centroid data) and `cnn`.

## Development results so far (tabular models)

On all 5,342 development KOIs (out-of-fold):

| model | PR-AUC | 95% CI | recall at precision 0.95 |
|---|---|---|---|
| tabular | 0.994 | 0.992-0.995 | 0.980 |
| tabular_lc | 0.969 | 0.962-0.974 | 0.819 |

Centroid features add 0.025 PR-AUC (95% CI 0.020-0.031). Without them, recall at 95%
precision falls most for low-SNR (0.67 vs 0.97) and short-period (0.66 vs 0.96) planets.
`tabular_lc` rejects centroid-offset false positives less often (0.946 vs 0.982) but
rejects stellar eclipses and non-transit-like signals slightly more often.

Notes from the error lists:

- 52 KOIs lack transit-fit values (SNR, depth, radius); 51 are false positives, so
  missingness carries a weak signal (missing-indicator AUC 0.506 in the leakage audit). One
  such confirmed planet (K07368.01) receives a near-zero score.
- One false positive (K00242.01) carries no Robovetter flag; it is among the tabular
  model's highest-scored false positives.

CNN results and the single test-set evaluation will be added once the light-curve views are
built and the CNN is trained.
