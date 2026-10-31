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

Models compared (`evaluation.models` in the config):

- `tabular`: gradient boosting on transit + stellar + centroid catalogue features;
- `tabular_lc`: transit + stellar only, since the light-curve CNN sees no centroid data;
- `tabular_lc_sub`: as `tabular_lc`, trained only on the development KOIs that have
  light-curve views. The capped view build leaves the CNN ~21% of the development KOIs, so
  this control separates "light curve vs catalogue" from "less training data";
- `cnn`: the two-tower CNN on global and local views.

All four were final, with thresholds fixed from development scores, before the test set was
scored. The test evaluation was run once (a first attempt crashed while loading the CNN
checkpoint, before any test score was computed or saved).

## Results

Full generated tables: [results.md](results.md). Test set: 1,334 of 1,335 KOIs (one star had
no usable light curve).

| model | dev PR-AUC (95% CI) | test PR-AUC (95% CI) | test recall / precision at dev threshold |
|---|---|---|---|
| tabular | 0.994 (0.989-0.997) | 0.993 (0.984-0.998) | 0.984 / 0.962 |
| tabular_lc | 0.960 (0.946-0.973) | 0.966 (0.947-0.982) | 0.734 / 0.978 |
| tabular_lc_sub | 0.931 (0.905-0.951) | 0.953 (0.935-0.969) | 0.394 / 0.977 |
| cnn | 0.917 (0.888-0.942) | 0.903 (0.870-0.934) | 0.518 / 0.937 |

Development figures are on the 1,123 development KOIs with views (the only ones every model
can score), so they differ slightly from the tabular-only numbers in
[tabular_baseline.md](tabular_baseline.md).

### Findings

1. **Centroid diagnostics dominate.** The full tabular model is best everywhere (test
   PR-AUC 0.993; recall 0.98 at precision 0.96). Removing centroid features costs 0.028
   PR-AUC (95% CI 0.012-0.046) and a quarter of the recall at the operating point.
2. **The CNN is behind a tabular model trained on the same data.** On the test set the CNN
   trails `tabular_lc_sub` by 0.051 PR-AUC (95% CI 0.023-0.079). On the smaller development
   comparison the gap (0.014, CI -0.013 to 0.042) was not significant; the test set is
   larger and is the definitive measurement.
3. **Training-set size matters, but explains only part of the gap.** Training the transit +
   stellar model on the CNN's 21% subset costs 0.013 test PR-AUC (CI 0.001-0.025). Most of
   the CNN's deficit relative to `tabular_lc` (0.063) remains after matching data.
4. **The CNN cannot see transit depth.** Each view is normalised so the transit minimum is
   -1, as in AstroNet, so a 1% stellar eclipse and a 100 ppm planet transit can look alike.
   Its most confident false positives include KOIs with implied radii of 54 and 57 Earth
   radii (K03873.01, K04939.01), which the tabular models reject easily: `koi_prad` is their
   most important feature. This is the most likely main cause of the CNN's weaker ranking
   and of the false positives among its highest scores (the ragged low-recall end of its
   curve).
5. **Where the CNN helps.** At the dev-chosen thresholds the CNN recovers more planets than
   `tabular_lc_sub` (0.52 vs 0.39 recall) at somewhat lower precision (0.94 vs 0.98), with
   the largest gains for high-SNR planets (0.77 vs 0.40 in the top SNR quartile). Its
   threshold also transferred less well: test precision 0.937 against the 0.95 target.
6. **Hard cases for every light-curve-only model** are low-SNR and short-period planets
   (CNN recall 0.20 and 0.24 in the lowest quartiles; `tabular_lc` 0.53 and 0.62).
   Several of the CNN's most confidently missed planets belong to multi-planet systems
   (K00904.02, K00904.03, K00282.03); one possible cause, not verified here, is sibling
   planets' transits left in the folded light curve.

### Most promising next steps

- Give the CNN the information its normalised views remove: transit depth or planet radius
  as scalar inputs, or an unnormalised view.
- Train the CNN on the full development set (build the remaining ~3,700 development stars).
- Add secondary-eclipse and odd/even views, and remove sibling planets' transits before
  folding.
