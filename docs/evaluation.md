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
4. **Missing transit depth explains only a few CNN errors.** Each view is normalised so
   the transit minimum is -1, as in AstroNet, so the CNN cannot see absolute depth. Its two
   most confident test false positives are giant-radius eclipsing binaries (K03873.01 and
   K04939.01, 54 and 57 Earth radii) that every tabular model scores below 0.08. But they
   are a minority: only 4 of the 19 test false positives the CNN accepts have radii above
   20 Earth radii, and the median accepted depth is ~1,300 ppm. Ten of the 19 carry the
   centroid-offset flag, i.e. contamination from a nearby star that photometry alone cannot
   reveal. A follow-up that gave the CNN depth explicitly gained little (see below). An
   earlier version of this note called missing depth the most likely main cause; the
   follow-up does not support that.
5. **Where the CNN helps.** At the dev-chosen thresholds the CNN recovers more planets than
   `tabular_lc_sub` (0.52 vs 0.39 recall) at somewhat lower precision (0.94 vs 0.98), with
   the largest gains for high-SNR planets (0.77 vs 0.40 in the top SNR quartile). Its
   threshold also transferred less well: test precision 0.937 against the 0.95 target.
6. **Hard cases for every light-curve-only model** are low-SNR and short-period planets
   (CNN recall 0.20 and 0.24 in the lowest quartiles; `tabular_lc` 0.53 and 0.62).
   Several of the CNN's most confidently missed planets belong to multi-planet systems
   (K00904.02, K00904.03, K00282.03); one possible cause, not verified here, is sibling
   planets' transits left in the folded light curve.

## Follow-up experiments (development folds only)

The test set has been used, so later experiments are judged on the development folds only,
on the 1,123 KOIs with views (`python scripts/evaluate.py --split dev --tag <name>`).

### Transit depth as an input (`cnn_depth`)

The CNN plus three log-scaled scalars, `koi_depth`, `koi_period` and `koi_duration`, joined
to its dense head (`python scripts/train_cnn.py --name cnn_depth --scalars koi_depth
koi_period koi_duration`). Period and duration are already used to fold the views, so the new
information is depth. Control `tabular_scalars_sub`: gradient boosting on those three
features alone, trained on the same KOIs.

| model | dev PR-AUC (95% CI) |
|---|---|
| tabular_lc_sub | 0.931 (0.905-0.951) |
| cnn_depth | 0.926 (0.900-0.948) |
| cnn | 0.917 (0.888-0.942) |
| tabular_scalars_sub | 0.795 (0.751-0.841) |

- Depth adds +0.009 PR-AUC to the CNN (95% CI 0.000-0.018, p = 0.048): a marginal gain, not
  a fix.
- It does not reduce deep false positives: the CNN already accepts few of them (1 of 336
  development false positives with radii above 20 Earth radii; 3 with depth as an input).
- The three scalars alone are far weaker (0.795), so the views carry most of the CNN's
  information.

### Odd/even and secondary-eclipse views (`cnn_oe`)

Three extra views on the local grid: the local view of odd-numbered transits, of
even-numbered transits, and a view centred at phase 0.5 (secondary eclipse for a circular
orbit). They are scaled with the primary local view's normalisation, so a binary's unequal
odd/even depths or its secondary eclipse keep their relative depth instead of each being
normalised to -1. The four local-grid views enter the CNN's local tower as channels
(`python scripts/train_cnn.py --name cnn_oe --local-views local_view odd_view even_view
secondary_view`). Global and local views were rebuilt bit-identically, so earlier models'
results stand.

| model | dev PR-AUC (95% CI) | recall at precision 0.95 |
|---|---|---|
| tabular_lc_sub | 0.931 (0.905-0.951) | 0.421 |
| cnn_depth | 0.926 (0.900-0.948) | 0.572 |
| cnn | 0.917 (0.888-0.942) | 0.364 |
| cnn_oe | 0.909 (0.874-0.938) | 0.501 |

- **No improvement in ranking:** `cnn` - `cnn_oe` = 0.008 PR-AUC (95% CI -0.009 to 0.027,
  p = 0.39).
- **The targeted false positives were already handled.** The plain CNN accepts only 3 of 365
  stellar-eclipse false positives at its operating threshold; `cnn_oe` accepts 2. On real
  data the views do show the intended signatures (e.g. a secondary eclipse at -0.3 for the
  eclipsing binary K07278.01; matching odd/even and a flat secondary for the planet
  K00723.01), but the CNN could already reject these cases.
- **The CNN's main weakness is missed planets**, especially at low SNR and short period, and
  these views are false-positive diagnostics; they do not make faint transits easier to see.
- Recall at precision 0.95 rose (0.36 to 0.50, and 0.57 for `cnn_depth`), but each threshold
  is chosen on the same development scores it is measured on, and at 95% precision recall
  swings with small changes among the top-ranked KOIs. The threshold-free PR-AUC shows no
  difference, so this is not evidence of a real gain.
- Known limitation: for very short-period variables such as the 1.75-day contact binary
  K01957.02, the views show flat plateaus at the brightest phases. Upward-outlier clipping
  (meant for flares) appears to remove genuine variability there. It predates these views,
  affects only extreme variables (false positives either way) and has not been changed.

### Most promising next steps

- Train the CNN on the full development set (build the remaining ~3,700 development stars):
  training-set size measurably limits the matched tabular model (finding 3), and is likely to
  limit the CNN too. Neither extra input tried so far (depth, odd/even/secondary views)
  produced a clear gain on ~1,100 training KOIs.
- Recover more low-SNR and short-period planets, where every light-curve-only model is
  weakest; one candidate is removing sibling planets' transits before folding.
- Centroid-offset false positives need pixel-level information (centroid or difference-image
  data); photometry alone cannot separate them.
