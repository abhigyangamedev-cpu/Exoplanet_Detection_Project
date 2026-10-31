# Tabular Baseline

Histogram gradient boosting (`exoplanet.models.tabular`) on the leakage-checked catalogue
feature groups `transit`, `stellar` and `centroid` (41 features). Missing values are handled
natively, so no imputation statistics can cross folds. Hyperparameters are fixed in
`configs/default.yaml` and early stopping is disabled, since its internal validation split
would not be grouped by host star. Generated report: `outputs/tabular_cv.md`.

## Cross-validation (development folds only)

Pooled out-of-fold PR-AUC **0.994** (positive fraction 0.409); per-fold PR-AUC 0.993-0.997,
ROC-AUC 0.995-0.998. The held-out test set has not been scored.

Feature-group ablation (pooled out-of-fold PR-AUC, same folds and parameters):

| groups | PR-AUC |
|---|---|
| transit | 0.959 |
| stellar | 0.768 |
| centroid | 0.888 |
| transit + stellar | 0.969 |
| transit + stellar + centroid | 0.994 |

## Why the score is so high, and why it is not leakage

- No feature dominates: the largest permutation-importance drop is 0.022 PR-AUC (`koi_prad`),
  and the univariate audit flagged no column.
- The false positives are largely easy. 48% have `koi_prad` > 20 Earth radii (stellar
  eclipsing binaries) against 0.6% of confirmed planets, and 45% carry the centroid-offset
  flag, which the `koi_dikco_*`/`koi_dicco_*` diagnostics measure directly.
- Every false positive in the sample has at least one Robovetter flag set, and those flags are
  computed from the same transit-shape and centroid diagnostics used here. The baseline is
  therefore largely re-learning the vetting rules that produced the labels. That is
  legitimate (these features were available before vetting), but it means the baseline sets a
  high bar.

## Implications for the CNN comparison

The CNN sees only light-curve views, not centroid diagnostics. The fair like-for-like
comparison is against `transit + stellar` (0.969); the full baseline (0.994) is the
reference for what catalogue diagnostics add. Both should be reported in the final
evaluation.
