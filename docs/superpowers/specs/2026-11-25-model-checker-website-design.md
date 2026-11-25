# Model checker website — design

Date: 2026-11-25. Status: awaiting approval.

## Intent

A local website where the user picks any catalogued Kepler candidate (KOI), sees what each
model said about it, and sees whether that was right or wrong against the true label, so
they can confirm the models with their own eyes. The same site also shows the final report.

## Goals

- Pick any of the ~6,677 labelled KOIs: search by KOI name or KIC number, or browse a
  filtered list.
- For the picked KOI: true label, each model's score and verdict, a clear RIGHT / WRONG mark
  per model, the light curve the CNN saw, and the key catalogue numbers.
- Filters that make checking fast: partition (dev / test), true label, "only mistakes" for a
  chosen model, SNR and period ranges.
- A Report page showing `docs/results.md` and `docs/evaluation.md` with their figures.
- Starts with one command, works offline, opens instantly.

## Non-goals

- New stars from MAST or uploaded light curves (no true label, so right/wrong can't be shown).
- Retraining, re-scoring, or tuning anything from the site.
- Hosting on the internet; this runs on the user's machine only.

## Anti-goals

- **Flattering numbers.** A final model run on a dev star it was trained on would look
  near-perfect. The site never shows that.
- **Using the test set again.** No model that has not already been scored on the test set
  gets scored on it by the site.

## How the scores stay honest (my call)

The site runs no model. It reads the scores the pipeline already saved:

| KOI is in | Score shown | Source |
|---|---|---|
| dev | out-of-fold (from the fold model that never saw that star) | `outputs/<model>_oof.csv` |
| test | the one-time held-out score | `outputs/evaluation_test_scores.csv` |

Every score is tagged "out-of-fold" or "held-out test" on the page. Consequences:

- Test KOIs show `tabular`, `tabular_lc`, `tabular_lc_sub`, `cnn` (the models scored on test).
  `cnn_full` and the weighted models show "not scored on test" there.
- Dev KOIs show every model that has an OOF file; a model with no score for that KOI (for
  example no light-curve view) shows "no score" rather than a guess.

## Verdict threshold (my call)

Verdict = planet if score >= threshold. Default per model: the dev-chosen "precision 0.95"
threshold from `outputs/operating_points.md` (config `target_precision: 0.95`), e.g.
`cnn_full` 0.831, `tabular_lc` 0.840, `tabular` 0.227. A slider lets the user move it and
watch the verdict and the running tallies change. Models with no row in that file default
to 0.5, labelled as such.

## Approaches considered

1. **Small local web server + one hand-built page (chosen).** A Flask app reads the saved
   CSVs once at start and loads one star's `.npz` view on demand; the page is plain
   HTML/CSS/JS with charts drawn in the browser. One small new dependency, fast, full control
   of the look, easy to test with Flask's test client.
2. **Streamlit app.** Quickest to write, but a heavy dependency, re-runs the whole script on
   every click, and looks like every other Streamlit demo.
3. **Fully static folder, no server.** Nothing to run, but the light-curve views are far too
   large to embed in one page (about 19 million numbers), and browsers block loading
   per-star files from disk without a server.

The user handed over the how; approach 1 is my pick.

## Design

```
+----------------------------------------------------------------------------+
|  Kepler KOI model checker                      [ Checker ]  [ Report ]     |
+---------------------------+------------------------------------------------+
| Search: [K00752.01    ]   |  K00752.01   KIC 10797460   dev, fold 3        |
| Partition: (all|dev|test) |  TRUE LABEL: CONFIRMED PLANET                  |
| Label: (all|planet|FP)    |                                                |
| Show: [x] only mistakes   |  model       score   verdict   result          |
|       of [cnn_full v]     |  cnn_full    0.91    planet    RIGHT  (OOF)    |
| SNR / period ranges       |  tabular_lc  0.42    FP        WRONG  (OOF)    |
|---------------------------|  tabular     0.98    planet    RIGHT  (OOF)    |
| K00752.01  planet  ok     |  threshold: [----o-----] per model             |
| K00753.01  FP      WRONG  |                                                |
| K00754.01  FP      ok     |  [ global view chart ]   [ local view chart ]  |
| ...  (6,677, paged)       |  [ odd | even ]          [ secondary ]         |
|                           |  period 9.49 d  depth 616 ppm  SNR 35.8  ...   |
| Shown: 412  right 371     |  FP flags: none                                |
+---------------------------+------------------------------------------------+
```

- **Code:** new package `src/exoplanet/web/` (data loading + Flask app + `templates/`,
  `static/`), launcher `scripts/serve.py` (prints the address and opens the browser), Flask
  added as an optional `web` extra in `pyproject.toml`.
- **Endpoints:** `/` (checker), `/report` (report), `/api/kois` (filtered, paged list with
  per-model right/wrong at the default thresholds), `/api/koi/<name>` (scores, label,
  catalogue numbers, views).
- **Tallies:** the left panel shows how many of the filtered KOIs the chosen model got right
  and wrong, so a filter such as "test, SNR < 15" gives a live accuracy count.
- **Report page:** the two markdown files rendered to HTML with their figures, a contents
  list down the side, and a tab for each document.
- **Missing pieces:** a KOI with no view shows the scores and a "no light curve built" note.
- **Tests:** pytest for the score-source rule (dev -> OOF, test -> held-out, never mixed),
  verdict/right-wrong logic, filters, and each endpoint through the Flask test client, on
  small fixtures.

## Left to the builder

Visual styling, chart drawing details, page size of the list, markdown rendering library.
