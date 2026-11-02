# Light-Curve Pipeline

Builds the CNN inputs: a global and a local view of each KOI's phase-folded light curve,
following Shallue & Vanderburg (2018). Code: `exoplanet.features.lightcurve` (pure NumPy,
unit-tested on synthetic light curves) and `exoplanet.data.lightcurves` (MAST retrieval and
view files). Entry point: `scripts/build_views.py`.

## Steps (per host star)

1. **Retrieve** every Kepler long-cadence quarter (PDCSAP flux, default quality bitmask) via
   lightkurve. FITS files are cached under `paths.lightcurve_dir` (~7 MB per star, ~41 GB for
   the 5,830 modelling-sample stars).
2. **Stitch**: drop non-finite points and divide each quarter by its median.
3. **Mask** the transits (+-1 duration) of *every* catalogue KOI on the star, including
   unvetted candidates, so no transit leaks into the trend.
4. **Detrend** each contiguous segment (split at gaps > 0.75 d) with a moving median of
   max(1.5 d, 3 x the longest transit duration on the star), interpolating across masked
   points. Divide the flux by the trend.
5. **Clip** upward outliers > 5 sigma (MAD) to remove flares and cosmic rays; downward
   points are kept because transits are downward.
6. **Fold and bin** each target KOI on `koi_period`/`koi_time0bk` (the epoch is only used for
   alignment and is never a model input):
   - global view: 2001 bins over the full orbit;
   - local view: 201 bins over +-2 transit durations, each bin 0.16 durations wide
     (overlapping). Kepler's 29.4-minute cadence is coarser than the local bin spacing for
     short transits, and narrow bins produced noisy, saw-toothed views in the trial build.
   Empty bins are filled by linear interpolation; each view is normalised to median 0 and
   minimum -1.

Each star's views go to `<views_dir>/<kepid>.npz`, written atomically; existing files are
skipped, so the build resumes after interruption. Quality metrics stored per KOI: the
fraction of empty global/local bins before interpolation and the number of in-transit
points. A truncated cached FITS file (from an interrupted download) triggers one cache
clear and re-download for that star.

## Trial build (25 stars, 26 KOIs)

All 25 stars succeeded after the retry fix. Views are centred on the transit, eclipsing
binaries show V-shaped deep eclipses and visible secondaries (e.g. K06252.01), and
phase-locked stellar variability is preserved in the global view (K00982.01). With
overlapping local bins, no trial KOI has empty local bins, including the 570-day K01008.01.
Throughput was ~7 s per star with 8 parallel workers (~12 h for the full sample).

## Build used for modelling (capped)

The full build ran at ~8.5 stars per minute (~11 h for all 5,830 stars), so it was stopped
after ~600 stars and replaced by a capped build (`--dev-stars 1000 --workers 16`, ~14 stars
per minute):

- **All 1,160 test-set stars** were built first, so the held-out evaluation covers the whole
  test set.
- **1,000 of 4,670 development stars**: the ~500 already built (the lowest KIC IDs, since the
  full build ran in ID order) plus a seeded random sample of the rest. The selection is saved
  in `data/processed/view_stars.csv`.

Result: 2,457 KOIs with views (1,123 development, 1,334 test); one star (KIC 10462462) had
no usable data after detrending. The CNN therefore trains on ~21% of the development KOIs the
tabular models use. Positive fractions of the sampled development folds range from 0.33 to
0.46 (overall 0.39, against 0.41 for the full sample), mostly sampling noise at ~220 KOIs per
fold. Building the remaining development stars (`--dev-stars` omitted) would let the CNN
train on the full development set.

## Full build (all development stars)

The remaining development stars were built later (`python scripts/build_views.py --workers
16`): views now exist for 6,665 of 6,677 KOIs (99.8%) on 5,820 of 5,830 stars. Ten stars
failed: KIC 10462462 (no usable data after detrending) and nine whose MAST downloads timed
out at 600 s during a server slowdown (`outputs/views_failures.csv`); they were not retried.
`cnn_full` trains on this full set (see [evaluation.md](evaluation.md)).

## Period check

Folding on a wrong period smears the transits across phase, leaving a view with no dip at
mid-transit, which a model would then learn from as if it were a real signal. Every view is
checked (`python scripts/check_periods.py`; `outputs/period_check.md`):

- **Metric.** `view_transit_snr` (`exoplanet.features.lightcurve`) divides the depth of the
  local view's centre (within a quarter duration of mid-transit) by the robust scatter of
  its out-of-transit bins. A unit test folds a synthetic light curve on its true period
  (value > 10) and on a period 2% off, which smears ~48 transits across phase (value < 3).
- **Result.** 190 of 6,665 views (2.9%) fall below 3. 184 of them are false positives, and
  155 of those carry the Robovetter's not-transit-like flag (22% of all false positives with
  views do): they are signals that never were coherent transits, which is exactly what the
  check should find. The CNN scores them near zero.
- **Planets.** Only 6 confirmed planets are flagged, all faint (catalogue SNR 8.5-13), where
  the folded dip is barely above the noise. No high-SNR planet has a flat view, so there is
  no sign of mis-stated catalogue periods among the planets.
- `view_transit_snr` follows the catalogue SNR (Spearman 0.66), as it should if the views
  are folded correctly.
- Flagged KOIs are reported, not removed: dropping them would remove mostly false
  positives, which is itself a form of label leakage.
