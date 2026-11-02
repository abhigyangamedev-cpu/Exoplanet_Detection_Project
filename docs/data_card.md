# Data Card

What data this project uses, exactly where it came from, how labels are defined, and how to
acknowledge it.

## Catalogue

| Item | Value |
|---|---|
| Source | NASA Exoplanet Archive, Kepler Objects of Interest **cumulative** table |
| Access | TAP service `https://exoplanetarchive.ipac.caltech.edu/TAP/sync` (`scripts/fetch_catalogue.py`) |
| Exact query | `select * from cumulative` (ADQL, CSV format) |
| Retrieved | 2026-10-31 03:14:15 UTC |
| Size | 9,564 KOIs × 153 columns |
| File | `data/raw/koi_cumulative.csv`, SHA-256 `a85348f20082d24fb14d4a05a9a44c644b801998fad7cec36f8980eb1f0a69ab` |
| Provenance record | `data/raw/koi_cumulative.meta.json` (query, URL, time, row count, checksum) |
| Parameter release used | Kepler **DR25** (`koi_parm_prov = q1_q17_dr25_koi`); see "Modelling sample" |

The cumulative table is revised as KOIs are re-vetted, so results are tied to this snapshot.
`scripts/fetch_catalogue.py` caches the file and re-downloads only with `--force`; every
later step reads the cached copy, and the checksum is carried into the splits, model files
and reports.

## Light curves

| Item | Value |
|---|---|
| Source | Kepler long-cadence (~29.4 min) light curves, MAST archive at STScI |
| Access | `lightkurve.search_lightcurve("KIC <kepid>", mission="Kepler", author="Kepler", exptime="long")`, then `download_all` |
| Flux | `pdcsap_flux` (Presearch Data Conditioning), `quality_bitmask="default"` |
| Quarters | Every available quarter (Q0–Q17) per star |
| Cache | Raw FITS under `C:/kepler_cache` (about 6 MB per star); binned views under `data/processed/views/<kepid>.npz` |
| Coverage | Views for 6,665 of 6,677 modelling KOIs (99.8%), 5,820 of 5,830 host stars |
| Failed stars | 10: KIC 10462462 (no usable data after detrending) and 9 MAST download timeouts (`outputs/views_failures.csv`) |
| Software | lightkurve 2.6.0, astropy 8.0.2 |

Builds read complete cached stars without contacting MAST, so the pipeline reruns offline
once the cache is filled.

## Label definition

| `koi_disposition` | Label | Count (all labelled) | Count (modelling sample) |
|---|---|---|---|
| CONFIRMED | 1 (planet) | 2,748 | 2,731 |
| FALSE POSITIVE | 0 | 4,839 | 3,946 |
| CANDIDATE | excluded | 1,977 | – |

- **Why CONFIRMED only for the positive class.** In the cumulative table, statistically
  validated planets are already dispositioned CONFIRMED, so "confirmed and validated" is one
  class. CANDIDATE KOIs have not been vetted to a final answer, so their true class is
  unknown; they are excluded rather than guessed.
- The label comes from `koi_disposition` (the archive's final disposition), never from
  `koi_pdisposition` (the Kepler pipeline's own verdict), which is a vetting output and is
  removed by the leakage audit.
- Any other or missing disposition raises an error (`exoplanet.data.labels`).

## Modelling sample

Labelled KOIs whose parameters come from older pipeline deliveries (`q1_q16_koi`,
`q1_q17_dr24_koi`) are 98% false positives (893 of 910) and have systematically missing
fields: keeping them would let a model learn delivery history instead of astrophysics.
The modelling sample keeps DR25 parameters only: **6,677 KOIs on 5,830 host stars, 40.9%
confirmed**.

| Partition | KOIs | Host stars | Planet fraction |
|---|---|---|---|
| Development (5 folds) | 5,342 | 4,670 | 0.409 |
| Test (held out, scored once) | 1,335 | 1,160 | 0.409 |

Splits are grouped by host star (`kepid`) and stratified by label, seed 42, created
2026-10-31 03:40:48 UTC (`data/processed/splits.csv`).

## What the labels and models mean

The labels are **vetting judgements** made by the Kepler team and follow-up programmes. A
model trained on them reproduces that judgement: a high score means "this signal resembles
KOIs that were confirmed", not "a planet has been detected". Confirming a planet requires
follow-up observation (radial velocities, high-resolution imaging, statistical validation).

## Known limitations

- The cumulative table changes over time; a later snapshot will give somewhat different
  labels and counts.
- CONFIRMED planets are biased toward bright hosts and short periods, where follow-up is
  easiest.
- Ten host stars have no light-curve views; the CNN cannot score their 12 KOIs.

## Acknowledgements

This research has made use of the NASA Exoplanet Archive, which is operated by the
California Institute of Technology, under contract with the National Aeronautics and Space
Administration under the Exoplanet Exploration Program.

This project includes data collected by the Kepler mission and obtained from the MAST data
archive at the Space Telescope Science Institute (STScI). Funding for the Kepler mission is
provided by the NASA Science Mission Directorate. STScI is operated by the Association of
Universities for Research in Astronomy, Inc., under NASA contract NAS 5-26555.

Light curves were retrieved and processed with Lightkurve (Lightkurve Collaboration, 2018,
Astrophysics Source Code Library, ascl:1812.013) and Astropy (Astropy Collaboration). The
two-view CNN follows Shallue & Vanderburg (2018), AJ 155, 94.
