"""Plain-language meaning of every KOI catalogue column, and the per-column audit table.

The table lists each column with its meaning, the keep-or-drop decision and the reason,
so the leakage audit can be checked column by column. Decisions come from the policy in
``exoplanet.features.tabular`` (allowed groups) and ``exoplanet.features.leakage`` (denied
columns); this module only explains them.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

from exoplanet.features.leakage import DENY_CATEGORIES
from exoplanet.features.tabular import classify_column

DESCRIPTIONS: dict[str, str] = {
    # Identifiers and names
    "kepid": "Kepler Input Catalog ID of the host star",
    "kepoi_name": "KOI designation (KNNNNN.NN)",
    "kepler_name": "Kepler planet name (Kepler-NNN b), assigned once a planet is confirmed",
    "ra": "Right ascension of the host star (deg)",
    "dec": "Declination of the host star (deg)",
    "ra_str": "Right ascension, sexagesimal text",
    "dec_str": "Declination, sexagesimal text",
    # Photometry
    "koi_kepmag": "Kepler-band magnitude of the host star",
    "koi_gmag": "g-band magnitude (KIC)",
    "koi_rmag": "r-band magnitude (KIC)",
    "koi_imag": "i-band magnitude (KIC)",
    "koi_zmag": "z-band magnitude (KIC)",
    "koi_jmag": "J-band magnitude (2MASS)",
    "koi_hmag": "H-band magnitude (2MASS)",
    "koi_kmag": "K-band magnitude (2MASS)",
    # Vetting and dispositions
    "koi_disposition": "Archive disposition: CONFIRMED, CANDIDATE or FALSE POSITIVE (the target)",
    "koi_pdisposition": "Kepler pipeline disposition (CANDIDATE or FALSE POSITIVE)",
    "koi_score": "Robovetter disposition score, 0-1 (confidence in the pipeline disposition)",
    "koi_fpflag_nt": "False-positive flag: not transit-like",
    "koi_fpflag_ss": "False-positive flag: stellar eclipse (significant secondary)",
    "koi_fpflag_co": "False-positive flag: centroid offset (signal from a nearby star)",
    "koi_fpflag_ec": "False-positive flag: ephemeris match (contamination from another source)",
    "koi_vet_stat": "Vetting status (Done or Active)",
    "koi_vet_date": "Date of the last vetting update",
    "koi_comment": "Free-text vetting comments",
    "koi_disp_prov": "Source of the disposition",
    # Provenance
    "koi_delivname": "Name of the KOI delivery (pipeline run) the row comes from",
    "koi_tce_delivname": "Name of the TCE delivery that produced the detection",
    "koi_parm_prov": "Provenance of the transit parameters (pipeline release)",
    "koi_sparprov": "Provenance of the stellar parameters",
    "koi_fittype": "Type of transit fit (e.g. MCMC, least squares)",
    "koi_limbdark_mod": "Limb-darkening model used in the fit",
    "koi_trans_mod": "Transit model used in the fit",
    "koi_quarters": "Bit string of the Kepler quarters searched",
    "koi_datalink_dvs": "Link to the Data Validation summary",
    "koi_datalink_dvr": "Link to the Data Validation report",
    # Detection statistics
    "koi_count": "Number of KOIs on the same host star",
    "koi_tce_plnt_num": "TCE planet number within the system (order of detection)",
    "koi_num_transits": "Number of transits observed",
    "koi_max_sngle_ev": "Maximum single-event statistic (strongest single transit)",
    "koi_max_mult_ev": "Maximum multiple-event statistic (folded detection strength)",
    "koi_bin_oedp_sig": "Odd-even depth comparison statistic (eclipsing-binary test)",
    "koi_model_snr": "Transit signal-to-noise ratio",
    "koi_model_dof": "Degrees of freedom of the transit fit",
    "koi_model_chisq": "Chi-square of the transit fit",
    # Transit fit
    "koi_period": "Orbital period (days)",
    "koi_time0bk": "Transit epoch (BJD - 2,454,833)",
    "koi_time0": "Transit epoch (BJD)",
    "koi_duration": "Transit duration (hours)",
    "koi_ingress": "Ingress duration (hours)",
    "koi_depth": "Transit depth (ppm)",
    "koi_ror": "Planet-to-star radius ratio",
    "koi_dor": "Planet-star distance over stellar radius (a/R*)",
    "koi_impact": "Impact parameter",
    "koi_incl": "Orbital inclination (deg)",
    "koi_eccen": "Orbital eccentricity (fixed at 0 in the fit)",
    "koi_longp": "Longitude of periastron (deg)",
    "koi_ldm_coeff1": "Limb-darkening coefficient 1",
    "koi_ldm_coeff2": "Limb-darkening coefficient 2",
    "koi_ldm_coeff3": "Limb-darkening coefficient 3",
    "koi_ldm_coeff4": "Limb-darkening coefficient 4",
    "koi_prad": "Planet radius (Earth radii)",
    "koi_sma": "Orbital semi-major axis (AU)",
    "koi_teq": "Planet equilibrium temperature (K)",
    "koi_insol": "Insolation flux (Earth = 1)",
    "koi_srho": "Stellar density fitted from the transit (g/cm3)",
    # Stellar
    "koi_steff": "Stellar effective temperature (K)",
    "koi_slogg": "Stellar surface gravity, log10(cm/s2)",
    "koi_smet": "Stellar metallicity [Fe/H]",
    "koi_srad": "Stellar radius (solar radii)",
    "koi_smass": "Stellar mass (solar masses)",
    "koi_sage": "Stellar age (Gyr)",
    # Centroids
    "koi_fwm_stat_sig": "Significance of flux-weighted centroid motion during transit",
    "koi_fwm_sra": "Flux-weighted centroid right ascension (absolute)",
    "koi_fwm_sdec": "Flux-weighted centroid declination (absolute)",
    "koi_fwm_srao": "Flux-weighted centroid shift in right ascension, in transit vs out",
    "koi_fwm_sdeco": "Flux-weighted centroid shift in declination, in transit vs out",
    "koi_fwm_prao": "Implied source offset in right ascension from the centroid shift",
    "koi_fwm_pdeco": "Implied source offset in declination from the centroid shift",
    "koi_dicco_mra": "Difference-image offset from the out-of-transit centroid, RA",
    "koi_dicco_mdec": "Difference-image offset from the out-of-transit centroid, Dec",
    "koi_dicco_msky": "Difference-image offset from the out-of-transit centroid, on sky",
    "koi_dikco_mra": "Difference-image offset from the catalogue (KIC) position, RA",
    "koi_dikco_mdec": "Difference-image offset from the catalogue (KIC) position, Dec",
    "koi_dikco_msky": "Difference-image offset from the catalogue (KIC) position, on sky",
}

KEEP_REASONS = {
    "transit": "Transit fit or detection statistic: measured from the light curve before vetting.",
    "stellar": "Host-star property or brightness: independent of the vetting decision.",
    "centroid": (
        "Centroid diagnostic measured from pixel data before vetting. Used by `tabular`; "
        "left out of `tabular_lc` because the light-curve CNN has no pixel data."
    ),
    "multiplicity": (
        "Legitimate system context, but very strong (multi-KOI stars are rarely false "
        "positives). Allowed, off by default so models stay comparable with the CNN."
    ),
}

# Column-specific evidence, added to the category reason.
NOTES: dict[str, str] = {
    "koi_disposition": "This is the label itself.",
    "kepler_name": "Present for 2,746 of 2,748 CONFIRMED and 2 of 4,839 FALSE POSITIVE KOIs.",
    "koi_pdisposition": (
        "4,838 of 4,839 FALSE POSITIVE KOIs are pipeline FALSE POSITIVE; all but 9 CONFIRMED "
        "are pipeline CANDIDATE."
    ),
    "koi_score": "Median 1.0 for CONFIRMED, 0.0 for FALSE POSITIVE.",
    "koi_fpflag_nt": "Any flag is set for 4,744 of 4,839 FALSE POSITIVE and 18 of 2,748 CONFIRMED.",
    "koi_parm_prov": "Older deliveries are 98% FALSE POSITIVE (893 of 910).",
    "koi_quarters": "Records which quarters were searched, which depends on delivery history.",
}

_ERR = re.compile(r"^(?P<base>.+?)_err(?P<kind>[12]?)$")
_ERR_KIND = {"": "Uncertainty", "1": "Upper uncertainty", "2": "Lower uncertainty"}


def describe(column: str) -> str:
    """Plain-language meaning of a catalogue column (uncertainty columns are derived)."""
    if column in DESCRIPTIONS:
        return DESCRIPTIONS[column]
    match = _ERR.match(column)
    if match and match["base"] in DESCRIPTIONS:
        base = DESCRIPTIONS[match["base"]]
        return f"{_ERR_KIND[match['kind']]} of: {base[0].lower()}{base[1:]}"
    raise KeyError(f"no description for catalogue column {column!r}")


def column_table(
    columns: Iterable[str], enabled_groups: Iterable[str], audit: pd.DataFrame | None = None
) -> pd.DataFrame:
    """One row per column: meaning, decision, class and reason, plus univariate audit AUCs
    (label separability from the value and from missingness) where available.
    """
    enabled = set(enabled_groups)
    rows = []
    for col in columns:
        cls = classify_column(col)
        if cls is None:
            raise KeyError(f"unclassified catalogue column {col!r}")
        kind, name = cls.split(":", 1)
        if kind == "allowed":
            decision = "keep" if name in enabled else "keep (optional, off)"
            reason = KEEP_REASONS[name]
        else:
            decision = "drop"
            reason = DENY_CATEGORIES[name]
        note = NOTES.get(col) or (NOTES["koi_fpflag_nt"] if col.startswith("koi_fpflag_") else "")
        rows.append(
            {
                "column": col,
                "meaning": describe(col),
                "decision": decision,
                "class": cls,
                "reason": f"{reason} {note}".strip(),
            }
        )
    table = pd.DataFrame(rows).set_index("column")
    if audit is not None:
        table = table.join(audit[["auc_value", "auc_missing"]], how="left")
    return table
