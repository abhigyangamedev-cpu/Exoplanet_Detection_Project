"""Allowed catalogue feature groups and leakage-checked feature selection."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from exoplanet.features.leakage import assert_no_leakage, deny_category

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    # Transit fit and detection statistics (inputs to vetting, not outputs of it).
    "transit": (
        "koi_period",
        "koi_duration",
        "koi_depth",
        "koi_ror",
        "koi_dor",
        "koi_impact",
        "koi_incl",
        "koi_model_snr",
        "koi_num_transits",
        "koi_max_sngle_ev",
        "koi_max_mult_ev",
        "koi_bin_oedp_sig",
        "koi_prad",
        "koi_sma",
        "koi_teq",
        "koi_insol",
        "koi_srho",
    ),
    # Host-star properties and photometry.
    "stellar": (
        "koi_steff",
        "koi_slogg",
        "koi_smet",
        "koi_srad",
        "koi_smass",
        "koi_kepmag",
        "koi_gmag",
        "koi_rmag",
        "koi_imag",
        "koi_zmag",
        "koi_jmag",
        "koi_hmag",
        "koi_kmag",
    ),
    # Centroid-offset diagnostics (the light-curve CNN does not see these).
    "centroid": (
        "koi_fwm_stat_sig",
        "koi_fwm_srao",
        "koi_fwm_sdeco",
        "koi_fwm_prao",
        "koi_fwm_pdeco",
        "koi_dicco_mra",
        "koi_dicco_mdec",
        "koi_dicco_msky",
        "koi_dikco_mra",
        "koi_dikco_mdec",
        "koi_dikco_msky",
    ),
    # System context: legitimate but very strong (multi-KOI stars are rarely false
    # positives). Off by default so the baseline is comparable with the light-curve CNN.
    "multiplicity": ("koi_count", "koi_tce_plnt_num"),
}


def classify_column(column: str) -> str | None:
    """``"allowed:<group>"``, ``"denied:<category>"``, or None if unclassified."""
    for group, cols in FEATURE_GROUPS.items():
        if column in cols:
            return f"allowed:{group}"
    category = deny_category(column)
    return f"denied:{category}" if category else None


def unclassified_columns(columns: Iterable[str]) -> list[str]:
    """Columns covered by neither an allowed group nor the deny policy."""
    return [c for c in columns if classify_column(c) is None]


def feature_columns(groups: Iterable[str]) -> list[str]:
    unknown = set(groups) - FEATURE_GROUPS.keys()
    if unknown:
        raise KeyError(f"Unknown feature groups: {sorted(unknown)}")
    return [c for g in groups for c in FEATURE_GROUPS[g]]


def select_features(df: pd.DataFrame, groups: Iterable[str]) -> pd.DataFrame:
    """Leakage-checked numeric feature matrix for the requested groups."""
    columns = feature_columns(groups)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Feature columns not in catalogue: {missing}")
    assert_no_leakage(columns)
    return df[columns].apply(pd.to_numeric, errors="coerce")
