"""Feature leakage policy for the KOI catalogue, and an empirical univariate audit.

Every catalogue column is either in an allowed feature group (``exoplanet.features.tabular``)
or denied here with a category explaining why. ``assert_no_leakage`` is called on every
feature matrix before it reaches a model.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

DENY_CATEGORIES = {
    "label": "The target, or a vetting decision/score derived from it.",
    "identifier": "Identifier or link; carries no physical information.",
    "provenance": (
        "Pipeline delivery or fit method. Older deliveries are ~98% FALSE POSITIVE, so "
        "provenance encodes the label."
    ),
    "uncertainty": (
        "Parameter uncertainty. Uncertainties separate the classes better than the values "
        "themselves, plausibly because planet hosts received follow-up characterisation: "
        "likely post-detection information."
    ),
    "epoch": "Transit epoch: an arbitrary reference time, not a physical property.",
    "sky_position": ("Absolute sky position; risks memorising field- or CCD-specific systematics."),
    "constant": "Constant or redundant with other features.",
    "empty": "Entirely empty in the cumulative table.",
}

DENIED: dict[str, str] = {
    # Target and vetting outputs.
    "koi_disposition": "label",
    "koi_pdisposition": "label",
    "koi_score": "label",
    "koi_fpflag_nt": "label",
    "koi_fpflag_ss": "label",
    "koi_fpflag_co": "label",
    "koi_fpflag_ec": "label",
    "koi_vet_stat": "label",
    "koi_vet_date": "label",
    "koi_comment": "label",
    "koi_disp_prov": "label",
    "kepler_name": "label",  # only assigned once a planet is confirmed
    # Identifiers and links.
    "kepid": "identifier",
    "kepoi_name": "identifier",
    "ra_str": "identifier",
    "dec_str": "identifier",
    "koi_datalink_dvs": "identifier",
    "koi_datalink_dvr": "identifier",
    # Provenance.
    "koi_delivname": "provenance",
    "koi_tce_delivname": "provenance",
    "koi_parm_prov": "provenance",
    "koi_sparprov": "provenance",
    "koi_fittype": "provenance",
    "koi_limbdark_mod": "provenance",
    "koi_trans_mod": "provenance",
    "koi_quarters": "provenance",
    # Epoch.
    "koi_time0bk": "epoch",
    "koi_time0": "epoch",
    # Sky position (flux-weighted centroid RA/Dec are absolute positions, not offsets).
    "ra": "sky_position",
    "dec": "sky_position",
    "koi_fwm_sra": "sky_position",
    "koi_fwm_sdec": "sky_position",
    # Constant or redundant with stellar parameters.
    "koi_eccen": "constant",
    "koi_ldm_coeff1": "constant",
    "koi_ldm_coeff2": "constant",
    "koi_ldm_coeff3": "constant",
    "koi_ldm_coeff4": "constant",
    # Entirely empty.
    "koi_model_dof": "empty",
    "koi_model_chisq": "empty",
    "koi_longp": "empty",
    "koi_ingress": "empty",
    "koi_sage": "empty",
}

_UNCERTAINTY = re.compile(r"_err[12]?$")


def deny_category(column: str) -> str | None:
    """Return the deny category for ``column``, or None if it is not denied."""
    if column in DENIED:
        return DENIED[column]
    if _UNCERTAINTY.search(column):
        return "uncertainty"
    return None


class LeakageError(ValueError):
    """A denied column reached a feature matrix."""


def assert_no_leakage(columns: Iterable[str]) -> None:
    """Raise ``LeakageError`` if any of ``columns`` is denied."""
    leaked = {c: deny_category(c) for c in columns if deny_category(c) is not None}
    if leaked:
        raise LeakageError(f"Denied columns in feature set: {leaked}")


def univariate_audit(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Per-column separability of the label, from the value and from missingness alone.

    AUCs are symmetrised (max(auc, 1 - auc)), so 0.5 means no signal. Values are
    median-imputed for scoring only.
    """
    y = np.asarray(y)
    rows = []
    for col in X.columns:
        x = pd.to_numeric(X[col], errors="coerce")
        missing = x.isna()
        rows.append(
            {
                "column": col,
                "auc_value": _symmetric_auc(y, x.fillna(x.median())) if x.notna().any() else 0.5,
                "auc_missing": _symmetric_auc(y, missing.astype(float)),
                "missing_fraction": float(missing.mean()),
            }
        )
    return pd.DataFrame(rows).set_index("column").sort_values("auc_value", ascending=False)


def flag_suspicious(
    audit: pd.DataFrame, max_value_auc: float, max_missing_auc: float
) -> pd.DataFrame:
    """Rows of ``univariate_audit`` output exceeding either threshold."""
    return audit[(audit["auc_value"] > max_value_auc) | (audit["auc_missing"] > max_missing_auc)]


def _symmetric_auc(y: np.ndarray, score: pd.Series) -> float:
    if score.nunique() < 2:
        return 0.5
    auc = roc_auc_score(y, score)
    return float(max(auc, 1.0 - auc))
