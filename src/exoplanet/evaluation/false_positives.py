"""Which false positives each model rejects, and which planets it misses.

The Robovetter flags are denied as model features (they are vetting outputs) but are the
natural vocabulary for analysing errors: they say why a KOI was ruled a false positive.
A KOI can carry several flags, so per-flag rows overlap.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from exoplanet.data.labels import LABEL

FP_FLAGS = {
    "koi_fpflag_nt": "not transit-like",
    "koi_fpflag_ss": "stellar eclipse",
    "koi_fpflag_co": "centroid offset",
    "koi_fpflag_ec": "ephemeris match",
}


def fp_rejection_by_flag(scores: pd.DataFrame, thresholds: Mapping[str, float]) -> pd.DataFrame:
    """Share of false positives scored below each model's threshold, overall and per flag."""
    fp = scores[scores[LABEL] == 0]
    groups = {"all false positives": np.ones(len(fp), bool)}
    groups.update({name: fp[col].to_numpy() == 1 for col, name in FP_FLAGS.items()})
    rows = []
    for name, sel in groups.items():
        row = {"flag": name, "n": int(sel.sum())}
        for model, thr in thresholds.items():
            row[model] = (fp.loc[sel, model] < thr).mean() if sel.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("flag")


def recall_by_bin(
    scores: pd.DataFrame, thresholds: Mapping[str, float], column: str, n_bins: int = 4
) -> pd.DataFrame:
    """Recall on confirmed planets within quantile bins of ``column`` (e.g. SNR, period)."""
    planets = scores[scores[LABEL] == 1].dropna(subset=[column])
    bins = pd.qcut(planets[column], n_bins, duplicates="drop")
    rows = []
    for interval, group in planets.groupby(bins, observed=True):
        row = {column: f"{interval.left:.3g} - {interval.right:.3g}", "n": len(group)}
        for model, thr in thresholds.items():
            row[model] = (group[model] >= thr).mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index(column)


def worst_errors(
    scores: pd.DataFrame, model: str, n: int, columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Most confident mistakes of ``model``: highest-scored false positives and
    lowest-scored confirmed planets, with the given context columns.
    """
    fp = scores[scores[LABEL] == 0].nlargest(n, model)
    planets = scores[scores[LABEL] == 1].nsmallest(n, model)
    return fp[[model, *columns]], planets[[model, *columns]]
