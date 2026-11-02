"""Label mapping and modelling-sample selection."""

from __future__ import annotations

from typing import Any

import pandas as pd

DISPOSITION = "koi_disposition"
PARAMETER_PROVENANCE = "koi_parm_prov"
LABEL = "label"


def label_catalogue(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Return the modelling sample with an integer ``label`` column (1 = positive).

    Excluded dispositions (e.g. CANDIDATE) are dropped, and, if ``sample.parameter_provenance``
    is set, so are KOIs whose parameters come from other pipeline deliveries.
    """
    labels = config["labels"]
    positive, negative, excluded = labels["positive"], labels["negative"], labels["exclude"]

    known = {positive, negative, *excluded}
    unexpected = set(df[DISPOSITION].dropna().unique()) - known
    if unexpected or df[DISPOSITION].isna().any():
        raise ValueError(
            f"Unmapped dispositions: {sorted(unexpected)}"
            + (" and missing values" if df[DISPOSITION].isna().any() else "")
        )

    out = df[df[DISPOSITION].isin([positive, negative])]
    provenance = config.get("sample", {}).get("parameter_provenance")
    if provenance:
        out = out[out[PARAMETER_PROVENANCE].isin(provenance)]

    out = out.copy()
    out[LABEL] = (out[DISPOSITION] == positive).astype(int)
    return out.reset_index(drop=True)
