"""Descriptive inspection of the KOI catalogue: classes, host-star structure, missingness."""

from __future__ import annotations

from typing import Any

import pandas as pd

DISPOSITION = "koi_disposition"
PIPELINE_DISPOSITION = "koi_pdisposition"
STAR = "kepid"

# Physical/transit quantities summarised per class.
KEY_NUMERIC = (
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_model_snr",
    "koi_steff",
    "koi_srad",
    "koi_kepmag",
)


def summarise_catalogue(
    df: pd.DataFrame,
    labelled: tuple[str, ...] = ("CONFIRMED", "FALSE POSITIVE"),
) -> dict[str, Any]:
    """Compute the catalogue summary used by the inspection report."""
    kois_per_star = df.groupby(STAR).size()
    multiplicity = df[STAR].map(kois_per_star)
    is_labelled = df[DISPOSITION].isin(labelled)
    lab = df[is_labelled]

    # Stars hosting both a confirmed planet and a false positive: these make grouped
    # splitting matter, since the two labels share stellar features and systematics.
    labels_per_star = lab.groupby(STAR)[DISPOSITION].nunique()

    multiplicity_by_class = pd.crosstab(
        multiplicity.where(multiplicity < 4, 4).rename("kois_on_star"), df[DISPOSITION]
    ).rename(index={4: "4+"})
    pos, neg = labelled
    if pos in multiplicity_by_class and neg in multiplicity_by_class:
        multiplicity_by_class[f"{pos} fraction"] = multiplicity_by_class[pos] / (
            multiplicity_by_class[pos] + multiplicity_by_class[neg]
        )

    missing = df.isna().mean().sort_values(ascending=False)
    numeric_present = [c for c in KEY_NUMERIC if c in df.columns]

    return {
        "n_kois": len(df),
        "n_columns": df.shape[1],
        "n_stars": int(kois_per_star.size),
        "disposition_counts": df[DISPOSITION].value_counts(dropna=False).to_dict(),
        "disposition_vs_pipeline": (
            pd.crosstab(df[DISPOSITION], df[PIPELINE_DISPOSITION])
            if PIPELINE_DISPOSITION in df.columns
            else None
        ),
        "n_labelled": int(is_labelled.sum()),
        "n_labelled_stars": int(lab[STAR].nunique()),
        "positive_fraction": float((lab[DISPOSITION] == labelled[0]).mean()),
        "kois_per_star": kois_per_star.value_counts().sort_index().to_dict(),
        "multiplicity_by_class": multiplicity_by_class,
        "n_mixed_label_stars": int((labels_per_star > 1).sum()),
        "n_columns_all_missing": int((missing == 1.0).sum()),
        "missing_fraction": missing[missing > 0],
        "numeric_by_class": (
            lab.groupby(DISPOSITION)[numeric_present].median().T if numeric_present else None
        ),
        "dtype_counts": df.dtypes.astype(str).value_counts().to_dict(),
    }


def _table(df: pd.DataFrame, floatfmt: str = ".4g") -> str:
    """Render a DataFrame as a Markdown table without the optional ``tabulate`` dependency."""
    header = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    # itertuples keeps per-column dtypes (iterrows would upcast integer counts to float).
    for idx, *values in df.itertuples(name=None):
        cells = [format(v, floatfmt) if isinstance(v, float) else str(v) for v in values]
        lines.append("| " + " | ".join([str(idx), *cells]) + " |")
    return "\n".join(lines)


def render_report(summary: dict[str, Any], metadata: dict[str, Any] | None = None) -> str:
    """Render ``summarise_catalogue`` output as a Markdown report."""
    s = summary
    out = ["# KOI Catalogue Inspection", ""]
    if metadata:
        out += [
            f"- Source: `{metadata['source']}`",
            f"- Query: `{metadata['query']}`",
            f"- Retrieved (UTC): {metadata['retrieved_utc']}",
            f"- SHA-256: `{metadata['sha256']}`",
            "",
        ]

    out += [
        "## Overview",
        "",
        f"- KOIs: {s['n_kois']:,} across {s['n_stars']:,} host stars ({s['n_columns']} columns)",
        f"- Labelled (CONFIRMED + FALSE POSITIVE): {s['n_labelled']:,} KOIs on "
        f"{s['n_labelled_stars']:,} stars",
        f"- Positive (CONFIRMED) fraction among labelled: {s['positive_fraction']:.1%}",
        f"- Stars hosting both a CONFIRMED and a FALSE POSITIVE KOI: {s['n_mixed_label_stars']:,}",
        "",
        "## Dispositions",
        "",
        _table(
            pd.DataFrame.from_dict(
                s["disposition_counts"], orient="index", columns=["n"]
            ).rename_axis("koi_disposition")
        ),
        "",
    ]
    if s["disposition_vs_pipeline"] is not None:
        out += [
            "Archive disposition (rows) vs. Kepler pipeline disposition (columns). "
            "`koi_pdisposition` is a vetting output and must not be used as a feature.",
            "",
            _table(s["disposition_vs_pipeline"]),
            "",
        ]

    kps = pd.DataFrame.from_dict(s["kois_per_star"], orient="index", columns=["n_stars"])
    out += [
        "## Host-star structure",
        "",
        _table(kps.rename_axis("kois_on_star")),
        "",
        "KOIs by number of KOIs on the same host star, with the confirmed fraction among "
        "labelled KOIs at each multiplicity.",
        "",
        _table(s["multiplicity_by_class"]),
        "",
    ]

    if s["numeric_by_class"] is not None:
        out += [
            "## Median key quantities by class (labelled KOIs)",
            "",
            _table(s["numeric_by_class"].rename_axis("column")),
            "",
        ]

    miss = s["missing_fraction"].to_frame("missing_fraction").rename_axis("column")
    out += [
        "## Missing values",
        "",
        f"{len(miss)} of {s['n_columns']} columns have missing values; "
        f"{s['n_columns_all_missing']} are entirely empty.",
        "",
        _table(miss, floatfmt=".1%"),
        "",
        "## Column types",
        "",
        ", ".join(f"{k}: {v}" for k, v in s["dtype_counts"].items()),
        "",
    ]
    return "\n".join(out)
