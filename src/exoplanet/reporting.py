"""Shared helpers for generated Markdown reports."""

from __future__ import annotations

import pandas as pd


def markdown_table(df: pd.DataFrame, floatfmt: str = ".4g") -> str:
    """Render a DataFrame as a Markdown table without the optional ``tabulate`` dependency."""
    header = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    # itertuples keeps per-column dtypes (iterrows would upcast integer counts to float).
    for idx, *values in df.itertuples(name=None):
        cells = [format(v, floatfmt) if isinstance(v, float) else str(v) for v in values]
        lines.append("| " + " | ".join([str(idx), *cells]) + " |")
    return "\n".join(lines)
