"""Precision-recall curve figure for model comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

from exoplanet.data.labels import LABEL  # noqa: E402

# Categorical slots 1-3 of the reference palette (blue, orange, aqua), validated for all
# pairs under colour-vision deficiency; series are assigned in this fixed order.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SURFACE, TEXT, TEXT_MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a64", "#e4e3de"


def plot_pr_curves(
    scores: pd.DataFrame,
    models: Sequence[str],
    operating_points: Mapping[str, Mapping[str, float]],
    title: str,
    path: Path,
) -> Path:
    """Full and zoomed (high-precision) PR curves, with each model's operating point."""
    if len(models) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} models per figure")
    y = scores[LABEL].to_numpy()
    prevalence = y.mean()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    zoom = {"xlim": (0.8, 1.0), "ylim": (0.8, 1.005)}
    for ax, limits, subtitle in [
        (axes[0], {"xlim": (0, 1.0), "ylim": (0.35, 1.02)}, "Full range"),
        (axes[1], zoom, "High-precision region"),
    ]:
        ax.set_facecolor(SURFACE)
        ax.axhline(prevalence, color=TEXT_MUTED, lw=1, ls=(0, (4, 3)))
        for model, color in zip(models, SERIES, strict=False):
            precision, recall, _ = precision_recall_curve(y, scores[model])
            ax.plot(recall, precision, color=color, lw=2, label=model, solid_capstyle="round")
            op = operating_points.get(model)
            if op:
                ax.plot(
                    op["recall"],
                    op["precision"],
                    "o",
                    ms=8,
                    color=color,
                    markeredgecolor=SURFACE,
                    markeredgewidth=2,
                    zorder=3,
                )
                if ax is axes[0]:
                    ax.annotate(
                        model,
                        (op["recall"], op["precision"]),
                        xytext=(-8, -14),
                        textcoords="offset points",
                        ha="right",
                        color=TEXT,
                        fontsize=8,
                    )
        ax.set(**limits)
        ax.set_title(subtitle, color=TEXT, fontsize=10, loc="left")
        ax.set_xlabel("Recall (confirmed planets recovered)", color=TEXT_MUTED, fontsize=9)
        ax.grid(color=GRID, lw=0.8)
        ax.tick_params(colors=TEXT_MUTED, labelsize=8, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0].set_ylabel("Precision", color=TEXT_MUTED, fontsize=9)
    axes[0].annotate(
        f"random classifier ({prevalence:.2f})",
        (0.02, prevalence),
        xytext=(0, 4),
        textcoords="offset points",
        color=TEXT_MUTED,
        fontsize=8,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    marker = plt.Line2D([], [], marker="o", ls="", color=TEXT_MUTED, ms=7)
    fig.legend(
        [*handles, marker],
        [*labels, "operating point (threshold set on dev folds)"],
        loc="lower center",
        ncol=len(labels) + 1,
        frameon=False,
        fontsize=9,
        labelcolor=TEXT,
    )
    fig.suptitle(title, color=TEXT, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return path
