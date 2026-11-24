"""Shared look for every scene: colours, fonts, type sizes and drawing helpers.

Layout units. Every figure is 16 inches wide with dpi = width / 16, so a size in points
looks the same at 1920x1080 (dpi 120) and the 960x540 draft (dpi 60). Lay out in figure
fractions (0-1); keep the bottom ``CAPTION_BAND`` of the height clear, the engine draws the
caption there after the scene draws.

Timing. Scenes get ``t`` in seconds and express timing as fractions of the scene:
``f = t / ctx.duration``; ``ease(progress(f, 0.2, 0.5))`` rises smoothly from 0 to 1 while
f goes from 0.2 to 0.5.

Importing this module sets matplotlib's rcParams (dark background, fonts, sizes), so plain
``fig.add_axes`` / ``ax.plot`` calls already match the style.
"""

from __future__ import annotations

from functools import lru_cache

import matplotlib
import numpy as np
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg, RendererAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, Rectangle

# --- colours (defined once; use these names, never literal hex in scenes) -----------
BG = "#070b1a"         # night sky
FG = "#e8ecf6"         # main text
MUTED = "#8b93ab"      # axes, secondary text
C_PLANET = "#46c2f0"   # confirmed planet (sky blue)
C_FALSE = "#ff8a3d"    # false positive (orange); blue/orange is colour-blind safe
C_ACCENT = "#ffd166"   # highlights, commands, the one thing to look at
C_WARN = "#ff4f6d"     # warnings only (the leaked score); never a class colour
TERM_BG = "#0e1424"    # terminal window body


def _first_font(*names: str) -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    return next(n for n in names if n in available)


FONT = _first_font("Segoe UI", "DejaVu Sans")        # DejaVu ships with matplotlib
MONO = _first_font("Consolas", "DejaVu Sans Mono")

# --- type sizes in points (same look at every resolution) ----------------------------
TITLE_SIZE = 56     # title / end card
HEADING_SIZE = 38   # scene heading (see heading())
BODY_SIZE = 28      # statements, big labels
LABEL_SIZE = 22     # axis labels, legends, ticks
CAPTION_SIZE = 26
TERM_SIZE = 20

# --- layout (figure fractions) --------------------------------------------------------
FIG_W_IN = 16.0
CAPTION_BAND = 0.15  # bottom band reserved for captions: scenes draw above this
MARGIN = 0.05        # side margin
HEADING_Y = 0.94     # top of the scene heading

matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": list(dict.fromkeys([FONT, "DejaVu Sans"])),
    "font.monospace": list(dict.fromkeys([MONO, "DejaVu Sans Mono"])),
    "font.size": LABEL_SIZE,
    "text.color": FG,
    "figure.facecolor": BG,
    "axes.facecolor": "none",
    "axes.edgecolor": MUTED,
    "axes.labelcolor": FG,
    "axes.labelsize": LABEL_SIZE,
    "axes.titlesize": BODY_SIZE,
    "axes.titlecolor": FG,
    "axes.linewidth": 1.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": MUTED,
    "grid.alpha": 0.25,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": FG,
    "ytick.labelcolor": FG,
    "xtick.labelsize": LABEL_SIZE - 4,
    "ytick.labelsize": LABEL_SIZE - 4,
    "legend.fontsize": LABEL_SIZE - 2,
    "legend.frameon": False,
    "lines.linewidth": 3.0,
    "lines.solid_capstyle": "round",
    "axes.prop_cycle": matplotlib.cycler(color=[C_PLANET, C_FALSE, C_ACCENT, FG]),
})


def new_figure(width: int, height: int) -> Figure:
    """A figure of exactly width x height pixels on an Agg canvas (no GUI, reusable)."""
    dpi = width / FIG_W_IN
    fig = Figure(figsize=(FIG_W_IN, height / dpi), dpi=dpi, facecolor=BG)
    FigureCanvasAgg(fig)
    return fig


def ease(x: float) -> float:
    """Smoothstep: 0 below 0, 1 above 1, slow-fast-slow in between."""
    x = min(max(float(x), 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def progress(t: float, start: float, end: float) -> float:
    """Linear 0-1 position of t between start and end (same units), clamped. Wrap in ease()."""
    if end <= start:
        return 1.0 if t >= end else 0.0
    return min(max((t - start) / (end - start), 0.0), 1.0)


def heading(fig: Figure, text: str, color: str = FG) -> None:
    """The scene heading, top-left, in the same place in every scene."""
    fig.text(MARGIN, HEADING_Y, text, ha="left", va="top", size=HEADING_SIZE,
             weight="semibold", color=color)


# --- captions -------------------------------------------------------------------------
_CAPTION_ONE_LINE = 0.66   # a caption narrower than this fraction of the width stays one line
_CAPTION_MAX_W = 0.90      # no caption line is wider than this
_CAPTION_MIN_SIZE = 14


@lru_cache(maxsize=512)
def _caption_layout(text: str, width_px: float, dpi: float) -> tuple[str, float]:
    """(text with at most one line break, font size) so that the caption fits the band."""
    renderer = RendererAgg(8, 8, dpi)
    words = text.split()
    one = " ".join(words)
    max_w = _CAPTION_MAX_W * width_px

    def width(s: str, size: float) -> float:
        prop = FontProperties(family="sans-serif", size=size)
        return renderer.get_text_width_height_descent(s, prop, ismath=False)[0]

    def layout(size: float) -> tuple[str, float]:
        """Best layout at this size and its widest line."""
        if width(one, size) <= _CAPTION_ONE_LINE * width_px or len(words) == 1:
            return one, width(one, size)
        splits = [(" ".join(words[:i]), " ".join(words[i:])) for i in range(1, len(words))]
        a, b = min(splits, key=lambda p: max(width(p[0], size), width(p[1], size)))
        return f"{a}\n{b}", max(width(a, size), width(b, size))

    size = float(CAPTION_SIZE)
    laid_out, widest = layout(size)
    while widest > max_w and size > _CAPTION_MIN_SIZE:  # too wide: shrink rather than overflow
        size -= 1.0
        laid_out, widest = layout(size)
    return laid_out, size


def draw_caption(fig: Figure, text: str) -> None:
    """Draw text centred in the bottom band, wrapped to at most two lines, shrunk if needed."""
    if not text or not text.strip():
        return
    laid_out, size = _caption_layout(text, fig.bbox.width, fig.dpi)
    fig.add_artist(Rectangle((0, 0), 1, CAPTION_BAND, transform=fig.transFigure,
                             facecolor=BG, alpha=0.82, edgecolor="none", gid="caption_band"))
    fig.text(0.5, CAPTION_BAND / 2, laid_out, ha="center", va="center", multialignment="center",
             size=size, color=FG, linespacing=1.25, gid="caption")


# --- star field -----------------------------------------------------------------------
@lru_cache(maxsize=16)
def _stars(seed: int, n: int = 320):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 1, (n, 2))
    size = 0.4 + 9.0 * rng.pareto(2.5, n).clip(0, 1.5)  # mostly faint, a few bright
    base = rng.uniform(0.35, 0.95, n)
    rate = rng.uniform(0.15, 0.6, n)   # twinkles per second
    phase = rng.uniform(0, 2 * np.pi, n)
    tint = rng.choice(3, n, p=[0.7, 0.18, 0.12])
    rgb = np.array([[1.0, 1.0, 1.0], [0.75, 0.85, 1.0], [1.0, 0.9, 0.75]])[tint]
    return xy, size, base, rate, phase, rgb


def starfield(ax, seed: int, t: float) -> None:
    """Fill ax (axes coordinates 0-1, axis hidden) with twinkling stars; t is seconds."""
    xy, size, base, rate, phase, rgb = _stars(seed)
    alpha = base * (0.7 + 0.3 * np.sin(2 * np.pi * rate * t + phase))
    ax.scatter(xy[:, 0], xy[:, 1], s=size, c=np.column_stack([rgb, alpha]), linewidths=0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()


# --- terminal -------------------------------------------------------------------------
_COMMAND_PREFIXES = ("$ ", "> ", "PS ")


def terminal(ax, lines: list[str], chars_shown: int, *, title: str = "PowerShell",
             fontsize: float = TERM_SIZE) -> None:
    """Draw a terminal window filling ax, with the first chars_shown characters typed.

    Each line counts its length plus one (the newline). Lines starting with "$ ", "> " or
    "PS " are commands and drawn in C_ACCENT. When the typed text is taller than the window,
    it scrolls: the last lines that fit are shown. Long lines are clipped at the window edge.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    px_per_pt = ax.figure.dpi / 72
    h_px = ax.bbox.height
    bar = fontsize * 1.9 * px_per_pt / h_px
    line_h = fontsize * 1.4 * px_per_pt / h_px
    pad_x = fontsize * 1.0 * px_per_pt / ax.bbox.width

    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.015",
                                facecolor=TERM_BG, edgecolor=MUTED, linewidth=1.2,
                                mutation_aspect=ax.bbox.width / h_px))
    ax.add_patch(Rectangle((0, 1 - bar), 1, bar, facecolor=MUTED, alpha=0.18, edgecolor="none"))
    for i, colour in enumerate(["#ff5f57", "#febc2e", "#28c840"]):
        ax.scatter([pad_x + i * pad_x * 1.2], [1 - bar / 2], s=(fontsize * 0.6) ** 2,
                   color=colour, linewidths=0)
    ax.text(0.5, 1 - bar / 2, title, ha="center", va="center", size=fontsize * 0.8,
            color=MUTED)

    typed: list[str] = []
    budget = max(int(chars_shown), 0)
    for line in lines:
        if budget <= 0:
            break
        typed.append(line[:budget])
        budget -= len(line) + 1
    done = budget > 0
    if not typed:
        typed = [""]
    if not done:
        typed[-1] += "█"  # block cursor while typing

    n_fit = max(int((1 - bar - line_h * 0.5) // line_h), 1)
    for row, line in enumerate(typed[-n_fit:]):
        colour = C_ACCENT if line.startswith(_COMMAND_PREFIXES) else FG
        ax.text(pad_x, 1 - bar - line_h * (row + 0.75), line, ha="left", va="center",
                family="monospace", size=fontsize, color=colour, clip_on=True)
