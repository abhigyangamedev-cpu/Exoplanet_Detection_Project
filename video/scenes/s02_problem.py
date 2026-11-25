"""Scene 2, the problem: what a transit is, then the judged Kepler signals.

Part 1: a planet disc crosses a star while the real local view of K00020.01 (its folded,
binned transit) draws in step with it. Part 2: one dot per judged signal ({n_kois}) gathers
into a grey cloud; the confirmed planets then fly to a blue cloud and the false positives to
an orange one, each counter counting the dots that have arrived.
"""

from __future__ import annotations

import numpy as np
import style
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle

from scenes import cue_fraction

# layout in "inch" units of a 16 x 9 overlay axes
STAR_XY, STAR_R = (4.3, 4.75), 2.0
TRACE_X, TRACE_W = 8.2, 6.9          # flux trace box
TRACE_BASE, TRACE_H = 6.1, 3.0       # baseline y and the full dip height below it
GREY_XY = (8.0, 4.3)
CONF_XY, FALSE_XY = (3.6, 3.75), (12.0, 3.75)
BINARY_XY = (14.9, 4.3)
BINARY_PERIOD_S = 2.5                # seconds per orbit of the eclipsing-binary icon
SPACING = 0.028                      # dot cloud: radius = SPACING * sqrt(index)
DOT_S = 7.0                          # dot size, points^2
FLIGHT = 0.045                       # each dot's flight, fraction of the scene
DIP_START = 0.02                     # fraction of the full dip that counts as "falling"
CHORD_DY = 0.25                      # the planet crosses this many star radii below centre
COUNT_LEAD = 0.09                    # a class's dots start flying this long before its number


def _spiral(n: int, centre) -> np.ndarray:
    """n points packed in a disc (sunflower spiral); the first points are the innermost."""
    i = np.arange(n)
    r = SPACING * np.sqrt(i + 0.5)
    a = i * np.pi * (3.0 - np.sqrt(5.0))
    return np.column_stack([centre[0] + r * np.cos(a), centre[1] + r * np.sin(a)])


def dip_edges(lv: np.ndarray, start: float = DIP_START) -> tuple[int, int]:
    """(first, last) local-view bin of the dip: the run of bins around the minimum that lie
    more than ``start`` of the full depth below the baseline. The planet touches the star's
    edge at the first and leaves it at the last (first and last contact)."""
    lv = np.asarray(lv, float)
    centre = int(np.argmin(lv))
    inside = lv < start * lv.min()
    lo = centre - int(np.argmin(inside[centre::-1])) + 1
    hi = centre + int(np.argmin(inside[centre:])) - 1
    return lo, hi


def planet_x(k: float, lo: int, hi: int, sx: float, star_r: float, planet_r: float) -> float:
    """Planet centre x when the trace cursor is at local-view bin k: the planet's disc
    touches the star's limb (on its chord, CHORD_DY star radii below centre) at bins lo and
    hi and is centred on the star half way between."""
    reach = np.sqrt((star_r + planet_r) ** 2 - (CHORD_DY * star_r) ** 2)
    return sx - reach + 2 * reach * (k - lo) / max(hi - lo, 1)


def _transit(ctx):
    star = ctx.load("star_planet.npz")
    lv = star["local_view"]
    lo, hi = dip_edges(lv)
    return lv, lo, hi, float(star["depth"])


def build(ctx):
    n = ctx.numbers
    n_kois, n_conf, n_false = n["n_kois"], n["n_confirmed"], n["n_false_positive"]
    lv, lo, hi, depth = _transit(ctx)
    planet_r = STAR_R * np.sqrt(depth)                 # true size ratio from the dip depth
    trace_x = TRACE_X + TRACE_W * np.arange(len(lv)) / (len(lv) - 1)
    trace_y = TRACE_BASE + TRACE_H * lv / -lv.min()

    # dots: grey slots, class targets, departure order
    grey = _spiral(n_kois, GREY_XY)
    is_conf = np.zeros(n_kois, bool)
    is_conf[np.random.default_rng(1).permutation(n_kois)[:n_conf]] = True
    target = np.empty_like(grey)
    target[is_conf] = _spiral(n_conf, CONF_XY)
    target[~is_conf] = _spiral(n_false, FALSE_XY)
    rank = np.empty(n_kois)                            # 0-1 position in its class's queue
    rank[is_conf] = np.arange(n_conf) / max(n_conf - 1, 1)
    rank[~is_conf] = np.arange(n_false) / max(n_false - 1, 1)
    grey_rgb = np.array(to_rgb(style.MUTED))
    class_rgb = np.where(is_conf[:, None], to_rgb(style.C_PLANET), to_rgb(style.C_FALSE))
    lift = -0.4                                        # flights arc downwards, under the labels

    # timing, tied to the spoken words
    f_tiny = cue_fraction(ctx, "tiny", 0.08)
    f_transit = cue_fraction(ctx, "called", 0.25)
    f_use = cue_fraction(ctx, "We use", 0.3)
    f_conf = cue_fraction(ctx, f"{n_conf:,}", 0.55)
    f_false = cue_fraction(ctx, f"{n_false:,}", 0.72)
    f_binary = cue_fraction(ctx, "two stars", 0.86)
    f_cross0, f_cross1 = 0.01, f_transit
    # the total has arrived as it is spoken ("We use <n_kois> ...")
    f_gather1 = max(cue_fraction(ctx, f"{n_kois:,}", f_use + 0.03), f_use + 0.01)
    # each class's dots fly so that its counter has arrived as its number is spoken
    conf_span = (max(f_conf - COUNT_LEAD, f_gather1 + 0.01), f_conf)
    false_span = (max(f_false - COUNT_LEAD, conf_span[1] + 0.005), f_false)
    f_left = false_span[1]                             # the grey cloud is empty
    depart = np.where(is_conf, conf_span[0] + (conf_span[1] - FLIGHT - conf_span[0]) * rank,
                      false_span[0] + (false_span[1] - FLIGHT - false_span[0]) * rank)

    def draw(fig, t):
        f = t / ctx.duration
        style.starfield(fig.add_axes((0, 0, 1, 1)), 2, t)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 9)
        ax.set_axis_off()

        a1 = style.ease(style.progress(f, 0.0, 0.04)) * (1 - style.ease(
            style.progress(f, f_use - 0.03, f_use + 0.01)))
        a2 = style.ease(style.progress(f, f_use - 0.01, f_use + 0.03))
        if a1 > 0:
            _draw_transit(ax, fig, f, a1)
        if a2 > 0:
            _draw_counts(ax, fig, f, a2)

    def _draw_transit(ax, fig, f, a):
        fig.text(style.MARGIN, style.HEADING_Y, "A transit", ha="left", va="top",
                 size=style.HEADING_SIZE, weight="semibold", color=style.FG, alpha=a)
        sweep = style.progress(f, f_cross0, f_cross1)
        k = sweep * (len(lv) - 1)                       # cursor, in local-view bins
        sx, sy = STAR_XY
        for scale in np.linspace(1.3, 1.02, 8):
            ax.add_patch(Circle((sx, sy), STAR_R * scale, color=style.C_ACCENT,
                                alpha=0.06 * a, lw=0))
        ax.add_patch(Circle((sx, sy), STAR_R, color=style.C_ACCENT, alpha=a, lw=0,
                            gid="star_disc"))
        for scale in np.linspace(0.95, 0.35, 7):
            ax.add_patch(Circle((sx, sy), STAR_R * scale, color=style.FG, alpha=0.07 * a,
                                lw=0))
        # first contact as the trace begins to fall (bin lo), last as it is back (bin hi)
        px = planet_x(k, lo, hi, sx, STAR_R, planet_r)
        py = sy - CHORD_DY * STAR_R
        a_p = a * (1 - style.ease(style.progress(abs(px - sx), 1.5 * STAR_R, 1.85 * STAR_R)))
        if a_p > 0:
            ax.add_patch(Circle((px, py), planet_r, facecolor=style.BG,
                                edgecolor=style.MUTED, lw=1.5, alpha=a_p, zorder=3,
                                gid="planet"))
            ax.text(px, py - planet_r - 0.3, "planet", ha="center",
                    va="center", size=style.LABEL_SIZE, color=style.MUTED, alpha=a_p,
                    zorder=3)

        # the trace, drawn up to the cursor
        ax.plot([TRACE_X, TRACE_X], [TRACE_BASE - TRACE_H - 0.3, TRACE_BASE + 0.6],
                color=style.MUTED, lw=1.5, alpha=a)
        ax.plot([TRACE_X, TRACE_X + TRACE_W], [TRACE_BASE - TRACE_H - 0.3] * 2,
                color=style.MUTED, lw=1.5, alpha=a)
        ax.text(TRACE_X - 0.25, TRACE_BASE - TRACE_H / 2, "brightness", rotation=90,
                ha="center", va="center", size=style.LABEL_SIZE, color=style.FG, alpha=a)
        ax.text(TRACE_X + TRACE_W, TRACE_BASE - TRACE_H - 0.6, "time", ha="right",
                va="center", size=style.LABEL_SIZE, color=style.FG, alpha=a)
        upto = int(np.floor(k)) + 1
        if upto >= 2:
            ax.plot(trace_x[:upto], trace_y[:upto], color=style.FG, lw=3, alpha=a,
                    gid="trace")
        if 0 < sweep < 1:
            ax.scatter([trace_x[upto - 1]], [trace_y[upto - 1]], s=90, color=style.C_ACCENT,
                       linewidths=0, alpha=a, zorder=4)

        a_tiny = a * style.ease(style.progress(f, f_tiny + 0.03, f_tiny + 0.08))
        ax.text(TRACE_X + TRACE_W * 0.5, TRACE_BASE + 0.45,
                f"the planet blocks {100 * depth:.1f}% of the light", ha="center",
                va="center", size=style.LABEL_SIZE, color=style.C_ACCENT, alpha=a_tiny)
        a_word = a * style.ease(style.progress(f, f_transit, f_transit + 0.04))
        x0, x1 = trace_x[lo], trace_x[hi]
        yb = TRACE_BASE - TRACE_H - 0.5                  # just under the time axis
        ax.plot([x0, x0, x1, x1], [yb + 0.15, yb, yb, yb + 0.15], color=style.C_ACCENT,
                lw=2, alpha=a_word)
        ax.text((x0 + x1) / 2, yb - 0.32, "transit", ha="center", va="center",
                size=style.BODY_SIZE, weight="semibold", color=style.C_ACCENT, alpha=a_word)

    def _draw_counts(ax, fig, f, a):
        fig.text(style.MARGIN, style.HEADING_Y, "Signals astronomers have already judged",
                 ha="left", va="top", size=style.HEADING_SIZE, weight="semibold",
                 color=style.FG, alpha=a)
        shown = int(round(n_kois * style.ease(style.progress(f, f_use - 0.01, f_gather1))))
        a_all = a * (1 - style.ease(style.progress(f, f_left, f_left + 0.03)))
        if a_all > 0:                                  # gone once its dots have left
            ax.text(GREY_XY[0], 7.4, f"{shown:,}", ha="right", va="center",
                    size=style.BODY_SIZE, weight="semibold", color=style.FG, alpha=a_all)
            ax.text(GREY_XY[0] + 0.12, 7.4, "Kepler signals, one dot each", ha="left",
                    va="center", size=style.BODY_SIZE, color=style.MUTED, alpha=a_all)

        # dot positions: grey slot -> class cloud along an arc
        p = np.clip((f - depart) / FLIGHT, 0.0, 1.0)
        p = p * p * (3 - 2 * p)
        xy = grey + (target - grey) * p[:, None]
        xy[:, 1] += lift * np.sin(np.pi * p)
        rgb = grey_rgb + (class_rgb - grey_rgb) * p[:, None]
        visible = np.arange(n_kois) < shown
        ax.scatter(xy[visible, 0], xy[visible, 1], s=DOT_S,
                   c=np.column_stack([rgb[visible], np.full(visible.sum(), a)]),
                   linewidths=0)

        for colour, xy0, count, label, f_on in (
                (style.C_PLANET, CONF_XY, int((p[is_conf] >= 1).sum()),
                 "confirmed planets", conf_span[0]),
                (style.C_FALSE, FALSE_XY, int((p[~is_conf] >= 1).sum()),
                 "false positives", false_span[0])):
            a_lab = a * style.ease(style.progress(f, f_on, f_on + 0.03))
            if a_lab <= 0:
                continue
            ax.text(xy0[0], 6.75, f"{count:,}", ha="center", va="center",
                    size=style.TITLE_SIZE, weight="semibold", color=colour, alpha=a_lab)
            ax.text(xy0[0], 6.1, label, ha="center", va="center", size=style.BODY_SIZE,
                    color=colour, alpha=a_lab)

        a_bin = a * style.ease(style.progress(f, f_binary - 0.01, f_binary + 0.04))
        if a_bin > 0:
            _draw_binary(ax, f, a_bin)

    def _draw_binary(ax, f, a):
        """Two stars orbiting each other, the nearer one eclipsing the other."""
        bx, by = BINARY_XY
        angle = 2 * np.pi * f * ctx.duration / BINARY_PERIOD_S   # ambient motion, like twinkle
        for sign, r in ((1, 0.32), (-1, 0.22)):
            x = bx + sign * 0.5 * np.cos(angle)
            depth_order = 3 + sign * np.sin(angle)       # the nearer star is drawn on top
            ax.add_patch(Circle((x, by), r, facecolor=style.C_ACCENT, edgecolor=style.BG,
                                lw=2, alpha=a, zorder=depth_order))
        ax.text(bx, by - 1.15, "e.g. two stars\neclipsing\neach other", ha="center",
                va="center", size=style.LABEL_SIZE, color=style.C_FALSE, alpha=a,
                linespacing=1.2)

    return draw
