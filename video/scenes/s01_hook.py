"""Scene 1, the hook: a star field, a distant star dimming as a planet crosses it, and the
real folded Kepler light curve of K00020.01 drawing in left to right, dip and all.

Frame 0 shows only the star field (``draw_starfield``): it is the loop target the end card
cross-fades into, so everything else fades in after t = 0.
"""

from __future__ import annotations

import numpy as np
import style
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator

from scenes import cue_fraction

STARFIELD_SEED = 7          # shared with the end card, which fades into this frame
HALF_WINDOW_H = 12.0        # light curve shown from -12 h to +12 h around mid-transit
N_POINTS = 2500             # light-curve points drawn (subsampled for speed)
BIN_H = 0.25                # binned trace resolution, hours

# overlay axes in "inch" units (16 x 9): the distant star on the right
STAR_XY = (12.6, 5.05)
STAR_R = 1.05
PLANET_R = 0.2
LC_RECT = (0.085, 0.27, 0.5, 0.41)   # light-curve axes, figure fractions


def draw_starfield(fig, t: float) -> None:
    """The star field alone, filling the figure: the whole of frame 0."""
    style.starfield(fig.add_axes((0, 0, 1, 1)), STARFIELD_SEED, t)


def _light_curve(ctx):
    star = ctx.load("star_planet.npz")
    hours = star["phase"] * 24.0
    flux = star["detrended_flux"]
    inside = np.abs(hours) < HALF_WINDOW_H
    hours, flux = hours[inside], flux[inside]
    rng = np.random.default_rng(0)
    pick = np.sort(rng.choice(len(hours), size=min(N_POINTS, len(hours)), replace=False))
    order = np.argsort(hours[pick])
    pts_h, pts_f = hours[pick][order], flux[pick][order]
    edges = np.arange(-HALF_WINDOW_H, HALF_WINDOW_H + BIN_H, BIN_H)
    idx = np.digitize(hours, edges) - 1
    bin_h = 0.5 * (edges[:-1] + edges[1:])
    bin_f = np.array([np.median(flux[idx == i]) if np.any(idx == i) else np.nan
                      for i in range(len(bin_h))])
    ok = np.isfinite(bin_f)
    return dict(pts_h=pts_h, pts_f=pts_f, bin_h=bin_h[ok], bin_f=bin_f[ok],
                name=str(star["koi_name"]), period=float(star["period"]),
                half_dur=float(star["duration_hours"]) / 2.0)


def build(ctx):
    lc = _light_curve(ctx)
    title = ctx.credits["title"]
    name = ctx.credits["name"]
    depth_pct = 100.0 * (1.0 - lc["bin_f"].min())
    f_lo = lc["bin_f"].min() - 0.25 * (1.0 - lc["bin_f"].min())
    f_hi = 1.0 + 0.2 * (1.0 - lc["bin_f"].min())

    # timing (fractions of the scene), tied to the spoken words
    f_mid = cue_fraction(ctx, "grows", 0.31)            # cursor at mid-transit
    f_sweep0 = 0.08
    f_sweep1 = f_mid + (f_mid - f_sweep0)                 # constant sweep speed
    f_dimmer = cue_fraction(ctx, "dimmer", 0.37)
    f_kepler = cue_fraction(ctx, "NASA", 0.42)
    f_planets = cue_fraction(ctx, "planets", 0.76)
    f_impostors = cue_fraction(ctx, "impostors", 0.85)

    def draw(fig, t):
        f = t / ctx.duration
        draw_starfield(fig, t)
        if f <= 0.0:
            return                                       # frame 0: only the star field

        a_title = style.ease(style.progress(f, 0.0, 0.1))
        a_panel = style.ease(style.progress(f, 0.03, 0.1))
        sweep = style.progress(f, f_sweep0, f_sweep1)
        cursor = -HALF_WINDOW_H + 2 * HALF_WINDOW_H * sweep

        fig.text(0.5, 0.9, title, ha="center", va="center", size=style.TITLE_SIZE,
                 weight="semibold", color=style.FG, alpha=a_title)
        fig.text(0.5, 0.815, name, ha="center", va="center", size=style.BODY_SIZE,
                 color=style.MUTED, alpha=a_title)

        # --- the real light curve, drawn up to the cursor -----------------------------
        ax = fig.add_axes(LC_RECT)
        shown = lc["pts_h"] <= cursor
        ax.scatter(lc["pts_h"][shown], lc["pts_f"][shown], s=5, color=style.MUTED,
                   alpha=0.45 * a_panel, linewidths=0)
        line = lc["bin_h"] <= cursor
        ax.plot(lc["bin_h"][line], lc["bin_f"][line], color=style.FG, lw=3, alpha=a_panel)
        if 0.0 < sweep < 1.0 and np.any(line):
            ax.scatter([lc["bin_h"][line][-1]], [lc["bin_f"][line][-1]], s=90,
                       color=style.C_ACCENT, linewidths=0, zorder=5)
        ax.set_xlim(-HALF_WINDOW_H, HALF_WINDOW_H)
        ax.set_ylim(f_lo, f_hi)
        ax.yaxis.set_major_locator(MaxNLocator(3))
        ax.set_xticks([-12, -6, 0, 6, 12])
        ax.set_xlabel("hours from the middle of the dip", alpha=a_panel)
        ax.set_ylabel("brightness", alpha=a_panel)
        for side in ("left", "bottom"):
            ax.spines[side].set_alpha(a_panel)
        ax.tick_params(colors=(*to_rgb(style.MUTED), a_panel),
                       labelcolor=(*to_rgb(style.FG), a_panel))

        a_label = style.ease(style.progress(f, f_kepler, f_kepler + 0.06))
        fig.text(LC_RECT[0], LC_RECT[1] + LC_RECT[3] + 0.035,
                 f"Real Kepler data: star {lc['name']}, folded on its "
                 f"{lc['period']:.2f}-day orbit",
                 ha="left", va="center", size=style.LABEL_SIZE, color=style.MUTED,
                 alpha=a_label)
        a_dip = style.ease(style.progress(f, f_dimmer, f_dimmer + 0.05))
        ax.annotate(f"the star dims by {depth_pct:.1f}%", xy=(0.0, lc["bin_f"].min()),
                    xytext=(5.0, lc["bin_f"].min() + 0.15 * (f_hi - f_lo)),
                    color=style.C_ACCENT, size=style.LABEL_SIZE, alpha=a_dip,
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="->", color=style.C_ACCENT, lw=2,
                                    alpha=a_dip, shrinkB=8))

        # --- the distant star and its planet -----------------------------------------
        ov = fig.add_axes((0, 0, 1, 1))
        ov.set_xlim(0, 16)
        ov.set_ylim(0, 9)
        ov.set_axis_off()
        drop = (1.0 - np.interp(cursor, lc["bin_h"], lc["bin_f"])) / (1.0 - lc["bin_f"].min())
        glow = (1.0 - 0.45 * float(np.clip(drop, 0.0, 1.0))) * a_panel  # exaggerated to show
        sx, sy = STAR_XY
        for scale in np.linspace(2.2, 1.04, 12):      # soft glow: many faint rings
            ov.add_patch(Circle((sx, sy), STAR_R * scale, color=style.C_ACCENT,
                                alpha=0.06 * glow, lw=0))
        ov.add_patch(Circle((sx, sy), STAR_R, color=style.C_ACCENT, alpha=glow, lw=0))
        for scale in np.linspace(0.95, 0.35, 7):      # brighter centre (limb darkening)
            ov.add_patch(Circle((sx, sy), STAR_R * scale, color=style.FG, alpha=0.07 * glow,
                                lw=0))
        # planet: at the contact points when the cursor is at +-half the transit duration
        dx = (STAR_R + PLANET_R) * cursor / lc["half_dur"]
        a_planet = a_panel * (1.0 - style.ease(style.progress(abs(dx), 1.7 * STAR_R,
                                                              2.3 * STAR_R)))
        if a_planet > 0:
            ov.add_patch(Circle((sx + dx, sy - 0.3 * STAR_R), PLANET_R, facecolor=style.BG,
                                edgecolor=style.MUTED, lw=1.5, alpha=a_planet))

        a_q1 = style.ease(style.progress(f, f_planets - 0.02, f_planets + 0.04))
        a_q2 = style.ease(style.progress(f, f_impostors - 0.02, f_impostors + 0.04))
        ov.text(sx, 2.45, "planet?", ha="center", va="center", size=style.HEADING_SIZE,
                weight="semibold", color=style.C_PLANET, alpha=a_q1)
        ov.text(sx, 1.8, "or impostor?", ha="center", va="center", size=style.HEADING_SIZE,
                weight="semibold", color=style.C_FALSE, alpha=a_q2)

    return draw

