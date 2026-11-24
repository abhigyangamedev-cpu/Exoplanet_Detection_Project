"""Scene 5, the light-curve pipeline, on the real Kepler data of K00020.01 (a planet).

Five stages, each labelled: raw brightness in three-month pieces (a few weeks first, so the
transits show, then all four years), stitched, slow changes removed (the trend drawn, then
divided out), folded on the orbit (points slide from their time to their phase position,
orbit after orbit landing on top of each other), and the two pictures the network sees
(global and local views). The last part compares the planet's local view with an eclipsing
binary's (K06252.01) in their class colours, on a shared depth scale.
"""

from __future__ import annotations

import numpy as np
import style
from matplotlib.colors import to_rgb, to_rgba

from scenes import cue_fraction

MAIN = (0.085, 0.27, 0.87, 0.54)        # light-curve axes, figure fractions
LEFT = (0.085, 0.33, 0.38, 0.42)        # the two panels at the end
RIGHT = (0.575, 0.33, 0.38, 0.42)
STEP = 3                                # draw every 3rd point (about 17,000), except
                                        # the first orbits, shown close up: all points
HEADING_STRIP_Y = 0.92                  # the five-step progress strip, top right
DOT_S, DOT_S_CLOSE = 2.5, 10.0          # points^2: whole light curve / close-up orbits
N_SLIDE = 3                             # orbits that visibly slide onto the first one;
                                        # the first look at the data shows N_SLIDE + 1 orbits
LOCAL_DURATIONS = 2.0                   # local view half-width in transit durations
                                        # (configs/default.yaml views.local_durations)
STAGES = ("Raw brightness, in three-month pieces", "Stitched: pieces lined up",
          "Slow changes removed", "Folded on the orbit", "Two pictures for the network")


def fold_x(phase, orbit, period, p):
    """x of each point while folding: its time position (p=0) to its phase position (p=1).

    Time is measured from the first transit, so a point's time is phase + orbit * period.
    """
    return phase + orbit * period * (1.0 - np.asarray(p))


def _fit_scale(phase, flux, view, lo, hi):
    """(baseline, depth) so that baseline + depth * view(phase) matches the folded flux."""
    x = np.linspace(lo, hi, len(view))
    inside = (phase >= lo) & (phase <= hi)
    g = np.interp(phase[inside], x, view)
    base = float(np.median(flux))
    return base, float(np.sum((flux[inside] - base) * g) / np.sum(g * g))


def _range(lo, hi, pad=0.06):
    span = hi - lo
    return lo - pad * span, hi + pad * span


def _lerp_box(a, b, p):
    """Camera between boxes (x0, x1, y0, y1). Spans change geometrically, so a zoom from a
    few days to four years looks steady; x keeps its left edge moving linearly (the start
    of the data stays in view), y its centre."""
    wa, wb = a[1] - a[0], b[1] - b[0]
    x0 = a[0] + (b[0] - a[0]) * p
    w = wa * (wb / wa) ** p
    ha, hb = a[3] - a[2], b[3] - b[2]
    yc = (a[2] + a[3]) / 2 + ((b[2] + b[3]) / 2 - (a[2] + a[3]) / 2) * p
    h = ha * (hb / ha) ** p
    return [x0, x0 + w, yc - h / 2, yc + h / 2]


def _style_axes(ax, a, xlabel="", ylabel=""):
    """Axis lines, ticks and labels faded to alpha a."""
    for side in ("left", "bottom"):
        ax.spines[side].set_alpha(a)
    ax.tick_params(colors=to_rgba(style.MUTED, a), labelcolor=to_rgba(style.FG, a),
                   length=5, width=1.2)
    if xlabel:
        ax.set_xlabel(xlabel, alpha=a, labelpad=4)
    if ylabel:
        ax.set_ylabel(ylabel, alpha=a, labelpad=6)


def _star(ctx, name):
    s = ctx.load(name)
    P = float(s["period"])
    k = np.flatnonzero(s["kept"])
    orbit = np.round((s["time"][k] - s["epoch"] - s["phase"]) / P).astype(int)
    t0 = float(s["epoch"]) + orbit.min() * P                 # first transit in the data
    half = min(LOCAL_DURATIONS * float(s["duration_hours"]) / 24.0, P / 2)
    return s, P, k, orbit - orbit.min(), t0, half


def build(ctx):
    n = ctx.numbers
    s, P, k, orbit_all, t0, half = _star(ctx, "star_planet.npz")
    pos = np.arange(len(k))
    sub = pos[(pos % STEP == 0) | (orbit_all <= N_SLIDE)]      # index into the kept points
    idx = k[sub]                                              # index into the full arrays
    x_time = s["time"][idx] - t0
    raw = s["raw_flux"][idx] / np.median(s["raw_flux"])
    stitched = s["stitched_flux"][idx]
    trend = s["trend"][idx]
    phase = s["phase"][sub]
    orbit = orbit_all[sub]
    n_orbits = int(orbit.max()) + 1
    quarter = s["quarter_index"][idx]
    close = orbit <= N_SLIDE                                 # drawn bigger when zoomed in
    in_transit = np.abs(phase) < s["duration_hours"] / 48.0
    years = (s["time"].max() - s["time"].min()) / 365.25

    # the trend as a line with breaks over the gaps between quarters
    tl_x = s["time"][::10] - t0
    tl_y = s["trend"][::10].astype(float)
    gap = np.diff(tl_x, prepend=tl_x[0]) > 0.5
    tl_x, tl_y = np.insert(tl_x, np.flatnonzero(gap), np.nan), np.insert(
        tl_y, np.flatnonzero(gap), np.nan)

    gv, lv = s["global_view"], s["local_view"]
    flux_all, phase_all = s["detrended_flux"], s["phase"]
    g_base, g_depth = _fit_scale(phase_all, flux_all, gv, -P / 2, P / 2)
    l_base, l_depth = _fit_scale(phase_all, flux_all, lv, -half, half)
    gv_x, lv_x = np.linspace(-P / 2, P / 2, len(gv)), np.linspace(-half, half, len(lv))

    b, bP, _, _, _, b_half = _star(ctx, "star_binary.npz")
    _, b_depth = _fit_scale(b["phase"], b["detrended_flux"], b["local_view"], -b_half, b_half)
    planet_pct = 100 * l_depth * lv                         # brightness change, %
    binary_pct = 100 * b_depth * b["local_view"]
    hours_p = np.linspace(-half, half, len(lv)) * 24
    hours_b = np.linspace(-b_half, b_half, len(b["local_view"])) * 24

    # colours
    q_cycle = np.array([to_rgb(c) for c in (style.C_ACCENT, style.FG, style.MUTED)])
    q_rgb = q_cycle[quarter % len(q_cycle)]
    fg, accent, muted = (np.array(to_rgb(c)) for c in (style.FG, style.C_ACCENT, style.MUTED))

    # camera boxes (x0, x1, y0, y1), x in days from the first transit
    x_four = (-P / 2, -P / 2 + (N_SLIDE + 1) * P)
    box_window = (*x_four, *_range(raw[close].min(), raw[close].max(), 0.12))
    x_all = _range(x_time.min(), x_time.max(), 0.02)
    box_raw = (*x_all, *_range(raw.min(), raw.max()))
    y_flat = _range(np.percentile(stitched, 0.01), np.percentile(stitched, 99.99))
    box_stitched = (*x_all, *y_flat)
    # zoomed on the baseline, so the slow changes are big (the dips drop out of view)
    box_trend = (*x_all, *_range(np.percentile(stitched, 6), np.percentile(stitched, 99.95),
                                 1.0))
    box_four = (*x_four, *y_flat)
    box_fold = (-P / 2, P / 2, *y_flat)
    box_local = (-1.15 * half, 1.15 * half, *y_flat)

    # timing, tied to the spoken words
    f_stitch = cue_fraction(ctx, "stitch", 0.1)
    f_flat = cue_fraction(ctx, "flatten", 0.19)
    f_keep = cue_fraction(ctx, "keeping the short", 0.25)
    f_fold = cue_fraction(ctx, "fold on", 0.29)
    f_lands = cue_fraction(ctx, "lands on top", 0.36)
    f_stands = cue_fraction(ctx, "stands out", 0.43)
    f_two = cue_fraction(ctx, "two pictures", 0.47)
    f_whole = cue_fraction(ctx, "whole orbit", 0.52)
    f_close = cue_fraction(ctx, "close-up", 0.55)
    f_built = cue_fraction(ctx, "We built", 0.63)
    f_planet = cue_fraction(ctx, "planet makes", 0.79)
    f_binary = cue_fraction(ctx, "Two stars", 0.87)

    f_reveal = 0.03                                          # the first orbits draw in
    f_zoom0, f_zoom1 = 0.035, f_stitch - 0.02                # four orbits -> four years
    f_div0, f_div1 = f_flat + 0.03, f_flat + 0.06            # the trend divided out
    f_unzoom = min(f_div1 + 0.025, f_fold)                   # back out: the dips stay
    f_dips = max(f_keep, f_unzoom - 0.01)
    f_fold0 = max(f_fold, f_unzoom)
    f_four = f_fold0 + 0.035                                 # camera on four orbits
    slide0, slide1 = f_four + 0.01, max(f_lands + 0.02, f_four + 0.06)
    pile0, pile1 = slide1 + 0.005, max(f_stands + 0.01, slide1 + 0.05)
    # each orbit's own fold progress window
    starts = np.where(orbit <= N_SLIDE, slide0 + (orbit - 1) * 0.008,
                      pile0 + (pile1 - pile0) * (orbit - N_SLIDE - 1)
                      / max(n_orbits - N_SLIDE - 1, 1))
    flights = np.where(orbit <= N_SLIDE, slide1 - slide0 - (N_SLIDE - 1) * 0.008, 0.012)
    f_panels = f_built - 0.01                                # two clean pictures
    f_compare = f_planet - 0.045                             # planet vs binary
    stage_starts = (0.0, f_stitch - 0.005, f_flat - 0.01, f_fold - 0.005, f_two - 0.01)
    camera = (  # (start, end, from, to); the latest move that has started wins
        (f_zoom0, f_zoom1, box_window, box_raw),
        (f_stitch, f_stitch + 0.05, box_raw, box_stitched),
        (f_flat - 0.01, f_flat + 0.02, box_stitched, box_trend),
        (f_div1 + 0.005, f_unzoom, box_trend, box_stitched),
        (f_fold0, f_four, box_stitched, box_four),
        (slide0, slide1, box_four, box_fold),
        (f_close, f_close + 0.03, box_fold, box_local),
    )

    def draw(fig, t):
        f = t / ctx.duration
        style.starfield(fig.add_axes((0, 0, 1, 1)), 5, t)
        _draw_titles(fig, f)
        a_main = 1 - style.ease(style.progress(f, f_panels - 0.02, f_panels))
        if a_main > 0:
            _draw_main(fig, f, a_main)
        if f >= f_panels:
            _draw_panels(fig, f)

    def _draw_titles(fig, f):
        stage = int(np.searchsorted(stage_starts, f, side="right")) - 1
        a = style.ease(style.progress(f, stage_starts[stage], stage_starts[stage] + 0.015))
        compare = style.ease(style.progress(f, f_compare - 0.01, f_compare + 0.01))
        title = STAGES[stage] if compare < 0.5 else "Planet, or two stars?"
        a = a * (1 - min(compare, 1 - compare) * 2)          # dip through zero at the swap
        fig.text(style.MARGIN, style.HEADING_Y, title, ha="left", va="top",
                 size=style.HEADING_SIZE, weight="semibold", color=style.FG, alpha=a)
        # progress strip: five steps, the current one lit
        for i in range(len(STAGES)):
            lit = i == stage and compare < 0.5
            done = i < stage or compare >= 0.5
            fig.text(0.80 + i * 0.033, HEADING_STRIP_Y, f"{i + 1}", ha="center",
                     va="center", size=style.LABEL_SIZE - 2, weight="bold",
                     color=style.C_ACCENT if lit else (style.FG if done else style.MUTED),
                     alpha=1.0 if lit else 0.55)
        a_star = 1 - style.ease(style.progress(f, f_panels - 0.02, f_panels))
        if a_star > 0:
            fig.text(style.MARGIN, 0.855, f"star {s['koi_name']}: a confirmed planet, "
                     f"{len(s['time']):,} brightness measurements over {years:.0f} years",
                     ha="left", va="center", size=style.LABEL_SIZE - 2, color=style.MUTED,
                     alpha=a_star)

    def _draw_main(fig, f, a_main):
        ax = fig.add_axes(MAIN)
        box = box_window
        for f0, f1, box_a, box_b in camera:
            if f >= f0:
                box = _lerp_box(box_a, box_b, style.ease(style.progress(f, f0, f1)))
        ax.set_xlim(box[0], box[1])
        ax.set_ylim(box[2], box[3])
        _style_axes(ax, a_main, "days from mid-transit" if f >= slide1 else "days",
                    "brightness")
        ax.set_yticks([])

        # y: raw -> stitched -> detrended
        p_st = style.ease(style.progress(f, f_stitch, f_stitch + 0.06))
        p_dt = style.ease(style.progress(f, f_div0, f_div1))
        y = raw + (stitched - raw) * p_st
        y = y / trend ** p_dt
        # x: time -> phase
        p_fold = np.clip((f - starts) / flights, 0.0, 1.0)
        p_fold = p_fold * p_fold * (3 - 2 * p_fold)
        x = fold_x(phase, orbit, P, p_fold)

        # colour: quarters -> white; while folding the moving orbits are lit
        rgb = q_rgb + (fg - q_rgb) * p_st
        a_lit = (style.ease(style.progress(f, f_four - 0.01, f_four + 0.01))
                 * (1 - style.ease(style.progress(f, pile1 - 0.005, pile1 + 0.02))))
        moving = (orbit >= 1) & (orbit <= N_SLIDE)
        rgb[moving] = rgb[moving] + (accent - rgb[moving]) * a_lit
        piling = orbit > N_SLIDE
        rgb[piling] = rgb[piling] + (muted - rgb[piling]) * a_lit
        # after flattening, the points inside a transit are lit: the dips that stay
        a_dips = (style.ease(style.progress(f, f_dips, f_dips + 0.015))
                  * (1 - style.ease(style.progress(f, f_four - 0.01, f_four))))
        rgb[in_transit] = rgb[in_transit] + (accent - rgb[in_transit]) * a_dips
        alpha = np.full(len(x), 0.9 * a_main)
        a_dim = style.ease(style.progress(f, f_two, f_two + 0.03))
        alpha *= 1 - 0.65 * a_dim
        reveal = x_four[0] + (x_four[1] - x_four[0]) * style.progress(f, 0.0, f_reveal)
        alpha[x_time > reveal] *= f > f_reveal
        zoomed = max(1 - style.progress(f, f_zoom0, f_zoom0 + 0.02),
                     style.progress(f, f_fold0, f_four)
                     * (1 - style.progress(f, slide1, pile1)))
        size = np.where(close, DOT_S + (DOT_S_CLOSE - DOT_S) * zoomed, DOT_S)
        size = np.where(in_transit, size + 4 * a_dips, size)
        ax.scatter(x, y, s=size, c=np.column_stack([rgb, alpha]), linewidths=0)

        if f < f_zoom0 + 0.01:
            _note(ax, f, 0.0, f_zoom0 + 0.01, "each dip: the planet passing in front",
                  0.5, 0.08)
        if f_stitch - 0.01 < f < f_flat:
            _note(ax, f, f_stitch + 0.06, f_flat - 0.01,
                  "Kepler's three-month pieces, joined into one record", 0.5, 0.94)
        if f_flat - 0.01 < f < f_fold0 + 0.01:
            _draw_trend(ax, f, a_main)
        if f_fold - 0.01 < f < f_two:
            _draw_fold_marks(ax, f, p_fold)
        if f >= f_two - 0.01:
            _draw_views(ax, f, a_main)

    def _note(ax, f, f0, f1, text, x, y, color=style.C_ACCENT):
        a = style.ease(style.progress(f, f0, f0 + 0.01)) * (
            1 - style.ease(style.progress(f, f1 - 0.01, f1)))
        ax.text(x, y, text, transform=ax.transAxes, ha="center", va="center",
                size=style.LABEL_SIZE, color=color, alpha=a)

    def _draw_trend(ax, f, a_main):
        p_draw = style.ease(style.progress(f, f_flat - 0.005, f_div0 - 0.005))
        p_dt = style.ease(style.progress(f, f_div0, f_div1))
        a = a_main * (1 - style.ease(style.progress(f, f_div1, f_div1 + 0.015)))
        upto = int(len(tl_x) * p_draw)
        if upto > 1:
            ax.plot(tl_x[:upto], 1 + (tl_y[:upto] - 1) * (1 - p_dt), color=style.C_ACCENT,
                    lw=3, alpha=a)
        _note(ax, f, f_flat + 0.005, f_div1, "the star's own slow changes", 0.5, 0.94)
        _note(ax, f, f_dips, f_fold0 + 0.01, "divided out: flat, but the short dips stay",
              0.5, 0.94)

    def _draw_fold_marks(ax, f, p_fold):
        a = (style.ease(style.progress(f, f_four - 0.01, f_four + 0.01))
             * (1 - style.ease(style.progress(f, slide0 + 0.01, slide0 + 0.03))))
        if a > 0:
            for i in range(1, N_SLIDE + 1):
                ax.axvline(-P / 2 + i * P, color=style.MUTED, lw=1.5, ls=(0, (5, 4)),
                           alpha=a)
            ax.text(0, 0.94, f"one orbit = {P:.1f} days", transform=ax.get_xaxis_transform(),
                    ha="center", va="center", size=style.LABEL_SIZE, color=style.C_ACCENT,
                    alpha=a)
        a_count = (style.ease(style.progress(f, slide0, slide0 + 0.02))
                   * (1 - style.ease(style.progress(f, f_two - 0.02, f_two))))
        if a_count > 0:
            stacked = 1 + int((p_fold.reshape(-1)[_orbit_first] >= 1).sum())
            ax.text(0.98, 0.94, f"orbits stacked: {stacked} of {n_orbits}",
                    transform=ax.transAxes, ha="right", va="center", size=style.LABEL_SIZE,
                    color=style.FG, alpha=a_count)
        _note(ax, f, f_stands, f_two - 0.005, "a faint dip stands out", 0.75, 0.35)

    def _draw_views(ax, f, a_main):
        p_g = style.ease(style.progress(f, f_two, f_whole + 0.02))
        upto = int(len(gv) * p_g)
        if upto > 1:
            ax.plot(gv_x[:upto], g_base + g_depth * gv[:upto], color=style.C_ACCENT, lw=3,
                    alpha=a_main * (1 - style.ease(style.progress(f, f_close, f_close + 0.02))))
        _note(ax, f, f_whole - 0.01, f_close, "picture 1: the whole orbit", 0.5, 0.94)
        p_l = style.ease(style.progress(f, f_close + 0.02, f_close + 0.05))
        upto = int(len(lv) * p_l)
        if upto > 1:
            ax.plot(lv_x[:upto], l_base + l_depth * lv[:upto], color=style.C_ACCENT, lw=3.5,
                    alpha=a_main)
        _note(ax, f, f_close + 0.01, f_panels, "picture 2: a close-up of the transit",
              0.5, 0.94)

    def _draw_panels(fig, f):
        a_two = (style.ease(style.progress(f, f_panels, f_panels + 0.02))
                 * (1 - style.ease(style.progress(f, f_compare - 0.02, f_compare))))
        a_cmp = style.ease(style.progress(f, f_compare, f_compare + 0.02))
        if a_two > 0:
            _two_pictures(fig, f, a_two)
        if a_cmp > 0:
            _compare(fig, f, a_cmp)

    def _two_pictures(fig, f, a):
        for box, x, v, xlabel, title in (
                (LEFT, gv_x / P, gv, "position in the orbit", "the whole orbit"),
                (RIGHT, lv_x * 24, lv, "hours from mid-transit", "close-up of the transit")):
            ax = fig.add_axes(box)
            ax.plot(x, v, color=style.FG, lw=2.5, alpha=a)
            ax.set_xlim(x[0], x[-1])
            ax.set_ylim(-1.2, 0.3)
            ax.set_yticks([])
            _style_axes(ax, a, xlabel)
            ax.set_title(f"{title}  ({len(v):,} values)", alpha=a, size=style.LABEL_SIZE,
                         pad=10)
        a_count = a * style.ease(style.progress(f, f_built, f_built + 0.02))
        fig.text(0.5, 0.835, f"Built for {n['n_views_built']:,} of the {n['n_kois']:,} "
                 "signals", ha="center", va="center", size=style.BODY_SIZE,
                 weight="semibold", color=style.C_ACCENT, alpha=a_count)

    def _compare(fig, f, a):
        lo = min(planet_pct.min(), binary_pct.min()) * 1.12
        hi = -0.12 * lo
        for box, hours, pct, colour, title, label, f_on in (
                (LEFT, hours_p, planet_pct, style.C_PLANET,
                 f"confirmed planet  {s['koi_name']}", "shallow, rounded dip", f_planet),
                (RIGHT, hours_b, binary_pct, style.C_FALSE,
                 f"two stars eclipsing  {b['koi_name']}", "deep,\nsharp V", f_binary)):
            a_line = a * style.ease(style.progress(f, f_on - 0.02, f_on + 0.02))
            ax = fig.add_axes(box)
            ax.set_xlim(hours[0], hours[-1])
            ax.set_ylim(lo, hi)
            _style_axes(ax, a, "hours from mid-transit",
                        "brightness change (%)" if box is LEFT else "")
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}".replace("-", "−"))
            if a_line > 0:
                upto = max(int(len(pct) * style.ease(style.progress(f, f_on - 0.02,
                                                                    f_on + 0.03))), 2)
                ax.plot(hours[:upto], pct[:upto], color=colour, lw=4, alpha=a_line)
            ax.set_title(title, color=colour, size=style.LABEL_SIZE, pad=10,
                         alpha=a * style.ease(style.progress(f, f_on - 0.04, f_on - 0.01)))
            ax.text(0.5 if box is LEFT else 0.83, 0.2 if box is LEFT else 0.35, label,
                    transform=ax.transAxes, ha="center", va="center", linespacing=1.15,
                    size=style.BODY_SIZE, weight="semibold",
                    color=colour, alpha=a * style.ease(style.progress(f, f_on, f_on + 0.02)))

    # one point per orbit, to count the orbits that have landed (orbit 0 never moves)
    _orbit_first = np.unique(orbit, return_index=True)[1][1:]
    return draw

