"""Scene 8, the results on the final exam (held-out test set), told honestly.

Left: the test precision-recall curves (pr_curves.npz) draw in one model at a time, each with
its bootstrap band and a direct label at its line end; on the right a table fills in each
test PR-AUC with its 95% interval (numbers.json test_pr_auc) as it is spoken. The catalogue
model trained on the network's training signals (tabular_lc_sub, not narrated) joins the
table last, as a row only: the narration leaves no time to draw its curve. Then the curves
give way to the network's recall by signal strength and orbit length (error_breakdown.json),
and the table to the false positives the network accepts, of which those with light from a
neighbouring star (the centroid-offset flag, numbers.json cnn_fp_accepted_centroid) are
marked. Last, how much data the network learned from, and the development-fold curves of
the retrained network (cnn_full) against the matching catalogue model, labelled as practice
groups so nobody reads them as test results.
"""

from __future__ import annotations

import numpy as np
import style
from matplotlib.patches import Rectangle

from scenes import cue_fraction

# layout in inches of the 16 x 9 overlay axes (caption band: y < 1.35)
PLOT = (1.3, 2.4, 5.5, 4.65)            # PR axes: x0, y0, w, h
Y_LIM = (0.38, 1.01)                     # precision axis (no model falls below 0.4)
LABEL_GAP = 0.06                         # min spacing of line-end labels, precision units
PANEL_X, PANEL_R = 9.0, 15.4             # right-hand panel
ROW_YS = (6.05, 4.95, 3.85, 2.75)        # test table rows, README order
DEV_ROW_YS = (3.6, 2.55)
SWATCH_W = 0.45
NAME_X = PANEL_X + SWATCH_W + 0.15
BAR_BASE, BAR_SCALE = 2.6, 3.4           # recall bars: base y, height of 100 %
BAR_W, BAR_STEP = 0.62, 0.86
CHART_X0 = (1.3, 5.3)                    # recall by SNR, recall by period
DRAW = 0.05                              # one curve's draw-in, fraction of the scene
FADE = 0.015

TEST_MODELS = ("tabular", "tabular_lc", "tabular_lc_sub", "cnn")   # table rows
NARRATED = ("tabular", "tabular_lc", "cnn")                         # curves drawn
DEV_MODELS = ("cnn_full", "tabular_lc")
# plain names on screen: (name, what it reads, line-end label); same words as scene 6
NAMES = {
    "tabular": ("Decision trees", "all catalogue numbers", "all numbers"),
    "tabular_lc": ("Decision trees", "without image-shift numbers", "no image shift"),
    "tabular_lc_sub": ("Decision trees", "same training signals as the network",
                       "network's signals"),
    "cnn": ("Neural network", "the two light-curve pictures", "neural network"),
    "cnn_full": ("Neural network", "retrained on all practice signals", "neural network"),
}
COLOURS = {"tabular": style.C_TREES, "tabular_lc": style.C_TREES,
           "tabular_lc_sub": style.C_TREES, "cnn": style.C_NET, "cnn_full": style.C_NET}
DASHES = {"tabular": "-", "tabular_lc": (0, (5, 2.5)),
          "cnn": "-", "cnn_full": "-"}


def spread_labels(ys, gap: float, lo: float, hi: float) -> np.ndarray:
    """Label positions near ys, in the same order, at least gap apart, inside [lo, hi]."""
    ys = np.asarray(ys, float)
    order = np.argsort(ys)
    out = np.clip(ys[order], lo, hi)
    for _ in range(50):                       # push neighbours apart, keep the centre
        moved = False
        for i in range(1, len(out)):
            short = gap - (out[i] - out[i - 1])
            if short > 1e-9:
                out[i - 1] -= short / 2
                out[i] += short / 2
                moved = True
        out = np.clip(out, lo, hi)
        if not moved:
            break
    for i in range(1, len(out)):              # bounds may squeeze: enforce the gap upwards
        out[i] = max(out[i], out[i - 1] + gap)
    res = np.empty_like(out)
    res[order] = out
    return res


def bin_label(text: str, last: bool) -> str:
    """'0.558 - 5.58' -> '0.6–5.6'; the top bin reads '27+'."""
    lo, hi = (float(v) for v in text.split(" - "))

    def fmt(x):
        return f"{x:.1f}" if x < 10 else f"{x:.0f}"

    return f"{fmt(lo)}+" if last else f"{fmt(lo)}–{fmt(hi)}"


def score_text(value: float) -> str:
    return f"{value:.3f}"


def range_text(lo: float, hi: float) -> str:
    return f"{lo:.3f}–{hi:.3f}"


def build(ctx):
    pr = ctx.load("pr_curves.npz")
    n = ctx.numbers
    errors = ctx.load("error_breakdown.json")
    test, dev = n["test_pr_auc"], n["dev_full_pr_auc"]

    def curve(key):
        return (np.asarray(pr[f"recall_{key}"], float), np.asarray(pr[f"precision_{key}"], float),
                np.asarray(pr[f"band_lo_{key}"], float), np.asarray(pr[f"band_hi_{key}"], float))

    curves = {m: curve(m) for m in NARRATED}
    dev_curves = {m: curve(f"dev_{m}") for m in DEV_MODELS}

    def end_labels(cs):
        ends = [c[1][-1] for c in cs.values()]
        return dict(zip(cs, spread_labels(ends, LABEL_GAP, Y_LIM[0] + 0.02, 0.97),
                        strict=True))

    label_y = end_labels(curves)
    dev_label_y = end_labels(dev_curves)

    bars = [[(bin_label(b["bin"], i == len(rows) - 1), b["recall"]) for i, b in enumerate(rows)]
            for rows in (errors["by_snr"], errors["by_period"])]
    n_fp, n_fp_co = int(n["cnn_fp_accepted"]), int(n["cnn_fp_accepted_centroid"])
    n_dev, n_cnn, n_full = int(n["n_dev"]), int(n["n_dev_cnn"]), int(n["n_dev_cnn_full"])

    # timing, tied to the spoken words
    def said(phrase, default):
        return cue_fraction(ctx, phrase, default)

    f_draw = {"tabular": said("On the final exam", 0.0) + 0.01,
              "tabular_lc": said("Without the image", 0.14),
              "cnn": said("The neural network", 0.26)}
    f_score = {m: said(score_text(test[m][0]), f_draw[m] + 0.08) for m in f_draw}
    # not narrated: joins the table (row only, no curve) once the network's score is said
    f_draw["tabular_lc_sub"] = f_score["cnn"] + 0.012
    f_score["tabular_lc_sub"] = f_draw["tabular_lc_sub"] + 0.012
    f_misses = said("It misses", 0.4)
    f_faint, f_short = said("faint", 0.42), said("short-orbit", 0.44)
    f_half = said("About half", 0.48)
    f_neighbour = said("neighbouring", 0.54)
    f_reveal = said("which a light curve", 0.58)
    f_learned = said("It learned", 0.64)
    f_retrain = said("Retrained", 0.715)
    # the retrained network first ("it scores ..."), then the catalogue model ("against ...")
    f_dev = {"cnn_full": f_retrain + 0.01,
             "tabular_lc": said("against", f_retrain + 0.14) - 0.04}
    f_dev_score = {m: said(score_text(dev[m][0]), f_dev[m] + 0.1) for m in DEV_MODELS}

    def a_in(f, start, dur=FADE):
        return style.ease(style.progress(f, start, start + dur))

    def a_out(f, start, dur=FADE):
        return 1.0 - style.ease(style.progress(f, start, start + dur))

    def draw(fig, t):
        f = t / ctx.duration
        style.starfield(fig.add_axes((0, 0, 1, 1)), 8, t)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 9)
        ax.set_axis_off()

        a_test = a_out(f, f_misses)                     # page 1: test curves
        a_bars = a_in(f, f_misses + 0.005) * a_out(f, f_retrain)
        a_dev = a_in(f, f_retrain)
        a_table = a_out(f, f_half - 0.005)              # right: test table
        a_fp = a_in(f, f_half) * a_out(f, f_learned - 0.005)
        a_data = a_in(f, f_learned)

        heads = [("Final exam: how well each model ranks planets", a_out(f, f_misses)),
                 ("Final exam: what the neural network misses",
                  a_in(f, f_misses) * a_out(f, f_retrain)),
                 ("Retrained: practice groups (development folds)", a_in(f, f_retrain))]
        for text, a in heads:
            if a > 0:
                style.heading(fig, text, alpha=a)

        if a_test > 0:
            pax = _pr_axes(fig, a_test)
            for m in NARRATED:
                _curve(pax, curves[m], m, label_y[m], f, f_draw[m], a_test, dim=_dim(f, m))
        if a_dev > 0:
            pax = _pr_axes(fig, a_dev)
            for m in DEV_MODELS:
                _curve(pax, dev_curves[m], m, dev_label_y[m], f, f_dev[m], a_dev)
            pax.text(0.03, 0.05, "practice groups (development folds)", transform=pax.transAxes,
                     ha="left", va="bottom", size=style.LABEL_SIZE - 4, color=style.C_ACCENT,
                     alpha=a_dev)
        if a_table > 0:
            _test_table(ax, f, a_table)
        if a_bars > 0:
            _recall_bars(ax, f, a_bars)
        if a_fp > 0:
            _false_positives(ax, f, a_fp)
        if a_data > 0:
            _training_data(ax, f, a_data)

    def _dim(f, m):
        """A curve dims once the next model is talked about."""
        if m == NARRATED[-1]:
            return 1.0
        return 1.0 - 0.55 * a_in(f, f_draw[NARRATED[NARRATED.index(m) + 1]])

    def _pr_axes(fig, alpha):
        x0, y0, w, h = PLOT
        pax = fig.add_axes((x0 / 16, y0 / 9, w / 16, h / 9))
        pax.set_xlim(0, 1.0)
        pax.set_ylim(*Y_LIM)
        ticks = [0.4, 0.6, 0.8, 1.0]
        pax.set_yticks(ticks, [f"{v:.0%}" for v in ticks])
        pax.set_xticks([0, 0.5, 1.0], ["0%", "50%", "100%"])
        pax.tick_params(length=6, width=1.5)
        for side in ("left", "bottom"):
            pax.spines[side].set_alpha(alpha)
        for label in (*pax.get_xticklabels(), *pax.get_yticklabels()):
            label.set_alpha(alpha)
        pax.grid(True, axis="y", alpha=0.15 * alpha)
        pax.text(0.5, -0.17, "planets found (recall)", transform=pax.transAxes, ha="center",
                 va="top", size=style.LABEL_SIZE - 2, color=style.FG, alpha=alpha)
        pax.text(-0.02, 1.04, "share of picks that are planets (precision)",
                 transform=pax.transAxes, ha="left", va="bottom", size=style.LABEL_SIZE - 2,
                 color=style.FG, alpha=alpha)
        return pax

    def _curve(pax, c, m, y_label, f, start, alpha, dim=1.0):
        p = style.ease(style.progress(f, start, start + DRAW))
        if p <= 0:
            return
        recall, precision, lo, hi = c
        keep = recall <= p + 1e-9
        a = alpha * dim
        pax.fill_between(recall[keep], lo[keep], hi[keep], color=COLOURS[m], alpha=0.16 * a,
                         linewidth=0)
        pax.plot(recall[keep], precision[keep], color=COLOURS[m], ls=DASHES[m], lw=3.0,
                 alpha=a, gid=f"pr_{m}", clip_on=False)
        a_label = a * style.ease(style.progress(p, 0.9, 1.0))
        if a_label > 0:
            pax.plot([1.0, 1.035], [precision[-1], y_label], color=COLOURS[m], lw=1.2,
                     alpha=a_label, clip_on=False)
            pax.text(1.045, y_label, NAMES[m][2], ha="left", va="center",
                     size=style.LABEL_SIZE - 4, color=COLOURS[m], alpha=a_label,
                     weight="semibold")

    def _row(ax, y, m, value, lo, hi, a_name, a_score):
        colour = COLOURS[m]
        if m in DASHES:                                  # a swatch only for drawn curves
            ax.plot([PANEL_X, PANEL_X + SWATCH_W], [y, y], color=colour, ls=DASHES[m],
                    lw=3.0, alpha=a_name, solid_capstyle="butt")
        name, reads, _ = NAMES[m]
        ax.text(NAME_X, y + 0.12, name, ha="left", va="center", size=style.LABEL_SIZE,
                weight="semibold", color=colour, alpha=a_name)
        ax.text(NAME_X, y - 0.3, reads, ha="left", va="center", size=style.LABEL_SIZE - 4,
                color=style.MUTED, alpha=a_name)
        if a_score > 0:
            ax.text(PANEL_R, y + 0.12, score_text(value), ha="right", va="center",
                    size=style.BODY_SIZE, weight="bold", color=style.FG, alpha=a_score)
            ax.text(PANEL_R, y - 0.3, range_text(lo, hi), ha="right", va="center",
                    size=style.LABEL_SIZE - 4, color=style.MUTED, alpha=a_score)

    def _table_header(ax, title, sub, alpha):
        ax.text(PANEL_X, 7.05, title, ha="left", va="center", size=style.LABEL_SIZE,
                weight="semibold", color=style.C_ACCENT, alpha=alpha)
        ax.text(PANEL_X, 6.65, sub, ha="left", va="center", size=style.LABEL_SIZE - 4,
                color=style.MUTED, alpha=alpha)
        ax.text(PANEL_R, 6.65, "score  (likely range)", ha="right", va="center",
                size=style.LABEL_SIZE - 4, color=style.MUTED, alpha=alpha)

    def _test_table(ax, f, alpha):
        _table_header(ax, "Final exam (held-out test set)", "1 = perfect ranking", alpha)
        for y, m in zip(ROW_YS, TEST_MODELS, strict=True):
            a_name = alpha * a_in(f, f_draw[m])
            if a_name > 0:
                _row(ax, y, m, *test[m], a_name, alpha * a_in(f, f_score[m] - 0.005))

    def _recall_bars(ax, f, alpha):
        ax.text(CHART_X0[0], 7.05, "Real planets the network flags", ha="left", va="center",
                size=style.LABEL_SIZE, weight="semibold", color=style.FG, alpha=alpha)
        ax.text(CHART_X0[0], 6.65, "share of confirmed planets in each group, final exam",
                ha="left", va="center", size=style.LABEL_SIZE - 4, color=style.MUTED,
                alpha=alpha)
        grow = style.ease(style.progress(f, f_misses + 0.005, f_misses + 0.04))
        charts = (("by signal strength", "fainter  →  stronger", f_faint),
                  ("by orbit length (days)", "shorter  →  longer", f_short))
        for x0, rows, (title, axis, f_hit) in zip(CHART_X0, bars, charts, strict=True):
            mid = x0 + (len(rows) * BAR_STEP - (BAR_STEP - BAR_W)) / 2
            ax.text(mid, 6.05, title, ha="center", va="center", size=style.LABEL_SIZE - 2,
                    color=style.FG, alpha=alpha)
            hit = a_in(f, f_hit - 0.005, 0.02)
            for i, (label, recall) in enumerate(rows):
                x = x0 + i * BAR_STEP
                a_bar = alpha * (1.0 - 0.55 * hit * (i > 0))  # the faintest / shortest stand out
                ax.add_patch(Rectangle((x, BAR_BASE), BAR_W, recall * BAR_SCALE * grow,
                                       facecolor=style.C_NET, edgecolor="none", alpha=a_bar,
                                       gid="recall_bar"))
                if i == 0 and hit > 0:
                    ax.add_patch(Rectangle((x, BAR_BASE), BAR_W, recall * BAR_SCALE * grow,
                                           facecolor="none", edgecolor=style.C_ACCENT, lw=2.5,
                                           alpha=alpha * hit))
                ax.text(x + BAR_W / 2, BAR_BASE + recall * BAR_SCALE * grow + 0.22,
                        f"{recall:.0%}", ha="center", va="center", size=style.LABEL_SIZE - 4,
                        weight="semibold", color=style.FG, alpha=alpha * (grow > 0.95))
                ax.text(x + BAR_W / 2, BAR_BASE - 0.25, label, ha="center", va="center",
                        size=style.LABEL_SIZE - 6, color=style.MUTED, alpha=alpha)
            ax.plot([x0 - 0.1, x0 + len(rows) * BAR_STEP - (BAR_STEP - BAR_W) + 0.1],
                    [BAR_BASE, BAR_BASE], color=style.MUTED, lw=1.5, alpha=alpha)
            ax.text(mid, BAR_BASE - 0.7, axis, ha="center", va="center",
                    size=style.LABEL_SIZE - 4, color=style.FG, alpha=alpha)

    def _false_positives(ax, f, alpha):
        ax.text(PANEL_X, 7.05, "False alarms the network let through", ha="left",
                va="center", size=style.LABEL_SIZE, weight="semibold", color=style.FG,
                alpha=alpha)
        ax.text(PANEL_X, 6.65, "false positives scored above its cut-off, final exam",
                ha="left", va="center", size=style.LABEL_SIZE - 4, color=style.MUTED,
                alpha=alpha)
        per_row = 10
        i = np.arange(n_fp)
        xs = PANEL_X + 0.25 + (i % per_row) * 0.6
        ys = 5.95 - (i // per_row) * 0.6
        pop = np.clip((f - f_half - i * 0.0015) / 0.01, 0, 1)
        ax.scatter(xs, ys, s=260, color=style.C_FALSE, alpha=alpha * pop, linewidths=0,
                   gid="fp_dots")
        a_co = alpha * a_in(f, f_neighbour - 0.005, 0.02)
        if a_co > 0:
            ax.scatter(xs[:n_fp_co], ys[:n_fp_co], s=620, facecolors="none",
                       edgecolors=style.C_ACCENT, linewidths=2.5, alpha=a_co, gid="fp_centroid")
            ax.text(PANEL_X, 4.55, f"{n_fp_co} of {n_fp}: light from\na neighbouring star",
                    ha="left", va="center", size=style.BODY_SIZE, weight="semibold",
                    color=style.C_ACCENT, alpha=a_co, linespacing=1.2)
        a_rev = alpha * a_in(f, f_reveal, 0.02)
        if a_rev > 0:
            ax.text(PANEL_X, 3.3, "a light curve alone\ncannot reveal this", ha="left",
                    va="center", size=style.BODY_SIZE, color=style.FG, alpha=a_rev,
                    linespacing=1.2)

    def _training_data(ax, f, alpha):
        ax.text(PANEL_X, 7.05, "Practice signals the network learned from", ha="left",
                va="center", size=style.LABEL_SIZE, weight="semibold", color=style.FG,
                alpha=alpha)
        width = PANEL_R - PANEL_X
        full = style.ease(style.progress(f, f_retrain, f_retrain + 0.04))
        share = (n_cnn + (n_full - n_cnn) * full) / n_dev
        ax.add_patch(Rectangle((PANEL_X, 6.15), width, 0.42, facecolor="none",
                               edgecolor=style.MUTED, lw=1.5, alpha=alpha))
        ax.add_patch(Rectangle((PANEL_X, 6.15), width * share, 0.42, facecolor=style.C_NET,
                               edgecolor="none", alpha=alpha, gid="data_bar"))
        ax.text(PANEL_X, 5.7, f"first: {n_cnn:,} of {n_dev:,} ({n_cnn / n_dev:.0%})",
                ha="left", va="center", size=style.LABEL_SIZE - 2, color=style.FG,
                alpha=alpha * (1 - full))
        ax.text(PANEL_X, 5.7, "retrained on all of them", ha="left", va="center",
                size=style.LABEL_SIZE - 2, color=style.FG, alpha=alpha * full)

        a_tab = a_in(f, f_retrain)
        if a_tab > 0:
            ax.text(PANEL_X, 4.85, "Practice groups (development folds)", ha="left",
                    va="center", size=style.LABEL_SIZE, weight="semibold",
                    color=style.C_ACCENT, alpha=a_tab)
            ax.text(PANEL_X, 4.45, "not the final exam", ha="left", va="center",
                    size=style.LABEL_SIZE - 4, color=style.MUTED, alpha=a_tab)
            ax.text(PANEL_R, 4.45, "score  (likely range)", ha="right", va="center",
                    size=style.LABEL_SIZE - 4, color=style.MUTED, alpha=a_tab)
            for y, m in zip(DEV_ROW_YS, DEV_MODELS, strict=True):
                a_name = a_tab * a_in(f, f_dev[m])
                if a_name > 0:
                    _row(ax, y, m, *dev[m], a_name, a_tab * a_in(f, f_dev_score[m] - 0.005))

    return draw
