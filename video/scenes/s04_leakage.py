"""Scene 4, the leakage audit: a model that copies the answer key.

One tile per catalogue column (audit_columns.json) sorts into "kept" and "removed". The six
vetting columns (NASA's own verdicts) are named. When the score is introduced, a bar grows
to the audited score of the light-curve-comparable model (its dip + star measurements). The
six lift out of "removed" and wait; they land in "kept" as the bar jumps, in the warning
colour, to a too-good score; then they are removed again and the bar returns.
"""

from __future__ import annotations

import numpy as np
import style
from matplotlib.colors import to_rgb
from matplotlib.patches import Rectangle

from scenes import cue_fraction

# layout in "inch" units of a 16 x 9 overlay axes
DX = 0.42                          # tile spacing
TILE_S = 23.0 ** 2                 # tile marker area, points^2 (about 0.32 in wide)
GRID_COLS, GRID_X0 = 17, 4.1       # all columns, before the audit
TOP_Y = 6.6                        # top tile row
KEPT_COLS, KEPT_X0 = 7, 0.9        # kept block; its last row has room for the six
REMOVED_COLS, REMOVED_X0 = 10, 4.6
HOVER_X = 4.02                     # the six wait in the gap between the blocks
NAMES_XY, NAMES_DY = (0.9, 3.0), 0.27
LABEL_Y = 7.05
BAR_BOX = (0.665, 2.4 / 9, 0.12, 4.2 / 9)   # score axes, figure fractions
TEXT_X = 12.85                               # explanation right of the bar
FLIGHT = 0.04                                # one tile's move, fraction of the scene

VETTING = ("koi_score", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
           "koi_pdisposition")


def vetting_columns(rows: list[dict]) -> list[str]:
    """The six vetting columns (Kepler's own disposition outputs), in audit order."""
    found = [r["column"] for r in rows if r["column"] in VETTING]
    if len(found) != len(VETTING):
        raise ValueError(f"vetting columns missing from the audit: {set(VETTING) - set(found)}")
    return found


def _slots(n: int, cols: int, x0: float, top: float = TOP_Y) -> np.ndarray:
    i = np.arange(n)
    return np.column_stack([x0 + (i % cols) * DX, top - (i // cols) * DX])


def _move(f, start, delay):
    """Per-tile eased 0-1 progress of a move starting at start + delay."""
    p = np.clip((f - start - delay) / FLIGHT, 0.0, 1.0)
    return p * p * (3 - 2 * p)


def build(ctx):
    rows = ctx.load("audit_columns.json")
    n = ctx.numbers
    audited, leaked = float(n["leak_lc_audited"]), float(n["leak_lc_plus_vetting"])
    n_lc = int(n["leak_lc_n_features"])                      # the light-curve-comparable
    n_added = int(n["leak_lc_plus_vetting_n_features"]) - n_lc   # model's inputs
    names = vetting_columns(rows)
    n_cols = len(rows)
    is_vet = np.array([r["column"] in VETTING for r in rows])
    kept = np.array([bool(r["kept"]) for r in rows])
    n_kept, n_removed = int(kept.sum()), int((~kept).sum())

    # positions: the full grid, the audit's two blocks, and the six back in the kept block
    grid = _slots(n_cols, GRID_COLS, GRID_X0)
    sort = np.empty_like(grid)
    sort[kept] = _slots(n_kept, KEPT_COLS, KEPT_X0)
    order = np.argsort(~is_vet[~kept], kind="stable")       # the six first, top row
    removed_slots = np.empty((n_removed, 2))
    removed_slots[order] = _slots(n_removed, REMOVED_COLS, REMOVED_X0)
    sort[~kept] = removed_slots
    back_in = _slots(n_kept + len(names), KEPT_COLS, KEPT_X0)[n_kept:]
    vet_idx = np.flatnonzero(is_vet)
    hover = np.column_stack([np.full(len(names), HOVER_X), TOP_Y - np.arange(len(names)) * DX])
    draw_order = np.argsort(is_vet, kind="stable")           # the six on top
    delay = np.random.default_rng(4).uniform(0, 0.05, n_cols)
    appear = np.random.default_rng(5).uniform(0.01, 0.07, n_cols)

    fg, muted, warn, accent = (np.array(to_rgb(c)) for c in
                               (style.FG, style.MUTED, style.C_WARN, style.C_ACCENT))
    lo = np.floor(audited * 10 - 0.5) / 10                   # zoomed score axis: lo .. 1

    # timing, tied to the spoken words
    f_cols = cue_fraction(ctx, "catalogue has", 0.11)
    f_verdict = cue_fraction(ctx, "verdicts", 0.2)
    f_answer = cue_fraction(ctx, "answer we want", 0.3)
    f_six = cue_fraction(ctx, "six verdict", 0.44)
    f_score = cue_fraction(ctx, "Our score", 0.5)
    f_ranks = cue_fraction(ctx, "ranks planets", 0.62)
    f_jump = cue_fraction(ctx, "jumps", 0.71)
    f_too = cue_fraction(ctx, "Too good", 0.8)
    f_model = cue_fraction(ctx, "The model was", 0.9)
    f_sort = f_answer                                        # tiles sort, done by "removed"
    f_grow0, f_grow1 = f_score, f_score + 0.06               # the bar appears with "score"
    f_land = f_jump - 0.015                                  # the six land as the bar jumps
    f_copy = f_too + 0.04                                    # the warning's second line
    f_out = min(f_model + 0.015, 0.93)                       # the six leave again
    f_settle = f_out + 0.01                                  # bar returns

    def draw(fig, t):
        f = t / ctx.duration
        style.starfield(fig.add_axes((0, 0, 1, 1)), 4, t)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 9)
        ax.set_axis_off()
        style.heading(fig, "A safety check: the leakage audit")

        _draw_tiles(ax, f, t)
        _draw_labels(ax, f)
        if f >= f_grow0:
            _draw_score(fig, ax, f)

    def _draw_tiles(ax, f, t):
        p_sort = _move(f, f_sort, delay)
        xy = grid + (sort - grid) * p_sort[:, None]
        # the six: lift out to wait between the blocks, land in "kept" as the score jumps,
        # then go back to "removed"
        steps = np.arange(len(names))
        p_lift = _move(f, f_six, steps * 0.004)
        p_land = _move(f, f_land, steps * 0.003)
        p_out = _move(f, f_out, steps * 0.002)
        bob = 0.04 * np.sin(2 * np.pi * 0.7 * t + steps) * p_lift * (1 - p_land)  # ambient
        vet_xy = sort[vet_idx] + (hover - sort[vet_idx]) * p_lift[:, None]
        vet_xy = vet_xy + (back_in - vet_xy) * p_land[:, None]
        vet_xy = vet_xy + (sort[vet_idx] - vet_xy) * p_out[:, None]
        vet_xy[:, 1] += bob
        xy[vet_idx] = vet_xy
        p_vet = np.maximum(p_lift, p_land) * (1 - p_out)

        rgb = np.where(kept[:, None], fg, fg + (muted - fg) * p_sort[:, None])
        alpha = 0.8 - np.where(kept, 0.0, 0.45 * p_sort)
        a_vet = style.ease(style.progress(f, f_verdict - 0.01, f_verdict + 0.03))
        rgb[vet_idx] = rgb[vet_idx] + (warn - rgb[vet_idx]) * a_vet
        alpha[vet_idx] = 0.8 + 0.2 * a_vet - 0.4 * (p_sort[vet_idx] - p_vet)
        alpha = alpha * np.clip((f - appear) / 0.02, 0.0, 1.0)
        rgba = np.column_stack([rgb, alpha])[draw_order]
        ax.scatter(xy[draw_order, 0], xy[draw_order, 1], s=TILE_S, marker="s", linewidths=0,
                   c=rgba)

        a_names = style.ease(style.progress(f, f_verdict, f_verdict + 0.04))
        if a_names > 0:
            x0, y0 = NAMES_XY
            ax.text(x0 - 0.15, y0 + 0.45, "NASA's verdicts:", ha="left", va="center",
                    size=style.LABEL_SIZE, color=style.C_WARN, alpha=a_names)
            for i, name in enumerate(names):
                ax.text(x0, y0 - i * NAMES_DY, name, ha="left", va="center",
                        family="monospace", size=style.LABEL_SIZE - 4, color=style.C_WARN,
                        alpha=a_names)

    def _draw_labels(ax, f):
        a_all = (style.ease(style.progress(f, f_cols - 0.01, f_cols + 0.03))
                 * (1 - style.ease(style.progress(f, f_sort - 0.02, f_sort + 0.01))))
        if a_all > 0:
            ax.text(GRID_X0 + (GRID_COLS - 1) * DX / 2, LABEL_Y, f"{n_cols}", ha="right",
                    va="center", size=style.BODY_SIZE, weight="semibold", alpha=a_all)
            ax.text(GRID_X0 + (GRID_COLS - 1) * DX / 2 + 0.12, LABEL_Y,
                    "catalogue columns, one tile each", ha="left", va="center",
                    size=style.BODY_SIZE, color=style.MUTED, alpha=a_all)
        a_blocks = style.ease(style.progress(f, f_sort + 0.06, f_sort + 0.09))
        if a_blocks > 0:
            kx = KEPT_X0 + (KEPT_COLS - 1) * DX / 2
            ax.text(kx - 0.1, LABEL_Y, f"{n_kept}", ha="right", va="center",
                    size=style.BODY_SIZE, weight="semibold", color=style.FG, alpha=a_blocks)
            ax.text(kx, LABEL_Y, "kept", ha="left", va="center", size=style.BODY_SIZE,
                    color=style.FG, alpha=a_blocks)
            a_plus = (style.ease(style.progress(f, f_land + 0.02, f_land + 0.04))
                      * (1 - style.ease(style.progress(f, f_out, f_out + 0.02))))
            if a_plus > 0:
                ax.text(kx + 0.95, LABEL_Y, f"+ {len(names)} verdicts", ha="left",
                        va="center", size=style.LABEL_SIZE, weight="semibold",
                        color=style.C_WARN, alpha=a_plus)
            rx = REMOVED_X0 + (REMOVED_COLS - 1) * DX / 2
            ax.text(rx - 0.1, LABEL_Y, f"{n_removed}", ha="right", va="center",
                    size=style.BODY_SIZE, weight="semibold", color=style.MUTED,
                    alpha=a_blocks)
            ax.text(rx, LABEL_Y, "removed", ha="left", va="center", size=style.BODY_SIZE,
                    color=style.MUTED, alpha=a_blocks)

    def _draw_score(fig, ax, f):
        a = style.ease(style.progress(f, f_grow0, f_grow0 + 0.02))
        grow = style.ease(style.progress(f, f_grow0, f_grow1))
        jump = style.ease(style.progress(f, f_jump, f_jump + 0.035))
        back = style.ease(style.progress(f, f_settle, f_settle + 0.025))
        p_warn = jump * (1 - back)
        value = lo + (audited - lo) * grow + (leaked - audited) * p_warn
        colour = fg + (warn - fg) * p_warn

        bax = fig.add_axes(BAR_BOX)
        bax.set_xlim(0, 1)
        bax.set_ylim(lo, 1.0)
        bax.spines["bottom"].set_visible(False)
        bax.spines["left"].set_alpha(a)
        bax.set_xticks([])
        ticks = np.round(np.arange(lo, 1.0 + 1e-9, 0.05), 2)
        bax.set_yticks(ticks, [f"{v:.2f}" for v in ticks], alpha=a)
        bax.tick_params(axis="y", length=6, width=1.5, colors=style.MUTED, labelcolor=style.FG)
        bax.add_patch(Rectangle((0.22, lo), 0.56, value - lo, facecolor=colour,
                                edgecolor="none", alpha=a, gid="score_bar"))
        a_perfect = a * style.ease(style.progress(f, f_score, f_score + 0.03))
        bax.axhline(1.0, color=style.C_ACCENT, lw=2, ls=(0, (4, 3)), alpha=a_perfect)

        # the zoom, said plainly: a break mark on the axis and its range in words
        x_axis = BAR_BOX[0] * 16
        y_base = BAR_BOX[1] * 9
        for dy in (0.18, 0.3):
            ax.plot([x_axis - 0.12, x_axis + 0.12], [y_base + dy - 0.06, y_base + dy + 0.06],
                    color=style.FG, lw=2, alpha=a)
        ax.text(x_axis + BAR_BOX[2] * 8, y_base - 0.3,
                f"axis zoomed in: {lo:.2f} to {1:.2f}", ha="center", va="center",
                size=style.LABEL_SIZE - 4, color=style.MUTED, alpha=a)
        ax.text(x_axis + BAR_BOX[2] * 8, y_base + BAR_BOX[3] * 9 + 0.35, "score", ha="center",
                va="center", size=style.BODY_SIZE, weight="semibold", alpha=a)

        # the number is shown only while the bar rests on a measured score, never in between
        at_rest = grow >= 1.0 and p_warn in (0.0, 1.0)
        if at_rest:
            score = leaked if p_warn == 1.0 else audited
            bax.text(0.5, score - 0.006, f"{score:.3f}", ha="center", va="top",
                     size=style.BODY_SIZE, weight="bold", color=style.BG, alpha=a,
                     gid="score_value")
        which = (f"this model uses {n_lc} of the {n_kept} kept columns" if p_warn < 0.5
                 else f"its {n_lc} columns + the {n_added} verdicts")
        ax.text(x_axis + BAR_BOX[2] * 8, y_base - 0.7, which, ha="center", va="center",
                size=style.LABEL_SIZE - 2,
                color=style.C_WARN if p_warn >= 0.5 else style.FG, alpha=a)

        a_expl = a * style.ease(style.progress(f, f_score, f_score + 0.03))
        ax.text(TEXT_X, 6.45, "1 = perfect", ha="left", va="center", size=style.LABEL_SIZE,
                color=style.C_ACCENT, alpha=a_expl)
        a_rank = a * style.ease(style.progress(f, f_ranks - 0.02, f_ranks + 0.02))
        ax.text(TEXT_X, 5.2, "how well a model\nranks planets\nabove false alarms",
                ha="left", va="center", size=style.LABEL_SIZE - 2, color=style.MUTED,
                linespacing=1.3, alpha=a_rank)

        a_too = (style.ease(style.progress(f, f_too - 0.01, f_too + 0.02))
                 * (1 - style.ease(style.progress(back, 0.0, 0.5))))
        a_honest = style.ease(style.progress(back, 0.5, 1.0))
        if a_too > 0:
            ax.text(TEXT_X, 3.75, "Too good\nto be true", ha="left", va="center",
                    size=style.BODY_SIZE, weight="bold", color=style.C_WARN, alpha=a_too,
                    linespacing=1.1)
            a_copy = a_too * style.ease(style.progress(f, f_copy - 0.03, f_copy))
            ax.text(TEXT_X, 2.65, "the model copied\nthe answer key", ha="left",
                    va="center", size=style.LABEL_SIZE, color=style.C_WARN, alpha=a_copy,
                    linespacing=1.2)
        if a_honest > 0:
            ax.text(TEXT_X, 3.4, "verdicts out:\nthe honest score", ha="left",
                    va="center", size=style.LABEL_SIZE, color=style.FG, alpha=a_honest,
                    linespacing=1.2)

    return draw
