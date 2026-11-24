"""Scene 3, data and splits: the catalogue is split into a locked final exam (test set) and
five practice groups (dev folds).

Sixty sample stars (splits_sample.json) start in the catalogue and fly to their real
partition. A star with several signals is drawn as a ring holding one dot per signal; the
whole ring moves as one, then those stars are highlighted: host-star grouping.
"""

from __future__ import annotations

import numpy as np
import style
from matplotlib.colors import to_rgb
from matplotlib.patches import Arc, FancyBboxPatch, Rectangle

from scenes import cue_fraction

# layout in "inch" units of a 16 x 9 overlay axes
CATALOGUE = (0.8, 1.8, 3.6, 4.5)              # x0, y0, w, h
CAT_COLS, CAT_DX, CAT_DY = 6, 0.58, 0.4
TEST_BOX = (9.0, 5.35, 6.4, 2.0)
TEST_SLOTS_X0, TEST_COLS, TEST_DX, TEST_DY = 12.75, 5, 0.55, 0.5
FOLDS_X0, FOLDS_Y0, FOLD_W, FOLD_H, FOLD_GAP = 5.6, 1.7, 1.8, 2.7, 0.2
FOLD_COLS, FOLD_DX, FOLD_DY = 3, 0.5, 0.55
LOCK_XY = (8.3, 6.35)
RING_R, DOT_OFFSET = 0.19, 0.085               # one star's ring; its signal dots
DOT_S = 26.0                                   # points^2
FLIGHT = 0.07                                  # one star's flight, fraction of the scene
N_FOLDS = 5


def partition_boxes() -> dict[str, tuple[float, float, float, float]]:
    """Box (x0, y0, w, h) of every partition, keyed as in splits_sample.json."""
    boxes = {"test": TEST_BOX}
    for k in range(N_FOLDS):
        boxes[f"dev fold {k}"] = (FOLDS_X0 + k * (FOLD_W + FOLD_GAP), FOLDS_Y0, FOLD_W, FOLD_H)
    return boxes


def star_targets(sample: list[dict]) -> np.ndarray:
    """(n, 2) final position of every sample star: a free slot inside its partition box."""
    boxes = partition_boxes()
    used: dict[str, int] = {}
    out = np.empty((len(sample), 2))
    for i, star in enumerate(sample):
        part = star["partition"]
        j = used.get(part, 0)
        used[part] = j + 1
        x0, y0, w, h = boxes[part]
        if part == "test":
            out[i] = (TEST_SLOTS_X0 + (j % TEST_COLS) * TEST_DX,
                      y0 + h - 0.5 - (j // TEST_COLS) * TEST_DY)
        else:
            out[i] = (x0 + 0.4 + (j % FOLD_COLS) * FOLD_DX,
                      y0 + h - 0.75 - (j // FOLD_COLS) * FOLD_DY)
    return out


def _catalogue_slots(n: int) -> np.ndarray:
    x0, y0, w, h = CATALOGUE
    j = np.arange(n)
    return np.column_stack([x0 + 0.35 + (j % CAT_COLS) * CAT_DX,
                            y0 + h - 0.6 - (j // CAT_COLS) * CAT_DY])


def _signal_offsets(n_kois: int) -> np.ndarray:
    """Positions of a star's signal dots relative to the star."""
    if n_kois == 1:
        return np.zeros((1, 2))
    a = np.pi / 2 + 2 * np.pi * np.arange(n_kois) / n_kois
    return DOT_OFFSET * np.column_stack([np.cos(a), np.sin(a)])


def _lock(ax, xy, closed: float, alpha: float) -> None:
    """A padlock; the shackle drops into the body as closed goes 0 -> 1."""
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - 0.38, y - 0.55), 0.76, 0.6,
                                boxstyle="round,pad=0,rounding_size=0.08",
                                facecolor=style.C_ACCENT, edgecolor="none", alpha=alpha))
    lift = 0.35 * (1 - closed)
    ax.add_patch(Arc((x, y + 0.05 + lift), 0.5, 0.75, theta1=0, theta2=180,
                     color=style.C_ACCENT, lw=5, alpha=alpha))
    ax.plot([x - 0.25, x - 0.25], [y + 0.05 + lift, y + 0.05], color=style.C_ACCENT, lw=5,
            alpha=alpha, solid_capstyle="butt")
    ax.plot([x + 0.25, x + 0.25], [y + 0.05 + lift, y + 0.05], color=style.C_ACCENT, lw=5,
            alpha=alpha, solid_capstyle="butt")
    ax.add_patch(Rectangle((x - 0.05, y - 0.38), 0.1, 0.2, facecolor=style.BG,
                           edgecolor="none", alpha=alpha))


def build(ctx):
    n = ctx.numbers
    sample = ctx.load("splits_sample.json")
    n_stars = len(sample)
    start = _catalogue_slots(n_stars)
    end = star_targets(sample)
    is_test = np.array([s["partition"] == "test" for s in sample])
    multi = np.array([s["n_kois"] > 1 for s in sample])
    offsets = [_signal_offsets(s["n_kois"]) for s in sample]
    owner = np.concatenate([np.full(len(o), i) for i, o in enumerate(offsets)])
    dot_offsets = np.concatenate(offsets)
    boxes = partition_boxes()

    # timing, tied to the spoken words
    f_locked = cue_fraction(ctx, "locked", 0.05)
    f_marked = cue_fraction(ctx, "marked", 0.26)
    f_other = cue_fraction(ctx, "The other", 0.4)
    f_five = cue_fraction(ctx, "five", 0.55)
    f_each = cue_fraction(ctx, "Each star", 0.65)
    f_recog = cue_fraction(ctx, "recognising", 0.82)
    test_span = (f_locked + 0.01, f_marked - 0.04)
    dev_span = (f_other + 0.01, max(f_five + 0.04, f_other + 0.12))

    # each star departs in queue order within its group
    depart = np.empty(n_stars)
    for mask, (s0, s1) in ((is_test, test_span), (~is_test, dev_span)):
        k = mask.sum()
        depart[mask] = s0 + (s1 - FLIGHT - s0) * np.arange(k) / max(k - 1, 1)

    def draw(fig, t):
        f = t / ctx.duration
        style.starfield(fig.add_axes((0, 0, 1, 1)), 3, t)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 9)
        ax.set_axis_off()

        a_in = style.ease(style.progress(f, 0.0, 0.04))
        fig.text(style.MARGIN, style.HEADING_Y, "Splitting the data", ha="left", va="top",
                 size=style.HEADING_SIZE, weight="semibold", color=style.FG, alpha=a_in)

        # --- catalogue (fades once it is empty) ------------------------------------------
        a_cat = a_in * (1 - style.ease(style.progress(f, dev_span[1], dev_span[1] + 0.03)))
        if a_cat > 0:
            x0, y0, w, h = CATALOGUE
            ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                        facecolor=style.TERM_BG, edgecolor=style.MUTED, lw=1.5,
                                        alpha=a_cat))
            ax.add_patch(Rectangle((x0, y0 + h - 0.32), w, 0.32, facecolor=style.MUTED,
                                   alpha=0.25 * a_cat, edgecolor="none"))
            ax.text(x0, 7.05, "Kepler catalogue", ha="left", va="center", size=style.BODY_SIZE,
                    color=style.FG, alpha=a_cat)
            ax.text(x0, 6.6, f"{n['n_kois']:,} signals, {n_stars} sample stars shown",
                    ha="left", va="center", size=style.LABEL_SIZE - 2, color=style.MUTED,
                    alpha=a_cat)

        # --- final exam ------------------------------------------------------------------
        a_test = style.ease(style.progress(f, f_locked - 0.02, f_locked + 0.03))
        if a_test > 0:
            x0, y0, w, h = boxes["test"]
            ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                        facecolor=style.TERM_BG, edgecolor=style.C_ACCENT,
                                        lw=2, alpha=a_test))
            ax.text(x0 + 0.25, y0 + h - 0.4, "final exam", ha="left", va="center",
                    size=style.BODY_SIZE, weight="semibold", color=style.C_ACCENT,
                    alpha=a_test)
            arrived = (f >= depart + FLIGHT) & is_test
            count = round(n["n_test"] * arrived.sum() / max(is_test.sum(), 1))
            ax.text(x0 + 0.25, y0 + h - 0.95, f"{count:,} signals", ha="left", va="center",
                    size=style.LABEL_SIZE, color=style.FG, alpha=a_test)
            a_once = a_test * style.ease(style.progress(f, f_marked, f_marked + 0.04))
            ax.text(x0 + 0.25, y0 + h - 1.45, "marked once, at the end", ha="left",
                    va="center", size=style.LABEL_SIZE - 2, color=style.MUTED, alpha=a_once)
            closed = style.ease(style.progress(f, test_span[1], test_span[1] + 0.04))
            _lock(ax, LOCK_XY, closed, a_test)

        # --- practice groups ---------------------------------------------------------------
        a_dev = style.ease(style.progress(f, f_other - 0.03, f_other + 0.02))
        if a_dev > 0:
            arrived = (f >= depart + FLIGHT) & ~is_test
            count = round(n["n_dev"] * arrived.sum() / max((~is_test).sum(), 1))
            ax.text(FOLDS_X0, FOLDS_Y0 + FOLD_H + 0.42,
                    f"{count:,} signals, five practice groups", ha="left", va="center",
                    size=style.BODY_SIZE, color=style.FG, alpha=a_dev)
            for k in range(N_FOLDS):
                x0, y0, w, h = boxes[f"dev fold {k}"]
                ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                                            boxstyle="round,pad=0,rounding_size=0.12",
                                            facecolor=style.TERM_BG, edgecolor=style.MUTED,
                                            lw=1.5, alpha=a_dev))
                ax.text(x0 + w / 2, y0 + h - 0.3, f"group {k + 1}", ha="center", va="center",
                        size=style.LABEL_SIZE - 2, color=style.MUTED, alpha=a_dev)

        # --- the sample stars ---------------------------------------------------------------
        p = np.clip((f - depart) / FLIGHT, 0.0, 1.0)
        p = p * p * (3 - 2 * p)
        xy = start + (end - start) * p[:, None]
        xy[:, 1] += 0.6 * np.sin(np.pi * p)                # arc over the boxes
        hi = style.ease(style.progress(f, f_each, f_each + 0.04))
        star_alpha = a_in * np.where(multi, 1.0, 1.0 - 0.6 * hi)
        dots = xy[owner] + dot_offsets
        dot_rgba = np.tile(np.append(to_rgb(style.FG), 1.0), (len(owner), 1))
        dot_rgba[multi[owner], :3] = (1 - hi) * np.array(to_rgb(style.FG)) + hi * np.array(
            to_rgb(style.C_ACCENT))
        dot_rgba[:, 3] = star_alpha[owner]
        ax.scatter(dots[:, 0], dots[:, 1], s=DOT_S, c=dot_rgba, linewidths=0, zorder=5)
        ring_rgb = (1 - hi) * np.array(to_rgb(style.MUTED)) + hi * np.array(to_rgb(style.C_ACCENT))
        ax.scatter(xy[multi, 0], xy[multi, 1], s=(2 * RING_R * 72) ** 2, facecolors="none",
                   edgecolors=[(*ring_rgb, a_in)], linewidths=1.5 + 1.0 * hi, zorder=5)

        # --- host-star grouping, explained where the catalogue was -------------------------
        a_note = style.ease(style.progress(f, f_each, f_each + 0.04))
        if a_note > 0:
            x0 = CATALOGUE[0]
            ax.scatter([x0 + 0.25], [6.85], s=(2 * RING_R * 72) ** 2, facecolors="none",
                       edgecolors=[(*to_rgb(style.C_ACCENT), a_note)], linewidths=2.5)
            off = _signal_offsets(3)
            ax.scatter(x0 + 0.25 + off[:, 0], 6.85 + off[:, 1], s=DOT_S,
                       color=style.C_ACCENT, alpha=a_note, linewidths=0)
            ax.text(x0 + 0.6, 6.85, "one star, several signals", ha="left", va="center",
                    size=style.LABEL_SIZE, color=style.C_ACCENT, alpha=a_note)
            ax.text(x0, 5.9, "Each star stays\nin one group", ha="left", va="top",
                    size=style.BODY_SIZE, weight="semibold", color=style.FG, alpha=a_note,
                    linespacing=1.2)
            a_why = style.ease(style.progress(f, f_recog - 0.04, f_recog))
            ax.text(x0, 4.3, "so a model can't pass\nby recognising the star\n"
                    "instead of the signal", ha="left", va="top", size=style.LABEL_SIZE,
                    color=style.MUTED, alpha=a_why, linespacing=1.3)

    return draw

