"""Scene 9, what a score means: worth a follow-up, not a discovery.

A score dial from 0 (looks like a false positive) to 1 (looks like the signals that became
planets). The needle swings into the high end, but the dial shows no number: it stands for
any high score, not a particular star's. As the narration says "worth a follow-up, not a
discovery", that sentence takes the right half of the screen, with the README's "What the
scores mean" sentence (numbers.json score_meaning) beneath it, and stays to the end. The
dial then gives way to the three next steps (numbers.json next_steps, from
docs/evaluation.md), each appearing as it is spoken.
"""

from __future__ import annotations

import textwrap

import numpy as np
import style
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

from scenes import cue_fraction

# layout in inches of the 16 x 9 overlay axes (caption band: y < 1.35)
DIAL_C, DIAL_R = (4.1, 3.55), 2.4
HIGH = 0.86                         # where the needle settles: "high", not any star's score
RIGHT_X = 8.4
STEP_X, STEP_YS = 0.9, (5.75, 4.45, 3.15)
BIG_SIZE = 50                       # "Worth a follow-up, not a discovery."
READOUT_DY = -1.55                  # the dial's reading, below the end labels
_CMAP = LinearSegmentedColormap.from_list("score", [style.C_FALSE, style.C_PLANET])


def wrap_step(text: str, one_line: int = 26) -> str:
    """A next step as drawn: one line if short, else two lines of balanced length."""
    words = text.split()
    if len(text) <= one_line or len(words) < 2:
        return text
    splits = [(" ".join(words[:i]), " ".join(words[i:])) for i in range(1, len(words))]
    a, b = min(splits, key=lambda p: max(len(p[0]), len(p[1])))
    return f"{a}\n{b}"


def wrap_meaning(text: str) -> str:
    """The README sentence as drawn under the big sentence."""
    return textwrap.fill(text, 46, break_on_hyphens=False)


def dial_point(value: float, radius: float) -> tuple[float, float]:
    """Point on the dial at value (0 at the left end, 1 at the right), radius from centre."""
    angle = np.pi * (1.0 - value)
    return DIAL_C[0] + radius * np.cos(angle), DIAL_C[1] + radius * np.sin(angle)


def build(ctx):
    n = ctx.numbers
    steps = [wrap_step(s) for s in n["next_steps"]]
    meaning = wrap_meaning(n["score_meaning"])

    # timing, tied to the spoken words
    f_high = cue_fraction(ctx, "a high score", 0.02)
    f_looks = cue_fraction(ctx, "looks like", 0.12)
    f_worth = cue_fraction(ctx, "worth a follow-up", 0.31)
    f_not = cue_fraction(ctx, "not a discovery", 0.42)
    f_next = cue_fraction(ctx, "Next:", 0.49)
    f_steps = [cue_fraction(ctx, phrase, d) for phrase, d in
               (("combine the two", 0.53), ("find more faint", 0.62), ("use pixel", 0.74))]

    # the dial's arc, coloured from false-positive-like to planet-like
    v = np.linspace(0, 1, 61)
    pts = np.array([dial_point(x, DIAL_R) for x in v])
    segments = np.stack([pts[:-1], pts[1:]], axis=1)
    seg_colours = _CMAP((v[:-1] + v[1:]) / 2)

    def a_in(f, start, dur=0.03):
        return style.ease(style.progress(f, start, start + dur))

    def draw(fig, t):
        f = t / ctx.duration
        style.starfield(fig.add_axes((0, 0, 1, 1)), 9, t)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 9)
        ax.set_axis_off()
        style.heading(fig, "What a high score means")

        a_dial = a_in(f, 0.0, 0.04) * (1.0 - a_in(f, f_next - 0.035, 0.03))  # gone by "Next"
        if a_dial > 0:
            _dial(ax, f, t, a_dial)
        _sentences(ax, f)
        _steps(ax, f)

    def _dial(ax, f, t, alpha):
        colours = seg_colours.copy()
        colours[:, 3] = alpha
        ax.add_collection(LineCollection(segments, colors=colours, linewidths=16,
                                         capstyle="butt"))
        for value, text, colour in ((0.0, "0", style.C_FALSE), (1.0, "1", style.C_PLANET)):
            x, y = dial_point(value, DIAL_R)
            ax.text(x, y - 0.35, text, ha="center", va="center", size=style.BODY_SIZE,
                    weight="bold", color=colour, alpha=alpha)
        ax.text(DIAL_C[0] - DIAL_R, DIAL_C[1] - 0.95, "looks like a\nfalse positive",
                ha="center", va="center", size=style.LABEL_SIZE - 4, color=style.C_FALSE,
                alpha=alpha, linespacing=1.15)
        ax.text(DIAL_C[0] + DIAL_R, DIAL_C[1] - 0.95, "looks like the\nconfirmed planets",
                ha="center", va="center", size=style.LABEL_SIZE - 4, color=style.C_PLANET,
                alpha=alpha, linespacing=1.15)

        swing = style.ease(style.progress(f, f_high, f_high + 0.1))
        value = 0.5 + (HIGH - 0.5) * swing + 0.008 * np.sin(2 * np.pi * 0.4 * t) * swing
        tip = dial_point(value, DIAL_R * 0.84)
        ax.plot([DIAL_C[0], tip[0]], [DIAL_C[1], tip[1]], color=style.FG, lw=5, alpha=alpha,
                gid="needle")
        ax.scatter([DIAL_C[0]], [DIAL_C[1]], s=260, color=style.FG, alpha=alpha,
                   linewidths=0)

        a_reading = alpha * a_in(f, f_worth - 0.01, 0.03)
        a_high = alpha * a_in(f, f_high + 0.07, 0.03) * (1.0 - a_in(f, f_worth - 0.01, 0.03))
        ax.text(DIAL_C[0], DIAL_C[1] + READOUT_DY, "high score", ha="center", va="center",
                size=style.BODY_SIZE, color=style.FG, alpha=a_high)
        ax.text(DIAL_C[0], DIAL_C[1] + READOUT_DY, "worth a follow-up", ha="center",
                va="center",
                size=style.BODY_SIZE, weight="bold", color=style.C_ACCENT, alpha=a_reading)

    def _sentences(ax, f):
        a_looks = a_in(f, f_looks - 0.01) * (1.0 - a_in(f, f_worth - 0.02, 0.02))
        if a_looks > 0:
            ax.text(RIGHT_X, 5.2, "the signal looks like the ones\nthat became planets",
                    ha="left", va="center", size=style.BODY_SIZE + 2, color=style.FG,
                    alpha=a_looks, linespacing=1.3)
        a_worth = a_in(f, f_worth - 0.005)
        if a_worth > 0:
            ax.text(RIGHT_X, 6.2, "Worth a follow-up,", ha="left", va="center",
                    size=BIG_SIZE, weight="bold", color=style.C_ACCENT, alpha=a_worth)
        a_not = a_in(f, f_not - 0.005)
        if a_not > 0:
            ax.text(RIGHT_X, 5.05, "not a discovery.", ha="left", va="center", size=BIG_SIZE,
                    weight="bold", color=style.FG, alpha=a_not)
            ax.text(RIGHT_X, 3.55, meaning, ha="left", va="center", size=style.LABEL_SIZE,
                    color=style.MUTED, alpha=a_in(f, f_not + 0.03), linespacing=1.3)

    def _steps(ax, f):
        a_head = a_in(f, f_next)
        if a_head <= 0:
            return
        ax.text(STEP_X, 7.0, "Next steps", ha="left", va="center", size=style.BODY_SIZE,
                weight="semibold", color=style.C_ACCENT, alpha=a_head)
        for i, (y, text, start) in enumerate(zip(STEP_YS, steps, f_steps, strict=True)):
            a = a_in(f, start - 0.01)
            if a <= 0:
                continue
            ax.scatter([STEP_X + 0.3], [y], s=900, color=style.C_ACCENT, alpha=a,
                       linewidths=0)
            ax.text(STEP_X + 0.3, y, str(i + 1), ha="center", va="center",
                    size=style.LABEL_SIZE, weight="bold", color=style.BG, alpha=a)
            ax.text(STEP_X + 0.85, y, text, ha="left", va="center", size=style.BODY_SIZE - 2,
                    color=style.FG, alpha=a, linespacing=1.2)

    return draw
