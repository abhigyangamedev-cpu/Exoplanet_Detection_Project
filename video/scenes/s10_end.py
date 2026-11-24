"""Scene 10, the end card, and the seamless loop back to scene 1.

The card shows the title, the credits from video/credits.json (name, team, institution,
guide; an empty field is left out with no gap), the tools the project is built with (its
real dependencies in pyproject.toml) and the data it uses. Tools appear as "Python" is
spoken, the data line as "NASA data" is.

The loop. Over the final 15% of the scene the card fades out and every star's brightness
moves from its running twinkle to its t = 0 value, on scene 1's star field (same seed, same
stars). The fade completes on the clip's last rendered frame, t = (n_frames - 1) / fps, which
is then exactly ``s01_hook.draw_starfield(fig, 0.0)``: scene 1's first frame. The video
plays on repeat, so that cut is invisible.
"""

from __future__ import annotations

import textwrap

import style
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties

from scenes import cue_fraction
from scenes.s01_hook import STARFIELD_SEED, draw_starfield

# (shown name, pyproject dependency); Python is the language itself
TOOLS = [("Python", None), ("NumPy", "numpy"), ("pandas", "pandas"),
         ("scikit-learn", "scikit-learn"), ("PyTorch", "torch"), ("lightkurve", "lightkurve"),
         ("matplotlib", "matplotlib")]
DATA_CREDIT = "Data: NASA Kepler mission, NASA Exoplanet Archive, MAST"
FADE_START = 0.85                   # the card fades out over the final 15%

# layout in inches of the 16 x 9 overlay axes (caption band: y < 1.35). The whole card is
# one column, centred vertically in [AREA_BOTTOM, AREA_TOP] whatever the credits hold.
CX = 8.0
AREA_TOP, AREA_BOTTOM = 8.7, 1.5
GAP = 0.2                           # between credit lines
TITLE_GAP = 0.35                    # title to name
SECTION_GAP = 0.6                   # credits to "Built with"
NAME_SIZE = style.BODY_SIZE + 4
CREDIT_SIZE = style.LABEL_SIZE
LINESPACING = 1.25
INSTITUTION_WRAP = 52               # characters per line
CHIP_SIZE = style.LABEL_SIZE - 2
CHIP_PAD = 0.2                      # inches around a chip's text
CHIP_GAP = 0.28
DATA_SIZE = CREDIT_SIZE - 2


def text_width(text: str, size: float, weight: str = "normal") -> float:
    """Width in inches of text drawn at size points."""
    prop = FontProperties(family="sans-serif", size=size, weight=weight)
    return RendererAgg(8, 8, 72).get_text_width_height_descent(text, prop, ismath=False)[0] / 72


def team_lines(team: list[str]) -> str:
    """The team, two names to a line, after "with"."""
    names = [m.strip() for m in team if m and m.strip()]
    rows = [" · ".join(names[i:i + 2]) for i in range(0, len(names), 2)]
    return "with " + "\n".join(rows) if rows else ""


MIN_SCALE = 0.7                     # credits shrink to at most 70%: names at 22.4 pt, rest 15.4
SCALE_STEP = 0.05
MAX_LINE_W = 15.0                   # inches: no credit line wider than this


class CreditsTooLong(ValueError):
    """The end card cannot fit the credits even at the smallest readable size."""


def credit_stack(credits: dict, scale: float = 1.0) -> list[tuple[str, float, str, str, str]]:
    """The credit lines under the title as (text, size, colour, weight, field); empty ones
    left out. scale shrinks the credit sizes (the title keeps its size)."""
    def field(key):
        value = credits.get(key) or ""
        return value.strip() if isinstance(value, str) else value

    name_size, credit_size = NAME_SIZE * scale, CREDIT_SIZE * scale
    stack = []
    if field("name"):
        stack.append((field("name"), name_size, style.FG, "semibold", "name"))
    if team_lines(credits.get("team") or []):
        stack.append((team_lines(credits["team"]), credit_size, style.FG, "normal", "team"))
    if field("institution"):
        stack.append((textwrap.fill(field("institution"), INSTITUTION_WRAP,
                                    break_on_hyphens=False), credit_size, style.MUTED, "normal",
                      "institution"))
    if field("guide"):
        stack.append((f"Guide: {field('guide')}", credit_size, style.MUTED, "normal", "guide"))
    return stack


def _height(text: str, size: float) -> float:
    """Height in inches of a (multi-line) text block at size points."""
    return size / 72 * (1 + LINESPACING * text.count("\n")) * 1.1


def _column(credits: dict, scale: float):
    """The column laid out downward from y = 0: (credit blocks, tools label, chips, data,
    bottom), the blocks as (y, text, size, colour, weight, field)."""
    chip_h = CHIP_SIZE / 72 * 1.1 + 2 * CHIP_PAD
    y = -_height("x", style.TITLE_SIZE)
    stack = []
    for text, size, colour, weight, key in credit_stack(credits, scale):
        y -= GAP if stack else TITLE_GAP
        stack.append((y, text, size, colour, weight, key))
        y -= _height(text, size)
    y -= SECTION_GAP
    tools_label = y
    y -= _height("x", CREDIT_SIZE) + GAP
    chips = y - chip_h / 2
    data = y - chip_h - GAP
    return stack, tools_label, chips, data, data - _height("x", DATA_SIZE)


def _too_wide(stack) -> list[str]:
    return [key for _, text, size, _, weight, key in stack
            if max(text_width(line, size, weight) for line in text.split("\n")) > MAX_LINE_W]


def card_layout(credits: dict) -> dict:
    """Top y of every block (title, credit lines, tools label, data line) and the chips'
    centre y, the column centred vertically in the area above the caption band.

    Credits too tall (or too wide) for the card are shrunk in steps of SCALE_STEP down to
    MIN_SCALE; if they still do not fit, CreditsTooLong names the field to shorten.
    """
    scale = 1.0
    while True:
        stack, tools_label, chips, data, bottom = _column(credits, scale)
        wide = _too_wide(stack)
        if -bottom <= AREA_TOP - AREA_BOTTOM and not wide:
            break
        if scale - SCALE_STEP < MIN_SCALE - 1e-9:
            if wide:
                key = wide[0]
                why = "a line is too wide"
            else:
                key = max(stack, key=lambda s: _height(s[1], s[2]))[5] if stack else "title"
                why = "the credits are too tall"
            raise CreditsTooLong(
                f"video/credits.json: the end card cannot fit the credits ({why}, even at "
                f"{MIN_SCALE:.0%} size). Shorten {key!r}.")
        scale = round(scale - SCALE_STEP, 4)
    top = min(AREA_TOP, (AREA_TOP + AREA_BOTTOM - bottom) / 2)
    return {"title": top, "stack": [(top + s[0], *s[1:5]) for s in stack], "scale": scale,
            "tools_label": top + tools_label, "chips": top + chips, "data": top + data}


def build(ctx):
    title = ctx.credits.get("title", "")
    lay = card_layout(ctx.credits)
    stack = lay["stack"]

    widths = [text_width(label, CHIP_SIZE) + 2 * CHIP_PAD for label, _ in TOOLS]
    x = CX - (sum(widths) + CHIP_GAP * (len(TOOLS) - 1)) / 2
    chips = []
    for (label, _), w in zip(TOOLS, widths, strict=True):
        chips.append((x + w / 2, label))
        x += w + CHIP_GAP

    # timing (fractions of the scene), tied to the spoken words
    f_tools = cue_fraction(ctx, "Python", 0.4)
    f_data = cue_fraction(ctx, "NASA data", 0.52)
    # the fade completes on the last frame the engine renders, t = (n_frames - 1) / fps
    n_frames = round(ctx.duration * ctx.fps)
    f_last = (n_frames - 1) / ctx.fps / ctx.duration

    def a_in(f, start, dur=0.06):
        return style.ease(style.progress(f, start, start + dur))

    def draw(fig, t):
        f = t / ctx.duration
        fade = style.ease(style.progress(f, FADE_START, f_last))
        _stars(fig, t, fade)
        keep = 1.0 - fade
        if keep <= 0.0:
            return                     # the last frame: scene 1's first frame, stars only

        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 9)
        ax.set_axis_off()

        a_title = keep * a_in(f, 0.0, 0.08)
        ax.text(CX, lay["title"], title, ha="center", va="top", size=style.TITLE_SIZE,
                weight="semibold", color=style.FG, alpha=a_title)
        a_credits = keep * a_in(f, 0.04, 0.08)
        for y, text, size, colour, weight in stack:
            ax.text(CX, y, text, ha="center", va="top", size=size, color=colour,
                    weight=weight, alpha=a_credits, linespacing=LINESPACING,
                    multialignment="center")

        a_tools = keep * a_in(f, f_tools - 0.02)
        if a_tools > 0:
            ax.text(CX, lay["tools_label"], "Built with", ha="center", va="top",
                    size=CREDIT_SIZE, color=style.C_ACCENT, alpha=a_tools)
            for i, (cx, label) in enumerate(chips):
                a = keep * a_in(f, f_tools - 0.02 + 0.012 * i, 0.04)
                ax.text(cx, lay["chips"], label, ha="center", va="center", size=CHIP_SIZE,
                        color=style.FG, alpha=a,
                        bbox=dict(boxstyle=f"round,pad={CHIP_PAD * 72 / CHIP_SIZE:.3f}",
                                  facecolor=style.TERM_BG, edgecolor=style.MUTED,
                                  linewidth=1.2, alpha=a))
        a_data = keep * a_in(f, f_data - 0.01)
        if a_data > 0:
            ax.text(CX, lay["data"], DATA_CREDIT, ha="center", va="top", size=DATA_SIZE,
                    color=style.MUTED, alpha=a_data)

    return draw


def _stars(fig, t: float, toward_zero: float) -> None:
    """Scene 1's star field, each star's brightness moved toward its t = 0 value by
    toward_zero (0: twinkling at t; 1: exactly scene 1's first frame)."""
    if toward_zero >= 1.0:
        draw_starfield(fig, 0.0)
        return
    draw_starfield(fig, t)
    if toward_zero <= 0.0:
        return
    ax = fig.axes[-1]
    style.starfield(ax, STARFIELD_SEED, 0.0)          # the same stars at t = 0
    now, zero = ax.collections[-2:]
    colours = now.get_facecolors().copy()
    colours[:, 3] += toward_zero * (zero.get_facecolors()[:, 3] - colours[:, 3])
    now.set_facecolors(colours)
    zero.remove()
