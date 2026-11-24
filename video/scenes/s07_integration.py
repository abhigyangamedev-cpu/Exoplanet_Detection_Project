"""Scene 7, putting it together: the pipeline as a chain of commands, replayed.

A row of seven steps (with their real script names in scripts/) lights up as the narration
lists them. Below, a terminal replays captured output (video/build/data/terminal/, written by
video/extract_data.py): the leakage audit and the report are typed into one session, then the
test suite, whose real summary line is called out beside the terminal while a check appears
on every step. The two long jobs are shown from their saved logs (the window title names the
log): the tail of the view build, captioned "several hours, resumable", and the training log,
captioned with the training time and the document it is quoted from.

Commands are shown exactly as they were run: the fast ones as video/extract_data.py runs them
(``capture_terminal``), the long ones as docs/evaluation.md records them. Output lines are
never changed; a line too long for the window is cut at the end with an ellipsis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import style
from matplotlib.patches import FancyBboxPatch

from scenes import cue_fraction
from scenes.s06_models import text_width

# the chain: (plain label, scripts, narrated phrase); the first cue containing the phrase
STEPS = (("fetch", ("fetch_catalogue.py",), "fetch"),
         ("audit", ("audit_leakage.py",), "audit"),
         ("split", ("make_splits.py",), "split"),
         ("build views", ("build_views.py",), "build the views"),
         ("train", ("train_cnn.py", "train_tabular.py"), "train,"),
         ("evaluate", ("evaluate.py",), "evaluate"),
         ("report", ("make_report.py",), "report"))

# commands as run (python = the project's .venv interpreter)
CMD_AUDIT = ("python scripts/audit_leakage.py --strict --output "
             "video/build/leakage_audit_results.md --columns-output "
             "video/build/leakage_columns.md")
CMD_REPORT = "python scripts/make_report.py --output video/build/report.md"
CMD_PYTEST = "python -m pytest -q -p no:cacheprovider"
CMD_BUILD = "python scripts/build_views.py"                  # docs/evaluation.md
CMD_TRAIN = "python scripts/train_cnn.py --name cnn_full"    # docs/evaluation.md
LOG_BUILD = "outputs/build_views_full_dev_resume4.log"       # read by extract_data.py
LOG_TRAIN = "outputs/train_cnn_full.log"
PROMPT = "> "

# layout: inches of the 16 x 9 overlay axes
ROW_X = (0.5, 15.5)
CHIP_Y, CHIP_H = 6.15, 1.05
CHIP_PAD, CHIP_GAP = 0.3, 0.22
LABEL_PT, SCRIPT_PT = style.LABEL_SIZE, 15
TERM_BOX = (0.5, 1.55, 11.7, 4.2)                             # x, y, w, h
TERM_PT = 18
MAX_CHARS = 80                                                # widest line in the window
NOTE_X = 12.55
TYPE_F = 0.0009                                               # typing: fraction per char


def fit_line(line: str, max_chars: int = MAX_CHARS) -> str:
    """The line unchanged if it fits, else cut to max_chars with an ellipsis at the end."""
    return line if len(line) <= max_chars else line[:max_chars - 1] + "…"


def session_state(segments, f: float) -> tuple[list[str], int]:
    """Terminal lines and typed characters at f for one session.

    segments: (f_show, f_type0, f_type1, text) per line. A line appears at f_show; a command
    line shows its prompt from then and types the rest between f_type0 and f_type1; an
    output line has f_show == f_type0 == f_type1 and appears whole. Once every line is
    complete, two extra characters are counted (the last newline and one more), so
    style.terminal draws no cursor (a finished log); end a live session with an idle
    prompt that never types to keep the cursor.
    """
    lines, chars = [], 0
    for f_show, f0, f1, text in segments:
        if f < f_show:
            break
        if lines:
            chars += len(lines[-1]) + 1
        lines.append(text)
        head = len(PROMPT) if text.startswith(PROMPT) and f1 > f0 else 0
        last = head + round((len(text) - head) * style.progress(f, f0, f1))
    if not lines:
        return [PROMPT], len(PROMPT)
    finished = len(lines) == len(segments) and f >= segments[-1][2]
    return lines, chars + last + (2 if finished else 0)


def _read_capture(data_dir: Path, name: str) -> list[str]:
    text = (Path(data_dir) / "terminal" / f"{name}.txt").read_text(encoding="utf-8")
    return [fit_line(line) for line in text.splitlines()]


def _command(f_show, f_type0, command):
    line = fit_line(PROMPT + command)
    return (f_show, f_type0, f_type0 + TYPE_F * len(line), line)


def _outputs(lines, f0, f1):
    fs = np.linspace(f0, f1, len(lines)) if len(lines) > 1 else [f0]
    return [(float(f), float(f), float(f), line) for f, line in zip(fs, lines, strict=True)]


def summary_line(pytest_lines: list[str]) -> str:
    """pytest's summary ("N passed in S s"): the last non-empty line of its output."""
    return next(line for line in reversed(pytest_lines) if line.strip()).strip()


def build(ctx):
    audit = _read_capture(ctx.data_dir, "audit_leakage")
    report = _read_capture(ctx.data_dir, "make_report")
    tests = _read_capture(ctx.data_dir, "pytest")
    build_tail = _read_capture(ctx.data_dir, "build_views_tail")
    train_log = _read_capture(ctx.data_dir, "train_cnn_full")
    passed = summary_line(tests)
    train_time = ctx.numbers["cnn_full_train_time_text"]
    train_source = ctx.numbers["sources"]["cnn_full_train_time_text"]

    # timing, tied to the spoken words
    f_step = [cue_fraction(ctx, phrase, (i + 1) * 0.045) for i, (_, _, phrase) in
              enumerate(STEPS)]
    f_chain = cue_fraction(ctx, "chain of commands", 0.06)
    f_tests = cue_fraction(ctx, "Automated tests", 0.35)
    f_check = cue_fraction(ctx, "check every step", 0.44)
    f_build = cue_fraction(ctx, "Building the views", 0.5)
    f_hours = cue_fraction(ctx, "several hours", 0.55)
    f_train = cue_fraction(ctx, "Training", 0.72)
    f_laptop = cue_fraction(ctx, "laptop", 0.8)
    f_took = cue_fraction(ctx, "card took", 0.9)
    f_audit, f_report = f_step[1], f_step[6]

    # session 1: audit and report, typed so their output lands as each step is named
    cmd_audit = _command(0.0, max(f_audit - TYPE_F * len(CMD_AUDIT) - 0.01, f_chain), CMD_AUDIT)
    out_audit = _outputs(audit, cmd_audit[2] + 0.01, cmd_audit[2] + 0.02)
    f_prompt = out_audit[-1][0] + 0.01
    cmd_report = _command(f_prompt, max(f_report - TYPE_F * len(CMD_REPORT) - 0.008,
                                        f_prompt + 0.01), CMD_REPORT)
    out_report = _outputs(report, cmd_report[2] + 0.008, cmd_report[2] + 0.008)
    cmd_tests = _command(out_report[-1][0] + 0.006, max(f_tests, out_report[-1][0] + 0.01),
                         CMD_PYTEST)
    f_passed = max(f_check, cmd_tests[2] + 0.03)
    out_tests = _outputs(tests, cmd_tests[2] + 0.01, f_passed)
    idle = (f_passed + 0.006, 2.0, 3.0, PROMPT)                 # waits, cursor blinking on
    live = [cmd_audit, *out_audit, cmd_report, *out_report, cmd_tests, *out_tests, idle]
    # sessions 2 and 3: the long jobs, from their saved logs
    cmd_build = _command(f_build, f_build + 0.005, CMD_BUILD)
    builds = [cmd_build, *_outputs(build_tail, cmd_build[2] + 0.01, f_train - 0.03)]
    cmd_train = _command(f_train, f_train + 0.005, CMD_TRAIN)
    trains = [cmd_train, *_outputs(train_log, cmd_train[2] + 0.01, min(f_took, 0.95))]
    sessions = ((0.0, "PowerShell", live), (f_build, f"saved log: {LOG_BUILD}", builds),
                (f_train, f"saved log: {LOG_TRAIN}", trains))
    # which step's output the terminal shows
    active = ((cmd_audit[1], 1), (cmd_report[1], 6), (cmd_tests[1], None), (f_build, 3),
              (f_train, 4))

    # chip layout, centred in the row
    widths = [max(text_width(label, LABEL_PT), *(text_width(s, SCRIPT_PT, mono=True)
                                                  for s in scripts)) + CHIP_PAD
              for label, scripts, _ in STEPS]
    total = sum(widths) + CHIP_GAP * (len(STEPS) - 1)
    if total > ROW_X[1] - ROW_X[0]:
        raise ValueError(f"step row is {total:.2f} in wide, more than the frame allows")
    lefts = ROW_X[0] + (ROW_X[1] - ROW_X[0] - total) / 2 + np.concatenate(
        [[0.0], np.cumsum(np.array(widths[:-1]) + CHIP_GAP)])

    def a_at(f, start, dur=0.02):
        return style.ease(style.progress(f, start, start + dur))

    def draw(fig, t):
        f = t / ctx.duration
        style.starfield(fig.add_axes((0, 0, 1, 1)), 7, t)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 9)
        ax.set_axis_off()
        style.heading(fig, "Putting it together")
        current = None
        for f_from, step in active:
            if f >= f_from:
                current = step
        _draw_steps(ax, f, current)
        _draw_terminal(fig, f)
        _draw_notes(ax, f)

    def _draw_steps(ax, f, current):
        for i, ((label, scripts, _), x, w) in enumerate(zip(STEPS, lefts, widths,
                                                            strict=True)):
            lit = a_at(f, f_step[i] - 0.01)
            on = i == current and f >= f_step[i] - 0.01
            ax.add_patch(FancyBboxPatch(
                (x, CHIP_Y), w, CHIP_H, boxstyle="round,pad=0,rounding_size=0.1",
                facecolor=style.C_ACCENT if on else style.TERM_BG,
                alpha=0.22 if on else 0.9, edgecolor="none"))
            ax.add_patch(FancyBboxPatch(
                (x, CHIP_Y), w, CHIP_H, boxstyle="round,pad=0,rounding_size=0.1",
                facecolor="none", edgecolor=style.C_ACCENT if on else style.FG,
                linewidth=2.6 if on else 1.4, alpha=1.0 if on else 0.3 + 0.6 * lit))
            cx = x + w / 2
            ax.text(cx, CHIP_Y + CHIP_H - 0.27, label, ha="center", va="center",
                    size=LABEL_PT, weight="semibold", color=style.FG, alpha=0.35 + 0.65 * lit)
            ys = ([CHIP_Y + 0.5, CHIP_Y + 0.24] if len(scripts) == 2 else [CHIP_Y + 0.37])
            for s, y in zip(scripts, ys, strict=True):
                ax.text(cx, y, s, ha="center", va="center", family="monospace",
                        size=SCRIPT_PT, color=style.C_ACCENT if lit > 0.5 else style.MUTED,
                        alpha=0.4 + 0.6 * lit)
            if i:
                ax.scatter([x - CHIP_GAP / 2], [CHIP_Y + CHIP_H / 2], marker=">", s=40,
                           color=style.MUTED, alpha=0.4 + 0.6 * lit, linewidths=0)
            a_tick = a_at(f, f_passed + 0.004 * i, 0.015)
            if a_tick > 0:   # "tests check every step": a check on each step
                ax.plot([cx - 0.13, cx - 0.04, cx + 0.14],
                        [CHIP_Y - 0.2, CHIP_Y - 0.29, CHIP_Y - 0.08],
                        color=style.C_ACCENT, lw=2.6, alpha=a_tick)

    def _draw_terminal(fig, f):
        start, title, segments = next(s for s in reversed(sessions) if f >= s[0])
        x, y, w, h = TERM_BOX
        tax = fig.add_axes((x / 16, y / 9, w / 16, h / 9))
        lines, chars = session_state(segments, f)
        style.terminal(tax, lines, chars, title=title, fontsize=TERM_PT)

    def _draw_notes(ax, f):
        a = a_at(f, f_passed)
        if a > 0:
            ax.text(NOTE_X, 5.35, "automated tests", ha="left", va="center",
                    size=style.LABEL_SIZE - 2, color=style.MUTED, alpha=a)
            ax.text(NOTE_X, 4.95, passed, ha="left", va="center", family="monospace",
                    size=TERM_PT, weight="bold", color=style.C_ACCENT, alpha=a)
        a = a_at(f, f_hours - 0.01)
        if a > 0:
            ax.text(NOTE_X, 4.2, "building the views", ha="left", va="center",
                    size=style.LABEL_SIZE - 2, color=style.MUTED, alpha=a)
            ax.text(NOTE_X, 3.65, "several hours,\nresumable", ha="left", va="center",
                    size=style.LABEL_SIZE + 2, weight="semibold", color=style.FG,
                    linespacing=1.15, alpha=a)
        a = a_at(f, f_laptop - 0.01)
        if a > 0:
            ax.text(NOTE_X, 2.85, "training, laptop GPU", ha="left", va="center",
                    size=style.LABEL_SIZE - 2, color=style.MUTED, alpha=a)
        a = a_at(f, f_took - 0.01)
        if a > 0:
            ax.text(NOTE_X, 2.35, train_time, ha="left", va="center",
                    size=style.BODY_SIZE, weight="bold", color=style.C_ACCENT, alpha=a)
            ax.text(NOTE_X, 1.85, f"source: {train_source}", ha="left", va="center",
                    size=style.LABEL_SIZE - 7, color=style.MUTED, alpha=a)

    return draw
