"""Shared tests for the ten scene modules (video/scenes/sNN_name.py).

The generic tests are parametrised over every scene id; a scene whose module does not exist
yet is skipped, so the suite stays green while later tasks add scenes. Scene-specific tests
follow below the generic ones.
"""

import re
from pathlib import Path

import engine
import numpy as np
import pytest
import scenes
import style
from matplotlib.text import Text
from narration import SCENE_IDS

SCENES_DIR = Path(scenes.__file__).resolve().parent
needs_data = pytest.mark.skipif(not (engine.DATA_DIR / "numbers.json").exists(),
                                reason="build data not generated (video/extract_data.py)")


def scene_path(scene_id: str) -> Path:
    return SCENES_DIR / f"s{scene_id}.py"


def scene_module(scene_id: str):
    """The scene module, or skip the test when it has not been written yet."""
    if not scene_path(scene_id).exists():
        pytest.skip(f"scene {scene_id} not implemented yet")
    return scenes.import_scene(scene_id)


def scene_source(scene_id: str) -> str:
    if not scene_path(scene_id).exists():
        pytest.skip(f"scene {scene_id} not implemented yet")
    return scene_path(scene_id).read_text(encoding="utf-8")


def make_ctx(scene_id: str, **kwargs) -> engine.Context:
    return engine.make_context(scene_id, **kwargs)


def render(draw, t: float, width: int = 960, height: int = 540) -> np.ndarray:
    return engine.render_frame(draw, t, width, height)


def drawn_texts(draw, t: float, width: int = 960, height: int = 540) -> list[str]:
    """Every text string the scene puts on screen at t (figure and axes texts)."""
    fig = style.new_figure(width, height)
    draw(fig, t)
    fig.canvas.draw()
    return [a.get_text() for a in fig.findobj(Text) if a.get_visible() and a.get_text()]


# --- every scene ----------------------------------------------------------------------

@needs_data
@pytest.mark.parametrize("scene_id", SCENE_IDS)
def test_scene_draws_start_middle_end(scene_id):
    module = scene_module(scene_id)
    ctx = make_ctx(scene_id, duration=20.0, width=960, height=540)
    draw = module.build(ctx)
    frames = [render(draw, t) for t in (0.0, 10.0, 19.99)]
    assert all(f.shape == (540, 960, 4) for f in frames)
    assert not np.array_equal(frames[0], frames[1])      # it animates
    assert frames[1].std() > 5                           # not blank


@pytest.mark.parametrize("scene_id", SCENE_IDS)
def test_scene_has_no_literal_results(scene_id):
    src = scene_source(scene_id)
    assert not re.search(r"0\.9\d\d|6[,]?677|2[,]?731|3[,]?946", src)


@needs_data
@pytest.mark.parametrize("scene_id", SCENE_IDS)
def test_scene_keeps_caption_band_clear(scene_id):
    """Nothing but the star field is drawn in the caption band (bottom CAPTION_BAND)."""
    module = scene_module(scene_id)
    ctx = make_ctx(scene_id, width=960, height=540)
    draw = module.build(ctx)
    band = int(round(540 * style.CAPTION_BAND))
    for frac in (0.1, 0.3, 0.5, 0.7, 0.9, 0.999):
        frame = render(draw, frac * ctx.duration)[-band:, :, :3].astype(int)
        background = np.array([int(style.BG[i:i + 2], 16) for i in (1, 3, 5)])
        # stars are tiny specks; anything else would cover many pixels
        changed = np.abs(frame - background).sum(axis=2) > 40
        assert changed.mean() < 0.01, f"{scene_id} draws in the caption band at {frac}"


# --- scene helpers --------------------------------------------------------------------

def test_cue_fraction_places_phrases_in_their_cue():
    ctx = engine.Context(scene_id="x", duration=11.0, fps=15, width=960, height=540,
                         numbers={}, data_dir=engine.DATA_DIR, credits={},
                         cues=[(0.0, 4.0, "alpha beta"), (4.0, 10.0, "gamma delta")])
    assert scenes.cue_fraction(ctx, "alpha", 0.5) == 0.0
    assert scenes.cue_fraction(ctx, "gamma", 0.5) == pytest.approx(4.0 / 11.0)
    # "delta" starts at character 6 of 11 in a 6 s cue
    assert scenes.cue_fraction(ctx, "delta", 0.5) == pytest.approx((4.0 + 6 * 6 / 11) / 11.0)
    assert scenes.cue_fraction(ctx, "missing", 0.5) == 0.5
    # a duration override shorter than the audio scales the cues down instead of past 1
    ctx.duration = 5.0
    assert scenes.cue_fraction(ctx, "gamma", 0.5) == pytest.approx(0.4)


# --- 01_hook ----------------------------------------------------------------------------

@needs_data
def test_hook_starts_with_only_the_star_field():
    """Frame 0 is the loop target: the end card fades into exactly this star field."""
    hook = scene_module("01_hook")
    draw = hook.build(make_ctx("01_hook", width=960, height=540))
    expected = render(lambda fig, t: hook.draw_starfield(fig, t), 0.0)
    assert np.array_equal(render(draw, 0.0), expected)
    assert not np.array_equal(render(draw, 0.0), render(lambda fig, t: None, 0.0))
    assert isinstance(hook.STARFIELD_SEED, int)


@needs_data
def test_hook_shows_title_and_name_from_credits():
    ctx = make_ctx("01_hook", width=960, height=540)
    texts = drawn_texts(scene_module("01_hook").build(ctx), 0.6 * ctx.duration)
    assert ctx.credits["title"] in texts
    assert any(ctx.credits["name"] in s for s in texts)


@needs_data
@pytest.mark.parametrize("scene_id,frac", [("01_hook", 0.6), ("05_pipeline", 0.05),
                                           ("05_pipeline", 0.97)])
def test_a_koi_id_is_called_a_signal_never_a_star(scene_id, frac):
    ctx = make_ctx(scene_id, width=960, height=540)
    texts = " ".join(drawn_texts(scene_module(scene_id).build(ctx), frac * ctx.duration))
    star = ctx.load("star_planet.npz")
    koi, planet = str(star["koi_name"]), str(star["kepler_name"])
    assert planet                                           # the catalogue names this planet
    assert f"signal {koi}" in texts and planet in texts
    assert not re.search(r"star\s+K\d{5}\.\d\d", texts)
    if frac > 0.9:                                          # the binary, too
        assert f"signal {ctx.load('star_binary.npz')['koi_name']}" in texts


# --- 02_problem -------------------------------------------------------------------------

@needs_data
def test_problem_counts_up_to_the_real_numbers():
    ctx = make_ctx("02_problem", width=960, height=540)
    draw = scene_module("02_problem").build(ctx)
    n = ctx.numbers
    final = {f"{n['n_confirmed']:,}", f"{n['n_false_positive']:,}"}
    total = f"{n['n_kois']:,}"
    assert final <= set(drawn_texts(draw, ctx.duration - 0.01))
    assert not (final | {total}) & set(drawn_texts(draw, 0.0))
    # each count has arrived as its number is spoken
    for number in (total, *final):
        t = scenes.cue_fraction(ctx, number, -1.0) * ctx.duration
        assert t >= 0, number
        assert number in drawn_texts(draw, t + 0.25), number
    # the total and its label leave with the last dot from the grey cloud
    end = drawn_texts(draw, ctx.duration - 0.01)
    assert total not in end and "Kepler signals, one dot each" not in end

    # the counters count up: every number on screen only grows, through values in between
    seen: dict[str, list[int]] = {}
    for frac in np.linspace(0.25, 0.999, 80):
        for label, count in labelled_counts(drawn_texts(draw, frac * ctx.duration)).items():
            seen.setdefault(label, []).append(count)
    for label, target in (("Kepler signals, one dot each", n["n_kois"]),
                          ("confirmed planets", n["n_confirmed"]),
                          ("false positives", n["n_false_positive"])):
        series = seen[label]
        assert series == sorted(series) and series[-1] == target, label
        assert any(0 < v < target for v in series), target


@needs_data
def test_problem_planet_touches_the_limb_as_the_trace_falls():
    """The planet's disc meets the star's edge exactly while the trace leaves (and returns
    to) its baseline, and is centred on the star at the bottom of the dip."""
    prob = scene_module("02_problem")
    ctx = make_ctx("02_problem", width=960, height=540)
    draw = prob.build(ctx)
    star = ctx.load("star_planet.npz")
    planet_r = prob.STAR_R * np.sqrt(float(star["depth"]))
    f_end = scenes.cue_fraction(ctx, "called", -1.0)
    assert f_end > 0
    tol = 0.05 * prob.TRACE_H                            # 5% of the full dip
    samples = []
    for frac in np.linspace(0.0, f_end, 500):
        fig = style.new_figure(960, 540)
        draw(fig, frac * ctx.duration)
        planets = fig.findobj(lambda a: a.get_gid() == "planet")
        traces = fig.findobj(lambda a: a.get_gid() == "trace")
        if not planets or not traces:
            continue
        (px, py), y = planets[0].center, traces[0].get_ydata()[-1]
        sx, sy = prob.STAR_XY
        samples.append((np.hypot(px - sx, py - sy), abs(px - sx), y))
    overlap = [s for s in samples if s[0] < prob.STAR_R + planet_r]
    assert overlap and len(overlap) < len(samples)
    # off the disc: the trace is at its baseline
    assert all(abs(y - prob.TRACE_BASE) < tol for d, _, y in samples
               if d > prob.STAR_R + planet_r)
    # first touch: the trace is only beginning to fall
    first = next(i for i, s in enumerate(samples) if s[0] < prob.STAR_R + planet_r)
    assert abs(samples[first][2] - prob.TRACE_BASE) < tol
    assert abs(samples[first - 1][2] - prob.TRACE_BASE) < tol
    # mid-transit: the trace is at the full depth
    _, _, y_mid = min(samples, key=lambda s: s[1])
    assert abs(y_mid - (prob.TRACE_BASE - prob.TRACE_H)) < tol


def test_problem_dip_edges_find_where_the_dip_starts():
    prob = scene_module("02_problem")
    lv = np.zeros(21)
    lv[7:14] = [-0.01, -0.3, -0.8, -1.0, -0.8, -0.3, -0.05]
    assert prob.dip_edges(lv) == (8, 13)                  # -0.01 is still baseline
    lo, hi = 8, 13
    sx, r_star, r_p = 4.0, 2.0, 0.25
    reach = np.sqrt((r_star + r_p) ** 2 - (prob.CHORD_DY * r_star) ** 2)
    assert prob.planet_x(lo, lo, hi, sx, r_star, r_p) == pytest.approx(sx - reach)
    assert prob.planet_x(hi, lo, hi, sx, r_star, r_p) == pytest.approx(sx + reach)
    assert prob.planet_x((lo + hi) / 2, lo, hi, sx, r_star, r_p) == pytest.approx(sx)


def labelled_counts(texts: list[str]) -> dict[str, int]:
    """The plain numbers on screen ("2,731"), keyed by the label drawn right after each."""
    return {texts[i + 1]: int(s.replace(",", "")) for i, s in enumerate(texts[:-1])
            if re.fullmatch(r"\d[\d,]*", s)}


# --- 03_data ------------------------------------------------------------------------------

@needs_data
def test_data_targets_keep_each_star_in_its_partition():
    data = scene_module("03_data")
    ctx = make_ctx("03_data", width=960, height=540)
    sample = ctx.load("splits_sample.json")
    boxes = data.partition_boxes()
    targets = data.star_targets(sample)
    assert len(targets) == len(sample)
    assert len({tuple(xy) for xy in targets}) == len(sample)   # one slot per star
    for star, (x, y) in zip(sample, targets, strict=True):
        x0, y0, w, h = boxes[star["partition"]]
        assert x0 < x < x0 + w and y0 < y < y0 + h, star


@needs_data
def test_data_shows_real_split_counts():
    ctx = make_ctx("03_data", width=960, height=540)
    texts = " ".join(drawn_texts(scene_module("03_data").build(ctx), ctx.duration - 0.01))
    assert f"{ctx.numbers['n_test']:,}" in texts
    assert f"{ctx.numbers['n_dev']:,}" in texts


# --- 04_leakage ---------------------------------------------------------------------------

VETTING = {"koi_score", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
           "koi_pdisposition"}


def score_bar(draw, t: float):
    """(top of the score bar in score units, its face colour) at t, or None if not drawn."""
    fig = style.new_figure(960, 540)
    draw(fig, t)
    bars = fig.findobj(lambda a: a.get_gid() == "score_bar")
    if not bars:
        return None
    return bars[0].get_y() + bars[0].get_height(), bars[0].get_facecolor()


@needs_data
def test_leakage_finds_the_six_vetting_columns_in_the_audit():
    leak = scene_module("04_leakage")
    rows = make_ctx("04_leakage", width=960, height=540).load("audit_columns.json")
    assert set(leak.vetting_columns(rows)) == VETTING
    assert not any(r["kept"] for r in rows if r["column"] in VETTING)


@needs_data
def test_leakage_shows_counts_names_and_the_jump():
    from matplotlib.colors import to_rgba
    ctx = make_ctx("04_leakage", width=960, height=540)
    draw = scene_module("04_leakage").build(ctx)
    n = ctx.numbers
    audited, leaked = f"{n['leak_lc_audited']:.3f}", f"{n['leak_lc_plus_vetting']:.3f}"
    n_kept = n["n_audit_columns"] - n["n_audit_removed"]
    n_lc = n["leak_lc_n_features"]
    n_added = n["leak_lc_plus_vetting_n_features"] - n_lc
    at = {phrase: scenes.cue_fraction(ctx, phrase, -1.0) * ctx.duration
          for phrase in ("Our score", "jumps", "Too good")}
    assert all(t >= 0 for t in at.values())             # the cues exist in the narration

    def texts(t):
        return " ".join(" ".join(drawn_texts(draw, t)).split())

    # no score before the narration introduces it, though the six are already waiting
    before = at["Our score"] - 0.3
    assert score_bar(draw, before) is None
    assert audited not in texts(before) and leaked not in texts(before)

    honest = at["jumps"] - 0.5                          # bar grown, not yet jumped
    shown = texts(honest)
    assert all(name in shown for name in VETTING)       # the six real names are on screen
    assert f"{n['n_audit_removed']} removed" in shown and f"{n_kept} kept" in shown
    assert f"model on {n_lc} dip + star measurements" in shown and audited in shown
    assert "verdicts" not in shown.replace("NASA's verdicts", "")
    top, colour = score_bar(draw, honest)
    assert top == pytest.approx(n["leak_lc_audited"])
    assert colour[:3] == pytest.approx(to_rgba(style.FG)[:3])

    # tiles and bar change state together: the six land as the score jumps
    jumped = texts(at["Too good"] + 1.0)
    assert leaked in jumped and "too good to be true" in jumped.lower()
    assert f"+ {n_added} verdicts" in jumped and f"{n_lc} + {n_added} verdicts" in jumped
    assert f"{n_kept + n_added} kept" not in jumped      # the counter never adds them in
    top, colour = score_bar(draw, at["Too good"] + 1.0)
    assert top == pytest.approx(n["leak_lc_plus_vetting"])
    assert colour[:3] == pytest.approx(to_rgba(style.C_WARN)[:3])

    # end state: the six are out again and the bar is back at the audited score
    end = texts(ctx.duration - 0.01)
    assert audited in end and leaked not in end and f"+ {n_added} verdicts" not in end
    top, colour = score_bar(draw, ctx.duration - 0.01)
    assert top == pytest.approx(n["leak_lc_audited"])
    assert colour[:3] == pytest.approx(to_rgba(style.FG)[:3])


@needs_data
def test_leakage_six_tiles_land_only_when_the_score_jumps():
    """Tile positions: the six wait outside 'kept' until the jump, then sit in it."""
    leak = scene_module("04_leakage")
    ctx = make_ctx("04_leakage", width=960, height=540)
    draw = leak.build(ctx)
    t_jump = scenes.cue_fraction(ctx, "jumps", -1.0) * ctx.duration
    kept_right = leak.KEPT_X0 + (leak.KEPT_COLS - 1) * leak.DX + 0.2
    n_cols = len(ctx.load("audit_columns.json"))         # one tile per column

    def warn_tiles_x(t):
        from matplotlib.collections import PathCollection
        from matplotlib.colors import to_rgb
        fig = style.new_figure(960, 540)
        draw(fig, t)
        tiles = next(c for c in fig.findobj(PathCollection) if len(c.get_offsets()) == n_cols)
        rgb = tiles.get_facecolors()[:, :3]
        is_warn = np.abs(rgb - np.array(to_rgb(style.C_WARN))).sum(axis=1) < 0.05
        return tiles.get_offsets()[is_warn, 0]

    waiting = warn_tiles_x(t_jump - 1.0)
    assert len(waiting) == 6 and np.all(waiting > kept_right)
    landed = warn_tiles_x(t_jump + 2.0)
    assert len(landed) == 6 and np.all(landed < kept_right)


# --- 05_pipeline --------------------------------------------------------------------------

STAGES = ["raw brightness, in three-month pieces", "stitched", "slow changes removed",
          "folded on the orbit", "two pictures for the network"]


def test_pipeline_fold_moves_points_from_time_to_phase():
    pipe = scene_module("05_pipeline")
    period = 4.0
    phase = np.array([-1.0, 0.0, 0.5, 1.9])
    orbit = np.array([0, 1, 2, 30])
    x_time = phase + orbit * period
    assert np.allclose(pipe.fold_x(phase, orbit, period, np.zeros(4)), x_time)
    assert np.allclose(pipe.fold_x(phase, orbit, period, np.ones(4)), phase)
    half = pipe.fold_x(phase, orbit, period, np.full(4, 0.5))
    assert np.all((half >= phase) & (half <= x_time))


@needs_data
def test_pipeline_labels_each_stage_in_order():
    ctx = make_ctx("05_pipeline", width=960, height=540)
    draw = scene_module("05_pipeline").build(ctx)
    first_seen = {}
    for frac in np.linspace(0.0, 0.999, 60):
        text = " ".join(drawn_texts(draw, frac * ctx.duration)).lower()
        for stage in STAGES:
            if stage in text:
                first_seen.setdefault(stage, frac)
    assert list(first_seen) == STAGES                   # every stage shown, in order
    assert sorted(first_seen.values()) == list(first_seen.values())


@needs_data
def test_pipeline_shows_view_counts_and_the_comparison():
    ctx = make_ctx("05_pipeline", width=960, height=540)
    draw = scene_module("05_pipeline").build(ctx)
    n = ctx.numbers
    counts = " ".join(drawn_texts(draw, 0.72 * ctx.duration))
    assert f"{n['n_views_built']:,}" in counts and f"{n['n_kois']:,}" in counts
    end = " ".join(" ".join(drawn_texts(draw, 0.985 * ctx.duration)).split()).lower()
    assert "shallow, rounded dip" in end and "deep, sharp v" in end


# --- 06_models ----------------------------------------------------------------------------

def drawn_text_alpha(draw, t: float, width: int = 960, height: int = 540) -> dict[str, float]:
    """Each on-screen text string and its alpha (1 when unset) at t."""
    fig = style.new_figure(width, height)
    draw(fig, t)
    out = {}
    for a in fig.findobj(Text):
        if a.get_visible() and a.get_text():
            alpha = a.get_alpha()
            out[a.get_text()] = max(out.get(a.get_text(), 0.0), 1.0 if alpha is None else alpha)
    return out


def shown(draw, t: float) -> set[str]:
    """Texts visible at t (alpha above a half)."""
    return {s for s, a in drawn_text_alpha(draw, t).items() if a > 0.5}


def test_models_polyline_point_walks_by_arc_length():
    models = scene_module("06_models")
    path = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 1.0]])           # length 4
    pts = models.polyline_point(path, np.array([0.0, 0.5, 0.75, 1.0]))
    assert np.allclose(pts, [[0, 0], [2, 0], [3, 0], [3, 1]])


@needs_data
def test_models_feed_the_real_views_into_the_network():
    from matplotlib.lines import Line2D
    ctx = make_ctx("06_models", width=960, height=540)
    draw = scene_module("06_models").build(ctx)
    star = ctx.load("star_planet.npz")
    fig = style.new_figure(960, 540)
    draw(fig, 0.99 * ctx.duration)
    lines = [ln for ln in fig.findobj(Line2D) if len(ln.get_ydata()) > 100]
    for key in ("global_view", "local_view"):
        view = star[key]
        match = [ln for ln in lines if len(ln.get_ydata()) == len(view)]
        assert len(match) == 1, key
        # the thumbnail is the view itself, only shifted and scaled
        assert np.corrcoef(match[0].get_ydata(), view)[0, 1] > 0.9999


@needs_data
def test_models_parts_appear_as_narrated_without_a_score_value():
    ctx = make_ctx("06_models", width=960, height=540)
    draw = scene_module("06_models").build(ctx)
    at = {p: scenes.cue_fraction(ctx, p, -1.0) * ctx.duration
          for p in ("A neural network", "zero to one", "A team", "each fixing",
                    "depth and length", "the star itself", "image shifts")}
    assert all(t >= 0 for t in at.values())             # the cues exist in the narration

    early = shown(draw, at["A neural network"] - 0.3)
    assert "Neural network" not in early and "Decision trees" not in early
    network = shown(draw, at["A team"] - 0.3)
    assert {"Neural network", "whole orbit", "close-up of the transit", "global view",
            "local view", "merged", "dense layers", "score", "0", "1"} <= network
    assert "Decision trees" not in network
    trees = shown(draw, at["each fixing"] + 0.8)
    assert {"Decision trees", "each fixes the last one's mistakes"} <= trees
    assert "the star itself" not in trees

    groups = [("the dip: depth, length, ...", "transit", "depth and length"),
              ("the star itself", "stellar", "the star itself"),
              ("image shift during the dip", "centroid", "image shifts")]
    for plain, term, phrase in groups:
        assert plain not in shown(draw, at[phrase] - 0.6), plain
        assert {plain, term} <= shown(draw, at[phrase] + 0.6), plain

    # a bare 0-1 scale: no model score for this star anywhere in the scene
    for frac in np.linspace(0.0, 0.999, 12):
        assert not any(re.search(r"\d\.\d", s) for s in drawn_texts(draw, frac * ctx.duration))


# --- 07_integration -----------------------------------------------------------------------

def extract_data_commands() -> dict[str, str]:
    """The fast commands video/extract_data.py runs (capture_terminal), as shown on screen."""
    import ast
    tree = ast.parse((engine.VIDEO_DIR / "extract_data.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "capture_terminal")
    table = next(n.value for n in ast.walk(fn) if isinstance(n, ast.Assign)
                 and getattr(n.targets[0], "id", "") == "commands")

    def arg(node):
        if isinstance(node, ast.Constant):
            return node.value
        parts = []
        for v in node.values:                            # f"{build}/name.md"
            if isinstance(v, ast.Constant):
                parts.append(v.value)
            else:
                assert v.value.id == "build"
                parts.append("video/build")
        return "".join(parts)

    return {k.value: "python " + " ".join(arg(e) for e in v.elts)
            for k, v in zip(table.keys, table.values, strict=True)}


def test_integration_commands_are_the_ones_that_were_run():
    integ = scene_module("07_integration")
    commands = extract_data_commands()
    assert integ.CMD_AUDIT == commands["audit_leakage"]
    assert integ.CMD_REPORT == commands["make_report"]
    assert integ.CMD_PYTEST == commands["pytest"]
    source = (engine.VIDEO_DIR / "extract_data.py").read_text(encoding="utf-8")
    for log in (integ.LOG_BUILD, integ.LOG_TRAIN):
        assert Path(log).name in source                  # the logs the captures copy
    docs = (engine.REPO_ROOT / "docs" / "evaluation.md").read_text(encoding="utf-8")
    assert f"`{integ.CMD_BUILD}`" in docs and f"`{integ.CMD_TRAIN}`" in docs
    scripts_dir = engine.REPO_ROOT / "scripts"
    for _, names, _ in integ.STEPS:
        assert all((scripts_dir / name).exists() for name in names)


def test_integration_fit_line_cuts_only_long_lines_at_the_end():
    integ = scene_module("07_integration")
    assert integ.fit_line("abc", 5) == "abc"
    assert integ.fit_line("abcde", 5) == "abcde"
    assert integ.fit_line("abcdefgh", 5) == "abcd…"


def test_integration_terminal_window_fits_the_widest_line():
    integ = scene_module("07_integration")
    models = scene_module("06_models")
    pad = 2 * integ.TERM_PT / 72                         # style.terminal: one em each side
    assert models.text_width("m" * integ.MAX_CHARS, integ.TERM_PT, mono=True) + pad \
        < integ.TERM_BOX[2]


def test_integration_session_types_commands_then_prints_output():
    integ = scene_module("07_integration")
    segs = [(0.0, 0.2, 0.4, "> abcd"), (0.5, 0.5, 0.5, "out"), (0.6, 0.7, 0.8, "> xy")]
    assert integ.session_state(segs, 0.1) == (["> abcd"], 2)        # prompt waits
    assert integ.session_state(segs, 0.3) == (["> abcd"], 4)        # half typed
    assert integ.session_state(segs, 0.45) == (["> abcd"], 6)
    assert integ.session_state(segs, 0.55) == (["> abcd", "out"], 7 + 3)
    # all typed: two extra characters, so the finished session draws no cursor
    assert integ.session_state(segs, 0.9) == (["> abcd", "out", "> xy"], 7 + 4 + 4 + 2)
    idle = [*segs, (0.85, 2.0, 3.0, "> ")]                          # an idle prompt
    assert integ.session_state(idle, 0.9) == (["> abcd", "out", "> xy", "> "], 7 + 4 + 5 + 2)


@needs_data
def test_integration_steps_light_up_in_narrated_order():
    integ = scene_module("07_integration")
    ctx = make_ctx("07_integration", width=960, height=540)
    draw = integ.build(ctx)
    labels = [label for label, _, _ in integ.STEPS]
    first_lit = {}
    for frac in np.linspace(0.0, 0.5, 70):
        alphas = drawn_text_alpha(draw, frac * ctx.duration)
        for label in labels:
            if alphas[label] > 0.9:
                first_lit.setdefault(label, frac)
    assert list(first_lit) == labels
    assert sorted(first_lit.values()) == list(first_lit.values())
    assert first_lit["fetch"] > 0.05                     # not lit before it is named


def assert_summary_called_out(texts: list[str], pytest_lines: list[str], summary: str):
    """The terminal shows pytest's raw last line; the call-out shows the summary. Runs over a
    minute end the raw line with the wall time again (" (0:02:01)"), which the call-out drops;
    shorter runs make the two strings identical."""
    last = next(line for line in reversed(pytest_lines) if line.strip()).strip()
    assert last.startswith(summary)
    same = last == summary
    assert texts.count(summary) == 1 + same             # call-out (and terminal if same)
    assert texts.count(last) == 1 + same                # terminal (and call-out if same)


def test_integration_summary_line_drops_only_the_wall_time():
    integ = scene_module("07_integration")
    assert integ.summary_line(["", "5 passed in 61.2s (0:01:01)", ""]) == "5 passed in 61.2s"
    assert integ.summary_line(["....", "5 passed in 3.2s", ""]) == "5 passed in 3.2s"


@needs_data
@pytest.mark.parametrize("last_line", ["132 passed in 41.07s", "132 passed in 121.56s (0:02:01)"])
def test_integration_calls_out_the_summary_for_short_and_long_runs(tmp_path, last_line):
    """Both pytest summary forms, independent of which capture is on disk."""
    import shutil
    integ = scene_module("07_integration")
    data = tmp_path / "data"
    shutil.copytree(engine.DATA_DIR, data)
    lines = ["." * 40 + " [100%]", last_line]
    (data / "terminal" / "pytest.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    ctx = make_ctx("07_integration", width=960, height=540)
    ctx.data_dir = data
    draw = integ.build(ctx)
    t = scenes.cue_fraction(ctx, "Building the views", -1.0) * ctx.duration
    assert t >= 0
    texts = drawn_texts(draw, t - 0.3)
    summary = integ.summary_line(lines)
    assert summary == "132 passed in " + last_line.split(" in ")[1].split(" (")[0]
    assert_summary_called_out(texts, lines, summary)


@needs_data
def test_integration_replays_the_real_captures_with_their_captions():
    integ = scene_module("07_integration")
    ctx = make_ctx("07_integration", width=960, height=540)
    draw = integ.build(ctx)
    terminal = ctx.data_dir / "terminal"

    def capture(name):
        return (terminal / f"{name}.txt").read_text(encoding="utf-8").splitlines()

    def at(phrase):
        t = scenes.cue_fraction(ctx, phrase, -1.0) * ctx.duration
        assert t >= 0, phrase
        return t

    tests = drawn_texts(draw, at("Building the views") - 0.3)
    summary = integ.summary_line(capture("pytest"))
    assert re.fullmatch(r"\d+ passed in [\d.]+s", summary)
    assert_summary_called_out(tests, capture("pytest"), summary)
    assert "> " + integ.CMD_PYTEST in tests
    assert all(line in tests for line in capture("make_report"))

    built = drawn_texts(draw, at("Training") - 0.2)
    assert capture("build_views_tail")[-1] in built
    assert "several hours,\nresumable" in built
    assert any(integ.LOG_BUILD in s for s in built)      # the window names the log

    end = drawn_texts(draw, ctx.duration - 0.01)
    log = capture("train_cnn_full")
    assert all(line in end for line in log[-10:])        # the whole fold table fits
    assert ctx.numbers["cnn_full_train_time_text"] in end
    assert f"source: {ctx.numbers['sources']['cnn_full_train_time_text']}" in end
    assert any(integ.LOG_TRAIN in s for s in end)
    assert ctx.numbers["cnn_full_train_time_text"] not in drawn_texts(draw, at("Training"))


# --- 08_results ---------------------------------------------------------------------------

def cue_times(ctx, *phrases) -> dict[str, float]:
    """Seconds at which each phrase is spoken; asserts every cue exists in the narration."""
    at = {p: scenes.cue_fraction(ctx, p, -1.0) * ctx.duration for p in phrases}
    assert all(t >= 0 for t in at.values()), at
    return at


def test_results_spread_labels_keeps_order_and_gap():
    res = scene_module("08_results")
    ys = np.array([0.754, 0.752, 0.643, 0.487])
    out = res.spread_labels(ys, 0.06, 0.4, 0.97)
    assert list(np.argsort(out)) == list(np.argsort(ys))          # same order as the lines
    assert np.all(np.diff(np.sort(out)) >= 0.06 - 1e-9)
    assert out[2] == pytest.approx(0.643, abs=0.03) and out[3] == pytest.approx(0.487)
    assert np.allclose(res.spread_labels([0.5, 0.8], 0.06, 0.4, 0.97), [0.5, 0.8])
    assert res.bin_label("0.558 - 5.58", False) == "0.6–5.6"
    assert res.bin_label("51.4 - 4.3e+03", True) == "51+"


@needs_data
def test_results_scores_appear_as_narrated():
    ctx = make_ctx("08_results", width=960, height=540)
    draw = scene_module("08_results").build(ctx)
    test = ctx.numbers["test_pr_auc"]
    at = cue_times(ctx, "Without the image", "The neural network", "It misses", "About half")

    def score(m):
        return f"{test[m][0]:.3f}"

    def interval(m):
        return f"{test[m][1]:.3f}–{test[m][2]:.3f}"

    first = shown(draw, at["Without the image"] - 0.3)
    assert {score("tabular"), interval("tabular"), "all catalogue numbers"} <= first
    assert score("tabular_lc") not in first and score("cnn") not in first
    second = shown(draw, at["The neural network"] - 0.3)
    assert {score("tabular_lc"), "without image-shift numbers"} <= second
    assert score("cnn") not in second
    # the table holds all four test models, named in plain words, until the false positives
    table = shown(draw, at["About half"] - 0.4)
    for m in ("tabular", "tabular_lc", "tabular_lc_sub", "cnn"):
        assert {score(m), interval(m)} <= table, m
    assert {"Decision trees", "Neural network", "same training signals as the network",
            "Final exam (held-out test set)"} <= table
    assert not any("development" in s for s in table)


@needs_data
def test_results_curves_are_the_test_curves_in_model_colours():
    from matplotlib.colors import to_rgb
    from matplotlib.lines import Line2D
    ctx = make_ctx("08_results", width=960, height=540)
    draw = scene_module("08_results").build(ctx)
    pr = ctx.load("pr_curves.npz")
    at = cue_times(ctx, "It misses", "Retrained")
    fig = style.new_figure(960, 540)
    draw(fig, at["It misses"] - 0.3)
    lines = {ln.get_gid(): ln for ln in fig.findobj(Line2D) if ln.get_gid()}
    assert set(lines) == {"pr_tabular", "pr_tabular_lc", "pr_cnn"}     # the narrated three
    for m in ("tabular", "tabular_lc", "cnn"):
        assert np.allclose(lines[f"pr_{m}"].get_ydata(), pr[f"precision_{m}"])
    colours = {m: to_rgb(ln.get_color()) for m, ln in lines.items()}
    assert colours["pr_cnn"] == to_rgb(style.C_NET)
    assert colours["pr_tabular"] == colours["pr_tabular_lc"] == to_rgb(style.C_TREES)
    assert not {to_rgb(style.C_PLANET), to_rgb(style.C_FALSE)} & set(colours.values())

    fig = style.new_figure(960, 540)
    draw(fig, ctx.duration - 0.01)
    lines = {ln.get_gid(): ln for ln in fig.findobj(Line2D) if ln.get_gid()}
    assert set(lines) == {"pr_cnn_full", "pr_tabular_lc"}
    assert np.allclose(lines["pr_cnn_full"].get_ydata(), pr["precision_dev_cnn_full"])
    assert np.allclose(lines["pr_tabular_lc"].get_ydata(), pr["precision_dev_tabular_lc"])


@needs_data
def test_results_recall_bars_and_neighbouring_star_share():
    ctx = make_ctx("08_results", width=960, height=540)
    draw = scene_module("08_results").build(ctx)
    n = ctx.numbers
    errors = ctx.load("error_breakdown.json")
    at = cue_times(ctx, "About half", "which a light curve", "It learned")

    fig = style.new_figure(960, 540)
    draw(fig, at["About half"] - 0.2)
    heights = [b.get_height() for b in fig.findobj(lambda a: a.get_gid() == "recall_bar")]
    recalls = [b["recall"] for b in errors["by_snr"] + errors["by_period"]]
    assert np.allclose(np.array(heights) / heights[0], np.array(recalls) / recalls[0])
    bar_texts = shown(draw, at["About half"] - 0.2)
    assert {f"{r:.0%}" for r in recalls} <= bar_texts
    assert {"fainter  →  stronger", "shorter  →  longer"} <= bar_texts

    share = f"{n['cnn_fp_accepted_centroid']} of {n['cnn_fp_accepted']}: light from"
    assert not any(share in s for s in shown(draw, at["About half"] - 0.2))
    revealed = shown(draw, at["which a light curve"] + 0.8)
    assert any(share in s for s in revealed)
    assert "a light curve alone\ncannot reveal this" in revealed
    fig = style.new_figure(960, 540)
    draw(fig, at["which a light curve"] + 0.8)
    (dots,) = fig.findobj(lambda a: a.get_gid() == "fp_dots")
    (rings,) = fig.findobj(lambda a: a.get_gid() == "fp_centroid")
    assert len(dots.get_offsets()) == n["cnn_fp_accepted"]
    assert len(rings.get_offsets()) == n["cnn_fp_accepted_centroid"]


@needs_data
def test_results_dev_scores_only_as_practice_groups():
    ctx = make_ctx("08_results", width=960, height=540)
    draw = scene_module("08_results").build(ctx)
    n = ctx.numbers
    dev = n["dev_full_pr_auc"]
    at = cue_times(ctx, "It learned", "Retrained", "The gap")
    dev_scores = {f"{dev[m][0]:.3f}" for m in ("cnn_full", "tabular_lc")}

    learned = shown(draw, at["It learned"] + 1.0)
    assert f"first: {n['n_dev_cnn']:,} of {n['n_dev']:,} (21%)" in learned
    assert "The network learned from only part of the data" in learned     # heading fits
    assert "Final exam: what the neural network misses" not in learned
    for frac in np.linspace(0.0, 0.999, 40):
        t = frac * ctx.duration
        texts = drawn_texts(draw, t)
        assert not any("0.999" in s for s in texts), t          # never the leaked score
        if dev_scores & shown(draw, t):
            assert t > at["Retrained"]
            assert {"Practice groups (development folds)",
                    "practice groups (development folds)", "not the final exam"} <= shown(draw, t)
    end = shown(draw, at["The gap"])
    assert dev_scores <= end and "retrained on all of them" in end
    assert "Retrained: practice groups (development folds)" in end


# --- 09_meaning ---------------------------------------------------------------------------

@needs_data
def test_meaning_says_follow_up_not_discovery_and_the_readme_sentence():
    meaning = scene_module("09_meaning")
    ctx = make_ctx("09_meaning", width=960, height=540)
    draw = meaning.build(ctx)
    at = cue_times(ctx, "worth a follow-up", "not a discovery", "Next:")
    before = shown(draw, at["worth a follow-up"] - 0.4)
    assert "Worth a follow-up," not in before and "worth a follow-up" not in before
    reading = shown(draw, at["worth a follow-up"] + 0.8)
    assert {"Worth a follow-up,", "worth a follow-up"} <= reading
    sentence = meaning.wrap_meaning(ctx.numbers["score_meaning"])
    assert " ".join(sentence.split()) == ctx.numbers["score_meaning"]
    for t in (at["not a discovery"] + 1.5, ctx.duration - 0.01):  # it stays to the end
        assert {"Worth a follow-up,", "not a discovery.", sentence} <= shown(draw, t)


@needs_data
def test_meaning_next_steps_appear_as_spoken_and_no_score_is_shown():
    meaning = scene_module("09_meaning")
    ctx = make_ctx("09_meaning", width=960, height=540)
    draw = meaning.build(ctx)
    phrases = ("combine the two", "find more faint", "use pixel")
    at = cue_times(ctx, *phrases)
    steps = [meaning.wrap_step(s) for s in ctx.numbers["next_steps"]]
    assert [" ".join(s.split()) for s in steps] == ctx.numbers["next_steps"]
    for i, phrase in enumerate(phrases):
        assert steps[i] not in shown(draw, at[phrase] - 0.5), phrase
        assert set(steps[: i + 1]) <= shown(draw, at[phrase] + 0.8), phrase
    for frac in np.linspace(0.0, 0.999, 15):                # the dial is a bare 0-1 scale
        assert not any(re.search(r"\d\.\d", s) for s in drawn_texts(draw, frac * ctx.duration))


# --- 10_end -------------------------------------------------------------------------------

def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(int) - b.astype(int)).mean())


@needs_data
def test_loop_seam():
    ctx10 = make_ctx("10_end", width=960, height=540)
    ctx01 = make_ctx("01_hook", width=960, height=540)
    last = render(scene_module("10_end").build(ctx10), ctx10.duration - 1e-3)
    first = render(scene_module("01_hook").build(ctx01), 0.0)
    assert np.abs(last.astype(int) - first.astype(int)).mean() < 2.0


@needs_data
@pytest.mark.parametrize("draft", [True, False], ids=["draft", "full"])
def test_loop_seam_at_the_last_rendered_frame(draft):
    """The clip's real last frame (t = (n - 1) / fps, caption logic applied) is scene 1's
    first frame, so the cut from the end card back to the opening is invisible."""
    ctx10 = make_ctx("10_end", draft=draft)
    ctx01 = make_ctx("01_hook", draft=draft)
    t_last = float(engine.frame_times(ctx10)[-1])
    assert t_last < ctx10.duration
    cap10, cap01 = engine.caption_at(ctx10.cues, t_last), engine.caption_at(ctx01.cues, 0.0)
    assert cap10 is None and cap01 is None              # no caption on either side of the cut
    last = engine.render_frame(scene_module("10_end").build(ctx10), t_last, ctx10.width,
                               ctx10.height, caption=cap10)
    first = engine.render_frame(scene_module("01_hook").build(ctx01), 0.0, ctx01.width,
                                ctx01.height, caption=cap01)
    assert last.shape == first.shape == (ctx10.height, ctx10.width, 4)
    assert mean_abs_diff(last, first) < 2.0
    # and not trivially: the card itself is on screen before the fade
    mid = engine.render_frame(scene_module("10_end").build(ctx10), 0.5 * ctx10.duration,
                              ctx10.width, ctx10.height)
    assert mean_abs_diff(mid, first) > 2.0


def test_end_tools_are_real_dependencies():
    import tomllib
    end = scene_module("10_end")
    project = tomllib.loads((engine.REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = {re.split(r"[<>=!~ \[]", d)[0].lower() for d in project["project"]["dependencies"]}
    assert end.TOOLS[0] == ("Python", None)              # the language, not a package
    for label, package in end.TOOLS[1:]:
        assert package in deps, label
    assert [label for label, _ in end.TOOLS] == ["Python", "NumPy", "pandas", "scikit-learn",
                                                 "PyTorch", "lightkurve", "matplotlib"]


FULL_CREDITS = {"title": "Kepler Exoplanet Transit Detection",
                "name": "Aaaaaaaaaaaa Bbbbbbbbbbbbbbbbb",                 # 30 characters
                "team": ["Cccccccccc Dddddddddddddddddd", "Eeeeeeeeeeeee Fffffffffffffff",
                         "Gggggggg Hhhhhhhhhhhhhhhhhhhhh", "Iiiiiiiiiiiiiii Jjjjjjjjjjjjj"],
                "institution": "Department of Physics and Astronomy, Kkkkkkkkkkk University "
                               "of Technology",                           # ~85 characters
                "guide": "Prof. Llllllll Mmmmmmmmmmmmmm", "voice": "en-IN-NeerjaNeural"}


def end_frame_texts(credits: dict, width: int = 960, height: int = 540):
    """(fig, renderer, visible texts) of the end card with these credits, all of it shown."""
    ctx = make_ctx("10_end", width=width, height=height)
    ctx.credits = credits
    draw = scene_module("10_end").build(ctx)
    t = scenes.cue_fraction(ctx, "Thank you", -1.0) * ctx.duration
    assert t > 0
    fig = style.new_figure(width, height)
    draw(fig, t + 0.3)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = [a for a in fig.findobj(Text) if a.get_visible() and a.get_text()]
    return fig, renderer, texts


def assert_fits_and_no_overlap(fig, renderer, texts):
    width, height = fig.bbox.width, fig.bbox.height
    band = style.CAPTION_BAND * height
    boxes = []
    for a in texts:
        bb = a.get_window_extent(renderer)
        patch = a.get_bbox_patch()
        if patch is not None:
            bb = patch.get_window_extent(renderer)
        assert bb.x0 >= 0 and bb.x1 <= width and bb.y1 <= height, a.get_text()
        assert bb.y0 >= band, f"{a.get_text()!r} enters the caption band"
        boxes.append((a.get_text(), bb))
    for i, (ta, a) in enumerate(boxes):
        for tb, b in boxes[i + 1:]:
            assert not a.overlaps(b), (ta, tb)


@needs_data
@pytest.mark.parametrize("width,height", [(960, 540), (1920, 1080)])
def test_end_shows_every_credit_and_fits(width, height):
    fig, renderer, texts = end_frame_texts(FULL_CREDITS, width, height)
    strings = [a.get_text() for a in texts]
    joined = " ".join(" ".join(strings).split())
    assert FULL_CREDITS["title"] in strings
    assert FULL_CREDITS["name"] in strings
    assert all(member in joined for member in FULL_CREDITS["team"])        # team of four
    assert FULL_CREDITS["institution"] in joined
    inst = next(s for s in strings if s.startswith("Department"))
    assert inst.count("\n") == 1                                           # wraps to two lines
    assert f"Guide: {FULL_CREDITS['guide']}" in strings
    for tool in ("Python", "NumPy", "pandas", "scikit-learn", "PyTorch", "lightkurve",
                 "matplotlib"):
        assert tool in strings
    assert "Data: NASA Kepler mission, NASA Exoplanet Archive, MAST" in strings
    assert_fits_and_no_overlap(fig, renderer, texts)


@needs_data
def test_end_renders_the_placeholder_credits():
    placeholders = {"title": "Kepler Exoplanet Transit Detection", "name": "Your Name",
                    "team": [], "institution": "Institution", "guide": "Guide Name"}
    _, _, texts = end_frame_texts(placeholders)
    strings = [a.get_text() for a in texts]
    assert placeholders["title"] in strings and placeholders["name"] in strings
    assert "Guide: Guide Name" in strings


@needs_data
def test_end_renders_the_credits_file_and_fits():
    """Whatever video/credits.json holds is shown and fits (an empty guide has no label)."""
    ctx = make_ctx("10_end", width=960, height=540)
    fig, renderer, texts = end_frame_texts(ctx.credits)
    strings = [a.get_text() for a in texts]
    assert ctx.credits["title"] in strings
    assert any(" ".join(ctx.credits["name"].split()) == " ".join(s.split()) for s in strings)
    if not (ctx.credits.get("guide") or "").strip():
        assert not any(s.startswith("Guide") for s in strings)
    assert_fits_and_no_overlap(fig, renderer, texts)


@needs_data
def test_end_omits_empty_fields_without_a_gap():
    sparse = {**FULL_CREDITS, "team": [], "guide": "", "institution": ""}
    fig, renderer, texts = end_frame_texts(sparse)
    strings = [a.get_text() for a in texts]
    assert FULL_CREDITS["name"] in strings
    assert not any(s.strip() in ("", "with", "Guide:", "Team:") for s in strings)
    assert not any("Guide" in s or "with" == s.split()[0] for s in strings)
    assert not any(m in " ".join(strings) for m in FULL_CREDITS["team"])
    assert_fits_and_no_overlap(fig, renderer, texts)

    # only the guide present: it sits right under the name, the same gap as any other line
    def gap_below_name(credits):
        fig, renderer, texts = end_frame_texts(credits)
        by_text = {a.get_text(): a.get_window_extent(renderer) for a in texts}
        name = by_text[credits["name"]]
        below = [bb for s, bb in by_text.items() if bb.y1 < name.y0 and s != credits["name"]]
        return name.y0 - max(bb.y1 for bb in below), fig.dpi

    gap_full, dpi = gap_below_name(FULL_CREDITS)
    gap_guide_only, _ = gap_below_name({**sparse, "guide": "Prof. X"})
    assert gap_guide_only == pytest.approx(gap_full, abs=0.05 * dpi)      # within 0.05 in
