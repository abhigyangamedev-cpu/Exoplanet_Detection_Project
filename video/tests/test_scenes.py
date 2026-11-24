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


# --- 02_problem -------------------------------------------------------------------------

@needs_data
def test_problem_counts_up_to_the_real_numbers():
    ctx = make_ctx("02_problem", width=960, height=540)
    draw = scene_module("02_problem").build(ctx)
    n = ctx.numbers
    final = {f"{n['n_confirmed']:,}", f"{n['n_false_positive']:,}", f"{n['n_kois']:,}"}
    assert final <= set(drawn_texts(draw, ctx.duration - 0.01))
    assert not final & set(drawn_texts(draw, 0.0))

    # the counters count up: every number on screen only grows, through values in between
    seen: dict[int, list[int]] = {}
    for frac in np.linspace(0.25, 0.999, 40):
        for i, s in enumerate(sorted_counts(drawn_texts(draw, frac * ctx.duration))):
            seen.setdefault(i, []).append(s)
    for target in (n["n_kois"], n["n_confirmed"], n["n_false_positive"]):
        series = next(v for v in seen.values() if v[-1] == target)
        assert series == sorted(series)
        assert any(0 < v < target for v in series), target


def sorted_counts(texts: list[str]) -> list[int]:
    """The plain numbers on screen ("2,731"), in drawing order."""
    return [int(s.replace(",", "")) for s in texts if re.fullmatch(r"\d[\d,]*", s)]


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
