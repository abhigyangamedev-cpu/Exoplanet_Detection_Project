"""Tests for the render engine (engine.py) and shared style (style.py).

The demo scene lives here, not in video/scenes/: it is registered in sys.modules as
``scenes.s99_demo`` so ``render_scene`` loads it through the real ``import_scene`` path.
"""

import json
import subprocess
import sys
import types
import wave

import engine
import imageio_ffmpeg
import numpy as np
import pytest
import scenes
import style
from narration import SCENE_IDS

# --- demo scene ---------------------------------------------------------------------

def demo_build(ctx):
    """A small scene: a title, a growing line and a dot, all timed in fractions."""
    rng = np.random.default_rng(0)
    xs = rng.uniform(0, 1, 200)
    ys = rng.uniform(0, 1, 200)

    def draw(fig, t):
        f = t / ctx.duration
        ax = fig.add_axes((0.08, style.CAPTION_BAND + 0.05, 0.84, 0.6))
        k = max(2, int(200 * style.ease(style.progress(f, 0.1, 0.8))))
        ax.scatter(xs[:k], ys[:k], s=20, c=style.C_PLANET)
        ax.plot([0, 1], [0, style.ease(style.progress(f, 0.0, 0.8))], color=style.C_FALSE)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        style.heading(fig, "Demo scene")

    return draw


@pytest.fixture
def demo_scene(monkeypatch):
    module = types.ModuleType("scenes.s99_demo")
    module.build = demo_build
    monkeypatch.setitem(sys.modules, "scenes.s99_demo", module)
    return "99_demo"


def demo_context(duration, width=960, height=540, fps=15, cues=None):
    return engine.Context(
        scene_id="99_demo", duration=duration, fps=fps, width=width, height=height,
        numbers={}, data_dir=engine.DATA_DIR, cues=cues or [], credits={},
    )


def silent_wav(path, seconds, rate=48000, beep=False):
    """A mono 16-bit WAV of silence; with beep, a 50 ms 1 kHz tone at its very start."""
    samples = np.zeros(int(seconds * rate))
    if beep:
        n = rate // 20
        samples[:n] = 0.5 * np.sin(2 * np.pi * 1000 * np.arange(n) / rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((samples * 32767).astype("<i2").tobytes())
    return path


def beep_onsets(path, rate=48000):
    """Seconds at which a tone starts after at least 0.2 s of quiet."""
    cmd = [engine.FFMPEG, "-loglevel", "error", "-i", str(path), "-f", "s16le", "-ac", "1",
           "-ar", str(rate), "-"]
    pcm = subprocess.run(cmd, capture_output=True, check=True).stdout
    loud = np.abs(np.frombuffer(pcm, "<i2").astype(float)) > 3000
    quiet = int(0.2 * rate)
    return [i / rate for i in np.flatnonzero(loud) if not loud[max(0, i - quiet):i].any()]


def frame_size(path):
    gen = imageio_ffmpeg.read_frames(str(path))
    meta = next(gen)
    gen.close()
    return tuple(meta["size"])


# --- style ---------------------------------------------------------------------------

def test_ease_and_progress_clamp():
    assert style.ease(-1) == 0.0 and style.ease(2) == 1.0
    assert style.ease(0.5) == pytest.approx(0.5)
    assert style.ease(0.25) < 0.25  # smoothstep starts slow
    assert style.progress(0.0, 0.2, 0.6) == 0.0
    assert style.progress(0.4, 0.2, 0.6) == pytest.approx(0.5)
    assert style.progress(0.9, 0.2, 0.6) == 1.0


def test_class_colours_are_distinct_and_defined_once():
    assert style.C_PLANET != style.C_FALSE
    assert len({style.BG, style.FG, style.C_PLANET, style.C_FALSE, style.C_ACCENT}) == 5


def test_fonts_exist():
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    assert style.FONT in names and style.MONO in names


def test_new_figure_has_requested_pixel_size():
    for w, h in [(1920, 1080), (960, 540)]:
        fig = style.new_figure(w, h)
        fig.canvas.draw()
        assert np.asarray(fig.canvas.buffer_rgba()).shape == (h, w, 4)


def caption_artists(fig):
    return [t for t in fig.texts if t.get_gid() == "caption"]


@pytest.mark.parametrize("text", [
    "word " * 17,
    "A considerably longer caption with unusually lengthy vocabulary demonstrating wrapping",
    "WWWWWWWWWW WWWWWWWWWW WWWWWWWWWW WWWWWWWWWW WWWWWWWWWW WWWWWWWWWW WWWWWWWWWW WWWW",
])
def test_caption_wraps_to_two_lines_within_frame(text):
    fig = style.new_figure(960, 540)
    style.draw_caption(fig, text)
    fig.canvas.draw()
    (caption,) = caption_artists(fig)
    assert caption.get_text().count("\n") <= 1
    assert caption.get_text().split() == text.split()  # nothing dropped
    box = caption.get_window_extent()
    assert box.x0 >= 0 and box.x1 <= 960
    assert box.y0 >= 0 and box.y1 <= style.CAPTION_BAND * 540  # inside the bottom band


def test_short_caption_stays_on_one_line():
    fig = style.new_figure(960, 540)
    style.draw_caption(fig, "Which are planets?")
    (caption,) = caption_artists(fig)
    assert "\n" not in caption.get_text()


def test_terminal_and_starfield_draw():
    fig = style.new_figure(960, 540)
    ax = fig.add_axes((0, 0, 1, 1))
    style.starfield(ax, seed=1, t=0.3)
    term = fig.add_axes((0.1, 0.3, 0.8, 0.6))
    lines = ["$ pytest -q", "........", "12 passed in 3.1s"]
    style.terminal(term, lines, chars_shown=14)
    fig.canvas.draw()
    shown = "\n".join(t.get_text() for t in term.texts)
    assert "$ pytest -q" in shown and "passed" not in shown  # typing stops after 14 chars


def test_terminal_scrolls_to_latest_lines():
    fig = style.new_figure(960, 540)
    term = fig.add_axes((0.1, 0.3, 0.8, 0.4))
    lines = [f"line {i:03d}" for i in range(200)]
    style.terminal(term, lines, chars_shown=10**6)
    shown = [t.get_text() for t in term.texts]
    assert "line 199" in shown and "line 000" not in shown


def test_starfield_is_deterministic():
    draw = lambda fig, t: style.starfield(fig.add_axes((0, 0, 1, 1)), seed=3, t=t)  # noqa: E731
    a = engine.render_frame(draw, 0.4, 320, 180)
    b = engine.render_frame(draw, 0.4, 320, 180)
    assert np.array_equal(a, b)


# --- engine --------------------------------------------------------------------------

def test_import_scene_missing_raises_importerror():
    with pytest.raises(ImportError):
        scenes.import_scene("98_does_not_exist")


def test_pad_matches_tts():
    import tts
    assert engine.PAD == tts.SCENE_PAD == 0.6


def test_render_frame_shape_and_dtype(demo_scene):
    draw = demo_build(demo_context(2.0))
    frame = engine.render_frame(draw, 1.0, 960, 540)
    assert frame.shape == (540, 960, 4) and frame.dtype == np.uint8


def test_caption_at():
    cues = [(0.0, 1.0, "one"), (1.0, 2.5, "two")]
    assert engine.caption_at(cues, 0.5) == "one"
    assert engine.caption_at(cues, 1.0) == "two"
    assert engine.caption_at(cues, 3.0) is None


def test_render_scene_length_follows_audio(tmp_path, demo_scene):
    wav = silent_wav(tmp_path / "a.wav", 2.0)
    out = tmp_path / "clip.mp4"
    ctx = demo_context(2.0, cues=[(0.0, 2.0, "A caption on the demo scene")])
    engine.render_scene(demo_scene, ctx, wav, out)
    video_s, audio_s = engine.media_durations(out)
    assert abs(video_s - 2.0) < 0.1
    assert abs(audio_s - video_s) < 0.03
    assert frame_size(out) == (960, 540)
    nframes, _ = imageio_ffmpeg.count_frames_and_secs(str(out))
    assert nframes == round(2.0 * 15)


def test_audio_longer_than_duration_is_trimmed(tmp_path, demo_scene):
    wav = silent_wav(tmp_path / "a.wav", 3.0)
    out = tmp_path / "clip.mp4"
    engine.render_scene(demo_scene, demo_context(1.0, 320, 180), wav, out)
    video_s, audio_s = engine.media_durations(out)
    assert abs(video_s - 1.0) < 0.05 and abs(audio_s - 1.0) < 0.03


def test_scene_stretches(demo_scene):
    """The same scene at 1 s and 5 s shows the same picture at the same fraction."""
    short_ctx, long_ctx = demo_context(1.0), demo_context(5.0)
    short, long = demo_build(short_ctx), demo_build(long_ctx)
    for frac in (0.0, 0.5):
        assert np.array_equal(engine.render_frame(short, frac * 1.0, 960, 540),
                              engine.render_frame(long, frac * 5.0, 960, 540))
    last_short = engine.frame_times(short_ctx)[-1]
    last_long = engine.frame_times(long_ctx)[-1]
    assert np.array_equal(engine.render_frame(short, last_short, 960, 540),
                          engine.render_frame(long, last_long, 960, 540))
    # and the scene really animates in between
    assert not np.array_equal(engine.render_frame(long, 0.0, 960, 540),
                              engine.render_frame(long, 2.5, 960, 540))


def test_frame_times():
    ctx = demo_context(2.0)
    times = engine.frame_times(ctx)
    assert len(times) == 30 and times[0] == 0.0 and times[1] == pytest.approx(1 / 15)


def test_concat_sums_durations(tmp_path, demo_scene):
    lengths = [1.0, 1.4, 0.8]
    clips = []
    for i, seconds in enumerate(lengths):
        wav = silent_wav(tmp_path / f"a{i}.wav", seconds - engine.PAD + 0.0123, beep=True)
        clips.append(tmp_path / f"c{i}.mp4")
        engine.render_scene(demo_scene, demo_context(seconds, 320, 180), wav, clips[-1])
    out = tmp_path / "all.mp4"
    total = engine.concat(clips, out)
    assert abs(total - sum(lengths)) < 0.1
    video_s, audio_s = engine.media_durations(out)
    assert abs(video_s - sum(lengths)) < 0.1
    assert abs(audio_s - video_s) < 0.01
    # each clip's beep sounds when its first frame shows: no offset, no drift
    clip_starts = np.cumsum([0.0] + lengths[:-1])
    onsets = beep_onsets(out)
    assert len(onsets) == len(lengths)
    assert np.max(np.abs(np.array(onsets) - clip_starts)) < 0.005


@pytest.mark.skipif(not (engine.DATA_DIR / "numbers.json").exists()
                    or not engine.MANIFEST.exists(), reason="build data/audio not generated")
def test_make_context_from_build_data():
    manifest = json.loads(engine.MANIFEST.read_text(encoding="utf-8"))
    ctx = engine.make_context("01_hook", draft=True)
    assert ctx.duration == pytest.approx(manifest["01_hook"]["duration"] + engine.PAD)
    assert (ctx.width, ctx.height, ctx.fps) == (960, 540, 15)
    assert ctx.numbers["n_kois"] > 0 and "title" in ctx.credits
    assert ctx.cues and all(len(text) <= 84 for _, _, text in ctx.cues)
    assert ctx.audio_wav.exists()
    full = engine.make_context("01_hook", duration=3.0)
    assert (full.width, full.height, full.fps, full.duration) == (1920, 1080, 30, 3.0)
    assert isinstance(ctx.load("pr_curves.npz")["recall_cnn"], np.ndarray)
    assert isinstance(ctx.load("error_breakdown.json"), (dict, list))


def test_cli_preview_and_render(tmp_path, monkeypatch, demo_scene, capsys):
    cues = [(0.0, 1.0, "A caption for the preview")]
    monkeypatch.setattr(engine, "make_context",
                        lambda scene_id, draft=False: demo_context(1.0, 320, 180, cues=cues))
    monkeypatch.setattr(engine, "PREVIEW_DIR", tmp_path / "preview")
    monkeypatch.setattr(engine, "CLIPS_DIR", tmp_path / "clips")
    engine.main(["preview", demo_scene, "--at", "0.5", "--draft"])
    png = tmp_path / "preview" / "99_demo_0.50_draft.png"
    assert png.exists() and str(png) in capsys.readouterr().out
    engine.main(["render", demo_scene, "--draft"])
    clip = tmp_path / "clips" / "99_demo_draft.mp4"
    assert abs(engine.media_durations(clip)[0] - 1.0) < 0.1
    assert "s/frame" in capsys.readouterr().out


def test_every_scene_id_maps_to_its_module_name(monkeypatch):
    for scene_id in SCENE_IDS:
        module = types.ModuleType(f"scenes.s{scene_id}")
        monkeypatch.setitem(sys.modules, f"scenes.s{scene_id}", module)
        assert scenes.import_scene(scene_id) is module
