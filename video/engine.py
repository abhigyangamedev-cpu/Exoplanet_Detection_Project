"""Render engine: one scene's matplotlib frames, timed to its narration, encoded to MP4.

A scene module (``scenes/sNN_name.py``) exposes::

    def build(ctx: Context) -> Callable[[Figure, float], None]

``build`` loads data once; the returned ``draw(fig, t)`` draws the frame at ``t`` seconds on
an already-cleared figure. Express timing as fractions of ``ctx.duration`` (``f = t /
ctx.duration``), never absolute seconds, so a scene stretches with its audio. Keep the bottom
``style.CAPTION_BAND`` clear: the engine draws the active caption there after ``draw``.

Frame i is drawn at ``t = i / fps`` for ``i in range(round(duration * fps))``. Frames are
piped as raw RGBA to ffmpeg (H.264 yuv420p + AAC); the clip's audio is the scene WAV padded
with silence (or trimmed) to exactly the video length, so clips concatenate without drift.

Look at work in progress (from the repo root, video env)::

    C:\\video_env\\Scripts\\python video/engine.py preview 04_leakage --at 0.5 [--draft]
    C:\\video_env\\Scripts\\python video/engine.py render 04_leakage [--draft]

``preview`` writes video/build/preview/<id>_<at>.png (with caption); ``render`` writes
video/build/clips/<id>[_draft].mp4 with audio and prints seconds per frame.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from matplotlib.figure import Figure

VIDEO_DIR = Path(__file__).resolve().parent
if str(VIDEO_DIR) not in sys.path:  # so `python video/engine.py` finds style, tts, scenes
    sys.path.insert(0, str(VIDEO_DIR))

import scenes  # noqa: E402
import style  # noqa: E402
import tts  # noqa: E402

REPO_ROOT = VIDEO_DIR.parent
BUILD_DIR = VIDEO_DIR / "build"
DATA_DIR = BUILD_DIR / "data"
MANIFEST = BUILD_DIR / "audio" / "manifest.json"
CREDITS = VIDEO_DIR / "credits.json"
PREVIEW_DIR = BUILD_DIR / "preview"
CLIPS_DIR = BUILD_DIR / "clips"

PAD = tts.SCENE_PAD  # 0.6 s of silence appended to each scene's audio
FULL = (1920, 1080, 30)   # width, height, fps
DRAFT = (960, 540, 15)
SAMPLE_RATE = 48000
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

Draw = Callable[[Figure, float], None]
Cue = tuple[float, float, str]


@dataclass
class Context:
    """Everything a scene's build() may use. Timing in fractions of ``duration``."""

    scene_id: str
    duration: float          # seconds: narration + PAD (or an override)
    fps: int
    width: int
    height: int
    numbers: dict            # video/build/data/numbers.json
    data_dir: Path
    cues: list[Cue]          # (start s, end s, caption text), drawn by the engine
    credits: dict            # video/credits.json
    audio_wav: Path | None = field(default=None)  # narration WAV, None without audio

    def load(self, name: str):
        """Load ``data_dir/name``: a .npz returns a dict of arrays, a .json the parsed value."""
        path = Path(self.data_dir) / name
        if path.suffix == ".npz":
            with np.load(path) as npz:
                return {k: npz[k] for k in npz.files}
        if path.suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        raise ValueError(f"cannot load {name!r}: expected .npz or .json")


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_context(scene_id: str, *, draft: bool = False, duration: float | None = None,
                 width: int | None = None, height: int | None = None) -> Context:
    """Context for a scene from the build data, credits and the audio manifest.

    duration defaults to the narration length + PAD; captions come from the narration's
    word timings. Draft is 960x540 at 15 fps, otherwise 1920x1080 at 30 fps.
    """
    w, h, fps = DRAFT if draft else FULL
    entry = _read_json(MANIFEST).get(scene_id) if MANIFEST.exists() else None
    wav, cues = None, []
    if entry:
        wav = REPO_ROOT / entry["wav"]
        audio_s = tts.wav_duration(wav)
        meta = _read_json(wav.with_suffix(".json"))
        words = [(float(a), float(b), str(x)) for a, b, x in meta["words"]]
        cues = tts.caption_cues(meta["text"], words, audio_s)
        if duration is None:
            duration = audio_s + PAD
    if duration is None:
        raise FileNotFoundError(f"no audio for {scene_id} in {MANIFEST}; run video/tts.py "
                                "or pass duration=")
    return Context(
        scene_id=scene_id, duration=float(duration), fps=fps, width=width or w,
        height=height or h, numbers=_read_json(DATA_DIR / "numbers.json"), data_dir=DATA_DIR,
        cues=cues, credits=_read_json(CREDITS), audio_wav=wav,
    )


def frame_times(ctx: Context) -> np.ndarray:
    """Time in seconds of every frame of the scene: i / fps."""
    return np.arange(round(ctx.duration * ctx.fps)) / ctx.fps


def caption_at(cues: list[Cue], t: float) -> str | None:
    """Text of the cue active at t (start <= t < end), else None."""
    for start, end, text in cues:
        if start <= t < end:
            return text
    return None


def _clear(fig: Figure) -> None:
    # Figure.clear() first clears each axes (rebuilding all its ticks) only to discard it;
    # removing the axes first skips that work, about 8 ms per axes per frame.
    for ax in list(fig.axes):
        fig.delaxes(ax)
    fig.clear()


def _draw_frame(fig: Figure, draw: Draw, t: float, caption: str | None) -> memoryview:
    _clear(fig)
    draw(fig, t)
    if caption:
        style.draw_caption(fig, caption)
    fig.canvas.draw()
    return fig.canvas.buffer_rgba()


def render_frame(draw: Draw, t: float, width: int, height: int,
                 caption: str | None = None) -> np.ndarray:
    """One frame as an (height, width, 4) uint8 RGBA array; no caption unless given."""
    fig = style.new_figure(width, height)
    return np.array(_draw_frame(fig, draw, t, caption))


def _close_pipe(stream) -> bool:
    """Close a buffered pipe even if its last flush fails; True if it closed cleanly."""
    try:
        stream.close()
        return True
    except OSError:
        stream.raw.close()  # the flush failed, so close() left the descriptor open
        return False


def render_scene(scene_id: str, ctx: Context, audio_wav: Path | None, out_mp4: Path) -> None:
    """Render scene ``scene_id`` to out_mp4: round(duration * fps) frames, audio muxed.

    The audio is audio_wav padded with silence (or trimmed) to exactly the video length;
    with audio_wav None the clip carries silence. Captions come from ctx.cues.
    """
    draw = scenes.import_scene(scene_id).build(ctx)
    times = frame_times(ctx)
    length = len(times) / ctx.fps
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    part = out_mp4.with_name(out_mp4.stem + ".part.mp4")

    if audio_wav is None:
        audio_in = ["-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=stereo"]
    else:
        audio_in = ["-i", str(audio_wav)]
    cmd = [
        FFMPEG, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{ctx.width}x{ctx.height}",
        "-r", str(ctx.fps), "-i", "-",
        *audio_in,
        "-map", "0:v", "-map", "1:a",
        "-af", f"apad=whole_dur={length:.6f},atrim=end={length:.6f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", str(SAMPLE_RATE), "-ac", "2",
        "-movflags", "+faststart", str(part),
    ]
    fig = style.new_figure(ctx.width, ctx.height)
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err)
        try:
            pipe_broken = False
            try:
                for t in times:
                    proc.stdin.write(_draw_frame(fig, draw, float(t), caption_at(ctx.cues, t)))
            except OSError:
                # ffmpeg exited early (Windows raises EINVAL, not always BrokenPipeError);
                # its own error message is reported below
                pipe_broken = True
            pipe_broken |= not _close_pipe(proc.stdin)
            code = proc.wait()
        except BaseException:
            # the scene (or caption drawing) failed: stop ffmpeg, which is blocked on stdin
            proc.kill()
            _close_pipe(proc.stdin)
            proc.wait()
            part.unlink(missing_ok=True)
            raise
        if code != 0 or pipe_broken:
            err.seek(0)
            part.unlink(missing_ok=True)
            message = err.read().decode(errors="replace").strip() or f"exit code {code}"
            raise RuntimeError(f"ffmpeg failed for {scene_id}: {message}")
    os.replace(part, out_mp4)


def media_durations(path: Path) -> tuple[float, float]:
    """(video seconds, audio seconds) of an MP4, from its decoded frames and samples."""
    gen = imageio_ffmpeg.read_frames(str(path))
    fps = next(gen)["fps"]
    gen.close()
    nframes, _ = imageio_ffmpeg.count_frames_and_secs(str(path))
    pcm = subprocess.run(
        [FFMPEG, "-loglevel", "error", "-i", str(path), "-map", "0:a:0", "-f", "s16le",
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-"],
        capture_output=True, check=True,
    ).stdout
    return nframes / fps, len(pcm) / 2 / SAMPLE_RATE


TARGET_LUFS = -16.0        # integrated loudness of the joined video (a hall, laptop speakers)
TARGET_PEAK_DB = -1.5      # true-peak ceiling, dBTP
MAX_GAIN_DB = 20.0


def loudness_gain(audio_inputs: list[str], joined: str, n: int) -> float:
    """dB of plain gain that brings the joined audio to TARGET_LUFS without its true peak
    passing TARGET_PEAK_DB, from ffmpeg's loudnorm measurement (EBU R128).

    A single static gain changes no timing (unlike loudnorm's dynamic mode), so the audio
    stays sample-aligned with the video. Silence gets no gain.
    """
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-nostats", *audio_inputs, "-filter_complex",
         f"{joined}concat=n={n}:v=0:a=1,loudnorm=I={TARGET_LUFS}:TP={TARGET_PEAK_DB}"
         ":print_format=json", "-f", "null", "-"],
        capture_output=True, check=True,
    )
    err = proc.stderr.decode(errors="replace")
    stats = json.loads(err[err.rindex("{"):err.rindex("}") + 1])
    measured_i, measured_tp = float(stats["input_i"]), float(stats["input_tp"])
    if not (np.isfinite(measured_i) and np.isfinite(measured_tp)) or measured_i < -70:
        return 0.0
    gain = min(TARGET_LUFS - measured_i, TARGET_PEAK_DB - measured_tp)
    return float(np.clip(gain, -MAX_GAIN_DB, MAX_GAIN_DB))


def concat(clips: list[Path], out_mp4: Path) -> float:
    """Join clips (same size and fps) into out_mp4; returns its length in seconds.

    Video packets are copied through the concat demuxer. Audio is decoded per clip and joined
    with the concat filter, then encoded once: the concat demuxer would keep each AAC
    stream's 1024 priming samples and shift all audio 21 ms late. The joined audio is
    brought to about TARGET_LUFS by one measured gain (``loudness_gain``).
    """
    out_mp4 = Path(out_mp4)
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    listing = "".join(
        "file '{}'\n".format(Path(c).resolve().as_posix().replace("'", "'\\''")) for c in clips
    )
    audio_inputs = [arg for c in clips for arg in ("-i", str(c))]
    joined = "".join(f"[{i + 1}:a]" for i in range(len(clips)))
    measure_joined = "".join(f"[{i}:a]" for i in range(len(clips)))
    part = out_mp4.with_name(out_mp4.stem + ".part.mp4")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write(listing)
    try:
        gain = loudness_gain(audio_inputs, measure_joined, len(clips))
        subprocess.run(
            [FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", f.name,
             *audio_inputs,
             "-filter_complex",
             f"{joined}concat=n={len(clips)}:v=0:a=1,volume={gain:.2f}dB[a]",
             "-map", "0:v", "-map", "[a]", "-c:v", "copy",
             "-c:a", "aac", "-b:a", "192k", "-ar", str(SAMPLE_RATE), "-ac", "2",
             "-movflags", "+faststart", str(part)],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg concat failed: {e.stderr.decode(errors='replace')}") from e
    finally:
        os.unlink(f.name)
    os.replace(part, out_mp4)
    return media_durations(out_mp4)[0]


# --- CLI --------------------------------------------------------------------------------

def _preview(scene_id: str, at: float, draft: bool) -> Path:
    ctx = make_context(scene_id, draft=draft)
    draw = scenes.import_scene(scene_id).build(ctx)
    t = min(max(at, 0.0), 1.0) * ctx.duration
    frame = render_frame(draw, t, ctx.width, ctx.height, caption=caption_at(ctx.cues, t))
    out = PREVIEW_DIR / f"{scene_id}_{at:.2f}{'_draft' if draft else ''}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.fromarray(frame).save(out)
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Preview or render one scene.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("preview", help="write a PNG of one frame (with caption)")
    p.add_argument("scene_id")
    p.add_argument("--at", type=float, default=0.5, help="fraction of the scene, 0-1")
    p.add_argument("--draft", action="store_true")
    r = sub.add_parser("render", help="write the scene clip with audio")
    r.add_argument("scene_id")
    r.add_argument("--draft", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "preview":
        print(_preview(args.scene_id, args.at, args.draft))
        return
    ctx = make_context(args.scene_id, draft=args.draft)
    out = CLIPS_DIR / f"{args.scene_id}{'_draft' if args.draft else ''}.mp4"
    start = time.perf_counter()
    render_scene(args.scene_id, ctx, ctx.audio_wav, out)
    elapsed = time.perf_counter() - start
    n = len(frame_times(ctx))
    print(f"{out}  {ctx.duration:.2f} s, {n} frames, {elapsed / n:.3f} s/frame")


if __name__ == "__main__":
    main()
