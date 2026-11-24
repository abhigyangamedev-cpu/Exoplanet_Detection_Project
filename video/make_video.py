"""Build the whole explainer video with one command (video env, from the repo root):

    C:\\video_env\\Scripts\\python video/make_video.py [--draft] [--scenes 4,5] [--voice NAME] [--jobs N]

Steps: check video/build/data exists; synthesize each scene's narration (cached, so only
changed scenes go online); render the scene clips in parallel worker processes, reusing a
clip whose inputs have not changed; join them in SCENE_IDS order; check the total length is
within 3:45-4:30; print a summary.

Output: video/exoplanet_project_video.mp4 (1920x1080, 30 fps), or with --draft
video/build/exoplanet_project_video_draft.mp4 (960x540, 15 fps). ``--scenes`` renders only
those scene clips (numbers 1-10 or scene ids) and prints their paths; it does not join them.

A clip is reused when its cache key matches the key stored beside it
(video/build/clips/<id>[_draft].key). The key covers the narration audio and word timings,
the resolution and fps, the scene's own source and every scene module it imports
(scene 10 imports scene 1), the shared sources (style.py, engine.py, tts.py,
scenes/__init__.py), credits.json and the content of every file under video/build/data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import re
import sys
import time
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

VIDEO_DIR = Path(__file__).resolve().parent
if str(VIDEO_DIR) not in sys.path:  # so `python video/make_video.py` finds engine, tts, ...
    sys.path.insert(0, str(VIDEO_DIR))

import engine  # noqa: E402
import tts  # noqa: E402
from narration import SCENE_IDS, load_narration  # noqa: E402

BUILD_DIR = engine.BUILD_DIR
DATA_DIR = engine.DATA_DIR
CLIPS_DIR = engine.CLIPS_DIR
AUDIO_DIR = tts.AUDIO_DIR
OUTPUT = VIDEO_DIR / "exoplanet_project_video.mp4"
DRAFT_OUTPUT = BUILD_DIR / "exoplanet_project_video_draft.mp4"

MIN_SECONDS, MAX_SECONDS = 225.0, 270.0  # 3:45-4:30
SHARED_SOURCES = ("style.py", "engine.py", "tts.py", "scenes/__init__.py", "credits.json")
EXTRACT_COMMAND = r".venv\Scripts\python video/extract_data.py"


def mmss(seconds: float) -> str:
    whole = int(round(seconds))
    return f"{whole // 60}:{whole % 60:02d}"


def check_length(total_seconds: float) -> None:
    """Raise SystemExit, loudly, when the video is outside 3:45-4:30 (225-270 s)."""
    if MIN_SECONDS <= total_seconds <= MAX_SECONDS:
        return
    if total_seconds > MAX_SECONDS:
        fix = f"cut {total_seconds - MAX_SECONDS:.1f} s"
    else:
        fix = f"add {MIN_SECONDS - total_seconds:.1f} s"
    raise SystemExit(
        "\n" + "!" * 72 + "\n"
        f"VIDEO LENGTH {mmss(total_seconds)} ({total_seconds:.1f} s) is outside the allowed "
        f"{mmss(MIN_SECONDS)}-{mmss(MAX_SECONDS)} ({MIN_SECONDS:.0f}-{MAX_SECONDS:.0f} s).\n"
        f"Edit video/narration.md to {fix} of narration (about 2.5 words per second), "
        "then run the build again.\nThe video file was still written so you can watch it.\n"
        + "!" * 72
    )


def parse_scenes(spec: str) -> list[str]:
    """'4,5' -> ['04_leakage', '05_pipeline']; scene numbers 1-10 or full scene ids."""
    out: list[str] = []
    for item in (part.strip() for part in spec.split(",")):
        if item in SCENE_IDS:
            scene_id = item
        elif item.isdigit() and 1 <= int(item) <= len(SCENE_IDS):
            scene_id = SCENE_IDS[int(item) - 1]
        else:
            raise ValueError(f"unknown scene {item!r}: use 1-{len(SCENE_IDS)} or a scene id "
                             f"({', '.join(SCENE_IDS)})")
        if scene_id in out:
            raise ValueError(f"scene {scene_id} given twice")
        out.append(scene_id)
    return out


# --- clip cache key -----------------------------------------------------------------------

_SCENE_IMPORT = re.compile(
    r"^\s*(?:from\s+scenes\.(s\d\d_\w+)\s+import|import\s+scenes\.(s\d\d_\w+)"
    r"|from\s+scenes\s+import\s+([^\n#]+))", re.M)


def scene_dependencies(scene_id: str, scenes_dir: Path = VIDEO_DIR / "scenes") -> list[Path]:
    """The scene's source file plus every scene module it imports, transitively, sorted."""
    found: set[Path] = set()
    todo = [f"s{scene_id}"]
    while todo:
        path = Path(scenes_dir) / f"{todo.pop()}.py"
        if path in found or not path.exists():
            continue
        found.add(path)
        for m in _SCENE_IMPORT.finditer(path.read_text(encoding="utf-8")):
            names = [m.group(1) or m.group(2)] if (m.group(1) or m.group(2)) else [
                n.strip().split(" ")[0] for n in m.group(3).strip("() ").split(",")]
            todo += [n for n in names if re.fullmatch(r"s\d\d_\w+", n)]
    return sorted(found)


def _file_digest(path: Path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest()


def clip_key(scene_id: str, audio_files: Sequence[Path], resolution: tuple[int, int, int], *,
             video_dir: Path = VIDEO_DIR, data_dir: Path = DATA_DIR) -> str:
    """Hash of everything a rendered clip depends on (see the module docstring)."""
    video_dir, data_dir = Path(video_dir), Path(data_dir)
    parts: dict = {"scene": scene_id, "resolution": list(resolution)}
    parts["audio"] = [_file_digest(p) for p in audio_files]
    sources = [video_dir / rel for rel in SHARED_SOURCES]
    sources += scene_dependencies(scene_id, video_dir / "scenes")
    parts["sources"] = {p.relative_to(video_dir).as_posix(): _file_digest(p) for p in sources}
    parts["data"] = {p.relative_to(data_dir).as_posix(): _file_digest(p)
                     for p in sorted(data_dir.rglob("*")) if p.is_file()}
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def clip_path(scene_id: str, draft: bool) -> Path:
    return CLIPS_DIR / f"{scene_id}{'_draft' if draft else ''}.mp4"


def key_path(clip: Path) -> Path:
    return Path(clip).with_suffix(".key")


# --- narration audio ------------------------------------------------------------------------

def prepare_audio(voice: str) -> dict[str, tts.Audio]:
    """Every scene's narration audio (cached by voice + text) and the audio manifest."""
    numbers = json.loads((DATA_DIR / "numbers.json").read_text(encoding="utf-8"))
    texts = load_narration(VIDEO_DIR / "narration.md", numbers)
    missing = [s for s in SCENE_IDS if s not in texts]
    if missing:
        raise SystemExit(f"video/narration.md has no '## <id>' section for: {', '.join(missing)}")
    audios: dict[str, tts.Audio] = {}
    for scene_id in SCENE_IDS:
        cached = all(p.exists() for p in tts.cache_paths(scene_id, texts[scene_id], voice,
                                                          AUDIO_DIR))
        if not cached:
            print(f"  narration {scene_id}: text or voice changed, synthesizing ...", flush=True)
        try:
            audios[scene_id] = tts.synthesize(scene_id, texts[scene_id], voice, AUDIO_DIR)
        except Exception as e:  # edge-tts raises several network error types
            raise SystemExit(
                f"Could not make the narration audio for scene {scene_id} with voice {voice}: "
                f"{type(e).__name__}: {e}\nThis scene's text or voice changed, so it needs the "
                "internet (edge-tts). Connect and run the build again; unchanged scenes are "
                "cached and work offline."
            ) from e
    tts.write_manifest(audios, texts, voice)
    return audios


# --- parallel rendering ---------------------------------------------------------------------

class RenderJob(NamedTuple):
    scene_id: str
    draft: bool
    out: Path
    key: str


class SceneRenderError(RuntimeError):
    """A scene failed to render; the message holds the scene id and the worker traceback."""


def _render_worker(job: RenderJob) -> tuple[str, float, int, float]:
    """Render one clip (runs in a worker process). Returns (id, seconds, frames, elapsed)."""
    try:
        key_path(job.out).unlink(missing_ok=True)
        start = time.perf_counter()
        ctx = engine.make_context(job.scene_id, draft=job.draft)
        engine.render_scene(job.scene_id, ctx, ctx.audio_wav, job.out)
        key_path(job.out).write_text(job.key, encoding="utf-8")  # only after a good render
        frames = len(engine.frame_times(ctx))
        return job.scene_id, frames / ctx.fps, frames, time.perf_counter() - start
    except BaseException:
        # re-raised as a plain picklable error that carries the scene id and full traceback
        raise SceneRenderError(f"scene {job.scene_id} failed to render:\n"
                               f"{traceback.format_exc()}") from None


def run_jobs(jobs: list[RenderJob], n_workers: int) -> dict[str, tuple[float, int, float]]:
    """Render the jobs in spawned worker processes; the first failure stops all of them."""
    results: dict[str, tuple[float, int, float]] = {}
    if not jobs:
        return results
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=max(1, min(n_workers, len(jobs)))) as pool:  # exit -> terminate
        for scene_id, seconds, frames, elapsed in pool.imap_unordered(_render_worker, jobs):
            results[scene_id] = (seconds, frames, elapsed)
            print(f"  rendered {scene_id:16s} {seconds:6.1f} s  {frames} frames  "
                  f"{elapsed / frames:.3f} s/frame", flush=True)
    return results


def default_jobs() -> int:
    return max(1, (os.cpu_count() or 2) // 2)


# --- the build ------------------------------------------------------------------------------

def build(scene_ids: list[str], *, draft: bool, voice: str, n_workers: int) -> list[Path]:
    """Audio, then the clips for scene_ids (reusing unchanged ones); returns clip paths."""
    print(f"Narration audio (voice {voice})", flush=True)
    audios = prepare_audio(voice)
    resolution = engine.DRAFT if draft else engine.FULL
    jobs, clips = [], []
    for scene_id in scene_ids:
        wav = audios[scene_id].wav_path
        key = clip_key(scene_id, [wav, wav.with_suffix(".json")], resolution)
        out = clip_path(scene_id, draft)
        clips.append(out)
        kp = key_path(out)
        if out.exists() and kp.exists() and kp.read_text(encoding="utf-8") == key:
            print(f"  reused   {scene_id} (unchanged)", flush=True)
        else:
            jobs.append(RenderJob(scene_id, draft, out, key))
    w, h, fps = resolution
    workers = f", parallel workers: {min(n_workers, len(jobs))}" if jobs else ""
    print(f"Rendering {len(jobs)} of {len(scene_ids)} scene clips at {w}x{h}, {fps} fps"
          f"{workers}", flush=True)
    run_jobs(jobs, n_workers)
    return clips


def summary(draft: bool, out: Path, total: float) -> str:
    lines = ["", "Scene            seconds"]
    for scene_id in SCENE_IDS:
        ctx = engine.make_context(scene_id, draft=draft)
        lines.append(f"{scene_id:16s} {len(engine.frame_times(ctx)) / ctx.fps:7.1f}")
    lines += [f"Total length     {mmss(total)} ({total:.1f} s; allowed "
              f"{mmss(MIN_SECONDS)}-{mmss(MAX_SECONDS)})",
              f"Output           {out}",
              f"Size             {out.stat().st_size / 1e6:.1f} MB"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    credits = json.loads(engine.CREDITS.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description="Build the explainer video.")
    parser.add_argument("--draft", action="store_true", help="960x540 at 15 fps, quick check")
    parser.add_argument("--scenes", help="only render these scene clips, e.g. 4,5 (no join)")
    parser.add_argument("--voice", default=credits.get("voice", tts.SAMPLE_VOICES["neerja"]),
                        help="edge-tts voice (default: credits.json 'voice')")
    parser.add_argument("--jobs", type=int, default=default_jobs(),
                        help=f"parallel render processes (default {default_jobs()})")
    args = parser.parse_args(argv)
    try:
        selected = parse_scenes(args.scenes) if args.scenes else list(SCENE_IDS)
    except ValueError as e:
        parser.error(str(e))
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")

    if not (DATA_DIR / "numbers.json").exists():
        raise SystemExit(f"{DATA_DIR} is missing. Create it first (project env, repo root):\n"
                         f"    {EXTRACT_COMMAND}")
    start = time.perf_counter()
    try:
        clips = build(selected, draft=args.draft, voice=args.voice, n_workers=args.jobs)
    except SceneRenderError as e:
        raise SystemExit(f"BUILD FAILED: {e}") from None
    if args.scenes:
        print("\nScene clips:")
        for clip in clips:
            print(f"  {clip}")
        return

    out = DRAFT_OUTPUT if args.draft else OUTPUT
    print(f"Joining {len(clips)} clips -> {out}", flush=True)
    total = engine.concat(clips, out)
    print(summary(args.draft, out, total))
    print(f"Build time       {mmss(time.perf_counter() - start)}")
    check_length(total)


if __name__ == "__main__":
    main()
