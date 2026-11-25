"""Narration audio with edge-tts, cached per scene, plus caption timing.

``synthesize`` turns one scene's text into a 48 kHz mono WAV and its word timings. Results
are cached on sha1(voice + text): ``<scene_id>_<hash>.wav`` plus a ``.json`` with the word
timings, so re-rendering visuals never calls the online service again.

Run as a script to build every scene's audio and video/build/audio/manifest.json:

    C:\\video_env\\Scripts\\python video/tts.py [--voice en-IN-NeerjaNeural] [--samples]

``--samples`` also writes scene 01_hook in both candidate voices to
video/build/voice_sample_neerja.wav and voice_sample_prabhat.wav.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import shutil
import subprocess
import time
import wave
from pathlib import Path
from typing import NamedTuple

import edge_tts
import imageio_ffmpeg
from narration import SCENE_IDS, load_narration

VIDEO_DIR = Path(__file__).resolve().parent
REPO_ROOT = VIDEO_DIR.parent
BUILD_DIR = VIDEO_DIR / "build"
AUDIO_DIR = BUILD_DIR / "audio"
SAMPLE_RATE = 48000
SCENE_PAD = 0.6  # seconds of silence each scene adds after its narration
SAMPLE_VOICES = {"neerja": "en-IN-NeerjaNeural", "prabhat": "en-IN-PrabhatNeural"}
RETRIES = 4

Word = tuple[float, float, str]
MIN_CUE_CHARS = 15            # a sentence fragment shorter than this is not left on its own
SENTENCE_END = (".", "?", "!")
CLAUSE_END = (",", ";", ":", ".", "?", "!")


class Audio(NamedTuple):
    wav_path: Path
    duration: float
    words: list[Word]


def cache_paths(scene_id: str, text: str, voice: str, out_dir: Path) -> tuple[Path, Path]:
    key = hashlib.sha1((voice + text).encode("utf-8")).hexdigest()[:16]
    stem = Path(out_dir) / f"{scene_id}_{key}"
    return stem.with_suffix(".wav"), stem.with_suffix(".json")


def wav_duration(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


async def _stream(text: str, voice: str) -> tuple[bytes, list[Word]]:
    audio = bytearray()
    words: list[Word] = []
    communicate = edge_tts.Communicate(text, voice, boundary="WordBoundary")
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
        elif chunk["type"] == "WordBoundary":
            start = chunk["offset"] / 1e7  # edge-tts reports 100 ns ticks
            words.append((start, start + chunk["duration"] / 1e7, chunk["text"]))
    return bytes(audio), words


def _fetch(text: str, voice: str) -> tuple[bytes, list[Word]]:
    for attempt in range(1, RETRIES + 1):
        try:
            return asyncio.run(_stream(text, voice))
        except Exception:
            if attempt == RETRIES:
                raise
            time.sleep(2 * attempt)
    raise AssertionError("unreachable")


def synthesize(scene_id: str, text: str, voice: str, out_dir: Path) -> Audio:
    """Return the scene's narration audio, from cache when voice and text are unchanged."""
    out_dir = Path(out_dir)
    wav, meta = cache_paths(scene_id, text, voice, out_dir)
    if not (wav.exists() and meta.exists()):
        out_dir.mkdir(parents=True, exist_ok=True)
        mp3_bytes, words = _fetch(text, voice)
        mp3 = wav.with_suffix(".mp3")
        mp3.write_bytes(mp3_bytes)
        try:
            subprocess.run(
                [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(mp3),
                 "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(wav)],
                check=True,
            )
        finally:
            mp3.unlink(missing_ok=True)
        # the metadata file is written last: its presence marks a complete cache entry
        meta.write_text(json.dumps({"voice": voice, "text": text, "words": words}, indent=1),
                        encoding="utf-8")
    words = [(float(a), float(b), str(w)) for a, b, w in json.loads(meta.read_text())["words"]]
    return Audio(wav, wav_duration(wav), words)


_NORM = re.compile(r"[^a-z0-9]")


def _norm(token: str) -> str:
    return _NORM.sub("", token.lower())


def _token_starts(tokens: list[str], words: list[Word], duration: float) -> list[float]:
    """Start time of each display token: matched to word timings, else interpolated."""
    n = len(tokens)
    if not words:
        lengths = [len(t) + 1 for t in tokens]
        total = sum(lengths)
        starts, acc = [], 0
        for length in lengths:
            starts.append(duration * acc / total)
            acc += length
        return starts
    times: list[float | None] = [None] * n
    j = 0
    for i, tok in enumerate(tokens):
        t = _norm(tok)
        if not t:
            continue
        for k in range(j, min(j + 8, len(words))):
            w = _norm(words[k][2])
            if w and (w == t or t.startswith(w)):
                times[i] = words[k][0]
                j = k + 1
                break
    # interpolate unmatched tokens between known anchors (index -1 -> first word, n -> end)
    anchors = [(-1, words[0][0])] + [(i, t) for i, t in enumerate(times) if t is not None]
    anchors.append((n, words[-1][1]))
    out = []
    a = 0
    for i in range(n):
        while anchors[a + 1][0] < i:
            a += 1
        (i0, t0), (i1, t1) = anchors[a], anchors[a + 1]
        out.append(times[i] if times[i] is not None else t0 + (t1 - t0) * (i - i0) / (i1 - i0))
    if times[0] is None:
        out[0] = words[0][0]
    for i in range(1, n):  # never step backwards in time
        out[i] = max(out[i], out[i - 1])
    return out


def caption_cues(text: str, words: list[Word], duration: float,
                 max_chars: int = 84) -> list[tuple[float, float, str]]:
    """Split text into on-screen cues of at most max_chars, timed by the word timings.

    A cue ends at a sentence end once it holds min(40, max_chars // 2) characters, or before a word
    that would overflow it. No fragment stands apart from its sentence: when the overflow
    break would leave fewer than MIN_CUE_CHARS of a sentence at the start of the next cue
    (or as the last cue), the break moves back to the previous comma or sentence end.
    Each cue lasts until the next begins; the last ends at
    ``duration``. Without word timings, time is shared out by character count.
    """
    tokens = text.split()
    if not tokens:
        return []
    starts = _token_starts(tokens, words, duration)

    def width(idx: list[int]) -> int:
        return len(" ".join(tokens[k] for k in idx))

    def ends_sentence(k: int) -> bool:
        return tokens[k].endswith(SENTENCE_END) or k == len(tokens) - 1

    def back_up(group: list[int]) -> int | None:
        """Position in group after its last comma or sentence end (not the group's own end)
        that leaves at least 20 characters before it; None if there is none."""
        for pos in range(len(group) - 1, 0, -1):
            if tokens[group[pos - 1]].endswith(CLAUSE_END) and width(group[:pos]) >= 20:
                return pos
        return None

    groups: list[list[int]] = [[]]
    for i, tok in enumerate(tokens):
        if groups[-1] and width(groups[-1]) + 1 + len(tok) > max_chars:
            # the words up to this sentence's end would sit as a fragment at the start of
            # the next cue: break earlier, at the last comma or sentence end, if there is one
            head = [i]
            while not ends_sentence(head[-1]):
                head.append(head[-1] + 1)
            pos = back_up(groups[-1]) if width(head) < MIN_CUE_CHARS else None
            if pos is not None and width(groups[-1][pos:] + head) > max_chars:
                pos = None
            moved = groups[-1][pos:] if pos is not None else []
            if pos is not None:
                del groups[-1][pos:]
            groups.append(moved)
        groups[-1].append(i)
        sentence_end = tok.endswith(SENTENCE_END) and i < len(tokens) - 1
        if sentence_end and width(groups[-1]) >= min(40, max_chars // 2):
            groups.append([])
    groups = [g for g in groups if g]
    # a short last cue: take the previous cue's words after its last comma or sentence end
    if len(groups) > 1 and width(groups[-1]) < MIN_CUE_CHARS:
        pos = back_up(groups[-2])
        if pos is not None and width(groups[-2][pos:] + groups[-1]) <= max_chars:
            groups[-1] = groups[-2][pos:] + groups[-1]
            del groups[-2][pos:]
    cue_starts = [starts[g[0]] for g in groups]
    cue_ends = cue_starts[1:] + [duration]
    return [
        (s, e, " ".join(tokens[i] for i in g))
        for s, e, g in zip(cue_starts, cue_ends, groups, strict=True)
    ]


def write_manifest(audios: dict[str, Audio], texts: dict[str, str], voice: str,
                   path: Path = AUDIO_DIR / "manifest.json") -> dict:
    """Write the audio manifest (per-scene wav, duration, n_words plus totals) for SCENE_IDS."""
    manifest: dict = {}
    for scene_id in SCENE_IDS:
        manifest[scene_id] = {
            "wav": audios[scene_id].wav_path.resolve().relative_to(REPO_ROOT).as_posix(),
            "duration": round(audios[scene_id].duration, 3),
            "n_words": len(texts[scene_id].split()),
        }
    total = sum(manifest[s]["duration"] for s in SCENE_IDS)
    manifest["voice"] = voice
    manifest["total_duration"] = round(total, 3)
    manifest["scene_pad"] = SCENE_PAD
    manifest["total_with_padding"] = round(total + SCENE_PAD * len(SCENE_IDS), 3)
    Path(path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_all(voice: str, samples: bool) -> dict:
    numbers = json.loads((BUILD_DIR / "data" / "numbers.json").read_text(encoding="utf-8"))
    scenes = load_narration(VIDEO_DIR / "narration.md", numbers)
    audios: dict[str, Audio] = {}
    for scene_id in SCENE_IDS:
        text = scenes[scene_id]
        audios[scene_id] = synthesize(scene_id, text, voice, AUDIO_DIR)
        print(f"{scene_id:16s} {audios[scene_id].duration:6.2f} s  {len(text.split()):4d} words")
    manifest = write_manifest(audios, scenes, voice)
    print(f"total {manifest['total_duration']:.2f} s; "
          f"with padding {manifest['total_with_padding']:.2f} s")
    if samples:
        for name, sample_voice in SAMPLE_VOICES.items():
            audio = synthesize("01_hook", scenes["01_hook"], sample_voice, AUDIO_DIR)
            shutil.copyfile(audio.wav_path, BUILD_DIR / f"voice_sample_{name}.wav")
            print(f"voice_sample_{name}.wav {audio.duration:.2f} s")
    return manifest


def main() -> None:
    credits = json.loads((VIDEO_DIR / "credits.json").read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--voice", default=credits.get("voice", SAMPLE_VOICES["neerja"]))
    parser.add_argument("--samples", action="store_true", help="also write the voice samples")
    args = parser.parse_args()
    build_all(args.voice, args.samples)


if __name__ == "__main__":
    main()
