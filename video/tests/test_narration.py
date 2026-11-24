import json
import subprocess
import wave
from pathlib import Path

import imageio_ffmpeg
import pytest
import tts
from narration import SCENE_IDS, format_value, load_narration
from tts import caption_cues, synthesize

VIDEO_DIR = Path(__file__).resolve().parents[1]
NARRATION = VIDEO_DIR / "narration.md"
NUMBERS = json.loads((VIDEO_DIR / "build" / "data" / "numbers.json").read_text(encoding="utf-8"))


def test_placeholders_filled_and_unknown_rejected(tmp_path):
    assert "6,677" in load_narration(NARRATION, NUMBERS)["02_problem"]
    p = tmp_path / "n.md"
    p.write_text("## 02_problem\nWe used {n_kois} signals.\n", encoding="utf-8")
    assert load_narration(p, {"n_kois": 6677})["02_problem"] == "We used 6,677 signals."
    p.write_text("## 01_hook\nScore {no_such_key}.\n", encoding="utf-8")
    with pytest.raises(KeyError, match="no_such_key"):
        load_narration(p, NUMBERS)
    p.write_text("## 01_hook\nScore {test_pr_auc.no_such_model}.\n", encoding="utf-8")
    with pytest.raises(KeyError, match="test_pr_auc.no_such_model"):
        load_narration(p, NUMBERS)


def test_spoken_format_convention(tmp_path):
    assert format_value(6677) == "6,677"
    assert format_value(0.9932661257179395) == "0.993"
    assert format_value([0.9026, 0.8695, 0.9339]) == "0.903"  # interval -> point estimate
    assert format_value("3 min 23 s") == "3 minutes 23 seconds"
    p = tmp_path / "n.md"
    p.write_text(
        "<!-- comment -->\n## 08_results\nA {test_pr_auc.tabular}, low {test_pr_auc.cnn.1}.\n",
        encoding="utf-8",
    )
    assert load_narration(p, NUMBERS)["08_results"] == "A 0.993, low 0.870."


def test_ten_scenes_in_order_and_word_budget():
    n = load_narration(NARRATION, NUMBERS)
    assert list(n) == SCENE_IDS
    assert all(t.strip() for t in n.values())
    assert 500 <= sum(len(t.split()) for t in n.values()) <= 640


def test_wording_rules():
    n = load_narration(NARRATION, NUMBERS)
    text = " ".join(n.values()).lower()
    assert "planet detected" not in text and "discovered a planet" not in text
    assert "{" not in text and "}" not in text
    # 0.999 is only ever the leakage warning
    assert [s for s, t in n.items() if "0.999" in t] == ["04_leakage"]


def test_caption_cues_fallback_without_word_timings():
    cues = caption_cues("one two three four " * 20, words=[], duration=20.0)
    assert cues[0][0] == 0.0 and abs(cues[-1][1] - 20.0) < 0.05
    assert all(len(c[2]) <= 84 for c in cues)
    assert all(a[1] <= b[0] + 1e-6 for a, b in zip(cues[:-1], cues[1:], strict=True))
    assert " ".join(c[2] for c in cues).split() == ("one two three four " * 20).split()


def test_caption_cues_follow_word_timings():
    text = "A planet crosses its star. The star dims, just a little."
    words = [
        (0.1, 0.2, "A"), (0.2, 0.6, "planet"), (0.6, 1.0, "crosses"), (1.0, 1.2, "its"),
        (1.2, 1.6, "star"), (2.0, 2.2, "The"), (2.2, 2.6, "star"), (2.6, 3.0, "dims"),
        (3.2, 3.5, "just"), (3.5, 3.6, "a"), (3.6, 4.0, "little"),
    ]
    cues = caption_cues(text, words, duration=4.5, max_chars=30)
    assert [c[2] for c in cues] == ["A planet crosses its star.", "The star dims, just a little."]
    assert cues[0][0] == pytest.approx(0.1)
    assert cues[1][0] == pytest.approx(2.0) and cues[0][1] == pytest.approx(2.0)
    assert cues[-1][1] == pytest.approx(4.5)


def _write_wav(path: Path, seconds: float, rate: int = 48000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))


def test_synthesize_uses_cache_without_network(monkeypatch, tmp_path):
    wav, meta = tts.cache_paths("01_hook", "Hello stars.", "en-IN-NeerjaNeural", tmp_path)
    _write_wav(wav, 1.5)
    meta.write_text(json.dumps({"words": [[0.1, 0.5, "Hello"], [0.5, 1.0, "stars"]]}))

    def no_network(*args, **kwargs):
        raise AssertionError("network used despite cache")

    monkeypatch.setattr(tts.edge_tts, "Communicate", no_network)
    audio = synthesize("01_hook", "Hello stars.", "en-IN-NeerjaNeural", tmp_path)
    assert audio.wav_path == wav
    assert audio.duration == pytest.approx(1.5)
    assert audio.words == [(0.1, 0.5, "Hello"), (0.5, 1.0, "stars")]
    # a different voice or text is a different cache entry
    assert tts.cache_paths("01_hook", "Hello stars.", "en-IN-PrabhatNeural", tmp_path)[0] != wav
    assert tts.cache_paths("01_hook", "Hello star.", "en-IN-NeerjaNeural", tmp_path)[0] != wav


def test_synthesize_converts_stream_to_48k_mono_wav(monkeypatch, tmp_path):
    mp3 = tmp_path / "tone.mp3"
    subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "lavfi",
         "-i", "sine=frequency=440:duration=1", "-ac", "1", "-ar", "24000", str(mp3)],
        check=True,
    )
    data = mp3.read_bytes()

    class FakeCommunicate:
        def __init__(self, text, voice, **kwargs):
            assert kwargs.get("boundary") == "WordBoundary"

        async def stream(self):
            yield {"type": "WordBoundary", "offset": 1_000_000, "duration": 4_000_000, "text": "Hi"}
            yield {"type": "audio", "data": data}

    monkeypatch.setattr(tts.edge_tts, "Communicate", FakeCommunicate)
    out = tmp_path / "out"
    audio = synthesize("10_end", "Hi", "en-IN-NeerjaNeural", out)
    with wave.open(str(audio.wav_path)) as w:
        assert w.getframerate() == 48000 and w.getnchannels() == 1
    assert audio.duration == pytest.approx(1.0, abs=0.1)
    assert audio.words == [(pytest.approx(0.1), pytest.approx(0.5), "Hi")]
    assert not list(out.glob("*.mp3"))
