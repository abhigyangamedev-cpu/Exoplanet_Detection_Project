"""Tests for the one-command build (video/make_video.py)."""

import subprocess
import sys

import make_video
import pytest
import scenes
import tts
from make_video import check_length, clip_key, parse_scenes, scene_dependencies
from narration import SCENE_IDS


def test_check_length_bounds():
    check_length(250.0)
    check_length(225.0)
    check_length(270.0)
    for bad in (224.0, 271.0):
        with pytest.raises(SystemExit, match=str(int(bad))):
            check_length(bad)


def test_check_length_says_the_range_and_how_much_to_change():
    with pytest.raises(SystemExit) as long:
        check_length(276.4)
    message = str(long.value)
    assert "4:36" in message and "3:45-4:30" in message
    assert "cut 6.4 s" in message
    with pytest.raises(SystemExit) as short:
        check_length(220.0)
    assert "add 5.0 s" in str(short.value)


def test_scene_selection_parses():
    assert parse_scenes("4,5") == ["04_leakage", "05_pipeline"]
    assert parse_scenes(" 10 , 1") == ["10_end", "01_hook"]
    assert parse_scenes("05_pipeline") == ["05_pipeline"]
    for bad in ("0", "11", "x", "4,4", ""):
        with pytest.raises(ValueError):
            parse_scenes(bad)


@pytest.mark.parametrize("scene_id", SCENE_IDS)
def test_every_scene_id_has_an_importable_scene_module(scene_id):
    module = scenes.import_scene(scene_id)
    assert callable(module.build)


def test_scene_10_depends_on_scene_1_source():
    deps = {p.name for p in scene_dependencies("10_end")}
    assert deps == {"s10_end.py", "s01_hook.py"}
    assert {p.name for p in scene_dependencies("07_integration")} == {
        "s07_integration.py", "s06_models.py"}
    assert {p.name for p in scene_dependencies("05_pipeline")} == {"s05_pipeline.py"}


def make_tree(root):
    """A minimal copy of the files a clip depends on: sources, credits, data, audio."""
    video = root / "video"
    (video / "scenes").mkdir(parents=True)
    for name in ("style.py", "engine.py", "tts.py", "credits.json"):
        (video / name).write_text(f"# {name}\n", encoding="utf-8")
    (video / "scenes" / "__init__.py").write_text("# init\n", encoding="utf-8")
    (video / "scenes" / "s01_hook.py").write_text("X = 1\n", encoding="utf-8")
    (video / "scenes" / "s05_pipeline.py").write_text("Y = 1\n", encoding="utf-8")
    (video / "scenes" / "s10_end.py").write_text(
        "from scenes.s01_hook import X\n", encoding="utf-8")
    data = video / "build" / "data"
    (data / "terminal").mkdir(parents=True)
    (data / "numbers.json").write_text("{}", encoding="utf-8")
    (data / "terminal" / "pytest.txt").write_text("ok\n", encoding="utf-8")
    audio = video / "build" / "audio"
    audio.mkdir()
    (audio / "10_end.wav").write_bytes(b"RIFF....")
    (audio / "10_end.json").write_text('{"words": []}', encoding="utf-8")
    return video


def key_of(video, scene_id="10_end", resolution=(960, 540, 15)):
    audio = [video / "build" / "audio" / "10_end.wav", video / "build" / "audio" / "10_end.json"]
    return clip_key(scene_id, audio, resolution, video_dir=video,
                    data_dir=video / "build" / "data")


def test_clip_key_is_stable_for_an_unchanged_tree(tmp_path):
    video = make_tree(tmp_path)
    assert key_of(video) == key_of(video)


@pytest.mark.parametrize("change", [
    "scenes/s10_end.py", "scenes/s01_hook.py", "scenes/__init__.py", "style.py", "engine.py",
    "tts.py", "credits.json", "build/data/numbers.json", "build/data/terminal/pytest.txt",
    "build/data/new_file.json", "build/audio/10_end.wav", "build/audio/10_end.json",
])
def test_clip_key_changes_with_each_dependency(tmp_path, change):
    video = make_tree(tmp_path)
    before = key_of(video)
    path = video / change
    path.write_bytes((path.read_bytes() if path.exists() else b"") + b"\n# edited\n")
    assert key_of(video) != before


def test_clip_key_ignores_unrelated_scenes_and_changes_with_resolution(tmp_path):
    video = make_tree(tmp_path)
    before = key_of(video)
    (video / "scenes" / "s05_pipeline.py").write_text("Y = 2\n", encoding="utf-8")
    assert key_of(video) == before
    assert key_of(video, resolution=(1920, 1080, 30)) != before


def test_build_reuses_a_clip_only_while_its_key_matches(tmp_path, monkeypatch):
    wav = tmp_path / "05_pipeline_x.wav"
    wav.write_bytes(b"audio one")
    wav.with_suffix(".json").write_text("{}", encoding="utf-8")
    audios = {s: tts.Audio(wav, 1.0, []) for s in SCENE_IDS}
    monkeypatch.setattr(make_video, "prepare_audio", lambda voice: audios)
    monkeypatch.setattr(make_video, "CLIPS_DIR", tmp_path / "clips")
    queued = []
    monkeypatch.setattr(make_video, "run_jobs", lambda jobs, n: queued.append(jobs))

    def build():
        make_video.build(["05_pipeline"], draft=True, voice="v", n_workers=2)
        return queued[-1]

    (job,) = build()
    assert job.scene_id == "05_pipeline" and job.out == tmp_path / "clips" / "05_pipeline_draft.mp4"
    job.out.parent.mkdir()
    job.out.write_bytes(b"clip")
    make_video.key_path(job.out).write_text(job.key, encoding="utf-8")
    assert build() == []                       # unchanged: reused
    wav.write_bytes(b"audio two")              # narration changed: rendered again
    (again,) = build()
    assert again.key != job.key


def test_missing_audio_without_network_names_the_scene(tmp_path, monkeypatch):
    monkeypatch.setattr(make_video, "AUDIO_DIR", tmp_path)

    def offline(text, voice):
        raise OSError("network is unreachable")

    monkeypatch.setattr(tts, "_fetch", offline)
    with pytest.raises(SystemExit, match="scene 01_hook") as err:
        make_video.prepare_audio("en-IN-NeerjaNeural")
    assert "internet" in str(err.value)


def test_a_failing_scene_fails_the_build_with_its_id_and_traceback(tmp_path):
    jobs = [make_video.RenderJob("99_missing", True, tmp_path / "99.mp4", "key")]
    with pytest.raises(make_video.SceneRenderError, match="99_missing") as err:
        make_video.run_jobs(jobs, n_workers=1)
    assert "Traceback" in str(err.value)
    assert not (tmp_path / "99.key").exists()


def test_make_video_does_not_import_the_project_package():
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import make_video; "
            "print(sorted(m for m in sys.modules if m.split('.')[0] in "
            "('exoplanet', 'extract_data')))")
    out = subprocess.run([sys.executable, "-c", code, str(make_video.VIDEO_DIR)],
                         capture_output=True, text=True, check=True).stdout
    assert out.strip() == "[]"
