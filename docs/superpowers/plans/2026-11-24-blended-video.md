# Blended Project Video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `video/exoplanet_project_video.mp4`, a 3:45-4:30 narrated, captioned, loopable 1080p video of the project's development journey, rendered from the project's real files.

**Architecture:** Two environments. The project `.venv` runs one extraction script that reads the catalogue, cache, views and result files and writes small `.npz`/`.json` files into `video/build/data/`. A separate environment at `C:\video_env` (numpy, matplotlib, imageio-ffmpeg, edge-tts, pytest only) turns `video/narration.md` into per-scene audio, renders each scene's frames with matplotlib to exactly its audio length, and joins everything with ffmpeg.

**Tech Stack:** Python 3.12, matplotlib (Agg), imageio-ffmpeg, edge-tts, pytest.

**Spec:** `docs/superpowers/specs/2026-11-24-blended-video-design.md`

## Global Constraints

- Output: one MP4, 3:45-4:30, 1920x1080, 30 fps, H.264 (`yuv420p`) + AAC. `--draft` renders 960x540 at 15 fps.
- Every number and curve on screen is read from project files via `video/build/data/`. No literal result numbers in scene code.
- The project `.venv` gains no packages. Video tooling lives in `C:\video_env`.
- No tracked file is overwritten. Commands that write reports are redirected into `video/build/`.
- The held-out test set is not re-scored and no model is retrained.
- Local commits only, on branch `feature/blended-video`; never push. `video/build/` and `video/*.mp4` are added to `.gitignore`.
- Voice: edge-tts, `en-IN-NeerjaNeural` or `en-IN-PrabhatNeural` (user picks). Audio is cached by a hash of voice + text.
- No music. Captions are drawn into the frames.
- Wording rules: a score means "worth follow-up", never "planet detected"; 0.999 appears only as the leakage warning.
- Title and end cards read names from `video/credits.json`; placeholders until the user supplies them.
- Colours: confirmed planet and false positive each have one colour, defined once in `style.py`.

## Review Focus

- Narration edited so one scene's audio becomes much longer or shorter: visuals must stretch to the new length, not cut off or freeze. (Task 3 test.)
- edge-tts unreachable or returning no word timings: render must still work from cached audio, and captions must fall back to proportional timing. (Task 2 tests.)
- A chosen example star missing from the cache or with a failed view: extraction must fail with the star's ID, not write an empty array. (Task 1 test.)
- A caption line too long for the frame at draft resolution: must wrap to at most two lines, never overflow. (Task 3 test.)
- Total length outside 3:45-4:30 after a script edit: the build must report it loudly. (Task 9 test.)

## User review gates

- After Task 2: narration script and two voice samples. Stop for the user.
- After Task 5: draft render of scenes 4 and 5. Stop for the user.
- After Task 9: full draft. Stop for the user, then final render.

---

### Task 1: Data extraction (project `.venv`)

**Files:**
- Create: `video/extract_data.py`, `video/credits.json`
- Modify: `.gitignore` (add `/video/build/`, `/video/*.mp4`)
- Test: `tests/test_video_extract.py`

**Interfaces:**
- Produces, in `video/build/data/`:
  - `numbers.json`: `{"n_kois", "n_confirmed", "n_false_positive", "n_test", "n_dev", "n_audit_columns", "n_audit_removed", "leak_lc_audited", "leak_lc_plus_vetting", "test_pr_auc": {model: [value, lo, hi]}, "dev_full_pr_auc": {model: [value, lo, hi]}, "n_views_built", "cnn_full_train_time_text", "next_steps": [str, str, str], "sources": {key: file}}`
  - `star_planet.npz`, `star_binary.npz`: `time, raw_flux, quarter_index, stitched_flux, trend, detrended_flux, phase, global_view, local_view`, plus scalars `kepid, koi_name, period, duration_hours`
  - `pr_curves.npz`: per model `precision_<m>, recall_<m>, band_lo_<m>, band_hi_<m>` for the test models and for dev `tabular_lc`, `cnn_full`
  - `audit_columns.json`: `[{"column", "group", "kept": bool}]` (153 rows)
  - `error_breakdown.json`: CNN recall by SNR bin and by period bin, from `outputs/evaluation_test.json`
  - `splits_sample.json`: 60 stars with their partition, for the splits animation
  - `terminal/<name>.txt`: captured stdout of `pytest -q`, `scripts/audit_leakage.py --strict`, `scripts/inspect_catalogue.py`, `scripts/make_report.py --output video/build/report.md`; plus copies of `outputs/train_cnn_full.log` and the tail of `outputs/build_views_full_dev_resume4.log`
- Functions: `extract_numbers(config) -> dict`, `extract_star(koi_name: str, config) -> dict[str, np.ndarray]`, `pick_examples(config) -> tuple[str, str]`, `main()`

- [ ] **Step 1: Write failing tests**

```python
def test_numbers_match_project_files(tmp_path):
    n = extract_numbers(load_config())
    assert n["n_confirmed"] + n["n_false_positive"] == n["n_kois"]
    assert n["n_audit_columns"] == 153
    assert 0.9 < n["leak_lc_audited"] < n["leak_lc_plus_vetting"] <= 1.0
    assert set(n["test_pr_auc"]) == {"tabular", "tabular_lc", "tabular_lc_sub", "cnn"}
    assert all(k in n["sources"] for k in ("leak_lc_audited", "test_pr_auc"))

def test_extract_star_missing_raises_with_id():
    with pytest.raises(LookupError, match="K99999.99"):
        extract_star("K99999.99", load_config())

@pytest.mark.skipif(not Path("C:/kepler_cache").exists(), reason="needs cache")
def test_extract_star_shapes():
    planet, _ = pick_examples(load_config())
    s = extract_star(planet, load_config())
    assert s["global_view"].shape == (2001,) and s["local_view"].shape == (201,)
    assert len(s["time"]) == len(s["raw_flux"]) == len(s["trend"]) > 10_000
    assert np.isfinite(s["detrended_flux"]).all()
```

- [ ] **Step 2: Run `.venv\Scripts\python -m pytest tests/test_video_extract.py -v`; expect import failure.**
- [ ] **Step 3: Implement `video/extract_data.py`.** Parse numbers from the markdown tables and JSON under `outputs/` (not from prose in `docs/`, except `cnn_full_train_time_text`, taken from `docs/evaluation.md` and recorded in `sources`). For stars, reuse `exoplanet.data.lightcurves.read_cached_star` and the functions in `exoplanet.features.lightcurve` step by step so each intermediate array is kept. `pick_examples` chooses a confirmed planet with SNR in the top decile, period 3-15 d, and a false positive whose local view has a visible secondary or V shape (start from K06252.01, named in `docs/lightcurve_pipeline.md`); both must have complete cache files. PR curves and bands come from the `*_scores.csv` files with `exoplanet.evaluation.pr`.
- [ ] **Step 4: Run the tests; expect pass. Run `.venv\Scripts\python video/extract_data.py`; expect every file listed above to exist, and `git status --short` to show no modified tracked file except `.gitignore`.**

### Task 2: Environment, narration script and voice

**Files:**
- Create: `video/requirements.txt`, `video/narration.md`, `video/narration.py`, `video/tts.py`
- Test: `video/tests/test_narration.py`

**Interfaces:**
- `narration.md`: one `## <scene_id>` heading per scene (`01_hook` ... `10_end`), body is the spoken text. `{name}` placeholders are filled from `numbers.json` keys so spoken numbers cannot drift from the data.
- `load_narration(path, numbers: dict) -> dict[str, str]` (ordered)
- `synthesize(scene_id: str, text: str, voice: str, out_dir: Path) -> Audio` where `Audio = (wav_path: Path, duration: float, words: list[tuple[float, float, str]])`; cached on `sha1(voice + text)`
- `caption_cues(text: str, words, duration: float, max_chars: int = 84) -> list[tuple[float, float, str]]`

- [ ] **Step 1: Create `C:\video_env` (`py -3.12 -m venv`), install `video/requirements.txt` (numpy, matplotlib, imageio-ffmpeg, edge-tts, pytest, pillow).**
- [ ] **Step 2: Write failing tests**

```python
def test_placeholders_filled_and_unknown_rejected():
    assert "6,677" in load_narration(p, {"n_kois": 6677})["02_problem"]
    with pytest.raises(KeyError, match="no_such_key"): ...

def test_ten_scenes_in_order_and_word_budget():
    n = load_narration(NARRATION, NUMBERS)
    assert list(n) == SCENE_IDS
    assert 500 <= sum(len(t.split()) for t in n.values()) <= 640

def test_wording_rules():
    text = " ".join(load_narration(NARRATION, NUMBERS).values()).lower()
    assert "planet detected" not in text and "discovered a planet" not in text

def test_caption_cues_fallback_without_word_timings():
    cues = caption_cues("one two three four " * 20, words=[], duration=20.0)
    assert cues[0][0] == 0.0 and abs(cues[-1][1] - 20.0) < 0.05
    assert all(len(c[2]) <= 84 for c in cues)
    assert all(a[1] <= b[0] + 1e-6 for a, b in zip(cues, cues[1:]))

def test_synthesize_uses_cache_without_network(monkeypatch, tmp_path):
    # pre-seed cache files, make edge_tts.Communicate raise, expect Audio returned
```

- [ ] **Step 3: Write `video/narration.md`** following the spec's storyboard, scene lengths and wording rules; about 570 words.
- [ ] **Step 4: Implement `narration.py` and `tts.py`.** edge-tts yields MP3 plus boundary events; convert to 48 kHz mono WAV with the imageio-ffmpeg binary; duration measured from the WAV. If no word boundaries are returned, `words` is `[]`.
- [ ] **Step 5: Run `C:\video_env\Scripts\python -m pytest video/tests/test_narration.py -v`; expect pass.**
- [ ] **Step 6: Render scene `01_hook` in both voices to `video/build/voice_sample_neerja.wav` and `voice_sample_prabhat.wav`.**
- [ ] **Step 7: USER GATE. Show the script and the two samples; record the chosen voice in `video/credits.json` under `"voice"`.**

### Task 3: Render engine and style

**Files:**
- Create: `video/style.py`, `video/engine.py`, `video/scenes/__init__.py`
- Test: `video/tests/test_engine.py`

**Interfaces:**
- `style.py`: `BG, FG, MUTED, C_PLANET, C_FALSE, C_ACCENT`, `FONT`, `new_figure(width: int, height: int) -> Figure`, `ease(x: float) -> float` (smoothstep, clamped to 0-1), `progress(t, start, end) -> float`, `draw_caption(fig, text: str) -> None` (bottom band, wraps to at most two lines), `starfield(ax, seed: int, t: float) -> None`, `terminal(ax, lines: list[str], chars_shown: int) -> None`
- `engine.py`:
  - `@dataclass Context: scene_id, duration, fps, width, height, numbers: dict, data_dir: Path, cues: list[tuple[float, float, str]], credits: dict`; method `load(name: str)` returns an npz or json from `data_dir`
  - a scene module exposes `build(ctx: Context) -> Callable[[Figure, float], None]`; the callable draws the frame for time `t` seconds on a cleared figure. Scenes express timing as fractions of `ctx.duration`, never as absolute seconds.
  - `render_scene(scene_id: str, ctx: Context, audio_wav: Path, out_mp4: Path) -> None`: frame count is `round(duration * fps)`; pipes `buffer_rgba` frames to ffmpeg rawvideo, muxes the WAV; the engine draws the caption active at `t` after the scene draws.
  - `concat(clips: list[Path], out_mp4: Path) -> float` returns total seconds
  - `PAD = 0.6` seconds of silence appended to each scene's audio

- [ ] **Step 1: Write failing tests**

```python
def test_render_scene_length_follows_audio(tmp_path):
    # 2.0 s silent wav, trivial scene; ffprobe-style check via imageio_ffmpeg
    assert abs(video_duration(out) - 2.0) < 0.1
    assert frame_size(out) == (960, 540)

def test_scene_stretches(tmp_path):
    # same trivial scene with duration 1.0 and 5.0: last frame identical in both
    assert np.array_equal(last_frame(short), last_frame(long))

def test_caption_wraps_to_two_lines_within_frame():
    fig = new_figure(960, 540); draw_caption(fig, "word " * 17)
    # rendered text bbox inside the figure, at most 2 lines

def test_concat_sums_durations(tmp_path): ...
```

- [ ] **Step 2: Run; expect failure. Step 3: implement. Step 4: run; expect pass.**

### Task 4: Scenes 1-3 (hook, problem, data and splits)

**Files:**
- Create: `video/scenes/s01_hook.py`, `s02_problem.py`, `s03_data.py`
- Test: `video/tests/test_scenes.py` (shared by Tasks 4-8)

**Interfaces:** consumes `Context`, `style`, `star_planet.npz`, `numbers.json`, `splits_sample.json`, `credits.json`. Produces `build(ctx)` per module.

- [ ] **Step 1: Write the shared scene test (fails for unbuilt scenes; parametrised over all ten IDs, each task makes its own pass)**

```python
@pytest.mark.parametrize("scene_id", SCENE_IDS)
def test_scene_draws_start_middle_end(scene_id):
    ctx = make_ctx(scene_id, duration=20.0, width=960, height=540)
    draw = import_scene(scene_id).build(ctx)
    frames = [render(draw, t) for t in (0.0, 10.0, 19.99)]
    assert all(f.shape == (540, 960, 4) for f in frames)
    assert not np.array_equal(frames[0], frames[1])      # it animates
    assert frames[1].std() > 5                           # not blank

@pytest.mark.parametrize("scene_id", SCENE_IDS)
def test_scene_has_no_literal_results(scene_id):
    src = scene_source(scene_id)
    assert not re.search(r"0\.9\d\d|6[,]?677|2[,]?731|3[,]?946", src)
```

- [ ] **Step 2: Implement.** s01: star field, real folded light curve draws in left to right and dips, title and placeholder name from `credits.json`. s02: planet disc crossing a star with a synchronised flux trace, then counts of confirmed and false positive growing as two coloured dot clouds. s03: catalogue icon to five dev folds and a test block; dots for the 60 sample stars fly to their partition, with multi-KOI stars moving together to show host-star grouping.
- [ ] **Step 3: Run the scene tests for these three IDs; expect pass. Render each in draft and look at frames at 0%, 50%, 100%.**

### Task 5: Scenes 4-5 (leakage audit, light-curve pipeline)

**Files:** Create `video/scenes/s04_leakage.py`, `s05_pipeline.py`

**Interfaces:** consumes `audit_columns.json`, `numbers.json` (`leak_lc_audited`, `leak_lc_plus_vetting`), `star_planet.npz`, `star_binary.npz`.

- [ ] **Step 1: Implement s04.** 153 column tiles sort into "kept" and "removed" groups; the six vetting columns are named; a bar grows to `leak_lc_audited`, then jumps to `leak_lc_plus_vetting` in the warning colour when the six tiles slide back in, then they are removed again.
- [ ] **Step 2: Implement s05.** Five stages on the planet star, each a fraction of the scene: raw quarters (coloured by `quarter_index`), stitched, trend line drawn over then divided out, time axis wrapping into phase (points animate from `time` to `phase` position), binning into the global and local views. Final third: planet and eclipsing binary local views side by side in their class colours.
- [ ] **Step 3: Run scene tests for both IDs; expect pass.**
- [ ] **Step 4: Draft-render scenes 4 and 5 with audio. USER GATE: confirm the look before the remaining scenes.**

### Task 6: Scenes 6-7 (models, putting it together)

**Files:** Create `video/scenes/s06_models.py`, `s07_integration.py`

**Interfaces:** consumes `star_planet.npz` (views feed the diagram), `terminal/*.txt`, `numbers.json` (`cnn_full_train_time_text`, `n_views_built`).

- [ ] **Step 1: Implement s06.** Two-tower diagram built from boxes: global view and local view enter separate conv towers, merge, output a score; beside it the tabular model as catalogue feature groups (transit, stellar, centroid) into gradient boosting. Layer counts are read from `configs/default.yaml` via `numbers.json` if shown; otherwise omitted.
- [ ] **Step 2: Implement s07.** `style.terminal` replays the captured commands in pipeline order with typed input and real output (trimmed to the last lines that fit); long jobs show their log tail with a caption quoting the duration and its source file.
- [ ] **Step 3: Run scene tests for both IDs; expect pass.**

### Task 7: Scenes 8-9 (results, what it means)

**Files:** Create `video/scenes/s08_results.py`, `s09_meaning.py`

**Interfaces:** consumes `pr_curves.npz`, `error_breakdown.json`, `numbers.json`.

- [ ] **Step 1: Implement s08.** Test PR curves draw in one model at a time with bootstrap bands, direct labels at line ends; table of test PR-AUC with intervals; recall-by-SNR and recall-by-period bars for the CNN; then dev curves for `tabular_lc` and `cnn_full`, labelled "development folds" on screen.
- [ ] **Step 2: Implement s09.** A score dial reading "worth a follow-up", with the sentence from the README's "What the scores mean"; three next steps from `docs/evaluation.md` held in `numbers.json["next_steps"]`.
- [ ] **Step 3: Run scene tests for both IDs; expect pass.**

### Task 8: Scene 10 (end card and loop)

**Files:** Create `video/scenes/s10_end.py`

- [ ] **Step 1: Add test**

```python
def test_loop_seam():
    last = render(import_scene("10_end").build(ctx10), ctx10.duration - 1e-3)
    first = render(import_scene("01_hook").build(ctx01), 0.0)
    assert np.abs(last.astype(int) - first.astype(int)).mean() < 2.0
```

- [ ] **Step 2: Implement.** Title, placeholder names, institution and guide from `credits.json`, tools used; the final 15% cross-fades to scene 1's first frame (same `starfield` seed and phase).
- [ ] **Step 3: Run; expect pass.**

### Task 9: Build command and final render

**Files:**
- Create: `video/make_video.py`, `video/README.md`
- Test: `video/tests/test_make_video.py`

**Interfaces:** `python video/make_video.py [--scenes 4,5] [--draft] [--voice NAME] [--jobs N]`; `check_length(total_seconds: float) -> None` raises `SystemExit` with the actual length outside 225-270 s; scenes render in parallel processes; unchanged scenes (same audio hash, source file mtime and data) are reused from `video/build/clips/`.

- [ ] **Step 1: Write failing tests**

```python
def test_check_length_bounds():
    check_length(250.0)
    for bad in (224.0, 271.0):
        with pytest.raises(SystemExit, match=str(int(bad))): check_length(bad)

def test_scene_selection_parses():
    assert parse_scenes("4,5") == ["04_leakage", "05_pipeline"]
```

- [ ] **Step 2: Implement; run tests; expect pass.**
- [ ] **Step 3: Run the whole suite: `C:\video_env\Scripts\python -m pytest video/tests -v` and `.venv\Scripts\python -m pytest -q`; expect all pass.**
- [ ] **Step 4: `make_video.py --draft`; watch it end to end. USER GATE: full draft.**
- [ ] **Step 5: Final render. Verify with the ffmpeg binary: 1920x1080, 30 fps, h264 + aac, length within 3:45-4:30; play it on repeat in the laptop's player and watch the loop seam once.**
- [ ] **Step 6: Write `video/README.md`: how to edit the script, set credits, re-render, and play on loop at the exhibition (including power and sleep settings).**
