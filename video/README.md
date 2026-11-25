# The project video

A narrated explainer of this project, about four minutes long, made to play on a loop at
the exhibition. It walks through the problem, the Kepler data, the leakage we found, the
light-curve pipeline, the two models, the results on the held-out test set and what a high
score means. Every number and curve in it is read from the project's own files, and the
last frame fades back into the first, so the loop has no visible jump.

## The two Python environments

| Environment | Used for | Python |
|---|---|---|
| Project env | reading the project's results into `video/build/data/` | `.venv\Scripts\python` |
| Video env | narration, drawing and encoding the video | `C:\video_env\Scripts\python` |

The video tools are never installed into the project env. If `C:\video_env` is missing:

```
py -3.12 -m venv C:\video_env
C:\video_env\Scripts\python -m pip install -r video/requirements.txt
```

## Make the video

Run these from the repository root (the folder that holds `video/`):

```
.venv\Scripts\python video/extract_data.py           # once, and again if results change
C:\video_env\Scripts\python video/make_video.py --draft  # quick check: 960x540, 15 fps
C:\video_env\Scripts\python video/make_video.py          # final: 1920x1080, 30 fps
```

The build prints how long each scene is, the total length, where the file is and its size.

- Final video: `video/exoplanet_project_video.mp4`
- Draft: `video/build/exoplanet_project_video_draft.mp4`

Rendering takes a while: a few minutes for the draft, longer for the final version
(slower still if the laptop is busy with other things). Scenes whose inputs have
not changed are reused, so a second run is much faster. `--jobs N` sets how many scenes
render at once (default: half the CPU cores); use `--jobs 2` if you need the laptop
meanwhile.

- To force every scene to render again, delete the folder `video/build/clips`.
- Re-running `extract_data.py` re-renders every clip, even if no result changed: it re-runs
  the test suite for the scene 7 terminal, and its timing line is part of the scene data.

## Change the script

1. Edit `video/narration.md`. Each `## 04_leakage` heading starts one scene's text. Text
   inside `<!-- ... -->` is a note and is not spoken. `{n_kois}` and similar placeholders
   are filled from the project's numbers; leave them in place.
2. Run the build again (draft first). Only scenes whose text changed get new audio, and
   making new audio needs the internet.

The whole video must be between **3:45 and 4:30**. The current script is about 4:26, so
there is little room to add. If the build says the video is too long (or too short), it
tells you by how many seconds; cut (or add) roughly 2.5 words per second of narration and
run it again. The file is still written, so you can watch it, but the build reports a
failure until the length fits.

## Put in your names

Edit `video/credits.json`:

```json
{
  "title": "Kepler Exoplanet Transit Detection",
  "name": "Your Name",
  "team": ["Second Member", "Third Member"],
  "institution": "Your School or College",
  "guide": "Your Guide's Name",
  "voice": "en-IN-NeerjaNeural"
}
```

The title card (scene 1) shows only `title` and `name`; the end card (scene 10) shows
all of them: name, team, institution and guide. Leave `team` as `[]` or a field as `""` to
leave it out. Then run the build again. If the credits are too long for the end card, the
build shrinks them a little; if they still do not fit, it stops and names the field to
shorten.

While `name`, `institution` or `guide` still hold the shipped placeholders ("Your Name",
"Institution", "Guide Name"), a draft build prints a loud warning, and the final build
stops before rendering anything, naming the fields. To render the final video with them
anyway (for a test), add `--allow-placeholder-credits`.

## Change the voice

Set `"voice"` in `video/credits.json`, or try one without editing anything:

```
C:\video_env\Scripts\python video/make_video.py --draft --voice en-IN-PrabhatNeural
```

Any edge-tts voice name works. A new voice needs the internet once; after that it is
cached.

## Work on one scene

```
C:\video_env\Scripts\python video/make_video.py --draft --scenes 4      # one scene
C:\video_env\Scripts\python video/make_video.py --draft --scenes 4,5    # several
C:\video_env\Scripts\python video/engine.py preview 04_leakage --at 0.5 --draft  # one frame as PNG
```

`--scenes` writes only those scene clips to `video/build/clips/` and prints their paths; it
does not join the full video. Run the full build afterwards. Scenes are numbered 1-10:
1 hook, 2 problem, 3 data, 4 leakage, 5 pipeline, 6 models, 7 integration, 8 results,
9 meaning, 10 end.

## Exhibition checklist

- [ ] Copy `video/exoplanet_project_video.mp4` to the laptop's desktop **and** to a USB
      stick as a backup.
- [ ] Play it on repeat:
      - Windows **Media Player**: open the file, click the **Repeat** button in the controls
        until it shows repeat-one or repeat-all.
      - Or **VLC**: click the **Loop** button below the video (or press `L`) until it
        shows loop-one or loop-all.
- [ ] Full screen (the full-screen button, or double-click the video). Move the mouse
      pointer off the picture.
- [ ] Laptop plugged into power.
- [ ] Settings > System > Power: screen off and sleep set to **Never** for the day (both
      "plugged in" values).
- [ ] Notifications off: Settings > System > Notifications, turn on **Do not disturb**
      (on older Windows, **Focus assist** to Alarms only).
- [ ] Volume checked at the stand, with people talking around you.
- [ ] Watch it loop once from the end back to the start before visitors arrive.
