# Blended Project Video: Design

Date: 2026-11-24. Status: awaiting approval.

## Intent

The exhibition brief (section B) requires a 3-5 minute "blended" video that plays continuously
on the laptop and explains the complete development journey of the Kepler exoplanet project.
It must be understandable with nobody explaining it, and include real project footage,
animations, voice-over narration, the integration process and final results.

## Goals

- One MP4, 3:45-4:30 long, 1920x1080, 30 fps, H.264 video + AAC audio, playable on repeat in
  the Windows Media Player / VLC that is already on the laptop.
- Tells the journey in order: problem, data, leakage audit, light-curve pipeline, models,
  pipeline run, results, what the scores mean.
- Every number and every plotted curve on screen is read from the project's own files
  (`outputs/*.md`, `docs/results.md`, `data/processed`, `models/`, the Kepler cache). Nothing
  is typed in by hand or invented.
- Narration by a synthesized neural voice (user's decision).
- Burned-in captions, because an exhibition hall is noisy and the brief says the video must
  work without a live explanation (my call).
- The last frame cuts cleanly to the first, so the loop has no dead air (my call).
- Re-renderable with one command after a script change.

## Non-goals

- A demo app for the live demonstration (brief section C). Separate job.
- Camera footage or hardware footage: this is a software project.
- Retraining or re-scoring any model. The held-out test set is not touched; the video only
  reads existing result files.

## Anti-goals

- Presenting a score as "planet detected". The README is explicit that the models reproduce
  a vetting judgement; the video must say the same.
- Showing the leaky 0.999 number as an achievement. It appears only as the leakage warning.
- Faked footage: terminal scenes show the real output of really running the command.
- A slideshow of bullet points read aloud.

## Constraints

- Windows 11, Python 3.12, RTX 2050. No ffmpeg or TTS installed today.
- The project `.venv` must not gain video dependencies. Video tooling goes in its own
  environment at `C:\video_env` (outside OneDrive, like `C:\kepler_cache`).
- The project is on branch `feature/brief-deliverables`; nothing is committed unless the user
  asks.
- Narration text is sent to an online speech service (see below). It contains only project
  content, no personal details.

## Approaches

1. **Rendered in Python from real data (chosen, my pick; the user left the how to me).**
   matplotlib draws every frame; frames are piped to ffmpeg; narration audio is laid
   underneath. Uses what the project already uses, is reproducible, and guarantees the
   visuals match the data. Trade-off: a clean scientific look, not motion-graphics polish.
2. **Animated web page, screen-recorded.** Better typography and transitions, but needs
   capture tooling, has fragile audio sync, and must be re-recorded for every revision.
3. **Narrated PowerPoint exported to video.** Fastest, but reads as slides, and the brief
   asks for footage and animations.

## How it is built (my calls)

- **Voice:** `edge-tts` (Microsoft neural voices, free, needs internet at render time only).
  Voice `en-IN-NeerjaNeural` or `en-IN-PrabhatNeural`; I render a 10-second sample of each
  and the user picks. Fallback if the service is unavailable: Piper, an offline neural voice.
  Audio files are cached, so re-rendering visuals does not call the service again.
- **Timing:** narration is generated per scene first; each scene's length is its audio length
  plus a short pad. Visuals are animated to fit that length, so audio and picture cannot
  drift. Caption timing comes from the word timings edge-tts returns.
- **Encoding:** `imageio-ffmpeg` supplies the ffmpeg binary (no system install).
- **Look:** dark night-sky background, one accent colour for confirmed planets and one for
  false positives, used consistently in every chart; large type readable from two metres.
- **Terminal scenes:** each command is actually run (the fast ones: tests, leakage audit,
  inspection, report); the captured output is replayed as a typed terminal. Long jobs
  (view build, training) are shown from their saved logs, labelled with their real duration.
  These are replays of real output, not a literal screen capture (my call). If the judges
  expect a literal screen recording, the user can record one with Win+Alt+R and it is cut
  into scene 7.
- **No tracked file is overwritten:** commands that write reports are run with their output
  redirected into `video/build/` (for example `make_report.py --output`). The on-screen
  command is shown as actually run.
- **Durations of long jobs** are quoted from the project's docs as a caption that names the
  source (for example "3 min 23 s on an RTX 2050, docs/evaluation.md"). The view build, which
  ran across several resumed sessions, is described as "several hours, resumable" with no
  exact figure.
- **Source of the brief:** sections B and C were pasted by the user in conversation; they are
  not in `Project details.pdf`.
- **Music:** none by default (licensing, and it competes with narration in a hall).

## Storyboard (about 4:10, about 570 words of narration)

| # | Scene | Length | On screen |
|---|---|---|---|
| 1 | Hook | 0:15 | Star field; one real confirmed planet's light curve draws in and dips. Title. |
| 2 | The problem | 0:25 | What a transit is (animated planet crossing a star, with the flux trace below). 6,677 Kepler signals: 2,731 confirmed, 3,946 false positives. |
| 3 | Data and splits | 0:25 | NASA catalogue to dev folds and held-out test; stars shown staying inside one partition (host-star grouping). |
| 4 | Leakage audit | 0:35 | The 153 catalogue columns the audit classifies (`docs/leakage_audit_results.md`) sorted into kept and removed; bar jumps 0.969 to 0.999 when the six vetting columns are let in. "Too good to be true, and it is." |
| 5 | Light-curve pipeline | 0:50 | One real star from the cache: raw quarters, stitched, detrended, folded on its period, then binned into the global (2001) and local (201) views. A planet beside an eclipsing binary. |
| 6 | Two models | 0:30 | Diagram of the 1D CNN's two towers; the gradient-boosted tabular model on catalogue features. |
| 7 | Putting it together | 0:25 | Typed terminal: pipeline commands in order with real output, tests passing, GPU training in 3 min 23 s. |
| 8 | Results | 0:40 | PR curves with bootstrap bands drawing in; the four-model table; where the CNN loses (faint and short-period planets, centroid-offset false positives); `cnn_full` closing the gap on dev. |
| 9 | What it means | 0:15 | "A high score means worth a follow-up, not a discovery." Next steps. |
| 10 | End card | 0:10 | Project title, name, tools used; fades to match scene 1's first frame. |

## Files

```
video/
  narration.md        the script, one section per scene (edit this to revise)
  make_video.py       entry point: python video/make_video.py [--scenes 4,5] [--draft]
  scenes/             one module per scene
  style.py            colours, fonts, shared drawing helpers
  build/              cached audio, captured terminal output, per-scene clips (git-ignored)
  exoplanet_project_video.mp4   the deliverable (git-ignored; ~50-100 MB)
```

`--draft` renders at 960x540 and 15 fps for quick review.

## Review steps with the user

1. Narration script and voice sample, before any visuals are built.
2. Draft render of scenes 4 and 5 (the two most complex) to confirm the look.
3. Full draft, then the final render.

## Left to the builder

Exact wording of the narration, which example stars to show (chosen for clean views from
`data/processed`), animation easing, font choice, and chart layout within the style above.

## Needed from the user

- The name (and team members, institution, guide) to show on the title and end cards.
