<!--
Narration for the exhibition video. One "## scene_id" section per scene; the text below
each heading is spoken as written. {placeholders} are filled from
video/build/data/numbers.json (format convention: video/narration.py). Rebuild the audio
after editing: C:\video_env\Scripts\python video/tts.py
Wording rules: a high score means "worth a follow-up", never a detection; 0.999 is spoken
only as the leakage warning.
-->

## 01_hook

Far away, a planet passes in front of its star, and the star grows a little dimmer. NASA's
Kepler telescope recorded thousands of dips like this. Which are planets, and which are
impostors?

## 02_problem

A planet crossing its star blocks a tiny fraction of the light. That brief dip is called a
transit. We use {n_kois} of Kepler's signals that astronomers have already judged.
{n_confirmed} are confirmed planets.
{n_false_positive} are false positives, such as two stars eclipsing each other.

## 03_data

First, we locked away {n_test} signals as a final exam, marked only once, at the very end.
The other {n_dev} were split into five practice groups. Each star stays in one group, so a
model cannot pass by recognising the star instead of the signal.

## 04_leakage

Next, a safety check called a leakage audit. The catalogue has {n_audit_columns} columns.
Some are verdicts from NASA's own checking software: the answer we want to predict. The
audit removed {n_audit_removed}. We let six verdict columns back in. Our score, where one is perfect, measures how well a model ranks planets above false
alarms. It jumps from {leak_lc_audited} to {leak_lc_plus_vetting}. Too good to be true, and
it is. The model was copying the answer key.

## 05_pipeline

For each star, we download its brightness measurements, and stitch Kepler's three-month
pieces together. We flatten the star's own slow changes, keeping the short dips. Then we
fold on the orbit, so every transit lands on top of the others and a faint dip stands out.
We draw two pictures: the whole orbit, and a close-up of the transit. We built these for {n_views_built} of the {n_kois} signals. A
planet makes a shallow, rounded dip. Two stars eclipsing each other make a deep, sharp V.

## 06_models

We trained two kinds of model. A neural network looks at the two pictures and gives a
score from zero to one. A team of small decision trees, each fixing the last one's
mistakes, reads catalogue numbers: the dip's depth and length, the star itself, and
whether its image shifts during the dip.

## 07_integration

The project runs as a chain of commands: fetch, audit, split, build the views, train,
evaluate, report. Automated tests check every step. Building the views took several hours,
and could resume after interruptions. Training the neural network on a laptop graphics
card took {cnn_full_train_time_text}.

## 08_results

On the final exam, the catalogue model wins, scoring {test_pr_auc.tabular}. Without the
image-shift numbers, it scores {test_pr_auc.tabular_lc}. The neural network, reading the
light alone, scores {test_pr_auc.cnn}. It misses more faint and short-orbit planets.
About half the false positives it lets through are light from a neighbouring star, which a light curve cannot reveal. It learned from only a fraction of the data.
Retrained on all of it, it scores {dev_full_pr_auc.cnn_full} on the practice groups,
against {dev_full_pr_auc.tabular_lc} for the matching catalogue model. More data closed most of the gap.

## 09_meaning

So a high score means a signal looks like the ones that became planets. It is worth a
follow-up, not a discovery. Next: combine the two models, find more faint planets, and use
pixel images to catch light from neighbouring stars.

## 10_end

This was Kepler Exoplanet Transit Detection, built in Python on real NASA data. Thank you
for watching.
