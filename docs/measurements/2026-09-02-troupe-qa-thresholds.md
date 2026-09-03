# Troupe's animation scores: what the thresholds are and where they came from

2026-09-02. `studio/troupe/qa.py` scores a rendered character sheet per cell
and the Troupe preview draws the result as a heatmap over the frame x direction
matrix. This document exists because the repo requires one before a keyed
constant exists, and `qa.THRESHOLDS` is eight of them.

## What is measured

Per run (one animation in one direction, read along the frame table's cells):

| metric | definition | range |
| --- | --- | --- |
| `shape_delta` | `1 - IoU` of the alpha mask against the previous frame | 0..1 |
| `seam_delta` | the same against the *last* frame, on frame 0 of a looping run only | 0..1 |
| `centroid_jitter` | distance the alpha centroid moved, as a fraction of the cell | 0..~1.4 |
| `foot_jitter` | how far the lowest opaque row moved, as a fraction of the cell height | 0..1 |
| `palette_flicker` | half the L1 distance between the two frames' per-colour pixel counts, over the larger sprite area | 0..1 |

Per `(animation, frame index)`, across directions:

| metric | definition |
| --- | --- |
| `drift_w`, `drift_h`, `drift_occupancy` | relative deviation of the alpha bbox width, height and opaque area from the median over the directions at that frame |

A cell with no opaque pixel is `blank`. Limb continuity is not measured: telling
a swapped limb from a swinging one needs segmentation or the rig, and the atlas
carries neither. A limb that vanishes shows up in `shape_delta` and
`palette_flicker`.

## The thresholds

| metric | warn | bad | why there |
| --- | ---: | ---: | --- |
| `shape_delta` | 0.35 | 0.60 | a third of the silhouette moving between two frames of a 6-12 frame cycle is a pop; more than half is a different pose |
| `seam_delta` | 0.35 | 0.60 | the same bar, on the one transition the preview plays most |
| `centroid_jitter` | 0.06 | 0.15 | at 32 px, 0.06 is two pixels of drift between frames; 0.15 is five |
| `foot_jitter` | 0.04 | 0.10 | the ground line: one pixel of bob at 32 px is 0.03 and is what a walk *should* do; three is a stumble |
| `palette_flicker` | 0.35 | 0.60 | a third of the sprite changing colour with the silhouette held is a shading pop |
| `drift_*` | 0.35 | 0.70 | deliberately lenient: a side view is legitimately narrower than a front view, so only a direction that is a third off the median in *width and height and area* is worth a look |

## How they were chosen, honestly

By construction against synthetic sheets, not by calibration against a rendered
corpus. `tests/troupe/test_troupe_qa.py` holds the constructions: a static sheet
scores zero on every metric; one blank frame flags itself and the frame after
it; a run whose last frame is shifted four pixels seams; a frame with half its
sprite recoloured flickers; a direction that is a mirror of another drifts by
exactly zero. Those are the cases whose answer is known in advance, and the
thresholds are set so each of them lands on the side it should.

What they are **not** yet is measured against a real 256-cell sheet from
`charsheet`. The shipped clips are placeholders (TODO.md P8), so a calibration
today would tune the bars to placeholder motion. When P8 lands, the procedure
is: score the sheet, sort cells by each metric, and look at the cells either
side of each bar in the Troupe preview -- the heatmap's click-to-jump is built
for exactly this -- then move the bar to where the eye agrees, and record the
new value here with the sheet it was tuned on.

## What the scores are for

Ranking, never gating. Nothing in `service/`, the queue, a worker or a sidecar
reads them, and `tests/troupe/test_troupe_mode.py` asserts the pane never
computes them in the frame loop. A flagged cell is a place to look first, not a
verdict on the sheet.
