# View calibration, 2026-08-04

`bench/views.py` carries two constants, `REFERENCE_YAW` and
`REFERENCE_ELEVATION`, and a comment admitting they are a guess. `bench/calibrate.py`
exists for one reason: to settle whether the thing they assume is even true --
that trellis-server aligns a reconstruction to the camera of the image it was
given, and that `normalize_glb` preserves that alignment, so that one of the
eight rendered views can be compared directly against `input.png`. Nothing in
this repo had established either. `PROMPT_TEMPLATE` asks SDXL for a "3/4
perspective view" rather than an axis-aligned front, and trellis-server.exe is a
vendored binary whose behaviour is confirmed by measurement rather than by
upstream documentation.

This matters beyond the bench. Part A Task 9 proposes a fidelity score on the
request path -- a number shown to the user about how well a mesh matches the
reference it came from. That number is only meaningful if a matched view exists.
If it does not, the score is a measurement of the camera.

So the sweep was run. This document is dated by the plan's schedule, per the
task brief; the sweep itself ran on **2026-08-05**, over roughly 25 minutes of
GPU (144 rendered and scored views per job, about 41 s per job).

## What was run

Step 1, to see whether there was anything to sweep:

```
uv run python -c "
from pathlib import Path
from warlock.config import get_config
root = Path(get_config().data_dir)
found = [d.name for d in root.iterdir() if (d/'model.glb').exists() and (d/'input.png').exists()]
print(len(found)); print(found)
"
```

**37** job directories under `assets/` carry both `model.glb` and `input.png`.
These are the meshes Part A Task 6's bench run left behind -- `core-v1` at seeds
42 and 1337 under `baseline-turbo-raw`, all 40 items -- so they span every
category the suite declares. By category: 12 vehicle, 10 prop, 7 character, 7
weapon, 1 environment. That is far above the brief's floor of 5-10 spanning
categories and well above `calibrate.verdict`'s three-job minimum, below which
it refuses to read a mode into the data at all.

Step 2, the sweep itself:

```
uv run python -m warlock.bench calibrate --all
```

Exit code 0. Full captured output at `.superpowers/sdd/UPDATE/calibrate-run.log`;
the structured per-cell data at `bench/calibrate/calibrate.json`.

The 37 job ids swept, in the order they appear in every list below:

```
024a9266c0dc  0637d71ef129  06816a103236  0b007a1facfb  0e636413a6f8
26cf522d98a9  2c2372e98171  304ed66774bf  3a2b24d9c4c3  3c71ff5e2aee
3ffe6cc6009c  428a79c9cc55  464eefc6e3aa  4ed5bfdc55d7  4f36f289d3a3
50a6b5680ea2  53c97ee8343f  5d14fe7a2592  62bbfb4651e2  6cb95de0cb11
732ba8258214  75da99410e09  7d3e978f3933  7dc30f469947  a23136069048
adf08df94029  b504b01fccc1  b6b7f6409682  ba70a161ffe5  c19f34c528a0
c427a8a5592b  ce1780be78ea  db71820e88db  ef426093b8c6  f82d943acb97
f98ac327e91d  ff79cc39fed5
```

## The sweep would not run at first: transformers 5.14 needs torchvision

Recorded here because it is a real finding with a consequence larger than this
task, and the next reader should not have to rediscover it.

`AutoImageProcessor.from_pretrained` for `models/dinov2-base` raised a bare
`ImportError`. **transformers 5.14 has dropped the PIL fallback for
`BitImageProcessor` and now requires torchvision**, which this project does not
declare in `pyproject.toml`. The `dino_cosine` half of the sweep could not run
at all until that was fixed.

It was unblocked by installing torchvision into the venv only, matching the
pinned `torch 2.11.0+cu128`:

```
uv pip install --index-url https://download.pytorch.org/whl/cu128 torchvision
# torchvision==0.26.0+cu128
```

That is **not** in `pyproject.toml`, so the next `uv sync` removes it again and
the sweep becomes un-rerunnable in the same way. It is deliberately left
undeclared here rather than fixed in passing, because adding a dependency is not
this task's remit and torch-family pins want their own change.

The consequence worth writing down is not the bench, though. **Part A Task 4's
`reference_cosine` -- the anchor half of the candidate rank score -- goes down
the same `AutoImageProcessor` path.** On any install without torchvision it
raises, the raise is caught, and the rank silently degrades to composition-only:
the anchor a user configured is quietly not consulted, and nothing says so. The
doctor's DINOv2 row does not catch this either, because it reports only that the
weights are on disk. All of that is outside this task and is not touched here,
but it is a live gap on every machine that has not had torchvision hand-installed.

## The numbers

`format_table` prints the eight best cells per job per metric; the log carries
all 74 of those blocks and `bench/calibrate/calibrate.json` carries every one of
the 144 cells per job with both scores. With 37 jobs, reproducing all 74 blocks
here would bury the finding under about six hundred lines of table, so this
document gives **every job's argmax row for both metrics in one summary table**,
then the **full `format_table` blocks for six representative jobs** -- one per
category, plus the two extremes of agreement. The remaining 31 jobs' blocks are
in `.superpowers/sdd/UPDATE/calibrate-run.log` verbatim, and the underlying
cells in `bench/calibrate/calibrate.json`.

Argmax per job. `sil` is `silhouette_iou`, `dino` is `dino_cosine`; yaw and
elevation in degrees.

| job | category | sil yaw | sil elev | sil score | dino yaw | dino elev | dino score |
|---|---|---|---|---|---|---|---|
| 024a9266c0dc | vehicle | 170 | 10 | 0.7257 | 80 | 10 | 0.8212 |
| 0637d71ef129 | vehicle | 250 | 30 | 0.5963 | 80 | 30 | 0.8304 |
| 06816a103236 | prop | 170 | 30 | 0.8745 | 20 | 20 | 0.7142 |
| 0b007a1facfb | vehicle | 40 | 30 | 0.4719 | 150 | 20 | 0.7943 |
| 0e636413a6f8 | character | 60 | 0 | 0.7075 | 0 | 30 | 0.8841 |
| 26cf522d98a9 | weapon | 200 | 30 | 0.3011 | 0 | 30 | 0.2132 |
| 2c2372e98171 | weapon | 150 | 10 | 0.4848 | 330 | 0 | 0.9053 |
| 304ed66774bf | character | 310 | 20 | 0.6672 | 180 | 20 | 0.6408 |
| 3a2b24d9c4c3 | vehicle | 350 | 30 | 0.4764 | 350 | 30 | 0.8025 |
| 3c71ff5e2aee | weapon | 30 | 20 | 0.5368 | 340 | 20 | 0.8423 |
| 3ffe6cc6009c | character | 230 | 30 | 0.6786 | 130 | 0 | 0.3324 |
| 428a79c9cc55 | weapon | 30 | 0 | 0.4913 | 140 | 20 | 0.0498 |
| 464eefc6e3aa | vehicle | 30 | 30 | 0.6489 | 330 | 10 | 0.2347 |
| 4ed5bfdc55d7 | prop | 110 | 30 | 0.4496 | 90 | 30 | 0.0995 |
| 4f36f289d3a3 | environment | 50 | 10 | 0.8054 | 50 | 0 | 0.8608 |
| 50a6b5680ea2 | vehicle | 320 | 30 | 0.5557 | 80 | 0 | 0.0726 |
| 53c97ee8343f | prop | 180 | 0 | 0.5650 | 40 | 0 | 0.5075 |
| 5d14fe7a2592 | character | 110 | 10 | 0.5424 | 150 | 0 | 0.5018 |
| 62bbfb4651e2 | prop | 200 | 10 | 0.7621 | 0 | 0 | 0.1917 |
| 6cb95de0cb11 | weapon | 200 | 20 | 0.3814 | 300 | 10 | 0.6648 |
| 732ba8258214 | vehicle | 30 | 0 | 0.5334 | 140 | 10 | 0.5936 |
| 75da99410e09 | prop | 220 | 20 | 0.7182 | 200 | 0 | 0.4020 |
| 7d3e978f3933 | prop | 70 | 30 | 0.5597 | 0 | 30 | 0.7849 |
| 7dc30f469947 | prop | 230 | 10 | 0.7508 | 50 | 0 | 0.4069 |
| a23136069048 | vehicle | 100 | 30 | 0.7954 | 90 | 10 | 0.9166 |
| adf08df94029 | vehicle | 300 | 30 | 0.8195 | 100 | 10 | 0.8717 |
| b504b01fccc1 | vehicle | 190 | 0 | 0.6428 | 70 | 10 | 0.8020 |
| b6b7f6409682 | character | 0 | 10 | 0.7730 | 0 | 10 | 0.4602 |
| ba70a161ffe5 | prop | 120 | 30 | 0.7202 | 110 | 20 | 0.8370 |
| c19f34c528a0 | vehicle | 100 | 30 | 0.6554 | 240 | 20 | 0.8761 |
| c427a8a5592b | character | 310 | 30 | 0.7177 | 0 | 0 | 0.8331 |
| ce1780be78ea | weapon | 130 | 0 | 0.8194 | 70 | 0 | 0.7016 |
| db71820e88db | prop | 180 | 30 | 0.6768 | 20 | 10 | 0.4421 |
| ef426093b8c6 | weapon | 280 | 30 | 0.4097 | 90 | 10 | 0.3006 |
| f82d943acb97 | vehicle | 100 | 30 | 0.6842 | 320 | 20 | 0.7070 |
| f98ac327e91d | character | 320 | 10 | 0.7513 | 10 | 10 | 0.5741 |
| ff79cc39fed5 | prop | 10 | 30 | 0.7049 | 20 | 20 | 0.8570 |

### Six representative jobs, in full

One per category (`4f36f289d3a3` is the only environment item in the sample),
plus `a23136069048`, which scores the highest `dino_cosine` in the whole sweep
and is therefore the strongest single candidate for "this one, at least, is
matched".

`024a9266c0dc` -- vehicle, "a military motorcycle with saddlebags":

```
024a9266c0dc  best silhouette_iou cells:
    yaw   elev    score
   170.0   10.0   0.7257
   170.0   20.0   0.7175
   190.0   10.0   0.7157
   170.0    0.0   0.7140
   190.0   20.0   0.7116
   190.0    0.0   0.7102
     0.0    0.0   0.7087
   180.0    0.0   0.7087
  (worst: yaw 130.0 elev 0.0 = 0.3972)
024a9266c0dc  best dino_cosine cells:
    yaw   elev    score
    80.0   10.0   0.8212
    80.0   20.0   0.7844
    90.0   10.0   0.7751
    70.0   10.0   0.7698
   280.0   20.0   0.7697
    80.0    0.0   0.7695
   280.0   10.0   0.7587
    70.0   20.0   0.7568
  (worst: yaw 180.0 elev 30.0 = 0.1506)
```

`06816a103236` -- prop, "a stack of leather-bound books":

```
06816a103236  best silhouette_iou cells:
    yaw   elev    score
   170.0   30.0   0.8745
   170.0   20.0   0.8739
   170.0   10.0   0.8608
    10.0   10.0   0.8549
   190.0   20.0   0.8536
    10.0    0.0   0.8524
   170.0    0.0   0.8462
   190.0   30.0   0.8398
  (worst: yaw 100.0 elev 0.0 = 0.5709)
06816a103236  best dino_cosine cells:
    yaw   elev    score
    20.0   20.0   0.7142
    10.0   20.0   0.6904
    20.0   30.0   0.6847
    10.0   10.0   0.6828
    30.0   20.0   0.6775
    30.0   10.0   0.6605
    20.0   10.0   0.6552
    40.0   20.0   0.6490
  (worst: yaw 180.0 elev 0.0 = 0.1222)
```

`b6b7f6409682` -- character, "a cartoon mushroom creature with stubby arms":

```
b6b7f6409682  best silhouette_iou cells:
    yaw   elev    score
     0.0   10.0   0.7730
    20.0   10.0   0.7697
    10.0   10.0   0.7675
   350.0   10.0   0.7586
     0.0    0.0   0.7561
   350.0    0.0   0.7518
    30.0   10.0   0.7515
    10.0    0.0   0.7495
  (worst: yaw 60.0 elev 30.0 = 0.5903)
b6b7f6409682  best dino_cosine cells:
    yaw   elev    score
     0.0   10.0   0.4602
   350.0   10.0   0.4492
   330.0   10.0   0.4398
   340.0   10.0   0.4307
    10.0   10.0   0.4255
    40.0   10.0   0.4246
    60.0   10.0   0.4217
    30.0   10.0   0.4151
  (worst: yaw 190.0 elev 30.0 = 0.1511)
```

`ce1780be78ea` -- weapon, "a riot shield with a reinforced rim":

```
ce1780be78ea  best silhouette_iou cells:
    yaw   elev    score
   130.0    0.0   0.8194
   310.0    0.0   0.8193
   230.0    0.0   0.8189
    50.0    0.0   0.8187
    50.0   10.0   0.8048
   230.0   10.0   0.8042
   310.0   10.0   0.8031
   130.0   10.0   0.8031
  (worst: yaw 270.0 elev 10.0 = 0.0876)
ce1780be78ea  best dino_cosine cells:
    yaw   elev    score
    70.0    0.0   0.7016
    60.0    0.0   0.6878
   310.0    0.0   0.6793
   350.0    0.0   0.6710
    10.0    0.0   0.6680
   320.0    0.0   0.6648
     0.0    0.0   0.6647
    50.0    0.0   0.6583
  (worst: yaw 90.0 elev 20.0 = 0.1100)
```

`4f36f289d3a3` -- environment, "a wooden watchtower on stilts":

```
4f36f289d3a3  best silhouette_iou cells:
    yaw   elev    score
    50.0   10.0   0.8054
    40.0   10.0   0.7982
    70.0   10.0   0.7937
    50.0   20.0   0.7869
    60.0   10.0   0.7852
    40.0   20.0   0.7756
    70.0    0.0   0.7744
    30.0   10.0   0.7712
  (worst: yaw 330.0 elev 0.0 = 0.2271)
4f36f289d3a3  best dino_cosine cells:
    yaw   elev    score
    50.0    0.0   0.8608
    40.0    0.0   0.8573
    30.0    0.0   0.8547
   240.0    0.0   0.8505
    50.0   10.0   0.8476
   220.0    0.0   0.8433
    30.0   10.0   0.8429
   230.0    0.0   0.8421
  (worst: yaw 170.0 elev 10.0 = 0.5155)
```

`a23136069048` -- vehicle, "a wooden handcart with two spoked wheels", the
sweep's best `dino_cosine`:

```
a23136069048  best silhouette_iou cells:
    yaw   elev    score
   100.0   30.0   0.7954
   100.0   20.0   0.7792
    10.0   30.0   0.7756
   110.0   30.0   0.7670
    80.0   30.0   0.7447
   110.0   20.0   0.7389
    80.0   20.0   0.7345
    10.0   20.0   0.7282
  (worst: yaw 240.0 elev 0.0 = 0.2914)
a23136069048  best dino_cosine cells:
    yaw   elev    score
    90.0   10.0   0.9166
   100.0   10.0   0.9130
    90.0    0.0   0.9112
   100.0    0.0   0.9068
    80.0   10.0   0.9012
   110.0   10.0   0.9004
    90.0   20.0   0.8965
    90.0   30.0   0.8952
  (worst: yaw 240.0 elev 10.0 = 0.6606)
```

## The verdict lines, verbatim

```
silhouette_iou: scattered: there is no fixed matched view -- use max-over-8-views (n=37, argmax yaws [170.0, 250.0, 170.0, 40.0, 60.0, 200.0, 150.0, 310.0, 350.0, 30.0, 230.0, 30.0, 30.0, 110.0, 50.0, 320.0, 180.0, 110.0, 200.0, 200.0, 30.0, 220.0, 70.0, 230.0, 100.0, 300.0, 190.0, 0.0, 120.0, 100.0, 310.0, 130.0, 180.0, 280.0, 100.0, 320.0, 10.0], spread 330.0 deg)
dino_cosine: scattered: there is no fixed matched view -- use max-over-8-views (n=37, argmax yaws [80.0, 80.0, 20.0, 150.0, 0.0, 0.0, 330.0, 180.0, 350.0, 340.0, 130.0, 140.0, 330.0, 90.0, 50.0, 80.0, 40.0, 150.0, 0.0, 300.0, 140.0, 200.0, 0.0, 50.0, 90.0, 100.0, 70.0, 0.0, 110.0, 240.0, 0.0, 70.0, 20.0, 90.0, 320.0, 10.0, 20.0], spread 300.0 deg)
```

## Verdict

Both metrics were measured, so both get a verdict, and they say the same thing.

> **Scattered.** For `silhouette_iou` the argmax yaws are
> `[170.0, 250.0, 170.0, 40.0, 60.0, 200.0, 150.0, 310.0, 350.0, 30.0, 230.0, 30.0, 30.0, 110.0, 50.0, 320.0, 180.0, 110.0, 200.0, 200.0, 30.0, 220.0, 70.0, 230.0, 100.0, 300.0, 190.0, 0.0, 120.0, 100.0, 310.0, 130.0, 180.0, 280.0, 100.0, 320.0, 10.0]`,
> spread **330** degrees. There is no fixed matched view, so a request-path
> fidelity score would be measuring the camera, not the mesh. Task 9 stops here;
> `views.py` keeps its UNCALIBRATED note, amended to say the sweep was run and
> what it found.

> **Scattered.** For `dino_cosine` the argmax yaws are
> `[80.0, 80.0, 20.0, 150.0, 0.0, 0.0, 330.0, 180.0, 350.0, 340.0, 130.0, 140.0, 330.0, 90.0, 50.0, 80.0, 40.0, 150.0, 0.0, 300.0, 140.0, 200.0, 0.0, 50.0, 90.0, 100.0, 70.0, 0.0, 110.0, 240.0, 0.0, 70.0, 20.0, 90.0, 320.0, 10.0, 20.0]`,
> spread **300** degrees. There is no fixed matched view, so a request-path
> fidelity score would be measuring the camera, not the mesh. Task 9 stops here;
> `views.py` keeps its UNCALIBRATED note, amended to say the sweep was run and
> what it found.

`calibrate.STABLE_YAW_SPREAD` is **30.0** degrees. Both figures are an order of
magnitude outside it, on 37 jobs. This is not a marginal call and no reasonable
choice of threshold rescues it: the spread is measured circularly, so 350 and 10
count as 20 degrees apart, and even so the argmaxes leave only a 30-degree gap
(`silhouette_iou`) or a 60-degree gap (`dino_cosine`) anywhere on the circle.
They are, to a first approximation, uniform.

## Three things that make the verdict stronger than the headline number

**The two metrics do not agree with each other either.** They were computed
over the same 144 renders, so if a matched view existed both should find it. In
8 of the 37 jobs their argmax yaws fall within 30 degrees of one another.
Chance alone would put seven of them there -- 7 of the 36 yaw bins are within
±30 degrees, so 19.4% of 37 is 7.2. The agreement between the two metrics is
statistically indistinguishable from none. That rules out the more hopeful
reading of the headline number, which would have been that one metric is simply
noisy while the other is tracking a real alignment.

**It scatters within every category, not just across them.** If trellis aligned
*some* subject classes and not others -- plausible, since a character has an
obvious front and a rock does not -- then the per-category spreads would be
tight and only the pooled figure would be wide. They are not:

| category | n | sil spread | dino spread |
|---|---|---|---|
| vehicle | 12 | 290 | 270 |
| prop | 10 | 220 | 200 |
| character | 7 | 240 | 180 |
| weapon | 7 | 250 | 200 |
| environment | 1 | -- | -- |

Every category with more than one item scatters by at least 180 degrees on both
metrics. There is no subset of the item space where a fixed reference yaw would
have worked.

**What `silhouette_iou` actually maximises is visible in its elevation.** Its
best elevation is 30 degrees -- the steepest the sweep offers -- in 19 of 37
jobs, against a roughly flat spread for `dino_cosine` (11 at 0, 11 at 10, 9 at
20, 6 at 30). A silhouette IoU against a reference image rewards whichever
camera makes the mesh's outline most compact, because a compact outline overlaps
a compact reference outline best; it is not preferentially finding the pose the
reference was drawn from. Several jobs show this outright. `ce1780be78ea`, the
riot shield, has its top four silhouette cells at 130, 310, 230 and 50 degrees
scoring 0.8194, 0.8193, 0.8189 and 0.8187 -- four yaws 80 to 180 degrees apart,
separated by seven ten-thousandths. `0e636413a6f8` has 60, 120, 240 and 300 all
at exactly 0.7075. A metric whose maximum is that flat has no argmax worth
reading; the reported one is a rounding artefact.

## What was changed

`src/warlock/bench/views.py`'s UNCALIBRATED paragraph, amended to record that
the sweep ran, when, over how many jobs, and how far it scattered.
`REFERENCE_YAW` and `REFERENCE_ELEVATION` are **deliberately left at 0.0 and
20.0**. They stay uncalibrated because there is nothing to calibrate them to,
and the guarded pre-calibration machinery already in `views.py` -- the separate
`yaw_offset` recorded in the sidecar, `"calibrated": False` -- is exactly right
as it stands. Nothing else in the repo was touched.

## Consequences

Part A **Task 9 is not built.** A request-path fidelity score comparing a fixed
rendered view against `input.png` would report the angle between two arbitrary
cameras, dressed up as a statement about mesh quality, and it would be shown to
a user. That is worse than no score.

The honest form of the metric survives, and `calibrate.py`'s own docstring named
it before the sweep ran: **max-over-eight-views**. Take the best of the eight
turntable views rather than the first, and the number stops depending on an
alignment that does not exist. It is weaker -- a max over eight samples is
biased upward and cannot distinguish "matched well" from "matched somewhere" --
but it is a statement about the mesh. Nothing in this task builds it; it is
recorded as the shape any future fidelity metric has to take.

One thing this sweep does *not* establish: that trellis ignores the input
camera. It establishes that no single yaw offset recovers the alignment across
subjects, which is compatible with several mechanisms -- trellis choosing a
canonical orientation per subject, the "3/4 perspective view" in
`PROMPT_TEMPLATE` giving different subjects different apparent fronts, or
`normalize_glb`'s centring interacting with an off-centre reconstruction.
Distinguishing them would need a different experiment (the same subject at
several deliberately rotated references) and no decision in this plan depends on
the answer.
