# The re-baselined render sweep, 2026-08-09

**Status: run taken 2026-08-09; both "Results" sections below are the outcome.**
Everything between "What will be run" and the first Results heading is the
pre-registration -- the decision rules were written *before* a single unit was
queued, which is the only thing that makes the answer worth anything. The numbers
went under "Results" and the rule that fired was applied verbatim, including
where it was the boring one: the confirm **FAILED** against the rule as written
(Amendment 1 records why the campaign proceeded anyway), the re-baseline gate
**PASSED**, every axis came back null, and `hole_worst` is no longer inverted.
This follows
[`2026-08-06-pixel-art-xl.md`](2026-08-06-pixel-art-xl.md), which is the
pre-registration exemplar in this directory.

**This document supersedes
[`2026-08-04-hole-rate-baseline.md`](2026-08-04-hole-rate-baseline.md).** That
document is not merely superseded on its numbers; its central metric has since
been shown to point the wrong way, and the section below says so before anything
else, so that nobody reads it forward.

## `hole_worst` is not weakly informative. It is backwards.

Over the 84-verdict review of the 2026-08-07 rogue sweep:

```
AUC(hole_worst predicts reject) = 0.115      (0.5 = a coin flip)
rejects with hole_worst EXACTLY 0.0:  48/81 = 59%
accepts with hole_worst exactly 0.0:   0/3
median hole_worst -- rejects 0.0000, accepts 0.0304
```

The accepted meshes have *more* measured holes than the median discarded one,
because the dominant failure mode is a solid slab and a slab has no visible
openings: `meshaudit` scores the worst outcome as perfect. Anywhere that reads a
low hole fraction as evidence of quality is wrong, and that includes reading it
as a sanity check *on* something else -- a probe that disagrees with `hole_worst`
is, on this evidence, more likely to be right than the metric is.

Two things follow and are already in the tree: `widgets.quality_badge` has no
green branch below `AUDIT_UNINFORMATIVE`, and the judge is trained on pixels
rather than on the audit scalars. **`hole_worst` may not rank anything in this
run.** The verdicts are human.

Separately, every **watertight** figure recorded before 2026-08-08 is void,
including the "0 of 83" once offered as a baseline: `meshreport` loaded with
`process=False`, so every xatlas UV-seam vertex split counted as a boundary
edge, and the figure was mostly measuring seams. It now welds by position before
judging. The triangle counts (177k-299k) stand.

## The question

**Does a baseline whose matte is the learned one produce acceptable output at a
workable rate?** And only if it does: which checkpoint, which style LoRA, and
which framing.

That ordering is the whole design, and it is the lesson of the run this one
replaces. Sweep B of the 2026-08-07 campaign spent 45 units varying four
depiction axes around a baseline pinned at `bg_removal=auto` -- the single
variable that dominated the verdict, held at its bad value -- and returned zero
accepts and every comparison a tie. That is a **floor effect, not a null
result**: one-factor-at-a-time around a baseline that fails ~96% of the time has
no headroom to detect an improvement in anything.

**The rule that follows, and that this run exists to obey: establish a baseline
that produces acceptable output at a workable rate before fanning out.**

## What will be run

```
uv run python scripts/sweep_confirm.py       # 10 units, blind -- FIRST
uv run python scripts/sweep_rebaseline.py    # 50 units
```

Both are headless submitters (`scripts/_campaign.py`); the app's worker drains
the rows. Roughly 22 minutes and 110 minutes of card time respectively, at the
2.2 min/unit the rogue sweep measured.

The confirm is `bg_removal` birefnet against auto, one knob, five seeds, and it
runs first because everything else leans on its answer. The re-baseline is
`base_model` x4, `style_lora` x4 and `framing` x1 over a birefnet baseline, at
the same five seeds, `category=character`.

**Review the confirm with Blind on.** `ReviewState.blind` renames every unit to
an id prefix *and* reorders them by a digest of the job id -- `sweeps.expand`
enqueues the baseline first, so position names the arm as plainly as a label
does. It defaults off and is never persisted. Forgetting it is unrecoverable.

## What the confirm is confirming, and why it needs confirming

The 2026-08-07 review produced exactly one signal in 84 verdicts: all three
accepts were `bg_removal=birefnet` and `auto` went 0 for 80. Three things make
it worth acting on rather than a curiosity at n=4:

- **The matched pairs are byte-identical upstream.** `input.png` hashes the same
  for `baseline s23` and `bg_removal=birefnet s23`, and for s42, s77 and s101 --
  the matte is applied at reconstruction time, so nothing about the picture
  differs. A controlled A/B, not a confounded marginal.
- **The failure mode changes, not just the rate.** 58 of 80 `auto` rejects are
  tagged `broken`; 0 of 4 birefnet are. A rate shift at n=4 is weak; a rate shift
  plus the disappearance of the dominant failure tag is a mechanism signature.
- **It is not review drift.** The accepts land at review positions 46, 48 and 83
  of 84, with 34 consecutive rejects between the second and the third.

What is nevertheless weak: **the clean 2x2 on its own is Fisher p=0.14.** The
p~4e-5 figure comes from using all 80 `auto` units as controls, which is
legitimate but leans on their comparability -- they varied checkpoint, LoRA and
four depiction axes. And the review was unblinded and single-reviewer. Ten units
with one knob, judged blind, lean on none of that.

## Decision rules, written in advance

**On the confirm.** Birefnet must take **at least 4 of the 5 matched pairs**,
*and* no birefnet reject may carry the `broken` tag. 3-2 or worse means the
2026-08-07 finding did not replicate, the re-baseline is not run as written, and
the campaign is re-planned around `auto`. This rule is checked before the
re-baseline is submitted, not after it drains.

**The go/no-go on the re-baseline: at least 12 accepts of 50 (25%).** Written
here before any label is filed, because a threshold chosen after the numbers are
in view is a threshold chosen to let the run count.

- **>= 12** -- the baseline is workable. Proceed to the depiction axes, and the
  tier-qualification corpus (`scripts/sweep_props.py`) is generated at these
  settings.
- **10-25%** -- harvest the accepts, and do **not** fan out to the depiction
  axes. There is not enough headroom for one-factor-at-a-time to detect anything,
  which is exactly the Sweep B failure.
- **< 10%** -- stop sweeping. The problem is upstream of every axis in this run,
  and no contrast measured through it means anything.

**On the per-axis contrasts.** Each is 5 matched pairs at 5 seeds.
**No checkpoint, style LoRA or framing is named a winner below 5-0** (one-sided
sign test, p=0.031); 4-1 is p=0.19 and is null, as is anything below it. This is
`sweep.MEANINGFUL_MARGIN`'s doctrine transposed to a count -- the number 0.20
does not transfer, because it is calibrated for a continuous metric measured once
per cell, but the refusal to name a winner inside the noise does.
`bench.findings.comparison_lines` already declines to print a contrast below
`min_pairs=5`, so every axis here sits exactly on its floor. **This run is
powered to detect a large effect and nothing smaller. Its deliverable is the
corpus; the axis findings are a bonus.**

**Refusal is reported separately from acceptance and never summed into one
score.** The rogue sweep is the reason: `playground` and `redmond3d` refused 0/5
each while `sdxl_cfg` refused 3/5 with flawless survivors, and `render3d` and
`pixelxl` passed the gate more often and produced the two worst meshes. Neither
number alone picks a checkpoint, and adding them picks the wrong one. Refusal
rates read off `findings.json`'s `refused_<code>` metrics, which now exist
because a terminal `error` writes an observation too -- the 17 refusals of the
rogue sweep wrote nothing at all, so a reader saw each checkpoint's accept rate
among the references that survived, which flatters exactly the checkpoints that
fail most often.

**On `hole_worst`.** AUC is recomputed against the new human labels and
reported. Only **AUC >= 0.65** would restore it as evidence of anything, and even
then `widgets.quality_badge`'s missing green branch is a separate decision owing
its own document. It is not used to rank any arm of this run under any outcome.

**On `Config.mesh_hole_max`.** It sits at 0.07 because
`2026-08-04-hole-rate-baseline.md` found a sharply bimodal distribution with a
completely empty gap from 0.0308 to 0.1010 and took the midpoint. It moves only
if the new corpus reproduces a bimodal distribution with an empty gap, in which
case it moves to that gap's midpoint. **If the distribution is unimodal, the
number is retired as a retry trigger rather than re-tuned** -- a threshold in the
middle of one population is false precision, and `Config.mesh_retries` is 0
either way.

**On comparability with the 2026-08-07 corpus.** `PROMPT_TEMPLATE` moved
(`PROMPT_VERSION` 3 -> 4) between the two runs, so a unit here is not comparable
with one from that run on the prompt axis either. Every change landed before this
run, which is the right order -- a sweep around a broken base measures the
brokenness -- but this is the first corpus in which any of them is measured, and
no cross-run comparison is drawn.

## Results — the confirm, 2026-08-09

Sweep `bebed19f8b39`, 10 units, **0 errors**. 08:09:53 to 08:42:21 local, 32.5
min of card time, 3.25 min/unit — half again the 2.2 min/unit the rogue sweep
measured, which is `sdxl` at full CFG rather than turbo. Reviewed 08:13 to 08:43.

| Seed | birefnet | auto | Pair |
| --- | --- | --- | --- |
| 11 | accept | reject `broken` | **birefnet** |
| 23 | reject `bad-shape` | reject `broken` | tie |
| 42 | reject `bad-shape` | reject `broken` | tie |
| 77 | accept | reject `broken` | **birefnet** |
| 101 | reject `bad-shape` | reject `bad-texture` | tie |

**birefnet 2, auto 0, 3 ties.**

### Verdict against the rule as written: FAILED

The rule required **at least 4 of the 5 matched pairs**. Birefnet took 2. A sign
test with ties discarded is n=2, p=0.250; Fisher on the accept rate (2/5 vs 0/5)
is p=0.222. Neither approaches the bar. **The strong form of the 2026-08-07
finding did not replicate.**

The rule's second clause passed, and decisively: **no birefnet reject carries the
`broken` tag — 0 of 5, against 4 of 5 for auto (Fisher p=0.024).** Every birefnet
failure is `bad-shape`. The mechanism signature that made the original finding
worth acting on — the dominant failure tag *disappearing* rather than thinning —
is exactly what reproduced.

### Amendment 1, 2026-08-09: the campaign proceeds anyway

Recorded as an amendment rather than by editing the rule above, so a later reader
sees both what was promised and what was done.

**The rule's stated remedy is inapplicable.** It said a failure means the
campaign "is re-planned around `auto`". Pooling the whole surviving corpus — the
`verdicts.vector` column outlives the pruned jobs, which is what it is for:

```
bg_removal   accept  reject    n   rate
  auto            0      90   90    0%
  birefnet        7       9   16   44%
```

**auto is 0 accepts in 90 units across two runs** (Fisher against birefnet,
p=4.7e-7, confounded — the auto units varied checkpoint, LoRA and four depiction
axes). There is no baseline to re-plan around. A rule whose failure branch names
an option that does not exist has not been followed by taking it.

**The instrument was mis-specified, and that is the honest reading.** A 5-pair
sign test can only reach significance at 5-0, so it requires the two arms to
*disagree* on nearly every pair. When both arms fail often — birefnet's own rate
is 40% here — ties are structurally guaranteed and the test has almost no power
against any effect short of total. The bar was chosen for a subtle difference;
the actual effect is a wipeout that shows up in the marginal and in the
failure-mode shift, and is invisible to pairs. **This is a flaw in the rule, not
evidence about the matte**, and it is written down here so the same instrument is
not reached for again in the same situation.

**What is *not* claimed.** The confirm did not establish birefnet at the strength
asked for, and this document does not pretend it did. What it establishes is
weaker and sufficient for the next step: birefnet is not worse, it removes the
dominant failure mode, and auto is unusable.

**Decision: `scripts/sweep_rebaseline.py` runs as written.** The birefnet arm's
2/5 is 40% (95% Wilson CI 12–77%) against this run's own **>= 12/50** gate, so
the 50-unit run is both the better-powered instrument and the real decision
point. The gate below is unchanged and is not amended.

## Results — the re-baseline, 2026-08-09

Sweep `5f26623d12aa`, 50 units submitted. 09:00 to 10:51 local, **1.84 h**, mean
2.54 min per finished unit.

```
done        41     all 41 reviewed
error        4     2 composition-gate refusals, 2 "interrupted by shutdown"
cancelled    5
```

### The gate: PASSED

**19 accepts, 22 rejects.** The pre-registered bar was **>= 12 of 50**. Against
the 41 units that produced a mesh that is **46%**; against all 50 submitted,
38%. Either denominator clears the 25% the bar was written as.

The stock of accepted meshes with files on disk went from **1 to 20**. That was
the campaign's actual objective, and it is met.

### Every axis is null, and the bar was unsatisfiable

Matched pairs against the baseline at the same seed:

| Arm | win | lose | tie | n/a | Accept rate |
| --- | --- | --- | --- | --- | --- |
| `base_model=playground` | 2 | 1 | 1 | 1 | 4/5 · 80% |
| `base_model=sdxl_cfg` | 1 | 0 | 1 | 3 | 3/3 · 100% |
| `style_lora=render3d` | 1 | 0 | 2 | 2 | 3/4 · 75% |
| `style_lora=redmond3d` | 2 | 1 | 1 | 1 | 3/5 · 60% |
| **baseline** (sdxl, no LoRA) | — | — | — | — | 2/4 · 50% |
| `style_lora=ps1` | 1 | 1 | 2 | 1 | 2/5 · 40% |
| `base_model=turbo` | 0 | 1 | 3 | 1 | 1/5 · 20% |
| `framing=front_ortho` | 1 | 2 | 1 | 1 | 1/5 · 20% |
| `style_lora=pixelxl` | 0 | 2 | 2 | 1 | 0/5 · 0%, all `broken` |
| `base_model=pixel` | — | — | — | 5 | no data |

Nothing reached 5-0. **Nothing could have**: `baseline s23` was refused at the
composition gate, so only **four** matched pairs ever existed, and the bar was
written as five. Every axis in this run was null before a single unit ran.

That is the second mis-specification of the same rule in one campaign — the
confirm's was that ties are structural when both arms fail often, this one is
that the bar assumed a baseline unit which is not guaranteed to exist. **Both
have one root cause: a sign test sized as though ties and losses were rare.**
Recorded here rather than repaired, because repairing a rule after seeing the
data it judged is the thing pre-registration forbids. A future OFAT run should
state its bar as a fraction of *usable* pairs, decided before the run and
evaluated after the losses are known.

The marginal accept rates in the right-hand column are **confounded and
underpowered** and no winner is claimed from them. They are reported because
they are the only evidence available for choosing settings for a corpus run, and
the choice is a practical one rather than a claim.

Two arms are informative despite the null. `style_lora=pixelxl` took 0 of 5 with
**every** rejection tagged `broken`, which is the prior that flat pixel art is
poor trellis input, confirmed at the only strength this design can offer. And
`base_model=turbo` at 1/5 is the weakest checkpoint that produced data.

### Refusals, reported separately

**2 of 50**, both `multi_object`, against 17 of 100 in the 2026-08-07 sweep. Both
fell on SDXL-family arms at full CFG (`baseline`, `base_model=sdxl_cfg`). n=2, so
this is a count and not a rate about any checkpoint.

### `hole_worst` is no longer inverted, and the matte is why

```
AUC(hole_worst predicts reject) = 0.756       (2026-08-07 corpus: 0.115)
median hole_worst — accepts 0.0141, rejects 0.0293   (was 0.0304 / 0.0000)
meshes at exactly 0.000: 1 of 41                     (was 48 of 81)
```

The pre-registered rule was that **only AUC >= 0.65 restores it as evidence**. It
clears, and the mechanism is legible rather than merely statistical: **the
inversion was an artefact of the `auto` matte, not a property of the metric.**
`auto` produced solid slabs, a slab has no visible openings, and `meshaudit`
therefore scored the dominant failure mode as perfect. Under birefnet the slab is
gone — one mesh at exactly zero against forty-eight — and with it gone the metric
ranks the right way round. This is the same mechanism the confirm sweep measured
through the `broken` tag, arrived at independently.

**What this does not license.** `widgets.quality_badge`'s missing green branch is
a separate decision, deferred by this document's own rule to its own measurement,
and it is **not** taken here. n=41 on one subject and one prompt is not grounds
for a UI that tells users a low hole count means a good mesh — the finding is
that the metric is informative *on a corpus whose matte is birefnet*, which is a
narrower claim than the badge would make.

### `mesh_hole_max`: retired as a retry trigger

The distribution is 0.000–0.052, **unimodal**, largest internal gap 0.007, and
**0 of 41 exceed 0.07**. The rule said the number moves only if a bimodal
distribution with an empty gap reproduces, and that **a unimodal distribution
retires it rather than re-tunes it**. It is retired. `Config.mesh_retries` is 0,
so nothing changes operationally and the constant is left in place with this
document as the reason it is not used.

### Watertight, re-measured

**2 of 41.** The first figure taken since `meshreport` began welding by position;
every watertight number before 2026-08-08 is void and none is carried forward.
Triangles 273,888 – 299,408 (median 288,440), unmoved by any axis, consistent
with the 177k–299k of the old run.

### Amendment 2, 2026-08-09: what the prop corpus is generated at

`scripts/sweep_props.py` needs a configuration and this run declined to name one.
The corpus does not need the *optimal* configuration — it needs one that produces
keepable meshes on three new subjects — so `WINNER` is set to the strongest
marginals, **`base_model=playground` + `style_lora=render3d`**, and the script
records that this is a marginal rather than a winner. No claim about either is
entered into any finding.

Seven units produced no terminal measurement (the whole `base_model=pixel` arm,
`base_model=sdxl_cfg s101`, `style_lora=render3d s11` — five cancelled, two lost
to an app shutdown). They are re-queued into this same sweep by
`scripts/sweep_refill.py`, so the pairs they belong to can still form. The two
genuine composition-gate refusals are **not** re-run: a refusal is a measurement.
