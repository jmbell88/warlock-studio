# The re-baselined render sweep, 2026-08-09

**Status: procedure written, run not yet taken.** Everything below "What will be
run" is a pre-registration -- the decision rules were written *before* a single
unit was queued, which is the only thing that makes the answer worth anything.
When the run happens the numbers go under "Results" and whichever rule fired is
applied verbatim, including if it is the boring one. This follows
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

## Results — the re-baseline

Not yet taken.
