# The graded prop corpus at the shipped default — 2026-08-30

**Status: run taken 2026-08-30. The middle decision rule fired, by one grade.**
Results for Q1 (P3) below; Q2 (P12) is graded but its verdict is not derivable
from these numbers and is still owed. The protocol, corpus, classes and every
decision rule were fixed in advance in
[`2026-08-30-art-verdicts-preregistration.md`](2026-08-30-art-verdicts-preregistration.md);
nothing below was chosen after seeing a grade.

This is the retry [`2026-08-13-tier-qualification.md`](2026-08-13-tier-qualification.md)
demanded — a new pre-registration, subjects re-framed to be trellis-friendly,
the bar declared against the graded scale.

## What was run

22 subjects, one seed (42), `text → sdxl_cfg → TRELLIS` at the shipped defaults
with `Config.mesh_profile = "raw"`. Submitted by `scripts/campaign_props.py`
from `docs/measurements/corpora/props-v1.txt`, unedited. **1 h 58 m wall clock**
for the 22 jobs — about 5.3 minutes each, against the ~2 min/unit the plan
estimated from a model-stage sweep unit; the reference stage is the difference.
Graded by one human on the −5..+5 scale, blind, cut at
`vectors.USABLE_GRADE = 3`.

## Results

| class | usable | rate | mean | grades |
|---|---|---|---|---|
| `easy` | 1/8 | 12.5% | −3.38 | −5 −5 −5 −5 −4 −4 −3 **+4** |
| `medium` | 3/8 | 37.5% | −1.12 | −5 −5 −5 −5 0 **+3 +3 +5** |
| `hard` | 0/3 | 0.0% | −0.67 | −5 +1 +2 |
| `humanoid` | 1/3 | 33.3% | +1.67 | +1 +1 **+3** |
| **easy + medium** | **4/16** | **25.0%** | — | the decision set |
| whole corpus | 5/22 | 22.7% | — | — |

The five usable meshes: a brass hand bell (+5), a round loaf of dark bread (+4),
a blacksmith's anvil (+3), a hanging oil lantern (+3), a dwarf blacksmith (+3).

## The decision rule that fired

**25.0% on the 16 `easy` + `medium` subjects → the middle rule**: *"the README
states the figure and calls generated 3D a draft path. No 'game-ready' claim."*

Applied verbatim, and **it fired by a single grade**. Had any one of the three
`+3`s been a `+2`, the set would read 3/16 = 18.8% and the bottom rule — *no
positive quality claim at all* — would have fired instead. That fragility is
part of the result and is not to be smoothed over in the README: the honest
sentence is that roughly a quarter of prop subjects reconstruct usably at the
shipped default, on a corpus of 16, with a boundary this close.

Nothing here moves `Config.mesh_profile`, and nothing here qualifies a gltfpack
tier.

## The control behaved, so the comparison holds

`a wooden treasure chest with iron banding` scored **−5**, against −5 at all
five seeds on 2026-08-13 and 0/30 on the Zelda crate. The pre-committed reading
of that outcome is *"the corpus is calibrated against the earlier run, and the
`easy`/`medium` classes carry the finding"* — so it does.

**This run produced five usable meshes where 2026-08-13 produced none.** That is
real but it is not a controlled improvement: the corpora differ by design (props
rather than the hard set), which was the whole point of re-framing. The chest is
the only directly comparable subject and it did not move at all.

## The prediction inverted, and that is the most useful thing here

The difficulty classes were a pre-registered prediction. They came out
**backwards**:

    predicted   easy  >  medium  >  hard
    measured    hard(−0.67) > medium(−1.12) > easy(−3.38)

`easy` was the *worst* class by mean and by rate. Seven of its eight subjects
scored −3 or below. Because the classes were fixed before the run, this is a
result rather than a rationalisation: **"compact, solid, rounded, closed
silhouette" does not predict reconstruction quality, and may be anti-predictive.**

Any future corpus that inherits these class definitions inherits a predictor
this run falsified.

## Where the failure lives

The reference stage was labelled as well, and it localises the collapse
precisely:

| | |
|---|---|
| reference (2D) accepted | **21/22 = 95%** |
| mesh usable | 5/22 = 23% |
| good reference → unusable mesh | **16 of 21** |

Exactly 2026-08-13's shape (18/20 references, 0/20 meshes) and the same
conclusion: **the 2D half is working and the reconstruction half is where the
corpus is lost.** Only one subject failed at the reference stage (the armoured
knight), so prompt quality, framing and the checkpoint are not what this corpus
is measuring.

The reviewer's tags say the same thing in another vocabulary, over 22 meshes:

    holes 10    bad-texture 5    broken 3    clean-shape 1    on-style 1

`holes` on nearly half the corpus is the dominant recorded defect.

## Hypotheses — post-hoc, recorded so a future run can pre-register them

Explicitly **not** findings. Each is a candidate question, not an answer:

1. **Smooth uniform surfaces may reconstruct worse than structured ones.** Every
   `easy` failure is a rounded, evenly-textured object (jug, barrel, pumpkin,
   skull, amphora, rock, cauldron); the successes carry hard edges and internal
   structure (bell, anvil, lantern, bread). This is the shape of the inverted
   prediction and would explain it, but it was read off the results.
2. **`holes` may be addressable without touching the model.** `op_remesh`'s
   voxel pre-pass (`close_holes`, `VOXEL_FRACTION`) exists for exactly the
   plate-crust gaps `meshaudit` counts. Whether it converts any of these ten
   into usable meshes is a measurable question and a cheap one — and note the
   remesh path only started producing quads at all on 2026-08-30, so no
   conclusion about it predates that fix.
3. **Single-view is the obvious suspect** for the hollow/holed failures, which
   is what the multi-view backend exists for. Untested here.

## What this does and does not unblock

**Does:** the release audit's "no positive quality evidence" item now has an
answer, in the middle of its range. The README can state a figure with the
draft-path caveat.

**Does not: the tier qualification's corpus gate still fires.**
`scripts/qualify_tiers.py` wants accepted meshes that are recognisably *a chest,
a sword and a rock*. This run accepted a bell, a loaf, an anvil, a lantern and a
dwarf — no chest (−5), no rock (−4), no sword in the corpus at all. Tier
qualification remains blocked, on subjects rather than on volume.

## Q2 (P12) — answered: the generated-character path is not viable

The three `humanoid` subjects scored +3, +1, +1 as *meshes* — the best mean of
any class — but that was never P12's instrument. Against its own rubric, limb
separation and silhouette judged by eye, the verdict is:

> **Limbs are bent and stretched.**

**The bottom rule fires:** *"0 or 1 of 3 → the generated-character path is not
viable at the shipped default, and Troupe's supplied-base-mesh path (P4) is the
only one worth investing in."*

The exact count does not need settling. The top rule needs *2 or 3 of 3* showing
separable limbs and a readable silhouette; a set described as bent and stretched
is not that at either 0 or 1, so the same rule fires either way.

**The distortion bug is ruled out as a confound.** `_sample_sdxl` was passing a
square frame to pipelines that resize the init image and mask to it, which
stretched non-square selections — and "stretched" is the word here. It does not
apply: these jobs carry no `init_image`, no `ip_adapter` and no `control`
(checked in their stored params; `resolution` 1024, `bg_removal` birefnet, and
nothing else). `_init_frame` only moves the frame when `uses_init`, so the code
path was never entered. The geometry is the reconstruction's own.

**This is the same failure the corpus already localised**, on a harder subject.
16 of 21 good references became unusable meshes and `holes` was the dominant
tag; a humanoid asks the reconstruction for separable limbs, which is more than
any prop in the corpus asked for, and it did not deliver them.

### What it decides

**Troupe Phase 7 is not worth planning on the generated-character path**, which
is exactly the decision P12 existed to take *before* the investment rather than
after. The supplied-base-mesh path is the one to build on — and it became
runnable the same day, which is the useful convergence here: a textured rigged
humanoid landed, three silent defects in the rig path were found and fixed by
putting it through, and P5 is now the live question.

This says nothing about whether a *better* reconstruction could carry
characters. It is a verdict on the shipped single-view default, which is what
was asked.

## Caveats

One reviewer, one seed, 16 subjects in the decision set. The single-reviewer
caveat is [`2026-08-09-rebaseline.md`](2026-08-09-rebaseline.md)'s and is
repeated rather than assumed read. Seed variance is not measured; a replicate
run is a new pre-registration.

**The corpus assets are retained**, tagged `props-v1`, per this programme's own
rule and 2026-08-13's instruction. Do not clean the library.
