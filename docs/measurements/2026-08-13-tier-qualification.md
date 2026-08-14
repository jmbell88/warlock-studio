# Qualifying the gltfpack tiers, 2026-08-13

**Status: run taken 2026-08-13. The corpus gate fired: zero of twenty meshes
were accepted, no tier was qualified or even measured, and `PROFILES` stays
raw-only.** The pre-registration below is unchanged; the numbers are under
"Results". It was written in the style of
[`2026-08-10-zelda-props.md`](2026-08-10-zelda-props.md): the decision rules
were written before a single unit was queued, and the rule that fired is applied
verbatim — it is the boring one, and it was planned for.

## The question

**Do `draft` (20k), `standard` (50k) and `detailed` (100k) keep what a tier must
keep, on meshes a human chose to keep?**

The bar has stood since the binary was vendored on 2026-08-07 and is recorded in
`../INVARIANTS.md`: a tier stays out of the generate form until it has been run
against a chest, a sword and a rock and shown to keep UVs, both PBR maps and
material assignment. The machinery has been live the whole time — the tiers run
today from the retarget panel — but a named tier in the generate form is a
button that silently ships a worse mesh if it fails, which is why the default
and the form still say `raw`. What has never existed is the corpus: the rogue
sweep rejected 80 of its 83 meshes, and whether a tier *preserves* something
cannot be judged on output that is already broken. The root `TODO.md` (deleted
2026-08-08) first wrote this down; the bar moved into the invariants and into
`scripts/qualify_tiers.py`'s docstring, and this document is the run it has been
waiting for.

## What will be run

```
uv run python scripts/sweep_props.py --dry-run   # plan and validate
uv run python scripts/sweep_props.py             # 20 units, 4 sweeps
```

The corpus generator: four no-axis plans at the re-baseline's strongest
marginals (`sweep_props.WINNER` — `playground` + `render3d` at 0.9 + `birefnet`
+ `three_quarter`, chosen per
[`2026-08-09-rebaseline.md`](2026-08-09-rebaseline.md) Amendment 2 as "a base,
not a winner"), five seeds each (11, 23, 42, 77, 101), `stage="model"`. The
app's worker drains the rows; roughly an hour of card time at the re-baseline's
measured pace. The subjects are deliberately unlike each other in the way the
tier check cares about: the chest is multi-material and can lose a material
assignment, the sword is thin and is what a simplify pass mangles first, the
rock is a single untextured material and is the control.

Then the human half: every mesh is graded in Review mode on **mesh usability
only**, accept or reject. And then the harness:

```
uv run python scripts/qualify_tiers.py --out bench/tiers --sheets
```

With no subjects named it reads the verdict table and takes **every accepted
mesh's `source.glb`** — the immutable reconstruction, not `model.glb`, because
simplifying a simplified mesh measures the second pass, not the tier. It writes
only under `--out` and never touches the corpus.

## Decision rules, written in advance

**The corpus gate.** A named subject — chest, sword, rock — with zero accepted
meshes across its five seeds **blocks the qualification**. The run stops, this
document records the corpus failure, and `PROFILES` stays as it is. No
substitution: the knight (below) or a second accepted chest does not stand in
for a missing sword, because the bar names three shapes precisely for their
three different failure modes. A retry at different settings or seeds is a new
pre-registration, not an amendment here.

**The tier gate is `warlock.tiercheck`, referenced rather than restated.**
`compare()` is the rule: not-worse-than-source on UV coverage, on primitive
material assignment, on material count, and on base-colour, metallic-roughness
and normal textures, plus no added required extensions. A source that was
*already* missing something lands in `notes` beside the verdict rather than
failing it — the rock's poverty is stated, not mistaken for a loss. A tier
qualifies only as `Report.qualified` computes it: **a pass on every accepted
subject.** Every accepted mesh participates, the knight included if it is
accepted — a character mesh is exactly what a user will point a tier at, and it
can only raise the bar, never lower it.

**The eye can veto, never admit.** `--sheets` renders each subject before and
after through the one sheet pipeline. A tier that fails the automated checks is
out regardless of how its sheet looks. A tier that passes them may still be
refused if its sheet shows a mangled silhouette — that refusal is recorded here
with the sheet named, because "the silhouette survived" is not a claim any
number in the report earns.

**Consequence, pre-stated.** Tiers that qualify enter
`studio/panes/settings_3d.py`'s `PROFILES` and with it the generate form.
`Config.mesh_profile` stays `raw` either way — the default flip is a separate
decision, deliberately not taken in this round. Nothing stored is
reinterpreted: a job that ran at `raw` stays a `raw` job.

**The knight's asymmetry, declared before any mesh is looked at.** The fourth
plan is not a qualification subject; it exists so the rig-handedness question
has a mesh whose two sides are distinguishable in a still. Its ground truth is
fixed now, from the prompt and not from any render: **the single large pauldron
is on the subject's right shoulder; the left shoulder is bare; the satchel is
on the subject's left hip.** Writing this down after seeing the rig is how you
talk yourself into whichever answer appeared, so it is written down first. The
handedness reading itself is out of scope here and gets its own document.

## Results

Taken 2026-08-13. Submitted as four sweeps — `679479b5249a` (chest),
`a26a7fc7f5ac` (sword), `c3e6495fce7c` (rock), `b0eb38819246` (knight) — and
drained by the app's worker the same afternoon: 20 of 20 done, 0 errors. Review
was two passes, both human: the reference images first (18 accepts, 2 rejects —
the 2D half of the pipeline behaved), then the meshes, graded on mesh usability
only.

**Every mesh was rejected.** Per subject, grades by seed (11/23/42/77/101):

| subject | grades | best |
|---|---|---|
| chest | −5 −5 −5 −5 −5 | −5 |
| sword | −2 0 −2 −5 −3 | 0 |
| rock | −5 −5 +2 −5 −5 | +2 |
| knight | −1 −2 −3 −3 −5 | −1 |

The best mesh of the run — the rock at seed 42, +2 — sits under any accept bar
this repo has used. Chest, sword and rock each have zero accepts across their
five seeds, so **the corpus gate fires**: the qualification does not run,
`scripts/qualify_tiers.py` was never invoked, no tier moved, and
`Config.mesh_profile` and the generate form are untouched. Per the gate as
written, a retry at different settings or seeds is a new pre-registration, not
an amendment here.

**The anomaly is the finding.** These are the same `WINNER` settings at which
the re-baseline accepted 19 of 41 on 2026-08-09
([`2026-08-09-rebaseline.md`](2026-08-09-rebaseline.md), Amendment 2's gate
cleared at 46%). Zero of twenty is not seed luck against that base rate, and
the reference-stage acceptance (18 of 20) localizes the collapse to the mesh
half of the pipeline. Candidate explanations — harder subjects than the
re-baseline's, or a regression in the reconstruction path somewhere in the four
days between (the audit-remediation and four-track rounds both touched
pipeline-adjacent code) — are deliberately not adjudicated here. Diagnosing it
is its own run; qualifying tiers against whatever comes out of *that* is the
retry this document's gate demands.

**The corpus assets were not retained.** The library was cleaned after grading:
the 20 job rows and every GLB are gone. The verdict rows survive (they are
denormalized off the jobs table for exactly this), so the grades above remain
the record of the run — but there is nothing on disk to re-inspect, which any
future diagnosis run should plan around by retaining its evidence until it is
written up.

The knight was rejected with the rest, so the rig-handedness question it was
queued for still has no mesh; its pre-declared asymmetry above stands for
whatever run next produces one.
