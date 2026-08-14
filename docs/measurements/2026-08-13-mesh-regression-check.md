# Is the mesh collapse a code regression? 2026-08-13

**Status: run taken 2026-08-13. The 0-1 rule fired as written — and the deeper
replay it triggered refutes that rule's own premise: there is no code
regression.** The 2D stage is byte-identical across the whole window at a fixed
seed, every layer around the reconstruction is unchanged, and the residual is
subject-intrinsic trellis failures plus the grading instrument itself. The
pre-registration below is unchanged; the evidence is under "Results". Written
following [`2026-08-13-tier-qualification.md`](2026-08-13-tier-qualification.md),
whose Results section is what this run exists to explain.

## The question

The tier-qualification corpus went 0 of 20 on mesh usability at settings
composed from a re-baseline that accepted 19 of 41 four days earlier
([`2026-08-09-rebaseline.md`](2026-08-09-rebaseline.md)). Whether that is a
code regression is unanswerable from the stored verdicts, because no subject
was ever run on both sides of the gap: the re-baseline's 46% was measured on
**one subject** (the SNES rogue adventurer, a `character`), and the two
campaigns since ran other subjects at other configurations. The same-day
evidence points at subject: the Zelda pot took 6/30 while the crate took 0/30
on identical code, and the tier chest went five straight −5. But the tier rock
— an organic control that should be trellis-friendly — also went 0/5, which
subject difficulty does not obviously explain. The dominant reject reason
everywhere is `holes`.

What is already excluded: the vendored `trellis-server.exe` and its weights are
unchanged since 2026-07-30; the mesh-stage pipeline diffs from 2026-08-09 to
HEAD contain lifecycle and admission work but no reconstruction-parameter
change; and `lora_fits(playground, render3d)` is True (both `FAMILY_SDXL`), so
the composed WINNER pair was not silently dropped.

## What will be run

One sweep script (`scripts/sweep_regression_check.py`), three plans, eight
units, all at `bg_removal=birefnet`, `framing=three_quarter`,
`reference_prep=True`, `platform=3d`, `stage=model`:

- **R1, re-baseline baseline replica** — `sweep_rogue.PROMPT` and `COMMON`
  verbatim, `base_model=sdxl`, no LoRA, seeds 11/42/77. Documented rate: 2 of 4
  accepts (seed 23 was refused at the composition gate then; it is left out
  here so every unit is expected to draw).
- **R2, playground-arm replica** — the same, `base_model=playground`.
  Documented rate: 4 of 5 accepts, the strongest arm the re-baseline measured.
- **T1, tier rock reproduction** — the tier corpus rock unit verbatim
  (`sweep_props.SUBJECTS` rock prompt and fields, `WINNER` config), seeds
  11/42. Today's rate: 0 of 5, four at −5.

Review is blind, mesh usability only, the usual −5..+5 scale — the same eyes
and the same bar as every corpus this compares against. **The library must not
be cleaned until this document's Results are written**: the tier run's
evidence was deleted before anything could be learned from it, and this run
exists partly to regenerate it.

## Decision rules, written in advance

Counting R1+R2 accepts out of 6 against the documented 6 of 9 on the same
subject at the same configurations:

- **0 or 1 of 6** — regression confirmed. The rogue subject collapsed with the
  code as the only moved variable. Next step is a bisect over 2026-08-09..13
  on this same replica, and nothing about tiers or subjects is concluded.
- **3 or more of 6** — no regression worth the name. The pipeline still
  produces on 2026-08-13 what it produced on 2026-08-09, and the tier corpus
  failed on subject and configuration grounds; the tier document's candidate
  list narrows accordingly, and the follow-up is subject-shaped (why boxes and
  this rock fail), not git-shaped.
- **Exactly 2 of 6** — ambiguous, extend R1 and R2 by seeds 23/101 each and
  re-apply the same rule on all 10 before reading anything into T1.

T1 carries no gate: it is the reproduction that supplies artifacts.
Whatever branch fires, its two meshes and their `meshreport`/`mesh_audit`
numbers are inspected and described here, because `holes` as a human reason
currently has no surviving mesh behind it anywhere.

A grade is not an axis reading: R2 outscoring R1 or vice versa is noted, not
acted on — n=3 per arm decides only the regression question, which is a
difference of kind (46%-class vs 0-of-20-class), not of degree.

## Results

Taken 2026-08-13. Submitted as sweeps `7e12b6731c82` (R1), `c01364a15e49`
(R2), `a175c62a298c` (T1); 8 of 8 drained, 0 errors, graded blind the same
evening. Evidence retained — the job rows and every artifact are still on disk.

### The grades

| arm | s11 | s42 | s77 | accepts |
|---|---|---|---|---|
| R1 rogue/sdxl | −4 | −1 | −2 | 0/3 |
| R2 rogue/playground | **+1** | 0 | −3 (`bad-texture`) | 0/3 |
| T1 tier rock | −5 (`holes`) | −5 (`holes`) | — | 0/2 |

R1+R2 = **0 accepts of 6**: the pre-registered 0-1 branch — "regression
confirmed, next step is a bisect." The bisect was replaced by something
strictly stronger, and what it found is below.

### The replay: the pipeline did not move

The 2D stage is seed-deterministic, so instead of bisecting by eye, the R1-s11
unit's reference generation was replayed through both trees — HEAD and a
worktree at `26d259b`, the commit the re-baseline drained under — with one
driver mirroring the worker's exact call. `uv.lock` differs between those
commits **only** in the project's own version string (same torch 2.11.0+cu128,
diffusers 0.39.0, transformers 5.14.1, peft 0.20.0). Result: **byte-identical
images, sha256 `faada1c3…` from both trees**, and byte-identical composed
prompts. Combined with what was excluded up front (server binary and weights
untouched since 2026-07-30; `ensure_config` defaults unchanged;
`reference.prepare` untouched in the window) and one more check —
`meshaudit.hole_fraction` is identical on `source.glb` and `model.glb` for
every unit here, so the optimize→normalize→grounding chain preserved the
reconstruction bit-for-what-matters — every layer between the prompt and the
graded mesh is accounted for. **For the same inputs, 2026-08-13 produces what
2026-08-09 produced.**

### What actually explains the three collapses

- **The rock (and by extension the chest, and the Zelda crate): the subject.**
  The rendered sheet of `rock s42` shows the failure plainly: trellis
  reconstructed the moss and foliage as a hollow vegetation lattice and the
  stone underneath does not exist — 29-40% see-through measured, `holes` by
  eye, −5 by any bar. An "overgrown mossy" prompt begs for exactly this. The
  composition gate even resisted: both rock units failed twice with "runs off
  the edge of the frame" before a tail draw scraped through, so the corpus
  meshed from the worst part of its own draw distribution.
- **The rogue replicas: the instrument, not the pipeline.** Tonight's rogue
  meshes sit *inside* the re-baseline's accepted population on every stored
  number — `hole_worst` 0.004-0.047 against the re-baseline's accept median
  0.0141 / reject median 0.0293 — and their rendered sheets read as clean,
  complete characters. One graded **+1 and was rejected**. The re-baseline's
  verdicts were binary accept/reject; the −5..+5 usability scale arrived with
  the Zelda review on 2026-08-10, and under it accept-class has meant ≥+3.
  A mesh a binary instrument called "accept" and a graded instrument calls
  "+1, reject" is one mesh and two rulers. n=6 keeps luck alive as a
  contributor — the reference draw is random whenever the pinned seed fails
  the gate (see below) — but no mechanism is left for the pipeline to have
  produced worse meshes, and the sheets say it did not.

### A finding about every sweep, discovered in passing

`reference_retries` defaults to 2 in both eras, and a pinned seed whose draw
fails the composition gate **rerolls at a random seed** — `reference_seed`
1554797768 on tonight's R1-s11, whose seed-11 draw (byte-identical in both
eras, so this was equally true on 2026-08-09) fails the gate with "more than
one object". Consequence: whenever a unit's pinned seed fails the gate, the
reference that actually meshes is a random draw, the unit is not reproducible
from its spec, and a matched pair whose two arms rerolled differently is
comparing two random pictures. Nothing is broken — the behaviour is deliberate
and documented in `config.reference_retries` — but sweep readings have been
assuming seed-pinned references that are not always seed-pinned, and
`reference_attempts` in params is the honest record of which units are which.

### Consequence

No bisect, no revert, nothing to fix in the pipeline. The tier document's
candidate list resolves to "subject and configuration, plus the instrument":
its retry should choose trellis-friendly subject framings (a bare rock, not an
overgrown one; the chest is expected to stay hard — flat panels are the
failure mode the Zelda crate already measured at 0/30) and should pre-register
its accept bar against the graded scale rather than against the binary-era
46%. A latent crash in `scripts/qualify_tiers.py --sheets` (it handed `pack` a
directory where a frame map is required — the flag had never been run) was
fixed in this change.
