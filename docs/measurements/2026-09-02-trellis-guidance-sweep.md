# Guidance strengths and the token budget against the holed subset — pre-registration, 2026-09-02

**Status: concluded 2026-09-02 — pre-registered, run and closed the same day; the grades that followed did not reopen it.** Ran after
[`2026-09-02-trellis-060-props.md`](2026-09-02-trellis-060-props.md) reports,
on whichever exe that document leaves in place.

## Why

`trellis-server.exe --help` lists three flags Warlock never passed:
`--gss F` and `--gsh F` (guidance strengths for the sparse-structure and
structured-latent stages) and `--max-tokens N` (the high-resolution token
budget, shipped at 49152). They act on exactly the stage the props-v1 corpus
is lost in ([`2026-08-30-sdxl-cfg-props.md`](2026-08-30-sdxl-cfg-props.md),
`holes` on 10 of 22). As of 2026-09-02 they are `Config.trellis_gss`,
`trellis_gsh` and `trellis_max_tokens`: `None` omits the flag, and the three
are `service.sweeps.SERVER_AXES` members, so a plan over them batches by
server restart the way `trellis_band` does.

**The exe does not print its defaults for these.** So the first rung of every
axis is *omitted* — the flag not passed — and the neighbours are relative to
the source's own values. Read from `include/trellis_args.h` at the v0.6.0 tag
on 2026-09-02: `gss = 7.5` (sparse-structure guidance), `gsh = 7.5`
(shape-SLAT guidance), `max_tokens = 49152`. So the rungs are 5.25 and 10.5
for each strength, and 98304 for the budget.

## What will be run

Subjects: the props-v1 prompts still above the 0.07 audit trigger on v0.6.0
([`2026-09-02-trellis-060-props.md`](2026-09-02-trellis-060-props.md)) —
**five**, not the ten the reviewer tagged on v0.5.4, because the exe bump
closed the other seven before this sweep ran: cauldron, pouch, lantern, well
head, dry branches. Seed 42, `mesh_profile=raw`, `trellis_tex_res=512`, band
auto. Submitted by `scripts/campaign_guidance.py`, tag `guidance-060`.

Plan, via `service.sweeps` (a sweep varies settings over one subject, so this
is five one-subject sweeps, one per prompt, from one `scripts/` submitter
following `_campaign.py`):

| axis | rungs |
|---|---|
| `trellis_gss` | omitted, default × 0.7, default × 1.4 |
| `trellis_gsh` | omitted, default × 0.7, default × 1.4 |
| `trellis_max_tokens` | omitted (49152), 98304 |

OFAT, not a full cross: 1 + 2 + 2 + 1 = 6 units per subject, 30 meshes,
roughly two hours at three to four minutes each.

Scoring, in this order:

1. `meshaudit.hole_fraction` worst-view, stored as `params["mesh_audit"]["worst"]`,
   tabulated by `scripts/hole_audit_vs_grade.py --tag <sweep tag>`. Machine
   evidence first, because it is free.
2. A blind grade on the −5..+5 scale **only if** the audit moves (rule below).
   [`2026-09-02-hole-audit-vs-grade.md`](2026-09-02-hole-audit-vs-grade.md)
   is what decides whether the audit is a proxy for the grade at all; if that
   document finds it is not, step 2 runs unconditionally.

## Decision rules

- A rung whose median `worst` over the five is below the omitted rung's median
  by more than 0.02 (the floor `trellis_band`'s sweep set for a real
  difference was ~0.3 % of faces; hole fraction has no such floor measured,
  so 0.02 is declared here and is the number to argue with) is graded. If it
  converts at least 2 of the 5 to usable (grade ≥ +3), it becomes the config
  default in `config.py`, citing this document.
- No rung clears 0.02: the flags stay `None`, remain sweep axes, and the
  result is recorded as a negative. The reroll question
  ([`2026-09-02-hole-audit-vs-grade.md`](2026-09-02-hole-audit-vs-grade.md))
  and the `close_holes` remesh question are the next instruments.
- `--max-tokens 98304` is also timed: if it costs more than 1.5× the omitted
  rung's wall clock it is not a candidate default even if it wins, and goes to
  `guidance.PLATFORMS` as an opt-in row instead.

## Results — machine audit, 2026-09-02

Tag `guidance-060`, 30 units queued, 28 finished. The two missing rows are
the pouch at `gsh = 5.25` (its reference failed the 2D gate — see the
reproducibility note) and the branches at `gss = 10.5` (the drain process
was killed mid-unit and the startup reconcile marked it failed; not re-run).
Worst-view hole fraction per unit; seconds are the whole job.

| subject | omitted | gss 5.25 | gss 10.5 | gsh 5.25 | gsh 10.5 | max_tok 98304 |
|---|---|---|---|---|---|---|
| cauldron | 0.222 | 0.226 | 0.230 | 0.223 | 0.222 | 0.222 |
| lantern | 0.245 | 0.235 | 0.233 | 0.242 | 0.235 | 0.245 |
| well head | 0.204 | 0.189 | 0.210 | 0.207 | 0.182 | 0.204 |
| pouch | 0.079 | 0.017 | 0.026 | — | 0.028 | 0.009 |
| branches | 0.367 | 0.018 | — | 0.351 | 0.096 | 0.076 |

**The three stable subjects do not move.** Cauldron, lantern and well head
sit within 0.02 of their omitted rung on every axis, and within 0.02 of
their `props-v1-060` corpus values (0.222 / 0.245 / 0.204) — the same
number twice, on separate runs. Neither guidance strength nor a doubled token
budget changes what the audit sees on them. `max_tokens = 98304` also
reproduces the omitted rung to three decimals on all three, which says the
HR pass never reached the 49152 ceiling for a 1024 prop; it costs nothing
and buys nothing here.

**The two that move are the two that do not reproduce.** The branches'
omitted rung measured 0.367 in this sweep and 0.215 in the corpus run, same
prompt, seed and exe; the pouch's reference passes or fails the 2D gate from
one identical submit to the next. Their spread across rungs (0.02–0.37) is
the same size as their spread across *identical* runs, so nothing in this
table can be attributed to a flag for either subject. Two repeat passes of
the five at the shipped defaults (tags `repro-a`, `repro-b`) put a number on
that variance, and located it:

| subject | corpus | repro-a | repro-b | `input.png` identical? |
|---|---|---|---|---|
| cauldron | 0.2217 | 0.2216 | 0.2217 | yes, all three |
| lantern | 0.2450 | 0.2451 | 0.2450 | yes, all three |
| well head | 0.2039 | 0.2038 | 0.2039 | yes, all three |
| pouch | 0.0912 | gate refused | 0.0642 | no — three different files |
| branches | 0.2148 | 0.1495 | 0.0694 | no — three different files |

**TRELLIS is deterministic to four decimals given the same image**, on
three of three subjects across three runs. The variance is entirely in the
image stage: for the pouch and the branches, SDXL at the same prompt and seed
produced a different reference PNG on each submit, and the gate's pass/fail
followed the picture. So the 2026-08-13 regression-check protocol's premise
("the 2D stage is byte-identical, so any delta is the reconstruction") is
true for some prompts and false for others on this machine, and a corpus
comparison has to check `input.png` hashes rather than assume them. Why SDXL
varies on those two prompts and not the other three is an open question for
the image pipeline (a non-deterministic kernel path is the usual suspect);
it is not this document's.

**Decision rule applied:** no rung clears 0.02 on the subjects where 0.02
means anything. The three flags stay `None`, remain sweep axes, and this is
recorded as a negative. The reroll and remesh questions are next.

**What the survivors have in common** is worth stating: a cauldron on three
legs, a lantern with open panes and a hanging loop, a well head whose roof
stands on posts, and a bundle of loose branches all have *real* enclosed
gaps in their silhouettes. The audit counts enclosed background inside the
silhouette, and on these it is very likely measuring topology the reference
actually shows rather than a reconstruction defect. That is a question for
the grading pass, and if it holds it is a finding about `meshaudit` rather
than about the exe: the audit trigger should not be used to reroll subjects
whose openings are intentional, and the 2026-08-30 reviewer's ten `holes`
tags may split into "perforated surface" (the seven v0.6.0 fixed) and
"open form" (these).

**After the grades, 2026-09-02.** Step 2's unconditional clause did not
trigger: [`2026-09-02-hole-audit-vs-grade.md`](2026-09-02-hole-audit-vs-grade.md)
found the audit *is* a proxy for the reviewer's `holes` tag on v0.5.4 (9 of
9 tagged above 0.07, 3 false positives), so the sweep rungs stay ungraded
and the negative stands on the audit alone. The split above is confirmed as
stated: the seven v0.6.0 fixed were the reviewer's `holes` (all graded −5 on
v0.5.4, +3 or better on v0.6.0 except two), and the five survivors carried no
`holes` tag on either binary and graded 2, 1, −3, 2, −4 on v0.6.0 — open forms
the reviewer rejects for shape, not skin. That is the `meshaudit` finding the
paragraph anticipated: the trigger should not reroll them, and it no longer
does, because `mesh_retries` stays 0.
