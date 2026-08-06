# Review

Review is where finished meshes get judged, and where those judgements start paying you back. Every
accept or reject is recorded as a **verdict** against the job it was filed on, together with a copy
of the settings that produced it — and the pool of verdicts is aggregated into the "What works"
findings that the generate panes surface as hints, ranked configurations you can apply to the forms
or save as presets. Ten seconds of judging a mesh teaches the app which settings earn their place.

There are two ways to file a verdict, feeding one pool:

- **In Review**, against a sweep's units or the recent-unreviewed bucket — built for judging many
  meshes quickly, with the keyboard.
- **In the 3D inspector**, against whatever asset is selected — the Accept / Reject buttons under
  the mesh quality section. One-off verdicts during ordinary use count exactly as much.

## The workspace

Review uses the same three-column skeleton as the generate modes:

- **Left**: the sweep list — a "Recent, unreviewed" bucket first, then every sweep, newest first —
  and below it the form that launches a new sweep.
- **Centre**: the shared asset viewer, showing the unit under judgement. Orbit, wireframe and
  framing work as they do in 3D.
- **Right**: the verdict panel — the unit's reference image, its mesh measurements, and the
  Accept / Reject / Skip controls — with the findings underneath.

"Recent, unreviewed" collects finished meshes from ordinary use that nobody has judged yet. It is
not a sweep: it has no settings axes and cannot be deleted, it simply empties as you file verdicts.

## Judging

`A` accepts the unit on screen. `R` arms a reject, and the next `1`–`5` picks the reason — holes,
bad shape, bad texture, wrong style, broken — because a reject without a reason teaches nothing.
`Esc` cancels an armed reject. `S` skips to the next unverdicted unit, and `Left` / `Right` step
through the list. The same controls exist as buttons in the verdict panel.

Verdicts are append-only and the latest one per job wins, so changing your mind is filing again,
not editing. And each verdict carries its own copy of the settings it judged — the learning corpus
survives the assets it was learned from, so pruning old jobs costs no knowledge.

## Sweeps

A sweep turns "which setting is better" from a hunch into rows you can judge side by side. The form
at the bottom of the sweep list takes:

- **Prompt and seeds** — one subject, one or more seeds (comma-separated). Every unit shares them,
  so the only thing that varies is what you asked to vary.
- **Axes** — a parameter name and a comma-separated list of values. The sweep plans a baseline unit
  plus one unit per differing value, per seed.
- **Start from current settings** — captures the 2D and 3D forms as the baseline the axes vary
  from. A sweep off an unstated baseline is not reproducible.

Because every unit shares the prompt and the seeds, the baseline and each axis unit form
**matched pairs** — same subject, same seed, one setting differing — which is what the axis
verdicts below are computed from. More seeds means more pairs per axis value.

Every unit goes through the same front door as an ordinary job — the same validation, the same
VRAM admission — and admission is all-or-nothing: one bad unit refuses the whole sweep, naming
itself, before anything is queued. Sweep units are hidden from the library so a launched sweep
does not bury your real assets; Review is where they live.

Deleting a sweep deletes its jobs and meshes but keeps every verdict filed on them — what the
sweep taught outlives what it built.

## What works

The findings section at the bottom of the verdict panel gives two kinds of answer.

**Axis verdicts** are the conclusive kind: matched pairs recovered from sweeps. Two units pair up
when they share a sweep, a seed and a prompt and differ in exactly one setting, so a line like
"lora_weight: 0.6 beat 0.9 in 7 of 8 matched pairs (2 sweeps, 2 prompts)" is an all-else-equal
comparison, not a correlation. Machine measurements pair the same way: every finished mesh is
measured automatically (worst-hole fraction, watertightness, triangle count), so a sweep shows
"worst-hole -4.1% over 12 paired runs" the moment it finishes, before a single verdict is filed.

**Ranked configurations** are whole settings vectors ordered by a conservative floor on their
accept rate, shown as "80% of 20 (61%+)": the first number is what happened, the parenthesised one
is the floor the evidence supports — which is what stops a lucky 5-for-5 from outranking a solid
19-of-20. A configuration needs five verdicts to appear, and carries a muted line of its machine
measurements when it has any. "Apply to forms" writes one into the 2D and 3D forms; "Save as
preset..." keeps it under a name.

The same findings feed the small hints next to controls in the generate panes, so the learning is
visible where the decisions are made: "accept 6/8 (41%+)" once a value has enough verdicts behind
it, and before that "holes 3% · watertight 71% (21 runs)" from the automatic measurements alone —
every finished mesh contributes those, reviewed or not. Verdicts, measurements and matched pairs
all survive pruning and sweep deletion; the corpus outlives the assets it was learned from.

Next: [Keyboard shortcuts](09-shortcuts.md) has the full Review table, and
[The library and jobs](08-library-and-jobs.md) covers where sweep units do and do not appear.
