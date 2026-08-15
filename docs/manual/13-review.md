# Review

Review is where finished meshes get judged, and where those judgements start paying you back. Every
grade you give a mesh is recorded as a **verdict** against the job it was filed on, together with a
copy of the settings that produced it — and the pool of verdicts is aggregated into the "What works"
findings that the generate panes surface as hints, ranked configurations you can apply to the forms
or save as presets. Ten seconds of judging a mesh teaches the app which settings earn their place.

There are two ways to file a verdict, feeding one pool:

- **In Review**, against a sweep's units or the recent-unreviewed bucket — built for judging many
  meshes quickly, with the keyboard.
- **In Create's inspector**, against whatever asset is selected — the grade row under the mesh
  quality section. One-off verdicts during ordinary use count exactly as much.

## The workspace

Review uses the same three-column skeleton as Create:

- **Left**: the sweep list — a "Recent, unreviewed" bucket first, then every sweep, newest first —
  and below it the form that launches a new sweep.
- **Centre**: the shared asset viewer, showing the unit under judgement. Orbit, wireframe and
  framing work as they do in 3D.
- **Right**: the verdict panel — the unit's reference image, its mesh measurements, the grade row
  and the tag toggles — with the findings underneath.

"Recent, unreviewed" collects finished meshes from ordinary use that nobody has judged yet. It is
not a sweep: it has no settings axes and cannot be deleted, it simply empties as you file verdicts.

## Judging

A mesh verdict is a **grade from −5 to +5**, not a yes or no. **+5** means it ships as-is, **+3**
means usable, **0** means no opinion either way, and **−5** means nothing about it is recoverable.

`1`–`5` file +1 to +5. `R` arms the negative sign, so the next `1`–`5` files −1 to −5. `0` is its
own key, because zero has no sign to arm and is a real answer rather than a refusal to give one.
That is eleven values inside six keys, which is what keeps a pass moving. `Esc` clears a pending
sign. `S` skips to the next unverdicted unit, and `Left` / `Right` step through the list. Every one
of these exists as a button in the verdict panel too.

Why a grade rather than an accept and a reject: a bit can say a mesh failed and can never say how
close it came. A solid slab with no geometry at all, a good shape with a smeared texture, and a mesh
a modeller would fix in five minutes were the same row — and that is the difference any ranking most
wants to know.

**Tags are optional at every grade.** `Ctrl` + `1`–`5` toggle the good ones — clean shape, good
texture, on style, sharp detail, good topology — and `Shift` + `1`–`5` the bad ones — holes, bad
shape, bad texture, wrong style, broken. They are staged until you press a grade, and anything that
moves you off the unit drops them, because a tag chosen while looking at one mesh must never be
filed against the next. A good tag on a negative grade is deliberately allowed: a mesh can have a
clean shape and still be unusable, and that pair of facts is worth more than either alone.

Where a single yes-or-no answer is still needed — deciding what pruning may reclaim, or what counts
towards a configuration's success rate — **grade +3 or better counts as usable**. Judgements filed
before grades existed were read as +3 for an accept and −3 for a reject: an accept asserted "usable"
and nothing stronger, so ±4 and ±5 are left for judgements nobody had the keys to make.

**Blind** hides which settings each unit ran. It renames every unit to a short id and presents them
in an order derived from the job id rather than the order they were queued — hiding only the label
would blind nothing, because a sweep queues its baseline first and position alone would name the
arm. Your own accepts and rejects still show, so a session is still resumable. Turn it on when the
verdict is the evidence for a decision rather than a note to yourself: knowing which arm you are
looking at is exactly what a confirming run cannot afford. It is per-session and starts off.

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

## Teaching the judge

Under the sweep list are two labelling passes. They are a different kind of work from judging a
mesh: an image takes about two seconds to answer and carries no reason, so the centre column becomes
a grid of pictures and `A` / `R` mean good and bad. While a pass is open it owns the keyboard, so
there is never any doubt which question a keypress answered.

There are two passes because the same image is two different things, and "good" means opposite
things about them:

- **Good 2D asset?** — judge the finished picture: composition, style, drama.
- **Good to reconstruct?** — judge it as input for the mesh: one subject, plain background, a
  neutral pose.

A dramatic plate with pillars and a cast shadow is a better asset and a worse blank, which is why
one label cannot stand in for the other. Both live on the same job, independently, and a mesh
verdict on that job says nothing about either.

Images that were **refused** at the composition gate appear here too, marked. They are the most
useful negatives there are: the picture exists, something was wrong with it, and a judge that has
only ever seen images the rules already liked has learned the rules rather than the quality.

Once there are enough of both answers the app trains a small classifier per question from your
labels — seconds of work, off the frame thread, and it retrains as you go.

What that classifier does today is score and sort. Every unit in a sweep is scored on the **Good to
reconstruct?** question, the verdict panel shows the number and says which question it answers, and
the review is presented best-scoring first. Unscored units come last rather than first, because "no
opinion" is not "bad". It is **advisory only**: it never refuses a job, deletes anything or retries,
and it only ever *sorts* — a judge that hid what it disliked would make its own mistakes invisible,
and you would never learn it was wrong.

It also does not file a verdict of its own yet. Doing that means picking a probability above which
the app says "accept", and that number would then be baked into everything already recorded, so it
waits on a measurement rather than on a guess. The **Good 2D asset?** question and the mesh judge are
the two still to come; the mesh one needs enough accepted meshes to learn from, which is what the
review loop is building.

Turning **Blind** on switches the judge off entirely for that session: no score is shown and the
order goes back to the blind one. A score is a quality signal that would identify the arms, and an
opinion on screen anchors the independent judgement a blind review exists to collect.

## What works

The findings section at the bottom of the verdict panel gives two kinds of answer.

**Axis verdicts** are the conclusive kind: matched pairs recovered from sweeps. Two units pair up
when they share a sweep, a seed and a prompt and differ in exactly one setting, so a line like
"lora_weight: 0.6 beat 0.9 in 7 of 8 matched pairs (2 sweeps, 2 prompts)" is an all-else-equal
comparison, not a correlation. The winner is whichever side graded higher, and the line says by how
much — "…in 7 of 8 matched pairs, avg +1.4 grade (2 sweeps, 2 prompts)". Machine measurements pair
the same way: every finished mesh is measured automatically (worst-hole fraction, watertightness,
triangle count), so a sweep shows "worst-hole -4.1% over 12 paired runs" the moment it finishes,
before a single verdict is filed.

**Ranked configurations** are whole settings vectors ordered by a conservative floor on their
usable rate, shown as "usable 80% of 20 (61%+) · avg +2.6": the first number is what happened, the
parenthesised one is the floor the evidence supports — which is what stops a lucky 5-for-5 from
outranking a solid 19-of-20 — and the average is the mean grade behind it. The average breaks ties
rather than doing the ranking: over one sample its own spread is zero, which would re-create exactly
the lucky-5-for-5 problem the floor exists to prevent. A configuration needs five verdicts to
appear, and carries muted lines of its machine measurements and its tag tallies when it has any.
"Apply to forms" writes one into the 2D and 3D forms; "Save as preset..." keeps it under a name.

The same findings feed the small hints next to controls in the generate panes, so the learning is
visible where the decisions are made: "usable 6/8 (41%+) · avg +2.6" once a value has enough
verdicts behind it, and before that "holes 3% · watertight 71% (21 meshes)" from the automatic
measurements alone —
every finished mesh contributes those, reviewed or not. Verdicts, measurements and matched pairs
all survive pruning and sweep deletion; the corpus outlives the assets it was learned from.

Next: [Keyboard shortcuts](14-shortcuts.md) has the full Review table, and
[The library and jobs](11-library-and-jobs.md) covers where sweep units do and do not appear.
