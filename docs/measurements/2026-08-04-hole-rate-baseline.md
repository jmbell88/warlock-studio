# Hole-rate baseline, 2026-08-04

> **Superseded in its interpretation, and read this first.**
> [`2026-08-09-rebaseline.md`](2026-08-09-rebaseline.md) is the successor. The
> distribution measured below is real and the arithmetic stands, but the metric
> it is about has since been shown to point the **wrong way**:
> AUC(`hole_worst` -> reject) = 0.115 over the 84-verdict review of 2026-08-07 --
> not weakly informative, backwards, because the dominant failure mode is a
> solid slab and a slab has no visible openings. **Nothing here licenses reading
> a low hole fraction as evidence of quality.** Separately, every *watertight*
> figure taken before 2026-08-08 is void: `meshreport` counted xatlas UV-seam
> splits as boundary edges, and it now welds by position before judging.

A measurement, taken before anything is built that reacts to it. Part A Task 7
proposes an automatic remesh-and-retry keyed on `params["mesh_audit"]["worst"]`,
and a retry threshold chosen against a number nobody has measured recently is
worse than no retry at all: it either never fires, or it burns two minutes of
GPU re-running meshes that were already fine. So the threshold is derived from
the distribution rather than guessed, the same way `Config.trellis_band` was
settled by a sweep rather than by argument.

Run started 2026-08-04 23:26:07 and finished 2026-08-05 02:20 local -- just
under three hours of GPU. The document is dated by the run's start, per the
task brief.

## What was run

Step 1, to see what exists:

```
$ uv run python -m warlock.bench suites
core-v1  40 items x 4 seeds  Core benchmark v1

$ uv run python -m warlock.bench recipes
baseline-turbo-raw   SDXL-Turbo, no style, raw mesh
playground-fidelity  Playground v2.5, 25 steps with CFG
sdxl-hyper-render3d  SDXL 1.0 + Hyper-SD, 3D-render LoRA
```

`core-v1` is the only suite, so the question of which one to use did not arise.
`baseline-turbo-raw` is the default recipe and the one that matches the shipped
configuration -- SDXL-Turbo, no style LoRA, `mesh_profile = raw` -- which is the
configuration a threshold has to hold for.

Step 2, the run itself:

```
uv run python -m warlock.bench run --suite core-v1 --recipe baseline-turbo-raw \
    --stage model --seeds 42,1337
```

Exit code 0. Run directory `bench/runs/20260804-232607-baseline-turbo-raw-core-v1`;
full log kept at `.superpowers/sdd/UPDATE/bench-run.log`. The manifest records
`trellis_band: null`, `trellis_tex_res: 512`, `mesh_profile: raw`,
`vram_exclusive: false`, torch 2.11.0+cu128, trellis-server as vendored.

Step 3, the collection, was the brief's own snippet run verbatim against the
data dir: it reads every row of `jobs`, pulls `mesh_audit.worst` out of `params`
where it is present, and sorts descending.

### The seed reduction, and why

`--seeds 42,1337` is a deviation from the brief and is recorded here rather
than buried. The suite declares four seeds (42, 1337, 20240701, 987654321),
which is 160 jobs; the CLI's own help puts `--stage model` at 6-8 hours of GPU
for that. Two seeds is 80 jobs and came in at under three hours.

The important part is *which* axis was cut. **All 40 items ran.** This is a
whole-suite run at reduced seed count, not a partial run over some of the
items -- every prop, weapon, character and environment in `core-v1` is
represented, and the reduction only costs resampling depth per item. That
matters because the failure mode this task exists to prevent is inferring a
baseline from a biased subset of *subjects*; a subset of seeds narrows the
confidence interval rather than tilting it. It does mean `n` is smaller than
it might have been, and the per-item variance across seeds is measured with
two samples rather than four. Anyone repeating this should run all four.

## The numbers

Of the 80 jobs, 38 finished and 43 errored (see the reference-refusal section
below). 37 of the 38 finished jobs carry a `mesh_audit.worst`.

```
n = 37
mean   0.1224
median 0.0120
max    0.5557
min    0.0000
```

Worst twenty, descending:

| worst | job |
|---|---|
| 0.5557 | 428a79c9cc55 |
| 0.5552 | 62bbfb4651e2 |
| 0.4701 | 75da99410e09 |
| 0.4305 | db71820e88db |
| 0.3210 | 464eefc6e3aa |
| 0.3061 | 7d3e978f3933 |
| 0.3020 | 53c97ee8343f |
| 0.2868 | b6b7f6409682 |
| 0.2383 | 4ed5bfdc55d7 |
| 0.2281 | c19f34c528a0 |
| 0.2223 | 732ba8258214 |
| 0.1521 | f82d943acb97 |
| 0.1255 | 4f36f289d3a3 |
| 0.1192 | adf08df94029 |
| 0.1010 | a23136069048 |
| 0.0308 | 06816a103236 |
| 0.0166 | b504b01fccc1 |
| 0.0133 | 50a6b5680ea2 |
| 0.0120 | 2c2372e98171 |
| 0.0115 | ff79cc39fed5 |

The remaining seventeen run from 0.0079 down to 0.0000, with eight of them
exactly 0.0000.

Counts above a range of candidate thresholds:

| threshold | above |
|---|---|
| 0.03 | 16 of 37 |
| 0.05 | 15 of 37 |
| 0.06 | 15 of 37 |
| 0.08 | 15 of 37 |
| 0.10 | 15 of 37 |
| 0.15 | 12 of 37 |
| 0.20 | 11 of 37 |

## The distribution is bimodal, with an empty gap

This is the finding that makes the threshold easy, and it is worth stating
before the verdict because it is the reason the verdict is not a judgement
call.

Nothing at all falls between **0.0308 and 0.1010**. Twenty-two of the 37 sit at
or below 0.031 -- a healthy cluster with a median of 0.012 and eight meshes at
a flat zero -- and fifteen sit at 0.101 and above, running out to 0.556. There
is no middle. The mean of 0.1224 is therefore describing nothing real: it is
the average of two populations, and no mesh in the sample resembles it.

That is not a statistical accident, it is the mechanism `meshaudit` was written
against. A trellis narrow-band remesh either produces a joined surface or it
produces a crust of disconnected plates, and the plates do not partially meet.
A mesh is perforated or it is not.

## The threshold, and how it was chosen

**0.07**, the midpoint of the empty gap.

It was *not* chosen as a percentile, and the distinction matters for anyone
tempted to re-derive it later from a different sample. Any threshold anywhere
inside `(0.031, 0.101)` selects exactly the same fifteen meshes -- the table
above shows 0.05, 0.06, 0.08 and 0.10 all returning 15 of 37 -- so the choice is
free within that interval, and the midpoint is simply the value furthest from
either edge of the gap and hence the most robust to a future sample shifting
one cluster slightly. A percentile would have been fragile in exactly the way a
gap midpoint is not: 15/37 happens to be the 59th percentile *today*, but that
figure is a property of this sample's mix of items, not of the defect.

There is independent corroboration, arrived at from the other direction:
`meshaudit.hole_fraction`'s docstring, written long before this run, says a
solid object measures ~0.0 and "the perforated meshes this was written for
measure 0.07-0.15 depending on the direction". A threshold derived here from a
gap in a 37-mesh distribution lands on the lower bound of a range someone else
wrote down from eyeballing perforated meshes. Two methods, one number.

## Verdict

**Warranted.** 15 of 37 meshes exceeded 0.07. A retry threshold of 0.07 catches
those and nothing else. Task 7 proceeds with `WARLOCK_MESH_HOLE_MAX`
defaulting to 0.07.

(This paragraph first named the variable `WARLOCK_REMESH_HOLE_MAX`, which was
never the name the code took: the shipped pair is `WARLOCK_MESH_RETRIES` and
`WARLOCK_MESH_HOLE_MAX`, backing `Config.mesh_retries` and
`Config.mesh_hole_max`. Corrected here rather than in the code, because the
name is arbitrary and the number is not -- and a document and a config field
naming different variables for the same ruling is how the two drift apart.)

## Observation: a >50% reference-stage refusal rate

Not this task's verdict, but the most striking thing in the run, and the next
reader should not have to rediscover it.

43 of the 80 jobs errored, and essentially every one of them failed at the same
place: the reference composition gate, refusing to hand the drawn image to
trellis.

```
39x  RuntimeError: The subject runs off the edge of the frame.
 4x  RuntimeError: The subject runs off the edge of the frame.;
     There is more than one object in the reference.
```

That is the entire failure set -- there were no trellis failures, no VRAM
refusals, no export errors. Under `baseline-turbo-raw` on the core suite, over
half of SDXL-Turbo's compositions put the subject off the edge of the frame,
and the gate correctly refuses them. The gate is doing its job; the generator
is not.

This is precisely the condition Part A Task 5's opt-in reference reroll
(`WARLOCK_REFERENCE_RETRIES`, off by default) was built for. The number here is
an argument about that default, and about whether a bench run should enable it
so that mesh-stage statistics are gathered over compositions that passed rather
than over the subset that happened to pass first try. It is left as an argument
rather than settled, because it is a different question from this one.

It also bounds the present measurement honestly: `n = 37` is a *survivor*
sample. Every mesh measured here came from a reference that already passed the
composition gate. If off-frame compositions correlate with worse reconstructions
-- which is plausible, since a clipped subject gives trellis less to work with
-- then the true hole-rate distribution over all 80 attempts has a heavier tail
than the one above, not a lighter one. That direction is safe for the threshold:
0.07 does not become too aggressive under a heavier tail.

## Correction to the recorded memory

Project memory carries a note titled "Band sweep result" stating that the old
"7-31% of the silhouette is holes" figure is *stale*, on the grounds that a
2026-08-01 band sweep "never measured worse than 1.7%". This run contradicts
that at 40% of the sample: fifteen of 37 meshes measured above 10%, four above
40%, and the worst measured 55.6%.

The earlier sweep's conclusion does not generalise, and it is worth being
precise about why rather than simply overturning it. The band sweep varied
`trellis_band` over a small set of values on a small set of subjects; it was
designed to answer "does widening the band help?" and it answered that. It was
never a survey of the item space. This run is the whole core suite -- forty
items across props, weapons, characters and environments -- at two seeds, and
it says the item space contains plenty of subjects that reconstruct as
perforated crusts at the default band.

Two things are therefore *not* being overturned. The band sweep's finding about
`DEFAULT_TRELLIS_BAND` itself stands: leaving it at `None` is still the right
default and widening the band still made meshes worse. And the sweep's own
measurements were not wrong -- they were correct about the subjects they
covered. What does not survive is the inference drawn from them, that high hole
rates are not a real phenomenon. They are real, they are common, and the
7-31% range that was called stale is if anything an *understatement* of the
tail.

The memory note should be read as scoped to the band question from here on.
