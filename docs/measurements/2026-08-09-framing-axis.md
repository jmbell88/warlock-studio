# The framing axis: `front_ortho` against `three_quarter`, 2026-08-09

**Status: run taken 2026-08-09; "Results" below is the outcome — the verdict was
null and directionally against `front_ortho`, so no `PROMPT_VERSION` bump was
made.** Everything between "The question" and the Results heading is a
pre-registration in the style of
[`2026-08-06-pixel-art-xl.md`](2026-08-06-pixel-art-xl.md): the decision rules
below were written before any unit was queued.

This is the most valuable pre-registration of the campaign and the reason is
arithmetic. The axis is **5 matched pairs on one prompt** -- exactly the size
where a rule chosen after seeing the split can be made to fit it -- and the prize
is the most expensive move available: a `PROMPT_VERSION` bump that re-keys the
entire findings corpus.

## The question

`guidance.FRAMINGS` carries `three_quarter` (the default) and `front_ortho`, and
`guidance.py`'s own comment calls `front_ortho` "a measurement axis rather than a
new default". So the code is finished and nothing has been measured. **Is a front
orthographic plate a better trellis input than the global three-quarter view, for
a character?**

It is a *camera* claim. It cannot contradict the T-pose fragment that
`category=character` already contributes, and it belongs in the render half of
the sweep rather than the depiction half because it changes how the subject is
presented to trellis rather than what the subject is.

## What will be run

`scripts/sweep_rebaseline.py` carries it:

```python
sweeps_mod.Axis("framing", ("front_ortho",))
```

Five `framing=front_ortho` units against the five baseline units, seed-matched on
`(11, 23, 42, 77, 101)`, one `prompt_hash`, `category=character`,
`bg_removal=birefnet`. Human verdicts through Review **with Blind on**. Read back
from `findings.json`'s `comparisons["framing"]`, via
`bench.findings.comparison_lines`.

## What a win would actually cost, decided in advance

`../LEFTOVERS.md` §2 *(deleted; git history)* says to "flip the per-category
default" if `front_ortho` wins.
**There is no per-category framing machinery.** `guidance.DEFAULT_FRAMING` is a
single global; the precedent for making one per-category is `default_size_m`,
which `CATEGORIES` options already carry. So a win is two decisions, not one, and
this document fixes which:

- **A win adds `default_framing` to the `character` entry of `CATEGORIES` and
  leaves the global `DEFAULT_FRAMING` at `three_quarter`.** The corpus speaks
  about exactly one character prompt; it says nothing whatever about a chest.
- It bumps `pipelines.prompt.PROMPT_VERSION` 4 -> 5. The reason that version sits
  at 4 today is that `three_quarter` carries the exact literal the template used
  to hold, so the composed default is byte-identical; changing which framing a
  category composes by default breaks that.
- **The cost is stated here rather than discovered at commit time:** every vector
  recorded under version 4 stops accumulating evidence, and `_LEGACY_ALIASES`
  does not help -- it renames keys, and this is a change of composed output under
  an unchanged key.

## Decision rules, written in advance

**The outcome is the matched-pair accept/reject count over the 5 seed-matched
pairs.**

- **`front_ortho` wins only at 5-0** (one-sided sign test, p=0.031).
- **4-1 is null** (p=0.19). So is 3-2, so is any tie, and so is 0-0.
- **Both arms at 0 accepts is a floor effect, not a null about framing.** This is
  the Sweep B lesson stated in advance so it cannot be forgotten in the moment:
  an axis measured around a baseline with no headroom has not been measured. The
  axis is re-asked on the next baseline and this document records only that.
- **A 5-0 inside a run that failed the >= 12/50 go/no-go is not a finding about
  framing.** The gate is upstream of every axis in the run.

**Even a 5-0 ships provisional.** Five pairs cannot survive Bonferroni over this
run's nine contrasts, and pretending otherwise is how a corpus-wide re-key gets
made on p=0.031. Fixed now, before the numbers: on a 5-0, either

- a 10-unit confirm is run first, in the `scripts/sweep_confirm.py` shape -- one
  knob, five seeds, blind -- and the default flips only if it replicates; **or**
- the default flips and this document's Results section says **provisional** in
  the first line, with the confirm named as owed.

The choice between those two is the reader's; what is not available is a
non-provisional flip off five pairs.

**Refusal is an observation, never a win.** `front_ortho`'s prompt fragment
exists partly to suppress turnaround and prop-sheet layouts, which were the sole
cause of all 17 refusals in the 2026-08-07 sweep. If it lowers
`refused_multi_object` without adding accepts, that is reported as a
refusal-rate finding and explicitly **not** as a quality finding. The two ranked
the checkpoints oppositely once already.

## Results, 2026-08-09

Sweep `5f26623d12aa`, the re-baseline's `framing` axis. **`front_ortho` did not
win. Nothing changes.**

| Seed | `front_ortho` | baseline (`three_quarter`) | Pair |
| --- | --- | --- | --- |
| 11 | reject `broken` | accept | baseline |
| 23 | reject `bad-shape` | *refused at the gate* | no pair |
| 42 | reject `broken` | reject `bad-texture` | tie |
| 77 | reject `broken` | accept | baseline |
| 101 | accept | reject `broken` | **front_ortho** |

**front_ortho 1, baseline 2, 1 tie, 1 unusable.** Accept rates 1/5 (20%) against
the baseline's 2/4 (50%).

### Verdict: null, and directionally against

The rule required 5-0 for a win, and this is 1-2. Two things are worth separating
in that.

**It is a genuine null, not a floor effect.** The pre-registration named that
distinction in advance precisely so it could not be blurred afterwards: both arms
at zero accepts would have meant the axis was measured around a baseline with no
headroom and therefore not measured at all. The baseline took 2 of 4. There was
headroom, and `front_ortho` did not use it.

**But the 5-0 bar was unsatisfiable here.** `baseline s23` was refused at the
composition gate, leaving only four usable pairs, so a win was arithmetically
impossible before the run began — see Amendment 1 in
[`2026-08-09-rebaseline.md`](2026-08-09-rebaseline.md), which records the same
defect across every axis in the run. **That does not rescue `front_ortho`**: it
lost the pairs it had and accepted less than half as often. A bar that could not
be met is a reason to distrust a *win*, not a reason to discount a loss.

### Consequences, all of them negative

- **`guidance.DEFAULT_FRAMING` stays `three_quarter`.** Unchanged.
- **No `default_framing` is added to the `character` entry of `CATEGORIES`.** The
  per-category machinery this document specified is not built, because there is
  nothing to put in it.
- **`pipelines.prompt.PROMPT_VERSION` stays at 4.** The corpus is not re-keyed,
  and every vector recorded under version 4 goes on accumulating. This is the
  outcome that costs nothing, which is exactly why the rule that produced it had
  to be written before the numbers.
- The provisional-flip machinery — the 10-unit confirm, the "provisional" first
  line — is not needed and is not invoked.

### The refusal clause did not fire

`front_ortho`'s prompt fragment exists partly to suppress turnaround and
prop-sheet layouts. The run's only two composition-gate refusals were `baseline
s23` and `base_model=sdxl_cfg s11`; **no `front_ortho` unit was refused.** With
5 units against 2 refusals corpus-wide there is no rate to compare, so this is
recorded as "not measured" rather than as a refusal-rate finding in either
direction.

### What stays owed

`front_ortho` remains in `guidance.FRAMINGS` as a selectable option. It is not
removed on 4 usable pairs of one prompt — the finding is that it is not a better
*default for characters on this subject*, which is narrower than "it is never
useful". Re-asking it would want a baseline with a higher accept rate and a bar
stated as a fraction of usable pairs.
