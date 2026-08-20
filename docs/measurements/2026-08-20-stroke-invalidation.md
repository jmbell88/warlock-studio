# Stroke invalidation: a union standing in for a set

**2026-08-20.** Closes the item the Aseprite parity programme's P1 appendix
carried as
"the measured `symmetry="xy"` 16× per-dab invalidation cliff (union-rect defect,
needs its own measurement doc before a fix — the constants rule)".

The measurement found the cliff, and found that it was the smaller half of the
problem.

## What was happening

`StrokeState._mark` folded every stamped rectangle into a single accumulating
box, `stroke.dirty`. Two callers read it:

| caller | when | is one box right? |
|---|---|---|
| `end_stroke` → `_commit_patch` | once, at release | **yes** — one patch over the union is the deliberate choice; a multi-rect edit type would need eviction accounting of its own for a saving measured in kilobytes |
| `_touch_stroke` → `Document.invalidate` | **after every dab** | no |

The second is the defect. A single box answers "what does the undo patch have to
cover", and it was being used to answer "what has to be recomposited right now",
which is a different question with a different answer.

Two consequences, and only one of them was the known item:

1. **Stroke growth.** The union grows as the stroke moves, so dab *N*
   recomposited the bounding box of dabs 1..*N* — for a dab that never covers
   more than the nib. Present on **every** symmetry setting, including `none`,
   which is to say on essentially every stroke anyone has ever drawn in this
   editor.
2. **The mirror cliff.** `symmetry="xy"` puts one dab in four places, far apart
   by construction, so the union spans both mirrored positions on both axes —
   near the whole canvas, from the very first dab.

## Method

`scratchpad/bench_symmetry.py` (reproduced below in substance): a 512×512
document, an 8px soft nib, a 200-move diagonal drag corner to corner — the
ordinary way a stroke is made. `Document.invalidate` is wrapped to record the
area of every rectangle it is handed. Wall clock is the whole drag.

Area is the primary number and wall clock the secondary one: the recomposite is
proportional to area, and wall clock on one machine is the thing that moves when
somebody else runs it.

## Before

| symmetry | wall | total area recomposited | mean per call | as a share of the canvas |
|---|---|---|---|---|
| none | 213.2 ms | 17,284,230 px | 85,565 | 0.33× |
| x | 313.0 ms | 25,699,502 px | 127,225 | 0.49× |
| y | 320.6 ms | 25,699,502 px | 127,225 | 0.49× |
| **xy** | **611.5 ms** | **50,302,202 px** | **249,021** | **0.95×** |
| radial | 705.2 ms | 52,650,799 px | 260,648 | 0.99× |

Growth, isolated (`symmetry="none"`, same stroke, first quarter against last):
**5,881 px → 196,511 px per dab, a 33× ramp** across a single stroke.

`xy` recomposited **95% of the canvas per dab**. The canvas here is 512×512; the
cliff gets worse with canvas size, since the nib does not grow with it.

## The change

`StrokeState` keeps `touched`, a **list** of the rectangles marked since the last
recomposite, alongside the existing `dirty` union. `_touch_stroke` drains it with
`take_touched()` and invalidates the pieces. `dirty` is untouched and still backs
the single undo patch, so nothing about history changes.

`take_touched` collapses the list back to its union when the union is no more
than `TOUCH_UNION_RATIO` (2.0) times the summed area of the parts. That one rule
covers both cases correctly and needs no special-casing of symmetry:

- Consecutive dabs along a stroke overlap heavily, so their union is barely
  bigger than their sum → one call, which is the better answer.
- Mirrored dabs at opposite corners have a union hundreds of times their sum →
  keep the parts.

`TOUCH_RECTS` (256) is a hard ceiling on calls per flush, for `spray`, which
emits its whole count before the flush and multiplies it by the mirrors.

**A first attempt used a plain count cap of 12 and it was wrong**: a single move
interpolates roughly four dabs, so `xy` produces ~17 rectangles per move and blew
straight through it, collapsing to the full-canvas union again. It measured 32%
of the canvas per dab — better than 95%, and still nothing like right. The
count-based rule was answering "how many pieces" when the question is "how
spread out are they". That is recorded here because the wrong version *looked*
like a large improvement.

## After

| symmetry | wall | total area | vs before (wall) | vs before (area) |
|---|---|---|---|---|
| none | 32.5 ms | 269,657 px | **6.6× faster** | **64× less** |
| x | 75.1 ms | 364,466 px | 4.2× | 71× |
| y | 74.7 ms | 364,466 px | 4.3× | 71× |
| **xy** | **155.6 ms** | **470,840 px** | **3.9×** | **107×** |
| radial | 205.8 ms | 538,606 px | 3.4× | 98× |

Every mode recomposites under 1% of the canvas per dab. The plain `none` stroke —
the common case, and the one nobody had listed as a defect — gained the most.

## What is not claimed

- **This is an engine measurement, not a frame-rate one.** It measures the
  compositing `Document.invalidate` does. The texture upload on the other side of
  it has its own rectangle accounting, and whether the app's frame time improves
  by the same factor was not measured here.
- **The 16× in the original note is neither confirmed nor refuted.** Whatever was
  measured then was measured on a different canvas, nib and stroke; 2.9× on area
  (`xy` against `none`) is what this configuration shows, and the absolute area
  is the number that matters rather than the ratio between two bad answers.
- **`TOUCH_UNION_RATIO = 2.0` is a default rather than a finding.** Nothing
  stored is keyed on it and no corpus depends on it; the two regimes it separates
  are hundreds of times apart, so anything between about 1.5 and 20 picks the
  same answer for both. It is written down here so a future change knows it was
  chosen, not derived.
