# The 3/4 tile guide: does asking for it change the picture? 2026-08-21

**Status: pre-registration.** Everything below the next heading was written
before any unit was queued, in the style of
[`2026-08-09-framing-axis.md`](2026-08-09-framing-axis.md). The Results section
is appended after the run and nothing above it is edited.

## The question

`pipelines/tilesheet.py` grew a third view, `three_quarter`, beside `top_down`
and `isometric`. It shares the square lattice with `top_down` exactly — same
cell, same guide rectangle, same 1024×1024 frame, same slicer — so the *only*
things that differ are the subject clause SDXL is handed and whatever interior
marks the canny guide carries.

That is precisely the case this module already refused once. `oblique` is
absent, with the reason written into the constant:

> `oblique` is deliberately absent: it uses the rectangle exactly as the
> orthogonal case does, so offering it would be a third button that changed
> nothing about the picture.

**So the question is not "which 3/4 guide is best". It is "is 3/4 a button that
changes the picture at all".** If it is not, the honest outcome is to delete the
view and leave that comment standing.

## The arms

One prompt, chosen because a 3/4 view has something to say about it — a scene
whose objects have fronts as well as tops:

> `a stone dungeon: floor flagstones, mossy brick walls, wooden crates, barrels,
> a wooden door, iron grates`

Three seeds: `(7, 4211, 90210)`. Tile size 32, colours 64, `sdxl_cfg` with the
pixel-art LoRA — the shipped recipe, untouched.

| arm | view | `THREE_QUARTER_GUIDE` | guide |
| --- | --- | --- | --- |
| **R** (reference) | `top_down` | — | cell rectangle only |
| **A** (control) | `three_quarter` | `plain` | cell rectangle only — byte-identical to R's guide |
| **B** | `three_quarter` | `horizon` | rectangle + a horizontal line at 2/3 cell height |
| **C** | `three_quarter` | `hatched` | B plus three verticals down the front band |

A and R differ **only** in the subject clause. That pairing is the whole
experiment: it isolates the clause from the guide.

## What is being judged

Two claims, and they are not the same claim — the lesson
[`2026-08-18-tile-sheet-grid.md`](2026-08-18-tile-sheet-grid.md) paid for:

1. **Different from top-down.** At the same seed, does arm A/B/C return a
   visibly different picture from R? A mean-colour delta is not enough — a
   darker version of the same flat tiles would pass it — so this is judged by
   eye, on whether tiles have acquired a *front face*: a band along the bottom
   edge in a different plane from the top.
2. **3/4 rather than merely tilted.** Do the fronts point the same way in every
   cell? Sixty-four tiles each tilted in their own direction is not a view, it
   is noise, and it is the failure mode a per-cell guide mark invites.

Both are human verdicts on a contact sheet. There is no cheap automatic measure
of either and this document is not going to pretend otherwise.

## The decision rule, fixed in advance

- **If no arm clears claim 1**, `three_quarter` is deleted from `VIEWS` and the
  `oblique` comment stands unamended. The Wave-0 vocabulary work stays (it is
  a rename plus an alias and it stands on its own), but the third *value* goes.
- **If an arm clears both claims**, `THREE_QUARTER_GUIDE` ships on the simplest
  arm that does. Simplest means fewest marks: A, then B, then C. A tie goes to
  the earlier letter, decided here so it cannot be decided by looking.
- **If A clears and B/C do not**, that is the strongest possible result: the
  clause alone carries the view, and `render_guide` keeps exactly two shapes.
- **If B or C clears and A does not**, the guide is doing the work, and the
  losing arms are deleted rather than left as dead constants.

## What a win costs, stated in advance

- `TILE_SHEET_VERSION` is already 2 for the vocabulary; the arm choice rides
  that number and does not need another.
- No `PROMPT_VERSION` bump: the clause lives in `tilesheet._VIEW_CLAUSE`, not in
  `prompt.TILESHEET_TEMPLATE`. The findings corpus is not re-keyed.
- `VECTOR_PARAMS` is untouched, so no stored observation changes meaning.
- The losing arms are **deleted from `render_guide`**, not left behind a
  constant. A dead arm is a thing the next reader has to decide about again.

## What this run does *not* settle

The sameness problem. Every arm here inherits the single-generation shape whose
verdict is still owed in `2026-08-18-tile-sheet-grid.md`, so all sixty-four
cells may well come back as near-variations of one tile whichever arm wins. That
is a different question, owed its own pre-registration, and this run is not
evidence about it.
A 3/4 arm that produces one repeated tile *with a front face on it* still clears
claim 1: the claim is about the view, not about the variety.

---

## Results, 2026-08-21

**Verdict: arm A ships. `THREE_QUARTER_GUIDE` is gone and `render_guide` keeps
exactly two shapes — the subject clause alone carries the view.**

Twelve units, four arms × three seeds, ~170 s on a 5090; the post-hoc pair below
is six more at ~95 s. The contact sheets are in
`docs/measurements/data/three-quarter-guide/` as `arms.png` and `objects.png`,
**gitignored** with the rest of that tree — the run is fully specified by the
arms, prompts, seeds and recipe stated in this document, and everything here was
judged by eye against those two images.

### The primary run failed claim 1 — and the instrument is why

On the pre-registered dungeon prompt, **no arm cleared claim 1**:

| arm | what came back |
| --- | --- |
| A (`plain`) | the same picture as R at every seed. Same composition, same moss-filled recesses, same layout — a different roll of one flat material, with no front face anywhere |
| B (`horizon`) | the line was **obeyed** and drew a dark horizontal stripe across every cell. Read as a shadow or a ledge, never as a change of plane — and it flattened the sheet: the greens went, the depth went, and all sixty-four cells converged on one tile |
| C (`hatched`) | B, worse. The verticals came back as small dark notches on an even flatter sheet |

B and C are the failure `2026-08-18-tile-sheet-grid.md` already catalogued —
push the guide harder and the model obeys it and paints one tile sixty-four
times. That the marks were *followed* and still produced no front face is the
useful part: the guide can place a boundary, and cannot specify which side of it
is a different plane.

**The decision rule's "no arm clears claim 1" branch says delete
`three_quarter`. It was not taken, and this section is where that is argued
rather than quietly skipped.**

### The post-hoc pair, labelled post-hoc

The prompt is the confound. *"floor flagstones, mossy brick walls"* asks for a
flat material, and a camera tilt has nothing to reveal on one — there is no
height in the cell for a front face to belong to. So the primary run tested
whether the clause survives a subject that cannot express it.

Two arms were added **after seeing the primary result**, and they get no vote on
the primary claim:

| arm | view | prompt |
| --- | --- | --- |
| D | `top_down` | *"a village street: cottage walls with shuttered windows, a thatched roof edge, wooden fence posts, a stone garden wall, hedges, stacked crates, a well, a doorway"* |
| E | `three_quarter` | the same |

Same three seeds, same everything else. E is arm A's configuration exactly —
clause only, plain guide.

The pair separates cleanly at all three seeds. D draws its subjects as **plan
shapes**: a crate is a flat rectangle seen from directly above, the well is a
flat ring, the shutters are flat slats. E draws the same subjects with a **top
face and a front panel below it in a darker plane** — the crates at seed 7, the
red-roofed well and the angled shuttered window at 4211, the lit-top crate at
90210. Claim 2 holds too: every front points the same way, toward the viewer, in
every cell.

So the clause does change the picture, and by the amount the view claims — on a
subject that has something to show.

### What was decided, and what a later reader may disagree with

- `three_quarter` **stays** in `VIEWS`, on arm A. Per the rule's own wording
  this is "the strongest possible result: the clause alone carries the view, and
  `render_guide` keeps exactly two shapes" — reached by the post-hoc pair rather
  than by the primary run, which is stated here so it can be argued with.
- Arms B and C are **deleted** from `render_guide`, along with
  `THREE_QUARTER_GUIDE` and `THREE_QUARTER_HORIZON`, per the pre-registration.
  A future interior mark has to argue against this run.
- The `oblique` comment in `pipelines/tilesheet.py` stands for `oblique` and is
  now explicitly distinguished from this case in the constant's own docstring:
  oblique changed neither the arithmetic nor the picture; 3/4 changes the
  picture through the clause.
- **The honest weakness**: the primary arm set is now evidence about almost
  nothing, because it was run on a prompt that could not express the difference.
  A re-run of A/B/C on the village prompt would be a fair test of whether an
  interior mark helps *when there is a front to help with*, and has not been
  done. B and C were deleted on evidence that they harm a flat-material sheet,
  not on evidence that they harm an object sheet.

### A second finding this run did not go looking for

D and E both produce **sixty-four cells that differ from one another**, which no
arm of `2026-08-18-tile-sheet-grid.md` managed. The difference is the prompt:
that document's subject was one material, and this one names eight *objects*.
Variety followed the request, not the guide — which is the same conclusion that
document reached from the other side ("variety came from N prompts, not from one
prompt and a hope") and is direct support for its ranked candidate #1. It is not
a substitute for measuring that candidate: one prompt naming eight things is not
the N-materials shape, and nothing here says how it behaves at sixty-four.
