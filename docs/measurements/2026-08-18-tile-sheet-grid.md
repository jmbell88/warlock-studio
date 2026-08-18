# The tile-sheet grid: what SDXL will and will not draw, 2026-08-18

**Status: run taken, decision owed.** The mechanism works and is shipped; the
art direction does not reach the bar and the constants ship **provisional**.
This document is what the next change to them argues against.

## The question

`pipelines/tilesheet.py` asks SDXL 1.0 (`sdxl_cfg`, pixel-art LoRA at 1.2) for
one 1024px generation laid out as an 8×8 grid of *different* tiles, forced onto
the grid by a canny ControlNet fed a picture of the cell boundaries. The
reference for "good" is `examples/tileset3.png`, `tileset8.png` and
`tileset10.png` — 64 distinct, correctly-framed terrain tiles sharing one
palette.

Two things had to be true and only one of them is:

1. **the guide is obeyed** — the model's discontinuities land on the rectangles
   the slicer cuts on, or every tile carries a sliver of its neighbour;
2. **the cells are different tiles** — 64 variations of one material is a
   texture sheet, not a tileset.

## The arms

One seed (7), one subject ("a damp stone dungeon…" plus `DETAIL_CLAUSE`), four
arms over the two constants most likely to be wrong — how thick the guide's
lines are, and how hard the template asks for *separate* cells.

| arm | guide | control scale | LoRA | result |
| --- | --- | --- | --- | --- |
| A: shipped constants | 2px borders | 0.65 | yes | **one continuous brick wall** through the grid; the guide reads as mortar |
| B: thick + gutters | 6px, 6px inset | 0.80 | yes | cells genuinely separated — and near-identical grey mush in every one |
| C: B without the LoRA | 6px, 6px inset | 0.80 | no | the model draws **the guide itself**: 64 blank plates with the border painted in |
| D: weak control | 2px borders | 0.35 | yes | one continuous scene again, grid barely visible |

The isometric case fails a fifth way: at 1024×512 with a diamond inscribed in
every cell, the guide is obeyed *perfectly* and all 64 cells come back as the
same tile.

## What the numbers say, and why they were not enough

The GPU test's grid check passes on arm A: median edge energy across the cell
boundaries is >1.5× the median across cell interiors, which is a true statement
— the guide *is* landing. It is simply not the property that matters. A
continuous wall with mortar lines on the grid satisfies it exactly as well as a
real tileset does.

So the measurement's own lesson is about the instrument: **"the seams are on the
grid" and "the cells are different tiles" are two claims, and only the first is
cheap to assert.** `test_the_cells_are_different_pictures` covers the second with
a colour-mean spread, which arm A also passes — a wall lit unevenly has spread.
A test that separated arm A from a real tileset would need per-cell *structural*
distinctness, which is the thing nobody has a cheap measure of.

## The diagnosis

Every cell of the guide is identical, so there is no per-cell signal for
variety. The model resolves that ambiguity in one of two ways depending on how
hard the guide is pushed — ignore it and paint one picture (A, D), or obey it
and paint one tile 64 times (B, and the isometric case).

This is the difference from `pipelines/spritesynth.py`, which uses the same
mechanism successfully: **each of its cells carries a *different* guide** (a
different stick-figure pose). A terrain tile has no canonical per-cell
silhouette to put there, so the trick does not transfer.

The retired ground path did not have this problem, and the reason is worth
keeping: it generated **one material per prompt** and composited afterwards.
Variety came from N prompts, not from one prompt and a hope.

## What ships, and what is owed

Shipped, and correct: the job kind, the door, the worker, the reduction (which
is measured — `2026-08-17-ground-reduction.md` — and is not implicated in any
of the above), the geometry, the exports and the UI. A sheet comes back on the
grid, slices cleanly, shares a palette and is a usable *material* sheet.

Provisional, and owed a decision: `prompt.TILESHEET_TEMPLATE`, the guide's line
width in `tilesheet.render_guide`, and whether one generation is the right shape
at all. The three candidates, in the order this run makes them look plausible:

1. **N materials, one grid.** Generate one texture per material the user names
   — the shape that already worked — and lay them out as a plain grid sheet
   rather than a 47-case atlas. Variety is then a property of the request rather
   than of the model's composition, which is the only arm here that addresses
   the diagnosis. Costs a form field (the materials) and N generations.
2. **A per-cell content guide.** Whatever would make cell (r, c) different from
   cell (r, c+1) in canny space. Nobody has proposed one for terrain.
3. **A stronger base.** The reference sheets in `examples/` are very likely not
   SDXL output; compositional control over 64 framed cells is a frontier-model
   capability. Out of scope while the app is offline-only.

Until one is taken, the feature is honest about what it is and the GPU test
pins the half that works.
