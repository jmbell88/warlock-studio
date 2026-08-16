# The three interactive defects: element drag, hover picking, terrain drag

2026-08-16, continuing from `2026-08-16-blend-modes-and-dither.md`. Same machine
(Windows 11, numpy 2.5.1, Python 3.13.13), same rule: every figure is the
minimum of 5–9 runs, and a before/after pair is quoted from one session where
the code allows it, because the spread on multi-megabyte float work is ±10 %.

These are P1–P3 from the C-rewrite review — the three worst *interactive*
defects, and the three the review marked as fixable in Python with no kernel at
all. None of them became a kernel. Two of the three were an algorithm doing work
whose answer it already had; the third was a cache key that included something it
should not have.

The test mesh throughout is a triangulated lattice: **199 712 triangles, 599 136
corners, 100 489 vertices** — the shape and size of a TRELLIS reconstruction,
which is what makes these defects visible at all. On an authored primitive none
of this is measurable.

## Summary

| # | Defect | Gate | Before | After | Verdict |
|---|---|---|---|---|---|
| P1 | element-drag preview rebuilds every render array per mouse-move | interactive at 200k tris | **368 ms** / frame | **92 ms** | **fixed, 4.0×** — §1, and see §1.3 |
| P2 | hover picking has no spatial index; two cheap fixes first | — | 5.6 ms overlay + 1.1 ms convert, in a **55 ms** frame | 0.07 ms + 0.0003 ms | **both fixed** — but §2.3 |
| P3 | terrain drag retiles the whole layer per painted cell | interactive at the map sizes the app offers | **7.9 ms** / cell at 512² | **0.36 ms** | **fixed, 22–41×** — §3 |

**P3 did not clear its gate at the default map size** and is reported that way in
§3.1 rather than quietly counted as a win: at 32² the paint cost 0.05 ms before
the change. It clears it comfortably at the sizes the new-map dialog will hand
out, which is why it was fixed anyway.

---

## §1 — The drag preview rebuilt what a moving vertex cannot change

### 1.1 What was measured

`_view_drag._preview_positions` ran the full `clay.document.to_primitives` on
every mouse-move of an element drag: material grouping, `_submesh`'s corner
gather, `render_arrays`, and the triangulation. Measured on the lattice above:

| | ms |
|---|---|
| `to_primitives`, one object, one material | **368** |
| … the same mesh split across three materials | **377–417** |
| of which `corner_triangles` | 197 |
| of which `face_normals` | 74 |
| of which `_accumulate` | 24 |
| of which `_submesh` | 20 |

At 368 ms a frame a drag runs at under three frames a second, and the cost is
paid per *object* in the drag.

The largest single item — 197 ms, over half the frame — is the triangulation,
and **it was being discarded**. `viewer.scene.GpuPrimitive.update_vertices`,
which is what the preview calls, rewrites the interleaved vertex buffer and
leaves the index buffer alone; it does not read `primitive.indices` at all. So
every frame of every drag ear-clip-screened 200k faces to produce an array that
was never uploaded and never looked at.

### 1.2 The fix

`render_arrays` split into two halves along the line the drag needs:

* `mesh.render_layout(mesh) -> RenderLayout` — everything a moved vertex cannot
  change: the corner permutation, the smooth/flat corner masks, the shared
  vertex table and its remap, and the index buffer.
* `mesh.render_from_layout(layout, positions)` — the two arrays it can: the
  positions and the normals.

`render_arrays(mesh)` is now their composition, so the ordinary rebuild path is
unchanged. `document.render_plan(mesh)` memoises the per-material layouts weakly
against the mesh — the same shape `adjacency.cached_triangulation` already uses,
and sound for the same reason: a `Mesh` is frozen and replaced whole, and a drag
holds the mesh it began on for its whole duration. `document.preview_primitives`
is what the drag calls.

| | before | after | speedup |
|---|---|---|---|
| one material | 368 ms | **92 ms** | **4.0×** |
| three materials | 377–417 ms | **109–120 ms** | **3.5×** |
| first frame of a drag (builds the layout) | — | 293 ms | — |

**Parity is byte identity**, not closeness, and it is asserted two ways in
`tests/clay/test_mesh.py` and `tests/clay/test_document.py`: `render_arrays`
against its own split over six mesh shapes (flat, smooth, mixed, textured,
concave, empty), and `preview_primitives` against `to_primitives` on the moved
mesh. The one deliberate exception is the index buffer, below.

### 1.3 What is still there, and it is most of what is left

92 ms is 11 fps, not a fixed drag. The remaining cost is `_newell` (50–55 ms) and
`_accumulate` plus its gather (31–37 ms), and both are still recomputed over
*every* face of the mesh although a drag moves a handful of vertices. The
next step is the obvious one — compute the affected faces from `drag.verts`,
recompute only those normals, and scatter into the previous frame's arrays. It
was left undone deliberately: it is a different optimisation from the one the
review asked for (a cache), it needs a mutable per-drag cache rather than a pure
memo, and the bit-identity argument for the incremental `_accumulate` rests on
`np.bincount` accumulating in ascending input order — true, and already relied on
and tested in `_accumulate`'s own docstring, but worth stating before leaning on
it a second time. **The measured ceiling for that work is the 92 ms above.**

### 1.4 The one deliberate output change

`preview_primitives` returns the layout's index buffer — the triangulation the
drag began with — rather than re-triangulating the deformed mesh. For every mesh
of triangles and quads, and for every convex n-gon, that is the same array. It
can differ only for a concave n-gon deformed far enough during a drag to change
which corners are reflex.

It is nonetheless exactly right, because **the index buffer on the GPU is the
pre-drag one either way**: `update_vertices` does not touch the IBO. The old
code computed a possibly-different triangulation and threw it away. The drag's
commit goes through the ordinary `render_arrays` path and re-triangulates. This
is stated on `RenderLayout` and in `preview_primitives`' docstring, and the
"a reused layout answers for moved vertices" test excludes the index field by
name rather than by accident.

---

## §2 — Hover picking: two cheap fixes, and the number that says they are not enough

### 2.1 The overlay rebuilt itself when the cursor moved

`_view_overlay._SelOverlay` keyed on `(id(mesh), id(sel), mode, hover)` — and its
own docstring claimed that "a *hover* change is nearly free: it rebuilds one tiny
index buffer and touches nothing else." With hover in the key, that was false in
the way that costs most: crossing onto the next face released the whole overlay
and rebuilt it, re-uploading the position VBO **and** the guide wireframe's index
buffer, which is two indices per edge — 300 200 edges, **2.4 MB**, on this mesh.

| element mode | hover moves one element, before | after |
|---|---|---|
| vertex | 5.6 ms | **0.065 ms** |
| edge | 4.3 ms | **0.098 ms** |
| face | 5.0 ms | **0.72 ms** |

Hover is now out of the key and the hover draws are held in their own buffer
list (`hover`, `hover_specs`, `release_hover`), which is the only thing a hover
change touches. Face mode's remaining 0.7 ms is `tris[tri_face == hover]`, a full
scan of the triangle list; a face→triangle CSR would remove it and is not worth
the table yet.

### 2.2 The uncached conversion

`_view_pick.pick_face` converted `obj.mesh.positions.astype("f8")` per object per
call — the ray tests work in f8, the mesh stores f4. **1.09 ms** per object per
mouse-move, for an array on a frozen mesh. `adjacency.cached_positions_f8`
memoises it weakly and frozen, like every other memo against a `Mesh`: **0.0003
ms**.

### 2.3 Neither of these is the defect, and the honest number says so

The two fixes together recover **≈6 ms**. A hover mouse-move in an element mode
costs about **55 ms** on this mesh, and **48 ms of it is `picking.ray_triangles`**
— an unindexed test of the ray against all 199 712 triangles, run through
`pick_element` on every motion event with no button down.

So P2's two cheap fixes are real (a 2.4 MB upload per mouse-move is not nothing,
and it was a cache key asserting a property it did not have), and they move the
frame from ~55 ms to ~49 ms. **They do not make hover picking interactive.** The
thing that would is the spatial index the standing ruling defers, and that ruling
is unchanged by these numbers — they only make it precise: the BVH is worth
48 ms of a 49 ms frame at 200k triangles.

---

## §3 — Terrain: one painted cell cost the map, not the brush

`terrain._retile_into` recomputed `rank_field` over the whole layer and then ran
`blob.indices_from` over the whole layer once per terrain, before slicing out the
box it was asked about. A terrain drag calls `paint_terrain` once per cell along
the interpolated line — three to eight cells a frame.

The masks genuinely need a ring of neighbours around the box, which is why every
caller already grows the box by one. They do not need anything further away. The
pass now runs over the box grown by one more ring and discards that ring's own
cases. Where the grown window meets the map edge, the padding `blob.masks_from`
applies there *is* the map edge, so `outside` lands the same way — the window is
an optimisation, not a rule change, and `tests/plotter/test_terrain.py` asserts
that against a literal copy of the old whole-layer pass at all four corners, all
four edges, one-cell boxes and the whole map, for both values of `outside`.

Five terrains, all five present, one painted cell:

| map | before | after | speedup | a 5-cell frame, after |
|---|---|---|---|---|
| 32² (the dialog's default) | 0.05 ms | 0.05 ms | — | 0.2 ms |
| 64² | 0.14 ms | 0.05 ms | 2.8× | 0.2 ms |
| 128² | 0.34 ms | 0.05 ms | 6.8× | 0.2 ms |
| 256² | 2.05 ms | 0.05 ms | **41×** | 0.3 ms |
| 512² (`plotter_setup.MAX_TILES`) | 7.90 ms | 0.36 ms | **22×** | 1.8 ms |

### 3.1 It did not clear its gate at the default size, and that is the point

At 32² — `MapDoc`'s default and the "Standard" preset — the whole-layer pass cost
0.05 ms and there was nothing to fix. The defect only exists on a large map, and
the review's "~50 whole-layer passes per painted cell" over-counts: the
`chosen.any()` guard already skipped every terrain not present in the box, so a
two-terrain map ran two passes, not five.

What is left at 512² is no longer the retile: **0.30 of the remaining 0.36 ms is
`work = np.array(data)`**, the whole-layer copy each of `paint_terrain_cells` and
`erase_terrain` takes so it can compare and produce a region. Cutting that means
working on a windowed copy and translating coordinates through `_finish` and the
`Region` protocol — a real change to a shared shape for 1.5 ms a frame at the
largest map the app will make. Not taken; recorded here so the next person has
the number.

---

## Verification

* `uv run pytest` — **8614 passed, 22 skipped** (8583 before this batch: 31 new
  tests).
* `WARLOCK_NATIVE=0 uv run pytest` — **8246 passed, 390 skipped** (8215 before).
* `uv run ruff check .` clean.
* No native code changed; the ABI stays at 8.
