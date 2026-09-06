# Union framing: how far a posed character leaves its rest bounding box

2026-09-05. `blender_worker.op_sheet` used to frame the ortho camera once from
the **rest** bounding box and only spin it. Every pose whose apex leaves that
box therefore rendered clipped, on every cell of that run, with nothing in the
sheet to say so. This document is the measurement behind replacing that with a
union pre-pass over every distinct `(pose, frame)` the sheet contains, and the
record of the rendered-scale change that comes with it.

## What was measured, and on what

The fixture is the one `tests/test_sheet.py`'s gpu section renders: a UV sphere
scaled to `0.3 x 0.2 x 1.0`, auto-rigged with `op_rig` against the shipped
`humanoid` template, exported and re-imported exactly as a real sheet's rig is.
The **poses are the real thing** — `clips.expand_clips("humanoid")` over the
default Troupe layout, i.e. the same 32 frames a character sheet renders.

That is the honest limit of this table: the skeleton, the clip library and the
skinning path are real, and the *subject* is a capsule rather than a generated
character. A real character's limbs reach further from its centre than a
capsule's do, so the horizontal numbers below are a floor and the vertical ones
are close to right (the jump's apex is the root translation, which does not
depend on the mesh).

Blender 5.2, `bpy` in-process, no rendering: the pre-pass is depsgraph
evaluation and bounding boxes, so the framing question is answerable without a
card. Reproduced with `blender_worker._pose_union` + `_union_framing` directly.

## Rest versus union

Rest box: `z ∈ [-1.0000, +1.0000]`, half-extents `0.30 x 0.20`.
Rest framing extent (the old formula, `max(hypot(sx, sy), sz, 1e-6) * 1.12`):
**2.2400**.

| animation | union z | union x | union y | centre z | extent | vs rest |
| --- | --- | --- | --- | ---: | ---: | ---: |
| idle | −1.0000 … +1.0000 | ±0.2494 | −0.6012 … +0.1848 | +0.0000 | 2.2400 | 1.0000 |
| walk | −0.9820 … +0.9994 | ±0.2708 | −0.6631 … +0.4315 | +0.0000 | 2.2400 | 1.0000 |
| run | −0.9681 … +0.9794 | ±0.2903 | −0.6508 … +0.7563 | +0.0000 | 2.2400 | 1.0000 |
| attack | −0.9993 … **+1.0175** | −0.4533 … +0.4303 | −0.6999 … +0.2728 | +0.0087 | 2.2596 | 1.0087 |
| jump | −0.9621 … **+1.2545** | ±0.2494 | −0.7049 … +0.4765 | +0.1272 | 2.5250 | 1.1272 |
| all five | −1.0000 … +1.2545 | −0.4533 … +0.4303 | −0.7049 … +0.7563 | +0.1272 | 2.5250 | 1.1272 |

Two rows leave the rest box vertically. **Jump** reaches `+1.2545` against a
rest top of `+1.0000` and a rest *window* top of `+1.12`: 0.1345 above the
frame, or 12% of the frame's upper half cut off at the apex — visible, and
exactly the defect the union fixes. **Attack**'s wind-up leaves the rest box by
1.75% and stays inside the 12% margin, so it was not clipped; it is in the
union because "inside the margin" is not a property anyone chose.

Horizontally nothing is close: the widest reach from the orbit axis is
`hypot(0.4533, 0.7563) = 0.882`, so `2r = 1.76` against a vertical span of
2.25. Height decides the window on this fixture, and on a wide-armed character
the same `max` picks the other term with no change to the rule.

## What this changes in a published sheet

A sheet containing a **jump** is framed 12.7% wider than it was, so its subject
renders 11% smaller in the cell — deliberately, because the alternative is the
apex cut off. Sheets whose animations all stay inside the rest box (idle, walk
and run, at every frame measured here) are framed to the same extent they
always were and their pixels are unchanged.

The pixeliser's byte-identity bar is **run to run, not version to version**:
two renders of the same sheet from this build must agree bit for bit, and that
still holds. A sheet rendered before today and re-rendered after it will differ
where the union differs, and this document is the record of why.

One further sub-pixel change ships with it. `_scene_bounds` now transforms its
corners in double precision (`_transform`) rather than through `mathutils`,
which is single: `_union_framing` takes a `max` over corners from both the rest
box and the posed union, and two arithmetics for one corner is a difference
nobody could later account for. The rest box moves by about a part in 10⁷ —
`1e-7` of a 2-unit subject, orders below a pixel at 512px.

## Still owed, on a real card and a real character

- The same table for a **generated** character mesh (the horizontal columns are
  the ones the capsule understates), across at least the three shipped body
  variants.
- How often the reframe retry actually fires end to end. The retry measures the
  packed, pre-quantisation trims, so it answers "did the silhouette touch the
  edge", which a capsule cannot exercise usefully: it is smooth, and a real
  character's fingers and weapon are what get to an edge first.
- Whether a socket's `reach` sphere ever dominates the window in practice. No
  door writes sockets yet, so the only evidence today is the unit fixture in
  `tests/test_sheet.py::test_sockets_are_projected_per_cell_with_a_depth_order`.
