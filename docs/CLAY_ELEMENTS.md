# Clay → Wings3D-class modeler

The plan being executed. Element editing (vertex/edge/face modes and a full op
set), GLB import with passive UV/texture carriage, and a Wings3D-style RMB
context menu. Status is tracked by the checkboxes; a task is done when its
implementation *and* its tests are in and the suite is green.

**Deferred, out of scope:** magnets/soft selection, tweak mode, edge loop/ring
selection, virtual mirror, UV editing/unwrap, non-destructive smooth preview,
quad-merge on import, repair-in-place (a re-export mints a new library row).

## Cross-cutting contracts

**CSR stays canonical.** No winged-edge storage. Adjacency, triangulation and
boundary loops are derived, cached artifacts keyed by the immutable mesh via
`weakref.WeakKeyDictionary` (`Mesh` is `eq=False`, so identity-hashable and
weak-referenceable). Never a plain `id()`-keyed global dict — id reuse is a
use-after-free. A cached value must not hold a reference back to its `Mesh`.

**One selection type.** `clay/elements.py`:

```python
@dataclass(frozen=True, eq=False)
class ElementSel:
    verts: np.ndarray   # (n,)  i4 sorted unique
    edges: np.ndarray   # (m,2) i4 low-vertex-first rows, lexsorted unique
    faces: np.ndarray   # (k,)  i4 sorted unique
```

Arrays copied and read-only (the `Mesh` idiom); replaced whole, never mutated,
so `id(sel)` keys the overlay VBO caches. **Edges are vertex pairs, not indices
into `edges()`** — canonical edge ids renumber globally on any topology change;
pairs survive and map on demand via `searchsorted`. Helpers: `empty`,
`is_empty`, `affected_verts`, `convert` (mode switch), `combine`
(replace|add|subtract). `OpError(ValueError)` lives here: a user-facing refusal
shown as a toast, with no edit recorded.

**One op signature.** Every element op is pure:
`op(mesh, sel, **params) -> tuple[Mesh, ElementSel]` — the mesh plus the
selection the UI shows next (extrude → the caps, bevel → the new faces, delete
→ empty). Refuse via `OpError`. Non-manifold policy per op: local/additive ops
(extrude, inset, subdivide, delete, flip, weld, CC) are best-effort;
merging/walking ops (dissolve, fill hole, bevel) refuse with a message naming
the element.

**Selection stays non-undoable** (the existing invariant). Reconciliation:
`ClayDoc.undo()`/`redo()` peek the step (`UndoStack.top`/`redo_top`, new
accessors) and drop `element_sel` for uids touched by mesh/object edits *only* —
a `TransformEdit` or a rename undo clears nothing. `set_mesh` gains
`select: ElementSel | None`, applied as a non-undoable side effect after the
`MeshEdit` push.

**Element mode is per-document** (`ClayDoc.element_mode`, `ClayDoc.element_sel:
dict[uid, ElementSel]`), not on `ClayState` — mode is the interpretation key for
the selection, which lives on the doc; an app-level mode would reinterpret every
tab's selection on a tab switch. Invariant: in element modes
`doc.selection == {uid: non-empty element_sel}` (the Wings3D body-selection
model), so `frame_selection`, props `_selected` and `world_bounds` keep working
unchanged. Neither mode nor selection is serialized.

**UV is per-corner and optional.** `Mesh.uv: np.ndarray | None`, `(L, 2)` f4 —
one row per face corner (a seam is two corners at one vertex). Every op states
its UV policy: **preserved** (delete/flip/dissolve/weld — travels with the
corner), **interpolated** (subdivide/CC/bevel/loop-cut/inset — a per-face lerp
of that face's own corner uvs, so it is seam-safe), **inherited** (extrude walls
copy their source corners).

**Ops registry** (`studio/clay_ops.py`, imgui-free): `Op(name, label, modes,
run, enabled, key, separator_before)` + `register()`. The RMB menu
(`panes/clay_menu.py`), the tools-pane buttons and `handle_key` op dispatch all
read the same registry — one source of truth for what is invocable per mode.

**Layering unchanged:** nothing under `clay/` imports imgui, moderngl, pygame or
`service`; `clay_mode.py` and `clay_ops.py` know no imgui; only panes and main
draw.

## Phase 0 — UV foundation

- [x] **T1 — `Mesh.uv` core.** Field + `_DTYPES["uv"]="f4"`; `__post_init__`
  skips `None`, else copies/coerces/freezes; `validate` requires `(L, 2)` when
  present. `edits.mesh_bytes` skips `None` fields. `render_arrays` → 4-tuple
  `(positions, normals, uvs, indices)`: `uv=None` keeps the existing
  split/share path byte-identical with `uvs=None`; `uv` present emits one
  render-vertex per corner (positions `[loops]`, uv verbatim, the same
  flat/smooth normals via the existing `np.add.at` accumulation), indices are
  the fan corners directly. `document._submesh` slices `uv` with the corner
  spans and the per-face gather becomes offset arithmetic (the worst hot spot
  for imported meshes). `to_primitives` passes `uvs` into `gltf.Primitive`.
  `transformed` carries uvs through and reverses them with the loops on a
  mirror.

## Phase 1 — Topology substrate

- [x] **T2 — `clay/adjacency.py`.** Corner-based half-edge over CSR, all
  vectorized (one O(L log L) sort): `next_corner`, `prev_corner`, `corner_face`,
  `edge_verts`, `corner_edge`, `edge_uses` (1=boundary / 2=interior /
  ≥3=non-manifold), `twin` (−1 on boundary/non-manifold/flipped),
  `vc_starts`/`vc_corners` (vertex→corner CSR), `flipped_pairs`. Cache: a
  module-level `WeakKeyDictionary[Mesh, Adjacency]`; arrays read-only;
  `Adjacency` holds no `Mesh` reference. Also `mesh.cached_triangulation(mesh)`
  (same idiom) — `ClayView.pick` and the overlays consume it. Tests: table
  correctness on box/plane/cylinder, cache identity (same mesh → same object,
  dead mesh → collected via `gc`).
- [x] **T3 — `boundary_loops` + `check_manifold`.** Boundary rings as ordered
  vertex arrays oriented in the hole direction (a cap wound with the returned
  order is consistent); pinched vertices (two outgoing boundary edges) reported
  and refused by fill-hole. `ManifoldReport(boundary_edges, nonmanifold_edges,
  flipped_edges, repeated_corner_faces, duplicate_faces, unused_verts, .clean)`
  — a report, not a gate; ops read `Adjacency` flags directly, UI and tests read
  the report.
- [x] **T4 — `clay/earclip.py` + `triangulate` dispatch.** Vectorized concavity
  screen (corner cross · Newell normal, `reduceat`-min per face); non-suspect
  faces keep the fan fast path (imported tri/quad soup never leaves it, and
  convex output stays byte-identical, so the existing tests pass). Suspect
  faces: project onto the best two axes, O(n²) ear search, tolerant of
  collinear/repeated corners, falling back to a fan if stalled — rendering never
  raises. `render_arrays` uses the same dispatcher so shading and picking agree.
  Retrofit `transformed`'s per-face Python loop-reversal with a vectorized
  `reversed_corner_perm(starts)`.
- [x] **T5 — `clay/elements.py` + `clay/topo.py` + the first ops.**
  `ElementSel`/`convert`/`combine`/`affected_verts`/`OpError` (see contracts).
  `topo.py`: `rebuild` (every op funnels through it, so rewriting loops forces
  an explicit uv decision), `take_faces`, `compact_vertices`, `splice_corners`
  (ragged mid-loop insert — serves subdivide cracks and loop-cut ends),
  `region_boundary_corners`, `reversed_corner_perm`. Prove the pipeline with the
  two simplest ops in `clay/ops_topo.py`: `delete_faces` (complement take +
  compact, uv preserved) and `flip_normals` (perm gather on `loops` and `uv`).
  Shared test helpers extracted to `tests/clay/topo_asserts.py` (directed-edge
  uniqueness, orientation — lifted from `test_primitives`, re-exported not
  moved).

## Phase 2 — The op set

T6–T8 are independent after T5. Order is easy → hard.

- [x] **T6 — `extrude_faces` + `inset_faces`** (`ops_topo.py`). Extrude is
  region-aware: caps keep their face indices, arity, material, smooth flag and
  uv verbatim; a new vertex block for the used verts; wall quads
  `[a, b, new_b, new_a]` per region-boundary corner (winding per
  `primitives._side_quads`); **zero offset by default** — the interactive flow
  is extrude-then-drag the returned cap selection with the W gizmo. Inset
  per-face (no shared ring verts between faces; inner corners lerp toward the
  face centroid in position *and* uv) and region (extrude@0 + a rim pull toward
  local centroids — a documented approximation). Best-effort on non-manifold.
- [x] **T7 — `weld` + `collapse` + `fill_hole`.** Weld: eps-grid quantize +
  27-neighbour union-find (selection-sized Python is acceptable; the all-verts
  path stays gridded); representative at the cluster centroid; the aftermath is
  shared with collapse — remap loops, drop consecutive-duplicate corners, drop
  <3-corner faces, compact. Doubled faces that welding creates are left and
  reported by `check_manifold` (deleting silently would remove visible
  geometry). UV preserved per corner, so seams survive. Fill hole: the seed edge
  must be a boundary; the ring comes from `boundary_loops`; one n-gon wound in
  the hole direction; refuses pinched and figure-eight rings; uv copied from the
  adjacent boundary corners (a documented placeholder).
- [x] **T8 — the dissolve family** (`ops_dissolve.py`). Component-merge core:
  removable interior edges → flood-fill components over `twin` → boundary ring
  chained head-to-tail → one n-gon per component (boundary corners keep their
  uv, interior corners are dropped). Refuse: an annulus (a face with a hole is
  unrepresentable), a bowtie ring, a non-manifold interior edge, boundary edges
  (for edge-dissolve), boundary- or non-manifold-touching verts (for
  vert-dissolve). Results may be concave — the first real ear-clip consumer.
  Kept 2-valence collinear verts are a stated limitation.
- [x] **T9 — `subdivide` (linear)** (`ops_subdiv.py`). Catmull-Clark topology
  with no smoothing: midpoint and face-centre verts, n quads per n-gon assembled
  vectorized; unselected neighbours of subdivided edges get midpoints spliced in
  (a pentagon, so no T-junction cracks) via `splice_corners`; material and
  smooth inherited; uv per-face means. Selection out: the child faces.
- [x] **T10 — `catmull_clark`.** T9's topology plus CC positions, all
  scatter-adds (`np.add.at`/`bincount`/`reduceat`, no per-face work). Face points
  are means; edge points interior `(a+b+F1+F2)/4`, with **boundary and
  non-manifold edges taking the midpoint** (the crease rule); vertex points
  interior `(Q+2R+(n−3)P)/n`, boundary-with-two-boundary-edges
  `(prev+6P+next)/8` (B-spline), corner and irregular kept. UV linear per face
  (not CC-smoothed, so seams stay exact — stated). Never refuses. Test the
  closed forms by hand on a cube, a torus and a plane grid.
- [x] **T11 — `loop_cut`** (`ops_bevel.py`). Quad-strip walk from the seed edge
  via `twin`, both directions; stop on a non-quad, a boundary, `edge_uses ≥ 3`
  or a closed ring. Cut verts lerped walk-oriented (so `t=0.25` stays on one
  side all the way round); crossed quads split in two; end faces spliced (no
  cracks); uv lerped per face. Selection out: the new ring edges as vertex pairs.
- [x] **T12 — `bevel_edges`** (`ops_bevel.py`) — the hard one, last.
  Slide-vertex fan reconstruction: per unbeveled incident edge a slide vert at
  `clamp(width/len, 0, .5)`, shared by both flanking faces so it is crack-free
  by construction; per both-beveled face corner a miter vert at
  `width/max(sin(θ/2), ε)` along the in-face bisector, clamped; a face-corner
  rewrite table (neither → 2 slides, one → 1 slide, both → 1 miter); a bevel
  quad per edge wound from f1's traversal; a vertex polygon iff ≥3 new verts at
  v. Reproduces the textbook checks: one edge at a cube corner → a pentagon and
  no polygon; three edges at a corner → a miter triangle; all twelve cube edges
  → 26 faces. Refuse: boundary edges, boundary vertices with ≥2 beveled edges,
  `edge_uses ≠ 2`, unclosable fans (bowtie). Stated limits: single segment,
  per-edge width clamp, in-face bisector miter, no self-intersection guard. UV:
  slides interpolated, miters keep the corner uv, new faces copy the nearest
  source corner (flat in UV, and honest about it).

## Phase 3 — Selection UI

Starts after T5; runs in parallel with Phase 2.

- [ ] **T13 — document layer.** `element_mode`/`element_sel` +
  `set_element_mode` (converting via `elements.convert`), `set_element_sel`
  (maintaining the `doc.selection` invariant; `remove_object` pops),
  `set_mesh(select=)`; `UndoStack.top`/`redo_top`; undo reconciliation (drop
  touched uids only, walking `CompoundEdit`); extend
  `tests/clay/test_serialize.py::test_a_restored_document_has_nothing_selected`.
- [ ] **T14 — `clay/pick.py` + face-pick refactor.** Pure numpy and
  headless-testable: `project` (one matmul), vertex/edge screen-space nearest
  (VERT 8px / EDGE 6px), occlusion as a depth test against the ray-picked
  surface plus a bias — which rejects far-side elements of closed meshes, picks
  both sides of open sheets, and picks rim elements on a silhouette click with
  no surface hit (the Wings3D behaviours, closed-form and testable). Marquee
  masks: a vert in the rect; an edge iff both endpoints; a face iff all corners
  — through-selection with no occlusion, because that is what a blockout marquee
  is for and id-buffer readback is rejected by `picking.py`'s own doctrine.
  `ClayView.pick_face` → `Hit(uid, t, face)` using `cached_triangulation`
  (finally consuming `tri_face`); `pick()` keeps returning a uid so existing
  tests pass; `_Entry` carries `tris`/`tri_face`/`edges`.
- [ ] **T15 — input rewiring.** `_press` button 3 → `_rmb_press` (menu on a <4px
  release; a drag does nothing); button 2 → pan; **RMB pan removed**. LMB
  dispatch order in element modes: gizmo hit → element pick (combine by
  modifier: none=replace, Shift=add, Ctrl=subtract; Alt+drag always orbits) →
  a surface hit with no element clears that object's selection → empty space:
  the Q tool is a marquee grab (zero-area clears), W/E/R orbit. Hover
  (`hover_element`) updated only in `_motion` with no grab, via a per-uid
  `_ScreenCache` keyed `(id(mesh), trs, cam, rect)` — no per-frame reprojection
  at rest. `handle_key`: 1/2/3/4 = vertex/edge/face/object (**4, not Tab** —
  imgui nav owns Tab; **not b** — the hand is on the number row); staged Esc
  (element sel → leave mode → object sel); mode-aware Ctrl+A; Ctrl+Shift+I
  invert; Delete in element modes dispatches the registry delete op and never
  falls through to deleting objects; Ctrl+D is object-mode only. Mods are read
  at press via `pygame.key.get_mods()` → add the autouse headless stub fixture
  to `tests/test_clay_view.py` (it raises headlessly; `test_clay_mode` has the
  pattern).
- [ ] **T16 — selection rendering.** `DrawItem` gains `depth: bool = False` and
  `point_size: float = 0.0`; the renderer draws depth-tested overlay items
  first, then depth-off as today; everything goes through the existing "solid"
  program (the gizmo idiom — no new shader). `_SelOverlay` per uid: one
  `pos_vbo` shared by index-buffer VAOs (all-verts/all-edges dim guides;
  selected verts POINTS red; selected edges LINES; selected faces a translucent
  fill from `tris[isin(tri_face, sel.faces)]` plus an outline; hover yellow),
  keyed `(id(mesh), id(sel), mode, hover)`; a hover change rebuilds only the
  tiny hover IBO; released on eviction and on `ClayView.release`. Face-fill
  z-fighting: a shrink-toward-eye bias baked into the model matrix, not
  `glPolygonOffset`. The marquee rectangle is drawn with
  `imgui.get_window_draw_list()` in `main._clay_viewport` (the pane layer).
- [ ] **T17 — element gizmos + live preview.** Gizmo at the `affected_verts`
  world centroid; Q shows no gizmo in element modes. Per-object
  `_ElementDrag(before_mesh, verts, local_positions, matrix, inverse)`;
  per-frame world affine (translate/rotate/scale with the existing
  `ops.snap_*`), applied `inverse @ W @ matrix` to the affected verts; the
  preview writes `entry.gpu.draws[i].vbo.write(...)` and `_SelOverlay.pos_vbo`
  in place — no new `Mesh` objects, no VAO rebuilds, `view.rebuilds` stays flat
  (assertable), and the byte size is unchanged because topology and material
  grouping are untouched. Release: one
  `doc.set_mesh(uid, replace(before, positions=final), select=sel)` per object
  (`before` = the press mesh, exactly one `MeshEdit` each); a zero-movement drag
  pushes nothing and evicts the previewed entries.

## Phase 4 — Ops UI

- [ ] **T18 — registry + menu + panes.** `studio/clay_ops.py` (`Op`, `OPS`,
  `menu(mode)`, `by_key(mode)`, `register`); move the Duplicate/Bake/Mirror/
  Delete/Frame bodies here (`clay_tools` and `_duplicate_selection` become thin
  calls); element-mode built-ins Select All/None/Invert. `panes/clay_menu.py`:
  `open_popup("clay-context")` on `view.menu_request`, `menu_item` rows with key
  hints and enablement, called from `main._clay_viewport` after the image.
  `clay_tools`: a mode row (Object/Verts/Edges/Faces buttons highlighting
  `doc.element_mode`), registry-driven action buttons, and wire the dead
  `state.grid` toggle through to `renderer.draw(show_grid=)` (an adjacent
  one-line fix). Outliner: Ctrl+click toggles, Shift+click ranges (anchor in
  `ClayState.outliner_anchor`). Props: an element-mode summary line ("vertex
  mode — 12 selected across 2 objects"); the frozen `generator is None` branch
  becomes reachable, needing no change.
- [ ] **T19 — register the element ops + parameter popups.** Register extrude
  (runs at 0 and selects the caps — the user drags), delete, dissolve, flip,
  fill hole, collapse, subdivide, smooth (CC). Ops with a parameter — bevel
  (width), inset (thickness/depth), weld (eps), loop cut (t), CC (levels, warn
  at ≥2 that it triples the mesh) — get a small imgui popup (the inker
  resize-popup idiom): value fields + Apply, remembering the last values on
  `ClayState`. The first topology op on a generated primitive sets
  `Obj.generator = None` via the op path in `clay_ops`, in one place. Every op
  run: `try: mesh, sel = op(...)` / `except OpError as e:
  ctx.toasts.error(str(e))`; on success `doc.set_mesh(uid, mesh, select=sel)`.

## Phase 5 — Import and textures

- [ ] **T20 — serialization v2.** `VERSION = 2` written unconditionally; the npz
  gains an optional `"uv"` member (the required five are unchanged; the reader
  defaults it to `None`); `textures/<n>.png` zip members (PIL PNG encode,
  `ZIP_STORED`, `_EPOCH` timestamps, so repeat saves are byte-identical within a
  process and PIL version), deduped by texture-tuple identity across the five
  slots (mirroring `scene.TEXTURE_SLOTS`); `scene.json` materials gain
  `"textures": {slot: index}` plus a top-level `"textures"` list (absent when
  untextured, so the JSON is v1-shaped apart from the version). v1 files load
  (uv `None`, no textures); >2 is refused; a missing texture member is refused
  (the half-read-is-worse-than-refused doctrine). The ~0.2–0.5s save-time PNG
  encode on the frame thread is accepted for v1 — saves are explicit.
  **Note:** `_MESH_FIELDS` currently drops `uv` silently on save. This task must
  land before T22 creates any.
- [ ] **T21 — `write_glb` textures.** `_Buffer.view_bytes`; `images` and
  `textures` arrays plus `baseColorTexture`/`metallicRoughnessTexture` under
  pbr, and normal/emissive/occlusion at material level; dedupe shared textures
  by tuple identity; correct the docstring's "Textures are not written at all".
  Round-trip test via `gltf.load`.
- [ ] **T22 — `clay/glbimport.py`.** `glb_to_claydoc(bytes, name) -> ClayDoc`
  via `gltf.load` (which preserves root and grounding transforms, decodes
  textures, and already refuses sparse and non-triangle modes). Refusals: skins
  ("This GLB is rigged…"), >2M triangles, no meshes. Per primitive → one `Obj`:
  a bitwise vertex merge (`np.unique` on 12-byte position rows — exporter-split
  corners are bit-identical, and tolerance welding is a repair tool, not an
  import default), per-corner uv kept (`uvs[indices]`), a smooth-per-face
  heuristic (no normals → smooth; corner·face normal > 0.999 → flat), materials
  deduped by loader identity into the palette, node world TRS decomposed onto
  the `Obj` (so the grounding transform survives the round-trip),
  `generator=None`, names deduped `.001`-style. Clean history, nothing selected.
  Tests built with `test_glbwrite`'s helpers plus `write_glb`.
- [ ] **T23 — round-trip + viewport verification.** The viewport pipeline
  (`_build` → `to_primitives` → `GpuPrimitive`/`GpuMaterial`) already uploads
  uvs and texture slots — verify with a GL smoke test, do not build. Fix
  `panes/clay_props.py` palette edits to copy the five texture slots into the
  replacement `Material` (today's fresh-`Material` rebuild would silently delete
  a baked texture). End-to-end test: an authored textured model → `write_glb` →
  `glb_to_claydoc` → `to_model` → `write_glb` → `gltf.load`, with uvs, texture
  pixels and world transforms equal.
- [ ] **T24 — entry wiring.** `clay_mode.import_glb_path(ctx, path)` /
  `edit_asset_in_clay(ctx, job)`; parse and merge on a task thread
  (`ctx.submit("clay-import:…")`); an `on_task_done` branch: >200k triangles →
  a confirm dialog ("Editing will be slow"), else adopt and switch to clay mode.
  `main._on_drop`: `.glb` routes to import (delete the refusal toast). The
  library card overflow menu gains "Edit in Clay" for done model rows: it
  prefers the `build.wblk` sidecar when present (reopening the authored
  document, closing the loop on a file written but never read back), else
  imports `model.glb` — the optimized, grounded, served mesh, not `source.glb`.

## Phase 6 — Docs and polish

- [ ] **T25 — manual + help.** `07-clay.md`: a new "Element modes" section
  (modes and keys, click and marquee modifiers, Alt-orbit, the RMB menu, and
  "element selection is transient — not saved, and an undo that changes geometry
  drops it"), a Transforming update (gizmos on elements, one step per object per
  drag), a Materials rewrite (baked textures carried, rendered and re-exported;
  a read-only chip; no painting), a new "Importing an asset" section, and delete
  the "dropping a .glb does not…" paragraph. `09-shortcuts.md`: 1/2/3/4,
  Ctrl+Shift+I, mode-aware Delete/Esc/Ctrl+A, a mouse paragraph (LMB
  orbit/Alt-orbit/marquee, MMB pans, RMB context menu), and delete the "1/2/3
  reserved" sentence. `main._shortcuts_popup` Clay rows;
  `studio/manual/targets.py` anchors. Gated by `tests/manual/test_docs.py`.
- [ ] **T26 — scale + polish pass.** Timing on a 100k-tri synthetic soup:
  adjacency build, triangulate, subdivide, and a marquee over 150k verts —
  assert no per-face Python path trips. A refusal-message wording sweep. A
  docstring truth pass (`mesh.py`'s `triangulate` promise fulfilled; `ops.py`'s
  "extrude belongs here" pointed at the new modules). Full suite and ruff.

## Known numbers and accepted costs

- A 100k-tri imported mesh is ≈4.9 MB, so a `MeshEdit` pair is ≈9.8 MB → about
  19 undo steps in the 192 MB budget (`UNDO_BYTES`/`UNDO_MIN_DEPTH` at
  `studio/undo.py:39-41`). A 500k-tri source-class mesh pins ≈380 MB RAM at
  `UNDO_MIN_DEPTH=8` — accepted for v1 as "usable but slower"; per-mesh delta
  encoding is the named follow-up.
- A full CPU rebuild plus GPU re-upload per committed `MeshEdit` is sub-50 ms at
  `model.glb` scale (per edit, not per frame) — accepted; partial updates are
  deferred. Live gizmo drags bypass this entirely via VBO writes.
- Bevel is single-segment, refuses boundary edges and miters via the in-face
  bisector — stated limits, not bugs.

## Verification

- `uv run pytest` and `uv run ruff check .`. The suite stood at **2339 passed,
  7 skipped** before T1 and **2352 / 7** after it. (Worktrees show fewer without
  the three `WARLOCK_*` env vars for the vendored paths — see the
  `warlock-worktree-setup` memory. This plan is being executed on `master`.)
- New test files: `tests/clay/test_adjacency.py`, `test_earclip.py`,
  `test_elements.py`, `test_ops_topo.py`, `test_ops_dissolve.py`,
  `test_ops_subdiv.py`, `test_ops_bevel.py`, `test_pick_elements.py`,
  `test_glbimport.py`, `tests/test_clay_ops.py`, plus
  `tests/clay/topo_asserts.py` helpers; extensions to `test_mesh`,
  `test_document`, `test_serialize`, `test_clay_view`, `test_clay_mode`,
  `test_glbwrite`.
- Manual gate: `tests/manual/test_docs.py` must pass after the 07/09 edits.
- Interactive check (end of Phase 4 and Phase 5): `uv run warlock` → Clay → box:
  press 3, select faces, RMB → Extrude, drag with W; bevel an edge with a width
  popup; Ctrl+Z restores and drops the selection. Import check: drop a library
  `model.glb`, confirm the textures render, weld and fill-hole, Export to
  library, and open the exported asset in 3D mode with the textures intact.
- Commits follow the `Warlock v0.0.8` subject convention (no version bump unless
  asked).
