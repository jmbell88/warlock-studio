# Plotter–Tiled Parity Master Plan

**Goal:** Bring Plotter to safe-interchange parity with Tiled 1.12.2 for typical
projects — richer layers/objects/tilesets/properties first, then infinite maps and
projections, then authoring workflow, then project-scale features — without breaking
the existing 339 passing Plotter tests, `.wmap` backward compatibility, or the studio's
invariants.

**Architecture:** Generalize the headless `src/warlock/studio/plotter/` document model in
place — a real layer tree with a `scene.py` resolver both renderers consume, tagged object
geometry with persistent Tiled ids beside process uids, one recursive property module
(`props.py`), one frozen `Tileset` class with a `tile_rect` indirection for collections, a
chunk store added beside the dense array (not under it), and a `Lattice` params object for
the new projections. TMX/TMJ keeps refuse-by-name-or-round-trip as the contract, machine-
checked by a feature matrix + fixture corpus; the studio layer stays a thin consumer.

**Spec:** The Plotter–Tiled Parity Roadmap (user-provided; compatibility target Tiled 1.12.2).

This is the program-level roadmap. Each milestone gets its own detailed task-by-task plan
when started (via `superpowers:writing-plans`), executed with
`superpowers:subagent-driven-development` or `superpowers:executing-plans`.

## Global constraints

- Tiled compatibility target pinned at 1.12.2 until a deliberate bump.
- Imported content is fully represented and round-tripped, or refused — no opaque
  pass-through fields that go stale after edits.
- The headless packages (`studio/plotter/` et al.) import no imgui/moderngl/pygame/`service`
  (import-pinning tests enforce this).
- Undo addresses content by uid, never index; one continuous gesture = one undo step.
- `.wmap` versions 1 and 2 stay readable; new writes are version 3; archive output stays
  deterministic.
- The current 339 Plotter tests keep passing throughout; refusal tests convert to positive
  round-trip tests as features land, never silently deleted.
- Never edit `src/` while pytest runs; full suite via `uv run pytest`; install via
  `uv sync --extra studio --extra text2image --extra rig`.
- Commit convention: `Warlock vN.N.N` (stay on the current version unless the user asks for
  a bump).

## Current state (verified by exploration, 2026-08-12)

**Engine** — `src/warlock/studio/plotter/`, 21 modules (~2,900 lines), pure: stdlib + numpy
+ lazy Pillow + `warlock.studio.undo` only, pinned by set-equality in
`tests/plotter/test_plotter_imports.py` (new modules must join the roster at `:131`).

- **Model:** `Layer = TileLayer | ObjectLayer` flat union (`_map_model.py`); no group/image
  layers, no offsets/parallax/tint/class. `MapObject` is rect|point only, pixel coords, no
  rotation/gid/polygon/text. `Prop` is a frozen 5-type value (string,int,float,bool,color),
  the tuple duplicated in `_map_model.py:38` and `tsx.py:45`. `Tileset` is frozen,
  atlas-only; terrains non-empty ⇒ 47-column blob set; `terrain_of = local // 47` (no stored
  terrain field). `TilesetRef` firstgids contiguous, append-only, never removed.
  `MAX_DIMENSION = 4096`.
- **MapDoc** (`tilemap.py`) = six mixins; public-method-records / `_`-hook-mutates keeps undo
  from recursing; `dirty = history.head != saved_head`; `renderorder`/`backgroundcolor`
  round-tripped but unhonoured.
- **Cells** are uint32; `FLIP_H 0x80000000` / `FLIP_V` / `FLIP_D`, `GID_MASK 0x1FFFFFFF`; hex
  120° bit `0x10000000` deliberately absent (refused by name in `tmx._finish`).
- **Undo:** 11 uid-addressed `Edit` classes in `edits.py`; edits own their arrays; stroke +
  object-edit sessions give one-gesture-one-step, closed idempotently at undo/redo and a
  canvas frame-top sweep.
- **`project.py`:** orthogonal + isometric only, free functions over 5 numbers, exact affine
  inverse; iso is placement-only (quads stay axis-aligned).
- **TMX/TMJ** (`tmx.py` 903 lines, `tsx.py`): reads CSV/base64/zlib/gzip/legacy tile-elements,
  host-supplied loader callbacks, refuses by name via `TiledUnsupported`; writes CSV-only,
  always external tilesets `/NN-*.tsx`+`.png`, byte-deterministic; emits `version="1.10"`
  `tiledversion="1.10.2"`. Wang sets recognise-or-refuse (exactly what `write_wangsets`
  emits). DTD-refusing XML door.
- **`.wmap`** (`wmap.py`): `VERSION=2` zip, 1980-epoch stamps, sorted manifest,
  byte-deterministic (pinned); stores indices never uids; forward-refusal + tolerant
  defaults, no migration machinery (`locked` was added without a bump, `wmap.py:170-178`).

**Studio** — `plotter_mode.py` (controller/facade) / `plotter_state.py` (imgui-free; 8
Tiled-lettered tools; brush = gid ndarray; selection = view-state cell rect, clamped at use,
not undoable) / `plotter_io.py` / five panes. `plotter_canvas.py` draws one imgui
`add_image_quad` per visible cell per layer (no composite; culling via `cell_bounds`;
`_TILESET_MEMO` on `tileset_epoch`; minimap cached on `(history.head, len(layers))`). Locks
enforced studio-side only. Workspace at `main.py:2850`.

**Tests/gates** — `tests/plotter/` 15 files (~301 tests) + `tests/test_plotter_mode.py`
(1,723 lines, FakeCtx/headless-pane pattern). `test_tmx_refusals.py` is the 26-case refusal
matrix. Manual gates: chapter list is a renumbering gate, help anchors + `help_button` ↔
`HELP_TARGETS` set equality both ways, prose ≤ 120 chars, every pane needs a help button or
exemption. `docs/INVARIANTS.md:111`/`:113` are the plotter invariants and must be edited when
one changes. `docs/manual/09-plotter.md:283-298` states the refusal boundary.

**Uncommitted tree** — entirely UX-05 (`studio/journal.py` crash-autosave generalisation, all
modes) + UX-06 (crash dialog); Plotter's share is small and additive. No plotter journal test
exists yet (`journal.py`'s docstring claims one).

## Milestone 0 — Commit the baseline

The working tree carries the UX-05/UX-06 journal + crash-dialog work. Before any parity work:
run the full suite, commit the current tree as its own `Warlock v0.0.21` commit(s) so parity
work is bisectable and reviewable. (A discipline test for `journal.py` is claimed by its
docstring but missing — fold into M0 or the first milestone.)

## Milestone overview

1. Compatibility contract (1.12.2 metadata, feature matrix, fixture corpus, refusal ledger)
2. Generalized document model + `.wmap` v3
3. Common-map interoperability (group/image layers, rich objects, property editors)
4. Tileset editor (per-tile metadata, animations, collisions, Wang sets, `.tsx`/`.tsj`)
5. Infinite maps + remaining projections (TileStore, chunks, hex/staggered/oblique)
6. Core authoring workflow (stamps, selections, layer ops, snapping)
7. Project-scale features (projects, templates, Automapping, worlds)
8. Extensibility (optional, out-of-process extension API)

**Per-milestone ritual (every milestone, no exceptions):** flip — never delete — the refusal
tests it obsoletes; move the matching `docs/PLOTTER_COMPAT.md` rows in the same PR; update
`docs/INVARIANTS.md` when an invariant sentence changes; update `docs/manual/09-plotter.md`
(+ `14-shortcuts.md` for new keys) within the existing chapter (no renumbering); add corpus
fixtures; register new engine modules in `tests/plotter/test_plotter_imports.py:131`; run the
full suite; commit as `Warlock vN.N.N` (current version unless a bump is requested).

---

## Milestone 1 — Compatibility contract

**Objective:** make the Tiled boundary stated and machine-checked before it moves: a published
feature matrix, a fixture corpus of real Tiled-1.12.2 TMX/TMJ pairs, a semantic comparator
later milestones gate on, and the version-metadata story settled. No behaviour change to what
loads or refuses.

**Exit gate:**

- `docs/PLOTTER_COMPAT.md` exists, one row per Tiled feature, each in exactly one state
  (round-trips / refused / preserved-verbatim); `tests/plotter/test_compat_matrix.py` fails if
  a refused row has no matching `TiledUnsupported` string in source, a round-trips row names no
  existing fixture, or a source refusal is missing from the matrix.
- Every pair in `tests/plotter/fixtures/tiled/` loads via both readers with
  `doc_facts(read_tmx(...)) == doc_facts(read_tmj(...))`, and export→re-read is semantically
  identical (`tests/plotter/test_fixture_corpus.py`).
- `tsx.TILED_VERSION == "1.12.2"`; the format-version attribute stays `1.10` (pinned by a test
  with the bump rule in a comment).
- All existing tests pass, except pinned-byte tests carrying the `tiledversion` string.

**Key decisions:**

- The semantic comparator is M1 work. `tests/plotter/_semantics.py` with
  `doc_facts(doc) -> dict` — everything a map *is*, minus process uids and byte encodings
  (projection, sizes, props, tileset facts incl. pixel hashes and terrains, per-layer facts
  incl. objects). Uid-free by construction; numeric comparison tolerant of float repr. M2 uses
  it to prove `.wmap` v3 lossless; M3's gate is built on it. Byte-determinism tests stay
  separate and stricter.
- Fixtures are files authored in Tiled 1.12.2 itself, committed under
  `tests/plotter/fixtures/tiled/` (basic-ortho, basic-iso, two-tilesets, objects-rect-point,
  typed-props, locked-layers, blob-terrain × `.tmx`/`.tmj` + tiny shared PNGs/`.tsx`). Tests
  supply loader callbacks against the fixture dir (engine stays pure). Refusal tests keep their
  inline-XML style.
- **Honest version bump:** `TILED_VERSION` (the targeted Tiled release) bumps to `"1.12.2"` in
  M1 after the corpus proves Tiled 1.12.2 opens our exports. `TSX_VERSION`/`MAP_VERSION` (the
  format version) moves only when we first emit a 1.12-only construct — that is M2's class/list
  properties; record what version Tiled 1.12.2 actually writes during fixture authoring and
  adopt that string in M2.
- The ledger is the matrix doc cross-checked against source refusal strings — no pytest-marker
  machinery.

**Work breakdown:** (1) comparator + corpus harness (`_semantics.py`, fixtures,
`test_fixture_corpus.py` incl. a wmap round-trip case); (2) matrix + ledger
(`docs/PLOTTER_COMPAT.md`, `test_compat_matrix.py`); (3) metadata bump (`tsx.py`
`TILED_VERSION`, pinned-byte expectations, manual cross-link to the matrix).

**Risks/dependencies:** none on other milestones; keep fixture atlases tiny (2–4 tiles); get
`doc_facts` right once — it is the substrate every later milestone extends.

---

## Milestone 2 — Generalized document model + `.wmap` v3

> **Executed 2026-08-14 (Wave 1), with two amendments discovered in flight.**
> (1) Tiled 1.12.2 has **no `list` property type** — its set is
> string/int/float/bool/color/file/object/class (+ project enums). `list` is
> modelled in `props.py` and stored in `.wmap`, but **refused by name at the
> TMX/TMJ doors both ways** (`a list-valued custom property`); the
> "TSX_VERSION bumps on first list emission" trigger below is therefore dead —
> the format-version bump waits for the corpus author to record what version
> string a real Tiled 1.12.2 writes. (2) Everything M2 modelled ahead of its
> writers is refused at the writer doors until M3 flips both doors together
> (rotation, draworder="index", the five exotic shapes, group/image layers,
> offsets/tint/parallax/class — see `docs/PLOTTER_COMPAT.md`).

**Objective:** the engine models what Tiled models — a layer tree (tile/object/image/group)
with per-layer class/tint/pixel-offset/parallax, objects with tagged geometry, rotation and
persistent ids, the full recursive property descriptor — and `.wmap` v3 stores all of it
deterministically while reading v1/v2. TMX/TMJ behaviour unchanged (interop is M3) except
properties, which flip here.

**Exit gate:**

- A `MapDoc` with nested groups, an image layer, all seven object shapes, rotations, and
  file/object/class/list properties survives `read_wmap(wmap_bytes(doc))` with `doc_facts`
  equality and byte-identical double save.
- Committed `tests/plotter/fixtures/wmap/{v1,v2}.wmap` open with correct defaults (v1 →
  orthogonal; v2 objects get fresh persistent ids; missing tint/offset/parallax → identity).
- Property refusal test flipped: file/object/class/list properties round-trip through
  TMX/TMJ/wmap; `TSX_VERSION`/`MAP_VERSION` bumps to Tiled 1.12.2's own format version on first
  list emission.
- Undo/redo over every new mutation (group add/remove with subtree, reparent, shape edit,
  rotation) with honest costs; full suite green.
- `docs/INVARIANTS.md:111` rewritten; imports pin updated (+`props`, +`scene`).

**Key decisions:**

- **Real tree, flat adapters at the seams.** `GroupLayer.children: list[Layer]`; `doc.layers`
  stays the root list. `LayerOps.layer(uid)`/`index_of` walk the tree; `tile_layers()` returns
  tile leaves in depth-first paint order (wmap `.npy` enumeration, `render_map`, minimap keep
  their loop shape). New engine module `plotter/scene.py` owns inherited-state resolution —
  both renderers iterate it (two-renderers invariant):

```python
@dataclass(frozen=True)
class Resolved:
    layer: Layer                    # a leaf: Tile/Object/Image
    offset: tuple[float, float]     # summed ancestor+own
    parallax: tuple[float, float]   # product
    opacity: float                  # product
    visible: bool                   # AND
    tint: RGBA                      # channel product
    locked: bool                    # OR (studio enforces; engine reports)

def resolve(doc: MapDoc, *, include_hidden: bool = False) -> list[Resolved]
```

- Hit-testing subtracts `Resolved.offset` before `_cell_under`/handle tests. Export composite
  treats parallax as 1.0 (a still has no camera; documented); the canvas applies it against the
  view origin. `LayerAddEdit`/`LayerRemoveEdit` gain `parent_uid` (subtree travels as one
  object, cost = subtree bytes); `LayerMoveEdit` becomes
  `(uid, before=(parent_uid, index), after=(parent_uid, index))` so reparenting is one step.
- **Persistent ids beside process uids.** Every layer and `MapObject` gains `id: int` (Tiled's
  persistent id); `MapDoc` gains `next_layer_id`/`next_object_id` (monotone, never reused, never
  decremented on undo — an object property may reference a deleted id). Exports stop minting ids
  per export (`itertools.count(1)` at `tmx.py:763` goes away) and write stored ids + the `next*`
  attributes. `.wmap` v3 stores ids, never uids; every read still mints fresh uids.
  `INVARIANTS.md:111` becomes: ".wmap stores Tiled ids and positions, never process uids — a uid
  is minted per process and means nothing in a file; the persistent id is an ordinary document
  field with its own monotone counters, which is what an object-typed property references."
  Stated cost: after add-then-undo, a re-save differs from saved bytes only in the `next*`
  fields — accepted; Tiled behaves identically.
- **Objects get a tagged shape;** `kind`/`w`/`h` become derived compat properties (keeps most of
  `test_plotter_mode.py` and the panes compiling through the transition):

```python
Shape = Rect | Point | Ellipse | Polygon | Polyline | TileShape | Text
# Rect/Ellipse(w,h); Polygon/Polyline(points); TileShape(gid,w,h — flips ride in gid);
# Text(text,w,h,family,pixel_size,wrap,color,halign,valign,bold,italic,underline,
#      strikeout,kerning)
```

- `MapObject` keeps `x`/`y`, gains `id`, `rotation: float`, `shape: Shape`. Shapes are frozen so
  `snapshot()` passes them through with no deep-copy cost. `ObjectLayer` gains `draworder`
  (`"topdown"`/`"index"`; list order is already the manual stacking order). `tmx._finish`
  validates `TileShape.gid` like cells.
- **One property model** in a new leaf module `plotter/props.py`, killing both duplicated
  `PROPERTY_TYPES` constants: `Prop(type, value, propertytype="")`, nine types
  (string,int,float,bool,color,file,object,class,list), `TiledUnsupported` moves here (`tsx.py`
  re-exports both names so existing imports keep working), plus the six codecs (XML/JSON/wmap
  read+write) replacing the three duplicated codec pairs. `file` → verbatim str (path resolution
  is the host's problem); `object` → int id (0 = none); `class` → recursive `dict[str, Prop]`
  with `propertytype` preserved verbatim even when unknown; `list` → `list[Prop]`. Class members
  written sorted — determinism holds.
- **`.wmap` v3 layout:** recursive manifest layer entries (`"type": "group"`, `"layers": [...]`);
  every layer adds id/class/tint/offset/parallax; image layers add `images/N.png` + repeat;
  objects add id/rotation/shape; top level adds `next_layer_id`/`next_object_id`; recursive
  property form. Tile-leaf `.npy` members stay flat depth-first. v1/v2 read via tolerant defaults
  (the `locked` precedent); v2 `kind` maps onto `Rect`/`Point`. Write v3 only.

**Work breakdown (each ~one PR):** (1) `props.py` + codec rewiring + refusal flip + version bump
+ minimal studio editor rows (full recursive editor is M3); (2) persistent ids
(`test_imported_object_ids_survive_a_tmx_round_trip`,
`test_new_objects_allocate_past_the_imported_ids`, `test_undo_of_an_add_does_not_reuse_its_id`);
(3) `Shape` union + rotation + draworder (engine only; refusals stand); (4) layer tree +
`scene.py` + edits + both renderers iterate `resolve` + minimap cache key gains tree shape
(`test_group_visibility_and_opacity_resolve_through_ancestors`,
`test_an_edit_inside_a_moved_group_still_lands_by_uid`,
`test_removing_a_group_removes_and_restores_the_subtree_as_one_step`); (5) `.wmap` v3 + v1/v2
fixtures; (6) docs sweep (`INVARIANTS:111`, manual, matrix).

**Risks/dependencies:** biggest churn is `test_plotter_mode.py` — the compat properties keep it
green; budget a follow-up to migrate call sites. Deep-subtree/image-layer edits are multi-MB —
honest costs, undo budget evicts. `scene.py` deliberately never assumes `(h,w)` storage (M5 seam
— flag in its docstring). Journal provider needs no change (`wmap_bytes`), but v3 must land
before autosave fixtures are versioned.

---

## Milestone 3 — Common-map interoperability

**Objective:** the constructs M2 models flow through TMX/TMJ both ways, draw, hit-test, and can
be authored: nested layers, image layers, every object shape with rotation and stacking, full
property editors. The refusal ledger visibly shrinks.

**Exit gate:**

- New corpus fixtures (nested-groups, image-layers, rich-objects, iso-rich-objects, class-props)
  authored in Tiled 1.12.2 pass TMX↔Plotter↔TMX and TMJ↔Plotter↔TMJ with `doc_facts` equality
  and byte-deterministic exports.
- Flipped: group layers, image layers, layer pixel offsets, ellipse/polygon/polyline/text objects
  (XML+JSON), tile objects, rotated objects.
- Re-asserted refusals: staggered/hex, infinite/chunks, zstd, templates, image-collection
  tilesets, foreign wangsets, `.tsj`, imageless embedded tilesets, DTD.
- Studio: multi-select with overlap cycling, arrow-key nudge, align/distribute as one undo step,
  grid/pixel snapping, rotation handles, stacking commands on `draworder="index"` layers, view
  toggles (names, shapes, reference arrows), "go to object", class/list editors preserving
  unknown `propertytype`s verbatim.

**Key decisions:**

- **Interop first, authoring second** — the milestone stages internally: (a) engine read/write +
  canvas render/hit-test (satisfies the semantic gate alone; the contract-bearing half), (b)
  authoring UX, (c) property editors.
- Readers/writers grow by **recursion, not duplication**: `_read_layer_list(parent, doc)`
  recursing on `<group>` / `"layers"`; one recursive `_write_layer` per format. Conditional
  attributes keep the diff-clean rule (`offsetx` only when nonzero, `tintcolor` only when set,
  `rotation` only when nonzero, `draworder` only when `"index"`).
- **Tile objects render through the existing quad machinery** — `Tileset.uv` + flip permutation,
  anchored bottom-left (Tiled's rule), rotated about `(x, y)` by permuting corners; rotation
  composes *after* orientation so transpose-then-mirror is untouched. Text objects render
  best-effort in imgui (correct box/alignment; fidelity limit stated in the manual — the data
  round-trips exactly).
- **Selection stays view state** (`INVARIANTS:113` unchanged in kind): `selected_object` →
  `selected_objects: list[int]`; overlap cycling walks the hit list; align/distribute/nudge-of-
  many = `doc.compound([ObjectPropsEdit, ...])`.
- **"Go to object":** engine gains `find_object(id) -> (layer_uid, obj_uid) | None`; pane centres
  + selects; unknown id shows "object #N (missing)" and the value is never zeroed on save.

**Work breakdown:** (1) container/decorated layers in TMX+TMJ + image-layer export
(`images/NN-name.png`) + `render.py` composites image layers; (2) object shapes in TMX+TMJ + iso
conversion of polygon/polyline points via `project.object_to_pixels` (verified against an iso
fixture authored in Tiled, not synthesized); (3) canvas render/hit-test per shape
(point-in-polygon, ellipse, rotated-rect via inverse rotation) + layers pane becomes a tree view
— extract `panes/plotter_objects_draw.py` (studio-side) rather than doubling the 1112-line
canvas; (4) authoring UX (vertex editing, ellipse/text creation, rotation handle, nudge,
snapping, stacking) through the existing object-edit session — no new undo mechanism; (5)
multi-select + align/distribute via `doc.compound`; (6) recursive `property_editor` (class
nesting with `propertytype` badge, list add/remove/reorder, file-as-text, object-ref with jump
button); (7) gate + docs (fixtures, `test_interop_roundtrip.py`, matrix, manual refusal section
rewrite, `INVARIANTS:113` multi-select sentence).

**Risks/dependencies:** depends entirely on M2; chunks 3–4 block M4's collision editor (reuses
shape authoring) — land them early. Iso polygon/text conversion is the likeliest
silent-corruption spot. New panes must satisfy the help-button coverage gate.

---

## Milestone 4 — Real tileset editor

**Objective:** tilesets stop being frozen atlas blobs: image-collection tilesets, per-tile records
(class, properties, probability, animation, collision), tileset-level metadata (drawing offset,
object alignment, grid/orientation, background colour, class), `.tsx` and `.tsj` read/write plus
embedded tilesets, generalized Wang sets with blob-47 as a preset, and safe tileset
removal/reorder with atomic gid remap.

**Exit gate:**

- A Tiled 1.12.2 tileset using collections, animations, collisions, per-tile properties,
  probabilities, and multiple mixed/edge/corner Wang sets round-trips through `.tsx`, `.tsj`, and
  embedded forms with `doc_facts` equality and deterministic bytes.
- Flipped: image-collection, embedded-wangset, foreign-TMJ-wangset, external-`.tsj` refusals, and
  the four per-tile refusals in `check_tileset_features` (animation, image, objectgroup,
  properties).
- `remove_tileset`/`move_tileset` rewrite every layer gid and every `TileShape.gid` in one undo
  step (`test_tileset_removal_remaps_every_gid_and_undoes_atomically`); painted cells of surviving
  tilesets render pixel-identical before/after.
- Palette shows animation and collision previews; canvas animates. `.wmap` v4 reads v1–v3.

**Key decisions:**

- **`Tileset` stays one frozen class; geometry gets one indirection, not a union** (a union would
  force branches into `resolve`/`tile_rect`/`uv`/palette/`render.py`/`wmap` and Packwright's
  `tsxout.py`). Collections are packed on import into a synthesized private atlas (deterministic
  shelf pack, order = local id); `Tileset` gains `tile_index` (per-slot rects; empty = grid
  arithmetic), `tile_ids` (slot → local id; empty = dense — Tiled collections keep holes),
  `sources` (per-slot original image paths, so collection export writes one file per tile and the
  synthesized atlas never leaks), `tiles: tuple[TileMeta, ...]`, `align`, `draw_offset`,
  `background`, `tileset_class`. **`tile_rect` becomes the single choke point** consulting
  `tile_index`; everything downstream inherits collections for free.
  `TilesetRef.last_gid`/`holds`/`local` consult sparse membership. Tiled 1.11+ sub-rectangle tiles
  fall out of the same field.
- **Per-tile records are frozen values:** `TileMeta(local_id, tile_class, properties, probability,
  animation: tuple[(local_id, duration_ms), ...], colliders: tuple[TileCollider, ...])` with
  `TileCollider(id, x, y, rotation, shape: Shape, obj_class, properties)` — reusing M2's `Shape`,
  not `MapObject` (colliders live inside a frozen tileset). `tileset.animation_frame(local_id,
  time_ms) -> int` is pure, used by the canvas (frame clock joins `_TILESET_MEMO` keying) and
  deliberately *not* by export (a still exports frame 0 — stated).
- **Every tileset mutation is a replace; count-changing mutations are a remap.** Relax
  `replace_tileset`'s same-count refusal for metadata/wang/tile-record edits. Removal/reorder get
  one new whole-snapshot edit (atomicity as a property of the type, not caller discipline),
  modeled on `ResizeEdit`:

```python
class TilesetRemapEdit(Edit):
    before_refs / after_refs                 # list[TilesetRef]
    before_layers / after_layers             # {layer_uid: whole array, _own-copied}
    before_objects / after_objects           # {obj_uid: TileShape gid snapshot}
```

- through new hook `_apply_remap(refs, layers, objects)` (bumps `tileset_epoch`). Remap is
  vectorized per layer under `GID_MASK` with flags OR'd back; removed set's gids zero;
  affected-cell count reported for the confirm dialog. Invariant rewrite: "firstgids are
  contiguous and append-only under ordinary editing; the only renumbering operations are remove
  and reorder, and each rewrites every gid on the map — layers and tile objects — in the same undo
  step, so no gid ever dangles."
- **Wang: represent everything, paint the preset.** New engine module `plotter/wang.py`:
  `WangColor(name, color, probability, representative_tile)`, `WangSet(name, wang_type, colors,
  wangids, properties)`. `Tileset.terrains` → `wangsets` (with `terrains` as a derived migration
  property). Recognition boundary moves from recognise-or-refuse to represent-fully: any
  well-formed wangset round-trips verbatim and deterministically. `tsx._expected_wangids` becomes
  `wang.blob_paintable(ws) -> int | None`; painting/`terrain_of`/precedence/47-blob brush work
  unchanged for preset-shaped sets; foreign sets show in the UI with the brush disabled and one
  sentence. The general greedy Wang filler (Tiled's `WangFiller` with probabilities) is the last
  chunk and explicitly allowed to slip — round-trip + editing parity is the safe-interchange
  requirement; a brush is workflow.
- **`.tsj` + embedded writing** live in `tsx.py` (`tsj_bytes`/`read_tsj`); `tmx_export(doc, *,
  embed_tilesets=False)` gains embedded emission and honours `ref.source` (an external tileset is
  referenced, not rewritten — the field has carried this since day one).

**Work breakdown:** (1) model (`TileMeta`/`TileCollider`/indirection fields/`wang.py`;
`blob_paintable` adapters; Packwright constructor parity — its tests are the canary, run in every
M4 PR); (2) `.tsx`/embedded interop + refusal flips + `rich-tileset.tsx` fixture; (3) `.tsj` +
export options + `plotter_io.py` loaders; (4) `.wmap` v4 (synthesized sheet + `tile_index` in the
manifest — one member, deterministic); (5) remap (`TilesetRemapEdit`,
`remove_tileset`/`move_tileset`, INVARIANTS + docstring rewrite,
`test_reorder_keeps_every_painted_cell_pixel_identical`,
`test_removal_zeroes_orphan_gids_and_reports_the_count`) — independent of chunks 2–4, can land
early; (6) studio tileset editor (`panes/plotter_tileset_editor.py`), palette upgrades (variable
tile sizes, previews), canvas animation clock, (stretch) general Wang brush. The editor pane edits
tilesets, never the brush — palette stays sole owner of what a stamp puts down (`INVARIANTS:107`).

**Risks/dependencies:** depends on M2 (`Shape`, props) and M3 chunks 3–4 (collision editing reuses
shape authoring); remap chunk only needs M2. `TilesetRemapEdit` cost = all atlases + all layers —
one remap can evict most of the undo budget (same accepted posture as `ResizeEdit`; say so in the
confirm dialog copy). `wang.py` stores wangids syntax-level, so hex/staggered (M5) doesn't block
on it.

---

## Milestone 5 — Infinite maps + remaining projections

**Objective:** a map may be infinite: painting allocates 16×16 chunks on demand at any integer
coordinate (negative included); `.wmap` stores only touched chunks; TMX/TMJ chunked maps
round-trip chunk-for-chunk against Tiled 1.12.2. Staggered-isometric and hexagonal projections
render, hit-test and round-trip, including the hex 120° rotation bit. **Oblique is descoped
permanently** — Tiled 1.12.2 has exactly four orientations; there is no oblique in TMX/TMJ, so the
roadmap item is parity with nothing (ledgered as permanent: not a Tiled feature).

**Exit gate:**

- Refusal cases `infinite`, `chunks`, `staggered`, `hexagonal` flip to acceptance in the same
  commits that add the features.
- A sparse map with content at (−2000,−2000) and (+2000,+2000) across 50 touched chunks paints
  interactively (a stroke allocates only the chunks under the brush — asserted by an allocation
  counter, not a stopwatch), saves a `.wmap` holding exactly 50 chunk members, and TMX round-trips
  against a Tiled 1.12.2 golden.
- Every orientation × stagger-axis × stagger-index has an inverse property test over a randomized
  lattice sweep, plus checked-in coordinate tables generated from Tiled 1.12.2.
- Two saves stay byte-identical; all older `.wmap` fixtures open; dense finite documents take zero
  behaviour change (existing engine tests pass except the mechanical `Lattice` signature update in
  `test_project.py`).

**Key decisions:**

- **A second store beside the array, not a rewrite under it.** The dense ndarray is load-bearing in
  five tested places (tools, edits, stroke snapshot, render, wmap); only the chokepoints branch.
  New pure module `plotter/store.py`:

```python
CHUNK = 16; MAX_COORD = 1 << 20; MAX_CHUNKS = 1 << 16   # ceilings refused by name
class DenseStore:   # aliases TileLayer.data, no copy; bounds = (0,0,w-1,h-1)
    read(x0, y0, w, h) -> np.ndarray      # dense copy; off-store reads 0
    write(x0, y0, block); cell(x, y); content_bounds(); chunks()
class ChunkStore:   # {(cx,cy): (16,16) uint32}; bounds = None; write allocates;
    ...             # all-zero chunks dropped; same signatures
```

- `TileLayer.store` — `DenseStore` aliasing `data` for finite layers (nothing indexing
  `layer.data` changes); `data is None` + `ChunkStore` for infinite docs. `MapDoc.infinite: bool`.
  `tools.py` survives 100% unchanged via **windowing**: the caller picks a working rect,
  `store.read(rect)` gives a dense window, the pure tool runs window-local, the returned `Region`
  translates by the window origin (`plotter_mode._layer_window(layer, rect)`). Flood on infinite:
  match window = `content_bounds ∪ seed` grown by one — a fill seeded in the void paints only the
  seed cell (documented + pinned).
- **Undo stays bounded — copy-on-write strokes.** `TilePatchEdit` untouched (negative `x0`/`y0`
  are already legal ints). On infinite layers `begin_stroke` records nothing; `stroke_write`
  snapshots each chunk before first touch; `end_stroke` assembles `before` from snapshots and
  pushes the same single `TilePatchEdit` over the union box. Cost ∝ what the gesture touched
  (`test_infinite_stroke_snapshots_only_touched_chunks`).
- **Negative coordinates:** `Region` unchanged (bounds check moves into the store);
  `MAX_DIMENSION` stays for finite maps; selection stays clamped-at-use (pass-through when
  `bounds is None`); resize is refused on infinite docs by name, replaced by `crop_to_content()` /
  `convert_to_fixed(x0,y0,w,h)` / `convert_to_infinite()` via one `StoreConvertEdit` holding both
  stores (the `ResizeEdit` honesty rule); minimap keys on content bounds; iso `_origin_x` premise
  dies for infinite (origin shift 0, canvas pans into negative map-pixel space — the view
  transform already allows it).
- **`project.py` grows a `Lattice` NamedTuple;** the five free numbers retire.
  `Lattice(projection, width, height, tile_w, tile_h, stagger_axis="y", stagger_index="odd",
  hex_side=0)`; every free function becomes `f(lat, ...)`; `ProjectionOps._lattice()` is the
  single funnel. Staggered/hex `cell_at` is **not** an affine inverse — it's Tiled's
  reference-point-plus-nearest-centre test, ported from `StaggeredRenderer`/`HexagonalRenderer`
  (total, deterministic, same tie-breaks); `INVARIANTS:111`'s "the inverse is exact rather than a
  hit test" gets scoped to ortho/iso in the same commit. `draw_order` generalizes to an
  inclusive-rect form (also what infinite needs); `renderorder` becomes honoured for orthogonal
  (currently preserved-but-ignored — parity requires honouring it).
- **Hex 120° bit is orientation-gated decode, not a new global flag.** `gid.ROTATE_120 =
  0x10000000`, `GID_MASK_HEX = 0x0FFFFFFF`; bit 28 stays illegal id space outside hexagonal maps
  and `tmx._finish`'s by-name probe is removed only for them (the exact pattern isometric
  followed). Hex rotation about the cell centre is drawable with `add_image_quad` (corners rotate,
  UVs stay) — no new canvas primitive; `render.py` composites via PIL rotate. Multi-tile hex brush
  arrangement rotation is descoped v1 with a stated manual line.
- **The quad canvas survives; a chunk-texture LOD is added behind a budget, not instead.** Culling
  already bounds work to the viewport (`_visible_range ∩ content_bounds`). Deep zoom-out breaks
  (~230k quads at 4px/tile on 1440p): (1) M5 core ships a visible-cell budget clamping minimum
  zoom to ≤ ~50k cells; (2) a final, profiling-gated PR adds a per-chunk texture cache — each
  visible chunk composited once via `render.py` keyed `(layer_uid, cx, cy, chunk_stamp)`, one quad
  per chunk. Not a third renderer: the LOD draws `render.py`'s pixels, so the two cannot disagree
  without the cache disagreeing with both (`INVARIANTS:111` sentence updated to say so).
- **`.wmap`:** this milestone bumps to v5 (reads v1–v4): `"infinite": true`, per-layer `"chunks":
  [[cx,cy],…]` sorted `(cy,cx)`, members `layers/{i}/{cx}_{cy}.npy`. (M2's v3 manifest reserves
  the `infinite`/`chunks` keys so the container doesn't churn — coordination item.)

**Work breakdown:** (1) `store.py` + finite adoption, zero behaviour change (`test_store.py`:
alias, zero-reads, dropped chunks, `MAX_COORD` refusal); (2) infinite `MapDoc` + stroke CoW +
convert/crop edits (`test_convert_to_fixed_is_one_undo_step`,
`test_undo_on_negative_coordinates_lands_where_it_was`); (3) serialization — wmap v5 + TMX/TMJ
chunk read (csv/base64/zlib/gzip) and csv chunk write; flip `infinite`/`chunks`
(`test_wmap_infinite_stores_only_touched_chunks`,
`test_tmx_chunks_round_trip_against_tiled_golden`); (4) canvas/minimap/state
(`test_visible_range_is_viewport_bounded_on_infinite_maps`,
`test_minimap_keys_on_content_bounds`); (5) `Lattice` refactor — its own zero-semantics commit (a
rename review), + rect `draw_order` + orthogonal `renderorder`; (6) staggered + hexagonal (ported
math, hex gid bits, orientation-gated probe, rotated quads, goldens; flip
`staggered`/`hexagonal`); (7) (gated) chunk-texture LOD with
`test_chunk_lod_matches_quad_path_pixelwise`.

**Risks/dependencies:** needs M2's container merged (chunks extend it) and M3's tmx plumbing;
staggered/hex edge behaviour at boundaries is mitigated by porting-not-deriving + golden tables;
the `Lattice` migration is wide — land it empty (no hex math) first; M6's selection/floating work
must land after PRs 1–2 here or it gets built against `layer.data` and redone.

---

## Milestone 6 — Core authoring workflow

**Objective:** capture, stamp libraries, random/probability stamps, shape gestures, symmetry, real
(mask) selections that float/move/transform as one step, select-by-tile/property, layer
duplicate/merge/flatten/multi-select/batch edits, snapping, status-bar hints — with
one-gesture-one-step and selection-is-view-state both intact.

**Exit gate:**

- Every new gesture is pinned by a mode test asserting exactly one history step; the tool-letter
  collision test still passes.
- A floated selection moved, flipped, and committed is one undo step; cancel restores pixels with
  zero history motion.
- A stamp saved to the library reopens in a fresh document with different tileset order and paints
  the same tiles (gid remap test).
- Merge-down and flatten each undo to the exact prior layer stack (uids preserved).
- Random stamp honours M4 probability weights under a seeded RNG.

**Key decisions:**

- **Selections become masks; they stay view state.** `Selection(x0, y0, mask: bool ndarray)` on the
  mode (a full mask is the rect fast path) — still not undoable, still clamped at use;
  `INVARIANTS:113` keeps its first sentence verbatim. `_constrained` generalizes: after
  `clip_region` cuts to the mask bbox, excluded cells restore from the layer window (`np.where` —
  the flood's keep-untouched idiom). `flood_fill` gains a mask overload applied to the match (the
  wall stays real). New pure helpers: `tools.match_mask(data, value)`, `tools.mask_region(before,
  region, allowed)`. Select-by-property is a mode-level walk over M4 per-tile properties feeding
  `match_mask`.
- **Floating selections are pure view state until commit** — no lift edit, no revoke.
  (Deliberately not Inker's `FloatingBuffer`: tiles are cheap to overlay; raster pixels are not.)
  `PlotterDoc.floating = {block, x, y, mask, src}` — the canvas draws source cells empty and the
  block at its offset; the layer is untouched, so cancel is `floating = None` (costs nothing,
  crash-safe). Commit pushes one step through a new engine door reused by symmetry, merge, and M7
  automapping:

```python
# _map_paint.py
def write_many(self, uid, regions: list[tuple[int, int, np.ndarray]]) -> bool
    # per-region TilePatchEdits, no-ops dropped, one CompoundEdit
```

- Flip/rotate while floating reuse `flip_brush_h`/`v`/`rotate_brush_cw` (arrangement + flags move
  together). Paste stays brush-based (Tiled's model, already shipped); float is the selection path
  — both stories stated in ch. 09.
- **Symmetry rides the stroke session.** Config (`none|x|y|xy|rotational`, pivot in half-cells) on
  mode state; `_apply` mirrors each region via new pure `tools.mirror_region_h/v(region, pivot2)`
  (doubled units so between-cell pivots are exact ints; blocks go through the brush flips so flags
  mirror with arrangement) and issues both `stroke_write`s inside the open session — the union box
  makes it one step. Zero engine change beyond two pure helpers.
- **Stamps are mini `.wmap`s.** Right-drag captures the layer window under the marquee as the
  brush. New engine module `plotter/stamps.py`: `stamp_bytes(brush, tilesets)`, `read_stamp(data)`,
  `remap_gids(block, src, dst_doc)` (matches tilesets by content identity — name + grid + pixel
  hash; appends missing sets via `TilesetAddEdit`; flags bit-exact), `weighted_pick(variants,
  weights, rng)` with the mode owning a per-stroke-seeded `random.Random`. Files at the data root
  under `plotter/stamps/*.wstamp` (journal's directory precedent); recents ride
  `studio/recents.py`.
- **Layer ops are compounds.** Duplicate = `LayerAddEdit` with a copy (fresh uid). Merge-down =
  `CompoundEdit([TilePatchEdit(lower, union content box, overlay), LayerRemoveEdit(upper)])` with
  pure `tools.overlay` (`np.where`, nonzero-of-upper-wins; opacity < 1 or non-tile layers refuse by
  name); flatten = fold of merge-downs in one compound; batch visibility/lock/props = one compound
  of `LayerPropsEdit`s; drag-reorder reuses M2's tree-aware `LayerMoveEdit`. Merge patches cover
  union content bounds, not the layer.
- **Snapping + hints are studio-only.** `state.snap` (`grid|half|pixel|bounds`), pure `_snap_point`;
  per-tool hint + live `(col,row)` in the existing canvas footer — no new pane, no new coverage
  exemption.

**Work breakdown:** (1) mask selections (`test_mask_region_keeps_excluded_cells`,
`test_subtract_selection_is_view_state_only`, `test_flood_respects_mask_walls`); (2) floating +
`write_many` (`test_float_move_commit_is_one_step`, `test_float_cancel_touches_no_history`,
`test_write_many_drops_noop_regions`); (3) capture + stamps + random
(`test_stamp_round_trips_bytes_identical`, `test_remap_matches_tilesets_by_content`,
`test_weighted_pick_honours_probability_under_seed`); (4) shape/symmetry gestures + terrain lines
(one re-fit region per stroke; `test_symmetry_stroke_is_one_step`, `test_terrain_line_refits_once`);
(5) layer ops + snapping + hints (`test_merge_down_undoes_to_prior_stack`,
`test_batch_visibility_is_one_compound`).

**Risks/dependencies:** PRs 1–2 need M5's store first; PRs 3–4 can run beside M5's projection
track; random stamps need M4 probability (fallback: uniform weights + ledger note).
`INVARIANTS:113` gains the mask + floating sentences (edited, not appended-around);
`14-shortcuts` rows for capture/float/symmetry.

---

## Milestone 7 — Project scale: projects, templates, Automapping, worlds

**Objective:** `.tiled-project` discovery with enums/classes/defaults consumed by the property
editors and preserved byte-safely; object templates (`.tx`) with inheritance/override/detach/
update; a headless Automapping engine matching Tiled's current conventions (one rule run = one
compound step, while-drawing mode); minimal world navigation.

**Exit gate:**

- A `.tiled-project` written by Tiled 1.12.2 loads, its enums/classes drive property-editor
  dropdowns, and an unchanged project saves back byte-identical (unknown keys verbatim).
- Template instances show inherited values live; editing the `.tx` updates undetached instances on
  explicit reload; detach is one undo step; TMX/TMJ template references match Tiled goldens. The
  `templates` refusal flips.
- The Automapping fixture suite (rule maps + expected outputs, several ported from Tiled's
  examples) produces cell-identical output; one rule run undoes in one Ctrl+Z; while-drawing
  automap undoes together with its stroke.
- A `.world` lists member maps and jumps between them; the file is never rewritten.

**Key decisions:**

- **Project state is cross-document: engine data, studio holder.** Pure `plotter/tproject.py`:
  `ProjectData(folders, property_types, extra)` where `extra` carries every unmodeled key verbatim;
  `read_project`/`write_project` (byte-stable, Tiled's key order)/`validate(doc, p) -> list[str]`
  (remedy sentences, not exceptions). Holder: `PlotterState.project` loaded by
  `plotter_io.open_project` (walk up from the opened map's directory — Tiled's discovery). The
  shared `property_editor` consults it for enum/class options; an unknown class is preserved and
  displayed opaque, never dropped.
- **Templates: resolved values + override set on the instance.** `plotter/template.py` (tsx-shaped
  XML). `MapObject` gains `template: str | None` and `overridden: frozenset[str]`; objects store
  resolved values (render/hit-test never chase a reference); writers emit only overridden fields —
  Tiled's exact semantics. `snapshot()` carries both, so `ObjectPropsEdit` covers detach and
  override edits for free. "Update instances" is an explicit command via one compound (no
  file-watcher — YAGNI).
- **Automapping: one pure module, rules-as-maps, edits assembled outside.** `plotter/automap.py`:
  `rules_from_map(doc) -> RuleMap` (`input_`/`inputnot_`/`output_` layers, rule regions, per-rule
  options from properties — Tiled 1.12 conventions; legacy pre-1.9 `regions` rule maps refused by
  name and ledgered), `read_rules_txt(text)`, and `plan(target, rule, region, rng) ->
  dict[layer_uid, list[Region]]` — pure: computes, never mutates, never pushes. The mode applies a
  plan through `write_many` into one `CompoundEdit` (missing output layers created inside the same
  compound). While-drawing: `end_stroke` gains a collect form — `end_stroke(into: list[Edit] | None
  = None)` — so paint and its automap consequence are one compound (Tiled's behaviour and the
  one-gesture rule); the automap region is the stroke's union box grown by the rule radius. Seeded
  rng; fixture tests pin outputs (`test_plan_is_pure` asserts the doc hash unchanged).
- **Worlds: navigate, never write.** `plotter/world.py` reads direct `maps` entries; the `patterns`
  regex feature is refused by name and ledgered. A world section in `plotter_bridge.py` lists
  members and jumps. No multi-map rendering, no world editing — input only, stated in ch. 09. Keep
  it this narrow.

**Work breakdown:** (1) `tproject.py` + discovery + editor consumption
(`test_unknown_keys_survive_verbatim`, `test_project_round_trip_byte_identical`,
`test_enum_options_reach_property_editor`); (2) `template.py` + `MapObject` fields +
create/place/detach/update + TMX/TMJ refs, flips `templates` (`test_overrides_only_are_written`,
`test_detach_is_one_props_edit`, `test_template_paths_are_posix_in_files`); (3) `automap.py` engine
+ `tests/plotter/fixtures/automap/` (`test_plan_matches_tiled_fixture_output`,
`test_probability_under_seeded_rng`); (4) automap in the mode (rules.txt via project,
run-on-map/region, `end_stroke(into=…)`, while-drawing toggle; `test_rule_run_is_one_compound`,
`test_while_drawing_joins_the_stroke_step`); (5) `world.py` + bridge navigation
(`test_patterns_refused_by_name`, `test_member_maps_listed_in_file_order`).

**Risks/dependencies:** PRs 1–2 need M2+M3; PR 3 needs only the engine and can start beside M6;
PRs 4–5 need M7.1 + M6's stroke work. `INVARIANTS:111` gains the automap-plans-are-pure and
template-resolved-values sentences.

---

## Milestone 8 — Extensibility (final, deliberately narrow)

**Objective:** a versioned, out-of-process format-adapter mechanism for custom
importers/exporters/validators — and an explicit, argued deferral of everything else (in-process
API, map actions, document mutation, Tiled plugin compatibility, any JS). A cross-process
"transactional document update API" means a merge engine built for an audience this tool doesn't
have — deferred, ledgered permanent-until-demand.

**Design:** manifest `plotter-adapters.json` (project folder or data root) parsed by pure
`plotter/adapters.py` — `Adapter(name, kind: export|import|validate, extension, argv with
{in}/{out} placeholders (no shell), format: tmx|tmj)`, version-gated with forward-refusal by name.
Execution is studio-side (`plotter_io.run_adapter`): export writes canonical TMX/TMJ to scratch,
spawns argv joined to the winjob (the scan test's rule), bounded timeout, staged output; import
runs the adapter to produce TMX/TMJ which enters through `tmx.py`'s existing door — the whole
refusal matrix guards adapter output too. Validators get `{in}` only, one-diagnostic-per-line
stdout surfaced as toasts. First run of an adapter shows the full argv in a confirm; acceptance
recorded keyed by a manifest-entry hash so an edited command re-asks. Documented as a ch. 09
section with two worked examples.

**Exit gate:** an adapter round-trip test using a checked-in Python adapter
(`tests/plotter/fixtures/adapters/csv_adapter.py`) passes; killing the app mid-export leaves no
orphan process (winjob scan test extended); unknown manifest version or an absolute-path escape in
`{out}` refused by name.

**Work breakdown:** (1) `adapters.py` + tests (`test_manifest_forward_refusal`,
`test_argv_placeholders_only`); (2) `run_adapter` + bridge UI + permission dialog + manual section;
(3) the ledgered deferral text in the compatibility matrix. Risk: scope creep toward "one little
in-process hook" — the ledger entry is the defence: widening must convert a named deferral.

---

## Program sequencing

```
M0 (commit journal/crash-dialog baseline)
 └─ M1 contract + refusal ledger
     └─ M2 document model + .wmap v3 ──┬─ M3 common-map interop (TMJ plumbing, objects, ext refs)
                                       └─ M4 tileset editor (per-tile metadata, Wang;
                                                             remap chunk needs only M2)
M5a store/infinite:        needs M2 (container) + M3 (tmx plumbing)
M5b projections (Lattice): needs M1 only (+ M3's writer for goldens)  — parallel with M5a
M6.1–2 selections/float:   needs M5a
M6.3–4 stamps/symmetry:    needs M4 probability only — parallel with M5
M6.5 layer ops:            needs M2 groups
M7.1–2 projects/templates: needs M2 + M3
M7.3 automap engine:       engine-only — can start beside M6
M7.4–5 automap-UI, worlds: needs M7.1 + M6 stroke work
M8 adapters:               needs M3 (canonical TMJ) — otherwise independent, last
```

**Serialization points** (wide, mechanical, land as single zero-semantics commits everything
rebases onto): the `Lattice` refactor (M5 PR 5) and store adoption (M5 PR 1).

**Named big-bang risks:** (1) M6 floating/masks built against `layer.data` before the store exists
— prevented by ordering; (2) the `Lattice` migration colliding with in-flight projection work —
land it empty first; (3) `.wmap` churn if chunk members are designed after the v3 container freezes
— M2's v3 manifest reserves the `infinite`/`chunks` keys; (4) the chunk-texture LOD ballooning —
explicitly profiling-gated.

**`.wmap` version ladder:** v3 (M2, tree/ids/props/shapes), v4 (M4, tileset extensions), v5 (M5,
chunks). Each version reads all earlier ones; write-latest-only; determinism pinned at every step.

## Test and fixture strategy

- `tests/plotter/fixtures/tiled/` — small maps authored in real Tiled 1.12.2, checked in as
  goldens (one per orientation × finite/infinite × TMX/TMJ, plus templates, projects, automap rule
  maps, Wang sets); a `FIXTURES.md` records the Tiled build + authoring steps so any fixture can be
  regenerated.
- Semantic equivalence against Tiled's output; byte identity only for our own writers (two saves
  identical — the existing pinned rule, extended to every new format). The `doc_facts` comparator
  (M1) is the shared gate substrate.
- Refusal fixtures stay minimal inline strings; each conversion moves the string into an acceptance
  test in the same commit as the feature.
- **Performance gates count, they don't time:** `test_sparse_paint_touches_only_written_chunks`
  (allocation counter), `test_visible_range_is_viewport_bounded_on_infinite_maps` (visible-cell
  count independent of content extent), `test_no_full_map_read_during_stroke` (instrumented store),
  one-memo-build-per-epoch regression for gid→atlas lookup; a single `@pytest.mark.slow` smoke
  (paint/undo/save on a 500-chunk map) as a canary, not a gate.
- Every model mutation is exercised through undo/redo, save/reopen, and the journal
  (crash-recovery) path; malformed-input + resource-ceiling tests for recursive groups, external
  references, collections, property lists, chunk payloads (the bounded-decompress precedent).

## Verification (every milestone close)

1. Refusal-ledger conversion done in the feature commits (never batched at the end);
   `docs/PLOTTER_COMPAT.md` rows moved in the same PRs.
2. `tests/plotter/test_plotter_imports.py` — new engine modules registered (`props`, `scene`,
   `wang`, `store`, `stamps`, `automap`, `tproject`, `template`, `world`, `adapters`); no new
   outward imports expected.
3. `docs/INVARIANTS.md` `:111`/`:113` — edited sentences, not appended contradictions.
4. `docs/manual/09-plotter.md` + `14-shortcuts.md` updated within existing chapters
   (`EXPECTED_KEYS` untouched all program long); help anchors resolve; new panes get `help_button`
   or a coverage exemption.
5. Fixtures + `FIXTURES.md` entries added.
6. Full suite: `uv run pytest` (never editing `src/` while it runs), including `tests/manual/`,
   scan tests, and Packwright's tests (the external consumer of `plotter.tsx`/`tileset` — the
   canary in every M4 PR).
7. Commit as `Warlock vN.N.N` at the current version; bump only when the user asks (their
   convention: patch bump, recorded in `pyproject.toml` — `tests/test_changelog.py` pins the
   `CHANGELOG.md` top entry against it, so a bump updates both).
