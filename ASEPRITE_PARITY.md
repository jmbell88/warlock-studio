# ASEPRITE_PARITY.md — the Aseprite-parity master program

**Status: Waves 0-5 DONE** (0-2 on 2026-08-17, 3 on 2026-08-18, 4 and 5 on 2026-08-19) — **the program's waves are complete.** The P1 backlog remains unscheduled by design (see "P1 backlog" below); items there are pulled into sessions individually, never waved.
**Progress is tracked by editing the wave status lines below (the PLOTTER_PLAN.md precedent). Each wave (or chunk) is a future session: its detailed implementation plan is written at execution time, arguing from this spec.**

Goal: make Inker a genuine Aseprite alternative. Six P0 gaps from the 2026-08-17 gap
analysis, ordered into five waves plus a landing wave. One P0 item — tablet/pen input —
is **explicitly dropped** (see Non-goals). The P1 list rides as an unscheduled appendix.

Two decisions taken with the user on 2026-08-17:

1. **ORA-first.** ORA remains the native and crash-recovery format and is extended to
   carry every new structure (index planes, tilesets). A `.aseprite` **writer** plus a
   round-trip corpus is Wave 5, after the document model has stabilized, together with
   an explicit lossy-interop report.
2. **No pen support.** The 2026-08-15 spike
   (`docs/measurements/2026-08-15-tablet-pressure-spike.md`) is the citable record: the
   vendored SDL2 exports no pen API, the one viable route is a Windows message hook,
   and the dev machine has no digitizer, so every pen claim would be recited rather
   than measured. The velocity taper stands unchanged.

## Rules of engagement (repo doctrine, restated so each wave's plan can cite one place)

- The suite is green after every chunk. A chunk that cannot end green is two chunks.
- **Standing negative control: an RGB document's bytes, history behavior, and ORA
  output never change in any wave.** The ORA byte-determinism pins are the proof.
- A refusal only ever moves because the editor learned to model the thing; the
  refusals that replace it are raised by name.
- The Aseprite-divergence numbering in `docs/INVARIANTS.md` is citable and
  append-only: amend text in place, never renumber, append new entries at the end.
- `docs/manual/09-inker.md` and `docs/INVARIANTS.md` are updated in the same wave as
  the behavior they describe, never later.
- One gesture = one Ctrl+Z; an op that changes nothing pushes nothing (or `dirty`
  lies). Edits own their data and are addressed by uid, never index.
- Headless engine packages import nothing from imgui/moderngl/pygame/`service`/the
  sibling engines; every import-pin change is made in the pin test *and* argued in its
  docstring *and* recorded in INVARIANTS.md.
- `vendor/` is gitignored: every native-kernel seam keeps its numpy fallback, and a
  change that is faster with the DLL but slower without it is rejected.

---

## Wave 0 — Landing zone + save safety — **DONE 2026-08-17** (`54753b7` + the commit after it)

Small, immediate, independent of everything below.

1. **Commit the uncommitted green batch** (bridge reorder, grid-32 + rulers, PS7
   layers, Packwright tileset import; green at 8876). Gated on the user's explicit go —
   the tree is theirs to sequence. Chunk 3.1 (tilegrid promotion) depends on this
   landing first.
2. **Save-safety fix.** Today `Document.load` stamps every non-ORA input
   `file_format="png"` (`src/warlock/studio/inker/document.py:292`) and `_write`
   dispatches on `tab.file_format`, never on the path suffix
   (`src/warlock/studio/inker_mode.py:1225-1239`) — so opening `foo.jpg` and pressing
   Ctrl+S silently writes PNG bytes into `foo.jpg`. JPG/WebP/BMP are advertised as
   openable (`src/warlock/studio/filetypes.py:33`).
   **Fix: force Save As when the backing file's suffix is not `.png`/`.ora`.**
   Re-encoding to JPEG would lose pixels silently; a refusal is more honest than a
   lossy write. `Document.load` (or the tab) records the true source suffix;
   `inker_mode.save` gains the gate; a toast explains ("this drawing came from a JPG;
   choose where to save the PNG/ORA copy"). `filetypes.py` stays as-is.
   Tests: open-jpg-then-save routes to Save As; PNG/ORA save behavior byte-identical.
3. Housekeeping riding along: `src/warlock/studio/packwright/__init__.py` stale
   "reaches outward exactly four times" docstring (the pin now holds seven entries);
   CHANGELOG entry for the tileset import.

**Gate:** suite green; a `.jpg` can no longer be silently overwritten with PNG bytes.

---

## Wave 1 — True indexed & grayscale color modes (P0 item 1) — **DONE 2026-08-17** (all six sub-waves)

### The architecture

**Storage: authoritative index plane on `Layer`, materialized RGBA beside it.**
`Layer` grows an optional `indices: (H, W) uint8` plane. When present it is
authoritative; `pixels` stays `(H, W, 4) uint8` **always** and becomes the derived
materialization (`pixels == lut[indices]`). The `(H,W,4)` hard invariant
(`layers.py:56-60`) is kept — demoted from "the record" to "the projection" on indexed
documents. Load-bearing consequences:

- The compositor (19 blend modes), the frame-flatten cache, texture upload, GIF/sheet
  flatten, thumbnails, and the native kernels change by **zero lines** — they read
  `pixels`, which stays valid RGBA.
- Linked cels share their index plane with zero new machinery, because a link is the
  same `Layer` object in two `Animation.cels` slots.
- Placeholders need nothing: the shared blank plane keeps `indices=None`; every write
  autovivifies a real cel first (`_ensure_cel_for`), which in an indexed document
  mints `indices` filled with the transparent index.
- Why not the alternatives: a parallel cel class forks every holder of `Layer`
  (`LayerStack`, `Animation.cels`, `CelSetEdit`, `ReplayEdit`, sessions, writers);
  reinterpreting `pixels` breaks every consumer at once. Constraint-on-write (today's
  mechanism) cannot deliver **identity** — two palette slots with the same RGB are
  indistinguishable in an RGBA plane, and identity is precisely what this wave adds.

**Document mode.**
```python
# document.py — new fields
color_mode: str = "rgb"            # "rgb" | "indexed" | "grayscale"
transparent_index: int = 0         # meaningful only in indexed mode
```
- `"rgb"` — today's document bit-for-bit, **including** today's constraint-on-write
  behavior when `palette` is set. Constraint-on-write is retained as an RGB-mode
  feature ("palette-constrained RGB"), not replaced: legacy "indexed" docs legally
  contain soft alpha, which a one-index-per-pixel plane cannot represent, so they open
  exactly as before and entering true indexed is always an explicit, undoable
  conversion.
- `"indexed"` — `palette` required, 1..**256** entries (>256 refused by name: the cap
  PNG-P, GIF and `.aseprite` share, and what makes the ORA representation possible).
  Palette entries keep RGBA (alpha survives where the `.gpl` projection cannot carry
  it). `transparent_index` materializes as `(0,0,0,0)`.
- `"grayscale"` — see below.
- `is_indexed` splits: `is_indexed` → `color_mode == "indexed"`; new
  `is_palette_locked` → `color_mode == "rgb" and bool(palette)`. Audit every studio
  reader of `doc.palette` truthiness (`inker_mode.py:1159/2627`, the palette pane) —
  GIF/gpl export paths read `bool(palette)`, which covers both.

**Write path: RGBA in, indices resolved at the one funnel.** Tools keep painting RGBA
exactly as today. `_commit_patch` (`document.py:776`) remains the single funnel and
becomes the index-resolution point (direct successor of the snap at lines 796-799):
the committed region resolves to indices (alpha < `OPAQUE_THRESHOLD` (128) → the
transparent index; else nearest in straight RGB reusing `snap`'s packed-`np.unique`
approach), `pixels` is re-materialized through the LUT, a no-op discards the pending
cel and pushes nothing, else one `IndexPatchEdit`.
- **The transparent index is never a nearest-match candidate** — it is reached only
  through alpha.
- **The painted color IS an index**: `resolve` takes `prefer: (RGBA, index)`;
  `begin_stroke` threads the active palette slot, so pixels the user painted take the
  preferred slot among duplicates; filter/paste/gradient output falls to
  lowest-index-wins deterministically.
- Soft brushes / feathered selections / AA shapes stay legal at the engine (the funnel
  thresholds them, so no invariant can break); the studio wave forces hard/pixel
  brushes in indexed docs, matching Aseprite's no-AA-in-indexed.
- Blend modes composite in RGBA over materializations (Aseprite composites indexed in
  RGB via lookup too). The screen may show off-palette colors where non-normal blends
  stack; the **stored cels never do**; flatten-for-export requantizes (GIF already
  snaps via `map_to_palette`).
- Geometry: exact ops (flip/rot90/crop/canvas-resize) transform the **index plane**
  with the same permutation (`_map_planes` gains `index_fn`; resize fills with the
  transparent index) so duplicate-slot identity survives geometry exactly. Smooth
  scale is inherently resampling: it runs on RGBA and re-resolves — the one stated
  place indices are re-inferred.

**Undo.** New edit types in `inker/undo.py`:
- `IndexPatchEdit(layer_uid, rect, before, after)` — index crops (¼ the bytes of an
  RGBA patch); undo/redo write indices then re-materialize through the document's
  **live** LUT — correct by history ordering, because palette state changes are
  themselves on the same stack in the same compounds.
- `IndexRemapEdit(fwd, inv)` — a permutation applied to every distinct plane (~1 KiB).
- `ColorStateEdit(before, after)` where each side is
  `(mode, palette, transparent_index)` — the generalization of `PaletteEdit`
  (which remains for constraint-mode ops).

**Palette operations under true indexing** (same public surface, mode-branched):

| Op | Indexed mechanism | Undo shape |
|---|---|---|
| `recolour_slot` | table change + re-materialize from unchanged indices — the instant-repaint payoff, no pixel rewrite | `ColorStateEdit` alone |
| `move_slot` / `sort_palette` | exact permutation `indices = fwd[indices]` + table reorder. **Acquires an undo step in indexed mode** (it moves index data) — stated in manual + docstrings | `CompoundEdit([ColorStateEdit, IndexRemapEdit])` |
| `remove_slot` (merge) | non-invertible remap → the existing `_palette_step` replay path | `CompoundEdit([ColorStateEdit, ReplayEdit])` |
| `add_slot` / `insert_ramp` | table-only; new refusal at 256 | `ColorStateEdit` |
| `set_transparent_index` | re-materialize (alpha moves between two indices) | `ColorStateEdit` |
| `palette_usage` / `histogram` | `np.bincount(indices)` — exact, finally per-slot | — |

Mode conversions (`convert_to_indexed(colours, method, *, transparent=0)`,
`convert_to_rgb()`, `convert_to_grayscale()`) go through the `_palette_step` idiom —
one Ctrl+Z across every layer and frame, links preserved by the grid snapshot. The
live convert session (`begin/preview/commit/cancel_convert`,
`_doc_paint.py:636-765`) is reused for preview; `commit_convert` grows a mode target.
`convert_to_rgb` drops `indices`, keeps `pixels`. `dither.py` gains
`convert_indices(...) -> (H,W) uint8` (nearest/FS/Bayer picking indices); the existing
`convert()` is kept byte-identical for constraint mode, with a visible-pixel parity
pin between the two.

New pure module `inker/index_plane.py` (added to the pin test's module list; no
outward imports): `lut`, `materialize`, `resolve`, `apply_remap`,
`permutation_tables`, `merge_table`, `histogram`, `OPAQUE_THRESHOLD = 128`.

**Grayscale: a funnel constraint over RGBA storage — argued.** Aseprite stores
`(v, a)`; we do not. `color_mode == "grayscale"` keeps RGBA storage with the invariant
*every visible pixel has r == g == b*, enforced at `_commit_patch` via the shared
Rec.709 luma coefficients. Rationale: grayscale has no identity problem — `(v,a)` and
`(v,v,v,a)` are informationally equivalent and `v` is exactly recoverable; a 2-channel
plane would fork every consumer this design was shaped to avoid forking; and all 19
blend modes preserve r=g=b on gray inputs (channelwise trivially; the HSL family
because gray has zero saturation), so even the composite stays gray. `.aseprite`
grayscale files open **as grayscale documents**. Divergence #2's grayscale half is
amended: behavior parity, storage divergence, lossless round-trip.

**Persistence (ORA).** Indexed cels/layers are written as **PNG mode-P images**
(`PLTE` = the palette in order, `tRNS` = per-entry alpha, pixel bytes = the index
plane verbatim). The record/projection doctrine a third time:
- The P-PNGs are the record for indices and the full-alpha table; `palette.gpl` stays
  the interop projection; `stack.xml` colors stay the picture foreign editors get.
- Old and foreign readers degrade perfectly: `_decode` already does
  `im.convert("RGBA")`, which honors PLTE+tRNS — an old Warlock opens with correct
  pixels and lands in constraint-on-write via `palette.gpl`. Krita/GIMP decode P-PNGs
  natively. Stated forward-compat cost (mirroring ora.py's existing animated-file
  one): an old build that *re-saves* writes RGBA planes and index identity is gone.
- `warlock.json` gains an **additive** `color` block
  (`{"mode": "indexed", "transparent": 0}`), written only when
  `color_mode != "rgb"` — `WARLOCK_VERSION` stays 1 (the bump rule — "a reader could
  get it wrong in a way it cannot detect" — is not met). Every RGB document's archive
  stays byte-for-byte what today's writer produces. A malformed `color` block costs
  the mode metadata, never the file: the P-PNGs are self-describing, the transparent
  index recovers from tRNS or defaults, with a log line.
- Grayscale writes ordinary RGBA PNGs + `{"mode": "grayscale"}`.
- **Crash recovery becomes honest for free**: the journal payload is
  `ora_bytes(doc)`.

**Interchange.** `asein.document_from_aseprite` stops LUT-flattening away the truth:
indexed files place the raw index planes (background-layer materialization keeps the
existing `_lut(palette, None)` distinction), set `color_mode="indexed"` +
`transparent_index` + full-alpha palette; grayscale files keep the exact `(v,v,v,a)`
expansion and set `color_mode="grayscale"`. `gifout.py`: when a frame's flatten equals
its materialization (single track, normal blend), pass indices through verbatim
instead of re-snapping; transparent-index alignment with `TRANSPARENT_INDEX`.

**Deferred, with rationale (recorded here so the decision is citable):**
- **Per-frame palettes.** Modern Aseprite authors one palette per sprite; per-frame
  tables are pre-1.0 legacy the format merely tolerates, so lossless interchange for
  files Aseprite writes today does not require them. They are also
  architecture-hostile: a linked cel is one object across frames, and its materialized
  `pixels` lives on that object — per-frame palettes would force materialization to
  per-slot render time, touching the compositor and every cache. The `.ase` reader
  warns and uses the final table (it effectively does today). → divergence #20.
- **ICC.** Divergence #3 re-examined and held: honest color management is a pipeline
  property, not a feature — the native blend kernels (bit-parity bar), dither metrics,
  and five export formats all assume sRGB bytes; the only cheap step (carrying an
  assigned profile blob) buys nothing a pixel-art user can see. Amend #3's text.

### Sub-waves (each shippable, suite green)

- **1.0 Enablers** (no behavior change): `layers.py` `indices` field (+`copy`,
  `__post_init__` validation), `document.py` mode fields + `_rematerialize` helpers,
  cost sites count `indices.nbytes` via one `plane_bytes(layer)` helper,
  `dither.convert_indices`, new `inker/index_plane.py`, pin-test module list. Tests:
  index_plane determinism; the visible-pixel parity pin for all five dither methods;
  ORA determinism pins green untouched.
- **1.1 Indexed engine** (headless): `_commit_patch` indexed branch, `_ensure_cel_for`
  fill, `_map_planes(index_fn=)`, stroke-prefer plumbing, mode-branched
  `_doc_indexed.py` + conversions + 256-cap + `is_indexed`/`is_palette_locked` split,
  the three new edit types, geometry index variants. Tests: duplicate-slot identity
  survives recolour/move/sort/flip/rotate/crop exactly; merge undoable to exact
  indices; undo/redo across a palette edit lands on identical indices *and* pixels;
  linked cels share one index plane; alpha-threshold and never-matched-transparent
  pins; refusals by name; no-op-pushes-nothing holds in indexed mode.
- **1.2 Persistence and recovery**: `ora.py` P-PNG writer + `color` block + reader
  degradation ladder. Tests: indexed/grayscale saved twice byte-identical; RGB archive
  byte-identical to a pre-change golden; round-trip exactness (indices, alpha palette,
  transparent index); degraded-read tests; `ora_bytes` recovery round-trip carries
  indices (the crash-journal honesty test, named as such).
- **1.3 Grayscale mode**: funnel luma branch, `convert_to_grayscale`. Tests: r==g==b
  after every write path; all-19-blends-preserve-grayness property test; ORA
  round-trip.
- **1.4 Interchange**: `asein.py` mode preservation, `gifout.py` index passthrough.
  Tests: `.ase` indexed fixture round-trips index-exactly incl. duplicate-color
  palette; grayscale fixture opens grayscale; GIF slot-stability pin.
- **1.5 Studio UI + docs**: mode menu via the convert session, transparent-index
  marker + context action on the palette pane, indexed foreground = a palette slot,
  hard-brush forcing; rewrite `docs/manual/09-inker.md` ~lines 362-373 (the "pixels
  stay full-colour RGBA underneath" paragraph is now only true of palette-constrained
  RGB) + new Indexed/Grayscale sections + the changed undo behavior of slot moves;
  INVARIANTS.md rewrite (below).

### Deviations from this spec, as executed (Wave 1)

Six, each argued at its site and recorded here so the spec stays the record:

1. **The paint slot is a document field, not a `begin_stroke` parameter.** Every
   tool paints the foreground -- fill, shape, gradient, text stamp -- so
   `Document.paint_slot` is set once per gesture at `inker_canvas._press` rather
   than threaded through twelve signatures. `InkerState.set_fg` is the one door
   that records or clears it, pinned by an AST scan.
2. **`grayscale()` lives in `indexed.py`, not `index_plane.py`.** It is a write
   constraint like `snap`, it reuses that module's `_LUMA`, and `index_plane` is
   about slots.
3. **`convert_to_rgb` keeps the palette**, landing the document in
   palette-constrained RGB rather than free RGB. Discarding a table the user
   authored would be the most destructive thing that menu could do;
   `set_palette(None)` is the separate, undoable act.
4. **The transparent slot's stored alpha is canonicalised on ORA read**, not
   normalised on the document. Its `tRNS` byte is *forced* to zero on write (the
   hole must be a hole to Krita/GIMP/a browser), so reading it back literally
   would read the writer's own requirement as user data -- and normalising the
   document instead left the *previous* hole stranded at alpha 0 when the index
   moved. The RGB survives; the alpha does not exist to survive.
5. **The `.aseprite` background-layer conflict is resolved by duplicating the
   slot**, not by the spec's "keep the `_lut(palette, None)` distinction" (which
   would have meant mixed planes in one document). A new entry with the
   transparent slot's colour is appended and the background re-pointed at it --
   only possible because slots are now what is stored. A full palette falls back
   to a warning by name.
6. **The GIF index passthrough is decided by equality, not by structure.**
   `sheetout.index_plane_one` picks the candidate structurally and then requires
   that materialising it reproduces the flatten byte for byte, which is safe
   without the condition list being complete.

Also fixed in passing: `tests/test_ground_gpu.py` cited `prompt.TILE_FIELDS`,
deleted by the taxonomy retirement, so the whole module had errored at fixture
setup and the gpu lane had been red since `2c8fe73`. It is green again (23
passed).

### INVARIANTS.md changes (Wave 1)

Rewrite the indexed paragraph into: (1) three color modes, one funnel —
`_commit_patch` applies each mode's constraint (snap / index-resolve / luma); (2) in
an indexed document the index planes are the record and `pixels` the materialization,
re-materialization has one owner; (3) palette identity is per-slot, never per-color —
remaps are index tables, never nearest re-inference; smooth scale is the stated
exception; (4) the transparent index materializes as alpha 0 and is never a
nearest-match candidate; (5) indexed palettes cap at 256, refused by name; (6) the
P-mode layer PNGs are the record, `palette.gpl`/`stack.xml` are projections — the
`animation.json` doctrine's third instance with the same stated forward-compat cost;
(7) a link shares its index plane because it shares its `Layer`; (8) grayscale is a
funnel constraint over RGBA storage, sound because all 19 blends preserve grayness;
(9) index-crop undo re-derives RGBA through the live LUT, correct because palette
state rides the same stack.

Divergences: **amend #2** (indexed + grayscale halves; grayscale storage stays
`(v,v,v,a)`, stated as behavior-parity/storage-divergence; update citing sites),
**amend #3** (re-examined and held), **append**: #19 palette-constrained RGB (a
Warlock mode Aseprite lacks; the migration story), #20 no per-frame palettes, #21
commit-time index resolution with an alpha threshold (the fg-slot `prefer` closes the
visible gap).

### Risks (Wave 1)
- **Materialization drift** — any code writing `pixels` without `indices`. Contained
  because live-write paths all end at `_commit_patch` or restore snapshots; add a
  test-only `_check_materialized` assert run after every session close in the suite.
- Undo across mode boundaries: guard `IndexPatchEdit` application with the animated
  snapshot precedent — refuse by name rather than corrupt.
- Pillow P-PNG determinism across versions — same exposure the RGBA pins carry; the
  new pins catch a Pillow bump.
- The `is_indexed` studio audit must be exhaustive or a pane treats constrained-RGB as
  truly indexed.
- Resist making tools index-native (divergence #21) until a user-visible failure
  justifies it — the brush-paints-RGBA decision is what keeps this wave tractable.

**Gate:** all sub-wave tests; determinism pins; the RGB negative control; gpu lane
green (no model-loading changes expected, but the lane rule stands).

---

## Wave 2 — Timeline target model: multi-cel/layer/frame editing (P0 item 4) — **DONE 2026-08-17** (nine commits, `eebbab5`..)

Built on the existing range machinery — the pattern is fixed and has three worked
examples (`_doc_ranges.py`: explicit ints in, dedupe by `id()` via
`unique_cel_layers`, per-cel edits collected into one `_push_range`, commit floating
before reading the grid, no-op pushes nothing). **No new undo machinery is needed.**
Ordered after Wave 1 so every new op is mode-aware from birth (they all write through
the funnel or `_map_planes`-style index-aware transforms).

- **New range ops**, each following the `filter_range` worked example
  (`_doc_ranges.py:641`): `flip_range(axis, t0,t1,f0,f1)`,
  `rotate_range(quarters, ...)` (square cels only for 90°; refuse by name otherwise —
  cels are canvas-sized so 90° needs W==H), `fill_range(colour, ...)`
  (selection-mask-weighted like `masked_apply`), `shift_range(dx, dy, wrap, ...)`
  (wraparound shift = Aseprite's Shift behavior; `np.roll` per cel when `wrap`,
  zero-fill otherwise), `clear_range` exists. The filter popup's "Apply to range"
  button generalizes to the other destructive image commands.
- **Range transform**: free transform over a cel range. Preview stays on the active
  cel through the existing `FloatingBuffer` session untouched; on commit the final
  affine (angle/scale/shear/flips, the buffer's own scale→shear→rotate contract) is
  replayed per distinct cel in the range — each cel lifts with the same mask,
  transforms, and commits; one `CompoundEdit` of per-cel patches; linked cels
  transformed once. Escape/cancel touches only the active cel as today.
- **Continuous layers**: `Track.continuous: bool = False` — new-cel autovivification
  copies the previous cel's content (continuous) vs blank (discontinuous). The cheap
  80% of Aseprite's feature. `animation.json` additive key written only when true
  (byte-stability for existing docs); toggle in the layers pane; `TrackPropsEdit`
  carries it.
- **Per-cel opacity and z-index are NOT added.** Divergences #1/#12 stand: cel opacity
  forks the compositor and the flatten caches for a rarely-used feature. Revisit on
  evidence; this spec is the citable decision.
- **Layers panel multi-select: minimal.** `range_sel` already spans tracks
  (`t0..t1`); expose the track-range in the layers pane (highlight member rows) and
  route flip/fill/clear/`TrackPropsEdit` batches to it. Arbitrary discontiguous
  multi-select is out of scope for this program (stated deferral).
- Timeline menu grows the new verbs, disabled-never-hidden, with the existing
  active-cell fallback.

Key files: `inker/_doc_ranges.py`, `inker/_doc_selection.py`, `inker/animation.py`,
`inker/anim_edits.py` (TrackPropsEdit field), `panes/inker_timeline.py`,
`panes/inker_layers.py`, `inker_mode.py`, `docs/manual/09-inker.md`.

**Gate:** for every new op — linked-cel-touched-once, one-undo-step,
no-op-pushes-nothing, indexed-document identity preserved (flip/rotate/shift are
exact index permutations); `animation.json` byte-stability for docs with no
continuous tracks. **All met.**

### Deviations from this spec, as executed (Wave 2)

Eight, each argued at its site and recorded here so the spec stays the record
(#8 recorded 2026-08-17, after the wave landed -- the numbering only appends):

1. **A range op does not go through the funnel; it goes through a new sibling.**
   The spec says "they all write through the funnel", which is not possible:
   `_commit_patch` pushes a step per call and a rect of five cels is one Ctrl+Z.
   `Document._patch_edit_for` applies the identical constraint in the identical
   order and *returns* the edit instead. Closing that gap fixed a pre-existing
   bug the spec did not know about: `filter_range` was writing raw RGBA onto
   indexed cels and leaving `layer.indices` stale.
2. **The permutations bypass the colour mode deliberately.** `flip`/`rotate`/
   `shift` permute the index plane and re-derive the pixels rather than
   resolving from the mapped colours -- `_map_planes`' `index_fn` discipline,
   applied one cel at a time. Re-resolving would collapse duplicate slots
   silently.
3. **`axis` is `transform.FLIPS` (`"horizontal"`/`"vertical"`)**, not the spec's
   shorthand: that tuple exists precisely so a menu cannot offer an axis the
   function refuses.
4. **The transform commit is automatic and ranged**, per the user's decision of
   2026-08-17: the normal commit (Enter, click outside) applies to the whole
   highlighted range. Cancel is never ranged. The **active cel is unioned in**
   even when the rect excludes it -- it is the one the user watched move. A
   **paste** commits plain, because it has no `source_box` to re-cut. Two
   supporting changes fell out: the buffer now records the *recipe*
   (`source_box`, `flips`, `resample`) rather than only the result, and
   `render_transform` was lifted out of `FloatingBuffer.transform` so the live
   render and the replay cannot drift.
5. **Continuous autovivification copies the nearest earlier occupied cel**
   (walking back from the current frame; none found means blank), and **copies
   rather than links** where Aseprite links. `asein` maps layer flag bit 16
   ("prefer linked cels") onto it, which is the nearest honest reading. The
   Timeline **Shift** menu verbs always wrap; the engine keeps `wrap=False` for
   a future caller.
6. **Flip/rotate/shift act on whole cels, ignoring the selection.** Only
   `fill_range` and `filter_range` honour it as a weight -- a weighted
   permutation would have to invent what goes in the part it did not move.
7. **The alpha lock is read off the track, not the cel.** Found while writing
   the `fill_range` test: track properties reach a cel only when its frame is
   materialised, so both range writes were painting through "preserve
   transparency" on every frame but the one on screen. `filter_range` had the
   same gap and is fixed with it.
8. **No range op consults the content lock.** A plain fill on a locked layer
   is refused at `write_colour`; the same fill from the timeline's range menu
   writes straight onto locked tracks -- and so do the flips, turns and shifts
   and the ranged transform commit. Nothing in `_doc_ranges.py` reads
   `track.locked` (only `track.alpha_lock`, per #7), and
   `commit_floating_range`'s replay writes to every target cel unasked; the
   one door that still holds in that path is the *lift* that starts the
   gesture, `float_pixels`' existing refusal on the active layer. This extends
   `filter_range`'s pre-existing posture rather than introducing it, and the
   argument is the gesture's shape: a canvas tool needs the lock because a
   stroke lands on whatever the active layer happens to be, while the range
   rect is an explicit bulk selection -- the user named those cells, locked
   ones included. Recorded in the content-lock entry of `docs/INVARIANTS.md`.

Also found in passing and **not** fixed, because it is neither Wave 2's nor a
regression: running `tests/test_studio_smoke.py` and `tests/test_studio_controls.py`
in one process crashes the interpreter with an access violation inside the second
file's `renderer.render` (`uv run pytest tests/test_studio_smoke.py
tests/test_studio_controls.py -n 0`). It reproduces on the tree before this wave.
Which files share a worker is `--dist loadfile` scheduling luck, so the full
parallel run goes red at random as tests are added anywhere; two `ImguiRenderer`
lifecycles in one process is the trigger, and a context of its own for the second
one is *not* enough to avoid it.

Two things learned during the Inker UX pass (2026-08-18), which added ~160 tests
and so shifted the scheduling this depends on:

- The prediction above came true -- the full run went red on roughly one attempt
  in two, on a *different* file each time, each passing alone.
- It does not always crash cleanly. One run left a worker hung rather than
  failing, so a suite that "takes too long" is a symptom of this and not only a
  slow machine.
- The faulting call is `imgui.new_frame()` on the second file's fresh context
  (`test_studio_controls.py:132`), not `renderer.render` as recorded above --
  i.e. the damage is already done by the time the second context draws, which
  points at what the *first* renderer's `shutdown` leaves behind rather than at
  the second one's setup.

Still not fixed here, and still not a regression, but it is now frequent enough
to be worth its own session.

---

## Wave 3 — Tilemap layers + tilesets (P0 item 3) — **DONE 2026-08-18** (twelve commits, `da94828`..)

### The architecture

**Chunk 3.1 — Promote `studio/tilegrid/` (pure refactor, no behavior change).**
Move `plotter/{gid,tileset,blob}.py` **verbatim** to a new shared leaf package
`src/warlock/studio/tilegrid/` (`__init__.py` re-exports the public names). The
`studio/undo.py` precedent: with three consumers (plotter, packwright, inker) the
definition belongs to a leaf, not an engine. This *strengthens* the doctrine — the
packwright→plotter exception shrinks to `tsx` + `pngio`, and the sibling ban becomes
absolute again with zero carve-outs. `blob.py` must come (Tileset validates terrain
sets against `blob.TILE_COUNT`; blob is pure numpy). `tsx.py` stays in plotter (Tiled
format code; inker's `.tsx` export happens above the engine pin).
Churn, honestly counted: 15 plotter modules re-import; plotter's OUTWARD_IMPORTS pin
grows to ~17 entries (the pin doing its job); packwright's pin swaps two entries;
tileset/gid/blob unit tests move to `tests/tilegrid/` with a new standard pin file
(tilegrid imports nothing under `warlock`); INVARIANTS lines about the exception and
the inker pin's SIBLING_PACKAGES docstring are rewritten. **No re-export shims** in
plotter — a shim is a second spelling. Determinism pins (wmap/tsx/tmx bytes) must
pass untouched: the proof the move changed nothing. **Must land after Wave 0 commits
the packwright tileset batch** (that work imports `plotter.tileset`; the promotion
rewrites its imports with everything else — interleaving does not work).

**Chunk 3.2 — Tile model core (headless, no persistence, no UI).**
```python
# inker/tiles.py
@dataclass
class TilesetSlot:
    """Mutable holder for an immutable Tileset. The frozen tilegrid.Tileset is
    replaced whole on every edit (texture caches key on id(pixels), so a
    frozen-replace IS the invalidation); the uid is what undo addresses."""
    tileset: Tileset                 # vertical strip, tile 0 = blank
    uid: int = field(default_factory=new_uid)

@dataclass
class TilemapCel(Layer):
    """A cel whose picture is derived. `refs` (grid_h, grid_w) uint32 in
    tilegrid.gid encoding + `tileset_uid` are authoritative; `pixels` is the
    canvas-sized RGBA materialization, kept in sync at edit time, so every
    reader of the stack — composite, flatten, ORA mergedimage, onion skin,
    thumbnails — needs no new case."""
    refs: np.ndarray | None = None   # validated non-None uint32 in __post_init__
    tileset_uid: int = 0
```
- `Animation.cels` typing unchanged — a `TilemapCel` *is* a `Layer`; linking a
  tilemap cel links the refs plane (correct parity semantics). `copy()` deep-copies
  `refs`, preserves the uid-override rule.
- Inker tilesets are Aseprite-style **vertical strips** (`image_w == tile_w`,
  spacing=0, margin=0 — `Tileset.columns` yields 1, `tile_rect` exact), **tile local
  id 0 is a real blank tile**: ref 0 means both "empty" and "tile 0" and they render
  identically (`gid.EMPTY == 0`); dedup maps fully-transparent content to 0. This
  makes `.aseprite` import an index-preserving copy and `.tsx` export a plain strip.
- `Document.tilesets: list[TilesetSlot]`; `Track.tileset_uid: int | None = None` —
  a tilemap **track** binds exactly one tileset (Aseprite's rule). Additive field;
  `animation.json` stays byte-identical (binding serializes in the new member).
- **Manual/Auto/Stack behavior is view state** — a toolbar toggle in `inker_state`,
  passed into document calls per gesture, never serialized (a toggle must not dirty a
  document).
- `inker/tile_edits.py`: `TilesetPatchEdit(tileset_uid, tiles=[(local_id, before,
  after)])` (tile-sized crops, never atlas-scale), `TilesetGrowEdit(tileset_uid,
  added)` (undo truncates the strip), `TileRefsEdit(layer_uid, rect, before, after)`
  (tile-unit rects, uint32), `TilesetListEdit(index, before, after)` (the slot minted
  once at op time — the unlink rule). Undo/redo call document hooks
  (`_apply_tileset_tiles` / `_apply_tileset_grow` / `_apply_refs` /
  `_apply_tileset_slot`) that do frozen-rebuild + re-materialization + invalidation.
- `inker/_doc_tiles.py` mixin: `add_tileset`, **`place_tiles(layer_uid, origin,
  patch)`** — the single door for every refs write (stamp tool, tile flood fill,
  flip-in-place, re-points) — plus the `_apply_*` hooks and one
  `_frames_of_tileset` invalidation walk (an Auto edit on frame 3 changes frame 7's
  picture if both place the tile) used by every hook.
- `charged()` in `anim_edits.py` learns `getattr(layer, "extra_nbytes", 0)`;
  `TilemapCel` provides it (refs are tiny — a 64×64 grid is 16 KiB).
- `_ensure_cel_for` grows one branch: a track with `tileset_uid` autovivifies a
  `TilemapCel` with all-zero refs, keeping the placeholder's uid.
- Whole-document flips/rotates on documents containing tilemap layers are **refused
  by name** initially ("a flip of a tilemap layer is not yet modeled"); Chunk 3.7 may
  replace the refusal with refs flag algebra (a 90° grid rotation plus
  `FLIP_D|FLIP_H` per cell — the eight-symmetry algebra plotter already trusts).
  Canvas resize re-grids refs (pad/crop in tile units).

**Chunk 3.3 — Pixel-edit routing + conversions.**
`_commit_patch` — the one funnel — checks `isinstance(layer, TilemapCel)` and diverts
to `_commit_tilemap_patch(layer, rect, before, behavior)`: compute touched tile
cells; cut the stroked result back to tile-local space (undoing placement flip flags
so tiles are edited in canonical orientation); apply behavior —
- **Manual**: never modify the tileset — the write is reverted (re-materialized from
  refs), the gesture pushes nothing, the UI reports why;
- **Auto**: edit existing tiles in place — every placement on every frame/layer
  sharing the tileset updates; dedup check first (edited content equal to another
  tile → re-point the ref instead);
- **Stack**: hash-dedup, append-never-modify — new content appends a tile and
  re-points; originals untouched;
then rebuild the frozen `Tileset`, re-materialize every bound cel's affected regions,
push one `CompoundEdit` (the `_pending_cels` machinery already makes the funnel
compound-aware). The content-hash index is a derived cache keyed on
`id(tileset.pixels)`, never serialized. An Auto stroke's undo restores one tile's
bytes and every placement reverts in the same step — the property falls out of the
model.
Conversions: `convert_layer_to_tilemap(layer_uid, tile_w, tile_h)` (cut every cel on
the track, dedup by hash, blank → 0; `CompoundEdit([TilesetListEdit,
track-bind, CelSetEdit per slot])`, replacement cels keep old uids) and
`convert_layer_to_raster(layer_uid)` (pixels already materialized — lossless by
construction, round-trips bit-exact).

**Chunk 3.4 — ORA.** New member `tiles.json` (`TILES_VERSION = 1`), written
only-when-non-empty, refused whole-member on wrong version, read **after** the grid
is rebuilt — failure containment per ora.py's own member doctrine:
- `data/tileset{i}.png` — each strip via the deterministic PNG writer;
- `data/tilerefs{n}.u32` — one per unique tilemap cel in `unique_cel_layers` order,
  raw little-endian uint32 row-major (dimensions in the JSON; kilobytes, no
  compression, trivially deterministic);
- `tiles.json` — tilesets, track bindings, cel refs; **indices not uids** (the
  `animation.json` convention).
Ordinary cel PNGs are **still written** (`TilemapCel.pixels` is honest RGBA), so an
old build, Krita, or GIMP opens the document looking exactly right, flattened per
layer; any `tiles.json` failure costs tile *structure*, never pixels or frames.
`WARLOCK_VERSION` stays 1 and `warlock.json` is untouched. The old-build-re-save trap
(member silently dropped) is the same accepted cost `animation.json` documents, and
gets a sentence in the manual. Byte-determinism: fixed member order, `_EPOCH` stamps,
stable names — `test_inker_ora_determinism.py` extended. **Journal crash recovery
needs nothing** — its payload is `ora_bytes(doc)`.

**Chunk 3.5 — `.aseprite` import.** The three refusals become readers (the
INVARIANTS rule: a refusal only moves because the editor learned to model the thing):
- **Tileset chunk 0x2023** (`asein.py:394-399`): id, flags, count, tile w/h, base
  index, name. External-file flag → **new refusal by name**; embedded →
  zlib-decompressed strip → `tilegrid.Tileset` directly (their strip layout is our
  columns=1 layout, index-for-index, tile 0 included). Base index ≠ 1 → warning
  (display-only).
- **Tilemap layer kind 2** (`asein.py:430-433`): read the tileset index, bind
  `Track.tileset_uid`.
- **Tilemap cel type 3** (`asein.py:504-507`): w, h, bits-per-tile (**refuse ≠ 32 by
  name**), the four bitmasks, zlib data → uint32; remap generically from *their
  declared masks* onto our `GID_MASK`/`FLIP_H`/`FLIP_V`/`FLIP_D` (numerically
  identical in every current file, so usually the identity — but reading the masks
  means a future Aseprite cannot silently scramble our flags). Non-tile-aligned cel
  offsets → refusal by name (unreachable from Aseprite's own writer).
Per-tile user data → warn ("per-tile properties are not kept; the tiles are").
Inker pin gains `("asein.py", "warlock.studio.tilegrid")`.

**Chunk 3.6 — Export + Plotter interop (UI layer, above the engine pin).**
"Shared tileset data" concretely: (1) **one type** — after 3.1 an Inker tileset *is*
a `tilegrid.Tileset`, no conversion layer to drift; (2) **file-level interop** —
`inker_mode` exports `.tsx` + PNG via `plotter.tsx.tsx_bytes` / `plotter.pngio`
(writers exist, take the shared type directly); import a `.tsx` as a tileset (any
grid geometry is handled; v1 keeps whatever grid arrives); (3) **in-app handoff** —
a studio command ("Use tileset in Plotter") hands the frozen `Tileset` to a Plotter
document as a `TilesetRef`, zero-copy, **snapshot semantics not subscription** (a
later Inker edit mints a new frozen object; Plotter keeps the old one — same
semantics as the file path, stated in the manual). **No live-linked registry** —
reconciling live edits across two undo stacks is a synchronization feature with its
own invariant load; the frozen-snapshot model is honest without it.

**Chunk 3.7 — UI panes + deferred geometry.** Tile picker pane, placement tool +
tile cursor in `panes/inker_canvas.py`, Manual/Auto/Stack toggle in
`panes/inker_tools.py`, canvas-resize re-gridding, optionally the flip refusal →
refs flag algebra. Manual tilemap section (no new chapter — no renumbering);
cross-reference in the Plotter chapter. *As executed:* the picker is
`panes/inker_tiles.py` (drawn in the left column once the document has a
tileset), the manual section is `docs/manual/09-inker.md#tilemap-layers` with the
cross-reference in `11-plotter.md#tilesets-from-inker`, and the flag algebra was
deferred (deviation 6 below).

**Indexed × tilemap composition** (Wave 1 landed first): a tilemap layer in an
indexed document materializes RGBA then resolves at the funnel like any cel; tileset
strips stay RGBA (noted as a divergence — Aseprite tilesets in indexed sprites hold
indices).

**Deferred:** terrains/Wang in Inker (plotter owns autotiling; the shared type
already carries `terrains`/`phases`, so terrain `.tsx` still loads and places),
per-tile properties, tile animation, external-file tilesets, non-32-bit tilemap
cels.

### INVARIANTS.md entries (Wave 3)
1. **tilegrid is the second shared leaf** — owns the gid word, the sliced-atlas type
   and the blob collapse; plotter/packwright/inker import it, none owns it, it
   imports nothing under `warlock`; the packwright→plotter exception shrinks to
   `tsx`+`pngio`; the sibling ban is again absolute.
2. **A tilemap cel's pixels are a materialization** — `refs` + the bound tileset are
   authoritative; `pixels` rebuilt only by the `_apply_*` hooks and the commit
   funnels; everything reading a stack sees ordinary RGBA. The flag-bit doctrine now
   spans both engines in tilegrid's words.
3. **`tiles.json` is versioned, only-when-non-empty, and fails alone** — a broken or
   unknown member costs tile structure, never pixels or frames.
4. **Manual/Auto/Stack is view state** — never serialized; a toolbar toggle cannot
   dirty a document.
5. **The asein refusal ledger** — 0x2023 / layer kind 2 / cel type 3 moved to
   support; external tilesets, bits≠32, non-aligned cels are the refusals that
   replaced them.

### Risks (Wave 3)
- Materialization divergence (pixels ≠ refs): the funnel is singular by pinned
  doctrine; test-only assert `materialize(refs, ts, size) == pixels` after every op
  in the tile suite; refuse-by-name any `_replay` op not yet taught.
- Chunk 3.1 blast radius (15 modules + 4 pin files + INVARIANTS in one commit):
  zero content edits to moved files; determinism pins as the no-change proof;
  explicit sequencing after the Wave 0 commit.
- Shared-tileset invalidation must reach every bound cel on every frame — one walk,
  used by every hook, tested with two layers sharing one tileset.
- Aseprite's Manual-mode nuance (our gloss: never touch the tileset) — contained in
  one routing function; a numbered divergence entry if feedback shows drift.

**Gate per chunk** as listed; overall: an `.aseprite` file with tilemaps opens,
edits, saves to ORA, reopens bit-exact (refs and flags), and exports a
Tiled-openable `.tsx`.

### Deviations from this spec, as executed (Wave 3)

Recorded here so the spec stays the record. Each is argued at its own site; this
is the citable list.

1. **`TilemapCel.plane_bytes` overrides the hook, instead of the spec's
   `charged()` + `getattr(layer, "extra_nbytes", 0)`.** `undo._plane_bytes`'s
   docstring already pre-argues exactly this: it asks a layer for its own
   `plane_bytes` precisely so a subclass with a third plane can answer for
   itself, and every cost site that charges a layer already goes through it. A
   parallel `extra_nbytes` protocol would have been a second spelling of one
   fact.
2. **The tile hash index holds the strip array and compares it with `is`,
   rather than keying on `id(pixels)`.** CPython recycles an id once the array
   it named is freed — the documented 2026-08-17 bug class — so a stamp that
   outlived its array could answer "unchanged" for a completely different
   atlas. Holding the array makes that impossible by construction and costs one
   strip per tileset. `panes/inker_textures.tileset_texture` uses the same
   stamp for the same reason.
3. **A dangling `Track.tileset_uid` is an early return, not a fallback.** The
   first cut of `_ensure_cel_for` synthesised a canvas-sized single-cell tileset
   when a track's binding named nothing; a dangling binding is an impossible
   state, and the honest answer is no cel, which is the idiom every sibling
   door in `document.py` already uses.
4. **The grayscale constraint and the palette snap are applied inside
   `_commit_tilemap_patch`.** The funnel's tilemap divert returns *before* the
   funnel's own two, so without this a blue stroke on a grayscale document put
   `[0, 0, 255, 255]` into the **tileset** and every placement of that tile drew
   it. Applied per touched cell, scoped to the funnel's own `rect` and before
   the dedup key, so the key, the no-op test and the recorded before/after all
   describe the pixels the document actually ends up with.
5. **Range ops on a tilemap track are refused this wave.** `flip_range`,
   `rotate_range`, `shift_range`, `fill_range` and `filter_range` all raise by
   name rather than growing a refs-aware path; they are the same
   eight-symmetry problem the whole-document geometry has, one scope down.
6. **The whole-document flip/rotate/scale/crop refusals are retained; the
   spec's optional "flip refusal → refs flag algebra" is explicitly deferred.**
   Only `resize_canvas` was taught (`_tile_regrid`: pad/crop in tile units,
   honouring the nine-way anchor, refusing a non-tile-aligned offset by name),
   because a resize translates by whole cells and needs no flag permutation at
   all. Turning `refs` by the eight-symmetry algebra is a Wave 4+ item.
7. **Door-level refusals beyond the spec's list.** `merge_down`, `apply_matte`,
   `lift`, cut, `_patch_edit_for` and the three whole-document colour
   conversions (`convert_to_indexed`, `convert_to_grayscale`,
   `convert_to_palette`) each refuse a tilemap layer *at the door*. Each writes
   pixels before the funnel's refusal could fire, so a guard at the funnel
   alone would leave a half-written layer behind an exception. `convert_to_rgb`
   is deliberately not among them: it drops index planes and rewrites no pixel,
   and a document that could not leave the mode it was in would simply be
   wedged.
8. **Two new edit types were minted**: `TrackTilesetEdit` (bind/unbind a
   track's tileset, ending in `_anim_changed` rather than a bare `rev` bump,
   because the binding is what `_ensure_cel_for` reads) and `LayerSwapEdit`
   (replace one still document's layer by uid — the conversions' still-document
   arm, where an animated document uses `CelSetEdit`).
9. **`.aseprite` per-tile user data is detected by a run counter.** The spec's
   own text says a tileset chunk "could be followed by a user data chunk (empty
   or not) and then all the user data chunks of the tiles": the *first*
   `USER_DATA` after a tileset belongs to the tileset, and only the second and
   later ones are per-tile. `_Parse.tileset_ud_run` tracks that, where the first
   cut had the detection inverted.
10. **Task 7's index-0 "Export tileset..." is superseded by the picker.** It
    reached `doc.tilesets[0]` because nothing yet had a selection to offer it;
    `panes/inker_tiles.py` now owns the export and addresses the picked
    tileset. What stayed in the bridge panel is the two doors that *acquire* a
    first tileset — the `.tsx` import and "Convert to tilemap..." — because the
    tile panel is only drawn once the document has one.
11. **The tile stamp's letter is `Y`, and it is arbitrary by elimination.**
    Every mnemonic in "tile", "stamp" and "tilemap" was taken; of the three
    letters no tool held, `X` swaps the colours and `Z` sits under Ctrl+Z where
    a slipped modifier would swap a tool for an undo. Aseprite has no letter to
    borrow: it has no tile *tool* at all, it puts the whole editor into a
    tilemap mode instead. No `SHIFT_TOOL_KEYS` entry was added, because that
    table's docstring claims its contents are Aseprite's own slot-mates.
12. **Manual mode reports through the pane, not the document.** The standing
    ruling from Task 4: `_commit_tilemap_patch` reverts silently and gains no
    channel, so `panes/inker_canvas` banks `history.head` at the press and
    raises the sentence once per gesture when the head has not moved. See the
    INVARIANTS entry.
13. **Two `.aseprite` tileset refusals beyond the spec's list**, both in
    `asein.py` and both by name. A tileset chunk with *neither* the
    external-file flag nor the embedded-tiles flag set — "holds no embedded
    pixel data this build can read" — and a tileset declaring **zero tiles** —
    "holds no tiles". The spec's list named only the external-file case. The
    first is a deliberate reading of a gap in the format documentation (bit 2's
    exact meaning when bit 1 is also clear could not be verified against the
    published spec, and real Aseprite always sets bit 2 for an embedded
    tileset), resolved on `asein.py`'s own doctrine — anything that would
    change what the pixels mean is a refusal, not a warning and a
    tileset-shaped hole. Revisit if a real file ever exercises it.
14. **Indexed x tilemap read order is stated, not fixed.** `ora._finish_colour`
    runs *before* `_read_tiles`, so on an indexed document the index planes are
    resolved on the old plain-`Layer` cels and the `TilemapCel` that replaces
    one copies `indices` through unchanged. That matches the model's own
    position — `TilemapCel.copy()` does the same, and the recorded divergence
    is that a tilemap cel materializes RGBA while its strip stays RGBA too — so
    it is recorded here rather than papered over, and indexed x tilemap
    composition remains explicitly out of Wave 3's test list.
15. **The picked tile is clamped at the panel *and* refused at the door.**
    `place_tiles` raises by name on a local id past the end of the bound
    tileset (masked through `GID_MASK`, checked before `_ensure_cel_for`),
    because `materialize` draws an out-of-range ref as blank — right for a ref
    that *became* stale, and silently wrong for one written that way, since the
    cell starts drawing a real tile the moment the tileset grows past that
    number. `InkerState.clamp_tile_pick` is what stops a user ever meeting the
    refusal: the panel applies it every frame and the stamp applies it again at
    the write, because a raise out of a press would be a window down rather
    than a refusal anyone can act on.

### Left open by Wave 3

- Refs-aware flip/rotate (the flag algebra) and the range ops that depend on
  it; the refusals name themselves and are the ledger.
- Terrains/Wang in Inker, per-tile properties, tile animation, external-file
  tilesets, non-32-bit tilemap cels — all deferred by the spec above and all
  still deferred.
- `_tile_hashes` entries for removed tilesets are never evicted (bounded by the
  tilesets ever created in a session, one strip each).
- `convert_layer_to_tilemap` is O(cells x cels) with a tile copy per cell —
  fine at Inker canvas sizes, would want vectorising for a large clip.

---

## Wave 4 — Production sprite-sheet export (P0 item 6) — **DONE 2026-08-19** (ten commits, `57d347a..c0dc334`)

Two tracks: Inker's own exporter grows the Aseprite-parity options; Packwright grows
the atlas-grade options; the existing "send frames to Packwright" bridge remains the
packed-layout path (Packwright's MaxRects already exists — Inker does not grow a
packer).

**Inker** (`inker/sheetout.py` + export UI in `inker_mode.py`/`panes/inker_timeline.py`):
- **Layout choice**: horizontal strip / vertical strip / rows N / columns N —
  `plan_frames` grows a layout parameter beside the existing row-wrap and
  directional-grid paths; `frame_size=0` sidecar convention unchanged.
- **Merge duplicate frames**: hash flattened frames; duplicate frames share a cell;
  sidecar cells already key by `cell_index`, so `animation_block`'s
  `{cell_index, duration_ms}` model carries it natively. Linked cels merge for free
  (already the same object) — the test proves it.
- **Ignore empty frames**: skip fully-transparent flattens; sidecar records the
  omission.
- **Split / filter by tag and by layer**: per-tag outputs reuse `export_tag`'s span
  logic and `rebase_tags`; per-layer outputs flatten track subsets via the existing
  `frame_stack` folding. (Split-by-slice deferred to P1.)
- **Trim as an option**: `measure_trim` exists and only records; `trim: bool` packs
  the trimmed rects (cell = max trimmed size, or per-frame rects in the sidecar).
- **Border / inner padding + extrusion**: small pure compose math (the
  `packwright/compose.py` `_extrude` order — sides then top/bottom — is the
  reference; port or share via `pipelines`).
- **Filename templates** for PNG-sequence and split outputs: `{title}`, `{tag}`,
  `{frame}`, `{layer}` — a small pure formatter in `sheetout.py`, unknown keys
  refused by name.
- **Export presets**: per-document memory of the last destination + options
  (`inker_state` fields, journal-exempt) — this is also the P1 "format-aware export
  presets" item's foundation.
- Sidecar stays `SHEET_VERSION = 1` — **additive keys only** (`layout`, `merged`,
  `trimmed` blocks written only when used, so pre-feature bytes are preserved).

**Packwright** (`packwright/layout.py` + settings pane):
- `columns: int | None` setting for grid packs (today columns are derived from the
  pow2-rounded width — `layout.py:238` — which is backwards for tileset authoring).
- **Shrink-to-fit / fix the pow2-growth trap**: `power_of_two` becomes optional per
  axis or off-by-default for grid packs; the preview compares output area to source
  area and says so (the manual's "a sparse sheet re-packs into a smaller one"
  promise must be one the defaults keep).
- TexturePacker JSON: schema choice (array — existing — vs hash), additive.

**Gate:** byte-determinism of every export; a Tiled-openable `.tsx` from a grid pack
(user-verified once, the TILED_VERSION rule); duplicate-merge correctness under
linked cels; `SHEET_VERSION` sidecar backward-compat pin (pre-feature exports
byte-identical); manual §07 + §10 updated.

### Deviations from this spec, as executed (Wave 4)

Recorded here so the spec stays the record. Each is argued at its own site; this
is the citable list.

1. **Three spec-location corrections**, found before implementation began and
   authoritative for every task that followed: `export_tag` lives in
   `inker_mode.py`, not `sheetout.py`; `measure_trim` lives in
   `pipelines/sheet.py`, not `sheetout.py`; and the gate's manual chapters are
   `08-inker.md`/`09-inker-animation.md`/`12-packwright.md` (post-renumbering),
   not the spec's `§07`/`§10`.
2. **Arrange semantics follow plain-English "rows N / columns N"** (N of that
   thing packed), not Aseprite's own crossed By-Rows/By-Columns labels, and the
   row count is ceil-derived (`rows = ceil(count/columns)`) rather than the
   spec's literal N — so a counted arrange never leaves a dead trailing row when
   N does not divide the frame count evenly.
3. **Merge and Skip empty are refused, by name, on a document with its own
   directional layout** — the same refusal Arrange already has. A
   turnaround/walk grid's cells are poses by yaws, not frames, and neither
   option has anything coherent to do to it.
4. **A merged cell's slices and pivot are remapped to its representative
   frame** (`_cell_representatives`) rather than left to key by cell index into
   whatever frame happens to share that index — a correctness fix the spec's
   merge description did not anticipate, since it assumed cells and frames
   stayed one-to-one.
5. **Trim delivers both spec readings, through different fields.** The atlas
   places each frame's trimmed pixels flush at its own cell's corner (a size
   reduction, not an offset-preserving placement), while the per-cell sidecar
   `trim` rectangle keeps recording the offset+size within the original,
   untrimmed frame, unchanged — so a consumer that wants the content back at
   its original position still can, from the sidecar rather than from the
   atlas layout.
6. **The pow2-growth fix landed as the spec's own second option** — off by
   default for grid packs, mode-resolved through a `None` sentinel that
   re-baselines whenever `mode` changes without also naming `power_of_two` —
   after measuring that the spec's literal instruction (delete both re-derives
   at `layout.py:238/249` unconditionally) breaks the Tiled `.tsx` contract in
   324 of 378 (86%) swept combinations: pow2 rounding is what keeps
   `Layout.columns`/`rows` agreeing with what a real Tiled reader independently
   derives from the finished PNG.
7. **A grid pack's `.tsx` refusal no longer aborts the whole export.** When an
   explicit `columns` pack's pow2-rounded geometry disagrees with Tiled's own
   derivation, the PNG and the TexturePacker JSON still write; only the `.tsx`
   is skipped, with a `"warn"`-level toast naming why.
8. **"Split by layer" writes one file per top-level row of the layer tree**
   (a lone track, or a whole group as one file), not one per leaf track —
   argued from what the layers panel shows as a single row, from a group being
   the only stack subset the pass-through opacity/visibility fold is exactly
   correct for, and from a document with no groups still degenerating to one
   file per layer, the plain Aseprite meaning.
9. **One `export_template` field customises either a split's per-file stem or
   a plain PNG sequence's per-frame numbering, never both at once** for a split
   PNG sequence — the two default constants the byte-identity pins require
   (`DEFAULT_TAG_TEMPLATE`/`DEFAULT_LAYER_TEMPLATE` vs. `DEFAULT_FRAME_TEMPLATE`)
   only reproduce exactly under this two-level scheme.
10. **Per-tab export memory is seeded once per tab** (`export_seed_uid`), not
    on every export click — a deliberate usability fix beyond the spec's
    literal words, so exporting the same tab twice in a row keeps a live edit
    instead of silently reverting it to what the tab last used.
11. **Trim/Padding/Extrude are not refused on a directional-layout document**,
    unlike Merge/Skip empty — a directional layout only fixes a cell's *count*,
    never its pixel size, so shrinking or padding a cell contradicts nothing
    the layout claims.

### Left open / owed by Wave 4

- **Tiled-openable `.tsx` from a grid pack, user-verified once** — re-owed.
  Item 6 above moved the geometry a real Tiled import would see (pow2 off by
  default for grid), so the standing verification needs a fresh look with the
  app installed (the TILED_VERSION rule: the constant only moves when a human
  with Tiled has looked).
- **Mode-level submit-test gap.** None of the arrange/merge/skip_empty/trim/
  padding/extrude early refusals in `inker_mode._submit_export` have a
  dedicated UI-level test; each is covered at the core (`sheetout`) layer
  only. A pre-existing gap the series has carried since before this wave,
  named again in the Task 1, 2, 4 and 5 reports.
- **A `"slices_conflict"` sidecar note was not built.** A merged-away frame's
  own authored slices, when they differ from its representative's, are
  silently dropped rather than recorded. Closing it needed comparing every
  duplicate frame's raw slice metadata against its representative's for every
  merge — measured past the close-out sweep's own ~20-line budget, so it
  stays a known gap rather than a rushed fix.
- **A visible group whose members are all hidden writes a transparent sheet**
  for a per-layer split. Flagged in Task 3's review but outside this
  close-out's own sweep list, so left untouched; `skip_empty` on the same
  export already raises rather than write a blank cell, which is the same
  failure reachable a different way.
- **`_slice_filenames` and `_split_stems` hold two different collision
  policies** (bump vs. refuse) in the same module, each argued at its own
  site. Noted in Task 3's report, not reconciled; revisit if a third naming
  helper is ever added.
- **The `{frame}`-in-split refusal has no wiring-level test (Task 5).**
  `filename_for` refuses `{frame}` on an export that has no frame, and
  `_split_stems` is the call that reaches it for a per-tag or per-layer batch
  — but the pin is at the `sheetout` layer only, so nothing proves the split
  runners actually pass the user's template down that path. Same shape as the
  mode-level submit-test gap above.

---

## Wave 5 — `.aseprite` writer + interop report (P0 item 2, deferred half) — **DONE 2026-08-19** (`c0dc334..HEAD`)

- **`inker/aseout.py`** (new, sibling of `asein.py`; pin: no new outward imports —
  pure struct packing + zlib): writes RGB, indexed and grayscale sprites — header,
  layer chunks (groups via child_level), cel chunks (compressed; linked cels as link
  cels preserving share structure), tags (direction/repeat), palette chunk 0x2019
  (+ old-format 0x0004 for compatibility), slices, and — from Wave 3 — tileset
  chunks (0x2023, embedded strips, index-for-index) and tilemap layers/cels (our gid
  bits written under explicitly-declared masks).
- **Round-trip corpus**: every fixture is written by `aseout` and re-read by
  `asein` — the automated gate is bit-exactness through our own reader (planes,
  indices, refs, flags, tags, palette incl. alpha where representable). A
  user-owed manual pass opens exports in real Aseprite (the Tiled-fixtures
  precedent: the constant only moves when a human with the app installed has
  looked).
- **`docs/ASEPRITE_INTEROP.md`**: the explicit lossy report — what ORA→aseprite and
  aseprite→ORA each drop (cel opacity #1, z-index #12, user data #14, per-frame
  palettes #20, color profiles #3, …), every line citing its divergence number.
- `document_from_aseprite`'s read-only stance (`path = None`) is **retired**: an
  opened `.aseprite` may save back to `.aseprite` via Save As; ordinary Save keeps
  routing to ORA/PNG per the Wave 0 gate. `filetypes`/`OPENABLE` wiring revisited
  (the deliberate exclusion at `inker_mode.py:41-49` is re-argued in place).

**Gate:** corpus round-trips bit-exact through our reader; suite green; manual §07
documents Save As → `.aseprite`; the interop report exists and is linked from the
manual. **All met** — `docs/ASEPRITE_INTEROP.md` is linked from both
`docs/manual/09-inker.md#saving` and `docs/manual/10-inker-animation.md#from-an-aseprite-file`.

### Deviations from this spec, as executed (Wave 5)

Five tasks (`aseout.py` core; tilesets/tilemaps/slices; the round-trip corpus; Save
As wiring; this interop report + manual + close-out sweep), each with its own fix
round after review. What follows is gathered from all five task reports
(`.superpowers/sdd/2026-08-18-aseprite-wave5-writer/task-*-report.md`), argued at
its own site in each and recorded here so the spec stays the record.

1. **The grayscale-violation check is masked to visible pixels, not a raw plane
   compare.** `_plane`'s greyness check first compared every channel unmasked,
   which refused an ordinary funnel-legal document (paint → erase → convert to
   grayscale leaves dead, non-grey RGB under alpha 0 — `indexed.grayscale`'s own
   documented behaviour). Fixed to mask on `pixels[..., 3] > 0` before comparing,
   so only a *visible* non-grey pixel refuses; a full, unstrided compare rather
   than a sampled one, because the correctness argument (a bypassed funnel's
   failure mode is exactly one stray coloured pixel) outweighs the ~8 MB extra
   compare against a strip that is about to be handed to zlib anyway.
2. **Palette-constrained RGB's loss is pinned, not fixed — a controller ruling,
   not a writer defect.** The 0x2019/0x0004 chunks are still written for an
   RGB-depth document carrying a constraint palette (a file Aseprite opens has its
   colour table), but `document_from_aseprite` only *installs* a palette as a
   constraint at indexed depth — adopting one at RGB depth would put the whole
   editor into palette-locked mode over a table nobody asked to be limited by.
   Divergence #19. `tests/inker/fixtures/aseprite/palette-constrained-rgb.aseprite`
   pins the loss in the corpus rather than treating it as a bug to close.
3. **The legacy WORD chunk-count field saturates rather than sentinels.**
   `_frame` originally wrote an unconditional `0xFFFF` sentinel whenever the
   modern DWORD count applied, which broke on a frame with **zero** chunks (every
   track's slot empty legally rounds to a frame with nothing in it): the DWORD
   correctly said 0, and the sentinel then made a reader falling back to the
   legacy WORD see sixty-five thousand chunks that were not there.
   `min(len(chunks), 0xFFFF)` dissolves the special case at the root instead of
   working around it, and is readable by anything that only knows the legacy
   field.
4. **Indexed tileset strips are resolved through the palette by
   `index_plane.resolve`'s own rule, exact match only — not option (b), refusing
   the whole feature.** The first pass built its own RGBA-equality lookup, which
   only placed a strip pixel that was already exactly one slot's full RGBA — so an
   ordinary funnel-painted strip (a half-coverage dab, an erased tile, alpha that
   is neither 0 nor 255) refused real documents outright. `_visible_lookup` now
   mirrors `resolve`'s alpha-first placement: below the opacity threshold goes to
   the transparent index regardless of the dead colour left under it, at or above
   it is placed by RGB alone among the visible slots. What still refuses, by name,
   is the one case `snap` and `resolve` genuinely disagree about — a visible
   colour that only the *transparent* slot holds — because there is no honest
   answer to write there.
5. **A slice's pivot and nine-patch centre are widened by `_first_set`, never
   invented.** The format declares their *presence* once per whole 0x2022 chunk,
   not per key, which is a constraint Aseprite's own file format shares, not a gap
   between the two document models — so this is an interop-report line, not a
   divergence: a key lacking what the chunk declares inherits the *first* value
   the slice carries anywhere (its own where it has one), and the zero branch that
   would otherwise invent a corner pivot on every unkeyed frame does not exist.
   Judged, per this task's own brief, **not** to warrant a new numbered
   divergence — see the note on divergence numbering below.
6. **The corpus's two funnel-painted fixtures are checked-in builders, not
   one-off generation scripts.** A first pass generated `grayscale-animated` and
   `tilemap-indexed` with scripts that were never committed, and both silently
   failed to reach the shape they were named for (a missing second frame; an
   eraser-only stroke that painted nothing) without the byte-level gate noticing,
   because a fully-normalised committed file round-trips identically whether or
   not the funnel-painted construct was ever present. `_asecorpus.py` now carries
   `build_grayscale_animated`/`build_tilemap_indexed` in a `BUILDERS` dict, and
   the corpus gate asserts the builder's own pre-write output carries the
   construct *and* that `aseprite_bytes(builder())` equals the committed file —
   so a regression in the shape itself, not just in the bytes, fails the suite.
7. **The corpus's committed fixtures are post-normalisation fixed points, not raw
   builder output, except where a fixture's whole point is that the two agree.**
   `palette-constrained-rgb` is the one fixture where this is load-bearing: its
   committed bytes are a *second*-pass write (build → write → read → write),
   because a freshly built palette-constrained document is not its own fixed
   point (divergence #19's read-side drop). Every other fixture's committed bytes
   equal its builder's first-pass output directly.
8. **`aseout` reaches into `..tilegrid.gid` for the four flip/rotate masks a
   tilemap cel chunk declares.** A new `OUTWARD_IMPORTS` entry,
   `("aseout.py", "warlock.studio.tilegrid")`, beyond the letter of "no new
   outward imports" the spec's own bullet stated for this module — argued in the
   pin test itself: the alternative was four literals restated locally, free to
   drift from the word every other tilegrid consumer shares, and a drifted flip
   mask silently reads a mirrored tile as a tile id in the billions.
9. **The reused Ctrl+S toast for an `.aseprite`-backed tab is its own sentence,
   not the shared "came from a {SUFFIX} file" one.** Fixed in this task's sweep:
   the shared sentence read "This drawing came from a ASEPRITE file" for a
   document that was *written* by Save As a moment ago, not imported —
   ungrammatical and backwards about how the document got there. It now reads
   "This file was written by Save As; Inker does not overwrite .aseprite in
   place. Save a copy or an .ora."
10. **Two defensive refusals added in this task's sweep, neither reachable through
    the studio today, both cheap insurance against a future relaxation of an
    invariant that currently prevents them.** A still-document `TilemapCel` whose
    `tileset_uid` is `None` used to hit `int(None)` — a bare, unnamed
    `TypeError` — because every real construction site sets a real slot uid;
    `_still_tileset_uid` now refuses it by name, `_rows`' own dangling-track
    refusal one layer earlier. A zero-tile tileset used to hit
    `np.concatenate([])` — a bare, unnamed `ValueError` — because a real
    `Tileset` cannot actually validate with fewer than one tile (its own
    `__post_init__` requires at least one column and one row); `_strip_bytes` now
    refuses it by name, exercised against a duck-typed stand-in since no real
    `Tileset` can reach the branch.
11. **`_slice_chunk`'s `max(0, ...)` on a bound's width/height is left as
    written, not clamped to 1×1.** Judged and documented rather than changed: the
    one way this package's own `Slice`/`SliceKey` genuinely holds a zero-size
    rectangle is `document_from_aseprite` reading a *hidden* key straight out of
    a foreign file (`asein._read_slice`'s own "no hidden state here" choice, which
    keeps the rectangle rather than inventing one). Writing that rectangle back
    as zero reproduces Aseprite's own "hidden" semantics faithfully on the next
    read; clamping to 1×1 would fabricate a visible box the source file never
    declared. Every rectangle this package's *own* editing funnel produces is
    already at least 1×1 (`transform.clamp_rect`'s floor), so an ordinary
    drawing never reaches the branch either way.
12. **One new numbered divergence, #22, not two.** A single-frame `.aseprite`
    opens as a still document (`Document.anim is None` has no room for a
    one-frame animation, where Aseprite's own document is always a timeline), and
    a tag declared over that lone frame is dropped with a warning — this is a
    genuine document-model divergence, cited in `asein.py` at the point it fires,
    and the INVARIANTS.md count word is updated to match. `_first_set`'s slice
    widening (item 5 above) was weighed against the same test and judged **not**
    to qualify: it is a mapping detail of the writer alone, mirroring a
    constraint Aseprite's own file format already imposes on itself, not a place
    the two *document models* disagree — so it stays an interop-report line with
    no number.

### Left open / owed by Wave 5 (and by the program as a whole)

- **The real-Aseprite manual pass, front and center.** Every gate above proves
  `aseout` and `asein` agree with each other; none of it proves real Aseprite
  agrees with either. `tests/inker/fixtures/aseprite/FIXTURES.md`'s "What is
  owed" section names the four fixtures worth authoring first, in priority
  order, and states plainly that every file in the corpus today is
  aseout-synthesized. `docs/ASEPRITE_INTEROP.md`'s closing paragraph states the
  same rule `docs/PLOTTER_COMPAT.md`'s `TILED_VERSION` gate does: the claim only
  strengthens once a human with the app installed has looked.
- **Wave 4's Tiled `.tsx` re-verification is still open**, carried forward
  unchanged from Wave 4's own "Left open / owed" section: a grid pack's `.tsx`
  geometry moved (pow2-rounding off by default) and needs a fresh look with real
  Tiled installed before the standing "no fixture is Tiled-authored" caveat in
  `docs/PLOTTER_COMPAT.md` can shrink. Unrelated to Aseprite; named here only
  because this is where the program's waves close out.
- **No palette chunk is written for an ordinary RGB document with no palette at
  all.** Noted as a concern in Task 1's report and never revisited: Aseprite
  writes a palette into every file it saves, and this writer omits the chunk
  entirely when the document has none. Tolerated by this reader and, by reading
  of the format, by Aseprite's own pre-1.0 fallback — but it is exactly the kind
  of detail only the manual pass above can actually confirm.
- **Tilemap/tileset chunks are the highest-risk part of the untested-against-real-
  Aseprite surface**, per Task 1's own concern: their field order was inverted
  from the *reader*, never checked against a file Aseprite itself wrote.
- **The mode-level submit-test gap** named in Wave 4's own "Left open" section
  (arrange/merge/skip_empty/trim/padding/extrude refusals have no dedicated
  UI-level test) is unrelated to this wave and remains exactly as open as Wave 4
  left it.

**Program status: Waves 0–5 are all DONE.** The P0 gap analysis this file opened
against is closed; the P1 backlog below remains unscheduled by design, pulled
into sessions individually rather than waved.

---

## P1 backlog (appendix — unscheduled; items are pulled into sessions individually, never waved)

From the gap analysis, with foundations noted:

- **Richer brushes**: angled square/line nibs; persistent custom-brush slots
  (`Stamp` exists; slots are `inker_state` + config persistence); source- vs
  destination-aligned patterns (`STAMP_ALIGN` exists — `stamp_align` is already the
  seam); pattern fills; **Shift-to-line-from-last-point** (pure `inker_canvas` press
  logic + `line_pixels`, small).
- **Selection ergonomics**: Copy Merged (flatten-then-copy — `frame_flat` exists);
  Color Range (global `similar` exists — needs the UI door); auto-select layer/cel
  under cursor; wraparound pixel shifting (lands with Wave 2's `shift_range`); new
  sprite from selection; transformations over timeline targets (lands in Wave 2).
- **Shapes**: rounded rectangles; rotate-before-commit (the `FloatingBuffer`
  session already re-renders from source — the seam exists); movable transform
  pivot; interactive skew handles (`shear` kernel exists; handles are
  `_transform_input` work).
- **Layer types**: real background layer (vs the `matte` divergence #6 — re-argue
  then); non-editable reference layers (asein already opens them hidden); empty
  groups; solo mode; per-layer/cel user colors + metadata (divergence #14 — needs a
  model first).
- **Color tools**: color curves; per-channel adjustment targeting; custom
  convolution matrices (`filters.py` FILTERS registry is the extension point); more
  palette color-model controls.
- **Onion-skin controls**: current-layer-only (trivial after `frame_stack` — filter
  the fold to one track); configurable tints (`ONION_BACK`/`ONION_FORWARD` become
  state); opacity progression curve; tag-aware wrapping (`advance`'s span logic is
  the reference).
- **Export presets**: partially land in Wave 4 (per-document destination+options
  memory); the P1 half is cross-format preset recall.

Also carried from exploration, unscheduled: the measured `symmetry="xy"` 16×
per-dab invalidation cliff (union-rect defect, needs its own measurement doc before
a fix — the constants rule).

## Non-goals (citable)

- **Pen/tablet pressure input** — dropped per user decision 2026-08-17. The spike
  (`docs/measurements/2026-08-15-tablet-pressure-spike.md`) is the record; its
  route-B prescription (`SDL_SetWindowsMessageHook` → `io.pen_pressure`) remains
  valid if hardware ever arrives, but no wave here builds it. The velocity taper
  stands.
- **ICC color management** — divergence #3 re-examined and held (Wave 1 rationale).
- **Per-frame palettes** — divergence #20 (Wave 1 rationale).
- **Per-cel opacity / z-index** — divergences #1/#12 stand (Wave 2 rationale).
- **Live cross-workspace tileset registry** — frozen-snapshot handoff only (Wave 3
  rationale).
- **Arbitrary discontiguous layer multi-select** — track ranges only (Wave 2).

## Execution model

Each wave (or chunk of Waves 1 and 3) is a session: write the detailed
implementation plan at execution time (superpowers:writing-plans), arguing from this
spec; execute with per-chunk suite runs; update the wave status line here and the
manual/INVARIANTS in the same change. The standing negative control (RGB documents
byte-identical) is checked at every wave boundary. Version numbers follow the
commit convention (stay at the current version unless the user asks for a bump).
