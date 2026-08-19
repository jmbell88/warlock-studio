# Artist tools batch & Plotter–Tiled parity: TileSplitter integration, grouped remap, the cleanup suite, and Parts K–Q

## Context

Two features for Warlock Studio, both serving the AI-tileset workflow (SDXL generates 1024×1024
tilesheets with dark separator lines and slightly irregular cells — see
`D:\Projects\TileSplitter\tileset5.png`):

1. **Tilesheet auto-split** — port `D:\Projects\TileSplitter` (detect dark grid-separator bands,
   crop each cell, recompose a clean sheet of uniform nearest-neighbor tiles). Warlock has no grid
   detection anywhere; the Plotter's add-tileset path today slices any image blindly at the map
   tile size with no dialog, which mangles AI sheets whose cells are irregular.
2. **Grouped palette remap** — convert an image onto a chosen K-color palette by clustering the
   source's distinct colors into ≤K groups *relative to each other* (count-weighted), then
   assigning groups one-to-one onto the palette. Existing `method="nearest"` maps each pixel to the
   absolutely-nearest palette color, so 15 mid-grays can collapse onto one target gray; grouping
   keeps dark grays → dark target, light grays → light target, per color family (falls out of
   clustering + min-distance one-to-one assignment, no hue special-casing).

Decisions settled with the user: detector lives in shared headless `studio/tilegrid/`; UI surface
is the Plotter add-tileset flow (detection is a *suggestion the user confirms* — repo rule against
silent re-detection, `inker/sheetin.py` docstring); palette remap is a new `"grouped"` method in
the Inker's existing Convert machinery; the pending working-tree batch commits first.

**Extended 2026-08-18 with Parts C–J:** seven more tools in the same seam, all selected by the
user from a whole-tree survey (Inker, Plotter/Packwright/tilegrid, and the generation→editor
hand-offs), plus a verified performance batch (Part J). Two survey facts drive the biggest ones: `pipelines/pixel.py` already holds a
pixel-scale detector and matte-cleanup ops that no editor can reach (the inker engine is pinned
against `pipelines`/PIL, so every reuse below is a numpy port, the same call A1 makes for
`slicing.py`), and since the ground-set retirement on 2026-08-18 nothing in Warlock can produce a
47-column terrain set at all — terrain painting is only reachable through files authored
elsewhere. Not proposed here because they are already planned elsewhere: everything in
`ASEPRITE_PARITY.md` Waves 3.6/3.7/4/5, its P1 backlog, and its named non-goals.

**Extended again 2026-08-18 with Parts K–Q:** a Plotter ↔ Tiled core-parity program from a
whole-surface gap analysis — the refused rows of `docs/PLOTTER_COMPAT.md` (target Tiled 1.12.2)
supplied the document-model gaps, and a full editor inventory (every citation verified against the
tree 2026-08-18) supplied the tool-side ones. In brief: tile-tool parity (stamp capture from the
map, pattern/random fill, mask selections with a wand, offset and autocrop), object & layer parity
(vertex editing, duplicates, merge down, the image-layer picker, the recursive class editor),
per-tile metadata (properties/class/animation/collision/probability), generic Wang sets,
image-collection tilesets and the tileset presentation fields, hexagonal & staggered lattices, and
infinite (chunked) maps — the M5 seams finally closing. Tiled's permanent non-goals and Plotter's
deliberate divergences stay; the ground rules are stated once at the head of the K–Q section.

## Step 0 — Housekeeping

(Originally: commit the pending ora.py batch — that shipped on 2026-08-18 as `9c531f4`.) What
remains: delete the stray `pytest_run2.out`; leave the uncommitted `inker_mode.py` `TSX_FILTER`
hunk alone — it is the user's in-progress Wave 3.6 start and collides with nothing here. Run
`uv run pytest` and `uv run ruff check .` to confirm the baseline is green before starting Part A;
if red, stop and report before proceeding. One more record, no action: the `plotter-wave-2` branch
is stale — its content was re-landed on master (master is ~11k lines ahead of it in the same
trees), but deleting a branch needs an explicit ask from the user, so it is only noted here.

## Part A — Tilesheet auto-split (Plotter)

### A1. New pure module `src/warlock/studio/tilegrid/slicing.py`

numpy-only, nothing under `warlock`, no PIL (nearest resize via integer index arrays). The tilegrid
import-pin test passes automatically for a new module. Module docstring states the sheetin rule:
detection is a suggestion; the caller confirms with the user.

- Constants: `DARK_THRESHOLDS = (20, 30, 40, 50, 60)`, `MIN_DARK_RATIO = 0.98`, `MAX_SIZE_CV = 0.15`.
- `SheetGrid` frozen dataclass: `rows`/`cols` (inclusive `(start, end)` content segments),
  `threshold`; properties `shape`, `tile_count`.
- `find_bands(mask) -> list[(start, end)]`; private `_segments(bands, length)`, `_size_cv(segments)`.
- `detect_grid(pixels, *, threshold=None, min_dark_ratio=..., max_size_cv=...) -> SheetGrid | None`:
  dark = `max(R,G,B) <= threshold`; row/col is a separator when ≥ ratio of its pixels are dark;
  valid grid needs ≥2 rows AND ≥2 cols with size-CV ≤ 0.15, else `None` (no fallback threshold —
  the caller's fallback is the blind slice). Auto mode tries all five thresholds, picks
  `(most tiles, lowest threshold)`. Raises `ValueError` on a non-(h,w,3|4) array.
- `recompose(pixels, grid, tile_w, tile_h) -> np.ndarray`: per-cell nearest-neighbor resize using
  center-sampled index maps `((2i+1)*src)//(2*dst)` (matches PIL NEAREST); returns a fresh
  `(rows*tile_h, cols*tile_w, 4)` uint8 array; never writes through the source.

### A2. Export from the leaf

Add `slicing` to `studio/tilegrid/__init__.py` imports/`__all__`; add `"slicing"` to the names list
in `tests/tilegrid/test_tilegrid_imports.py::test_the_public_names_are_present`.

### A3. State — `src/warlock/studio/plotter_state.py`

Add to `PlotterState`: `sheet_import: tuple[str, str, str, np.ndarray, Any] | None = None`
(`(uid, name, source, pixels, grid)`) and `sheet_import_open: bool = False`. Clear both in
`_forget_document_state()` (a parked 4 MB sheet names one tab; tab switch drops it). No size field
— the target is the map's own tile size.

### A4. Controller — `src/warlock/studio/plotter_tilesets.py`

- `_sheet_or_tileset(name, source, pixels, uid, tile_w, tile_h) -> dict` (task thread): runs
  `slicing.detect_grid(pixels)`; grid → `{"sheet": (uid, name, source, pixels, grid), "uid": uid}`;
  no grid → today's exact `{"tileset": Tileset(...), "source": ..., "uid": ...}`. Lazy imports
  inside the function, matching existing closures.
- Rewire the three image doors' `run()` closures to call it: `add_tileset_path` (non-.tsx branch),
  `ask_add_tileset`, `use_as_tileset` (Create's own sheets have no separator lines → detect
  nothing → blind slice, unchanged). `.tsx` branch and `invalid_from` wrapping untouched.
- `land_tileset(ctx, state, tab, result)` — extract the adoption tail verbatim from
  `plotter_mode.on_task_done`'s tileset arm (projection/append, `tileset_index`, `brush = None`,
  terrain hand-off, `view.fitted = False`, toast). Load-bearing: the popup's Import must reproduce
  all of it, and the terrain hand-off is easy to miss in a hand copy.
- `import_detected_sheet(ctx) -> bool` / `import_sheet_blind(ctx) -> bool` (frame thread; the
  Packwright `import_tileset` shape — recompose is milliseconds): resolve the tab **by parked
  uid** (`state.get(uid)`, not active; gone → clear fields, False). Detected: `recompose` at
  `tab.doc.tile_w/tile_h` → `Tileset` → `land_tileset` with **`source=""`** (deliberate:
  `TilesetRef.source` feeds `.tmx` export's external-file reference and the recomposed atlas no
  longer matches the file — comment this). Blind: original source kept, byte-for-byte today's
  slice. `ValueError` → error toast + clear + False. Success → clear both fields, True.

### A5. Routing — `src/warlock/studio/plotter_mode.py`

In the `"plotter-tileset"` arm (~line 292): first, `result.get("sheet")` → park on
`state.sheet_import`, `sheet_import_open = False`, return; then the existing tileset arm delegates
to `land_tileset`. The existing closed-tab guard (lines 288–290) already drops parked results.
Re-export `import_detected_sheet`, `import_sheet_blind`, `land_tileset`.

### A6. Pane popup — `src/warlock/studio/panes/plotter_tileset.py`

Model on `panes/packwright_sources.py::_tileset_popup`. `SHEET_POPUP = "plotter-sheet-import"`.
**Trap:** the pump (`if state.sheet_import is not None and not sheet_import_open: open_popup`) plus
`_sheet_popup(ctx, state, tab)` must sit **above** the `if not doc.tilesets:` early return at
line 52 — first-tileset-onto-a-fresh-map is the most common case for this feature.
Popup: name + dimensions; "Detected a {cols} × {rows} tile grid with separator lines (threshold
N)"; muted line explaining Import strips separators and redraws each cell at map tile size.
Buttons: Import → `import_detected_sheet`; "Slice at {W} × {H} instead" → `import_sheet_blind`;
Cancel → clear fields. `begin_popup` False while open-flag set → clear both (click-outside drops
the pixels). Dark art that false-positives the detector is mitigated by this popup's blind button.

### A7. Tests

- New `tests/tilegrid/test_slicing.py` (synthesized sheets via a `_sheet(row_sizes, col_sizes,
  sep, line, fill)` builder): band grouping; uniform grid detects; border lines make no phantom
  cells; slightly-irregular cells (CV < 0.15) detect; wildly-irregular refused; flat image and
  all-dark image detect nothing; one-axis-only is not a grid; darkness is the max channel; the
  0.98 ratio edge is inclusive; auto threshold prefers most tiles then lowest; fixed threshold
  used verbatim; recompose shape/per-cell content; recompose is nearest (output values ⊆ input);
  every separator pixel stripped; source untouched (read-only array).
- Extend `tests/test_plotter_mode.py` (existing FakeCtx inline-submit pattern): gridded sheet parks
  for confirmation; plain image still slices blind; Import lands recomposed tiles at map tile size
  with `source == ""` and state cleared; blind answer matches today's slice with source kept;
  parked sheet for a closed tab dropped on import; tab switch drops the parked sheet; import with
  nothing parked is a no-op; `use_as_tileset` routes through detection; `.tsx` never parks.
  (Existing blind-slice test uses an all-black png → zero content segments → still blind; verified,
  no edit needed.)
- Extend `tests/test_studio_smoke.py`: popup opens over a parked import **on a map with no
  tilesets** (pins the early-return placement); a later frame with popup absent dropped the pixels.

### A8. Docs

`docs/manual/11-plotter.md` "Tilesets" section: one paragraph on the confirm-gated split (help
target unchanged; prose inside an existing section passes `tests/manual/`). `docs/INVARIANTS.md`:
one sentence in the "studio/tilegrid is the second shared leaf" paragraph noting the leaf owns the
separator detector and that detection at the import door is confirm-gated — the existing "generated
sheets are sliced on imposed rectangles, never re-detected" invariant is untouched (this detector
runs only at the user-file door behind a popup).

## Part B — Grouped palette conversion (Inker)

Design: `grouped` is a pure per-distinct-color map (like `nearest`, no dither texture). A **table**
`(keys, targets)` — sorted packed-uint32 distinct visible RGBs → exact palette RGB each lands on —
is built once per conversion scope and applied per plane via the house packed-unique/searchsorted
idiom. **Metric: squared-Euclidean integer RGB throughout** (clustering + assignment); no Oklab —
keeps `dither.py`'s one-metric identity with the write-path snap, exact integer tie-breaks, and the
quality win comes from clustering + one-to-one assignment, not the metric. Final pixels are exact
palette members, so `indexed.snap` cannot move them (hard repo constraint).

### B1. Core — `src/warlock/studio/inker/dither.py`

- `METHODS = ("nearest", "grouped", "floyd-steinberg", *ORDERED)` (line 71; only
  `METHODS[0] == "nearest"` is pinned by `test_ui_tables.py`). Update the comment; add new names
  to `__all__`; docstring paragraph on grouped (and Oklab as rejected-for-now).
- Refactor `_median_cut` (line 416): move the splitting loop verbatim into
  `_median_boxes(colours, counts, want) -> list[np.ndarray]` (disjoint index arrays);
  `_median_cut` = `_median_boxes` + unchanged representative loop. Bit-identity verified by
  existing `build_palette` pins.
- `grouped_table(planes, palette) -> (keys, targets)`:
  1. `keys, counts = _distinct(planes)`; empty → empty table.
  2. Unpack keys to `(N,3) int64`; `K = len(palette)` (duplicates allowed — distinct slots).
  3. Clustering: `N <= K` → each distinct color its own group (injective map — exactly the
     structure-preserving degenerate case); else `_median_boxes(colours, counts, K)` (may return
     M < K boxes; fine).
  4. Representatives: count-weighted mean with `_median_cut`'s exact rounding (one definition of
     "a box's color").
  5. Box rank = lowest sorted-key index in the box (distinct, deterministic).
  6. Assignment: integer `d2[box, target]`; lexsort all M·K pairs by `(d2, box_rank,
     palette_index)`; greedy walk taking pairs whose box and target are both free until all M
     boxes assigned (M ≤ K guarantees completion; distinct targets). Distance-0 pairs win first ⇒
     identity on already-palettized input (a `METHODS`-parametrized test *forces* this).
  7. Scatter membership → `targets` `(N,3) uint8`.
- `_candidate_slots(count, transparent)` — extract the existing hole/candidates rule from
  `convert_indices` so it exists once. `grouped_index_table(planes, palette, *, transparent=0)` =
  `grouped_table` over the non-hole slots (transparent slot never a target, by construction).
- `convert(..., *, table=None)` / `convert_indices(..., *, table=None)`: `ValueError` if table
  given with a non-grouped method; grouped standalone builds from `[pixels]` (seven
  `METHODS`-parametrized tests call `convert` on one plane). `_grouped(out, keys, targets,
  entries)`: visible mask mirrors `_ordered`'s alpha handling exactly (semi-transparent converted,
  fully transparent verbatim); searchsorted lookup; defensive not-found remainder nearest-snaps
  against the palette entries (keeps `convert` total for a stale table). `convert_indices` passes
  `table` through to its inner `convert` — existing parity pins then cover grouped automatically.

### B2. Document layer — `src/warlock/studio/inker/_doc_indexed.py`

- `convert_to_palette` (~line 199): build the table **inside the replay closure** from
  `self._palette_planes()` (whole document, linked cels once, session snapshots substituted),
  then `_map_planes(lambda p: dither.convert(p, wanted, method, table=table), mask_fn=None)`.
  Inside `run()` (not captured) so redo recomputes against restored planes and the undo stack
  never pins a possibly-megabyte table. Leave the `nearest`-only no-op door (~line 232) alone
  (grouped is idempotent on a snapped doc; comment why).
- `_resolve_planes` (~line 671, indexed path): build `grouped_index_table` from
  `[layer.pixels for layer in self._index_planes()]` **strictly before** the loop (the loop
  mutates pixels via `materialize`), pass to each `convert_indices`. No new undo plumbing;
  `check_materialized()` / ORA / one-undo-step flow through existing machinery.

### B3. Preview session — `src/warlock/studio/inker/_doc_paint.py`

`preview_convert` (~line 689): on memo miss, build grouped table from `self._palette_planes()`
(whole document — during preview that is current-frame snapshots + other frames' true planes,
byte-for-byte what commit reads after restore, so preview == commit on the visible frame). Memo
key unchanged. One docstring sentence: grouped's table is document-wide though only the frame is
shown. Known pre-existing indexed preview/commit candidate-table discrepancy applies to all
methods; grouped matches its siblings (do not fix here).

### B4. UI — verify only, no code

The Convert popup combo is built from `dither.METHODS` at the call site (`inker_bridge.py:758–759`);
`"grouped"` appears automatically, labeled by key. `state.convert_method` fallback keys off
`METHODS`. No import-pin edits.

### B5. Tests

- New `tests/inker/test_dither_grouped.py`: fifteen-grays-onto-two keeps contrast (both targets
  used, map monotone in luma); color families don't cross-map; grouped is a flat map (no texture);
  count-weighting decides the split; fewer distinct colors than swatches map one-to-one; one
  distinct color takes its nearest swatch; table deterministic; palette order doesn't change the
  picture (tie-free fixture); a shared table overrides per-plane grouping (disjoint gray ranges);
  missing-from-table color snaps to nearest; table with another method refused.
- Existing `METHODS`-parametrized suites (`test_dither.py`, `test_index_plane.py`,
  `test_palette_convert.py`) pick grouped up automatically — each constraint verified satisfiable
  (identity on palettized input, alpha rules, transparent slot, determinism). The FS/ordered
  interleaving negative control deliberately excludes it.
- `test_palette_convert.py`: grouped builds one table for the whole document (two layers with
  disjoint gray ranges map consistently); redo lands on the same pixels.
- `test_inker_convert_session.py`: grouped preview matches commit on the visible frame (two frames
  whose union changes the visible frame's grouping).
- Indexed invariants: grouped `convert_to_indexed` holds `check_materialized()`, ORA round-trip,
  no opaque pixel on the hole, single undo step.

### B6. Docs

`docs/manual/08-inker.md`, "Converting a drawing onto a palette": one bullet after **nearest** —
grouped clusters the drawing's own colors into as many groups as the palette has entries, one to
one; where nearest collapses fifteen grays onto the closest gray, grouped keeps a darker and a
lighter apart; flat regions stay flat.

## Part C — Pixel-scale descale (Inker)

The single most common AI artifact after ragged tilesheets: an SDXL "pixel art" render at 1024²
whose real art is 64². The measurement already exists (`pipelines/pixel.py::detect_grid` — luma
lattice period **and phase**; reducing off-phase or by plain resample gives "a small photograph
of pixel art", that module's own words) but the editor cannot reach it; `transform.scale` with
nearest is a guess. This part puts the phase-aware reduction on the open document.

### C1. Engine — `src/warlock/studio/inker/transform.py`

numpy ports of the grid trio, in the module that already owns flip/rotate/scale/crop/resize:

- Constants copied verbatim with a comment naming the sibling: `GRID_SCALES = (4, 5, 6, 8, 10,
  12, 16)`, `GRID_RESIDUAL_MAX = 0.05`, `_MIN_CELLS = 4`. The residual threshold is provisional
  in `pipelines/pixel.py` (docs/measurements/2026-08-06-pixel-art-xl.md, pre-registered) —
  copying it moves nothing; both copies cite that document.
- Docstring paragraph disambiguating the **three detectors** this file now ships beside:
  `pipelines/pixel.py` (same lattice measure, PIL, export path), `tilegrid/slicing.py`
  (separator *bands* at import doors, Part A), and this one (the lattice measure inside the
  editor). Name all three in each; a future reader must not merge them.
- `detect_pixel_grid(pixels) -> dict` (`scale`/`phase`/`residual`/`candidate`, `scale=None` when
  nothing passes): vectorized luma with pixel.py's 0.299/0.587/0.114 coefficients — deliberately
  not `dither.luma` (a per-colour helper, and Rec. 709) so the two lattice detectors give one
  answer on one image. Port `_axis_phase` (argmax of gradient-by-position-mod-period) and
  `grid_residual` (interior-over-total gradient ratio; **the worse axis decides**). Keep the
  largest-passing-scale rule and carry its comment: every divisor of a true period passes just
  as cleanly, and the smallest would halve every authored pixel.
- `descale(pixels, scale, phase) -> np.ndarray`: cell-centre selection `arr[np.ix_(ys, xs)]`
  with centres `phase + scale // 2` stepping `scale` — pure element selection, no arithmetic,
  so it can never mint a colour and is exact on index planes by construction. `ValueError` on
  `scale < 2` or when the grid leaves no cells.

### C2. Document — `src/warlock/studio/inker/_doc_geometry.py`

`descale_to_grid(scale, phase)` shaped like the existing `scale()`: `_map_planes` with the same
selection for pixels, index planes and the selection mask (one function — selection is
mode-agnostic), one undo step.

### C3. UI — the Resize popup

`panes/inker_bridge.py::_resize_popup` (line 272). Detection runs **once, when the popup opens**
— a 2048² gradient sweep per frame is a frame-thread stall — over `tab.doc.flatten(matte=False)`,
parked in `ctx.state.preview` under `f"inker_grid:{tab.uid}"` beside the existing `inker_resize:`
key. When a scale is found: a muted "Detected a {scale} px pixel grid — true size {W} × {H}" line
and a "Descale" button (`descale_to_grid` → `tab.view.fitted = False` → close popup), inside the
existing `tab.busy` disabled scope. `scale=None` draws nothing — the ordinary-image case stays
exactly today's popup. Never applied silently (sheetin rule).

### C4. Tests

New `tests/inker/test_descale.py`: `np.repeat`-synthesized k× art at a known phase round-trips
bit-exactly; a smooth gradient detects nothing; the divisor rule prefers the largest passing
scale; output values ⊆ input values; the source array is untouched; the document op moves
layers + mask + index planes together and one undo restores all of them; tiny-image refusal.
`tests/test_studio_smoke.py`: the popup shows the detection line over a synthetic pixel-art doc.

### C5. Docs

One sentence in `docs/manual/08-inker.md`'s scale/resize passage (existing section, no new
heading).

## Part D — Matte-cleanup filter pack (Inker)

BiRefNet cutouts land with halos and semi-transparent fringes (`service/matte.py::alpha_plane`
hands the raw mask to `apply_matte`); the remedy today is the eraser. Four new entries in
`src/warlock/studio/inker/filters.py::FILTERS` (line 513) — the module's own contract ("a sixth
filter is an entry here and no edit in any pane") buys live preview, feathered selection
weighting, alpha-lock respect, one-undo commit and Apply-to-range for free.

### D1. Filters — `src/warlock/studio/inker/filters.py`

All four: pure `(H, W, 4) uint8 → same`, identity defaults (safe frame-1 preview), params
registered in the tables beside the registry.

- **"alpha threshold"** — `alpha_threshold(pixels, *, threshold=0.0)`: 0.0 is the identity;
  else alpha becomes 255 where `>= threshold`, 0 below (`pipelines/pixel.snap_alpha`'s rule at
  the editor). `POPUP_VALUES["alpha threshold"] = {"threshold": 128.0}` — invert's precedent:
  nobody opens it to snap nothing. `RANGES["threshold"] = (0.0, 255.0)`. Touches alpha by
  design — extend the module head's exception paragraph.
- **"defringe"** — `defringe(pixels, *, fringe=0.0)`: pixels with `0 < alpha < 255` take their
  RGB **copied** from the nearest fully-opaque pixel within `fringe` steps (iterative
  8-neighbour propagation); alpha untouched, so this is a colour filter, not an alpha
  exception. Copy, never average: averaging is exactly the halo being removed, and on a
  palette-locked document a copied colour is already a member where a blend fights the snap
  (Part B's one-metric identity). Its own parameter with a small span — `RANGES["fringe"] =
  (0.0, 8.0)` — for despeckle's stated reason (each step is a full-image pass on the frame
  thread; the shared `radius` tops out at 32). `POPUP_VALUES` seeds 2.0.
- **"grow / shrink matte"** — `matte_grow(pixels, *, grow=0.0)`: signed steps; negative erodes
  coverage (rim alpha → 0), positive dilates (new rim pixels copy the nearest opaque
  neighbour's colour, alpha 255 — the defringe propagation with coverage). `RANGES["grow"] =
  (-8.0, 8.0)`. An alpha exception on purpose: coverage is the point.
- **"remove orphans"** — `remove_orphans(pixels, *, orphans=0.0)`: toggle (`TOGGLE_PARAMS` +=
  `"orphans"`, `POPUP_VALUES` seeds 1.0). `pipelines/pixel.clean_orphans`'s rule: an opaque
  pixel with no same-colour 8-neighbour takes its neighbours' most common colour; transparent
  pixels are left alone in both roles (an isolated hole is a silhouette decision). Docstring
  carries the despeckle contrast: a median deletes 1 px detail wholesale; this deletes only
  friendless pixels, so a deliberate two-pixel highlight survives. **Port note:** pixel.py's
  per-lonely-pixel Python loop is fine — lonely pixels are few by definition — but the lonely
  *test* must stay the vectorized neighbour-stack comparison; a per-frame-pixel loop under a
  live preview is not acceptable.

### D2. Tests

New `tests/inker/test_filters_matte.py`: threshold 0.0 is bit-identity and the snap is exact at
the boundary; defringe recolors a synthesized halo ring from the subject colour, alpha untouched,
written colours ⊆ existing opaque colours; grow-then-shrink is *not* asserted as identity
(corners are lost — assert per-sign coverage monotonicity instead); orphan removal: lone pixel
recoloured, a 2 px pair survives, holes untouched. The existing `FILTERS`-parametrized
identity/shape sweeps pick all four up automatically — verify they do before writing duplicates.

### D3. Docs

Four bullets in `docs/manual/08-inker.md`'s filter list; the module-head alpha-exception
paragraph names threshold and grow/shrink as the new deliberate exceptions.

## Part E — Detection at the Inker's sheet door + tile-size mismatch (riders on A)

### E1. Prefill the Inker's import-sheet popup

The Inker has the same blind-grid problem A fixes for the Plotter: `inker_mode.ask_import_sheet`
(run() at :395) decodes the atlas and `panes/inker_bridge.py::_sheet_import_popup` (:612) makes
the user type cell/offset/padding/count. After the decode, on the same task thread, run
`tilegrid.slicing.detect_grid(atlas)` (the leaf is deliberately importable from inker). When a
grid comes back **with uniform segments** — one cell size per axis; the popup's
cell/offset/padding model cannot express an irregular grid, and recompose is the Plotter door's
move, not this one's — attach `{"suggest": (cell, offset, padding)}` to the result: cell = the
segment length, offset = the first segment's start, padding = the inter-segment gap. The
`inker-sheetin` arm of `on_task_done` seeds `state.sheet_cell` / `sheet_offset` /
`sheet_padding` from it. The popup itself is untouched: fields arrive filled, the live
`grid_rects` recompute stays the single truth, the user still confirms. **Trap:** the three
fields deliberately persist across imports as a convenience — overwrite them only when a
suggestion fired; a `None` detection keeps the last values, never resets to defaults.

### E2. Tile-size mismatch at the library door

`use_as_tileset` (`plotter_tilesets.py:197`) slices at the *map's* tile size and ignores the
sheet's own record: a 32 px AI sheet on a 16 px map silently mis-slices, and Part A's detector
cannot catch it (Create sheets are drawn on imposed rectangles with no separator lines → detect
nothing → blind slice). In A4's `_sheet_or_tileset`, for the library door only: also read the
job's `sheet.json` (`svc_files.job_dir_file(ctx.svc, job_id, "sheet.json")`;
`pipelines/tilesheet.py::sheet_sidecar` records `tile_w`/`tile_h` at :426–427). When the
recorded size differs from the map's, park a mismatch variant for A6's popup: "This sheet was
generated at {32} × {32}; this map's tiles are {16} × {16}." Import slices on the sheet's
recorded grid and `recompose`s each cell to the map tile size (A1's machinery, nearest;
`source=""` is already the library-door rule); the alternate button is byte-for-byte today's
blind slice; Cancel clears. Equal sizes park nothing — today's path exactly. Doors without a
sidecar are untouched.

### E3. Tests

Inker (`tests/inker/` + the mode tests): a uniform detection seeds the three fields; an
irregular grid seeds nothing; no detection keeps the previous values. `tests/test_plotter_mode.py`:
mismatch parks; equal size does not; Import lands recomposed tiles at map size with
`source == ""`; the alternate button matches today's slice; a job without `sheet.json` never
parks.

## Part F — Terrain-set inference (Plotter) — the big one, last

Nothing in Warlock produces a 47-column terrain set since the generators were retired
(2026-08-18, `plotter_tilesets.py` module head), and Tiled Wang import is recognise-or-refuse —
so the whole terrain feature is reachable only through files authored elsewhere. This part
reopens it for user- and AI-made sheets. Settled: **single terrain per sheet, phases = 1** in
v1; suggestion-only, behind the Part A popup family; Tiled import stays recognise-or-refuse
(this analyzer runs only at the user-file door).

### F1. New pure module `src/warlock/studio/tilegrid/roles.py`

numpy-only, sibling to A1's `slicing.py`, same docstring rules (sheetin suggestion clause; the
three-detector disambiguation from C1 — this is the fourth measurement and must name the others).

- Background resolution: if the sheet carries any transparency, background = `alpha == 0`;
  otherwise the most common colour over every cell's outer ring (an opaque mockup sheet on a
  flat backdrop). Record which mode fired.
- Per tile: edge strips (outer `max(1, min(tile_w, tile_h) // 8)` px per side) and corner
  squares of the same depth. A side is *open* when ≥ `EDGE_OPEN_RATIO = 0.6` of its strip is
  background; a corner likewise. Assemble the 8-bit neighbour mask from `blob.py`'s bit
  vocabulary (closed side ⇒ neighbour present), `blob.normalise`, then index into
  `BLOB_MASKS`. These two constants are import-door heuristics behind a confirm popup — nothing
  stored is keyed on them, so no `docs/measurements/` document is owed (state this in the
  module docstring; the CLAUDE.md rule is about corpus-keyed constants).
- `infer_roles(pixels, tile_w, tile_h) -> SheetRoles | None`: `None` unless **every one of the
  47** canonical masks is matched by at least one tile (a partial set is refused silently — the
  popup only ever *offers*, it never nags). First tile in reading order wins a contested mask;
  extras are counted, not used (phase variants are the v2 that would use them). `SheetRoles`
  frozen dataclass: `order` (length-47 tuple of source cell indices, ascending `BLOB_MASKS`),
  `extras`, `background`.
- `reorder(pixels, roles, tile_w, tile_h) -> np.ndarray`: rebuild the atlas as 47 columns × 1
  row in canonical order — fresh array, never writes through the source (A1's discipline).

### F2. Controller — rides A4's `_sheet_or_tileset`

After the separator arm resolves a cell grid (detected-and-recomposed when A found bands, else
the blind map-size grid): when the grid holds ≥ 47 cells, run `infer_roles` on the uniform
cells. Found → park `{"terrain": (uid, name, source, pixels, tile, roles)}` for the popup.
Import → `reorder` → `Tileset(name, pixels, tile_w, tile_h, terrains=(TerrainSpec(...),))` →
`land_tileset`, `source=""` (a reordered atlas no longer matches any file — A4's rule, same
comment). "Import as plain tiles" falls through to whichever plain path applied; Cancel clears.
**Trap:** terrain layout is positional (`Tileset.terrain_of`/`local_for` are the only source of
a cell's role, exactly 47 columns enforced at construction) — the reorder is not cosmetic, it
*is* the role assignment.

### F3. Popup — third variant in A6's `_sheet_popup` family

"These {N} tiles look like a complete 47-case terrain set" + a muted line saying Import reorders
the tiles into the canonical blob layout and enables terrain painting; buttons: "Import as
terrain set" / "Import as plain tiles" / Cancel. The plain button is the false-positive
mitigation, same as A's dark-art case.

### F4. Tests

`tests/tilegrid/test_roles.py`, with the corpus **derived from blob itself** (draw each of the
47 roles programmatically: fill the tile, clear each open edge strip and notch each open corner
per `open_edges`/`open_corners`): canonical sheet infers the identity order; a shuffled sheet
recovers the permutation; one role removed → `None`; 64 random tiles → `None`; the
opaque-background variant resolves via border colour; a 48-tile sheet with one duplicated role →
first wins, `extras == 1`; ratio edge cases at 0.6. `tests/test_plotter_mode.py`: parks; plain
button = plain path; terrain import lands `Tileset.terrains` and `terrain.paint_terrain` works
on the imported set. Smoke: the popup variant renders.

### F5. Docs

`docs/manual/11-plotter.md` terrain section paragraph (existing section); `docs/INVARIANTS.md`
tilegrid paragraph — the leaf now owns the blob-role analyzer, confirm-gated at the user-file
door, same clause as the separator detector.

## Part G — Seamless-tile toolkit (Inker)

Tiled mode renders the 3×3 wrap but offers no numeric truth and no offset move.

### G1. Wrap offset

`transform.translate` already does the whole job — `wrap=True` is an exact permutation, and its
docstring already argues the index-plane fill. Missing is the funnel and the button:
`_doc_geometry.py::offset_layer(dx, dy)` on the active layer through the standard patch funnel,
wrap always on (an un-wrapped offset is the move tool), content-lock refusal like every
structural op, selection mask untouched (the op moves the layer, not the selection), one undo
step. UI: a "Wrap ½" button beside the Tiled combo (`panes/inker_canvas.py:301`), applying
`(w // 2, h // 2)`, enabled only when `tab.tiled != "off"` — the classic put-the-seam-in-the-
middle move; pressing it twice on even dimensions is the identity.

### G2. Seam readout

`src/warlock/studio/inker/tiling.py` gains `seam_ratio(pixels) -> (horizontal, vertical)` — a
numpy port of `pipelines/seam.py::_ratios` (edge-vs-interior mean-absolute ratio; carry the
flat-image 0.0 rule *and* its mean-not-median argument verbatim) — plus `SEAM_MAX = 3.5` copied
with its citation (`docs/measurements/2026-08-08-seam-threshold.md`; a copy at a new surface
moves nothing — the same document governs both, say so in the comment). Drawn beside the Tiled
combo while tiled ≠ off: muted "seam ×{worst:.1f}", `theme.WARN`-coloured above `SEAM_MAX`.
Computed over `tab.doc.flatten(matte=False)` and **cached against the undo serial**
(`UndoStack.head`) — a full-image diff per frame is a frame-thread stall; recompute only when
the serial moves.

### G3. Tests & docs

`offset_layer`: four half-wraps = identity; index planes permute exactly; content-lock refusal;
one undo step. `seam_ratio`: pins on synthesized fixtures matching seam.py's semantics —
wrap-constructed noise ≈ 1, two flat halves with a hard join high, flat image 0.0. Smoke: the
readout and button appear only in tiled mode. Docs: two sentences in `08-inker.md`'s Tiled-mode
section.

## Part H — Sheet-panel hand-offs (wiring only)

The 3D-rendered 8-direction sheet and its pixel restyle both dead-end at Save PNG
(`panes/sheet_panel.py::_saved` buttons :389–395; pixel pair :610–614), though everything needed
to open them exists. Two doors, no new algorithms:

- **Edit in Inker.** The rendered sheet is a uniform grid its sidecar fully describes
  (`pipelines/sheet.py::sidecar` — `columns`, `rows`, `frame_w`/`frame_h` or square
  `frame_size`, `cells`) but it is **not** a `DirectionalLayout` kind (`animation.SHEET_KINDS`
  is `turnaround`/`walk` only) — so the door is `sheetin.document_from_grid` on the sidecar's
  own geometry, **not** `document_from_atlas`. New `inker_mode.open_rendered_sheet(ctx, job_id,
  sheet_id, *, pixel=False)` mirroring `open_sprite_draft` (:516): task-thread load of PNG +
  sidecar, `document_from_grid`, routed through an `inker-open:` key so `on_task_done` adopts
  with no routing change; **unlinked** — no path, first Ctrl+S is a Save As; the sheet on disk
  is where the document came from, not its file. The pixel-restyle block gets the same button
  against its own sidecar.
- **Add to Packwright.** The house pattern is confirm-at-the-door: park the sheet's pixels and
  recorded cell size through Packwright's existing `tileset_import`/`tileset_import_open`/
  `tileset_cell` trio (`packwright_state.py:126–128`) via a new
  `packwright_mode.add_rendered_sheet(ctx, job_id, sheet_id)`, so `_tileset_popup` opens with
  the cell prefilled and the existing occupancy/import path does the rest.

Tests: smoke — both buttons exist and the Inker tab opens with rows × cols frames; a mode test
for the Packwright parking. Docs: one sentence in `docs/manual/07-sprite-sheets.md`'s existing
prose.

## Part I — Tile dedup in Packwright's sheet import

### I1. Pure helper — `src/warlock/studio/packwright/sources.py`

`dedup_tiles(sprites, *, orientations=False) -> tuple[list[Sprite], int]`: exact-bytes key
`sprite.pixels.tobytes()` (`inker/tiles.py::content_key`'s rule, :303); first in reading order
wins (`tile_key`'s zero-padding already makes lexical order the reading order). With
`orientations`, the key is the lexicographic min over the 8 dihedral variants — packwright may
not import inker, so the eight `np.rot90`/flip compositions are written here, with a comment
naming `inker/tiles.py::oriented` (the orientation order) and `tilegrid.gid` (the flag
vocabulary) so three copies of D4 don't drift silently.

### I2. Wire + popup

`packwright_state.py`: `tileset_dedup: bool = False`, `tileset_dedup_flips: bool = False` beside
`tileset_cell`. `panes/packwright_sources.py::_tileset_popup` (:88): a "Drop duplicate tiles"
checkbox with a nested "match flipped / rotated" toggle; the counts line becomes
"{kept} unique of {total} tiles ({dropped} duplicates dropped)" and **must** be computed by the
same `dedup_tiles` call the import runs — the popup-promise contract `tileset_occupancy` already
enforces for emptiness. Default **off**: a repack stays byte-faithful unless asked. The import
door dedups the `sprites_from_tileset` result when the flag is set.

### I3. Tests & docs

`tests/packwright/`: exact duplicates dropped, reading-order winner kept; a flipped duplicate
survives without the toggle and drops with it; rotation variants; an RGB-opaque sheet; popup
preview and import agree on the dropped count; default-off is byte-identical to today. Docs: one
sentence in `docs/manual/12-packwright.md`'s sources section.

## Part J — Performance batch: two C kernels, four algorithmic fixes

From a separate perf review (measurements are that review's own; every code citation verified
against the tree 2026-08-18). The shape matches the house record exactly — native batch 2's
lesson was that most "make it C" candidates are really algorithm or caching problems, and this
review reached the same split on its own. Verified before adoption: current `WARLOCKC_ABI` is 8
(`native/warlockc.h:49`), `WARLOCK_NATIVE=0` forces the fallbacks (`src/warlock/native.py:66` —
the A/B lane the test plan needs), and `scipy>=1.11` is already a **main** dependency
(`pyproject.toml:21`), so the weld fix invents no dependency.

### J1. C kernel — palette nearest (`native/`, ABI 8 → 9)

One kernel, three call sites that today each build an N×P×3 int32 temporary:
`indexed.snap` (:89, distinct-colours idiom), the nearest search inside `index_plane.resolve`
(:128), and **both** searches in `dither._ordered` (:221) — `p1` = nearest entry, `p2` = entry
nearest to the reflection `2c − p1`.

- **Signature correction to the review:** the query colours must be **int32, not u8**. The
  reflection `2c − p1` spans −255..510 by construction, and a u8 kernel would silently clamp
  the very search `_ordered` exists to run. `warlockc_palette_nearest_i32(queries N×3 i32,
  palette P×3 i32, out N i32)`: exact integer squared-Euclidean RGB, **lowest index wins ties**
  (numpy `argmin`'s first-minimum rule — the parity bar is bit-identical results, and integer
  math makes that achievable; no SIMD needed, none wanted).
- The Python side keeps everything else: `snap`'s packed-uint32 `np.unique` reduction,
  `resolve`'s alpha gate / candidate slots / `prefer` override, `_ordered`'s mix arithmetic.
  The kernel replaces only the distance-matrix argmin.
- numpy fallback stays, verbatim, as the reference (`native/*.c` invariant); ABI bump to 9.

### J2. C kernel — four-connected flood (`native/`)

Target is `plotter/tools.py:27` — which is `flood_mask`, not a per-cell Python queue: the deque
was already replaced by frontier dilation, whose pass count is the **path distance** to the
furthest cell. That is exactly why the review's serpentine-corridor case is pathological
(distance ≈ cell count → quadratic work) while the open room is fine. The C kernel restores
O(cells) with a real queue: `warlockc_flood_u8(match H×W u8, seed, out H×W u8)`, iterative
queue over caller-provided scratch, four-connected, bounded by `match`, nothing wraps.
`flood_mask` is shared by `flood_fill` **and** `terrain.fill_terrain` — one kernel serves both.
Parity bar: the exact reached-set, byte-identical to the dilation result by construction.

### J3. Not C — algorithmic fixes (all four citations verified)

- **Ray picking** (`viewer/picking.py:137` `ray_triangles` — vectorised Möller–Trumbore over
  *every* triangle): build a cached BVH; a faster linear scan has the wrong complexity. Two
  traps from the house record: the cache stamp must be a mesh **revision counter**, never
  `id(array)` (that bug class was live in three caches, fixed 2026-08-17), and a drag moves
  positions every frame — picking mid-drag must not rebuild the BVH per frame (keep the linear
  path for a dirty tree, or refit rather than rebuild).
- **Drag-preview normals** (`clay/mesh.py:502` `render_from_layout` — `_newell` over every
  loop per call): make incremental over the moved vertices' faces only.
- **MaxRects** (`packwright/maxrects.py:145` `pack` — linear free-list scan per item over a
  quadratically pruned list): improve pruning/indexing before any native thought.
- **Weld clustering** (`clay/ops_topo.py:424` `_clusters` — Python dict spatial hash +
  union-find under `WELD_SEARCH_LIMIT`): replace with the already-present SciPy stack
  (`cKDTree.query_pairs` + `scipy.sparse.csgraph.connected_components`), keeping the
  above-limit cell-grouping arm as is.
- **Do not revisit** blur, brush resolution, TMX decoding, or per-cell ctypes compositing —
  prior measurements rejected those ports (native batch 2: a *correct* blur kernel was
  reverted; the fused dab was rejected at 6.4%).

### J4. Tests & verification

- Parity (the non-negotiable lane): byte-identical palette results across ties, duplicate
  entries, transparency, 1- and 256-entry palettes, random images, and ordered dithering —
  including negative/overflowing reflection queries; exact flood masks for empty/full maps,
  holes, diagonal contact, corridors, boundaries, out-of-bounds and non-matching seeds. Full
  suite green **twice**: with the DLL and with `WARLOCK_NATIVE=0`.
- Perf lane (`uv run pytest -m perf -n 0`, serial): ≥ 2× palette speedup on high-entropy 512²,
  no meaningful low-colour regression (the flat palette-size curve is the dispatch-bound
  signature — check it stays flat), linear flood behaviour on a 512² serpentine corridor.
- Build via `pwsh native\build.ps1` under the existing `/fp:precise`, no-FMA rules (moot for
  integer kernels, kept for uniformity); clang parity per the existing native test lane.

## Plotter ↔ Tiled core parity — Parts K–Q

Gap analysis anchored on two sources: `docs/PLOTTER_COMPAT.md`'s refused rows (the document-model
gaps, against Tiled 1.12.2) and a full inventory of the editor surface (the tool gaps; all
citations verified 2026-08-18). Ground rules for the whole program, stated once:

- **The permanent non-goals stay non-goals** — worlds/projects, Automapping, object templates,
  plugins/extensions, custom exporter APIs (`PLOTTER_COMPAT.md`'s last table) — and the deprecated
  Tiled features stay refused (pre-Wang terrain types, image-layer transparent colour, tile-space
  layer coordinates): Tiled itself is retiring them and a refusal is the honest state.
- **The deliberate divergences are kept, not "fixed":** tilesets are add-only, paste loads the
  brush rather than dropping a block, the projection is fixed once anything is painted, selections
  stay view state (not undoable, not saved), and Ctrl+D stays deselect (Tiled's duplicate binding
  moves — L2).
- **Every lifted refusal follows the ledger rule** (`docs/INVARIANTS.md`, Plotter paragraph): a
  refusal only ever moves because the editor learned to model the thing — reader, writer, a
  fixture pair under `tests/plotter/fixtures/tiled/`, the `test_tmx_refusals.py` case swapped for
  an acceptance case, and the `PLOTTER_COMPAT.md` row moved, all in the same change.
- **Every new stored field is named in `wmap._document_version`'s gate** (`plotter/wmap.py:112`,
  `VERSION` is 5 today). Version numbers are assigned in ship order, one bump per part that adds
  fields; a document using none of a part's additions keeps writing the oldest readable version —
  the existing rule, restated because five of these seven parts touch the container.
- **`tsx.TILED_VERSION` stays `1.10.2`** whatever lands here: the gate needs a human with a real
  Tiled installed (`PLOTTER_COMPAT.md` head) and nothing in this program can satisfy it.
- **Docs per part:** prose inside existing `docs/manual/11-plotter.md` sections only (a new
  chapter is a renumbering); the "What Plotter refuses" list shrinks as rows move; the shortcut
  table (`main.py:4515`) gains any new binding; `INVARIANTS.md`'s Plotter paragraph gains a
  sentence wherever a stated rule moves (terrain derivation in N, bit 28 in P, the resolver's
  no-dense-shape promise cashing in at Q).

### Part K — Tile-tool parity: capture, pattern fill, real selections, map ops

- **K1 Capture a stamp from the map.** Pick (`I`) is one cell only today (`plotter/tools.py:333`,
  wired `panes/plotter_canvas.py:1405-1418`); Tiled's capture-drag is the gap. New pure
  `tools.capture(layer, rect) -> ndarray` (encoded gids verbatim — the flags travel bit-exactly,
  the standing rule); gesture: with Pick held, press anchors and drag draws the marquee rectangle
  (reuse the selection rect drawing), release sets the brush and switches to Stamp — the paste
  precedent exactly (`plotter_mode.py:588-611`). A plain click stays today's single-cell pick
  byte-for-byte. An all-empty capture is refused with a toast rather than arming an all-erase
  brush.
- **K2 Pattern + random fill.** `flood_fill` (`tools.py:270`) and both shape fills (`:201`,
  `:216`) gain `brush=`: the value written at `(x, y)` is `brush[y % bh, x % bw]` anchored to
  *map* coordinates — the "position mod k" rule terrain phases already use
  (`plotter/terrain.py:176-191`) — so two overlapping fills continue one pattern. A 1×1 brush is
  byte-identical to today (regression pin). Random mode (`plotter_state.py:170`) now applies to
  Fill and Shape as it does to Stamp, per landed cell, `random_stamp`'s choice rule
  (`tools.py:183`). The fill's *match* is still the single target cell's encoded gid — the brush
  decides what is written, never what bounds.
- **K3 Selection becomes a mask; Wand arrives.** The central refactor. `state.select` is one
  inclusive rect (`plotter_state.py:198-206`); it becomes rect-or-mask: a bounding rect plus an
  optional bool array (mask `None` = solid rect, so today's marquee keeps today's cost), still
  view state, still clamped at use (`selection_in`, `plotter_state.py:341-356`). New tool **Wand
  (`W`**, Tiled's letter, unused here): click selects the contiguous same-gid region via the
  shared `flood_mask` (`tools.py:27` — the same helper Part J2's C kernel accelerates; one kernel,
  three callers), Ctrl+click selects every cell of that gid map-wide (Tiled's Select Same Tile,
  folded in because `S` is taken by Objects). Shift adds to the selection, Alt subtracts — on
  Wand and on the marquee both. The doors keep their shape: `_constrained`
  (`panes/plotter_canvas.py:1368`) and `clip_region` (`tools.py:109`) take the mask, and the flood
  keeps its bounded-up-front rule (`bounds=`, canvas `:1464-1470`) by masking the *match* — never
  trimming afterwards, or the fill escapes, runs around the outside and comes back in with the
  trip hidden (the INVARIANTS argument, unchanged). Ctrl+A/D/Esc semantics untouched.
- **K4 Offset command.** `_map_geometry.offset(dx, dy, *, wrap, scope)` — Tiled's Map ▸ Offset
  Map. Scope = active layer or whole map; `wrap` rolls (an exact permutation), un-wrapped shifts
  fill vacated cells with 0; whole-map scope moves objects by the pixel equivalent (`resize`'s own
  rule, `_map_geometry.py:28`); one snapshot edit (`ResizeEdit`'s shape, `edits.py:379`). UI: an
  Offset row inside the existing Resize section (`panes/plotter_tools.py:254-301`).
- **K5 Autocrop.** Bounding box of nonzero cells across all tile leaves (`tile_layers()` walk) →
  delegate to `resize(w, h, off_x, off_y)` verbatim, so objects travel by the rule that already
  exists. A wholly-empty map is refused with a toast. Button beside Resize.
- **K6 Kill the dead generator button.** The terrain empty state's "Open the generator" requests
  section key `plotter/generate`, which nothing has registered since the generator deletion
  (`panes/plotter_tools.py:119-121`; `plotter_tilesets.py:13-17`) — a dead route live today.
  Repoint it at the add-tileset door (`ask_add_tileset`, `plotter_tilesets.py:193`) with copy
  saying terrain sets arrive from a `.tsx` carrying Wang sets (or, once Part F ships, from the
  47-tile import popup).
- **K7 Tests.** Capture→stamp round-trips byte-equal with flags preserved; 1×1 fill byte-identity
  pin; pattern continuity across two separate fills; random fill writes only brush members; wand
  set == `flood_mask`'s reached set; Ctrl+wand == global gid equality; Shift/Alt selection
  algebra; a bounded flood cannot escape a concave mask; offset wrap ×4 = identity, no-wrap
  zero-fills, objects move only under whole-map scope; autocrop is the tight box and refuses an
  empty map; smoke: Offset/Autocrop controls render, Wand row renders, and no
  `widgets.request_open("plotter/generate")` remains anywhere. Manual: a Wand row in the tools
  table plus capture/pattern sentences in the existing Tools section.

### Part L — Object and layer parity

- **L1 Polygon/polyline vertex editing.** Today polygons are created as a fixed 4-point box and
  polylines as 2 points with no way to move a point (`panes/plotter_layers.py:764-767`). Selected
  polygon/polyline objects draw a handle per vertex (the corner-handle idiom,
  `panes/plotter_canvas.py:1646-1691`); dragging one moves it through the object session
  (`begin_object_edit`/`place_object`, `plotter/_map_objects.py:94`/`:143` — one undo step per
  drag), **computed in the object's own frame** — the `_resized` trap by name (canvas `:1691`;
  INVARIANTS): a vertex on a rotated polygon converts through the rotation, never min/max in map
  space. Double-click on a segment inserts a vertex at the projection; a click makes a vertex hot
  and Delete removes it, floored at 3 points (polygon) / 2 (polyline), refused by toast at the
  floor. Shapes are frozen records — a moved vertex is a new `Polygon`/`Polyline` through
  `merged_object_values`, the one reconciliation door.
- **L2 Object duplicate and copy/paste.** **Ctrl+J** duplicates the selected object one cell
  down-right (Ctrl+D is deselect here, `plotter_mode.py:679-687` — a kept divergence from Tiled's
  binding, stated in the manual); the copy takes a fresh persistent id from the monotone counter
  (ids are never reused) and its object-typed properties keep pointing where the original's
  pointed. Ctrl+C/V with the Objects tool in hand routes to an object clipboard (kind-tagged
  beside the tile clipboard, `plotter_state.py:215-216`); cross-map object paste is refused whole
  by the same rule — a tile object carries a gid and an object property may name an id, and both
  mean something else elsewhere. **Multi-object selection is deliberately out of scope** this
  round (marquee-over-objects, group transforms and a mixed form are a program of their own) —
  recorded as an open decision, not quietly half-done.
- **L3 Image-layer image picker.** An image layer created in Plotter stays empty and the pane says
  so (`panes/plotter_layers.py:272-290`). Add **Choose image…** on the image-layer row: OS picker
  (`studio/dialogs`, minding the (label, patterns) pairing rule), task-thread decode (the
  `plotter_tilesets` closure shape), landing as a `LayerPropsEdit` grown to carry pixels
  (`edits.py:168`; cost = the array's bytes via `_own`, `edits.py:35`). `.wmap` already stores an
  image layer's pixels as `images/N.png` — no container change, no version bump from this item.
- **L4 Duplicate layer + merge down.** Context-menu rows beside delete
  (`panes/plotter_layers.py:194-195`). Duplicate copies the subtree with fresh uids and fresh
  persistent ids, landing as one `LayerAddEdit` (`edits.py:96` — the subtree travels whole, and no
  array is shared with the original). Merge down: the active tile leaf onto the tile leaf directly
  below it in paint order, **gid-level, nonzero-above-wins per cell** — a data merge, deliberately
  not a pixel composite: opacity/tint/blend stay presentation, because a merge that baked them
  would disagree with the renderer the first time a tint changed. Refused by name when the layer
  below is not a tile layer. One snapshot edit.
- **L5 Recursive class-property editor.** The property editor (`panes/plotter_layers.py:482`)
  shows a class value as a read-only summary; the manual already promises "until the recursive
  editor lands" — this is that landing. Recursive draw with `push_id` per depth; members editable
  by their stored types, nested classes and dialect lists included; one editor serves map, layer
  and object properties since all three route through `props.py`'s single model.
- **L6 Highlight current layer.** A Highlight toggle beside Grid/Minimap
  (`panes/plotter_tools.py:184-185`); the canvas dims every non-active resolved leaf (draw
  opacity × 0.3) — **in `panes/plotter_canvas` only, never in `scene.resolve`**: `render.py`
  composites exports off the same resolver, and a dim living there would export dimmed.
- **L7 Tests & docs.** Vertex drag on a rotated polygon lands where the outline says (the
  object-frame pin); insert/remove floors; duplicate mints fresh ids; cross-map object paste
  refused by name; merge-down equals the pre-merge pair's render for plain layers and refuses
  across kinds; image-layer pixels survive a `.wmap` round trip; export is byte-equal with
  Highlight on; the class editor round-trips a nested class + list without changing a type.
  Manual: sentences in the existing Layers/Objects sections.

### Part M — Per-tile metadata: properties, class, animation, collision, probability

The largest single unlock in the compat ledger: six refused rows fall to one model.

- **M1 Model.** `tilegrid/tileset.py` grows a frozen `TileMeta` — `class_name`, `properties`,
  `probability`, `animation` (ordered `(local_id, duration_ms)` pairs), `collision` (shape
  records) — stored sparsely as `tiles: dict[int, TileMeta]` on `Tileset` (Tiled's own sparseness;
  most tiles carry nothing). The leaf stays pure: **no upward import** — the collision shapes are
  tilegrid's own small frozen records (rect/ellipse/polygon), and `plotter` converts to/from its
  `_map_model` dataclasses at the codec, or the tilegrid import pin fails. Packwright and Inker
  see only additive fields.
- **M2 Codecs.** `tsx.py` reads and writes per-tile `<tile>` blocks — `properties`, `type`
  (class), `probability`, `<animation>`, `<objectgroup>` (collision) — XML and the JSON twin; the
  refusals at `tsx.py:159-171` / `:208-220` lift one at a time, each with its fixture pair and its
  refusal-case swap. "Per-tile terrain assignment" (deprecated) stays refused. `.wmap` tileset
  entries gain a `tiles` record — **named in the `_document_version` gate** (the program's first
  bump).
- **M3 Edit.** New `TileMetaEdit` in `edits.py`, addressed the way `TilesetReplaceEdit`
  (`edits.py:282`) addresses its tileset, snapshotting one tile's `TileMeta` on both sides —
  kilobytes, so no document snapshot.
- **M4 UI.** A **Tile** header under the tileset palette (below the picker,
  `panes/plotter_tileset.py:89-155`), shown when the palette selection is 1×1: class, probability
  and typed properties (L5's editor generalized), plus **Animation…** and **Collision…** popups.
  Animation editor: ordered frame rows (local id, duration ms), *Add frame from selection* (the
  palette pick is the frame picker), remove, reorder. Collision editor: the tile drawn zoomed in a
  popup with rect/ellipse/polygon creation, move and resize reusing the canvas's object math (the
  object-frame rules again); stored on the tile, drawn as outlines, **never hit-tested against the
  map** — collision is metadata an engine reads, the object-layer sentence a second time.
- **M5 Playback + random weights.** The canvas substitutes an animated tile's current frame at
  draw time — gid → the frame's local id with the cell's flip flags preserved — from a clock the
  pane passes in (injectable for tests); **the document arrays never move** (a clock that wrote
  gids would dirty a saved map every frame). `render.py`, the minimap and every export draw frame
  1: an export is a still (the parallax precedent — canvas and export deliberately disagree, and
  the disagreement is stated). `random_stamp` (`tools.py:183`) takes per-tile probability as
  weights; probability 0 is never chosen by random paint and always placeable by hand — Tiled's
  rule.
- **M6 Tests & docs.** Round-trip per lifted row (XML + JSON + `.wmap`); substitution is
  draw-only (document bytes identical across animation frames); frame-1 export pin; weighted
  random distribution under a forced seed; probability-0 exclusion; each metadata edit is one
  undo step; collision/animation popup smoke. Manual: the Tilesets section grows a per-tile
  paragraph; the refusal list shrinks by six.

### Part N — Generic Wang sets (terrain beyond the blob preset)

The deepest engine change of the program; the regression bar is byte-identity on everything that
exists.

- **N1 Model.** `tilegrid` grows a `WangSet` record — name, kind (`corner`/`edge`/`mixed`),
  colours (name, rgba, probability) and per-tile eight-position wangids — beside `blob.py`, which
  stays: the 47-case collapse becomes the *derivation of the blob preset* (a two-colour set whose
  wangids are generated from `BLOB_MASKS`), and `Tileset.terrain_of`/`local_for`'s positional
  47-column rule becomes that preset's layout rather than the general case.
- **N2 Semantics.** `terrain.py` generalizes with the invariant intact: membership is still
  **derived from the gid, never stored per cell** — the wangid is looked up by tile id, and the
  map goes on storing gids alone. Painting becomes wangid constraint-matching over the touched
  ring (Tiled's terrain-brush algorithm): collect each neighbour's implied edge/corner colours,
  choose a tile whose wangid matches exactly, ties broken by probability then lowest id; **no
  match leaves the cell untouched** rather than writing a near-miss — a wrong tile is the silent
  half-read in picture form. The blob path keeps `_retile_into`'s exact machinery, and the whole
  existing terrain corpus (`tests/plotter/test_terrain.py`) must pass byte-identical.
- **N3 Interop.** The recognise-or-refuse asymmetry ends deliberately: `read_wangsets`
  (`tsx.py:374`, JSON twin `:331`) accepts foreign corner/edge/mixed sets whole;
  `write_wangsets` writes data-driven sets with existing blob-preset exports byte-identical
  (determinism pin). Phases stay a blob-preset feature (the v5 gate unchanged; foreign sets have
  none). `.wmap` stores the wang model → version gate.
- **N4 UI.** The terrain swatch list (`terrains_of`, `panes/plotter_tools.py:88`) becomes one row
  per wang *colour*; tool, status line and shortcuts unchanged.
- **N5 Part-F interplay.** F's role inference still emits the canonical 47-column blob preset —
  its contract does not move; after N, that import constructs the blob-preset `WangSet` rather
  than positional terrain rows.
- **N6 Tests.** Fixture pairs for a corner-only, an edge-only and a three-colour mixed set; paint
  on each with exact-match assertions and the no-match-leaves-cell rule; probability tie-break
  pins; blob-corpus byte-identity; export determinism pin.

### Part O — Tileset kinds and presentation fields

- **O1 Image-collection tilesets.** `tilegrid.Tileset` is one atlas array; add a collection
  variant carrying per-tile images with sparse ids (Tiled permits gaps). `tile_rect`, the palette
  and both renderers branch on it; an oversized tile draws by the anchor rule the renderer
  already has ("a 32px map with 48px trees", bottom-left). Packwright and Inker are untouched —
  additive, and `maxrects` determinism never sees the new type. The refusal lifts with fixtures.
- **O2 External `.tsj` + embedded tileset images.** `.tsj` decoding reuses the JSON tileset
  reading `tmx.py` already runs for embedded JSON tilesets; embedded `<image>` payloads decode
  instead of refusing. Both lift with fixture pairs.
- **O3 Tileset transparent colour.** `trans` is applied at decode (colour-key → alpha 0) — the
  *tileset* row only; the deprecated image-layer twin stays refused.
- **O4 Presentation fields**, each read, written, stored and rendered: `tileoffset` (a draw
  offset in the canvas and `render.py`; the minimap ignores it by its one-pixel-per-cell rule,
  already stated for layer offsets), object alignment (the tile-object anchor, through
  `project.object_to_pixels`), `rendersize`/`fillmode` (grid-sized / preserve-aspect draw), and
  tileset `backgroundcolor` (palette presentation, preserved on the trip). Version gate for what
  `.wmap` stores.
- **O5 zstd.** `zstandard` joins the studio extra — a wheel dependency, not a download; the
  offline invariant is untouched. zstd-compressed layer data reads; writes stay zlib (every Tiled
  reads zlib; the row's note says the claim is read-without-loss).
- **O6 Tests.** Collection round trip plus palette/paint/pick over sparse ids; oversized-tile
  anchor pin; `.tsj`/embedded-image fixtures; colour-key alpha pin; `tileoffset` agreement between
  canvas placement and the flat renderer; a zstd fixture decoding byte-equal to its zlib twin.

### Part P — Hexagonal and staggered maps

- **P1 Lattices.** `project.py` gains `hexagonal` and `staggered` — `Lattice.stagger_axis` /
  `stagger_index` / `hex_side` are already reserved seams (`PLOTTER_COMPAT.md`'s M5 paragraph
  names them). The inverse (`cell_at`) is not affine on these lattices: resolve by candidate
  check — compute the two or three cells whose region can contain the point and test containment
  exactly. The refusal's own wording was "named rather than projected approximately"; the
  acceptance bar is therefore an exact hit test, not a nearest-centre guess.
- **P2 Doors and drawing.** New-map presets and stagger/hex-side fields (`plotter_setup.py` owns
  presets and caps); hex outlines / staggered diamonds in the canvas grid pass
  (`panes/plotter_canvas.py:769`); draw order row-by-row (Tiled's staggered order); `renderorder`
  stays orthogonal-only, as in Tiled.
- **P3 Bit 28.** The hexagonal 120° rotation flag stops being a by-name refusal (`tmx._finish`
  and `wmap._validate` both probe it today) and becomes modeled **on hexagonal maps only** —
  elsewhere it stays refused, Tiled's own rule. Rendering trap, stated up front: 120°/240° is
  **not** expressible as the transpose-then-mirror quad permutation — it is a rotation of the
  four corner *positions* about the cell centre, composed with the flip permutation, and both
  renderers implement it identically. The brush turn (`Z`) steps 60° on a hex map via the
  bit-28 + flip-flag composition — port Tiled's algebra exactly and pin it against a table of all
  twelve states.
- **P4 Terrain on hex.** Wangids on hex flow through Part N's matcher unchanged; if P somehow
  lands first, the Terrain tool refuses on a hex/staggered map by name until N does.
- **P5 Storage, compat, tests.** Projection value + stagger/hex fields in the `.wmap` manifest →
  version gate; the staggered/hex-map and hex-rotation rows move with fixture pairs; new-map
  smoke; hit-test pins on shared hex corners (a click lands in exactly one cell — the isometric
  standard).

### Part Q — Infinite maps (the M5 seams close) — the big one, last

Every seam this needs was held open by name: `scene.resolve` never asks a layer for a dense
rectangle, `.wmap` reserves `infinite`/`chunks`, and the two `WmapUnstorable` handlers keep a
writer-door refusal off the frame thread (`PLOTTER_COMPAT.md`, the M5 paragraph).

- **Q1 Storage.** An infinite `TileLayer` stores 16×16 chunks (`dict[(cx, cy)] → (16, 16)
  uint32`; 16 is what Tiled writes, and its reader takes any size) instead of one dense array.
  `write_region`, the stroke session and `TilePatchEdit` span chunks; content bounds = the union
  of nonempty chunks.
- **Q2 Tools.** Painting allocates chunks on touch; the flood bounds to content bounds ∪ the
  selection (a truly unbounded flood does not terminate); erase leaves emptied chunks to be
  dropped at save (Tiled's shape). The `MAX_TILES` cap becomes a cap on the *populated extent* —
  the same on-a-slip-of-the-keyboard argument (`plotter_setup.py`).
- **Q3 View.** Unbounded scroll; the grid drawn over the viewport; the minimap over the content
  bounding box; the Resize form hidden for infinite maps; autocrop (K5) becomes "shrink to
  content" and is the infinite→finite conversion's core.
- **Q4 Doors.** The new-map dialog gains an Infinite toggle; conversion both ways —
  finite→infinite is a re-chunking, infinite→finite crops to content behind a confirm.
- **Q5 Interop.** `tmx.py:133`'s refusal lifts: chunked XML/JSON layer data in CSV and
  base64(+compression) reads and writes; `.wmap`'s reserved keys activate exactly as reserved —
  additive to the container, the design promise — behind the version gate; the `WmapUnstorable`
  frame-thread handlers stay.
- **Q6 Tests & the M5 ledger.** A chunk-spanning stroke is one patch step; flood termination and
  bounds; save drops empty chunks; `.wmap` and `.tmx`/`.tmj` round trips including negative
  coordinates; canvas smoke over negative-origin content; conversion both ways. Closing M5 also
  retires the code comments that cite it: `PLOTTER_COMPAT.md`'s "`M{n}` citations" section says
  M5 is the last one cited from code — update those comments to describe what shipped, and mint
  no new `M{n}` numbers.

## Order & verification

1. Step 0 (housekeeping; verify green baseline).
2. Part A: A1→A2 + `uv run pytest tests/tilegrid -n 8` → A3–A5 + controller tests → A6 + smoke →
   A7 remainder → A8.
3. Part B: B1 (`_median_boxes` refactor first, confirm `tests/inker/test_dither.py` bit-identity)
   → B5 algorithm tests → B2 → B3 → B4 verify → B6.
4. Part E (rides A's detector and A6's popup): E1 → E2 → E3.
5. Part C: C1 + algorithm tests → C2 → C3 + smoke → C5.
6. Part D: one filter at a time, each with its tests → D2 sweep check → D3.
7. Part G: G1 → G2 → G3.
8. Part H: both doors → tests.
9. Part I: I1 → I2 → I3.
10. Part J: J1 kernel + parity tests → J2 kernel + parity tests → ABI 9 + both-ways full suite →
    J3 items one at a time (each is independent) → J4 perf lane.
11. Part F closes the artist batch — largest of A–I, and the only one with real detection risk:
    F1 + corpus tests → F2 → F3 → F4 → F5.
12. Parts K–Q are a second program and run after the artist batch, in this sequence: **K → L → M
    → O → N → P → Q**. K and L are tool/UX work with no format risk; M opens the tile-metadata
    era and is a prerequisite for N (wangids ride the same per-tile model); O is independent and
    slots between them; N carries the engine risk (byte-identity bar); P rides N for hex terrain;
    Q is the largest change in the whole file and goes last of everything.
13. After every part: full `uv run pytest` (parallel, ~55 s; never edit src/ while it runs) +
    `uv run ruff check .`.
14. End-to-end sanity (artist batch): add `D:\Projects\TileSplitter\tileset5.png` as a Plotter
    tileset → popup detects 16×16 → Import → clean tiles; Convert a tileset in the Inker onto a
    2-gray palette via "grouped" → contrast preserved; open a 1024² AI "pixel art" reference →
    Resize popup detects → Descale → alpha threshold + defringe on a matted cutout; import a
    sprite-sheet PNG → fields arrive prefilled; render an 8-direction sheet from a mesh → Edit in
    Inker; repack a sheet in Packwright with dedup on → the count line drops; add a 47-tile blob
    sheet → terrain popup → paint terrain on the imported set.
15. End-to-end sanity (parity program): capture a 3×2 stamp off the map and fill a wand
    selection with it in random mode; drag a vertex on a rotated polygon; give a tile an
    animation and watch it play on the canvas while the exported PNG stays frame 1; open a
    Tiled-authored map carrying a corner-only Wang set and paint with its brush; open an
    image-collection tileset; author a hex map, rotate the brush through all twelve states, and
    round-trip it; toggle a map to infinite, paint past the old edge, save and reopen.
16. Commits: one `Warlock v0.0.24` commit per part (no version bump, no push unless asked).

## Key risks (carried from design review + the survey)

- Popup pump must precede the `if not doc.tilesets:` early return in the pane (most common case).
- `land_tileset` extraction is load-bearing (terrain hand-off easy to miss in a hand copy).
- Recomposed/reordered imports land with `source=""` (`.tmx` export must not reference a file
  the atlas no longer matches) — Parts A, E2 and F all inherit the rule.
- Genuinely dark art can false-positive detection — the confirm popup is the mitigation, and
  Part F's "Import as plain tiles" button is the same mitigation for silhouette coincidences.
- Grouped table built per plane instead of per document would group each layer differently — the
  whole-document table is the central correctness requirement of Part B.
- **Four detectors now exist** (pixel-scale in `pipelines/` and in the editor, separator bands,
  blob roles) — every docstring names its siblings; they are different mechanisms at different
  doors and must not be merged.
- Frame-thread cost: Part C detects once per popup open and Part G caches against the undo
  serial; Part D's spatial filters preview per slider-frame, so every new parameter gets its
  own small span (despeckle's ceiling argument), never the shared `radius`.
- Defringe and grow **copy** colours, never average — a blend on a palette-locked document
  fights the snap identity.
- E1 must never reset the persisted popup fields on a `None` detection.
- Part H slices through `document_from_grid`, not `document_from_atlas` — the render sidecar has
  no `DirectionalLayout` kind (`SHEET_KINDS` is `turnaround`/`walk` only), and `of()` returning
  `None` would refuse the whole door.
- Part F's reorder *is* the role assignment (terrain layout is positional; 47 columns enforced
  at `Tileset` construction) — importing without reordering would assign every tile a wrong role
  silently.
- Part J's palette kernel takes **int32 queries** — a u8 signature silently clamps `_ordered`'s
  reflection search (`2c − p1` spans −255..510), and the clamp would pass every test that only
  feeds in-gamut colours.
- Part J's BVH cache stamp is a revision counter, never `id(array)`; picking mid-drag must not
  rebuild the tree per frame.
- **Parity-program risks (K–Q):** K3's mask must preserve the flood's bounded-up-front rule
  (mask the match, never trim after — the escape-around-the-outside failure); L6's dim lives in
  the canvas only, never `scene.resolve`, or exports dim; L4's merge-down is gid-level, never a
  pixel composite; M5's animation substitution is draw-only — a clock that wrote gids would dirty
  a saved map every frame; N's bar is byte-identity on the whole existing blob-terrain corpus,
  and its matcher leaves an unmatched cell untouched rather than writing a near-miss; P's 120°
  rotation is a corner transform, not a quad permutation, and bit 28 stays refused off hex maps;
  Q's flood must be bounded (content ∪ selection) or it does not terminate, and `MAX_TILES`
  becomes an extent cap.
- Five of K–Q add stored fields: **every one must be named in `wmap._document_version`'s gate**
  (the field-by-name rule, `plotter/wmap.py`), with version numbers assigned in ship order; and
  `tsx.TILED_VERSION` stays `1.10.2` throughout — the bump needs a human with Tiled installed,
  and it has been wrongly bumped once already.
- `TileMeta` and `WangSet` live in the `tilegrid` shared leaf, which imports nothing under
  `warlock` — collision shapes get tilegrid-local records with conversion at the plotter codec,
  or the import pins fail; Packwright's determinism must never see the new types.
- Deliberate divergences are load-bearing, not backlog: add-only tilesets, paste-loads-the-brush,
  fixed projection, view-state selections and Ctrl+D-as-deselect are all kept by decision, and a
  parity change that "fixes" one is a regression.
