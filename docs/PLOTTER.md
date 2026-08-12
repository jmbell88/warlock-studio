Plotter → Tiled parity: editing ergonomics & UX familiarity
  
 Context

 Warlock's Plotter mode (shipped v0.0.14, engine deepened in v0.0.20–21) is a deliberately narrow
 tile-map editor: orthogonal + isometric fixed-size maps, GID cells with Tiled flip flags, tile/object
 layers, typed custom properties, a 47-case blob autotile system, and byte-deterministic
 .wmap/TMX/TMJ round-trips. Compared to Tiled, its data model is respectable but its editing
 ergonomics lag badly: no tile selection, no copy/paste, no brush flip/rotate, no line or ellipse
 drawing, no layer lock, no UI to edit map/layer custom properties (the model fully supports them),
 no way to move or resize an existing object on the canvas, and no minimap. Keybindings also diverge
 from Tiled's where they don't have to.

 User-approved scope: editing ergonomics + Tiled UX familiarity, one implementation plan,
 refuse-by-name invariant kept for everything else.

 Out of scope (decided)

 - Data-model breadth: hex/staggered maps, group/image layers, offsets/parallax/tint,
 polygon/ellipse/text/tile objects, rotation, image-collection tilesets, per-tile
 properties/animation/collision — remain named refusals per docs/INVARIANTS.md.
 - Autotiling depth: foreign Wang sets, terrain-brush over arbitrary tilesets, automapping.
 - Ecosystem: infinite maps, worlds, projects, .tsj/zstd, plugin export formats.
 - Autosave/crash recovery — already planned as the cross-mode journal service (audit item 4D,
 AUDIT_PLAN.md:105); nothing here grows a mode-local crash copy.

 Architecture (verified)

 - Headless engine src/warlock/studio/plotter/ — purity pinned by
 tests/plotter/test_plotter_imports.py (stdlib + numpy + lazy Pillow + studio.undo only).
 - Studio layer: plotter_mode.py / plotter_state.py / plotter_io.py / plotter_tilesets.py,
 five panes (panes/plotter_canvas.py, _tools, _tileset, _layers, _bridge), GL cache
 panes/plotter_textures.py.
 - Binding house rules: undo by uid never index; no-op writes push nothing; a whole drag is one undo
 step (stroke sessions, tilemap.py:152-173); pure tools.py region functions that never mutate
 input; two renderers (canvas quads vs render.py) must agree
 (test_plotter_mode.py:588); byte-deterministic saves/exports; plotter_state.TOOLS drives
 buttons + shortcut table; disabled-not-hidden while saving; manual chapters
 docs/manual/09-plotter.md + 14-shortcuts.md are test-gated (tests/manual/).

 Key design decisions

 - D1 Selection is view-state on PlotterState (select: tuple[int,int,int,int] | None,
 normalized inclusive cell rect), like brush/selected_object — Tiled selections aren't undoable
 either, and an Edit would dirty a saved doc from a marquee drag. Cleared in
 _forget_document_state (plotter_state.py:213) and by staged Esc / Ctrl+D; NOT cleared by undo;
 clamped at use time against current map size.
 - D2 Clipboard is per-document, refused-by-name across maps (clipboard: np.ndarray | None +
 clipboard_doc: str = source PlotterDoc.uid). Gids are global to a map's firstgids; remapping
 across maps is unreliable. Paste in another tab → toast naming the reason.
 - D3 Paste becomes a stamp (Tiled's model): Ctrl+V sets state.brush = clipboard.copy() +
 state.tool = "stamp". No new Edit type; reuses stamp path, stroke sessions, no-op rule. Stated
 divergence (manual): our stamp replaces wholesale, so pasted empties erase.
 - D4 Selection constrains tile tools except Terrain — stamp/erase/shape/line via a pure bbox
 intersection on the returned region; flood fill masks the match (post-hoc clipping would let a
 fill leave and re-enter). Terrain exempt: the blob re-fit touches the 8-ring by design.
 - D5 Lock is document-state, undoable, serialized; enforced at the studio layer. The engine's
 write_region/set_object don't check it (like visible; undo must land on locked layers).
 Lock blocks content edits; rename/opacity/reorder/delete/unlock stay allowed (Tiled semantics).
 - D6 .wmap stays VERSION=2 with "locked" written unconditionally, read via
 entry.get("locked", False) — old builds reading new files just drop the lock on resave.
 TMX/TMJ write locked="1" / "locked": true only when locked (the visible="0" idiom) so
 every existing export stays byte-identical.
 - D7 Line = Shift modifier on Stamp (Tiled's model), one stroke session = one undo step; the
 same tools.line interpolates fast drags (fixes skipped cells).
 - D8 Rect tool becomes "Shape" (key P) with rect/ellipse mode toggle — Tiled's Shape Fill.
 - D9 Object drag/resize uses an engine edit session (begin_object_edit/end_object_edit)
 mirroring stroke sessions: snapshot at press, live mutation while dragging, one ObjectPropsEdit
 at release, idempotent end, committed by undo()/redo().
 - D10 Minimap is a canvas-corner overlay, one pixel per cell from a per-tileset mean-color LUT
 (not a render_map downscale — a 512² map is a few numpy ops, not a 268-MP blend), drawn as one
 add_image_quad through the four projected map corners (isometric diamond falls out free).
 No new pane → no 1100px-window risk.
 - D11 Busy-guarding: _MUTATING_CTRL (plotter_mode.py:363) gains only "x" (cut mutates).
 Paste, Ctrl+A/D, and X/Y/Z brush transforms are view-state → deliberately unguarded. Delete key
 gets an inline tab.busy guard.

 Proposed keymap

 ┌──────────────────────────────────────┬──────────────────────────────┬──────────────────────────────────┬────────────────────────────────────────────────────┐
 │                Action                │             Old              │               New                │                        Why                         │
 ├──────────────────────────────────────┼──────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────────────────┤
 │ Stamp / Erase / Terrain / Pick       │ B / E / T / I                │ unchanged                        │ B, E, T already Tiled's; Tiled has no eyedropper   │
 │                                      │                              │                                  │ letter                                             │
 ├──────────────────────────────────────┼──────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────────────────┤
 │ Bucket fill                          │ G                            │ F                                │ Tiled (Ctrl+G grid toggle unaffected)              │
 ├──────────────────────────────────────┼──────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────────────────┤
 │ Shape fill (rect+ellipse)            │ R (rect only)                │ P                                │ Tiled Shape Fill                                   │
 ├──────────────────────────────────────┼──────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────────────────┤
 │ Rectangular select                   │ —                            │ R                                │ Tiled (freed by the move above)                    │
 ├──────────────────────────────────────┼──────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────────────────┤
 │ Select objects                       │ O                            │ S                                │ Tiled (plain S; Ctrl+S save is a chord)            │
 ├──────────────────────────────────────┼──────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────────────────┤
 │ Flip brush H / V, rotate CW          │ —                            │ X / Y / Z                        │ Tiled                                              │
 │ (Shift=CCW)                          │                              │                                  │                                                    │
 ├──────────────────────────────────────┼──────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────────────────┤
 │ Line from last painted cell          │ —                            │ Shift+click (Stamp)              │ Tiled                                              │
 ├──────────────────────────────────────┼──────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────────────────┤
 │ Copy / cut / paste                   │ —                            │ Ctrl+C / X / V                   │ Tiled + Inker                                      │
 ├──────────────────────────────────────┼──────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────────────────┤
 │ Select all / deselect                │ —                            │ Ctrl+A / Ctrl+D (also            │ Tiled / Inker                                      │
 │                                      │                              │ Ctrl+Shift+A)                    │                                                    │
 ├──────────────────────────────────────┼──────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────────────────┤
 │ Delete selection or selected object  │ —                            │ Delete                           │ Tiled + Inker                                      │
 ├──────────────────────────────────────┼──────────────────────────────┼──────────────────────────────────┼────────────────────────────────────────────────────┤
 │ Esc                                  │ drag + object selection at   │ staged: drag → object →          │ Inker idiom                                        │
 │                                      │ once                         │ selection                        │                                                    │
 └──────────────────────────────────────┴──────────────────────────────┴──────────────────────────────────┴────────────────────────────────────────────────────┘

 New TOOLS tuple (order = grid order): stamp B, erase E, fill F, terrain T, shape P, select R, pick I, object S. Everything derived updates itself (tool grid,
 TOOL_KEYS, shortcut-table tool
 rows, test_a_tool_letter_picks_that_tool). Hand-written copies that must move: the TOOLS
 comment at plotter_state.py:53-55 (currently justifies G/R by Inker parity — rewrite for Tiled
 parity), the drag_kind comment at plotter_state.py:159, docs/manual/09-plotter.md tool table,
 docs/manual/14-shortcuts.md, and the literal extra rows in main.py:3567-3585.

 Icons (no codepoint guessing — catalogue rule, icons.py:1-19): shape → existing icons.SQUARE,
 select → existing icons.SQUARE_DASHED. Lock toggle needs LOCK/LOCK_OPEN added to
 icons.py — transcribe from lucide-static 0.525.0 font/info.json (dev-time fetch, e.g.
 unpkg/jsdelivr; this is the docstring's own prescribed procedure, valid against the vendored
 lucide.ttf).

 Tasks (ordered; each lands green on uv run pytest)

 T1 — Perf enabler: kill the per-cell tileset scan in plotter_canvas._layers

 Recorded defect (docs/measurements/2026-08-09-native-batch-2.md:138-147): per visible cell per
 layer, linear scan over refs.items() calling ref.holds() (plotter_canvas.py:265-270).
 - Engine: MapDoc.tileset_epoch: int = 0, bumped in _attach_tileset/_detach_tileset/
 _swap_tileset (_map_tilesets.py — the hooks edits replay through, so undo/redo bump too).
 - Canvas: module memo {tab_uid: (epoch, {tile_id: tileset_index | None})}; drop entry when epoch
 moves; evict in plotter_mode.close_tab's drop() beside plotter_textures.release_doc.
 - Tests: tests/plotter/test_tilemap.py (epoch moves with attach/detach/swap/undo/redo);
 tests/test_plotter_mode.py (seeded memo not re-scanned; stale epoch refills).

 T2 — Brush flip/rotate (X/Y/Z)

 - Engine tools.py: pure flip_brush_h/flip_brush_v/rotate_brush_cw — np.fliplr/flipud/rot90 +
 vectorized flag algebra (mirror toggles the other axis where FLIP_D is set; rotation is the
 exact 8-state permutation table).
 - Oracle tests (tests/plotter/test_tools.py): for all 8 flag masks on a non-symmetric tile,
 render.orient of the transformed gid pixel-equals the numpy transform of render.orient of the
 original. Plus: 4 rotations = identity, double flips = identity, non-square brushes, gid 0 stays 0.
 - Studio: plain-key x/y/z branches in handle_key when state.brush is not None
 (Shift+Z = 3× CW). View-state → no busy guard. _cursor footprint follows shape for free.
 - Mode tests: head unchanged by X/Y/Z; rotate-then-stamp matches flat renderer (extends the
 two-renderer family).

 T3 — Line drawing (Shift+click Stamp) + drag interpolation

 - Engine tools.py: pure line(x0, y0, x1, y1) -> list[(x, y)], integer Bresenham, endpoint-
 inclusive, no clipping (stamp clips per placement).
 - State: last_paint: tuple[int,int] | None (cleared in _forget_document_state),
 drag_last_cell cleared in clear_drag.
 - Canvas: Shift+click with last_paint → begin_stroke, brush at every line(...) cell,
 end_stroke (one step); plain click sets last_paint. In the paint drag branch, interpolate
 non-adjacent consecutive cells through line(...) inside the open session. Preview line in
 _cursor (lattice-corner math → isometric works).
 - Tests: 8-connectivity, steep/shallow/degenerate, reversal symmetry; one-undo-step mode test.

 T4 — Shape fill (rect + ellipse)

 - Engine tools.py: fill_ellipse(data, x0, y0, x1, y1, value) -> Region | None — centre-of-cell
 test via np.ogrid against the inscribed ellipse; degenerate 1-wide/tall drags → line/cell.
 - State: tool key "rect" → "shape"; shape_mode: str = "rect" (app-level, like tool).
 - Panes: rect/ellipse toggle in plotter_tools.py when shape is active; _apply_rect →
 _apply_shape dispatch; add a drag preview outline (rect = 4 lattice corners, ellipse =
 ~32-sample polyline through cell_corner space) — today's rect tool has no preview at all.
 - Grep hazard: state.tool == "rect" at plotter_canvas.py:355,427 must become "shape";
 drag_kind == "rect" (the gesture) and _object_input's object kind = "rect" stay.
 - Tests: ellipse bbox-only, interior-preserved, edge clipping, degenerates; dispatch test; smoke
 toggles both modes.

 T5 — Keymap pass + Select tool skeleton (one commit — R swaps owners atomically)

 - plotter_state.py: new TOOLS; select field; _forget_document_state clears
 select/last_paint; rewrite the two comments named above.
 - plotter_mode.handle_key: staged Esc (drag → selected_object → select, consumed always);
 Ctrl+A select-all, Ctrl+D / Ctrl+Shift+A deselect (view-state, unguarded).
 - Canvas: drag_kind = "select" — press anchors, drag updates normalized state.select live,
 release keeps, plain click clears (Tiled). Marquee overlay via cell_corner parallelogram,
 low-alpha accent fill + outline, above _grid, below _cursor.
 - main.py:3567+: add literal rows (X/Y/Z, Shift+click, Ctrl+C/X/V, Ctrl+A/D, Delete, Esc).
 - Docs: both tables rewritten; 09-plotter.md gains "Selection and the clipboard" section stub.
 - Tests: test_a_tool_letter_picks_that_tool self-updates; staged-Esc test; Ctrl+A-moves-no-head
 test; test_switching_tabs_drops_the_palette_and_the_object_selection keeps passing.

 T6 — Copy / cut / paste / delete + selection-constrained painting

 - State: clipboard, clipboard_doc (app-level, survive tab switches so the refusal can name its
 source).
 - _ctrl_key: c copy (active tile layer ∩ clamped selection → owned copy; toasts for no
 selection / not a tile layer); x copy + one zero write_region (one step; add "x" to
 _MUTATING_CTRL); v paste per D2/D3 (cross-doc refusal toast). Delete (plain branch, tab.busy
 guarded): object tool + selected_object → remove_object (lock-checked after T7); else
 selection on a tile layer → one erase write_region.
 - Engine tools.py: pure clip_region(x0, y0, region, bounds) -> Region | None;
 flood_fill(..., bounds=None) masks the match array (seed outside → None). flood_mask
 signature unchanged → terrain.fill_terrain untouched by construction.
 - Canvas: apply clip_region/bounded fill when state.select is set; Terrain ignores selection
 (docstring + manual note).
 - Tests: clip_region (disjoint/partial/full), bounded flood can't escape or re-enter, cut is one
 step, paste-is-a-stamp, cross-map paste refused by name, selection constrains stamp but not
 terrain.

 T7 — Layer lock

 - Engine: locked: bool = False on both layer dataclasses (_map_model.py), added to both
 snapshot() dicts (:125-131, :143-149) and set in _apply_layer_props
 (_map_layers.py:135-142) — then set_layer_props(uid, locked=True) is undoable for free
 (verified: set_layer_props filters through snapshot() keys).
 - Serialization: wmap.py writer/reader per D6; tmx.py — write only-when-locked at all four
 sites (TMX/TMJ × tile/object layer), read tolerantly into both constructors.
 - Enforcement (studio): plotter_canvas._layer_for_paint toasts "That layer is locked.";
 _object_input refuses add/move/resize on locked layers; layers-pane object form + delete
 wrapped in begin_disabled when the owning layer is locked; Delete key path checks.
 - UI: lock icon toggle beside the eye in plotter_layers._row using new icons.LOCK/LOCK_OPEN
 (transcribed, see Icons above), greyed while saving.
 - Tests: lock toggles undoably + no-op pushes nothing (test_tilemap.py); .wmap round-trip +
 test_a_version_2_file_without_the_locked_key_opens_unlocked + reopened-doc-same-bytes still
 green (test_wmap.py); TMX round-trip keeps lock AND an unlocked doc's export carries no
 locked attribute — pins write-only-when-set (test_tmx.py); painting a locked layer toasts
 and pushes nothing (test_plotter_mode.py).

 T8 — Map & layer custom-property editors

 - Engine: new MapPropsEdit(before, after) in edits.py (dicts owned in __post_init__ like
 LayerPropsEdit); hook _apply_map_properties; public MapDoc.set_map_properties on
 tilemap.py with the no-op rule.
 - Widget: extract the object property editor (plotter_layers.py:159-201) into a reusable
 property_editor(ctx, form_key, props, on_change) kept in plotter_layers.py (it stays the
 owner; cross-pane imports are house-normal). Whole-replacement-dict discipline preserved.
 - Layer properties: collapsed widgets.header("Properties", persist_key="plotter/layer-props") in
 the active-layer expansion → set_layer_props(uid, properties=...) (rides LayerPropsEdit).
 - Map properties: collapsed header in plotter_tools.py's map section → set_map_properties.
 - Tests: MapPropsEdit undo/redo + dict ownership (test_edits.py); no-op pushes nothing
 (test_tilemap.py); smoke opens both headers and adds a property through each.

 T9 — Object manipulation on canvas (move, resize, Delete)

 - Engine: _object_edit session in tilemap.py/_map_objects.py — begin_object_edit(layer_uid, obj_uid) captures obj.snapshot() (re-begin ends previous);
 end_object_edit() idempotent,
 pushes one ObjectPropsEdit iff changed; MapDoc.undo/redo call it beside end_stroke().
 - Canvas (_object_input/_objects): press hit priority — corner handles (sp(5) squares, sp(6)
 hit radius, selected rects only) → drag_kind = "object-resize" + drag_handle corner name;
 body/point → select + drag_kind = "object-move" with pointer offset; empty space → existing
 draw-new gesture. Move mutates live x/y (Ctrl-held = snap to lattice); resize pins the opposite
 corner, crossing over swaps anchors, normalize at release. A click that never moves pushes
 nothing (session finds no diff). Stale-release sweep at plotter_canvas.py:55-57 grows
 end_object_edit().
 - Tests: session semantics (one edit/drag, idempotent, undo mid-drag commits, no-op drag);
 hit-zone arithmetic (corner beats body, topmost first); crossed-over resize normalizes; smoke
 selects/moves/resizes.

 T10 — Minimap

 - Engine render.py: pure minimap(doc) -> (h, w, 4) uint8 — per-tileset mean-color LUT memoized
 on id(tileset.pixels) (safe: pixels frozen at construction), composited bottom-first with
 _over, honours visible/opacity, ignores flip flags (flips preserve a mean).
 - Canvas overlay _minimap (after _cursor): bottom-right, max sp(160) long edge; array cached in
 ctx.state.preview[f"plotter_minimap:{tab.uid}"] keyed (history.head, len(layers)); texture
 via new plotter_textures.image_texture (released in release_doc/release_all); drawn through
 the four projected map corners (ortho rectangle / iso diamond, one path); viewport rect from
 _visible_range; click/drag recentres view.pan — claimed before tool dispatch in _events.
 Headless (texture is None): outline only (the smoke-suite pattern). Toggle state.minimap
 beside Grid/Show objects.
 - Tests: minimap shape/painted-pixel/hidden-layer/LUT-memo (test_render.py); cache invalidation
 on head movement, click-to-cell round-trip (test_plotter_mode.py); smoke draws both
 projections + toggled off.

 T11 — Docs and manual-gate closure

 Sweep docs/manual/09-plotter.md (tool table, new sections: selection/clipboard, lock, properties,
 minimap, line/shape, divergences from Tiled), docs/manual/14-shortcuts.md, main.py shortcut
 rows, INVARIANTS.md (selection ownership: canvas owns the marquee gesture, tools pane owns shape
 mode — one control one owner; the object-edit session joins the stroke-session sentence).
 CHANGELOG.md entry under the current version. Run uv run pytest tests/manual/ last — it is the
 gate.

 Risks

 1. Export byte-identity: TMX/TMJ tests pin bytes of lock-free docs — write locked
 only-when-set. .wmap reader+writer for locked must land in one change
 (test_a_reopened_document_saves_back_to_the_same_bytes).
 2. Two-renderer agreement: T2's rotation flags tested against render.orient output, never a
 hand-derived table; test_the_canvas_and_the_flat_renderer_agree_about_every_flag stays green
 untouched.
 3. Busy-state: only x joins _MUTATING_CTRL; Delete gets an inline guard; object drags/lock
 toggles already inside tab.busy gates. Extend the refused-while-busy test family.
 4. Esc behaviour change: staged Esc alters observable coupling (mid-drag press no longer
 deselects the object) — pin with a new test.
 5. String drift: tool "rect" → "shape" at exactly plotter_canvas.py:355,427; the drag
 kind "rect" and object kind "rect" are different strings that must NOT change.
 6. Smoke/headless traps: every new overlay needs the texture is None branch; add_rect
 thickness is the 5th positional — prefer _backdrop's polyline idiom.
 7. Session leaks: line stroke + object session closed at the frame-top sweep
 (plotter_canvas.py:55-57) and in undo()/redo() — both chokepoints tested.
 8. Icons: LOCK/LOCK_OPEN codepoints must come from lucide-static 0.525.0 font/info.json;
 if unfetchable, fall back to a text toggle rather than guessing a codepoint.
 9. Suite discipline: never edit src/ while pytest runs (source-scanning tests); Python 3.13
 dedents docstrings (source-scanning tests); commit subjects stay Warlock v0.0.21.

 Verification

 Per task and at the end:
 uv run pytest tests/plotter/
 uv run pytest tests/test_plotter_mode.py
 uv run pytest tests/test_studio_smoke.py -k plotter
 uv run pytest tests/manual/
 uv run pytest            # full suite before any commit
 uv run ruff check .
 uv run python scripts/screenshot_modes.py   # visual: marquee, handles, minimap, shape preview
 Manual in-app checks: paint with a marquee active (constrained); Shift-click a 40-cell line, one
 Ctrl+Z removes it; X/Y/Z a 2×3 brush and stamp on an isometric map, compare with the Ctrl+E library
 render; lock a layer and try every tool + object drag; copy in map A / paste in map B → refusal
 toast; resize an object past its opposite corner; minimap click-navigation at far zoom; save a
 locked-layer map as .tmx and open in Tiled (lock survives, nothing else moved).