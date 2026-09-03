# FINDINGS.md — Warlock Studio review, 2026-09-02

A whole-tree review of the Studio at v0.0.30 on `feature/plotter-tiled-clay-ergonomics`
(working tree, including the uncommitted trellis v0.6.0, engine-flag and Sirens
clipboard work). Seven parallel passes, one per subsystem, each verifying claims
against the code; the two Critical crashes and the Plotter export overwrite were
re-verified by hand. `uv run ruff check .` is clean and the targeted test files
touched by in-flight work pass (373 passed, 1 environmental skip).

This is a findings list, not a plan. Per the repository's rule, anything here
that can be built should be built and the entry removed, not ticked. Severity
tags: **Critical** (crash, data loss, or corrupts a document), **High** (wrong
result the user will hit, or a promise in the docs the code breaks), **Medium**,
**Low**. Line numbers are against the working tree at review time.

---

## 0. Do these first

All twelve entries that stood here (the two Critical crashes, the Plotter export
overwrite, the shared undo-gesture bug and the eight High data-loss items) were
built on 2026-09-03 and struck; `tests/test_findings_section0.py` is the
regression for each. What remains below is everything else.

---

## 1. Cross-cutting themes

These recur in more than one mode and are cheaper to fix once.

**T1. Slider and drag doors push one undo step per frame.** Sirens tempo/speed/rows
(`panes/sirens_transport.py:73-83`, `sirens_orders.py:120-125`, `sirens_effects.py:201-211`),
Sirens envelope drags (`sirens_envelopes.py:498-506`), Sirens name fields, Inker
`recolour_slot`, `set_cel_opacity`, `set_cel_z`, `set_group_props`
(`inker_colors.py:301`, `inker_picker.py:142`, `inker_timeline.py:1976-2004, 1566`).
With `UNDO_MAX_DEPTH = 64`, one second of dragging evicts every earlier edit in
the document. `controls.slider_int(commit=True)` and the `was=` +
`is_item_deactivated_after_edit` idiom in `inker_menu.header_controls` already exist.
Make "one gesture, one step" a rule with a test per door.

**T2. Blocking work on the frame thread.** Review's `create_sweep` (20–40 `create_job`
calls with NVML reads, `review_mode.py:1455-1493`); Review mesh loads
(`main.py:5200-5215`); the reference PNG decode on job completion
(`viewer_embed.load_reference:183-192`); Clay crash-recovery decode
(`clay_mode.py:1047`); Troupe atlas decode+upload (`troupe_mode.py:827-836`, up to
1024×8192 RGBA); Inker `_reload_linked` (`inker_mode.py:2817`); Packwright save/export
PNG encode (`packwright_io.py:153, 179, 282`); Sirens re-encoding every sample to WAV
per render snapshot and per journal tick (`sirens_mode.py:729` → `wsng_bytes`);
settings JSON encode + write during splitter drags (`main.py:1616`, `settings.py:267-298`).
Each has a sibling in the tree that already does it right (GLB parse/adopt split,
`inker-recover:` task). Add a perf-lane test that fails when these run on the frame
thread.

**T3. Task-thread writes into UI state.** `journal.write` sets tab marks from the task
closure (`journal.py:427-446`); `jobs_cache.measure_one` mutates `_dir_sizes`
(`jobs_cache.py:244-248`); Clay `reserve_uid` races `new_uid` on an unlocked
`itertools.count` (`clay/document.py:123`, called from `serialize.read_wblk` on a
task). Hand results back through `Done.result` and apply on the frame thread.

**T4. Greyed controls with no or wrong reason.** Progress-card Cancel
(`panes/overlay.py:367`), Plotter layer menu passing `BUSY` for an `active`/`many` gate
(`plotter_menu.py:151-173`), error/cancelled rows in the results tray saying "not ready
yet" (`generation_workspace.py:198-223`), Inker tileset doors saying "Open a drawing
first" when the tab is saving (`inker_mode.py:4319, 4359`). The app's own rule is that a
disabled control explains itself.

**T5. Documented behaviour the code does not have.** Sirens "instrument kind vs
channel kind is a refusal" (nothing checks); Inker Repeat Last Export "no dialog" (PNG
sequences reopen it, slices can never repeat); `_refresh`'s "lazy import" of
`inker_mode` (unconditional on frame 1). Either fix the code or fix the sentence;
never leave both. (The X-ray pick, the stale-buffer `play`, the imgui-nav sentences,
`TILED_VERSION` and "six workspaces" were settled on 2026-09-03.)

**T6. Per-frame recomputation nobody looks at.** Menu bar rebuilds every command spec
and evaluates every `enabled()` each frame including a 200-job scan (`menus.py:180`,
`palette.py:660`); Troupe `sendable_meshes` runs an uncached 400-row `list_jobs` per
frame while the picker header is open (`troupe_settings.py:98`); `Model.bounds()`
min/max over every vertex per frame for the inspector (`viewer_embed.py:211`,
`gltf.py:236-263`); Home rebuilds Resume rows several times a frame
(`landing.py:153-211`); Review stats the reference image three times per frame
(`review_mode.py:1693-1699`); Reference-stage validation runs twice per frame
(`create_brief.py:89`, `settings_2d.py:1836`). Memoise on `cache._generation` or the
menu-open state.

**T7. `main.py` at 6,345 lines and `inker_mode.py` at 4,466.** `main.py` still holds ~900
lines of Review pane drawing (`_review_*`), the Clay and Poser viewports, and ~300 lines
of shortcut data; these are the only surfaces not under their own `guard`. `inker_mode.py`
holds export (~900 lines), keys, palette IO, playback, open, tilesets and journal.
`inker_canvas.py` (3.6k) mixes input, transform, slices, text, overlays and scrollbars.
`sirens_mode.py` (1,250) holds editing, playback, keys and journal. The seams are
listed per mode below.

**T8. Test blind spots.** Zero test references for `studio/_view_cache/_view_drag/
_view_overlay/_view_pick.py`, `sizeguard.py`, `panes/thumbs.py`, `panes/inker_menu.py`,
`packwright_items.py`, `troupe_bridge.py`, `troupe_sheets.py`, and **all six Sirens
panes**. `_layouts_popup` has no test, which is why finding 1 shipped. Several suites
are `inspect.getsource` string assertions rather than behaviour
(`tests/test_generation_workspace.py`). `tests/plotter/_semantics.py::_tileset_facts`
does not compare `tiles`, `wangsets` or presentation fields, which is why the TMX/TMJ
readers can disagree and still pass.

---

## 2. Core shell (`main.py`, layout, dialogs, palette, journal, Home)

### Correctness
- **[Medium]** `_layouts_popup`, `component_gallery.draw()` and `_shortcuts_popup` all
  draw at host scope with no `guard.run` (`main.py:3422, 3427, 3440`). Wrap each.
- **[High] `Layout.save()` can skip persisting rail/sidebar.** `layout.py:427-455`
  `set_share` sets `_workspace_share_saved = True` unconditionally; a drag clamped at
  `SHARE_MIN/MAX` never calls `save()`, the latch stays armed, and the next
  `set_rail()`/`set_sidebar_width()`/"Reset pane sizes" returns early without writing.
  Return whether it persisted rather than latching.
- **[Medium] Ctrl chords leak through the command palette.** `palette_open` is not in
  `modal_open` (`main.py:218`); Ctrl+Enter with the palette open in Create generates.
- **[Medium] The Manual overlay does not own the keyboard.** Only Esc is intercepted
  (`main.py:2777`); with the reference open, `Delete` in Create trashes the selected
  asset unconfirmed and tool letters switch Inker tools.
- **[Medium] Prompt dialog cannot be tabbed.** `dialogs.py:494` calls
  `set_keyboard_focus_here` every frame the field is not active; use the `Confirm._focused`
  one-shot.
- **[Medium]** `journal.write` sets three tab attributes from the task thread with no
  lock on the read side (`journal.py:427-446`). See T3.
- **[Low]** `_MODIFIER_MAP`/`_KEY_MAP` lack GUI/super and `K_APPLICATION`
  (`imgui_backend.py:288-359`).

### UX
- **[High] Settings → Advanced "Sidebar width" is dead.** `set_sidebar_width` animates
  `layout.SIDEBAR_W`, but `_build_ui` always calls `layout_mod.measure(...)`, so the only
  reader (`layout.py:235`) never runs; `Library._width_seed` is read once at
  construction. Remove the control or make it live and say "next launch" for dragged
  workspaces.
- **[High] "Reset pane sizes" does nothing for a split ever dragged.**
  `app_settings.py:657-663` clears the legacy `lay.shares`; `Layout.share` consults the
  bound `Library.share` first, and every drag persists there. Reset must also clear the
  library's per-workspace shares.
- **[Medium]** Delete key is bound in the Create sidebar library but not in Library mode
  (`main.py:2964` vs the `library` branch at ~2826); the shortcuts sheet advertises it.
- **[Medium]** The window title reflects only unsaved *pose* edits (`main.py:2272`); a
  dirty Inker/Clay/Plotter/Sirens document leaves the caption clean. Derive from the
  quit-guard predicate.
- **[Low]** Progress-card Cancel greys with no reason (`overlay.py:367`).

### Structure
- **[High]** Extract `panes/review_*.py`, `panes/clay_viewport.py`,
  `panes/poser_viewport.py` and `shortcuts.py` from `main.py:259-554, 4400-5610`.
- **[Medium]** Panes import `main` (`panes/tour.py:329`, `panes/landing.py:554`) for
  `modal_open` and `_version`. Move them to `dialogs.py` / `warlock/__init__`.
- **[Medium]** `layouts.py:156-157, 212, 263` re-spell `layout.SIDEBAR_WIDTHS`,
  `PANEL_MIN/MAX`; put them in `tokens`.
- **[Medium]** The ~150 function-local `from . import` statements in `main.py` buy
  nothing once `ensure_providers()` imports all six mode modules on frame 1. Accept eager
  loading and say so, or actually gate on `state.inker is not None`.
- **[Low]** `App` reaches `viewer._grab`/`clay_view._grab` (`main.py:2601, ~2683`);
  expose `dragging`. Three copies of the hover/grab routing rule at `main.py:2590-2640`,
  only the Clay one carrying the `tab.saving` press gate.
- **[Low]** Surface sizes off the token scale: `_LABEL_CELL = 132`, `sp(520)/sp(720)`,
  `sp(430)`, `sp(320)`, `sp(210)`, `landing.py:469-470`.

### Performance
- **[Medium]** Menu bar spec rebuild per frame (T6). **[Medium]** Settings flush on the
  frame thread during drags (T2). **[Low]** `ctypes.string_at` copies vertex/index
  buffers per command list per frame (`imgui_backend.py:239, 281`);
  `(c_ubyte * size).from_address` is zero-copy. **[Low]** `guard.enter` allocates an
  `ErrorRecoveryState` per pane per frame (`guard.py:164-188`).

---

## 3. Create, Review, Library, Home

### Correctness
- **[High]** `review_mode.launch` runs `create_sweep` inline on the frame thread
  (`review_mode.py:1455-1493`); its docstring cites a `settings_2d._submit` that no
  longer exists. Submit under a `review-launch` key and keep `form.submitting` true until
  it lands.
- **[Medium]** `_sync_viewer` decodes the reference PNG on the frame thread the moment a
  job finishes (`viewer_embed.load_reference:183-192`). Split like `parse_model`/`_adopt_model`.
- **[Medium]** Results-tray "Open" bypasses `asset_open.open_asset`
  (`generation_workspace.py:200`): `source_job` stays stale on a reference and a mesh
  result shows `input.png` on the Reference stage.
- **[Medium]** `jobs_cache.measure_one` mutates `_dir_sizes` from a task (T3).
- **[Low]** `create_stages._reached_export` imports imgui-bearing `widgets` every frame
  from a module whose docstring says it imports nothing from imgui
  (`create_stages.py:120`). Move `artifacts_for`'s table to a headless module.

### UX
- **[Medium]** Error/cancelled rows in the tray say "not ready yet" and disable Rerun,
  though `rerun_job` needs only `input.png` and the library card offers "Try again" on
  the same rows. Enable Rerun with the failure as caption or drop the rows.
- **[Medium]** A VRAM door refusal is a fading toast while the plan block keeps saying
  "Ready to generate" (`validation.check_vram:385`, `main._collect_tasks:1641-1681`,
  `_generation_plan:1869`). `vram.shortfall_message` is a multi-remedy sentence a toast
  cannot hold. Record the last refusal on `AppState` and draw it in the plan block.
- **[Medium]** `hole_worst` is presented as a ranking in the inspector's remesh line
  (`panes/inspector.py:1214-1224`, "kept the best of 12.3%, 4.5%"), which INVARIANTS
  forbids. Say what was compared and carry the `AUDIT_UNINFORMATIVE` caveat. The caveat
  itself exists three times with three wordings (`widgets.quality_badge:592`,
  `inspector._quality:1189`, `review_mode.mesh_lines:1749`) and `review_mode` imports it
  from `widgets` inside a per-frame function. One headless helper.
- **[Medium]** "Working now" in the tray duplicates the global progress card without its
  Cancel (`generation_workspace._progress:124`, `overlay.py:339-372`).
- **[Medium]** Review's sweep-axis help covers 3 of 14 axes (`review_mode.py:1566`,
  `AXIS_HELP` vs `sweeps.KWARG_AXES`); the three new engine axes got tooltips, the older
  ones have none. Add help and a parity test; `36-review.md` names no axis at all.
- **[Low]** "Keep → Delete the others" raises one undo toast per loser
  (`candidates_panel.keep:110`). **[Low]** `rank.score` shown as an unqualified "score N%"
  (`generation_workspace.py:184`). **[Low]** Toast "Show" from Sirens/Settings only
  selects, no mode change (`asset_open.open_asset:142`).

### Structure / performance
- **[Low]** Private reach-across: `generation_workspace._vary` → `library._copy_settings`,
  `settings_2d` → `generation_workspace._queue_position`, `plan_for` reaching
  `service.sprites` through a pane. **[Low]** `generation_workspace.should_draw` is
  permanently true after the first done job, so the empty state is unreachable and the
  viewer always loses `tray_height`.

### Tests
`test_generation_workspace.py` is five source-string assertions; nothing drives
`_result_card`'s enable table, error-row inclusion, or Open routing. No test asserts a
refused `"submit"` is reported. Nothing pins where the reference PNG decode happens.

---

## 4. First run, doctor, docs, tour, accessibility

- **[Low]** The tour welcome step omits music (`tour/scripts.py:43`); the "click the
  rail" step has no keyboard path and should mention Ctrl+K.
- **[Medium] Clay HUD colours are untokenised** (`panes/clay_hud.py:100-102`,
  `(1,1,1,0.2)` washes): invisible on the light theme, and `test_accessibility.py` only
  measures `tokens.PALETTES`. **[Low]** Toggle knob hard-coded white
  (`widgets.py:2108`, `controls.py:871`), ~1.3:1 against the light `EDGE` track.
  **[Low]** Raw px sizes at `candidates_panel.py:61,63` and `inspector.py:640`.
  **[Low]** Tour card has no Enter/Right/Left binding (`panes/tour.py:412`).
- **[Low]** No i18n layer exists; fine, but record the decision in INVARIANTS so nobody
  half-starts extraction.
- **[Low]** Stale `src/warlock/sirens/`, `src/warlock/plotter/`, `src/warlock/packwright/`
  directories hold only `__pycache__`; the engines live under `studio/`. Delete them and
  fix CLAUDE.md's shorthand.
- **[Low]** Seven files in the diff carry `LF will be replaced by CRLF` warnings
  (`README.md`, `config.py`, `doctor.py`, `generation.py`, `review_mode.py`,
  `test_doctor.py`, `test_progress.py`). Harmless, noisy in every diff.

---

## 5. Inker

### Engine correctness (`studio/inker/`)
- *(Withdrawn 2026-09-03.)* "Mirror symmetry is one pixel off": measured through
  `StrokeState` on a 32-wide canvas, a diameter-4 pixel dab at x=6.0 covers columns
  5–8 and its mirror covers 23–26, which is exact about `default_axis` (column c ↔
  column 31−c). The review's arithmetic assumed a different axis.
- **[Medium]** `animation.layers_for:862` mutates the shared `Layer.opacity` of a linked
  cel; onion skin / preview of another frame corrupts the current stack's opacity.
- **[Medium]** `merge_down` drops per-cel opacity and z (`_doc_layers.py:700`), and bakes
  the lower layer's own opacity/blend then resets it.
- **[Medium]** Interactive Move re-resolves palette indices by nearest colour
  (`_doc_paint.py:625, 652`) instead of permuting them like `offset_layer` does.
- **[Medium]** Bucket fill `refer="layer"` stops at erased pixels because `similar`
  compares dead RGB under alpha 0 (`selection.py:494`).
- **[Low]** `FloatingBuffer.flip` seeds `source` but not `source_offset` (pivot lost);
  `duplicate_layer` drops `cel_opacity/cel_z/cel_notes/note/continuous`; `paste()` with
  no selection lands at (0,0) and the clipboard records no source box.

### App-layer correctness
- **[High]** The context-bar combine mode is overwritten by every press
  (`inker_canvas.py:1447` vs `inker_context.py:215`): set *Add*, drag → replace.
- **[High]** A click with a select tool leaves a 1-pixel selection instead of
  deselecting (`inker_canvas.py:2407-2481`, floor/ceil on a zero-size rect).
- **[High]** Regenerate selection on an empty animated cel captures the placeholder uid
  and is refused on landing with "The layer it was for is gone." (`inker_bridge.py:646`).
- **[High]** `fg_slot` goes stale after palette move/sort/remove/ramp insert
  (`inker_colors.py:313-425`); the next stroke lands in another colour's slot.
- **[High]** Repeat Last Export re-opens the dialog for PNG sequences and can never
  repeat slices (`inker_mode.py:1666-1672, 2284, 1372`).
- **[High]** `_reload_linked` decodes on the frame thread (`inker_mode.py:2817`).
- **[Medium]** Space mid-stroke kills its own pan: `clear_drag` resets `space_held`
  (`inker_state.py:2636`). Keyboard state is not gesture state.
- **[Medium]** `fill_selection`/`stroke_selection`/`shift_selected` skip the busy gate
  (`inker_ops.py:1101-1134`) and can write while `write_ora` walks the stack.
- **[Medium]** Save As `.aseprite` marks clean and drops the journal though the write is
  lossy (`inker_mode.py:1232, 2766`).
- **[Medium]** Four sliders push a step per frame (T1). **[Medium]** Cursor readout and
  shape commit truncate where the press floors (`inker_canvas.py:713, 2450`).
  **[Medium]** `toggle_play` lacks `step_frame`'s open-gesture guard
  (`inker_mode.py:3290`). **[Medium]** `_settle` mutates before a save that may be
  refused (`:1224`). **[Medium]** Sixteen `ACTION_MODIFIERS` are remappable and never
  read (`inker_ops.py:2409-2491`; `action_active` has two callers); modifiers are read
  four different ways in the canvas.
- **[Medium]** Frame-spread exports refuse silently during playback; a second tab's
  export is refused with no message (`inker_mode.py:1723, 1351, 1725`).
- **[Low]** Timeline prompt callbacks and drag-reorder carry stack indices into deferred
  paths (`inker_timeline.py:1732-1833`). Fractional wheel notches leave the 5 % zoom
  lattice. "Queued a mesh…" toasted before the job exists (`:2489`).

### File IO
- **[High]** GIF export ignores per-frame palette overrides: frames are nearest-matched
  against the document table (`inker_mode.py:2215`, `gifout.py`).
- **[High]** `ora._read_tiles` never validates the refs grid against the canvas, and the
  "atomic apply" swaps cels before `materialize` (`ora.py:1814-1918`).
- **[Medium]** Hand-listed `Layer` fields drop `background`/`reference` on open
  (`ora.py:1814-1861`, `asein.py:1515`); use `fields(Layer)`.
- **[Medium]** `gifin`/`sheetin` set `duration_ms` past the clamp; `aseout._frame` then
  dies with a bare `struct.error` on `<H` (`gifin.py:102`, `sheetin.py:355`, `aseout.py:278`).
- **[Medium]** `asein` decodes tilesets with the raw transparent index and cels with the
  clamped one (`asein.py:1713`). **[Medium]** `write_ora`/`write_aseprite` hand-roll a
  `.tmp` without `try/finally` instead of `studio/atomic.py`.
- **[Low]** `asein` warns the reference layer "opens hidden" while reading VISIBLE
  verbatim; skipped header fields have no COMPAT row. `textstamp` allocates unbounded
  (`textstamp.py:130`) with no `pixelguard` check.

### UX parity
- **[High]** Max zoom is 1000 % (`INKER_MAX_ZOOM = 10.0`); a 16 px sprite is 160 px wide
  on 1440p. Aseprite: 6400 %; Plotter already has 32×.
- **[Medium]** Brush cursor is a floating ring at raw mouse coordinates, not the
  footprint, with no mirrored twin (`inker_canvas.py:3602`).
- **[Medium]** Grid, symmetry, layer-edge and pixel-cell lines at fractional screen
  coordinates antialias across two pixels (`:3026, 3030, 3320-3344`).
- **[Medium]** No chord for flip H/V, merge down, duplicate/delete layer, delete frame,
  or Ctrl+Shift+Z redo. `SHIFT_TOOL_KEYS`/`ALT_TOOL_CHORDS` are a hand copy checked
  against `TOOLS`, not `BINDINGS`, so a rebinding leaves the sheet wrong.
- **[Medium]** Picker Hue/Saturation sliders are dead on greys and drift on dark colours
  because HSV is re-derived from 8-bit RGB per frame (`inker_picker.py:186-203`).
- **[Low]** Onion skin on a two-frame clip draws the other frame twice; `,`/`.`/Enter
  bound twice; `_MUTATING_CTRL` duplicates `inker_ops.ready`'s gate.

### Structure
- **[High]** Extract from `inker_mode.py` in payoff order: `inker_export.py`
  (`:1243-2329`), `inker_keys.py` (`:3206-3830`), `inker_palette_io.py`,
  `inker_playback.py`, `inker_open.py`; replace the fifteen-arm `on_task_done` chain
  with a prefix table.
- **[Medium]** Split `inker_canvas.py` into input / transform / slices / overlays; move
  pure helpers (`marquee_rect`, `closes_gesture`, `_onion_index`) into `inker_state`;
  `0x1FFFFFFF` → `gid.GID_MASK`. One `held_chord()`.
- **[Medium]** Three `over`-style commits and three cut-arithmetic copies in
  `_doc_selection.py` (the root of the alpha-lock bug).

### Performance
- **[Medium]** `preview_filter` re-blends, invalidates and re-uploads the box every frame
  the popup is open (`_doc_paint.py:457-500`); a 2048² filter = 16 MB upload per frame.
- **[Medium]** Every brush-down copies the whole layer twice (`_doc_paint.py:927`,
  `brush.py:777, 789`): 32 MiB per click at 2048².
- **[Medium]** A brush-down that draws nothing triggers `_stamp_all` + full recomposite
  (`document.py:1119-1131`); `_ensure_cel_for` swaps in place.
- **[Medium]** `_content_box` rescans four alpha passes per painted frame, keyed on a
  never-pruned `id(doc)` (`inker_canvas.py:2877`). LRUs are Python lists with
  `remove()` per visible cell per frame (`inker_textures.py:239-357`).
- **[Low]** `_below` cache is full-canvas float32 (64 MiB at 2048²); mirror preview does
  up to 4096 `add_rect_filled` per frame; `_adopt` writes the settings file on every tab
  open; `gifout.map_to_palette` is O(palette × pixels).

---

## 6. Clay, Poser, Troupe, shared viewer

### Correctness
- **[Medium]** Esc on a Clay element drag leaves the overlay VBO at the previewed
  positions (`_view_drag.py:625-630, 905`); same in `_restart_keyboard_drag`.
- **[Medium]** `reserve_uid` on the task thread races `new_uid` (T3). **[Medium]** Clay
  crash-recovery decode on the frame thread (`clay_mode.py:1047`).
- **[Medium]** Poser `_blend` interpolates a partial key from identity, which is not
  rest in node space (`pipelines/sheet.py:207-222`); a bone present in one key swings
  through parent-alignment mid-segment.
- **[Medium]** Zero-sum or unnormalised skin weights collapse vertices in the viewer
  (`viewer/programs.py:141`, `gltf.py:462-471` never renormalises).
- **[Medium]** Clip-library key quaternions are never validated (`rigging.py:272`,
  `service/clips.py:723`); a 3-element list raises inside `_blend` and a NaN is written
  to disk. `validate_pose` exists.
- **[Medium]** Troupe atlas decoded and uploaded on the frame thread, then decoded again
  for scoring (`troupe_mode.py:827, 725`).

### UX
- **[Medium]** Poser right-click "Reset the whole pose" bypasses the guard the button
  has (`main.py:3750`).
- **[Medium]** Poser and the shared viewer have no reframe key, no axis views, and a
  click on empty space orbits rather than deselecting; the gizmo is dismissed only with
  an undocumented Esc.
- **[Low]** No pan without a middle button in either viewer. Shade Auto's
  "whole document" branch is unreachable (`clay_ops.py:680`, gated on `has_objects`).
  `insert_key` has no caller.

### Structure / performance
- **[Medium]** The node/rest rotation conversion lives only in `poser_mode.py:816-847`
  and is restated in `blender_worker.py:497`; `service.clips.preview_frames` has zero
  callers while `rebuild_frames` reimplements it.
- **[Medium]** Bevel rewrites every face in a Python loop (`ops_bevel.py:414-428`);
  Select More/Less reimplements the vectorised `_face_corner_mask`
  (`clay_ops.py:828`); `Model.bounds()` per frame (T6); hover motion redraws the MSAA
  scene (`viewer_embed.py:481`); pose mode rebuilds every overlay per frame.
- **[Low]** `dissolve_edges` is O(selected × corners).

### Tests
No Ctrl chord-during-drag beyond Ctrl+J and Ctrl+Z; nothing drives
`Viewer._press/_motion/_release`; `screen_ray` orthographic branch untested outside Clay.
`panes/troupe_bridge.py` and `panes/troupe_sheets.py` still have no test references.

---

## 7. Plotter and Packwright

### Correctness
- **[High] Embedded XML tilesets drop presentation fields, Wang sets and `trans`.**
  `tmx.py:615-669` builds `Tileset` without `presentation_of`, `wangsets` or
  `colour_key`; the JSON twin reads all three. `presentation-112.tmx` reads
  `object_alignment='unspecified'` while the `.tmj` reads `bottomleft`. This falsifies
  several COMPAT rows. Factor `tileset_from_element` out of `read_tsx`.
- **[High] A failed repack leaves the previous atlas on screen and exportable**
  (`packwright_mode.py:568`, `packwright_preview.py:39`, `packwright_io.py:222`).
- **[High] Grid packs relocate trimmed tiles to the cell's top-left**, so the `.tsx` is
  misaligned (`packwright/layout.py:319-357`, `compose.py:53`); `trim` defaults True.
- **[High] Dropping several files onto Packwright keeps only the first**: each drop
  submits under the same key and the refusal is ignored (`main.py:3088`,
  `packwright_mode.py:253`).
- **[High]** Re-adding a changed source PNG is skipped as a duplicate
  (`packwright_mode.py:329`), contradicting `wpack.py:16`.
- **[Medium]** `tileset_usage` misses grouped tile layers and tile objects, so
  `remove_tileset` can renumber painted cells (`_map_tilesets.py:104`).
- **[Medium]** Closing the active tab carries brush/palette/terrain/selection onto the
  neighbour (`plotter_state.py:717`, no `_forget_document_state`).
- **[Medium]** Image-layer texture stamped on `id(pixels)`, which CPython recycles
  (`plotter_canvas.py:1091`). **[Medium]** Ctrl+Shift+Up/Down reorders layers while
  saving (`_MUTATING_CTRL` omits them). **[Medium]** The object clipboard is a live
  reference (`plotter_mode.py:948`). **[Medium]** Terrain re-fit treats an infinite
  map's window edge as the map edge (`terrain.py:166`). **[Medium]** Foreign Wang set
  `class`/properties/representative tile dropped and written back as `tile="-1"`.
  **[Medium]** Deprecated image-layer `x`/`y` folded on XML, dropped on JSON.
  **[Medium]** Absolute filesystem paths persisted as sprite keys in `.wpack`.
- **[Low]** Tile-object `gid` unchecked; `_image_layer_files` can be shadowed by the map
  name; unknown `staggeraxis` silently replaced; MaxRects tries only POT sizes even with
  `power_of_two=False`; `wpack` coerces `"false"` to True.

### UX
- **[Medium]** Tileset removal is modelled and undoable but has no UI. **[Medium]** "Add
  to Packwright" from Library/Troupe toasts "Start or open an atlas first" instead of
  minting one. **[Medium]** No pivot/anchor control or preview. **[Medium]** Export
  encode errors reach the user as "see the log" (`texturepacker.py:145`).
- **[Low]** Every repack resets the view; undo drops the object selection; deleting a
  layer leaves `selected_objects` pointing into it; layer-menu reasons name the wrong
  cause; bridge label always says "JSON (Array)".

### Structure / performance
- **[Low]** `tileset_from_inker` bypasses `land_tileset`; `_shift_layer` duplicates
  `shift_layer`.
- **[Medium]** Save/Export encode on the frame thread (T2). **[Medium]** Per-visible-cell
  Python in the canvas rebuilding an 11-field `Lattice` per call, with no LOD, so zooming
  out is the worst case (`plotter_canvas.py:1037-1071`, `_map_project.py:35`); same on
  the export path (`render.py:302`). **[Low]** Objects drawn without culling; Wang fill is
  per-cell Python; source/packed lists have no `list_clipper` at 1024 rows.

### Tests
`_tileset_facts` blind to `tiles`, `wangsets`, presentation; written XML spellings never
asserted; no Tiled-authored fixture despite the 2026-08-29 manual verification.

---

## 8. Sirens

### Correctness
- **[High]** Tempo/Speed/Rows sliders and name fields push a step per frame (T1).
- **[High] The playhead is a row index into an imaginary single-pattern timeline**
  (`playhead_row:858`): seconds ÷ seconds-per-row at the document tempo, ignoring the
  order list, pattern lengths, `Fxx` and `Bxx`. With two patterns the highlight is off
  screen at second 5 and follow mode pins the grid to the bottom. Have `_render` emit a
  `(sample_offset, order_index, row)` map and bisect it.
- **[Medium]** A dragged release marker can land on step 0 and silence the held note
  (`sirens_envelopes.moved:154`); a loop dragged past the release vanishes from the
  graph but stays in the document.
- **[Medium]** Row-scoped effects (`1xx`, `2xx`, `4xy`, `Axy` last one row) differ
  silently from FamiTracker's persistent ones; the manual gives no persistence rule
  (`synth._apply_row:209`).
- **[Medium]** The "instrument kind vs channel kind" refusal in `document.py:169` is
  documented, not enforced. **[Low]** `read_wsng` collapses duplicate sample keys silently.

### UX (what a FamiTracker/Furnace/OpenMPT user cannot do)
- **[High]** No delete/rename/reorder/retarget in the order list, no loop point other
  than 0, no duplicate pattern; `remove_pattern`, `set_order`, `loop_order=N` all exist
  in `SongDoc`. An intro-then-loop song cannot be expressed from the UI. "+ To order"
  with the caret on a sound effect appends the effect's pattern into the song.
- **[High]** Grid click does not pick the column (`sirens_patterns.py:258`); click on
  `Fxx`, type, get a note.
- **[High]** No mute/solo, no channel pan/rename/kind UI; `update_channel` has no caller
  and `_stem_render` already shows the masked-render pattern.
- **[Medium]** Playback cannot start from the caret, play one pattern
  (`render_pattern` has no caller), or loop (`tab.loop` is stored, `sirens_audio.play`
  plays once).
- **[Medium]** No note preview on entry; no keyboard instrument selection; no
  Home/End, Insert/shift-rows, interpolate, or step increment.
- **[Medium]** Follow mode hides the caret and edits go to an off-screen row.
- **[Medium]** Channels past the pane width are invisible with no scroll or marker
  (`_grid:220`); Left/Right happily move the caret into them.

### Structure / performance
- **[Medium]** Split `sirens_mode.py` into `sirens_edit.py`, `sirens_play.py` and keys,
  keeping the re-export block as the compatibility surface.
- **[Low]** The envelope editor's pure half (`span`, `painted`, `moved`, `toggled`,
  `grabbed`) lives in an imgui-importing pane; move it under `studio/sirens/`.
- **[Medium]** Every accepted render and every journal tick re-encodes every sample to
  WAV on the frame thread (T2). **[Low]** `_cell_text` re-imports `synth` per cell
  (`sirens_patterns.py:133`).

### Tests
No test for undo granularity of a slider or long drag; `playhead_row`
against a multi-entry order; envelope marker invariants; effect persistence; grid click → column; paste
wider than the remaining channels. None of the six Sirens panes has a test.

---

## 9. Suggested order of work

1. The "one gesture, one step" sweep (T1), now that `UndoStack.mark()` folds a run
   exactly. This is the single change that most improves how the editors feel.
2. The frame-thread sweep (T2) with a perf-lane guard, then the task-thread writes (T3).
3. The three doc/code contradictions left in T5: each is a one-line decision.
4. Plotter interop: embedded tileset reader, JSON zero coercion, `_tileset_facts`
   parity, one Tiled-authored fixture. Packwright: failed-pack state, grid+trim, batch
   drop, replace-on-readd.
5. Sirens playback: row map, follow-mode caret, order-list editing,
   mute/solo, play-from-caret. This is what the "Experimental" chip coming off
   implicitly promised.
6. Inker parity: zoom ceiling, footprint cursor, snapped overlay lines, mirror fix,
   missing chords.
7. Extractions (T7) once the behaviour above is pinned, so the moves are pure.
