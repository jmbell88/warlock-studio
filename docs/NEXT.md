Inker Animation Timeline — Aseprite-style frames × layers
  
 Context

 Warlock Studio's Inker mode is a layered raster editor (pure engine under src/warlock/studio/inker/, UI in studio/inker_mode.py + panes) with no notion of time: one document = one static image. The goal is to
 author simple 2D animations frame by frame, Aseprite-style, so Warlock can export game-ready sprite sheets and keep animation sources inside the job system.

 Decisions settled with the user:

 - Purpose: sprite-sheet export (packed PNG + JSON sidecar) for game engines, and animations living in the saved .ora / linked paint.ora. No GIF in v1; no new 3D behaviour in v1 (a linked save's input.png is the
 current frame's flatten, unchanged mechanics).
 - Model: full Aseprite frames × layers grid — Track (layer identity: name/opacity/visible/blend, uid) × Frame (duration ms, uid), sparse cel map dict[(track_uid, frame_uid)] -> Layer; absent key = empty cel; a
 linked cel is one Layer object referenced from several frame slots. Layers stay canvas-sized.
 - V1 scope: timeline pane, playback, onion skinning, per-frame durations, linked cels, tags/loop ranges.
 - File format: .ora with one nested <stack> group per frame (foreign editors see frame groups) + a Warlock-only animation.json zip member (authoritative grid: tracks, durations, tags, link identity). Reader
 falls back to today's flat read when the JSON is absent/bad.
 - Compatibility: Document.anim is None is exactly today's document — existing files, tools, undo, saves bit-identical; timeline UI appears only once a document is animated.

 Design

 Engine model (studio/inker/animation.py, new; document.py)

 - Track, Frame, Tag dataclasses + Animation (tracks bottom-first mirroring LayerStack, frames in timeline order, cels dict, tags, current frame index as view state). Uids share layers.new_uid()'s space.
 Pure/headless like the rest of inker/.
 - Document grows anim: Animation | None = None. The existing LayerStack becomes the materialized view of the current frame: _materialize_frame() rebuilds self.stack from the current frame's cels (absent cel →
 transient transparent placeholder Layer; track props synced onto each Layer, track authoritative). Panes, tools, floating buffer, selection, _below cache, and the compositor all keep working unmodified — they
 see an ordinary LayerStack.
 - set_current_frame(index) mirrors set_active_layer: not undoable (view state), commit_floating() first, materialize, invalidate_all(). Gated on tab.saving at call sites.
 - Cross-frame undo needs two surgical changes: Document.layer_by_uid(uid) (current stack first, then anim.cels values) used by PatchEdit._put; and invalidate(rect, layer_uid=) must not KeyError on a layer
 absent from the current stack — it bumps rev + the frame stamps of every frame carrying that layer and returns (verified: today's code raises there).
 - Linked-cel identity is by Python object (id(layer)), so an edit to a linked cel shows on every frame it occupies for free; every whole-grid iteration (transforms, ORA write, replay snapshot) walks
 anim.unique_cel_layers() — each shared Layer exactly once (the double-transform trap).

 Edits (studio/inker/anim_edits.py, new)

 All uid-addressed dataclasses with cost in __post_init__, ending in a doc._anim_changed(...) hook (re-materialize + invalidate when the current frame is affected):

 - FrameAdd/Remove/Move/DurationEdit, TrackAdd/Remove/Move/PropsEdit, CelAdd/Remove/Link/UnlinkEdit, AnimateEdit.
 - Costs follow the LayerAddEdit lesson: carried Layers report their nbytes; linked duplicates cost ~0; CelUnlinkEdit mints its copy's uid once at op time (the flatten_layers replay rule).
 - First "Add frame" = CompoundEdit([AnimateEdit, FrameAddEdit]) — one Ctrl+Z returns to a plain document. Existing add/remove/move/duplicate_layer/set_layer_props branch onto Track* edits when animated (stack
 index == track index, so inker_layers works verbatim).
 - Empty-cel autovivify: _ensure_active_cel() at the top of every active-layer write path; _commit_patch wraps the pending CelAddEdit + PatchEdit into one CompoundEdit; a no-op write removes the cel again and
 pushes nothing (the "a step that changes nothing is not pushed" invariant).
 - Whole-canvas ops (crop/resize/flip/rotate/scale) via _map_planes/_replay apply to unique_cel_layers() and snapshot the whole grid with an id(layer) -> copy map so link structure survives undo.
 merge_down/flatten_layers are disabled for animated docs in v1 (Document.can_restructure, read by the layers panel).

 Onion skin + frame cache

 - Document.frame_flat(frame_uid) — per-frame flattened RGBA, LRU-capped (FRAME_CACHE_BYTES = 128 MiB), invalidated by per-frame stamps (_frame_stamps, bumped by invalidate/anim edits/invalidate_all via
 anim.frame_uids_of_layer).
 - inker_textures.frame_texture(ctx, tab, frame_uid) under slot inker_tex:{uid}:frame{fuid} (rides the existing prefix sweep). Canvas draws prev/next N frame textures tinted red/green at reduced alpha beneath
 the live composite. Onion settings on InkerState (app-level, like tool settings).

 Playback

 - Per-tab transient state on InkerDoc: playing, play_index, play_accum_ms. Entering playback commits the float once; all mutating input refused while playing (same gating tier as tab.saving).
 - Ticked from the timeline pane's draw() on imgui.get_io().delta_time (the motion.py idiom — there is no per-mode update hook). Pure animation.advance(frames, index, accum, dt_ms, loop) helper, headless-tested.
 Loop range = active tag else whole timeline.
 - While playing the canvas draws frame_texture(play_frame_uid) instead of the live composite — no set_current_frame/invalidate_all per tick. Stop = one set_current_frame(play_index).

 Timeline pane (studio/panes/inker_timeline.py, new)

 - Bottom strip (sp(150)) of the centre column in main.py::_inker_workspace, reserved only when doc.anim is set — otherwise layout is byte-for-byte today's. Entry point: an "Animate / + Frame" button in
 inker_bridge (gated on tab.saving).
 - Rows: transport (play/pause, first/last), tags bar, frame headers (click = go to frame; context: insert / duplicate-linked / duplicate-copied / delete / duration…), then track rows × frame columns (filled
 square = cel, chain glyph = linked, outline = empty; click drives set_active_layer + set_current_frame; context: link with previous, unlink, clear). Cel thumbnails deferred past v1 (dots first).
 - Keyboard in inker_mode.handle_key (only when animated): pygame.K_COMMA/K_PERIOD prev/next frame (match on key constants — pygame.key.name() spelling is SDL-version-dependent), K_RETURN toggles playback
 (transform's modal branch already consumes Enter first). Gated on tab.saving. Timeline's per-tab UI state key prefix joins inker_textures._PER_TAB_KEYS.

 ORA format (studio/inker/ora.py)

 - stack.xml (animated only): one <stack name="frame:0001"> per frame, cels as layers inside (top-first), frames 2+ written visibility="hidden" on both the group and its layer elements — so Krita/GIMP and an old
 Warlock build (whose reader flattens groups but honours layer visibility) display frame 1 correctly. Linked cels share one data/cel{uid}.png, written once (dedup on id(layer)), referenced by src from every
 slot.
 - animation.json (versioned; indices not uids — uids are per-process): frames (durations), tracks (props, bottom-first), cels ({track, frame, data} — shared data path = link), tags. JSON is authoritative;
 stack.xml is the interop projection.
 - Read: JSON-first grid rebuild (decode each distinct data PNG once, share the Layer across slots); any inconsistency logs and falls through to the existing flat read. No JSON → today's path exactly.
 - mergedimage.png/thumbnail = frame 1's flatten (deterministic — the playhead is view state and must not change what a save contains).
 - Accepted risk: an old Warlock build that opens and then saves an animated file rewrites it flat and loses the animation — inherent to forward compat; hidden groups limit display damage only.

 Sprite-sheet export

 - Additive-only pipelines/sheet.py changes: Plan.frame_w/frame_h (default 0 = square frame_size; plan() never sets them, so all existing 3D output is byte-identical) and sidecar(..., animation=None) emitting an
 "animation" key (frames: [{cell_index, duration_ms}], tags) only when given. All existing consumers are key-based readers.
 - New pure studio/inker/sheetout.py: build_inker_plan(doc) constructs Plan/Cell rows directly (never plan(), whose square FRAME_SIZES check would refuse the canvas) — one Cell per frame, frame=i, yaw=0.0,
 frames from doc.frame_flat; reuses measure_trim / the MAX_ATLAS_PX guard. (First deliberate studio/inker/ → pipelines.sheet import; sheet.py is pure stdlib+PIL, so the no-imgui/moderngl/pygame/service rule
 holds.)
 - inker_mode.export_sheet mirrors export_png: gated on tab.saving, commit_floating() first, runs on a task thread via _start(ctx, tab, f"inker-export:{tab.uid}", run), returns {"exported": path} — the existing
 on_task_done branch toasts it unchanged.

 Linked-doc save

 No changes: _save_linked writes input.png first (current frame's flatten) then paint.ora (now carrying the grid). Staleness rule and save_inker_working ORA-magic validation hold verbatim.

 Phases (each independently landable, suite green throughout)

 1. Engine model & materialization — animation.py; Document.anim/ensure_animation/_materialize_frame/set_current_frame/layer_by_uid; invalidate guard; PatchEdit._put switch. Full existing suite must pass
 untouched with anim is None.
 2. Edits & grid ops — anim_edits.py; layer-op branching; can_restructure; autovivify compound; grid-aware _map_planes/_replay/restore_snapshot; Document frame/cel ops.
 3. ORA — frame-group + shared-PNG writer, animation.json, JSON-first tolerant reader. Plain docs byte-compatible.
 4. Timeline UI, onion, caches — frame_flat + stamps + LRU; frame_texture; inker_timeline.py; workspace split; bridge button; comma/period keys; onion drawing.
 5. Playback — InkerDoc play state; delta-time tick; cached-texture draw path; input refusal; Enter toggle.
 6. Sprite-sheet export — additive sheet.py fields; sheetout.py; export_sheet.
 7. Docs & pins — CLAUDE.md Inker invariants (identity-by-object links, unique-cel iteration, frame-switch-is-view-state, playback-reads-caches); manual page (docs/manual/, validated by
 tests/manual/test_docs.py).

 Critical files

 - src/warlock/studio/inker/document.py, layers.py, undo.py, ora.py (+ new animation.py, anim_edits.py, sheetout.py)
 - src/warlock/studio/inker_mode.py, inker_state.py
 - src/warlock/studio/panes/inker_canvas.py, inker_textures.py, inker_bridge.py (+ new inker_timeline.py)
 - src/warlock/main.py (_inker_workspace, ~L1296)
 - src/warlock/pipelines/sheet.py (additive only)

 Verification

 - Headless engine tests (new tests/inker/test_animation.py): materialization preserves active index/props; autovivify is one compound step, undoes fully, no-op leaves nothing; linked-cel edit visible on all its
 frames; unlink uid stable across undo/redo; undo after frame reorder lands on the right cel; crop/rotate transforms each linked cel exactly once; replay-undo restores link structure; advance()
 timing/loop/tags; frame stamps invalidate exactly the frames carrying an edited linked layer; LRU respects its cap; set_current_frame moves no history head (dirty stays honest).
 - tests/inker/test_ora.py: animated round-trip (grid/links/tags/durations); link identity survives reload; corrupt/absent JSON falls back flat; plain-doc output unchanged (regression pin); frames 2+ written
 hidden.
 - tests/inker/test_undo.py: new edit costs (linked dup ≈ 0, CelAddEdit = nbytes), eviction with an animated ReplayEdit.
 - Sheet tests: sidecar with animation block; square sidecar without it byte-identical; non-square frame_w/frame_h arithmetic. Existing tests/test_sheet*.py untouched and green.
 - End-to-end: uv run pytest both with and without the native DLL (WARLOCK_NATIVE=0); manual smoke via /run — animate a doc, draw across 3 frames with a linked background, play with a tag loop, save/reopen the
 .ora, export a sheet, open the linked job's paint.ora path.

 Deliberate v1 trade-offs (flagged for approval)

 - merge_down/flatten disabled on animated documents (per-frame flatten is a follow-up).
 - Timeline cells are dots/chain glyphs, not pixel thumbnails (the layer_thumb stamp pattern extends later).
 - mergedimage.png and a linked save's input.png reflect frame 1 / the current frame respectively.
 - An old Warlock build saving an animated .ora flattens it (forward-compat limit).