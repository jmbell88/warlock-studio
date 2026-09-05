# Subsystems — the slice table

One row per scope key. Paste a row whole into the explorer brief. Paths are under
`src/warlock/` unless they start with `tests/`, `docs/` or `scripts/`. *INVARIANTS* gives
the line of each bold lead-in in `docs/INVARIANTS.md` at 2026-09-05; if the line has
moved, grep the quoted words. *Gates* are tests that already refuse a class of defect in
this slice; an explorer checks them before reporting.

Mode keys (`warlock.studio.modes.KEYS`): home library create inker clay poser troupe
plotter packwright muse sirens review settings. `home`, `library`, `review`, `settings`
→ `shell`.

## shell

- **Source:** `studio/main.py`, `runtime.py`, `app_ctx.py`, `state.py`, `tasks.py`,
  `layout.py`, `layouts.py`, `layout_edit.py`, `layout_skeleton.py`, `dialogs.py`,
  `palette.py`, `menus.py`, `rail.py`, `status_bar.py`, `toolbar.py`, `shortcuts.py`,
  `journal.py`, `recents.py`, `docmodes.py`, `focus.py`, `dpi.py`, `fonts.py`, `theme.py`,
  `tokens.py`, `motion.py`, `guard.py`, `probe.py`, `splash.py`, `review_mode.py`,
  `review_panes.py`, `jobs_cache.py`, `asset_open.py`, `artifacts.py`, `imgui_backend.py`;
  panes `landing.py`, `library.py`, `library_full.py`, `inspector.py`, `overlay.py`,
  `app_settings.py`, `first_run.py`, `model_gate.py`, `palette.py`, `thumbs.py`,
  `candidates_panel.py`, `stamps.py`, `tour.py`.
- **INVARIANTS:** 11, 13 (three threads), 21, 23 (absent models), 43 (refusal names its
  control), 151–169 (shortcut arms, ConfirmQueue, toast ladder, filter box, trash, palette,
  motion, thirteen modes, menu/status bar, Home), 203 (layouts), 320–324, 332 (GL, fonts,
  untrusted files, startup dialog), 334–346 (allocation checks, task prefixes, broad
  except, ceilings, caches), 350–352 (screenshots, startup phases), 400–404
  (accessibility, boundaries, imgui ids), 410–422 (file picker, i18n, Tab, TEXTINPUT,
  columns, DPI), 452–458 (rail rungs, drag-drop, probe, exercise driver), 485 (panel
  section), 489–503 (journal, crash recovery).
- **Tests:** `tests/test_studio_smoke.py`, `tests/test_editor_shell.py`,
  `tests/test_journal.py`, `tests/test_layouts.py`, `tests/test_panes_*.py`,
  `tests/test_landing_*.py`, `tests/test_library_*.py`, `tests/test_inspector_*.py`,
  `tests/test_review_mode.py`, `tests/test_settings_*.py`, `tests/test_notifications.py`,
  `tests/test_dialogs_prompt.py`, `tests/test_palette.py`, `tests/ui/`, `tests/app/`.
- **Manual:** 03, 04, 20, 21, 36, 37, 38, 41.
- **Gates:** `tests/test_undo_gesture_doors.py`, `tests/test_frame_thread_doors.py`,
  `tests/test_task_thread_writes.py`, `tests/test_pane_guard.py` (imgui id collisions,
  raising panes), `tests/test_accessibility.py`, `tests/test_ux_todo_fixes.py` (toast
  levels), `tests/test_findings_themes.py`, `tests/test_findings_blind_spots.py`.

## create

- **Source:** `studio/create_brief.py`, `create_stages.py`, `create_assets.py`,
  `generation_workspace.py`, `matte_preview.py`, `quality.py`, `candidates.py`,
  `studio/viewer/`, `studio/viewer_embed.py`, `_view_*.py`; panes `settings_2d.py`,
  `settings_3d.py`, `texture_panel.py`, `remesh_panel.py`, `retarget_panel.py`,
  `sprite_panel.py`, `sheet_panel.py`, `stage_rig.py`, `pose_panel.py`; `guidance.py`,
  `judge.py`, `generation.py`, `sweep.py`, `provenance.py`.
- **INVARIANTS:** 49 (`hole_worst`), 98–100 (LoRA, conditioning axes), 108 (chunked
  prompt), 136–138 (glTF loader ceilings), 146–149 (taxonomy, quality tier), 318
  (painted reference), 354 (rendering parity), 384–388 (derivation, sweep, candidates),
  398 (GLB loader).
- **Tests:** `tests/test_create_*.py`, `tests/test_generation_*.py`,
  `tests/test_viewer_*.py`, `tests/test_gltf_loader.py`, `tests/test_candidates.py`,
  `tests/test_judge*.py`, `tests/test_matte_*.py`, `tests/test_settings_*.py`,
  `tests/test_prompt_*.py`, `tests/test_conditioning_*.py`.
- **Manual:** 02, 12, 22, 23, 24.
- **Gates:** `tests/test_create_stages.py` (no control in both bar and column),
  `tests/test_resource_ceilings.py`, `tests/test_field_error_wiring.py`.

## inker

- **Source:** `studio/inker/` (and `inker/flourish/`), `studio/inker_*.py`,
  `inker_flourish.py`, `pixelguard.py`, `ninepatch.py`, `anchors.py`, `ants.py`; panes
  `inker_*.py`; `pipelines/sheet.py`, `pixel.py`, `pixelize.py`, `pixelsheet.py`.
- **INVARIANTS:** 53 (zoom), 187 (selection is view state), 193 (asein ledger), 209–261
  (Inker: context bar, layout, pan, quarter turns, timeline, undo, animation, colour modes,
  indexed palettes, links, grayscale, GIF, frame cache, playback, clip export, the 24
  divergences), 288–294 (Aseprite parity record), 424 (layer groups), 466–468 (Flourish),
  550.
- **Tests:** `tests/inker/` (149 files), `tests/test_inker_*.py`, `tests/test_pixel*.py`,
  `tests/test_sheet*.py`, `tests/test_undo_budget.py`.
- **Manual:** 05, 06, 15, 28, 29.
- **Gates:** headless import pin (grep `tests/inker/` and `tests/test_inker_*.py` for
  `imgui`), `tests/test_undo_gesture_doors.py`, `tests/test_inker_busy_guards.py`,
  `tests/test_inker_export_refusals.py`, `docs/COMPAT.md` Aseprite ledger (not executable).

## clay

- **Source:** `studio/clay/`, `studio/clay_*.py`, `clay_view.py`, `clay_viewport.py`,
  `clay_hints.py`; panes `clay_*.py`; `glbio.py`, `studio/viewer/gltf.py`.
- **INVARIANTS:** 136–138 (loader ceilings, `MAX_TOTAL_BYTES`), 296–308 (CSR mesh, one
  conversion out, uid undo, freezing, two merges, drag delta, boundary ops), 398.
- **Tests:** `tests/clay/`, `tests/test_clay_*.py`, `tests/test_mesh_import.py`,
  `tests/test_gltf_loader.py`.
- **Manual:** 07, 30.
- **Gates:** headless import pin (`tests/test_clay_ops.py`), `tests/test_clay_view.py`
  (outward imports), `tests/test_resource_ceilings.py`.

## poser

- **Source:** `studio/poser_mode.py`, `poser_viewport.py`, `skeletons.py`,
  `_viewer_pose.py`; panes `poser_*.py`, `pose_panel.py`, `retarget_panel.py`;
  `rigging.py`, `poselib.py`, `clips.py`, `service/rig.py`, `service/poses.py`,
  `service/clips.py`, `pipelines/blender_worker.py`, `jointfit.py`, `pose2d.py`.
- **INVARIANTS:** 116 (Blender out of process), 124–134 (weld before heat, deformation
  battery, joint sources, pose contract, poses are files, validated at read), 140–142
  (sheet grid on host, importer invents objects), 436–438 (rotation frames, measured
  joints), 515 (skeleton is JSON).
- **Tests:** `tests/test_poser_*.py`, `tests/test_rig*.py`, `tests/test_rigging.py`,
  `tests/test_inspector_rig.py`, `tests/test_panes_rig_bone_count.py`.
- **Manual:** 08, 25, 26.
- **Gates:** `tests/test_poser_imports.py` (only `blender_worker` imports `bpy`),
  `tests/test_poser_panes_smoke.py`.

## troupe

- **Source:** `studio/troupe/`, `studio/troupe_mode.py`, `troupe_state.py`; panes
  `troupe_*.py`; `_q_troupe.py`, `_q_sprite.py`, `service/troupe.py`, `service/sprites.py`,
  `service/sheets.py`, `pipelines/charsheet.py`, `spritesynth.py`, `sheet.py`.
- **INVARIANTS:** 390–394 (sprite draft, cell order, layout), 426–448 (Troupe programme,
  frame tables, clips, sidecar, pixeliser, joints, atlas size, four jobs and a gate, T-pose
  guide, character sheet, no document), 460–464 (tag names, corrections, three-way
  re-render merge), 470 (scores rank, never gate).
- **Tests:** `tests/troupe/`, `tests/test_sprite_*.py`, `tests/test_effect_sprites.py`.
- **Manual:** 11, 27, 33.
- **Gates:** headless import pin (`tests/troupe/`), `tests/test_effect_sprites.py`
  (outward imports).

## plotter

- **Source:** `studio/plotter/`, `studio/tilegrid/`, `studio/plotter_*.py`; panes
  `plotter_*.py`; `_q_tileset.py`, `_q_tilesheet.py`, `service/tilesheets.py`,
  `pipelines/tileatlas.py`, `tilemask.py`, `tilesheet.py`.
- **INVARIANTS:** 78–80 (grid pack never trims, one tileset per syntax), 171–173
  (Plotter is a mode with a pure engine, AI tile sheet is one job), 191 (`tilegrid` shared
  leaf), 195–201 (tilemap cel materialisation, `tiles.json`, view-state toggles,
  reopenable exports), 406–408 (new map asked for, tile size vs projection).
- **Tests:** `tests/plotter/` (53), `tests/tilegrid/`, `tests/test_plotter_*.py`,
  `tests/test_tileset_*.py`, `tests/test_tilesheet_*.py`.
- **Manual:** 09, 13, 31.
- **Gates:** `tests/plotter/test_compat_matrix.py` (parses `docs/COMPAT.md` Tiled rows
  as data), headless import pins (`tests/plotter/`, `tests/tilegrid/`).

## packwright

- **Source:** `studio/packwright/`, `studio/packwright_*.py`; panes `packwright_*.py`;
  `studio/atomic.py`.
- **INVARIANTS:** 102, 122 (staged writes), 189 (deterministic packer), 201 (reopenable
  exports).
- **Tests:** `tests/packwright/`, `tests/test_packwright_*.py`,
  `tests/test_atomic_writes.py`.
- **Manual:** 10, 32.
- **Gates:** headless import pin (`tests/packwright/`), `tests/test_undo_gesture_doors.py`.

## muse

- **Source:** `studio/muse/`, `studio/muse_*.py`; panes `muse_*.py`; `_q_music.py`,
  `service/_jobs_music.py`, `pipelines/music_worker.py`, `music_client.py`,
  `separation_worker.py`, `audioout.py`, `pipelines/acestep/`.
- **INVARIANTS:** 56–76 (Muse is a job-row mode, no document, the reverse Sirens bridge,
  `sirens_audio` gains volume not offset, loop rotation, headless with scipy banned,
  crossfade vs `smpl`, crossfade declines, separation is a one-shot child, derived
  formats, Demucs digest).
- **Tests:** `tests/muse/`, `tests/test_muse_*.py`, `tests/test_music_*.py`,
  `tests/test_separation.py`.
- **Manual:** 14, 16, 35.
- **Gates:** headless import pin and scipy ban (`tests/muse/`),
  `tests/test_muse_panes_smoke.py`.

## sirens

- **Source:** `studio/sirens/`, `studio/sirens_*.py` (`sirens_audio.py` is the only
  `pygame.mixer` toucher); panes `sirens_*.py`.
- **INVARIANTS:** 55 (playhead is a bisect), 60–62 (one bridge door), 521–548 (headless
  engine, per-tick parameters, equal temperament, numpy patterns, uid order list, `.wsng`
  is the composition, column-keyed keys, render-then-play, device isolation,
  `render_dirty`, audition task key, pure export, panes smoke).
- **Tests:** `tests/sirens/`, `tests/test_sirens_*.py`, `tests/test_findings_sirens.py`.
- **Manual:** 34.
- **Gates:** headless import pin and scipy ban (`tests/sirens/`),
  `tests/test_sirens_panes_smoke.py`, `tests/test_findings_sirens.py` (`_MOVED` tables).

## service

- **Source:** `service/` (all), `queue.py`, `db.py`, `vectors.py`, `vram.py`, `leases.py`,
  `memlog.py`, `_q_*.py`, `followups.py`, `asset_workflows.py`, `publish.py`,
  `migrate.py`, `hashes.py`, `progress.py`, `instance.py`, `errors.py`, `models.py`,
  `config.py`.
- **INVARIANTS:** 15, 17 (service is the only business logic; one sqlite connection),
  31–35 (VRAM modes, coexist, last reference), 41 (admission at the door), 47
  (persistent matte), 51 (`config.effective`), 92–96 (`source.glb`/`model.glb`, TRELLIS
  only, remesh), 102–106 (staged writes, grounding), 112–114 (two mesh measurements,
  gltfpack tiers), 120–122 (cancel token, served names), 144 (derived values), 177–185
  (job kind sweep, `source_job`, follow-ups, VRAM halves agree, `t2i_sample` window),
  314–316 (`clean_jobs`, built asset), 348 (job store recovery), 356–380 (verdicts,
  observations, terminal status, dispatch loop, backup, findings, judge, labelling,
  blinding, tier qualification), 396 (a new job kind is four edits).
- **Tests:** `tests/service/`, `tests/test_service*.py`, `tests/test_queue.py`,
  `tests/test_db_*.py`, `tests/test_jobs_*.py`, `tests/test_job_durability.py`,
  `tests/test_vram*.py`, `tests/test_api.py`, `tests/test_failure_paths.py`,
  `tests/test_atomic_writes.py`, `tests/test_authored_sources.py`.
- **Manual:** 36, 43, 44.
- **Gates:** `tests/test_field_error_wiring.py` (refusals carry a `field`), the stage-keyed
  sweep (grep tests for `_q_` and `STAGE`), `tests/test_atomic_writes.py`,
  `tests/test_resource_ceilings.py`.

## pipelines

- **Source:** `pipelines/` (all; `_workerio.py`, `t2i_client.py`, `text2image_worker.py`,
  `matting_worker.py`, `blender_worker.py`, `music_worker.py`, `separation_worker.py`,
  `lora_train_worker.py`, `pack_worker.py`, `fetch_worker.py`, `recipe_worker.py`,
  `loadprobe.py`, `trellis.py`), `winjob.py`, `doctor.py`, `packs.py`, `fetch.py`,
  `native.py`, `tiercheck.py`, `native/*.c`, `installer/`, `scripts/`.
- **INVARIANTS:** 19, 25–29 (offline; the two user-initiated exceptions; fetch is planned
  before performed), 45 (doctor probe in a child), 86–90 (image pipeline out of process,
  stdin reader, winjob), 110 (native kernel has a reference), 116 (Blender), 310–312 (one
  home, the move), 505–511 (checkout-shaped runtime, verified downloads, one transaction,
  no downloaded Python), 552–554 (unsigned installer, 3.13 floor).
- **Tests:** `tests/test_winjob*.py`, `tests/test_offline.py`, `tests/test_doctor.py`,
  `tests/test_fetch*.py`, `tests/test_packs.py`, `tests/test_pack_worker.py`,
  `tests/test_t2i_*.py`, `tests/test_matting*.py`, `tests/test_workerio.py`,
  `tests/test_native_*.py`, `tests/test_installer.py`, `tests/test_runtime_dependencies.py`,
  `tests/test_lora*.py`.
- **Manual:** 01, 39, 40, 41, 42, 44.
- **Gates:** `tests/test_offline.py` (`HF_HUB_OFFLINE`), the winjob scan (grep tests for
  `kill-on-close`), `tests/test_changelog.py`, `scripts/preflight.py` (version lockstep),
  native parity tests (`tests/test_native_*.py`, `tests/test_bvh_native.py`).

## docs

- **Source:** `README.md`, `INSTALL.md`, `CONTRIBUTING.md`, `SECURITY.md`,
  `THIRD-PARTY-NOTICES.md`, `CHANGELOG.md`, `CLAUDE.md`, `TODO.md`, `docs/INVARIANTS.md`,
  `docs/COMPAT.md`, `docs/MODELS.md`, `docs/measurements/`, `docs/manual/*.md`,
  `studio/manual/` (the loader), `changelog.py`.
- **INVARIANTS:** 82–84 (chapters 01–19 reserved; number decides order and part), 261
  (the divergence numbering is a citable API), 290 (non-goals are decisions).
- **Tests:** `tests/manual/`, `tests/test_changelog*.py`, `tests/test_external_doc_links.py`,
  `tests/test_ux_todo_fixes.py`, `tests/test_findings_followups.py`.
- **Manual:** all.
- **Gates:** see `docs-checkup.md`; it lists which rows are already gated.

## tour

- **Source:** `studio/tour/`, `studio/_view_overlay.py` (the drawing side), panes `tour.py`.
- **INVARIANTS:** 450 (points and waits; never acts for the reader).
- **Tests:** `tests/tour/`.
- **Manual:** 01.
- **Gates:** `tests/tour/test_tour_imports.py` (no outward imports),
  `tests/tour/test_tour_conditions.py` (every wait is a named condition).
