140 Improvements for Warlock Studio — UX/UI, Speed, Clarity
  
 Context                                                                                                                                                                               

 Requested: a list of 100+ improvements (cap 250) targeting UX/UI, speed, and clarity. This list was built from three code surveys (studio UI panes, performance hotspots, roadmap/docs/error-message audit) so
 every item is grounded in a real observation with a file reference — not generic advice. Notably, the codebase already solves most first-order UX problems (toasts, confirm dialogs, disabled-with-reason,
 empty-state widget, DPI tokens); these are the second-order gaps that remain.

 Items are grouped by theme and tagged [Speed], [UX], or [Clarity]. Rough priority within each group is top-down.

 ---
 A. Database & data layer — [Speed]

 1. Enable PRAGMA journal_mode=WAL; synchronous=NORMAL on the JobStore connection (db.py:256). Today every commit() is a rollback-journal write plus two fsyncs, and seven methods commit individually — a sweep
 submit is N fsyncs holding the RLock the frame thread queues behind. One line, affects every write path; WAL also stops readers blocking on writers.
 2. Memoize params JSON parsing in list() (db.py:837): 200+ json.loads every 500 ms for rows that almost never change. Key the memo on the raw params string, or narrow the list query to the columns the library
 actually reads (rank, mesh_report).
 3. Reduce COUNT(*) cadence (jobs_cache.py:88 → db.py:355): a full index scan twice a second just to render "showing newest N of M". Refresh on a long cadence or only when len(jobs) == limit; invalidate on
 submit/delete.
 4. Add migration 6 indexes: verdicts(job_id, source, id DESC) and observations(job_id, id DESC) — latest_verdicts/latest_observations (db.py:696, 792) are full scans with a temp b-tree GROUP BY, and
 findings.refresh runs on every finished generation.
 5. Add jobs(status, created_at, id) composite index — serves both next_queued (every dispatch tick) and unverdicted_models without a sort step (db.py:36-37, 529).
 6. Make idx_jobs_sweep a partial index (WHERE sweep_id IS NOT NULL) so list_sweeps' unit tally (db.py:593) becomes an index-only scan instead of a full-table walk on every Review rescan.
 7. Rewrite unverdicted_models' NOT IN subquery as NOT EXISTS with an index on verdicts(source, job_id) (db.py:743-749).
 8. Short-circuit attach_progress on worker.current_job_id (service/jobs.py:485, core.py:134): at most one job runs, yet 200 rows each take the progress lock twice a second.
 9. Replace dataclasses.asdict in ProgressBus.snapshot (progress.py:321) with an explicit dict literal — it's a recursive deep-copy called per frame for the running card.
 10. Batch sweep-unit inserts under one transaction where the directory-before-row ordering allows (service/sweeps.py → create_job per unit); WAL (item 1) removes most of the cost either way.

 B. Frame loop & rendering — [Speed]

 11. Add an idle-frame throttle: the loop renders 60 fps unconditionally (main.py:331-335) even on the static Home chooser. Gate full redraw on: pending input, live toast/progress, camera not settled (epsilon
 compare against goals — camera.py:130 is asymptotic and never exactly settles), strip render in flight, task in flight; otherwise tick at 10–15 fps.
 12. Skip viewer.render when inputs are unchanged (main.py:1992-2016): when camera/model/placement/wireframe/rect are identical, re-show the previous viewport.texture. Cheapest slice of item 11, lands
 independently.
 13. Same skip for the Clay viewport (main.py:1239 → clay_view.draw).
 14. Same skip for the compare pane — it re-renders the second scene at full res every frame whenever comparing (viewer_embed.py:320-331).
 15. Cache normal matrices and hoist constant uniforms in _draw_model (render.py:141-164): today it's a numpy 3×3 inverse per primitive per frame plus re-writes of u_view/u_proj/u_exposure that are constant
 across the pass (already hoisted in the overlay pass — mirror that).
 16. Pre-sort GpuPrimitive.defines at build time so ProgramCache.get (programs.py:459) stops doing tuple(sorted(...)) once per primitive per frame.
 17. Frame-scoped stat() memo in ThumbnailCache.get (textures.py:67): 40 visible cards × 60 fps = ~2400 stats/second on the frame thread. Use the existing self._frame counter.
 18. Cache stats() vertex totals on the immutable model (viewer_embed.py:164-179) instead of walking every primitive whenever the inspector draws.
 19. Cache library visible() + failures() per job-list generation (library.py:32, 166): the full filter predicate runs twice per frame over every job, allocating a lowercase haystack per row when the text filter
 is set.
 20. Hoist the queue-position map out of the card loop (library.py:257): it's O(queued²) — a 60-unit sweep means 60 × 200-row scans per frame.
 21. Move _review_findings' findings.json load below the widgets.header guard (main.py:1661) — one stat per frame that only matters when the header is open.
 22. Iterate only the visible index range in the Inker grid (inker_canvas.py:639-646): a 4096 canvas at grid 8 is ~1024 Python iterations per axis per frame, mostly discarded.
 23. Viewport-cull marching ants: AABB pre-filter per loop plus a per-run visibility mask (ants.py:82-136, inker_canvas.py:703) — the documented optimisation still computes and draws off-screen contour at high
 zoom.
 24. Gate layer_thumb on a per-layer revision or throttle it (inker_textures.py:118): keyed on doc.rev, so every dab re-downsamples the full canvas per layer while the layers panel is open.
 25. Cache Clay element_centre/selection_centre on (mesh id, selection id, transform) (clay_view.py:561-630) — full re-projection of every selected vertex per frame.
 26. Compute Clay world matrices once per frame as {uid: world}: _composite and _element_overlays each call m3.compose for the same objects (clay_view.py:405, 472); also stop synthesising a fresh empty selection
 per unselected object.
 27. Cache world_bounds the same way (clay_view.py:632) — currently O(objects) with an 8-corner matmul each, hit every frame in object mode with a gizmo up.
 28. Avoid rebuilding _below as a full extra canvas composite on every structural change (inker/document.py:217-233), and confirm the shared UndoStack byte budget actually accounts for Clay's whole-Mesh
 snapshots.

 C. Startup & job latency — [Speed]

 29. Move the torch import (CUDA check) off the startup path (runtime.py:89, doctor.py:152): 2–10 s before the first frame for a status dot. _resolve_vram only needs vram.probe(), which degrades gracefully
 without torch.
 30. Move the bpy subprocess probe off the startup path (doctor.py:59-95, 120 s timeout budget) — let it arrive via TaskRunner on frame ~30; runtime.checks is already replaced atomically.
 31. Split doctor checks volatile vs static (doctor.py:30-52, system.py:31): the 5 s health poll re-runs ~60–100 Path.exists() on model weights that cannot change mid-session. Poll only
 port/disk/VRAM/job-object; compute weights once and merge.
 32. Defer the blocking startup storage walk (main.py:254 → measure_storage rglob of every job file before the window opens) — submit it as the first _request_storage instead; the footer shows nothing for one
 frame.
 33. Incremental storage accounting: announce fires a full recursive disk walk per finished job (main.py:601, files.py:379) — a 40-unit sweep finishing means repeated multi-second walks. Add the finished dir's
 size to a running total; full re-measure only after delete/prune.
 34. Parallelize doctor with the loop-thread start in Runtime._start (runtime.py:83-111) — the only real dependency is _resolve_vram before _make_worker, and that needs vram.probe(), not the check list.
 35. Tighten the trellis health poll from 1.0 s to ~0.1 s (trellis.py:231-237): expected ~0.5 s wasted per spawn and per ensure_config restart — which mixed-config sweeps trigger repeatedly.
 36. Tighten port-reclaim polling from 250 ms to ~50 ms (trellis.py:312-316).
 37. Hard-link instead of copy in the remesh-retry staging (queue.py:1178, 1264): four shutil.copyfiles of tens of MB per retry round; os.link makes the keep-a-copy step free on NTFS (verify _apply_scale's
 in-place rewrite doesn't defeat it; fall back to one to_thread wrapping both copies).
 38. Skip the idle worker's per-second executor hop + sqlite query when nothing woke it (queue.py:384-401).

 D. Viewer & asset loading — [Speed] / [Clarity]

 39. [Speed] Memoize glTF image decode by source index (gltf.py:458): ORM packing points metallicRoughness and occlusion at the same image, so a 2048² PNG is decoded into a 16 MB buffer twice or more. This is
 the dominant cost of parse_model — directly shortens "job finished → mesh on screen".
 40. [Speed] De-duplicate GPU textures by decoded buffer (scene.py:44-58, 249): materials are de-duped by id() but images are not — shared maps get duplicate ctx.texture + mipmap builds.
 41. [Speed] Reduce read_rgba's stall (glctx.py:85-96): blocking glReadPixels plus a second full-frame copy just to flip rows; fold the flip into Image.frombuffer + transpose, and move the PNG encode
 (capture.py:23) to the TaskRunner — only the readback needs GL.
 42. [Clarity] Surface skipped textures to the user (gltf.py:441-477): a GLB with external-file URIs or an undecodable map silently renders grey — the likely "why is my model untextured" report. One
 toast/inspector note: "loaded untextured: N textures skipped".

 E. Errors surfaced, not just logged — [Clarity]

 43. Fix the silent job-count failure (jobs_cache.py:90): on error the "Load older" button disappears and history looks complete. Show a warning state instead.
 44. Distinguish picker crash from user cancel (dialogs.py:50, 60): a failed file picker returns None exactly like changing your mind. Toast the failure.
 45. Surface storage-measurement failure (jobs_cache.py:117) rather than an empty footer.
 46. Show ctx.cache.error's text and offer Retry in the library's "Could not read the job list." (library.py:41).
 47. Toast thumbnail-capture failures (app_ctx.py:213) instead of leaving the card image-less with no explanation.
 48. Surface Clay thumbnail / reference-render failures (main.py:1122, 1144).
 49. Wrap the raise Invalid(str(exc)) passthroughs (jobs.py:48, rig.py:48/86/128, sheets.py:98/112/242) so raw library text never reaches the user unedited.
 50. Fix f"reference is {job['status']}" — it leaks a raw status enum into user-facing text; map to a sentence.
 51. Make "file not ready" actionable (three sites in derive.py): say what is still running and that it will appear when done.
 52. Give Failed errors a pointer at the log: "could not export FBX" / "could not bake this pose" say retry-or-report but never where to look — the toast action system (TOAST_ACTIONS["log"]) already exists; use
 it here.
 53. List accepted formats in "that is not a readable image".

 F. Onboarding & doctor — [Clarity]

 54. Give the two fatal doctor checks a remedy: trellis-server.exe and GGUF-weights failures are bare "not found at <path>" (doctor.py _exe_check/_gguf_check) — the only two blockers that guarantee zero output
 are the only checks without a download command or manual pointer, while every non-fatal model row carries its exact hf download line.
 55. Add a submit-time guard on missing weights: create_job validates everything except whether the selected checkpoint is on disk (jobs.py:198); the only protection is a cosmetic " - weights missing" combo
 suffix (main.py:282). The doctor knew at startup — refuse with the download command.
 56. Add a "Diagnostics / Set up models" entry on the Home screen: six tiles and no repair path; a first-run user with nothing downloaded gets a fully-operational-looking form.
 57. Link docs/manual/12-troubleshooting.md from the failing UI — the manual documents all of this well, but the red banner and diagnostics popup never point at it, nor at uv run warlock doctor.
 58. Make the health dot a real control: it's a 16 px invisible button at the far right of the header, and it's the only door to the actionable hf download text. Enlarge the target, add a label or badge when
 checks are failing.
 59. Keep a residual affordance after banner dismissal — once dismissed, the only remaining signal for a fatal condition is the dot's colour.

 G. Manual & docs — [Clarity]

 60. Write the missing manual page for the Home chooser (panes/landing.py) — the first screen every launch, mentioned once in the manual.
 61. Write the app-Settings pane page (panes/app_settings.py) — UI scale, FPS toggle, layout resets, model list.
 62. Write the Profiles page (panes/profiles_panel.py) — a top-level Home tile scattered across 5 files with no owner chapter.
 63. Document landmark-informed rigging: zero manual hits for "landmark"/"pose2d"; 15-extending.md describes the old bbox-fit behaviour; WARLOCK_POSE_FIT is the only env var with no manual mention.
 64. Fix stale 11-configuration.md entries: WARLOCK_GLTFPACK ("not yet vendored" — it is), WARLOCK_REFERENCE_RETRIES (documents default 0; config.py:132 says 2), WARLOCK_MESH_PROFILE (right conclusion, wrong
 reason).
 65. Document the retarget panel as a user feature — it appears only inside architecture chapters.
 66. Fix the stale CLAUDE.md line calling app-Settings "a read-only placeholder" — it now has three working sections; only the model list is read-only.

 H. Notifications & feedback — [UX]

 67. Add a toast history view — a toast that expires while you're in another window is gone forever; a small "recent messages" list (in the diagnostics popup or a bell icon) closes that.
 68. Add success/warning toast levels — only info/error exist, so warnings must be dressed as errors.
 69. Pause toast TTL on hover (expire_toasts is time-only) and show "+N more" when over the 5-visible cap.
 70. Add drop-target visual feedback: there is one drop target (main.py:934) and no hover highlight or "what will happen" overlay — the user learns the result only from a post-drop toast.
 71. Fix the wrong-mode drop message (main.py:954): dropping a non-image in 2D says "Drop an image to start a mesh from it" — but in 2D a drop is a conditioning reference, not a mesh.
 72. Show a placeholder glyph for missing thumbnails (library.py:236) instead of a blank imgui.dummy square.
 73. Extend empty_state to the six lists lacking one: Inker layers, Profiles (no saved profiles), sheet panel, pose panel, Review's run list, manual search-with-no-results — the widget exists and is good
 (widgets.py), it's used in only 4 places.
 74. Add an inker key to overlay.placeholder's dict (overlay.py:222) and upgrade the bare centred string to the richer icon+title+hint empty_state style.

 I. Keyboard & input — [UX]

 75. Add mode-switch shortcuts (Ctrl+1..8) — the mode switch is mouse-only; no shortcuts exist at all outside WORK_MODES (main.py:868).
 76. Make Esc leave Home/Manual/Settings (currently no keys are handled there at all).
 77. Give modals keyboard access: Enter-to-confirm, Esc-to-cancel, and default focus on the Confirm dialog (Prompt focuses its field; Confirm focuses nothing).
 78. Make ConfirmQueue/PromptQueue actually queue (dialogs.py:96, 145): a second question while one is open is silently dropped — _request_quit hand-nests three guards to work around it (main.py:969), and any
 future double-ask loses one with no trace.
 79. Add arrow-key navigation in the library (selection doesn't move from the keyboard).
 80. Add a command palette / quick-open — 8 modes and ~20 panels, all mouse-driven.
 81. Label the shortcuts button (main.py:1812): the keyboard-shortcut list hides behind an unlabelled ? icon next to the 9 px health dot; nothing on screen says F1 opens the Manual.
 82. Right-click context menu on library cards — actions live only behind the ... small-button today.
 83. Drag a library card onto the 3D source slot — no in-app drag-and-drop exists at all.
 84. Drag-reorder Inker layers (buttons only today) and multi-select in the Clay outliner.

 J. Library & browsing — [UX]

 85. More sort options (library.py:131): only newest/best exist — add name, size, duration, kind, and an asc/desc toggle.
 86. Search boxes for the small lists: Clay outliner, Inker layer stack, Profiles, Review's run list, the pose list — only the library and manual have one.
 87. Filter syntax (state.py:167): the free-text filter is substring-only; support tag: / status: / kind: prefixes.
 88. Resolve "filters only apply to the loaded window" (library.py:65): either auto-widen the window while a filter is active, or say "N older jobs not searched — Load older" inline.
 89. A compact/list density toggle for the card view, and date grouping (Today / This week / Older).
 90. Show the full name in a tooltip when a card label truncates (hard cut at 46 chars, library.py:243).
 91. Soft-delete / trash for library deletes — confirmed but permanent today; undo exists only inside Inker/Clay documents.

 K. Forms & panes — [UX] / [Clarity]

 92. Sticky submit footer on the 2D form — 28 controls in four non-collapsible guidance groups mean Generate scrolls away; pin the validate/cost/submit block.
 93. Tooltip coverage for the dense panes: inspector.py has one tooltip in 810 lines; landing, app-settings, profiles, pose, retarget, inker-layers, clay-props/outliner, sheet panel have zero — versus 4–6 in the
 library and generate panes.
 94. Hide or disable-with-reason the one-option Budget combo (settings_3d.py, PROFILES = [("raw", ...)]) — every other control in the codebase already follows the disabled-with-reason idiom.
 95. Add the missing custom_triangles widget (state.py:118 — state exists, no control) or remove the state.
 96. Give 3D "Size (m)" a slider or unit-hinted drag instead of a raw input_float.
 97. Route the ~15 hardcoded pixel sizes through sp() — dialogs.py:117/123/175/177 (150,0) buttons, profiles_panel.py:162/165/187, inker_canvas.py:149/182/185, inker_bridge.py:124/130, settings_2d.py:630/661,
 settings_3d.py:65/179, inker_colors.py:17 SWATCH, plus the magic negative widths (settings_2d.py:299/166) — all wrong at 150–200 % display scale.
 98. Derive the library footer reservation from the bulk bar's real height (library.py:45) instead of a hand-measured constant.
 99. Rebuild the font atlas on UI-scale change — app_settings.py:63's "Text sharpens fully after a restart" is a known papercut.
 100. An in-UI settings surface for common config — ~30 env vars (config.py), none editable in the app; even a read-only "effective configuration" table showing each var, its current value, and whether it came
 from the environment or a default would end the "which env var, set where?" round-trips (an editable config file is the larger follow-on).

 L. Speed odds and ends — [Speed]

 101. Fix ThumbnailCache._supersede's full-key scan (textures.py:105): it walks all ~120 cache keys on every new decode; keep a per-job index instead.
 102. Make the job-list poll adaptive (jobs_cache.py:25, 500 ms fixed): poll slowly (2–5 s) when nothing is queued or running and fast only while a job is live — the terminal-transition toast path already tells
 you when state can change.
 103. Show the window before setup finishes: everything in section C runs serially before the first frame; open the window immediately with a lightweight loading state so a cold start doesn't look hung.
 104. Show progress on the sprite-sheet direction strip while StripRender steps one cell per frame — the viewer.stripping state already exists; render the filled/remaining cell count instead of a partially blank
 strip.

 M. Layout & theming — [UX]

 105. A light theme, or at least a theme hook — tokens.py:82 is explicitly dark-only; the palette is already centralized in tokens, which is most of the work.
 106. Resizable or configurable sidebar width: both sidebars are fixed at 300 design px (layout.py) with MIN_SIZE = (1100, 700) — wasteful centre on 4K, tight on a small laptop. Even a settings-pane width option
 (narrow/default/wide) without live dragging would cover both ends.
 107. Keyboard navigation on the Home tiles (arrows + Enter) — the chooser is the first screen every launch and is mouse-only.

 N. Notifications & doctor, round two — [UX] / [Clarity]

 108. Extend TOAST_ACTIONS beyond "log": give job-transition toasts a "Show" action that selects the job in the library — the action plumbing already exists (widgets.py:648), only "Open log" uses it.
 109. Aggregate sweep-completion toasts: a 60-unit sweep finishing emits per-job terminal toasts into a 5-visible stack; collapse them into one "Sweep finished — N done, M refused" toast with a "Review" action.
 110. Health-dot hover tooltip summarizing failing checks, so a glance doesn't require opening the diagnostics popup.
 111. "Run checks again" button in the diagnostics popup — today the 5 s poller is the only refresh, and after item 31 splits static checks out, a manual re-probe is the way to confirm a fix (e.g. weights just
 downloaded) without restarting.
 112. One-time load-check for matting/pose models: doctor's green rows deliberately under-claim ("weights present … not checked: whether the model loads") — a deferred TaskRunner probe would upgrade them to a
 real answer.
 113. Match dispatch-time refusal messages to check_vram's remedy pattern: Worker._check_resources' free-VRAM and COMMIT_CEILING refusals should name the numbers and an action the way vram.shortfall_message does
 (the codebase's best error).
 114. "Copy error" on the inspector's error box — the Copy-details idiom exists in the diagnostics popup and doctor banner; the per-job error display should have it too.
 115. "Open folder in Explorer" and "Copy job id" on the library card overflow / inspector — the job directory is the real artifact store and there's no in-app door to it.

 O. Browsing & housekeeping, round two — [UX]

 116. Make the prune keep-count configurable — Prune… is hardwired to keep the newest 20 (library.py storage footer); a spinner in the confirm covers everyone.
 117. Distinct message for the honest empty-path case in deliberately-ambiguous errors: "unknown file" / "no such job" are intentionally vague for malformed ids, but the user who selected nothing gets the same
 string — branch before the ambiguity, not after.
 118. HELP_TARGETS coverage audit: the (?) context-help system (manual/targets.py) exists — verify every pane has a target and add the missing ones alongside the new manual pages (items 60–65).
 119. Shrink or shortcut the library's grown window: "Load older" widens LIST_LIMIT by 200 permanently for the session (jobs_cache.py:64) — after item 2 this is cheap, but a "jump back to newest" reset keeps the
 poll small again.

 P. Findings & evidence clarity — [Clarity]

 (These come from TODO.md §0/§7 — the sweep evidence that daily UI hints are built on. They're clarity work in the truest sense: the numbers the UI shows are currently misleading.)

 120. Audit every reader of hole_worst for the inversion: AUC 0.115 against human verdicts — a slab has no holes, so 48/81 rejects scored a perfect 0.0. Anywhere the UI (quality badge, findings hints,
 "worst-hole" lines) implies low-hole = good is actively wrong until the metric is gated on "not a slab".
 121. Run the blind birefnet-vs-auto confirm (8–12 units, labels hidden): the 3-accepts-all-birefnet finding is the single most consequential number in the corpus and it rests on unblinded single-reviewer
 judging; the clean 2×2 alone is p=0.14.
 122. Re-run the render sweep with bg_removal=birefnet as baseline before trusting any other axis — Sweep B measured nothing because every unit sat on the floor (auto), and the re-run also covers PROMPT_TEMPLATE
 v3→v4, so mark old units non-comparable in findings.
 123. Redo the checkpoint ranking on the re-run: refusal rate and mesh quality currently rank checkpoints oppositely, and all of it ran under the broken auto matting, so the hints the 2D pane shows per
 checkpoint stand on nothing.
 124. Surface style-vs-palette conflicts in the composed prompt: art_style=snes injects "vivid saturated colours" against an explicit "black and silver" brief (§7) — the "Prompt actually sent" tree could flag
 fragments that contradict the user's own colour words.

 Q. Roadmap-aligned: the quality judge's UX seams — [UX] / [Clarity]

 (TODO.md §8's Phase-1 deliverables, listed here because each is UI/clarity work with the seams already built.)

 125. Migration: verdicts.stage (reference|blank|model, backfill model) — the same PNG is both deliverable and TRELLIS input, and "good" means opposite things; one label pool across both is useless for each.
 126. db.unlabelled_references() — the existing unverdicted_models excludes errored jobs and sweep units, the two things a labelling pass needs most.
 127. Stage filter inside findings._marginals (not at call sites), so reference verdicts and mesh verdicts stop pooling.
 128. A thumbnail-grid labelling UI in Review with paced texture uploads (the StripRender one-cell-per-step lesson) — the human labelling pass is the explicit blocker for the whole judge.
 129. Retrain via a findings_dirty-style pumped flag, never a direct ctx.submit — the dropped-final-recompute bug shape is already documented; don't reintroduce it.
 130. judge.py as a pure module (stdlib+numpy, None when weights absent, never fails a job) — the vram.py/memlog.py pattern applied again.
 131. Judge staleness made visible: the probe .npz should carry corpus size, label count and schema version, and the UI should say "trained on N labels, D days ago" — silent staleness is a named standing risk.
 132. Sort Review by probe score as the first, advisory-only rung of the authority ladder — no filtering, no refusal, until §11's measurement doc exists.

 R. Mesh pipeline qualification — unlocks UI already built — [UX]

 133. Qualify the gltfpack tiers against a chest, a sword and a rock (UVs, both PBR maps, material assignment kept) — the binary is now vendored and the code paths live; each qualified tier unlocks the Budget
 combo (item 94) and the retarget panel's tier list for real. Corpus must come from the §0 re-run (80 of 83 current meshes are rejects).
 134. Check rig handedness on the first asymmetric re-run mesh: whether trellis reconstructs with the same handedness as the COCO-left → template-+X mapping is unverified and invisible if wrong — one asymmetric
 subject, check which side the .L bones landed, one-line sign fix if flipped, then a measurement doc.
 135. Judge whether landmark placement improves skin weights — the detector tracks the inner edge of bulky armour; landmark rigging ships unjudged on the thing it exists to improve.

 S. Error-message polish, round two — [Clarity]

 136. Sweep admission errors name the unit and the field: admission is all-or-nothing and the refusal "names itself" — verify the message carries unit label + offending field + value so a 60-unit refusal isn't a
 hunt.
 137. Invalid carries field= wherever a control exists: the UI highlights the named field; audit the Invalid raises that omit it (several of the passthroughs in item 49 lose it by construction).
 138. A "what will this cost" line on rig/sheet/pose submissions — the 2D form says "4 references — a few seconds each" and 3D says "roughly two minutes of GPU"; the rig and sheet queue buttons make no estimate
 at all.
 139. Name the stale-artifact consequence at the point of action: the retarget panel already warns before the button about rig artifacts going stale — apply the same before-not-after pattern to pose overwrite
 (deletes the cached bake) and reference re-generation (marks paint.ora stale).
 140. A one-page "effective configuration" doctor section: fold item 100's env-var table into the diagnostics popup too, so "what is this install actually running with" has one answer in-app and one in warlock
 doctor.

 ---
 Verification

 This deliverable is a list, not a code change — verification is that each item traces to a real observation. Spot-check any item by opening the cited file/line. If the user wants execution, the natural next
 step is picking a slice (e.g. A1–A5 + B11–B12 + C29–C32 as a "speed sprint", or E+F as a "clarity sprint") and planning that slice properly, with uv run pytest (427–433 tests) as the regression gate.


Inker & Clay review — 25 improvements
  
 Context                                                                                                                                                 

 A usability/feature review of Warlock Studio's two authoring modes: Inker (the layered raster editor under studio/inker/ + inker_mode.py/panes) and Clay (the CSR-mesh modeler under studio/clay/ +
 clay_mode.py/clay_view.py/panes). Both engines are deliberately pure (no imgui/moderngl/service imports) and both share the uid-addressed studio/undo.py engine — every proposal below respects that split.
 TODO.md currently plans zero Inker or Clay work, so all 25 items are net-new. Items are ordered roughly by usability-payoff-per-effort within each section.

 Inker — 13 improvements

 1. More blend modes. composite.BLEND_MODES is only normal, multiply, screen, overlay, add (inker/composite.py:31). Add darken, lighten, soft-light, hard-light, color-dodge, color-burn, difference — all are pure
 per-pixel formulas that slot into the existing numpy-reference-plus-optional-C-kernel pattern, and each gets an ORA_OPS mapping so Krita/GIMP .ora files stop silently degrading to normal. (composite._MODE_IDS
 stays written out, per the native-kernel invariant.)
 2. Per-tool option memory. Brush, eraser, blur, smudge and the shape tools all share one brush_size/hardness/opacity (panes/inker_tools.py:87-110), so switching eraser→brush inherits the eraser's size. Store a
 settings dict per tool in InkerState; this is the single most-felt daily annoyance in a paint app.
 3. Layer alpha lock and a "sample current layer" eyedropper. Layer carries only pixels/name/opacity/visible/blend/uid (inker/layers.py:36). Alpha lock (paint only where alpha > 0) is one mask in the stroke
 write path and is the cheapest 80% of what people use clipping masks for; Document.eyedrop gains a layer_only= flag beside the existing composite sample.
 4. Whole-layer filters: brightness/contrast, hue/saturation, levels, gaussian blur, sharpen. The document has zero image-wide adjustments (nothing between gradient and undo in document.py). Each is a pure
 pixels -> pixels function in a new inker/filters.py, applied through the existing PatchEdit path (respecting the active selection as the region), with a live-preview popup in inker_bridge. No non-destructive
 machinery needed — patches already make it undoable.
 5. Selection grow / shrink / border. Feather and invert are the only refinements (panes/inker_tools.py:164-183). Grow/shrink are morphological dilate/erode on the 8-bit mask — small additions to selection.py
 with obvious tests — and "select layer alpha" (Ctrl+click thumbnail) falls out of the same work.
 6. Multi-stop gradients. gradient.KINDS is linear/radial with exactly two stops (inker/gradient.py:13). Extend ramp to a stop list and add a minimal stop editor row in the tool options; keep fg→bg as the
 default preset so the current path is unchanged.
 7. Custom canvas size + anchor on resize. New-canvas offers only 512/1024/2048 presets (inker_mode.py:31), and the Resize popup always passes offset=(0,0) even though resize_canvas already supports anchoring
 (document.py:1017). Add width/height fields to the new popup and a 3×3 anchor widget to resize — the engine half already exists.
 8. A proper color picker. The fg/bg control is a bare color_edit4 with no_inputs set (panes/inker_colors.py:18,39) — no hex entry, no HSV control at all. Add hex + HSV fields (imgui has flags for both) and
 palette import/export (.gpl is a trivial text format) for the 24-swatch row.
 9. Canvas rotation and flipped view. PaintView is zoom+pan only (inker_state.py:82). View rotate (R+drag or shortcut steps) and horizontal-flip preview are display transforms in inker_canvas only — pixels
 untouched — and flipping the view is the classic way to spot drawing errors.
 10. Brush stabilisation + a size-by-speed taper. No tablet pressure exists anywhere (grep confirms), and true pressure needs a windowing-layer change — but a stroke smoother (average the last N mouse points)
 and velocity-based width/opacity taper live entirely in StrokeState and fake 70% of the benefit for mouse users too.
 11. Snapping to the grid and symmetry axis placement. The grid is display-only (panes/inker_tools.py:189-197) and the symmetry guides are fixed at canvas centre (inker_canvas.py:649). Snap shape/marquee
 endpoints to grid intersections when the grid is on; let the user drag the symmetry axis. Both are coordinate arithmetic in the canvas pane.
 12. Radial/mandala symmetry. brush.SYMMETRY is none/x/y/xy (inker/brush.py:42). N-way rotational symmetry is a rotation of stamp positions around the (now movable, per item 11) axis point — a natural fit for
 the texture/tile workflows the 2D pane feeds.
 13. Crash-safe autosave. Dirty state is only a tab warning (inker_canvas.py:94); a crash loses the session. Periodically write paint.autosave.ora beside the target through the existing TaskRunner save path
 (gated on saving, deleted on clean save/close), and offer recovery on next open. write_ora's already-documented mid-write hazards make this the mode where a crash costs the most.

 Clay — 12 improvements

 14. Surface manifold/hole diagnostics in the UI. adjacency.check_manifold/ManifoldReport exist and nothing shows them (clay/adjacency.py:307). A row in clay_bridge beside the triangle count ("2 holes · 3
 non-manifold edges", click → select the offending elements) directly serves the export-to-pipeline flow, where meshreport will complain later anyway — better at authoring time. Near-zero engine work.
 15. Material palette management. clay_props._material can pick and tweak a slot but there is no Add/Remove/Rename anywhere (panes/clay_props.py:201) — a new empty document literally says "the palette is empty"
 with no fix. Add the three buttons via MaterialEdit-style undoable edits; replacement-not-mutation is already the palette rule.
 16. Shade smooth/flat as an op. Mesh.smooth is per-face and drives rendering, but no UI sets it — flags come only from import inference and generators. Register "Shade Smooth"/"Shade Flat" (plus an
 auto-by-angle variant reusing glbimport's FLAT_COSINE idea) in clay_ops.py; it's a per-face bool write through set_mesh.
 17. Axis views and an orthographic toggle. ClayView is one yaw/pitch/distance (clay_state.py:49) — no front/side/top, no ortho. Numpad-style 1/3/7 presets (on available keys, since 1-4 are element modes — e.g.
 Ctrl+1/3/7) and an ortho projection flag in the camera. Blockout modeling without ortho side views is genuinely hard.
 18. Axis constraints and numeric entry during transforms. clay_view._about always pivots on selection centre and a gizmo drag has no X/Y/Z lock or typed value. Add axis keys during a drag plus a small HUD
 showing the live delta, and typed-value commit. The single-commit-on-release model (set_transform(was=…)) already fits.
 19. Vertex/edge snapping during element drags. Snap is grid/angle only and applies on the gizmo path but not _commit_element_drag (clay/ops.py:76-125). First apply the existing grid snap to element drags
 (consistency bug as much as feature), then add snap-to-vertex using pick.nearest_vertex, which already exists.
 20. Proportional editing (soft falloff) for element drags. _ElementDrag moves affected_verts at full weight (clay_view.py:73). A radius + smooth falloff weighting is a small change to the drag preview/commit
 math and transforms Clay from blockout-only to organic-adjustment-capable — the highest feature-set-per-line item in this list.
 21. Bridge edge loops + extrude for edges/vertices. Extrude/inset are in_mode("face")-gated; there's no bridge. Edge extrude (rip an edge chain into a quad strip) and bridge (connect two loops) are the two ops
 whose absence blocks common kitbash workflows; topo.py's CSR surgery primitives (splice_corners, region_boundary_corners) are the right substrate.
 22. More primitives: icosphere, capsule, grid-plane. GENERATORS has box/plane/cylinder/cone/uv_sphere/torus (clay/primitives.py:313); plane is a single quad. Icosphere (even triangle distribution for sculpt-ish
 edits), capsule (collision proxies for the game-asset audience), and a subdivided grid plane (terrain patches) — each is one builder + defaults dict, exactly how the registry was designed to grow.
 23. Bounding-box dimensions readout + camera persisted in .wblk. Properties shows TRS only — no world-space W×D×H, in an app whose whole pipeline cares about size_m. Add a read-only dimensions row to
 clay_props. And serialize.scene.json carries no view, so every reopen loses the camera; persist yaw/pitch/distance/target per document (additive JSON key, version-safe since read_wblk validates).
 24. Outliner ergonomics: drag-reorder, isolate, duplicate/delete in context. The outliner is a flat list with eye/rename/delete only (panes/clay_outliner.py). Drag to reorder (display order is meaningful in
 exports), "isolate" (solo visibility, one click vs N eye toggles), and duplicate from the row's context menu — all UI-layer, no engine change.
 25. UV support for authored geometry — generator UVs + a box/planar unwrap op. Mesh.uv exists but only imports populate it (glbimport.py:165); every primitive builds uv=None, so a Clay-built asset exports with
 no UVs and can never be textured downstream. Give each primitive canonical UVs in its builder and add a "Box Unwrap" op (project by dominant normal axis) — deliberately not full LSCM unwrapping, just enough
 that a built asset is texturable after export.

 Verification

 This is a review deliverable, not an implementation. If any subset is picked up for implementation: each engine-level item lands with headless tests in tests/inker/ / tests/clay/ (both engines are pure by
 invariant, so every pixel/geometry rule is assertable without a GPU), UI items follow the existing pane test patterns (tests/test_inker_mode.py, tests/test_clay_mode.py), and blend-mode / kernel work keeps the
 numpy reference and bit-parity bar per the native-kernel invariant.
