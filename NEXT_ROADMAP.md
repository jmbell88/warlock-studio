# NEXT_ROADMAP — the LIST.md items, phased

Reconciled 2026-08-08 from `docs/LIST.md` (165 items: the 140-item UX/UI/speed/clarity
survey plus the 25-item Inker & Clay review). LIST.md remains the survey and the source
of each item's file/line citation — item numbers below are LIST.md's, so any entry here
can be traced back to its observation. **`TODO.md` overrides this file wherever the two
disagree**, the same clause LIST.md carries; the gated items below deliberately point at
TODO.md's sections rather than restating their specs.

Ordering is **dependency + payoff**: Phase 1 makes the app fast and the base honest,
Phase 2 makes failures diagnosable, Phase 3 polishes interaction on that base, Phase 4
adds net-new capability. Within each phase items are listed in recommended order.
Phase 5 (the texture generator) is additive beyond LIST.md — designed 2026-08-08, no
survey item numbers — and sits last because it depends on Phase 4's Clay25 and on
TODO §11's bake op.

## Excluded — already done (verified against TODO.md, 2026-08-08)

- **Q125–Q132** (the quality-judge UX: migration 7 `verdicts.stage`,
  `db.unlabelled_references`, the findings stage filter, `warlock/judge.py`, the
  labelling grid with paced uploads, the pumped-flag retrain, probe staleness metadata,
  Review sort-by-score) — all shipped, code-complete per TODO §7. What remains of that
  thread is a human labelling session, listed under Phase 4's gated work.
- **G66** (stale CLAUDE.md "read-only placeholder" line about app-Settings) — already
  corrected; CLAUDE.md now documents the downloader.

Every other item — 156 of them — appears exactly once below: 150 as phased code work,
6 as gated (GPU/network/human) work.

---

## Phase 1 — Foundations: data layer, startup, frame loop

*All [Speed]. Everything else sits on these: the DB lock the frame thread queues
behind, the serial startup the window waits on, and the unconditional 60 fps redraw.
Low-risk, measurable, and felt daily — land these before adding any new surface area.*

**Implementation order:** A1 → A2/A3 → A4–A7 → C29–C32 → B11–B14 → D39 → remainder by
payoff. A1 (WAL) first because it cheapens every other write-path item, including
Phase 3's features. Any new index or schema change is an append-only migration —
confirm `len(MIGRATIONS)` before writing the entry (TODO §7's warning: numbers have
collided once already).

### A. Database & data layer — `db.py`, `jobs_cache.py`, `progress.py`, `service/sweeps.py`

1. **A1** — `PRAGMA journal_mode=WAL; synchronous=NORMAL` on the JobStore connection
   (`db.py:256`). One line; removes two fsyncs per commit and stops readers blocking on
   writers. Everything below is cheaper after it.
2. **A2** — Memoize params-JSON parsing in `list()` (`db.py:837`), keyed on the raw
   params string (or narrow the query to the columns the library reads).
3. **A3** — Reduce `COUNT(*)` cadence (`jobs_cache.py:88` → `db.py:355`): long cadence
   or only when `len(jobs) == limit`; invalidate on submit/delete.
4. **A4** — Indexes `verdicts(job_id, source, id DESC)` and
   `observations(job_id, id DESC)` for `latest_verdicts`/`latest_observations`
   (`db.py:696, 792`).
5. **A5** — Composite index `jobs(status, created_at, id)` serving `next_queued` and
   `unverdicted_models` (`db.py:36-37, 529`).
6. **A6** — Make `idx_jobs_sweep` partial (`WHERE sweep_id IS NOT NULL`) so
   `list_sweeps`' tally (`db.py:593`) is index-only.
7. **A7** — Rewrite `unverdicted_models`' `NOT IN` as `NOT EXISTS` with an index on
   `verdicts(source, job_id)` (`db.py:743-749`).
8. **A8** — Short-circuit `attach_progress` on `worker.current_job_id`
   (`service/jobs.py:485`, `core.py:134`).
9. **A9** — Replace `dataclasses.asdict` in `ProgressBus.snapshot` (`progress.py:321`)
   with an explicit dict literal.
10. **A10** — Batch sweep-unit inserts under one transaction where the
    directory-before-row ordering allows (`service/sweeps.py`).

### C. Startup & job latency — `runtime.py`, `doctor.py`, `main.py`, `trellis.py`, `queue.py`

11. **C29** — Move the torch import (CUDA check) off the startup path
    (`runtime.py:89`, `doctor.py:152`); `_resolve_vram` needs only `vram.probe()`.
12. **C30** — Move the bpy subprocess probe off the startup path (`doctor.py:59-95`);
    deliver via TaskRunner — `runtime.checks` is already replaced atomically.
13. **C31** — Split doctor checks volatile vs static (`doctor.py:30-52`,
    `system.py:31`): poll only port/disk/VRAM/job-object; compute weights once and merge.
14. **C32** — Defer the blocking startup storage walk (`main.py:254`) to the first
    `_request_storage`.
15. **C33** — Incremental storage accounting (`main.py:601`, `files.py:379`): add the
    finished dir's size to a running total; full re-measure only after delete/prune.
16. **C34** — Parallelize doctor with the loop-thread start in `Runtime._start`
    (`runtime.py:83-111`).
17. **C35** — Tighten the trellis health poll 1.0 s → ~0.1 s (`trellis.py:231-237`).
18. **C36** — Tighten port-reclaim polling 250 ms → ~50 ms (`trellis.py:312-316`).
19. **C37** — Hard-link instead of copy in remesh-retry staging (`queue.py:1178, 1264`);
    verify `_apply_scale`'s in-place rewrite doesn't defeat it, else fall back to one
    `to_thread` wrapping both copies.
20. **C38** — Skip the idle worker's per-second executor hop + sqlite query when nothing
    woke it (`queue.py:384-401`).

### B. Frame loop & rendering — `main.py`, `render.py`, `programs.py`, `textures.py`, `library.py`, Inker/Clay view files

21. **B11** — Idle-frame throttle (`main.py:331-335`): gate full redraw on pending
    input, live toast/progress, camera not settled (epsilon vs goals — `camera.py:130`
    never exactly settles), strip render or task in flight; otherwise 10–15 fps.
22. **B12** — Skip `viewer.render` when inputs unchanged (`main.py:1992-2016`) — the
    cheapest slice of B11, lands independently.
23. **B13** — Same skip for the Clay viewport (`main.py:1239` → `clay_view.draw`).
24. **B14** — Same skip for the compare pane (`viewer_embed.py:320-331`).
25. **B15** — Cache normal matrices, hoist constant uniforms in `_draw_model`
    (`render.py:141-164`) — mirror the overlay pass.
26. **B16** — Pre-sort `GpuPrimitive.defines` at build time (`programs.py:459`).
27. **B17** — Frame-scoped `stat()` memo in `ThumbnailCache.get` (`textures.py:67`).
28. **B18** — Cache `stats()` vertex totals on the immutable model
    (`viewer_embed.py:164-179`).
29. **B19** — Cache library `visible()` + `failures()` per job-list generation
    (`library.py:32, 166`).
30. **B20** — Hoist the queue-position map out of the card loop (`library.py:257`).
31. **B21** — Move `_review_findings`' findings.json load below the `widgets.header`
    guard (`main.py:1661`).
32. **B22** — Iterate only the visible index range in the Inker grid
    (`inker_canvas.py:639-646`).
33. **B23** — Viewport-cull marching ants: AABB pre-filter per loop + per-run visibility
    mask (`ants.py:82-136`, `inker_canvas.py:703`).
34. **B24** — Gate `layer_thumb` on a per-layer revision or throttle it
    (`inker_textures.py:118`).
35. **B25** — Cache Clay `element_centre`/`selection_centre` on
    (mesh id, selection id, transform) (`clay_view.py:561-630`).
36. **B26** — Compute Clay world matrices once per frame as `{uid: world}`
    (`clay_view.py:405, 472`); stop synthesising empty selections per unselected object.
37. **B27** — Cache `world_bounds` the same way (`clay_view.py:632`).
38. **B28** — Avoid rebuilding `_below` as a full extra composite on every structural
    change (`inker/document.py:217-233`); confirm the shared UndoStack byte budget
    accounts for Clay's whole-Mesh snapshots.

### D/L. Viewer loading & odds and ends — `gltf.py`, `scene.py`, `glctx.py`, `capture.py`, `textures.py`, `jobs_cache.py`

39. **D39** — Memoize glTF image decode by source index (`gltf.py:458`) — the dominant
    cost of `parse_model`; directly shortens "job finished → mesh on screen".
40. **D40** — De-duplicate GPU textures by decoded buffer (`scene.py:44-58, 249`).
41. **D41** — Reduce `read_rgba`'s stall (`glctx.py:85-96`): fold the row flip into
    `Image.frombuffer` + transpose; move the PNG encode (`capture.py:23`) to TaskRunner.
42. **L101** — Fix `ThumbnailCache._supersede`'s full-key scan (`textures.py:105`);
    keep a per-job index.
43. **L102** — Adaptive job-list poll (`jobs_cache.py:25`): 2–5 s idle, fast only while
    a job is live.
44. **L103** — Show the window before setup finishes with a lightweight loading state
    (composes with C29–C34; the splash already covers part of this).

**Phase 1 verification:** `uv run pytest` green after each landing; B11–B14 verified by
eye (no visual regression on Home/2D/3D/Clay); A-section items spot-checked with a
timed sweep submit and a 200-row library scroll. The mtime-keyed caches introduced by
any item here follow the racily-clean rule (`CLAUDE.md`; directory mtime is not a
reliable change signal on Windows).

---

## Phase 2 — Clarity: errors surfaced, onboarding, doctor, docs — **done**

*A record, not a queue. What each item concluded is in `CLAUDE.md` and in the comments at each
site; the tests are `tests/test_error_surfacing.py`, `tests/test_onboarding.py`,
`tests/test_message_clarity.py`, `tests/test_evidence_clarity.py` and `tests/manual/test_coverage.py`.
All 38 items landed. Six things are worth carrying forward because they are decisions rather than
edits:*

- **`guidance.GuidanceError` is a `ValueError` subclass that names its own control** (E49/S137), so
  every `except ValueError` in the repo is unchanged and the nine `Invalid(str(exc))` passthroughs
  could gain a field without a translation table. `errors.invalid_from` frames library text in a
  sentence and carries the field through.
- **`validation.check_weights` refuses a text job whose selected weights are absent**, at the door,
  beside `check_vram` and for its reason. The style LoRA is the case that mattered: `_load_loras`
  *skips* a missing style adapter, so the job used to finish wearing a `style_lora` param that never
  ran and join the findings corpus as evidence about it. `tests/conftest.py` materialises the probe
  paths, so the suite's pinned-empty model root does not refuse every text job.
- **`config.effective` is the one data source** for the diagnostics popup and `warlock doctor`
  (S140, and the thing K100 in Phase 3 reads). Its value is the `from_env` column, and it reads the
  environment rather than diffing against a fresh `Config`: a variable explicitly set to its default
  is exactly when somebody is asking whether their setting took. `config.SETTINGS` is asserted
  against the dataclass in both directions.
- **P120 removed a claim rather than adding one.** `AUC(hole_worst → reject) = 0.115` over the
  reviewed corpus, so the quality badge's green branch below 2% is gone — 48 of 81 rejected meshes
  measured exactly 0.0 and would have worn it, and none of the 3 accepted ones did. The escalation
  stays, because a *high* reading is still real evidence. `widgets.AUDIT_UNINFORMATIVE` is the one
  threshold. Two dead readers turned up on the way (`mesh_audit["verdict"]` in `review_mode`, whose
  test pinned a shape the worker has never written).
- **N112's load probes are keyed on the resolved weights directory, not on the kind.** The bpy
  answer can be a bare module global because it is a fact about the interpreter; these are facts
  about a path, and `WARLOCK_T2I_ROOT` moves it.
- **The manual was renumbered** to fit Home, Style profiles and App settings in the right places
  (01→19; Review moved out of Architecture into the user chapters). Chapter numbers decide both
  order and part, so a new chapter *is* a renumbering — and every cross-link, the index and
  `HELP_TARGETS` are asserted in both directions, which is what made it mechanical.
  `tests/manual/test_coverage.py` now also fails a pane with neither a (?) nor a stated exemption,
  which is the gap the O118 sweep found (inker-colors, candidates).

Two items were already satisfied and are recorded as such rather than re-done: G65 (the retarget
panel was documented as a user feature) and the reference-regeneration half of S139 — a reroll
creates a *new* job and copies `input.png` across, so it never rewrites the source job's reference
and leaves no `paint.ora` stale. The one path that does rewrite it in place is Inker's Revert, whose
confirm now names the layered document instead of "every edit".

## Phase 3 — UX: keyboard, notifications, library, forms, layout — **done**

*A record, not a queue. What each item concluded is in `CLAUDE.md` and in the comments at each
site; the tests are `tests/test_mode_keys.py`, `tests/test_palette.py`,
`tests/test_notifications.py`, `tests/test_library_browsing.py`, `tests/test_forms_and_layout.py`,
and the additions to `tests/test_studio_smoke.py`. All 43 items landed. Nine things are worth
carrying forward because they are decisions rather than edits:*

- **The mode switch is `Alt`+1..8, not Ctrl.** It has to be checked *above* Inker, Clay and Review,
  which consume every key they are given — so whatever it names, it takes from them permanently.
  Inker already bound Ctrl+0/Ctrl+1 to its zoom and Phase 4's Clay17 wants Ctrl+1/3/7. The digits
  come from `modes.MODES` positionally, so the binding is the picture on screen.
- **`previous_mode` is sampled once per key event, not once per frame** (`App._note_mode`). F1
  changes the mode from inside `_shortcut`, so a frame-start sample would still hold the mode from
  before it and Esc would go two steps back. Home is the floor rather than a place you escape from.
- **`ConfirmQueue`/`PromptQueue` are real queues and `pending` is read-only.** Assigning `None` to
  it was how a caller cancelled, which on a queue discards everything behind it too (`dismiss()`).
  `_request_quit`'s three hand-nested lambdas became a list walked by index — the *chain* stays,
  because cancelling the first must not leave two more questions to dismiss.
- **A modal owns the keyboard while it is up**, so the frame loop stops dispatching shortcuts:
  otherwise one Esc both cancels the dialog and leaves the mode behind it. Releases still pass,
  because Inker's space-to-pan is a hold.
- **Four toast levels as a table of dwell times, and the ladder is reading time rather than
  severity** — which is why success and info share one. An unknown level renders as an ordinary
  notice, the rule `action` already followed. A sweep raises exactly *one* toast, judged against
  the loaded window, which fails in the safe direction.
- **`parse_query` treats an unrecognised prefix as free text.** Silently reinterpreting a colon
  somebody typed on purpose is how a search starts returning nothing with no explanation. Field
  terms *add* to the combos rather than overriding them.
- **Every sort buckets its unanswerable rows last, in both directions** — the rule `best` already
  followed for an unranked job, generalised. "Unknown" is not a value at one end of a scale.
- **The trash is migration 9's `deleted_at`, and card Delete stopped asking**: the trash is the
  confirmation and a better one. Trashing a *queued* job cancels it, because deleting used to remove
  the row so the worker never saw it. Prune deliberately still deletes from disk.
- **The theme hook is a PEP 562 `__getattr__` on `theme`**, so `theme.ACCENT` — written at dozens of
  call sites — is a live lookup. A module constant would have repainted imgui's style and left every
  hand-drawn rect on the old colours. Two palettes of the same names; the light one keeps the
  elevation *roles* rather than inverting the numbers.

Two items were already satisfied and are recorded as such rather than re-done: **I84** (Inker's
layer drag-reorder and the Clay outliner's Ctrl/Shift multi-select both existed, and both are gated
on `busy`). K94's *disable-with-reason* landed; offering more tiers stays gated on R133.

**Visually verified 2026-08-08**, by `scripts/screenshot_modes.py` — the real `App`, every
mode, both palettes, rendered and read back off the framebuffer. Four defects, none of which
GL smoke coverage could have seen:

- **41 of 83 icons drew Inter glyphs.** imgui resolves a merged font's sources in order, so
  the base wins; Inter ships 745 PUA stylistic-set alternates and Lucide 0.525.0 spans
  U+E038–E682, overlapping on 478 codepoints. Settings was a division sign, Clay's ruler a
  question mark, Quit a digit six. `scripts/strip_font_pua.py` + `tests/test_fonts.py`.
- **The smoke suite runs at UI scale 1.0 and nothing else does.** `theme.apply` sets
  `item_spacing` through `sp()`, so grids subtracting a literal `8` per gap were exact at
  1.0 and short at 1.5 — the Inker toolbox lost its fifth column, Clay its fourth button,
  the Clay outliner reserved 32 px for a 58 px button. `widgets.grid_width`.
- **Nine full-width sliders had no visible label**, because imgui draws one outside the
  widget. `widgets.labeled_slider_int`/`_float`, `labeled_combo`'s rule.
- **Every landing tile drew a scrollbar** over a few pixels of overflow, on the app's first
  screen every launch. `widgets.card` takes `no_scrollbar`.

One thing was found and deliberately **not** changed, because it is a design call rather than
a defect: `viewer/env.BACKGROUND_HEX` is the literal dark `0x0F1014` and its comment says it
"matches tokens.BG so the viewport and the panels around it read as one surface" — which the
light palette breaks, leaving both 3D viewports black in a light UI. Making it follow
`theme.BG` means threading a colour through `viewer/render.draw` (which already takes one)
*and* adding it to `Viewer.render`'s B12 skip key, or a theme switch would not redraw.

---

## Phase 4 — Feature expansion + gated work — **17 of 25 landed**

*Net-new capability. TODO.md plans zero Inker or Clay work, so all 25 review items are
additive; both engines are pure by invariant, so every engine-level item lands with
headless tests. The gated subsection is not code — it is the GPU/network/human work
TODO.md §2–§8 already specifies, listed here so the roadmap accounts for every LIST.md
item.*

**Landed (2026-08-08):** Clay14, Ink1–Ink8, Ink10–Ink13, Clay15, Clay16, Clay17, Clay25.
Each is one commit; the tests are `tests/inker/test_filters.py`, `test_gpl.py`,
`tests/clay/test_diagnose.py`, `test_uv.py` and additions to the existing engine and
mode suites. Six decisions are worth carrying forward:

- **`_MODE_IDS` is written out and the numbers are free of `BLEND_MODES`' order.** The
  list is now menu order (grouped by family) and the kernel's numbering is 0–11
  regardless, so the menu can be regrouped without invalidating a DLL. `WARLOCKC_ABI`
  is 6. `hard-light` is written as `overlay` with the operands swapped **on both
  paths**, so the identity is exact rather than exact-to-an-ulp.
- **A per-tool option is a property over the active tool's dictionary**, not a rewrite
  of nine call sites. Each of those call sites meant "the tool in my hand" all along;
  the state layer was the thing that disagreed. Symmetry, the grid and the colours stay
  app-level, and a test pins that split.
- **Alpha lock restores the channel after the formula**, never folds the layer alpha
  into the weight — a weight-based version still raises the alpha of a half-transparent
  pixel, which is the whole thing the lock exists to prevent. It rides to disk as
  `warlock-alpha-lock`, with a hyphen: a colon is an XML namespace prefix and an
  undeclared one makes the whole `stack.xml` unparseable.
- **An autosave is never a save.** It does not clear dirty, move `saved_head`, retitle
  the tab or set `saving` — locking the editor for a second every two minutes would be
  worse than no autosave, and the others would answer "where should this go" on the
  user's behalf. Recovered documents open untitled *and* dirty.
- **Clay25's unwrap normalises over the mesh's own bbox, not per island.** Per-island
  scaling makes texel density depend on how big a face happens to be. The generators
  pack rather than overlap, because a canonical unwrap whose islands overlap cannot be
  baked to — which is the only reason these coordinates exist.
- **Shade Auto's limitation is written down rather than discovered**: a capped cylinder
  comes out entirely flat, because a per-face flag can only say "this face has no sharp
  edge". That is also the correct answer for this renderer, since smoothing the band
  would average the cap normals into the rim.

The smoke test grew **three columns** on the way through Ink2: stacked in one column the
Inker canvas child sits below the tools pane, imgui culls a child pushed past the visible
area, and a culled canvas uploads no textures — so one added row failed the animated-inker
test with an empty texture list and nothing to do with frames.

**Not started, and why:**

- **Ink9** (canvas rotation and flipped view). The only Inker item whose correct
  implementation is a viewport-wide coordinate change: `ants.py`'s fast path is written
  against "`to_screen` is a uniform scale plus this offset" (its own comment), and the
  composite, the floating buffer, the onion frames, the grid, the symmetry lines and the
  transform handles are all drawn axis-aligned. A half-done version silently misplaces
  every overlay, which is worse than not having it.
- **Clay18–Clay21** (axis constraints and numeric entry, snapping during element drags,
  proportional editing, bridge/extrude on `topo.py`). The four heaviest interaction
  items; none is blocked, they simply were not reached. One correction for whoever
  takes Clay19: `clay_view._element_world_transform` **already** applies both
  `snap_translation` and `snap_rotation` to an element drag, so the "consistency bug"
  half of that item appears to be already satisfied — check the object path against it
  before writing anything, and if they agree the item is snap-to-vertex alone.
- **Clay22** (icosphere, capsule, grid plane) and **Clay23**/**Clay24** (bbox readout
  and camera in `.wblk`; outliner ergonomics). Small; Clay22 now also owes canonical
  UVs, since every generator produces them and `tests/clay/test_uv.py` asserts it over
  the whole registry.

### Inker — 13 features (LIST.md's payoff-per-effort order). **Ink9 is the one left.**

1. **Ink1** — More blend modes (darken, lighten, soft-light, hard-light, color-dodge,
   color-burn, difference) in `inker/composite.py`: numpy reference first, optional C
   kernel second per the native-kernel invariant; `_MODE_IDS` stays written out; each
   mode gets an `ORA_OPS` mapping.
2. **Ink2** — Per-tool option memory (settings dict per tool in `InkerState`).
3. **Ink3** — Layer alpha lock + "sample current layer" eyedropper
   (`Document.eyedrop(layer_only=...)`).
4. **Ink4** — Whole-layer filters (brightness/contrast, hue/sat, levels, blur,
   sharpen) as pure functions in a new `inker/filters.py`, applied through the
   existing `PatchEdit` path, respecting the active selection; live-preview popup in
   `inker_bridge`.
5. **Ink5** — Selection grow/shrink/border (morphological ops on the 8-bit mask in
   `selection.py`); "select layer alpha" falls out of the same work.
6. **Ink6** — Multi-stop gradients (`gradient.ramp` takes a stop list; fg→bg stays the
   default preset).
7. **Ink7** — Custom canvas size + 3×3 anchor on resize (`resize_canvas` already
   supports anchoring; the popup just never passes it).
8. **Ink8** — A proper color picker: hex + HSV fields, `.gpl` palette import/export
   (`panes/inker_colors.py`).
9. **Ink9** — Canvas rotation and flipped view (display transforms in `inker_canvas`
   only — pixels untouched).
10. **Ink10** — Brush stabilisation + size-by-speed taper in `StrokeState`.
11. **Ink11** — Grid snapping + movable symmetry axis (`inker_canvas`).
12. **Ink12** — Radial/mandala symmetry (N-way rotation of stamp positions in
    `inker/brush.py`, around the now-movable axis).
13. **Ink13** — Crash-safe autosave: periodic `paint.autosave.ora` through the
    existing TaskRunner save path, gated on `busy`, deleted on clean save/close,
    recovery offer on open.

### Clay — 12 features. **Clay18–24 are the ones left.**

14. **Clay14** — Surface `check_manifold`/`ManifoldReport` in the UI
    (`clay/adjacency.py:307`): "2 holes · 3 non-manifold edges", click selects the
    offenders. Near-zero engine work; do first.
15. **Clay15** — Material palette Add/Remove/Rename via undoable edits
    (replacement-not-mutation stays the palette rule).
16. **Clay16** — Shade Smooth/Flat as ops in `clay_ops.py` (+ auto-by-angle reusing
    `glbimport`'s `FLAT_COSINE` idea).
17. **Clay17** — Axis views (Ctrl+1/3/7 presets) and an orthographic toggle in
    `ClayView`.
18. **Clay18** — Axis constraints + numeric entry during transforms, with a live-delta
    HUD; fits the single-commit-on-release model (`set_transform(was=…)`).
19. **Clay19** — Snapping during element drags: first apply the existing grid snap to
    `_commit_element_drag` (consistency bug as much as feature), then snap-to-vertex
    via the existing `pick.nearest_vertex`.
20. **Clay20** — Proportional editing (radius + smooth falloff on `_ElementDrag`) —
    the highest feature-per-line item in the list.
21. **Clay21** — Bridge edge loops + edge/vertex extrude on `topo.py`'s CSR surgery
    primitives.
22. **Clay22** — New primitives: icosphere, capsule, subdivided grid plane (one
    builder + defaults dict each, per the `GENERATORS` registry design).
23. **Clay23** — Bounding-box dimensions readout in `clay_props`; persist camera
    (yaw/pitch/distance/target) in `.wblk` (additive JSON key; `read_wblk` validates).
24. **Clay24** — Outliner ergonomics: drag-reorder, isolate (solo visibility),
    duplicate/delete in context menu.
25. **Clay25** — UV support for authored geometry: canonical UVs per primitive builder
    + a "Box Unwrap" op (project by dominant normal axis) — deliberately not LSCM.

### Gated — needs GPU, network, or a human (specs live in TODO.md; run any time the
resource is available, independent of the code phases)

- **P121** — Blind birefnet-vs-auto confirm, 8–12 units, labels hidden
  (`scripts/sweep_confirm.py` exists; TODO §2 run 1). Run **first** — everything else
  in this list leans on its answer.
- **P122** — Re-run the render sweep with `bg_removal=birefnet` as baseline, carrying
  the framing axis (`scripts/sweep_rebaseline.py`; TODO §2 run 2). Mark old units
  non-comparable (PROMPT_VERSION 3→4).
- **P123** — Redo the checkpoint ranking on the re-run's output; the per-checkpoint
  refusal rates now land in `findings.json` automatically (TODO §2).
- **R133** — Qualify the gltfpack tiers against the re-run's accepted meshes
  (`scripts/qualify_tiers.py` refuses to run on an empty corpus by design; TODO §3).
  On a pass: expose the tiers in `panes/settings_3d.PROFILES` and lift K94's
  disable-with-reason.
- **R134** — Rig handedness check on the first asymmetric re-run mesh: which side the
  `.L` bones landed; one-line sign fix if flipped, then a measurement doc (TODO §4).
- **R135** — Judge whether landmark placement improves skin weights, by eye, via the
  deformation battery's `rig_qa.png` (TODO §4).
- **Judge labelling session** — the human pass in Review's *Teach the judge* grids
  (both passes over the same images: product and blank are different questions), then
  the held-out accuracy measurement doc (TODO §7, §10). Unlocks §8's mesh probe once
  the re-run supplies positives in the tens.

**Phase 4 verification:** every Inker/Clay engine item lands with headless tests in
`tests/inker/` / `tests/clay/` (both engines are pure by invariant); UI items follow
`tests/test_inker_mode.py` / `tests/test_clay_mode.py` patterns; blend-mode/kernel
work keeps the numpy reference and the bit-parity bar; new blend modes preserve `.ora`
interop (Krita/GIMP round-trip). Gated items each end in a measurement doc under
`docs/measurements/` where they move a corpus-keyed constant, per the repo rule.

---

## Phase 5 — Dedicated texture generator

*Additive beyond LIST.md (designed 2026-08-08). Three tiers, in dependency order:
materials first (mostly wiring over the existing seamless-tile path), mesh re-texturing
second (the real feature; its spec is largely written across `optimize_job`, the rig
publication pattern and TODO §11's bake op — build the bake infrastructure once for
both), a dedicated texture model third and only if the bake path proves insufficient.
Hard prerequisite: **Clay25** (generator UVs + box unwrap, Phase 4) — Clay geometry
currently has no UVs, so nothing user-modelled is texturable until it lands.*

### Tier 1 — Material generator (tileable PBR sets) — **done (2026-08-08)**

*A record, not a queue. T1 is one commit, T3+T4 one, T5 one; the tests are
`tests/test_material.py`, the additions to `tests/test_derive_2d.py`,
`tests/test_tile_form.py` and `tests/test_inspector_exports.py`. Five things are worth
carrying forward because they are decisions rather than edits:*

- **`SEAM_MAX` is 3.5, and the corpus is the reason** — `docs/measurements/2026-08-08-seam-threshold.md`.
  2.0 raised two false alarms on tiles that wrap perfectly, and the shape that breaks
  the ratio is large flat cells with thin hard lines: the denominator is a mean over
  every adjacent pair and so is tiny on a mostly-flat picture, while the numerator is
  one column that may land on a grout line. The measured empty band is 2.50–5.52 and
  3.5 is its geometric centre rounded. `scripts/calibrate_seam.py` and
  `calibrate_seam_hard.py` rebuild the corpus from fixed seeds.
- **T2 was already shipped as `output=tile`** and is recorded rather than re-done. It
  goes through `create_job`, takes the same validation door, VRAM admission and
  directory-before-row ordering, refuses non-SDXL families at the door and at the pipe,
  and `seam_report` was already in `DERIVED_PARAMS`. Adding an `output=texture` spelling
  would have been two names for one stage.
- **`pipelines/material.py` says it is estimating.** Every map carries its one stated
  assumption (darker is deeper; fine detail is roughness) *and* the case where that
  assumption fails honestly. Metalness is not estimated at all — a metal and a dark
  dielectric of the same colour are the same pixels — roughness is squeezed into
  0.35–0.95 because neither end is a claim it has earned, and the height map is
  deliberately not offered as `occlusionTexture`. Every Export label says "est." and the
  zip repeats it in a README, because those are the only places the user reads.
- **The wrap is the load-bearing part and the shift test is the proof.** Derivation
  commutes with a cyclic shift iff every neighbourhood wraps, so
  `derive(roll(x)) == roll(derive(x))` catches a clamped edge anywhere in the chain
  without knowing where the chain's edges are. A clamped edge would put a seam in the
  derived map that the albedo does not have — invisible until the material is on a
  surface.
- **`material.zip` re-derives rather than asking `get_file`.** It already holds its own
  artifact lock, and reaching for three more would establish an ordering between four
  locks where the module has one stated rule about two.
- **T5's toggle already existed** (Object / Seamless tile) and the maps derive with
  nothing to opt into, so what T5 actually needed was both directions of working with
  the albedo: a tile is now openable in the Inker (`files.EDITABLE_STAGES`, with
  `_remeasure` re-measuring the *seam* rather than a composition report a tile has no
  subject for), and the 2D viewport can draw it repeated — off by default, because every
  other view of an asset shows one cell and a viewport that silently showed four would
  make the texture look a quarter of its size.

### Tier 2 — Mesh re-texturing (new look for an existing `model.glb`'s UVs) — **done (2026-08-08)**

*A record, not a queue. T6 is two commits (the pure host half, then `op_views`/`op_project`), T7 one,
T8 one, T9/T10/T11 one; the tests are `tests/test_retexture.py` plus the re-texture sections of
`tests/test_service.py`, `tests/test_vram.py` and `tests/test_studio_plumbing.py`. Seven decisions
are worth carrying forward:*

- **It is a queue job, and T7's "near-sibling of `optimize_job`" means the refusals, not the
  placement.** A retarget is a two-second gltfpack subprocess and belongs inline; six SDXL passes
  around two Blender runs need the resident pipe, so a TaskRunner thread would be the OOM that only
  reproduces under load. What *is* the sibling: the `Conflict` on a queued or running source, the
  untouched `source.glb`, and every refusal at the door rather than in the worker.
- **The mesh is published into the source job's directory, by rename.** The rig's pattern verbatim,
  and the same three consequences: a cancel deletes only the temp (`rigging.RETEXTURE_GLB_TMP`) and
  this job's own renders, `_discard_artifacts` resolves `source_job`, and nothing ever sees a
  partial `model.glb`.
- **`SURFACE_DERIVED` is a stated subset of `files.DERIVED`, asserted in both directions.** STL and
  the convex collision hull carry no material, and a re-texture changes no geometry, so deleting
  them would cost a re-export to produce a byte-identical file. It lives in
  `pipelines/retexture.py` because `queue.py` deletes them and may not import `service`; the
  per-artifact lock it deletes them under is injected by `studio.runtime` for the same reason, and
  defaults to a null lock.
- **The atlas swap appends, and it goes through `glbio`.** Re-exporting through trimesh would
  re-encode every texture and rewrite the node graph, which is where the grounding transform lives.
  Overwriting the image in place is wrong the moment anything else references it, so a new image, a
  new texture, and every material sampling the old one repointed. An untextured but unwrapped mesh
  *gains* the slot; a mesh with no `TEXCOORD_0` is refused.
- **T10's assertion has two halves.** The rig, its poses and its sheets are asserted not stale, and
  the retarget's own report is asserted still to name all of them — equal-by-accident is the failure
  a bare "is empty" check cannot see.
- **T11: no observations row.** The corpus is about what *generation* settings produce, which is
  `import_mesh`'s and the retarget re-audit's argument unchanged. It falls out of
  `_observe_finished`'s existing kind gate, which is exactly why it is pinned by a test.
- **The default atlas size is the mesh's own, and that is a measured correction.** A flat 1024
  resampled every trellis mesh's 2048 atlas down by half — including the ~63% that no view covers
  and that therefore *keeps its old colour*, which is the half a re-texture must leave alone.

**The verdict on the bake: `docs/measurements/2026-08-08-retexture-bake.md`.** Sufficient as a
recolour, insufficient as a re-texture, and the cause is **coverage** — 36–37% of the atlas over four
real reconstructions, because a perforated mesh's interior walls project onto holes in the render and
are masked out. Neither atlas resolution nor view count is the limit: 1024 and 2048 are
indistinguishable, and front-plus-back reach 36.1% while the other four views add 1.0 pp between
them. Two findings from the same runs are load-bearing elsewhere: the bake is **faithful** where it
has something to bake (every plank seam and bolt mark survives in the one valid positive control),
and **`clay/uv.py`'s box projection stacks all six faces of `primitives.box()` on one unit square**
(24 corners, 4 distinct UVs) — so a Clay-authored asset has UVs and cannot carry a texture, and
Phase 5's stated Clay25 prerequisite is only half met.

### Tier 3 — **Superseded by the Tier 2 measurement.** Coverage, not fidelity

*T12 as written — "a UV-space diffusion or material-decomposition model" — is no longer the
indicated next step, and the measurement is the reason rather than a change of mind: a better
texture model would improve the third of the atlas that is already being painted and would do
nothing about the two thirds that are not. Kept here rather than deleted because it is a decision
with evidence behind it, and the evidence is what a future reader needs. In order:*

12. **Pack Clay's box unwrap.** The smallest of the three, a prerequisite Phase 5's own preamble
    already names, and the difference between a feature that silently does nothing on authored
    geometry and one that works. `clay/uv.py` is explicit that overlapping islands are what a cube
    projection *is* and that the primitives able to pack cheaply should do so in their own
    generator; `primitives.box()` is the one that does not. The strength-0.85 control shows the bake
    is ready for it.
13. **Attack coverage.** More axis views are measured not to help. What would: weighting by
    *visibility* rather than by facing — a depth test from the camera, which also removes the
    `occlusion_tested: false` overhang smear — and views chosen from the mesh rather than from a
    fixed basis.
14. **Only then reconsider a texture model**, as a registry spec with family/residency declared, a
    `models.Fetch` deduped on `(repo_id, destination)`, download only through `fetch_worker`, and
    `local_files_only=True` at load — the T12 shape, unchanged — A/B'd through Review against the
    bake on a corpus whose coverage is high enough for the difference to be attributable to the
    model. It gets its own spec first.

**Phase 5 verification:** pure modules (T3) get headless tests; T6/T7 follow the rig
worker's test patterns (temp-name publication, cancel semantics, `_discard_artifacts`
resolving the right directory); T2's admission and DERIVED_PARAMS additions get the
existing validation-test treatment; T1 and any threshold constant land with a
measurement doc. Full suite green both native and `WARLOCK_NATIVE=0`, as everywhere.

---

## Verification (whole roadmap)

- `uv run pytest` (427–433 tests) is the regression gate after every landing, in every
  phase. Suites must pass with `WARLOCK_NATIVE=0` as well wherever a kernel is touched.
- Item accounting: 165 LIST.md items = 150 phased above (Phase 1: 44, Phase 2: 38,
  Phase 3: 43, Phase 4: 25) + 6 gated (P121–123, R133–135) + 9 excluded as done
  (Q125–132, G66). The judge labelling session is gated work too but is TODO-tracked
  rather than a LIST.md item.
- Any schema change is an append-only migration; confirm `len(MIGRATIONS)` first.
- Where this file and `TODO.md` disagree, TODO.md is right.
