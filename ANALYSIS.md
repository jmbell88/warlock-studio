# QA Audit — 2026-08-06 (post-v0.0.9)

Six parallel audit passes over the whole codebase: concurrency/DB/queue, the service
layer, studio UI/modes/viewer, inker+clay+undo, pipelines/offline, and
docs/tests/config. Baseline before any change: **2612 passed, 7 skipped, ruff clean.**
Every finding below was verified in source (several reproduced empirically) before
being acted on. Known-deliberate behaviours (reference_prep pinning, the band default,
single-image conditioning, the missing gltfpack binary) were excluded up front.

## Fixed in this audit

### Correctness bugs

- **`flatten_layers` stranded a floating buffer** (`studio/inker/document.py`). Every
  other canvas-level op commits the float first; flatten replaced the whole stack, so a
  lifted selection's undo record named a layer that no longer existed — the next Ctrl+Z
  raised `KeyError` out of `by_uid` and the lifted pixels were gone. Now commits first;
  regression test added.
- **`.wblk` with objects but no `materials` array** (`studio/clay/serialize.py`) got an
  empty palette instead of the default one — everything rendered and exported magenta,
  and the properties panel raised `IndexError`. The reader now substitutes the default
  palette (an empty *scene* still round-trips its empty palette); regression test added.
- **Verdicts could be filed against unfinished jobs** (`service/verdicts.py`). Review
  lists queued/running/failed units, and pressing A recorded a permanent, denormalized
  accept for a mesh that never existed — poisoning the findings corpus forever.
  `record_verdict` now refuses any job that is not `done`; the Review UI already
  catches the error and toasts.
- **Sweep admission wasn't all-or-nothing** (`service/sweeps.py`). `_check_unit` missed
  two of `create_job`'s refusals (resolution allowlist; conditioning-needs-a-reference),
  so a sweep over `ip_adapter`/`control` or a bad resolution minted the sweep row and
  died on the first unit with an unnamed error. Both checks now mirrored.
- **Remesh skipped VRAM admission** (`service/jobs.py`). `rerun_job` in remesh mode
  promotes a reference (admitted against SDXL cost only) to a model job without
  `check_vram` — the exact path the refusal message itself recommends. On a card where
  promote correctly refuses, remesh queued the same reconstruction and failed minutes
  later (or overcommitted, which on Windows is the host-commit crash). Now checked.
- **`prune_jobs`/`delete_job` raced the worker's claim** (`service/jobs.py`, `db.py`).
  The running-check was against a page snapshot; a queued job claimed in the gap had
  its row and directory deleted under a live reconstruction. Added
  `JobStore.delete_if_not_running` (status check and delete in one statement under the
  lock, the same shape as `claim`) and routed both callers through it.
- **`normalize_glb` grounding transform** — see *Deferred*, this one is real and
  reproduced but needs a decision.

### Races and thread-safety

- **`TrellisServer.running` / `_reap_if_dead` check-then-act** (`pipelines/trellis.py`):
  re-read `self._proc` after the None test while `stop()` nulls it from another thread
  — a cancel during remesh could die with `AttributeError` instead of unwinding. Both
  now read `_proc` once.
- **`TrellisServer.generate` blocked the event loop** writing the returned GLB (tens of
  MB) inline — during which cancellation and every `call_on_loop` stalled. Validation
  and the atomic write now run through `asyncio.to_thread`.
- **Worker poll loop had no backoff on persistent errors** (`queue.py`): a corrupt DB
  page or full disk spun the loop flat out, writing a traceback per pass to the disk
  that caused it. Error branch now sleeps `POLL_INTERVAL`.
- **`winjob._ensure_job` was unlocked** while genuinely called from four threads — two
  overlapping first calls could mint two job objects and leak one, with `armed()`
  reporting on a handle some children weren't in. Now under a lock. Also
  `CreateJobObject`'s failure log always printed `(0)` (`ctypes.get_last_error` with
  `windll`'s `use_last_error=False`); now calls `GetLastError` directly.

### UI state and GPU-resource lifetime

- **Applying a saved pose (and picking a preset) bypassed the discard-confirm**
  (`panes/pose_panel.py`): both overwrite hand-made rotations, and Apply also cleared
  the dirty flag so the loss was silent and permanent. Both now go through `guard`,
  like every other exit route.
- **Selection change mid-compare leaked the split's GPU half** (`state.py` clears the
  flag but can't reach the viewer): the stale mesh kept rendering a full second scene
  draw every frame, invisibly, forever. `_draw_viewport_image` now reconciles.
- **Two forget-before-release violations**: `viewer_embed.clear_reference` and the
  sheet panel's strip texture both released registered textures without
  `forget_texture`, leaving the imgui backend holding dead objects under GL names the
  driver reuses — the exact "unrelated image renders here" / use-after-release hazard
  the rule exists to prevent. Both fixed; the strip texture is also now released at
  teardown (it leaked on every exit).
- **Inker toolbar Undo/Redo weren't gated on `saving`** (the keyboard path and every
  other control are): an undo of a replay-edit mid-save could rebind the stack while
  `write_ora` was between `stack.xml` and the layer PNGs — a torn archive. Gated.
- **Save/Save-as/Export were enabled mid-free-transform**: saving committed the
  transform with no confirm and left the transform mode stuck pointing at nothing.
  Gated on `state.transforming`.
- **Clay's viewport accepted new drags while a save was in flight**, contrary to the
  rule every panel follows (and to `clay_mode`'s own docstring). New presses are now
  refused while saving; an in-flight drag keeps its release (the save's bytes were
  captured before it started, and swallowing the release would strand `_grab`).

### Hardening and hygiene

- `optimize_job` now iterates `files.DERIVED` instead of a duplicated literal tuple
  (a sixth export added to one and not the other would serve stale pre-retarget copies).
- `save_clay_source` now calls `check_job_id` like every other path-building entry
  point (was defended only by the DB lookup).
- All five pre-row asset writers (`create_job`, `import_reference`, `import_mesh`,
  `rerun_job`, `promote_to_model`) now clean up their directory when the *payload
  write* fails, not just the DB insert — a disk-full mid-write no longer leaves a
  truncated orphan directory.
- `load_lora_weights` (both call sites) and `load_ip_adapter` now pass
  `local_files_only=True`, closing the last three weight loads that would be free to
  touch the hub on a host with `HF_HUB_OFFLINE=0` in the environment.
- Stale comment in `models.py` pointing at a `_SCHEDULERS` table that doesn't exist.

### Docs, tests, config

- **The documented setup command failed outright**: `dev` moved to
  `[dependency-groups]` but README, CLAUDE.md, and the installation chapter (three
  places) all still said `uv sync --extra dev …`, which errors on a fresh clone. All
  fixed; the extras table now explains dev is a default group.
- **Manual TOC dropped chapters 15–16**: `loader.PARTS` stopped at 14, so *Extending*
  and *Review* rendered as bare "Contents" entries. Ranges now cover 1–16 (and
  Troubleshooting moved from "Architecture" to "Setup & operations"); a test now
  asserts every real chapter lands in a part.
- CLAUDE.md pointed forward-planning at `docs/TODO.md`, which v0.0.9 deleted; the
  `docs/` inventory sentence also missed `docs/REPORT.md`. Both corrected.
- Six live env vars were missing from the configuration manual's "every variable"
  table (`WARLOCK_RANK`, `WARLOCK_REFERENCE_RETRIES`, `WARLOCK_MESH_RETRIES`,
  `WARLOCK_MESH_HOLE_MAX`, `WARLOCK_VRAM_BUDGET`, `WARLOCK_VRAM_TOTAL`); added. The
  README's "Everything is env-overridable" overclaim now defers to that table.
- A bench-suite fixture note cited the deleted `docs/NEXT.md`; reworded.
- `pyproject.toml`: the `rig` extra's explanatory comment was stranded above the
  `studio` extra; moved home.
- `tests/inker/test_document.py` renamed to `test_inker_document.py` — it collided
  with `tests/clay/test_document.py` and collection survived only because of an
  undocumented `__init__.py` asymmetry.
- `.gitignore`: `assets/`, `models/`, `sweep/`, `scratch/` root-anchored (matching
  `/bench/`, `/vendor/`) so a future same-named package directory can't silently
  vanish from the index.

## Deferred — needs your input

1. **`normalize_glb` grounds in the wrong frame for GLBs whose roots carry a
   transform** (`pipelines/postprocess.py`, `_insert_transform_below`). Reproduced: a
   unit box under a rotated, translated root comes back sunk half its height and
   offset. Trellis output is safe (trimesh emits an identity root), but **Clay exports
   and Blender-authored uploads put real TRS on root nodes**, so `import_mesh` grounds
   them wrong and `meshreport` immediately files a pivot complaint against a mesh the
   user just authored. Proposed fix: insert the grounding node *above* the scene roots
   (one new root wrapping them) instead of below each root; needs care because
   `optimize_job` reapplies the transform after gltfpack rewrites the node graph, and
   the viewer/`stale_rig_artifacts` both read the node structure. Say the word and
   I'll implement it with a regression test matrix (identity root / rotated root /
   multi-root / size_m).

2. **The pose editor can bind one job's rig while the panel targets another**
   (`panes/pose_panel.py` + `inspector.py`). Enter pose mode on job A, click job B in
   the library: Save writes A's rotations into B's poses, and joints mode re-skins B
   against A's skeleton — cross-job data corruption. Fix direction: bind the editor to
   the job id it was opened on (or guard selection changes out of pose mode). Both
   change UI flow, so choose: **(a)** panel follows the editor's bound job regardless
   of selection, or **(b)** selection change guards/exits pose mode.

3. **`delete_sweep` deletes running units** (`service/sweeps.py`) — `cancel_job`
   returns before the worker stops writing, so the rmtree'd directory can be
   resurrected as an orphan (reproducible when cancelling during the t2i phase).
   Options: refuse while any unit is running (mirror `delete_job`'s `Conflict`), or
   cancel-and-wait. Related: deleting a *done* mesh job while a rig/sheet job for it
   is running recreates its directory the same way (`finalize_rig` renames into a
   deleted dir). Both want the same decision about dependent/running work.

4. **Three frame-thread stalls** the audit measured as real but whose fixes are
   architectural: `_sync_viewer` runs `load_model` (full GLB + texture decode) on the
   frame thread at the exact moment a job finishes; `jobs_cache.tick` re-stats 9 files
   × every listed row twice a second (unbounded after "load more"); the sheet panel's
   "Refresh preview" does N offscreen renders + GPU readbacks inline (16 directions =
   a visible freeze). All three need a load-off-thread/hand-over pattern (the GL
   upload must stay on the frame thread) or caching; worth one focused pass.

5. **`runtime.shutdown` closes the store after a *timed-out* task-pool shutdown** —
   a task that survives the 30 s grace and then touches the DB gets
   `ProgrammingError` swallowed into a dead future. Either skip `store.close()` when
   the pool timed out, or correct the comment that claims the ordering is sufficient.

6. **Smaller judgement calls**, cheap to do on a nod: `_sync_viewer` never clears a
   mesh left by Review when the 3D selection has nothing to show (stale sweep mesh
   under another job's inspector); sweeping `ip_scale`/`control_scale` with no adapter
   in the base produces N byte-identical units (guidance.normalize drops the orphan
   scales) — `expand` could refuse or collapse them; `prompt.chunk` canonicalizes
   comma spacing so the recorded prompt can differ from the encoded one
   (provenance/recipe mismatch on re-run); `guidance.py`'s `consumable` fragment is
   5 words against the module's own 2–4-word rule (trimming changes generated output,
   so it should ride a deliberate prompt-version change); deleting a source asset
   isn't blocked while a rig for it runs (subset of #3); `inker_colors` is the one
   drawn pane with no help button, invisible to the help-coverage test.

## Verification

- Full suite after all fixes: see commit — run was green (`uv run pytest`), ruff clean.
- New regression tests: flatten-with-floating-buffer, objects-without-materials
  palette, every-manual-chapter-has-a-part.
