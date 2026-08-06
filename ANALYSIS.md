# QA Audit — 2026-08-06 (second pass)

Four passes over the codebase: a line-by-line audit of the uncommitted evidence and
pixel-art features, a re-verification of everything the first audit deferred, a
repo-wide sweep for half-wired features, and a manual trace of the diff. Baseline
before any change: **2680 passed, 7 skipped, ruff clean**. After: **2736 passed, 7
skipped, ruff clean**.

The explicit-marker sweep came back empty — no `TODO`/`FIXME`/`XXX`/`HACK` anywhere in
`src/`, no `NotImplementedError` outside the two abstract `Edit` hooks in
`studio/undo.py`, no bare `...` bodies, and every `pass` a commented deliberate swallow.
Every defect below is a boundary mismatch or a dead half, not unfinished code.

The first audit's own record is in git (`git show d6fd407:ANALYSIS.md`); everything it
deferred is resolved here.

## Fixed

### The uncommitted features, before committing them

- **Machine evidence never reached `findings.json`.** `service.findings.refresh` had two
  call sites, both inside a verdict path, so the observations the worker appends on every
  finished model job arrived only when somebody next pressed A or R — and on an install
  with no verdict ever filed, the file did not exist and the whole observation hint tier
  was unreachable. Both claims to the contrary shipped in the same diff (`CLAUDE.md` and
  `docs/manual/16-review.md`). The frame loop's `announce` callback now marks
  `AppState.findings_dirty` for a finished model-stage job, mirroring
  `_observe_finished`'s own condition, and `review_mode.pump_findings` submits it. **A
  flag pumped every frame, never a direct submit**: `TaskRunner.submit` refuses a key
  already in flight and nothing re-armed it, so a burst of verdicts recomputed over the
  set as it stood at the first press and the request following the *last unit of a sweep*
  was dropped for good. The tests missed it because every one of them called `aggregate`
  directly; the new one asserts the file carries an observation with no verdict filed.
- **The pixel preview claimed to save a file.** It submitted `derive.get_file` under
  `save:<job>:<name>`, and the app toasts `"Saved to <result>"` for every finished `save:`
  key — while `get_file` returns the path *inside* the job directory and is never None. So
  "Preview pixels" reported a save to a path the user never chose, with no dialog shown.
  `app_ctx.derive_key` is a namespace of its own now, and `Ctx.artifact_busy` checks both
  so a preview and an export of one artifact still cannot describe different files.
- **An existing v1 `findings.json` rendered every recipe as "(0%+)"** — the pane defaulted
  a missing `wilson_low` to 0.0, under a heading claiming the Wilson ranking.
  `bench.findings.vector_line` omits the bound instead, which is what `hint` already did.
- **Every mean was printed beside the bucket's `n`, not its own.** A `meshreport` that
  failed or returned `status: "invalid"` leaves an observation with holes and no watertight
  flag, so one reading in twenty-one was advertised as twenty-one. `_metric_summary` emits
  `counts` per metric; the reader thresholds and labels per measurement, and drops a thin
  one rather than letting a fat one carry it.
- **A payload that was not JSON took down the whole refresh.** The type checks in
  `aggregate` never ran, because the decode is in the store: `JobStore._blob` returns
  `None` for an unparseable blob, which every reader already treats as a row to skip.
- **The machine-evidence hints were absent from the pane they were made for.** The 3D pane
  hinted one control out of five while observations measure *geometry*; `platform`,
  `bg_removal` and `reference_prep` now carry hints (`size_m` and `custom_triangles`
  deliberately do not — continuous values never meet the threshold). The reader says
  "(21 meshes)" rather than "runs", because a model-stage observation credits the whole 2D
  taxonomy its job was promoted with, so these lines appear under prompt controls where
  the number has to name what it measured.
- Smaller: `ThumbnailCache` retires the superseded *version* of a key instead of leaving
  it to LRU (a palette-tuning session evicted library thumbnails to make room for its own
  dead textures) and takes the sampling filter into the key; `_record_observation` moved
  after `_maybe_queue_rig`, so the new await cannot cost a job its auto-rig on shutdown;
  `comparison_lines` is built after the collapsing header rather than every frame with the
  section closed; `oriented()` refuses a contrast whose two sides render identically;
  `idx_observations_sweep` is gone (nothing queries that table by sweep); the `sweep_unit`
  columns and the migration's "one transaction" comment now say what is actually true.

### Correctness

- **`normalize_glb` grounded in the wrong frame, and once per root.** Bounds are measured
  in *world* space but the transform was applied under each root, composing as
  `M_root · T · S` instead of `T · S · M_root` — so a rotated root rotated the grounding
  offset. And the same world-space translation was applied once per root while each root's
  own offset stayed unscaled above it, so a Clay export (**one root per object**) grounded
  every object as though it alone were the scene: three boxes asked for 2 m came out 8.2 m
  and reported 2. trellis output hid both, because trimesh writes one identity root.
  `_insert_transform_below` now empties the root and folds its TRS into the inserted child.
  A wrapper node above the roots does not work — trimesh's base frame is named `world` and
  so is the root its own exporter writes, and the collision resolves into a cycle that
  drops the transform again. Pinned across identity / rotated / translated / non-uniformly
  scaled / matrix / multi-root, with and without `size_m`.
- **The pose editor wrote one job's rotations into another.** `_enter` used the job id and
  discarded it, and `_sync_viewer` deliberately does not reload while posing — so clicking
  another asset left the rig on screen under that asset's inspector, and Save wrote there.
  Nothing downstream could catch it: rig validation checks bone *names*, and two jobs on
  one template have the same ones. The editor records `viewer.pose_job_id`, and the panel
  withholds every control (rather than retargeting them) when the selection has moved.
- **`delete_sweep` deleted running units.** It called the unguarded `store.delete` and then
  rmtree'd, where `delete_job` and `prune_jobs` both use `delete_if_not_running` — and a
  status check would not have been enough anyway, because `cancel_job` writes `cancelled`
  and only *asks* the worker to stop. `worker_is_inside` reads `current_job_id`, which the
  worker clears in its own `finally`, after the last write. A unit still being written is
  cancelled and left, and so is the sweep row, so the second press finishes the job.
- **Deleting an asset while a rig or sheet for it ran** recreated its directory as an
  orphan: those are separate job rows whose artifacts land beside the `model.glb` they were
  fitted to, and the target's own status says nothing (it is `done`, which is why a rig
  could be queued for it). `dependent_jobs` refuses in `delete_job` and skips in
  `prune_jobs`.
- **Three Clay paths edited geometry without freezing the generator.** `clay_ops.run`'s
  docstring claimed it cleared `generator` "in one place, for every op"; only
  `run_mesh_op` and Smooth did, so Delete, Bake Transform, Mirror — and an element drag,
  which nobody had listed — left the object still claiming to be "box, size 1". The
  properties panel kept offering that size field, and touching it rebuilt a pristine box
  over the edit, with no warning. The freeze is `Document.set_mesh`'s now, with
  `keep_generator=True` for the one caller whose new mesh *is* what the generator makes.
- **A sweep whose units are all the same job is refused.** `expand` compares each unit
  against the base only, and `guidance.normalize` drops a setting with nothing to apply it
  to — so an `ip_scale` axis with no adapter in the base (or a `lora_weight` axis with no
  style LoRA) produced up to 64 byte-identical jobs at one seed. Hours of GPU redrawing one
  picture, and N bogus "distinct configs" in the verdict corpus.
- **`provenance.trellis_recipe` recorded the wrong optimise tier.** It read
  `params["mesh_profile"]`, which nothing writes — a job stores it as `params["profile"]` —
  so every recipe recorded the config default and a bench recipe carrying "standard"
  recorded "raw". The two keys beside it were right, which is what made it look like a name.

### Frame-thread stalls

All three measured, all fixed by doing less per frame rather than by moving GL work:

- `attach_files` takes a caller-owned cache stamped `(status, job-dir mtime)` — one stat
  per row instead of ten, on a two-hundred-row page, twice a second, unbounded after "load
  more".
- `_sync_viewer` parses the GLB off-thread (`Viewer.parse_model`) and adopts it on the
  frame thread (`_adopt_model`), where the GPU upload must stay. It used to run the whole
  load inline on the frame a job transitioned to done. `load_model` survives for the
  callers where the wait *is* the response.
- `viewer/sheet.StripRender` renders one cell per `step()`. Sixteen draws each followed by
  a synchronous `read_rgba` was a visible freeze; sixteen frames of one is not. The
  `ctx.busy("sheet-preview")` guard that stood there was scaffolding against a key nothing
  ever submitted.

### Controls that did nothing, and dead halves

- The Clay context menu never consulted `tab.saving`, so every row stayed clickable during
  a save and the click was swallowed silently; the parameter popup's Apply did the same.
  Both are greyed now, and the menu says why once at the top.
- `clay_tools`' Delete checked its predicate *after* the click, drawing a live red button
  that did nothing — the one button where "nothing happened" is hardest to tell from
  "something irreversible happened". `widgets.destructive_button` takes `enabled` now.
- `clay_ops.Param.integer` was declared, set on Smooth's `levels`, and never read, so the
  field accepted 1.5 and the op truncated it.
- A blank vector-preset name was refused silently: the modal closed and nothing was said.
- Vector presets could be saved and applied but never deleted, and nothing capped the list.
- A bare-letter key bound to a *parameterised* op set `pending_op` and could not open the
  popup from the event layer, leaving the mode holding a request it could not act on.
  Unreachable today; `open_op_popup` is the request the pane acts on.
- `pose_panel`, `sheet_panel` and `review_mode.preview_units` swallowed exceptions with no
  log, where every comparable site writes one.
- **`Runtime.shutdown` closed the store after a *timed-out* pool shutdown.** The comment
  said the ordering means "a task still calling into the service finds the store open" and
  the next one said the wait is bounded; the two are only compatible while the pool
  drains. `TaskRunner.shutdown` returns whether it did (it was discarding
  `concurrent.futures.wait`'s `not_done`), and the connection is left open when it did
  not — the process is exiting either way, and leaking a handle for the last second of a
  shutdown costs less than a `ProgrammingError` captured into a future nobody will poll.
- Deleted: `glctx.read_image`, `glctx.create_standalone`, `picking.world_positions`,
  `picking.rotation_between`, `gltf.Model.reset_rotation`, `inker.Document.stroking`,
  `Document.checkpoint`, the eight-member "compatibility with the flat editor" block (the
  pane it served is gone), `JobsCache.children`, and the empty "Add-ons" settings section.
- Kept with the reason written down: `service.system.health` (each half has a cheaper
  reader now; it survives as the one "state of everything" answer),
  `clay.adjacency.check_manifold` (O(corners), and `import_mesh` already measures the
  exported GLB through `meshreport`), `JobStore.children` and its index.
- `panes/app_settings.py` no longer promises a future release.

### Documentation

- **The 2D export family had no manual coverage at all** — `icon.png`, `sprite.png`, the
  three pixel artifacts, `manifest.json`, `wrap_preview.png`. The overview tells the reader
  2D mode has an Export tab; nothing said what was in it. New `## 2D exports` section in
  `02-generating-references.md`, including the pixel size/colours knob and the manifest.
- **The seamless-tile stage had none either** — the words *tile* and *seamless* appeared
  nowhere in `docs/manual/`, for a first-class output with its own library filter, its own
  seam measurement and its own export. New `## Seamless tiles` section.
- `11-configuration.md` now mentions what `studio_settings.json` holds beyond the pane.
- `CLAUDE.md` gained the findings-refresh seam, the grounding composition rule, the
  set_mesh freeze, the derive-vs-save key split and the frame-thread costs.

## Deferred — needs a deliberate decision

Both of these change generated output, so they must ride a `pipelines/prompt.PROMPT_VERSION`
bump in the same change or `provenance.versions()` will call two incomparable runs
comparable. Do them together or not at all.

1. **`prompt.chunk` canonicalizes comma spacing** (`pipelines/prompt.py:136-165`) — it
   strips around commas and drops empty phrases — but `queue.py:801` records the
   *pre-chunk* string. Provenance fidelity only: `chunk` is deterministic and idempotent,
   so re-running the recorded string reproduces the encoding exactly.
2. **`guidance.py`'s `consumable` fragment is five words** where the module's own rule is
   2–4, and it is the only over-length phrase in the file. Trimming changes output for
   every `category=consumable` job.

Also noted and left: **Clay's snap applies to rotation and translation but not scale**
(`clay_view.py`). There is no scale-increment field in the tools pane, so this is almost
certainly intended — it just says so nowhere.

## Verification

- `uv run pytest` — 2736 passed, 7 skipped. `uv run ruff check .` — clean.
- New regression tests, each failing before its fix: the findings refresh reaching the file
  with no verdict filed and its re-arm after a refused submit; the derive/save key split;
  per-metric counts and the thin-measurement drop; a non-JSON payload costing one row; the
  six-case grounding matrix and the size-target-per-root-count pair; the pose editor's
  bound job; a rig in flight blocking a delete and a busy unit surviving `delete_sweep`;
  the generator freeze on `_delete`/`_bake`/`mirror`; a degenerate sweep refused; the
  recipe's optimise tier; the file-list cache and its pruning; the viewer parse/adopt
  hand-over and its stale-result drop; `StripRender` matching `strip` cell for cell; the store staying open after a
  timed-out pool shutdown.
- `tests/manual/test_docs.py` gates the manual edits.
