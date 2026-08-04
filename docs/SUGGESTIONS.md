# SUGGESTIONS.md — 50 improvements for easier, better, faster asset creation

Grounded in a survey of the codebase on 2026-08-02 (app.py, queue.py, pipelines/, static/, recovered
docs/NEXT.md). Items marked with the dormant-roadmap origin where they overlap the deleted NEXT.md.
Effort tags are rough: **S** = hours, **M** = a day or two, **L** = multi-day.

Items marked **[done]** have been implemented (2026-08-02); everything else is
still a review-list entry. The ones left undone are not leftovers — each either
needs a product decision (which tiers to expose, what a collection is, how a
pose editor should feel) or is multi-day work with a real design surface.

**Implemented:** 3, 6, 10, 16, 17, 23, 32, 37, 38, 39, 40, 43, plus both latent
bugs called out in the notes (`Config.mesh_profile` defaulting to a tier that
errors, and the `/optimize`-vs-worker race) and the restored `docs/NEXT.md`.

---

## Faster generation (throughput & latency)

1. **Batch candidates in one diffusers call.** `count=8` becomes 8 separate serial queue jobs
   (`app.py:376-409`); `Text2Image.generate` hardcodes `.images[0]` and never uses
   `num_images_per_prompt` (`text2image.py:312-323`). One batched call would turn 8× ~1.5 s + 8×
   queue overhead into a single pass. **M**

2. **Overlap the two GPU stages in coexist mode.** Both models are resident simultaneously
   (~7 GB SDXL + ~16 GB trellis), yet the worker is strictly serial (`queue.py:127-193`) — a
   2-minute trellis reconstruction blocks a 1.5-second SDXL reference job. Run the next job's SDXL
   stage while trellis reconstructs the current one. **L**

3. **[done] Wake the worker on submit instead of polling.** Dispatch polls `next_queued` every 1.0 s
   (`queue.py:41,180`) — up to a second of dead time per job. An `asyncio.Event` set in
   `create_job` removes it. **S**
   → `Worker.wake()` + `_wait_for_work()`; every route that inserts a queued row calls it.
   The poll interval stays as the backstop. Side effect worth knowing: `test_api.py`'s
   client fixture now patches `next_queued`, as its two siblings already did — it used to
   be *accidentally* safe because dispatch was a second away.

4. **Parse the GLB once per job.** `model.glb`/`source.glb` (~22 MB, ~290 k faces) is fully loaded
   by trimesh 4–5 separate times, all serially on the queue: `optimize._triangles`,
   `normalize_glb`, `meshaudit.load_mesh`, `meshreport.build` (`optimize.py:163`,
   `postprocess.py:203`, `meshaudit.py:93`, `meshreport.py:50`). Load once, pass the scene
   through. **M**

5. **Move the silhouette hole audit off the queue.** `meshaudit.hole_fraction` is a Python/numpy
   rasteriser with two superlinear flood-fill loops (`meshaudit.py:22,219-256`), purely
   diagnostic, and it blocks the next job. Run it after `finish()` in a background thread, and/or
   vectorise the flood fill. **M**

6. **[done] Skip the source triangle count for the `raw` profile.** `optimize.run` pays a full source
   load even when the answer is only recorded as a number (`optimize.py:87-101`). **S**
   → `target_triangles is None` now short-circuits before `_triangles`, and reports the
   counts as `null` rather than guessing. The mesh report measures the finished model a
   step later anyway, so nothing user-visible was lost.

7. **Keep-warm control + warm-up on submit.** After idle eviction the first job pays a ~14 s CUDA
   reload inside trellis stage 1 (`progress.py:128-134`). Preload models the moment a job enters
   the queue; offer a "keep warm" toggle; make the two idle timeouts independently
   configurable. **M**

8. **Fast-preview mesh tier.** Expose a "draft mesh" option combining low resolution with the
   exe's `--box-uv` (explicitly faster than the default xatlas unwrap — the single longest stage)
   and optionally `--no-texture`, for a silhouette check before committing to the full run.
   The flags exist in the exe but `_argv`/the POST never pass them (`trellis.py:71-85,223-225`). **M**

9. **Automatic retries.** No retries exist anywhere: a transient httpx failure from trellis fails
   the job outright (`trellis.py:233`) and OOM produces a friendly message but no retry at lower
   resolution (`errors.py:16-21`). Add bounded retry for transient network errors and one
   step-down retry for OOM. **M**

10. **[done] Cheap wins on the hot paths.** Add the missing index on `created_at` (`db.py:31,162,294`
    sort without one) and stop `/api/health` running the full doctor suite — socket bind
    included — on every call (`app.py:150-165`, `doctor.py:30-47`); cache with a short TTL. **S**
    → `idx_jobs_created` (in `_SCHEMA` and as migration 3, so fresh and existing DBs
    converge) and a 5-second health cache keyed on `trellis_running`, since that is the
    one input whose change flips the port check's answer.

## Queue visibility & control

11. **A real queue view.** Queued jobs render a static "queued" pill with no position, no wait
    estimate (`_attach_progress` fills in only the running job, `app.py:1480-1482`). Show queue
    position and a rough ETA derived from recent per-kind durations. **M**

12. **Priority / "run this next".** Dispatch is FIFO-only (`ORDER BY created_at LIMIT 1`,
    `db.py:294-299`). A priority column plus a reorder control would let a quick reference jump a
    2-minute mesh job. **M**

13. **Pause queue and cancel-all.** Both are currently impossible; cancel is one job at a time via
    a card menu. **S**

14. **Worker fatal recovery.** `worker.fatal` surfaces on `/api/health` but there is no restart
    path — queued jobs sit at `queued` forever (`queue.py`, `app.py:158`). Auto-restart the worker
    task, or at minimum expose a restart action. **M**

15. **SSE (or WebSocket) progress.** The UI polls `/api/progress` every 600 ms and the job list
    every few seconds (`app.js:3231,3406-3408`). A server-sent event stream cuts latency and
    removes the polling load. **M**

16. **[done] Cancel from the progress overlay.** During a run, the only cancel lives inside the job
    card's collapsed "Actions" menu behind a confirm (`app.js:1614-1651`). Put a cancel button on
    the viewport progress card itself. **S**
    → `#ph-cancel`, shown while the overlay narrates a live job. Deliberately without a
    `confirm()`: a blocking dialog would freeze the very progress bar behind it (see 41).

17. **[done] Cost/time preview before submit.** Nothing warns that "Candidates: 8" is eight serial runs
    or that resolution 1536 roughly doubles trellis time. The sheet panel already does exactly
    this arithmetic (`app.js:2970-2975`) — extend the pattern to the submit dock. **S**
    → `#submit-cost`, one line under the validation summary, per mode.

18. **[done] Progress in the tab title.** A backgrounded tab shows nothing until the 45-second
    completion notification (`app.js:1132-1144`); `#ph-progress` is `aria-live="off"`
    (`index.html:898`). **S**
    → `syncTitle()` on every poll. It narrates whatever is *running*, not the selected job:
    a background tab has no selection. The favicon half is not done — it needs a canvas
    favicon generator, which is more machinery than the title deserved.

## Reference iteration (stage 1 — where quality is decided)

19. **Candidate groups.** `POST /api/jobs` returns `{"id", "ids"}` but the client reads only
    `body.id` (`app.js:1006`) and siblings get no `parent_id` (`app.py:404-408`) — asking for 8
    candidates produces 8 unrelated rows and follows one. Link siblings, render them as one
    contact-sheet group with pick-to-promote. **M**

20. **Image-vs-image compare.** Side-by-side compare is gated on `model.glb` (`app.js:1886`) —
    mesh-only — yet choosing between reference images is exactly the decision stage 1 asks the
    user to make. **M**

21. **Reference editing before the expensive run.** Upload is a bare file input
    (`index.html:758`): no crop, rotate, re-centre, and no preview of what `bg_removal` will do
    before paying the 2-minute reconstruction. Even crop + bg-preview alone would save many wasted
    runs. **L**

22. **img2img refinement.** A near-miss reference can only be rerolled from scratch. An img2img
    mode with a strength slider (diffusers supports this on the same pipe) lets the user nudge a
    good candidate instead of gambling on a new seed. **L**

23. **[done] Retry with the same seed.** `POST /api/jobs/{id}/rerun` accepts a `seed` form field
    (`app.py:586`) but `rerunJob()` never sends one (`app.js:1408-1410`). Add "reroll (new seed)"
    vs "retry (same seed)" as distinct actions. **S**
    → `rerunJob(id, how, {seed})`; "Retry with the same seed" on the card menu and first in
    the inspector's actions for a failed job.

24. **User-saved presets.** `guidance.PRESETS` ships fixed presets; there is no way to save the
    current form (prompt skeleton + 12 guidance fields + model/LoRA/weight) as a named user
    preset. localStorage or a small DB table both work. **M**

## Mesh quality & control (stage 2)

25. **Vendor gltfpack and qualify the tiers.** The whole optimize path is built and dormant
    (`pipelines/optimize.py`, doctor check, API) but `vendor/gltfpack/gltfpack.exe` is absent and
    every named tier is unqualified — worse, `Config.mesh_profile` defaults to `"standard"`
    (`config.py:66-69`), a tier that currently *errors* without the binary. Recovered NEXT.md §1
    has the full qualification procedure (chest/sword/rock, verify UVs + PBR maps survive). **M**

26. **Retarget UI for `POST /api/jobs/{id}/optimize`.** The route exists (`app.py:852`) and is
    reachable by hand-POST only — zero references in `app.js`. A triangle-budget control next to
    the downloads row makes re-budgeting a click instead of a curl. **S**

27. **LOD chain export.** `optimize.run` targets a single budget and overwrites `model.glb`.
    Since `source.glb` is never touched, emitting LOD0/1/2 at three budgets is cheap and exactly
    what game imports want. **M**

28. **Expose the mesh seed in 3D mode + seed sweep.** `#seed` lives in the 2D pane, hidden in 3D
    mode (`index.html:69,727-736`); `meshFields()` never sends `mesh_seed` (`app.js:873-885`), so
    promotion always randomises it (`app.py:733`). Surface it, and allow "same reference, N mesh
    seeds" batch runs (`/model` takes no `count` today). **M**

29. **Expose per-job trellis knobs.** The exe offers `--max-tokens`, `--gss`/`--gsh` guidance
    strengths, `--atlas`, `--decim` — none are passed (`trellis.py:71-85,223-225`). An "advanced
    mesh" disclosure unlocks quality/speed trades without a rebuild. **M**

30. **Normal-map baking.** Surface detail currently rides on vertex normals only (README). Baking
    `source.glb` detail onto the optimized mesh's normal map (Blender worker already exists as a
    subprocess pattern) would let low-budget tiers keep visual detail. **L**

31. **GLB structure verification + provenance.** `normalize_glb` rewrites the JSON chunk without
    verifying the one-scene/one-root/no-skin shape it assumes, and records its transform only in
    job params — not in an `asset.extras.warlock` block inside the file (NEXT.md §2.1,
    `postprocess.py:187-237`). **M**

32. **[done] Fix the `/optimize`-vs-worker race.** The route checks only that `source.glb` exists
    (`app.py:872-874`), not job status, and both paths write `set_params` from independently-read
    copies (`app.py:938` vs `queue.py:755,800,872`) — a full-blob last-write-wins lost update
    (`db.py:189`). Require terminal status and add a read-modify-write guard. **M**
    → 409 on a `queued`/`running` job, and `JobStore.merge_params` (read and write under one
    hold of the lock) in place of the stale-copy `set_params`. `/optimize` also takes its
    profile from `Config.mesh_profile` rather than hardcoding `standard`.

## Viewer & inspection

33. **Material/texture inspector.** No material list, no basecolor/normal/roughness channel
    toggles, no UV/atlas view, no texture-resolution readout — `textures.zip` is a blind
    download. **M**

34. **Lighting environments.** Lighting and exposure are hardcoded (`app.js:12-16,68-76`). A few
    switchable presets (studio / outdoor / dark + exposure slider) is how users verify PBR
    textures actually read correctly. **M**

35. **Camera presets and ortho views.** Orbit only; an ortho camera already exists for the sheet
    preview (`app.js:2921`) but is never offered in the viewer. Add front/side/top presets, saved
    viewpoints, fullscreen. **M**

36. **Better shading toggles + scale reference.** Wireframe is a hard replace, not an overlay
    (`app.js:315-323`); no matcap/unlit/normals modes, no grid/axes, no human-silhouette scale
    reference despite `size_m` being a first-class input. **M**

37. **[done] Screenshot button.** `canvas.toBlob` is already used for the internal thumbnail
    (`app.js:2184`) — expose the same capture as a user-facing "save image" for sharing/review. **S**
    → "Save image" in the viewer tool row.

## Errors & trust

38. **[done] Surface full error text and logs.** Job errors are truncated to one ellipsised line with no
    tooltip or copy (`index.html:118-121`, `app.js:1855`); `error.log` is written per job
    (`errors.py:27`) and `trellis.log` exists, but neither is in the `_MEDIA` whitelist
    (`app.py:1225`) so users cannot retrieve a traceback at all. **S**
    → `error.log` joins `_MEDIA` and `_attach_files`; `GET /api/logs/trellis` returns the
    last 64 KB of the shared log (a tail, not a `FileResponse` — the file is unbounded). The
    inspector shows the full sentence plus a lazily-fetched "Full traceback" disclosure, and
    the card's one-liner carries the whole message as its tooltip.

39. **[done] Retry button on failed jobs.** A failed job offers no retry affordance; rerun exists in the
    API. One click, optionally same-seed (see 23). **S**
    → the card's re-run actions now gate on *settled* (done **or** error) rather than done,
    and a failed job's inspector leads with retry-same-seed.

40. **[done] Fix invisible validation errors.** `reportValidation()` can target `#seed` or
    `#negative-prompt` inside the collapsed `<details id="advanced-settings">`; nothing opens the
    disclosure and the hidden-check misses closed `<details>` (`app.js:902-925,921`), so the user
    sees "Fix this field" with no visible field. Auto-open ancestor disclosures/panes before
    focusing. **S**
    → `revealField()` opens every ancestor `<details>` and brings the settings pane forward,
    for *every* flagged field — a message beside an invisible field is as invisible as it is.

41. **Replace the eight blocking `confirm`/`prompt` dialogs.** They freeze the 600 ms poll loop
    and the progress bar behind them (reroll `app.js:1407`, delete `:1624`, discard-edits `:2266`,
    pose delete `:2563`, sheet delete `:3040`, prune `:3210`, rename `:1530`, pose naming
    `:2818`) — the file's own comment at `:1301` gives this exact reason for avoiding
    `alert()`. **M**

42. **Soft-delete with an undo window.** Delete removes the job directory immediately
    (`app.py:1211`) and prune deletes N jobs behind one confirm (`app.py:437`); pose delete and
    "apply joints" are likewise irreversible. A trash/grace-period pattern makes destructive
    actions survivable. **M**

43. **[done] Show an ETA on cold starts.** The ETA is suppressed until `percent >= 10` and `!live.cold`
    (`app.js:3370`) — i.e. exactly during the slowest, most confusing case (the ~8 GB CUDA
    load). Even "warming up, typically ~15 s" beats silence. **S**
    → the cold branch says so instead of going quiet. Not an estimate of the job — an
    explanation of the silence, which is the thing the suppression was hiding.

## Library & organization

44. **Server-side pagination, filtering, sorting.** `/api/jobs` has no offset/filter/sort
    (`app.py:418-425`) despite indexed `favorite` and normalized `tags` columns (`db.py:56-58`);
    the client filters over the full list, which carries every job's full `params` including
    `mesh_report`. Slim the list payload too. **M**

45. **Grid/contact-sheet library view.** Thumbnails exist, but the library is a single list with
    no sort control, no select-all, no shift-range selection (`app.js:1541-1546`). A thumbnail
    grid is the natural way to scan candidates and past work. **M**

46. **Lineage view.** `parent_id` is recorded and `db.children()` exists but is dead code
    (`db.py:141`). Show reference → mesh → rig → sheets as a small tree in the inspector so
    related assets stop looking like unrelated rows. **M**

47. **Collections and tag chips.** Tags are a raw comma string filterable only through free text
    (`index.html:923`, `app.js:1380-1382`); there are no projects/collections. Tag chips with
    autocomplete + a collection grouping cover most "where did my sword set go" needs. **M**

## Rig, pose & sheet

48. **Pose editor depth.** FK only, one joint at a time, no numeric per-axis entry, no symmetric
    editing (mirror is a one-shot button, `app.js:2788-2802`), no simple IK for limbs. Numeric
    entry + live-mirrored editing are the cheap 80%. **L**

49. **Cross-asset pose reuse.** Shipped preset poses already apply to every rig of the same
    template (`app.py:766-775`); user-saved poses are trapped in one job's `poses/` dir
    (`rigging.pose_path`). Let a saved pose be applied to any asset with the same template. **M**

50. **Resizable workspace panes.** Side panes are fixed at 320 px with `overflow:hidden`
    (`index.html:369`); between 901–1199 px the inspector becomes an overlay drawer, so on a
    typical laptop the model and its controls cannot be seen at once (`index.html:545-561`).
    Drag-to-resize plus a collapsible settings pane. **M**

---

## Notes for review

- **[done]** Items 25, 26, 31 and the process-tree-kill/capability-health ideas come from the deleted
  `docs/NEXT.md` (recoverable via `git show 3c718e5:docs/NEXT.md`). CLAUDE.md:25,54 and
  `index.html:780` still reference that file — either restore it or update the references.
  → restored from `3c718e5`; all three references are live again. It stays what it says it
  is: forward planning, not a description of the app.
- **[done]** Two latent bugs surfaced during the survey that are worth fixing regardless of this list:
  `Config.mesh_profile` defaults to a tier that errors without gltfpack (item 25), and the
  `/optimize` route races the worker (item 32).
  → the default is now `raw`, which is also the only tier the UI offers; the race is item 32
  above. Still open, because renaming a shipped field is a compatibility call rather than a
  fix: `/api/sheets/options` returns `yaws` as a list of angles while `POST /sheets` takes
  `yaws` as an integer count (`app.py:1067,1091`) — same name, two types.
- Highest leverage per effort, if you want a shortlist: **1, 3, 16, 17, 19, 23, 26, 38, 40, 43**
  (all S/M, all directly felt in the daily loop). Of these, 1, 19 and 26 remain: batching
  candidates changes what a "job" is, candidate groups need a grouping model (`parent_id` is
  already spoken for by promotion), and a retarget UI would expose tiers that CLAUDE.md says
  stay hidden until they are qualified against a chest, a sword and a rock.
