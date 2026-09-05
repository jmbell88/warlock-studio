# Defect classes — what this codebase has grown before

Paste the checklist (from the rule to the end) whole into every explorer brief. Each
item is a kind of defect a previous audit found more than once, with the instance that
named it and the guard that now exists. The **theme key** in brackets is what
`/warlock-audit <theme>` accepts.

The rule for using it: walk every item against every file in your slice. Where a guard
exists, check whether it *covers this file*; a guard written for eight doors does not
cover the ninth. Where no guard exists and you find the defect, `suggested_fix` names the
guard as well as the fix.

---

1. **[undo] One gesture, one undo step, and every edit has one.** A slider, drag or
   colour-picker door that pushes history once per value-change frame (T1, 2026-09-02;
   M06, 2026-09-04), and its mirror: a live edit that is never recorded at all, or only
   on a release that may never render (M05). Guard: `controls.fold_undo` (draw, fold,
   act); `tests/test_undo_gesture_doors.py`. Look for `is_item_deactivated_after_edit`
   as the only recorder, and for `push`/`record` inside a per-frame `changed:` branch.

2. **[frame-thread] Nothing blocks the frame loop.** A decode, encode, file write,
   sqlite query, subprocess wait or numpy pass on the pygame thread (T2). Guard:
   `TaskRunner.submit` with a frame-thread adopt; `tests/test_frame_thread_doors.py`.
   Look for `open(`, `Image.open`, `np.load`, `json.load`, `subprocess`, `.result()` in
   `draw` or `handle_key` paths.

3. **[task-thread] Task threads publish, they do not write.** A task half that sets UI
   state, tab attributes or module dictionaries directly (T3). Guard: results handed
   back through `Done.result` and adopted on the frame thread;
   `tests/test_task_thread_writes.py`. Look for `ctx.`/`tab.`/`state.` assignments
   inside functions submitted to `TaskRunner`.

4. **[greyed] A greyed control says why, truthfully.** `disabled=True` with an empty,
   stale or wrong reason, or a reason chosen inline where it cannot be tested (T4).
   Guard: a pure `*_reason` function beside the draw call;
   `tests/test_findings_themes.py`. The exercise driver reports these as
   `disabled-no-reason`.

5. **[docs-vs-code] The manual, INVARIANTS and docstrings describe the code that
   exists.** A sentence the code contradicts (T5: "refuses" where it silently accepts;
   "no dialog" where one opens; "lazy import" where it is eager), and the reverse: a
   behaviour the manual never mentions. Guard: none general; `tests/manual/` covers
   structure, not truth. Compare each promise in the chapter against the handler.

6. **[per-frame] Nothing is recomputed per frame that nobody reads.** A full pass over
   a mesh, a directory listing, a job-table query or a layout solve inside `draw` with
   no memo key (T6). Guard: memoise on a generation counter (`jobs_cache`,
   `AppState.frame_index`), never on wall time.

7. **[layering] Imports go one way and headless packages stay headless.** A `studio/`
   editor package (`inker`, `clay`, `plotter`, `packwright`, `sirens`, `troupe`, `muse`,
   `tour`) importing imgui, moderngl, pygame or `service`; `sirens`/`muse` importing
   scipy; a pane importing another pane's internals; `queue.py` importing `service`;
   anything but `blender_worker` importing `bpy` (T7). Guard: an import-pin test per
   package; a `_MOVED` table for every relocated name. Report an import a pin test does
   not cover, not one it already refuses.

8. **[blind-spots] The decidable half of every pane is tested.** A pure helper, cache
   key, predicate or piece of arithmetic in a pane with no test (T8). Panes cannot be
   driven headlessly; their *decisions* can. Guard: `tests/test_findings_blind_spots.py`
   names the modules covered so far.

9. **[staged-writes] Every write onto a served name is staged and replaced.** A write
   in place onto a file something may be reading; a temp name shared by two concurrent
   writers (M03: `.name.tmp`); a sidecar left stale beside a replaced artifact (M12).
   Guard: `studio/atomic.py`; `tests/test_atomic_writes.py`. Look for `open(path, "w")`
   on a path the app serves, and for exports that write more than one file.

10. **[ceilings] A refusal comes before the allocation, and per-item bounds are
    counted.** A ceiling per accessor with none per document (H01); a cache bounded by
    count when item size is an argument; a number read from a file deciding an
    allocation before it is checked (INVARIANTS 334–346). Guard:
    `tests/test_resource_ceilings.py`, `tests/test_gltf_loader.py`. Look for loops that
    allocate from a count the file supplied.

11. **[sweep-tables] A new kind is a sweep, and every prefix list is complete.** A job
    kind missing from one stage-keyed table; a task prefix missing from a guard list
    (H02: `pack:` absent from the quit protection); a recents kind with no opener; an
    `on_task_done` chain that lets a key fall through unreported (T7 found two). Guard:
    the stage-keyed sweep test, `_TASK_HANDLERS` prefix tables, INVARIANTS 336. Look
    for hand-written lists of kinds or prefixes and check each against its source of
    truth.

12. **[presence-checks] Present is not usable.** `exists()` where `is_file()` is meant
    (L01); a zero-byte weight reported healthy (M04); `find_spec` standing in for an
    import (M01); "installed" with no probe. Guard: `doctor.py`'s suspect-file check,
    `pack_worker`'s import probe. Look for every path check that gates a "ready" verdict.

13. **[stale-completion] A late result cannot override a newer intent.** A decode or
    load whose completion adopts unconditionally, with no request token, so an older
    request landing last wins (M11); a Stop that does not invalidate what is in flight.
    Guard: a monotonically increasing request id compared on adopt. Look for every
    `on_task_done` that replaces a player, document or preview.

14. **[vocabulary] Closed vocabularies stay closed.** A toast level outside
    `TOAST_LEVELS` (falls back to `info` silently); a refusal without a `field`; a mode
    or stage name spelled by hand rather than read from `modes.KEYS`; a shortcut not in
    `shortcuts.py`. Guard: `tests/test_ux_todo_fixes.py`, `tests/test_field_error_wiring.py`,
    `tests/manual/test_shortcuts.py`.

15. **[subprocess] Every child is in the job, and nothing reaches the network.** A
    `subprocess.Popen` outside the `winjob` kill-on-close wrapper; anything that could
    open a socket besides `fetch_worker` and `pack_worker`; a stdin reader that leaves a
    read pending (the t2i deadlock). Guard: the winjob scan test,
    `tests/test_offline.py`. Look for `Popen`, `urllib`, `requests`, `huggingface_hub`.

16. **[inventories] Counts and lists agree with their source of truth.** A mode
    inventory in README or the manual that disagrees with `modes.KEYS` (L04); an extras
    list that disagrees with `pyproject.toml` (L05); a test count; a "six workspaces"
    sentence; a notices table that disagrees with `docs/MODELS.md` (L06); a security
    policy that names one network exception when there are two (L07). Guard:
    `tests/test_findings_followups.py` (workspace-count words). For the `docs` slice this
    item is expanded in `docs-checkup.md`.

17. **[cancel] Cancel means stopped, and the next job is clean.** A cancel that lets a
    child run to completion (the 2026-09-03 Blender cancel); a cancel token minted by a
    cleanup that deletes served names (INVARIANTS 120); a quit path that bypasses a
    "cannot be interrupted" phase (H02). Guard: `tests/test_failure_paths.py`,
    `tests/test_job_durability.py`. Look at every cancel handler and every quit route.

18. **[recovery] Every authored thing survives a crash the same way.** A document kind
    with no journal provider, or a journal write whose `.meta.json` sidecar is not the
    last thing written (INVARIANTS 489–503). Guard: `tests/test_journal.py`. Look for a
    new document kind and check `journal.ensure_providers` knows it.
