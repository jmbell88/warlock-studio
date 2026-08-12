# Warlock Studio — Full Audit Report

**Date:** 2026-08-11 · **Tree audited:** HEAD `de87838` (subject "Warlock v0.0.21", working tree clean)
**Method:** six parallel read-only auditors with disjoint scopes — (1) model/LoRA/VRAM lifecycle, (2) concurrency/DB/job lifecycle, (3) service layer & pipelines, (4) studio UI/UX, (5) stubs/dead code/hygiene, (6) documentation consistency. ~102k lines of `src/warlock` Python, 255 test files, all of `docs/`. Nothing was executed except `uv run ruff check .` (clean); pytest was not run — the two test-failure claims below are from static reading of the assertions. Overlapping findings were cross-checked and merged; each finding carries the confidence the evidence supports.  
**Status (2026-08-12):** superseded by `MASTER_AUDIT.md`, which merges this report with `AUDIT2.md` and records what was actually fixed. Read the findings here as evidence about `de87838`, not as a description of the current tree.

---

## Executive summary

The codebase is in unusually good shape. The hygiene sweep found **zero dead code** (all 1,702 top-level definitions have callers), **zero stale FIXME/HACK markers**, no stub bodies, a clean ruff run, and every big invariant it could check statically — staged writes, offline integrity, `bpy` containment, winjob coverage, lock discipline, headless-package purity, texture lifecycle, uid-addressed undo — **verified holding**.

The problems cluster in three places:

1. **The HEAD commit `de87838` shipped a broken release state.** It bumped `pyproject.toml` to 0.0.22 but left `CHANGELOG.md` and `warlock.__version__` at 0.0.21 (the repo's own lockstep tests failed on that tree, statically), and it deleted `docs/TODO.md` while CLAUDE.md, INVARIANTS.md, and a link-checker source list still named that file as "the only roadmap." *(Fixed 2026-08-12: pyproject reverted to 0.0.21 with `scripts/preflight.py` gating the lockstep; the roadmap deletion is now stated in CLAUDE.md and INVARIANTS.md.)*
2. **The 2026-08-11 base-model default change didn't propagate to the authoritative docs.** INVARIANTS.md, CLAUDE.md's pipeline one-liner, and `queue.py`'s module docstring all still say SDXL-Turbo is the default; code moved it to `sdxl_cfg`.
3. **Model/LoRA lifecycle: safe, with two real defects.** Nothing corrupts state, leaks unboundedly, or crashes the app — but a warm pipe causes spurious VRAM refusals for three job kinds, and a LoRA installed while the pipe is resident is silently dropped while the job's records claim it ran.

**Totals: 0 critical · 4 high · 15 medium · ~26 low.**

---

## The headline question: is model/LoRA loading/unloading safe?

**Verdict: structurally safe — no path found that corrupts state, permanently leaks VRAM, or crashes the process. The defects are wrong-answer and wasted-work bugs, not safety failures.**

What was verified to hold (each pinned by existing tests or traced end-to-end in code):

- **Transactional load.** `Text2Image.load` configures on a local and publishes `self._pipe` as its last statement; every failure point (from_pretrained, scheduler, placement, LoRA) unwinds through `del pipe` + `_reclaim()`.
- **The reference-drop doctrine.** `_reclaim` takes no pipe reference; `unload()` drops `self._pipe` first; `_evict_t2i`, the base-switch path, `_release_t2i`, and `shutdown` all drop or bypass their references before collecting. Every unload/trellis-stop runs off the loop thread.
- **Handoff.** `_needs_handoff` has one owner, consulted from the single `_acquire_t2i` preamble all four t2i stages share; offloaded specs hand off unconditionally; `unload()` vs `trim()` is used exactly as the invariant states; `vram.estimate` prices the offload sequentially so the gate and the enforcement agree.
- **Trellis subprocess.** `stop()` waits after kill and *keeps* the handle on failure (`TrellisStopFailed`), so nothing loads into unreleased VRAM; spawn → `winjob.assign` immediately; port reclaim kills only a provably-ours orphan; respawn backoff present; GLB responses are size-capped, validated, staged-then-renamed.
- **LoRA state machine on the cached-pipe path.** Adapters and weights are re-applied on **every** `generate` call, so same-base-different-LoRA jobs cannot inherit each other's state; adapter-name collisions are structurally prevented and tested; applied-not-requested is what gets recorded by `Text2Image._recipe`.
- **Admission.** Every door (create, rerun both modes, promotion, pixel sheet, sprite synthesis, retexture, sweeps per-unit, bench via `service.jobs.create_job`) takes the VRAM and/or weights checks; host commit is re-checked at dispatch and at the image→mesh boundary.
- **Cancel windows.** Mid-sample (step callback), mid-trellis (kill, with phase restored across remesh retries), between-load-and-sample — all checked; cancelled jobs publish nothing.

What does not hold — the four lifecycle findings (details in the severity sections below): the text-only dispatch VRAM credit (**H3**), the fetch-while-resident LoRA drop (**M1**), the retexture family-check gap (**M4**), and three bounded LOW leak/shutdown windows (shutdown racing a slow load, exception tracebacks pinning the conditioning stack, `vram_gib` excluding eagerly-loaded adapters).

---

## High findings

### H1. Release lockstep is broken at HEAD: pyproject 0.0.22 vs CHANGELOG/`__version__` 0.0.21
**bug / packaging · confidence high · found independently by two auditors**
- `pyproject.toml:3` = `0.0.22` (bumped in `de87838`); `CHANGELOG.md:9` top heading = `## 0.0.21 — 2026-08-11`; `src/warlock/__init__.py` `__version__ = "0.0.21"`.
- `tests/test_changelog.py:90` asserts the changelog's top release equals the pyproject version — a literal string comparison that **fails on this tree**. `tests/test_offline.py:46` pins `warlock.__version__ == version("warlock")` and fails once the editable install's metadata reflects 0.0.22 (i.e., after the next `uv sync`).
- User-visible: Home would print 0.0.22 beside the title while "What's new" falls back to the 0.0.21 entry — the exact failure the changelog test's docstring describes. The commit subject (`Warlock v0.0.21`) also drifts from the manifest it bumped.
- Also: the comment above `__version__` names the wrong pinning test ("tests/test_models.py"; it's `tests/test_offline.py`).
- **Fix:** either write the `## 0.0.22` CHANGELOG entry and bump `__version__`, or revert the pyproject bump — whichever matches intent; correct the comment.

### H2. `docs/TODO.md` was deleted, but four places still declare it the live roadmap
**doc-conflict · confidence high · found independently by two auditors**
- `de87838` deleted `docs/TODO.md` (186 lines). Still pointing at it in the present tense: `CLAUDE.md:31` ("**`docs/TODO.md` is the only roadmap.**"), `docs/INVARIANTS.md:218` ("The live roadmap, and the only one…"), `tests/test_ux_todo_fixes.py:1` (docstring), and `tests/test_external_doc_links.py:43` (`SOURCES` still lists it — silently skipped by the `if not source.exists(): continue` at line 101, which is exactly the silent-shrink failure that file's own comment warns about).
- Risk: the next session follows CLAUDE.md to a roadmap that doesn't exist — or "helpfully" recreates a TODO.md, re-legitimising `§N` citations that are defined to resolve against the *deleted* LEFTOVERS.md.
- **Fix:** update CLAUDE.md and INVARIANTS.md to say the roadmap file was deleted 2026-08-11 (git history keeps it; `§N` resolves via `git log --diff-filter=D`); fix the test docstring; drop or annotate the `SOURCES` entry.

### H3. Dispatch-time VRAM credit only covers `kind == "text"` — pixel-sheet, sprite-synthesis, and retexture double-charge the resident pipe
**bug · `src/warlock/queue.py:998-1013` with `src/warlock/vram.py:173-196` · confidence medium-high**
- `_check_resources` credits the resident pipe's measured reserved memory back against the estimate **only when `job["kind"] == "text"`**. `vram.estimate` prices `pixel_sheet`, `sprite_synthesis`, and `retexture` with a full image-model term (+ ControlNet, + IP encoder, + `TRELLIS_GIB` under coexist).
- On the flagship 32 GiB coexist card, the *natural* flow — generate a 2D reference (leaves a ~7.5 GiB pipe warm; the coexist path deliberately keeps it), then start a sprite synthesis — computes `need ≈ 26.7` vs `free ≈ 23.5` and refuses at dispatch with "close other GPU applications", even though the job would **reuse** the very pipe being counted against it (true incremental need ≈ 3.7 GiB). Pixel sheets refuse by ~2.5 GiB; retexture is knife-edge. Waiting out the 600 s idle eviction "fixes" it, which presents as a phantom VRAM leak.
- This is the exact defect class the credit was added to fix for text jobs; the fix never reached the three later kinds. Fails safe (refusal, not crash), but refuses a supported workflow on default hardware. Headless tests skip this branch entirely (`device_memory()` is None), which is why no test caught it.
- **Fix:** widen the credit gate to every kind whose estimate carries an image-model term — or better, derive the credit from `vram.estimate` itself rather than a kind list. (Confidence note: WDDM free-VRAM readings are virtualized, so the refusal may be intermittent rather than deterministic.)

### H4. "SDXL-Turbo is the default" survives in the authoritative docs; code moved the default to `sdxl_cfg`
**stale-doc · confidence high**
- Code: `src/warlock/models.py:38` `DEFAULT_BASE_MODEL = "sdxl_cfg"`, citing `docs/measurements/2026-08-11-default-base-model.md`. README, MODELS.md, and manual chapters 03/16 all agree.
- Against that: `docs/INVARIANTS.md:214` ("SDXL-Turbo … is the default because FLUX is gated"), `INVARIANTS.md:7` and CLAUDE.md's "What this is" pipeline one-liner ("text or image prompt → SDXL-Turbo → …"), `INVARIANTS.md:25` ("Coexist: SDXL-Turbo (~7 GB…)"), and `src/warlock/queue.py:3`'s module docstring.
- INVARIANTS.md instructs readers to consult it before modifying a subsystem — and hands them the pre-2026-08-11 answer with a wrong causal story ("because FLUX is gated" was never why `sdxl_cfg` won).
- **Fix:** reword the four spots to name SDXL 1.0 full CFG as the default with turbo as the fast option.

---

## Medium findings

### Model / LoRA lifecycle

**M1. A LoRA installed while the base pipe is resident never loads — and the records claim it ran.** bug · `src/warlock/pipelines/text2image.py:298-349, 369-376`, `src/warlock/service/downloads.py:170-206`, `src/warlock/_q_sprite.py:97-112, 218-228` · confidence high.
`_load_loras` runs once inside `load()`; `_adapters` is frozen for the pipe's life. The in-app model install (`service.downloads.download`) never evicts or reloads the pipe. Flow: generate once on `sdxl_cfg` (pipe stays warm ≤ 600 s), install `pixelxl` in Settings, submit a styled job → `check_weights` passes (file on disk), the cached pipe is reused, and `_apply_adapters` drops the adapter with a *wrong* "not downloaded" warning. Consequences: the output silently lacks the style (the outcome `check_weights` exists to prevent); `params["style_lora"]` is in the `VECTOR_PARAMS` allowlist, so the findings corpus records a style that never ran; the pixel-sheet sidecar writes `recipe["style_lora"]` from the *request* — resurrecting verbatim the "sidecar naming a LoRA that never loaded" bug the 2026-08-11 door work closed. **Fix:** on a successful fetch of a `lora:` row (or a base's own `base_lora`), evict the resident pipe or re-run `_load_loras` on it; alternatively let `_apply_adapters` attempt a late `load_lora_weights` when the key is registry-valid, family-fitting, and now on disk.

**M4. Retexture door has no family check — a FLUX/klein base passes admission and dies mid-job.** gap · `src/warlock/service/_jobs_rework.py:206-208` vs `src/warlock/pipelines/text2image.py:414-421` · confidence high (mechanism), medium (reachability from the shipped pane).
A retexture is an img2img pass and `Text2Image._conditioned` refuses non-SDXL families **at runtime** — but `retexture_job` checks only that the base key exists in the registry, and `_retexture` resolves the base through the general chain (an unset `base_model` on a host whose `WARLOCK_T2I_MODEL` names a klein entry also qualifies). The job queues cleanly, renders all six Blender views, stops trellis (offload handoff is unconditional), loads ~16 GiB of host commit, then errors on the first restyle pass — minutes of work plus a trellis restart, violating the stated door principle ("every refusal is still *here* rather than in the worker"). `rerun_job --reroll` repeats it. **Fix:** refuse non-`FAMILY_SDXL` bases in `retexture_job` and its rerun path, the way `create_pixel_sheet` refuses the pairing.

### Concurrency / job lifecycle

**M2. `optimize_job`/`retexture_job` never consult `dependent_jobs()` — a retarget races a running re-texture or rig on the same mesh.** bug · `src/warlock/service/_jobs_rework.py:68-70`, `src/warlock/_q_sprite.py:655-664` · confidence high (code), medium (frequency).
Both doors refuse only on the *target job's own* status, but a retexture/rig/sheet is a separate row writing into that done job's directory. Queue a re-texture for done mesh J, then retarget J inline (allowed — rewrites `model.glb` from `source.glb`): when `_retexture` finishes, its `os.replace` publishes a skin baked from the **pre-retarget** geometry over the retargeted mesh, silently reverting the triangle budget while `params["profile"]` claims the tier ran. A rig in flight finalizes a skeleton bound to the old mesh, and `stale_rig_artifacts` only reports files that already exist, so no warning. The per-artifact locks don't cover this (`optimize_job` locks `(J,"optimize")`; the re-texture's replace takes no lock). **Fix:** raise `Conflict` when `dependent_jobs(svc, job_id)` is non-empty — the helper exists and `_jobs_lifecycle.py`'s docstring names it as the answer to exactly this shape.

**M3. Pixel-sheet and re-texture jobs drive the progress bar to 100 % during the first sampling pass.** bug · `src/warlock/_q_sprite.py:170-171, 590-591`, `src/warlock/progress.py:88-98` · confidence high.
Both kinds hand the pipeline the generic `_t2i_state`/`_t2i_step` callbacks, but `_PHASES_BY_KIND` has no entry for them — `phases_for` falls back to `PHASES_IMAGE` (no `t2i_load`/`t2i_sample`), and `ProgressBus.update` maps an unknown phase onto the whole bar. The last sampling step of the *first* band/view emits 100 %, and the never-regress creep pins it there for the rest of a multi-minute job. This is the exact trap INVARIANTS.md documents for new job kinds; `_sprite_synthesis` explicitly routes around it while its two siblings walked into the mirror-image failure. No test covers these kinds' phases. **Fix:** add `PHASES_PIXEL_SHEET`/`PHASES_RETEXTURE` tables (or a `_sprite_step`-style per-phase mapper) and pin them in `tests/test_progress.py`.

**M5. `apply_library_pose` bypasses the pose-cap lock its sibling exists to hold.** bug · `src/warlock/service/poses.py:220-233` vs `service/rig.py:148-155` · confidence high (code), medium (impact).
`rig.save_pose` wraps the check-then-write of `MAX_POSES` in `svc.convert_lock(job_id, "poses")` with a comment explaining why; `apply_library_pose` performs the identical check-then-write with no lock. Since a lock only excludes writers that take it, this voids the sibling's guarantee too: an apply racing a save both read `MAX_POSES − 1` and both write. **Fix:** wrap the cap check + `rigging.save_pose` in the same `convert_lock`.

**M6. Home-migration's "no live writer" lock is released before the multi-minute copy begins.** hardening · `src/warlock/migrate.py:159-173, 192-224` · confidence high (mechanism), low (likelihood).
`BEGIN EXCLUSIVE` / `ROLLBACK` proves nothing is live *at that instant*, then the lock is dropped before `_move` starts the cross-volume `copytree` (the real migration is ~95 GB). A second Warlock launched mid-copy passes its own precondition, writes into legacy `assets/`, and the first process's post-verify `rmtree(legacy)` deletes those writes. The recount catches added files; an equal-size in-place modification is invisible. Windows sharing semantics blunt the worst case to a silently split library rather than data loss. **Fix:** hold the exclusive connection open across `_move` (close in a `finally` after the deletes), or re-take it immediately before each `rmtree`.

**M7. `uninstall`'s live-jobs guard: wrong read, check-then-act, and an unload that can land under a running generate.** hardening · `src/warlock/service/downloads.py:267-291` · confidence medium-high · **flagged independently by three auditors**.
(a) It filters `store.list(limit=MAX_LIST_LIMIT)` — a 5,000-row page — instead of the purpose-built, deliberately unbounded `store.active_jobs()`, so a queued row older than the page is invisible and its weights get deleted. (b) The check is a snapshot: a `create_job` in the gap gets claimed, and `unload_text2image` then runs `pipe.unload()` concurrently with a `generate()` in flight on that pipe (the loop is free while `_generate` awaits `to_thread`); the weights deletion under a reading pipe at least fails loud (`PermissionError` → `Failed`). **Fix:** use `active_jobs()`, and re-verify `worker.current_job_id is None` (or re-run the live check) inside the same loop-side callable that unloads.

### Studio UI

**M8. Restore, purge, and empty-trash results never refresh the job cache or the storage figure.** bug · `src/warlock/studio/main.py:1083`, `panes/library.py:1049-1059, 1290` · confidence high.
`_on_task_done` invalidates on `delete:/prune/rename:/name:/tags:/fav:` only; `restore:{id}`, `purge:{id}`, and `"empty-trash"` match nothing. So the toast-Undo and trash-Restore look inert for up to the 3 s idle backstop — and permanent purge and Empty trash, the two actions that actually free disk, never re-measure, so the "N jobs – X GB" footer keeps the pre-delete figure for the session (trashing, which frees nothing, *does* re-measure). **Fix:** add `restore:`/`purge:` to the invalidate prefixes and `purge:`/`empty-trash` to the storage re-measure.

**M9. Ctrl+K and Ctrl+Enter read `pygame.key.get_mods()` instead of `event.mod` — the exact hazard the same file documents.** bug · `src/warlock/studio/main.py:1627-1631, 1735` · confidence high (inconsistency), medium (frequency).
`_passes_text_field` (lines 1541-1547) reads `event.mod` and its docstring states the rule: events are drained in a batch, so a modifier released between press and processing makes `get_mods()` lie and the shortcut is silently dropped — "only when the typist was fast." `_shortcut` still uses `get_mods()`: a fast Ctrl+K in Inker falls through to bare `k`, which is the **Rect tool** — palette fails to open *and* the active tool changes; a fast Ctrl+Enter submit is silently dropped. **Fix:** use `event.mod`, as `_passes_text_field` and `review_mode.handle_key` already do.

### Documentation

**M10. Download-root spelling split after the `~/.warlock` home move.** doc-conflict · confidence high (conflict), medium (severity).
`README.md:59-64` now says `--local-dir $HOME/.warlock/models/...`; `docs/MODELS.md`, manual `15-installation.md:75-82`, and `models.py:155`'s `download_text` (which doctor's paste-able remedies emit) still say relative `models/...`. The code default is `~/.warlock/models` (`config.py:211-212`). A command pasted from MODELS.md/ch. 15 lands weights in `<cwd>/models/`, rescued only by the one-time migration — which skips a root "if the destination already holds something," so on any host with a non-empty `~/.warlock/models` (e.g. after one in-app fetch) the pasted download is stranded and doctor keeps reporting it missing. `INVARIANTS.md:23` is also now wrong about itself ("the literal `--local-dir models/...` is **the README's spelling**" — the README no longer spells it that way). **Fix:** standardize on the explicit `$HOME/.warlock/models` spelling in MODELS.md, ch. 15, and `download_text`; update INVARIANTS.md:23.

**M11. Manual index and the in-app loader disagree about which part chapter 14 belongs to — and the manual gate doesn't cover grouping.** doc-conflict · `docs/manual/00-index.md:24` vs `src/warlock/studio/manual/loader.py:27-31` · confidence high.
The index lists Keyboard shortcuts under "Using Warlock Studio" (Part I); `PARTS` puts `range(14, 19)` under "Setup & operations," so the in-app contents shows it under the other heading. CLAUDE.md claims the number decides order *and part* and that `tests/manual/` gates both directions — but `test_index_links_every_chapter` only asserts linkage, not grouping, which is exactly how this drifted. **Fix:** align `PARTS` or the index; extend `tests/manual/test_docs.py` to assert index-section membership against `PARTS`.

**M12. INVARIANTS.md names the live `src/warlock/sweep.py` as deleted.** doc-conflict · `docs/INVARIANTS.md:217` · confidence high.
The bench bullet says "The parameter-sweep half (`src/warlock/sweep.py`, `verdicts.py`, `report.py`, …) is gone" — but the files actually deleted (commit `1fc1573`, v0.0.8) were `src/warlock/bench/sweep.py`, `bench/report.py`, `bench/verdicts.py`. Today's `src/warlock/sweep.py` is the trellis `--band` sweep: CLI-wired (`cli.py:48,68`) and tested (`tests/test_sweep.py`). The authoritative doc names a live module as deleted — an invitation for a future cleanup pass to delete it. **Fix:** correct the paths (and disambiguate from the live `service/verdicts.py`).

### UX (what would make the app easier)

**M13. A multi-gigabyte model download has no cancel.** ux · `src/warlock/studio/panes/app_settings.py:489-521` · confidence high.
Progress is wired per-row, and the concurrency policy disables every *other* row while a fetch runs — but there is no Cancel control anywhere in the pane. A mistaken 16 GB SDXL fetch on a slow line can only be stopped by quitting the app. The kill-on-close job already reaps the child on exit, and staging means a cancel leaves no half-installed model — the safe mechanism exists; only the button is missing. **Fix:** a Cancel beside the progress bar that kills the fetch child.

**M14. The manual's "full list" of shortcuts is missing Clay bindings that exist — and the in-app popup hides Clay's Ctrl+W.** ux/doc · `src/warlock/studio/clay_mode.py:607, 756-778` vs `docs/manual/14-shortcuts.md:100-121`; popup rows `main.py:3541` vs `3556, 3580, 3596` · confidence high.
Clay binds Ctrl+1/3/7 (±Shift = opposite axis view), Ctrl+5 (orthographic toggle), and Ctrl+W (close tab, matching its three sibling modes); the manual's Clay table has none of them despite the chapter opening with "the tables below are the full list," and the popup lists "Ctrl+N / O" for Clay where the siblings say "Ctrl+N / O / W." Six axis views are effectively undiscoverable. **Fix:** add the rows to the manual's Clay table (the `tests/manual` gate covers numbering, not table contents) and make the popup rows uniform.

**M15. No bulk retry for failed jobs — the one bulk action the failures affordance sets you up for.** ux · `src/warlock/studio/panes/library.py:484-505, 1065-1135` · confidence high (gap), medium (priority).
The library deliberately surfaces "N jobs failed – show" (the stated overnight-batch use case) and Select-all ticks everything shown — but the bulk bar offers only Export zip / Save to project / Delete. Having jumped to the failures and selected them, the user must open each card and press "Try again" one at a time; `run_action(..., "retry")` already exists per job. **Fix:** when the ticked set contains failed jobs, add "Try again (N)" looping the existing action.

---

## Low findings

### Model / VRAM lifecycle
- **L1. Shutdown can strand a fully-loaded pipe when a load outlives `SHUTDOWN_TIMEOUT`.** `queue.py:704-731`, `text2image.py:241-296`. Cancelling the worker task doesn't stop the `to_thread` loader (no interruption point inside `load()`, and a cold 16 GB klein load can exceed 20 s); the final `if …loaded: unload()` guard then reads `loaded == False` an instant before the loader publishes the pipe — up to ~16 GiB of host commit stays resident for process life on paths where the interpreter survives. Fix: join the outstanding load before the `.loaded` check, or have `load()` re-check a stop flag before publishing. *Confidence medium — timing-dependent and rare.*
- **L2. On a failed conditioned job, the exception traceback pins the ControlNet/IP-encoder stack past every reclaim point.** `text2image.py:434-447, 575-577, 794-796`, `queue.py:1065-1076`. The `gc.collect()`/`empty_cache()` passes run while the propagating exception still references the frames holding the conditioning tensors; ~2.5–3.7 GiB returns to the allocator pool only when `_process`'s except block ends, and to the driver only at the next job's trim/unload. Bounded and self-correcting, but a real gap in the drop-every-reference doctrine on the error path. Fix: strip `exc.__traceback__` (the codebase already does this in `_on_task_done`) and run one pool trim in the error branch.
- **L3. `BaseModel.vram_gib` excludes the adapters `_load_loras` eagerly places on-device.** `models.py:222-225`, `text2image.py:271-344`. The base distillation LoRA (~0.8 GB) plus every fitting style LoRA on disk load into device memory unconditionally; the "deliberately conservative" 7.0 GiB figure under-prices by up to ~1.5 GiB — the direction of error the registry's own comment forbids. Fix: fold a per-family adapter allowance into the estimate, or document the exclusion beside `vram_gib`.

### Concurrency / DB
- **L4. `_stage_link`'s fallback writes served GLB names in place on filesystems without hard links.** `queue.py:99-108` (used by `_q_generate.py:621-625`'s remesh-restore). Where `os.link` fails wholesale (exFAT/network `WARLOCK_DATA_DIR` — exactly the case the fallback exists for), `shutil.copyfile` truncates the served `model.glb` in place; a crash mid-copy leaves a torn file on a job about to be `done`. The adjacent docstring claims `os.replace` semantics that only the link branch has. Fix: copy to a temp sibling + `os.replace`, mirroring `optimize.staged_copy`.
- **L5. `trash_job`'s claimed-in-the-gap comment describes behavior the code doesn't have.** `service/_jobs_lifecycle.py:309-319`. If the worker claims between the status snapshot and `cancel()`, the cancel *succeeds* (its WHERE covers queued *and* running) and the job is trashed — not "refused and left running and untrashed" as the comment says — while `request_cancel` was never sent, so the worker burns the full reconstruction before discovering `finish()` returns False. Final state correct; minutes of GPU wasted; comment asserts the opposite of reality. Fix: route the queued branch through `cancel_job` (which asks the worker to stop), or fix the comment.
- **L6. A corrupt `jobs.params` blob on a queued row wedges dispatch permanently.** `db.py:1239-1243`, `queue.py:735-753`. `_to_dict` parses `params` bare; `next_queued()` raises, the worker loop logs/sleeps/retries the same oldest row forever, starving everything behind it. The `_blob` tolerance was added for the other three JSON columns after this exact class of failure; params was left out. Reachable only via hand-edited DB or disk corruption. Fix: log-and-substitute `{}` on parse failure, or mark the row `error` at dispatch.
- **L7. `_discard_artifacts` runs blocking file I/O on the event loop.** `queue.py:1081, 1093`. Both call sites in `_process`'s finally invoke it directly; for a cancelled re-texture it `rmtree`s a directory of ~24 rendered/baked images on the `warlock-loop` thread that hosts dispatch and cancellation. Every DB write in the same finally goes through `to_thread`; the unlinks don't. Fix: `await asyncio.to_thread(self._discard_artifacts, job)`.

### Service layer / pipelines
- **L8. Seven writers still stage through a non-dotfile `.tmp` with no `finally` cleanup — the defect `_save_source` documents as fixed.** `service/files.py:151-153, 209-212, 276-280, 513-515`, `service/derive.py:494-496`, `pipelines/postprocess.py:250-252`, `judge.py:213-226`. An ENOSPC/PermissionError mid-write strands a visible `.tmp` (up to ~22 MB) in the job directory forever; `save_edited_image`'s *fixed* tmp name is additionally shared by concurrent callers, so two racing saves could rename a torn file onto the served `input.png`. Fix: adopt the `_save_source` shape (dot-prefixed unique temp + `finally` unlink) at all seven sites.
- **L9. `doctor` checks only the *existence* of the two vendored executables, never their version.** `doctor.py:196-203, 233-240`. `_warlockc_check` exists precisely because "vendor/ is gitignored, so a checkout routinely holds a DLL built from older sources" — an argument that applies verbatim to `trellis-server.exe` (hint pins "vendored build: v0.5.4" with nothing verifying it) and gltfpack. A stale exe behaves differently with no row saying why. Also noted: no doctor row detects a second Warlock instance holding `jobs.sqlite`. Fix: record/compare a version string or hash pin as a non-fatal "stale vendored build" detail.
- **L10. `warlock sweep` prints a pointer to a deleted file.** `sweep.py:184` — "See LEFTOVERS.md section 2." is user-facing terminal output; the file was deleted 2026-08-10. Fix: point at the live measurements doc.
- **L11. `service/judge.py` refuses a bad stage with a raw `ValueError`.** `service/judge.py:49-57` — the only service-layer refusal outside the `service.errors` hierarchy; today it guards programmer error (all callers pass constants), but if a pane ever wires a stage from state, the user gets "see the log" instead of an `Invalid` with `field="stage"`. Fix: raise `errors.Invalid(..., field="stage")` or document it as an assertion.
- **L12. Fetch child's stderr is discarded, so a crash before `main()` reports only an exit code.** `service/downloads.py:361-370, 453-456`; `pipelines/fetch_worker.py:218-219`. `fetch_worker.main` converts exceptions into `result.json`, but a child dying before `main` (broken venv import, and the `json.loads(sys.stdin.read())` that sits *outside* the try) writes no result and its traceback goes to DEVNULL — the user gets "exited with code 1" with the cause recorded nowhere. Fix: pipe stderr into the fallback detail (as `rigging.run_worker` does) and move the stdin parse inside the error-reporting path.
- **L13. `export_to_folder` copies onto consumed names in place.** `service/export.py:101-105`. `WARLOCK_EXPORT_DIR` exists to be watched by a game project; `copyfile` truncates the destination first, so a hot-reloading engine can read a torn GLB mid-re-export — outside the staged-writes invariant's letter (served files), squarely inside its reasoning. Fix: temp + `os.replace` per file.
- **L14. `prompt_preview` tolerates only `(ImportError, OSError)` around the tokenizer load.** `service/system.py:158-172`. A half-downloaded/corrupt tokenizer directory raises `ValueError`/`JSONDecodeError` out of transformers, escaping the degrade-not-fail contract and turning the live prompt preview into an error toast on every refresh. Fix: widen to `except Exception` with a debug log, or route through `fetch.present` first. *Confidence medium.*

### Studio UI
- **L15. `Viewer.compare()` parses a whole GLB synchronously on the frame thread.** `viewer_embed.py:246-249`. `_sync_viewer` was given a parse-off-thread split precisely because inline parsing froze the frame; the blocking survivors are argued for by name, and "Compare with selected" is not among them. Fix: reuse the `pending`/task split.
- **L16. The "Queued — N jobs in line" toast counts from the stale page.** `main.py:1046-1051`. `invalidate()` only marks dirty; the count misses the job this submit created. Fix: `waiting + 1` or toast next tick.
- **L17. `submit_promotion` drops a refused submit silently.** `panes/settings_3d.py:477`. The `"submit"` task key is shared with 2D Generate; an Accept from the matte modal landing while a create is in flight is refused by key-dedupe and nothing is queued or toasted — the modal closes looking like success. Narrow window. Fix: check the return and toast/retry.
- **L18. The Quit button always asks, with a message that is usually untrue.** `main.py:3427-3439`. With nothing generating and nothing unsaved — the common state — the modal still warns "Anything still generating is cancelled," and confirming can trigger up to six more guard questions. The guards already protect everything real. Fix: skip or reword the generic confirm when no job is running and no guard is dirty.
- **L19. The library keyboard stops at navigation.** `main.py:1751-1757`, `panes/library.py:807-813`. Up/Down/Enter navigate and open, but Delete/favourite/rename are mouse-only. Since delete-to-trash is deliberately confirm-free ("the trash *is* the confirmation"), a Delete-key binding would be exactly as safe as the menu item and much faster during a cull pass. Fix: bind Delete (and perhaps F for favourite) in the fall-through where the arrows live.

### Documentation
- **L20. INVARIANTS.md's inventory of `docs/` omits MODELS.md** (`INVARIANTS.md:218` — "four things now"; there are five, and README links the fifth). Same bullet's "eighteen files still cite `TODO.md §N`" is now 17. Fix together with H2, which edits the same paragraph.
- **L21. Manual ch. 21 tells extenders to declare a `download` field that is actually a derived property.** `21-extending.md:34-35` vs `models.py:233-237` — the declarable field is `fetch: tuple[Fetch, ...]`; writing `download=` on the frozen dataclass is a TypeError, and omitting `fetch` makes the model unfetchable in-app. Fix: reword the bullet.
- **L22. "Only two fatal startup checks" is wrong on small-VRAM hosts.** `15-installation.md:67-68`, `18-troubleshooting.md:78` vs `doctor.py:302` — `_vram_check` is fatal whenever the budget can't hold a lone trellis run. Fix: "two fatal checks, plus a fatal VRAM row on a card that cannot hold a reconstruction at all."
- **L23. CHANGELOG has no 0.0.15 entry** though `Warlock v0.0.15` commits exist and the preamble claims the file is "the only record of what a version actually changed." *Confidence medium that the skip wasn't deliberate.*
- **L24. `docs/REPORT.md` is an undated pre-implementation research note whose recommendation (ComfyUI-as-backend over HTTP/WebSocket) the project rejected** — nothing marks it as historical; a cold reader could take it for the current plan. Fix: one-line status header.
- **L25. INVARIANTS.md overstates `promote_to_model`'s admission** ("admits on weights as well" — it calls only `check_vram`; behaviorally correct since a promotion is an image job and `check_weights` is text-only by design, pinned by test — but the authoritative file claims a check the code doesn't make). Fix: correct the sentence.
- **L26. Two stale prose fragments in code:** `service/__init__.py:5-7` still describes "(transitional) HTTP routes" that were deleted (`tests/test_api.py` confirms "This *was* the HTTP suite"); `scripts/screenshot_modes.py:7` says "eight modes" — there are thirteen (the script itself is drift-proof, deriving from `modes.KEYS`).

### Hygiene
- **L27. `examples/` PNGs (player, two sprite sheets, tileset) are referenced by nothing** — no hit in src/tests/scripts/docs/README. Plausibly deliberate demo inputs, but nothing tells anyone they exist. Fix: link from the sprite-synthesis/Plotter manual chapters, or delete. *Confidence medium.*

---

## What's healthy (verified, not assumed)

Each of these was actively checked by at least one auditor — they are the audit's positive results, useful as a baseline for future passes:

- **Hygiene:** zero orphaned top-level definitions (two whole-repo scans over 1,702 public + all private defs); zero live FIXME/XXX/HACK; no stub bodies; no tracked bytecode; ruff fully clean; all 14 `scripts/` resolve statically against current APIs; bench recipes/suites match their schemas; all 9 native exports bound both ways at ABI 7 with fallbacks intact and `native.available()` guards at every call site.
- **Invariant anchors:** `HF_HUB_OFFLINE` first thing in the package; `merge_params` under one lock hold; `DERIVED_PARAMS` complete against the full enumerated set of worker-recorded params; `VECTOR_PARAMS` in `vectors.py` with the import direction pinned by an AST test; winjob scan covering all seven spawn sites (no other spawn primitive exists in `src/` at all); headless purity of all four packages pinned by dedicated tests; manual numbering gated; test basenames unique; `fetch_worker` imported only inside a fixture.
- **Model lifecycle:** transactional load, reference-drop doctrine, single-owner handoff, trellis stop/keep-handle discipline, per-generate adapter re-application, admission at every door, cancel windows — see the headline section.
- **DB/queue:** every `JobStore` method under the RLock; single connection (migrate's probe targets the legacy DB before the store opens); cancel-vs-finish closed at the DB and tested; `reconcile_startup` wired and tested; staged writes on every served name in the worker; `ProgressBus` id-checked so progress can't be misattributed; DB migrations append-only and idempotent.
- **Service:** export invariants hold end-to-end (source.glb written once; optimize-then-normalize; grounding on every path; derived exports under per-artifact locks and invalidated by re-optimize); refusal contract holds at every user-reachable door except L11; no studio code writes the store directly; no pipeline imports service; offline integrity confirmed by grep (`httpx` reaches only the local trellis port; LPIPS deliberately excluded because it downloads at import; matting/pose/DINO loads are `local_files_only=True`).
- **UI:** texture lifecycle fully paired (thumbnail cache, mode-prefix sweeps, viewer resize, shutdown); frame-loop discipline (GLB parse off-thread, storage walks/exports/pickers through TaskRunner, stat-gated per-frame caches); uid-addressed undo with serial head and byte budget; modal/dialog queue discipline; field-error routing with fold-opening; `push_id` discipline in every loop read.

---

## Suggested fix order

1. **H1** — decide 0.0.22 vs 0.0.21 and restore lockstep (suite is statically red until then).
2. **H2 + M12 + L20 + L25** — one editing pass over CLAUDE.md and INVARIANTS.md §§ 217-218 fixes the roadmap pointers, the sweep.py misattribution, the docs inventory, and the promote_to_model overstatement together; **H4** (SDXL-Turbo default) is the same class in the same files plus `queue.py`'s docstring.
3. **H3** — widen the dispatch VRAM credit (small change in `_check_resources`, big workflow unblock on default hardware).
4. **M1** — evict/reload the pipe on LoRA fetch (correctness of outputs *and* of the findings corpus).
5. **M2, M4** — two `Conflict` checks at existing doors (`dependent_jobs()` already exists; family check mirrors `create_pixel_sheet`).
6. **M8, M9, M3** — small UI/progress fixes with outsized perceived-quality impact.
7. **M13, M15, M14** — the three UX mediums (download cancel; bulk retry; shortcut documentation).
8. The remaining mediums (M5, M6, M7, M10, M11), then lows opportunistically — L8 (staging sites) and L12 (fetch stderr) first, as they affect debuggability of everything else.

## Coverage limits

Honest boundaries of this audit: nothing was executed (no pytest run, no GPU measurement — VRAM arithmetic is from the code's own constants; WDDM free-memory readings virtualize, so H3's refusal may be intermittent); the dead-code scans are whole-word textual and do not cover class *methods* with no callers or code reachable only from dead code; `review_mode.py`, the inker/plotter canvas panes, and `viewer/` internals got targeted greps rather than full reads on the UI pass; manual chapters 05–11 were only lightly checked (most were updated in HEAD); the six auditors' full per-file coverage notes are preserved in their reports.
