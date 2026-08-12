# Warlock Studio — Master Audit (consolidated)

**Date:** 2026-08-11 · **Tree audited:** HEAD `de87838` ("Warlock v0.0.21")
**Supersedes:** `AUDIT.md` (six-agent static audit, read-only, pytest not run) and `AUDIT2.md` (primary + three sub-agent audit, dynamic — ruff, full pytest, and targeted GPU tests were run). Both source reports remain in the tree until this document is accepted; every finding from both appears here exactly once (see the source index at the end).

## How the two audits were merged

- **IDs.** Category-prefixed IDs (`REL`, `RUN`, `MDL`, `ART`, `CON`, `SVC`, `UX`, `DST`, `DOC`, `TST`, `HYG`), renumbered. Every entry carries its source IDs as `[A1-…]` (AUDIT.md) and `[A2-…]` (AUDIT2.md).
- **Severity.** Four buckets: **Blocker / High / Medium / Low**. Where the audits disagreed, the merged rating follows the stronger evidence (dynamic verification and blast-radius argument beat a static read); the entry notes both original ratings. A2's "medium-high" maps to Medium and "low-medium"/"medium-low" to Low, tagged in the entry.
- **Duplicates.** 12 findings were reported by both audits (or by two A2 IDs); each was merged into one entry keeping all distinct evidence and both recommendations. 99 raw findings → **87 unified findings: 1 Blocker · 18 High · 37 Medium · 31 Low.**
- **Current-tree spot check (2026-08-11, post-audit).** The working tree has moved past `de87838` (uncommitted bench/metrics work unrelated to either audit). Re-verified today: `pyproject.toml:3` still `0.0.22`, `__version__` still `0.0.21`, CHANGELOG top entry still `0.0.21`, `docs/TODO.md` still absent. The headline findings are not stale.

## Verification snapshot (from A2's dynamic run, at `de87838`)

| Check | Result |
|---|---|
| `uv run ruff check .` | Passed (both audits) |
| `uv run pytest -q` | **2 failed**, 6,244 passed, 17 skipped (452.9 s) |
| Loader/VRAM/fetch/uninstall/Trellis/WinJob tests | 168 passed; 19 targeted queue resource/shutdown tests passed |
| UX-focused tests | 127 passed |
| Docs/manual tests | 42 passed, 3 skipped — despite the missing declared roadmap (see DOC-01) |

The two failures are exactly REL-01: `tests/test_changelog.py:99` (changelog 0.0.21 ≠ project 0.0.22) and `tests/test_offline.py:46` (`__version__` 0.0.21 ≠ installed 0.0.22). The full suite also unexpectedly ran real GPU tests on this provisioned host (TST-02).

---

## Executive summary

The codebase is substantially implemented and in unusually good hygiene: zero dead top-level definitions (1,702 checked), zero stale FIXME/HACK markers, no stub bodies, clean ruff, and most of the big invariants — staged writes, offline integrity, `bpy` containment, winjob coverage, lock discipline, headless-package purity, texture lifecycle, uid-addressed undo — verified holding by direct check.

The problems cluster at the **seams between otherwise capable systems**:

1. **The release state at HEAD is incoherent and the suite is red** (REL-01): pyproject says 0.0.22, everything else says 0.0.21, and `de87838` also deleted `docs/TODO.md` while four authoritative places still declare it the live roadmap (DOC-01).
2. **Model/LoRA lifecycle is safe during an ordinary serial job but not at maintenance and shutdown boundaries**: uninstall races admission and generation (MDL-01), bounded shutdown can race or outlive blocking model threads (MDL-02), a LoRA installed while the pipe is resident is silently dropped while records claim it ran (MDL-05), and the dispatch-time VRAM credit only covers `kind=="text"`, so three job kinds spuriously refuse on default hardware (MDL-06).
3. **Supply chain and install integrity are thinner than the docs imply**: model downloads are unpinned and BiRefNet runs repository Python with `trust_remote_code=True` (MDL-03); "installed" means a few filenames exist (MDL-08); pasteable model commands target a different directory from the runtime default (DST-02); a clean wheel cannot find the vendored binaries (DST-01).
4. **A `done` mesh can be quietly degraded** — normalization/grounding failures are swallowed while the job stays `done` (ART-01) — and a second Studio instance is only logged, though it can kill the first instance's Trellis server (RUN-01).
5. **Accessibility and recovery have structural gaps**: keyboard navigation is disabled app-wide (UX-02), high UI scale can make controls unreachable (UX-01), crash recovery exists only for Inker (UX-05).

### The headline question: is model/LoRA loading/unloading safe?

**Merged verdict: safe on the ordinary serial-job path; not safe at the uninstall/shutdown/download seams; and provenance is currently outside the definition of "safe" entirely.**

Verified holding (by static trace in A1, and by 168 passing lifecycle tests plus real-GPU conditioning/VRAM tests in A2): transactional load (`Text2Image.load` publishes only a fully configured pipe; every failure path unwinds through `del pipe` + `_reclaim()`); the reference-drop doctrine at every unload site; single-owner handoff via `queue._needs_handoff` with `unload()`-never-`trim()` teardown; Trellis stop-confirms-death-and-keeps-handle-on-failure; per-`generate` adapter re-application (no cross-job LoRA bleed); admission checks at every door with dispatch-time re-check; all cancel windows.

Not holding: MDL-01, MDL-02, MDL-04, MDL-05, MDL-06 (all High, below), plus the bounded Medium/Low gaps MDL-07 through MDL-17. The core structural insight (A2): **event-loop serialization does not serialize the underlying model thread** — the actual model operation runs in `asyncio.to_thread`, so scheduling an unload on the worker loop proves nothing about the thread still inside `from_pretrained` or sampling. A model-store read/write lease that covers the *thread*, not the coroutine, is the missing primitive.

---

## Blocker

### REL-01 — Release identity is inconsistent; the suite is red at HEAD
**bug/packaging · confidence high · [A1-H1, A2-REL-01] · verified dynamically and re-verified on today's tree**
- `pyproject.toml:3` = `0.0.22` (bumped in `de87838`); `CHANGELOG.md:9` and `src/warlock/__init__.py` (`__version__`, line 15) = `0.0.21`; `uv.lock:1697-1698` follows pyproject.
- `tests/test_changelog.py:99` and `tests/test_offline.py:46` fail — confirmed by an actual pytest run, not just static reading.
- User-visible: Home prints 0.0.22 beside the title while "What's new" falls back to the 0.0.21 entry — the exact failure the changelog test's docstring describes. The commit subject (`Warlock v0.0.21`) also drifts from the manifest it bumped.
- Also: the comment above `__version__` names the wrong pinning test ("tests/test_models.py"; it is `tests/test_offline.py`).
- **Fix:** decide direction — either write the `## 0.0.22` CHANGELOG entry and bump `__version__`, or revert the pyproject bump to 0.0.21 — then correct the comment. A2 additionally recommends deriving runtime version from installed metadata with a source-tree fallback, and making the two tests a mandatory release preflight (see TST-03).

---

## High

### RUN-01 — A second Studio instance is only logged, and can kill the first instance's Trellis server
**bug/process · confidence high · [A2-RUN-01; A1-L9 noted the missing doctor row] · `studio/main.py:4180-4228`, `pipelines/trellis.py:210-216,312-352`, manual `18-troubleshooting.md:157-158`**
Startup sees a live session marker and writes only a log warning; the second instance continues, sharing the job DB and engine port (the manual itself says the second "will lose fights over both"). Worse: Trellis port reclamation treats a matching configured executable path as proof the listener is *our* orphan — two normal instances share that path, so the second can terminate the first's live server. The session marker is a single file either process can overwrite/remove, weakening crash attribution. No doctor row detects a second instance holding `jobs.sqlite`.
**Fix:** acquire a real per-home OS-level single-instance lock before migration/DB/runtime startup, surfaced as a native dialog, not just `warlock.log`. Never use executable identity alone to kill a listener. (If multi-instance is ever wanted: per-instance ports, process-safe model-store mutation, per-process markers, explicit DB/queue ownership.)

### MDL-01 — Model uninstall races queue admission and in-flight generation
**bug/hardening · merged severity High (A1: Medium; A2: High) · [A1-M7, A2-MDL-01] · `service/downloads.py:241-299`, `queue.py:786-812`, `_q_generate.py:196-216`, `panes/app_settings.py:310-317`**
Three defects compound: (a) the live-jobs guard filters `store.list(limit=MAX_LIST_LIMIT)` — a 5,000-row page — instead of the purpose-built unbounded `store.active_jobs()`, so a queued row older than the page is invisible and its weights get deleted; (b) the check is a snapshot — a `create_job` in the gap gets claimed, and `unload_text2image` then runs `pipe.unload()` concurrently with a `generate()` in flight (the loop is free while `_generate` awaits `to_thread`); (c) the Settings pane's disabled-buttons convention is UI-only — headless callers, shutdown races, or a second process share no backend lock (see MDL-11). Possible outcomes: unload concurrent with `load()`/`generate()`, deleted files under a load, two pipes resident, Windows file-lock failures, OOM. Weight deletion under a reading pipe at least fails loud (`PermissionError` → `Failed`).
**Fix:** immediately — use `active_jobs()` and re-verify `worker.current_job_id is None` inside the same loop-side callable that unloads. Structurally — a service-level model-store read/write lease: dispatch and every model operation hold a read/use lease; download/uninstall/repair acquire an exclusive maintenance lease, pause claims, recheck, unload, mutate, then resume. The lease must cover the model *thread*, not merely the event-loop coroutine.

### MDL-02 — Bounded shutdown can race or outlive blocking model and task threads
**bug · merged severity High (A1: Low for its narrower strand-a-pipe window; A2: High with a partial repro) · [A1-L1, A2-MDL-02] · `queue.py:704-731`, `pipelines/text2image.py:241-296`, `studio/runtime.py:196-234`, `studio/tasks.py:188-229`, `service/downloads.py:44`**
Cancelling an asyncio task awaiting `to_thread` does not stop the worker thread. After the 20 s grace, shutdown cancels the coroutine and unloads the pipe while `from_pretrained` or sampling may still be running: a late load can publish after the final `if …loaded: unload()` guard read `loaded == False` (up to ~16 GiB host commit stays resident for process life where the interpreter survives); an active sample can hold references while another thread clears the owner and calls `empty_cache()`. TaskRunner uses `ThreadPoolExecutor.shutdown(wait=False)` plus mutation of private `_threads_queues` — A2 reproduced that a blocked non-daemon worker still keeps the interpreter alive (its subprocess outlived the "prompt" shutdown until the worker was released). Downloads can wait up to four hours with no cancellation (MDL-14).
**Fix:** join the outstanding load before the `.loaded` check and/or have `load()` re-check a stop flag before publishing; provide cooperative cancellation for every long task; track subprocess handles so shutdown can kill and reap; never unload a pipe until its model-operation lease (MDL-01) is released; replace private `concurrent.futures` mutation with supported lifecycle control; add subprocess-level tests asserting the interpreter actually exits while load/sample/dialog/download are blocked.

### MDL-03 — Model repositories are unpinned, and BiRefNet executes repository code
**security/supply-chain · confidence high · [A2-MDL-03] · `models.py:105-176,909-944`, `pipelines/fetch_worker.py:164-189`, `pipelines/matting.py:225-247`**
`Fetch` has no revision or digest; `snapshot_download()` follows the repo's current default revision. BiRefNet explicitly downloads `*.py` and later loads with `trust_remote_code=True`. A future Download click can retrieve different Python and weights from those audited. The fetch child is crash/RSS isolation, not a security sandbox — it runs with the user's filesystem/network permissions; `uv.lock` does not pin the downloaded code.
**Fix:** give every registry fetch an immutable commit SHA carried through the worker and the rendered manual command; verify an artifact manifest of expected paths/sizes/hashes; store and display the installed revision and include it in generation provenance; prefer vendored/audited model implementation code over `trust_remote_code` (if remote code remains, make trust visible and restrict the boundary).

### MDL-04 — Offloaded FLUX loads can cross the Windows commit limit despite passing admission
**bug · confidence high · [A2-MDL-04] · `models.py:524-559`, `pipelines/text2image.py:273-280`, `queue.py:391-410`, `vram.py:110-129`**
FLUX.2 is ~16 GiB in host memory under CPU offload. Host-commit admission refuses only at the 90 % ceiling: a machine at 80–89 % can have far less than 16 GiB left, be admitted, and cross the limit during checkpoint allocation — on Windows plausibly OS process termination, the exact failure the check exists to prevent. (Tension with A1's healthy-list item "host commit is re-checked at dispatch": the re-check exists but its threshold is a percentage, not the bytes the next operation needs.)
**Fix:** model host peak footprint per base spec; require absolute free commit of `host_peak + margin` immediately before `from_pretrained` and before large placement transitions; keep percentage telemetry but not as the admission criterion.

### MDL-05 — A LoRA installed while the base pipe is resident is silently dropped — and the records claim it ran
**bug · merged severity High (A1: Medium; A2: High — silent wrong output plus corpus contamination) · [A1-M1, A2-MDL-05] · `pipelines/text2image.py:298-396`, `service/downloads.py:170-206`, `_q_sprite.py:97-112,218-228`, `studio/main.py:957-979`**
`_load_loras` runs once inside `load()`; `_adapters` is frozen for the pipe's life, and the in-app install path never evicts or reloads the pipe. Flow: generate once on `sdxl_cfg` (pipe stays warm ≤ 600 s), install `pixelxl` in Settings, submit a styled job → `check_weights` passes (file on disk), the cached pipe is reused, and `_apply_adapters` drops the adapter with a *wrong* "not downloaded" warning. Consequences: the output silently lacks the style (the outcome `check_weights` exists to prevent); `params["style_lora"]` is in `VECTOR_PARAMS`, so the findings corpus records a style that never ran; the pixel-sheet sidecar writes `recipe["style_lora"]` from the *request* — resurrecting verbatim the "sidecar naming a LoRA that never loaded" bug the 2026-08-11 door work closed.
**Fix:** increment a model-store generation on successful download/uninstall/repair and have the worker invalidate or refresh the resident pipe before the next job (or let `_apply_adapters` late-load a registry-valid, family-fitting, on-disk adapter at a safe boundary). A missing selected style must be a user-visible refusal, never a warning plus a successful-looking job. Add a resident-pipe → download → immediate-generate regression test.

### MDL-06 — Dispatch-time VRAM credit only covers `kind == "text"`; pixel-sheet, sprite-synthesis, and retexture double-charge the resident pipe
**bug · confidence medium-high · [A1-H3; A2 listed "resident-memory crediting" among strengths without per-kind analysis] · `queue.py:998-1013`, `vram.py:173-196`**
`_check_resources` credits the resident pipe's measured reserved memory back against the estimate only for `kind=="text"`, while `vram.estimate` prices the other three kinds with a full image-model term (+ ControlNet, + IP encoder, + `TRELLIS_GIB` under coexist). On the flagship 32 GiB coexist card the *natural* flow — generate a 2D reference (leaves a ~7.5 GiB pipe warm by design), then start a sprite synthesis — computes need ≈ 26.7 vs free ≈ 23.5 and refuses at dispatch with "close other GPU applications", even though the job would **reuse** the pipe being counted against it (true incremental need ≈ 3.7 GiB). Pixel sheets refuse by ~2.5 GiB; retexture is knife-edge. Waiting out the 600 s idle eviction "fixes" it, presenting as a phantom VRAM leak. Fails safe (refusal, not crash) but refuses a supported workflow on default hardware; headless tests skip the branch (`device_memory()` is None). WDDM free-VRAM virtualization may make the refusal intermittent rather than deterministic.
**Fix:** widen the credit to every kind whose estimate carries an image-model term — or better, derive the credit from `vram.estimate` itself rather than a kind list.

### ART-01 — Canonical mesh normalization can fail while the job is still `done`
**bug · confidence high · [A2-ART-01] · `_q_mesh.py:109-152`, `service/_jobs_create.py:475-504`, `CLAUDE.md:26`, `README.md:34,111`**
Normalization (requested size, center X/Z, floor grounding) is wrapped in a catch-everything; the worker logs and continues, and built/imported models do the same yet still insert `status="done"`. Mesh reporting may also fail without changing the terminal state. The non-fatal design protects a successfully reconstructed `source.glb`, but overstates `model.glb`: a user can export a visibly successful asset with the wrong pivot or scale, with the only explanation in logs — contradicting the documented "grounding always runs" claim (A1's healthy list verified grounding *runs* on every path; ART-01 is about its failures being swallowed — both are true, and the invariant wording should be qualified).
**Fix:** distinguish `done` from `done_degraded` (or carry structured artifact health), surface which canonical steps failed, offer Retry/Repair, and never present an unnormalized file as the canonical engine artifact without a warning. Record normalization/report failures in provenance. Note: any new worker-recorded health field must join `DERIVED_PARAMS` or a reroll wears a stale verdict.

### UX-01 — High UI scale can make workspace controls unreachable
**accessibility · confidence high · [A2-UX-01] · `studio/main.py:46-51,112-120,2016-2031`, `studio/tokens.py:14-24`, `studio/layout.py:143-152`**
The resize floor follows monitor DPI, but workspace geometry follows DPI × user UI scale. Three-column workspaces reserve two 300-design-px sidebars and force a 300-px center: at the 1100-px minimum window, 1.5×/2× scale needs ~1350/1800 px before gutters. The header has a compact fallback; workspaces do not collapse, scroll, or convert sidebars to drawers — users who enlarge the UI for accessibility can lose the right pane entirely.
**Fix:** responsive breakpoints — collapsible/drawer sidebars, one-pane navigation at narrow widths, an always-reachable pane switcher; render/interaction-test every mode at minimum size × {1.0, 1.5, 2.0} scale in both themes.

### UX-02 — Keyboard navigation is disabled across most of the application
**accessibility · confidence high · [A2-UX-02] · `studio/dialogs.py:246-263`, `studio/imgui_backend.py:68-98,248-286`, `studio/main.py:3444-3624`, `panes/app_settings.py:463-475`, `panes/inker_layers.py:122-125`**
ImGui keyboard navigation is off and the backend maps only a narrow key set; dialogs manually emulate Enter; custom Tab logic covers the primary 2D/3D forms but Settings, Profiles, Library, inspector, mode switch, and editor controls have no focus traversal; some controls are icon-only or hidden-ID with no semantic label. Keyboard-only users and many users with motor impairments cannot complete normal workflows, and the shortcut sheet makes support look broader than it is.
**Fix:** enable ImGui keyboard nav, complete the key mapping, define focus order, show a contrast-qualified focus indicator, name icon/hidden-ID controls, and add end-to-end keyboard-only traversal tests per top-level mode.

### UX-03 — Library selection, inspector, and viewport can describe different assets
**bug · confidence high · [A2-UX-03] · `panes/library.py:886-895`, `studio/main.py:1173-1174,1299-1347,1928-1929`, `studio/jobs_cache.py:26-30,111-130`**
Selecting a card updates `state.selected` immediately, but viewer sync waits for a cache reread (3 s idle interval) or mode transition; parsing is async while the previous model stays visible and `viewer.pending` is never surfaced. The inspector can describe asset B while the viewport shows asset A — export/compare/edit decisions become untrustworthy.
**Fix:** synchronize on selection change; while B loads, label "Loading B…", dim/clear A; keep an explicit relationship between selection ID, displayed ID, pending ID, and failure. Never show an unlabeled stale model beside a new inspector.

### UX-04 — Compare can self-compare, freeze the frame, or crash Studio on malformed input
**bug · merged (A2 High; A1-L15 found the sync-parse half) · [A1-L15, A2-UX-04] · `panes/library.py:697-702,788-798,1012-1020`, `studio/viewer_embed.py:246-255`, `studio/state.py:1022-1028`, `studio/main.py:592-602,1341-1374`**
Right-click selects the target before "Compare with selected" runs, so the baseline can be replaced by the target and later sync can make both sides identical. `Viewer.compare()` parses the GLB and uploads GPU resources synchronously on the frame thread with no local error boundary — unlike the primary viewer's async, stale-guarded, contained path (`_sync_viewer` got the `pending`/task split precisely because inline parsing froze the frame, and the blocking survivors are argued for by name; compare is not among them). Outcomes: misleading comparisons, visible freezes on large models, whole-app exit if parsing or GPU upload raises.
**Fix:** capture baseline and target IDs explicitly before menu selection, refuse identical IDs, label it "Compare selected with this", parse off-thread with stale-result guards, adopt GPU resources only after validation inside a contained error surface — i.e., reuse the primary viewer's path.

### UX-05 — Crash recovery covers only Inker
**gap · confidence high · [A2-UX-05] · `studio/main.py:1186-1200,1856-1889`, `studio/inker_mode.py:1433-1602`, manual `07-inker.md:393-405`**
Orderly quit guards several authored-document modes, but periodic atomic autosave/recovery exists only for Inker. Clay, Plotter, Packwright, inspector pose edits, Poser, and profile drafts (UX-17) can all hold substantial unsaved work.
**Fix:** one document-journal/recovery service — atomic snapshots, origin identity, schema/version, last-good cleanup, startup recovery chooser — rather than per-mode reimplementation. Add profile drafts to dirty/recovery state.

### UX-06 — Startup and fatal frame-loop failures disappear without a user-facing report
**gap · confidence high · [A2-UX-06] · `studio/main.py:544-603,4232-4258`**
Fatal exceptions are logged and teardown runs, but the splash/window can simply vanish — no message, no log location, no recovery status.
**Fix:** a minimal native fallback dialog independent of ImGui/GL: friendly summary, `warlock.log`/`crash.log` paths, Copy Details, Open Log Folder, recovery status; full traceback only in the log.

### DST-01 — Wheel layout cannot satisfy the runtime's native-binary lookup
**packaging · confidence high · [A2-DST-01] · `config.py:10-11,76-110`, `.gitignore:20`, `pyproject.toml:95-107`**
Native defaults derive from `Path(__file__).parents[2] / "vendor"` — correct for an editable checkout, wrong in a wheel (points under the environment, e.g. `Lib/vendor`); the wheel includes the package, manual, and changelog but not the gitignored `vendor/` binaries, and setup docs tell users to populate the checkout's `vendor/`, which a non-editable install never inspects.
**Fix:** explicitly choose the supported distribution model — either install pinned native assets under a versioned `WARLOCK_HOME`/app-data location, or package redistributable binaries as wheel resources resolved via `importlib.resources`; or officially declare source-checkout-only and make a wheel install fail with an actionable message. Add a clean built-wheel smoke test that runs Doctor and resolves every resource (or a guard that refuses wheel installs).

### DST-02 — Pasteable model commands target a different directory from the runtime default
**bug/doc · merged severity High (A1: Medium; A2: High — gigabytes land where Warlock does not look) · [A1-M10, A2-DST-02] · `config.py:195-229`, `models.py:105-176,155`, `doctor.py:190-192,350-405`, `docs/MODELS.md`, manual `15-installation.md:74-158`, `README.md:47-57`, `tests/test_fetch.py:144-155`**
Runtime defaults to `~/.warlock/models` (`config.py:211-212`); README now says `--local-dir $HOME/.warlock/models/...`, but MODELS.md, manual ch. 15, and `models.py` `download_text` (which doctor's paste-able remedies emit) still render relative `models/...`. A command pasted from those lands weights in `<cwd>/models/`, rescued only by the one-time migration — which skips a root "if the destination already holds something", so on any host with a non-empty `~/.warlock/models` (e.g. after one in-app fetch) the pasted download is stranded and doctor keeps reporting it missing. `INVARIANTS.md:23` is also now wrong about itself ("the literal `--local-dir models/...` is the README's spelling" — it no longer is). Tests pin the relative spelling rather than resolved behavior.
**Fix:** render commands from the effective `Config` (quote resolved paths for PowerShell; honor `WARLOCK_T2I_ROOT`/`WARLOCK_T2I_DIR`/Trellis overrides); standardize MODELS.md, ch. 15, and `download_text` on the explicit home path; update `INVARIANTS.md:23`; test copy/paste from a non-project CWD.

### DOC-01 — `docs/TODO.md` was deleted, but four places still declare it the live roadmap — and the docs tests silently miss it
**doc-conflict · confidence high · [A1-H2, A2-DOC-01] · still absent on today's tree**
`de87838` deleted `docs/TODO.md` (186 lines). Still pointing at it in the present tense: `CLAUDE.md:31` ("the only roadmap"), `docs/INVARIANTS.md:218`, `tests/test_ux_todo_fixes.py:1` (docstring), and `tests/test_external_doc_links.py:43` — where `SOURCES` still lists it and the `if not source.exists(): continue` at line 101 silently skips it, the exact silent-shrink failure that file's own comment warns about (the docs suite passed anyway; the link test enforces only a global count). Risk: the next session follows CLAUDE.md to a roadmap that doesn't exist — or recreates one, re-legitimising `§N` citations that are defined to resolve against the *deleted* LEFTOVERS.md.
**Fix:** either restore a tracked roadmap or update CLAUDE.md/INVARIANTS.md to say the file was deleted 2026-08-11 (git history keeps it; `§N` resolves via `git log --diff-filter=D`); fix the test docstring; drop or annotate the `SOURCES` entry; make the link test assert every declared source exists and enforce per-source coverage, not only global.

### DOC-02 — "SDXL-Turbo is the default" survives in the authoritative docs; code moved the default to `sdxl_cfg`
**stale-doc · confidence high · [A1-H4, A2-DOC-02]**
Code: `models.py:38` `DEFAULT_BASE_MODEL = "sdxl_cfg"`, citing `docs/measurements/2026-08-11-default-base-model.md`; README, MODELS.md, and manual ch. 03/16 agree. Against that: `docs/INVARIANTS.md:7,25,214` (including the wrong causal story — "because FLUX is gated" was never why `sdxl_cfg` won), CLAUDE.md's pipeline one-liner, and `queue.py:3`'s module docstring. INVARIANTS.md instructs readers to consult it before modifying a subsystem — and hands them the pre-2026-08-11 answer.
**Fix:** reword the four doc spots plus the queue docstring to name SDXL 1.0 full-CFG as the default with Turbo as the fast option; A2 also suggests a small doc assertion that declared defaults match the registry.

---

## Medium

### Runtime / process

- **RUN-02 — Home-migration's "no live writer" lock is released before the multi-minute copy begins.** [A1-M6] · `migrate.py:159-173,192-224` · confidence high (mechanism), low (likelihood). `BEGIN EXCLUSIVE`/`ROLLBACK` proves nothing is live *at that instant*, then the lock drops before `_move` starts the ~95 GB cross-volume `copytree`. A second Warlock launched mid-copy passes its own precondition, writes into legacy `assets/`, and the first process's post-verify `rmtree(legacy)` deletes those writes (recount catches added files; equal-size in-place modification is invisible; Windows sharing semantics blunt the worst case to a silently split library). Related: RUN-01. **Fix:** hold the exclusive connection open across `_move` (close in a `finally` after the deletes), or re-take it immediately before each `rmtree`.
- **RUN-03 — Malformed environment configuration can crash before diagnostics.** [A2-CFG-01] · `config.py:30-67,155-177,195-205,328-351`. Several fields call raw `int()`/`float()` inside dataclass factories — a typo in a port/retry/timeout/threshold raises before Doctor/startup messaging exists; other fields (optional VRAM floats) silently turn malformed input into `None`, so the policy is inconsistent — a malformed explicit safety limit becomes "unset". **Fix:** centralize typed parsing and range validation, retain the env-var name, report all invalid values in one actionable startup/Doctor surface.

### Model / LoRA / downloads

- **MDL-07 — Optional LoRAs load eagerly; one corrupt unused file bricks every compatible base.** [A2-MDL-06] · `pipelines/text2image.py:298-344`. Every compatible style LoRA on disk loads during base init; missing files are tolerated but a corrupt/incompatible optional file raises inside the load transaction — the user cannot generate without a style they never selected. **Fix:** lazily load only selected adapters, or pre-validate/quarantine per-adapter with health shown per Settings row; keep required step-distillation LoRAs fatal. Test corrupt-optional + healthy-base + unrelated-selected-style.
- **MDL-08 — "Present" means a few filenames exist, not that weights are complete or trustworthy.** [A2-MDL-07] · severity medium-high · `fetch.py:438-497`, `tests/conftest.py:140-175`, `doctor.py:350-406`. Presence checks are `Path.exists()` on a small sentinel set; fixtures create empty files and production admission accepts them. Manual commands, copied directories, corruption, upstream layout changes, or a hard-killed publication bypass the staging guarantee. **Fix:** completion manifest written only after verification against the pinned revision (MDL-03), required for "installed"; at minimum reject zero-length safetensors, parse required JSON, and separate "files found" from "load probe passed" in Doctor/Settings.
- **MDL-09 — Multi-model free-space admission charges the whole plan to the first volume.** [A2-MDL-08] · `fetch.py:142-171,298-315`, `tests/test_fetch.py:218-244`. `disk_refusal()` totals every job but calls `free_gib()` only for `jobs[0].dest`; `WARLOCK_T2I_DIR` can split volumes — a roomy first drive approves a large write to a nearly full one, and the inverse falsely refuses. **Fix:** group planned bytes per resolved volume, enforce headroom per group, report which destination lacks space; add mixed-drive tests.
- **MDL-10 — Whole-selection download atomicity is promised but not implemented.** [A2-MDL-09] · `service/downloads.py:170-206`, `pipelines/fetch_worker.py:105-160,164-213`. Repos publish sequentially: repo 1 stays installed if repo 2 fails (checkpoint + required distillation LoRA can strand GiB unusable). Per-repo staged publication is careful under Python unwinding, but a hard kill strands deterministic `.fetch.part`/`.fetch.bak` paths; rollback failures are suppressed and the backup deleted. **Fix:** stage the entire selection, validate, publish via a journaled transaction with unique txn dirs and an incomplete marker; recover/quarantine on startup; at minimum return a precise partial-success result with safe Resume/Rollback.
- **MDL-11 — Model mutation serialization exists only as a UI convention.** [A2-MDL-10] · `panes/app_settings.py:310-317`, `service/downloads.py:170-299`. `download()`/`uninstall()` share no backend lock — headless callers, future panes, shutdown races, or a second process can collide on staging/backup/destination paths. **Fix:** per-model-root mutation lock in the service layer (in addition to the MDL-01 lease); UI disabled state should reflect backend ownership, not define it.
- **MDL-12 — Idle cache eviction is latched and misses caches loaded outside the GPU queue.** [A2-MDL-11] · `queue.py:816-850,1133-1139`, `service/derive.py:403`, `service/matte.py:225-230`. After one idle pass `_caches_evicted` suppresses future eviction until a GPU job completes; matting loaded later via a TaskRunner operation leaves its ~1.5 GiB child resident indefinitely. **Fix:** per-cache loaded/last-used state (or notify a central cache manager on every populate); test evict → matte-via-task → evict.
- **MDL-13 — Trellis error responses bypass the response-size cap.** [A2-MDL-12] · `pipelines/trellis.py:464-491`. Success bodies stream under a ceiling; HTTP error bodies use unbounded `await r.aread()` — a wedged/compromised local server can exhaust host memory. The success path also holds chunks then joins, transiently doubling up to the 512 MiB limit. **Fix:** stream and cap error bodies; stream success to a temp file or bounded buffer; set a measured ceiling from legitimate maximum artifacts.
- **MDL-14 — No Cancel for multi-gigabyte downloads; no Verify/Repair/Update for installed models.** [A1-M13, A2-MDL-13, A2-UX-11] · `panes/app_settings.py:450-521`, `studio/tasks.py:71-140`, `service/downloads.py:44`. Progress is wired per-row and other rows are disabled during a fetch, but there is no Cancel anywhere — a mistaken 16 GB fetch on a slow line (timeout up to four hours) can only be stopped by quitting the app. The safe mechanism already exists: the kill-on-close job reaps the child, and staging means cancel leaves no half-installed model — only the button is missing. Installed rows offer Remove but no Verify/Repair/Update, so a corrupt-but-present checkpoint has no recovery besides remove/re-download. **Fix:** Cancel/Cancelling beside the progress bar backed by child termination and journal-safe cleanup (Resume preferable for large immutable downloads); after pinning (MDL-03), add installed/current/outdated/corrupt state with Verify and Repair.
- **MDL-15 — Retexture door has no family check; a FLUX/klein base passes admission and dies mid-job.** [A1-M4] · `service/_jobs_rework.py:206-208` vs `pipelines/text2image.py:414-421` · confidence high (mechanism), medium (reachability from the shipped pane). `Text2Image._conditioned` refuses non-SDXL families at *runtime*, but `retexture_job` checks only registry existence (an unset `base_model` on a host whose `WARLOCK_T2I_MODEL` names a klein entry also qualifies). The job queues, renders all six Blender views, stops trellis, loads ~16 GiB host commit, then errors on the first restyle pass — minutes of work plus a trellis restart, violating the stated door principle; `rerun_job --reroll` repeats it. **Fix:** refuse non-`FAMILY_SDXL` bases in `retexture_job` and its rerun path, mirroring `create_pixel_sheet`.

### Artifacts / testability

- **ART-02 — Queue happy-path fakes mostly exercise swallowed degradation paths.** [A2-ART-02] · `tests/conftest.py:239-272,313-364`, `_q_mesh.py:100-153,229-259`, `tests/test_fakes_match_real_signatures.py:16-34`. The fake Trellis writes `fake-glb` and fake t2i writes `fake-png`; post-processing catches parse/normalize/audit/report failures, so many tests asserting `done` never test successful artifact processing. **Fix:** default fakes emit tiny structurally valid PNG/GLB; malformed artifacts move behind explicit degraded-behavior tests; a successful queue test asserts normalization/audit/report ran without swallowed exceptions. (Pairs with ART-01.)

### Concurrency / job lifecycle

- **CON-01 — `optimize_job`/`retexture_job` never consult `dependent_jobs()`; a retarget races a running re-texture or rig on the same mesh.** [A1-M2] · `service/_jobs_rework.py:68-70`, `_q_sprite.py:655-664` · confidence high (code), medium (frequency). Both doors refuse only on the target job's own status, but a retexture/rig/sheet is a separate row writing into that done job's directory. Queue a re-texture for done mesh J, retarget J inline: the re-texture's `os.replace` publishes a skin baked from the pre-retarget geometry over the retargeted mesh, silently reverting the triangle budget while `params["profile"]` claims the tier ran; a rig in flight binds the old mesh and `stale_rig_artifacts` reports nothing. Per-artifact locks don't cover this. **Fix:** raise `Conflict` when `dependent_jobs(svc, job_id)` is non-empty — the helper exists and `_jobs_lifecycle.py`'s docstring names it as the answer to exactly this shape.
- **CON-02 — Pixel-sheet and re-texture jobs drive the progress bar to 100 % during the first sampling pass.** [A1-M3] · `_q_sprite.py:170-171,590-591`, `progress.py:88-98` · confidence high. Both kinds use the generic `_t2i_state`/`_t2i_step` callbacks, but `_PHASES_BY_KIND` has no entry for them — `phases_for` falls back to `PHASES_IMAGE` and an unknown phase maps onto the whole bar; the last sampling step of the first band/view emits 100 % and never-regress creep pins it for the rest of a multi-minute job. The exact trap INVARIANTS.md documents; `_sprite_synthesis` routes around it while its siblings walked in. **Fix:** add `PHASES_PIXEL_SHEET`/`PHASES_RETEXTURE` tables (or a `_sprite_step`-style mapper) and pin them in `tests/test_progress.py`.
- **CON-03 — `apply_library_pose` bypasses the pose-cap lock its sibling exists to hold.** [A1-M5] · `service/poses.py:220-233` vs `service/rig.py:148-155` · confidence high (code), medium (impact). `rig.save_pose` wraps the `MAX_POSES` check-then-write in `svc.convert_lock(job_id, "poses")`; `apply_library_pose` performs the identical check-then-write with no lock, voiding the sibling's guarantee (both read `MAX_POSES − 1`, both write). **Fix:** wrap the cap check + `rigging.save_pose` in the same `convert_lock`.

### Studio UI

- **UX-07 — Destructive Delete/Discard gets default focus and Enter confirms it.** [A2-UX-07] · severity medium-high · `studio/dialogs.py:246-263`. **Fix:** make Keep/Cancel the safe default; require explicit focus/activation for destruction.
- **UX-08 — Global shortcuts leak through the matte modal** — suppression knows only confirmation/prompt queues. [A2-UX-08] · `studio/main.py:1458-1476,1566-1569,3193-3209`, `panes/settings_3d.py:480-508`. **Fix:** centralize modal ownership/input suppression for every modal surface.
- **UX-09 — The download-complete frame synchronously runs forced diagnostics** including slow Torch/Blender probes. [A2-UX-09] · `studio/main.py:957-979`, `doctor.py:77-94,123-170,255-273`. **Fix:** run verification as a task with "Verifying installation…".
- **UX-10 — Corrupt settings reset silently; failed settings writes are ignored.** [A2-UX-10] · `studio/settings.py:65-83,101-126`, `studio/main.py:827,3899-3904`. **Fix:** persistent warning, preserve/rename corrupt input, expose retry, confirm defaults were restored.
- **UX-11 — Restore, purge, and empty-trash never refresh the job cache or the storage figure.** [A1-M8] · `studio/main.py:1083`, `panes/library.py:1049-1059,1290` · confidence high. `_on_task_done` invalidates on `delete:/prune/rename:/name:/tags:/fav:` only — toast-Undo and trash-Restore look inert for up to the 3 s backstop, and purge/Empty-trash (the actions that actually free disk) never re-measure, so the "N jobs – X GB" footer keeps the pre-delete figure all session. **Fix:** add `restore:`/`purge:` to the invalidate prefixes and `purge:`/`empty-trash` to the storage re-measure.
- **UX-12 — Ctrl+K and Ctrl+Enter read `pygame.key.get_mods()` instead of `event.mod` — the hazard the same file documents.** [A1-M9] · `studio/main.py:1627-1631,1735` vs `_passes_text_field` (1541-1547) · confidence high (inconsistency), medium (frequency). Events drain in a batch; a modifier released between press and processing makes `get_mods()` lie: a fast Ctrl+K in Inker falls through to bare `k` — the **Rect tool** — so the palette fails *and* the tool changes; a fast Ctrl+Enter submit silently drops. **Fix:** use `event.mod`, as `_passes_text_field` and `review_mode.handle_key` already do.
- **UX-13 — The manual's "full list" of shortcuts is missing Clay bindings that exist; the popup hides Clay's Ctrl+W.** [A1-M14] · `studio/clay_mode.py:607,756-778` vs manual `14-shortcuts.md:100-121`; popup rows `main.py:3541` vs `3556,3580,3596`. Clay binds Ctrl+1/3/7 (±Shift opposite axis), Ctrl+5 (ortho), Ctrl+W (close tab) — none in the manual's Clay table; six axis views are effectively undiscoverable. **Fix:** add the rows (the `tests/manual` gate covers numbering, not table contents) and make the popup rows uniform.
- **UX-14 — No bulk retry for failed jobs — the one bulk action the failures affordance sets you up for.** [A1-M15] · `panes/library.py:484-505,1065-1135` · confidence high (gap), medium (priority). "N jobs failed – show" plus Select-all exist, but the bulk bar offers only Export/Save/Delete; `run_action(..., "retry")` already exists per job. **Fix:** when the ticked set contains failed jobs, add "Try again (N)" looping the existing action.
- **UX-15 — Profiles' contextual help can attach to and overlap the global row** — `help_button` assumes a preceding local item. [A2-UX-12] · `panes/profiles_panel.py:24-29`, `manual/render.py:62-76`; compare `panes/app_settings.py:54-60`. **Fix:** draw a local heading first or make help layout independent of prior-item state.
- **UX-16 — Profile taxonomy combos are unlabeled** though the 2D form already owns labels/grouping. [A2-UX-13] · `panes/profiles_panel.py:174-180`, `panes/settings_2d.py:90-127`. **Fix:** reuse `GUIDANCE_GROUPS`, field labels, and labeled controls.
- **UX-17 — Cancel/Quit can silently discard profile drafts; drafts are absent from the dirty-document guard.** [A2-UX-14] · `panes/profiles_panel.py:137-192,293-296`, `studio/main.py:1874-1881`. **Fix:** compare draft to origin, guard Cancel/Quit, include drafts in recovery (UX-05).
- **UX-18 — Meaningful muted copy uses disabled text at ~3.20:1 (dark) / 2.55:1 (light) contrast.** [A2-UX-15] · accessibility · `studio/theme.py:136-146`, `studio/tokens.py:135-161`, `studio/widgets.py:1331-1342`. **Fix:** an opaque WCAG-qualified secondary-copy role; reserve disabled styling for disabled controls.
- **UX-19 — Text input ignores `TEXTINPUT`/`TEXTEDITING` and drops non-BMP code points.** [A2-UX-16] · accessibility · `studio/imgui_backend.py:291-327`. **Fix:** IME composition + positioning, full Unicode input, font-fallback tests.
- **UX-20 — Home suppresses the only global progress card with ETA and Cancel**, leaving a non-clickable queue sentence. [A2-UX-17] · `studio/main.py:3193-3199`, `panes/landing.py:258-271,440-454`, `panes/overlay.py:268-362`. **Fix:** keep the card on Home or make the queue row open a cancel/detail surface.
- **UX-21 — Quit asks a generic, usually untrue question, then chains per-document dialogs; it under-describes active downloads/exports.** [A1-L18, A2-UX-20] · `studio/main.py:3427-3439,1856-1889`. With nothing generating and nothing unsaved — the common state — the modal still warns "Anything still generating is cancelled," and confirming can trigger up to six more guard questions; the guards already protect everything real. **Fix:** one preflight summary listing active operations and every unsaved document, then confirm once; skip the generic confirm when nothing is running or dirty.
- **UX-22 — DPI is sampled only at startup; moving between mixed-DPI monitors does not rebuild fonts/style.** [A2-UX-21] · accessibility · `studio/tokens.py:14-16`, `studio/main.py:1439-1455`. **Fix:** handle display/DPI change, resample, rebuild scaled resources without losing layout state.

### Documentation

- **DOC-03 — gltfpack is called shipped/present but absent from a clean tracked checkout.** [A2-DOC-03] · `.gitignore:20`, manual `04-generating-meshes.md:157-159`, `16-configuration.md:28,118-119`, `doctor.py:233-240`, `tests/conftest.py:67-74`. The manual says vendored; `vendor/` is wholly ignored and a clean-checkout fixture assumes absence; setup covers Trellis but no gltfpack acquisition/checksum. **Fix:** ship/install it, or document an exact pinned release + checksum with an actionable Doctor remedy. (See also SVC-02, DST-01.)
- **DOC-04 — Manual index and the in-app loader disagree which part chapter 14 belongs to — and the manual gate doesn't cover grouping.** [A1-M11] · `docs/manual/00-index.md:24` vs `studio/manual/loader.py:27-31`. The index lists Keyboard shortcuts under Part I; `PARTS` puts `range(14, 19)` under "Setup & operations". CLAUDE.md claims the number decides order *and* part and that `tests/manual/` gates both directions — but `test_index_links_every_chapter` asserts only linkage. **Fix:** align `PARTS` or the index; extend `tests/manual/test_docs.py` to assert index-section membership against `PARTS`.
- **DOC-05 — INVARIANTS.md names the live `src/warlock/sweep.py` as deleted.** [A1-M12] · `docs/INVARIANTS.md:217`. The files actually deleted (commit `1fc1573`, v0.0.8) were `src/warlock/bench/{sweep,report,verdicts}.py`; today's `sweep.py` is the trellis `--band` sweep, CLI-wired and tested — the authoritative doc invites a future cleanup pass to delete a live module. **Fix:** correct the paths and disambiguate from the live `service/verdicts.py`.

### Tests / CI / product boundaries

- **TST-01 — The model-stage learned judge is declared but intentionally unbuilt.** [A2-TST-01] · `judge.py:75-88`, `service/judge.py:35-55`, `README.md:23`. The stage registry declares `model`; the service refuses training it as "declared and unbuilt", citing the now-missing `TODO.md §8`; README's judge language implies mesh judging exists. **Fix:** once the positive-corpus gate is met, implement the planned graded eight-view model probe (see the CAMERA.md plan); otherwise hide the declaration and label the judge reference/image-only. Keep the blocker in a live document.
- **TST-02 — GPU/Blender tests are not deselected by default despite saying they are.** [A2-TST-03] · `pyproject.toml:128-131`, `tests/test_conditioning_gpu.py:3-8`, `tests/test_prompt_encode_gpu.py:10-13`, `README.md:106-109`, `CLAUDE.md:10`. The `gpu` marker is registered with no default exclusion — on a provisioned host plain `uv run pytest` ran real GPU tests (observed in A2's run); on an unprovisioned host the critical paths silently skip; the test modules claim default deselection. **Fix:** make integration tests explicitly opt-in (e.g. default `-m "not gpu"`), and define a provisioned GPU/Blender lane run on a known schedule or before lifecycle changes.
- **TST-03 — No tracked CI/release workflow enforces the existing guards.** [A2-TST-04] · no `.github` workflow or equivalent exists; REL-01 is exactly what the tests were written to stop and they weren't run. **Fix:** a Windows CI matrix (lint, non-GPU tests, manual/docs checks, wheel/entry-point smoke if wheels are supported, explicit GPU/Blender lane) — or, minimally, a mandatory local release-preflight script gating version commits.

---

## Low

### Release / process

- **REL-02 — CHANGELOG has no 0.0.15 entry** though `Warlock v0.0.15` commits exist and the preamble claims the file is "the only record of what a version actually changed". [A1-L23] · confidence medium the skip wasn't deliberate. Fix: backfill from git or annotate the gap.
- **RUN-04 — Some winjob ctypes prototypes rely on implicit conversions.** [A2-NAT-01] · `winjob.py:101-123,284-318`. Handle-consuming Win32 calls lack complete `argtypes`/`restype`; implicit integer conversion for pointer-sized handles is fragile. Fix: declare full signatures; capture `GetLastError` before it can be overwritten.

### Model / VRAM lifecycle

- **MDL-16 — On a failed conditioned job, the exception traceback pins the ControlNet/IP-encoder stack past every reclaim point.** [A1-L2] · `text2image.py:434-447,575-577,794-796`, `queue.py:1065-1076`. The `gc.collect()`/`empty_cache()` passes run while the propagating exception still references the frames holding ~2.5–3.7 GiB of conditioning tensors; bounded and self-correcting, but a real gap in the drop-every-reference doctrine on the error path. Fix: strip `exc.__traceback__` (as `_on_task_done` already does) and run one pool trim in the error branch.
- **MDL-17 — `BaseModel.vram_gib` excludes the adapters `_load_loras` eagerly places on-device.** [A1-L3] · `models.py:222-225`, `text2image.py:271-344`. Base distillation LoRA (~0.8 GB) plus every fitting style LoRA load unconditionally; the "deliberately conservative" 7.0 GiB under-prices by up to ~1.5 GiB — the direction of error the registry's own comment forbids. Fix: fold a per-family adapter allowance into the estimate, or document the exclusion beside `vram_gib`. (Interacts with MDL-07's move to lazy loading.)
- **MDL-18 — Registry-only LoRA discovery is deliberate but not framed as a product boundary.** [A2-MDL-14] · `text2image.py:298-344`, `models.py:618-703`. Users of local image tools may assume dropping in a safetensors works. Fix: document registry-only support prominently, or build a manifest-based importer (validate family, sanitize adapter names, record hashes/trust, never execute code on import).

### Concurrency / DB

- **CON-04 — `_stage_link`'s fallback writes served GLB names in place on filesystems without hard links.** [A1-L4] · `queue.py:99-108` (used by `_q_generate.py:621-625` remesh-restore). Where `os.link` fails wholesale (exFAT/network `WARLOCK_DATA_DIR` — the case the fallback exists for), `shutil.copyfile` truncates the served `model.glb` in place; a crash mid-copy leaves a torn file on a job about to be `done`; the docstring claims `os.replace` semantics only the link branch has. Fix: temp sibling + `os.replace`, mirroring `optimize.staged_copy`.
- **CON-05 — `trash_job`'s claimed-in-the-gap comment describes behavior the code doesn't have.** [A1-L5] · `service/_jobs_lifecycle.py:309-319`. If the worker claims between the snapshot and `cancel()`, the cancel *succeeds* and the job is trashed — but `request_cancel` was never sent, so the worker burns the full reconstruction before `finish()` returns False. Final state correct; minutes of GPU wasted; the comment asserts the opposite. Fix: route the queued branch through `cancel_job`, or fix the comment.
- **CON-06 — A corrupt `jobs.params` blob on a queued row wedges dispatch permanently.** [A1-L6] · `db.py:1239-1243`, `queue.py:735-753`. `_to_dict` parses `params` bare; `next_queued()` raises and the worker loop retries the same oldest row forever, starving everything behind it. The `_blob` tolerance exists for the other three JSON columns after this exact failure class. Reachable only via hand-edited DB or disk corruption. Fix: log-and-substitute `{}`, or mark the row `error` at dispatch.
- **CON-07 — `_discard_artifacts` runs blocking file I/O on the event loop.** [A1-L7] · `queue.py:1081,1093`. For a cancelled re-texture it `rmtree`s ~24 rendered/baked images on the `warlock-loop` thread that hosts dispatch and cancellation; every DB write in the same `finally` goes through `to_thread`, the unlinks don't. Fix: `await asyncio.to_thread(self._discard_artifacts, job)`.

### Service layer / pipelines

- **SVC-01 — Seven writers still stage through a non-dotfile `.tmp` with no `finally` cleanup — the defect `_save_source` documents as fixed.** [A1-L8] · `service/files.py:151-153,209-212,276-280,513-515`, `service/derive.py:494-496`, `pipelines/postprocess.py:250-252`, `judge.py:213-226`. ENOSPC/PermissionError mid-write strands a visible `.tmp` (up to ~22 MB) forever; `save_edited_image`'s *fixed* tmp name is shared by concurrent callers, so two racing saves could rename a torn file onto the served `input.png`. Fix: adopt the `_save_source` shape (dot-prefixed unique temp + `finally` unlink) at all seven sites.
- **SVC-02 — `doctor` checks only the existence of the two vendored executables, never their version.** [A1-L9] · `doctor.py:196-203,233-240`. `_warlockc_check` exists precisely because gitignored vendor dirs routinely hold stale builds — an argument that applies verbatim to `trellis-server.exe` (hint pins "vendored build: v0.5.4" with nothing verifying it) and gltfpack. Fix: record/compare a version string or hash pin as a non-fatal "stale vendored build" detail. (Second-instance detection moved to RUN-01; see also DOC-03.)
- **SVC-03 — `warlock sweep` prints a pointer to a deleted file.** [A1-L10] · `sweep.py:184` — "See LEFTOVERS.md section 2." is user-facing terminal output; the file was deleted 2026-08-10. Fix: point at the live measurements doc.
- **SVC-04 — `service/judge.py` refuses a bad stage with a raw `ValueError`** — the only service-layer refusal outside the `service.errors` hierarchy. [A1-L11] · `service/judge.py:49-57`. Fix: `errors.Invalid(..., field="stage")` or document it as an assertion.
- **SVC-05 — The fetch child's stderr is discarded, so a crash before `main()` reports only an exit code.** [A1-L12] · `service/downloads.py:361-370,453-456`, `pipelines/fetch_worker.py:218-219`. A child dying before `main` (broken venv import; the `json.loads(sys.stdin.read())` outside the try) writes no `result.json` and its traceback goes to DEVNULL. Fix: pipe stderr into the fallback detail (as `rigging.run_worker` does); move the stdin parse inside the error-reporting path.
- **SVC-06 — `export_to_folder` copies onto consumed names in place.** [A1-L13] · `service/export.py:101-105`. `WARLOCK_EXPORT_DIR` exists to be watched by a game project; `copyfile` truncates first, so a hot-reloading engine can read a torn GLB — outside the staged-writes invariant's letter, inside its reasoning. Fix: temp + `os.replace` per file.
- **SVC-07 — `prompt_preview` tolerates only `(ImportError, OSError)` around the tokenizer load.** [A1-L14] · `service/system.py:158-172` · confidence medium. A corrupt tokenizer dir raises `ValueError`/`JSONDecodeError` out of transformers, turning the live preview into an error toast every refresh. Fix: widen to `except Exception` with a debug log, or route through `fetch.present` first.

### Studio UI

- **UX-23 — "New 3D model" and "New Model" sit together but the latter means Clay.** [A2-UX-18] · `panes/landing.py:509-524`. Fix: "Generate 3D" / "Model in Clay" with one-line descriptions.
- **UX-24 — Blank prompt/name submission stays enabled and silently does nothing.** [A2-UX-19] · `studio/dialogs.py:342-365`. Fix: disable Save and show "Name required."
- **UX-25 — The "Queued — N jobs in line" toast counts from the stale page.** [A1-L16] · `main.py:1046-1051`. `invalidate()` only marks dirty; the count misses the job this submit created. Fix: `waiting + 1` or toast next tick.
- **UX-26 — `submit_promotion` drops a refused submit silently.** [A1-L17] · `panes/settings_3d.py:477`. The shared `"submit"` task key means an Accept from the matte modal landing during an in-flight create is refused by key-dedupe — nothing queued, nothing toasted, modal closes looking like success. Narrow window. Fix: check the return and toast/retry.
- **UX-27 — The library keyboard stops at navigation.** [A1-L19] · `main.py:1751-1757`, `panes/library.py:807-813`. Up/Down/Enter work; Delete/favourite/rename are mouse-only, though delete-to-trash is deliberately confirm-free ("the trash *is* the confirmation") so a Delete binding is exactly as safe as the menu item. Fix: bind Delete (and perhaps F) in the fall-through where the arrows live. (Subsumed by UX-02 if the full keyboard-nav program runs.)

### Distribution

- **DST-03 — Base install exposes the Studio command even when Studio dependencies are optional.** [A2-DST-03] · severity medium-low · `pyproject.toml:13-86`, `cli.py:14-20,53-62`. A base-only install ends in an import traceback rather than an actionable remedy. Fix: catch the optional-dependency import failure and print the supported install command; if wheels go public, complete project metadata and license file.

### Documentation

- **DOC-06 — INVARIANTS.md's inventory of `docs/` omits MODELS.md** ("four things now"; there are five) and its "eighteen files still cite `TODO.md §N`" is now 17. [A1-L20] · `INVARIANTS.md:218`. Fix together with DOC-01 (same paragraph).
- **DOC-07 — Manual ch. 21 tells extenders to declare a `download` field that is actually a derived property.** [A1-L21] · `21-extending.md:34-35` vs `models.py:233-237`. `download=` on the frozen dataclass is a TypeError; omitting `fetch` makes the model unfetchable in-app. Fix: reword to `fetch: tuple[Fetch, ...]`.
- **DOC-08 — "Only two fatal startup checks" is wrong on small-VRAM hosts.** [A1-L22] · `15-installation.md:67-68`, `18-troubleshooting.md:78` vs `doctor.py:302` — `_vram_check` is fatal whenever the budget can't hold a lone trellis run. Fix: qualify the sentence.
- **DOC-09 — `docs/REPORT.md` is an undated pre-implementation research note whose recommendation the project rejected** (ComfyUI-as-backend over HTTP/WebSocket); nothing marks it historical. [A1-L24, A2-DOC-05]. Fix: status/date/supersession banner; optionally move under an archive/research namespace.
- **DOC-10 — INVARIANTS.md overstates `promote_to_model`'s admission** ("admits on weights as well" — it calls only `check_vram`; behaviorally correct and test-pinned, but the authoritative file claims a check the code doesn't make). [A1-L25]. Fix: correct the sentence.
- **DOC-11 — Two stale prose fragments in code:** `service/__init__.py:5-7` still describes deleted "(transitional) HTTP routes"; `scripts/screenshot_modes.py:7` says "eight modes" — there are thirteen (the script itself derives from `modes.KEYS`). [A1-L26].
- **DOC-12 — Persisted preferences conflict with "there is no config file" wording.** [A2-DOC-04] · `README.md:100`, `16-configuration.md:3`, `studio/settings.py:1-27`, `17-app-settings.md:3-5`. Fix: "no engine config file; runtime configuration is env vars; Studio UI preferences persist separately."

### Tests / hygiene

- **TST-04 — Maintenance hotspots are very large** — no production stubs found, but frame loop, navigation, lifecycle, mode dispatch, shortcuts, viewer adoption, and modal flows converge in a ~4,000-line `main.py` (~196 KB); `widgets.py`, `library.py`, `inker_mode.py`, `db.py`, `review_mode.py`, `state.py`, `queue.py` are similarly concentrated. [A2-TST-02] · severity low-medium. Fix: extract cohesive seams opportunistically — modal/input ownership, selection/view sync, recovery, model maintenance, shutdown — without duplicating business logic into panes.
- **HYG-01 — `examples/` PNGs are referenced by nothing** (player, two sprite sheets, tileset — no hit in src/tests/scripts/docs/README). [A1-L27] · confidence medium. Plausibly deliberate demo inputs. Fix: link from the sprite-synthesis/Plotter manual chapters, or delete.

---

## Cross-audit reconciliation

**Severity disagreements (resolved above, both ratings recorded):**

| Finding | A1 said | A2 said | Merged | Why |
|---|---|---|---|---|
| MDL-01 uninstall race | Medium (hardening) | High | High | A2's structural argument: loop serialization ≠ thread serialization; three compounding defects |
| MDL-02 shutdown races | Low (narrow window) | High | High | A2 partially reproduced the TaskRunner half; merged scope is broader than A1's single window |
| MDL-05 LoRA drop | Medium | High | High | Silent wrong output + `VECTOR_PARAMS` corpus contamination + sidecar regression |
| DST-02 model dir split | Medium | High | High | Gigabytes stranded from doctor's own paste-able remedy; migration skip makes it sticky |

**Tensions between one audit's "healthy" list and the other's findings** (all three resolved in favor of the finding, with the healthy claim qualified rather than deleted):

- A1 verified "grounding always runs on every path" — true, but ART-01 shows its *failures are swallowed* while the job stays `done`. Both hold; the invariant wording should say "grounding is attempted on every path; failures are currently non-fatal and invisible" until ART-01 is fixed.
- A1 verified "host commit is re-checked at dispatch" — true, but MDL-04 shows the re-check is a percentage ceiling, not the bytes the next allocation needs.
- A2 listed "resident-memory crediting" among strengths — the mechanism exists, but A1's MDL-06 shows it is gated to `kind=="text"` only.

**Method differences worth remembering:** A1 ran nothing (its two test-failure claims were static reads — subsequently confirmed by A2's actual run). A2 ran the full suite, targeted GPU tests, and a TaskRunner shutdown repro. Where only one audit looked at a subsystem, its findings are single-source: A1 alone covered hygiene/dead-code, the pose/library/progress internals, and the low-level service staging sites; A2 alone covered supply-chain, packaging/wheel, accessibility, IME, and DPI.

---

## What's healthy (verified, not assumed — merged)

- **Hygiene (A1):** zero orphaned top-level definitions (1,702 public + all private defs, two whole-repo scans); zero live FIXME/XXX/HACK; no stub bodies; no tracked bytecode; ruff clean; all 14 `scripts/` resolve statically; bench recipes/suites match schemas; all 9 native exports bound at ABI 7 with fallbacks intact and `native.available()` guards at every call site.
- **Invariant anchors (A1, several re-confirmed by A2's test runs):** `HF_HUB_OFFLINE` first thing in the package; `merge_params` under one lock hold; `DERIVED_PARAMS` complete against the enumerated worker-recorded set; `VECTOR_PARAMS` in `vectors.py` with import direction pinned by AST test; winjob scan covering all seven spawn sites (no other spawn primitive in `src/`); headless purity of all four packages pinned; manual numbering gated; test basenames unique; `fetch_worker` imported only inside a fixture.
- **Model lifecycle:** transactional load; reference-drop doctrine; single-owner handoff; trellis stop-confirms-death and keeps the handle on failure; per-generate adapter re-application (no cross-job bleed, collisions structurally prevented and tested); admission at every door with dispatch re-check *(qualified by MDL-04, MDL-06)*; all cancel windows; adapter re-enable/disable ordering accounts for persistent PEFT state; per-call conditioning cleanup verified on real GPU (A2).
- **DB/queue:** every `JobStore` method under the RLock; single connection; cancel-vs-finish closed at the DB and tested; `reconcile_startup` wired and tested; staged writes on every served name in the worker; `ProgressBus` id-checked; migrations append-only and idempotent.
- **Service:** export invariants hold end-to-end (`source.glb` written once; optimize-then-normalize; grounding attempted on every path *(see ART-01)*; derived exports under per-artifact locks, invalidated by re-optimize); refusal contract holds at every user-reachable door except SVC-04; no studio code writes the store directly; no pipeline imports service; offline integrity confirmed (`httpx` reaches only the local trellis port; LPIPS deliberately excluded; matting/pose/DINO loads `local_files_only=True`); uninstall has strong path containment and shared-weight claim accounting.
- **UI:** texture lifecycle fully paired; frame-loop discipline (GLB parse off-thread *(except Compare — UX-04)*, storage walks/exports/pickers through TaskRunner, stat-gated per-frame caches); uid-addressed undo with serial head and byte budget; modal/dialog queue discipline; field-error routing with fold-opening; `push_id` discipline; task errors, toast history, contextual manual links, reduced-motion, themes, soft-delete/Undo thoughtfully implemented.
- **Process isolation:** Blender, matting, fetch, and Trellis behind process boundaries; children assigned to the kill-on-close job; the manual has real parser/link/coverage tests (DOC-01 is a specific blind spot, not absent discipline); the release tests correctly caught the version drift when actually run.

## Coverage limits (merged)

A1 executed nothing (its VRAM arithmetic is from the code's own constants; WDDM free-memory virtualizes, so MDL-06's refusal may be intermittent); its dead-code scans are whole-word textual and miss uncalled class methods; `review_mode.py`, the inker/plotter canvas panes, and `viewer/` internals got targeted greps; manual chapters 05–11 were lightly checked. A2 ran the suite and targeted GPU tests but its dynamic coverage was one host/one configuration; its concurrent-work note records that unrelated bench/metrics edits appeared in the tree during its final verification (they are still present, uncommitted, and are not covered by either audit). Neither audit reviewed the uncommitted bench work or `docs/measurements/2026-08-11-perceptual-hash-floor.md`.

---

## Source-finding index

Every source ID and where it landed:

**AUDIT.md (A1):** H1→REL-01 · H2→DOC-01 · H3→MDL-06 · H4→DOC-02 · M1→MDL-05 · M2→CON-01 · M3→CON-02 · M4→MDL-15 · M5→CON-03 · M6→RUN-02 · M7→MDL-01 · M8→UX-11 · M9→UX-12 · M10→DST-02 · M11→DOC-04 · M12→DOC-05 · M13→MDL-14 · M14→UX-13 · M15→UX-14 · L1→MDL-02 · L2→MDL-16 · L3→MDL-17 · L4→CON-04 · L5→CON-05 · L6→CON-06 · L7→CON-07 · L8→SVC-01 · L9→SVC-02 (+RUN-01 note) · L10→SVC-03 · L11→SVC-04 · L12→SVC-05 · L13→SVC-06 · L14→SVC-07 · L15→UX-04 · L16→UX-25 · L17→UX-26 · L18→UX-21 · L19→UX-27 · L20→DOC-06 · L21→DOC-07 · L22→DOC-08 · L23→REL-02 · L24→DOC-09 · L25→DOC-10 · L26→DOC-11 · L27→HYG-01

**AUDIT2.md (A2):** REL-01→REL-01 · RUN-01→RUN-01 · ART-01→ART-01 · ART-02→ART-02 · CFG-01→RUN-03 · MDL-01→MDL-01 · MDL-02→MDL-02 · MDL-03→MDL-03 · MDL-04→MDL-04 · MDL-05→MDL-05 · MDL-06→MDL-07 · MDL-07→MDL-08 · MDL-08→MDL-09 · MDL-09→MDL-10 · MDL-10→MDL-11 · MDL-11→MDL-12 · MDL-12→MDL-13 · MDL-13→MDL-14 · MDL-14→MDL-18 · NAT-01→RUN-04 · UX-01..10→UX-01..10 · UX-11→MDL-14 · UX-12→UX-15 · UX-13→UX-16 · UX-14→UX-17 · UX-15→UX-18 · UX-16→UX-19 · UX-17→UX-20 · UX-18→UX-23 · UX-19→UX-24 · UX-20→UX-21 · UX-21→UX-22 · DST-01→DST-01 · DST-02→DST-02 · DST-03→DST-03 · DOC-01→DOC-01 · DOC-02→DOC-02 · DOC-03→DOC-03 · DOC-04→DOC-12 · DOC-05→DOC-09 · TST-01→TST-01 · TST-02→TST-04 · TST-03→TST-02 · TST-04→TST-03

---

## Outcomes (2026-08-12)

Worked through `AUDIT_PLAN.md` phases 0–5. Decision gates were answered by the
user: **D1** revert `pyproject` to 0.0.21 · **D2** keep `docs/TODO.md` deleted and
correct the docs · **D3** supply chain at full depth (pin + manifest + vendored
BiRefNet) · **D4** source-checkout-only · **D5** skip the accessibility program ·
**D6** a local release-preflight script.

Suite at close: **6349 passed, 17 skipped, 17 deselected** (the deselected are the
new opt-in `gpu` lane), `ruff` clean. Every phase ended on a green full run.

### Fixed

**Phase 0** — REL-01, REL-02, DOC-01…DOC-12, DST-02, TST-02, SVC-03.
**Phase 1** — MDL-01, MDL-02, MDL-04, MDL-05, MDL-06, MDL-07, MDL-12, MDL-15,
MDL-16, MDL-17.
**Phase 2** — RUN-01, RUN-02, RUN-03, ART-01, ART-02, CON-01, CON-03, CON-04,
CON-05, CON-06, CON-07, MDL-09, MDL-13, MDL-10 *(partial — see below)*.
**Phase 3** — CON-02, UX-03, UX-04, UX-07, UX-08, UX-09, UX-10, UX-11, UX-12,
UX-13, UX-14, UX-15, UX-16, UX-17, UX-21, UX-23, UX-24, UX-25, UX-26, UX-27,
MDL-14 *(cancel half)*.
**Phase 4** — DST-01, DST-03, TST-03, MDL-03 *(partial)*, MDL-08 *(partial)*.
**Phase 5** — SVC-01, SVC-04, SVC-05, SVC-06, SVC-07, RUN-04, MDL-18, HYG-01.

New primitives, each with its own tests: `warlock/leases.py` (the read/use vs
exclusive-maintenance model-store lease, held by *threads*), `warlock/instance.py`
(per-home OS single-instance lock + native alert), `scripts/preflight.py` (the
release gate, verified to reproduce REL-01's exact failure).

### Three findings whose fix differed from the recommendation

* **UX-07.** The plan says make Keep/Cancel the safe default. It also has to
  change what *Enter* does: focus alone is not the hazard, `_enter_pressed()`
  mapping to the destructive button is. Enter now cancels.
* **MDL-07 / MDL-17 interact.** Making optional adapters lazy shrank MDL-17's
  under-pricing from "every fitting adapter" to "the required one plus at most
  the selected one", so `vram._adapter_cost` prices exactly that rather than the
  eager sum the finding assumed.
* **MDL-05.** Implemented *both* halves the plan offered as alternatives — the
  store-generation counter *and* late loading at a safe boundary — because lazy
  adapters (MDL-07) make the second one nearly free, and it is the half that
  removes the failure rather than detecting it.

### Deferred, with reasons

* **UX-01, UX-02, UX-18, UX-19, UX-22** (accessibility and responsive UI) — D5,
  explicitly skipped. Still open, still correct; nothing here works around them.
* **UX-05, UX-06, UX-20** (Phase 4D crash-recovery service, the fatal-error
  dialog, and Home's progress card). UX-06 is partly served by
  `instance.alert`, which is a native dialog independent of imgui and GL and is
  the piece 4D would build on. The document-journal service itself is unbuilt.
* **MDL-03 / MDL-08 remainder.** `Fetch.revision` is threaded end to end
  (registry → planner → worker → rendered command) and a completion manifest is
  written after publication, with Doctor separating "files found" from "files
  usable" and rejecting zero-length weights. **Not** done: actual commit SHAs
  are not yet filled in for any entry (`revision=""` throughout — the mechanism
  is pinned open, the values need a maintainer to choose them), hashes are not
  verified against the manifest, and BiRefNet is **not** vendored, so
  `trust_remote_code=True` still stands. D3 asked for that depth; it is a
  standalone project and this pass built the carrier for it.
* **MDL-10 remainder.** Partial-success is now reported precisely and stranded
  staging is swept; the journaled whole-selection transaction is not built.
* **MDL-11** is covered *in effect* by the exclusive maintenance lease rather
  than by a separate per-model-root lock — one primitive, both guarantees.
* **TST-01** (the model-stage judge) and **TST-04** (file-size hotspots) are
  product/standing items rather than defects; unchanged.
* **DOC-03** gltfpack is now documented honestly with a pinned build and
  checksum, but the binary is still not shipped — D4 says source-checkout-only,
  so acquiring it by hand is the supported path.

### One correction to the audit itself

DOC-10 said INVARIANTS.md "overstates `promote_to_model`'s admission". True, but
the underlying behaviour is *right*: a promotion reconstructs an `input.png`
that already exists, loads no image model, and so has no per-job weights to
admit on. The doc now says that explicitly, so the next reader does not "fix"
the code to match the old sentence.
