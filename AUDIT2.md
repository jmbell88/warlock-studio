# Warlock Studio Audit 2

**Audit date:** 2026-08-11  
**Repository revision reviewed:** `de87838` (`Warlock v0.0.21`)  
**Review posture:** read-only. This audit changed only this report; it did not change application code, tests, configuration, or existing documentation. Unrelated concurrent edits appeared in the shared workspace during the review and were left untouched.  
**Review team:** primary review plus three read-only sub-agents covering runtime/model safety, UX/UI, and documentation/test/repository quality.

## Executive summary

Warlock Studio is an unusually ambitious and substantially implemented desktop application. It has real depth rather than a facade: a serial GPU pipeline, an offline-first model registry, Blender and native-worker isolation, thirteen coherent modes, staged artifact writes, a service boundary, extensive headless editors, and more than six thousand tests. Many previously dangerous details have explicit tests and written rationale.

The current checkout should nevertheless **not be treated as release-ready**. The most important reasons are:

1. The release identity is inconsistent and the full test suite is red: package metadata says `0.0.22`, while the application and changelog say `0.0.21`.
2. Model lifecycle safety is strong during an ordinary serial job, but unsafe at maintenance and shutdown boundaries. Model uninstall can race queue admission; shutdown can unload a pipeline while its blocking model thread continues; a newly downloaded LoRA can be silently ignored by a resident pipeline.
3. Model downloads are neither pinned nor cryptographically verified. BiRefNet includes repository Python that is later run with `trust_remote_code=True`; the worker process is a crash/memory boundary, not a security sandbox.
4. A second Studio instance is only logged, not refused. It shares the job database and model store and may identify the first instance's Trellis server as an orphan and terminate it.
5. Canonical mesh normalization/grounding can fail while the job is still marked `done`, leaving a user with an engine-facing artifact at the wrong scale or origin and no visible degraded state.
6. The documented/source-checkout installation model conflicts with wheel behavior and ignored native binaries. A clean wheel cannot find the binaries at the paths runtime derives.
7. At high UI scale, fixed workspace geometry can make controls unreachable. Full keyboard navigation is disabled, creating a broad accessibility gap.
8. Library selection and viewport state can disagree, and the Compare path performs synchronous parsing/GPU upload without the primary viewer's error containment.
9. Crash recovery exists only for Inker, although Clay, Plotter, Packwright, Poser, inspector pose edits, and profile drafts can all contain unsaved work.
10. Authoritative documentation is internally inconsistent: the declared sole roadmap is missing, current guidance disagrees about the default image model, and pasteable model commands target a different directory from the actual default.

### Direct answer: are model and LoRA loading/unloading safe?

**The ordinary, single-job loading path is largely well designed.** `Text2Image.load()` constructs and configures a local pipeline before publishing it; failed loads catch `BaseException`, drop references, collect garbage, and clear the CUDA cache. Loads use local paths and `local_files_only=True`. Base-model switches unload first. Adapter enable/disable ordering accounts for PEFT state persisting across jobs. Per-call ControlNet/IP-Adapter state is torn down in a `finally`-equivalent path. Real GPU tests exercised conditioning identity and VRAM return successfully in this audit.

**Unloading is not safe under every path that can currently call it.** Idle eviction is reasonably safe because the serial worker is idle. Uninstall and shutdown are not equivalently protected: the actual model operation runs in `asyncio.to_thread`, and event-loop serialization does not serialize the underlying model thread. A model-store mutation barrier and explicit in-flight model-operation ownership are needed before unload/delete can be called safe. Supply-chain provenance and integrity also need to be part of the definition of a safe load.

## Verification snapshot

| Check | Result |
|---|---|
| `uv run ruff check .` | Passed |
| `uv run pytest -q` | **2 failed, 6,244 passed, 17 skipped, 58 warnings** in 452.87 seconds |
| Runtime/model focused tests | 168 loader/VRAM/fetch/uninstall/Trellis/WinJob tests passed; 19 targeted queue resource/shutdown tests passed |
| UX-focused tests | 127 passed |
| Documentation/manual focused tests | 42 passed, 3 skipped, despite the missing declared roadmap |
| Working tree before report | Clean |

The two full-suite failures are:

- `tests/test_changelog.py:99`: changelog `0.0.21` does not match project `0.0.22`.
- `tests/test_offline.py:46`: `warlock.__version__ == 0.0.21` does not match installed distribution `0.0.22`.

The full suite unexpectedly ran provisioned GPU tests. This is discussed under TST-03.

**Concurrent-work note:** final status checks showed separate work appearing under `AUDIT.md`, `src/warlock/bench/`, `tests/test_bench_metrics.py`, and `docs/measurements/2026-08-11-perceptual-hash-floor.md`. This activity began after the audit's initial clean-tree check and was still changing during final verification. It was not produced by this audit, was not reviewed as part of revision `de87838`, and was left untouched.

## Severity and priority

- **Release blocker:** the current revision should not be published or represented as a coherent release.
- **High:** credible data loss, process crash/hang, security/supply-chain exposure, incorrect artifact publication, or a core workflow/accessibility failure.
- **Medium:** substantial reliability, maintenance, integrity, or recurring UX cost with a narrower trigger or usable workaround.
- **Low:** hardening, clarity, or maintainability improvement whose current failure impact is limited.

The IDs below are stable references for remediation work; they are not implied issue numbers.

---

## 1. Release, process, and artifact integrity

### REL-01 — Release identity is inconsistent and guarded tests fail

**Severity:** Release blocker  
**Evidence:** `pyproject.toml:3`, `uv.lock:1697-1698`, `src/warlock/__init__.py:12-15`, `CHANGELOG.md:9`, `tests/test_changelog.py:99`, `tests/test_offline.py:46`

Project and installed metadata report `0.0.22`; `warlock.__version__`, the title/About surface, and the leading changelog entry report `0.0.21`. Home's “What's new” content therefore describes the previous version while packaging identifies a new one.

**Recommendation:** add the real `0.0.22` changelog entry, make runtime version reporting derive from installed metadata with a source-tree fallback, and enforce the existing tests as a mandatory release preflight. This is the clearest evidence that the repository also needs tracked CI/release automation (TST-04).

### RUN-01 — Multiple Studio instances are detected but allowed to corrupt each other's assumptions

**Severity:** High  
**Evidence:** `src/warlock/studio/main.py:4180-4228`, `src/warlock/pipelines/trellis.py:210-216,312-352`, `docs/manual/18-troubleshooting.md:157-158`

Startup sees a live session marker and writes only a warning to the log. The second instance continues. The manual itself states that both instances share a database and engine port and “the second will lose fights over both.” This is worse than a vague contention risk: Trellis port reclamation considers a matching configured executable path proof that the listener is Warlock's orphan. Two normal instances use the same executable path, so the second can terminate the first instance's live server.

The session marker is also a single file. A second process overwrites it; either process can later remove it, weakening crash attribution.

**Recommendation:** acquire a real per-home, OS-level single-instance lock before migration/database/runtime startup. If multi-instance support is desired instead, assign per-instance ports, make model-store mutations process-safe, preserve per-process markers, and specify database/queue ownership. Do not use executable identity alone to kill a listener owned by another live Studio process. Surface the refusal in a native dialog, not only `warlock.log`.

### ART-01 — Canonical mesh normalization can fail while the asset is marked done

**Severity:** High  
**Evidence:** `src/warlock/_q_mesh.py:109-152`, `src/warlock/service/_jobs_create.py:475-504`, `CLAUDE.md:26`, `README.md:34,111`

Normalization is supposed to apply requested size, center X/Z, and put the model on the floor. The worker catches every exception from this step, logs it, and continues. Built/imported models do the same and still insert a `status="done"` model row. Mesh reporting may also fail without changing the terminal state.

The non-fatal design protects a successfully reconstructed `source.glb`, but it overstates the status of `model.glb`. A user can export a visibly successful asset whose pivot or scale is wrong in an engine, with the only explanation hidden in logs. This contradicts current documentation saying grounding always runs and outputs are grounded.

**Recommendation:** keep the reconstruction, but distinguish `done` from `done_degraded` (or carry structured artifact health). Surface which canonical steps failed, offer Retry/Repair, and prevent an unnormalized file from being presented as the canonical engine artifact without a warning. Record normalization and report failures in provenance.

### ART-02 — Queue “happy path” fakes mostly exercise swallowed degradation paths

**Severity:** Medium  
**Evidence:** `tests/conftest.py:239-272,313-364`, `src/warlock/_q_mesh.py:100-153,229-259`, `tests/test_fakes_match_real_signatures.py:16-34`

The shared fake Trellis server writes the bytes `fake-glb`, and fake text-to-image commonly writes `fake-png`. Normal post-processing catches parse, normalize, audit, and report failures and allows completion, so many tests asserting a job becomes `done` are not testing successful artifact processing at all. The fake contract test checks signatures, not artifact validity or semantic behavior.

**Recommendation:** make default fakes emit tiny structurally valid PNG and GLB fixtures. Put malformed artifacts behind explicit tests that assert degraded/error behavior and logged diagnostics. A successful queue test should assert that normalization/audit/report were reached without swallowed exceptions.

### CFG-01 — Malformed environment configuration can crash before diagnostics

**Severity:** Medium  
**Evidence:** `src/warlock/config.py:30-67,155-177,195-205,328-351`

Several fields call raw `int()` or `float()` inside dataclass factories. A typo in a port, retry count, timeout, or threshold can raise during configuration construction, before normal Doctor/startup messaging exists. Other fields, such as optional VRAM floats, silently turn malformed input into `None`, so the handling policy is inconsistent.

**Recommendation:** centralize typed parsing and range validation, retain the source environment-variable name, and report all invalid values in one actionable startup/Doctor surface. Do not silently turn a malformed explicit safety limit into “unset.”

---

## 2. Model, LoRA, VRAM, download, and native-runtime safety

### MDL-01 — Model uninstall has a check/use race with queue admission and generation

**Severity:** High  
**Evidence:** `src/warlock/service/downloads.py:241-299`, `src/warlock/queue.py:786-812`, `src/warlock/_q_generate.py:196-216`, `src/warlock/studio/panes/app_settings.py:310-317`

Uninstall queries the database once for queued/running jobs, then later asks the worker loop to unload and starts deleting. A job can be admitted after the query. More subtly, the image operation itself is in `asyncio.to_thread`; scheduling unload on the worker loop does not serialize it with that underlying thread. The Settings pane prevents two download/remove UI tasks from overlapping each other, but it does not create a queue/model-use barrier.

Possible outcomes include an unload concurrent with `load()` or `generate()`, deleted files during a load, two pipeline objects resident after a race, Windows file-lock failures, or an OOM.

**Recommendation:** introduce a service-level model-store read/write lease. Dispatch and every model operation hold a read/use lease; download, uninstall, repair, and update acquire an exclusive maintenance lease, pause claims, recheck queue/current state after acquiring it, unload, mutate, refresh model generation state, then resume. The lock must cover the model thread, not merely the event-loop coroutine.

### MDL-02 — Bounded shutdown can race or outlive blocking model and task threads

**Severity:** High  
**Evidence:** `src/warlock/queue.py:704-730`, `src/warlock/_q_generate.py:196-216`, `src/warlock/studio/runtime.py:196-234`, `src/warlock/studio/tasks.py:188-229`, `src/warlock/service/downloads.py:44`

Cancelling an asyncio task that awaits `to_thread` does not stop the worker thread. After the queue's 20-second grace period, shutdown cancels the coroutine and then unloads the image pipeline even though `from_pretrained` or sampling may still be running. A late load can publish after unload did nothing; an active sample can hold references while another thread clears the owner and calls `empty_cache()`.

TaskRunner similarly uses `ThreadPoolExecutor.shutdown(wait=False)` and removes pool threads from private `_threads_queues`. The audit reproduced that this does not make a blocked non-daemon worker stop keeping the interpreter alive: `shutdown(timeout=.05)` returned promptly, but the subprocess stayed alive until the worker was released. Downloads can wait up to four hours and currently have no cancellation UI.

**Recommendation:** provide cooperative cancellation for every long task and track subprocess handles so shutdown can kill and reap them. Never unload a pipeline until its model-operation lease is released. Move uninterruptible CUDA/native work to a killable process if bounded exit is a product requirement. Replace private `concurrent.futures` mutation with supported lifecycle control, and add subprocess-level tests that assert the interpreter actually exits while load, sample, dialog, and download operations are blocked.

### MDL-03 — Model repositories are unpinned and BiRefNet executes repository code

**Severity:** High  
**Evidence:** `src/warlock/models.py:105-176,909-944`, `src/warlock/pipelines/fetch_worker.py:164-189`, `src/warlock/pipelines/matting.py:225-247`

`Fetch` has no revision or digest. `snapshot_download()` follows the repository's current default revision. BiRefNet explicitly downloads `*.py`; loading later uses `trust_remote_code=True`. A future explicit Download click can therefore retrieve different Python and weights from those audited for this release.

The child process is valuable isolation for crashes and unreclaimable RSS, but it runs with the user's filesystem/network permissions. It is not a security sandbox. `uv.lock` does not pin the downloaded code.

**Recommendation:** give every registry fetch an immutable commit SHA, carry it through the worker and rendered manual command, and verify a signed/maintained artifact manifest of expected paths, sizes, and hashes. Store and display the installed revision, include it in generation provenance, and expose an update decision explicitly. Prefer vendored/audited model implementation code over `trust_remote_code`; if remote code remains, make trust visible and run it under a restricted low-privilege boundary.

### MDL-04 — Offloaded FLUX models can cross the Windows commit limit during admission-approved load

**Severity:** High  
**Evidence:** `src/warlock/models.py:524-559`, `src/warlock/pipelines/text2image.py:273-280`, `src/warlock/queue.py:391-410`, `src/warlock/vram.py:110-129`

The FLUX.2 checkpoint is approximately 16 GiB in host memory under model CPU offload. Current host-commit admission refuses only when the machine is already at the 90% ceiling. A machine at 80–89% can have far less than 16 GiB left, be admitted, and cross the limit during checkpoint allocation. On Windows that may result in OS process termination rather than a recoverable Python OOM—the exact failure the check is intended to prevent.

**Recommendation:** model both device and host peak footprint in every base spec. Require absolute free commit of `host_peak + measured safety margin` immediately before `from_pretrained` and again before large placement transitions. Continue percentage telemetry, but do not use a percentage threshold as a substitute for the bytes the next operation needs.

### MDL-05 — A newly downloaded LoRA can be silently ignored until the base pipeline reloads

**Severity:** High  
**Evidence:** `src/warlock/pipelines/text2image.py:298-396`, `src/warlock/service/downloads.py:170-206`, `src/warlock/studio/main.py:957-979`

Compatible style LoRAs are discovered and attached only while the base pipeline loads. A pipeline that was loaded before a LoRA download retains its old `_adapters` set. After Settings reports the LoRA installed, selecting it and generating immediately logs “not downloaded” and produces an unstyled image. The request is not visibly refused; the user's expected style is silently lost.

**Recommendation:** increment a model-store generation after successful download/uninstall/repair and make the worker invalidate or refresh the resident pipeline before the next job. Alternatively attach the selected LoRA lazily on demand at a safe model-operation boundary. Missing selected style should be a user-visible refusal, never a warning followed by a successful-looking job. Add a resident-pipeline → download → immediate-generation regression test.

### MDL-06 — Optional LoRAs are loaded eagerly; one corrupt unused file can brick every compatible base

**Severity:** Medium  
**Evidence:** `src/warlock/pipelines/text2image.py:298-344`

Every compatible style LoRA found on disk is loaded during base-pipeline initialization. Missing files are tolerated, but a corrupt or incompatible optional file raises inside the base load transaction. The user cannot generate without that style even if it was never selected.

**Recommendation:** lazily load only selected adapters, or pre-validate/quarantine each optional adapter and expose its health per Settings row. Keep required step-distillation LoRAs fatal because they define the base recipe. Test a corrupt optional adapter alongside a healthy base and an unrelated selected style.

### MDL-07 — “Present” means a few filenames exist, not that weights are complete or trustworthy

**Severity:** Medium-high  
**Evidence:** `src/warlock/fetch.py:438-497`, `tests/conftest.py:140-175`, `src/warlock/doctor.py:350-406`

Presence checks inspect a small sentinel set with `Path.exists()`. Test fixtures intentionally create empty files and production admission accepts them. In-app staging reduces interrupted-fetch risk, but manual commands, copied directories, disk corruption, an upstream layout change, or a hard-killed publication can all bypass that guarantee.

**Recommendation:** write a completion manifest only after verification against the pinned revision, and require it for “installed.” Validate every required artifact's expected size/hash. At minimum, reject zero-length safetensors, parse required JSON, and separate “files found” from “load probe passed” in Doctor and Settings.

### MDL-08 — Multi-model free-space admission charges the whole plan to only the first volume

**Severity:** Medium  
**Evidence:** `src/warlock/fetch.py:142-171,298-315`, `tests/test_fetch.py:218-244`

`disk_refusal()` totals every job but calls `free_gib()` only for `jobs[0].dest`. `WARLOCK_T2I_DIR` can put Turbo on another volume from `WARLOCK_T2I_ROOT`. A roomy first drive can therefore approve a large later write to a nearly full drive; the inverse can falsely refuse a valid plan.

**Recommendation:** resolve the volume identity for each destination, group planned bytes per volume, and enforce the headroom independently on every group. Add mixed-drive tests and report which destination lacks space.

### MDL-09 — Whole-selection download atomicity is promised but not implemented

**Severity:** Medium  
**Evidence:** `src/warlock/service/downloads.py:170-206`, `src/warlock/pipelines/fetch_worker.py:105-160,164-213`

The service describes half a plan as unacceptable, but it publishes each repository sequentially. If repository one succeeds and repository two fails, the first remains installed. For a checkpoint plus a required distillation LoRA, this can consume several GiB while the model remains unusable.

Within one repository, staged per-file publication is careful during Python exception unwinding. A hard process kill does not run that rollback, however, and deterministic `.fetch.part`/`.fetch.bak` locations can be stranded. Rollback failures are suppressed and the backup is deleted, which favors hiding the rollback failure over retaining recoverable prior data.

**Recommendation:** stage the entire selection, validate it, then publish through a journaled transaction. Use unique transaction directories and an incomplete marker. On startup, recover or quarantine interrupted transactions before presence checks. At minimum, return a precise partial-success result and provide safe Resume/Rollback instead of claiming all-or-nothing.

### MDL-10 — Model mutation serialization exists only as a UI convention

**Severity:** Medium  
**Evidence:** `src/warlock/studio/panes/app_settings.py:310-317`, `src/warlock/service/downloads.py:170-299`

The Models pane disables concurrent download/remove actions, but `service.downloads.download()` and `uninstall()` share no backend lock. Headless callers, future panes, shutdown races, or a second process can collide on staging, backup, or destination paths.

**Recommendation:** enforce a per-model-root mutation lock in the service/model-store layer, in addition to the broader generation/use lease in MDL-01. UI disabled state should reflect backend ownership, not define it.

### MDL-11 — Idle cache eviction is latched and can miss caches loaded later outside the GPU queue

**Severity:** Medium  
**Evidence:** `src/warlock/queue.py:816-850,1133-1139`, `src/warlock/service/derive.py:403`, `src/warlock/service/matte.py:225-230`

After one idle pass, `_caches_evicted` prevents future cache eviction until a GPU queue job completes. Matting can be loaded later by a TaskRunner service operation without a GPU job, leaving its roughly 1.5 GiB child resident indefinitely.

**Recommendation:** give each cache explicit loaded/last-used state and evict by that state, or notify a central cache manager whenever any path populates a cache. Test evict → load matte through a task-pool path → evict again.

### MDL-12 — Trellis error responses bypass response-size limits

**Severity:** Medium  
**Evidence:** `src/warlock/pipelines/trellis.py:464-491`

Successful bodies are streamed under a ceiling, but HTTP error responses use unbounded `await r.aread()`. A wedged or compromised local native server can exhaust host memory with a huge error body. The success path also holds chunks and then joins them, temporarily duplicating a response up to the configured 512 MiB limit—far above normal output sizes.

**Recommendation:** stream and cap error bodies too. Stream success directly to a temporary file or bounded buffer, validate the GLB incrementally, and set a measured ceiling based on legitimate maximum artifacts.

### MDL-13 — Model repair, update, verification, and download cancellation are missing

**Severity:** Medium  
**Evidence:** `src/warlock/studio/panes/app_settings.py:450-500`, `src/warlock/studio/tasks.py:71-140`, `src/warlock/service/downloads.py:44`

An installed row offers Remove but not Verify, Repair, or Update. A corrupt-but-present checkpoint has no direct recovery besides remove/re-download. Multi-gigabyte downloads can run for four hours with a progress bar but no Cancel action.

**Recommendation:** after pinning revisions, show installed/current/outdated/corrupt state, add Verify and Repair, and expose Cancel/Cancelling backed by child termination and journal-safe staging cleanup. Resume is preferable for large immutable downloads.

### MDL-14 — Custom model/LoRA discovery is deliberately absent but not framed as a product boundary

**Severity:** Low/product gap  
**Evidence:** `src/warlock/pipelines/text2image.py:298-344`, `src/warlock/models.py:618-703`

Only registry entries are loaded; arbitrary local safetensors are ignored. This is safer than blind directory scanning, but users familiar with local image tools may assume dropping in a LoRA is supported.

**Recommendation:** either document registry-only support prominently or build a manifest-based importer that inspects safetensors metadata, validates base/family compatibility, assigns sanitized adapter names, records hashes and trust state, and never executes arbitrary code during import.

### NAT-01 — Some Windows ctypes prototypes rely on implicit conversions

**Severity:** Low hardening  
**Evidence:** `src/warlock/winjob.py:101-123,284-318`

Several handle-consuming Win32 calls lack complete `argtypes`/`restype` declarations. Current Windows tests pass, but implicit integer conversion for pointer-sized handles is fragile across architectures and Python/runtime changes.

**Recommendation:** declare every invoked API's complete signature and consistently capture/log `GetLastError` before another call can overwrite it.

---

## 3. UX/UI and end-user workflow

### UX-01 — High UI scale can make workspace controls unreachable

**Severity:** High  
**Evidence:** `src/warlock/studio/main.py:46-51,112-120,2016-2031`, `src/warlock/studio/tokens.py:14-24`, `src/warlock/studio/layout.py:143-152`

The window's resize floor follows monitor DPI, but workspace geometry follows monitor DPI multiplied by the user's UI scale. The three-column workspaces reserve two 300-design-pixel sidebars and force a 300-pixel center. At the 1100-pixel minimum window, a 1.5× or 2× UI scale needs roughly 1350 or 1800 pixels before gutters. The header has a compact fallback; workspaces do not collapse, horizontally scroll, or convert sidebars to drawers.

**Impact:** users who enlarge the UI for accessibility can lose access to the right pane and its controls.

**Recommendation:** create responsive breakpoints: collapsible/drawer sidebars, one-pane navigation at narrow widths, and an always-reachable pane switcher. Render and interaction-test every mode at minimum size and 1.0×/1.5×/2.0× scale in both themes.

### UX-02 — Keyboard navigation is disabled across most of the application

**Severity:** High  
**Evidence:** `src/warlock/studio/dialogs.py:246-263`, `src/warlock/studio/imgui_backend.py:68-98,248-286`, `src/warlock/studio/main.py:3444-3624`, `src/warlock/studio/panes/app_settings.py:463-475`, `src/warlock/studio/panes/inker_layers.py:122-125`

ImGui keyboard navigation is not enabled and the backend maps only a narrow key set. Confirmation dialogs manually emulate Enter because navigation is off. Custom Tab logic covers primary 2D/3D forms, but Settings, Profiles, Library, inspector, mode switch, and editor controls do not receive equivalent focus traversal. Some controls use hidden IDs or icon-only interaction without a semantic keyboard label.

**Impact:** keyboard-only users and many users with motor impairments cannot complete normal workflows. The shortcut sheet can make keyboard support appear broader than it is.

**Recommendation:** enable ImGui keyboard navigation, complete backend key mapping, define predictable focus order, show a contrast-qualified focus indicator, and give icon/hidden-ID controls accessible names. Add end-to-end keyboard-only traversal for every top-level mode.

### UX-03 — Library selection, inspector, and viewport can describe different assets

**Severity:** High  
**Evidence:** `src/warlock/studio/panes/library.py:886-895`, `src/warlock/studio/main.py:1173-1174,1299-1347,1928-1929`, `src/warlock/studio/jobs_cache.py:26-30,111-130`

Selecting a card updates `state.selected` immediately but viewer synchronization waits for a cache reread or mode transition. The idle cache interval is three seconds. Parsing then happens asynchronously while the previous model remains visible, and `viewer.pending` is not surfaced.

**Impact:** the inspector can describe asset B while the viewport still shows asset A, making export, comparison, or edit decisions untrustworthy.

**Recommendation:** synchronize on selection change. While B loads, visibly label “Loading B…”, dim/clear A, and retain an explicit relationship between selection ID, displayed ID, pending ID, and failure. Never show an unlabeled stale model beside a new inspector.

### UX-04 — Compare can self-compare, freeze the frame, or close Studio on malformed input

**Severity:** High  
**Evidence:** `src/warlock/studio/panes/library.py:697-702,788-798,1012-1020`, `src/warlock/studio/viewer_embed.py:246-255`, `src/warlock/studio/state.py:1022-1028`, `src/warlock/studio/main.py:592-602,1341-1374`

Right-click selects the target before “Compare with selected” runs, so the baseline can be replaced by the target and later synchronization can make both sides identical. Compare parses the GLB and uploads GPU resources synchronously on the frame thread with no local error boundary, unlike the primary viewer's asynchronous guarded path.

**Impact:** misleading comparisons, visible freezes on large models, and a whole-app exit if parsing or GPU upload raises.

**Recommendation:** capture baseline and target IDs explicitly before menu selection, refuse identical IDs, label the command “Compare selected with this,” parse off-thread with stale-result guards, and adopt GPU resources only after validation inside a contained error surface.

### UX-05 — Crash recovery covers only Inker

**Severity:** High  
**Evidence:** `src/warlock/studio/main.py:1186-1200,1856-1889`, `src/warlock/studio/inker_mode.py:1433-1602`, `docs/manual/07-inker.md:393-405`

Orderly quit guards several authored-document modes, but periodic atomic autosave/recovery exists only for Inker. Clay, Plotter, Packwright, inspector pose edits, Poser, and profile drafts can all hold substantial unsaved work.

**Recommendation:** define one document-journal/recovery service with atomic snapshots, origin identity, schema/version, last-good cleanup, and a startup recovery chooser. Add profile drafts to dirty/recovery state. Do not implement each mode's crash safety independently.

### UX-06 — Startup and fatal frame-loop failures disappear without a user-facing report

**Severity:** High  
**Evidence:** `src/warlock/studio/main.py:544-603,4232-4258`

Fatal exceptions are logged and teardown runs, but the splash/window can simply disappear. Users are not told what failed, where the log lives, or whether recovery data exists.

**Recommendation:** use a minimal native fallback dialog independent of ImGui/GL. Include a friendly summary, `warlock.log`/`crash.log` paths, Copy Details, Open Log Folder, and recovery status. Preserve the full traceback only in the log.

### Additional UX findings

| ID | Severity | Finding and evidence | Recommendation |
|---|---|---|---|
| UX-07 | Medium-high | Destructive Delete/Discard receives default focus and Enter confirms it (`studio/dialogs.py:246-263`). | Make Keep/Cancel the safe default; require explicit focus/activation for destruction. |
| UX-08 | Medium | Global shortcuts leak through the matte modal because shortcut suppression knows only confirmation/prompt queues (`studio/main.py:1458-1476,1566-1569,3193-3209`; `panes/settings_3d.py:480-508`). | Centralize modal ownership/input suppression for every modal surface. |
| UX-09 | Medium | The frame that reports a model download complete synchronously runs forced diagnostics, including slow Torch/Blender probes (`studio/main.py:957-979`; `doctor.py:77-94,123-170,255-273`). | Submit verification as a task and show “Verifying installation…”. |
| UX-10 | Medium | Corrupt settings reset silently and failed settings writes are ignored by callers (`studio/settings.py:65-83,101-126`; `studio/main.py:827,3899-3904`). | Show a persistent warning, preserve/rename corrupt input, expose retry, and confirm when defaults were restored. |
| UX-11 | Medium | Running multi-GB downloads have no Cancel control (`panes/app_settings.py:478-500`; `studio/tasks.py:71-140`). | Add Cancel/Cancelling/Resume backed by real worker-process cancellation; see MDL-13. |
| UX-12 | Medium | Profiles' contextual help can attach to and overlap the global row because `help_button` assumes a preceding local item (`panes/profiles_panel.py:24-29`; `manual/render.py:62-76`; compare `panes/app_settings.py:54-60`). | Draw a local heading first or make help layout independent of prior-item state. |
| UX-13 | Medium | Profile taxonomy combos are unlabeled even though the 2D form already owns labels/grouping (`panes/profiles_panel.py:174-180`; `panes/settings_2d.py:90-127`). | Reuse the same `GUIDANCE_GROUPS`, field labels, and labeled controls. |
| UX-14 | Medium | Cancel/Quit can silently discard profile drafts; drafts are absent from the dirty-document guard (`panes/profiles_panel.py:137-192,293-296`; `studio/main.py:1874-1881`). | Compare draft to origin, guard Cancel/Quit, and include drafts in recovery. |
| UX-15 | Medium/accessibility | Meaningful muted copy uses disabled text at about 3.20:1 contrast in dark mode and 2.55:1 in light mode (`studio/theme.py:136-146`; `studio/tokens.py:135-161`; `studio/widgets.py:1331-1342`). | Create an opaque, WCAG-qualified secondary-copy role; reserve disabled styling for disabled controls. |
| UX-16 | Medium/accessibility | Text input ignores `TEXTINPUT`/`TEXTEDITING` and drops non-BMP code points (`studio/imgui_backend.py:291-327`). | Implement IME composition, positioning, full Unicode input, and font fallback tests. |
| UX-17 | Medium | Home suppresses the only global progress card with ETA and Cancel, leaving a non-clickable queue sentence (`studio/main.py:3193-3199`; `panes/landing.py:258-271,440-454`; `panes/overlay.py:268-362`). | Keep the progress card on Home or make the Home queue row open a cancel/detail surface. |
| UX-18 | Low-medium | “New 3D model” and “New Model” sit together but the latter means Clay (`panes/landing.py:509-524`). | Rename to “Generate 3D” and “Model in Clay,” with one-line descriptions. |
| UX-19 | Low-medium | Blank prompt/name submission stays enabled and silently does nothing (`studio/dialogs.py:342-365`). | Disable Save and show “Name required.” |
| UX-20 | Medium | Quit first asks a generic question and then starts a sequence of per-document dialogs; it under-describes active downloads/exports (`studio/main.py:3428-3438,1856-1889`). | Use one preflight summary listing active operations and every unsaved document, then confirm once. |
| UX-21 | Medium/accessibility | DPI is sampled only at startup; moving between mixed-DPI monitors does not rebuild fonts/style (`studio/tokens.py:14-16`; `studio/main.py:1439-1455`). | Handle display/DPI change, resample, and rebuild scaled resources without losing layout state. |

### UX test gaps

Existing tests cover many local controls and state transitions, but not the highest-risk interactions above. Add:

- screenshot and interaction matrices for all modes at 1100×700 and common larger sizes, 1.0×/1.5×/2.0× UI scale, both themes;
- keyboard-only completion of representative workflows in every mode;
- selection → viewer synchronization and stale-load failure tests;
- compare baseline/target semantics, large-load non-blocking behavior, and corrupt-GLB containment;
- modal shortcut suppression;
- IME/CJK/emoji composition;
- settings write/corruption failure UX;
- mixed-DPI monitor transitions.

---

## 4. Packaging, installation, documentation, and test-system gaps

### DST-01 — Wheel layout cannot satisfy the runtime's native-binary lookup

**Severity:** High  
**Evidence:** `src/warlock/config.py:10-11,76-110`, `.gitignore:20`, `pyproject.toml:95-107`

Native defaults are derived from `Path(__file__).parents[2] / "vendor"`. In an editable source checkout that points at the project. In a wheel it points under the environment (for example `Lib/vendor`). The wheel includes the Python package, manual, and changelog, but not ignored `vendor/` binaries. Setup documentation tells users to populate the checkout's `vendor/`, which a non-editable wheel will never inspect.

**Recommendation:** explicitly choose a supported distribution model. Prefer installing pinned native assets under a versioned `WARLOCK_HOME`/application data location through an installer, or package the legally redistributable binaries as wheel/application resources and resolve them with `importlib.resources`. Add a clean built-wheel smoke test that runs Doctor and starts far enough to resolve every resource.

### DST-02 — Pasteable model commands target a different default directory from runtime

**Severity:** High  
**Evidence:** `src/warlock/config.py:195-229`, `src/warlock/models.py:105-176`, `src/warlock/doctor.py:190-192,350-405`, `docs/MODELS.md`, `docs/manual/15-installation.md:74-158`, `README.md:47-57`

Runtime defaults to `~/.warlock/models`. Registry/Doctor/manual commands render relative `models/...`. Running them from an arbitrary working directory downloads gigabytes where Warlock does not look. README's primary setup correctly uses `$HOME/.warlock/models`, increasing the contradiction. Tests currently pin the relative spelling rather than the resolved behavior (`tests/test_fetch.py:144-155`).

**Recommendation:** render commands from the effective `Config`, quote the resolved path for PowerShell, and honor `WARLOCK_T2I_ROOT`, `WARLOCK_T2I_DIR`, and Trellis overrides. Test copy/paste commands from a non-project CWD. If commands are intended as examples rather than exact remedies, stop presenting them as exact Doctor fixes.

### DOC-01 — The declared sole roadmap is missing, and docs tests silently miss it

**Severity:** High documentation/maintenance risk  
**Evidence:** `CLAUDE.md:31`, `docs/INVARIANTS.md:218`, `tests/test_external_doc_links.py:36-44,98-103,118-122`

Both contributor guidance and invariants say `docs/TODO.md` is the sole live roadmap. It is absent. Numerous code/tests/scripts cite numbered sections that now require git archaeology into a deleted historical file. The focused documentation suite still passed because the link test skips missing source files and enforces only a global link count.

**Recommendation:** restore a tracked live roadmap or replace every historical citation with a durable decision/measurement document. Make documentation tests assert that every declared source exists and enforce coverage per source, not only globally. A historical plan can live under an explicitly archived path; source comments should not name a nonexistent current file.

### DOC-02 — “Authoritative” current guidance disagrees about the default model

**Severity:** Medium  
**Evidence:** `src/warlock/models.py:23-38`, `src/warlock/queue.py:1-5`, `CLAUDE.md:7`, `docs/INVARIANTS.md:7,25,214`

Implementation and current user docs use `sdxl_cfg`; the queue module header, contributor guide, and invariants still describe SDXL-Turbo as the default and use the old default model root. These files are specifically presented as authoritative engineering guidance.

**Recommendation:** update current-state documents and module headers. Keep dated measurements historical, but label them as such. Add a small documentation assertion that declared current defaults match the registry.

### DOC-03 — gltfpack is called shipped/present but is absent from a clean tracked checkout

**Severity:** Medium  
**Evidence:** `.gitignore:20`, `docs/manual/04-generating-meshes.md:157-159`, `docs/manual/16-configuration.md:28,118-119`, `src/warlock/doctor.py:233-240`, `tests/conftest.py:67-74`

The manual and invariants say gltfpack is vendored and present; `vendor/` is wholly ignored and a clean-checkout fixture assumes it is absent. Setup covers Trellis but provides no gltfpack acquisition/checksum. Doctor explains only that meshes remain full density.

**Recommendation:** ship/install it, or document an exact pinned release and checksum with an actionable Doctor remedy. Do not call an untracked local machine asset part of the repository.

### DOC-04 — Persisted preferences conflict with “there is no config file” wording

**Severity:** Low  
**Evidence:** `README.md:100`, `docs/manual/16-configuration.md:3`, `src/warlock/studio/settings.py:1-27`, `docs/manual/17-app-settings.md:3-5`

The intended distinction is environment-based engine configuration versus `studio_settings.json` UI preferences, but the absolute wording is misleading.

**Recommendation:** say “there is no engine/configuration file; runtime configuration is via environment variables. Studio UI preferences are persisted separately.”

### DOC-05 — Research and current architecture are not clearly separated

**Severity:** Low  
**Evidence:** `docs/REPORT.md:25,44`, `docs/INVARIANTS.md:218`

`docs/REPORT.md` is an undated long ComfyUI recommendation that can read like a live plan, while shipped architecture deliberately uses no ComfyUI and the declared roadmap is elsewhere/missing.

**Recommendation:** add status, date, and supersession banners; move historical research under an archive/research namespace.

### TST-01 — The model-stage learned judge is declared but intentionally unbuilt

**Severity:** Medium feature/documentation gap  
**Evidence:** `src/warlock/judge.py:75-88`, `src/warlock/service/judge.py:35-55`, `README.md:23`

The core stage registry declares `model`, but the service explicitly refuses training it as “declared and unbuilt.” Its rationale cites missing `TODO.md §8`. README's learned-quality-judge language is broad enough to imply mesh judging is available.

**Recommendation:** after the positive-corpus gate is met, implement the planned graded eight-view model probe; otherwise remove/hide the declaration and clearly label the existing judge as reference/image-only. Keep the blocker in a live document.

### TST-02 — Production has few obvious stubs, but maintenance hotspots are very large

**Severity:** Low-medium maintainability  
**Evidence:** `src/warlock/studio/main.py` (~196 KB), `studio/widgets.py` (~88 KB), `studio/panes/library.py` (~63 KB), `studio/inker_mode.py` (~62 KB), `db.py` (~60 KB), `studio/review_mode.py` (~57 KB), `studio/state.py` (~55 KB), `queue.py` (~54 KB)

A scan for `TODO`, `FIXME`, `NotImplemented`, `pass`, and placeholder patterns found no obvious production feature stubs masquerading as complete. `studio/undo.py:58-62` contains appropriate abstract-interface `NotImplementedError`s. Many `pass` statements are documented exception suppression or intentional control flow. The substantive declared-but-unbuilt feature is the model judge above.

The maintenance risk is concentration: frame loop, navigation, lifecycle, mode dispatch, shortcuts, viewer adoption, and many modal flows converge in a roughly 4,000-line `main.py`; other cross-cutting files are similarly large.

**Recommendation:** continue extracting cohesive state machines and services—modal/input ownership, selection/view synchronization, recovery, model maintenance, and shutdown are natural seams—without duplicating business logic into panes.

### TST-03 — GPU/Blender tests are not deselected by default despite saying they are

**Severity:** Medium  
**Evidence:** `pyproject.toml:128-131`, `tests/test_conditioning_gpu.py:3-8`, `tests/test_prompt_encode_gpu.py:10-13`, `README.md:106-109`, `CLAUDE.md:10`

Pytest registers the `gpu` marker but has no default exclusion. On this provisioned host, plain `uv run pytest -q` ran real GPU tests, loaded adapters, and emitted PEFT/CUDA-stack warnings. On an unprovisioned host those critical paths mostly skip. The test modules claim default deselection, so contributor expectations and actual cost differ.

**Recommendation:** make integration tests explicitly opt-in (for example a dedicated flag or default `not gpu`), and define a separate provisioned GPU/Blender job that must be run on a known schedule or before changes to lifecycle code. Keep the fast suite deterministic and the integration suite visible rather than silently skipped.

### TST-04 — No tracked CI/release workflow enforces the existing guards

**Severity:** Medium  
**Evidence:** no tracked `.github` workflow or equivalent CI/release configuration; current REL-01 failures

The repository has strong tests but no visible mechanism guaranteeing they run before a version commit/release. The current mismatch is exactly what the tests were written to stop.

**Recommendation:** add a Windows CI matrix for supported Python versions, lint, non-GPU tests, manual/docs checks, built-wheel resource/entry-point smoke, and an explicit provisioned GPU/Blender lane. Make release creation depend on version/changelog/wheel/Doctor checks.

### DST-03 — Base install exposes the Studio command even when Studio dependencies are optional

**Severity:** Medium-low  
**Evidence:** `pyproject.toml:13-86`, `src/warlock/cli.py:14-20,53-62`

The base package exposes `warlock`; no subcommand opens Studio and imports optional pygame/ModernGL/imgui dependencies. A base-only install can therefore end in an import traceback rather than an actionable install remedy.

**Recommendation:** catch the optional-dependency import failure and explain `uv sync --extra studio`/the supported install command. If wheels are public, add complete project metadata (readme, license, URLs, supported platforms/Python classifiers) and a root license file as appropriate.

---

## 5. Conflicting claims and implementation gaps at a glance

| Claim or expectation | Actual state | Relevant finding |
|---|---|---|
| Project is a coherent `0.0.22` release | Runtime and changelog are `0.0.21`; tests fail | REL-01 |
| Another instance is merely an informative warning | Both instances share state; second can kill the first server | RUN-01 |
| A `done` model is canonical/grounded | Normalization/report errors are swallowed | ART-01 |
| Download selection is all-or-nothing | Earlier repositories remain after a later failure | MDL-09 |
| An installed LoRA is immediately usable | Resident pipeline does not discover it | MDL-05 |
| Installed model rows establish health | A few files, even empty fixtures, establish “present” | MDL-07 |
| Model downloads are reproducible | Repository HEAD is mutable; no hashes/revisions | MDL-03 |
| Download free-space check covers the plan | Only first destination volume is measured | MDL-08 |
| Native binaries are vendored/shipped | `vendor/` is ignored and not in wheels | DST-01, DOC-03 |
| Doctor commands put files where runtime loads them | Commands use relative `models/`; runtime uses `~/.warlock/models` | DST-02 |
| `docs/TODO.md` is the sole live roadmap | File is absent and docs tests pass anyway | DOC-01 |
| SDXL-Turbo is the default | Implementation default is `sdxl_cfg` | DOC-02 |
| GPU tests are deselected by default | Plain pytest runs them when provisioned | TST-03 |
| All authored work has a safe quit/recovery story | Only Inker has crash autosave/recovery | UX-05, UX-14 |
| Larger UI scale improves accessibility | Fixed columns can hide controls; keyboard nav is off | UX-01, UX-02 |

---

## 6. What is already strong

The audit found substantial practices worth preserving:

- Offline defaults are established before model imports; normal model loads use local paths and `local_files_only=True`.
- The explicit fetch worker keeps the application process offline and stages repository files before publication.
- `Text2Image.load()` publishes only a fully configured pipeline and reclaims failed load attempts.
- Adapter re-enable/disable ordering accounts for persistent PEFT state; per-call conditioning cleanup is unusually careful.
- Base-model changes, idle eviction, exclusive Trellis handoff, submit-time VRAM admission, dispatch-time free-memory checks, and resident-memory crediting form a strong baseline.
- Trellis stop confirms death before treating VRAM as free, and port reclamation checks executable identity. RUN-01 is about ownership ambiguity between live instances, not careless arbitrary-process killing.
- Blender, matting, fetch, and Trellis work use process boundaries where native/import-time state makes in-process cleanup unreliable. Children are assigned to a Windows kill-on-close job.
- SQLite operations are guarded, and queue claim/status transitions are intentional.
- Staged/atomic writes are common for served artifacts and authored formats.
- Model uninstall has strong path-containment and shared-weight claim accounting; it will not recursively delete arbitrary override paths.
- Headless Inker, Clay, Plotter, and Packwright engines have meaningful import boundaries and extensive tests.
- Native raster kernels retain NumPy references, ABI checks, and parity tests.
- UI task errors, field-error rings, toast history, diagnostics, contextual manual links, reduced-motion support, themes, soft-delete/Undo, and many empty states are thoughtfully implemented.
- Primary viewer loads are asynchronous and guarded against stale results; the Compare path should be brought up to that standard.
- The manual has meaningful parser/link/coverage tests; DOC-01 is a specific blind spot, not an absence of documentation discipline.
- The existing release tests correctly caught the version drift when actually run.

---

## 7. Recommended remediation sequence

### P0 — Before publishing the current release

1. Resolve REL-01 and require the full non-GPU release preflight.
2. Refuse unsafe second instances (RUN-01).
3. Add the model-operation/model-maintenance barrier (MDL-01) and ensure shutdown never unloads an in-flight model thread (MDL-02).
4. Pin model repository revisions and address the BiRefNet remote-code trust boundary (MDL-03).
5. Make canonical normalization failure visible and non-success-equivalent (ART-01).
6. Decide and verify the supported installation/distribution shape; fix wheel/native lookup and exact model download destinations (DST-01, DST-02).
7. Correct the missing/stale authoritative documentation (DOC-01, DOC-02).

### P1 — Reliability and core user experience

1. Add host-commit byte admission for offloaded bases (MDL-04).
2. Refresh/invalidate the resident pipeline after model-store changes and isolate corrupt optional LoRAs (MDL-05, MDL-06).
3. Add pinned integrity manifests, per-volume disk admission, transaction recovery, and backend mutation serialization (MDL-07 through MDL-10).
4. Add Verify/Repair/Cancel/Resume to Models (MDL-13, UX-11).
5. Implement responsive workspace breakpoints and real keyboard navigation (UX-01, UX-02).
6. Fix selection/view synchronization and rebuild Compare on the safe async viewer path (UX-03, UX-04).
7. Generalize crash recovery and add a native fatal-error surface (UX-05, UX-06).
8. Replace invalid-artifact happy-path fakes and split opt-in integration testing from the default suite (ART-02, TST-03).

### P2 — Accessibility, polish, and maintainability

1. Fix destructive defaults, modal input ownership, settings failure feedback, profile draft/form issues, contrast, IME, Home progress, quit preflight, and mixed-DPI handling (UX-07 through UX-21).
2. Fix cache re-arming, response caps, and ctypes prototypes (MDL-11, MDL-12, NAT-01).
3. Make gltfpack installation honest/actionable and clarify configuration/research documentation (DOC-03 through DOC-05).
4. Decide the model-judge and custom-model product boundaries (TST-01, MDL-14).
5. Add tracked CI, wheel smoke testing, optional-extra error UX, and carefully extract the largest cross-cutting modules (TST-02, TST-04, DST-03).

## Final assessment

Warlock Studio is not suffering from a lack of implementation. Its dominant risks are now at the seams between otherwise capable systems: event-loop ownership versus blocking model threads, UI convention versus service-level serialization, filename presence versus artifact integrity, source checkout versus installed distribution, selection state versus displayed state, and “recoverable in an orderly quit” versus “recoverable after a crash.”

The highest-value work is therefore not adding another mode. It is making those seams explicit and enforceable: one process owner, one model-store maintenance protocol, one cancellable shutdown model, one artifact health state, one verified installation layout, one responsive/focusable UI system, and one recovery mechanism for authored work. With those foundations hardened, the existing breadth becomes a strength rather than a larger surface for rare lifecycle failures.
