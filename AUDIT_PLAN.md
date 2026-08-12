# Warlock Studio — Audit Remediation Plan

**Date:** 2026-08-11 · **Addresses:** every finding in `MASTER_AUDIT.md` (87 findings: 1 Blocker, 18 High, 37 Medium, 31 Low). Finding IDs below refer to that document.
**Status:** plan only — nothing here has been executed. This was produced in a planning session; no code was changed.

## Ground rules (apply to every phase)

1. **Verify before fixing.** Each of the last three executed audits contained ~3 findings that turned out to be wrong or deliberate behavior. Before changing anything, reproduce or re-trace the finding against the current tree; if the code turns out to be deliberate (check `docs/INVARIANTS.md` and git history first), record the finding as *rejected* with a note instead of "fixing" it.
2. **Baseline discipline.** The audits reviewed `de87838`. The working tree also carries unrelated uncommitted bench/metrics work (`src/warlock/bench/`, `tests/test_bench_metrics.py`, `docs/measurements/2026-08-11-perceptual-hash-floor.md`) — leave it untouched; don't fold it into audit commits.
3. **Suite gates.** End every phase with `uv run ruff check .` and a full `uv run pytest` green (modulo the pre-existing REL-01 failures until Phase 0 lands). Never edit `src/` while the suite runs — several tests read module source. A bare `uv sync` prunes the extras; always `uv sync --extra studio --extra text2image --extra rig`.
4. **Invariant bookkeeping.** Any fix that changes an invariant updates `docs/INVARIANTS.md` in the same change. Any new worker-recorded param (ART-01's artifact health, MDL-03's installed revision in provenance) **must join `DERIVED_PARAMS`** or a reroll wears a stale verdict. No corpus-keyed constants (`trellis_band`, `SEAM_MAX`, grade scale) change under this plan, so no new `docs/measurements/` docs are required by it.
5. **Commit convention.** Subjects stay `Warlock v0.0.21` (or whatever D1 resolves to); no version bumps unless explicitly requested.
6. **Every fix ships with its regression test** where the finding names one (they mostly do). Findings that are pure doc edits are covered by the extended docs tests in Phase 0.

## Decision gates (need a human call before the affected work starts)

- **D1 — Version direction (blocks Phase 0, REL-01).** Revert `pyproject.toml` to 0.0.21, or complete the bump (write the `## 0.0.22` CHANGELOG entry + `__version__`). *Recommendation: revert to 0.0.21* — every other surface (commit subjects, changelog, `__version__`) says 0.0.21, and the house convention is that versions bump only on explicit request; if the pyproject bump was intentional, complete it instead.
- **D2 — Roadmap fate (blocks Phase 0, DOC-01).** Recreate a tracked `docs/TODO.md`, or update CLAUDE.md/INVARIANTS.md to state the roadmap file was deleted 2026-08-11 and `§N` citations resolve via git history. *Recommendation: update the docs* — the deletion in `de87838` looks intentional; recreating the file re-legitimises stale `§N` citations.
- **D3 — Supply-chain program scope (blocks the Phase 4 supply-chain track).** Pinned revisions + hash manifests + Verify/Repair (MDL-03, MDL-08, MDL-10) is a sizeable program touching the registry schema, fetch worker, doctor, and Settings. Go/no-go and depth (pin-only vs pin+manifest vs pin+manifest+vendored BiRefNet code to drop `trust_remote_code`).
- **D4 — Distribution model (blocks the Phase 4 distribution track, DST-01).** Officially declare source-checkout-only (cheap: a guard + docs), or support wheels (installer/resource resolution + wheel smoke test). *Recommendation: declare source-checkout-only now*; wheels later if ever wanted.
- **D5 — Accessibility/responsive program (blocks the Phase 4 accessibility track).** UX-01 + UX-02 (+ UX-18/19/22) is multi-week UI work. Go/no-go and ordering.
- **D6 — CI (blocks the Phase 4 CI track, TST-03).** Tracked GitHub Actions vs a mandatory local release-preflight script. Even the minimal script would have caught REL-01.

---

## Phase 0 — Restore a green, truthful baseline

*All text/config edits plus small test extensions. No behavior changes. Ends with the full suite green.*

| Item | Findings | Work |
|---|---|---|
| 0.1 | **REL-01** | Per D1: reconcile pyproject / CHANGELOG / `__version__`; fix the wrong test-name comment above `__version__`. Suite goes green here. |
| 0.2 | **DOC-01**, DOC-06 | Per D2: one editing pass over `CLAUDE.md:31` and `INVARIANTS.md:218` (same paragraph also fixes the docs inventory and the `§N` count); fix `tests/test_ux_todo_fixes.py` docstring; fix the `SOURCES` entry in `tests/test_external_doc_links.py` **and make the test fail on a missing declared source** (per-source coverage, not global count). |
| 0.3 | **DOC-02** | Reword INVARIANTS.md ×3, CLAUDE.md pipeline one-liner, `queue.py:3` docstring: SDXL 1.0 full-CFG is default, Turbo is the fast option; drop the wrong "because FLUX is gated" causal story. Optional: a doc assertion that declared defaults match `models.DEFAULT_BASE_MODEL`. |
| 0.4 | **DST-02** | Standardize `$HOME/.warlock/models` in MODELS.md, manual ch. 15, and `models.py` `download_text` — render from the effective `Config` (honor `WARLOCK_T2I_ROOT`/`WARLOCK_T2I_DIR`/Trellis overrides, quote for PowerShell); update `INVARIANTS.md:23`; retarget `tests/test_fetch.py:144-155` to pin resolved behavior; add a non-project-CWD paste test. |
| 0.5 | DOC-05, DOC-10 | Same INVARIANTS.md pass: correct the sweep.py deletion misattribution (the deleted files were `bench/{sweep,report,verdicts}.py`); correct the `promote_to_model` admission sentence. |
| 0.6 | DOC-04 | Align manual `00-index.md` with `studio/manual/loader.py` `PARTS` (pick one; the index's Part I placement of ch. 14 looks right — verify against the loader's intent); extend `tests/manual/test_docs.py` to assert index-section membership against `PARTS`. |
| 0.7 | DOC-03, DOC-07, DOC-08, DOC-09, DOC-11, DOC-12, REL-02 | Small doc fixes: gltfpack honesty (pinned release + checksum in docs until D4 decides shipping), ch. 21 `fetch=` wording, "two fatal checks" qualifier, REPORT.md status banner, two stale prose fragments, config-file wording, 0.0.15 changelog gap (backfill or annotate). |
| 0.8 | **TST-02** | Default-exclude GPU/Blender integration tests (e.g. `addopts = -m "not gpu"` or equivalent), keep an explicit opt-in lane; align README/CLAUDE wording. |
| 0.9 | SVC-03 | `sweep.py:184`: point the user-facing message at the live measurements doc instead of the deleted LEFTOVERS.md. |

**Acceptance:** `uv run pytest` fully green; docs tests now fail if a declared link source goes missing; every doc statement spot-checked in the master audit reads true against the tree.

## Phase 1 — Model/LoRA lifecycle correctness

*The highest-value code fixes. Each lands with its named regression test.*

| Item | Findings | Work |
|---|---|---|
| 1.1 | **MDL-06** | Widen the dispatch-time VRAM credit in `queue._check_resources` to every kind whose estimate carries an image-model term — preferably derive the credit from `vram.estimate` itself, killing the kind list. Tests: fake `device_memory()` scenarios for pixel-sheet / sprite-synthesis / retexture with a warm pipe (the branch headless tests currently skip). |
| 1.2 | **MDL-05** | Add a model-store generation counter bumped on successful download/uninstall/repair; worker compares at dispatch and evicts/reloads (or re-runs `_load_loras`) before the next job. Make a missing selected style a *refusal*, not a warning. Regression test: resident pipe → install LoRA → immediate styled generate. Also fixes the pixel-sheet sidecar recording a request-side `style_lora`. |
| 1.3 | **MDL-01** (quick half) | Replace `store.list(limit=…)` with `store.active_jobs()` in `uninstall`; re-verify `worker.current_job_id is None` inside the same loop-side callable that unloads. Test: uninstall racing a create. |
| 1.4 | **MDL-01/MDL-02/MDL-11** (structural half) | Design + implement the model-store lease: read/use lease held by dispatch and every model operation (covering the `to_thread` *thread*, not the coroutine); exclusive maintenance lease for download/uninstall/repair; per-model-root mutation lock in the service layer. Shutdown never unloads until the lease is released; `load()` re-checks a stop flag before publishing; join the outstanding load before the final `.loaded` check. Subprocess-level test asserting the interpreter exits while a load/sample is blocked. |
| 1.5 | **MDL-04** | Byte-based host-commit admission: per-base `host_peak` in the spec; require absolute free commit ≥ `host_peak + margin` immediately before `from_pretrained` and before large placement transitions. |
| 1.6 | **MDL-15** | Family check in `retexture_job` + its rerun path, mirroring `create_pixel_sheet`'s refusal. Test both doors. |
| 1.7 | MDL-07 | Lazy-load only selected adapters (or pre-validate/quarantine optional ones, health per Settings row); required distillation LoRAs stay fatal. Test: corrupt optional adapter + healthy base + unrelated selected style. Interacts with 1.2 and MDL-17 (lazy loading changes the eager-adapter VRAM story). |
| 1.8 | MDL-12 | Cache eviction by per-cache loaded/last-used state instead of the `_caches_evicted` latch. Test: evict → load matte via task path → evict again. |
| 1.9 | MDL-16, MDL-17 | Error-path traceback strip + pool trim in `_process`'s error branch; fold an adapter allowance into `vram_gib` (or document the exclusion) — reconcile with whatever 1.7 decides about eager loading. |

**Acceptance:** the natural 2D-then-3D flow on a 32 GiB coexist card no longer refuses; the LoRA-install-while-resident regression test passes; shutdown/uninstall race tests pass; INVARIANTS.md's VRAM section updated to describe the lease and the byte-based admission.

## Phase 2 — Concurrency, artifacts, process safety

| Item | Findings | Work |
|---|---|---|
| 2.1 | **RUN-01** | Per-home OS-level single-instance lock acquired before migration/DB/runtime startup; refusal surfaced in a native dialog. Stop treating executable-path identity alone as proof a listener is our orphan. Doctor row for a second instance (from A1-L9). |
| 2.2 | RUN-02 | Hold the migration's exclusive connection open across `_move` (close in `finally` after the deletes), or re-take immediately before each `rmtree`. |
| 2.3 | **ART-01** | Artifact-health state: `done_degraded` or structured health params (join `DERIVED_PARAMS`); surface failed canonical steps in library/inspector; Retry/Repair action; warning before exporting an unnormalized `model.glb`. Update the "grounding always runs" wording in INVARIANTS/CLAUDE to "attempted; failures now visible". |
| 2.4 | ART-02 | Default fakes emit tiny structurally valid PNG/GLB; malformed artifacts move behind explicit degraded-path tests; happy-path queue tests assert normalization/audit/report actually ran. (Do together with 2.3 — the health state gives these tests something to assert.) |
| 2.5 | CON-01 | `Conflict` when `dependent_jobs(svc, job_id)` is non-empty in `optimize_job`/`retexture_job` — the helper exists; `_jobs_lifecycle.py`'s docstring already names it as the answer. |
| 2.6 | CON-03 | Wrap `apply_library_pose`'s cap check + `rigging.save_pose` in the same `convert_lock(job_id, "poses")` its sibling holds. |
| 2.7 | RUN-03 | Centralized typed env parsing with range validation; report all invalid values in one Doctor/startup surface; stop silently nulling malformed safety limits. |
| 2.8 | MDL-09, MDL-13 | Per-volume disk admission in `disk_refusal()` (+ mixed-drive tests); stream-and-cap Trellis error bodies, stream success to temp file/bounded buffer. |
| 2.9 | MDL-10 | Whole-selection download transaction: stage all, validate, journaled publish, startup recovery/quarantine of interrupted transactions — or, minimum viable, a precise partial-success result with safe Resume/Rollback. Scope per D3 (pairs with the manifest work). |
| 2.10 | CON-04, CON-05, CON-06, CON-07 | Staged copy in `_stage_link` fallback; route `trash_job`'s queued branch through `cancel_job` (or fix the comment); tolerate corrupt `params` at dispatch; `to_thread` the `_discard_artifacts` I/O. |

**Acceptance:** race tests for 2.1/2.5/2.6 pass; a queue happy-path test now fails if normalization silently breaks; the staged-writes invariant paragraph extended to cover the new sites.

## Phase 3 — Studio UI correctness and quick wins

| Item | Findings | Work |
|---|---|---|
| 3.1 | UX-11 | Add `restore:`/`purge:` to `_on_task_done`'s invalidate prefixes and `purge:`/`empty-trash` to the storage re-measure. |
| 3.2 | UX-12 | `event.mod` instead of `get_mods()` in `_shortcut` (Ctrl+K, Ctrl+Enter), matching `_passes_text_field`. |
| 3.3 | CON-02 | `PHASES_PIXEL_SHEET`/`PHASES_RETEXTURE` tables (or a per-phase mapper), pinned in `tests/test_progress.py`. |
| 3.4 | **UX-04** | Rebuild Compare on the primary viewer's async path: explicit baseline/target capture before the menu, refuse identical IDs, off-thread parse with stale guards, GPU adoption inside a contained error surface, relabel "Compare selected with this". |
| 3.5 | **UX-03** | Viewer sync on selection change; "Loading B…" state; explicit selection/displayed/pending/failed ID relationship. |
| 3.6 | MDL-14 (cancel half) | Cancel button beside the download progress bar, backed by fetch-child termination (kill-on-close job already reaps; staging already makes cancel safe). Verify/Repair waits for D3. |
| 3.7 | UX-14, UX-13 | Bulk "Try again (N)" over the existing per-job retry; add Clay's Ctrl+1/3/5/7/W rows to manual ch. 14 and make the popup rows uniform. |
| 3.8 | UX-07, UX-08 | Safe default focus in destructive dialogs; centralized modal ownership/input suppression covering the matte modal. |
| 3.9 | UX-09, UX-10, UX-20, UX-21 | Post-download verification as a task; settings corruption/write-failure surfaced; keep the progress card on Home (or clickable queue row); single quit preflight summary replacing the generic-confirm + dialog chain. |
| 3.10 | UX-15, UX-16, UX-17 | Profiles: local heading before `help_button`; reuse `GUIDANCE_GROUPS`/labels for taxonomy combos; draft-vs-origin dirty tracking guarded on Cancel/Quit. |
| 3.11 | UX-23, UX-24, UX-25, UX-26, UX-27 | Landing rename; disable blank-submit with "Name required"; queued-toast `waiting + 1`; check `submit_promotion`'s refused return; bind Delete (and F) in the library key fall-through. |

**Acceptance:** progress tests pin the two new phase tables; a corrupt-GLB Compare no longer exits the app (containment test); UI-affecting fixes each get a state-transition test where the pane tests already have precedent.

## Phase 4 — Gated programs (each starts only after its decision gate)

**4A. Supply chain & install integrity (D3):** MDL-03 (pinned revisions carried through worker + rendered commands; installed revision in provenance — joins `DERIVED_PARAMS`; BiRefNet `trust_remote_code` boundary per D3 depth), MDL-08 (completion manifest required for "installed"; zero-length safetensors rejected; "files found" vs "load probe passed" split in Doctor/Settings), MDL-14 (Verify/Repair/Update rows — needs pinning first), SVC-02 (version/hash pins for `trellis-server.exe` and gltfpack as a non-fatal doctor detail), remainder of MDL-10 if 2.9 shipped the minimum.

**4B. Distribution (D4):** DST-01 (per D4: source-checkout-only guard + docs, or wheel resource resolution + installer), DST-03 (actionable optional-dependency error for the Studio command; project metadata/license if wheels go public), DOC-03 follow-through (ship or formally document gltfpack per the same decision).

**4C. Accessibility & responsive UI (D5):** UX-01 (breakpoints: collapsible/drawer sidebars, one-pane narrow mode; render/interaction matrix at min-size × {1.0, 1.5, 2.0} × both themes), UX-02 (ImGui keyboard nav enabled, full key mapping, focus order, focus indicator, accessible names; keyboard-only traversal test per mode — subsumes UX-27), UX-18 (WCAG secondary-copy role), UX-19 (IME/TEXTINPUT/non-BMP), UX-22 (DPI-change rebuild).

**4D. Crash recovery service:** UX-05 (one document-journal/recovery service — atomic snapshots, origin identity, schema/version, startup chooser — adopted by Clay/Plotter/Packwright/Poser/inspector pose edits; Inker migrates to it), UX-17 (profile drafts join it), UX-06 (native fatal-error dialog independent of ImGui/GL, wired to recovery status).

**4E. CI / release automation (D6):** TST-03 (per D6: workflow or mandatory preflight script — lint, non-GPU suite, docs/manual checks, changelog/version lockstep, wheel smoke if 4B chose wheels; explicit GPU/Blender lane on schedule), enforcing the guards that would have caught REL-01.

## Phase 5 — Remaining lows, boundaries, closure

| Item | Findings | Work |
|---|---|---|
| 5.1 | SVC-01 | Adopt the `_save_source` staging shape (dot-prefixed unique temp + `finally` unlink) at all seven listed sites; fix `save_edited_image`'s shared tmp name. |
| 5.2 | SVC-05 | Pipe fetch-child stderr into the fallback detail (as `rigging.run_worker` does); move the stdin parse inside the error-reporting path. |
| 5.3 | SVC-04, SVC-06, SVC-07 | `errors.Invalid(field="stage")` in judge; temp + `os.replace` in `export_to_folder`; widen `prompt_preview`'s tolerance. |
| 5.4 | RUN-04 | Complete `argtypes`/`restype` on the winjob Win32 calls; capture `GetLastError` promptly. |
| 5.5 | MDL-18, TST-01 | Product-boundary decisions: document registry-only LoRA support (or scope an importer); build the model-stage judge per the existing CAMERA.md plan once its corpus gate is met, or hide the declaration and label the judge image-only. |
| 5.6 | HYG-01 | Link `examples/` PNGs from the sprite-synthesis/Plotter manual chapters, or delete them. |
| 5.7 | TST-04 | Standing guideline, not a task: when Phases 1–4 touch `main.py`/`state.py`/`queue.py` seams (modal ownership, selection sync, recovery, model maintenance, shutdown), extract the seam rather than growing the file. No big-bang refactor. |
| 5.8 | Closure | Final full suite + ruff; update MASTER_AUDIT.md with an outcome column (fixed / rejected-with-reason / deferred) per finding; record rejected findings and any new invariants in `docs/INVARIANTS.md`; update session memory. |

---

## Traceability — every finding has a home

| Phase | Findings |
|---|---|
| 0 | REL-01, REL-02, DOC-01…12, DST-02, TST-02, SVC-03 |
| 1 | MDL-01, MDL-02, MDL-04, MDL-05, MDL-06, MDL-07, MDL-12, MDL-15, MDL-16, MDL-17 |
| 2 | RUN-01, RUN-02, RUN-03, ART-01, ART-02, CON-01, CON-03…07, MDL-09, MDL-10, MDL-13 |
| 3 | CON-02, MDL-14 (cancel), UX-03, UX-04, UX-07…17, UX-20, UX-21, UX-23…27 |
| 4 (gated) | MDL-03, MDL-08, MDL-14 (verify/repair), MDL-10 (remainder), SVC-02, DST-01, DST-03, UX-01, UX-02, UX-05, UX-06, UX-18, UX-19, UX-22, TST-03 |
| 5 | SVC-01, SVC-04…07, RUN-04, MDL-18, TST-01, TST-04, HYG-01 |

Sizing, roughly: Phase 0 is one focused session; Phases 1–2 are the substantive engineering (the model-store lease in 1.4 is the largest single design); Phase 3 is many small independent fixes (parallelizable); Phase 4 tracks are each their own project and only start on an explicit go; Phase 5 is cleanup. Phases 0→1→2 are ordered (green suite first, then lifecycle, then the things that build on lifecycle primitives); Phase 3 can interleave with 2.
