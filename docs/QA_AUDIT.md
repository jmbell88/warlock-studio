QA Audit — Warlock Studio (2026-08-05)
  
 Context

 The user asked for a QA audit of Warlock Studio. Scope agreed via questions: full local-hardware verification (complete test
 suite incl. GL/GPU/bpy tests, ruff, doctor, live app launch), systematic verification of the CLAUDE.md hard invariants, report
 only — no fixes applied, report at docs/measurements/2026-08-05-qa-audit.md with actionable items appended to TODO.md.

 Exploration established: 2,340 tests collected; no CI, no type checker, no coverage tooling; uv.lock stale (committed lock says
 0.0.6 vs pyproject 0.0.8, fix uncommitted); torchvision is venv-only (a clean uv sync silently degrades bench ranking); TODO.md
 carries a ledger of known small defects; REPORT.md at root is a ComfyUI research doc, not a prior audit — ignore it. The venv is
 Python 3.13.13 with bpy, torch+torchvision, moderngl, pygame all installed, so one pytest command covers the whole suite
 (gpu/bpy tests self-gate via importorskip, which succeeds here). vendor/trellis/trellis-server.exe present;
 vendor/gltfpack/gltfpack.exe and BiRefNet weights absent (both documented as expected).

 Safety rails (apply to every step)

 - Never run uv sync — it would strip venv-only torchvision and rewrite the lock. All commands use uv run --no-sync … (fallback
 --frozen; document which was used, and verify uv.lock wasn't rewritten after the first run).
 - App smoke launch uses a scratch data dir (WARLOCK_DATA_DIR/WARLOCK_DB → session scratchpad; config.py honours both,
 weights/vendor resolve from PROJECT_ROOT regardless). The user's assets/ is never touched.
 - Evidence files (junit xml, doctor output, logs) go to the scratchpad, never the repo.
 - The only repo writes in the whole audit: the report file and the TODO.md append, at the end. No source edits, no commits.

 Phase 0 — Preflight snapshot (foreground, ~5 min)

 - git rev-parse HEAD (expect 6d7f03b), git status --porcelain (expect only M uv.lock, ?? REPORT.md).
 - Pin the supply-chain finding: git show HEAD:uv.lock warlock version (expect 0.0.6) vs pyproject.toml (0.0.8); torchvision
 absent from lock but import torchvision succeeds in venv (0.26.0+cu128).

 Phase 1 — Verification runs (parallel)

 1a. Full test suite (background Bash, started first, timed — wall time is itself a finding since no CI exists):
 uv run --no-sync pytest -rs --junitxml=<scratchpad>/pytest-full.xml -q
 Baseline: ~2,340 collected, 0 failures. Do not assert a skip count (known baselines conflict) — instead audit every skip reason
 from -rs: BiRefNet/gltfpack-gated skips are legitimate; any GL or bpy skip on this machine is a finding (hardware is present, so
 a skip means the gate broke). First check sub-conftests under tests/clay//tests/inker/ don't deselect anything before trusting
 single-command coverage.

 1b. Fast gates (foreground, no GPU contention):
 - uv run --no-sync ruff check . — baseline clean; any output is a regression. Do not run ruff format (known 113-file diff,
 deliberately not a gate).
 - uv run --no-sync warlock doctor — capture rows + exit code (expect 0). Expected: trellis exe OK, gguf OK, gltfpack WARN, CUDA
 OK, VRAM OK, winjob armed OK, matting WARN (BiRefNet absent), blender OK. Check whether the DINOv2 metric row overclaims
 (doesn't check torchvision — TODO decision #3). Run doctor before the app smoke (port check would flip with the app running).
 Any FATAL row halts for diagnosis.

 1c. Invariant audit — 4 parallel read-only Explore subagents while pytest runs, one per group. Rule: where a named pinning test
 exists, verify it exists, is collected, and still targets the mechanism — don't re-prove what a green suite proves. Manual reads
 prioritized by churn (bench/, studio/main.py, review_mode.py, settings_2d/3d, clay/ are hot;
 queue.py/db.py/rigging.py/trellis.py/config.py stable since 07-20).

 - Group A — concurrency/DB/VRAM/subprocess/offline: db.py every self._conn under self._lock + merge_params single-hold; vram
 tri-state resolve before _make_worker; exclusive-mode stop-before-load/unload-before-next-start in Worker._generate; check_vram
 placement in create_job; winjob scanner test (tests/test_vram.py:175) scan root covers bench/ and studio/; HF_HUB_OFFLINE first
 in __init__.py; test_offline.py covers matting.py and bench/metrics.py; frame loop in main.py has no new blocking calls.
 - Group B — artifact pipeline: source.glb never overwritten; gltfpack-before-normalize ordering; _discard_artifacts deletes
 both; staged writes (optimize.staged_copy, postprocess._staged); optimize_job Conflict on queued/running; unconditional
 grounding; prompt-chunk single-chunk bit-identity; meshaudit/meshreport "watertight" separation; input caps at every
 byte-accepting entry point; DERIVED_PARAMS completeness — diff the strip list against every params[...] write in queue.py
 (highest-value manual check; most likely silently broken by recent churn).
 - Group C — rigging/pose/sheet + modes/review: import bpy only in blender_worker.py; temp-name rig writes + GLB-first finalize +
 cancel semantics; bone_heuristic="BLENDER" / export_rest_position_armature=False still literal; pose_path regex gate;
 _purge_import_helpers after every gltf import; mode sets partition modes.KEYS exactly; Review out of VIEWPORT_MODES; viewer.path
 (never a unit key) decides Review loads; _review_viewport honours pose_mode; _shortcut fall-through returns. review_mode.py +
 main.py dispatch read directly (top churn).
 - Group D — inker/clay/GL: inker purity (no imgui/moderngl/pygame/service imports under studio/inker/); uid-addressed undo +
 revoke semantics + save-commits-floating-first + saving gates; clay immutable CSR mesh + id(mesh) cache key + to_model single
 conversion; texture_ref/forget_texture pairing; ThumbnailCache one-frame deferral; manual reads of viewer_embed.py (known
 missing forget_texture), panes/inker_textures.py (fully untested), studio/dpi.py:38 (silent except:pass driving all UI scaling).
 Verify meta-pin test_fakes_match_real_signatures.py still targets current signatures.

 Phase 2 — Known-defect triage (foreground, after invariant groups)

 Re-verify each TODO.md ledger item with file:line evidence, classify severity:
 1. _pick_anchor decompression-bomb hole (profiles_panel.py) — likely High.
 2. Dead hole_ratio read in inspector._quality vs _audit_mesh's worst — Low.
 3. Truncating copyfile in _restore (queue.py mesh-retry) — Medium.
 4. viewer_embed missing forget_texture vs clay_view.py's fixed pattern — Medium-High.
 5. SEAM_MAX=2.0 uncalibrated — open measurement, not a code defect.
 6. BiRefNet path never executed — residual risk + doctor-row honesty question.
 7. Low-priority items (anchor re-embed, ClayView.pick re-triangulation, matting.unload empty_cache, dual thumbnail paths) — one
 line of evidence each.
 8. bench/findings.json empty — verdict loop never fed; process finding. Check bench/findings.py (churniest module) coherence.

 Phase 3 — App launch smoke (serial, after pytest finishes — GPU contention)

 1. Read studio/runtime.py::_start first to know whether trellis-server spawns at startup or lazily (sets the orphan-check
 expectation).
 2. Launch with scratch env (WARLOCK_DATA_DIR/WARLOCK_DB → scratchpad), background: uv run --no-sync warlock. Observe ~90–120 s
 via <scratch>/warlock.log: startup completes, doctor rows surface, no tracebacks, idle ticks, then kill the process.
 3. Orphan verification: after kill, Get-Process trellis-server must return nothing — an orphaned trellis-server.exe is a
 Critical finding (winjob invariant broken in practice). No generate job submitted (minutes of GPU, not needed for a smoke).

 Phase 4 — Gap analysis (desk work)

 - Untested: studio/dpi.py, panes/inker_textures.py; thin: app_settings, inker_colors/layers, clay panes; no direct tests:
 studio/main.py (1999 LOC), widgets.py, config.py, glbio.py.
 - Infra: no CI (Phase 1a wall time = the cost estimate for adding one), no type checker, no coverage, format never adopted (a
 decision to record, not a fix).
 - Supply chain: stale uv.lock + torchvision venv-only.

 Phase 5 — Deliverables (the only repo writes)

 docs/measurements/2026-08-05-qa-audit.md in the house style of the two existing measurement docs (dated, verdict-first):
 verdict paragraph → What was run (commands, exit codes, wall times, skip audit) → Invariant audit table (invariant → pin test
 file::name → pinned-green / manually-verified / FINDING) → Findings (severity-ranked, file:line evidence + failure scenario) →
 Ledger re-verification → Gaps → Recommended fix order → Environment (HEAD, python, GPU, weights present/absent).

 TODO.md append: new ## QA audit 2026-08-05 section with the actionable fix-order list cross-referencing the report. No version
 bump; no commit unless the user asks.

 Provisional fix order (final ranking depends on evidence): any test/invariant regression found → torchvision/uv.lock supply
 chain → _pick_anchor bomb gate → viewer_embed forget_texture → _restore staged write → doctor row honesty → dead reads /
 low-priority ledger → infra proposals (CI, typing, format).

 Verification

 The audit is the verification; its own success criteria: pytest exit code + every skip reason explained; ruff clean; doctor exit
 0 with expected WARNs only; app launches, idles, exits with zero orphaned processes; every CLAUDE.md invariant row has a
 disposition; report + TODO append written and internally consistent with the captured evidence.
