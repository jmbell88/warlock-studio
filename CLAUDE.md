# CLAUDE.md

Guidance for Claude Code in this repository. See `D:\Projects\CLAUDE.md` for how this project fits into the wider workspace.

## What this is

Local AI 3D asset generator: text or image prompt → SDXL 1.0 at full CFG (reference PNG; `sdxl_cfg`, with Turbo as the fast option) → `trellis-server.exe` (vendored native binary, not the Python TRELLIS package) → textured GLB, in a single-process desktop app: a pygame window, one ModernGL context, imgui panels drawn through it. No HTTP, no browser.

## Commands

- Install: `uv sync --extra studio --extra text2image --extra rig` — dev tooling is a dependency *group*, not an extra, and a bare `uv sync` prunes the extras (breaking ~10 test files at collection). `rig` needs a Python 3.13 venv or it silently installs nothing.
- Tests: `uv run pytest` (single test: `uv run pytest tests/test_x.py::test_name`). Never edit `src/` while the suite runs — several tests read module source. The `gpu` marker is excluded by default; `uv run pytest -m gpu` is the opt-in lane (real card, real weights) and is what to run before changing model loading, VRAM accounting or conditioning.
- Lint: `uv run ruff check .`
- Optional native kernels: `pwsh native\build.ps1` → `vendor/warlockc/warlockc.dll` (gitignored; not needed to run anything).

## The invariants live in `docs/INVARIANTS.md`

That file is the full, authoritative record of this codebase's hard invariants **and the measured reasoning behind them** — read the section covering a subsystem before modifying it, and update it when an invariant changes. The one-liners below are only the catastrophic-mistake preventers; every one has a longer argument there.

- **Fully offline.** `HF_HUB_OFFLINE=1` is the first thing the package does; nothing downloads at runtime. The single exception is the user-initiated `fetch_worker` subprocess, which goes online in its own environment only.
- **Three threads.** The pygame frame loop never blocks; the asyncio `Worker` lives on the `warlock-loop` thread; everything blocking goes through `TaskRunner`. One GL context — imgui draws through moderngl (`studio/imgui_backend.py`), and textures must be registered/forgotten with the backend.
- **One sqlite connection** behind an RLock; every `JobStore` method takes the lock; a partial `params` write goes through `merge_params` (read-modify-write under one hold).
- **`service/` is the only business-logic layer** — panes and tests both call it; refusals raise `service.errors` exceptions carrying `field`.
- **VRAM.** Coexist vs exclusive handoff is `queue._needs_handoff` (offloaded specs hand off unconditionally); teardown is `unload()` never `trim()`; admission at the door (`service.validation.check_vram`/`check_weights`) and re-checked at dispatch against free VRAM and host commit. Drop every reference before `_reclaim` — a release helper may not hold one.
- **`bpy` never runs in the app process** (`pipelines/blender_worker.py` is the only module that imports it), and **every subprocess goes in the `winjob` kill-on-close job** — a scan test enforces this.
- **`source.glb` is the reconstruction; `model.glb` is derived from it** (optimize-then-normalize, in that order); grounding always runs; every other export is a pure function of `model.glb`. Writes onto served files are staged, never in place.
- **`DERIVED_PARAMS`** strips worker-recorded values on rerun/promotion — anything the worker records about artifacts joins it, or a reroll wears a stale verdict. **`VECTOR_PARAMS`** is an allowlist (in `warlock/vectors.py`, because queue.py may not import `service`).
- **Native kernels** (`native/*.c`): the numpy fallback is never deleted; the bar is bit-identical parity (`/fp:precise`, no FMA), except `contours.c`, whose bar is the unit-edge set.
- **The headless packages** — `studio/inker/`, `clay/`, `plotter/`, `packwright/` — import no imgui/moderngl/pygame/`service` (import-pinning tests enforce the outward sets). Undo is addressed by uid, never index.
- **Crash recovery is `studio/journal.py`, one mechanism for six document kinds** — payload plus a `.meta.json` sidecar written *last* as the completion gate; declining keeps the files and there is no age-out.
- **A constant the stored corpus is keyed on gets a `docs/measurements/` document before it changes** (`trellis_band`, `SEAM_MAX`, the grade scale…). `hole_worst` is corpus-dependent — never present it as a quality scale.
- **There is no roadmap file.** `docs/TODO.md` was deleted on 2026-08-11 (commit `de87838`); git history keeps it. A `TODO.md §N` citation in code refers to the *deleted* `docs/LEFTOVERS.md` — chase both through `git log --diff-filter=D`, and do not mint new `§N` citations.
- **A manual chapter's number decides its order and part** — adding one is a renumbering; `tests/manual/` gates it in both directions.

See memory (`warlock-stack.md`) for more background on this stack.
