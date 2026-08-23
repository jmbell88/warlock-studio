# CLAUDE.md

Guidance for Claude Code in this repository. See `D:\Projects\CLAUDE.md` for how this project fits into the wider workspace.

## What this is

Local AI 3D asset generator: text or image prompt → SDXL 1.0 at full CFG (reference PNG; `sdxl_cfg`, with Turbo as the fast option) → `trellis-server.exe` (vendored native binary, not the Python TRELLIS package) → textured GLB, in a single-process desktop app: a pygame window, one ModernGL context, imgui panels drawn through it. No HTTP, no browser.

Eleven modes (`studio/modes.py` is authoritative): the rail runs Home/Library/Create, then the six workspaces Inker, Clay, Poser, Troupe, Plotter and Packwright, then a footer holding Review and Settings. Manual, Profiles and the guided tour are deliberately *not* modes — an overlay, a sheet and a second overlay respectively. Generation is one staged mode (Create), not the separate 2D/3D pair the older docs describe.

## Commands

- Install: `uv sync --extra studio --extra text2image --extra rig` — dev tooling is a dependency *group*, not an extra, and a bare `uv sync` prunes the extras (breaking ~10 test files at collection). `rig` needs a Python 3.13 venv or it silently installs nothing.
- Tests: `uv run pytest` — parallel by default (`-n 8 --dist loadfile`, ~45 s for 10,900+ tests). Never edit `src/` while the suite runs — several tests read module source. Three lanes are excluded from the default run and each is opt-in:
  - `uv run pytest -m gpu -n 0` — real card, real weights. Run it before changing model loading, VRAM accounting or conditioning. **Serial is enforced**: `-m gpu` under xdist is refused, because N workers means N simultaneous 7 GB loads onto one card. This is also the one lane that sees the real `~/.warlock` (see below).
  - `uv run pytest -m perf -n 0` — the wall-clock budget assertions. Meaningless under contention, so they are never in the parallel run.
  - `uv run pytest tests/test_x.py::test_name -n 0` — a single test is quicker without paying for worker startup.
- `tests/conftest.py` pins `WARLOCK_HOME` at a throwaway directory for the whole session. Every config root resolves under it, so a test that builds a bare `Config` no longer reads the developer's real model library — which is what kept ~150 s of genuine BiRefNet CPU inference in the suite, and made those tests produce a different matte on a machine that had the weights. It also pins `memlog.system_memory`, so no test's verdict depends on how loaded the machine is. The gpu lane is exempt from both, deliberately: pinned, it would go green as skips.
- Lint: `uv run ruff check .`
- Optional native kernels: `pwsh native\build.ps1` → `vendor/warlockc/warlockc.dll` (gitignored; not needed to run anything).

## The invariants live in `docs/INVARIANTS.md`

That file is the full, authoritative record of this codebase's hard invariants **and the measured reasoning behind them** — read the section covering a subsystem before modifying it, and update it when an invariant changes. The one-liners below are only the catastrophic-mistake preventers; every one has a longer argument there.

- **Fully offline.** `HF_HUB_OFFLINE=1` is the first thing the package does; nothing downloads at runtime. The single exception is the user-initiated `fetch_worker` subprocess, which goes online in its own environment only.
- **Three threads.** The pygame frame loop never blocks; the asyncio `Worker` lives on the `warlock-loop` thread; everything blocking goes through `TaskRunner`. One GL context — imgui draws through moderngl (`studio/imgui_backend.py`), and textures must be registered/forgotten with the backend.
- **One sqlite connection** behind an RLock; every `JobStore` method takes the lock; a partial `params` write goes through `merge_params` (read-modify-write under one hold).
- **`service/` is the only business-logic layer** — panes and tests both call it; refusals raise `service.errors` exceptions carrying `field`.
- **VRAM.** Coexist vs exclusive handoff is `queue._needs_handoff` (offloaded specs hand off unconditionally); teardown is `unload()` never `trim()`; admission at the door (`service.validation.check_vram`/`check_weights`) and re-checked at dispatch against free VRAM and host commit. Drop every reference before `_reclaim` — a release helper may not hold one.
- **The image pipeline never runs in the app process** either — `pipelines/text2image_worker.py` holds the checkpoint and `t2i_client.Text2ImageClient` is the handle, because `unload()` in-process returned the VRAM and kept 24 GiB of host commit. Its stdin reader must never leave a read *pending*: a blocking one deadlocks the child's next native import on Windows.
- **`bpy` never runs in the app process** (`pipelines/blender_worker.py` is the only module that imports it), and **every subprocess goes in the `winjob` kill-on-close job** — a scan test enforces this.
- **`source.glb` is the reconstruction; `model.glb` is derived from it** (optimize-then-normalize, in that order); grounding always runs; every other export is a pure function of `model.glb`. Writes onto served files are staged, never in place.
- **`DERIVED_PARAMS`** strips worker-recorded values on rerun/promotion — anything the worker records about artifacts joins it, or a reroll wears a stale verdict. **`VECTOR_PARAMS`** is an allowlist (in `warlock/vectors.py`, because queue.py may not import `service`).
- **Native kernels** (`native/*.c`): the numpy fallback is never deleted; the bar is bit-identical parity (`/fp:precise`, no FMA), except `contours.c`, whose bar is the unit-edge set.
- **The headless packages** — `studio/inker/`, `clay/`, `plotter/`, `packwright/` — import no imgui/moderngl/pygame/`service` (import-pinning tests enforce the outward sets). Undo is addressed by uid, never index.
- **Crash recovery is `studio/journal.py`, one mechanism for six document kinds** — payload plus a `.meta.json` sidecar written *last* as the completion gate; declining keeps the files and there is no age-out.
- **A constant the stored corpus is keyed on gets a `docs/measurements/` document before it changes** (`trellis_band`, `SEAM_MAX`, the grade scale…). `hole_worst` is corpus-dependent — never present it as a quality scale.
- **A finished plan is deleted rather than ticked.** The old `docs/TODO.md` went on 2026-08-11 (`de87838`); `REDESIGN.md`/`INKER_UPDATE.md`/`NEXT_SESSION.md` on 2026-08-15; `ASEPRITE_PARITY.md` and `UPDATE_2.md` on 2026-08-20; `LPC_ALT.md`, `EXE_PLAN.md` and `MY_TODO.md` on 2026-08-21, consolidated into the one file below. Git history keeps them all (`git log --diff-filter=D`), sections worth keeping are folded into `docs/INVARIANTS.md`, and citations name the *programme* rather than the file — `tests/test_ux_todo_fixes.py` sweeps the filenames back out of `src/` and `scripts/`. The `TODO.md §N` numbering is dead: all fourteen citations were rewritten on 2026-08-21 and no `§N` API exists.
- **There is one plan file: `TODO.md`, and it is not a roadmap.** It holds only what *a human* has to do — art direction, keyframe authoring, the real-Aseprite and real-Tiled passes, GPU runs, the design decisions that are the user's — plus the three fully-specified unstarted programmes (the installer, Troupe's phases 0e/6/7/8, and the host-commit defects). If an item there could be built, build it and strike it out rather than tracking it, and never cite the file from `src/` or `scripts/`.
- **The two interop ledgers are one file.** `docs/COMPAT.md` holds Plotter↔Tiled and Inker↔Aseprite. Only the Tiled part is executable — `tests/plotter/test_compat_matrix.py` parses its rows as data and its scan deliberately stops at the Aseprite heading.
- **A manual chapter's number decides its order and part** — adding one is a renumbering; `tests/manual/` gates it in both directions. **Chapters 01–19 are reserved for the tutorial series** (01–13 used), so a new tutorial chapter is a file and an `EXPECTED_KEYS` line rather than another twenty-five renames.
- **The guided tour points and waits, and never acts for the reader.** `studio/tour/` is data with *no* outward imports; `panes/tour.py` draws it from `_overlays`, above the Manual and below the palette, and is deliberately not in `_modal_open`. A step waits on a *named* condition, and the vocabulary and its evaluator are asserted against each other. The scrim fills around holes because draw data cannot be split — and it needs one for the card too, or the tour dims itself. `anchors` is `layout.FRAME_PANES` for controls; positions are read, never computed.

See memory (`warlock-stack.md`) for more background on this stack.
