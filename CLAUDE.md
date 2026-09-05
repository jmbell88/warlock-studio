# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Warlock Studio: a fully offline, single-process desktop app for making game assets. Text or image prompt → SDXL 1.0 reference PNG (`pipelines/text2image_worker.py`, a child process) → `trellis-server.exe` (vendored native binary under `vendor/trellis/`, not the Python TRELLIS package) → textured GLB. One pygame window, one ModernGL context, imgui panels drawn through it (`studio/imgui_backend.py`). No HTTP, no browser. Windows-first; CI is Windows-only.

Thirteen modes, and `src/warlock/studio/modes.py` is the authoritative list: Home, Library, Create, then the workspaces Inker (pixel art), Clay (mesh primitives/booleans), Poser (rig + clips), Troupe (3D → sprite sheet, the one Experimental mode), Plotter (tile maps → Tiled), Packwright (atlas packing), Muse (ACE-Step music generation, output is a job row not a document), Sirens (chiptune tracker), then Review and Settings. The Manual, the guided tour (`studio/tour/`) and Flourish (procedural VFX inside Inker, `studio/inker/flourish/`) are deliberately *not* modes.

## Commands

```powershell
uv sync --extra studio --extra text2image --extra rig --extra music   # dev group installs by default; a bare `uv sync` prunes the extras and breaks ~10 test files at collection
uv run warlock            # the app
uv run warlock doctor     # deps, weights, config, which native kernels are live
uv run pytest             # default lane: -n 8 --dist loadfile, gpu+perf excluded, ~2 min for 16k+ tests
uv run pytest tests/test_x.py::test_name -n 0     # single test: skip worker startup
uv run pytest -m gpu -n 0   # real card + weights; xdist is refused for this lane (N workers = N 7 GB loads)
uv run pytest -m perf -n 0  # wall-clock budgets; meaningless under contention
uv run ruff check .
uv run python scripts/preflight.py   # release gate; run before a PR (`--fast` is what CI runs)
pwsh scripts\rebuild.ps1  # all of Windows CI locally, then the native DLL, then the installer, then prunes dist\
pwsh native\build.ps1     # optional C kernels → vendor/warlockc/warlockc.dll (gitignored); WARLOCK_NATIVE=0 forces numpy
```

- Python 3.13 is required (`bpy` ships 3.13 wheels only); the `rig` extra silently installs nothing elsewhere.
- **Never edit `src/` while the suite runs** — several tests read module source.
- `--dist loadfile` is load-bearing, not tuning: it keeps at most one imgui context alive per process and preserves module-level cache couplings. Don't switch to `--dist load` or `-n auto` (measured slower and OOM-prone; the reasoning is in `pyproject.toml`).
- `tests/conftest.py` pins `WARLOCK_HOME` to a throwaway dir and pins `memlog.system_memory`, so no test sees the real `~/.warlock` model library or depends on machine load. The gpu lane is exempt on purpose.

## Where the truth lives

- **`docs/INVARIANTS.md`** is the authoritative record of hard constraints *and the measured incident behind each*. Read the section for a subsystem before touching it; update it when an invariant changes. The bullets below are only the catastrophic-mistake preventers.
- **`docs/measurements/`** — a constant the stored corpus is keyed on (`trellis_band`, `SEAM_MAX`, the grade scale…) gets a dated document there *before* it changes.
- **`docs/COMPAT.md`** — the two interop ledgers (Plotter↔Tiled, Inker↔Aseprite) in one file. Only the Tiled rows are executable (`tests/plotter/test_compat_matrix.py` parses them as data).
- **`docs/MODELS.md`** — optional weights, their licences, and the SDXL recipe registry.
- **`TODO.md`** is the one plan file and it is not a roadmap: only what a human must do (hardware validation, art direction, design decisions) plus fully-specified, deliberately unstarted work. If an item there could be built, build it and strike it out. Never cite it from `src/` or `scripts/`; finished plan files are deleted, not ticked (`tests/test_ux_todo_fixes.py` sweeps dead filenames out).
- Style: comments explain *why*, naming the incident that motivated a guard. A regression test's name is the claim, and it must fail against the unfixed code. Manual changes ship in the same commit as the behaviour change.

## Architecture in one pass

- **Layers.** `service/` is the only business-logic layer; panes (`studio/panes/`) and tests both call it, never each other's internals. Refusals raise `service.errors` exceptions carrying a `field` so the UI can point at a control. `queue.py` schedules jobs; `db.py`'s `JobStore` is one sqlite connection behind an RLock, and partial `params` writes go through `merge_params`. Job kinds live in `_q_*.py` modules; adding one is a sweep of every stage-keyed table (an invariant, test-gated).
- **Three threads.** The pygame frame loop never blocks; the asyncio `Worker` lives on the `warlock-loop` thread; everything blocking goes through `TaskRunner` (`studio/tasks.py`). One GL context — textures must be registered/forgotten with the imgui backend.
- **Heavy work is always a child process, inside the `winjob` kill-on-close job**: text2image (`t2i_client` is the in-app handle; its stdin reader must never leave a read pending or the child deadlocks on its next native import), matting (`matting_worker.py`), Blender (`blender_worker.py` is the only module that imports `bpy`), music (`music_worker.py`, `separation_worker.py`), LoRA training, doctor's load probe. A scan test enforces the job wrapper.
- **Offline.** `HF_HUB_OFFLINE=1` is set in `warlock/__init__.py` before anything imports. The single exception is the user-initiated `fetch_worker` subprocess.
- **VRAM.** Admission at the door (`service.validation.check_vram`/`check_weights`), re-checked at dispatch; coexist-vs-exclusive handoff is `queue._needs_handoff`; teardown is `unload()` never `trim()`; drop every reference before `_reclaim`.
- **Artifacts.** `source.glb` is the reconstruction; `model.glb` is derived (optimize, then normalize; grounding always runs); every other export is a pure function of `model.glb`. Every write onto a served name is staged to a temp and `os.replace`d, never in place. `DERIVED_PARAMS` strips worker-recorded values on rerun/promotion; `VECTOR_PARAMS` is an allowlist in `vectors.py` (queue.py may not import `service`).
- **Headless editor packages** under `studio/` — `inker/` (and `inker/flourish/`), `clay/`, `plotter/`, `packwright/`, `sirens/`, `troupe/`, `muse/` — import no imgui, moderngl, pygame or `service` (`sirens/` and `muse/` also ban scipy so renders are byte-identical across a `uv sync`). Import-pinning tests enforce the exact outward set; adding an import means updating the pin deliberately. `studio/sirens_audio.py` is the only module that touches `pygame.mixer`. Undo is addressed by uid, never index.
- **Crash recovery** is `studio/journal.py`, one mechanism for every document kind: payload plus a `.meta.json` sidecar written *last* as the completion gate.
- **Native kernels** (`native/*.c`) are optimisations with a reference: the numpy fallback is never deleted and the bar is bit-identical parity (`/fp:precise`, no FMA), except `contours.c` whose bar is the unit-edge set.
- **Manual** (`studio/manual/`, `docs/manual/`): a chapter's number decides its order and part, so adding one is a renumbering, gated by `tests/manual/` in both directions. Chapters 01–19 are reserved for the tutorial series.
- **Tour** (`studio/tour/`): pure data with no outward imports, drawn from `_overlays`; steps wait on *named* conditions and never act for the reader.
- **Create** is one staged mode: the Reference stage is a command bar (`studio/create_brief.py`, *what* to make) over a recipe column (`panes/settings_2d.py`, *how*), and no control appears in both.

## Tooling notes

- `scripts/exercise_mode.py` (also the `/exercise-mode` skill) drives every control in a mode through the real input path and screenshots each press; `scripts/screenshot_modes.py` captures all modes.
- Subagents working in parallel must never `git stash`, `checkout` or `reset` — a stash from one fixer reverts the shared tree for the others.
