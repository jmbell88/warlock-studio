# Warlock Review — Implementation Plan Index

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement each plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

The 25 review recommendations cover four independent subsystems. Each has its own
plan and each produces working, testable software on its own. Execute them in any
order **except** the one hard dependency noted below.

| Plan | Items | File |
|---|---|---|
| A — Faster iteration | 1–6 | `2026-08-02-warlock-a-faster-iteration.md` |
| B — Game-ready meshes | 7–13 | `2026-08-02-warlock-b-game-ready-meshes.md` |
| C — Character pipeline | 14–19 | `2026-08-02-warlock-c-character-pipeline.md` |
| D — Workshop ergonomics | 20–25 | `2026-08-02-warlock-d-workshop-ergonomics.md` |

## Cross-plan dependency

**Plan A Task 4 creates the first entry in `db.MIGRATIONS` and the pattern for
adding a column.** Plan D Task 1 (`name`/`tags`/`favorite`) adds the second
entry. If Plan D runs first, it must create the migration itself following the
identical pattern; `MIGRATIONS` is append-only and entries must never be edited
once shipped, so whichever plan runs second appends rather than edits.

Plan B Task 3 (`source.glb` + optimize) changes what `model.glb` *is*. Plan C's
sheet work reads `model.glb`/`rig.glb` by name and is unaffected.

## Global Constraints

Every task in every plan is bound by these. They are the hard invariants from
`CLAUDE.md` plus the project's toolchain.

- **Python 3.12+**, uv-managed. Install: `uv sync --extra dev --extra text2image --extra rig`
  (the `rig` extra needs a **Python 3.13** venv or bpy silently installs nothing).
- **Verify every task with `uv run pytest -q` and `uv run ruff check .`** — both must
  be green before the commit step. Baseline is ~332 passing tests.
- **Single sqlite3 connection, serialized by `JobStore._lock` (an `RLock`).** Any new
  method that reaches `self._conn` takes `self._lock` first. `asyncio.to_thread`
  serializes nothing.
- **Fully offline.** No runtime network calls, ever. All weights load with
  `local_files_only=True`. Missing weights fail with a one-time manual `hf download`
  instruction. `HF_HUB_OFFLINE=1` in `src/warlock/__init__.py` stays first.
- **VRAM handoff.** When `Config.vram_exclusive` is set, `Worker._generate` must keep
  stop-before-load / unload-before-next-start for text jobs.
- **`bpy` only in `pipelines/blender_worker.py`.** `rigging.py` stays importable with
  no bpy. Every Blender op runs out-of-process via `rigging.run_worker`, dispatched
  from callers through `asyncio.to_thread`.
- **No frontend build step.** `static/index.html` and `static/app.js` are served as-is.
  No npm, no bundler, no `package.json`. three.js r170 addons come from
  `static/vendor/three/` only.
- **A rig writes into the *source* job's directory**, not the rig job's own.
- **A pose is a map of glTF node-local quaternions, stored XYZW**; the worker converts
  to WXYZ. `bone_heuristic="BLENDER"` on import and
  `export_rest_position_armature=False` on export are load-bearing.
- **The sprite-sheet grid is decided on the host** in `pipelines/sheet.py`, never in
  Blender. Every `import_scene.gltf` in the worker is followed by
  `_purge_import_helpers`.
- **Commit subject format:** `Warlock v0.0.1` (version stays fixed unless the user
  asks for a bump).

## Engine acceptance (run once per plan, at the end)

- Import a produced GLB into Godot 4 (`D:\Projects\arpg` is available as a testbed).
- Load a sheet + sidecar in Pygame-CE per the `docs/NEXT.md` acceptance checks.
