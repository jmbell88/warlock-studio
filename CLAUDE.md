# CLAUDE.md

This file provides guidance to Claude Code when working in this repository. See `D:\Projects\CLAUDE.md` for how this project fits into the wider workspace.

## What this is

Local AI 3D asset generator: text or image prompt → SDXL-Turbo (reference PNG) → `trellis-server.exe` (vendored native binary, not the Python TRELLIS package) → textured GLB, served from a FastAPI app with a plain-JS/three.js frontend.

## Hard invariants

**Single sqlite3 connection is safe only because of one event loop.** `JobStore` (`db.py`) wraps one unsynchronized `sqlite3` connection with no locking. This is safe *only* because every access is funneled through `asyncio.to_thread` from the single asyncio event loop in `queue.py` — that serializes all DB calls onto one worker thread at a time. `progress.py` is explicitly in-memory-only and never touches the DB from its own reader thread for the same reason (see `progress.py:14-17`). Do not call `JobStore` methods directly from a thread that isn't dispatched via `asyncio.to_thread`, and do not add a second writer thread without rethinking this.

**VRAM handoff order (queue.py:1-5, 158-174).** The 3D server (`trellis-server.exe`) and SDXL-Turbo cannot fit in VRAM together. For a text job: `trellis.stop()` first, then load and run SDXL, then unload SDXL (`finally: t2i.unload()`) before the next job can start `trellis-server.exe` again. Any change to job dispatch must preserve stop-before-load / unload-before-next-start, or you get an OOM that only reproduces under load, not in tests.

**No frontend build step.** `static/index.html` / `static/app.js` are served as-is; there is no npm, no bundler, no `package.json`. three.js and its addons are vendored directly under `static/vendor/three/`. Adding a build step here is a scope change, not a fix.

## Stack

- Python 3.12+, FastAPI, uv-managed (`uv sync`, `uv run pytest`, `uv run ruff check .`)
- `trellis-server.exe` is a vendored compiled binary (`vendor/`), not a Python package — confirm behavior against its actual CLI help / exposed routes, not against the upstream TRELLIS project's docs
- SDXL-Turbo is the default image model because Flux is gated (requires HF auth); see `pipelines/text2image.py`
- `docs/DESIGN.md` is the design reference

See memory (`animancer3d-stack.md`) for more background on this stack.
