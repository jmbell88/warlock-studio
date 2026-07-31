# Animancer3D — Local Meshy.ai Clone (v1 Design & Plan)

## Context

The user wants a self-hosted alternative to Meshy.ai (text-to-3D / image-to-3D SaaS) instead of paying for a provider API. `D:\Projects\Animancer3D` is empty — this is a greenfield project. Decisions settled in brainstorming:

- **Purpose:** game assets for Godot (characters/props for the arpg project et al.)
- **V1 scope:** image-to-3D and text-to-3D. NOT in v1: retexturing existing meshes, auto-rig/animation, 3D-print prep.
- **Shape:** local web UI + FastAPI backend (matches the user's house stack: Python 3.12, uv, src layout, FastAPI, SQLite, ruff, pytest).
- **Runtime:** native Python pipelines loaded in-process (no ComfyUI dependency).
- **Models:** Microsoft **TRELLIS.2** (MIT) for image→3D; **Flux.1-schnell** (or SDXL fallback) via `diffusers` for the text→image stage of text-to-3D.
- **Hardware:** RTX 5090 (32 GB VRAM), 64 GB RAM, Windows 11 — comfortably fits either model, but not both resident at once alongside headroom; needs load/unload management.

## Architecture

```
animancer3d/
├── pyproject.toml            # uv, hatchling, ruff, pytest — same pattern as DocForge/Panthera
├── src/animancer3d/
│   ├── app.py                # FastAPI app factory + routes
│   ├── config.py             # paths, model ids, VRAM options (env-overridable)
│   ├── db.py                 # SQLite (sqlite3 stdlib or SQLAlchemy) — jobs + assets tables
│   ├── queue.py              # asyncio single-worker GPU job queue
│   ├── pipelines/
│   │   ├── manager.py        # loads/unloads models; guarantees one model resident at a time
│   │   ├── text2image.py     # Flux.1-schnell via diffusers → reference image
│   │   ├── image2model.py    # TRELLIS.2 → textured mesh (GLB)
│   │   └── postprocess.py    # trimesh/pymeshlab: decimation (low-poly target), format conversion, thumbnail render
│   └── static/               # web UI: vanilla JS + three.js (vendored), no build step for v1
│       ├── index.html        # generate form (prompt or image upload), job list, asset gallery
│       └── viewer.js         # three.js GLTFLoader orbit viewer
├── assets/                   # gitignored — per-job output dirs (input.png, model.glb, model.obj, thumb.png)
└── tests/
```

### Flow

1. UI POSTs a job (`prompt` or uploaded image + options: target polycount, seed, texture resolution).
2. Job row inserted (`queued`); asyncio worker picks it up (one GPU job at a time).
3. Text jobs: `text2image` generates a clean single-object reference image (prompt template biases toward "single object, neutral background, 3/4 view"); image jobs skip this.
4. `image2model` runs TRELLIS.2 → GLB with PBR textures.
5. `postprocess`: optional decimation to target polycount (game-ready low-poly), export OBJ/STL variants on demand, render a thumbnail.
6. UI polls job status; finished assets appear in the gallery with an interactive three.js preview and download buttons (GLB primary, OBJ/STL secondary).

### VRAM management

`pipelines/manager.py` owns the GPU: load model → run → move to CPU/unload before the next stage loads. Text-to-3D is sequential (Flux, unload, TRELLIS.2). Keep TRELLIS.2 resident between consecutive 3D jobs to avoid ~30–60 s reload cost; evict on idle timeout or when Flux is needed.

## Known risk (address first)

**TRELLIS-on-Windows.** TRELLIS historically depends on custom CUDA extensions (nvdiffrast, spconv/flex-attention kernels) that are painful on native Windows, and the RTX 5090 (Blackwell, sm_120) requires a recent torch (cu128+) build. Phase 1 is a standalone spike proving `image → GLB` end-to-end on this machine **before** any app code. Fallbacks if native Windows fails: (a) run the model worker under WSL2 with the FastAPI app talking to it over localhost, (b) switch to Hunyuan3D 2.x whose Windows story is better. The job-queue/pipeline API is model-agnostic so a swap only touches `image2model.py`.

## Implementation phases

1. **Model spike (do first):** scratch script that installs TRELLIS.2 deps and converts one test image → textured GLB on the 5090. Record working torch/CUDA/extension versions. Go/no-go gate for the native-Windows approach.
2. **Scaffold:** uv project, FastAPI app factory, config, SQLite schema (`jobs`, `assets`), health route. `uv run pytest` + `uv run ruff check .` green.
3. **Job queue + image-to-3D:** asyncio worker, model manager, wrap the spike code as `image2model.py`, `POST /api/jobs` (image upload) → GLB in `assets/<job_id>/`.
4. **Text-to-3D:** Flux.1-schnell `text2image.py` + sequential VRAM handoff; prompt template for object-centric reference images.
5. **Post-processing:** trimesh/pymeshlab decimation to user-set polycount, OBJ/STL export endpoints, offscreen thumbnail.
6. **Web UI:** single-page `static/` app — generation form, job status list (polling), gallery, three.js GLB viewer. Vendored three.js, no bundler.
7. **Polish:** cancel/delete jobs, disk cleanup, README with setup instructions (model weights download on first run via `huggingface_hub`).

## Verification

- Phase 1: visually inspect the spike's GLB in the three.js viewer or Blender/Godot import.
- Each phase: `uv run pytest` (unit tests mock the GPU pipelines; one optional `-m gpu` marked integration test runs a real tiny generation) and `uv run ruff check .`.
- End-to-end: start server (`uv run uvicorn animancer3d.app:app`), submit a text job and an image job through the UI, download GLB, import into Godot 4 and confirm textures/scale.

## Out of scope for v1 (explicit)

Retexturing user meshes, auto-rigging/animation, watertight repair / 3D-print prep, multi-model selection, auth/multi-user, hosted deployment.

## Implementation note (post-plan)

The Phase-1 spike found [trellis.cpp](https://github.com/pwilkin/trellis.cpp) — a C++/GGML port of TRELLIS.2-4B with prebuilt Windows CUDA binaries, GGUF weights, and a resident HTTP server, claiming output parity with the reference pipeline. `image2model` therefore drives a managed `trellis-server.exe` subprocess over localhost HTTP instead of loading Python TRELLIS in-process. This supersedes the "native Python pipelines" choice for the 3D stage only (Flux remains in-process via diffusers); it avoids the Linux-only conda stack entirely and is strictly better than the WSL2 fallback the plan anticipated.
