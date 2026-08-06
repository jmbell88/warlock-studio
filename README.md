# Warlock Studio

A local, self-hosted alternative to Meshy.ai for generating game-ready 3D assets — no provider API, everything runs on your own GPU.

- **Image → 3D**: reference image → textured GLB (base colour plus a combined metallic/roughness texture; surface detail rides on vertex normals, not a normal map), powered by Microsoft **TRELLIS.2-4B** running natively via [trellis.cpp](https://github.com/pwilkin/trellis.cpp) (C++/GGML, CUDA).
- **Text → 3D**: prompt → reference image via **SDXL-Turbo** (diffusers, loaded from a local weights dir; FLUX.1-schnell swappable) → same image-to-3D pipeline.
- **Rig → pose → sprite sheet**: fit a template skeleton, pose it with 3D gizmos in the viewport, and bake the poses into a 2D sprite sheet with a JSON sidecar.
- **Desktop app**: one window — no server, no browser, no localhost. A settings pane, an interactive ModernGL preview and an asset inspector, with GLB download ready to import into Godot, Blender, Unity, or Unreal. Text jobs stop at the reference by default: the image is shown full-size for approval (with candidate fan-out and per-stage seeds) before anything pays for a trellis run.

## Requirements

- Windows, NVIDIA GPU (tested: RTX 5090 / 32 GB; TRELLIS.2 alone fits in 16 GB)
- [uv](https://docs.astral.sh/uv/)
- ~16 GB disk for TRELLIS.2 GGUF weights (+ ~7 GB for SDXL-Turbo if using text-to-3D)

## Setup

```powershell
# 1. Python deps. --extra studio is the app's window and renderer; add
#    --extra text2image for text-to-3D (pulls torch cu128). The dev tools
#    (pytest, ruff) are uv's default dependency group and come along free.
uv sync --extra studio

# 2. trellis.cpp CUDA server binary -> vendor/trellis/
#    https://github.com/pwilkin/trellis.cpp/releases (trellis-cuda-windows-x64.zip)
#    vendored build: v0.5.4 (2026-07-27)

# 3. TRELLIS.2 GGUF weights -> models/trellis2-gguf/
uvx hf download ilintar/trellis2-gguf --include "*.gguf" --exclude "q4/*" --exclude "q8/*" `
  --local-dir models/trellis2-gguf

# 4. SDXL-Turbo weights (fp16 variant, ~7 GB) -> models/sdxl-turbo/  (text-to-3D only)
uvx hf download stabilityai/sdxl-turbo --include "*.json" --include "*.txt" --include "*fp16.safetensors" `
  --exclude "sd_xl_turbo_1.0*" --local-dir models/sdxl-turbo
```

These one-time downloads are the only network use. The app itself is fully offline — it never downloads anything (`HF_HUB_OFFLINE=1` is set at import, all model loads are `local_files_only`); missing weights produce a clear error and a `doctor` warning instead of a download.

### Optional image models and style LoRAs

The reference image is the single biggest lever on final mesh quality — TRELLIS can only be as good as the picture it is handed — so the image model and an optional style LoRA are per-job choices in the guidance panel. Everything below is optional and independently skippable; `warlock doctor` lists each one with the exact command to fetch it. Base models are one-resident-at-a-time (a 32 GB card holds trellis plus a single SDXL-class pipe, not two), so switching between jobs costs a reload; style LoRAs are adapters on the resident pipe and switch for free.

```powershell
# SDXL 1.0 + Hyper-SD (~7 GB + 787 MB). Style LoRAs are trained against full
# SDXL at 20-25 steps with CFG, so they land noticeably stronger here than on
# Turbo's 4 steps at guidance 0. Hyper-SD buys the step count back.
uvx hf download stabilityai/stable-diffusion-xl-base-1.0 `
  --include "*.json" --include "*.txt" --include "*fp16.safetensors" --local-dir models/sdxl-base-1.0
uvx hf download ByteDance/Hyper-SD Hyper-SDXL-4steps-lora.safetensors --local-dir models/loras

# Playground v2.5 (~7 GB): highest fidelity, ~25 steps with CFG, correspondingly slower.
uvx hf download playgroundai/playground-v2.5-1024px-aesthetic `
  --include "*.json" --include "*.txt" --include "*fp16.safetensors" --local-dir models/playground-v2.5

# Style LoRAs -> models/loras/
uvx hf download goofyai/3d_render_style_xl 3d_render_style_xl.safetensors --local-dir models/loras
uvx hf download artificialguybr/3DRedmond-V1 `
  3DRedmond-3DRenderStyle-3DRenderAF.safetensors --local-dir models/loras
uvx hf download artificialguybr/ps1redmond-ps1-game-graphics-lora-for-sdxl `
  PS1Redmond-PS1Game-Playstation1Graphics.safetensors --local-dir models/loras
```

FLUX is not offered: both `dev` and `schnell` are click-through gated on Hugging Face, and 12B parameters will not coexist with trellis on one card. To use a local FLUX copy anyway: download it yourself (`uvx hf auth login` for the download only), point `WARLOCK_T2I_DIR` at it, and set `WARLOCK_VRAM_EXCLUSIVE=1`. Note that `WARLOCK_T2I_DIR` only redirects *where* the built-in `turbo` entry loads from — it still runs at that entry's settings (512 px, 4 steps, guidance 0), which suit schnell-like distilled checkpoints and nothing else. A model that needs different settings wants a `models.py` entry, not this variable.

### Optional: rigging

```powershell
uv sync --extra rig
```

Fits a template skeleton (humanoid, quadruped, bird, fish, insect, serpent, or tailed biped) to a finished mesh and skins it with Blender's automatic weights, producing `rig.glb` alongside `model.glb`. Rig it on demand from the job's **rig** button, or tick "Fit a skeleton when the mesh is done" on the generate form to queue it automatically.

Blender runs as a subprocess, never inside the app — `bpy` is process-global and can take the interpreter down on the kind of non-manifold geometry trellis sometimes produces. Bone-heat weighting fails outright on such meshes; the worker catches that and falls back to envelope weights, recording which was used in `rig.json` rather than failing the job.

Once a job is rigged the inspector gains a **Pose** panel: click **Edit pose** to swap the preview to the rig, click a joint to attach a rotation gizmo, and save the result under a name. Poses are forward-kinematic only — each joint's local rotation, nothing else — and each one can be downloaded as its own posed GLB, baked by Blender on first request and cached afterwards. Saving under an existing name replaces that pose rather than adding a near-duplicate.

### Sprite sheets

Any finished mesh can be baked to a 2D sprite sheet: **poses down, eight compass directions across**. The panel previews the eight views live in the viewport so the framing, camera elevation and flat/lit choice can be judged before committing; the sheet that ships is rendered in Blender's EEVEE with a transparent film, queued like any other job.

Each sheet is a PNG plus an engine-neutral JSON sidecar — pixel rectangles and what each one shows, with no Godot `AtlasTexture` or Unity `SpriteMetaData` opinions baked in. Cells are a flat list rather than a nested grid, so animated clips can join later as cells with a `frame` above zero without a format change:

```json
{"version": 1, "columns": 8, "rows": 2, "frame_size": 128,
 "yaws": [0, 45, 90, 135, 180, 225, 270, 315],
 "cells": [{"index": 0, "x": 0, "y": 0, "w": 128, "h": 128,
            "pose": "751cf6147291", "pose_name": "idle", "yaw": 0, "frame": 0}]}
```

Sheets need Blender, so they live behind the same `rig` extra. An unrigged prop still gets one — a turnaround of its rest pose.

`bpy` ships **CPython 3.13 wheels only**. On any other Python the extra installs nothing, `warlock doctor` reports rigging as unavailable, and the app hides the rig controls — everything else works unchanged.

## Run

```powershell
uv run warlock          # opens the desktop app
uv run warlock doctor   # checks dependencies, weights, and configuration
```

`warlock sweep` is the measurement tool behind the band decision below: it regenerates one reference image at several trellis `--band` values with a fixed seed and audits each resulting mesh, writing the GLBs and a log to `--out` (default `sweep/`).

```powershell
uv run warlock sweep --image assets/<job-id>/input.png --bands auto,4,8 --seed 42
```

### Inker

The third top-level mode is a layered raster editor -- soft brushes, layers with
blend modes, a full selection suite (rectangle, ellipse, lasso, magic wand, with
Shift to add and Alt to subtract), free transform, gradients, blur and smudge,
symmetry and a grid. It opens any image, keeps several documents in tabs, and
saves natively as [OpenRaster](https://www.openraster.org/) (`.ora`) -- a zip of
layer PNGs that Krita and GIMP both read and write -- or exports a flattened PNG.

It is wired into the pipeline in both directions. **Open in Inker** on a finished
reference edits it in place: saving writes `input.png` through the same path the
old inline editor used (the untouched original is kept once, as `input.orig.png`,
so *Revert to original* always works) and keeps the layers beside it in
`paint.ora`, which is internal working state and never served. Going the other
way, **Save as reference** adds what you painted to the library as a finished
reference -- measured, so the quality gate has real data -- and **Send to 3D**
queues the mesh stage from the flattened image.

Keys follow Aseprite where it has one: `B`/`E`/`G`/`M`/`L`/`W`/`V`/`I` pick tools,
`X` swaps the two colours, `[` and `]` size the brush (with Shift, its hardness),
space-drag or middle-drag pans, `Ctrl+T` transforms, `Ctrl+0` fits and `Ctrl+1` is
100%.

The trellis server subprocess starts on the first 3D job and by default stays resident in VRAM alongside SDXL-Turbo (~16 GB + ~7 GB on a 32 GB card); both are evicted after 10 minutes idle (configurable). Set `WARLOCK_VRAM_EXCLUSIVE=1` to restore sequential VRAM use for text jobs (trellis stopped → image model loads, generates, unloads → trellis restarts) — needed for smaller GPUs, resolution 1536, or a resident Flux.

The main knobs are env-overridable (the full table lives in
[docs/manual/11-configuration.md](docs/manual/11-configuration.md)): `WARLOCK_DATA_DIR`, `WARLOCK_DB`, `WARLOCK_TRELLIS_EXE`, `WARLOCK_TRELLIS_MODELS`, `WARLOCK_TRELLIS_PORT`, `WARLOCK_TRELLIS_IDLE`, `WARLOCK_T2I_ROOT` (where image models and `loras/` live), `WARLOCK_T2I_MODEL` (default base model key), `WARLOCK_T2I_DIR` (redirects the `turbo` entry only), `WARLOCK_VRAM_EXCLUSIVE`, `WARLOCK_RIG_TEMPLATE`, `WARLOCK_RIG_TIMEOUT`, `WARLOCK_POSE_TIMEOUT`, `WARLOCK_SHEET_TIMEOUT`, `WARLOCK_TRELLIS_WEBP` (WebP rather than PNG for trellis textures), `WARLOCK_TRELLIS_TEX_RES` (texture resolution), `WARLOCK_TRELLIS_BAND` (mesh extraction band; unset by default, and measurement says leave it that way).

## Development

```powershell
uv run pytest -q            # unit tests; the renderer's skip without a GL 3.3 context
uv run ruff check .
```

The app is a single process: a pygame window, one ModernGL context, and
[imgui-bundle](https://github.com/pthom/imgui_bundle) panels drawn through that
same context (the 3D viewport is a texture the panels show). The queue, the job
store and every pipeline are unchanged from the server version — they were
always transport-agnostic, and `warlock.service` is the layer the app and the
old HTTP routes both called.

Outputs land in `assets/<job_id>/` (`input.png`, `model.glb`, `rig.glb`/`rig.json` once rigged — the rig lives beside the mesh it was fitted to, not in the rig job's own directory — `poses/<pose_id>.json` plus its baked `.glb` for each saved pose, and `sheets/<sheet_id>.png` plus its sidecar for each sprite sheet). The SQLite job store lives at `assets/jobs.sqlite`.
