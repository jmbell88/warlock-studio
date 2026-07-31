# Animancer3D

A local, self-hosted alternative to Meshy.ai for generating game-ready 3D assets — no provider API, everything runs on your own GPU.

- **Image → 3D**: reference image → textured GLB with full PBR maps (albedo, normal, roughness, metallic), powered by Microsoft **TRELLIS.2-4B** running natively via [trellis.cpp](https://github.com/pwilkin/trellis.cpp) (C++/GGML, CUDA).
- **Text → 3D**: prompt → reference image via **SDXL-Turbo** (diffusers; swap in FLUX.1-schnell with HF auth) → same image-to-3D pipeline.
- **Web UI**: generation form, job queue, and an interactive three.js preview with GLB download — ready to import into Godot, Blender, Unity, or Unreal.

## Requirements

- Windows, NVIDIA GPU (tested: RTX 5090 / 32 GB; TRELLIS.2 alone fits in 16 GB)
- [uv](https://docs.astral.sh/uv/)
- ~16 GB disk for TRELLIS.2 GGUF weights (+ ~7 GB for SDXL-Turbo if using text-to-3D)

## Setup

```powershell
# 1. Python deps (add --extra text2image for text-to-3D; pulls torch cu128)
uv sync --extra dev

# 2. trellis.cpp CUDA server binary -> vendor/trellis/
#    https://github.com/pwilkin/trellis.cpp/releases (trellis-cuda-windows-x64.zip)

# 3. TRELLIS.2 GGUF weights -> models/trellis2-gguf/
uvx hf download ilintar/trellis2-gguf --include "*.gguf" --exclude "q4/*" --exclude "q8/*" `
  --local-dir models/trellis2-gguf
```

Text-to-image weights (SDXL-Turbo, ~7 GB) download automatically from HuggingFace on the first text job. For higher quality set `ANIMANCER3D_T2I_MODEL=black-forest-labs/FLUX.1-schnell` and `ANIMANCER3D_T2I_SIZE=1024` (gated repo — needs `uvx hf auth login`).

## Run

```powershell
uv run animancer3d          # serves http://127.0.0.1:8420
```

The trellis server subprocess starts on the first 3D job, stays warm between jobs, and is evicted after 10 minutes idle (configurable). For text jobs the worker sequences VRAM: trellis is stopped, Flux generates the reference image and unloads, then trellis runs.

Everything is env-overridable: `ANIMANCER3D_DATA_DIR`, `ANIMANCER3D_DB`, `ANIMANCER3D_TRELLIS_EXE`, `ANIMANCER3D_TRELLIS_MODELS`, `ANIMANCER3D_TRELLIS_PORT`, `ANIMANCER3D_TRELLIS_IDLE`, `ANIMANCER3D_T2I_MODEL`, `ANIMANCER3D_T2I_SIZE`.

## Development

```powershell
uv run pytest -q            # unit tests (no GPU needed)
uv run ruff check .
```

Outputs land in `assets/<job_id>/` (`input.png`, `model.glb`). The SQLite job store lives at `assets/jobs.sqlite`.
