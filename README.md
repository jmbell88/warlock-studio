# Warlock Studio

A local, fully offline indie art studio for game assets — one desktop window, everything on your own GPU, no provider API, no account, no network. It folds the jobs of several tools into one pipeline-aware app: AI 2D/3D asset generation you would otherwise rent from Meshy.ai, polygon modelling, rigging and posing in the spirit of Blender, layered painting and animation in the spirit of Krita and Aseprite, tile-map editing that speaks Tiled's formats, and atlas packing that speaks TexturePacker's — and because they share one library, a mesh built in one workspace can be rigged, posed, sheeted and packed by the others.

The generation pipeline:

- **Image → 3D**: reference image → textured GLB (base colour plus a combined metallic/roughness texture; surface detail rides on vertex normals, not a normal map), powered by Microsoft **TRELLIS.2-4B** running natively via [trellis.cpp](https://github.com/pwilkin/trellis.cpp) (C++/GGML, CUDA).
- **Text → 3D**: prompt → reference image via a diffusers pipeline, loaded from a local weights dir. **SDXL-Turbo** is the default; ten base models are registered (`src/warlock/models.py`) from 4-step distillations to full-CFG SDXL, Playground, Juggernaut, DreamShaper and FLUX.2 klein, with per-job style LoRAs, IP-Adapter appearance conditioning, ControlNet silhouette lock, and a seamless-tile mode with seam measurement. See [docs/MODELS.md](docs/MODELS.md).
- **Rig → pose → sprite sheet**: fit one of seven template skeletons (humanoid, quadruped, bird, fish, insect, serpent, tailed biped), pose it with 3D gizmos or reusable poses from the Poser's global library, and bake poses into sprite sheets — flat or lit, 4/8/16 directions, optionally restyled into pixel art.
- **The approval gate**: text jobs stop at the reference by default — the image is shown full-size for approval (with candidate fan-out and per-stage seeds) before anything pays for a trellis run.

## The modes

A switch at the top of the window chooses between **thirteen** top-level modes (`src/warlock/studio/modes.py` is the authoritative list) drawn as three groups: the two ways in, the eight workspaces, and the three shelves. There is no per-mode key — the `Ctrl+K` command palette is the keyboard route, and `F1` opens the manual.

1. **Home** — what changed in this build, machine status and diagnostics, and everything you were recently working on. The app opens here every launch; no mode is remembered.
2. **Manual** — the full manual (`docs/manual/`) embedded in the window. `F1`, and every pane's (?) button, come here.
3. **2D** — the reference stage. Owns the prompt and every control that composes it: guidance selects, model and style LoRA, conditioning, seeds and candidates.
4. **3D** — the mesh stage. Owns no prompt controls at all: mesh, rig, pose and sprite-sheet decisions, plus surface re-texture of a finished mesh.
5. **Inker** — a layered raster editor *and* animation workspace: soft/pixel/square brushes with symmetry, 12 blend modes, a full selection suite, filters, gradients — and a timeline with tracks and cels, linked cels, per-frame durations, onion skinning, and tags with forward/reverse/ping-pong playback. Saves native [OpenRaster](https://www.openraster.org/) (`.ora`, Krita/GIMP-compatible; the animation rides inside), exports flattened PNG, animated GIF, or a sprite sheet with JSON sidecar. Autosaves every two minutes with crash recovery, and bridges the pipeline in both directions (edit a reference in place, or send a painting to the mesh stage).
6. **Clay** — modelling from primitives: vertex/edge/face element modes, extrude/bevel/subdivide/dissolve, UVs, a material palette, GLB import, and a diffable `.wblk` native format. Two ways out: export to the library as an ordinary asset (rigging, posing, sheets and every mesh export then work on it unchanged), or render it flat and send it to 3D.
7. **Poser** — authoring reusable poses against a skeleton template, kept in a global pose library rather than belonging to any one asset; poses can move their root.
8. **Review** — judging finished meshes with graded verdicts (−5..+5 plus tags), parameter sweeps over arbitrary setting axes, an advisory DINOv2-probe quality judge taught by in-app labelling, and the "What works" findings the verdicts add up to — which surface as hints beside the generate controls.
9. **Plotter** — a tile-map editor: grid, layer stack, tilesets and object layers, native `.wmap`, and Tiled interop in both directions (`.tmx`/`.tmj` import and export; unsupported Tiled features are refused explicitly, never partially loaded).
10. **Packwright** — a sprite-atlas packer: files, drops, Inker documents or library assets in; a deterministic atlas out (Grid or MaxRects, with trim/padding/extrude/power-of-two), as PNG plus TexturePacker JSON, and a `.tsx` for grid packs. Re-export of an unchanged document is byte-identical.
11. **Settings** — the app's own preferences: theme, UI scale, layout, and the model list, from which a missing one can be downloaded.
12. **Library** — every job ever generated, with filters, rerun and promotion, the trash and the prune.
13. **Profiles** — saved style profiles for the 2D form: nine fields (base model, LoRA and strength, negative prompt, platform, genre, era, setting, palette) plus an anchor image.

## What comes out

Everything but the primary artifacts is derived lazily on first request and cached (`service/derive.py`):

- **Per mesh**: `model.glb` (optimised, grounded), `source.glb` (raw reconstruction), STL, OBJ (zip), FBX, `collision.glb` (convex hull), `textures.zip`, `rig.glb` once rigged, and a baked GLB per saved pose.
- **Per reference**: `icon.png` (512 transparent cutout), `sprite.png` (trimmed, pivot recorded), `pixel_{32,64,128}.png` (palette-capped or mapped to a user palette in Oklab, optional dither), and a `manifest.json` carrying sizes, trim boxes, pivots and the recipe. Tiles additionally get an estimated PBR material set.
- **Per workspace**: sprite sheets as PNG plus an engine-neutral JSON sidecar (poses down, compass directions across; animated clips are cells with a `frame` above zero), plus a **Pixelate** variant restyling the whole sheet under one seed and palette; Inker's ORA/PNG/GIF/sheet; Plotter's TMX/TMJ; Packwright's atlases.
- **Bulk**: zip named artifacts across many jobs, or mirror exports into `WARLOCK_EXPORT_DIR` (e.g. a Godot project's `assets/`).

## Requirements

- Windows, NVIDIA GPU (tested: RTX 5090 / 32 GB; TRELLIS.2 alone fits in 16 GB)
- [uv](https://docs.astral.sh/uv/); Python ≥ 3.12, but **rigging needs 3.13** — `bpy` ships CPython 3.13 wheels only. On any other Python the rig extra installs nothing, `warlock doctor` reports rigging unavailable, and the app hides the rig controls; everything else works unchanged.
- ~16 GB disk for TRELLIS.2 GGUF weights (+ ~7 GB for SDXL-Turbo if using text-to-3D)

## Setup

```powershell
# 1. Python deps. --extra studio is the app's window and renderer; add
#    --extra text2image for text-to-3D (pulls torch cu128) and --extra rig
#    for rigging/posing/sheets. Contributors running the test suite want all
#    three -- a bare `uv sync` prunes the extras and breaks ~10 test files.
uv sync --extra studio --extra text2image --extra rig

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

These downloads are the only network use there is. The generation pipeline is fully offline — the app process never downloads anything (`HF_HUB_OFFLINE=1` is set at import, all model loads are `local_files_only`), and a missing weight produces a clear error and a `doctor` warning naming the exact command rather than a silent fetch.

Everything below the first two steps can also be fetched from inside the app, in **Settings → Models**: tick the rows you want and press Download. That does not make the app online-capable, and the mechanism is the point — the button spawns a separate `python -m warlock.pipelines.fetch_worker` process which sets `HF_HUB_OFFLINE=0` in its own environment, fetches one repository into a staging directory beside the destination, moves the files in only if it succeeded, and exits. The app process keeps `HF_HUB_OFFLINE=1` for its entire life. Free disk is checked against the whole plan before anything is spawned, and a failed fetch leaves no half-populated model directory.

**Optional models** — alternative base models (full SDXL, Playground, Juggernaut, DreamShaper, FLUX.2 klein), style LoRAs (3D render, PS1, pixel art), IP-Adapter, ControlNet, BiRefNet matting, DINOv2, ViTPose — live in [docs/MODELS.md](docs/MODELS.md) with the exact commands and the rationale for each recipe.

### Optional: rigging

Rigging (the `rig` extra) fits a template skeleton to a finished mesh and skins it with Blender's automatic weights, producing `rig.glb` alongside `model.glb` — on demand from the job's **rig** button, or queued automatically when the mesh finishes. Blender runs as a subprocess, never inside the app — `bpy` is process-global and can take the interpreter down on the kind of non-manifold geometry trellis sometimes produces; bone-heat failures fall back to envelope weights, recorded in `rig.json`. A rigged job gains the pose editor (forward-kinematic joint rotations, saved under names, each downloadable as a baked GLB) and sprite sheets; an unrigged prop still gets a rest-pose turnaround sheet. With the ViTPose weights present ([docs/MODELS.md](docs/MODELS.md)), humanoid joint placement reads the subject's actual landmarks off the reference image instead of assuming a T-pose.

### Optional: native kernels

```powershell
pwsh native\build.ps1
```

Builds `vendor/warlockc/warlockc.dll` from the C in `native/`. Entirely optional — every kernel has a numpy implementation it falls back to, and `warlock doctor` shows which is in use. What it buys is speed on hot raster paths (the four-view mesh audit drops from seconds to ~0.13 s); the bar is bit-identical parity with the numpy path, so old and new measurements stay comparable. Needs MSVC Build Tools, LLVM/clang, or zig; the script finds whichever is present. `WARLOCK_NATIVE=0` forces the numpy path.

## Run

```powershell
uv run warlock          # opens the desktop app
uv run warlock doctor   # checks dependencies, weights, and configuration
```

`warlock sweep --image assets/<job-id>/input.png --bands auto,4,8 --seed 42` regenerates one reference at several trellis `--band` values with a fixed seed and audits each resulting mesh.

`python -m warlock.bench` is the developer measurement suite behind quality decisions: versioned suites (`core-v1`, `pixel-v1`) run under named recipes, rendered to eight views per mesh and scored on silhouette IoU and DINOv2 identity (always A-against-B, never as an absolute). Subcommands: `suites`, `recipes`, `run`, `score`, `calibrate`, `prune`, `purge`.

### Configuration

There is no config file — everything is a `WARLOCK_*` env var; the full table lives in [docs/manual/16-configuration.md](docs/manual/16-configuration.md). The main knobs: `WARLOCK_DATA_DIR` (where assets and the job store live), `WARLOCK_EXPORT_DIR`, `WARLOCK_T2I_ROOT`/`WARLOCK_T2I_MODEL` (image-model home and default), and `WARLOCK_VRAM_EXCLUSIVE`.

On VRAM: the trellis server subprocess starts on the first 3D job and by default stays resident alongside SDXL-Turbo (~16 GB + ~7 GB on a 32 GB card); both are evicted after 10 minutes idle. `WARLOCK_VRAM_EXCLUSIVE=1` restores sequential VRAM use for text jobs (trellis stopped → image model loads, generates, unloads → trellis restarts) — needed for smaller GPUs, resolution 1536, or a resident FLUX.

## Development

```powershell
uv run pytest -q            # unit tests; the renderer's skip without a GL 3.3 context
uv run ruff check .
```

The app is a single process: a pygame window, one ModernGL context, and [imgui-bundle](https://github.com/pthom/imgui_bundle) panels drawn through that same context (the 3D viewport is a texture the panels show). Three threads — the frame loop, an asyncio worker for the GPU queue, and a task pool for blocking calls; jobs run one at a time. `warlock.service` is the single business-logic layer the panes and the tests both call. Model loads that would bloat the app process run in subprocesses that end (Blender, BiRefNet matting, the fetch worker), all tied to a kill-on-close job object.

Outputs land in `assets/<job_id>/` (`input.png`, `model.glb`, `rig.glb`/`rig.json`, `poses/`, `sheets/`); the SQLite job store lives at `assets/jobs.sqlite`.

Where to read more: the user manual is [docs/manual/00-index.md](docs/manual/00-index.md) (21 chapters, also embedded in the app), the hard invariants and their measured reasoning are `docs/INVARIANTS.md`, measurement write-ups are `docs/measurements/`, and `CHANGELOG.md` tracks releases.
