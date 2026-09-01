# Warlock Studio

A local, fully offline indie art studio for game assets — one desktop window, everything on your own GPU, no provider API, no account, no network. It folds the jobs of several tools into one pipeline-aware app: AI 2D/3D asset generation you would otherwise rent from Meshy.ai, polygon modelling, rigging and posing in the spirit of Blender, layered painting and animation in the spirit of Krita and Aseprite, tile-map editing that speaks Tiled's formats, and atlas packing that speaks TexturePacker's — and because they share one library, a mesh built in one workspace can be rigged, posed, sheeted and packed by the others. The newest of those paths runs end to end: a described character becomes a reference, a mesh, a rig, and a directional sprite sheet you can open on a timeline and paint over.

The generation pipeline:

- **Image → 3D**: reference image → textured GLB (base colour plus a combined metallic/roughness texture; surface detail rides on vertex normals, not a normal map), powered by Microsoft **TRELLIS.2-4B** running natively via [trellis.cpp](https://github.com/pwilkin/trellis.cpp) (C++/GGML, CUDA).
- **Text → 3D**: prompt → reference image via a diffusers pipeline, loaded from a local weights dir. **SDXL 1.0 at full CFG** is the default and the one base download the setup below asks for — 30 steps at 1024 px, with the negative prompt and ControlNet live; the same 7 GB also powers three faster recipes over the same weights, and **SDXL-Turbo** remains the 4-step fast option one install away. Eleven base models are registered (`src/warlock/models.py`) from 4-step distillations to full-CFG SDXL, Playground, Juggernaut, DreamShaper and FLUX.2 klein, with per-job style LoRAs, IP-Adapter appearance conditioning, ControlNet silhouette lock, and a seamless-tile mode with seam measurement. See [docs/MODELS.md](docs/MODELS.md).
- **Text → 2D sheets**: the same prompt as a **tileset**, in one of three layouts — *Materials* (1–16 surface descriptions × 1–4 draws, capped at 64 cells, each cell its own **seamless** generation, so the tiles genuinely repeat), *Terrain set* (an inside and an outside surface composited into a complete 47-case blob autotile that lands in Plotter with the Terrain tool live, by record, with no import prompt), or *Grid (legacy)* (the original 8×8 single generation onto a ControlNet grid guide — the only layout offering 3/4 and 2:1 isometric, and the only one offering a 48 px tile; the seamless two are top-down at 16/32/64 px, since a seamless tile must divide the 1024 px material and must wrap a square). Or as a **sprite sheet**: pick an action (idle, walk, run, attack, cast, hurt, jump) and a direction count, and it draws the character first, keeps it as its own asset, then imagines candidate sheets from it with animation tags and frame durations baked in. Neither is reconstructed into a mesh; a tileset goes on to Plotter or Packwright. The legacy grid mechanism is measured (`docs/measurements/2026-08-18-tile-sheet-grid.md`); its art direction is not settled, and that document says so.
- **Rig → pose → sprite sheet**: fit one of seven template skeletons (humanoid, quadruped, bird, fish, insect, serpent, tailed biped), pose it with 3D gizmos or reusable poses from the Poser's global library, and bake poses into sprite sheets — flat or lit, 4/8/16 directions, optionally restyled into pixel art. Beyond single poses, **Troupe** renders whole animation clips: keyframes authored in the Poser, interpolated into a 256-cell character sheet of five animations across eight directions.
- **The approval gate**: text jobs stop at the reference by default — the image is shown full-size for approval (with candidate fan-out and per-stage seeds) before anything pays for a trellis run.

## The modes

A rail down the left of the window chooses between **twelve** top-level modes
(`src/warlock/studio/modes.py` is the authoritative list, and `RAIL_GROUPS` is
the grouping) in three sections: **Pipeline**, **Workspaces**, and an
unlabelled footer. There is no per-mode key — the `Ctrl+K` command palette is
the keyboard route, and `F1` opens the manual as an overlay over whatever you
are looking at.

**Pipeline** — what you have, and making another one:

1. **Home** — what changed in this build, machine status and diagnostics, and
   everything you were recently working on. The app opens here every launch; no
   mode is remembered.
2. **Library** — every job ever generated, with filters, rerun and promotion,
   the trash and the prune. It sits *before* Create because that is the order
   of the question: what do I have, then make another one.
3. **Create** — the whole generation pipeline in one mode, staged: the
   reference (prompt, guidance, base model and style LoRA, conditioning, seeds
   and candidates), then the mesh, then rig, pose, sprite sheet and surface
   re-texture. Text jobs stop at the reference for approval by default, before
   anything pays for a trellis run. Saved **style profiles** for the reference
   form live here as a sheet rather than as a mode of their own.

**Workspaces** — each fills the window with its own three-column layout:

4. **Inker** — a layered raster editor *and* animation workspace. Soft, pixel
   and square brushes with symmetry, 19 blend modes, a full selection suite,
   filters and gradients; true **indexed and grayscale colour modes**; **tilemap
   layers** over shared tilesets; and a timeline with tracks and cels, linked
   cels, per-frame durations, onion skinning, and tags with forward/reverse/
   ping-pong playback. Saves native [OpenRaster](https://www.openraster.org/)
   (`.ora`, Krita/GIMP-compatible; the animation rides inside), **reads and
   writes `.aseprite`**, and exports flattened PNG, animated GIF, or a
   production sprite sheet with a JSON sidecar (arrange, merge duplicate frames,
   skip empties, trim, padding, extrude, and per-tag or per-layer splits).
   Autosaves every two minutes with crash recovery, and bridges the pipeline in
   both directions.
5. **Clay** — modelling from primitives: vertex/edge/face element modes,
   extrude/bevel/subdivide/dissolve, UVs, a material palette, GLB import, and a
   diffable `.wblk` native format. Two ways out: export to the library as an
   ordinary asset (rigging, posing, sheets and every mesh export then work on it
   unchanged), or render it flat and send it to Create.
6. **Poser** — authoring reusable poses against a skeleton template, kept in a
   global pose library rather than belonging to any one asset; poses can move
   their root. Also the **clip editor**: the keyframes a character sheet
   animates — which keys, in what order, how many frames apart — with
   onion-skinned neighbours, a scrubber that plays the renderer's own
   interpolation, and your edits saved beside the shipped clips rather than over
   them.
7. **Troupe** — character sprite sheets from a 3D model, as a chain rather than
   a button: a prompt draws a reference against a drawn pose guide — **A-pose by
   default**, because the shipped humanoid rig template is itself an A-pose, with
   T-pose still on offer for the limb separation a single-view reconstruction
   prefers — you approve it, and the same asset then goes through reconstruction,
   the auto-rig and a 256-cell render without being asked again. Five animations
   (idle, walk, run, attack, jump) across eight directions, rendered large and
   reduced to the pixel size you asked for, quantised against one palette. The
   sidecar carries a tag per animation and direction, so **Edit in Inker** opens
   the whole sheet on its own timeline with the spans already set.
8. **Plotter** — a tile-map editor: grid, layer stack, tilesets and object
   layers, terrain/Wang sets, per-tile metadata, hexagonal and staggered maps,
   infinite maps, native `.wmap`, and Tiled interop in both directions
   (`.tmx`/`.tmj` import and export; unsupported Tiled features are refused
   explicitly, never partially loaded).
9. **Packwright** — a sprite-atlas packer: files, drops, Inker documents or
   library assets in; a deterministic atlas out (Grid or MaxRects, with
   trim/padding/extrude/power-of-two), as PNG plus TexturePacker JSON, and a
   `.tsx` for grid packs. Re-export of an unchanged document is byte-identical.
10. **Sirens** — a chiptune tracker: the synthesis engine, a five-column pattern
   grid, an envelope editor, sample import and sound effects, with WAV, stems
   and sfx export. Marked Experimental: a block selection can be transposed and
   cleared, but not yet copied, cut or pasted.

**The footer** carries no caption, and holds the two destinations where you are
not making something — entered rarely and left again:

11. **Review** — judging finished meshes with graded verdicts (−5..+5 plus
    tags), parameter sweeps over arbitrary setting axes, an advisory DINOv2-probe
    quality judge taught by in-app labelling, and the "What works" findings the
    verdicts add up to — which surface as hints beside the generate controls.
12. **Settings** — the app's own preferences: theme, UI scale, layout, and the
    model list, from which a missing one can be downloaded.

Two things are deliberately *not* modes. The **manual** is an overlay
(`F1`, and every pane's (?) button) because help is consulted *about* a screen,
and taking that screen away to show it answers the question by removing it.
**Profiles** are a sheet over the reference form, because a shelf of saved
settings sitting beside six creative workspaces would say that "manage my
styles" is a place you travel to.

## What comes out

Everything but the primary artifacts is derived lazily on first request and cached (`service/derive.py`):

- **Per mesh**: `model.glb` (optimised, grounded), `source.glb` (raw reconstruction), STL, OBJ (zip), FBX, `collision.glb` (convex hull), `textures.zip`, `rig.glb` once rigged, and a baked GLB per saved pose.
- **Per reference**: `icon.png` (512 transparent cutout), `sprite.png` (trimmed, pivot recorded), `pixel_{32,64,128}.png` (palette-capped or mapped to a user palette in Oklab, optional dither), and a `manifest.json` carrying sizes, trim boxes, pivots and the recipe. Tiles additionally get an estimated PBR material set.
- **Per workspace**: sprite sheets as PNG plus an engine-neutral JSON sidecar (poses down, compass directions across; animated clips are cells with a `frame` above zero), plus a **Pixelate** variant restyling the whole sheet under one seed and palette; Inker's ORA/PNG/GIF/sheet; Plotter's TMX/TMJ; Packwright's atlases.
- **Bulk**: zip named artifacts across many jobs, or mirror exports into `WARLOCK_EXPORT_DIR` (e.g. a Godot project's `assets/`).

## Requirements

- **Windows 10/11, 64-bit, and an NVIDIA GPU with CUDA.** There is no macOS or Linux build and no CPU fallback — without a CUDA device the 3D path cannot run at all.
- **16 GB VRAM** for 3D reconstruction (`vram.py`'s `TRELLIS_GIB = 16.0`). Tested on an RTX 5090 / 32 GB; a 4080/5080-class card or better is the comfortable range.
- **32 GB system RAM.** More than the GPU figure suggests it should need: Windows charges trellis's ~16 GiB device allocation against *host* commit, so admission control refuses jobs at 96% commit on a 63.5 GB machine even with 24 GB physically free. 16 GB will fight you.
- **~23 GB disk before the first asset** — 16.1 GB of TRELLIS.2 GGUF weights plus 7.0 GB for SDXL 1.0 — then roughly 35–50 MB per generated 3D job. There is no automatic age-out; pruning is manual.
- **A 1920×1080 display or larger at 100% scaling.** The window opens at 1600×950 (scaled by your DPI setting) and is clamped to the desktop, so it fits smaller panels, but below that the six workspaces get cramped.
- [uv](https://docs.astral.sh/uv/); Python ≥ 3.12, but **rigging needs 3.13** — `bpy` ships CPython 3.13 wheels only. On any other Python the rig extra installs nothing, `warlock doctor` reports rigging unavailable, and the app hides the rig controls; everything else works unchanged.

**How long a generation takes:** roughly two minutes of GPU per 3D attempt on the tested card — a reference image in seconds, then the reconstruction. Budget for more than one attempt: the approval gate exists because the first reference is often not the one you want.

## Setup

Two steps in a terminal, then the app fetches its own weights.

```powershell
# 1. Python deps. --extra studio is the app's window and renderer; add
#    --extra text2image for text-to-3D (pulls torch cu128) and --extra rig
#    for rigging/posing/sheets. Contributors running the test suite want all
#    three -- a bare `uv sync` prunes the extras and breaks ~10 test files.
uv sync --extra studio --extra text2image --extra rig

# 2. trellis.cpp CUDA server binary -> vendor/trellis/
#    https://github.com/pwilkin/trellis.cpp/releases (trellis-cuda-windows-x64.zip)
#    vendored build: v0.5.4 (2026-07-27)
```

Then start the app. **It will offer you the ~23 GB of model weights on first
run** — a first-run panel names your GPU and its VRAM, lists exactly what needs
downloading with the combined size, and refuses up front if the disk cannot
hold it. The same rows live in **Settings → Models** afterwards, where you can
add or remove individual models; a removal tells you what it would actually
free before you confirm, which matters because four of the registered recipes
share one 7 GB checkpoint.

> Some models restrict what you may do with what they generate. SDXL-Turbo is
> under a **non-commercial** research licence and Playground v2.5 has a
> monthly-active-user cap; both are labelled in the picker and listed in
> [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md). The default, SDXL 1.0, is
> OpenRAIL++-M and commercially permissive.

**This does not make the app online-capable, and the mechanism is the point.**
The Download button spawns a separate `python -m warlock.pipelines.fetch_worker`
process which sets `HF_HUB_OFFLINE=0` *in its own environment*, fetches one
repository into a staging directory beside the destination, verifies the hub's
recorded digests, moves the files in only if it succeeded, and exits. The app
process keeps `HF_HUB_OFFLINE=1` for its entire life. Free disk is checked
against the whole plan before anything is spawned, and a killed download leaves
a partial directory that `present()` treats as absent rather than as installed.

### The same downloads, from a terminal

For a headless box, a scripted setup, or if you would rather see the commands:

```powershell
# TRELLIS.2 GGUF weights (16.1 GB) -> ~/.warlock/models/trellis2-gguf/
uvx hf download ilintar/trellis2-gguf --revision a57397bd3d351599d9729fc144b3f87c3f87d65b --include "*.gguf" --exclude "q4/*" --exclude "q8/*" `
  --local-dir $HOME/.warlock/models/trellis2-gguf

# SDXL 1.0 weights (fp16 variant, 7.0 GB) -> ~/.warlock/models/sdxl-base-1.0/  (text-to-3D only)
uvx hf download stabilityai/stable-diffusion-xl-base-1.0 --revision 462165984030d82259a11f4367a4eed129e94a7b `
  --include "*.json" --include "*.txt" --include "*fp16.safetensors" --local-dir $HOME/.warlock/models/sdxl-base-1.0
```

`--include` must be **repeated per pattern**. The space-separated form is
accepted by the CLI but only the last pattern takes effect, which fetches the
safetensors and silently leaves out every `config.json` — producing a directory
that looks downloaded and fails the `model_index.json` check.

This one SDXL download powers four of the registered recipes — full CFG (the
default), Hyper-SD, LCM and Lightning are the same weights run four ways, so the
three faster ones cost only a small LoRA each. SDXL-Turbo is a separate 7 GB
checkpoint and is optional; its command is in [docs/MODELS.md](docs/MODELS.md).

These downloads are the only network use there is. The generation pipeline is
fully offline — the app process never downloads anything (`HF_HUB_OFFLINE=1` is
set at import, all model loads are `local_files_only`), and a missing weight
produces a clear error and a `doctor` warning naming the exact command rather
than a silent fetch.

**Optional models** — alternative base models (SDXL-Turbo, the Hyper-SD/LCM/Lightning recipes over the base you already have, Playground, Juggernaut, DreamShaper, FLUX.2 klein), style LoRAs (3D render, PS1, pixel art), IP-Adapter, ControlNet, BiRefNet matting, DINOv2, ViTPose — live in [docs/MODELS.md](docs/MODELS.md) with the exact commands and the rationale for each recipe.

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

`warlock sweep --image ~/.warlock/assets/<job-id>/input.png --bands auto,4,8 --seed 42` regenerates one reference at several trellis `--band` values with a fixed seed and audits each resulting mesh.

`python -m warlock.bench` is the developer measurement suite behind quality decisions: versioned suites (`core-v1`, `pixel-v1`) run under named recipes, rendered to eight views per mesh and scored on silhouette IoU and DINOv2 identity (always A-against-B, never as an absolute). Subcommands: `suites`, `recipes`, `run`, `score`, `calibrate`, `prune`, `purge`.

### Configuration

There is no *engine* config file — every path, port, timeout and mode is a `WARLOCK_*` env var, and the full table lives in [docs/manual/39-configuration.md](docs/manual/39-configuration.md). Studio's own UI preferences (theme, UI scale, pane layout, remembered form fields) are a separate thing and do persist, in `studio_settings.json` in the data directory; they are edited in the app rather than in a file. The main knobs: `WARLOCK_DATA_DIR` (where assets and the job store live), `WARLOCK_EXPORT_DIR`, `WARLOCK_T2I_ROOT`/`WARLOCK_T2I_MODEL` (image-model home and default), and `WARLOCK_VRAM_EXCLUSIVE`.

On VRAM: the trellis server subprocess starts on the first 3D job and by default stays resident alongside the image model (~16 GB + ~7 GB on a 32 GB card); both are evicted after 10 minutes idle. `WARLOCK_VRAM_EXCLUSIVE=1` restores sequential VRAM use for text jobs (trellis stopped → image model loads, generates, unloads → trellis restarts) — needed for smaller GPUs, resolution 1536, or a resident FLUX.

## Development

```powershell
uv run pytest -q            # unit tests; the renderer's skip without a GL 3.3 context
uv run pytest -m gpu -n 0 -q  # opt-in lane: real card/weights; must run serially
uv run ruff check .
```

The default run excludes the `gpu` marker (`addopts = -m "not gpu"`): those tests load real
checkpoints onto a real card, so they belong to a deliberate lane rather than to every `pytest`.
Run that lane before changing model loading, VRAM accounting or conditioning.

The app is a single process: a pygame window, one ModernGL context, and [imgui-bundle](https://github.com/pthom/imgui_bundle) panels drawn through that same context (the 3D viewport is a texture the panels show). Three threads — the frame loop, an asyncio worker for the GPU queue, and a task pool for blocking calls; jobs run one at a time. `warlock.service` is the single business-logic layer the panes and the tests both call. Model loads that would bloat the app process run in subprocesses that end (Blender, BiRefNet matting, the fetch worker), all tied to a kill-on-close job object.

Outputs land in `~/.warlock/assets/<job_id>/` (`input.png`, `model.glb`, `rig.glb`/`rig.json`, `poses/`, `sheets/`); the SQLite job store lives at `~/.warlock/assets/jobs.sqlite`. Everything the app generates — the library, benchmark runs, palettes and model weights — sits under that one home directory rather than inside the checkout; an install that predates it has its directories moved there on the next start (copy, verify, then delete), and `WARLOCK_HOME` or `WARLOCK_NO_MIGRATE` opts out. See [Data locations](docs/manual/39-configuration.md#data-locations).

Where to read more: the user manual is [docs/manual/00-index.md](docs/manual/00-index.md) (38 chapters, also embedded in the app), the hard invariants and their measured reasoning are `docs/INVARIANTS.md`, measurement write-ups are `docs/measurements/`, and `CHANGELOG.md` tracks releases.

## Licence

Warlock Studio is **GPL-3.0-or-later** — see [LICENSE](LICENSE).

GPL rather than something permissive because the Windows installer bundles
[`bpy`](https://pypi.org/project/bpy/) (Blender as a Python module, GPL-3.0) to
provide rigging. The subprocess boundary is real — only
`src/warlock/pipelines/blender_worker.py` imports `bpy`, and it never runs in
the app process — but the installer distributes both inside one executable, so
the combined work is GPL. A source checkout without `--extra rig` contains no
GPL dependency; the licence on this project is unchanged either way.

Third-party components — the vendored `trellis-server.exe` and ggml DLLs, the
NVIDIA CUDA redistributables, `gltfpack`, the bundled fonts and the vendored
BiRefNet code — carry their own terms, collected in
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

**Model weights are not part of this project.** They are downloaded by you, from
Hugging Face, and licensed by their publishers — two of them restrict commercial
use of what you generate. The app shows each model's licence in the picker and
at download; the full table is in the notices file above.
