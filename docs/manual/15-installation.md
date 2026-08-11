# Installation

Warlock Studio is a local desktop app with two heavy dependencies it does not ship: a vendored
native binary and several gigabytes of model weights. Everything below is done once, by hand, and
after it the app never touches the network again.

## Requirements

- Windows, with an NVIDIA GPU. The tested machine is an RTX 5090 with 32 GB; the reconstruction
  engine alone fits in 16 GB.
- [uv](https://docs.astral.sh/uv/), which manages the Python environment.
- Roughly 16 GB of disk for the TRELLIS.2 GGUF weights, plus about 7 GB for SDXL 1.0 if you want
  text-to-3D.

Python 3.12 or newer. Rigging is the one part that wants a specific version — see
[Python dependencies](#python-dependencies) below.

## Python dependencies

The base install is the app and its renderer:

```powershell
uv sync --extra studio
```

The `dev` tooling (pytest, pytest-asyncio, ruff) is a dependency *group*, not an extra, and uv
installs it by default — no flag needed. The extras are separable on purpose, and each one buys a
different capability:

| Extra | What it adds | Skipping it costs |
| --- | --- | --- |
| `studio` | moderngl, pygame-ce, imgui-bundle | The window itself. Without it only `warlock doctor` and `warlock sweep` run. |
| `text2image` | torch cu128, torchvision, diffusers, transformers, accelerate, peft, sentencepiece, protobuf, and BiRefNet's own einops/kornia/timm | Text-to-3D. Image-to-3D from an upload still works. |
| `rig` | bpy | Rigging, posing and sprite sheets. |

`text2image`'s tail is longer than it looks because two of the things it pulls in are not declared
by anything else. BiRefNet — the learned matting model — is loaded with `trust_remote_code`, so the
modelling code that builds it is the checkpoint's own and no resolver can see its imports: `einops`,
`kornia` and `timm` are what that code reaches for, and without them the matting silently fell back
to a corner fill on a machine where `warlock doctor` could see every weight on disk. `torchvision` is
the other: `transformers` builds its fast image processors on it, and the DINOv2 embedding behind
candidate ranking needs it, so leaving it undeclared meant any `uv sync` removed it and candidate
ranking quietly degraded to composition alone.

`studio` is an extra rather than a core dependency because `warlock doctor` and `warlock sweep` have
to run on a machine with no display — the command line only imports the window on the path that
opens one.

`rig` carries one constraint worth knowing before you install it. `bpy` ships **CPython 3.13 wheels
only**, so the requirement is marked `python_version >= '3.13' and python_version < '3.14'`. On
Python 3.12 the extra installs
nothing at all: `warlock doctor` reports rigging as unavailable, the app hides the rig controls, and
everything else works unchanged. The marker is not decoration — without it, `bpy`'s own
`Requires-Python` would make the whole project unresolvable on 3.12 rather than merely leaving
rigging out.

## The trellis binary

The reconstruction engine is `trellis-server.exe`, a compiled CUDA binary from
[trellis.cpp](https://github.com/pwilkin/trellis.cpp) — not the Python TRELLIS package. Download
`trellis-cuda-windows-x64.zip` from that project's releases and unpack it into `vendor/trellis/`, so
that `vendor/trellis/trellis-server.exe` exists.

The vendored build is **v0.5.4** (2026-07-27). If you keep the binary somewhere else, point
`WARLOCK_TRELLIS_EXE` at it — see [Environment variables](16-configuration.md#environment-variables).

A missing binary is one of only two **fatal** startup checks: no reconstruction engine means no
mesh, and there is nothing to degrade to.

## Model weights

Two downloads are enough to make the app work end to end. Both are one-time.

```powershell
# TRELLIS.2 GGUF weights -> models/trellis2-gguf/
uvx hf download ilintar/trellis2-gguf --include "*.gguf" --exclude "q4/*" --exclude "q8/*" `
  --local-dir models/trellis2-gguf

# SDXL 1.0 weights (fp16 variant, ~7 GB) -> models/sdxl-base-1.0/  (text-to-3D only,
# needs `uv sync --extra studio --extra text2image` to pull torch cu128)
uvx hf download stabilityai/stable-diffusion-xl-base-1.0 `
  --include "*.json" --include "*.txt" --include "*fp16.safetensors" --local-dir models/sdxl-base-1.0
```

That second download is the default image model, and it is also three others: the Hyper-SD, LCM
and Lightning recipes are the same weights run differently, so each of them costs only a small
adapter on top. SDXL-Turbo is a separate checkpoint and is optional now — the models page has its
command.

The GGUF download also brings `birefnet.gguf`, the background-matting model. It is optional: without
it the engine falls back to a threshold cutout, which is worse on anything with a soft edge.

### Optional image models and style LoRAs

The reference image is the single biggest lever on final mesh quality — the reconstruction engine
can only be as good as the picture it is handed — so the image model and an optional style LoRA are
per-job choices in the guidance panel. Everything below is optional and independently skippable, and
`warlock doctor` lists each one with the exact command to fetch it.

Base models are one-resident-at-a-time: a 32 GB card holds the reconstruction engine plus a single
SDXL-class pipeline, not two, so switching between jobs costs a reload. Style LoRAs are the
opposite — adapters on the resident pipeline, switched for free.

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

# SDXL 1.0 + LCM (pixel art): the same base weights again, run at 8 steps with
# guidance 1.0 -- the recipe the pixel-art LoRA below was trained against. The
# LCM LoRA has to be renamed: loras/ is flat, and the upstream filename is
# generic enough that any other repo's default-named adapter would overwrite it.
uvx hf download latent-consistency/lcm-lora-sdxl `
  pytorch_lora_weights.safetensors --local-dir models/loras
Rename-Item models/loras/pytorch_lora_weights.safetensors lcm-lora-sdxl.safetensors

# Style LoRAs -> models/loras/
uvx hf download goofyai/3d_render_style_xl 3d_render_style_xl.safetensors --local-dir models/loras
uvx hf download artificialguybr/3DRedmond-V1 `
  3DRedmond-3DRenderStyle-3DRenderAF.safetensors --local-dir models/loras
uvx hf download artificialguybr/ps1redmond-ps1-game-graphics-lora-for-sdxl `
  PS1Redmond-PS1Game-Playstation1Graphics.safetensors --local-dir models/loras
# Pixel art: generates on a pixel grid rather than being downscaled into one.
uvx hf download nerijs/pixel-art-xl pixel-art-xl.safetensors --local-dir models/loras
# Pixel art for FLUX.2 klein. Renamed for the same reason the LCM LoRA above is:
# loras/ is flat and shared across architectures, and this repo ships the same
# generic filename. It is offered only on the two klein entries.
uvx hf download Limbicnation/pixel-art-lora `
  pytorch_lora_weights.safetensors --local-dir models/loras
Rename-Item models/loras/pytorch_lora_weights.safetensors pixel-art-klein.safetensors
```

The SDXL 1.0 weights serve three entries in the model list — the Hyper-SD one above, and a full-CFG
one that runs the same checkpoint at 30 steps with real classifier-free guidance, and a pixel-art
one that runs it at 8 steps under an LCM adapter. Downloading them once gets you all three.

### Optional conditioning models

Image conditioning needs its own weights. Neither is required to generate anything; each one simply
makes its control unavailable until it is present.

```powershell
# IP-Adapter Plus: condition on an image's appearance. Both halves are needed --
# the weights alone load fine and then fail at the first call.
uvx hf download h94/IP-Adapter sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors `
  --local-dir models/ip-adapter
uvx hf download h94/IP-Adapter --include "models/image_encoder/*" --local-dir models/ip-adapter

# ControlNet (Canny): lock the silhouette to an image's edges.
uvx hf download diffusers/controlnet-canny-sdxl-1.0 `
  --include "*.json" --include "*fp16.safetensors" --local-dir models/controlnet-canny-sdxl
```

See [Conditioning on an image](03-generating-references.md#conditioning-on-an-image) for what these
actually do.

FLUX is not offered. Both `dev` and `schnell` are click-through gated on Hugging Face, and 12B
parameters will not coexist with the reconstruction engine on one card. Using a local copy anyway is
possible but constrained — see
[Using a different image model](16-configuration.md#using-a-different-image-model).

## Checking the install

```powershell
uv run warlock doctor   # checks dependencies, weights and configuration
uv run warlock          # opens the desktop app
```

`doctor` prints one row per check, and the split between **fatal** and non-fatal is the whole point
of reading it:

- **Fatal** — `trellis-server.exe` and the TRELLIS GGUF weights. Without either, no mesh job can
  run at all.
- **Non-fatal** — everything else: `birefnet.gguf`, `gltfpack`, CUDA, free disk space (it wants at
  least 5 GB), the trellis port, every image model, style LoRA, IP-Adapter, ControlNet and metric
  model, and Blender. Each of these costs you one capability and nothing more, so each is reported
  individually with the command that fixes it. A single "weights" row could not tell you *which* of
  five downloads you skipped.

The same checks run when the app starts, and their result is the **health dot** at the far right of
the top bar: green when everything passed, amber when a non-fatal check failed, red for a fatal one
or a dead GPU worker. Click it for the full list, a **Copy details** button and a shortcut to the
log file.

One non-fatal check gets a red banner anyway: the trellis port. The app is perfectly usable without
ever running a mesh job, but a port already held at startup means an orphaned server from a previous
crash, and every 3D job will fail — or worse, be served by the orphan — until it is stopped.

## Offline by design

Those one-time downloads are the only network use there is. The app itself never downloads anything:
`HF_HUB_OFFLINE=1` and `HF_HUB_DISABLE_TELEMETRY=1` are set the moment the package is imported, and
every model load is `local_files_only`.

The consequence is worth stating plainly, because it is a design decision rather than an oversight:
a missing set of weights produces a clear error and a `doctor` warning naming the exact command to
fetch it, never a silent download. There is no provider API, no account, and nothing about your
prompts or your images leaves the machine.

Every optional model on this page can also be fetched from **Settings → Models**, and that does not
weaken any of the above. The button spawns a separate process which is allowed online, fetches one
repository into a staging folder beside its destination, moves the files in only on success, and
exits; the app process keeps `HF_HUB_OFFLINE=1` for its whole life and nothing on the generation
path can start a fetch. It is the same download you would run by hand, run for you, once, because
you asked.
