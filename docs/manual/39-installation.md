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
`WARLOCK_TRELLIS_EXE` at it — see [Environment variables](40-configuration.md#environment-variables).

A missing binary is one of the **fatal** startup checks: no reconstruction engine means no mesh, and
there is nothing to degrade to. The other two are the TRELLIS GGUF weights, for the same reason, and
the VRAM budget — that last one is fatal only when the budget cannot hold a lone reconstruction, so
on a card large enough it never fires and only two rows can ever be red.

## gltfpack

`gltfpack` is the mesh optimiser every decimating triangle tier runs through. It is a second
vendored binary and it is acquired exactly the way the trellis one is — by hand, once. `vendor/` is
git-ignored in its entirety, so a fresh clone has neither binary and the manual should not be read
as saying otherwise.

Get `gltfpack.exe` from [meshoptimizer](https://github.com/zeux/meshoptimizer/releases) and put it
at `vendor/gltfpack/gltfpack.exe`, or point `WARLOCK_GLTFPACK` at a copy you keep elsewhere.

The build this project is qualified against reports **gltfpack 1.2**; the exact file measured is
2,966,528 bytes with SHA-256
`ff64f45e84aac9a1f58880e40934b3f29277413e2d0b3ed257322261ec021d2b`. A different build is very likely
fine — the checksum is here so a stale or mismatched copy can be *identified*, not so it can be
enforced.

Unlike the trellis binary, a missing `gltfpack` is not fatal and not even a warning about a broken
install: `warlock doctor` reports it, the generate form is unaffected, and every job simply ships
the raw reconstruction at full density instead of decimating it.

## Model weights

Two downloads are enough to make the app work end to end. Both are one-time.

```powershell
# TRELLIS.2 GGUF weights -> ~/.warlock/models/trellis2-gguf/
uvx hf download ilintar/trellis2-gguf --revision a57397bd3d351599d9729fc144b3f87c3f87d65b --include "*.gguf" --exclude "q4/*" --exclude "q8/*" `
  --local-dir $HOME/.warlock/models/trellis2-gguf

# SDXL 1.0 weights (fp16 variant, ~7 GB) -> ~/.warlock/models/sdxl-base-1.0/  (text-to-3D only,
# needs `uv sync --extra studio --extra text2image` to pull torch cu128)
uvx hf download stabilityai/stable-diffusion-xl-base-1.0 --revision 462165984030d82259a11f4367a4eed129e94a7b `
  --include "*.json" --include "*.txt" --include "*fp16.safetensors" --local-dir $HOME/.warlock/models/sdxl-base-1.0
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
per-job choices in the settings pane. Everything below is optional and independently skippable, and
`warlock doctor` lists each one with the exact command to fetch it.

**Only the models listed here can be selected, and that is deliberate.** Dropping a
`.safetensors` into `loras/` does *not* add a style: Warlock's picker is driven by a registry in
`src/warlock/models.py`, not by a directory listing. Every entry there declares the architecture it
was fitted to, the weight it was *measured* at, and its trigger words — none of which a bare file
carries, and each of which is wrong-by-default rather than merely missing. A LoRA trained with
`use_rslora` needs a default weight an order of magnitude smaller than an ordinary one; an adapter
fitted to another architecture raises with the checkpoint already resident in VRAM. Adding a model
is [an ordinary code change](45-extending.md), and a small one.

Base models are one-resident-at-a-time: a 32 GB card holds the reconstruction engine plus a single
SDXL-class pipeline, not two, so switching between jobs costs a reload. Style LoRAs are the
opposite — adapters on the resident pipeline, switched for free.

```powershell
# SDXL-Turbo (~7 GB): the 4-step fast option, at 512 px and guidance 0. Its own
# checkpoint rather than a recipe over the base weights, so it is a full second
# download -- worth it when iteration speed matters more than fidelity. This is
# also the entry WARLOCK_T2I_DIR redirects.
uvx hf download stabilityai/sdxl-turbo --revision 71153311d3dbb46851df1931d3ca6e939de83304 `
  --include "*.json" --include "*.txt" --include "*fp16.safetensors" `
  --exclude "sd_xl_turbo_1.0*" --local-dir $HOME/.warlock/models/sdxl-turbo

# SDXL 1.0 + Hyper-SD (~7 GB + 787 MB). Style LoRAs are trained against full
# SDXL at 20-25 steps with CFG, so they land noticeably stronger here than on
# Turbo's 4 steps at guidance 0. Hyper-SD buys the step count back.
uvx hf download stabilityai/stable-diffusion-xl-base-1.0 --revision 462165984030d82259a11f4367a4eed129e94a7b `
  --include "*.json" --include "*.txt" --include "*fp16.safetensors" --local-dir $HOME/.warlock/models/sdxl-base-1.0
uvx hf download ByteDance/Hyper-SD --revision bc08d970a87c74c71209491d64e3525845698863 Hyper-SDXL-4steps-lora.safetensors --local-dir $HOME/.warlock/models/loras

# Playground v2.5 (~7 GB): highest fidelity, ~25 steps with CFG, correspondingly slower.
uvx hf download playgroundai/playground-v2.5-1024px-aesthetic --revision 1e032f13f2fe6db2dc49947dbdbd196e753de573 `
  --include "*.json" --include "*.txt" --include "*fp16.safetensors" --local-dir $HOME/.warlock/models/playground-v2.5

# FLUX.2 klein-base 4B (~16 GB): a different architecture entirely -- one Qwen3
# text encoder at 512 tokens instead of two CLIPs at 77, and a DiT instead of a
# UNet. Honours a negative prompt, 50 steps at guidance 4. Offloaded rather than
# resident: ~16 GB stays in host memory and about 10 GB is on the card at peak.
uvx hf download black-forest-labs/FLUX.2-klein-base-4B --revision a3b4f4849157f664bdbc776fd7453c2783562f4d `
  --include "*.json" --include "*.txt" --include "*.jinja" --include "*.safetensors" `
  --exclude "flux-2-klein-base-4b.safetensors" --local-dir $HOME/.warlock/models/flux2-klein-base-4b

# FLUX.2 klein 4B distilled (~16 GB): the 4-step sibling, at guidance 1.0. It
# registers as distilled, so the negative prompt is inert on it -- pick
# klein-base when you want one. It exists so the pixel-art klein LoRA below can
# run at the recipe it was trained on.
uvx hf download black-forest-labs/FLUX.2-klein-4B --revision e7b7dc27f91deacad38e78976d1f2b499d76a294 `
  --include "*.json" --include "*.txt" --include "*.jinja" --include "*.safetensors" `
  --exclude "flux-2-klein-4b.safetensors" --local-dir $HOME/.warlock/models/flux2-klein-4b

# SDXL 1.0 + LCM (pixel art): the same base weights again, run at 8 steps with
# guidance 1.0 -- the recipe the pixel-art LoRA below was trained against. The
# LCM LoRA has to be renamed: loras/ is flat, and the upstream filename is
# generic enough that any other repo's default-named adapter would overwrite it.
uvx hf download latent-consistency/lcm-lora-sdxl --revision a18548dd4956b174ec5b0d78d340c8dae0a129cd `
  pytorch_lora_weights.safetensors --local-dir $HOME/.warlock/models/loras
Rename-Item $HOME/.warlock/models/loras/pytorch_lora_weights.safetensors lcm-lora-sdxl.safetensors

# SDXL 1.0 + Lightning (394 MB on top of the base weights): a second 4-step
# distillation, adversarial where Hyper-SD is trajectory-consistency -- so the
# two are directly comparable with everything else held fixed.
uvx hf download ByteDance/SDXL-Lightning --revision c9a24f48e1c025556787b0c58dd67a091ece2e44 `
  sdxl_lightning_4step_lora.safetensors --local-dir $HOME/.warlock/models/loras

# Juggernaut XL v9 (~6.9 GB): a photoreal SDXL finetune, DPM++ 2M Karras at 35
# steps with CFG 4.0. Materials and lighting read as photographed rather than
# illustrated, which suits a prop that will be reconstructed and then lit.
uvx hf download RunDiffusion/Juggernaut-XL-v9 --revision cf419233522daa0b9ea36c3aff98fa2cab1fb0fb `
  --include "*.json" --include "*.txt" --include "*fp16.safetensors" --local-dir $HOME/.warlock/models/juggernaut-xl-v9

# DreamShaper XL (~6.9 GB): the stylised counterpart, DEIS at 25 steps per its card.
uvx hf download Lykon/dreamshaper-xl-1-0 --revision 41e6644752a8c9aa63930e6043c4fd83c7708420 `
  --include "*.json" --include "*.txt" --include "*fp16.safetensors" --local-dir $HOME/.warlock/models/dreamshaper-xl

# Style LoRAs -> ~/.warlock/models/loras/
uvx hf download goofyai/3d_render_style_xl --revision 5ec74a57db5e244a2157173781a7b29045f88237 3d_render_style_xl.safetensors --local-dir $HOME/.warlock/models/loras
uvx hf download artificialguybr/3DRedmond-V1 --revision f4b4b980972566aea7c71af9d4e170d7fcb6c404 `
  3DRedmond-3DRenderStyle-3DRenderAF.safetensors --local-dir $HOME/.warlock/models/loras
uvx hf download artificialguybr/ps1redmond-ps1-game-graphics-lora-for-sdxl --revision 74bb3a6e2efd47ead698ff3ac2695ab63bbd2d5c `
  PS1Redmond-PS1Game-Playstation1Graphics.safetensors --local-dir $HOME/.warlock/models/loras
# Pixel art: generates on a pixel grid rather than being downscaled into one.
uvx hf download nerijs/pixel-art-xl --revision 8bf4a4d9ea283e00a51fafda8e0539f8248ea037 pixel-art-xl.safetensors --local-dir $HOME/.warlock/models/loras
# Pixel art for FLUX.2 klein. Renamed for the same reason the LCM LoRA above is:
# loras/ is flat and shared across architectures, and this repo ships the same
# generic filename. It is offered only on the two klein entries.
uvx hf download Limbicnation/pixel-art-lora --revision 0ac8e5c3400af68228811edc324721e25fc26777 `
  pytorch_lora_weights.safetensors --local-dir $HOME/.warlock/models/loras
Rename-Item $HOME/.warlock/models/loras/pytorch_lora_weights.safetensors pixel-art-klein.safetensors
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
uvx hf download h94/IP-Adapter --revision 018e402774aeeddd60609b4ecdb7e298259dc729 sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors `
  --local-dir $HOME/.warlock/models/ip-adapter
uvx hf download h94/IP-Adapter --revision 018e402774aeeddd60609b4ecdb7e298259dc729 --include "models/image_encoder/*" --local-dir $HOME/.warlock/models/ip-adapter

# ControlNet (Canny): lock the silhouette to an image's edges.
uvx hf download diffusers/controlnet-canny-sdxl-1.0 --revision eb115a19a10d14909256db740ed109532ab1483c `
  --include "*.json" --include "*fp16.safetensors" --local-dir $HOME/.warlock/models/controlnet-canny-sdxl

# ControlNet (Depth): anchors a re-texture's restyle passes to the mesh's own
# geometry. The hint is rendered by Blender from the mesh rather than estimated
# from a photo, which is why this one belongs to the re-texture stage alone and
# never appears in the reference stage's conditioning pickers.
uvx hf download diffusers/controlnet-depth-sdxl-1.0 --revision 17bb97973f29801224cd66f192c5ffacf82648b4 `
  --include "*.json" --include "*fp16.safetensors" --local-dir $HOME/.warlock/models/controlnet-depth-sdxl
```

See [Conditioning on an image](22-generating-references.md#conditioning-on-an-image) for what these
actually do.

### Optional measuring and helper models

These five never make a picture. They measure one, cut one out, rewrite the prompt that asks for
one, or place a skeleton — and each is absent-changes-nothing: without it the feature it powers
falls back to what the app did before it existed, rather than failing.

```powershell
# DINOv2 base (~0.4 GB): the identity embedding behind style-anchor similarity
# and the Review judge's probes.
uvx hf download facebook/dinov2-base --revision f9e44c814b77203eaa57a6bdbbd535f21ede1415 `
  --include "*.json" --include "*.safetensors" --local-dir $HOME/.warlock/models/dinov2-base

# PickScore v1 (~3.8 GB): a CLIP-H fine-tuned on 500k human A/B preferences,
# which scores an image *for its prompt*. It is the "would a person pick this"
# term in candidate ranking. Two commands against one repository and both are
# needed: `hf download` silently ignores --include when a positional filename
# is given too, so a single merged command fetches the weights without the
# processor and the model then cannot load.
uvx hf download yuvalkirstain/PickScore_v1 --revision a4e4367c6dfa7288a00c550414478f865b875800 model.safetensors `
  --local-dir $HOME/.warlock/models/pickscore-v1
uvx hf download yuvalkirstain/PickScore_v1 --revision a4e4367c6dfa7288a00c550414478f865b875800 `
  --include "*.json" --include "*.txt" --local-dir $HOME/.warlock/models/pickscore-v1

# Prompt expander, Fooocus GPT-2 (~0.7 GB): turns a short prompt into a fuller
# one. See Prompt enrichment in the references chapter.
uvx hf download LykosAI/GPT-Prompt-Expansion-Fooocus-v2 --revision a1fa8a394dc4614ec944ed188d8873648df85a10 `
  --include "*.json" --include "*.txt" --include "*.bin" --local-dir $HOME/.warlock/models/prompt-expander-fooocus

# ViTPose base (~0.4 GB): measures a humanoid subject's joints off the reference
# image, so a rig starts from where the limbs actually are rather than from the
# template's proportions.
uvx hf download usyd-community/vitpose-base-simple --revision a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45 `
  --include "*.json" --include "*.safetensors" --local-dir $HOME/.warlock/models/vitpose-base

# BiRefNet (~1 GB): host-side background matting for 2D exports. Without it the
# alpha comes from a corner flood fill, with visibly rougher edges around hair,
# spokes and anything thin. Weights only -- the repo's own modelling code is
# vendored in this app, so nothing is executed out of the downloaded directory.
# That vendored code imports einops, kornia, timm and torchvision, so this one
# also wants `uv sync --extra text2image`.
uvx hf download ZhengPeng7/BiRefNet --revision e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4 `
  --include "*.json" --include "*.safetensors" --local-dir $HOME/.warlock/models/birefnet
```

All five run on the **CPU**, deliberately: a measurement must not take VRAM from the models making
the asset. What each one changes when present:

| Model | Without it | With it |
| --- | --- | --- |
| DINOv2 | A style profile has no anchor similarity; the judge has no probes. | [Style profiles](36-profiles.md) and [Review](37-review.md) work fully. |
| PickScore | Candidates rank on composition and style anchor alone. | A human-preference term joins the ranking — see [Seeds and candidates](22-generating-references.md#seeds-and-candidates). |
| Prompt expander | The prompt is used as typed. | [Prompt enrichment](22-generating-references.md#prompt-enrichment) can rewrite it. |
| ViTPose | Skeletons are fitted by bounding box. | Humanoid rigs start from measured joints — see [Where the joints come from](25-rigging-and-posing.md#where-the-joints-come-from). |
| BiRefNet | A 2D export's alpha comes from a corner flood fill. | The cutout is matted properly, which shows on hair and anything thin. |

**FLUX.1 is not offered; FLUX.2 klein is.** The two `FLUX.1` checkpoints — `dev` and `schnell` —
are click-through gated on Hugging Face, and 12B parameters will not coexist with the reconstruction
engine on one card. Using a local copy anyway is possible but constrained — see
[Using a different image model](40-configuration.md#using-a-different-image-model). The FLUX.2 klein
pair above is neither gated nor 12B, which is why those two are ordinary entries in the model list.

Because klein is a different architecture, the controls fitted to SDXL do not apply to it, and the
pane says so rather than letting a submit fail: the style-LoRA picker lists only adapters fitted to
the architecture you have chosen — for klein, the pixel-art LoRA above and nothing else — and
structure control explains that it needs a full-CFG model. Changing the base model clears a
selection the new one cannot run and tells you which one it cleared, so switching to klein and back
never leaves a stale choice behind.

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

The same checks run when the app starts, and their result is the **Issues** badge in the navigation
rail's footer: amber when a non-fatal check failed, red for a fatal one or a dead GPU worker. It is
not there at all when everything passed — a permanent "OK" badge is noise. Click it for the full
list, a **Copy details** button and a shortcut to the log file. When nothing is failing, the same
list is reachable from the command palette (Ctrl+K, "Issues").

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
