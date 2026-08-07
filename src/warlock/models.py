"""The image-model registry: which base models and style LoRAs a job may pick.

Kept separate from guidance.py on purpose. That module owns *prompt fragments*
-- taxonomy that only ever ends up as text. Model identity is a different
concern: it decides what is resident in VRAM, what has to be downloaded by
hand, and what doctor.py reports on. Mixing the two would blur a boundary
guidance.py's docstring states explicitly.

Everything here is optional and independently skippable. A missing base model
fails that one job with its download command; a missing LoRA is skipped at load
time. Nothing in this file is ever downloaded at runtime -- see the offline
invariant in __init__.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_BASE_MODEL = "turbo"

# Which architecture a checkpoint is, and therefore which encode/sample path
# runs. Declared per model rather than sniffed, for the same reason
# BaseModel.controlnet is: a future checkpoint must not become "SDXL-shaped"
# because it happened to ship two text encoders and a unet/.
FAMILY_SDXL = "sdxl"
FAMILY_FLUX2_KLEIN = "flux2klein"
FAMILIES = (FAMILY_SDXL, FAMILY_FLUX2_KLEIN)

# How a pipeline is placed on the device.
#
# "resident" is .to("cuda") -- the whole pipe, which is what every SDXL-class
# checkpoint has always done. "offload" is enable_model_cpu_offload(), which
# keeps one submodule on the device at a time; the two are mutually exclusive
# (accelerate's hooks assume the modules start on the host, so a preceding
# .to("cuda") defeats the whole mechanism).
RESIDENT = "resident"
OFFLOAD = "offload"

# LoRA adapter names are per-pipeline, so the always-on step-distillation
# adapter needs a name no style LoRA key can collide with.
BASE_LORA_ADAPTER = "_base"

LORA_WEIGHT_MIN = 0.0
LORA_WEIGHT_MAX = 1.5
DEFAULT_LORA_WEIGHT = 0.9

# Conditioning strengths, bounded the same way LoRA weight is: the API's
# 400-on-out-of-range comes from one place, and the UI's sliders read their
# ends from here rather than repeating numbers.
IP_SCALE_MIN = 0.0
IP_SCALE_MAX = 1.5
DEFAULT_IP_SCALE = 0.6
CONTROL_SCALE_MIN = 0.0
CONTROL_SCALE_MAX = 2.0
DEFAULT_CONTROL_SCALE = 0.65
# How far an img2img denoise is taken. Below ~0.3 the init image survives
# essentially unchanged and above ~0.65 it stops constraining the result, which
# for a restyle is the whole point of starting from one. The effective step
# count is steps x strength, which is why a 4-step distilled base cannot do
# img2img at all: 0.45 of four steps is under two.
IMG2IMG_STRENGTH_MIN = 0.30
IMG2IMG_STRENGTH_MAX = 0.65
DEFAULT_IMG2IMG_STRENGTH = 0.45

# The style LoRA a pixel sheet restyle is fixed to in v1. Named here rather
# than in the sheet code so the registry stays the only place a model key is
# written down.
PIXEL_SHEET_LORA = "pixelxl"

# How far into the denoise the ControlNet keeps acting. Ending early lets the
# last steps add detail the hint image never had; 1.0 holds the structure to
# the final step and tends to look traced.
CONTROL_END_MIN = 0.0
CONTROL_END_MAX = 1.0
DEFAULT_CONTROL_END = 0.8


@dataclass(frozen=True, slots=True)
class BaseModel:
    """A diffusers text2image checkpoint plus the settings it must be run at.

    Sampler settings are part of the model's identity, not user preference: a
    4-step distilled model run at 25 steps with CFG produces mush, and Hyper-SD
    silently degrades unless its scheduler uses trailing timestep spacing. So
    steps/guidance/scheduler travel with the checkpoint.
    """

    # NOTE on the download strings below: --include must be repeated per
    # pattern. The space-separated form (--include "*.json" "*.txt" ...) is
    # accepted by the CLI but only the *last* pattern takes effect, which
    # fetches the safetensors and silently leaves out every config.json --
    # producing a directory that looks downloaded and fails the
    # model_index.json check. Verified against hf 0.36 on 2026-08-01.
    key: str
    label: str
    dir_name: str
    image_size: int
    steps: int
    guidance_scale: float
    variant: str | None = "fp16"
    # A name pipelines/text2image._scheduler knows (it raises on any other);
    # None keeps whatever the checkpoint's own scheduler_config.json specifies.
    scheduler: str | None = None
    # A step-distillation LoRA fused on at load, never user-facing.
    base_lora: str | None = None
    # Whether a ControlNet may be attached to this checkpoint. Explicit rather
    # than derived from guidance_scale: a ControlNet at guidance 0 on a 4-step
    # distilled base fights the structure hint instead of honouring it, and a
    # future base at CFG 1.5 must not silently become "controllable" because it
    # cleared a threshold nobody qualified it against.
    controlnet: bool = False
    # Which architecture this is, from FAMILIES. Everything the image half of
    # the app does -- chunked CLIP encoding, the pooled embeddings, style
    # LoRAs, ControlNet, IP-Adapter, img2img -- is an SDXL fact, so a
    # non-"sdxl" family takes a different sample path and loses all of it.
    family: str = FAMILY_SDXL
    # RESIDENT or OFFLOAD. A 16 GB checkpoint fully resident cannot coexist
    # with trellis-server on a 32 GB card; offloaded, its peak is roughly the
    # larger of its two big submodules and it fits.
    residency: str = RESIDENT
    # Peak device footprint under ``residency``, in GiB -- what vram.estimate
    # charges a text job for this checkpoint. Deliberately conservative:
    # refusing a job is the good outcome.
    vram_gib: float = 7.0
    # Files (relative to the model directory) whose presence means "downloaded",
    # for doctor. Empty keeps the default unet/-shaped formula, which is right
    # for every SDXL checkpoint and wrong for anything that has no unet/.
    probe: tuple[str, ...] = ()
    download: str = ""


@dataclass(frozen=True, slots=True)
class StyleLora:
    key: str
    label: str
    filename: str
    # Trained trigger words. Prepended to the composed prompt in
    # text2image.generate(), alongside PROMPT_TEMPLATE -- these are model-facing
    # scaffolding, not creative direction, so they don't belong in guidance.py.
    trigger: str = ""
    default_weight: float = DEFAULT_LORA_WEIGHT
    download: str = ""


@dataclass(frozen=True, slots=True)
class IPAdapter:
    """An appearance/identity adapter plus the CLIP vision encoder it needs.

    The encoder is a separate download from the weights and loading one
    without the other succeeds and then fails at the first call, which is not
    a failure a user can read -- hence two download lines and a doctor row
    that checks both.
    """

    key: str
    label: str
    dir_name: str
    subfolder: str
    weight_name: str
    # Where the CLIP vision encoder sits *relative to dir_name*, matching the
    # layout the download command produces.
    image_encoder_dir: str
    default_scale: float = 0.6
    download: str = ""

    @property
    def encoder_folder(self) -> str:
        """What diffusers' image_encoder_folder wants: a path *relative to the
        adapter root*, POSIX-separated. Built here so a Windows backslash can
        never reach load_ip_adapter.
        """
        return self.image_encoder_dir.replace("\\", "/")


@dataclass(frozen=True, slots=True)
class ControlNet:
    key: str
    label: str
    dir_name: str
    # Which preprocessor turns the reference into a hint image. Only "canny"
    # exists today (``pipelines/control.PREPROCESSORS``); a depth hint would
    # need a torch model and therefore a module of its own, since control.py
    # must stay torch-free.
    preprocessor: str
    variant: str | None = "fp16"
    default_scale: float = 0.65
    default_end: float = 0.8
    download: str = ""


def _table(*items):
    return {item.key: item for item in items}


BASE_MODELS: dict[str, BaseModel] = _table(
    BaseModel(
        "turbo",
        "SDXL-Turbo (fast)",
        "sdxl-turbo",
        image_size=512,
        steps=4,
        guidance_scale=0.0,
        download=(
            "uvx hf download stabilityai/sdxl-turbo "
            '--include "*.json" --include "*.txt" --include "*fp16.safetensors" '
            '--exclude "sd_xl_turbo_1.0*" --local-dir models/sdxl-turbo'
        ),
    ),
    BaseModel(
        # The backend where style LoRAs behave as trained: they are fitted
        # against full SDXL at 20-25 steps with CFG, and Turbo's 4 steps at
        # guidance 0 applies them only weakly. Hyper-SD buys back the step
        # count without changing the base weights the LoRAs were fitted to.
        "sdxl",
        "SDXL 1.0 + Hyper-SD (best LoRA response)",
        "sdxl-base-1.0",
        image_size=1024,
        steps=4,
        guidance_scale=0.0,
        scheduler="ddim_trailing",
        base_lora="Hyper-SDXL-4steps-lora.safetensors",
        download=(
            "uvx hf download stabilityai/stable-diffusion-xl-base-1.0 "
            '--include "*.json" --include "*.txt" --include "*fp16.safetensors" '
            "--local-dir models/sdxl-base-1.0\n"
            "  uvx hf download ByteDance/Hyper-SD Hyper-SDXL-4steps-lora.safetensors "
            "--local-dir models/loras"
        ),
    ),
    BaseModel(
        # Its own scheduler_config.json is already EDMDPMSolverMultistep, so no
        # override. The card recommends 50 steps; 25 is the point where extra
        # steps stop visibly changing a plain-background single object, and
        # this is a reference image for TRELLIS, not a final render.
        "playground",
        "Playground v2.5 (highest fidelity, slow)",
        "playground-v2.5",
        image_size=1024,
        steps=25,
        guidance_scale=3.0,
        controlnet=True,
        download=(
            "uvx hf download playgroundai/playground-v2.5-1024px-aesthetic "
            '--include "*.json" --include "*.txt" --include "*fp16.safetensors" '
            "--local-dir models/playground-v2.5"
        ),
    ),
    BaseModel(
        # The same weights as "sdxl", run the way the checkpoint was trained:
        # 30 steps with real classifier-free guidance and no Hyper-SD. This is
        # the only reason it exists as a separate entry -- a distilled 4-step
        # base at guidance 0 discards the negative prompt entirely
        # (text2image encodes it only when guidance_scale > 1.0) and gives a
        # ControlNet nothing to steer. No new weights: same dir_name.
        "sdxl_cfg",
        "SDXL 1.0 (full CFG, structural control)",
        "sdxl-base-1.0",
        image_size=1024,
        steps=30,
        guidance_scale=7.0,
        controlnet=True,
        download=(
            "uvx hf download stabilityai/stable-diffusion-xl-base-1.0 "
            '--include "*.json" --include "*.txt" --include "*fp16.safetensors" '
            "--local-dir models/sdxl-base-1.0"
        ),
    ),
    BaseModel(
        # The pixel-art profile's null hypothesis: the same SDXL 1.0 weights as
        # "sdxl"/"sdxl_cfg" (no new checkpoint), run the way the pixel-art-xl
        # author documents -- LCM at 8 steps, guidance 1.0. Distinct from the
        # Hyper-SD arm because stacking a 1.0-weight distillation LoRA under a
        # 1.2-weight style LoRA is unproven; both arms exist so bench/pixel-v1
        # can decide which the preset keeps.
        #
        # No VAE override on purpose. sdxl-vae-fp16-fix patches fp16's
        # overflow-to-NaN, and everything here loads bfloat16 (see
        # pipelines/text2image; variant="fp16" only names the weight *files*),
        # which has fp32's exponent range -- so that failure mode does not
        # exist here. If bench images ever show VAE decode artifacts, add a
        # BaseModel.vae field then, with a docs/measurements/ note.
        "pixel",
        "SDXL 1.0 + LCM (pixel art)",
        "sdxl-base-1.0",
        image_size=1024,
        steps=8,
        guidance_scale=1.0,
        scheduler="lcm",
        base_lora="lcm-lora-sdxl.safetensors",
        download=(
            "uvx hf download stabilityai/stable-diffusion-xl-base-1.0 "
            '--include "*.json" --include "*.txt" --include "*fp16.safetensors" '
            "--local-dir models/sdxl-base-1.0\n"
            "  uvx hf download latent-consistency/lcm-lora-sdxl "
            "pytorch_lora_weights.safetensors --local-dir models/loras\n"
            # Renamed because the upstream filename is generic: any other repo's
            # default-named LoRA downloaded into the flat loras/ directory would
            # silently overwrite it.
            "  then rename models/loras/pytorch_lora_weights.safetensors "
            "to lcm-lora-sdxl.safetensors"
        ),
    ),
    BaseModel(
        # The second distillation arm, and the reason it is worth a row: it is
        # a genuinely different method from "sdxl"'s Hyper-SD -- adversarial
        # rather than trajectory-consistency -- over the same base weights, so
        # the sweep/verdict machinery can compare the two with everything else
        # held fixed. 394 MB of new download, and no new checkpoint.
        #
        # Trailing timestep spacing is not optional, exactly as it is not for
        # Hyper-SD: on the default leading spacing a Lightning LoRA produces a
        # washed-out image and no error saying so. Euler rather than DDIM
        # because that is the pairing the model card documents.
        "lightning",
        "SDXL 1.0 + Lightning (4-step)",
        "sdxl-base-1.0",
        image_size=1024,
        steps=4,
        guidance_scale=0.0,
        scheduler="euler_trailing",
        base_lora="sdxl_lightning_4step_lora.safetensors",
        download=(
            "uvx hf download stabilityai/stable-diffusion-xl-base-1.0 "
            '--include "*.json" --include "*.txt" --include "*fp16.safetensors" '
            "--local-dir models/sdxl-base-1.0\n"
            "  uvx hf download ByteDance/SDXL-Lightning "
            "sdxl_lightning_4step_lora.safetensors --local-dir models/loras"
        ),
    ),
    BaseModel(
        # A photoreal SDXL finetune, run at its card's own recipe: DPM++ 2M
        # Karras, 30-40 steps, CFG 3-7. 35/4.0 is the middle of both ranges.
        # Full CFG, so the negative prompt is live and a ControlNet has
        # something to steer.
        "juggernaut",
        "Juggernaut XL v9 (photoreal)",
        "juggernaut-xl-v9",
        image_size=1024,
        steps=35,
        guidance_scale=4.0,
        scheduler="dpm_karras",
        controlnet=True,
        download=(
            "uvx hf download RunDiffusion/Juggernaut-XL-v9 "
            '--include "*.json" --include "*.txt" --include "*fp16.safetensors" '
            "--local-dir models/juggernaut-xl-v9"
        ),
    ),
    BaseModel(
        # The stylised counterpart to juggernaut, and the card's own snippet is
        # DEIS at 25 steps. It states no CFG, so this takes SDXL's own 7.0
        # rather than inventing one.
        "dreamshaper",
        "DreamShaper XL (stylised)",
        "dreamshaper-xl",
        image_size=1024,
        steps=25,
        guidance_scale=7.0,
        scheduler="deis",
        controlnet=True,
        download=(
            "uvx hf download Lykon/dreamshaper-xl-1-0 "
            '--include "*.json" --include "*.txt" --include "*fp16.safetensors" '
            "--local-dir models/dreamshaper-xl"
        ),
    ),
    BaseModel(
        # The first non-SDXL architecture in the registry, and the reason
        # BaseModel has a ``family`` at all. One Qwen3 text encoder at 512
        # tokens instead of two CLIPs at 77, a DiT instead of a UNet, and no
        # pooled embedding -- so the chunker, the style LoRAs, the ControlNet,
        # the IP-Adapter and img2img are all inapplicable to it.
        #
        # The *base* variant rather than the distilled FLUX.2-klein-4B, and
        # deliberately: the distilled checkpoint registers is_distilled=True,
        # and Flux2KleinPipeline.do_classifier_free_guidance is
        # ``guidance_scale > 1 and not is_distilled``, so a negative prompt is
        # impossible on it. This one carries no is_distilled in its
        # model_index.json, takes the class default of False, and therefore
        # honours the negative prompt the app fills in by default.
        #
        # variant=None because the repo ships no *.fp16.safetensors, and a
        # ``probe`` because the default doctor formula looks for a unet/ this
        # has no equivalent of. Both halves of the probe matter: the text
        # encoder is half the download, and checking only the transformer would
        # call a half-fetched model present.
        #
        # OFFLOAD, not RESIDENT. Transformer 7.75 GB + text encoder 8.04 GB +
        # VAE 0.17 GB is ~16 GB fully resident -- the same as trellis-server,
        # so .to("cuda") would put coexist out of reach on a 32 GB card
        # (16 + 16 + 1.5 headroom > 32) and make this usable only under
        # WARLOCK_VRAM_EXCLUSIVE=1. Offloaded, the peak is roughly the larger
        # submodule plus activations; 10.0 is that, rounded up.
        "flux_klein",
        "FLUX.2 klein-base 4B (full CFG)",
        "flux2-klein-base-4b",
        image_size=1024,
        steps=50,
        guidance_scale=4.0,
        variant=None,
        family=FAMILY_FLUX2_KLEIN,
        residency=OFFLOAD,
        vram_gib=10.0,
        probe=(
            "transformer/diffusion_pytorch_model.safetensors",
            "text_encoder/model-00001-of-00002.safetensors",
        ),
        download=(
            "uvx hf download black-forest-labs/FLUX.2-klein-base-4B "
            '--include "*.json" --include "*.txt" --include "*.jinja" '
            '--include "*.safetensors" '
            # The repo carries a redundant 7.75 GB single-file checkpoint
            # beside the diffusers layout; nothing here reads it.
            '--exclude "flux-2-klein-base-4b.safetensors" '
            "--local-dir models/flux2-klein-base-4b"
        ),
    ),
)

STYLE_LORAS: dict[str, StyleLora] = _table(
    StyleLora(
        "render3d",
        "3D render",
        "3d_render_style_xl.safetensors",
        trigger="3d style, 3d render",
        download=(
            "uvx hf download goofyai/3d_render_style_xl 3d_render_style_xl.safetensors "
            "--local-dir models/loras"
        ),
    ),
    StyleLora(
        "redmond3d",
        "3D render (Redmond)",
        "3DRedmond-3DRenderStyle-3DRenderAF.safetensors",
        trigger="3D Render Style, 3DRenderAF",
        download=(
            "uvx hf download artificialguybr/3DRedmond-V1 "
            "3DRedmond-3DRenderStyle-3DRenderAF.safetensors --local-dir models/loras"
        ),
    ),
    StyleLora(
        # Pairs with the PS1-era art style: chunky untextured geometry
        # is exactly what TRELLIS reconstructs most cleanly.
        "ps1",
        "PS1 / low-poly game",
        "PS1Redmond-PS1Game-Playstation1Graphics.safetensors",
        trigger="Ps1 game graphics",
        download=(
            "uvx hf download artificialguybr/ps1redmond-ps1-game-graphics-lora-for-sdxl "
            "PS1Redmond-PS1Game-Playstation1Graphics.safetensors --local-dir models/loras"
        ),
    ),
    StyleLora(
        # The one LoRA that generates pixel art *natively* rather than being
        # downscaled into it. Its default weight is the author's documented
        # recipe (1.2), not this module's DEFAULT_LORA_WEIGHT -- below it the
        # output keeps SDXL's anti-aliased gradients and no downscale recovers
        # a clean grid from them.
        #
        # The trigger carries both spellings the model card uses; they are
        # model-facing scaffolding, which is exactly why guidance.py's fragments
        # may never contain the word (tests/test_guidance.py pins that).
        "pixelxl",
        "Pixel art (pixel-art-xl)",
        "pixel-art-xl.safetensors",
        trigger="pixel, pixel art",
        default_weight=1.2,
        download=(
            "uvx hf download nerijs/pixel-art-xl pixel-art-xl.safetensors "
            "--local-dir models/loras"
        ),
    ),
)


IP_ADAPTERS: dict[str, IPAdapter] = _table(
    IPAdapter(
        # "plus" rather than the base adapter: it conditions on 16 patch
        # tokens instead of one pooled embedding, which is the difference
        # between "same kind of object" and "this object".
        "plus",
        "Appearance reference (IP-Adapter Plus)",
        "ip-adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
        image_encoder_dir="models/image_encoder",
        download=(
            "uvx hf download h94/IP-Adapter sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors "
            "--local-dir models/ip-adapter\n"
            '  uvx hf download h94/IP-Adapter --include "models/image_encoder/*" '
            "--local-dir models/ip-adapter"
        ),
    ),
)

CONTROLNETS: dict[str, ControlNet] = _table(
    ControlNet(
        "canny",
        "Edge / silhouette lock (Canny)",
        "controlnet-canny-sdxl",
        preprocessor="canny",
        download=(
            "uvx hf download diffusers/controlnet-canny-sdxl-1.0 "
            '--include "*.json" --include "*fp16.safetensors" '
            "--local-dir models/controlnet-canny-sdxl"
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class MetricModel:
    """A model used to *measure* an asset rather than make one.

    Same registry shape as the rest so doctor reports it the same way, but
    nothing in the app's generation path touches these -- only the benchmark
    does, and a missing one costs a metric, not a job.
    """

    key: str
    label: str
    dir_name: str
    download: str = ""


METRIC_MODELS: dict[str, MetricModel] = _table(
    MetricModel(
        "dinov2",
        "DINOv2 base (identity metric)",
        "dinov2-base",
        download=(
            "uvx hf download facebook/dinov2-base "
            '--include "*.json" --include "*.safetensors" '
            "--local-dir models/dinov2-base"
        ),
    ),
)


DEFAULT_POSE_MODEL = "vitpose"


@dataclass(frozen=True, slots=True)
class PoseModel:
    """A model that finds a subject's joints in a picture of it.

    Its own table for the same reason MattingModel is not a MetricModel: this
    one is read by the *generation* path -- the rig fitter asks it where the
    shoulders are -- while a metric only ever grades a finished asset. What
    they share is that every one of them is optional, none is ever downloaded
    at runtime, and a missing one costs quality rather than a job.
    """

    key: str
    label: str
    dir_name: str
    download: str = ""


POSE_MODELS: dict[str, PoseModel] = _table(
    PoseModel(
        # The plain ViTPose-base, not one of the MoE variants: those need a
        # dataset_index passed with every forward and buy accuracy on datasets
        # this project never sees. COCO-17 on a single centred subject is the
        # easiest case a pose estimator has, and base clears it.
        "vitpose",
        "ViTPose base (rig joint placement)",
        "vitpose-base",
        download=(
            "uvx hf download usyd-community/vitpose-base-simple "
            '--include "*.json" --include "*.safetensors" '
            "--local-dir models/vitpose-base"
        ),
    ),
)


DEFAULT_MATTING = "birefnet"


@dataclass(frozen=True, slots=True)
class MattingModel:
    """A model that separates a subject from its background, host-side.

    Its own table rather than another MetricModel: a metric measures a finished
    asset and its absence costs a number, while this one *produces* an asset's
    alpha and its absence costs edge quality on every 2D export. Both are
    optional and neither is ever downloaded at runtime, which is all they have
    in common.

    ``remote_code`` is stated rather than implied. The published repo ships its
    own modelling code and transformers executes it on load. It comes from the
    snapshot the user downloaded once -- nothing is fetched, and the offline
    invariant holds -- but it is third-party Python running in this process,
    and doctor says so out loud rather than leaving it in a docstring.
    """

    key: str
    label: str
    dir_name: str
    remote_code: bool = False
    download: str = ""


MATTING_MODELS: dict[str, MattingModel] = _table(
    MattingModel(
        # BiRefNet and not something smaller, because trellis-server already
        # uses BiRefNet internally for bg_removal="birefnet" -- so a 2D export
        # and the 3D input derived from the same reference agree about where
        # the subject ends, which two different matting models would not.
        "birefnet",
        "BiRefNet",
        "birefnet",
        remote_code=True,
        download=(
            "uvx hf download ZhengPeng7/BiRefNet "
            '--include "*.json" --include "*.py" --include "*.safetensors" '
            "--local-dir models/birefnet\n"
            # The weights are not the whole download. The repo's own modelling
            # code builds its backbone through packages this project does not
            # depend on, so a user who runs only the line above gets a
            # directory doctor can see and a model that cannot import. Phrased
            # as "may" because which of them is needed is a property of that
            # repo's code, not something Warlock can assert from here.
            "  you may also need: uv pip install timm torchvision "
            "-- BiRefNet's modelling code imports them and Warlock does not ship them"
        ),
    ),
)


def controlnet_bases() -> list[str]:
    """Base models a ControlNet may be attached to -- the UI hides the whole
    Structure group when the chosen base is not one of these, rather than
    offering a control that cannot do anything."""
    return [m.key for m in BASE_MODELS.values() if m.controlnet]


def cfg_bases() -> list[str]:
    """Base models where a negative prompt actually does something.

    text2image encodes the negative branch only when ``guidance_scale > 1.0``
    -- there is no unconditional branch to steer at CFG 0 -- so on a distilled
    4-step base the field is inert. Derived from the number rather than
    declared per model, unlike ``controlnet`` above: "does classifier-free
    guidance run" *is* the guidance scale, whereas "is a ControlNet qualified
    against this checkpoint" is a judgement no threshold can make.
    """
    return [m.key for m in BASE_MODELS.values() if m.guidance_scale > 1.0]


def lora_bases() -> list[str]:
    """Base models a style LoRA may be applied to.

    Derived from ``family`` rather than declared, unlike ``controlnet``: "does
    an SDXL LoRA fit this pipe" *is* the architecture -- the adapter's tensors
    name UNet modules -- whereas "is a ControlNet qualified against this
    checkpoint" is a judgement no property can make. Loading one onto a Flux
    transformer is a load error, not a weak result, which is why this refuses
    rather than degrading the way a missing LoRA file does.
    """
    return [m.key for m in BASE_MODELS.values() if m.family == FAMILY_SDXL]


def tile_bases() -> list[str]:
    """Base models that can produce a seamless tile.

    Its own name rather than a second reader of ``lora_bases()``, which today
    returns the same list: seamlessness is circular padding over ``Conv2d``,
    and a LoRA is a set of UNet tensors. They agree only because both are SDXL
    facts right now, and a shared list is how two questions quietly become one
    wrong answer.
    """
    return [m.key for m in BASE_MODELS.values() if m.family == FAMILY_SDXL]


def catalog() -> dict[str, Any]:
    """The two tables in guidance.catalog()'s field shape, for the same selects."""
    return {
        "base_model": [
            {"key": m.key, "label": m.label} for m in BASE_MODELS.values()
        ],
        "style_lora": [
            {"key": lora.key, "label": lora.label, "default_weight": lora.default_weight}
            for lora in STYLE_LORAS.values()
        ],
        "ip_adapter": [
            {"key": a.key, "label": a.label, "default_scale": a.default_scale}
            for a in IP_ADAPTERS.values()
        ],
        "control": [
            {
                "key": c.key,
                "label": c.label,
                "preprocessor": c.preprocessor,
                "default_scale": c.default_scale,
                "default_end": c.default_end,
            }
            for c in CONTROLNETS.values()
        ],
    }
