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
    # Key into _SCHEDULERS in pipelines/text2image; None keeps whatever the
    # checkpoint's own scheduler_config.json specifies.
    scheduler: str | None = None
    # A step-distillation LoRA fused on at load, never user-facing.
    base_lora: str | None = None
    # Whether a ControlNet may be attached to this checkpoint. Explicit rather
    # than derived from guidance_scale: a ControlNet at guidance 0 on a 4-step
    # distilled base fights the structure hint instead of honouring it, and a
    # future base at CFG 1.5 must not silently become "controllable" because it
    # cleared a threshold nobody qualified it against.
    controlnet: bool = False
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
    # Which preprocessor turns the reference into a hint image. "canny" is
    # pipelines/control; "depth" is pipelines/depth, which needs torch.
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
        # Pairs with the existing lowpoly art style: chunky untextured geometry
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
