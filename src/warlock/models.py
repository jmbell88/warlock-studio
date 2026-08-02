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
        download=(
            "uvx hf download playgroundai/playground-v2.5-1024px-aesthetic "
            '--include "*.json" --include "*.txt" --include "*fp16.safetensors" '
            "--local-dir models/playground-v2.5"
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
    }
