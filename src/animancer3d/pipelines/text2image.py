"""Text -> reference image via a distilled diffusion model (diffusers). Lazy, heavy import.

Requires the 'text2image' extra: uv sync --extra text2image

Default model is SDXL-Turbo (ungated). Set ANIMANCER3D_T2I_MODEL to e.g.
black-forest-labs/FLUX.1-schnell if you are authenticated with HuggingFace.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# Bias generations toward images TRELLIS handles well: one object, clean silhouette.
PROMPT_TEMPLATE = (
    "{prompt}, single object centered on a plain light gray background, "
    "3/4 perspective view, studio lighting, game asset concept art, "
    "full object in frame, no cropping, no text, no watermark"
)


class Text2Image:
    def __init__(self, model_id: str, image_size: int = 1024) -> None:
        self._model_id = model_id
        self._image_size = image_size
        self._pipe = None

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    def load(self) -> None:
        if self._pipe is not None:
            return
        import torch
        from diffusers import AutoPipelineForText2Image

        log.info("loading %s", self._model_id)
        self._pipe = AutoPipelineForText2Image.from_pretrained(
            self._model_id, torch_dtype=torch.bfloat16, variant="fp16"
        )
        # Large models (Flux: 12B DiT + T5-XXL) can peak past 32 GB fully resident;
        # offload sequences components through VRAM and costs little at 1-4 steps.
        self._pipe.enable_model_cpu_offload()

    def unload(self) -> None:
        if self._pipe is None:
            return
        import torch

        log.info("unloading %s", self._model_id)
        self._pipe = None
        gc.collect()
        torch.cuda.empty_cache()

    def generate(self, prompt: str, output_path: Path, *, seed: int = 42) -> Path:
        import torch

        self.load()
        assert self._pipe is not None
        image = self._pipe(
            PROMPT_TEMPLATE.format(prompt=prompt),
            num_inference_steps=4,  # turbo/schnell models are distilled for 1-4 steps
            guidance_scale=0.0,
            width=self._image_size,
            height=self._image_size,
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return output_path
