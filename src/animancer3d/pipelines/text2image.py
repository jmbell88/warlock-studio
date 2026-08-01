"""Text -> reference image via a distilled diffusion model (diffusers). Lazy, heavy import.

Requires the 'text2image' extra: uv sync --extra text2image

Default model is SDXL-Turbo (ungated). Set ANIMANCER3D_T2I_MODEL to e.g.
black-forest-labs/FLUX.1-schnell if you are authenticated with HuggingFace.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger(__name__)

STEPS = 4  # turbo/schnell models are distilled for 1-4 steps

# Bias generations toward images TRELLIS handles well: one object, clean silhouette.
PROMPT_TEMPLATE = (
    "{prompt}, single object centered on a plain light gray background, "
    "3/4 perspective view, studio lighting, game asset concept art, "
    "full object in frame, no cropping, no text, no watermark"
)


class JobCancelled(Exception):
    """Raised from inside a diffusers step callback to abort mid-sample."""


class Text2Image:
    def __init__(self, model_id: str, image_size: int = 1024) -> None:
        self._model_id = model_id
        self._image_size = image_size
        self._pipe = None

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    def cached(self) -> bool:
        """True if the weights are already in the HuggingFace cache.

        The first run downloads ~7 GB with no output of its own, which otherwise
        looks like a hang. Intercepting huggingface_hub's tqdm is fragile, so we
        just check the cache directory and report an indeterminate phase.
        """
        home = os.environ.get("HF_HOME")
        hub = Path(home) / "hub" if home else Path.home() / ".cache" / "huggingface" / "hub"
        return (hub / f"models--{self._model_id.replace('/', '--')}").exists()

    def load(self, on_state: Callable[[str], None] | None = None) -> None:
        if self._pipe is not None:
            return
        import torch
        from diffusers import AutoPipelineForText2Image

        if on_state is not None:
            on_state("load" if self.cached() else "download")
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

    def generate(
        self,
        prompt: str,
        output_path: Path,
        *,
        seed: int = 42,
        on_state: Callable[[str], None] | None = None,
        on_step: Callable[[int, int], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Path:
        import torch

        self.load(on_state)
        assert self._pipe is not None
        # load()/download() have no interruption point of their own; check
        # once here so a cancel requested during either isn't silently lost.
        if cancel_event is not None and cancel_event.is_set():
            raise JobCancelled
        if on_state is not None:
            on_state("sample")

        def step_cb(_pipe, i, _t, kwargs):
            if cancel_event is not None and cancel_event.is_set():
                raise JobCancelled
            if on_step is not None:
                on_step(i + 1, STEPS)
            return kwargs  # diffusers requires the kwargs dict back, not None

        image = self._pipe(
            PROMPT_TEMPLATE.format(prompt=prompt),
            num_inference_steps=STEPS,
            guidance_scale=0.0,
            width=self._image_size,
            height=self._image_size,
            generator=torch.Generator("cuda").manual_seed(seed),
            callback_on_step_end=step_cb,
        ).images[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        return output_path
