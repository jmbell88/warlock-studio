"""Text -> reference image via a local diffusion model (diffusers). Lazy, heavy import.

Requires the 'text2image' extra: uv sync --extra text2image

Which checkpoint runs, at what size/steps/CFG, is a models.BaseModel spec
rather than module constants -- see models.py for why those settings travel
with the checkpoint. Weights load from a local diffusers directory under
config.t2i_model_root with local_files_only=True; the app never downloads. See
the README for the one-time `hf download` commands.

Style LoRAs are loaded as named adapters at load() time and switched per job
with set_adapters(), which costs nothing -- so a job can change style without
paying a pipeline reload. Changing the *base* model does require a reload, and
the caller (queue.Worker) handles that by unloading first.
"""

from __future__ import annotations

import gc
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

from .. import models

log = logging.getLogger(__name__)

# Bias generations toward images TRELLIS handles well: one object, clean silhouette.
PROMPT_TEMPLATE = (
    "{prompt}, single object centered on a plain light gray background, "
    "3/4 perspective view, studio lighting, game asset concept art, "
    "full object in frame, no cropping, no text, no watermark"
)


def _scheduler(name: str, current):
    """Build a replacement scheduler from the pipeline's existing config.

    Only step-distilled bases need one: Hyper-SD is trained against trailing
    timestep spacing and produces washed-out images on the default leading
    spacing, with no error to tell you that is what happened. Checkpoints whose
    own scheduler_config.json is already correct (Playground v2.5 ships
    EDMDPMSolverMultistep) pass scheduler=None and keep it.
    """
    if name == "ddim_trailing":
        from diffusers import DDIMScheduler

        return DDIMScheduler.from_config(current.config, timestep_spacing="trailing")
    raise ValueError(f"unknown scheduler {name!r}")


class JobCancelled(Exception):
    """Raised from inside a diffusers step callback to abort mid-sample."""


class Text2Image:
    def __init__(
        self, spec: models.BaseModel, model_root: Path, model_dir: Path | None = None
    ) -> None:
        self.spec = spec
        self._model_root = model_root
        # model_dir exists for config's WARLOCK_T2I_DIR escape hatch, which
        # predates the registry; normally the spec's dir_name resolves it.
        self._model_dir = model_dir or (model_root / spec.dir_name)
        self._lora_dir = model_root / "loras"
        # Style LoRA keys that actually loaded, so generate() can tell "not
        # downloaded" from "not selected" without touching the disk again.
        self._adapters: set[str] = set()
        self._base_adapter: str | None = None
        self._pipe = None
        self.last_used: float = 0.0

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    def load(self, on_state: Callable[[str], None] | None = None) -> None:
        if self._pipe is not None:
            return
        # Checked before the (slow) torch import: instant, actionable failure,
        # and unit-testable without the text2image extra installed.
        if not (self._model_dir / "model_index.json").exists():
            raise RuntimeError(
                f"{self.spec.label} weights not found at {self._model_dir}. "
                f"Download once with:\n  {self.spec.download}"
            )
        import torch
        from diffusers import AutoPipelineForText2Image

        if on_state is not None:
            on_state("load")
        log.info("loading %s", self._model_dir)
        self._pipe = AutoPipelineForText2Image.from_pretrained(
            str(self._model_dir),
            torch_dtype=torch.bfloat16,
            variant=self.spec.variant,
            local_files_only=True,
        )
        if self.spec.scheduler is not None:
            self._pipe.scheduler = _scheduler(self.spec.scheduler, self._pipe.scheduler)
        # Fully resident: SDXL bf16 is ~7 GB, which fits the 32 GB card even
        # alongside trellis-server. (cpu_offload was for 30+ GB Flux; a local
        # Flux user should set WARLOCK_VRAM_EXCLUSIVE=1 instead.)
        self._pipe.to("cuda")
        self._load_loras()

    def _load_loras(self) -> None:
        """Attach the base step-distillation LoRA and every style LoRA on disk.

        A missing file is skipped, not fatal: the LoRAs are separate optional
        downloads and a user who fetched one of the three must still be able to
        generate. The base LoRA is the exception -- without it a Hyper-SD base
        is a 4-step run of undistilled SDXL, i.e. noise -- so that one raises.
        """
        assert self._pipe is not None
        if self.spec.base_lora is not None:
            path = self._lora_dir / self.spec.base_lora
            if not path.exists():
                raise RuntimeError(
                    f"{self.spec.label} requires {self.spec.base_lora}, missing at "
                    f"{path}. Download once with:\n  {self.spec.download}"
                )
            self._pipe.load_lora_weights(
                str(path.parent),
                weight_name=path.name,
                adapter_name=models.BASE_LORA_ADAPTER,
            )
            self._base_adapter = models.BASE_LORA_ADAPTER
        for lora in models.STYLE_LORAS.values():
            path = self._lora_dir / lora.filename
            if not path.exists():
                log.info("style LoRA %s not downloaded (%s); skipping", lora.key, path)
                continue
            self._pipe.load_lora_weights(
                str(path.parent), weight_name=path.name, adapter_name=lora.key
            )
            self._adapters.add(lora.key)
        log.info(
            "loaded LoRAs: base=%s styles=%s",
            self._base_adapter,
            sorted(self._adapters) or "none",
        )

    def _apply_adapters(self, lora: str | None, weight: float) -> None:
        assert self._pipe is not None
        names: list[str] = []
        weights: list[float] = []
        if self._base_adapter is not None:
            # Always on and always at full strength: this is step distillation,
            # not style, and scaling it down just costs sampling quality.
            names.append(self._base_adapter)
            weights.append(1.0)
        if lora is not None:
            if lora not in self._adapters:
                # Not fatal: it validated against the registry, it just is not
                # on disk. Losing the style beats failing a job over it.
                log.warning("style LoRA %s not downloaded; generating without it", lora)
            else:
                names.append(lora)
                weights.append(weight)
        if names:
            self._pipe.set_adapters(names, weights)
        else:
            self._pipe.disable_lora()

    def unload(self) -> None:
        """Drop the pipeline and give its VRAM back.

        The before/after numbers are logged rather than assumed: in exclusive
        mode trellis-server restarts immediately after this returns, and if
        anything still holds a reference to the pipe (or to a tensor it
        produced) empty_cache() frees nothing and the restart OOMs. That
        failure only reproduces under load, so the log line is the only way to
        tell it apart from an unrelated OOM after the fact. The same applies to
        a base-model switch, which unloads one 7 GB pipe to make room for the
        next -- a leak there means two resident pipes, not one.
        """
        if self._pipe is None:
            return
        import torch

        gib = 1024**3
        before = torch.cuda.memory_allocated() / gib if torch.cuda.is_available() else 0.0
        self._pipe = None
        self._adapters = set()
        self._base_adapter = None
        gc.collect()
        torch.cuda.empty_cache()
        after = torch.cuda.memory_allocated() / gib if torch.cuda.is_available() else 0.0
        log.info(
            "unloaded %s: %.2f -> %.2f GiB allocated", self._model_dir, before, after
        )

    def generate(
        self,
        prompt: str,
        output_path: Path,
        *,
        seed: int = 42,
        lora: str | None = None,
        lora_weight: float = models.DEFAULT_LORA_WEIGHT,
        negative_prompt: str | None = None,
        on_state: Callable[[str], None] | None = None,
        on_step: Callable[[int, int], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> Path:
        """Generate a reference image and save it to ``output_path``.

        ``negative_prompt`` only bites when the checkpoint runs with CFG: a
        4-step distilled base at guidance_scale 0 discards it, which is why the
        UI notes it applies to the CFG bases (playground) rather than silently
        doing nothing.
        """
        import torch

        self.load(on_state)
        assert self._pipe is not None
        # load()/download() have no interruption point of their own; check
        # once here so a cancel requested during either isn't silently lost.
        if cancel_event is not None and cancel_event.is_set():
            raise JobCancelled
        self._apply_adapters(lora, lora_weight)
        if on_state is not None:
            on_state("sample")

        steps = self.spec.steps

        def step_cb(_pipe, i, _t, kwargs):
            if cancel_event is not None and cancel_event.is_set():
                raise JobCancelled
            if on_step is not None:
                on_step(i + 1, steps)
            return kwargs  # diffusers requires the kwargs dict back, not None

        # The LoRA's trigger words sit here rather than in guidance.py: they are
        # what the adapter was fitted on, so they belong with the rest of the
        # model-facing scaffolding.
        style = models.STYLE_LORAS.get(lora or "")
        text = PROMPT_TEMPLATE.format(prompt=prompt)
        if style is not None and style.trigger and lora in self._adapters:
            text = f"{style.trigger}, {text}"

        image = self._pipe(
            text,
            negative_prompt=negative_prompt or None,
            num_inference_steps=steps,
            guidance_scale=self.spec.guidance_scale,
            width=self.spec.image_size,
            height=self.spec.image_size,
            generator=torch.Generator("cuda").manual_seed(seed),
            callback_on_step_end=step_cb,
        ).images[0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        self.last_used = time.monotonic()
        return output_path
