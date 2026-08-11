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

import contextlib
import gc
import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .. import models
from .prompt import (
    PROMPT_TEMPLATE,
    SHEET_TEMPLATE,
    TILE_TEMPLATE,
    chunk,
    pad_pair,
    view_clause,
)

log = logging.getLogger(__name__)

# Flux2KleinPipeline's own tokenizer_max_length, restated rather than read off
# the pipe: it is the number ``_sample_flux2`` must pass to both the negative
# encode and the call, and the two disagreeing would silently truncate one side
# of the guidance pair.
FLUX2_MAX_SEQUENCE = 512


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
    if name == "euler_trailing":
        # SDXL-Lightning's pairing, and trailing spacing is required for the
        # same reason ddim_trailing is: an adversarially distilled 4-step LoRA
        # sampled on leading spacing washes out, silently.
        from diffusers import EulerDiscreteScheduler

        return EulerDiscreteScheduler.from_config(
            current.config, timestep_spacing="trailing"
        )
    if name == "lcm":
        # The LCM LoRA is a consistency distillation: it is only sampled
        # correctly by LCMScheduler, and on the checkpoint's own sampler it
        # produces flat grey rather than an error.
        from diffusers import LCMScheduler

        return LCMScheduler.from_config(current.config)
    if name == "dpm_karras":
        # Juggernaut's card: DPM++ 2M with Karras sigmas. Neither half is the
        # from_config default, so both are named.
        from diffusers import DPMSolverMultistepScheduler

        return DPMSolverMultistepScheduler.from_config(
            current.config, use_karras_sigmas=True, algorithm_type="dpmsolver++"
        )
    if name == "deis":
        # DreamShaper XL's card ships its own snippet using this one.
        from diffusers import DEISMultistepScheduler

        return DEISMultistepScheduler.from_config(current.config)
    raise ValueError(f"unknown scheduler {name!r}")


def _noop() -> None:
    """The unconditioned path's teardown."""


@contextlib.contextmanager
def circular_padding(*modules: Any):
    """Make every Conv2d in ``modules`` pad circularly, then put it back.

    This is the whole seamlessness mechanism, and it is three lines because
    torch already does the work: ``_ConvNd`` computes
    ``_reversed_padding_repeated_twice`` in ``__init__`` regardless of mode and
    ``_conv_forward`` branches on ``padding_mode != 'zeros'`` at call time, so
    flipping the attribute is enough -- no rebuild, no reload, no second
    checkpoint. With it, the model's receptive field wraps, and SDXL produces
    an image whose left edge continues into its right.

    Reverted in a finally, and to each module's *own* previous mode rather than
    to a blanket "zeros". The pipe is resident and shared with ordinary jobs:
    a patch that leaked would make every later reference tile, which is
    invisible until someone looks at the edges of an image nobody looks at the
    edges of. A ``None`` module is skipped -- a pipeline component can
    legitimately be absent, and this is the wrong place to find out.

    Each conv is recorded at most once, keyed by identity. Two of the passed
    modules can legitimately reach the same child (a pipeline that shares an
    encoder between components), and recording it twice would capture
    "circular" the second time round and then restore that -- the exact leak
    the finally exists to prevent, arrived at from the other direction. Raw
    ``id()`` values are sound as keys only because ``previous`` holds a strong
    reference to every conv whose id is in ``seen``, so nothing recorded can be
    collected and have its id recycled for the block's duration: make
    ``previous`` weak, or clear it before the restore, and the de-dup silently
    starts skipping live modules.
    """
    import torch

    previous: list[tuple[Any, str]] = []
    seen: set[int] = set()
    try:
        for module in modules:
            if module is None:
                continue
            for child in module.modules():
                if isinstance(child, torch.nn.Conv2d) and id(child) not in seen:
                    seen.add(id(child))
                    previous.append((child, child.padding_mode))
                    child.padding_mode = "circular"
        yield
    finally:
        for child, mode in previous:
            child.padding_mode = mode


class JobCancelled(Exception):
    """Raised from inside a diffusers step callback to abort mid-sample."""


def _encode_long_prompt(pipe, chunks: list[str]):
    """Encode ``chunks`` through both SDXL text encoders and concatenate on
    the sequence axis, replicating per-chunk what
    StableDiffusionXLPipeline.encode_prompt does for a single <=77-token
    prompt: padding="max_length", max_length=77, truncation=True,
    hidden_states[-2], concatenated across the two encoders on the feature
    axis. The UNet's cross-attention has no sequence-length limit, so N
    encoded 77-token chunks concatenated on the sequence axis is a valid,
    longer conditioning sequence -- this is what A1111, ComfyUI and compel
    do to work around the same CLIP limit.

    Pooled is a single vector, not a sequence, so it cannot be concatenated
    across chunks: it comes from text_encoder_2's output on the first chunk
    only, matching encode_prompt's own "only ALWAYS interested in the pooled
    output of the final text encoder."
    """
    import torch

    device = pipe._execution_device
    per_chunk_embeds = []
    pooled = None
    for i, chunk_text in enumerate(chunks):
        chunk_embeds = []
        for tokenizer, text_encoder in (
            (pipe.tokenizer, pipe.text_encoder),
            (pipe.tokenizer_2, pipe.text_encoder_2),
        ):
            inputs = tokenizer(
                chunk_text,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                output = text_encoder(
                    inputs.input_ids.to(device), output_hidden_states=True
                )
            if i == 0 and text_encoder is pipe.text_encoder_2:
                pooled = output[0]
            chunk_embeds.append(output.hidden_states[-2])
        per_chunk_embeds.append(torch.cat(chunk_embeds, dim=-1))
    embeds = torch.cat(per_chunk_embeds, dim=1)
    assert pooled is not None
    dtype = pipe.text_encoder_2.dtype
    return embeds.to(dtype=dtype, device=device), pooled.to(dtype=dtype, device=device)


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
        self.last_prompt: str = ""
        self.last_recipe: dict[str, Any] = {}

    @property
    def _has_adapters(self) -> bool:
        """Whether anything was ever loaded -- i.e. whether there is PEFT state
        for ``disable_lora`` to disable.

        The family early return this replaces answered the question by proxy,
        and got one case wrong even for SDXL: ``turbo`` has no base LoRA, so on
        a host with no loras/ directory the empty branch already called
        disable_lora() on a pipe with no adapters. The correct predicate was
        never "is this SDXL", it was "is there anything to disable".
        """
        return self._base_adapter is not None or bool(self._adapters)

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    @property
    def model_dir(self) -> Path:
        """Where this base model's weights live -- same resolution Text2Image
        itself uses, exposed so a caller (the prompt-preview endpoint) can
        load matching tokenizers without reimplementing WARLOCK_T2I_ROOT."""
        return self._model_dir

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
        # Configured on a *local* and published as the last statement. self._pipe
        # used to be assigned straight out of from_pretrained, so `loaded` went
        # true before the scheduler swap, the device placement and the LoRA
        # attachment had run -- and a failure in any of those (a missing base
        # LoRA raises by design, an OOM in .to("cuda"), a cancel) left a
        # half-configured pipe that every later load() returned early on, with
        # the weights resident and nothing able to reach them.
        pipe = AutoPipelineForText2Image.from_pretrained(
            str(self._model_dir),
            torch_dtype=torch.bfloat16,
            variant=self.spec.variant,
            local_files_only=True,
        )
        try:
            if self.spec.scheduler is not None:
                pipe.scheduler = _scheduler(self.spec.scheduler, pipe.scheduler)
            if self.spec.residency == models.OFFLOAD:
                # One submodule on the device at a time, so a ~16 GB checkpoint
                # peaks near the larger of its two big modules and still coexists
                # with trellis-server. Mutually exclusive with the branch below:
                # accelerate's hooks assume the modules start on the host, and a
                # preceding .to("cuda") defeats the whole mechanism.
                pipe.enable_model_cpu_offload()
            else:
                # Fully resident: SDXL bf16 is ~7 GB, which fits the 32 GB card
                # even alongside trellis-server.
                pipe.to("cuda")
            self._load_loras(pipe)
        except BaseException:
            # BaseException, not Exception: a cancel landing mid-load is
            # precisely the case that most needs the weights dropped, and
            # KeyboardInterrupt/CancelledError are not Exceptions.
            #
            # `del pipe` before _reclaim(), never inside it: this frame's local
            # is a live reference, so a collect that ran with it still bound
            # would free nothing at all.
            del pipe
            self._reclaim()
            raise
        self._pipe = pipe

    def _load_loras(self, pipe) -> None:
        """Attach the base step-distillation LoRA and every style LoRA on disk.

        A missing file is skipped, not fatal: the LoRAs are separate optional
        downloads and a user who fetched one of the three must still be able to
        generate. The base LoRA is the exception -- without it a Hyper-SD base
        is a 4-step run of undistilled SDXL, i.e. noise -- so that one raises.

        Takes ``pipe`` explicitly rather than reading ``self._pipe``: that is
        what lets ``load`` configure everything on a local and publish it only
        once this has returned, so the raise above cannot leave a resident,
        half-configured pipe behind.
        """
        # ``spec.base_lora`` needs no family guard of its own: it is a declared
        # field, None on every non-SDXL entry, so the ``is not None`` below
        # already is the guard.
        if self.spec.base_lora is not None:
            path = self._lora_dir / self.spec.base_lora
            if not path.exists():
                raise RuntimeError(
                    f"{self.spec.label} requires {self.spec.base_lora}, missing at "
                    f"{path}. Download once with:\n  {self.spec.download}"
                )
            pipe.load_lora_weights(
                str(path.parent),
                weight_name=path.name,
                adapter_name=models.BASE_LORA_ADAPTER,
                local_files_only=True,
            )
            self._base_adapter = models.BASE_LORA_ADAPTER
        for lora in models.STYLE_LORAS.values():
            if not models.lora_fits(self.spec, lora):
                # Deliberately *not* the missing-file tolerance below: the file
                # is on disk and names another architecture's modules, so
                # loading it would raise with the checkpoint already resident.
                continue
            path = self._lora_dir / lora.filename
            if not path.exists():
                log.info("style LoRA %s not downloaded (%s); skipping", lora.key, path)
                continue
            pipe.load_lora_weights(
                str(path.parent),
                weight_name=path.name,
                adapter_name=lora.key,
                local_files_only=True,
            )
            self._adapters.add(lora.key)
        log.info(
            "loaded LoRAs: base=%s styles=%s",
            self._base_adapter,
            sorted(self._adapters) or "none",
        )

    def _apply_adapters(self, pipe, lora: str | None, weight: float) -> None:
        """Set the active adapters on ``pipe``.

        Takes the pipe rather than using self._pipe because a ControlNet run
        goes through a *different* pipeline object built by from_pipe() over
        the same components. PEFT adapter state is attached to the UNet, which
        the two share, but set_adapters/disable_lora are pipeline methods --
        calling them on the wrong object is the kind of thing that silently
        generates without the style LoRA.
        """
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
            # enable_lora() first, and it is not belt-and-braces: disable_lora()
            # sets ``_disable_adapters`` on every PEFT layer, and set_adapters()
            # writes the *scaling* without ever clearing that flag. So one job
            # generated with no style LoRA silently switched every later job in
            # the same process to no style LoRA too -- the pipe stays resident
            # across jobs, so the state outlives the job that set it, and the
            # only recovery was the idle unload. It reads as working because
            # the trigger words are still prepended, so the output does change
            # when a style is picked; it just is not the adapter doing it.
            #
            # Measured on FLUX.2 klein (docs/measurements/2026-08-10-pixel-art
            # -klein.md): three weights after a no-LoRA run came back
            # byte-identical to each other, and none of them matched the same
            # weight run first. Nothing is family-specific about it -- this is
            # a plain PEFT state machine, so SDXL was affected identically.
            pipe.enable_lora()
            pipe.set_adapters(names, weights)
        elif self._has_adapters:
            pipe.disable_lora()

    def _conditioned(self, cond) -> tuple[Any, dict[str, Any], Callable[[], None]]:
        """Attach a Conditioning to the resident pipe for one call.

        Returns ``(pipe to call, extra call kwargs, teardown)``.

        Everything here loads per-call and is dropped by teardown. A resident
        ControlNet (~2.5 GB) plus the CLIP-ViT-H image encoder (~1.2 GB) pushes
        the coexist budget past what a 32 GB card holds alongside
        trellis-server, and a ~3 s reload is far cheaper than the ~90 s trellis
        restart that running out of room would cost. The unconditioned path
        never reaches this function at all -- that is the bit-identity
        contract.
        """
        from PIL import Image

        assert self._pipe is not None
        if self.spec.family != models.FAMILY_SDXL:
            # Belt and braces behind guidance.normalize and queue._conditioning:
            # every pipeline class built below is a StableDiffusionXL* one, so
            # there is nothing here that could attach to another architecture.
            raise RuntimeError(
                f"{self.spec.label} cannot take conditioning; "
                f"it is not an SDXL-family checkpoint"
            )
        target = self._pipe
        extra: dict[str, Any] = {}
        teardown_control = False
        teardown_ip = False

        # Defined before anything is loaded, not after. The ControlNet (~2.5 GB)
        # and the CLIP-ViT-H encoder (~1.2 GB) are attached in the block below,
        # and a failure part-way through -- a corrupt checkpoint, an OOM on the
        # second load -- used to escape before the teardown closure existed, so
        # whatever had already been attached stayed attached for process life.
        # The flags are set as each step succeeds, so this undoes exactly what
        # was actually done.
        def teardown() -> None:
            import torch

            if teardown_ip:
                # The restore has to be exact: from_pipe shares the UNet, so
                # the next unconditioned job's bit-identity rests on this
                # putting the attention processors back as they were.
                with contextlib.suppress(Exception):
                    target.unload_ip_adapter()
            if teardown_control:
                target.controlnet = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        try:
            if cond.uses_control:
                spec = models.CONTROLNETS[cond.control]
                root = self._model_root / spec.dir_name
                # Checked before the torch work, the same shape as load()'s
                # check: an instant, actionable failure rather than a stack
                # trace out of from_pretrained.
                if not (root / "config.json").exists():
                    raise RuntimeError(
                        f"{spec.label} weights not found at {root}. "
                        f"Download once with:\n  {spec.download}"
                    )
                import torch
                from diffusers import (
                    ControlNetModel,
                    StableDiffusionXLControlNetPipeline,
                )

                # bf16 to match the UNet. The published checkpoint is fp16, so
                # the residuals are marginally coarser -- converting the whole
                # pipe to fp16 instead would change the unconditioned path's
                # bytes, which is not on the table.
                controlnet = ControlNetModel.from_pretrained(
                    str(root),
                    torch_dtype=torch.bfloat16,
                    variant=spec.variant,
                    local_files_only=True,
                ).to("cuda")
                # Component reuse: no second UNet, VAE or text encoder is built.
                #
                # torch_dtype is not optional here, and its absence is not a
                # rounding difference. from_pipe defaults to float32 and applies
                # it with an in-place .to() on the components it was handed --
                # which are the *resident* pipe's, shared by identity. Omitting
                # it silently upcasts the live UNet/VAE/text encoders from bf16
                # to fp32: every later unconditioned job then produces different
                # bytes (breaking the bit-identity rule), and the resident pipe
                # doubles from ~7 GB to ~14 GB, which does not fit beside
                # trellis. Measured 2026-08-03: build the pipe without this and
                # never call it, and the next plain generate() already differs.
                if cond.uses_init:
                    # Structure hint *and* a starting picture: a different
                    # pipeline class again, and the kwargs swap places -- see
                    # the routing note below.
                    from diffusers import (
                        StableDiffusionXLControlNetImg2ImgPipeline,
                    )

                    target = StableDiffusionXLControlNetImg2ImgPipeline.from_pipe(
                        self._pipe, controlnet=controlnet, torch_dtype=torch.bfloat16
                    )
                else:
                    target = StableDiffusionXLControlNetPipeline.from_pipe(
                        self._pipe, controlnet=controlnet, torch_dtype=torch.bfloat16
                    )
                # Set the moment the ControlNet is attached, so a failure in the
                # IP-Adapter block below still frees these 2.5 GB.
                teardown_control = True
                if self._pipe.unet.dtype != torch.bfloat16:
                    # The guard for the above, in case a diffusers upgrade
                    # changes how from_pipe applies dtype. Loud, because the
                    # symptom otherwise is "images changed for no reason".
                    log.error(
                        "from_pipe changed the resident pipe's dtype to %s; "
                        "unconditioned output is no longer reproducible",
                        self._pipe.unet.dtype,
                    )
                if target.scheduler is not self._pipe.scheduler:
                    # A silently reset scheduler is exactly the failure
                    # _scheduler's docstring warns about on the Hyper-SD path.
                    log.warning(
                        "from_pipe did not carry the scheduler across; restoring"
                    )
                    target.scheduler = self._pipe.scheduler
                with Image.open(cond.control_image) as im:
                    # The routing that is easy to get exactly backwards. On a
                    # ControlNet-only pipeline the hint *is* ``image``; add an
                    # init picture and ``image`` becomes the thing being
                    # denoised, with the hint demoted to ``control_image``.
                    # Diffusers accepts either shape without complaint and
                    # produces a traced copy of the wrong picture.
                    extra["control_image" if cond.uses_init else "image"] = im.convert(
                        "RGB"
                    )
                extra["controlnet_conditioning_scale"] = float(cond.control_scale)
                extra["control_guidance_end"] = float(cond.control_end)

            if cond.uses_init:
                if not cond.uses_control:
                    # No ControlNet in play, so the plain img2img class. Pure
                    # component reuse like the branch above: no second UNet,
                    # VAE or text encoder, and therefore no teardown flag --
                    # there is nothing attached to detach.
                    import torch
                    from diffusers import StableDiffusionXLImg2ImgPipeline

                    target = StableDiffusionXLImg2ImgPipeline.from_pipe(
                        self._pipe, torch_dtype=torch.bfloat16
                    )
                with Image.open(cond.init_image) as im:
                    extra["image"] = im.convert("RGB")
                extra["strength"] = float(cond.strength)

            if cond.uses_ip:
                spec = models.IP_ADAPTERS[cond.ip_adapter]
                root = self._model_root / spec.dir_name
                weights = root / spec.subfolder / spec.weight_name
                if not weights.exists():
                    raise RuntimeError(
                        f"{spec.label} weights not found at {weights}. "
                        f"Download once with:\n  {spec.download}"
                    )
                # Onto the target: diffusers explicitly supports loading an
                # IP-Adapter onto a ControlNet pipeline, and the adapter's
                # attention processors have to live on the pipe being called.
                target.load_ip_adapter(
                    str(root),
                    subfolder=spec.subfolder,
                    weight_name=spec.weight_name,
                    image_encoder_folder=spec.encoder_folder,
                    local_files_only=True,
                )
                teardown_ip = True
                target.set_ip_adapter_scale(float(cond.ip_scale))
                with Image.open(cond.ip_image) as im:
                    extra["ip_adapter_image"] = im.convert("RGB")
        except BaseException:
            teardown()
            raise

        return target, extra, teardown

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
        self._reclaim()
        after = torch.cuda.memory_allocated() / gib if torch.cuda.is_available() else 0.0
        log.info(
            "unloaded %s: %.2f -> %.2f GiB allocated", self._model_dir, before, after
        )

    def _reclaim(self) -> None:
        """Reset the adapter bookkeeping and give the allocator's pool back.

        Factored out of ``unload`` so the failure path in ``load`` releases by
        the *same* implementation rather than a second, drifting copy of it.

        **It deliberately takes no pipe.** The obvious shape -- ``_release(pipe)``
        with a ``del pipe`` inside -- frees nothing: a parameter binding is a
        live reference for exactly as long as this frame runs, and the caller's
        own local is alive too, so the ``gc.collect()`` below reaches a pipe
        that two frames still hold. Every caller therefore drops its reference
        *first* and then calls this. (Measured, not reasoned: with the
        argument-taking version, the GPU tests' pipes stayed on the card and the
        next thirty queue tests were refused by dispatch-time VRAM admission.)
        """
        import torch

        self._adapters = set()
        self._base_adapter = None
        gc.collect()
        # Guarded: this is reachable on a CPU-only or driver-lost host --
        # notably from shutdown() -- where a bare empty_cache() raises and
        # turns a cleanup path into a crash.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def trim(self) -> None:
        """Return cached-but-unused memory to the driver. Keeps the pipe loaded.

        The middle ground unload() and "do nothing" leave out. In coexist mode
        the pipeline is meant to stay resident between jobs, so nothing ever
        ran after a job -- and torch's caching allocator never shrinks on its
        own, so its reserved pool grows to the high-water mark of the largest
        job and stays there. Under WDDM the driver charges those device blocks
        against *system* commit, which is the limit the 2026-08-03 crash hit.

        Cheap enough for a per-job finally: a gc pass plus a driver call, with
        no weights touched, so the next job still gets its warm-pipe latency.
        """
        if self._pipe is None:
            return
        import torch

        gib = 1024**3
        if not torch.cuda.is_available():
            gc.collect()
            return
        before = torch.cuda.memory_reserved() / gib
        gc.collect()
        torch.cuda.empty_cache()
        after = torch.cuda.memory_reserved() / gib
        log.info("trimmed cache: %.2f -> %.2f GiB reserved", before, after)

    def generate(
        self,
        prompt: str,
        output_path: Path,
        *,
        seed: int = 42,
        lora: str | None = None,
        lora_weight: float = models.DEFAULT_LORA_WEIGHT,
        negative_prompt: str | None = None,
        conditioning: Any | None = None,
        on_state: Callable[[str], None] | None = None,
        on_step: Callable[[int, int], None] | None = None,
        cancel_event: threading.Event | None = None,
        tile: bool = False,
        sheet: bool = False,
        framing: str = "",
    ) -> Path:
        """Generate a reference image and save it to ``output_path``.

        ``negative_prompt`` only bites when the checkpoint runs with CFG: a
        4-step distilled base at guidance_scale 0 discards it, which is why the
        UI notes it applies to the CFG bases (playground, sdxl_cfg) rather than
        silently doing nothing.

        ``conditioning`` is a pipelines.conditioning.Conditioning or None. None
        -- and a Conditioning with neither half in play -- takes a path that is
        byte-for-byte what this method did before conditioning existed.

        ``tile`` switches every Conv2d in the UNet, the VAE and an attached
        ControlNet to circular padding for the duration of this call, which
        makes the output natively seamless. The ControlNet is included because
        its residuals are added into the UNet at every block, so a zero-padded
        hint branch would contribute a seam to a sample whose every other
        convolution wraps. It is a property of one job, never of the pipe: the
        same resident pipeline serves ordinary references, so the patch is
        applied here and reverted before this method returns.
        """
        self.load(on_state)
        assert self._pipe is not None
        # load()/download() have no interruption point of their own; check
        # once here so a cancel requested during either isn't silently lost.
        if cancel_event is not None and cancel_event.is_set():
            raise JobCancelled

        if conditioning:
            if on_state is not None:
                on_state("condition")
            target, extra, teardown = self._conditioned(conditioning)
        else:
            target, extra, teardown = self._pipe, {}, _noop
        # The try starts here, not at the pipeline call. Everything between the
        # conditioning attach and that call -- adapter application, two long-
        # prompt encodes -- can raise, and each of those failures used to skip
        # teardown() entirely and strand the ControlNet and image encoder in
        # VRAM for the life of the process.
        stack = contextlib.ExitStack()
        try:
            if tile and self.spec.family != models.FAMILY_SDXL:
                # Refused rather than degraded. Circular padding is a property
                # of Conv2d, and a DiT has none -- so patching what a Flux pipe
                # does have (its VAE) would produce an image whose latent
                # never wrapped and whose decode did, which is a tile that
                # looks seamless in a thumbnail and seams in a material.
                # service.jobs.create_job refuses this at the door; this is the
                # other half.
                raise RuntimeError(
                    f"{self.spec.label} cannot generate a seamless tile; "
                    f"it is not an SDXL-family checkpoint"
                )
            if tile:
                # The VAE decoder as well as the UNet: a seamless latent
                # decoded through zero-padded convolutions grows a visible
                # border, which is the failure that makes people reach for an
                # inpainting pass they do not need.
                #
                # And the ControlNet, when one is attached. It is a separate
                # module tree with its own Conv2d stack, and its residuals are
                # added into the UNet at every block -- so a hint branch left
                # zero-padded contributes a seam of its own to a sample whose
                # every other convolution wraps. The de-dup in circular_padding
                # is what makes passing it alongside the UNet safe: from_pipe
                # shares components by identity, so the same conv can be
                # reachable through both.
                stack.enter_context(
                    circular_padding(
                        self._pipe.unet,
                        getattr(self._pipe, "vae", None),
                        getattr(target, "controlnet", None),
                    )
                )
            # After from_pipe, never before: the adapters have to be set on the
            # pipeline object that is actually called.
            self._apply_adapters(target, lora, lora_weight)
            if on_state is not None:
                on_state("sample")

            steps = self.spec.steps

            def step_cb(_pipe, i, _t, kwargs):
                if cancel_event is not None and cancel_event.is_set():
                    raise JobCancelled
                if on_step is not None:
                    on_step(i + 1, steps)
                return kwargs  # diffusers requires the kwargs dict back, not None

            # The LoRA's trigger words sit here rather than in guidance.py: they
            # are what the adapter was fitted on, so they belong with the rest
            # of the model-facing scaffolding.
            style = models.STYLE_LORAS.get(lora or "")
            # Only the framing template belongs here; the guidance-field subset
            # a tile wants is applied by the caller, which composes ``prompt``,
            # exactly as it is for an object today.
            # Template only -- ``sheet`` deliberately does not imply the
            # circular padding ``tile`` does. A contact sheet must not wrap:
            # its left and right edges are different directions of the same
            # subject, and making them continuous would bleed one cell into
            # another.
            template = (
                SHEET_TEMPLATE if sheet else (TILE_TEMPLATE if tile else PROMPT_TEMPLATE)
            )
            # ``view`` fills PROMPT_TEMPLATE's framing slot and is inert in the
            # other two, which carry framings of their own. A framing this
            # build does not carry composes the default rather than raising:
            # the row is already written by the time we are here.
            text = template.format(prompt=prompt, view=view_clause(framing))
            if style is not None and style.trigger and lora in self._adapters:
                text = f"{style.trigger}, {text}"
            self.last_prompt = text

            sample = (
                self._sample_flux2
                if self.spec.family == models.FAMILY_FLUX2_KLEIN
                else self._sample_sdxl
            )
            image, chunks = sample(target, extra, text, negative_prompt, seed, step_cb)
        finally:
            stack.close()
            teardown()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Staged and renamed. The reroll loop calls this with the *served*
        # input.png as its target, once per attempt, so the rejected candidates
        # of a retrying job are written straight over a name the library grid
        # and every 2D derivation already read -- and a PNG encode of a 1024²
        # image is long enough to be caught half-written. Renaming in makes
        # each attempt land whole or not at all.
        # The staging name keeps the real suffix (``.a.tmp.png``, not
        # ``.a.png.tmp``) so Pillow still infers the format from it. Passing the
        # format explicitly would have been the same thing for every caller
        # today -- all four hand this a ``.png`` -- and would have made the
        # encode depend on an argument rather than on the destination.
        tmp = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
        try:
            image.save(tmp)
            os.replace(tmp, output_path)
        finally:
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)
        self.last_used = time.monotonic()
        self.last_recipe = self._recipe(seed, text, negative_prompt, lora, lora_weight,
                                        conditioning, chunks, tile)
        return output_path

    def _sample_sdxl(self, target, extra, text, negative_prompt, seed, step_cb):
        """The SDXL sample: chunked CLIP encoding, then the pipeline call.

        Lifted out of generate() without a single expression moving. The
        unconditioned path's bit-identity is the reason ``_conditioned`` is
        written the way it is, and a refactor that "tidied" any line of this
        while splitting it is exactly how that guarantee would be lost.
        """
        import torch

        assert self._pipe is not None
        tokenizers = [self._pipe.tokenizer, self._pipe.tokenizer_2]
        positive_chunks = chunk(text, tokenizers)
        # Only playground (guidance_scale > 1) runs classifier-free
        # guidance; turbo and sdxl+Hyper-SD run at guidance_scale=0.0, where
        # diffusers ignores the negative prompt outright
        # (force_zeros_for_empty_prompt), so skipping the extra encode on
        # the default path costs nothing.
        negative_chunks: list[str] | None = None
        if self.spec.guidance_scale > 1.0:
            negative_chunks = chunk(negative_prompt or "", tokenizers)
            positive_chunks, negative_chunks = pad_pair(
                positive_chunks, negative_chunks
            )

        prompt_embeds, pooled_prompt_embeds = _encode_long_prompt(
            self._pipe, positive_chunks
        )
        negative_embeds = negative_pooled = None
        if negative_chunks is not None:
            negative_embeds, negative_pooled = _encode_long_prompt(
                self._pipe, negative_chunks
            )

        # extra is splatted rather than passed as explicit None kwargs:
        # diffusers branches on `is not None`, and "probably identical"
        # is not what the bit-identity rule asks for.
        image = target(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            negative_pooled_prompt_embeds=negative_pooled,
            num_inference_steps=self.spec.steps,
            guidance_scale=self.spec.guidance_scale,
            width=self.spec.image_size,
            height=self.spec.image_size,
            generator=torch.Generator("cuda").manual_seed(seed),
            callback_on_step_end=step_cb,
            **extra,
        ).images[0]
        return image, len(positive_chunks)

    def _sample_flux2(self, target, extra, text, negative_prompt, seed, step_cb):
        """The FLUX.2 klein sample. Four things differ, all forced by the
        pipeline's real signature rather than chosen:

        *No chunking.* One Qwen3 text encoder at ``tokenizer_max_length`` 512,
        so the whole prompt goes in as a string. The 77-token limit ``chunk``
        exists for is a CLIP fact and does not apply here -- hence a recorded
        ``prompt_chunks`` of 1.

        *No pooled embedding.* ``__call__`` accepts neither
        ``pooled_prompt_embeds`` nor ``negative_pooled_prompt_embeds``; passing
        either is a TypeError.

        *The negative prompt arrives as embeddings.* There is no
        ``negative_prompt`` string parameter -- with CFG on, the pipeline
        hardcodes ``""`` and encodes that. But it passes whatever
        ``negative_prompt_embeds`` it was given straight through to
        ``encode_prompt``, which returns non-None embeds verbatim, so encoding
        the text here is how the field is honoured at all. Only under CFG:
        below guidance 1.0 the pipeline never looks at them.

        *``extra`` is asserted empty.* ``_conditioned`` refuses this family
        outright, so a non-empty ``extra`` means something upstream built
        SDXL kwargs for a Flux call.
        """
        import torch

        assert not extra, f"conditioning kwargs on a {self.spec.family} sample: {sorted(extra)}"
        negative_embeds = None
        if self.spec.guidance_scale > 1.0:
            negative_embeds = target.encode_prompt(
                prompt=negative_prompt or "",
                num_images_per_prompt=1,
                max_sequence_length=FLUX2_MAX_SEQUENCE,
            )[0]
        image = target(
            prompt=text,
            negative_prompt_embeds=negative_embeds,
            num_inference_steps=self.spec.steps,
            guidance_scale=self.spec.guidance_scale,
            width=self.spec.image_size,
            height=self.spec.image_size,
            max_sequence_length=FLUX2_MAX_SEQUENCE,
            generator=torch.Generator("cuda").manual_seed(seed),
            callback_on_step_end=step_cb,
        ).images[0]
        return image, 1

    def _recipe(
        self, seed, text, negative_prompt, lora, lora_weight, conditioning, chunks,
        tile,
    ) -> dict[str, Any]:
        """Everything that decided this image, assembled the same way (and at
        the same point) last_prompt is -- so a job can record what produced it
        without the caller reaching into the spec."""
        from .. import provenance

        out: dict[str, Any] = {
            "base_model": self.spec.key,
            "family": self.spec.family,
            "residency": self.spec.residency,
            "steps": self.spec.steps,
            "guidance_scale": self.spec.guidance_scale,
            "image_size": self.spec.image_size,
            "variant": self.spec.variant,
            "scheduler": type(self._pipe.scheduler).__name__ if self._pipe else None,
            "seed": seed,
            "prompt": text,
            "negative_prompt": negative_prompt or "",
            "prompt_chunks": chunks,
            "tile": bool(tile),
            "models": provenance.model_fingerprints({"base_model": self._model_dir}),
            "versions": provenance.versions(),
        }
        if lora and lora in self._adapters:
            # Applied, not merely requested -- the predicate the trigger
            # prepend already uses. A style that never loaded (not downloaded,
            # or fitted to another family) did not shape this image, and
            # recording it anyway is how a row comes to claim a style that
            # never ran.
            out["style_lora"] = lora
            out["lora_weight"] = lora_weight
        if conditioning:
            out["conditioning"] = conditioning.as_dict()
        return out
