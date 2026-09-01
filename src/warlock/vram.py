"""How large the card actually is, what a job will cost, and which mode fits.

Everything the app believed about VRAM was prose. ``queue.py`` opens by naming
a 32 GB card, ``text2image.py`` names ~7 GB for SDXL and ~2.5/1.2 GB for the
conditioning stack, and the coexist default loads all of it next to
trellis-server's ~16 GB -- on whatever card is present, having never asked how
large it is. On Windows (WDDM) overcommitting VRAM does not raise: the driver
spills into shared system memory, which becomes host *commit*, which is
precisely the wall :mod:`~warlock.memlog` exists to document.

This module is the measurement those numbers were missing, plus the cost model
that turns them into a decision. It is pure in the way ``memlog`` is pure:
returns ``None`` when a reading is unavailable, never raises, and imports
nothing from ``service``, ``queue`` or ``studio`` -- so the whole gate is
testable on a machine with no GPU by way of ``WARLOCK_VRAM_TOTAL``.

One honest caveat, carried here rather than in a commit message: on WDDM
``cudaMemGetInfo``'s *free* figure is the driver's virtualized view and is less
trustworthy than it is on Linux -- it can report room that only exists as
shared system memory. ``total`` is exact. So admission control leans on
total-minus-headroom, and treats free as a secondary, dispatch-time check.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

_GIB = 1024**3


@dataclass(frozen=True, slots=True)
class DeviceMemory:
    """Device-wide figures for CUDA device 0, in GiB."""

    total_gib: float
    free_gib: float
    # Read during the same CUDA query as the figures. Empty for test doubles
    # and overrides where no physical device supplied the total.
    name: str = ""


# --- what things cost --------------------------------------------------------
#
# None of these numbers is invented. Each was already asserted in a comment
# somewhere; collecting them here is what makes them testable.

TRELLIS_GIB = 16.0
"""trellis-server.exe resident, at res 1024 (queue.py's module docstring)."""

SDXL_GIB = 7.0
"""An SDXL-class pipe fully .to("cuda") in bf16 (text2image.py:160).

Also the fallback for a ``base_model`` the registry no longer carries: params
outlive the registry, and ``queue._generate`` already applies that tolerance.
"""

CONTROLNET_GIB = 2.5
"""A ControlNet attached for one call (text2image._conditioned)."""

# An SDXL LoRA training step at 1024 px, batch 1, fp16 UNet with gradient
# checkpointing and fp32 rank-16 adapter weights: ~14 GiB measured on the
# reference configuration, declared with the same headroom every other figure
# here carries. Always exclusive -- the worker stops trellis and evicts the
# image pipe before the trainer spawns -- so this is the whole card's budget
# for the job, whatever the coexist policy says.
LORA_TRAIN_GIB = 18.0

IP_ENCODER_GIB = 1.2
"""The CLIP-ViT-H image encoder an IP-Adapter needs (same place)."""

TRELLIS_RES_MULT: dict[int, float] = {512: 0.85, 1024: 1.0, 1536: 1.5}
"""Reconstruction resolution scales the trellis footprint.

1024 is the baseline the 16 GB figure was measured at; 1536 is the case
``config.trellis_band``'s neighbours already name as an OOM situation.
"""

HEADROOM_GIB = 1.5
"""Reserved for the desktop compositor and the app's own GL context.

A card whose every byte is handed to the models still has to draw a window.
"""

COEXIST_GIB = TRELLIS_GIB + SDXL_GIB
"""What the default (coexist) mode asks a card to hold at once."""


def _resolution(params: dict[str, Any] | None) -> int:
    raw = (params or {}).get("resolution")
    try:
        res = int(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 1024
    return res if res in TRELLIS_RES_MULT else 1024


def _trellis_cost(params: dict[str, Any] | None) -> float:
    return TRELLIS_GIB * TRELLIS_RES_MULT[_resolution(params)]


# A step-distillation LoRA's device footprint, in GiB. Hyper-SDXL-4steps is
# 787 MB on disk and the weights are what land on the card, so 0.8 rounded up.
# The three distilled recipes are within a few tens of MB of each other, which
# is why this is one number and not a per-entry field.
BASE_LORA_GIB = 0.8


def _adapter_cost(spec: Any, params: dict[str, Any] | None) -> float:
    """What the adapters attached alongside this checkpoint cost, in GiB.

    ``BaseModel.vram_gib`` is the checkpoint alone, and its own comment says
    under-pricing is the direction not to err in -- but the required
    step-distillation LoRA is loaded unconditionally inside ``load()`` and was
    charged nowhere, so a Hyper-SD/LCM/Lightning recipe was under-priced by
    ~0.8 GiB every time (MDL-17).

    A *style* adapter is now attached only when a job selects one
    (``Text2Image._ensure_adapter``, MDL-07), so at most one is on the device
    and only when ``params`` names it -- which is why this reads the selection
    rather than summing every fitting adapter in the registry, as the eager
    load would have required.

    A style's size comes from its own ``Fetch`` records rather than a constant:
    a LoRA's device footprint is its weights, and ``size_gib`` is the field that
    already records how big those are. ``StyleLora`` carries no size of its own,
    which is why this sums the fetches instead of reading one attribute.
    """
    from . import models

    total = BASE_LORA_GIB if getattr(spec, "base_lora", None) else 0.0
    style = models.STYLE_LORAS.get(str((params or {}).get("style_lora") or ""))
    if style is not None and models.lora_fits(spec, style):
        total += sum(one.size_gib for one in style.fetch)
    return total


def _image_model_cost(params: dict[str, Any] | None) -> float:
    """What this job's chosen checkpoint costs, in GiB, adapters included.

    Not every image model is 7 GB any more: a spec carries its own
    ``vram_gib``, measured under its own ``residency``. Importing ``models``
    here is consistent with this module's purity rule -- the ban is on
    ``service``/``queue``/``studio``, and ``models`` is stdlib-only.
    """
    from . import models

    spec = models.BASE_MODELS.get(str((params or {}).get("base_model") or ""))
    if spec is None:
        return SDXL_GIB
    return spec.vram_gib + _adapter_cost(spec, params)


def offloaded_base(params: dict[str, Any] | None) -> bool:
    """Whether this job's checkpoint runs under ``models.OFFLOAD``.

    An offloaded pipe is the opposite shape from a resident one:
    ``enable_model_cpu_offload()`` keeps the whole checkpoint in *host* RAM for
    the life of the pipe and streams one submodule to the device at a time. So
    its ``vram_gib`` is honest about VRAM and says nothing about commit -- and
    under WDDM trellis-server's ~16 GiB device allocation is charged against
    the same commit limit as those host weights. Running the two side by side
    is the 2026-08-03 Resource-Exhaustion crash approached from the host side.

    So an offloaded base is a *mandatory* sequential handoff, whatever
    ``WARLOCK_VRAM_EXCLUSIVE`` says, and ``estimate`` has to price it that way
    or the gate over-charges the one job that is careful about VRAM.
    ``queue._needs_handoff`` is the enforcing half; this is the accounting one.
    """
    from . import models

    spec = models.BASE_MODELS.get(str((params or {}).get("base_model") or ""))
    return spec is not None and spec.residency == models.OFFLOAD


def estimate(
    kind: str,
    stage: str,
    params: dict[str, Any] | None = None,
    *,
    exclusive: bool = False,
) -> float:
    """Projected peak VRAM, in GiB, for one job.

    The two modes differ in exactly the way ``Worker._generate`` does: under
    ``exclusive`` the stages hand off (trellis is stopped before SDXL loads and
    SDXL is unloaded before trellis restarts), so the peak is the *larger* of
    the two stages; under coexist they are resident together and it is the
    *sum*. That is the whole reason auto-selecting the mode can rescue a
    smaller card rather than only refusing it.
    """
    return estimate_parts(kind, stage, params, exclusive=exclusive)[0]


def estimate_parts(
    kind: str,
    stage: str,
    params: dict[str, Any] | None = None,
    *,
    exclusive: bool = False,
) -> tuple[float, float]:
    """``estimate``, plus the checkpoint term it charged.

    Returns ``(total, image_model_gib)``. The second number is what a *resident*
    pipe has already been debited from free VRAM for, and therefore what
    ``queue._check_resources`` must credit back so the same weights are not
    charged twice. It is returned from here rather than recomputed there on
    purpose: the caller used to decide by kind (``kind == "text"``), and three
    kinds whose estimate does carry a checkpoint term -- ``pixel_sheet``,
    ``sprite_synthesis``, ``retexture`` -- were missing from that list, so on
    the flagship 32 GiB coexist card the natural flow (generate a reference,
    which leaves a ~7.5 GiB pipe warm *by design*, then start a sprite sheet)
    computed need 26.7 against free 23.5 and refused a job that would have
    **reused** the very pipe being counted against it. Waiting out the 600 s
    idle eviction "fixed" it, which presents as a phantom VRAM leak.

    Deriving it from the same expression that charges it means a new kind cannot
    be added to one half and forgotten in the other: whatever
    ``_image_model_cost`` is added to the total is what comes back here.
    """
    params = params or {}
    image = 0.0
    # An offloaded checkpoint hands off whether or not the flag is set, because
    # its cost is in host commit rather than in VRAM and nothing about the
    # coexist budget describes it (see ``offloaded_base``). Applied here, at the
    # top, so every kind below inherits it -- for ``image``/``rig`` the image
    # term is zero and the two forms are arithmetically identical anyway.
    exclusive = exclusive or offloaded_base(params)
    if kind == "lora_train":
        # The trainer runs alone: see LORA_TRAIN_GIB. No checkpoint term comes
        # back, because nothing resident survives into the run.
        return LORA_TRAIN_GIB, 0.0
    if kind == "retexture":
        # Six img2img passes over one mesh's renders, one at a time, through
        # the same resident pipe -- so the peak is one pass, not six, and the
        # view count is deliberately *not* a term. The checkpoint is priced
        # from the registry rather than assumed to be 7 GB, because a re-texture
        # is exactly the job somebody points at an offloaded FLUX.2 spec.
        #
        # The two Blender halves are out-of-process and CPU-side (`op_views`
        # renders in EEVEE at 512px, `op_project` bakes emission at one sample),
        # so they cost this budget nothing -- the same reason a rig job
        # estimates zero below.
        image = _image_model_cost(params)
        pass_gib = image
        if params.get("control"):
            pass_gib += CONTROLNET_GIB
        # Never trellis: the mesh was reconstructed long before, and nothing
        # here goes near it. Under coexist a warm trellis is still holding its
        # memory, which is what the reference stage below accounts for too.
        return (pass_gib if exclusive else pass_gib + TRELLIS_GIB), image
    if kind == "pixel_sheet":
        # An img2img restyle: the same resident pipe as a text job, plus a
        # ControlNet, and never trellis -- the mesh was reconstructed long
        # before. Under coexist a warm trellis is still holding its memory,
        # exactly as the reference stage below accounts for.
        #
        # Priced from the registry like every other checkpoint-loading kind:
        # `queue._pixel_sheet` reads params["base_model"], so charging a flat
        # SDXL_GIB was correct only for as long as the one caller kept writing
        # "sdxl_cfg" into it.
        image = _image_model_cost(params)
        pixel = image + CONTROLNET_GIB
        return (pixel if exclusive else pixel + TRELLIS_GIB), image
    if kind == "sprite_synthesis":
        # Two txt2img passes, one after the other through the same resident
        # pipe, each carrying both adapters at once -- so the peak is one pass
        # with a ControlNet *and* an IP-Adapter resident, not two of anything.
        # Unconditional rather than read from params: unlike a text job, this
        # kind always conditions on both (the pose guide and the reference are
        # the entire idea), so a params-gated term could only ever be wrong.
        #
        # The matte is a CPU flood fill in this process and the reduction is
        # Pillow, so neither costs this budget anything.
        image = _image_model_cost(params)
        sprite = image + CONTROLNET_GIB + IP_ENCODER_GIB
        return (sprite if exclusive else sprite + TRELLIS_GIB), image
    if kind == "tile_sheet":
        # One txt2img pass through the resident pipe -- or N of them, one after
        # the other, on the seamless path. Either way the peak is *one* pass:
        # the grid mode slices sixty-four tiles out of a single frame, and the
        # materials mode runs its N generations sequentially through one pipe
        # with ``num_images_per_prompt`` pinned at 1. So neither the grid nor
        # the material count is a term here.
        #
        # **The ControlNet is params-gated, not unconditional.** It was
        # unconditional while the grid guide was the only mechanism this kind
        # had, and that stopped being true when the seamless modes landed: they
        # impose nothing and attach no hint, so a job drawn through
        # ``text2image``'s circular-padding path would be charged 2.5 GiB for a
        # module it never loads -- refusing it on a card that fits it. The door
        # writes ``control`` only in grid mode, which is what makes the gate
        # sound; it is the same gate ``retexture`` and ``text`` already use.
        #
        # The IP-Adapter is gated the same way and for the same reason: a
        # reference is optional on this kind. ``style_lock`` is the second way
        # to reach the encoder -- it hands the first material to the adapter as
        # the reference for every pass after it -- and the door has no
        # ``ip_adapter`` to write for that case, because the image does not
        # exist until the job is half over.
        #
        # The reduction and the palette are numpy and Pillow in this process,
        # so neither costs this budget anything. Never trellis: a tile sheet
        # has no mesh anywhere near it.
        image = _image_model_cost(params)
        sheet = image
        if params.get("control"):
            sheet += CONTROLNET_GIB
        sheet_block = params.get("sheet")
        locked = bool(isinstance(sheet_block, dict) and sheet_block.get("style_lock"))
        if params.get("ip_adapter") or locked:
            sheet += IP_ENCODER_GIB
        return (sheet if exclusive else sheet + TRELLIS_GIB), image
    if kind not in ("text", "image"):
        # rig / pose / sheet / charsheet are Blender, out of process and
        # CPU-side -- and the character sheet's pixel-art pass after it is
        # numpy and Pillow in this process, which costs this budget nothing
        # for the same reason the tile sheet's reduction does not.
        return 0.0, 0.0

    sdxl = 0.0
    if kind == "text":
        image = _image_model_cost(params)
        sdxl = image
        if params.get("control"):
            sdxl += CONTROLNET_GIB
        if params.get("ip_adapter"):
            sdxl += IP_ENCODER_GIB

    if stage in ("reference", "tile"):
        # No reconstruction at all -- a tile never reaches trellis, and the
        # circular-padding patch allocates nothing. Under coexist a warm
        # trellis is still resident beside the pipe and its memory is not
        # given back.
        return (sdxl if exclusive else sdxl + TRELLIS_GIB), image

    trellis = _trellis_cost(params)
    return (max(sdxl, trellis) if exclusive else sdxl + trellis), image


def estimate_job(job: dict[str, Any], *, exclusive: bool = False) -> float:
    """``estimate`` for a job row as the store hands it back."""
    return estimate_job_parts(job, exclusive=exclusive)[0]


def estimate_job_parts(job: dict[str, Any], *, exclusive: bool = False) -> tuple[float, float]:
    """``estimate_parts`` for a job row as the store hands it back."""
    return estimate_parts(
        job.get("kind", ""),
        job.get("stage", "model"),
        job.get("params") or {},
        exclusive=exclusive,
    )


# --- measuring the device ----------------------------------------------------


def _read(torch: Any) -> DeviceMemory | None:
    try:
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
    except Exception:  # noqa: BLE001 -- a reading must never raise
        log.debug("could not read CUDA memory", exc_info=True)
        return None
    try:
        get_name = getattr(torch.cuda, "get_device_name", None)
        name = str(get_name(0)) if callable(get_name) else ""
    except Exception:  # noqa: BLE001 -- a name must not erase valid figures
        name = ""
    return DeviceMemory(total_gib=total / _GIB, free_gib=free / _GIB, name=name)


_published: DeviceMemory | None = None
_published_lock = threading.Lock()


def publish(free_gib: float, total_gib: float, name: str = "") -> None:
    """Record a device reading taken in a child process.

    Since the image pipeline moved into ``text2image_worker``, the app process
    has no reason to import torch -- and ``device_memory`` returns None without
    it, which would leave admission with no device figure at all for the whole
    session. The child takes the reading it is already positioned to take
    (``mem_get_info`` reports the *device*, so its answer counts trellis-server
    too) and ``t2i_client`` hands it here.

    Last-writer-wins under a lock, and deliberately not timestamped: a stale
    reading is what "the card as of the last time anyone looked" means, and it
    is strictly better than the None it replaces. The live torch path below
    still wins whenever torch is in fact loaded, so a GPU test and the
    in-process fallback are unaffected.
    """
    global _published
    with _published_lock:
        _published = DeviceMemory(
            total_gib=float(total_gib), free_gib=float(free_gib), name=name
        )


def device_memory() -> DeviceMemory | None:
    """Device-wide free/total, or None -- without importing torch.

    Reached through ``sys.modules`` for the same reason ``queue.vram_gib``
    does: importing torch costs seconds and this is called from the event loop.
    Unlike ``vram_gib``, the number it returns *does* see trellis-server's
    allocation -- ``mem_get_info`` wraps ``cudaMemGetInfo``, which reports the
    device, not the process.

    Falls back to the last reading ``publish`` was given. Torch first, always:
    a reading this process can take now beats one a child took a minute ago.
    """
    torch = sys.modules.get("torch")
    live = None if torch is None else _read(torch)
    if live is not None:
        return live
    with _published_lock:
        return _published


#: NVML's own success code, and the two calls this module makes. The device
#: memory struct is three ``unsigned long long`` in bytes: total, free, used.
_NVML_SUCCESS = 0


#: A sentinel distinct from None, which is a *cached failure* here.
_UNSET: Any = object()
_NVML_SESSION: Any = _UNSET


class _NvmlMemory(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


def live_memory() -> DeviceMemory | None:
    """A reading taken *now*, through the driver, without torch. None if absent.

    :func:`device_memory` cannot answer this. It reads ``sys.modules`` and
    deliberately does not import torch -- and torch now lives in the
    ``text2image`` child process, so in the app process that function is either
    absent for a session that has never generated or is reporting the figure
    the *last* generation ended on. ``publish``'s own docstring says so. A
    meter drawn from it would be stale in exactly the state the user is asking
    about: "can I start something right now".

    NVML is ``nvidia-smi``'s own interface and ships with the driver, so this
    needs no package, no import cost and no child -- the same trade
    :mod:`~warlock.memlog` makes with psapi. NVIDIA-only: every other card
    returns None and the caller omits the figure rather than inventing one.

    **Its number may legitimately differ from admission control's.** On WDDM
    ``nvmlDeviceGetMemoryInfo`` reports *dedicated device* memory, while
    ``cudaMemGetInfo``'s free is the driver's virtualized view (see this
    module's docstring) -- so the meter can read fuller than the gate does.
    That is the meter being the better figure, not either one being broken.

    :func:`device_memory` is deliberately left alone. Promoting NVML into its
    ladder would improve admission control too, and that is a VRAM-admission
    change belonging in its own commit, verified under ``-m gpu``.
    """
    session = _nvml()
    if session is None:
        return None
    nvml, handle, name = session
    try:
        info = _NvmlMemory()
        if nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(info)) != _NVML_SUCCESS:
            return None
    except (OSError, AttributeError, ValueError):
        return None
    return DeviceMemory(
        total_gib=info.total / _GIB,
        free_gib=info.free / _GIB,
        name=name,
    )


def _nvml() -> tuple[Any, Any, str] | None:
    """The library, device 0's handle and its name, initialised once.

    **Held open deliberately.** ``nvmlInit_v2``/``nvmlShutdown`` around every
    reading measured **15.7 ms** on this machine (RTX 5090, 2026-08-23) --
    fifteen times the budget a frame-loop sampler has and nearly all of it the
    init/shutdown pair, which enumerates the device set. Held, a reading is
    ~0.02 ms. NVML is refcounted and the process is the natural scope, so
    there is nothing to release: the driver drops the session with the
    process, exactly as the CUDA context does.

    Cached negatively as well as positively -- a machine with no NVIDIA driver
    must not pay a failing ``CDLL`` every second.
    """
    global _NVML_SESSION

    if _NVML_SESSION is not _UNSET:
        return _NVML_SESSION
    _NVML_SESSION = None
    if sys.platform != "win32":
        return None
    try:
        nvml = ctypes.CDLL("nvml.dll")
        if nvml.nvmlInit_v2() != _NVML_SUCCESS:
            return None
        handle = ctypes.c_void_p()
        if nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(handle)) != _NVML_SUCCESS:
            nvml.nvmlShutdown()
            return None
        buffer = ctypes.create_string_buffer(96)
        if nvml.nvmlDeviceGetName(handle, buffer, len(buffer)) != _NVML_SUCCESS:
            buffer.value = b""
    except (OSError, AttributeError, ValueError):
        return None
    _NVML_SESSION = (nvml, handle, buffer.value.decode("utf-8", "replace"))
    return _NVML_SESSION


def probe() -> DeviceMemory | None:
    """The variant permitted to import torch. **Not the startup path.**

    The docstring here used to say "the startup variant" and justify itself
    with "``doctor._cuda_check`` already pays that import during
    ``run_checks``, so this costs nothing extra". That stopped being true when
    ``_cuda_check`` learned to defer the import (C29) and nobody revisited this
    -- leaving ``_vram_check`` as the one row in ``run_checks`` that still
    imported torch unconditionally, so the deferral bought nothing at all on
    the recommended install. Measured on 2026-08-24: 1,511 ms per cold start,
    against 47 ms for the whole check set once this stopped being called from
    it.

    ``doctor._vram_check`` now calls :func:`device_memory` instead unless it
    has actually been asked to probe. Use this only where paying seconds for a
    definitive answer is the right trade -- and never from the frame thread or
    the event loop.
    """
    try:
        import torch
    except ImportError:
        return None
    return _read(torch)


# --- resolving a plan --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Plan:
    """The resolved VRAM policy for this process."""

    total_gib: float | None
    """Device total, or the WARLOCK_VRAM_TOTAL override. None = unknown."""
    budget_gib: float | None
    """What a job may ask for. None disables admission control entirely."""
    exclusive: bool
    """The resolved value of ``Config.vram_exclusive``."""
    reason: str
    """Why, in one line, for the log and the doctor row."""
    explicit: bool = False
    """Whether the mode came from ``WARLOCK_VRAM_EXCLUSIVE`` rather than the card.

    Carried rather than re-derived, because it stops being derivable the moment
    the decision is written back: ``Runtime._resolve_vram`` puts a plain bool
    onto the tri-state ``Config.vram_exclusive``, so every later ``plan()``
    call -- every health poll -- saw a set value and reported an auto-selected
    mode as "set explicitly by WARLOCK_VRAM_EXCLUSIVE".
    """

    @property
    def enforced(self) -> bool:
        return self.budget_gib is not None

    def fits(self, need_gib: float) -> bool:
        return self.budget_gib is None or need_gib <= self.budget_gib


def plan(
    *,
    exclusive: bool | None,
    budget_gib: float | None = None,
    total_gib: float | None = None,
    device: DeviceMemory | None = None,
    explicit: bool | None = None,
) -> Plan:
    """Resolve the mode and the budget.

    An explicitly set ``WARLOCK_VRAM_EXCLUSIVE`` (i.e. ``exclusive`` not None)
    is honoured verbatim and never overridden -- auto-detection exists to give
    an unconfigured host a sane default, not to argue with one that was
    configured.

    ``explicit`` says whether that set value came from the environment, for a
    caller resolving a config the startup pass has already written a bool back
    onto. Left None it is inferred, which is right the first time and only the
    first time.
    """
    if explicit is None:
        explicit = exclusive is not None
    total = total_gib if total_gib is not None else (device.total_gib if device else None)
    if total is None:
        return Plan(
            None,
            None,
            bool(exclusive),
            "no CUDA device detected; VRAM admission control is off",
            explicit,
        )

    budget = budget_gib if budget_gib is not None else max(total - HEADROOM_GIB, 0.0)

    if exclusive is not None and explicit:
        mode = "exclusive" if exclusive else "coexist"
        return Plan(
            total,
            budget,
            exclusive,
            f"{total:.1f} GiB card, {budget:.1f} GiB budget; "
            f"{mode} set explicitly by WARLOCK_VRAM_EXCLUSIVE",
            True,
        )
    if exclusive is not None:
        # Already resolved once, and told so. Report the mode without claiming
        # an origin: re-deriving it here is what invented the environment
        # variable nobody set.
        mode = "exclusive" if exclusive else "coexist"
        return Plan(
            total,
            budget,
            exclusive,
            f"{total:.1f} GiB card, {budget:.1f} GiB budget; {mode}",
            False,
        )

    auto = budget < COEXIST_GIB
    mode = "exclusive" if auto else "coexist"
    why = (
        f"{budget:.1f} GiB budget cannot hold trellis and an image model "
        f"together ({COEXIST_GIB:.1f} GiB)"
        if auto
        else f"{budget:.1f} GiB budget holds trellis and an image model "
        f"together ({COEXIST_GIB:.1f} GiB)"
    )
    return Plan(total, budget, auto, f"{total:.1f} GiB card, {mode} auto-selected: {why}")


# --- does a checkpoint fit this card -----------------------------------------

FIT_OK = "fits"
FIT_TIGHT = "tight"
FIT_NO = "no"


def fits(plan_: Plan, spec: Any) -> str:
    """FIT_OK / FIT_TIGHT / FIT_NO for one checkpoint on this card.

    A three-valued answer rather than a bool, because "it will load but only if
    nothing else is resident" is the honest description of an SDXL-class pipe on
    a 24 GB card and is neither of the other two: refusing it would be wrong,
    and calling it fine would hide the reload that will happen on every 3D job.

    The rules, in order:

    * over the budget outright -- ``no``. Nothing rescues this; the pipe alone
      does not fit.
    * handing off anyway -- ``fits``. Under ``exclusive`` trellis is stopped
      before the checkpoint loads, and an ``OFFLOAD`` spec hands off whatever
      the flag says (``offloaded_base``), so in both cases the checkpoint is
      alone on the card and its own figure is the whole story.
    * would not sit beside trellis -- ``tight``. This is the coexist case, and
      the cost is a stop/start per job rather than a failure.
    * otherwise ``fits``.

    An unknown budget is ``fits``, deliberately: the absence of a measurement is
    not evidence of a shortfall, which is the same rule ``Plan.fits`` and
    ``disk_refusal`` already follow.

    What this does **not** model is host commit for an offloaded spec. Those
    weights live in host RAM for the life of the pipe and no figure here
    describes that; :mod:`~warlock.memlog` owns commit, and inventing a margin
    for it in a badge would be a number nobody measured.
    """
    from . import models

    budget = plan_.budget_gib
    if budget is None:
        return FIT_OK
    if spec.vram_gib > budget:
        return FIT_NO
    if plan_.exclusive or spec.residency == models.OFFLOAD:
        return FIT_OK
    if spec.vram_gib + TRELLIS_GIB > budget:
        return FIT_TIGHT
    return FIT_OK


# Which base to suggest, best first. Quality order, from the 2026-08-09
# re-baseline's marginal accept rates (`sdxl_cfg` 3/3, baseline `sdxl` 2/4,
# `turbo` 1/5) with `lightning` placed beside `sdxl` as the other distillation
# arm over the same weights. Only SDXL-family entries, and only ones that share
# a download with the shipped default or are a plain checkpoint of their own --
# a recommendation that named a 16 GB model the user has not got would be a
# worse answer than the one they already picked.
RECOMMENDED_BASES: tuple[str, ...] = ("sdxl_cfg", "sdxl", "lightning", "turbo")


def recommended_base(plan_: Plan) -> str:
    """The best base this card can actually hold, by key.

    Preference order first, then fit: a ``fits`` entry anywhere in the list
    beats a ``tight`` one earlier in it, because "tight" means every 3D job
    pays a trellis restart and that cost outweighs the ranking gap between two
    recipes over the same weights.

    Falls back to ``models.DEFAULT_BASE_MODEL`` when nothing fits, rather than
    to nothing: a card that can hold no checkpoint at all has a problem no
    picker can solve, and the caller's job is to show a badge, not to refuse.
    """
    from . import models

    scored = [
        (key, fits(plan_, models.BASE_MODELS[key]))
        for key in RECOMMENDED_BASES
        if key in models.BASE_MODELS
    ]
    for wanted in (FIT_OK, FIT_TIGHT):
        for key, verdict in scored:
            if verdict == wanted:
                return key
    return models.DEFAULT_BASE_MODEL


def _smaller_base(plan_: Plan, params: dict[str, Any]) -> Any | None:
    """A recommendable checkpoint strictly cheaper than the one this job picked.

    Deliberately **not** restricted to the picked model's own family. The two
    families in the registry are each internally flat in ``vram_gib`` -- every
    SDXL entry is 7.0 and both FLUX.2 klein entries are 10.0 -- so a
    same-family rule could never fire, and the case that actually costs a user
    a refusal is exactly the cross-family one: a 10 GiB offloaded klein picked
    on a card that has room for an SDXL pipe and nothing larger.
    """
    from . import models

    spec = models.BASE_MODELS.get(str(params.get("base_model") or ""))
    if spec is None:
        return None
    other = models.BASE_MODELS[recommended_base(plan_)]
    if other.key == spec.key or other.vram_gib >= spec.vram_gib:
        return None
    return other if fits(plan_, other) != FIT_NO else None


def remedies(
    params: dict[str, Any] | None = None,
    *,
    exclusive: bool = False,
    extra: Sequence[str] = (),
    plan_: Plan | None = None,
) -> str:
    """The "to run it, ..." clause: the things about *this job* that cost VRAM.

    Extracted so the submit-time refusal and the dispatch-time one share it
    (N113). They were two messages about one condition with two different sets
    of advice -- the door named the conditioning to turn off and the resolution
    to drop, and the dispatch check, which is the one that fires after the user
    has already waited in the queue, said "close other GPU applications" and
    nothing else. The remedies are a property of the job, not of which check
    noticed.

    ``extra`` goes first, for the remedy that belongs to one caller only.

    ``plan_`` unlocks one more remedy: when the job named a checkpoint this card
    is too small for and a cheaper one is available, say so. It goes *before*
    the environment-variable clause deliberately -- picking a smaller model is
    a click, and setting ``WARLOCK_VRAM_EXCLUSIVE`` is a restart, so the last
    thing offered stays the most drastic one. Optional because
    ``dispatch_shortfall_message`` has no plan to give: that check fires on
    *free* memory, where the budget the fit is computed against is not the
    thing that ran out.
    """
    params = params or {}
    out: list[str] = list(extra)
    if params.get("control"):
        out.append("turn off ControlNet conditioning")
    if params.get("ip_adapter"):
        out.append("turn off the IP-Adapter reference")
    if _resolution(params) > 1024:
        out.append("drop the platform preset to a lower resolution")
    if plan_ is not None:
        cheaper = _smaller_base(plan_, params)
        if cheaper is not None:
            out.append(f"pick a smaller base model such as {cheaper.label!r}")
    if not exclusive:
        out.append("set WARLOCK_VRAM_EXCLUSIVE=1 and restart")
    if not out:
        out.append("raise WARLOCK_VRAM_BUDGET if this card has more room than reported")
    if len(out) > 1:
        out[-1] = "or " + out[-1]
    return ", ".join(out) if len(out) > 2 else " ".join(out)


def shortfall_message(need_gib: float, plan_: Plan, params: dict[str, Any] | None = None) -> str:
    """Why the job was refused, and what to change. One sentence, then remedies."""
    budget = plan_.budget_gib if plan_.budget_gib is not None else 0.0
    tail = remedies(params, exclusive=plan_.exclusive, plan_=plan_)
    return (
        f"this job needs about {need_gib:.1f} GiB of VRAM; the budget is "
        f"{budget:.1f} GiB. To run it, {tail}."
    )


def dispatch_shortfall_message(
    need_gib: float,
    headroom_gib: float,
    free_gib: float,
    params: dict[str, Any] | None = None,
    *,
    exclusive: bool = False,
) -> str:
    """The same shape for the refusal at dispatch (N113).

    It says a different *thing* from ``shortfall_message`` -- what is free now,
    rather than what the budget allows -- because that is genuinely the
    condition it caught: the card was big enough when the job was submitted and
    something took the room while it waited. But the remedy half is the job's,
    so it is the same half, plus the one remedy only this check can offer.
    """
    tail = remedies(
        params, exclusive=exclusive, extra=("close other GPU applications and try again",)
    )
    held = headroom_gib - free_gib
    return (
        f"this job needs about {need_gib:.1f} GiB of VRAM and only "
        f"{headroom_gib:.1f} GiB is available to it ({free_gib:.1f} GiB free plus "
        f"{held:.1f} GiB in models Warlock already holds). To run it, {tail}."
    )
