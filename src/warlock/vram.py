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

import logging
import sys
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


def _image_model_cost(params: dict[str, Any] | None) -> float:
    """What this job's chosen checkpoint costs, in GiB.

    Not every image model is 7 GB any more: a spec carries its own
    ``vram_gib``, measured under its own ``residency``. Importing ``models``
    here is consistent with this module's purity rule -- the ban is on
    ``service``/``queue``/``studio``, and ``models`` is stdlib-only.
    """
    from . import models

    spec = models.BASE_MODELS.get(str((params or {}).get("base_model") or ""))
    return SDXL_GIB if spec is None else spec.vram_gib


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
    params = params or {}
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
        pass_gib = _image_model_cost(params)
        if params.get("control"):
            pass_gib += CONTROLNET_GIB
        # Never trellis: the mesh was reconstructed long before, and nothing
        # here goes near it. Under coexist a warm trellis is still holding its
        # memory, which is what the reference stage below accounts for too.
        return pass_gib if exclusive else pass_gib + TRELLIS_GIB
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
        pixel = _image_model_cost(params) + CONTROLNET_GIB
        return pixel if exclusive else pixel + TRELLIS_GIB
    if kind not in ("text", "image"):
        # rig / pose / sheet are Blender, out of process and CPU-side.
        return 0.0

    sdxl = 0.0
    if kind == "text":
        sdxl = _image_model_cost(params)
        if params.get("control"):
            sdxl += CONTROLNET_GIB
        if params.get("ip_adapter"):
            sdxl += IP_ENCODER_GIB

    if stage in ("reference", "tile"):
        # No reconstruction at all -- a tile never reaches trellis, and the
        # circular-padding patch allocates nothing. Under coexist a warm
        # trellis is still resident beside the pipe and its memory is not
        # given back.
        return sdxl if exclusive else sdxl + TRELLIS_GIB

    trellis = _trellis_cost(params)
    return max(sdxl, trellis) if exclusive else sdxl + trellis


def estimate_job(job: dict[str, Any], *, exclusive: bool = False) -> float:
    """``estimate`` for a job row as the store hands it back."""
    return estimate(
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
    return DeviceMemory(total_gib=total / _GIB, free_gib=free / _GIB)


def device_memory() -> DeviceMemory | None:
    """Device-wide free/total, or None -- without importing torch.

    Reached through ``sys.modules`` for the same reason ``queue.vram_gib``
    does: importing torch costs seconds and this is called from the event loop.
    Unlike ``vram_gib``, the number it returns *does* see trellis-server's
    allocation -- ``mem_get_info`` wraps ``cudaMemGetInfo``, which reports the
    device, not the process.
    """
    torch = sys.modules.get("torch")
    return None if torch is None else _read(torch)


def probe() -> DeviceMemory | None:
    """The startup variant, permitted to import torch.

    ``doctor._cuda_check`` already pays that import during ``run_checks``, so
    by the time the plan is resolved this costs nothing extra.
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


def remedies(
    params: dict[str, Any] | None = None,
    *,
    exclusive: bool = False,
    extra: Sequence[str] = (),
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
    """
    params = params or {}
    out: list[str] = list(extra)
    if params.get("control"):
        out.append("turn off ControlNet conditioning")
    if params.get("ip_adapter"):
        out.append("turn off the IP-Adapter reference")
    if _resolution(params) > 1024:
        out.append("drop the platform preset to a lower resolution")
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
    tail = remedies(params, exclusive=plan_.exclusive)
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
