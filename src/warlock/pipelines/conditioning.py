"""The frozen value object describing one job's SDXL conditioning.

Deliberately inert: no torch, no Pillow, no registry lookups, nothing that
could fail. It is built on the worker thread from a job's params and crosses
the ``asyncio.to_thread`` boundary into Text2Image.generate, so it has to be
immutable and cheap to construct -- the resolution work (does this key exist,
does the checkpoint allow a ControlNet, where does the hint image go) belongs
to guidance.normalize and queue.Worker, not here.

``None`` and a falsy Conditioning mean the same thing to the pipeline: take
the unconditioned path, which is bit-identical to what the app produced
before any of this existed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Re-exported from the registry rather than restated, so a default can only
# ever be changed in one place. models.py is torch-free too.
from ..models import (  # noqa: F401
    DEFAULT_CONTROL_END,
    DEFAULT_CONTROL_SCALE,
    DEFAULT_IMG2IMG_STRENGTH,
    DEFAULT_IP_SCALE,
)


@dataclass(frozen=True, slots=True)
class Conditioning:
    """What to attach to the resident SDXL pipe for one generate() call.

    ``ip_*`` is appearance/identity (IP-Adapter over a reference image);
    ``control_*`` is structure (a ControlNet over a preprocessed hint image).
    The two are independent and either may be absent.
    """

    ip_adapter: str | None = None
    ip_image: Path | None = None
    ip_scale: float = DEFAULT_IP_SCALE
    control: str | None = None
    control_image: Path | None = None
    control_scale: float = DEFAULT_CONTROL_SCALE
    control_end: float = DEFAULT_CONTROL_END
    # img2img: the picture the denoise *starts from*, and how far it is taken.
    # A third independent axis rather than a variant of the two above -- it
    # changes which pipeline class runs, where the other two only change what
    # is attached to one.
    init_image: Path | None = None
    strength: float = DEFAULT_IMG2IMG_STRENGTH

    @property
    def uses_ip(self) -> bool:
        return bool(self.ip_adapter and self.ip_image)

    @property
    def uses_control(self) -> bool:
        return bool(self.control and self.control_image)

    @property
    def uses_init(self) -> bool:
        return self.init_image is not None

    def __bool__(self) -> bool:
        return self.uses_ip or self.uses_control or self.uses_init

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe, and only the halves that are actually in play -- a scale
        with nothing to scale reads as a live setting when the recipe is read
        back, which is the same rule guidance.normalize follows for
        style_lora/lora_weight.
        """
        out: dict[str, Any] = {}
        if self.uses_ip:
            out["ip_adapter"] = self.ip_adapter
            out["ip_image"] = self.ip_image.name if self.ip_image else None
            out["ip_scale"] = self.ip_scale
        if self.uses_control:
            out["control"] = self.control
            out["control_image"] = self.control_image.name if self.control_image else None
            out["control_scale"] = self.control_scale
            out["control_end"] = self.control_end
        if self.uses_init:
            out["init_image"] = self.init_image.name if self.init_image else None
            out["strength"] = self.strength
        return out
