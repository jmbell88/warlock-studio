"""What goes into an atlas: one immutable RGBA image with a stable key.

``key`` is the sprite's identity and the packer's total order. It has to be
stable across a repack -- the whole determinism contract in :mod:`.maxrects`
bottoms out in sorting by it -- so it is derived from *where the sprite came
from* rather than from its name: names legitimately repeat, and two layers
called "Layer 1" packing into one slot would be a silent data loss.

**An Inker document is enumerated, never interpreted.** An animated document
gives one sprite per frame, through ``Document.frame_flat``, which is the same
flatten the timeline plays and the onion skin draws -- so a packed frame is
pixel-identical to what the user was looking at. A still document gives one
sprite per layer, hidden layers included: the pane chooses what to include, and
an enumerator that silently dropped rows would make "why is my sprite missing"
a question about two places at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Shared rather than copied. This was a byte-identical second spelling of
# ``plotter.tileset``'s helper, differing only in the noun in its error
# message, which is now a parameter. The edge is already pinned in both
# directions -- ``tsxout`` imports the same module for the .tsx writer -- so
# this adds no dependency the package did not already have.
from ..plotter.tileset import frozen_rgba


@dataclass(frozen=True)
class Sprite:
    """One image to be packed. ``name`` is for humans; ``key`` is identity."""

    key: str
    name: str
    pixels: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "pixels", frozen_rgba(self.pixels, "a sprite"))
        if not self.key:
            raise ValueError("a sprite needs a key")

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])


def frame_key(prefix: str, index: int) -> str:
    """One animation frame's key. Zero-padded so a lexical sort is a temporal
    one -- which matters because the packer's canonical order is by key."""
    return f"{prefix}#frame{index:04d}"


def layer_key(prefix: str, index: int, name: str) -> str:
    """One layer's key. The *index* is in it as well as the name, because two
    layers may legitimately share a name and two sprites may not share a key."""
    return f"{prefix}#layer{index:02d}:{name}"


def sprites_from_document(doc: Any, *, prefix: str) -> list[Sprite]:
    """Every sprite an Inker document offers, in its own natural order.

    Animated: one per frame, bottom to top of the timeline. Still: one per
    layer, in stack order, which is bottom-first exactly as the layers panel
    shows it upside down.
    """
    anim = getattr(doc, "anim", None)
    if anim is not None and anim.frames:
        out = []
        for index, frame in enumerate(anim.frames):
            flat = doc.frame_flat(frame.uid)
            if flat is None:
                # A frame the document declines to flatten is not a frame we
                # can pack; skipping it silently would leave a clip one cell
                # short with nothing to say which one.
                raise ValueError(f"frame {index + 1} of {prefix!r} could not be flattened")
            out.append(
                Sprite(key=frame_key(prefix, index), name=f"{prefix} {index + 1}", pixels=flat)
            )
        return out

    return [
        Sprite(
            key=layer_key(prefix, index, layer.name),
            name=layer.name or f"{prefix} {index + 1}",
            pixels=layer.pixels,
        )
        for index, layer in enumerate(doc.stack)
    ]


def sprite_from_image(pixels: Any, *, key: str, name: str = "") -> Sprite:
    """A loose image file, already decoded. Here rather than in the host so the
    RGBA check and the copy happen in exactly one place."""
    return Sprite(key=key, name=name or key, pixels=pixels)
