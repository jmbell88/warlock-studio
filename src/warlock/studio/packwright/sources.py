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

**Metadata arrives as plain data and is frozen here.** A sprite can carry a
pivot and a list of named rectangles, which is what lets an atlas say where a
sprite is placed from and which part of a UI panel stretches. Packwright reads
them through one duck-typed method -- ``Document.sprite_meta_for_frame`` -- and
never learns what a frame, a per-frame key or a ``Slice`` is: the resolving is
the drawing editor's business and happens on its side of the call, and what
crosses is dicts and tuples that :func:`sprite_meta` turns into the frozen types
below. That is why this module imports nothing from the raster editor, which its
own package pin requires.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Shared rather than copied. This was a byte-identical second spelling of
# ``plotter.tileset``'s helper, differing only in the noun in its error
# message, which is now a parameter. The edge is already pinned in both
# directions -- ``tsxout`` imports the same module for the .tsx writer -- so
# this adds no dependency the package did not already have.
from ..plotter.tileset import frozen_rgba


@dataclass(frozen=True, slots=True)
class SliceSpec:
    """One named rectangle on a sprite, in **source-image** coordinates.

    Source-image and not trimmed: a slice describes the picture the artist drew,
    and trimming is something the packer did afterwards. A consumer that wants
    the trimmed frame already has ``spriteSourceSize`` to subtract, and baking
    the trim in here would make the numbers wrong the moment trimming is turned
    off.

    ``center`` is the stretchable middle of a nine-slice panel, as
    ``(x, y, w, h)`` -- the same spelling as the four fields above it, so a
    reader has one rectangle convention rather than two.
    """

    name: str
    x: int
    y: int
    w: int
    h: int
    pivot: tuple[float, float] | None = None
    center: tuple[int, int, int, int] | None = None


@dataclass(frozen=True, slots=True)
class SpriteMeta:
    """Everything about a sprite that is not its pixels.

    A frozen record with defaults, so a sprite that has none is exactly the
    sprite this packer has always had -- and so the layout, the sidecars and the
    document format each get to be additive rather than versioned.
    """

    pivot: tuple[float, float] | None = None
    slices: tuple[SliceSpec, ...] = ()

    def __bool__(self) -> bool:
        return self.pivot is not None or bool(self.slices)


#: The shared "nothing to say" instance. Frozen, so sharing is safe, and one
#: object rather than a default factory keeps ``Sprite`` cheap to build in the
#: overwhelmingly common case.
EMPTY_META = SpriteMeta()


def _point(value: Any) -> tuple[float, float] | None:
    return None if value is None else (float(value[0]), float(value[1]))


def _rect(value: Any) -> tuple[int, int, int, int] | None:
    return None if value is None else tuple(int(v) for v in value)  # type: ignore[return-value]


def sprite_meta(raw: Any) -> SpriteMeta:
    """Plain dicts from a document into the frozen record above.

    The coercion lives on *this* side of the call deliberately. The producer is
    free to be a raster editor with per-frame keys or a loose PNG with nothing
    at all, and the packer's types stay the packer's -- which is what keeps this
    package importable with no editor present and its import pin honest.

    Anything missing is simply absent: this is a read of somebody else's data,
    and a metadata field that will not parse must cost the pivot rather than the
    sprite. ``.wpack``'s reader is the one place that refuses instead, because
    there a malformed field means a file that is wrong about itself.
    """
    if not isinstance(raw, dict):
        return EMPTY_META
    slices = []
    for entry in raw.get("slices") or ():
        try:
            slices.append(
                SliceSpec(
                    name=str(entry.get("name", "")),
                    x=int(entry["x"]),
                    y=int(entry["y"]),
                    w=int(entry["w"]),
                    h=int(entry["h"]),
                    pivot=_point(entry.get("pivot")),
                    center=_rect(entry.get("center")),
                )
            )
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    try:
        pivot = _point(raw.get("pivot"))
    except (TypeError, ValueError, IndexError):
        pivot = None
    return SpriteMeta(pivot=pivot, slices=tuple(slices))


@dataclass(frozen=True)
class Sprite:
    """One image to be packed. ``name`` is for humans; ``key`` is identity."""

    key: str
    name: str
    pixels: np.ndarray
    #: Trailing and defaulted, so every existing construction site -- and every
    #: caller that builds one from a loose file -- is unchanged.
    meta: SpriteMeta = field(default=EMPTY_META)

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


def _meta_of(doc: Any, frame_uid: Any) -> SpriteMeta:
    """One frame's metadata, if the document has any to give.

    Duck-typed rather than isinstance-checked: the packer takes sprites from
    Inker documents, from loose files and from test doubles, and a document
    without the method is one with nothing to say -- not an error.
    """
    read = getattr(doc, "sprite_meta_for_frame", None)
    return EMPTY_META if read is None else sprite_meta(read(frame_uid))


def sprites_from_document(doc: Any, *, prefix: str) -> list[Sprite]:
    """Every sprite an Inker document offers, in its own natural order.

    Animated: one per frame, bottom to top of the timeline. Still: one per
    layer, in stack order, which is bottom-first exactly as the layers panel
    shows it upside down.

    Metadata comes off the document per *frame*, and every layer of a still
    document gets the same one: a slice is a rectangle on the canvas rather than
    on a layer, so a still document's layers share it -- which is also what a
    nine-slice panel drawn on two layers means.
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
                Sprite(
                    key=frame_key(prefix, index),
                    name=f"{prefix} {index + 1}",
                    pixels=flat,
                    meta=_meta_of(doc, frame.uid),
                )
            )
        return out

    meta = _meta_of(doc, None)
    return [
        Sprite(
            key=layer_key(prefix, index, layer.name),
            name=layer.name or f"{prefix} {index + 1}",
            pixels=layer.pixels,
            meta=meta,
        )
        for index, layer in enumerate(doc.stack)
    ]


def sprite_from_image(pixels: Any, *, key: str, name: str = "") -> Sprite:
    """A loose image file, already decoded. Here rather than in the host so the
    RGBA check and the copy happen in exactly one place."""
    return Sprite(key=key, name=name or key, pixels=pixels)
