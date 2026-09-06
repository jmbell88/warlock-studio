"""The cycle, landed as a new Inker document.

**Built directly, not through undoable ops.** ``sheetin._document_from_rects``
settled the shape and the argument is the same one: a document a generator hands
you should open the way a freshly imported sheet opens -- no path, an empty undo
stack, and a first Ctrl+Z that does nothing because there is nothing before it.
Landing it through ``add_frame``/``add_layer`` would push a dozen steps a user
could undo into a half-built walk.

Which is also why this is a *new* document rather than a group in the old one.
Flourish inserts into the open document because an effect belongs to the drawing
it is drawn over; a walk cycle is a different sprite, and the still drawing the
user started from is left exactly as it was -- nothing in this whole feature ever
edits it, so cancelling has nothing to undo.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..animation import Animation, Frame, Tag, Track
from ..composite import over, to_float, to_uint8
from ..layers import Layer, LayerStack
from . import gait, render
from . import rig as R

#: The name of the one tag a bake writes.
TAG_NAME = "walk"

#: Canvas ceiling. Fourteen tracks times eight frames is fourteen times eight
#: canvas-sized planes, which is the document's inherent weight and not a cost
#: this module could optimise away -- at 1024 square it is most of a gigabyte.
#: Pixel art is small by definition, so this refuses a mistake rather than a
#: workflow.
WALK_MAX_PIXELS = 1024 * 1024


def too_large(size: tuple[int, int]) -> str:
    """Why this canvas is too big to bake, or ``""``."""
    if size[0] * size[1] > WALK_MAX_PIXELS:
        return (
            f"A walk cycle holds {len(R.PART_NAMES)} layers on every frame, so a "
            f"{size[0]}x{size[1]} canvas is too large to bake. Scale the drawing "
            "down first."
        )
    return ""


def composite_frames(
    rest: R.Rig,
    settings: gait.WalkSettings,
    size: tuple[int, int],
    count: int = gait.WALK_FRAMES,
) -> list[np.ndarray]:
    """One flattened RGBA plane per frame -- what the preview shows.

    Folded through the same ``composite.over`` the editor composites layers with,
    off the same placed planes the bake writes, so the preview and the baked
    document cannot disagree about what the walk looks like.
    """
    out: list[np.ndarray] = []
    for frame in render.frames(rest, settings, count):
        flat = to_float(np.zeros((size[1], size[0], 4), dtype=np.uint8))
        for name in rest.order:
            drawn = frame.get(name)
            if drawn is None:
                continue
            flat = over(flat, to_float(render.place(drawn, size)))
        out.append(to_uint8(flat))
    return out


def document(
    rest: R.Rig,
    settings: gait.WalkSettings,
    size: tuple[int, int],
    *,
    matte: tuple[int, int, int, int] | None = None,
    count: int = gait.WALK_FRAMES,
) -> Any:
    """The walk as an ordinary animated Inker document.

    One track per assigned part in draw order, one cel per part per frame, every
    cel its own ``Layer`` -- **never linked**, because the brief's promise is
    that each frame is independently editable and two slots holding one object
    would make a stroke on frame three appear on frame five.

    One looping tag over the whole cycle. Durations from the settings, uniform,
    because a walk's frames are evenly spaced by construction here.
    """
    from ..document import Document
    from ..undo import UNDO_BYTES, UndoStack

    refused = R.refusal(rest) or too_large(size)
    if refused:
        raise ValueError(refused)

    rendered = render.frames(rest, settings, count)
    duration = max(1, int(settings.duration_ms))
    frames = [Frame(duration_ms=duration) for _ in rendered]
    drawn = [name for name in rest.order if rest.parts[name].assigned]
    tracks = [Track(name=R.label(name)) for name in drawn]

    cels: dict[tuple[int, int], Layer] = {}
    for track, name in zip(tracks, drawn, strict=True):
        for frame, rendered_frame in zip(frames, rendered, strict=True):
            placed = rendered_frame.get(name)
            if placed is None:
                continue
            cels[(track.uid, frame.uid)] = Layer(
                pixels=render.place(placed, size), name=track.name
            )

    anim = Animation(
        tracks=tracks,
        frames=frames,
        cels=cels,
        tags=[Tag(name=TAG_NAME, start=0, end=len(frames) - 1, loop=True)],
        current=0,
    )
    doc = Document(
        stack=LayerStack(anim.layers_for(anim.frames[0], size), 0),
        history=UndoStack(UNDO_BYTES),
        anim=anim,
    )
    # Carried from the source rather than inferred: the drawing this was cut out
    # of already answered "is this a photo", and a walk made of its pixels is the
    # same answer. ``None`` is the transparent default an ``.ora`` round-trips.
    doc.matte = matte
    # ``ora`` and no path, for ``sheetin``'s reason: the document is an animation
    # and the still formats cannot hold one, so the format it *would* be saved as
    # is the one that can.
    doc.file_format = "ora"
    doc.path = None
    return doc
