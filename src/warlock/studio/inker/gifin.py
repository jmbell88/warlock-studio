"""An animated GIF as an editable document: ``gifout``'s inverse.

Inker could write a GIF and not open one. That made the export a one-way door
-- the file a user had just shared was a file this editor refused -- and it is
the only export in the app with that shape, so this closes it rather than
documenting it.

Two things the format does are the whole of the work here.

**A frame is not a picture, it is a patch.** GIF frames carry an offset, a size
and a disposal method, and a decoder has to compose them to arrive at what is
on screen. Pillow does exactly that when it is seeked, so every frame is read
*through* a seek and converted to RGBA there -- never assembled by hand. A
naive read of the raw frames gives a clip that is correct on frame one and
increasingly wrong after it.

**Time is stored in hundredths of a second**, so the durations that come back
are the rounded ones ``gifout`` wrote, not the exact ones the timeline held.
That is a property of the file and not a loss here; the round trip pins it.

A *one frame* GIF opens as a still drawing. A timeline with a single row in it
would be a claim about the file that the file does not make, and every other
single-picture format this editor opens arrives as a still.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .animation import DEFAULT_DURATION_MS
from .sheetin import document_from_grid

__all__ = ["frames_of_gif", "read_gif"]


def frames_of_gif(path: Any) -> tuple[list[np.ndarray], list[int]]:
    """Every composed frame as RGBA, with its duration in milliseconds.

    Blocking and Pillow-only; the caller is on a task thread. A frame with no
    duration recorded -- which a still GIF and some encoders both produce --
    gets the timeline's own default rather than zero, because a zero-length
    frame is one no player agrees about.
    """
    from PIL import Image, ImageSequence

    planes: list[np.ndarray] = []
    durations: list[int] = []
    with Image.open(path) as im:
        for frame in ImageSequence.Iterator(im):
            planes.append(np.asarray(frame.convert("RGBA"), dtype=np.uint8).copy())
            duration = int(frame.info.get("duration") or 0)
            durations.append(duration if duration > 0 else DEFAULT_DURATION_MS)
    if not planes:
        raise ValueError("a gif needs at least one frame")
    return planes, durations


def read_gif(path: Any, *, track_name: str = "Sprite") -> Any:
    """One GIF as a document: a still for one frame, a clip for more.

    The clip is built by stacking the frames into a one-column atlas and
    handing that to :func:`.sheetin.document_from_grid` -- the same door an
    imported sprite sheet comes through. Reusing it rather than assembling an
    ``Animation`` here is what keeps the cel map, the track and the refusals in
    one place; the atlas is the only copy this costs, and a GIF is small.
    """
    from .document import Document

    planes, durations = frames_of_gif(path)
    if len(planes) == 1:
        doc = Document.from_pixels(planes[0])
        doc.path = Path(path)
        return doc
    atlas = np.concatenate(planes, axis=0)
    height, width = planes[0].shape[:2]
    doc = document_from_grid(atlas, (width, height), track_name=track_name)
    for frame, duration in zip(doc.anim.frames, durations, strict=True):
        frame.duration_ms = duration
    doc.path = Path(path)
    return doc
