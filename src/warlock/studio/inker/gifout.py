"""An Inker animation as an animated GIF.

The other export is the atlas ``sheetout`` writes, and the two are not
alternatives: a sheet is what an engine consumes, and a GIF is what a person
looks at. Nothing in this repository could produce a file that plays a clip
without an engine around it -- the sheet needs a runtime that reads the sidecar,
and an ``.ora`` needs this editor -- so a finished animation could not be shown
to anybody who was not running Warlock.

Pure in the ``sheetout`` sense minus its one exception: stdlib, numpy and a lazy
Pillow, and *no* import from ``pipelines`` either, because a GIF carries its own
timing in its own frames and there is no sidecar format to be the authority on.

Three things about the format decide everything below, and each of them is a
place a naive encode goes quietly wrong.

**A GIF pixel is a palette index, and one index may be transparent.** So alpha
is not stored, it is *thresholded*: a pixel is either fully there or fully gone,
with no partial coverage anywhere. That is the reason ``ALPHA_THRESHOLD`` is a
constant with an argument attached rather than a parameter, and it is also why
the soft brush and this exporter are a poor pairing -- an antialiased rim
becomes a hard one at whatever cut is chosen. It is exactly what the pixel nibs
produce naturally.

**Time is stored in hundredths of a second.** A 100 ms frame survives; a 33 ms
one does not, and the file plays at a rate the editor never showed. Rounding is
done *here*, and reported, rather than left to Pillow's truncation -- which
would turn a 15 ms frame into 10 ms and a whole clip 33% fast.

**Frame one's palette is not every frame's palette.** Each frame is quantised
independently, which is what keeps a colour appearing halfway through a clip
from being mapped onto the nearest colour of frame one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = ["ALPHA_THRESHOLD", "GIF_TICK_MS", "TRANSPARENT_INDEX", "quantise", "write_gif"]

#: The alpha at and above which a pixel is drawn at all. Halfway, because the
#: format offers no other answer: the choice is which side of the fence a
#: half-covered pixel falls on, and any other cut is a claim that one kind of
#: edge is more common than the other.
ALPHA_THRESHOLD = 128

#: The palette slot given to transparency. The last one, so the 255 colours the
#: quantiser is allowed to choose are the low indices and nothing it picks can
#: collide with it.
TRANSPARENT_INDEX = 255

#: The resolution of a GIF's own clock, in milliseconds.
GIF_TICK_MS = 10


def quantise(frame: np.ndarray) -> Any:
    """One RGBA plane as a palette image with ``TRANSPARENT_INDEX`` punched out.

    The alpha is taken off *before* quantising and pasted back after: handing
    the quantiser an RGBA image lets it spend palette entries distinguishing
    colours that differ only in an alpha the format cannot store, and the fully
    transparent pixels -- whose colour is arbitrary and frequently black -- drag
    a colour into the palette that nothing visible uses.
    """
    from PIL import Image

    rgba = Image.fromarray(np.ascontiguousarray(frame), "RGBA")
    try:
        alpha = rgba.getchannel("A")
        # 255 colours, not 256: one slot is reserved above.
        indexed = rgba.convert("RGB").convert(
            "P", palette=Image.ADAPTIVE, colors=TRANSPARENT_INDEX
        )
        indexed.paste(
            TRANSPARENT_INDEX, alpha.point(lambda a: 255 if a < ALPHA_THRESHOLD else 0)
        )
        indexed.info["transparency"] = TRANSPARENT_INDEX
        return indexed
    finally:
        rgba.close()


def tick_durations(durations_ms: Sequence[int]) -> list[int]:
    """Durations rounded to what a GIF can actually hold, never to zero.

    Rounded rather than truncated -- Pillow truncates, so 15 ms would become 10
    and a clip drawn at 66 fps would play a third fast -- and floored at one
    tick, because a zero-delay frame is left to the viewer to interpret and most
    of them pick 100 ms, which is the opposite of what a very short frame meant.
    """
    return [
        max(GIF_TICK_MS, int(round(float(ms) / GIF_TICK_MS)) * GIF_TICK_MS)
        for ms in durations_ms
    ]


def write_gif(
    path: Any,
    frames: Sequence[np.ndarray],
    durations_ms: Sequence[int],
    *,
    loop: bool = True,
) -> None:
    """Write one animated GIF. Off-thread: takes arrays, not a document."""
    if not frames:
        raise ValueError("a gif needs at least one frame")
    if len(frames) != len(durations_ms):
        raise ValueError("every frame needs a duration")
    shape = frames[0].shape[:2]
    if any(plane.shape[:2] != shape for plane in frames):
        raise ValueError("every frame of a gif is the same size")

    images = [quantise(plane) for plane in frames]
    try:
        images[0].save(
            path,
            "GIF",
            save_all=True,
            append_images=images[1:],
            duration=tick_durations(durations_ms),
            # 0 means forever; omitting the key entirely means play once.
            **({"loop": 0} if loop else {}),
            transparency=TRANSPARENT_INDEX,
            # Restore to the background before each frame. Without it a frame
            # whose subject has moved is drawn over the previous one and every
            # transparent pixel keeps showing what was under it, so a walk cycle
            # smears into a crowd.
            disposal=2,
            optimize=False,
        )
    finally:
        for image in images:
            image.close()
