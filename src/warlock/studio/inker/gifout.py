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

That last paragraph is about an *adaptive* quantise, and it is exactly what an
indexed document does not want: the whole point of authoring against a palette
is that the file carries the table you chose. So ``write_gif`` takes an
optional ``palette``, and with one it skips the quantiser entirely -- every
frame gets that table, in that order, and a colour is written to the slot the
user put it in rather than to whichever slot a per-frame histogram happened to
give it. The pixels already sit exactly on those colours (``indexed.snap`` put
them there as they were painted), so this is a lookup rather than an
approximation, and the per-frame drift the paragraph above worries about cannot
arise.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

__all__ = [
    "ALPHA_THRESHOLD",
    "GIF_TICK_MS",
    "MAX_PALETTE",
    "TRANSPARENT_INDEX",
    "loop_option",
    "map_to_palette",
    "quantise",
    "write_gif",
]

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

#: How many document colours a GIF can carry. ``TRANSPARENT_INDEX`` takes the
#: 256th slot, so a palette longer than this cannot be written verbatim and the
#: caller falls back to the adaptive quantiser rather than silently truncating
#: the table -- a dropped swatch is pixels changing colour in the export.
MAX_PALETTE = TRANSPARENT_INDEX


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


def map_to_palette(
    frame: np.ndarray, palette: Sequence[Any], indices: np.ndarray | None = None
) -> Any:
    """One RGBA plane as a palette image using *palette* verbatim.

    The indexed path. Every pixel is looked up in the table rather than
    quantised against a histogram, so the exported colour table is the authored
    one and slot *n* means the same colour in every frame of the clip.

    ``nearest`` rather than an exact match even though the document is snapped:
    a frame handed here has been *flattened*, and flattening composites layer
    opacities and blend modes, which can land a pixel between two swatches. The
    nearest one is the only answer that keeps the export a picture of the
    document.

    ``indices`` short-circuits all of that. A truly indexed document already
    knows which slot each pixel is in, and a colour lookup is strictly worse
    than being told: two palette slots holding the same colour are one colour in
    a flattened plane, so the lookup has to pick one of them and the export
    stops being slot-stable. ``sheetout.index_plane_one`` supplies it only when
    the flatten *is* a cel's materialisation, so passing it is never a guess.
    Transparency still comes off the frame's alpha, because that is what decides
    which pixels the format draws at all.
    """
    from PIL import Image

    from . import indexed as ix

    table = [tuple(colour)[:3] for colour in palette]
    if not table or len(table) > MAX_PALETTE:
        raise ValueError(f"a gif palette holds 1..{MAX_PALETTE} colours")
    if indices is not None and indices.shape != frame.shape[:2]:
        raise ValueError("an index plane is the size of the frame it indexes")
    flat = np.ascontiguousarray(frame) if indices is not None else ix.snap(
        np.ascontiguousarray(frame), palette
    )
    if indices is not None:
        picks = np.minimum(indices, len(table) - 1).astype(np.uint8)
    else:
        rgb = flat[..., :3].reshape(-1, 3)
        # ``setdefault``, so a duplicate colour keeps the **lowest** slot rather
        # than whichever one the loop reached last. Deterministic for the same
        # reason ``index_plane.resolve`` is: an export that moved a pixel between
        # two identical swatches from one run to the next would make "two
        # exports of an unchanged document are byte-identical" false.
        lookup: dict[tuple[int, ...], int] = {}
        for index, colour in enumerate(table):
            lookup.setdefault(colour, index)
        picks = np.zeros(rgb.shape[0], dtype=np.uint8)
        for colour, index in lookup.items():
            picks[(rgb == np.asarray(colour, dtype=np.uint8)).all(axis=1)] = index
        picks = picks.reshape(flat.shape[:2])
    picks[flat[..., 3] < ALPHA_THRESHOLD] = TRANSPARENT_INDEX

    indexed = Image.fromarray(picks, "P")
    # 256 entries: the authored table, then padding, then transparency in the
    # last slot. Pillow wants a flat RGB triple list of exactly that length.
    raw: list[int] = []
    for colour in table:
        raw.extend(colour)
    raw.extend([0] * (3 * (256 - len(table))))
    indexed.putpalette(raw)
    indexed.info["transparency"] = TRANSPARENT_INDEX
    return indexed


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


def loop_option(loop: bool | int) -> dict[str, int]:
    """The ``loop=`` keyword Pillow wants, or nothing at all.

    **The netscape extension counts *additional* plays, not total ones.** That
    off-by-one is the whole reason this is a named function with a test rather
    than an expression at the call site: a tag set to play three times has to
    be written as ``loop=2``, and the block is emitted at all only when the
    file should play more than once.

    So: True is forever (``0``); False is play-once, which the format spells by
    *omitting* the block; a count of 1 is play-once too and omits it for the
    same reason; and a count of two or more asks for that many minus one.

    ``isinstance`` before any arithmetic, because ``True == 1`` in Python and
    the two mean opposite things here -- loop forever versus play exactly once.
    """
    if isinstance(loop, bool):
        return {"loop": 0} if loop else {}
    count = int(loop)
    return {"loop": count - 1} if count >= 2 else {}


def write_gif(
    path: Any,
    frames: Sequence[np.ndarray],
    durations_ms: Sequence[int],
    *,
    loop: bool | int = True,
    palette: Sequence[Any] | None = None,
    indices: Sequence[Any] | None = None,
    palettes: Sequence[Any] | None = None,
) -> None:
    """Write one animated GIF. Off-thread: takes arrays, not a document.

    With *palette*, the document's own colour table is written verbatim; see
    the module docstring. A palette too long for the format falls back to the
    adaptive quantiser rather than dropping swatches.

    *indices* is one exact index plane per frame, or None in the slots where
    there is not one -- a truly indexed document supplies them where the flatten
    is a cel's own materialisation, and a colour lookup is used everywhere else.
    Per frame rather than all-or-nothing, because a clip routinely has one frame
    with a blended layer on it and forty without.

    *palettes* is one **per-frame override** table, or None in the slots with
    no override -- ``animation.frame_palettes``, which a document grows the
    moment somebody gives one frame its own colours. A frame with an override
    is written with that table and the format's *local* colour table; the rest
    share *palette* as the global one. Without this every frame was resolved
    through the document's table, so an overridden frame exported the right
    slots in the wrong colours -- and GIF was the only exporter with that bug,
    since ``aseout`` and ``ora`` both round-trip the overrides.

    ``loop`` takes a tag's repeat count as well as a flag; see
    :func:`loop_option` for what each spelling writes.
    """
    if not frames:
        raise ValueError("a gif needs at least one frame")
    if len(frames) != len(durations_ms):
        raise ValueError("every frame needs a duration")
    shape = frames[0].shape[:2]
    if any(plane.shape[:2] != shape for plane in frames):
        raise ValueError("every frame of a gif is the same size")
    if indices is not None and len(indices) != len(frames):
        raise ValueError("every frame needs an index plane or None")
    if palettes is not None and len(palettes) != len(frames):
        raise ValueError("every frame needs a palette or None")

    planes = list(indices) if indices is not None else [None] * len(frames)
    tables = list(palettes) if palettes is not None else [None] * len(frames)
    images = []
    for plane, slots, own in zip(frames, planes, tables, strict=True):
        # The frame's own table wins, then the document's, then the adaptive
        # quantiser. A table too long for the format falls through rather than
        # dropping swatches, exactly as the global one always has.
        table = own or palette
        usable = bool(table) and len(table) <= MAX_PALETTE
        images.append(map_to_palette(plane, table, slots) if usable else quantise(plane))
    try:
        images[0].save(
            path,
            "GIF",
            save_all=True,
            append_images=images[1:],
            duration=tick_durations(durations_ms),
            # 0 means forever; omitting the key entirely means play once.
            **loop_option(loop),
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
