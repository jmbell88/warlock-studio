"""Palette files in and out, and the colour mode the document is in.

One place for the four spellings a palette arrives and leaves in -- a JASC
.pal, a GIMP .gpl, an image the colours are read out of, and an image they are
written into -- beside set_color_mode, which is what decides whether the table
is a *constraint* or a storage change.

Lifted out of ``studio/inker_mode`` on 2026-09-04 (T7 of the 2026-09-02
review), after every behavioural finding that touches it was closed, so the
move is code motion over tested behaviour rather than a rewrite.

``inker_mode`` is imported as a *module* and never ``from``-imported: every
attribute is resolved at call time, so this file and its parent may be
imported in either order. The parent serves these names back through a PEP
562 ``__getattr__``, which is what keeps ``inker_mode.export_png`` and the
rest working for every caller and every test.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import atomic, dialogs, inker_mode, sizeguard
from .inker import gpl


def _write_palette(path: Any, colours: list[tuple[int, int, int, int]], name: str) -> None:
    """Write *colours* in the format the chosen filename asks for.

    The suffix decides, and a name with no suffix at all gets ``.gpl`` -- the
    picker's filter list does not tell us which entry was selected, and the
    filename is the only thing the user actually said.

    ``newline=""`` is load-bearing on Windows, not tidiness. Python's text mode
    defaults to ``newline=None``, which rewrites every line feed it is handed as
    ``os.linesep`` -- so ``dumps_jasc``'s already-correct CRLF reached the disk
    as CR CR LF. Our own ``parse_jasc`` reads that file back perfectly, because
    ``splitlines`` shrugs at anything, which is precisely the trap: the only
    reason to write this format at all is the strict third-party readers that
    will not. Line endings are the serialiser's decision and are made once, in
    ``gpl``; this call writes exactly what it was handed.
    """

    dest = Path(path)
    if dest.suffix.lower() not in inker_mode.PALETTE_SUFFIXES:
        dest = dest.with_suffix(".gpl")
    atomic.write_text(
        dest, gpl.dumps_for(dest.suffix, colours, name), encoding="utf-8", newline=""
    )


def _palette_text(path: Path) -> str:
    """One palette file's text, under a ceiling. Task thread only.

    A ``.gpl`` is a few hundred lines of "R G B name" and both readers of one
    went straight to ``read_text`` -- which decodes the whole file into a
    Python string before anything asks how big it was. ``MAX_UPLOAD_BYTES``
    rather than Inker's own document ceiling: this is a row of swatches, not a
    drawing, and twenty megabytes of it is already four orders past any palette
    anyone has.
    """
    from ..service.files import MAX_UPLOAD_BYTES

    return sizeguard.within_ceiling(path, MAX_UPLOAD_BYTES).read_text(
        encoding="utf-8", errors="replace"
    )


def import_palette(ctx: Any) -> None:

    inker_mode.ensure(ctx)

    def run() -> list[tuple[int, int, int, int]] | None:
        path = dialogs.open_file("Import a palette", inker_mode.PALETTE_FILTER)
        if path is None:
            return None
        return gpl.parse_any(_palette_text(path))

    ctx.submit("inker-palette", run)


def export_palette(ctx: Any) -> None:
    state = inker_mode.ensure(ctx)
    colours = list(state.swatches)

    def run() -> str | None:
        path = dialogs.save_file("Export the palette", "palette.gpl", inker_mode.PALETTE_FILTER)
        if path is None:
            return None
        _write_palette(path, colours, "Warlock")
        return str(path)

    # A key of its own. It used to share ``inker-palette-export`` with
    # ``export_document_palette``, and ``tasks.submit`` refuses a duplicate key
    # -- so whichever picker was already up made the other command do nothing at
    # all, silently, because neither call site reads the bool it answers.
    ctx.submit("inker-palette-export", run)


# --- indexed colour -----------------------------------------------------------
#
# The palette belongs to the *document* -- it is saved with the file and it is
# what every write snaps to -- so everything here takes a tab and goes through
# ``Document``. The swatch row above is a different thing and stays one: a
# session's favourite colours, persisted in settings, no bearing on any file.
#
# All of these run **inline on the frame thread**, gated on ``tab.busy``, which
# is exactly what the canvas geometry ops in ``panes/inker_bridge`` do and for
# the same reason: they rebind whole layer planes, so one landing mid-save
# writes an archive whose parts disagree. The cost is the same class as a
# rotate, and ``indexed.snap`` works over the region's *distinct* colours
# rather than its pixels, which is what keeps a 40-frame clip inside a frame.


#: What the mode picker offers, in the order it draws them.
COLOR_MODES = ("rgb", "indexed", "grayscale")

#: How each mode is written on a button.
COLOR_MODE_LABELS = {"rgb": "RGB", "indexed": "Indexed", "grayscale": "Grayscale"}


def set_color_mode(
    ctx: Any, tab: Any, mode: str, *, max_colours: int = 32, method: str = "nearest"
) -> bool:
    """Move a document between RGB, true indexed and grayscale. -> whether it moved.

    One door for all three, because they are one question and because the
    refusals belong together: each conversion is a whole-document rewrite, one
    undo step, and inline on the frame thread for ``index_to``'s reason (a
    partial rewrite landing mid-save writes an archive whose parts disagree).

    Entering **indexed** with no palette builds one from the drawing's own
    colours, exactly as ``palette_from_document`` does for constrained mode --
    two published operations and no third one. Entering it *with* a palette
    keeps the table the user authored.

    ``method`` is one of ``dither.METHODS``. It defaulted to ``"nearest"`` and
    was not a parameter at all, which made the Convert popup's matrix reachable
    for a *snap onto a palette* and unreachable for the one operation that
    changes mode -- so "convert to indexed" was the only conversion in the app
    with no dither. The pane routes the popup's choice through here now.
    """
    if tab is None or tab.busy or mode not in COLOR_MODES:
        return False
    state = inker_mode.ensure(ctx)
    doc = tab.doc
    if doc.color_mode == mode:
        return False
    try:
        if mode == "indexed":
            moved = doc.convert_to_indexed(
                doc.palette or None, method, max_colours=max_colours
            )
        elif mode == "grayscale":
            moved = doc.convert_to_grayscale()
        else:
            moved = doc.convert_to_rgb()
    except ValueError as exc:
        # By name, and with the attempt in front of it: every refusal this can
        # raise is about the palette the user can see (too many colours, a
        # transparent index naming no slot), and a silent False would leave a
        # button that does nothing. The frame is the house rule -- library text
        # with no subject makes the reader work out what was being tried.
        ctx.toast(f"Cannot switch to {COLOR_MODE_LABELS[mode]}: {exc}.", "warn")
        return False
    if not moved:
        return False
    state.palette_slot = 0
    state.palette_slots = []
    state.palette_usage = None
    state.fg_slot = None
    if mode == "indexed":
        ctx.toast(
            f"Indexed: {len(doc.palette)} colours, slot {doc.transparent_index}"
            " is transparent.",
            "success",
        )
    elif mode == "grayscale":
        ctx.toast("Grayscale. Every write lands on a grey from here.", "success")
    else:
        # Worth saying, because the pixels do not move: leaving a mode lifts a
        # constraint, and the drawing looks exactly as it did a moment ago.
        ctx.toast("RGB colour. The pixels are unchanged.")
    return True


def set_transparent_slot(ctx: Any, tab: Any, index: int) -> bool:
    """Move which palette slot means "hole". Indexed documents only."""
    if tab is None or tab.busy or not tab.doc.set_transparent_index(index):
        return False
    inker_mode.ensure(ctx).palette_usage = None
    ctx.toast(f"Slot {index} is transparent now.", "success")
    return True


def index_to(ctx: Any, tab: Any, colours: Any) -> bool:
    """Make *tab* indexed against *colours*, or plain RGBA with ``None``."""
    if tab is None or tab.busy:
        return False
    state = inker_mode.ensure(ctx)
    if not tab.doc.set_palette(colours):
        return False
    state.palette_slot = 0
    state.palette_slots = []
    state.palette_usage = None
    # ``set_color_mode`` clears it too, and for the same reason: the slot the
    # brush was claiming indexes a table that has just been replaced, so left
    # standing it would land the next stroke in whatever colour inherited
    # that number.
    state.fg_slot = None
    if colours:
        ctx.toast(f"Indexed to {len(list(colours))} colour(s).", "success")
    else:
        # Worth saying, because nothing on the canvas changes: leaving indexed
        # mode lifts the constraint and repaints nothing.
        ctx.toast("Indexed colour off. The pixels are unchanged.")
    return True


def import_document_palette(ctx: Any) -> None:
    """Open a ``.gpl`` and index the inker_mode.active document to it.

    A second task key from ``import_palette``'s, because they are different
    acts on different subjects -- one adds to the session's swatch row, the
    other rewrites every pixel of a file -- and sharing a key would let the
    landing handler guess wrong about which one came back.
    """

    inker_mode.ensure(ctx)
    tab = inker_mode.active(ctx)
    if tab is None or tab.busy:
        return

    def run() -> list[tuple[int, int, int, int]] | None:
        path = dialogs.open_file("Index to a palette", inker_mode.PALETTE_FILTER)
        if path is None:
            return None
        return gpl.parse_any(_palette_text(path))

    ctx.submit(f"inker-index:{tab.uid}", run)


#: The ceiling on a palette read out of an image. The GIF colour table's own
#: limit and the largest number of swatches any of this is useful at -- a
#: photograph has tens of thousands of distinct colours, so an image import
#: *always* median-cuts unless it was pixel art already.
IMAGE_PALETTE_MAX = 256


def palette_from_image(ctx: Any) -> None:
    """Read a palette out of any image and index the inker_mode.active document to it.

    Never refuses on colour count: an image with more colours than the ceiling
    is median-cut down to it and the toast says so. Refusing would mean the
    command works on pixel art and fails on every photograph, which is the half
    of its input the user is least able to predict.

    The decode is on the task thread with the picker, for the reason every
    dialog in this module is: a native picker is modal to the OS, and a JPEG the
    size of a phone photo is not a frame's worth of work either.
    """
    from .inker import dither

    inker_mode.ensure(ctx)
    tab = inker_mode.active(ctx)
    if tab is None or tab.busy:
        return

    def run() -> dict[str, Any] | None:
        path = dialogs.open_file("Palette from an image", inker_mode.OPEN_FILTER)
        if path is None:
            return None

        from . import pixelguard

        # A picked file, so the same ceiling as the sheet import above.
        pixels = pixelguard.decode_rgba(path, Path(path).name)
        # Counted through ``np.unique`` on a packed uint32 rather than through a
        # set of tuples: a phone photo is twelve million pixels, and the set
        # costs a gigabyte to answer one number.
        rgb = pixels[..., :3][pixels[..., 3] > 0]
        packed = (
            rgb[:, 0].astype(np.uint32) << 16 | rgb[:, 1].astype(np.uint32) << 8 | rgb[:, 2]
        )
        distinct = int(np.unique(packed).size)
        return {
            "colours": dither.build_palette([pixels], IMAGE_PALETTE_MAX),
            "distinct": distinct,
        }

    ctx.submit(f"inker-palimg:{tab.uid}", run)


def export_document_palette(ctx: Any) -> None:
    """Write the *document's* table out as a ``.gpl`` or a JASC ``.pal``."""
    tab = inker_mode.active(ctx)
    if tab is None or not tab.doc.palette:
        return
    colours = [tuple(c) for c in tab.doc.palette]
    stem = tab.path.stem if tab.path else "palette"

    def run() -> str | None:
        path = dialogs.save_file(
            "Export the document palette", f"{stem}.gpl", inker_mode.PALETTE_FILTER
        )
        if path is None:
            return None
        _write_palette(path, colours, stem)
        return str(path)

    # Not ``inker-palette-export``: see the sibling above for why sharing it
    # made one of the two commands inert whenever the other was open.
    ctx.submit("inker-palette-export-doc", run)


#: One swatch of an exported palette strip, in real pixels. Bigger than the one
#: pixel per colour some tools write, because the file is looked at as often as
#: it is read back: at 1px a sixteen-colour palette is a 16x1 image that every
#: viewer renders as a smear. ``palette_from_image`` reads the strip back to
#: exactly the same colours either way -- it works over the *distinct* colours
#: in the image, not over its pixels -- so the size costs nothing but bytes.
PALETTE_STRIP_CELL = 16


def palette_strip(colours: Sequence[tuple[int, int, int, int]], cell: int) -> Any:
    """A one-row swatch strip for *colours* as an RGBA array. Pure.

    Separate from the command below and taking no ``ctx`` so that the pixels
    are testable without a picker: this is the half that can be wrong, and the
    half a round trip through ``palette_from_image`` has to hold for.
    """

    side = max(1, int(cell))
    row = np.asarray([tuple(c)[:4] for c in colours], dtype=np.uint8)
    if not row.size:
        raise ValueError("there is nothing to write")
    return np.repeat(np.repeat(row[None, :, :], side, axis=0), side, axis=1)


def export_palette_image(ctx: Any) -> None:
    """Write the document's palette out as a PNG swatch strip.

    The mirror of :func:`palette_from_image`, and the reason it is worth having
    at all: a palette that can only arrive as a picture and never leave as one
    is a one-way door. It is also how a palette reaches a tool that reads no
    palette format -- every image editor there is opens a PNG.

    The *pixels* are built on the frame thread and the picker is not, which is
    ``save_as``' rule: an array built after an unbounded modal is an array of
    whatever the user changed while it was up.
    """
    tab = inker_mode.active(ctx)
    if tab is None or not tab.doc.palette:
        return
    strip = palette_strip(
        [tuple(c) for c in tab.doc.palette], PALETTE_STRIP_CELL
    )
    stem = tab.path.stem if tab.path else "palette"

    def run() -> str | None:
        path = dialogs.save_file(
            "Export the palette as an image", f"{stem}-palette.png", inker_mode.PNG_FILTER
        )
        if path is None:
            return None
        from PIL import Image

        dest = Path(path)
        if dest.suffix.lower() != ".png":
            dest = dest.with_suffix(".png")
        # Through ``atomic`` like every other write onto a path the user named:
        # a half-written PNG over a file they already had is the one outcome an
        # export may not produce.
        atomic.save_image(dest, Image.fromarray(strip, "RGBA"), "PNG")
        return str(dest)

    # The palette-export keys' rule: one key per command, because ``submit``
    # refuses a duplicate and a shared key makes whichever picker is already up
    # silently swallow the other command.
    ctx.submit("inker-palette-export-image", run)
