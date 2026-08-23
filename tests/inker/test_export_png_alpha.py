"""What a flat PNG export actually puts where the user erased.

The gap this closes: nothing opened an exported PNG and asked whether a
hand-erased pixel came back transparent. ``Document.matte`` -- white for any
opaque image opened here, by ``matte_for`` -- is composited under the stack by
``flatten``, so every erased area exported as opaque white while the canvas
showed a checkerboard there. That is correct behaviour for a photo and it is
the documented intent; it was simply invisible, and until ``toggle_matte``
there was no way out of it.

End to end through ``inker_mode.export_png``, with the native save dialog
monkeypatched, because the defect lived in the whole chain rather than in
``flatten``: the composite the canvas draws, the matte the flatten consults and
the bytes the exporter writes are three different reads of the same document.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from warlock.studio import inker, inker_mode
from warlock.studio.inker_state import InkerDoc, InkerState


class _Ctx:
    """The four members ``export_png`` touches."""

    def __init__(self, state: Any) -> None:
        self.state = type("S", (), {"inker": state})()
        self.toasts: list[tuple[str, str]] = []
        self.run: Any = None

    def toast(self, message: str, kind: str = "info", **_kw: Any) -> None:
        self.toasts.append((message, kind))

    def submit(self, key: str, run: Any) -> bool:
        self.run = run
        return True


def _opaque_photo() -> Any:
    """What opening a JPEG gives: no alpha anywhere, so ``matte_for`` says white."""
    pixels = np.zeros((4, 4, 4), dtype=np.uint8)
    pixels[..., 0] = 200
    pixels[..., 3] = 255
    doc = inker.Document.from_pixels(pixels)
    assert doc.matte == (255, 255, 255, 255)
    return doc


def _erase_a_hole(doc: Any) -> None:
    """One transparent pixel. Written straight into the layer rather than
    through the eraser: which tool cut the alpha is not what is under test, and
    every one of them (the eraser, delete-selection, the alpha filters) leaves
    exactly this."""
    doc.stack.active.pixels[1, 1, 3] = 0
    doc.invalidate_all()


def _export(monkeypatch, tmp_path: Path, doc: Any) -> np.ndarray:
    from warlock.studio import dialogs

    dest = tmp_path / "out.png"
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: dest)
    tab = InkerDoc(doc=doc, title="photo.png")
    tab.path = None
    state = InkerState()
    state.add(tab)
    ctx = _Ctx(state)
    inker_mode.export_png(ctx, tab)
    assert ctx.run is not None
    assert ctx.run()["exported"] == dest
    with Image.open(dest) as im:
        return np.asarray(im.convert("RGBA"), dtype=np.uint8)


def test_the_matte_fills_an_erased_hole_with_white(monkeypatch, tmp_path):
    """The documented default, now asserted rather than assumed."""
    doc = _opaque_photo()
    _erase_a_hole(doc)

    out = _export(monkeypatch, tmp_path, doc)

    assert tuple(out[1, 1]) == (255, 255, 255, 255)


def test_turning_the_matte_off_exports_the_hole_transparent(monkeypatch, tmp_path):
    """The bug the control fixes: alpha 0 in the file, not white."""
    doc = _opaque_photo()
    _erase_a_hole(doc)
    assert doc.toggle_matte() is True
    assert doc.matte is None

    out = _export(monkeypatch, tmp_path, doc)

    assert int(out[1, 1, 3]) == 0
    # And nothing else moved: the untouched pixels are the photo.
    assert tuple(out[0, 0]) == (200, 0, 0, 255)


# --- removing a background means removing it ---------------------------------


def _photo(size=8):
    """An opaque flat image, the shape ``matte_for`` stamps white."""
    doc = inker.Document.blank(size, size)
    px = np.zeros((size, size, 4), np.uint8)
    px[..., 3] = 255
    px[..., :3] = 255
    px[3:5, 3:5, :3] = (200, 30, 30)
    doc.stack[0].pixels[...] = px
    doc.invalidate_all()
    doc.matte = inker.matte_for(doc.composite)
    assert doc.matte == (255, 255, 255, 255)
    return doc


def test_deleting_a_selection_clears_the_flatten_matte():
    """The reported bug. ``matte_for`` stamps white on any opaque import, and
    only the AI cutout and ``to_background`` used to clear it -- so selecting
    the white and pressing Delete cut alpha the export filled straight back in.
    The pixels really were transparent and the file really was white."""
    doc = _photo()
    doc.select_wand((0, 0), tolerance=8)
    assert doc.delete_selection()
    assert doc.matte is None
    flat = doc.flatten(matte=True)
    assert tuple(int(v) for v in flat[0, 0]) == (0, 0, 0, 0)
    assert tuple(int(v) for v in flat[3, 3]) == (200, 30, 30, 255), "the subject stays"


def test_the_eraser_clears_the_flatten_matte():
    doc = _photo(16)
    doc.begin_stroke((4, 4), (0, 0, 0, 0), size=6, mode="erase", nib="square", hardness=1.0)
    doc.stroke_to((10, 10))
    assert doc.end_stroke()
    assert doc.matte is None


def test_one_undo_puts_back_both_the_pixels_and_the_matte():
    """``CompoundEdit.undo`` walks ``reversed()``, so the ``MatteEdit`` is
    appended last and restores the colour before the pixels come back -- the
    ordering ``apply_matte`` learned the hard way, where an undone cutout
    restored the pixels and lost the colour for good."""
    doc = _photo()
    doc.select_wand((0, 0), tolerance=8)
    assert doc.delete_selection()
    assert doc.matte is None
    doc.undo()
    assert doc.matte == (255, 255, 255, 255)
    assert int(doc.stack[0].pixels[..., 3].min()) == 255
    assert tuple(int(v) for v in doc.flatten(matte=True)[0, 0]) == (255, 255, 255, 255)


def test_a_soft_paint_stroke_does_not_disturb_the_matte():
    """The test is opaque-to-*fully*-transparent, so an antialiased edge -- on
    every ordinary stroke -- leaves a photo's matte exactly where it was."""
    doc = _photo(16)
    doc.begin_stroke((4, 4), (10, 20, 30, 255), size=6, hardness=0.2)
    doc.stroke_to((10, 10))
    assert doc.end_stroke()
    assert doc.matte == (255, 255, 255, 255)


def test_a_background_document_is_left_alone():
    """``flatten`` does not consult the matte where there is a real background
    layer and ``_shown_pixels`` forces that layer opaque, so there is no hole
    to answer for -- the reason ``set_matte`` refuses there too."""
    doc = _photo()
    assert doc.to_background()
    assert doc.matte is None, "to_background folds it into pixels"
    doc.matte = (255, 255, 255, 255)
    head = doc.history.head
    doc.select_wand((0, 0), tolerance=8)
    doc.delete_selection()
    assert doc.matte == (255, 255, 255, 255)
    assert doc.history.head != head, "the cut itself is still a step"
