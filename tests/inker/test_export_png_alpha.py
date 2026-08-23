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
