"""The reader's degradation contract, at the seams a corrupt member reaches.

Every optional member of an ``.ora`` fails to a default with a log line and
never takes the file with it -- the rule ``ora.py`` states per member. What is
pinned here are the two ways that contract used to leak: an exception class the
``except`` tuples did not carry (``AttributeError`` for a wrong-shaped payload,
``OSError`` for Pillow's ``UnidentifiedImageError`` on non-image bytes), and
the grayscale branch of ``_finish_colour`` dropping the palette the writer had
recorded.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from warlock.studio.inker import ora
from warlock.studio.inker.document import Document


def _rewrite_member(path: Path, member: str, data: bytes) -> None:
    """Replace one member's bytes, keeping every other member and the order."""
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        items = {name: zf.read(name) for name in names}
    items[member] = data
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in names:
            zf.writestr(name, items[name])


def _animated(tmp_path: Path) -> Path:
    doc = Document.blank(8, 8)
    doc.stack.active.pixels[0:2, 0:2] = (5, 6, 7, 255)
    doc.invalidate_all()
    doc.ensure_animation()
    doc.add_frame()
    path = tmp_path / "anim.ora"
    ora.write_ora(doc, path)
    return path


def test_a_wrong_shaped_animation_json_falls_back_flat(tmp_path: Path):
    """``tracks`` as a string iterates into characters and the first ``.get``
    raises ``AttributeError`` -- which the member's ``except`` tuple omitted,
    so a shape error crashed the open instead of costing the timeline."""
    path = _animated(tmp_path)
    with zipfile.ZipFile(path) as zf:
        payload = json.loads(zf.read(ora.ANIMATION_MEMBER))
    payload["tracks"] = "not-a-list"
    _rewrite_member(path, ora.ANIMATION_MEMBER, json.dumps(payload).encode("utf-8"))

    back = ora.read_ora(path)
    assert back.anim is None, "the grid fell back flat rather than raising"
    assert len(back.stack) >= 1


def test_a_cel_member_of_non_image_bytes_still_opens(tmp_path: Path):
    """Non-image bytes in a cel PNG are Pillow's ``UnidentifiedImageError``,
    an ``OSError`` outside the old tuple -- the open crashed on both the
    animated read and the flat fallback."""
    path = _animated(tmp_path)
    with zipfile.ZipFile(path) as zf:
        payload = json.loads(zf.read(ora.ANIMATION_MEMBER))
    member = payload["cels"][0]["data"]
    _rewrite_member(path, member, b"these are not png bytes")

    back = ora.read_ora(path)  # must not raise
    assert back is not None


def test_a_grayscale_document_keeps_its_palette_through_a_round_trip(tmp_path: Path):
    """The writer records ``palette.gpl`` for a grayscale document -- "a
    grayscale document with a palette gets both" is the funnel's own rule --
    but the reader's grayscale early-return dropped it, so the constraint was
    lost forever on the first save-open-save."""
    greys = [(0, 0, 0, 255), (128, 128, 128, 255), (255, 255, 255, 255)]
    doc = Document.blank(8, 8)
    doc.stack.active.pixels[0:2, 0:2] = (128, 128, 128, 255)
    doc.invalidate_all()
    assert doc.convert_to_grayscale()
    assert doc.set_palette(greys)
    path = tmp_path / "grey.ora"
    ora.write_ora(doc, path)

    back = ora.read_ora(path)
    assert back.color_mode == "grayscale"
    assert back.palette == greys
