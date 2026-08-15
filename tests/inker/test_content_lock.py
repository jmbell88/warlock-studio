"""The content lock: a layer that refuses every tool-level write.

The lock is only worth anything if *every* door agrees on it, so this file is
organised as one test per door rather than one test per feature -- a new write
path that forgets to ask is exactly the bug this is here to catch, and the
"legal" half below is the other side of the same statement: what a lock must
**not** cost you.

The refusal is at the ops themselves and undo writes raw pixels underneath
them, which is deliberate and asserted here: a lock switched on after an edit
must not wedge the history that already holds it.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
from PIL import Image

from warlock.studio import inker
from warlock.studio.inker import ora as inker_ora
from warlock.studio.inker.document import Document
from warlock.studio.inker.selection import SelectionMask

RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)


def _doc(*, locked: bool = True) -> Document:
    doc = Document.blank(8, 8)
    doc.stack.active.pixels[:, :] = RED
    doc.stack.active.locked = locked
    doc.invalidate_all()
    return doc


def _all(doc: Document) -> SelectionMask:
    return SelectionMask.full(*doc.size)


def _unchanged(doc: Document, before: np.ndarray) -> bool:
    return np.array_equal(doc.stack.active.pixels, before)


# --- every door -------------------------------------------------------------


def test_a_stroke_is_refused():
    doc = _doc()
    before = doc.stack.active.pixels.copy()
    head = doc.history.head

    assert doc.begin_stroke((4.0, 4.0), BLUE, size=4) is False
    doc.stroke_to((5.0, 5.0))
    assert doc.end_stroke() is False

    assert _unchanged(doc, before)
    assert doc.history.head == head


def test_a_stroke_on_a_locked_track_autovivifies_no_cel():
    """The refusal is *before* ``_ensure_active_cel``, or a brush-down on a
    locked track leaves a blank cel nothing may then write to -- and a document
    asking to be saved for a gesture that was refused."""
    doc = _doc(locked=False)
    doc.add_frame()
    doc.set_layer_props(locked=True)
    assert doc.anim is not None
    slots = len(doc.anim.cels)

    assert doc.begin_stroke((4.0, 4.0), BLUE, size=4) is False

    assert len(doc.anim.cels) == slots
    assert doc._pending_cels == []


def test_write_colour_is_refused_which_covers_fill_and_the_shapes():
    doc = _doc()
    before = doc.stack.active.pixels.copy()
    weight = np.ones((8, 8), dtype=np.float32)

    assert doc.write_colour((0, 0, 8, 8), BLUE, weight) is False
    assert doc.fill((4, 4), BLUE) is False
    assert doc.shape("rect", (1, 1), (6, 6), BLUE, 1, filled=True) is False

    assert _unchanged(doc, before)


def test_a_gradient_is_refused():
    doc = _doc()
    before = doc.stack.active.pixels.copy()
    assert doc.gradient((0.0, 0.0), (8.0, 8.0), BLUE, RED) is False
    assert _unchanged(doc, before)


def test_a_filter_session_never_opens():
    """Refused at ``begin_filter`` rather than at commit, so no preview is
    shown for a filter that could never be applied."""
    doc = _doc()
    assert doc.begin_filter() is None
    assert doc._filter is None
    assert doc.preview_filter("blur", radius=2.0) is False


def test_a_lift_is_refused():
    doc = _doc()
    before = doc.stack.active.pixels.copy()
    doc.select(_all(doc))
    assert doc.lift() is False
    assert doc.floating is None
    assert _unchanged(doc, before)


def test_begin_transform_is_refused_because_it_lifts():
    doc = _doc()
    assert doc.begin_transform() is False
    assert doc.floating is None


def test_delete_selection_is_refused():
    doc = _doc()
    before = doc.stack.active.pixels.copy()
    doc.select(_all(doc))
    assert doc.delete_selection() is False
    assert _unchanged(doc, before)


def test_a_paste_is_refused_but_paste_as_layer_is_not():
    doc = _doc(locked=False)
    doc.select(SelectionMask.from_rect(doc.size, (0, 0, 4, 4)))
    assert doc.copy()
    doc.set_layer_props(locked=True)

    assert doc.paste() is False
    assert doc.floating is None
    # The escape hatch: pasting *beside* a locked layer writes to a layer of
    # its own and stays legal.
    assert doc.paste_as_layer()
    assert len(doc.stack) == 2


def test_merge_down_is_refused_for_either_participant():
    for locked_index in (0, 1):
        doc = Document.blank(8, 8)
        doc.stack[0].pixels[:, :] = RED
        doc.add_layer("Top")
        doc.stack[1].pixels[0:4, 0:4] = BLUE
        doc.stack[locked_index].locked = True
        doc.stack.active_index = 1

        assert doc.merge_down() is False
        assert len(doc.stack) == 2


# --- what a lock must not cost you ------------------------------------------


def test_copying_off_a_locked_layer_is_legal():
    """Reading pixels changes nothing, so there is nothing to refuse."""
    doc = _doc()
    doc.select(_all(doc))
    assert doc.copy()
    assert doc.clipboard.pixels is not None


def test_committing_a_float_lands_after_the_lock_goes_on():
    """A float outlives lock toggles, and every save commits it first -- so
    refusing here would wedge the document rather than protect it."""
    doc = _doc(locked=False)
    doc.select(SelectionMask.from_rect(doc.size, (0, 0, 4, 4)))
    assert doc.lift()
    doc.move_floating(2, 2)
    doc.set_layer_props(locked=True)

    assert doc.commit_floating()
    assert doc.floating is None
    assert tuple(doc.stack.active.pixels[3, 3]) == RED


def test_managing_a_locked_layer_stays_legal():
    doc = _doc()
    doc.add_layer("Top")
    doc.stack.active_index = 0

    assert doc.set_layer_props(0, name="Renamed")
    assert doc.set_layer_props(0, visible=False)
    assert doc.move_layer(0, 1)
    assert doc.remove_layer(1)
    assert len(doc.stack) == 1


def test_document_scope_operations_apply_regardless():
    """Geometry, the matte and flatten are statements about the whole document;
    one locked layer vetoing a crop is a document that cannot be worked on."""
    doc = _doc()
    doc.rotate90()
    assert doc.size == (8, 8)
    doc.scale((4, 4))
    assert doc.size == (4, 4)
    assert doc.crop((0, 0, 2, 2))
    assert doc.apply_matte(np.zeros((2, 2), dtype=np.uint8))
    assert int(doc.stack[0].pixels[..., 3].max()) == 0


def test_flatten_ignores_the_lock():
    doc = _doc()
    doc.add_layer("Top")
    doc.stack[1].pixels[0:4, 0:4] = BLUE
    doc.flatten_layers()
    assert len(doc.stack) == 1


# --- undo lives below the doors ---------------------------------------------


def test_undo_still_restores_a_layer_locked_after_the_edit():
    """Patches write raw pixels underneath the doors on purpose: a lock is not
    a reason to wedge history that already holds an edit."""
    doc = _doc(locked=False)
    weight = np.ones((4, 4), dtype=np.float32)
    assert doc.write_colour((0, 0, 4, 4), BLUE, weight)
    assert tuple(doc.stack.active.pixels[1, 1]) == BLUE

    # Set on the layer rather than through ``set_layer_props``, so the write is
    # still the top of the stack and this is a test about the *write*.
    doc.stack.active.locked = True
    assert doc.undo()
    assert tuple(doc.stack.active.pixels[1, 1]) == RED
    assert doc.stack.active.locked

    assert doc.redo()
    assert tuple(doc.stack.active.pixels[1, 1]) == BLUE


# --- the property travels ---------------------------------------------------


def test_a_copy_carries_the_lock():
    doc = _doc()
    assert doc.stack.active.copy().locked


def test_the_track_is_authoritative_on_an_animated_document():
    doc = _doc(locked=False)
    doc.add_frame()
    doc.set_layer_props(locked=True)
    assert doc.anim is not None
    assert doc.anim.tracks[0].locked
    # The materialised cel and an empty slot's placeholder both carry it down,
    # which is what makes ``write_locked`` reading the layer read the track.
    assert doc.stack[0].locked
    assert doc.anim.placeholder(doc.anim.tracks[0], doc.anim.frames[0], doc.size).locked


def test_an_autovivified_cel_carries_the_lock():
    """The sixth property in ``_ensure_cel_for``'s copy. Reached by unlocking,
    autovivifying and reading it back -- a cel built with the wrong properties
    is only visible one line later."""
    doc = _doc(locked=False)
    doc.add_frame()
    doc.add_layer("Ink")
    doc.set_layer_props(locked=False)
    doc._ensure_active_cel()
    assert doc.stack.active.locked is False

    doc.set_layer_props(locked=True)
    assert doc.stack.active.locked is True


# --- the alpha lock's own gap ------------------------------------------------


def test_a_gradient_honours_the_alpha_lock():
    """The one write path that did not. ``gradient`` sets alpha from the ramp's
    own coverage, so "preserve transparency" painted straight past the edge of
    the shape it was meant to stay inside."""
    doc = Document.blank(8, 8)
    layer = doc.stack.active
    layer.pixels[2:6, 2:6] = RED
    layer.alpha_lock = True
    before = layer.pixels[..., 3].copy()

    assert doc.gradient((0.0, 0.0), (8.0, 0.0), BLUE, RED)

    assert np.array_equal(layer.pixels[..., 3], before)
    # And the colours really did change where there was something to change.
    assert tuple(layer.pixels[3, 3]) != RED


# --- persistence -------------------------------------------------------------


def test_the_attribute_is_written_only_when_set(tmp_path: Path):
    """Byte-identity: a document nobody has locked anything in must produce the
    XML this writer produced before the attribute existed."""
    doc = _doc(locked=False)
    path = tmp_path / "plain.ora"
    inker.write_ora(doc, path)
    with zipfile.ZipFile(path) as zf:
        xml = zf.read("stack.xml").decode("utf-8")
    assert inker_ora.CONTENT_LOCK_ATTR not in xml


def test_a_lock_round_trips_through_a_still_ora(tmp_path: Path):
    doc = _doc()
    doc.add_layer("Top")
    doc.stack[1].locked = False
    path = tmp_path / "locked.ora"
    inker.write_ora(doc, path)

    with zipfile.ZipFile(path) as zf:
        xml = zf.read("stack.xml").decode("utf-8")
    assert inker_ora.CONTENT_LOCK_ATTR in xml

    back = inker.Document.load(path)
    assert [layer.locked for layer in back.stack] == [True, False]


def test_the_two_locks_are_independent_through_a_round_trip(tmp_path: Path):
    doc = Document.blank(8, 8)
    doc.stack[0].alpha_lock = True
    doc.add_layer("Top")
    doc.stack[1].locked = True
    path = tmp_path / "both.ora"
    inker.write_ora(doc, path)

    back = inker.Document.load(path)
    assert [(layer.alpha_lock, layer.locked) for layer in back.stack] == [
        (True, False),
        (False, True),
    ]


def test_a_lock_round_trips_through_an_animated_ora(tmp_path: Path):
    doc = _doc(locked=False)
    doc.add_frame(copy=True)
    doc.set_layer_props(0, locked=True)
    path = tmp_path / "anim.ora"
    inker.write_ora(doc, path)

    back = inker.Document.load(path)
    assert back.anim is not None
    assert back.anim.tracks[0].locked
    assert back.stack[0].locked

    with zipfile.ZipFile(path) as zf:
        xml = zf.read("stack.xml").decode("utf-8")
    assert inker_ora.CONTENT_LOCK_ATTR in xml


def test_the_animated_version_stays_one(tmp_path: Path):
    """Adding a ``.get``-based key does not qualify for a version bump, and an
    older build reading this file must simply see an unlocked track."""
    import json

    doc = _doc(locked=True)
    doc.add_frame()
    path = tmp_path / "v.ora"
    inker.write_ora(doc, path)
    with zipfile.ZipFile(path) as zf:
        payload = json.loads(zf.read(inker_ora.ANIMATION_MEMBER))
    assert payload["version"] == 1
    assert payload["tracks"][0]["locked"] is True


def test_a_foreign_ora_opens_unlocked(tmp_path: Path):
    """The negative control. Krita and GIMP write neither attribute, so every
    layer in a foreign file has to open editable -- a reader that defaulted the
    other way would hand the user a document nothing works on."""
    path = tmp_path / "foreign.ora"
    pixels = np.zeros((8, 8, 4), dtype=np.uint8)
    pixels[:, :] = RED
    buf = io.BytesIO()
    Image.fromarray(pixels, "RGBA").save(buf, "PNG")
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", b"image/openraster")
        zf.writestr(
            "stack.xml",
            '<image version="0.0.3" w="8" h="8"><stack>'
            '<layer name="L" src="data/a.png" locked="1" edit-locked="true"/>'
            "</stack></image>",
        )
        zf.writestr("data/a.png", buf.getvalue())

    doc = inker.Document.load(path)
    assert doc.stack[0].locked is False
    assert doc.write_locked() is False
    # And it is genuinely editable, not merely reported as such.
    assert doc.write_colour((0, 0, 4, 4), BLUE, np.ones((4, 4), dtype=np.float32))


def test_the_lock_survives_a_stack_xml_reread(tmp_path: Path):
    """The attribute is ours and prefixed to say so, and a hyphen rather than a
    colon -- an undeclared namespace prefix makes the whole file unparseable."""
    doc = _doc()
    path = tmp_path / "parse.ora"
    inker.write_ora(doc, path)
    with zipfile.ZipFile(path) as zf:
        root = ElementTree.fromstring(zf.read("stack.xml"))
    layer = root.find("stack").find("layer")
    assert layer.get(inker_ora.CONTENT_LOCK_ATTR) == "1"
