"""A real background layer, reference layers, and solo. Wave 6.5.

**This package re-argues divergence #6 rather than working around it.** The
matte was a colour composited under the stack at flatten time: invisible to the
layer model, and therefore dropped by every writer -- an `.aseprite` written
here said nothing about it because there was nothing in the model to say. A
flag on the bottom layer is what Aseprite has, what the file format carries
(layer flag 0x08), and a thing the user can paint on.

What the flag means is one sentence, and it is worth stating because it is
*not* what a naive reading gives: **a background layer is opaque at the
composite**, not at the write doors. Erasing on it reveals the colour under the
eraser rather than a hole, and the pixels the user painted are still the pixels
in the file -- so unmarking the layer gives back exactly the drawing that was
there.
"""

from __future__ import annotations

import numpy as np

from warlock.studio import inker


def _doc(**kw):
    return inker.Document.blank(8, 8, **kw)


def test_a_fresh_document_has_no_background_layer():
    assert _doc().has_background is False


def test_converting_folds_the_matte_into_the_pixels_and_clears_it():
    """The stand-in becomes a real layer, which is the whole re-argument."""

    doc = _doc(matte=(10, 20, 30, 255))
    assert doc.to_background() is True
    assert doc.has_background is True
    assert doc.matte is None
    assert tuple(int(c) for c in doc.stack[0].pixels[0, 0]) == (10, 20, 30, 255)


def test_a_background_layer_composites_opaque():
    doc = _doc()
    doc.to_background()
    doc.stack[0].pixels[...] = (0, 0, 0, 0)
    doc.invalidate_all()
    assert int(doc.flatten()[0, 0, 3]) == 255


def test_the_matte_is_not_composited_under_a_background():
    """A document with an opaque bottom has nothing for a second colour to
    show through, and compositing one would be a colour nothing can display."""

    doc = _doc(matte=(255, 0, 0, 255))
    doc.to_background()
    doc.matte = (0, 255, 0, 255)  # as a stale file might carry
    assert tuple(int(c) for c in doc.flatten()[0, 0]) != (0, 255, 0, 255)


def test_the_flag_changes_the_composite_and_not_the_pixels():
    doc = _doc()
    doc.stack[0].pixels[2, 2] = (9, 9, 9, 128)
    before = doc.stack[0].pixels.copy()
    doc.to_background()
    doc.from_background()
    assert doc.has_background is False
    # The conversion folded a matte in (there is none here), so what comes back
    # is what was painted.
    assert np.array_equal(doc.stack[0].pixels[2, 2, :3], before[2, 2, :3])


def test_converting_twice_is_refused_rather_than_repeated():
    doc = _doc()
    assert doc.to_background() is True
    assert doc.to_background() is False
    assert doc.from_background() is True
    assert doc.from_background() is False


def test_a_reference_layer_refuses_every_tool_write():
    """The content lock plus a statement of intent: ``write_locked`` reads it,
    so no door has to be taught about it one at a time."""

    doc = _doc()
    doc.set_reference(0, True)
    assert doc.write_locked() is True
    assert doc.begin_stroke((4.0, 4.0), (255, 0, 0, 255)) is False
    doc.set_reference(0, False)
    assert doc.write_locked() is False


def test_solo_hides_everything_else_and_then_brings_it_back():
    doc = _doc()
    doc.add_layer()
    doc.add_layer()
    assert doc.solo(1) is True
    assert [layer.visible for layer in doc.stack] == [False, True, False]
    assert doc.solo(1) is True
    assert [layer.visible for layer in doc.stack] == [True, True, True]


def test_solo_writes_the_ordinary_visibility_flags():
    """Deliberately not a mode: a second visibility state would have to be
    learnt by the compositor, every export path and the ORA writer."""

    import inspect

    source = inspect.getsource(inker.Document.solo)
    assert "set_layer_props" in source
    assert "solo_mode" not in source


def test_both_flags_survive_an_ora_round_trip(tmp_path):
    from warlock.studio.inker import ora

    doc = _doc()
    doc.add_layer()
    doc.to_background()
    doc.set_reference(1, True)
    path = tmp_path / "layers.ora"
    ora.write_ora(doc, path)
    again = ora.read_ora(path)
    assert again.stack[0].background is True
    assert again.stack[1].reference is True


def test_an_ordinary_document_writes_the_xml_it_always_did(tmp_path):
    """Written only when set, the content lock's rule and its reason."""

    import zipfile

    from warlock.studio.inker import ora

    path = tmp_path / "plain.ora"
    ora.write_ora(_doc(), path)
    with zipfile.ZipFile(path) as archive:
        xml = archive.read("stack.xml").decode("utf-8")
    assert "warlock-background" not in xml
    assert "warlock-reference" not in xml


# --- 6.8: trim, the 45-degree mirrors, grid from selection -------------------


def test_trim_crops_to_what_is_drawn():
    doc = inker.Document.blank(16, 16)
    doc.stack.active.pixels[4:8, 4:8] = (255, 0, 0, 255)
    doc.invalidate_all()
    assert doc.trim() is True
    assert doc.size == (4, 4)


def test_trimming_an_empty_document_leaves_it_alone():
    """An empty canvas is a size the user chose, and a trim that turned it
    into 1x1 would be a command that destroys a setting."""

    doc = inker.Document.blank(16, 16)
    assert doc.trim() is False
    assert doc.size == (16, 16)


def test_trimming_a_full_document_is_a_no_op_rather_than_a_step():
    doc = inker.Document.blank(8, 8)
    doc.stack.active.pixels[...] = (255, 0, 0, 255)
    doc.invalidate_all()
    head = doc.history.head
    assert doc.trim() is False
    assert doc.history.head == head


def test_the_diagonal_mirrors_reflect_about_the_forty_five_degree_lines():
    """Neither is expressible as a combination of the axis-aligned pair, which
    is why they are two more modes rather than a flag on the existing ones."""

    from warlock.studio.inker import brush

    size = (9, 9)
    axis = (4.0, 4.0)
    (_p, mirrored) = brush._mirror((6.0, 5.0), size, "diag", axis=axis)
    assert mirrored == (5.0, 6.0)
    (_p, anti) = brush._mirror((6.0, 5.0), size, "anti", axis=axis)
    assert anti == (3.0, 2.0)


def test_every_symmetry_mode_reflects_the_axis_onto_itself():
    from warlock.studio.inker import brush

    for mode in brush.SYMMETRY:
        points = brush._mirror((4.0, 4.0), (9, 9), mode, axis=(4.0, 4.0))
        assert all(point == (4.0, 4.0) for point in points), mode
