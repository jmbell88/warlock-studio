"""Per-cel opacity: the value a *slot* carries on top of its track's.

Divergence 1 (``docs/INVARIANTS.md``) used to read "opacity is a Track property;
per-cel skipped", and this file is what retired it on 2026-08-30.

The whole design turns on one fact, and most of the tests below are about it: a
**linked cel is two keys mapping to one ``Layer`` object**. That sharing is what
makes the import worth more than a folder of PNGs, and it is also why the value
cannot live on the ``Layer`` -- two slots may legitimately want two different
opacities, and Aseprite's own format agrees, giving every cel chunk (linked ones
included) its own opacity byte. So it is a dict on ``Animation``, keyed exactly
like ``cels``, folded down in ``layers_for`` and nowhere else.

The standing negative control for the whole track is here too: a document that
never uses the feature must be byte-for-byte the document it was.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.inker import asein, aseout
from warlock.studio.inker.document import Document


def _animated(frames: int = 2) -> Document:
    doc = Document.blank(4, 4)
    doc.stack[0].name = "Art"
    doc.stack[0].pixels[:, :] = (255, 0, 0, 255)
    doc.invalidate_all()
    doc.ensure_animation()
    for _ in range(frames - 1):
        doc.add_frame(link=True)
    doc.set_current_frame(0)
    return doc


def _slot(doc, ti=0, fi=0):
    return doc.anim.tracks[ti].uid, doc.anim.frames[fi].uid


# --- the model ---------------------------------------------------------------


def test_a_fresh_grid_names_no_slot_and_every_slot_reads_one():
    doc = _animated()
    assert doc.anim.cel_opacity == {}
    assert doc.anim.cel_alpha(*_slot(doc, 0, 0)) == 1.0


def test_a_cel_opacity_multiplies_the_tracks_rather_than_replacing_it():
    """The difference the feature is: the row still says what the row says."""
    doc = _animated()
    doc.set_layer_props(0, opacity=0.5)
    assert doc.set_cel_opacity(0.5, track_index=0, frame_index=0)
    doc.set_current_frame(0)
    assert doc.stack[0].opacity == pytest.approx(0.25)
    assert doc.anim.tracks[0].opacity == 0.5


def test_setting_one_back_to_full_removes_the_key_rather_than_storing_one():
    """Sparse, because both writers ask whether a slot is *named* here."""
    doc = _animated()
    doc.set_cel_opacity(0.4, track_index=0, frame_index=0)
    assert doc.anim.cel_opacity
    doc.set_cel_opacity(1.0, track_index=0, frame_index=0)
    assert doc.anim.cel_opacity == {}


def test_an_out_of_range_value_is_clamped_rather_than_refused():
    doc = _animated()
    doc.anim.cel_opacity[_slot(doc)] = 1.4
    assert doc.anim.cel_alpha(*_slot(doc)) == 1.0
    doc.anim.cel_opacity[_slot(doc)] = -2.0
    assert doc.anim.cel_alpha(*_slot(doc)) == 0.0


def test_an_empty_slot_is_refused_rather_than_given_a_number_nothing_draws():
    doc = _animated()
    doc.clear_cel(track_index=0, frame_index=1)
    assert not doc.set_cel_opacity(0.5, track_index=0, frame_index=1)
    assert doc.anim.cel_opacity == {}


def test_setting_the_value_it_already_has_pushes_nothing():
    doc = _animated()
    before = len(doc.history)
    assert not doc.set_cel_opacity(1.0, track_index=0, frame_index=0)
    assert len(doc.history) == before


# --- the linked case, which is the reason this is a dict ---------------------


def test_one_linked_cel_carries_two_different_opacities_in_its_two_slots():
    """**The** test. Two keys, one ``Layer``, two numbers.

    An opacity stored on the layer could not express this at all -- the object
    is shared, so writing it through either slot would move both. Keying by
    ``(track uid, frame uid)`` is what makes the two independent, and folding
    the multiply into ``layers_for`` per *call* is what makes the shared object
    composite differently on the two frames.
    """
    doc = _animated()
    track, frame0, frame1 = doc.anim.tracks[0].uid, *(
        f.uid for f in doc.anim.frames[:2]
    )
    # One object, genuinely: this is the link, not two equal planes.
    assert doc.anim.cels[(track, frame0)] is doc.anim.cels[(track, frame1)]

    assert doc.set_cel_opacity(0.25, track_index=0, frame_index=0)
    assert doc.set_cel_opacity(0.75, track_index=0, frame_index=1)
    assert doc.anim.cel_alpha(track, frame0) == 0.25
    assert doc.anim.cel_alpha(track, frame1) == 0.75

    # And it reaches the materialised stack, which is what the compositor sees.
    doc.set_current_frame(0)
    assert doc.stack[0].opacity == pytest.approx(0.25)
    doc.set_current_frame(1)
    assert doc.stack[0].opacity == pytest.approx(0.75)
    # Still one object, after all that.
    assert doc.anim.cels[(track, frame0)] is doc.anim.cels[(track, frame1)]


def test_the_two_slots_of_a_link_composite_to_two_different_pictures():
    """The value is not merely stored: it changes pixels, through the ordinary
    ``(pixels, opacity, blend)`` triple the stack kernel already consumed."""
    doc = _animated()
    doc.set_cel_opacity(0.25, track_index=0, frame_index=0)
    doc.set_cel_opacity(1.0, track_index=0, frame_index=1)
    doc.set_current_frame(0)
    dim = doc.composite.copy()
    doc.set_current_frame(1)
    full = doc.composite.copy()
    assert dim[..., 3].max() < full[..., 3].max()


# --- undo, by uid ------------------------------------------------------------


def test_a_cel_opacity_change_is_one_undoable_step_addressed_by_uid():
    doc = _animated()
    track, frame = _slot(doc, 0, 1)
    before = len(doc.history)
    doc.set_cel_opacity(0.3, track_index=0, frame_index=1)
    assert len(doc.history) == before + 1
    doc.undo()
    assert doc.anim.cel_alpha(track, frame) == 1.0
    doc.redo()
    assert doc.anim.cel_alpha(track, frame) == pytest.approx(0.3)


def test_the_undo_lands_on_the_slot_it_was_made_to_after_a_reorder():
    """Addressed by uid, never by index -- the grid's standing rule."""
    doc = _animated(3)
    doc.set_cel_opacity(0.3, track_index=0, frame_index=2)
    moved = _slot(doc, 0, 2)
    doc.move_frame(2, 0)
    doc.undo()  # the move
    doc.undo()  # the opacity
    assert doc.anim.cel_alpha(*moved) == 1.0


def test_an_undone_step_costs_the_history_no_bytes():
    doc = _animated()
    before = doc.history.bytes
    doc.set_cel_opacity(0.3, track_index=0, frame_index=0)
    assert doc.history.bytes == before


# --- .aseprite -------------------------------------------------------------


def test_a_per_cel_opacity_round_trips_through_aseprite():
    doc = _animated()
    doc.unlink_cel(track_index=0, frame_index=1)
    doc.set_cel_opacity(0.6, track_index=0, frame_index=0)
    back, warnings = asein.document_from_aseprite(aseout.aseprite_bytes(doc))
    assert "per-cel opacity" not in " ".join(warnings)
    track, frame = _slot(back, 0, 0)
    # 0.6 -> 153/255, which is the nearest the format's byte can hold.
    assert back.anim.cel_alpha(track, frame) == pytest.approx(153 / 255)
    assert back.anim.cel_alpha(*_slot(back, 0, 1)) == 1.0


def test_a_linked_cels_two_opacities_round_trip_through_aseprite():
    """The linked chunk carries its own opacity byte, and so do we."""
    doc = _animated()
    doc.set_cel_opacity(0.25, track_index=0, frame_index=0)
    doc.set_cel_opacity(0.75, track_index=0, frame_index=1)
    back, _ = asein.document_from_aseprite(aseout.aseprite_bytes(doc))
    track = back.anim.tracks[0].uid
    frames = [frame.uid for frame in back.anim.frames]
    assert back.anim.cels[(track, frames[0])] is back.anim.cels[(track, frames[1])]
    assert back.anim.cel_alpha(track, frames[0]) == pytest.approx(0.25, abs=0.01)
    assert back.anim.cel_alpha(track, frames[1]) == pytest.approx(0.75, abs=0.01)


def test_a_document_that_never_dimmed_a_cel_writes_the_bytes_it_always_wrote():
    """The standing negative control, on the ``.aseprite`` writer."""
    doc = _animated()
    plain = aseout.aseprite_bytes(doc)
    doc.set_cel_opacity(0.5, track_index=0, frame_index=0)
    assert aseout.aseprite_bytes(doc) != plain
    doc.undo()
    assert aseout.aseprite_bytes(doc) == plain


# --- .ora --------------------------------------------------------------------


def test_a_per_cel_opacity_round_trips_through_ora(tmp_path):
    from warlock.studio.inker import ora

    doc = _animated()
    doc.unlink_cel(track_index=0, frame_index=1)
    doc.set_cel_opacity(0.4, track_index=0, frame_index=1)
    path = tmp_path / "a.ora"
    ora.write_ora(doc, path)
    back = ora.read_ora(path)
    assert back.anim.cel_alpha(*_slot(back, 0, 1)) == pytest.approx(0.4)
    assert back.anim.cel_alpha(*_slot(back, 0, 0)) == 1.0


def test_a_linked_cels_two_opacities_round_trip_through_ora(tmp_path):
    from warlock.studio.inker import ora

    doc = _animated()
    doc.set_cel_opacity(0.25, track_index=0, frame_index=0)
    doc.set_cel_opacity(0.75, track_index=0, frame_index=1)
    path = tmp_path / "a.ora"
    ora.write_ora(doc, path)
    back = ora.read_ora(path)
    track = back.anim.tracks[0].uid
    frames = [frame.uid for frame in back.anim.frames]
    # One PNG named twice -- the link survives -- wearing two numbers.
    assert back.anim.cels[(track, frames[0])] is back.anim.cels[(track, frames[1])]
    assert back.anim.cel_alpha(track, frames[0]) == pytest.approx(0.25)
    assert back.anim.cel_alpha(track, frames[1]) == pytest.approx(0.75)


def test_an_undimmed_document_writes_a_byte_identical_ora(tmp_path):
    """The standing negative control, on the determinism pin's own terms."""
    from warlock.studio.inker import ora

    doc = _animated()
    first = tmp_path / "a.ora"
    ora.write_ora(doc, first)
    plain = first.read_bytes()

    doc.set_cel_opacity(0.5, track_index=0, frame_index=0)
    doc.undo()
    second = tmp_path / "b.ora"
    ora.write_ora(doc, second)
    assert second.read_bytes() == plain


def test_an_opacity_key_an_older_build_never_wrote_reads_back_at_full(tmp_path):
    """Additive key, ``.get``-based reader, version still 1."""
    import json
    import zipfile

    from warlock.studio.inker import ora

    doc = _animated()
    path = tmp_path / "a.ora"
    ora.write_ora(doc, path)
    with zipfile.ZipFile(path) as zf:
        payload = json.loads(zf.read(ora.ANIMATION_MEMBER))
    assert payload["version"] == 1
    assert all("opacity" not in cel for cel in payload["cels"])
    back = ora.read_ora(path)
    assert back.anim.cel_opacity == {}


# --- the compositing path is untouched --------------------------------------


def test_the_fold_is_the_only_place_the_grid_reaches_the_compositor():
    """``layers.py`` and ``composite.py`` know nothing about a cel.

    The claim the wave was built on, asserted rather than remembered: the whole
    feature lives in ``Animation.layers_for``, so the stack, the blend
    arithmetic and the native kernel below it never learn the word.
    """
    import inspect

    from warlock.studio.inker import composite, layers

    for module in (layers, composite):
        assert "cel_opacity" not in inspect.getsource(module)
        assert "cel_alpha" not in inspect.getsource(module)


def test_a_still_document_has_no_grid_and_no_per_cel_opacity():
    doc = Document.blank(4, 4)
    assert doc.anim is None
    assert not doc.set_cel_opacity(0.5)
    assert doc.stack[0].opacity == 1.0
    assert np.asarray(doc.composite).shape[:2] == (4, 4)
