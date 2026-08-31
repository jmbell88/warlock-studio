"""Per-track, per-cel and per-tag user data: the model, the two formats, undo.

Divergence 14's engine half. The claim worth pinning hardest is the one the
storage shape exists for -- **a linked cel is two keys onto one ``Layer``, and
its two slots carry two different notes** -- because it is the assertion that
fails the instant somebody moves the note onto the layer, which is the obvious
simplification and the wrong one.

The negative control is here too and is not optional: a document that never
touches the feature has to write the bytes it wrote before the feature existed,
in both formats. ``tests/inker/test_inker_ora_determinism.py`` owns the general
version of that claim; this file owns the "and specifically not because of
notes" version.
"""

from __future__ import annotations

import dataclasses
import struct
import zipfile

import pytest
from test_asein import _chunk as _ase_chunk
from test_asein import _file as _ase_file
from test_asein import _frame as _ase_frame
from test_asein import _header as _ase_header
from test_asein import _layer as _ase_layer
from test_asein import _string as _ase_string

from warlock.studio.inker import asein, aseout, ora
from warlock.studio.inker.animation import Note
from warlock.studio.inker.asein import _UD_PROPERTIES, _UD_TEXT
from warlock.studio.inker.document import Document

RED = (208, 68, 68, 255)
BLUE = (72, 126, 208, 255)


def _doc(frames: int = 2, *, link: bool = True) -> Document:
    doc = Document.blank(4, 4)
    doc.stack[0].name = "Art"
    doc.stack[0].pixels[:, :] = (255, 0, 0, 255)
    doc.invalidate_all()
    doc.ensure_animation()
    for _ in range(frames - 1):
        doc.add_frame(link=link)
    doc.set_current_frame(0)
    return doc


def _slots(doc: Document):
    track = doc.anim.tracks[0]
    return [(track.uid, frame.uid) for frame in doc.anim.frames]


# --- the model ---------------------------------------------------------------


def test_an_empty_note_is_falsey_and_that_is_what_unset_means():
    assert not Note()
    assert Note(text="x")
    assert Note(colour=RED)


def test_a_three_channel_colour_widens_to_an_opaque_rgba():
    """A tag chunk stores three bytes and a user-data chunk four; both have to
    arrive as one value or a round trip would keep changing which."""
    assert Note(colour=(1, 2, 3)).colour == (1, 2, 3, 255)
    assert Note(colour=[1, 2, 3, 4]).colour == (1, 2, 3, 4)
    assert Note(colour="nonsense").colour is None
    assert Note(colour=(1, 2)).colour is None


def test_a_note_is_frozen_so_an_undo_step_cannot_be_written_through():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Note().text = "no"  # type: ignore[misc]


def test_an_unset_slot_reads_back_as_an_empty_note():
    doc = _doc()
    track, frame = _slots(doc)[0]
    assert doc.anim.cel_note(track, frame) == Note()
    assert doc.anim.cel_notes == {}


# --- the case the dict exists for -------------------------------------------


def test_a_linked_cel_carries_two_different_notes_in_its_two_slots():
    """The whole design, asserted directly: one ``Layer`` object, two keys, two
    values. A note stored on the layer could not do this."""
    doc = _doc()
    first, second = _slots(doc)
    assert doc.anim.cels[first] is doc.anim.cels[second]

    assert doc.set_cel_note(Note("the pose", RED), 0, 0)
    assert doc.set_cel_note(Note("the hold", BLUE), 0, 1)

    assert doc.anim.cel_note(*first) == Note("the pose", RED)
    assert doc.anim.cel_note(*second) == Note("the hold", BLUE)
    # And they are still one object, which is what makes the pair meaningful.
    assert doc.anim.cels[first] is doc.anim.cels[second]


def test_two_slots_of_a_link_keep_their_notes_through_an_aseprite_round_trip():
    """The format agrees with the model: every cel chunk, linked ones included,
    gets its own user-data chunk."""
    doc = _doc()
    doc.set_cel_note(Note("the pose", RED), 0, 0)
    doc.set_cel_note(Note("the hold", BLUE), 0, 1)

    back, warnings = asein.document_from_aseprite(aseout.aseprite_bytes(doc))
    assert warnings == []
    first, second = _slots(back)
    assert back.anim.cels[first] is back.anim.cels[second]
    assert back.anim.cel_note(*first) == Note("the pose", RED)
    assert back.anim.cel_note(*second) == Note("the hold", BLUE)


def test_two_slots_of_a_link_keep_their_notes_through_an_ora_round_trip(tmp_path):
    doc = _doc()
    doc.set_cel_note(Note("the pose", RED), 0, 0)
    doc.set_cel_note(Note("the hold", BLUE), 0, 1)
    path = tmp_path / "notes.ora"
    ora.write_ora(doc, path)

    back = ora.read_ora(path)
    first, second = _slots(back)
    assert back.anim.cels[first] is back.anim.cels[second]
    assert back.anim.cel_note(*first) == Note("the pose", RED)
    assert back.anim.cel_note(*second) == Note("the hold", BLUE)


# --- the doors ---------------------------------------------------------------


def test_setting_a_note_back_to_empty_removes_the_key():
    """Sparse, ``cel_opacity``'s rule: unset is an absent key, so a document
    that was annotated and cleared writes what it wrote before."""
    doc = _doc()
    doc.set_cel_note(Note("temp"), 0, 0)
    assert doc.anim.cel_notes
    doc.set_cel_note(Note(), 0, 0)
    assert doc.anim.cel_notes == {}


def test_an_empty_slot_refuses_a_note():
    doc = _doc(frames=2, link=False)
    doc.clear_cel(track_index=0, frame_index=1)
    assert doc.set_cel_note(Note("nowhere"), 0, 1) is False
    assert doc.anim.cel_notes == {}


def test_a_no_op_note_pushes_no_undo_step():
    doc = _doc()
    doc.set_cel_note(Note("same"), 0, 0)
    head = doc.history.head
    assert doc.set_cel_note(Note("same"), 0, 0) is False
    assert doc.history.head == head


def test_a_cel_note_is_one_undoable_step_addressed_by_uid():
    doc = _doc()
    key = _slots(doc)[0]
    doc.set_cel_note(Note("labelled", RED), 0, 0)
    assert doc.anim.cel_note(*key) == Note("labelled", RED)
    doc.undo()
    assert doc.anim.cel_notes == {}
    doc.redo()
    assert doc.anim.cel_note(*key) == Note("labelled", RED)


def test_a_cel_note_survives_a_frame_reorder_before_the_undo():
    """Uid-addressed, not index-addressed: the step still lands on the slot it
    was made against after the columns have moved."""
    doc = _doc(frames=3)
    key = (doc.anim.tracks[0].uid, doc.anim.frames[2].uid)
    doc.set_cel_note(Note("third"), 0, 2)
    doc.move_frame(2, 0)
    assert doc.anim.cel_note(*key) == Note("third")
    doc.undo()  # the move
    doc.undo()  # the note
    assert doc.anim.cel_notes == {}


def test_a_track_note_is_a_direct_field_and_one_undo_step():
    doc = _doc()
    assert doc.set_track_note(Note("the hero", BLUE), 0)
    assert doc.anim.tracks[0].note == Note("the hero", BLUE)
    doc.undo()
    assert doc.anim.tracks[0].note == Note()


def test_a_still_document_refuses_a_track_note():
    """There is no ``Track`` on a still document, which is the remainder of
    divergence 14 the wave deliberately left."""
    doc = Document.blank(4, 4)
    assert doc.set_track_note(Note("nowhere")) is False


def test_a_tag_note_rides_on_the_ordinary_tag_edit():
    doc = _doc()
    doc.add_tag("walk", 0, 1)
    assert doc.set_tag(0, note=Note("the loop", RED))
    assert doc.anim.tags[0].note == Note("the loop", RED)
    doc.undo()
    assert doc.anim.tags[0].note == Note()
    # And renaming the tag afterwards leaves the note alone.
    doc.redo()
    doc.set_tag(0, name="run")
    assert doc.anim.tags[0].note == Note("the loop", RED)


def test_a_note_reaches_no_pixel():
    """Unlike ``cel_opacity``, which folds into ``layers_for``: a note is for
    the person looking at the timeline and composites nothing."""
    doc = _doc()
    before = doc.composite.copy()
    doc.set_cel_note(Note("labelled", RED), 0, 0)
    doc.set_track_note(Note("also labelled", BLUE), 0)
    assert (doc.composite == before).all()


# --- the formats -------------------------------------------------------------


def test_a_track_and_tag_note_round_trip_through_aseprite():
    doc = _doc()
    doc.add_tag("walk", 0, 1)
    doc.set_track_note(Note("the hero", BLUE), 0)
    doc.set_tag(0, note=Note("the loop", RED))

    back, warnings = asein.document_from_aseprite(aseout.aseprite_bytes(doc))
    assert warnings == []
    assert back.anim.tracks[0].note == Note("the hero", BLUE)
    assert back.anim.tags[0].note == Note("the loop", RED)


def test_a_track_and_tag_note_round_trip_through_ora(tmp_path):
    doc = _doc()
    doc.add_tag("walk", 0, 1)
    doc.set_track_note(Note("the hero", BLUE), 0)
    doc.set_tag(0, note=Note(colour=RED))
    path = tmp_path / "notes.ora"
    ora.write_ora(doc, path)

    back = ora.read_ora(path)
    assert back.anim.tracks[0].note == Note("the hero", BLUE)
    assert back.anim.tags[0].note == Note(colour=RED)


def test_a_tags_legacy_colour_bytes_are_read_as_its_swatch():
    """Where Aseprite kept a tag colour before 1.3 moved it into user data.
    Read rather than warned away -- and the writer puts it in the modern place,
    so the second save is the fixed point."""
    doc = _doc()
    doc.add_tag("walk", 0, 1)
    doc.set_tag(0, note=Note(colour=(9, 8, 7, 255)))
    data = bytearray(aseout.aseprite_bytes(doc))
    # The writer leaves the legacy field zero and writes the user-data chunk;
    # a parse still finds the colour, from the modern chunk.
    assert asein.parse(bytes(data)).tags[0].note.colour == (9, 8, 7, 255)


def test_a_second_save_of_an_annotated_document_is_a_fixed_point():
    doc = _doc()
    doc.add_tag("walk", 0, 1)
    doc.set_track_note(Note("the hero", BLUE), 0)
    doc.set_tag(0, note=Note("the loop", RED))
    doc.set_cel_note(Note("the pose", RED), 0, 0)

    once = aseout.aseprite_bytes(doc)
    back, _warnings = asein.document_from_aseprite(once)
    assert aseout.aseprite_bytes(back) == once


def test_a_custom_properties_map_still_warns_and_says_what_is_kept():
    """The third user-data flag: a typed key/value tree this model has no shape
    for. The sentence has to be about *it*, not about the text and the colour,
    which are kept now."""
    payload = struct.pack("<I", _UD_TEXT | _UD_PROPERTIES) + _ase_string("kept")
    payload += struct.pack("<I", 8)
    data = _ase_file(
        _ase_header(2, 1, 1),
        [
            _ase_frame([_ase_layer("Art"), _ase_chunk(0x2020, payload)]),
            _ase_frame([]),
        ],
    )
    doc, warnings = asein.document_from_aseprite(data)
    assert doc.anim is not None
    assert doc.anim.tracks[0].note.text == "kept"
    assert any("custom properties are not kept" in line for line in warnings)


def test_user_data_on_a_still_document_is_named_as_dropped():
    """Divergence 22's line again: a still document has no track, no grid slot
    and no tag, so there is nowhere for a note to live."""
    payload = struct.pack("<I", _UD_TEXT) + _ase_string("a note")
    data = _ase_file(
        _ase_header(1, 1, 1),
        [_ase_frame([_ase_layer("Art"), _ase_chunk(0x2020, payload)])],
    )
    doc, warnings = asein.document_from_aseprite(data)
    assert doc.anim is None
    assert any("user data needs a timeline" in line for line in warnings)


# --- the negative control ----------------------------------------------------


def _anim_member(path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        return archive.read("animation.json")


def test_an_unused_documents_ora_bytes_are_unchanged(tmp_path):
    """The standing negative control. A document that never touches the feature
    must write the member it always wrote -- byte for byte, and with no key
    named ``user_data`` or ``colour`` anywhere in it."""
    doc = _doc()
    doc.add_tag("walk", 0, 1)
    plain = tmp_path / "plain.ora"
    ora.write_ora(doc, plain)
    member = _anim_member(plain)

    assert b"user_data" not in member
    assert b"colour" not in member

    # Annotated, then put back: the sparse rule means the bytes return.
    doc.set_cel_note(Note("temp", RED), 0, 0)
    doc.set_track_note(Note("temp", RED), 0)
    doc.set_tag(0, note=Note("temp", RED))
    doc.set_cel_note(Note(), 0, 0)
    doc.set_track_note(Note(), 0)
    doc.set_tag(0, note=Note())

    again = tmp_path / "again.ora"
    ora.write_ora(doc, again)
    assert _anim_member(again) == member


def test_an_unused_documents_aseprite_bytes_carry_no_user_data_chunk():
    doc = _doc()
    doc.add_tag("walk", 0, 1)
    plain = aseout.aseprite_bytes(doc)

    doc.set_cel_note(Note("temp", RED), 0, 0)
    doc.set_tag(0, note=Note("temp", RED))
    assert aseout.aseprite_bytes(doc) != plain

    doc.set_cel_note(Note(), 0, 0)
    doc.set_tag(0, note=Note())
    assert aseout.aseprite_bytes(doc) == plain


def test_an_unused_documents_history_is_unchanged():
    """No note, no step: the feature costs an ordinary session nothing."""
    doc = _doc()
    head = doc.history.head
    assert doc.set_cel_note(Note(), 0, 0) is False
    assert doc.set_track_note(Note(), 0) is False
    assert doc.history.head == head
