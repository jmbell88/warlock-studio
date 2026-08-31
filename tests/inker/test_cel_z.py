"""Per-cel z-index: the offset a *slot* carries from its track's position.

Divergence 12 (``docs/INVARIANTS.md``) used to read "track order IS stack order
(compositor + native kernel contract)", and this file is what retired it on
2026-08-30.

The storage is Wave 10's (``cel_opacity``) and Wave 12's (``cel_notes``): a
sparse dict on ``Animation`` keyed ``(track uid, frame uid)``, because a
**linked cel is two keys mapping to one ``Layer`` object** and the two slots may
legitimately sit at two different heights -- which Aseprite's own format agrees
with, giving every cel chunk, linked ones included, its own ``i16`` z-index.

Where this wave is *not* Wave 10 is the compositor. Opacity had somewhere to
land (the materialised layer's own ``opacity``), so ``layers_for`` folded it in
and nothing downstream learned the word "cel". A reorder has nowhere to land:
``layers_for`` must go on returning the list in **track** order, because
``active_index`` and every editing call site read list position as track
position. So the offset rides alongside the stack (``LayerStack.cel_z``) and is
applied at the one place that may reorder -- ``LayerStack._entries``, and only
when it is building the whole stack.

**And that costs a cache.** ``Document._below`` holds the composite of the
layers under the active one and is patched rather than rebuilt on a dab; its
premise is that those rows are finished business, and a lift is exactly the
thing that makes them not. So any nonzero z on the frame being drawn on turns
the cache off. The failure mode of getting that wrong is the nastiest kind --
right on the first dab, wrong on the second -- so the tests for it are here and
so is the standing negative control, whose reference hashes were captured
against the tree as it stood **before** this wave (a throwaway script,
Wave 13's method, pasted in below as literals).

The measured cost of the disabled cache is
``docs/measurements/2026-08-30-cel-z-below-cache.md``.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from warlock.studio.inker import asein, aseout, ora
from warlock.studio.inker.document import Document


def _animated(frames: int = 2, tracks: int = 3) -> Document:
    doc = Document.blank(4, 4)
    doc.stack[0].name = "Art"
    doc.stack[0].pixels[:, :] = (255, 0, 0, 255)
    for _ in range(tracks - 1):
        doc.add_layer()
    doc.invalidate_all()
    doc.ensure_animation()
    for _ in range(frames - 1):
        doc.add_frame(link=True)
    doc.set_current_frame(0)
    return doc


def _slot(doc, ti=0, fi=0):
    return doc.anim.tracks[ti].uid, doc.anim.frames[fi].uid


# --- the model ---------------------------------------------------------------


def test_a_fresh_grid_names_no_slot_and_every_slot_reads_zero():
    doc = _animated()
    assert doc.anim.cel_z == {}
    assert doc.anim.cel_zindex(*_slot(doc)) == 0


def test_setting_one_back_to_zero_removes_the_key_rather_than_storing_one():
    """Sparse, because both writers ask whether a slot is *named* here."""
    doc = _animated()
    assert doc.set_cel_z(2, track_index=0, frame_index=0)
    assert doc.anim.cel_z
    assert doc.set_cel_z(0, track_index=0, frame_index=0)
    assert doc.anim.cel_z == {}


def test_a_value_past_the_formats_field_is_clamped_rather_than_refused():
    """A number this build could save but not re-read is worse than a clamp."""
    doc = _animated()
    doc.anim.cel_z[_slot(doc)] = 99999
    assert doc.anim.cel_zindex(*_slot(doc)) == 32767
    doc.anim.cel_z[_slot(doc)] = -99999
    assert doc.anim.cel_zindex(*_slot(doc)) == -32768


def test_an_empty_slot_is_refused_rather_than_given_a_number_nothing_draws():
    doc = _animated()
    doc.clear_cel(track_index=1, frame_index=0)
    assert not doc.set_cel_z(1, track_index=1, frame_index=0)
    assert doc.anim.cel_z == {}


def test_the_edit_is_one_undoable_step_addressed_by_slot():
    doc = _animated()
    assert doc.set_cel_z(2, track_index=0, frame_index=0)
    assert doc.history.can_undo
    doc.undo()
    assert doc.anim.cel_z == {}
    doc.redo()
    assert doc.anim.cel_zindex(*_slot(doc)) == 2


def test_a_linked_cel_carries_a_height_per_slot_not_per_object():
    """The whole reason this is a dict and not a ``Layer`` field."""
    doc = _animated()
    track = doc.anim.tracks[0].uid
    first, second = (frame.uid for frame in doc.anim.frames)
    assert doc.anim.cels[(track, first)] is doc.anim.cels[(track, second)]
    assert doc.set_cel_z(2, track_index=0, frame_index=1)
    assert doc.anim.cel_zindex(track, second) == 2
    assert doc.anim.cel_zindex(track, first) == 0
    assert doc.anim.cels[(track, first)] is doc.anim.cels[(track, second)]


def test_the_track_does_not_move_when_a_cel_does():
    """An offset, not a reorder: everything addressed by track index goes on
    meaning what it meant, which is what the editing surface leans on."""
    doc = _animated()
    names = [track.name for track in doc.anim.tracks]
    doc.set_cel_z(2, track_index=0, frame_index=0)
    assert [track.name for track in doc.anim.tracks] == names
    assert [layer.name for layer in doc.stack] == names


# --- what it does to the picture ---------------------------------------------


def _three_colours(doc: Document) -> None:
    """Bottom red, middle green, top blue -- all opaque, all full canvas, so
    the flatten is exactly whichever one blends last."""
    for layer, colour in zip(
        doc.stack, [(255, 0, 0, 255), (0, 255, 0, 255), (0, 0, 255, 255)], strict=True
    ):
        layer.pixels[:, :] = colour
    doc.invalidate_all()


def test_with_no_z_the_top_track_is_what_shows():
    doc = _animated()
    _three_colours(doc)
    assert tuple(doc.flatten()[0, 0]) == (0, 0, 255, 255)


def test_lifting_the_bottom_cel_over_the_top_one_puts_it_on_top():
    """Three rows, so clearing the top one takes ``+3`` and not ``+2``: an
    offset that lands on the top row's own height ties with it, and a tie keeps
    track order -- which puts the lifted row *under* the one it matched."""
    doc = _animated()
    _three_colours(doc)
    doc.set_cel_z(3, track_index=0, frame_index=0)
    assert tuple(doc.flatten()[0, 0]) == (255, 0, 0, 255)


def test_dropping_the_top_cel_under_the_bottom_one_buries_it():
    doc = _animated()
    _three_colours(doc)
    doc.set_cel_z(-2, track_index=2, frame_index=0)
    assert tuple(doc.flatten()[0, 0]) == (0, 255, 0, 255)


def test_equal_effective_heights_keep_track_order_between_them():
    """The sort's tiebreak, which is why an unused feature is the identity
    permutation and not merely a cheap one."""
    doc = _animated()
    _three_colours(doc)
    # Row 0 lifted to 1 and row 1 already at 1: green is the later track, so
    # green stays over red, and blue at 2 is still over both.
    doc.set_cel_z(1, track_index=0, frame_index=0)
    assert tuple(doc.flatten()[0, 0]) == (0, 0, 255, 255)
    doc.set_cel_z(-2, track_index=2, frame_index=0)
    assert tuple(doc.flatten()[0, 0]) == (0, 255, 0, 255)


def test_a_lift_belongs_to_one_frame_and_not_the_next():
    doc = _animated()
    _three_colours(doc)
    doc.set_cel_z(3, track_index=0, frame_index=1)
    assert tuple(doc.flatten()[0, 0]) == (0, 0, 255, 255)
    doc.set_current_frame(1)
    assert tuple(doc.flatten()[0, 0]) == (255, 0, 0, 255)


def test_the_off_frame_flatten_honours_it_too():
    """Onion skinning, playback and every export read ``frame_stack``."""
    doc = _animated()
    _three_colours(doc)
    doc.set_cel_z(3, track_index=0, frame_index=1)
    flat = doc.frame_flat(doc.anim.frames[1].uid)
    assert tuple(flat[0, 0]) == (255, 0, 0, 255)


def test_a_still_document_has_nowhere_to_put_one_and_says_so_by_having_none():
    doc = Document.blank(4, 4)
    assert doc.anim is None
    assert doc.cel_z_rows() is None
    assert doc.stack.cel_z is None
    assert not doc.set_cel_z(1, track_index=0, frame_index=0)


def test_a_dead_rows_leftover_entry_does_not_arm_the_feature():
    """``cel_z`` keeps entries for deleted rows on purpose (undo brings them
    back); a stale key must not disable a cache for a document that has none."""
    doc = _animated()
    doc.set_cel_z(2, track_index=0, frame_index=0)
    doc.remove_layer(0)
    assert doc.anim.cel_z, "the entry is kept, which is the field's own rule"
    assert doc.cel_z_rows() is None
    assert doc.stack.cel_z is None


# --- the below-cache, which is what this wave really costs --------------------


def _dab(doc: Document, colour=(0, 0, 0, 255), box=(0, 0, 4, 4)) -> None:
    weight = np.zeros((4, 4), dtype=np.float32)
    x0, y0, x1, y1 = box
    weight[y0:y1, x0:x1] = 1.0
    doc.write_colour((0, 0, 4, 4), colour, weight)


def test_an_ordinary_document_still_uses_the_below_cache():
    doc = _animated()
    _three_colours(doc)
    doc.set_active_layer(1)
    _dab(doc)
    assert doc._below is not None


def test_a_nonzero_z_turns_the_below_cache_off_for_that_frame():
    doc = _animated()
    _three_colours(doc)
    doc.set_cel_z(2, track_index=0, frame_index=0)
    doc.set_active_layer(1)
    _dab(doc)
    assert doc.stack.cel_z is not None
    assert doc._below is None, "the cache's premise is broken; it must not fill"


def test_the_second_dab_is_as_right_as_the_first():
    """**The failure mode this wave exists to avoid.** With the cache left on,
    the first dab rebuilds it and looks correct and the second reuses a base
    that no longer describes the picture."""
    doc = _animated()
    _three_colours(doc)
    doc.set_cel_z(3, track_index=0, frame_index=0)
    doc.set_active_layer(1)
    for _ in range(3):
        _dab(doc, (10, 20, 30, 255))
        # Red is lifted over everything, so no dab on the middle row can show.
        assert np.array_equal(doc._composite, doc.stack.flatten())
        assert tuple(doc._composite[0, 0]) == (255, 0, 0, 255)


def test_turning_the_last_z_back_off_gives_the_cache_back():
    doc = _animated()
    _three_colours(doc)
    doc.set_cel_z(2, track_index=0, frame_index=0)
    doc.set_active_layer(1)
    _dab(doc)
    assert doc._below is None
    doc.set_cel_z(0, track_index=0, frame_index=0)
    _dab(doc, (1, 2, 3, 255))
    assert doc.stack.cel_z is None
    assert doc._below is not None


def test_a_write_below_the_active_layer_is_right_with_the_cache_off():
    """The path the cache's ``composite_below_region`` repair used to serve."""
    doc = _animated()
    _three_colours(doc)
    doc.set_cel_z(2, track_index=2, frame_index=0)
    doc.set_active_layer(2)
    doc.stack[0].pixels[:, :] = (7, 8, 9, 255)
    doc.invalidate((0, 0, 4, 4), layer_uid=doc.stack[0].uid)
    assert np.array_equal(doc._composite, doc.stack.flatten())


def test_a_partial_range_is_never_asked_to_sort():
    """``_order`` refuses a partial range even with a z in play, because a lift
    can cross the active layer and there is no correct answer for a slice."""
    doc = _animated()
    _three_colours(doc)
    doc.set_cel_z(2, track_index=0, frame_index=0)
    stack = doc.stack
    assert stack.cel_z is not None
    assert stack._order(0, len(stack)) is not None
    assert stack._order(0, 1) is None
    assert stack._order(1, len(stack)) is None


# --- the formats -------------------------------------------------------------


def test_a_z_survives_a_round_trip_through_aseprite():
    doc = _animated()
    doc.set_cel_z(2, track_index=0, frame_index=0)
    doc.set_cel_z(-1, track_index=2, frame_index=1)
    back, _warnings = asein.document_from_aseprite(aseout.aseprite_bytes(doc))
    assert back.anim is not None
    assert back.anim.cel_zindex(*_slot(back, 0, 0)) == 2
    assert back.anim.cel_zindex(*_slot(back, 2, 1)) == -1
    assert back.anim.cel_zindex(*_slot(back, 1, 0)) == 0


def test_a_links_two_chunks_carry_two_heights_over_one_image():
    """The linked-cel chunk has its own z field, and so does this package."""
    doc = _animated()
    doc.set_cel_z(2, track_index=0, frame_index=1)
    back, _warnings = asein.document_from_aseprite(aseout.aseprite_bytes(doc))
    track = back.anim.tracks[0].uid
    first, second = (frame.uid for frame in back.anim.frames)
    assert back.anim.cels[(track, first)] is back.anim.cels[(track, second)]
    assert back.anim.cel_zindex(track, first) == 0
    assert back.anim.cel_zindex(track, second) == 2


def test_a_z_survives_a_round_trip_through_ora(tmp_path):
    doc = _animated()
    _three_colours(doc)
    doc.set_cel_z(3, track_index=0, frame_index=0)
    path = tmp_path / "z.ora"
    path.write_bytes(ora.ora_bytes(doc))
    back = ora.read_ora(path)
    assert back.anim.cel_zindex(*_slot(back, 0, 0)) == 3
    # And the *loaded* document composites it, which is the half a dictionary
    # assertion cannot see: ``Document.__post_init__`` ends in
    # ``invalidate_all``, which is what arms ``LayerStack.cel_z``.
    assert back.stack.cel_z is not None
    assert tuple(back.flatten()[0, 0]) == (255, 0, 0, 255)


def test_an_ora_without_the_key_reads_back_flat_rather_than_failing(tmp_path):
    """Additive: the version stays 1, so an older file still opens."""
    doc = _animated()
    path = tmp_path / "flat.ora"
    path.write_bytes(ora.ora_bytes(doc))
    back = ora.read_ora(path)
    assert back.anim.cel_z == {}


# --- the standing negative control -------------------------------------------
#
# Captured against the tree as it stood at cc7ee724, BEFORE this wave, by a
# throwaway script (Wave 13's method). Every hash below is the output of a run
# that actually happened; the builder underneath is a transcription of that
# script's, and the assertion is that the wave changed none of it.

_COMPOSITE_SHA = "5d4904c2a051e4c33988317175896acbf420b9c62a91b5b4579962dadec2cc4e"
_FLATTEN_SHA = "5d4904c2a051e4c33988317175896acbf420b9c62a91b5b4579962dadec2cc4e"
_FRAME1_FLAT_SHA = "4a1d647d8ddeefffbd7a9f154bb9f8aeed7edc6ecd251ec4967f20a0a90d0eee"
_ORA_SHA = "ced2313ac8e227a96bcbadbaea360dcf2ef6b41e2ec0ff50577d48704d513826"
_ASE_SHA = "b58b3569dad83046a86898e46703c671e1615763694e029ba2d6dac11275fc8c"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _reference_doc() -> Document:
    """The script's document, verbatim: three opaque-ish rows, two frames that
    are not linked, and **two** dabs on the middle row -- the first builds the
    below-cache and the second reuses it, which is the whole reason this
    reference exists."""
    doc = Document.blank(16, 16)
    doc.stack[0].name = "Art"
    doc.stack[0].pixels[:, :] = (200, 30, 30, 255)
    doc.add_layer()
    doc.stack[1].name = "Mid"
    doc.stack[1].pixels[4:12, 4:12] = (30, 200, 30, 128)
    doc.add_layer()
    doc.stack[2].name = "Top"
    doc.stack[2].pixels[6:10, 6:10] = (30, 30, 200, 200)
    doc.invalidate_all()
    doc.ensure_animation()
    doc.add_frame(link=False)
    doc.set_current_frame(1)
    doc.set_active_layer(2)
    weight = np.zeros((16, 16), dtype=np.float32)
    weight[1:6, 9:15] = 1.0
    doc.write_colour((0, 0, 16, 16), (5, 250, 250, 255), weight)
    doc.set_current_frame(0)

    doc.set_active_layer(1)
    first = np.zeros((16, 16), dtype=np.float32)
    first[2:8, 2:8] = 0.75
    doc.write_colour((0, 0, 16, 16), (10, 20, 30, 255), first)
    second = np.zeros((16, 16), dtype=np.float32)
    second[5:14, 5:14] = 0.5
    doc.write_colour((0, 0, 16, 16), (240, 240, 10, 255), second)
    return doc


def test_a_document_with_no_z_still_uses_the_cache_and_draws_the_same_pixels():
    """The standing negative control, half one: the cached path is still the
    cached path, and it still produces the bytes it produced before the wave."""
    doc = _reference_doc()
    assert doc.stack.cel_z is None
    assert doc._below is not None, "the cache must still be in play"
    assert _sha(doc._composite.tobytes()) == _COMPOSITE_SHA
    assert _sha(doc.flatten().tobytes()) == _FLATTEN_SHA
    assert _sha(doc.frame_flat(doc.anim.frames[1].uid).tobytes()) == _FRAME1_FLAT_SHA


def test_a_document_with_no_z_writes_the_same_files():
    """Half two: the ORA and ``.aseprite`` determinism pins, against literals
    captured before the wave rather than against this same tree twice."""
    doc = _reference_doc()
    assert _sha(ora.ora_bytes(doc)) == _ORA_SHA
    assert _sha(aseout.aseprite_bytes(doc)) == _ASE_SHA


def test_the_control_is_worth_nothing_without_its_positive():
    doc = _reference_doc()
    doc.set_cel_z(2, track_index=0, frame_index=0)
    assert _sha(ora.ora_bytes(doc)) != _ORA_SHA
    assert _sha(aseout.aseprite_bytes(doc)) != _ASE_SHA
    assert _sha(doc.flatten().tobytes()) != _FLATTEN_SHA


def test_setting_a_z_and_putting_it_back_writes_the_reference_bytes_again():
    """Sparse storage is what makes this true, and it is the property both
    writers lean on."""
    doc = _reference_doc()
    doc.set_cel_z(2, track_index=0, frame_index=0)
    doc.set_cel_z(0, track_index=0, frame_index=0)
    assert doc.anim.cel_z == {}
    assert _sha(ora.ora_bytes(doc)) == _ORA_SHA
    assert _sha(aseout.aseprite_bytes(doc)) == _ASE_SHA


def test_the_entries_a_flat_stack_hands_the_compositor_are_the_old_ones():
    """The native stack kernel's bar is bit parity, so what it *receives* for a
    document that does not use the feature must be the identical list."""
    doc = _animated()
    _three_colours(doc)
    rows = doc.stack._entries(0, len(doc.stack))
    doc.stack.cel_z = [0, 0, 0]
    same = doc.stack._entries(0, len(doc.stack))
    assert [(id(p), o, b) for p, o, b in rows] == [(id(p), o, b) for p, o, b in same]


@pytest.mark.parametrize("value", [3, -3])
def test_an_offset_past_the_ends_of_the_stack_is_the_end_of_the_stack(value):
    """Clamping is the sort's, not a refusal: nothing above the top row exists
    for a cel to sit between."""
    doc = _animated()
    _three_colours(doc)
    doc.set_cel_z(value, track_index=1, frame_index=0)
    expected = (0, 255, 0, 255) if value > 0 else (0, 0, 255, 255)
    assert tuple(doc.flatten()[0, 0]) == expected
