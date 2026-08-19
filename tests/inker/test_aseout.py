"""Writing a ``.aseprite``, gated by reading it back with our own parser.

The gate is deliberately ``asein`` and not a checked-in binary. A fixture blob
would pin these bytes against *nothing* -- a writer and a matching golden file
can be wrong together forever -- where a round trip asserts the one property
the writer exists for: what ``aseout`` puts on disk is what this build reads
back, plane for plane and flag for flag. The half a round trip cannot reach --
whether real Aseprite agrees -- is a user-owed manual pass, the Tiled-fixtures
precedent, and is named in ``ASEPRITE_PARITY.md``'s Wave 5 gate rather than
faked here.

``test_asein.py`` is the mirror of this file and builds its fixtures by hand
with ``struct.pack``; nothing here reaches for those builders, because a test
that wrote the file with the reader's own helpers would be asserting that two
copies of one idea agree.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from warlock.studio.inker import asein, aseout
from warlock.studio.inker import groups as gp
from warlock.studio.inker.animation import DIRECTIONS, Frame, Tag, Track
from warlock.studio.inker.composite import BLEND_MODES
from warlock.studio.inker.document import Document

RED = (255, 0, 0, 255)
BLUE = (0, 0, 255, 255)


def _round_trip(doc) -> tuple[Document, list[str]]:
    return asein.document_from_aseprite(aseout.aseprite_bytes(doc))


def _still() -> Document:
    doc = Document.blank(4, 4)
    doc.stack[0].name = "Background"
    doc.stack[0].pixels[:, :] = RED
    top = doc.add_layer("Ink")
    top.pixels[0:2, 0:2] = BLUE
    top.opacity = 0.6
    top.blend = "multiply"
    top.visible = False
    top.locked = True
    doc.invalidate_all()
    return doc


def _animated() -> Document:
    """Two tracks over two frames: a linked background and a per-frame top."""
    doc = Document.blank(4, 4)
    doc.stack[0].name = "Background"
    doc.stack[0].pixels[:, :] = RED
    doc.add_layer("Ink")
    doc.invalidate_all()
    doc.ensure_animation()
    doc.anim.frames[0].duration_ms = 40
    doc.set_active_layer(1)
    doc.write_colour((0, 0, 2, 2), BLUE, np.ones((2, 2), dtype=np.float32))
    doc.add_frame(link=True)
    doc.set_frame_duration(1, 120)
    doc.unlink_cel(track_index=1, frame_index=1)
    doc.set_current_frame(1)
    doc.set_active_layer(1)
    doc.write_colour((2, 2, 4, 4), BLUE, np.ones((2, 2), dtype=np.float32))
    doc.set_current_frame(0)
    return doc


# --- RGB, still and animated -------------------------------------------------


def test_a_still_rgb_document_round_trips_bit_exact():
    doc = _still()
    back, warnings = _round_trip(doc)

    assert warnings == []
    assert back.anim is None
    assert back.size == (4, 4)
    assert [layer.name for layer in back.stack] == ["Background", "Ink"]
    for before, after in zip(doc.stack, back.stack, strict=True):
        assert np.array_equal(before.pixels, after.pixels)


def test_a_still_layers_properties_survive():
    doc = _still()
    back, _ = _round_trip(doc)
    top = back.stack[1]
    assert top.opacity == pytest.approx(0.6, abs=1 / 255)
    assert top.blend == "multiply"
    assert top.visible is False
    assert top.locked is True


def test_the_layer_opacity_flag_is_set():
    """Bit 0 of the header's flags. Off, every layer reads back fully opaque
    whatever byte was stored -- which is a picture that composites wrong rather
    than a file that fails to open, so nothing but an opacity assertion sees it."""
    doc = _still()
    doc.stack[0].opacity = 0.4
    back, _ = _round_trip(doc)
    assert back.stack[0].opacity == pytest.approx(0.4, abs=1 / 255)


def test_an_animated_document_keeps_its_frames_and_durations():
    doc = _animated()
    back, warnings = _round_trip(doc)

    assert warnings == []
    assert back.anim is not None
    assert [frame.duration_ms for frame in back.anim.frames] == [40, 120]
    assert [track.name for track in back.anim.tracks] == ["Background", "Ink"]


def test_every_cel_of_an_animated_document_is_bit_exact():
    doc = _animated()
    back, _ = _round_trip(doc)
    anim, other = doc.anim, back.anim
    for ti, track in enumerate(anim.tracks):
        for fi, frame in enumerate(anim.frames):
            before = anim.cels.get((track.uid, frame.uid))
            after = other.cels.get((other.tracks[ti].uid, other.frames[fi].uid))
            assert (before is None) == (after is None)
            if before is not None:
                assert np.array_equal(before.pixels, after.pixels)


def test_the_six_track_properties_survive():
    doc = _animated()
    track = doc.anim.tracks[1]
    track.name = "Ink"
    track.opacity = 0.6
    track.visible = False
    track.blend = "screen"
    track.locked = True
    track.continuous = True

    back, _ = _round_trip(doc)
    after = back.anim.tracks[1]
    assert after.name == "Ink"
    assert after.opacity == pytest.approx(0.6, abs=1 / 255)
    assert after.visible is False
    assert after.blend == "screen"
    assert after.locked is True
    assert after.continuous is True


def test_an_empty_slot_stays_empty():
    """A track with no cel on a frame writes no cel chunk, and the sparse grid
    comes back sparse -- a full-canvas transparent cel would read as drawn."""
    doc = _animated()
    doc.anim.cels.pop((doc.anim.tracks[1].uid, doc.anim.frames[1].uid))
    back, _ = _round_trip(doc)
    assert back.anim.cels.get((back.anim.tracks[1].uid, back.anim.frames[1].uid)) is None


def test_alpha_lock_is_dropped_and_the_read_is_clean():
    """The one track property Aseprite has no bit for. It is an *editing aid*,
    not picture data, so it goes silently -- the interop report is where the
    loss is written down, not a toast on every save."""
    doc = _animated()
    doc.anim.tracks[0].alpha_lock = True
    back, warnings = _round_trip(doc)
    assert warnings == []
    assert back.anim.tracks[0].alpha_lock is False


# --- grayscale ---------------------------------------------------------------


def test_a_grayscale_document_round_trips_as_grayscale():
    doc = Document.blank(4, 4)
    doc.stack[0].pixels[:, :] = (200, 30, 30, 255)
    doc.stack[0].pixels[0, 0] = (0, 0, 0, 0)
    doc.invalidate_all()
    doc.convert_to_grayscale()

    back, warnings = _round_trip(doc)
    assert warnings == []
    assert back.color_mode == "grayscale"
    assert np.array_equal(back.stack[0].pixels, doc.stack[0].pixels)


def test_a_grayscale_document_holding_a_colour_is_refused_by_name():
    """Only reachable on a corrupted document -- the write funnel enforces
    ``r == g == b`` -- so the check is what stops two of the three channels
    being thrown away without a word."""
    doc = Document.blank(4, 4)
    doc.invalidate_all()
    doc.convert_to_grayscale()
    doc.stack[0].pixels[1, 1] = BLUE

    with pytest.raises(ValueError, match="Background"):
        aseout.aseprite_bytes(doc)


# --- indexed -----------------------------------------------------------------


def _indexed() -> Document:
    """A palette with **two identical browns** and a hole that is not slot 0."""
    doc = Document.blank(4, 4)
    doc.stack[0].pixels[:, :] = (10, 20, 30, 255)
    doc.invalidate_all()
    palette = [(9, 9, 9, 255), (10, 20, 30, 255), (10, 20, 30, 255), (200, 10, 10, 128)]
    doc.convert_to_indexed(palette, transparent=1)
    layer = doc.stack[0]
    layer.indices[0, :] = 0
    layer.indices[1, :] = 2
    layer.indices[2, :] = 3
    layer.indices[3, :] = 1
    doc._rematerialize(layer, doc._index_lut(), notify=False)
    doc.invalidate_all()
    return doc


def test_an_indexed_document_keeps_the_slot_each_pixel_is_in():
    """The whole reason index planes exist: slot 1 and slot 2 hold the same
    brown, and a writer that re-quantised the pixels would collapse them."""
    doc = _indexed()
    back, warnings = _round_trip(doc)

    assert warnings == []
    assert back.color_mode == "indexed"
    assert np.array_equal(back.stack[0].indices, doc.stack[0].indices)
    assert np.array_equal(back.stack[0].pixels, doc.stack[0].pixels)


def test_an_indexed_document_keeps_its_palette_and_its_hole():
    doc = _indexed()
    back, _ = _round_trip(doc)
    assert [tuple(c) for c in back.palette] == [tuple(c) for c in doc.palette]
    assert back.transparent_index == 1


def test_an_indexed_palette_keeps_its_alpha():
    doc = _indexed()
    back, _ = _round_trip(doc)
    assert tuple(back.palette[3]) == (200, 10, 10, 128)


def test_an_indexed_animated_document_round_trips():
    doc = _animated()
    doc.convert_to_indexed([(0, 0, 0, 0), RED, BLUE], transparent=0)
    back, _ = _round_trip(doc)
    assert back.color_mode == "indexed"
    for layer, after in zip(
        doc.anim.unique_cel_layers(), back.anim.unique_cel_layers(), strict=True
    ):
        assert np.array_equal(layer.indices, after.indices)


# --- groups ------------------------------------------------------------------


def _grouped() -> Document:
    """Three layers, the top two in a group, and *that* group inside another.

    The second ``group_layers`` over the same rows nests **inward**: the new
    group takes the run's existing parent, so ``Art`` ends up inside ``Ink``
    rather than around it. Two levels either way, which is what this file needs.
    """
    doc = Document.blank(4, 4)
    doc.stack[0].name = "L0"
    for i in (1, 2):
        doc.add_layer(f"L{i}")
    doc.invalidate_all()
    outer = doc.group_layers([1, 2], name="Ink")
    assert outer is not None
    inner = doc.group_layers([1, 2], name="Art")
    assert inner is not None
    assert doc.group_of[inner.uid] == outer.uid
    return doc


def test_nested_groups_survive_the_round_trip():
    doc = _grouped()
    back, warnings = _round_trip(doc)

    assert warnings == []
    assert [layer.name for layer in back.stack] == ["L0", "L1", "L2"]
    names = sorted(node.name for node in back.groups.values())
    assert names == ["Art", "Ink"]
    gp.check(back.groups, back.group_of, back.member_uids())


def test_a_nested_group_is_still_inside_its_parent():
    doc = _grouped()
    back, _ = _round_trip(doc)
    by_name = {node.name: uid for uid, node in back.groups.items()}
    inner, outer = by_name["Art"], by_name["Ink"]
    assert back.group_of.get(inner) == outer
    order = back.member_uids()
    assert gp.leaves_of(back.group_of, order, inner) == order[1:]


def test_a_groups_own_properties_survive_except_its_opacity():
    """Aseprite's UI offers a group no opacity, and ``_group_tree`` reads one
    back as 1.0 whatever the byte says -- so the byte is written as 255 and the
    loss is a line in the interop report rather than a folder that dims."""
    doc = _grouped()
    node = next(n for n in doc.groups.values() if n.name == "Ink")
    node.visible = False
    node.locked = True
    node.opacity = 0.5
    back, _ = _round_trip(doc)
    after = next(n for n in back.groups.values() if n.name == "Ink")
    assert after.visible is False
    assert after.locked is True
    assert after.opacity == 1.0


def test_a_grouped_animated_document_round_trips():
    """The cel chunks' layer indices count group rows, so a group above a track
    shifts every index below it -- the one arithmetic a still document with one
    frame could never catch."""
    doc = _animated()
    node = doc.group_layers([0, 1], name="All")
    assert node is not None
    back, _ = _round_trip(doc)
    assert [t.name for t in back.anim.tracks] == ["Background", "Ink"]
    for ti, track in enumerate(doc.anim.tracks):
        for fi, frame in enumerate(doc.anim.frames):
            before = doc.anim.cels.get((track.uid, frame.uid))
            after = back.anim.cels.get(
                (back.anim.tracks[ti].uid, back.anim.frames[fi].uid)
            )
            assert (before is None) == (after is None)
            if before is not None:
                assert np.array_equal(before.pixels, after.pixels)


# --- links -------------------------------------------------------------------


def test_a_linked_cel_comes_back_as_one_object():
    doc = _animated()
    anim = doc.anim
    background = anim.tracks[0].uid
    assert anim.cels[(background, anim.frames[0].uid)] is anim.cels[
        (background, anim.frames[1].uid)
    ]

    back, _ = _round_trip(doc)
    other = back.anim
    row = other.tracks[0].uid
    assert other.cels[(row, other.frames[0].uid)] is other.cels[
        (row, other.frames[1].uid)
    ]
    # And the unlinked row is still two objects, or the assertion above would
    # pass on a writer that linked everything.
    ink = other.tracks[1].uid
    assert other.cels[(ink, other.frames[0].uid)] is not other.cels[
        (ink, other.frames[1].uid)
    ]


def test_a_link_over_three_frames_stores_its_pixels_once():
    doc = _animated()
    doc.add_frame(link=True)
    back, _ = _round_trip(doc)
    other = back.anim
    row = other.tracks[0].uid
    shared = {id(other.cels[(row, frame.uid)]) for frame in other.frames}
    assert len(shared) == 1


# --- tags --------------------------------------------------------------------


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_a_tags_direction_survives(direction: str):
    doc = _animated()
    doc.anim.tags.append(
        Tag(name="walk", start=0, end=1, loop=True, direction=direction, repeat=3)
    )
    back, warnings = _round_trip(doc)
    assert warnings == []
    tag = back.anim.tags[0]
    assert tag.name == "walk"
    assert (tag.start, tag.end) == (0, 1)
    assert tag.direction == direction
    assert tag.repeat == 3


def test_several_tags_keep_their_order():
    doc = _animated()
    doc.add_frame()
    doc.anim.tags.append(Tag(name="idle", start=0, end=0))
    doc.anim.tags.append(Tag(name="walk", start=1, end=2))
    back, _ = _round_trip(doc)
    assert [tag.name for tag in back.anim.tags] == ["idle", "walk"]


# --- blend modes -------------------------------------------------------------


def test_the_two_blend_tables_are_the_same_nineteen_modes():
    """No refusal is needed for a blend mode, and this is why -- asserted
    rather than asserted-in-a-docstring, since a twentieth mode added to
    ``BLEND_MODES`` would otherwise be written as ``normal`` in silence."""
    assert set(BLEND_MODES) == set(asein._BLEND_BY_INDEX)
    assert len(BLEND_MODES) == len(asein._BLEND_BY_INDEX) == 19


@pytest.mark.parametrize("mode", BLEND_MODES)
def test_every_blend_mode_survives(mode: str):
    doc = _still()
    doc.stack[1].blend = mode
    back, warnings = _round_trip(doc)
    assert warnings == []
    assert back.stack[1].blend == mode


# --- stability and the file door ---------------------------------------------


def test_a_re_imported_document_writes_the_same_bytes_twice():
    """The fixed point. A writer whose output depends on anything the reader
    does not restore -- a uid, a dictionary order, a cel offset -- diverges
    here on the second pass and nowhere else."""
    doc = _animated()
    doc.group_layers([0, 1], name="All")
    doc.anim.tags.append(Tag(name="walk", start=0, end=1, repeat=2))

    once, _ = asein.document_from_aseprite(aseout.aseprite_bytes(doc))
    first = aseout.aseprite_bytes(once)
    twice, _ = asein.document_from_aseprite(first)
    assert aseout.aseprite_bytes(twice) == first


def test_an_indexed_document_is_byte_stable_too():
    doc = _indexed()
    once, _ = asein.document_from_aseprite(aseout.aseprite_bytes(doc))
    first = aseout.aseprite_bytes(once)
    twice, _ = asein.document_from_aseprite(first)
    assert aseout.aseprite_bytes(twice) == first


def test_write_aseprite_puts_those_bytes_on_disk(tmp_path: Path):
    doc = _still()
    path = tmp_path / "sprite.aseprite"
    aseout.write_aseprite(doc, path)
    assert path.read_bytes() == aseout.aseprite_bytes(doc)


def test_the_declared_file_size_is_the_files_own():
    """The header's first DWORD. Wrong, and our own reader warns the file may
    be truncated -- which is the ``warnings == []`` assertions above, said once
    where the field is."""
    data = aseout.aseprite_bytes(_still())
    assert int.from_bytes(data[:4], "little") == len(data)


# --- refusals ----------------------------------------------------------------


def test_too_many_frames_is_refused_by_name():
    doc = _animated()
    doc.anim.frames = [Frame()] * 65_536
    with pytest.raises(ValueError, match="65535 frames"):
        aseout.aseprite_bytes(doc)


def test_too_many_layers_is_refused_by_name():
    doc = _animated()
    doc.anim.tracks = [Track()] * 65_536
    with pytest.raises(ValueError, match="65535 layers"):
        aseout.aseprite_bytes(doc)


def test_a_palette_over_256_colours_is_refused_by_name():
    doc = _still()
    doc.palette = [(i, i, i, 255) for i in range(257)]
    with pytest.raises(ValueError, match="256 colours"):
        aseout.aseprite_bytes(doc)


def test_a_colour_mode_this_writer_has_no_depth_for_is_refused_by_name():
    doc = _still()
    doc.color_mode = "cmyk"
    with pytest.raises(ValueError, match="cmyk"):
        aseout.aseprite_bytes(doc)


def test_an_indexed_layer_with_no_index_plane_is_refused_by_name():
    doc = _indexed()
    doc.stack[0].indices = None
    with pytest.raises(ValueError, match="Background"):
        aseout.aseprite_bytes(doc)


def test_a_canvas_too_big_for_the_format_is_refused_by_name():
    doc = _still()
    doc.stack.layers[0].pixels = np.zeros((4, 70_000, 4), dtype=np.uint8)
    doc.stack.layers[1].pixels = np.zeros((4, 70_000, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="65535"):
        aseout.aseprite_bytes(doc)
