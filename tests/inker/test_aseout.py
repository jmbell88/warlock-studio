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

import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from warlock.studio.inker import asein, aseout
from warlock.studio.inker import groups as gp
from warlock.studio.inker import index_plane as ixp
from warlock.studio.inker.animation import DIRECTIONS, Frame, Tag, Track
from warlock.studio.inker.composite import BLEND_MODES
from warlock.studio.inker.document import Document
from warlock.studio.inker.slices import SliceKey
from warlock.studio.inker.tiles import TilemapCel, materialize, strip
from warlock.studio.tilegrid import gid

RED = (255, 0, 0, 255)
GREEN = (0, 255, 0, 255)
BLUE = (0, 0, 255, 255)
MAGENTA = (255, 0, 255, 255)


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


def test_a_frame_holding_nothing_at_all_round_trips():
    """A frame with **no chunks**, which the sparse grid allows and which the
    two chunk-count fields disagree about most cheaply: 0 in the modern DWORD
    falls through to the legacy WORD, so a WORD carrying the "read the other
    one" sentinel would be read as sixty-five thousand chunks that are not
    there. Both fields carry the count, saturated, and this is the frame that
    proves it."""
    doc = _animated()
    empty = doc.add_frame()
    assert not [key for key in doc.anim.cels if key[1] == empty.uid]

    back, warnings = _round_trip(doc)
    assert warnings == []
    assert len(back.anim.frames) == 3
    last = back.anim.frames[2].uid
    assert not [key for key in back.anim.cels if key[1] == last]


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


def _as_stored(pixels: np.ndarray) -> np.ndarray:
    """``pixels`` as a grayscale file can hold them: ``(value, alpha)``.

    A grayscale cel stores two channels, so the *dead* RGB a fully transparent
    pixel carries survives only as its red. On a visible pixel this is the
    identity -- ``r == g == b`` is the mode's whole constraint -- so one helper
    covers both halves of the comparison and the assertions below say which
    half they mean.
    """
    out = pixels.copy()
    out[..., 1] = out[..., 0]
    out[..., 2] = out[..., 0]
    return out


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


def test_erasing_before_converting_to_grayscale_still_saves():
    """The regression. The eraser cuts alpha and leaves the colour it was drawn
    in behind it, and ``indexed.grayscale`` returns a fully transparent pixel
    *verbatim* -- so this document is entirely funnel-legal and carries blue
    under alpha 0. A greyness check that did not mask itself to the visible
    pixels would refuse to save it, which is a drawing the user cannot get out
    of the app over pixels nobody can see.
    """
    doc = Document.blank(8, 8)
    doc.stack[0].pixels[:, :] = BLUE
    doc.invalidate_all()
    doc.begin_stroke((0, 0), BLUE, size=4, hardness=1.0, mode="erase")
    doc.stroke_to((7, 7))
    doc.end_stroke()
    doc.convert_to_grayscale()
    erased = doc.stack[0].pixels[..., 3] == 0
    assert erased.any()
    assert (doc.stack[0].pixels[..., 2][erased] != doc.stack[0].pixels[..., 0][erased]).any()

    back, warnings = _round_trip(doc)
    assert warnings == []
    assert back.color_mode == "grayscale"
    # The visible half is bit-exact; the invisible half keeps its alpha and
    # loses two channels of colour that were never drawn.
    visible = ~erased
    assert np.array_equal(back.stack[0].pixels[visible], doc.stack[0].pixels[visible])
    assert np.array_equal(back.stack[0].pixels, _as_stored(doc.stack[0].pixels))


def test_a_grayscale_document_holding_a_visible_colour_is_refused_by_name():
    """Reachable only past the funnel, which enforces ``r == g == b`` on every
    write -- so the check is what stops two of the three channels being thrown
    away without a word."""
    doc = Document.blank(4, 4)
    doc.invalidate_all()
    doc.convert_to_grayscale()
    doc.stack[0].pixels[1, 1] = BLUE

    with pytest.raises(ValueError, match="Background"):
        aseout.aseprite_bytes(doc)


def test_an_invisible_colour_is_not_a_grayscale_violation():
    """The mask's other side, said directly: the same blue at alpha 0 saves."""
    doc = Document.blank(4, 4)
    doc.invalidate_all()
    doc.convert_to_grayscale()
    doc.stack[0].pixels[1, 1] = (0, 0, 255, 0)

    back, _ = _round_trip(doc)
    assert tuple(back.stack[0].pixels[1, 1]) == (0, 0, 0, 0)


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


def test_a_palette_constrained_rgb_document_loses_its_palette():
    """**A pinned loss, not a defect** (divergence 19, and Task 5's interop
    report is where it is written down).

    The chunks are written -- a file Aseprite reads has its colour table, which
    is what the second half of this test asserts against the parse. What does
    not come back is the *constraint*: ``document_from_aseprite`` installs a
    palette only at indexed depth, and it must keep doing so, because a real
    RGB ``.aseprite`` legitimately carries a palette chunk and adopting it would
    silently put the whole editor into palette-constrained mode over a table
    nobody asked to be limited by.
    """
    doc = _still()
    doc.set_palette([(0, 0, 0, 255), RED, BLUE])
    assert doc.color_mode == "rgb" and doc.palette

    sprite = asein.parse(aseout.aseprite_bytes(doc))
    assert sprite.palette == [(0, 0, 0, 255), RED, BLUE]

    back, warnings = _round_trip(doc)
    assert warnings == []
    assert back.color_mode == "rgb"
    assert not back.palette


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

    Each layer is painted a *different* colour on purpose. Two group rows sit
    between L0 and L1 in the emitted layer list, so every cel index below them
    is shifted by two -- and an off-by-two there puts each layer's pixels on the
    row above or below, which names alone cannot see (they come from the layer
    chunks, which are not the thing being shifted).
    """
    doc = Document.blank(4, 4)
    doc.stack[0].name = "L0"
    doc.stack[0].pixels[:, :] = (10, 0, 0, 255)
    for i in (1, 2):
        doc.add_layer(f"L{i}").pixels[:, :] = (0, 10 * i, 0, 255)
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
    # The pixels, not just the names: two group rows shift every cel index
    # below them, and only distinct colours per layer can see that landing
    # wrong.
    for before, after in zip(doc.stack, back.stack, strict=True):
        assert np.array_equal(before.pixels, after.pixels)


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


def test_an_indexed_document_with_no_palette_is_refused_by_name():
    doc = _still()
    doc.color_mode = "indexed"
    doc.palette = None
    with pytest.raises(ValueError, match="nothing to write its pixels through"):
        aseout.aseprite_bytes(doc)


def test_a_transparent_index_no_header_byte_can_name_is_refused_by_name():
    doc = _indexed()
    doc.transparent_index = 300
    with pytest.raises(ValueError, match="300"):
        aseout.aseprite_bytes(doc)


def test_a_cel_that_is_not_canvas_sized_is_refused_by_name():
    """Layers are canvas-sized here (``layers.py``'s first decision), so this
    is a broken document rather than a small cel -- and a writer that shrugged
    would put a plane on the file whose declared size was the canvas's."""
    doc = _animated()
    cel = doc.anim.cels[(doc.anim.tracks[1].uid, doc.anim.frames[1].uid)]
    cel.pixels = np.zeros((2, 2, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="canvas-sized"):
        aseout.aseprite_bytes(doc)


def test_a_tag_repeating_past_a_word_is_refused_by_name():
    """A count that wrapped would silently stop a clip that was set to run, and
    a function returning bytes has no warning to raise instead."""
    doc = _animated()
    doc.anim.tags.append(Tag(name="walk", start=0, end=1, repeat=70_000))
    with pytest.raises(ValueError, match="walk"):
        aseout.aseprite_bytes(doc)


def test_a_blend_mode_the_format_has_no_number_for_is_refused_by_name():
    """Dead code while the two tables hold the same nineteen modes -- and the
    point of it is the day they do not, since the alternative is writing the
    layer out as ``normal`` in silence. ``Track.blend`` is unvalidated, which is
    how a mode gets this far."""
    doc = _animated()
    doc.anim.tracks[1].blend = "glow"
    with pytest.raises(ValueError, match="glow"):
        aseout.aseprite_bytes(doc)


def test_a_canvas_too_big_for_the_format_is_refused_by_name():
    doc = _still()
    doc.stack.layers[0].pixels = np.zeros((4, 70_000, 4), dtype=np.uint8)
    doc.stack.layers[1].pixels = np.zeros((4, 70_000, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="65535"):
        aseout.aseprite_bytes(doc)


# --- tilesets, tilemap layers and tilemap cels --------------------------------


def _tile(colour: tuple[int, int, int, int], w: int = 4, h: int = 4) -> np.ndarray:
    tile = np.zeros((h, w, 4), dtype=np.uint8)
    tile[..., 0], tile[..., 1], tile[..., 2], tile[..., 3] = colour
    return tile


def _tileset(*colours: tuple[int, int, int, int], name: str = "tiles"):
    """A vertical strip whose local id 0 is the required blank tile."""
    blank = np.zeros((4, 4, 4), dtype=np.uint8)
    built = strip(np.stack([blank, *[_tile(c) for c in colours]], axis=0))
    return replace(built, name=name)


def _tilemap_still() -> Document:
    """8x8 at 4x4 tiles (a 2x2 grid): a raster Background under a tilemap
    layer holding one plain and one horizontally flipped placement."""
    doc = Document.blank(8, 8)
    doc.stack[0].name = "Background"
    doc.stack[0].pixels[:, :] = RED
    doc.invalidate_all()
    slot = doc.add_tileset(_tileset(RED, GREEN))
    cel = doc.add_tilemap_layer(slot.uid, name="Tiles")
    doc.place_tiles(
        cel.uid, (0, 0), np.array([[1, gid.compose(2, flip_h=True)]], dtype=np.uint32)
    )
    return doc


def _tilemap_animated() -> Document:
    """The same shape over two frames, the tilemap cel linked into both."""
    doc = Document.blank(8, 8)
    doc.stack[0].name = "Background"
    doc.stack[0].pixels[:, :] = RED
    doc.invalidate_all()
    doc.ensure_animation()
    slot = doc.add_tileset(_tileset(RED, GREEN))
    cel = doc.add_tilemap_layer(slot.uid, name="Tiles")
    doc.place_tiles(
        cel.uid,
        (0, 0),
        np.array([[gid.compose(2, flip_v=True), 1]], dtype=np.uint32),
    )
    doc.add_frame(link=True)
    return doc


def _synced(doc: Document) -> None:
    """Every tilemap cel's pixels still agree with ``materialize(refs, ts)``."""
    layers = doc.stack if doc.anim is None else doc.anim.unique_cel_layers()
    for layer in layers:
        if isinstance(layer, TilemapCel):
            slot = doc.tileset_slot(layer.tileset_uid)
            assert np.array_equal(
                layer.pixels, materialize(layer.refs, slot.tileset, layer.size)
            )


def test_a_still_tilemap_document_round_trips_refs_bit_exact():
    doc = _tilemap_still()
    back, warnings = _round_trip(doc)

    assert warnings == []
    assert [layer.name for layer in back.stack] == ["Background", "Tiles"]
    cel = back.stack[1]
    assert isinstance(cel, TilemapCel)
    assert np.array_equal(cel.refs, doc.stack[1].refs)
    assert cel.tileset_uid == back.tilesets[0].uid
    _synced(back)


def test_a_tilemap_cels_flag_bits_survive_the_declared_masks():
    """The flags ride the same uint32 as the id, so a writer that dropped them
    -- or declared masks that did not match the bits it wrote -- would land a
    mirrored tile the right way round and nothing else would notice."""
    doc = _tilemap_still()
    flipped = int(doc.stack[1].refs[0, 1])
    assert flipped & gid.FLIP_H

    back, _ = _round_trip(doc)
    assert int(back.stack[1].refs[0, 1]) == flipped


def test_the_tilemap_cel_declares_our_own_gid_masks():
    """Declared, not assumed: ``asein._remap_tile_refs`` remaps onto our bits
    using the masks *the chunk itself carries*, so writing the bits without
    saying which they are reads back as a tile id of two billion."""
    data = aseout.aseprite_bytes(_tilemap_still())
    declared = struct.pack("<IIII", gid.GID_MASK, gid.FLIP_H, gid.FLIP_V, gid.FLIP_D)
    assert declared in data


def test_a_tileset_strip_is_bit_exact():
    doc = _tilemap_still()
    back, _ = _round_trip(doc)
    assert len(back.tilesets) == 1
    before = doc.tilesets[0].tileset
    after = back.tilesets[0].tileset
    assert (after.tile_w, after.tile_h) == (before.tile_w, before.tile_h)
    assert after.tile_count == before.tile_count
    assert np.array_equal(after.pixels, before.pixels)
    assert after.name == before.name


def test_an_animated_tilemap_document_keeps_its_binding_and_its_link():
    doc = _tilemap_animated()
    back, warnings = _round_trip(doc)

    assert warnings == []
    track = back.anim.tracks[1]
    assert track.tileset_uid == back.tilesets[0].uid
    first = back.anim.cels[(track.uid, back.anim.frames[0].uid)]
    second = back.anim.cels[(track.uid, back.anim.frames[1].uid)]
    assert isinstance(first, TilemapCel)
    assert second is first, "a linked tilemap cel must come back as one object"
    original = doc.anim.cels[(doc.anim.tracks[1].uid, doc.anim.frames[0].uid)]
    assert np.array_equal(first.refs, original.refs)
    _synced(back)


def test_a_spare_tileset_survives_with_no_tilemap_layer_at_all():
    """``ora``'s "not garbage" rule, restated for this writer: a tileset the
    user has built but not placed yet is document state, and a save that
    dropped it would lose the work at the moment it is most likely to exist."""
    doc = _still()
    doc.add_tileset(_tileset(GREEN, name="spare"))

    back, warnings = _round_trip(doc)
    assert warnings == []
    assert len(back.tilesets) == 1
    assert back.tilesets[0].tileset.name == "spare"
    assert np.array_equal(
        back.tilesets[0].tileset.pixels, doc.tilesets[0].tileset.pixels
    )
    assert not [layer for layer in back.stack if isinstance(layer, TilemapCel)]


def test_two_tilesets_come_back_in_the_order_they_were_written():
    doc = _tilemap_still()
    doc.add_tileset(_tileset(BLUE, name="second"))
    back, _ = _round_trip(doc)
    assert [slot.tileset.name for slot in back.tilesets] == ["tiles", "second"]
    assert back.stack[1].tileset_uid == back.tilesets[0].uid


def _indexed_tilemap() -> Document:
    """An indexed document that then grew a tilemap layer.

    The only order that reaches this state: ``_refuse_tilemap_convert`` closes
    the other one, because converting a document whose picture is *derived*
    would convert the materialisation and leave the tileset behind it.
    """
    doc = Document.blank(8, 8)
    doc.stack[0].name = "Background"
    doc.stack[0].pixels[:, :] = RED
    doc.invalidate_all()
    doc.convert_to_indexed([(0, 0, 0, 0), RED, GREEN], transparent=0)
    slot = doc.add_tileset(_tileset(RED, GREEN))
    cel = doc.add_tilemap_layer(slot.uid, name="Tiles")
    doc.place_tiles(
        cel.uid, (0, 0), np.array([[1, gid.compose(2, flip_h=True)]], dtype=np.uint32)
    )
    return doc


def test_an_indexed_tilemap_document_round_trips_exact():
    """The decision this task turned on: the format stores a tileset's pixels
    at the *sprite's* depth, so an 8-bit sprite carries an 8-bit strip, while
    this model keeps every strip RGBA (the Wave 3 divergence). The strip is
    resolved back through the palette on the way out -- the exact inverse of
    ``asein._decode_tileset`` -- so what comes back is the same picture."""
    doc = _indexed_tilemap()
    back, warnings = _round_trip(doc)

    assert warnings == []
    assert back.color_mode == "indexed"
    assert np.array_equal(
        back.tilesets[0].tileset.pixels, doc.tilesets[0].tileset.pixels
    )
    assert np.array_equal(back.stack[1].refs, doc.stack[1].refs)
    _synced(back)


def test_a_tilemap_cel_in_an_indexed_document_never_needs_an_index_plane():
    """A tilemap cel carries no ``indices`` in any colour mode -- its picture
    is derived -- so the cel chunk that would have demanded one is never the
    chunk this layer writes."""
    doc = _indexed_tilemap()
    assert doc.stack[1].indices is None
    back, _ = _round_trip(doc)
    assert back.stack[1].indices is None


def test_an_off_palette_tileset_strip_is_refused_by_name():
    """The one hole in resolving strips through the palette, and it is named:
    a strip *imported* into an indexed document (from a ``.tsx``, from
    Plotter) can hold a colour the document's table has no slot for. The
    funnel resolves everything a user paints, so this is the import case
    alone -- and the alternative to refusing is a nearest match that would
    silently repaint somebody else's atlas."""
    doc = _indexed_tilemap()
    doc.add_tileset(_tileset(BLUE, name="stray"))
    with pytest.raises(ValueError, match="stray"):
        aseout.aseprite_bytes(doc)


def test_a_grayscale_tileset_strip_holding_a_visible_colour_is_refused_by_name():
    doc = Document.blank(8, 8)
    doc.invalidate_all()
    doc.convert_to_grayscale()
    doc.add_tileset(_tileset(BLUE, name="colourful"))
    with pytest.raises(ValueError, match="colourful"):
        aseout.aseprite_bytes(doc)


def test_a_tilemap_track_whose_cel_holds_no_refs_is_refused_by_name():
    """A layer declared kind 2 whose cel is an ordinary raster one would be
    read back through ``_build_tilemap_cel`` with no refs at all -- an empty
    grid where a drawing was. Refused rather than emptied."""
    from warlock.studio.inker.layers import Layer

    doc = _tilemap_animated()
    anim = doc.anim
    key = (anim.tracks[1].uid, anim.frames[0].uid)
    anim.cels[key] = Layer.empty(8, 8, "Tiles")
    with pytest.raises(ValueError, match="Tiles"):
        aseout.aseprite_bytes(doc)


def test_a_still_tilemap_cel_with_no_tileset_binding_is_refused_by_name():
    """``TilemapCel.tileset_uid`` defaults to ``0``, and every real
    construction site sets it to a real slot uid -- but a layer built by hand
    with ``refs`` set and ``tileset_uid=None`` used to hit ``int(None)``, a
    bare ``TypeError`` naming nothing. This is the still-document half of
    ``_rows``' dangling-track refusal, one layer earlier."""
    doc = _tilemap_still()
    doc.stack[1].tileset_uid = None
    with pytest.raises(ValueError, match="Tiles"):
        aseout.aseprite_bytes(doc)


def test_a_zero_tile_tileset_is_refused_by_name():
    """Defensive: a real ``Tileset`` cannot actually hold zero tiles (its own
    validation requires at least one column and one row), so this exercises
    ``_strip_bytes`` directly against a stand-in that lies about it, the way a
    future caller of the private function might. Without the guard,
    ``np.concatenate([])`` raises a bare, unnamed ``ValueError``."""

    class _EmptyTileset:
        name = "hollow"
        tile_count = 0

        def tile_pixels(self, index):  # pragma: no cover - must not be called
            raise AssertionError("a zero-tile strip has no tile to ask for")

    with pytest.raises(ValueError, match="hollow"):
        aseout._strip_bytes(_EmptyTileset(), "rgb", None, 0)


def test_a_tilemap_document_is_byte_stable():
    doc = _tilemap_animated()
    once, _ = asein.document_from_aseprite(aseout.aseprite_bytes(doc))
    first = aseout.aseprite_bytes(once)
    twice, _ = asein.document_from_aseprite(first)
    assert aseout.aseprite_bytes(twice) == first


# --- slices -------------------------------------------------------------------


def test_a_slice_round_trips_with_its_pivot_and_its_nine_patch():
    doc = _still()
    entry = doc.add_slice(
        (1, 1, 4, 4), name="Button", pivot=(1.0, 2.0), center=(1, 1, 2, 2)
    )
    back, warnings = _round_trip(doc)

    assert warnings == []
    assert len(back.slices) == 1
    got = back.slices[0]
    assert got.name == "Button"
    assert got.bounds == entry.bounds
    assert got.pivot == entry.pivot
    assert got.center == entry.center


def test_a_plain_slice_comes_back_with_neither_flag_set():
    """The two flags are written only when the slice carries the thing, so a
    plain rectangle reads back as one -- a zero centre is a real nine-patch as
    far as the reader is concerned, and inventing one would put a stretch box
    on every slice in the document."""
    doc = _still()
    doc.add_slice((0, 0, 2, 2), name="Plain")
    back, _ = _round_trip(doc)
    got = back.slices[0]
    assert got.pivot is None
    assert got.center is None


def test_several_slices_keep_their_order_and_their_names():
    doc = _still()
    doc.add_slice((0, 0, 2, 2), name="One")
    doc.add_slice((1, 1, 3, 3), name="Two")
    back, _ = _round_trip(doc)
    assert [entry.name for entry in back.slices] == ["One", "Two"]


def test_a_slice_keyed_per_frame_round_trips_frame_for_frame():
    doc = _animated()
    doc.add_frame()
    entry = doc.add_slice((0, 0, 2, 2), name="Head")
    doc.set_slice_key(
        entry.uid, doc.anim.frames[1].uid, key=SliceKey(bounds=(1, 1, 3, 3))
    )
    doc.set_slice_key(
        entry.uid, doc.anim.frames[2].uid, key=SliceKey(bounds=(1, 1, 3, 3))
    )
    back, warnings = _round_trip(doc)

    assert warnings == []
    got = back.slices[0]
    seen = [got.at(frame.uid).bounds for frame in back.anim.frames]
    want = [entry.at(frame.uid).bounds for frame in doc.anim.frames]
    assert seen == want == [(0, 0, 2, 2), (1, 1, 3, 3), (1, 1, 3, 3)]


def test_a_key_that_returns_to_the_base_is_written_as_its_own_run():
    """Aseprite's keys run to the *end* of the timeline, this model's override
    one frame -- so the emission is run-length: a key wherever the resolved
    rectangle changes. A frame that goes back to the base needs one of its
    own, or the middle frame's rectangle would run to the end of the clip."""
    doc = _animated()
    doc.add_frame()
    entry = doc.add_slice((0, 0, 2, 2), name="Head")
    doc.set_slice_key(
        entry.uid, doc.anim.frames[1].uid, key=SliceKey(bounds=(1, 1, 3, 3))
    )
    back, _ = _round_trip(doc)
    got = back.slices[0]
    assert [got.at(frame.uid).bounds for frame in back.anim.frames] == [
        (0, 0, 2, 2),
        (1, 1, 3, 3),
        (0, 0, 2, 2),
    ]


def test_a_key_carrying_no_pivot_inherits_the_slices_own():
    """The format stores pivot and nine-patch *presence* once per slice, not
    per key, so a document whose keys disagree is one the file cannot spell.
    The union decides the two flags and a key missing what the chunk declares
    is written with the slice's own value -- never a zero, which the reader
    could not tell from a rectangle somebody meant."""
    doc = _animated()
    entry = doc.add_slice((0, 0, 2, 2), name="Head", pivot=(1.0, 1.0))
    doc.set_slice_key(
        entry.uid, doc.anim.frames[1].uid, key=SliceKey(bounds=(1, 1, 3, 3))
    )
    assert entry.keys[doc.anim.frames[1].uid].pivot is None

    back, _ = _round_trip(doc)
    got = back.slices[0]
    assert got.at(back.anim.frames[0].uid).pivot == (1.0, 1.0)
    assert got.at(back.anim.frames[1].uid).pivot == (1.0, 1.0)


def test_a_slice_on_a_still_document_round_trips():
    doc = _still()
    doc.add_slice((0, 0, 3, 3), name="Whole", center=(1, 1, 2, 2))
    back, _ = _round_trip(doc)
    assert back.anim is None
    assert back.slices[0].center == doc.slices[0].center


def test_a_document_with_slices_is_byte_stable():
    doc = _animated()
    entry = doc.add_slice(
        (0, 0, 2, 2), name="Head", pivot=(1.0, 1.0), center=(0, 0, 1, 1)
    )
    doc.set_slice_key(
        entry.uid, doc.anim.frames[1].uid, key=SliceKey(bounds=(1, 1, 3, 3))
    )
    once, _ = asein.document_from_aseprite(aseout.aseprite_bytes(doc))
    first = aseout.aseprite_bytes(once)
    twice, _ = asein.document_from_aseprite(first)
    assert aseout.aseprite_bytes(twice) == first


# --- an indexed strip is resolved the way every other plane is ---------------
#
# The tile funnel constrains a strip with ``indexed.snap``, which -- unlike the
# ``index_plane.resolve`` every *raster* plane goes through -- leaves alpha
# untouched and returns a fully transparent pixel verbatim. So an ordinary
# gesture leaves strip pixels no RGBA equality can place, and a writer that
# demanded one refused to save documents the editor itself had just produced.
# These are those documents.


def _indexed_tile_doc(palette, transparent=0, behavior="stack") -> Document:
    """An indexed 8x8 document with a tilemap layer over a two-colour strip,
    with the tile behaviour set so a paint reaches the tileset at all (Manual
    -- the field's default -- re-derives every write away)."""
    doc = Document.blank(8, 8)
    doc.stack[0].name = "Background"
    doc.stack[0].pixels[:, :] = RED
    doc.invalidate_all()
    doc.convert_to_indexed(palette, transparent=transparent)
    slot = doc.add_tileset(_tileset(RED, GREEN))
    cel = doc.add_tilemap_layer(slot.uid, name="Tiles")
    doc.place_tiles(cel.uid, (0, 0), np.array([[1, 2]], dtype=np.uint32))
    doc.set_active_layer(doc.stack.index_of(cel.uid))
    doc.tile_behavior = behavior
    return doc


def _as_stored_indexed(pixels: np.ndarray, doc: Document) -> np.ndarray:
    """``pixels`` as an 8-bit sprite can hold them.

    Written through ``index_plane`` and not through ``aseout``'s own helper on
    purpose: the claim being asserted is that a strip normalises **the way
    every other plane in an indexed document already did**, so the comparison
    has to be built from the document's own machinery rather than from the
    writer's -- otherwise it would only assert that two copies of one idea
    agree. There is no per-pixel alpha at one byte per pixel, so a
    sub-threshold pixel is the hole and a visible one takes its slot's alpha.
    """
    table = ixp.lut(doc.palette, doc.transparent_index)
    plane = ixp.resolve(pixels, table, doc.transparent_index)
    return ixp.materialize(plane, table)


def test_a_half_coverage_dab_on_a_tile_still_saves():
    """The reproduction. A soft dab onto a blank cell leaves ``(0, 255, 0,
    128)`` in the strip -- a *visible* pixel whose RGB is a palette colour and
    whose alpha is nothing the table can hold. Alpha decides first, exactly as
    ``resolve`` decides it, so the pixel is placed and the document saves."""
    doc = _indexed_tile_doc([(0, 0, 0, 0), RED, GREEN])
    doc.write_colour((0, 4, 4, 8), GREEN, np.full((4, 4), 0.5, dtype=np.float32))
    strip_px = doc.tilesets[0].tileset.pixels
    assert (strip_px[..., 3] == 128).any(), "the fixture must produce a soft pixel"

    back, warnings = _round_trip(doc)
    assert warnings == []
    assert np.array_equal(
        back.tilesets[0].tileset.pixels, _as_stored_indexed(strip_px, doc)
    )


def test_an_erased_tile_still_saves():
    """The eraser cuts alpha and leaves the colour it was drawn in behind, so
    the strip carries ``(255, 0, 0, 0)`` -- invisible, and refused by an RGBA
    equality that had no entry for it. It writes the transparent index and
    comes back as the hole, which is the grayscale path's own bargain."""
    doc = _indexed_tile_doc([(0, 0, 0, 0), RED, GREEN], behavior="auto")
    doc.begin_stroke((1, 1), RED, size=3, hardness=1.0, mode="erase")
    doc.stroke_to((2, 2))
    doc.end_stroke()
    strip_px = doc.tilesets[0].tileset.pixels
    dead = (strip_px[..., 3] == 0) & (strip_px[..., 0] != 0)
    assert dead.any(), "the fixture must leave dead colour under alpha 0"

    back, warnings = _round_trip(doc)
    assert warnings == []
    assert np.array_equal(
        back.tilesets[0].tileset.pixels, _as_stored_indexed(strip_px, doc)
    )
    assert not back.tilesets[0].tileset.pixels[dead].any()


def test_a_partly_erased_pixel_takes_its_slots_alpha():
    """The third shape the eraser leaves: alpha 235, above the threshold, on a
    palette colour. It is visible, so it takes the slot -- and the slot's own
    alpha with it, which is what one byte per pixel means."""
    doc = _indexed_tile_doc([(0, 0, 0, 0), RED, GREEN], behavior="auto")
    doc.begin_stroke((1, 1), RED, size=3, hardness=0.2, mode="erase")
    doc.stroke_to((2, 2))
    doc.end_stroke()
    strip_px = doc.tilesets[0].tileset.pixels
    soft = (strip_px[..., 3] >= 128) & (strip_px[..., 3] < 255)
    assert soft.any(), "the fixture must leave a partly erased pixel"

    back, _ = _round_trip(doc)
    assert np.array_equal(
        back.tilesets[0].tileset.pixels, _as_stored_indexed(strip_px, doc)
    )
    assert (back.tilesets[0].tileset.pixels[soft][:, 3] == 255).all()


def test_an_off_palette_visible_colour_still_refuses_by_name():
    """The refusal that must survive the fix: an *imported* atlas holding a
    visible colour this document has no slot for. Nothing a user paints can
    reach it, which is exactly why nearest-matching it would be repainting
    somebody else's tiles."""
    doc = _indexed_tile_doc([(0, 0, 0, 0), RED, GREEN])
    doc.add_tileset(_tileset(BLUE, name="stray"))
    with pytest.raises(ValueError, match="stray"):
        aseout.aseprite_bytes(doc)


def test_a_visible_pixel_in_the_transparent_slots_colour_is_refused_by_name():
    """The one state ``indexed.snap`` and ``index_plane.resolve`` disagree
    about: ``snap`` matches against every slot, so a paint near the transparent
    slot's *stored* colour lands on it, where ``resolve`` would have excluded
    it and no raster plane in the same document can hold it. There is no honest
    answer -- writing that slot turns a pixel the user just painted into a hole
    -- so it is named as its own case rather than folded into "off palette"."""
    doc = _indexed_tile_doc([MAGENTA, RED, GREEN], transparent=0)
    doc.write_colour(
        (0, 4, 4, 8), (250, 0, 250, 255), np.ones((4, 4), dtype=np.float32)
    )
    assert (doc.tilesets[0].tileset.pixels[..., :3] == MAGENTA[:3]).all(axis=-1).any()

    with pytest.raises(ValueError, match="transparent slot"):
        aseout.aseprite_bytes(doc)


def test_a_palette_of_one_transparent_colour_still_saves():
    """``index_plane.resolve``'s own allowance for the degenerate document,
    mirrored: with nothing but the transparent slot there is no candidate set,
    so the whole table becomes one and every pixel lands on slot 0. A refusal
    here would be a refusal over a picture with one colour and one meaning."""
    doc = Document.blank(8, 8)
    doc.invalidate_all()
    doc.convert_to_indexed([(0, 0, 0, 0)], transparent=0)
    doc.add_tileset(_tileset(name="flat"))
    back, warnings = _round_trip(doc)
    assert warnings == []
    assert not back.tilesets[0].tileset.pixels.any()


def test_a_key_only_pivot_becomes_the_slices_own():
    """The base carries no pivot and a key does. The format stores that
    presence once per slice, so there is no writing it per key -- and the
    fallback must not be a zero, which the reader cannot tell from a pivot
    somebody set. The first value the slice carries becomes every frame's."""
    doc = _animated()
    entry = doc.add_slice((0, 0, 2, 2), name="Head")
    doc.set_slice_key(
        entry.uid,
        doc.anim.frames[1].uid,
        key=SliceKey(bounds=(1, 1, 3, 3), pivot=(2.0, 3.0)),
    )
    assert entry.pivot is None

    back, warnings = _round_trip(doc)
    assert warnings == []
    got = back.slices[0]
    assert got.pivot == (2.0, 3.0)
    assert [got.at(frame.uid).pivot for frame in back.anim.frames] == [
        (2.0, 3.0),
        (2.0, 3.0),
    ]


def test_a_key_only_nine_patch_becomes_the_slices_own():
    """The centre's half of the same rule, and the one a zero fallback would
    have shown most plainly: ``(0, 0, 0, 0)`` is an empty stretch box, which a
    nine-slice panel draws quite differently from the one that was set."""
    doc = _animated()
    entry = doc.add_slice((0, 0, 4, 4), name="Panel")
    doc.set_slice_key(
        entry.uid,
        doc.anim.frames[1].uid,
        key=SliceKey(bounds=(0, 0, 4, 4), center=(1, 1, 3, 3)),
    )
    assert entry.center is None

    back, _ = _round_trip(doc)
    got = back.slices[0]
    assert got.center == (1, 1, 3, 3)
    assert all(got.at(frame.uid).center == (1, 1, 3, 3) for frame in back.anim.frames)


def test_a_key_only_pivot_document_is_byte_stable():
    doc = _animated()
    entry = doc.add_slice((0, 0, 2, 2), name="Head")
    doc.set_slice_key(
        entry.uid,
        doc.anim.frames[1].uid,
        key=SliceKey(bounds=(1, 1, 3, 3), pivot=(2.0, 3.0)),
    )
    once, _ = asein.document_from_aseprite(aseout.aseprite_bytes(doc))
    first = aseout.aseprite_bytes(once)
    twice, _ = asein.document_from_aseprite(first)
    assert aseout.aseprite_bytes(twice) == first
