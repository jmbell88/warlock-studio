"""The GIF export, asserted by reading the file back.

Every property below is one the format can quietly get wrong: a duration
truncated to the tick below, a frame drawn over its predecessor so a moving
subject smears, a palette taken from frame one so a colour appearing later is
mapped onto the nearest thing that was there at the start. None of them raises,
and none of them is visible until somebody plays the file somewhere else.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.inker import gifout

Image = pytest.importorskip("PIL.Image")


def _plane(size, colour, *, box=None):
    """A transparent canvas with an opaque rectangle of ``colour`` in it."""
    out = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    x0, y0, x1, y1 = box or (0, 0, size[0], size[1])
    out[y0:y1, x0:x1] = colour
    return out


def _frames_of(path):
    with Image.open(path) as im:
        return [
            (im.seek(i) or im.convert("RGBA").copy(), im.info.get("duration"))
            for i in range(im.n_frames)
        ]


def test_a_clip_round_trips_its_frame_count_size_and_colours(tmp_path):
    dest = tmp_path / "clip.gif"
    red = _plane((8, 6), (220, 30, 30, 255), box=(0, 0, 4, 6))
    blue = _plane((8, 6), (30, 30, 220, 255), box=(4, 0, 8, 6))
    gifout.write_gif(dest, [red, blue], [100, 200])

    frames = _frames_of(dest)
    assert len(frames) == 2
    assert frames[0][0].size == (8, 6)
    assert frames[0][0].getpixel((1, 1))[:3] == (220, 30, 30)
    # The second frame's colour is not in the first frame's palette, which is
    # exactly what a shared palette would have got wrong.
    assert frames[1][0].getpixel((5, 1))[:3] == (30, 30, 220)


def test_durations_survive_at_the_formats_own_resolution(tmp_path):
    dest = tmp_path / "clip.gif"
    # Three *distinct* frames. Pillow collapses consecutive identical ones into
    # a single frame holding the sum of their durations -- which is a correct
    # optimisation and would make this assertion about that instead.
    frames = [
        _plane((6, 4), (255, 255, 255, 255), box=(i, 0, i + 2, 4)) for i in range(3)
    ]
    gifout.write_gif(dest, frames, [100, 40, 250])
    assert [d for _, d in _frames_of(dest)] == [100, 40, 250]


def test_a_duration_is_rounded_rather_than_truncated():
    """Pillow truncates, so 15 ms would be written as 10 and a clip drawn at
    66 fps would play a third fast. The floor is one tick, because a zero-delay
    frame is left to the viewer and most of them pick 100 ms -- the opposite of
    what a very short frame meant."""
    assert gifout.tick_durations([15, 14, 1, 0, 104]) == [20, 10, 10, 10, 100]


def test_transparency_is_thresholded_rather_than_dropped(tmp_path):
    """A GIF pixel is an index and one index may be transparent, so there is no
    partial coverage to keep. What must not happen is the alpha being ignored
    and the whole canvas coming back opaque."""
    dest = tmp_path / "clip.gif"
    plane = _plane((6, 6), (10, 200, 10, 255), box=(2, 2, 4, 4))
    gifout.write_gif(dest, [plane], [100])

    frame = _frames_of(dest)[0][0]
    assert frame.getpixel((0, 0))[3] == 0
    assert frame.getpixel((3, 3))[3] == 255


def test_a_half_covered_pixel_falls_on_one_side_of_the_threshold(tmp_path):
    dest = tmp_path / "clip.gif"
    plane = _plane((4, 1), (255, 0, 0, 255))
    plane[0, 0, 3] = gifout.ALPHA_THRESHOLD - 1
    plane[0, 1, 3] = gifout.ALPHA_THRESHOLD
    gifout.write_gif(dest, [plane], [100])

    frame = _frames_of(dest)[0][0]
    assert frame.getpixel((0, 0))[3] == 0
    assert frame.getpixel((1, 0))[3] == 255


def test_a_frame_does_not_show_what_the_one_before_it_had(tmp_path):
    """``disposal=2``. Without it every transparent pixel keeps showing what was
    under it, so a subject that moves smears into a crowd."""
    dest = tmp_path / "clip.gif"
    left = _plane((8, 4), (255, 0, 0, 255), box=(0, 0, 2, 4))
    right = _plane((8, 4), (255, 0, 0, 255), box=(6, 0, 8, 4))
    gifout.write_gif(dest, [left, right], [100, 100])

    frame = _frames_of(dest)[1][0]
    assert frame.getpixel((7, 1))[3] == 255
    assert frame.getpixel((0, 1))[3] == 0


def test_a_clip_loops_forever_by_default(tmp_path):
    dest = tmp_path / "clip.gif"
    plane = _plane((4, 4), (255, 255, 255, 255))
    gifout.write_gif(dest, [plane] * 2, [100, 100])
    with Image.open(dest) as im:
        assert im.info.get("loop") == 0
    once = tmp_path / "once.gif"
    gifout.write_gif(once, [plane] * 2, [100, 100], loop=False)
    with Image.open(once) as im:
        # Absent, rather than present and zero: zero *is* forever.
        assert "loop" not in im.info


def test_a_mismatched_frame_or_duration_is_refused_rather_than_written(tmp_path):
    dest = tmp_path / "clip.gif"
    plane = _plane((4, 4), (255, 255, 255, 255))
    with pytest.raises(ValueError):
        gifout.write_gif(dest, [plane], [100, 100])
    with pytest.raises(ValueError):
        gifout.write_gif(dest, [plane, _plane((5, 5), (0, 0, 0, 255))], [100, 100])
    with pytest.raises(ValueError):
        gifout.write_gif(dest, [], [])
    assert not dest.exists()


def test_the_reserved_index_is_never_spent_on_a_colour():
    """255 colours, not 256. Letting the quantiser use every slot means the one
    it picks for the last index is the one punched out as transparent, which
    deletes that colour from the picture."""
    rng = np.random.default_rng(3)
    noisy = rng.integers(0, 256, (32, 32, 4), dtype=np.uint8)
    noisy[..., 3] = 255
    indexed = gifout.quantise(noisy)
    try:
        assert int(np.asarray(indexed).max()) < gifout.TRANSPARENT_INDEX
    finally:
        indexed.close()


# --- repeat counts (C7) ------------------------------------------------------
#
# A GIF says "play this many more times" in a 19-byte netscape application
# extension, and the count in it is *additional* plays rather than total ones.
# Every one of the assertions below is about that off-by-one, and none of them
# is visible in a viewer that happens to loop everything anyway -- so they are
# pinned as bytes in the file rather than as whatever Pillow reports back.

#: ``21 FF 0B "NETSCAPE2.0" 03 01 <count lo> <count hi> 00``.
NETSCAPE = b"\x21\xff\x0bNETSCAPE2.0\x03\x01"


def _netscape_block(path):
    """The 19 bytes of the loop extension, or None when the file has none."""
    data = path.read_bytes()
    at = data.find(NETSCAPE)
    return None if at < 0 else data[at : at + 19]


def test_the_loop_option_counts_additional_plays_not_total_ones():
    # True and 1 are the same integer in Python and mean opposite things here,
    # which is exactly why the function tests the type before the value.
    assert gifout.loop_option(True) == {"loop": 0}
    assert gifout.loop_option(False) == {}
    assert gifout.loop_option(1) == {}
    assert gifout.loop_option(2) == {"loop": 1}
    assert gifout.loop_option(3) == {"loop": 2}


def test_a_forever_loop_writes_a_zero_count_netscape_block(tmp_path):
    dest = tmp_path / "forever.gif"
    plane = _plane((4, 4), (10, 200, 10, 255))
    gifout.write_gif(dest, [plane] * 2, [100, 100], loop=True)
    assert _netscape_block(dest) == NETSCAPE + b"\x00\x00\x00"


def test_a_repeat_of_one_omits_the_block_entirely(tmp_path):
    dest = tmp_path / "once.gif"
    plane = _plane((4, 4), (10, 200, 10, 255))
    gifout.write_gif(dest, [plane] * 2, [100, 100], loop=1)
    # Play once is spelled by the block's *absence*; a block saying "0 more
    # plays" is the format's way of saying forever.
    assert _netscape_block(dest) is None


def test_a_repeat_of_three_asks_for_two_more_plays(tmp_path):
    dest = tmp_path / "thrice.gif"
    plane = _plane((4, 4), (10, 200, 10, 255))
    gifout.write_gif(dest, [plane] * 2, [100, 100], loop=3)
    assert _netscape_block(dest) == NETSCAPE + b"\x02\x00\x00"


# --- the exact index path ----------------------------------------------------


def _read_slots(path):
    """Every frame's raw palette indices, without converting to RGB."""
    with Image.open(path) as im:
        out = []
        for index in range(getattr(im, "n_frames", 1)):
            im.seek(index)
            out.append(np.asarray(im.convert("P"), dtype=np.uint8).copy())
        return out


def test_an_exact_index_plane_is_written_slot_for_slot(tmp_path):
    """The gap a colour lookup cannot close: slots 1 and 3 are the same red, so
    a flattened plane is one red and the lookup has to pick one of them. Told
    which slot each pixel is in, the export keeps both -- which is what makes a
    GIF's colour table mean the same thing as the document's."""
    palette = [(0, 0, 0, 255), (200, 20, 20, 255), (20, 200, 20, 255), (200, 20, 20, 255)]
    plane = np.zeros((1, 4, 4), np.uint8)
    plane[0, 0] = plane[0, 1] = (200, 20, 20, 255)
    plane[0, 2] = (20, 200, 20, 255)
    plane[0, 3] = (200, 20, 20, 255)
    slots = np.asarray([[1, 3, 2, 1]], dtype=np.uint8)

    dest = tmp_path / "exact.gif"
    gifout.write_gif(dest, [plane], [100], palette=palette, indices=[slots])
    assert _read_slots(dest)[0].tolist() == [[1, 3, 2, 1]]


def test_without_a_plane_a_duplicate_colour_falls_to_the_lowest_slot(tmp_path):
    """Deterministic, and stated: an export that moved a pixel between two
    identical swatches from one run to the next would make "two exports of an
    unchanged document are byte-identical" false."""
    palette = [(0, 0, 0, 255), (200, 20, 20, 255), (20, 200, 20, 255), (200, 20, 20, 255)]
    plane = np.zeros((1, 2, 4), np.uint8)
    plane[0, 0] = plane[0, 1] = (200, 20, 20, 255)

    dest = tmp_path / "dup.gif"
    gifout.write_gif(dest, [plane], [100], palette=palette)
    assert _read_slots(dest)[0].tolist() == [[1, 1]]


def test_alpha_still_decides_transparency_even_with_a_plane(tmp_path):
    """The index plane says which *colour* a pixel is; the format's own
    transparency is a separate slot, and what puts a pixel in it is alpha."""
    palette = [(0, 0, 0, 255), (200, 20, 20, 255)]
    plane = np.zeros((1, 2, 4), np.uint8)
    plane[0, 0] = (200, 20, 20, 255)
    slots = np.asarray([[1, 1]], dtype=np.uint8)

    dest = tmp_path / "hole.gif"
    gifout.write_gif(dest, [plane], [100], palette=palette, indices=[slots])
    read = _read_slots(dest)[0]
    assert read[0, 0] == 1
    assert read[0, 1] == gifout.TRANSPARENT_INDEX


def test_planes_are_per_frame_and_may_be_none(tmp_path):
    """A clip routinely has one frame with a blended layer on it and forty
    without, so all-or-nothing would throw away the forty."""
    palette = [(0, 0, 0, 255), (200, 20, 20, 255), (200, 20, 20, 255), (20, 200, 20, 255)]
    first = np.zeros((1, 1, 4), np.uint8)
    first[0, 0] = (200, 20, 20, 255)
    # A *different* colour, so Pillow's inter-frame delta cannot collapse the
    # second frame to nothing and leave the reader looking at the background.
    second = np.zeros((1, 1, 4), np.uint8)
    second[0, 0] = (20, 200, 20, 255)

    dest = tmp_path / "mixed.gif"
    gifout.write_gif(
        dest,
        [first, second],
        [100, 100],
        palette=palette,
        indices=[np.asarray([[2]], np.uint8), None],
    )
    assert _read_slots(dest)[0][0, 0] == 2, "told, so the second of the two identical reds"
    # The second frame is asserted by *colour* rather than by slot. Pillow hands
    # back later frames of a transparent GIF already promoted to RGBA -- the raw
    # indices are gone by the time a reader sees them -- so a slot assertion here
    # would be testing Pillow's re-quantise, not what was written. The colour is
    # the part that has to be right when there is no plane to be told about.
    with Image.open(dest) as im:
        im.seek(1)
        assert tuple(np.asarray(im.convert("RGBA"))[0, 0]) == (20, 200, 20, 255)


def test_a_mismatched_plane_is_refused_by_name():
    plane = np.zeros((2, 2, 4), np.uint8)
    with pytest.raises(ValueError, match="the size of the frame"):
        gifout.map_to_palette(plane, [(0, 0, 0, 255)], np.zeros((3, 3), np.uint8))


def test_a_short_plane_list_is_refused_by_name(tmp_path):
    plane = np.zeros((1, 1, 4), np.uint8)
    with pytest.raises(ValueError, match="an index plane or None"):
        gifout.write_gif(
            tmp_path / "x.gif", [plane, plane], [10, 10], palette=[(0, 0, 0, 255)],
            indices=[None],
        )


# --- per-frame colour tables -------------------------------------------------


def test_a_frame_with_its_own_table_is_written_in_its_own_colours(tmp_path):
    """Palette animation: the drawing does not move and the colours do. Every
    frame used to be resolved through the one document table, so an overridden
    frame exported the right slots in the wrong colours -- read back here
    rather than asserted at the call, because that is the only way to tell a
    local colour table from a global one."""
    dest = tmp_path / "flash.gif"
    slots = np.zeros((4, 4), dtype=np.uint8)
    slots[:] = 1
    plane = _plane((4, 4), (10, 10, 10, 255))
    document = [(0, 0, 0, 255), (220, 30, 30, 255)]
    override = [(0, 0, 0, 255), (30, 30, 220, 255)]

    gifout.write_gif(
        dest,
        [plane, plane],
        [100, 100],
        palette=document,
        indices=[slots, slots],
        palettes=[None, override],
    )

    frames = _frames_of(dest)
    assert tuple(np.asarray(frames[0][0])[0, 0])[:3] == (220, 30, 30)
    assert tuple(np.asarray(frames[1][0])[0, 0])[:3] == (30, 30, 220)


def test_no_override_writes_exactly_what_it_always_did(tmp_path):
    """The ordinary document must not pay for the feature or change output."""
    plane = _plane((4, 4), (10, 10, 10, 255))
    slots = np.ones((4, 4), dtype=np.uint8)
    table = [(0, 0, 0, 255), (220, 30, 30, 255)]

    before = tmp_path / "before.gif"
    after = tmp_path / "after.gif"
    gifout.write_gif(before, [plane, plane], [100, 100], palette=table, indices=[slots, slots])
    gifout.write_gif(
        after,
        [plane, plane],
        [100, 100],
        palette=table,
        indices=[slots, slots],
        palettes=[None, None],
    )

    assert before.read_bytes() == after.read_bytes()


def test_a_palettes_list_the_wrong_length_is_refused(tmp_path):
    plane = _plane((4, 4), (10, 10, 10, 255))
    with pytest.raises(ValueError):
        gifout.write_gif(tmp_path / "x.gif", [plane, plane], [10, 10], palettes=[None])
