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
