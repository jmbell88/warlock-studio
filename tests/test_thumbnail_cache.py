"""The thumbnail cache's sampling filter and its rewrite detection.

Both matter to the pixel-art preview: NEAREST is what keeps a 32 px sprite
crisp on screen, and the (key, mtime) keying is the whole mechanism by which a
re-derived artifact replaces itself in the UI without anyone invalidating
anything.
"""

from __future__ import annotations

import os

from PIL import Image

from warlock.studio.textures import ThumbnailCache


class _FakeTexture:
    def __init__(self, size):
        self.size = size
        self.filter = None
        self.repeat_x = True
        self.repeat_y = True
        self.released = False
        # The GL name the imgui backend keys its registration on. Present
        # because release goes through ``forget_texture`` first -- so a fake
        # without one only passes while no test has left a renderer registered.
        self.glo = id(self)

    def release(self):
        self.released = True


class _FakeGL:
    NEAREST = "nearest"
    LINEAR = "linear"

    def texture(self, size, components, data):
        return _FakeTexture(size)


def _png(path, color):
    Image.new("RGBA", (8, 8), color).save(path)


def test_a_texture_samples_linear_unless_asked_for_nearest(tmp_path):
    cache = ThumbnailCache(_FakeGL())
    _png(tmp_path / "thumb.png", (10, 20, 30, 255))
    _png(tmp_path / "pixel_32.png", (10, 20, 30, 255))

    smooth = cache.get("job:thumb", tmp_path / "thumb.png")
    crisp = cache.get("job:pixel_32.png", tmp_path / "pixel_32.png", nearest=True)

    assert smooth.filter == ("linear", "linear")
    assert crisp.filter == ("nearest", "nearest")


def test_a_rewritten_file_hands_out_a_new_texture(tmp_path):
    # A re-derive lands via os.replace, so the path stays the same and only the
    # mtime moves. The bumped mtime must produce a fresh texture, or the
    # preview keeps showing the palette the user just changed away from.
    cache = ThumbnailCache(_FakeGL())
    path = tmp_path / "pixel_32.png"
    _png(path, (10, 20, 30, 255))

    first = cache.get("job:pixel_32.png", path, nearest=True)

    _png(path, (200, 0, 0, 255))
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))

    # A new frame: the rewrite lands on a task thread and is drawn on a later
    # frame, and the per-frame stat memo (B17) is scoped to exactly one.
    cache.begin_frame()
    again = cache.get("job:pixel_32.png", path, nearest=True)
    assert again is not first


def test_the_superseded_texture_is_retired_rather_than_left_to_lru(tmp_path):
    """"Replaces itself" was half true: the new mtime minted a new entry and
    the old one sat in the cache until ordinary LRU pressure reached it. A
    palette-tuning session left one dead texture behind per turn of the knob,
    evicting library thumbnails to make room for them."""
    cache = ThumbnailCache(_FakeGL())
    path = tmp_path / "pixel_32.png"
    _png(path, (10, 20, 30, 255))

    first = cache.get("job:pixel_32.png", path, nearest=True)

    _png(path, (200, 0, 0, 255))
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    cache.begin_frame()  # the rewrite is seen on the next frame (B17)
    cache.get("job:pixel_32.png", path, nearest=True)

    assert len(cache._entries) == 1
    # Retired, not released: the frame that drew it may still be in flight.
    assert first.released is False
    cache.begin_frame()
    assert first.released is True


def test_one_key_asked_for_with_two_filters_gets_two_textures(tmp_path):
    """The filter is baked in at decode time, so leaving it out of the key
    meant whichever call decoded first decided the sampling for every other --
    silently, and differently depending on which pane drew first."""
    cache = ThumbnailCache(_FakeGL())
    path = tmp_path / "pixel_32.png"
    _png(path, (10, 20, 30, 255))

    crisp = cache.get("job:pixel_32.png", path, nearest=True)
    smooth = cache.get("job:pixel_32.png", path)

    assert crisp.filter == ("nearest", "nearest")
    assert smooth.filter == ("linear", "linear")
    # And both stay live: they describe the same bytes, so neither supersedes
    # the other and two panes drawing one file cannot thrash each other.
    assert len(cache._entries) == 2
    assert crisp.released is False
