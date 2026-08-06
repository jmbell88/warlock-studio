"""Paint mode's GPU texture cache reads pixels only when something changed.

``Document.image.tobytes()`` is a full-canvas copy -- 16 MB at 2048 square --
and the composite is asked for every frame. The rule these pin is that a frame
where the texture already exists and the revision has not moved performs *no*
pixel flatten at all: creation payloads are thunks, evaluated only when a
texture is actually created, and uploads are gated on ``rev`` as before.

Headless on purpose: the module touches GL only through ``ctx.viewer.ctx``, so
a recording fake stands in for moderngl and the tests assert copies, not
pixels.
"""

from __future__ import annotations

from typing import Any

from warlock.studio.panes import inker_textures


class _Texture:
    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size
        self.writes: list[tuple[bytes, Any]] = []
        self.filter: Any = None
        self.released = False

    def write(self, data: bytes, viewport: Any = None) -> None:
        self.writes.append((data, viewport))

    def release(self) -> None:
        self.released = True


class _GL:
    NEAREST = "nearest"
    LINEAR = "linear"

    def __init__(self) -> None:
        self.created: list[_Texture] = []

    def texture(self, size: tuple[int, int], components: int, data: bytes) -> _Texture:
        texture = _Texture(size)
        self.created.append(texture)
        return texture


class _Ctx:
    def __init__(self) -> None:
        self.viewer = type("V", (), {"ctx": _GL()})()
        self.state = type("S", (), {"preview": {}})()


class _Image:
    """A Pillow stand-in that counts every ``tobytes`` through a shared list,
    so a crop's flatten is charged to the same counter as the full image's."""

    def __init__(self, size: tuple[int, int] = (8, 8), counter: list | None = None):
        self.size = size
        self.counter = counter if counter is not None else []

    def tobytes(self) -> bytes:
        self.counter.append(self.size)
        return b"\x00" * (self.size[0] * self.size[1] * 4)

    def crop(self, region: tuple[int, int, int, int]) -> _Image:
        x0, y0, x1, y1 = region
        return _Image((x1 - x0, y1 - y0), self.counter)


class _Doc:
    def __init__(self, image: _Image) -> None:
        self.image = image
        self.rev = 1
        self._region: tuple[int, int, int, int] | None = None

    def mark(self, region: tuple[int, int, int, int] | None) -> None:
        self.rev += 1
        self._region = region

    def take_dirty(self) -> tuple[int, int, int, int] | None:
        region, self._region = self._region, None
        return region


def _tab(doc: Any) -> Any:
    return type("Tab", (), {"uid": "t1", "doc": doc})()


def test_an_unchanged_frame_flattens_no_pixels() -> None:
    ctx = _Ctx()
    doc = _Doc(_Image())
    tab = _tab(doc)

    first = inker_textures.composite(ctx, tab, nearest=True)
    assert len(doc.image.counter) == 1, "creation is the one full flatten"

    again = inker_textures.composite(ctx, tab, nearest=True)
    assert again is first
    assert doc.image.counter == [(8, 8)], "an unchanged frame reads nothing"
    assert first.writes == []


def test_a_dirty_rect_uploads_only_the_rect() -> None:
    ctx = _Ctx()
    doc = _Doc(_Image())
    tab = _tab(doc)
    texture = inker_textures.composite(ctx, tab, nearest=True)

    doc.mark((1, 2, 3, 6))
    inker_textures.composite(ctx, tab, nearest=True)
    assert doc.image.counter[-1] == (2, 4), "the crop was flattened, not the canvas"
    assert texture.writes[-1][1] == (1, 2, 2, 4)


def test_a_structural_change_uploads_everything() -> None:
    ctx = _Ctx()
    doc = _Doc(_Image())
    tab = _tab(doc)
    texture = inker_textures.composite(ctx, tab, nearest=True)

    doc.mark(None)  # take_dirty's None means "all"
    inker_textures.composite(ctx, tab, nearest=True)
    assert texture.writes[-1][1] is None
    assert doc.image.counter[-1] == (8, 8)


class _Floating:
    def __init__(self) -> None:
        self.size = (4, 4)
        self.rev = 1
        self.pixels = _Image(self.size)


def test_an_unchanged_floating_buffer_flattens_no_pixels() -> None:
    ctx = _Ctx()
    buf = _Floating()
    doc = type("D", (), {"floating": buf})()
    tab = _tab(doc)

    first = inker_textures.floating(ctx, tab, nearest=True)
    assert len(buf.pixels.counter) == 1

    again = inker_textures.floating(ctx, tab, nearest=True)
    assert again is first
    assert len(buf.pixels.counter) == 1, "an unchanged buffer reads nothing"

    buf.rev += 1
    inker_textures.floating(ctx, tab, nearest=True)
    assert first.writes, "a revision bump still uploads"
