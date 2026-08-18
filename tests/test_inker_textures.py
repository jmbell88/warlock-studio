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
        self.dropped: list[int] = []

    def mark(self, region: tuple[int, int, int, int] | None) -> None:
        self.rev += 1
        self._region = region

    def take_dirty(self) -> tuple[int, int, int, int] | None:
        region, self._region = self._region, None
        return region

    def take_dropped_frames(self) -> list[int]:
        gone, self.dropped = self.dropped, []
        return gone


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


# --- cel thumbnail eviction ---------------------------------------------------
#
# The cap eviction must never release a texture drawn this frame: a thumbnail
# handed out during the UI build already has an ``add_image`` in the live draw
# list, and freeing it before the backend draws is ``ThumbnailCache``'s exact
# deferred-release hazard. Frame numbers are stubbed rather than taken from
# imgui -- these tests have no context, and ambient context state elsewhere in
# a worker must not change their verdict.


class _CelDoc:
    anim = None

    def layer_stamp(self, uid: str) -> int:
        return 1


class _CelLayer:
    def __init__(self, uid: str) -> None:
        import numpy as np

        self.uid = uid
        self.pixels = np.zeros((4, 4, 4), dtype=np.uint8)


def _cel(ctx: Any, tab: Any, uid: str) -> Any:
    return inker_textures.cel_thumb(ctx, tab, _CelLayer(uid), size=4)


def test_asking_for_the_frame_number_off_context_is_safe() -> None:
    """Not a ``try``/``except`` around ``get_frame_count``: with no context that
    call is an access violation, not an exception -- imgui's null check is an
    assert compiled out of the release build -- so it takes the process down
    and no handler ever runs. The context has to be checked first, which is
    ``widgets._has_context``'s rule.

    (Vacuous in a process that happens to hold a context; the file needs none,
    and under ``--dist loadfile`` it gets a worker of its own.)
    """
    import ast
    import inspect

    value = inker_textures._frame_number()
    assert value is None or isinstance(value, int)

    # On the code, not on the source text: the docstring names both calls.
    tree = ast.parse(inspect.getsource(inker_textures._frame_number).lstrip())
    at = {
        node.attr: node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and node.attr in ("get_current_context", "get_frame_count")
    }
    assert at["get_current_context"] < at["get_frame_count"]


def test_off_context_the_cel_cap_evicts_immediately(monkeypatch) -> None:
    """Headless there is no draw list to protect, so the old behaviour holds:
    the least recently drawn thumbnail past the cap is released on the spot."""
    monkeypatch.setattr(inker_textures, "CEL_THUMB_CAP", 2)
    monkeypatch.setattr(inker_textures, "_frame_number", lambda: None)
    ctx = _Ctx()
    tab = _tab(_CelDoc())

    first = _cel(ctx, tab, "a")
    _cel(ctx, tab, "b")
    _cel(ctx, tab, "c")

    assert first.released, "the oldest thumbnail past the cap is freed"
    assert "inker_tex:t1:cela" not in ctx.state.preview
    assert len(inker_textures._cel_lru(ctx, "t1")) == 2


def test_a_cel_drawn_this_frame_is_never_evicted(monkeypatch) -> None:
    """More cells visible than the cap holds: every one was touched this
    frame, so the cache overshoots for the frame rather than releasing a
    texture whose draw command is already queued."""
    monkeypatch.setattr(inker_textures, "CEL_THUMB_CAP", 2)
    monkeypatch.setattr(inker_textures, "_frame_number", lambda: 7)
    ctx = _Ctx()
    tab = _tab(_CelDoc())

    thumbs = [_cel(ctx, tab, uid) for uid in ("a", "b", "c", "d")]

    assert not any(t.released for t in thumbs), "all were drawn this frame"
    assert len(inker_textures._cel_lru(ctx, "t1")) == 4, "overshoot, not release"


def test_last_frames_cels_drain_once_a_new_frame_touches(monkeypatch) -> None:
    """The overshoot from a crowded frame drains on the next frame that draws:
    last frame's draw list has been submitted by then, so its textures are
    finally safe to free -- and the one drawn *now* is still protected."""
    monkeypatch.setattr(inker_textures, "CEL_THUMB_CAP", 2)
    monkeypatch.setattr(inker_textures, "_frame_number", lambda: 7)
    ctx = _Ctx()
    tab = _tab(_CelDoc())
    crowded = {uid: _cel(ctx, tab, uid) for uid in ("a", "b", "c", "d")}

    monkeypatch.setattr(inker_textures, "_frame_number", lambda: 8)
    again = _cel(ctx, tab, "d")

    assert again is crowded["d"] and not again.released
    assert crowded["a"].released and crowded["b"].released
    assert len(inker_textures._cel_lru(ctx, "t1")) == 2
