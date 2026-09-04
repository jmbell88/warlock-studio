"""What one document, one cache or one gesture is allowed to cost.

Batch 6's subject was a *file* declaring a number nothing checked. This one is
the same question asked of the app's own state: a control with a floor and no
ceiling, a cache counting the wrong unit, an operation that multiplies, a list
that only ever grows. None of these is reachable by a hostile file -- they are
all reachable by a user with a keyboard and a long session, which is why they
matter differently rather than less.

The pattern worth naming, because five of the cases below share it: **a bound
that exists at one door and not at the one the user actually comes through.**
``MAX_TRIANGLES`` gated import and not growth. ``MAX_SPRITES`` gated packing
and not adding. ``NEW_MAX`` gated the new-canvas form and not the resize popup.
``MAX_LIST_LIMIT`` gated the query and not the counter that drives it. In each
case the number was already chosen and already argued; what was missing was the
second call site.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

# --- the resize popup ---------------------------------------------------------


def test_the_resize_popup_caps_growth_and_leaves_shrinking_free():
    """The popup stored ``(max(1, w), max(1, h))`` -- a floor and no ceiling --
    and fed it to ``doc.scale``, which has none either. ``100000`` is 40 GB a
    layer, on the frame thread, holding an unsaved document."""
    from warlock.studio import inker_mode

    assert inker_mode.clamp_resize((512, 512), 100_000, 100_000) == (
        inker_mode.NEW_MAX,
        inker_mode.NEW_MAX,
    )
    # Shrinking is the usual reason to open this popup, and the comment that
    # used to sit on ``NEW_MAX`` said so. It is not capped at all.
    assert inker_mode.clamp_resize((512, 512), 8, 8) == (8, 8)
    assert inker_mode.clamp_resize((512, 512), 0, -4) == (1, 1)


def test_an_oversized_document_stays_resizable():
    """A canvas imported at 12,000 px must not snap to 8192 the moment its
    owner opens the popup to crop it."""
    from warlock.studio import inker_mode

    assert inker_mode.clamp_resize((12_000, 12_000), 12_000, 12_000) == (12_000, 12_000)
    assert inker_mode.clamp_resize((12_000, 12_000), 200_000, 4)[0] == 12_000


def test_a_typed_nonsense_size_keeps_what_the_document_has():
    from warlock.studio import inker_mode

    assert inker_mode.clamp_resize((64, 32), None, "x") == (64, 32)


# --- the thumbnail cache ------------------------------------------------------


class _FakeTexture:
    def __init__(self, size):
        self.size = size
        self.components = 4
        self.filter = None
        self.repeat_x = True
        self.repeat_y = True
        self.released = False
        self.glo = id(self)

    def release(self):
        self.released = True


class _FakeGL:
    NEAREST = "nearest"
    LINEAR = "linear"

    def texture(self, size, components, data):
        return _FakeTexture(size)


def test_the_thumbnail_cache_is_bounded_by_bytes_as_well_as_by_count():
    """``max_side`` is a per-call argument and the manual passes 1600, so the
    same "limit of 120" spans 31 MiB of VRAM and 1.2 GiB."""
    from warlock.studio.textures import ThumbnailCache

    cache = ThumbnailCache(_FakeGL(), limit=100, budget=4 * 64 * 64 * 4)
    for i in range(8):
        # One insert a frame: ``_evict`` deliberately never frees a texture
        # this frame has already handed out, so eight inserts on one frame is
        # the documented overshoot rather than a budget that does not work.
        cache.begin_frame()
        cache.from_pixels(f"k{i}", 1.0, (64, 64), b"\x00" * (64 * 64 * 3))
    assert len(cache._entries) < 8, "the count was nowhere near its limit"
    assert cache._bytes <= cache.budget


def test_the_byte_total_comes_back_down_when_entries_go():
    """A total maintained beside every insert is a total that can drift."""
    from warlock.studio.textures import ThumbnailCache

    cache = ThumbnailCache(_FakeGL())
    cache.begin_frame()
    for i in range(4):
        cache.from_pixels(f"k{i}", 1.0, (16, 16), b"")
    assert cache._bytes == 4 * 16 * 16 * 4
    # A new version of one key retires the old one through ``_supersede``.
    cache.from_pixels("k0", 2.0, (16, 16), b"")
    assert cache._bytes == 4 * 16 * 16 * 4


# --- inker frame textures -----------------------------------------------------


class _Texture:
    def __init__(self, size):
        self.size = size
        self.filter = None
        self.released = False
        self.glo = id(self)

    def write(self, data, viewport=None):
        pass

    def release(self):
        self.released = True


class _GL:
    NEAREST = "nearest"
    LINEAR = "linear"

    def texture(self, size, components, data):
        return _Texture(size)


class _Ctx:
    def __init__(self):
        self.viewer = SimpleNamespace(ctx=_GL())
        self.state = SimpleNamespace(preview={})


class _AnimDoc:
    def __init__(self, side=64):
        self.side = side

    def frame_flat(self, frame_uid, track_uid=None):
        return np.zeros((self.side, self.side, 4), dtype=np.uint8)

    def frame_stamp(self, frame_uid):
        return frame_uid


def test_inker_frame_textures_have_a_vram_budget(monkeypatch):
    """The CPU flatten cache is bounded (``document.FRAME_CACHE_BYTES``) and
    the GL side was not, while the frame count is capped at nothing at all."""
    from warlock.studio.panes import inker_textures

    monkeypatch.setattr(inker_textures, "FRAME_TEXTURE_BYTES", 4 * 64 * 64 * 4)
    ctx = _Ctx()
    tab = SimpleNamespace(uid="t1", doc=_AnimDoc())
    for uid in range(10):
        inker_textures.frame_texture(ctx, tab, uid)
    live = [k for k in ctx.state.preview if k.startswith("inker_tex:t1:frame")]
    # Four frames' worth: a texture and a stamp each, plus the two bookkeeping
    # entries (the order and the touched map) that ride under the same prefix
    # so ``release_doc``'s sweep collects them with the tab.
    assert len(live) <= 4 * 2 + 2
    # A dict used as an ordered set since 2026-09-03 -- ``list.remove`` was a
    # linear scan per visible cell per frame; see ``_frame_lru``.
    order = list(ctx.state.preview["inker_tex:t1:frame-lru"])
    assert len(order) <= 4
    assert order[-1].endswith("frame9"), "the newest is the one kept"


def test_the_frame_texture_count_is_bounded_for_a_tiny_document(monkeypatch):
    """A 64-square document is 16 KB a frame, so the byte budget alone would
    let thousands of entries into the list the sweep walks."""
    from warlock.studio.panes import inker_textures

    monkeypatch.setattr(inker_textures, "FRAME_TEXTURE_CAP", 5)
    ctx = _Ctx()
    tab = SimpleNamespace(uid="t2", doc=_AnimDoc(side=4))
    for uid in range(40):
        inker_textures.frame_texture(ctx, tab, uid)
    assert len(ctx.state.preview["inker_tex:t2:frame-lru"]) <= 5


# --- growth that multiplies ---------------------------------------------------


def _grid(levels: int):
    """A cube subdivided *levels* times: 6, 24, 96, 384 faces."""
    from warlock.studio.clay import elements as el
    from warlock.studio.clay import ops_subdiv
    from warlock.studio.clay import primitives as bp

    mesh = bp.box()
    for _ in range(levels):
        mesh, _ = ops_subdiv.subdivide(mesh, el.empty())
    return mesh


def test_smooth_refuses_to_multiply_past_the_triangle_budget(monkeypatch):
    """Catmull-Clark is four times the faces per press, and ``MAX_TRIANGLES``
    gated the import door only."""
    from warlock.studio.clay import elements as el
    from warlock.studio.clay import ops_subdiv

    mesh = _grid(2)
    monkeypatch.setattr(ops_subdiv, "MAX_SUBDIVIDED_FACES", 16)
    with pytest.raises(el.OpError, match="Smoothing"):
        ops_subdiv.catmull_clark(mesh, el.empty())


def test_the_budget_stops_the_second_press_as_well_as_the_first(monkeypatch):
    """Per level, not once: each level is its own allocation."""
    from warlock.studio.clay import elements as el
    from warlock.studio.clay import mesh as bm
    from warlock.studio.clay import ops_subdiv

    mesh = _grid(0)
    monkeypatch.setattr(ops_subdiv, "MAX_SUBDIVIDED_FACES", 40)
    once, _ = ops_subdiv.catmull_clark(mesh, el.empty(), levels=1)
    assert bm.face_count(once) == 24
    with pytest.raises(el.OpError, match="Smoothing"):
        ops_subdiv.catmull_clark(mesh, el.empty(), levels=2)


def test_linear_subdivide_shares_the_budget(monkeypatch):
    from warlock.studio.clay import elements as el
    from warlock.studio.clay import ops_subdiv

    monkeypatch.setattr(ops_subdiv, "MAX_SUBDIVIDED_FACES", 4)
    with pytest.raises(el.OpError, match="Subdividing"):
        ops_subdiv.subdivide(_grid(1), el.empty())


def _sprite(key: str):
    from warlock.studio.packwright.sources import Sprite

    return Sprite(key=key, name=key, pixels=np.zeros((2, 2, 4), np.uint8))


def test_packwright_refuses_a_sprite_past_the_pack_ceiling(monkeypatch):
    """``MAX_SPRITES`` was asked at pack time, of a document that had already
    accepted them -- so the only way past a full pack was to delete some."""
    from warlock.studio.packwright import document as pd

    monkeypatch.setattr(pd, "MAX_SPRITES", 3)
    doc = pd.PackDoc()
    for i in range(3):
        doc.add_source(_sprite(f"s{i}"))
    with pytest.raises(ValueError, match="most"):
        doc.add_source(_sprite("s9"))


# --- the library window -------------------------------------------------------


def test_load_more_stops_at_the_service_ceiling(monkeypatch):
    """``limit`` grew forever while ``list_jobs`` clamped the read, so past the
    ceiling every press moved a number and changed nothing else."""
    from warlock.service import jobs as svc_jobs
    from warlock.studio import jobs_cache

    monkeypatch.setattr(svc_jobs, "MAX_LIST_LIMIT", 500)
    cache = jobs_cache.JobsCache(svc=None)
    for _ in range(20):
        cache.load_more()
    assert cache.limit == 500
    assert not cache.can_load_more()


def test_the_window_can_still_widen_below_the_ceiling(monkeypatch):
    from warlock.service import jobs as svc_jobs
    from warlock.studio import jobs_cache

    monkeypatch.setattr(svc_jobs, "MAX_LIST_LIMIT", 5000)
    cache = jobs_cache.JobsCache(svc=None)
    assert cache.can_load_more()
    cache.load_more()
    assert cache.limit == jobs_cache.LIST_LIMIT * 2


class _FakeImgui:
    """Just enough of imgui for ``library._clipper``'s arithmetic."""

    def __init__(self, view: float, scroll: float) -> None:
        self._view = view
        self._scroll = scroll
        self.cursor = 0.0
        self.dummies: list[float] = []

    def get_window_size(self):
        return SimpleNamespace(y=self._view)

    def get_scroll_y(self):
        return self._scroll

    def get_cursor_pos_y(self):
        return self.cursor

    def dummy(self, size):
        self.dummies.append(size[1])
        self.cursor += size[1]


def _clipped(monkeypatch, count: int, view: float, scroll: float, selected=None):
    from warlock.studio.panes import library

    fake = _FakeImgui(view, scroll)
    monkeypatch.setattr(library, "imgui", fake)
    monkeypatch.setattr(library, "sp", lambda v: v)
    monkeypatch.setattr(library, "filters_compact", lambda _ctx: False)
    ctx = SimpleNamespace(
        state=SimpleNamespace(library_scroll_to=None, selected=selected)
    )
    skip = library._clipper(ctx, count)
    if skip is None:
        return None, fake
    drawn = []
    for i in range(count):
        if not skip(f"job{i}"):
            drawn.append(i)
            fake.cursor += library.CARD_HEIGHT
    return drawn, fake


def test_a_short_library_is_never_clipped(monkeypatch):
    """The threshold is not about correctness -- it keeps a scroll-dependent
    path from underneath the ordinary case."""
    from warlock.studio.panes import library

    drawn, _ = _clipped(monkeypatch, library.CLIP_THRESHOLD - 1, 400.0, 0.0)
    assert drawn is None


def test_a_long_library_draws_only_what_is_on_screen(monkeypatch):
    from warlock.studio.panes import library

    count = 400
    drawn, fake = _clipped(monkeypatch, count, 400.0, 0.0)
    assert drawn is not None
    assert len(drawn) < 10, "a 400-pixel viewport holds four 96-pixel cards"
    assert drawn[0] == 0
    # Every skipped card still takes its own height, so the scrollbar and every
    # row's place in the list are what they would have been.
    assert len(fake.dummies) == count - len(drawn)
    assert fake.cursor == pytest.approx(count * library.CARD_HEIGHT)


def test_the_selected_card_is_drawn_even_when_it_is_off_screen(monkeypatch):
    """Arriving at a selection by any route has to land."""
    drawn, _ = _clipped(monkeypatch, 400, 400.0, 0.0, selected="job300")
    assert 300 in drawn


def test_scrolling_moves_which_cards_are_drawn(monkeypatch):
    from warlock.studio.panes import library

    drawn, _ = _clipped(monkeypatch, 400, 400.0, 100.0 * library.CARD_HEIGHT)
    assert drawn is not None and drawn[0] >= 99


# --- the backstops ------------------------------------------------------------


def test_motion_forgets_keys_nothing_is_drawing_any_more():
    """Four dictionaries keyed on strings with job ids in them, and ``forget``
    had one caller. ``_FRAME`` is already an exact record of liveness."""
    from warlock.studio import motion

    motion.reset()
    try:
        motion._STATE["library/sel/old"] = 1.0
        motion._TARGET["library/sel/old"] = 1.0
        motion._FRAME["library/sel/old"] = 10
        motion._STATE["library/sel/live"] = 1.0
        motion._FRAME["library/sel/live"] = 5_000
        assert motion.sweep(frame=5_000) == 1
        assert "library/sel/old" not in motion._STATE
        assert "library/sel/live" in motion._STATE
    finally:
        motion.reset()


def test_a_key_that_was_snapped_but_never_asked_for_is_swept_too():
    """``snap`` writes ``_STATE`` with no ``_FRAME`` entry, so a liveness test
    that only looked at ``_FRAME`` would never reach it."""
    from warlock.studio import motion

    motion.reset()
    try:
        motion._STATE["orphan"] = 0.5
        motion.sweep(frame=10_000)
        assert "orphan" not in motion._STATE
    finally:
        motion.reset()


def test_the_sweep_runs_rarely_rather_than_every_frame():
    from warlock.studio import motion

    motion.reset()
    try:
        motion._FRAME["a"] = 0
        motion.sweep(frame=5_000)
        motion._FRAME["b"] = 0
        assert motion.sweep(frame=5_001) == 0, "not swept again a frame later"
        assert "b" in motion._FRAME
    finally:
        motion.reset()


def test_the_confirm_queue_refuses_a_runaway_caller():
    """A cap and not a smaller queue: dropping the *second* question was the
    bug this queue exists to have fixed."""
    from warlock.studio import dialogs

    queue = dialogs.ConfirmQueue()
    for i in range(dialogs.MAX_QUEUED + 20):
        queue.ask(dialogs.Confirm(title=f"q{i}", message="?"))
    assert len(queue._queue) == dialogs.MAX_QUEUED
    # The three a quit asks are nowhere near it.
    assert dialogs.MAX_QUEUED > 3


def test_the_prompt_queue_shares_the_cap():
    from warlock.studio import dialogs

    queue = dialogs.PromptQueue()
    for i in range(dialogs.MAX_QUEUED + 5):
        queue.ask(dialogs.Prompt(title=f"p{i}", label="name"))
    assert len(queue._queue) == dialogs.MAX_QUEUED


def test_the_texture_registry_says_so_when_it_outgrows_itself(caplog):
    """No sweep, deliberately -- the renderer cannot tell whether an owner
    still wants an entry. What was missing was any signal at all."""
    from warlock.studio import imgui_backend

    backend = imgui_backend.ImguiRenderer.__new__(imgui_backend.ImguiRenderer)
    backend._textures = {}
    backend._warn_at = 4
    with caplog.at_level("WARNING"):
        for i in range(6):
            backend._textures[i] = object()
            backend._note_registry()
    assert any("registered textures" in r.message for r in caplog.records)
    assert backend._warn_at > 4, "the threshold doubles rather than repeating"


# --- the convention that had no test ------------------------------------------


def test_every_per_tab_preview_key_is_swept_when_the_tab_closes():
    """``_PER_TAB_KEYS`` is a list somebody has to remember to add to, and its
    own comment records that four keys were quietly outside it.

    So this is the scan that makes forgetting fail rather than accumulate. A
    per-tab entry in ``state.preview`` is written as an f-string whose literal
    head is a ``name:`` prefix and whose interpolations mention a uid -- which
    is a shape narrow enough to find every one of them and wide enough that a
    new one cannot be written any other way without looking deliberate.
    """
    import ast
    import re
    from pathlib import Path

    from warlock.studio.panes import inker_textures

    root = Path(inker_textures.__file__).parent.parent
    files = [root / "inker_mode.py", root / "inker_ops.py", root / "inker_state.py"]
    files += sorted((root / "panes").glob("inker_*.py"))
    assert len(files) >= 6, "the vacuous-pass guard: a moved module must fail here"

    head = re.compile(r"^[a-z][a-z0-9_]*:$")
    allowed = set(inker_textures._PER_TAB_KEYS) | {"inker_tex:"}
    found: set[tuple[str, str]] = set()
    for path in files:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.JoinedStr) or not node.values:
                continue
            first = node.values[0]
            if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
                continue
            if not head.match(first.value):
                continue
            rest = ast.dump(
                ast.Module(body=[ast.Expr(v) for v in node.values[1:]], type_ignores=[])
            )
            if "uid" in rest:
                found.add((path.name, first.value))
    assert found, "the scan found nothing at all, which means it is not scanning"
    stray = sorted({key for _name, key in found} - allowed)
    assert not stray, (
        f"{stray} are per-tab preview keys that ``release_doc`` will not sweep -- "
        "add each to inker_textures._PER_TAB_KEYS"
    )


def test_the_per_tab_key_list_holds_no_prefix_nothing_writes():
    """The other direction, so the list cannot rot into a list of names that
    used to mean something."""
    from pathlib import Path

    from warlock.studio.panes import inker_textures

    root = Path(inker_textures.__file__).parent.parent
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in [root / "inker_mode.py", *sorted((root / "panes").glob("inker_*.py"))]
    )
    for prefix in inker_textures._PER_TAB_KEYS:
        assert text.count(f'"{prefix}') >= 2, f"nothing writes {prefix}"
