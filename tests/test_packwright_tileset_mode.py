"""The tile-sheet door on Packwright's controller.

The decode task parks the sheet on the state, the popup's answer turns it into
sprites through :func:`packwright_mode.import_tileset`, and the key carries the
tile size so a re-cut is not a duplicate. Fakes follow
``tests/test_packwright_mode.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from warlock.studio import packwright_mode


class _Settings:
    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.store.get(key)

    def set(self, key: str, value: Any) -> None:
        self.store[key] = value


class _AppState:
    def __init__(self) -> None:
        self.packwright = None
        self.inker = None
        self.mode = "home"
        self.preview: dict[str, Any] = {}


class FakeCtx:
    def __init__(self, *, accept: bool = True) -> None:
        self.state = _AppState()
        self.settings = _Settings()
        self.submitted: list[str] = []
        self.toasts: list[tuple[str, str]] = []
        self.accept = accept
        self.result: Any = None

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        self.submitted.append(key)
        if not self.accept:
            return False
        self.result = run(*args)
        return True

    def toast(self, message: str, kind: str = "info") -> None:
        self.toasts.append((message, kind))


class _Done:
    def __init__(self, key: str, result: Any = None, message: str = "") -> None:
        self.key = key
        self.result = result
        self.message = message


def _sheet() -> np.ndarray:
    """2 x 2 cells of 4 px; the bottom-right cell is empty."""
    pixels = np.zeros((8, 8, 4), dtype=np.uint8)
    for row, column in ((0, 0), (0, 1), (1, 0)):
        pixels[row * 4 + 1, column * 4 + 1] = (9, 9, 9, 255)
    return pixels


def _park(ctx: FakeCtx, tab: Any, pixels: np.ndarray) -> None:
    packwright_mode.on_task_done(
        ctx,
        _Done(
            f"packwright-tileset:{tab.uid}",
            {"tileset": ("sheet.png", "sheet", pixels), "uid": tab.uid},
        ),
    )


def test_the_ask_submits_under_its_own_prefix() -> None:
    ctx = FakeCtx(accept=False)  # never run the dialog-opening task in a test
    tab = packwright_mode.new_document(ctx)
    packwright_mode.ask_add_tileset(ctx)
    assert ctx.submitted == [f"packwright-tileset:{tab.uid}"]


def test_the_decode_landing_parks_the_sheet_on_the_state() -> None:
    ctx = FakeCtx()
    tab = packwright_mode.new_document(ctx)
    _park(ctx, tab, _sheet())
    state = packwright_mode.ensure(ctx)
    assert state.tileset_import is not None
    assert state.tileset_import[1] == "sheet"
    assert state.tileset_import_open is False


def test_a_landing_for_a_closed_tab_is_dropped() -> None:
    ctx = FakeCtx()
    packwright_mode.new_document(ctx)
    packwright_mode.on_task_done(
        ctx, _Done("packwright-tileset:gone", {"tileset": ("p", "p", _sheet()), "uid": "gone"})
    )
    assert packwright_mode.ensure(ctx).tileset_import is None


def test_import_adds_only_occupied_cells_and_rearms_the_pack() -> None:
    ctx = FakeCtx()
    tab = packwright_mode.new_document(ctx)
    tab.pack_dirty = False
    state = packwright_mode.ensure(ctx)
    _park(ctx, tab, _sheet())
    state.tileset_cell = (4, 4)

    assert packwright_mode.import_tileset(ctx) is True
    assert len(tab.doc.sources) == 3
    assert tab.pack_dirty is True
    assert state.tileset_import is None
    assert ctx.toasts[-1][0] == "Added 3 tile(s)."


def test_a_recut_at_another_tile_size_is_not_a_duplicate() -> None:
    ctx = FakeCtx()
    tab = packwright_mode.new_document(ctx)
    state = packwright_mode.ensure(ctx)
    _park(ctx, tab, _sheet())
    state.tileset_cell = (4, 4)
    packwright_mode.import_tileset(ctx)
    before = len(tab.doc.sources)

    _park(ctx, tab, _sheet())
    state.tileset_cell = (2, 2)
    packwright_mode.import_tileset(ctx)
    assert len(tab.doc.sources) > before


def test_the_same_cut_twice_is_a_duplicate_and_says_so() -> None:
    ctx = FakeCtx()
    tab = packwright_mode.new_document(ctx)
    state = packwright_mode.ensure(ctx)
    for _ in range(2):
        _park(ctx, tab, _sheet())
        state.tileset_cell = (4, 4)
        packwright_mode.import_tileset(ctx)
    assert len(tab.doc.sources) == 3
    assert ctx.toasts[-1][0] == "Those tiles are already in this atlas."


def test_import_with_nothing_parked_is_a_no_op() -> None:
    ctx = FakeCtx()
    packwright_mode.new_document(ctx)
    assert packwright_mode.import_tileset(ctx) is False
