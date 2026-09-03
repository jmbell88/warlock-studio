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
    assert ctx.toasts[-1][0] == "Those tiles are already in this atlas, unchanged."


def test_import_with_nothing_parked_is_a_no_op() -> None:
    ctx = FakeCtx()
    packwright_mode.new_document(ctx)
    assert packwright_mode.import_tileset(ctx) is False


# --- dedup at the import door (Part I) ----------------------------------------


def _dup_sheet(size: int = 4) -> np.ndarray:
    """Four cells: two identical, one their mirror, one genuinely different."""
    one = np.zeros((size, size, 4), dtype=np.uint8)
    one[..., :3] = (30, 30, 30)
    one[..., 3] = 255
    one[0, 0, :3] = (200, 40, 40)
    other = one.copy()
    other[size - 1, size - 1, :3] = (40, 200, 40)
    flipped = np.ascontiguousarray(one[:, ::-1])
    top = np.concatenate([one, one.copy()], axis=1)
    bottom = np.concatenate([flipped, other], axis=1)
    return np.concatenate([top, bottom], axis=0)


def _park_dups(ctx: FakeCtx, pixels: np.ndarray, cell: int = 4) -> Any:
    tab = packwright_mode.new_document(ctx)
    state = packwright_mode.ensure(ctx)
    state.tileset_import = ("sheet.png", "sheet", pixels)
    state.tileset_cell = (cell, cell)
    return tab


def test_dedup_is_off_by_default_and_the_import_stays_byte_faithful():
    """A repack is a faithful repack unless somebody asks otherwise: an atlas
    quietly missing four tiles that happened to be rotations of each other is a
    bug report nobody can reproduce."""
    ctx = FakeCtx()
    tab = _park_dups(ctx, _dup_sheet())
    state = packwright_mode.ensure(ctx)
    assert state.tileset_dedup is False

    assert packwright_mode.import_tileset(ctx) is True
    assert len(tab.doc.sources) == 4


def test_exact_duplicates_drop_when_the_flag_is_set():
    ctx = FakeCtx()
    tab = _park_dups(ctx, _dup_sheet())
    state = packwright_mode.ensure(ctx)
    state.tileset_dedup = True

    assert packwright_mode.import_tileset(ctx) is True
    # The two identical cells become one; the mirror and the odd one survive.
    assert len(tab.doc.sources) == 3


def test_the_orientation_toggle_also_drops_the_mirror():
    ctx = FakeCtx()
    tab = _park_dups(ctx, _dup_sheet())
    state = packwright_mode.ensure(ctx)
    state.tileset_dedup = True
    state.tileset_dedup_flips = True

    assert packwright_mode.import_tileset(ctx) is True
    assert len(tab.doc.sources) == 2


def test_the_popup_preview_and_the_import_agree_on_the_dropped_count():
    """The popup-promise contract ``tileset_occupancy`` already enforces for
    emptiness, applied to the dedup the popup now also promises: both sides run
    the same ``dedup_tiles`` call over the same sprites."""
    from warlock.studio.packwright.sources import dedup_tiles, sprites_from_tileset

    ctx = FakeCtx()
    tab = _park_dups(ctx, _dup_sheet())
    state = packwright_mode.ensure(ctx)
    state.tileset_dedup = True
    state.tileset_dedup_flips = True

    tile = state.tileset_cell
    promised, dropped = dedup_tiles(
        sprites_from_tileset(
            state.tileset_import[2],
            tile=tile,
            prefix=f"sheet.png@{tile[0]}x{tile[1]}",
            name="sheet",
        ),
        orientations=True,
    )
    packwright_mode.import_tileset(ctx)
    assert len(tab.doc.sources) == len(promised)
    assert dropped == 2
