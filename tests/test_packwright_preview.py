"""The preview pane's shrink-to-fit note.

Pure-function tests: :func:`~warlock.studio.panes.packwright_preview._area_note`
has no imgui in it precisely so it can be checked here rather than only through
the smoke suite that draws it -- the ``compose.py``/``layout.py`` split applied
to a pane.
"""

from __future__ import annotations

from test_packwright_mode import FakeCtx, _pack, _tab

from warlock.studio.panes import packwright_preview


def test_a_sparse_tileset_import_shrinks_at_default_settings():
    """The manual's promise, measured: a mostly-empty source sheet -- an 8x8
    grid of 32x32 tiles with only 10 kept -- re-packs into a visibly smaller
    atlas at the defaults (grid, power-of-two, padding 2, trim on)."""
    from warlock.studio.packwright.sources import sprites_from_tileset

    ctx = FakeCtx()
    tab = _tab(ctx, sources=0)  # a fresh document: PackSettings() defaults
    sheet = _tileset_pixels(cols=8, rows=8, tile=32, kept=10)
    for sprite in sprites_from_tileset(sheet, tile=(32, 32), prefix="sheet"):
        tab.doc.add_source(sprite)
    tab.pack_dirty = True
    _pack(ctx, tab)

    size_px = (int(tab.atlas.shape[1]), int(tab.atlas.shape[0]))
    source_area = 8 * 32 * 8 * 32
    output_area = size_px[0] * size_px[1]
    assert output_area < source_area, (output_area, source_area)

    note = packwright_preview._area_note(tab, size_px)
    assert note is not None and note.startswith("Packed to")
    assert "%" in note


def test_the_note_says_shrink_grow_or_parity_correctly():
    shrunk = _fake_tab([(64, 64)])
    note = packwright_preview._area_note(shrunk, (32, 32))
    assert note is not None and "Packed to" in note and "%" in note

    grown = _fake_tab([(16, 16)])
    note = packwright_preview._area_note(grown, (64, 64))
    assert note is not None and "turn off power-of-two" in note

    parity = _fake_tab([(32, 32)])
    assert packwright_preview._area_note(parity, (32, 32)) == (
        "Packed to the same area as the source pixels."
    )


def test_no_note_with_nothing_to_compare():
    assert packwright_preview._area_note(_fake_tab([]), (32, 32)) is None


# --- fakes ------------------------------------------------------------------


class _FakeSprite:
    def __init__(self, w: int, h: int) -> None:
        self.width = w
        self.height = h


class _FakeSource:
    def __init__(self, w: int, h: int) -> None:
        self.sprite = _FakeSprite(w, h)


class _FakeDoc:
    def __init__(self, source_sizes: list[tuple[int, int]]) -> None:
        self.sources = [_FakeSource(w, h) for w, h in source_sizes]


class _FakeTab:
    def __init__(self, doc: _FakeDoc) -> None:
        self.doc = doc


def _fake_tab(source_sizes: list[tuple[int, int]]) -> _FakeTab:
    return _FakeTab(_FakeDoc(source_sizes))


def _tileset_pixels(*, cols: int, rows: int, tile: int, kept: int):
    import numpy as np

    pixels = np.zeros((rows * tile, cols * tile, 4), dtype=np.uint8)
    filled = 0
    for row in range(rows):
        for col in range(cols):
            if filled >= kept:
                break
            y0, x0 = row * tile, col * tile
            pixels[y0 + 4 : y0 + tile - 4, x0 + 4 : x0 + tile - 4] = (200, 30, 30, 255)
            filled += 1
        if filled >= kept:
            break
    return pixels
