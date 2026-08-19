"""Reading blob roles off a sheet nobody in this program made.

**The corpus is derived from ``blob`` itself**, not typed: each of the 47 roles
is drawn programmatically from ``open_edges`` and ``open_corners``, so the
fixture and the thing under test agree about what a role *looks like* by
construction. A hand-drawn corpus would be a second, quieter statement of the
same table, and the first renumbering would put the two out of step with nothing
to say so.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.tilegrid import blob, roles

TILE = 16
FILL = (60, 140, 70, 255)
BACKDROP = (200, 40, 200, 255)


def _role_tile(blob_index: int, *, opaque: bool = False) -> np.ndarray:
    """One tile drawn as its role: filled, with its open sides cleared.

    The strips cleared here are the strips :mod:`.roles` measures -- an eighth
    of the tile -- plus a pixel, so the ratio comfortably clears the threshold
    rather than sitting on it. The threshold's own edge is tested separately.
    """
    depth = max(1, TILE // 8) + 1
    tile = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    tile[:, :] = BACKDROP if opaque else (0, 0, 0, 0)
    tile[:, :] = FILL
    north, east, south, west = blob.open_edges(blob_index)
    if north:
        tile[:depth, :] = BACKDROP if opaque else (0, 0, 0, 0)
    if south:
        tile[TILE - depth:, :] = BACKDROP if opaque else (0, 0, 0, 0)
    if west:
        tile[:, :depth] = BACKDROP if opaque else (0, 0, 0, 0)
    if east:
        tile[:, TILE - depth:] = BACKDROP if opaque else (0, 0, 0, 0)
    ne, se, sw, nw = blob.open_corners(blob_index)
    blank = BACKDROP if opaque else (0, 0, 0, 0)
    if ne:
        tile[:depth, TILE - depth:] = blank
    if se:
        tile[TILE - depth:, TILE - depth:] = blank
    if sw:
        tile[TILE - depth:, :depth] = blank
    if nw:
        tile[:depth, :depth] = blank
    return tile


def _sheet(indices: list[int], cols: int = 47, *, opaque: bool = False) -> np.ndarray:
    rows = -(-len(indices) // cols)
    out = np.zeros((rows * TILE, cols * TILE, 4), dtype=np.uint8)
    if opaque:
        out[:, :] = BACKDROP
    for position, blob_index in enumerate(indices):
        y, x = (position // cols) * TILE, (position % cols) * TILE
        out[y:y + TILE, x:x + TILE] = _role_tile(blob_index, opaque=opaque)
    return out


def _canonical(**kwargs) -> np.ndarray:
    return _sheet(list(range(blob.TILE_COUNT)), **kwargs)


def test_the_corpus_really_covers_every_role() -> None:
    """The fixture's own premise: 47 distinct silhouettes, derived not typed."""
    assert blob.TILE_COUNT == 47
    seen = {
        (blob.open_edges(i), blob.open_corners(i)) for i in range(blob.TILE_COUNT)
    }
    assert len(seen) == blob.TILE_COUNT


def test_a_canonical_sheet_infers_the_identity_order() -> None:
    found = roles.infer_roles(_canonical(), TILE, TILE)
    assert found is not None
    assert found.order == tuple(range(blob.TILE_COUNT))
    assert found.extras == 0
    assert found.background == "alpha"


def test_a_shuffled_sheet_recovers_the_permutation() -> None:
    rng = np.random.default_rng(4)
    permutation = rng.permutation(blob.TILE_COUNT).tolist()
    found = roles.infer_roles(_sheet(permutation), TILE, TILE)
    assert found is not None
    # ``order[case]`` is the *cell* playing that case, so composing it with the
    # permutation must give the identity.
    assert [permutation[cell] for cell in found.order] == list(range(blob.TILE_COUNT))


def test_one_role_missing_is_refused() -> None:
    """A partial set is refused silently: a half-set that cannot be painted
    with is not an offer worth making."""
    short = list(range(blob.TILE_COUNT))
    short[10] = short[11]  # role 10 replaced by a duplicate of 11
    assert roles.infer_roles(_sheet(short), TILE, TILE) is None


def test_random_tiles_are_refused() -> None:
    rng = np.random.default_rng(7)
    noise = np.zeros((8 * TILE, 8 * TILE, 4), dtype=np.uint8)
    noise[..., :3] = rng.integers(0, 256, (8 * TILE, 8 * TILE, 3), dtype=np.uint8)
    noise[..., 3] = 255
    assert roles.infer_roles(noise, TILE, TILE) is None


def test_a_sheet_too_small_to_hold_47_is_refused() -> None:
    assert roles.infer_roles(_sheet(list(range(20))), TILE, TILE) is None


def test_the_opaque_background_variant_resolves_by_border_colour() -> None:
    """A mockup sheet on a flat backdrop: the backdrop is by construction what
    every tile's border is mostly made of."""
    found = roles.infer_roles(_canonical(opaque=True), TILE, TILE)
    assert found is not None
    assert found.background == "border"
    assert found.order == tuple(range(blob.TILE_COUNT))


def test_a_duplicated_role_lets_the_first_win_and_counts_an_extra() -> None:
    indices = [*range(blob.TILE_COUNT), 5]
    # 48 columns for 48 tiles: one row, no padding cell to confuse the count.
    found = roles.infer_roles(_sheet(indices, cols=48), TILE, TILE)
    assert found is not None
    assert found.extras == 1
    assert found.order[5] == 5, "the first tile in reading order wins"


def test_a_padded_grid_counts_its_empty_cells_as_extras() -> None:
    """47 is prime, so a sheet laid out on any other width has empty cells at
    the end -- and an empty cell is a legitimate all-background silhouette whose
    role the first tile already took."""
    found = roles.infer_roles(_sheet(list(range(blob.TILE_COUNT)), cols=8), TILE, TILE)
    assert found is not None
    assert found.extras == 1  # 48 cells for 47 roles
    assert found.order == tuple(range(blob.TILE_COUNT))


def test_the_ratio_edge_decides_a_side() -> None:
    """A side is open when at least ``EDGE_OPEN_RATIO`` of its strip is
    background, and the inclusive edge is the case a rewrite gets wrong."""
    depth = max(1, TILE // 8)
    filled = np.zeros((TILE, TILE, 4), dtype=np.uint8)
    filled[:, :] = FILL
    background = filled[:, :, 3] == 0

    # A north strip exactly at the ratio counts as open.
    at_ratio = background.copy()
    cells = depth * TILE
    wanted = int(np.ceil(cells * roles.EDGE_OPEN_RATIO))
    flat = at_ratio[:depth, :].reshape(-1)
    flat[:wanted] = True
    at_ratio[:depth, :] = flat.reshape(depth, TILE)
    mask = roles._cell_mask(at_ratio, TILE, TILE)
    assert not mask & blob.N, "at the ratio the side is open"

    # One pixel short of it is not.
    under = background.copy()
    flat = under[:depth, :].reshape(-1)
    flat[: wanted - 1] = True
    under[:depth, :] = flat.reshape(depth, TILE)
    assert roles._cell_mask(under, TILE, TILE) & blob.N


def test_a_closed_side_means_a_neighbour_is_present() -> None:
    """The inversion between "what the picture looks like" and "what the blob
    table is indexed by" -- getting it backwards makes every tile its own
    opposite."""
    solid = np.zeros((TILE, TILE), dtype=bool)  # nothing is background
    assert roles._cell_mask(solid, TILE, TILE) == blob.BLOB_MASKS[blob.FULL]
    empty = np.ones((TILE, TILE), dtype=bool)  # everything is background
    assert roles._cell_mask(empty, TILE, TILE) == blob.BLOB_MASKS[blob.LONE]


# --- reorder -----------------------------------------------------------------


def test_reorder_lays_the_sheet_out_as_47_columns() -> None:
    sheet = _canonical()
    found = roles.infer_roles(sheet, TILE, TILE)
    assert found is not None
    out = roles.reorder(sheet, found, TILE, TILE)
    assert out.shape == (TILE, blob.TILE_COUNT * TILE, 4)
    assert out.dtype == np.uint8


def test_reorder_puts_each_role_in_its_canonical_column() -> None:
    rng = np.random.default_rng(11)
    permutation = rng.permutation(blob.TILE_COUNT).tolist()
    sheet = _sheet(permutation)
    found = roles.infer_roles(sheet, TILE, TILE)
    assert found is not None
    out = roles.reorder(sheet, found, TILE, TILE)
    for case in range(blob.TILE_COUNT):
        column = out[:, case * TILE:(case + 1) * TILE]
        assert np.array_equal(column, _role_tile(case)), case


def test_reorder_never_writes_through_the_source() -> None:
    sheet = _canonical()
    sheet.flags.writeable = False
    found = roles.infer_roles(sheet, TILE, TILE)
    assert found is not None
    out = roles.reorder(sheet, found, TILE, TILE)
    out[:] = 3  # would raise if it aliased the read-only source


def test_reorder_fills_alpha_for_an_rgb_sheet() -> None:
    sheet = _canonical(opaque=True)[:, :, :3]
    found = roles.infer_roles(sheet, TILE, TILE)
    assert found is not None
    out = roles.reorder(sheet, found, TILE, TILE)
    assert out.shape[2] == 4
    assert np.all(out[:, :, 3] == 255)


def test_a_non_image_array_is_refused() -> None:
    with pytest.raises(ValueError):
        roles.infer_roles(np.zeros((8, 8), dtype=np.uint8), TILE, TILE)
    with pytest.raises(ValueError):
        roles.infer_roles(np.zeros((8, 8, 4), dtype=np.uint8), 0, TILE)
