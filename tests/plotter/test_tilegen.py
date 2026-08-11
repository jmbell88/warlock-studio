"""The generated base set: its geometry, its art, and its determinism.

Determinism is the one that matters most and is easiest to lose. The point of a
generated base set is that a polished set is painted over it and can be diffed
against it, so a generator that drifted by a pixel between runs -- through an
RNG, a float, or an antialiased edge going through the platform's libm -- would
quietly make that comparison meaningless.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.plotter import blob, project, tilegen
from warlock.studio.plotter.tileset import TerrainSpec

SPECS = [
    tilegen.GenSpec(tile_w=16, tile_h=16, projection=project.ORTHOGONAL),
    tilegen.GenSpec(tile_w=32, tile_h=16, projection=project.ISOMETRIC),
]


@pytest.mark.parametrize("spec", SPECS)
def test_the_same_spec_gives_the_same_bytes(spec):
    assert tilegen.atlas_pixels(spec).tobytes() == tilegen.atlas_pixels(spec).tobytes()


@pytest.mark.parametrize("spec", SPECS)
def test_the_atlas_is_one_row_per_terrain_and_one_column_per_case(spec):
    ts = tilegen.generate(spec)
    assert ts.columns == blob.TILE_COUNT
    assert ts.rows == len(spec.terrains)
    # No filler tiles: a dead cell in a published atlas is one somebody
    # eventually paints something into.
    assert ts.tile_count == blob.TILE_COUNT * len(spec.terrains)


@pytest.mark.parametrize("spec", SPECS)
def test_a_tiles_position_is_its_role(spec):
    ts = tilegen.generate(spec)
    for rank in range(len(spec.terrains)):
        for case in (blob.LONE, 20, blob.FULL):
            assert ts.terrain_of(ts.local_for(rank, case)) == rank


def test_a_generated_set_declares_the_terrains_it_was_given():
    spec = tilegen.GenSpec(terrains=tilegen.DEFAULT_TERRAINS[:3], tile_w=8, tile_h=8)
    assert [entry.name for entry in tilegen.generate(spec).terrains] == [
        entry.name for entry in tilegen.DEFAULT_TERRAINS[:3]
    ]


def test_an_orthogonal_interior_tile_has_no_outline_at_all():
    spec = tilegen.GenSpec(tile_w=8, tile_h=8)
    codes = tilegen.tile_coverage(spec, blob.FULL)
    assert (codes == tilegen.FILL).all()


def test_an_orthogonal_lone_tile_is_outlined_on_all_four_sides():
    spec = tilegen.GenSpec(tile_w=8, tile_h=8)
    codes = tilegen.tile_coverage(spec, blob.LONE)
    assert (codes[0, :] == tilegen.OUTLINE).all()
    assert (codes[-1, :] == tilegen.OUTLINE).all()
    assert (codes[:, 0] == tilegen.OUTLINE).all()
    assert (codes[:, -1] == tilegen.OUTLINE).all()
    assert codes[3, 3] == tilegen.FILL


def test_the_outline_follows_the_side_with_no_neighbour():
    """North and east have neighbours here, so the outline is on south and
    west -- which is the whole of what the blob case means."""
    spec = tilegen.GenSpec(tile_w=8, tile_h=8)
    codes = tilegen.tile_coverage(spec, int(blob.BLOB_INDEX[blob.N | blob.E]))
    # The two open sides are outlined edge to edge, corners included -- so the
    # closed sides are clear only *between* them.
    assert (codes[-1, :] == tilegen.OUTLINE).all()
    assert (codes[:, 0] == tilegen.OUTLINE).all()
    assert (codes[0, 1:-1] == tilegen.FILL).all()
    assert (codes[1:-1, -1] == tilegen.FILL).all()


def test_a_notched_corner_is_one_pixel():
    """The single pixel that separates the forty-seven from the sixteen: both
    flanks have neighbours, so neither side drew an outline, and the corner
    still belongs to the boundary."""
    spec = tilegen.GenSpec(tile_w=8, tile_h=8)
    codes = tilegen.tile_coverage(spec, int(blob.BLOB_INDEX[blob.N | blob.E]))
    assert codes[0, -1] == tilegen.OUTLINE
    assert codes[1, -1] == tilegen.FILL
    assert codes[0, -2] == tilegen.FILL


def test_an_isometric_tile_is_a_diamond_inside_its_rectangle():
    spec = tilegen.GenSpec(tile_w=16, tile_h=8, projection=project.ISOMETRIC)
    codes = tilegen.tile_coverage(spec, blob.FULL)
    covered = codes != tilegen.VOID
    # The corners of the rectangle are outside the diamond; the middle is in.
    assert not covered[0, 0] and not covered[0, -1]
    assert not covered[-1, 0] and not covered[-1, -1]
    assert covered[spec.tile_h // 2, spec.tile_w // 2]
    # Widest in the middle, narrowest at the ends.
    runs = covered.sum(axis=1)
    assert runs[0] < runs[spec.tile_h // 2]


def test_an_isometric_diamond_is_symmetric_left_to_right():
    spec = tilegen.GenSpec(tile_w=16, tile_h=8, projection=project.ISOMETRIC)
    covered = tilegen.tile_coverage(spec, blob.FULL) != tilegen.VOID
    assert np.array_equal(covered, covered[:, ::-1])


def test_the_atlas_uses_only_the_colours_it_was_given():
    """No antialiasing anywhere, which is both the art style and the reason the
    output does not depend on a platform's floating point."""
    spec = tilegen.GenSpec(terrains=tilegen.DEFAULT_TERRAINS[:2], tile_w=8, tile_h=8)
    pixels = tilegen.atlas_pixels(spec)
    seen = {tuple(int(part) for part in colour) for colour in np.unique(
        pixels.reshape(-1, 4), axis=0
    )}
    allowed = {(0, 0, 0, 0)}
    for entry in spec.terrains:
        allowed.add(tuple(entry.fill))
        allowed.add(tuple(entry.outline))
    assert seen <= allowed


def test_repolish_keeps_the_roles_and_takes_the_art():
    spec = tilegen.GenSpec(tile_w=8, tile_h=8)
    original = tilegen.generate(spec)
    painted = np.array(original.pixels)
    painted[..., 1] = 3
    out = tilegen.repolish(original, painted)
    assert [entry.name for entry in out.terrains] == [e.name for e in original.terrains]
    assert int(out.pixels[..., 1].max()) == 3
    # A new array identity, which is what makes the pane's texture cache -- keyed
    # on ``id(pixels)`` -- re-upload with no invalidation call anywhere.
    assert out.pixels is not original.pixels


def test_repolish_refuses_an_atlas_that_changed_size():
    """The roles are positional, so a cropped atlas is a set whose tile 93 is no
    longer the tile the map thinks it is."""
    original = tilegen.generate(tilegen.GenSpec(tile_w=8, tile_h=8))
    with pytest.raises(ValueError, match="resize it"):
        tilegen.repolish(original, np.zeros((4, 4, 4), dtype=np.uint8))


def test_a_spec_refuses_what_it_cannot_draw():
    with pytest.raises(ValueError, match="at least one terrain"):
        tilegen.GenSpec(terrains=())
    with pytest.raises(ValueError, match="cannot share a name"):
        tilegen.GenSpec(
            terrains=(
                TerrainSpec("Grass", (1, 1, 1, 255), (0, 0, 0, 255)),
                TerrainSpec("grass", (2, 2, 2, 255), (0, 0, 0, 255)),
            )
        )
    with pytest.raises(ValueError, match="needs a name"):
        tilegen.GenSpec(terrains=(TerrainSpec("  ", (1, 1, 1, 255), (0, 0, 0, 255)),))
    with pytest.raises(ValueError, match="four pixels"):
        tilegen.GenSpec(tile_w=2, tile_h=2)


def test_a_tile_edge_past_the_ceiling_is_refused_and_the_ceiling_itself_is_not():
    """The atlas is 47 tiles wide per terrain, so the edge is multiplied by 376
    before anything is allocated -- which is why the refusal is on the spec
    rather than on the array it would have asked for."""
    with pytest.raises(ValueError, match="at most 512 pixels"):
        tilegen.GenSpec(tile_w=tilegen.MAX_TILE_EDGE + 1, tile_h=8)
    with pytest.raises(ValueError, match="at most 512 pixels"):
        tilegen.GenSpec(tile_w=8, tile_h=tilegen.MAX_TILE_EDGE + 1)
    spec = tilegen.GenSpec(tile_w=tilegen.MAX_TILE_EDGE, tile_h=tilegen.MAX_TILE_EDGE)
    assert spec.tile_w == 512


def test_a_tileset_refuses_a_terrain_declaration_its_geometry_denies():
    """Validated when the tileset is made, not at the first paint -- the same
    rule the slicing follows and for the same reason."""
    from warlock.studio.plotter.tileset import Tileset

    with pytest.raises(ValueError, match="columns wide"):
        Tileset(
            name="wrong",
            pixels=np.zeros((8, 16, 4), dtype=np.uint8),
            tile_w=8,
            tile_h=8,
            terrains=tilegen.DEFAULT_TERRAINS[:1],
        )
