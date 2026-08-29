"""The blob-47 coverage field: that one atlas is the whole map.

The claim this module makes is not "these tiles look nice". It is that a 47-tile
atlas, composited once, renders any membership field *identically* to computing
the field over the whole map at once -- which is what lets a terrain set be 47
pictures instead of a per-map render. If that ever stops being true, every set
generated afterwards paints maps with a hairline seam at cell boundaries, and
nothing in the data says why. So the round-trip test below is the important one
and everything else is scaffolding for it.

The other standing risk is drift: ``tilemask`` restates ``blob``'s eight bits
and its collapse rule because a pipeline may not import ``studio``, and a test
is the only thing holding the two copies together. Tests are not layered, so
this one imports both.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from warlock.pipelines import tilemask
from warlock.studio.tilegrid import blob

TILE = 32
CELLS = 12


def _material(seed: float, *, warm: bool) -> np.ndarray:
    """A smooth wrap-periodic gradient as a stand-in for a generated material.

    **Deliberately not a flat colour.** A flat A over a flat B produces the same
    picture for a large family of wrong masks -- every pixel is one of two
    values whatever the boundary does -- so the round-trip identity would pass
    with the coverage field scrambled. Two sinusoids at integer frequencies are
    exactly periodic over the tile, which is the property the whole construction
    assumes of a real material.
    """
    u = np.arange(TILE)[None, :] * (2 * np.pi / TILE)
    v = np.arange(TILE)[:, None] * (2 * np.pi / TILE)
    first = 0.5 + 0.5 * np.sin(u + seed) * np.cos(v * 2 + seed)
    second = 0.5 + 0.5 * np.cos(u * 3 - seed) * np.sin(v + seed)
    if warm:
        rgb = np.stack([first, second * 0.4 + 0.3, np.full_like(first, 0.15)], axis=2)
    else:
        rgb = np.stack([np.full_like(first, 0.1), second * 0.3 + 0.2, first], axis=2)
    out = np.empty((TILE, TILE, 4), dtype=np.uint8)
    out[:, :, :3] = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    out[:, :, 3] = 255
    return out


def _distances(size: int) -> tuple[float, float, float]:
    """The three ratios in pixels, as :func:`tilemask.blob_coverages` resolves them."""
    return (
        tilemask.BLOB_INSET_RATIO * size,
        tilemask.NOISE_AMPLITUDE_RATIO * size,
        tilemask.FEATHER_RATIO * size,
    )


# -- drift --------------------------------------------------------------------


def test_the_restated_bits_are_the_same_bits():
    """``tilemask`` may not import ``studio``, so this is the only thing holding
    its copy of the eight neighbour bits to ``blob``'s. The values are
    positional -- the atlas order derives from them -- so a divergence here
    renumbers every generated set against every set already on disk."""
    for name in ("N", "NE", "E", "SE", "S", "SW", "W", "NW"):
        assert getattr(tilemask, name) == getattr(blob, name), name
    assert tilemask.NEIGHBOURS == blob.NEIGHBOURS
    assert tilemask.CORNER_FLANKS == blob.CORNER_FLANKS


def test_the_restated_collapse_is_the_same_collapse():
    """All 256 raw masks, not a sample: the two ``normalise`` bodies are
    separate source, and a rule that agreed on 255 cases would be worse than one
    that disagreed on all of them."""
    for mask in range(256):
        assert tilemask.normalise(mask) == blob.normalise(mask), mask


def test_the_restated_table_is_the_same_table():
    assert tilemask.BLOB_MASKS == blob.BLOB_MASKS
    assert tilemask.TILE_COUNT == blob.TILE_COUNT == 47
    assert np.array_equal(tilemask.BLOB_INDEX, blob.BLOB_INDEX)


def test_the_index_table_is_a_read_only_uint8_array():
    """Built lazily, because this module's rule is that numpy is imported inside
    the functions -- and read-only for ``blob``'s reason: the hot caller indexes
    it with a whole mask field at once and must not be able to write through
    it."""
    table = tilemask.BLOB_INDEX
    assert table.dtype == np.uint8
    assert table.shape == (256,)
    assert table.flags.writeable is False
    assert tilemask.BLOB_INDEX is table  # cached, not rebuilt per access


# -- the map round trip -------------------------------------------------------


def test_the_atlas_renders_the_whole_map_exactly():
    """**The claim the 47-tile atlas rests on.** A 12x12 membership field is
    rendered twice: once as one 384px signed-distance field over all 144 cell
    rects, and once by blitting columns out of a 47-tile atlas. The two are
    byte-identical.

    Why the non-member cells can simply be filled with B, without consulting
    this module at all: inside a non-member cell ``d_out`` is 0 (the pixel is
    inside that cell's own rect) and ``d_in`` is at least 0, so ``D <= 0``,
    while the threshold is at least ``inset - amplitude`` -- a positive number,
    and larger than half the feather at the shipped ratios. So alpha clips to
    exactly 0, and :func:`tilemask.composite` at alpha 0 is ``outer`` bit for
    bit. That is asserted below rather than assumed.

    Why the member cells agree: a cell outside the 3x3 block is at least ``T``
    from every pixel of the tile, and the precondition puts the whole ramp
    inside ``T/2``, so a far cell can only ever arrive at an already-saturated
    alpha. The materials need no such argument -- they are seamless, so a tile
    sampling ``position mod T`` samples the same phase either way.

    The assertion that ships is on the **uint8** composite, because that is the
    claim that matters (the rendered map is the same picture) and because a
    float assertion invites a tolerance argument. The float fields are checked
    too, at a tolerance: they measure exactly equal today -- min-reduction
    *selects* one of its inputs rather than combining them, and these
    coordinates are all small integers plus a half, so both paths produce the
    same float64 before the same cast -- but that exactness is a property of
    these particular numbers rather than of the construction, and the picture is
    the promise.
    """
    inner, outer = _material(0.0, warm=True), _material(2.0, warm=False)
    inset, amplitude, feather = _distances(TILE)

    field = np.random.default_rng(0).random((CELLS, CELLS)) > 0.45
    assert 0 < int(field.sum()) < field.size  # a field with something in it

    span = CELLS * TILE
    member_rects, outer_rects = [], []
    for cy in range(CELLS):
        for cx in range(CELLS):
            rect = (
                float(cx * TILE),
                float(cy * TILE),
                float((cx + 1) * TILE),
                float((cy + 1) * TILE),
            )
            (member_rects if field[cy, cx] else outer_rects).append(rect)

    tile_noise = tilemask.wrap_noise(TILE, seed=0)
    whole_alpha = tilemask.coverage(
        member_rects,
        outer_rects,
        span,
        noise=np.tile(tile_noise, (CELLS, CELLS)),
        inset=inset,
        amplitude=amplitude,
        feather=feather,
    )
    whole = tilemask.composite(
        np.tile(inner, (CELLS, CELLS, 1)), np.tile(outer, (CELLS, CELLS, 1)), whole_alpha
    )

    # Every pixel of every non-member cell is exactly zero coverage, which is
    # what makes "just blit B there" the same operation as compositing.
    outside = np.repeat(np.repeat(~field, TILE, axis=0), TILE, axis=1)
    assert float(whole_alpha[outside].max()) == 0.0

    atlas = tilemask.blob_atlas(inner, outer, TILE, seed=0)
    assert atlas.shape == (TILE, blob.TILE_COUNT * TILE, 4)
    coverages = tilemask.blob_coverages(TILE, seed=0)
    indices = blob.indices_from(field)

    assembled = np.empty_like(whole)
    assembled_alpha = np.empty_like(whole_alpha)
    for cy in range(CELLS):
        for cx in range(CELLS):
            box = (slice(cy * TILE, (cy + 1) * TILE), slice(cx * TILE, (cx + 1) * TILE))
            if field[cy, cx]:
                case = int(indices[cy, cx])
                assembled[box] = atlas[:, case * TILE : (case + 1) * TILE]
                assembled_alpha[box] = coverages[case]
            else:
                assembled[box] = outer
                assembled_alpha[box] = 0.0

    assert np.array_equal(whole, assembled)
    assert np.allclose(whole_alpha, assembled_alpha, atol=1e-6)


# -- the collapse is geometry, not bookkeeping --------------------------------


def test_the_forty_seven_cases_fall_out_of_the_distance_field():
    """All 256 raw masks give the coverage of their normalised form, bit for bit.

    This is what makes the construction the *right* one rather than merely
    self-consistent: nothing in :func:`tilemask.blob_rects` calls
    ``normalise``, and the collapse still happens. A diagonal neighbour's
    nearest point to any pixel of the tile is the shared corner, so its distance
    is ``hypot(dx, dy)`` where ``dy`` alone is the distance to its northern
    flank -- and whenever that flank is absent, the flank's rect is in the outer
    set and is never further away. The diagonal therefore never wins the
    minimum, and moving it between the two sets changes nothing.

    Compared like with like: both sides come from ``coverage``, which has one
    documented return dtype. A float64 answer measured against a float32 store
    would report 255 disagreements for no reason at all.
    """
    noise = tilemask.wrap_noise(TILE, seed=3)
    inset, amplitude, feather = _distances(TILE)
    for raw in range(256):
        raw_member, raw_outer = tilemask.blob_rects(raw, TILE)
        can_member, can_outer = tilemask.blob_rects(tilemask.normalise(raw), TILE)
        kwargs = dict(noise=noise, inset=inset, amplitude=amplitude, feather=feather)
        assert np.array_equal(
            tilemask.coverage(raw_member, raw_outer, TILE, **kwargs),
            tilemask.coverage(can_member, can_outer, TILE, **kwargs),
        ), raw


# -- preconditions -------------------------------------------------------------


def test_a_ramp_wider_than_half_a_tile_is_refused_with_its_numbers():
    """The bound is the hypothesis of the locality argument, not a taste limit:
    past it a cell two away can move this tile's coverage, and the 47-column
    atlas stops describing the map. The message names the numbers because a
    refusal that does not is a refusal somebody works around."""
    with pytest.raises(ValueError) as caught:
        tilemask.blob_coverages(16, seed=0, inset=10.0)
    text = str(caught.value)
    assert "10" in text and "16" in text and "8" in text

    with pytest.raises(ValueError):
        tilemask.coverage(
            [(0.0, 0.0, 16.0, 16.0)],
            [(16.0, 0.0, 32.0, 16.0)],
            16,
            noise=np.zeros((16, 16), dtype=np.float32),
            inset=6.0,
            amplitude=2.0,
            feather=1.0,
        )


def test_a_tile_below_the_floor_is_refused_with_its_numbers():
    """Below 16px the inset is under 3px and inside and outside stop reading as
    different roles -- a smudge, produced silently, is worse than an error."""
    with pytest.raises(ValueError) as caught:
        tilemask.blob_coverages(8, seed=0)
    text = str(caught.value)
    assert "8" in text and str(tilemask.MIN_TILE) in text


def test_a_field_with_no_cells_at_all_is_refused():
    """``inf - inf``. There is no boundary to measure and no honest answer."""
    with pytest.raises(ValueError):
        tilemask.signed_distance([], [], TILE)


# -- the roles read the way the index says they do ----------------------------


def _margin(size: int) -> int:
    """How far from a border a pixel must be for alpha to be saturated.

    Alpha is 1 once the distance exceeds ``inset + amplitude + 0.5*feather``, so
    the whole sum rounded up is a bound with room in it -- and it is derived
    from the constants rather than typed, so re-tuning a ratio moves the test's
    idea of "the middle of an edge" with it.
    """
    inset, amplitude, feather = _distances(size)
    return int(np.ceil(inset + amplitude + feather))


def test_the_lone_case_is_an_island():
    """``blob.LONE``: no neighbour at all, so every border pixel is one pixel
    from an outer cell and the tile is a patch of A floating in B."""
    coverages = tilemask.blob_coverages(TILE, seed=0)
    lone = coverages[blob.LONE]
    for border in (lone[0], lone[-1], lone[:, 0], lone[:, -1]):
        assert float(border.max()) == 0.0
    assert float(lone[TILE // 2, TILE // 2]) == 1.0


def test_the_interior_fill_is_solid():
    """``blob.FULL``: no outer rect exists, so ``d_out`` is ``+inf`` and there
    is no branch anywhere that special-cases it -- the clip does the work."""
    coverages = tilemask.blob_coverages(TILE, seed=0)
    assert float(coverages[blob.FULL].min()) == 1.0
    assert float(coverages[blob.FULL].max()) == 1.0


def test_a_north_only_case_reaches_the_north_edge_and_no_other():
    """Read off ``blob.open_edges``, which until now only tests read: north is
    the one side with a neighbour, so it is the one side coverage runs out to.
    North is ``dy = -1`` -- row zero -- in the y-down convention.
    """
    case = int(blob.BLOB_INDEX[blob.N])
    assert blob.open_edges(case) == (False, True, True, True)
    coverages = tilemask.blob_coverages(TILE, seed=0)
    north = coverages[case]
    for border in (north[-1], north[:, 0], north[:, -1]):
        assert float(border.max()) == 0.0
    margin = _margin(TILE)
    # Away from the two corners, where the open east and west sides are close
    # enough to pull the coverage down themselves.
    assert float(north[0, margin : TILE - margin].min()) == 1.0


def test_a_notched_case_bites_exactly_one_corner():
    """The narrow case that separates the 47 from the 16: every edge has a
    neighbour and one diagonal does not, so the only thing missing is a bite out
    of that corner. Read off ``blob.open_corners``."""
    case = int(blob.BLOB_INDEX[0xFF & ~blob.NE])
    assert blob.open_corners(case) == (True, False, False, False)
    assert blob.open_edges(case) == (False, False, False, False)

    notched = tilemask.blob_coverages(TILE, seed=0)[case]
    # North-east in a y-down array is row 0, last column.
    assert float(notched[0, TILE - 1]) == 0.0
    for corner in ((0, 0), (TILE - 1, 0), (TILE - 1, TILE - 1)):
        assert float(notched[corner]) == 1.0
    # And nowhere else: every pixel that is not fully covered is inside the
    # north-east quadrant.
    bitten = np.argwhere(notched < 1.0)
    assert bitten.size
    assert int(bitten[:, 0].max()) < TILE // 2
    assert int(bitten[:, 1].min()) > TILE // 2


# -- noise ---------------------------------------------------------------------


def _seam_ratio(field: np.ndarray) -> float:
    """``seam.py``'s statistic: the wrap discontinuity over the field's own grain.

    A hard number says nothing on its own -- a fine-grained field differs by a
    lot between *any* two adjacent columns -- so the seam is divided by the mean
    interior difference. Near 1 means the wrap is no more of a discontinuity
    than the field already contains.
    """
    array = np.asarray(field, dtype=np.float64)
    seam = float(np.abs(array[:, 0] - array[:, -1]).mean())
    interior = float(np.abs(np.diff(array, axis=1)).mean())
    return seam / interior


def test_the_noise_is_continuous_across_its_own_wrap():
    """The assertion with content, and it is deliberately not ``np.roll``.

    Rolling a ``(T, T)`` field by ``T`` is the identity for *any* array, so
    "``np.allclose(noise, np.roll(noise, T))``" passes on white noise and proves
    nothing. What has to be true is that the field is continuous *across* the
    wrap: the column after the last is the first, and the lattice interpolation
    has to have known that. So this measures the seam against the field's own
    grain, which is what ``seam.py`` measures on a generated tile.

    Measured: a wrapping field scores at worst 1.16 over four tile sizes and
    twelve seeds, while replacing the last column with an independent draw --
    which is what a lattice that clamped instead of taking its indices mod k
    would effectively do -- scores 4.43. The bound sits in the empty band.
    """
    for size in (16, 32, 64):
        for seed in (0, 1, 7):
            field = tilemask.wrap_noise(size, seed=seed)
            assert field.shape == (size, size)
            assert field.dtype == np.float32
            # The bound is load-bearing: ``coverage``'s precondition assumes it.
            assert float(np.abs(field).max()) <= 1.0
            assert _seam_ratio(field) < 1.5
            assert _seam_ratio(field.T) < 1.5


def test_a_longer_period_is_a_longer_period():
    """``period_tiles`` is the door to phase variants, and this is what says it
    is a real door rather than an ignored argument: at 2 the field is twice as
    wide, and its four quadrants differ from each other -- which is exactly the
    repetition along a long straight edge that phases exist to break."""
    one = tilemask.wrap_noise(TILE, seed=0)
    two = tilemask.wrap_noise(TILE, seed=0, period_tiles=2)
    assert two.shape == (2 * TILE, 2 * TILE)
    assert not np.array_equal(two, np.tile(one, (2, 2)))
    assert not np.array_equal(two[:TILE, :TILE], two[TILE:, TILE:])
    assert _seam_ratio(two) < 1.5
    assert _seam_ratio(two.T) < 1.5


def test_the_field_is_deterministic_and_the_seed_reaches_it():
    """Same seed twice is the same bytes -- a set regenerated from its sidecar
    has to be the set it claims to be -- and a different seed is a different
    coastline, or the seed control on the form does nothing."""
    first = tilemask.blob_coverages(TILE, seed=0)
    assert np.array_equal(first, tilemask.blob_coverages(TILE, seed=0))
    assert not np.array_equal(first, tilemask.blob_coverages(TILE, seed=1))

    inner, outer = _material(0.0, warm=True), _material(2.0, warm=False)
    atlas = tilemask.blob_atlas(inner, outer, TILE, seed=0)
    assert np.array_equal(atlas, tilemask.blob_atlas(inner, outer, TILE, seed=0))


# -- composite -----------------------------------------------------------------


def test_the_ends_of_the_composite_are_exact():
    """Coverage 0 is ``outer`` bit for bit and coverage 1 is ``inner`` bit for
    bit. Not a rounding nicety: the painter fills a non-member cell with the
    outer material without calling this at all, and the round-trip identity is
    only true because those two paths agree."""
    inner, outer = _material(0.0, warm=True), _material(2.0, warm=False)
    zeros = np.zeros((TILE, TILE), dtype=np.float32)
    assert np.array_equal(tilemask.composite(inner, outer, zeros), outer)
    assert np.array_equal(tilemask.composite(inner, outer, zeros + 1.0), inner)


def test_the_composite_refuses_mismatched_shapes():
    inner, outer = _material(0.0, warm=True), _material(2.0, warm=False)
    with pytest.raises(ValueError):
        tilemask.composite(inner[:, :, :3], outer[:, :, :3], np.zeros((TILE, TILE), np.float32))
    with pytest.raises(ValueError):
        tilemask.composite(inner, outer, np.zeros((TILE, TILE + 1), np.float32))
    with pytest.raises(ValueError):
        tilemask.blob_atlas(inner[:16], outer, TILE, seed=0)


# -- the change detector --------------------------------------------------------


def test_the_field_has_not_moved():
    """A **change detector**, not a correctness claim. Nothing about this digest
    says the field is right; the tests above are what say that. What it says is
    that a change to the distance, the noise, the constants or the dtype is a
    red test rather than a silent re-render of every terrain set anyone has ever
    generated -- sets already on disk keep the field they were drawn with, and
    a map painted from an old set and repainted from a regenerated one would
    disagree along every coastline.

    If this fails and the change was intended, bump
    :data:`tilemask.MASK_VERSION` and put the new digest here.
    """
    digest = hashlib.sha256(tilemask.blob_coverages(32, seed=0).tobytes()).hexdigest()
    assert digest == (
        "3ad23f54e442593edc670c31a08c84c26beac285a078a7553ea38b70def2bb64"
    )
    assert tilemask.MASK_VERSION == 1
