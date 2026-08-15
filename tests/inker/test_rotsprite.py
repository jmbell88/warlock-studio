"""RotSprite: turning pixel art without inventing colours.

Two claims are worth pinning and they pull in opposite directions. The filter
must be **exact** -- every stage is an integer copy, so nothing it produces may
be a colour the source did not have, and the same input must give the same
bytes every time. And it must be *right*, which for EPX means one hand-computed
3x3 case: the rule is four two-term comparisons per quadrant and every
implementation of it that is subtly wrong is wrong in the same way, by getting
one of the "and not" terms backwards, which shows only on a diagonal.
"""

from __future__ import annotations

import numpy as np

from warlock.studio.inker import transform as tf

A = (255, 0, 0, 255)
B = (0, 0, 255, 255)


def _plane(rows: list[list[tuple[int, int, int, int]]]) -> np.ndarray:
    return np.array(rows, dtype=np.uint8)


# --- EPX itself --------------------------------------------------------------


def test_a_flat_plane_is_only_doubled():
    """The interesting half of the rule is the "and not": a pixel whose
    neighbours all agree has no edge to round, so every quadrant is itself."""
    plane = _plane([[A, A, A], [A, A, A], [A, A, A]])
    out = tf.epx(plane)
    assert out.shape == (6, 6, 4)
    assert np.array_equal(out, np.repeat(np.repeat(plane, 2, axis=0), 2, axis=1))


def test_the_hand_computed_diagonal_case():
    """A 3x3 with a corner of B, worked through by hand.

    The centre pixel is A with ``up = B``, ``left = B``, ``right = A``,
    ``down = A`` (edges replicate, but the centre's four neighbours are all
    real). Its top-left quadrant is the one that fires::

        1 = A_up  if C == A and C != D and A != B   ->  B == B, B != A, B != A

    and the other three do not: 2 needs ``up == right`` (B != A), 3 needs
    ``down == left`` (A != B), 4 needs ``right == down`` (A == A) *and*
    ``right != up`` (A != B) *and* ``down != left`` (A != B) -- so 4 fires too,
    with ``down``, which is A, i.e. no visible change. The one substantive
    answer is that the centre's top-left corner becomes B: the step is rounded.
    """
    plane = _plane(
        [
            [B, B, A],
            [B, A, A],
            [A, A, A],
        ]
    )
    out = tf.epx(plane)

    # The centre pixel (1, 1) expands to out[2:4, 2:4].
    assert tuple(out[2, 2]) == B  # quadrant 1, taken from ``up``
    assert tuple(out[2, 3]) == A  # quadrant 2, unchanged
    assert tuple(out[3, 2]) == A  # quadrant 3, unchanged
    assert tuple(out[3, 3]) == A  # quadrant 4, ``down`` which is also A


def test_epx_invents_no_colour():
    """Every stage is a copy, so the output's palette is a subset of the
    input's -- which is the whole reason this is the pixel-art filter."""
    rng = np.random.default_rng(11)
    plane = rng.integers(0, 4, size=(9, 7, 4), dtype=np.uint8) * 85
    before = {tuple(px) for px in plane.reshape(-1, 4).tolist()}
    after = {tuple(px) for px in tf.epx(plane).reshape(-1, 4).tolist()}
    assert after <= before


def test_epx_works_on_a_mask_plane_too():
    """The comparison goes through ``_packed`` and the copy through fancy
    indexing, so a two-dimensional mask takes the same path -- and it has to,
    or a rotated buffer's coverage would stop matching its pixels."""
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[0:2, 0:2] = 255
    out = tf.epx(mask)
    assert out.shape == (8, 8)
    assert set(out.ravel().tolist()) <= {0, 255}


def test_the_border_replicates_rather_than_padding_with_nothing():
    """A transparent pad would make the border's neighbour a colour that is not
    in the drawing, rounding off every shape that touches the edge of its cel."""
    plane = _plane([[A, A], [A, A]])
    out = tf.epx(plane)
    assert {tuple(px) for px in out.reshape(-1, 4).tolist()} == {A}


# --- the whole turn ----------------------------------------------------------


def test_three_rounds_and_a_centre_sample():
    assert tf.ROTSPRITE_ROUNDS == 3
    assert tf.ROTSPRITE_SCALE == 8
    plane = _plane([[A, B], [B, A]])
    assert tf.rotsprite(plane, 0.0, expand=True).shape == (2, 2, 4)


def test_a_quarter_turn_is_exactly_the_lossless_one():
    """The strongest available check on the whole pipeline. ``rotate90`` is
    ``np.rot90`` and resamples nothing at all, so an 8x upscale, a nearest
    turn and a centre-of-block downsample agreeing with it byte for byte means
    the lattice and the sample offset are both right -- an off-by-one in either
    shows here and nowhere else."""
    rng = np.random.default_rng(3)
    plane = rng.integers(0, 255, size=(6, 5, 4), dtype=np.uint8)
    for quarters in range(4):
        turned = tf.rotsprite(plane, 90.0 * quarters, expand=True)
        assert np.array_equal(turned, tf.rotate90(plane, quarters)), quarters


def test_the_turn_is_deterministic():
    """A free transform re-renders from the lifted pixels on every mouse-move,
    so a result that wobbled would look like a bug in the drag."""
    rng = np.random.default_rng(5)
    plane = rng.integers(0, 3, size=(12, 12, 4), dtype=np.uint8) * 127
    first = tf.rotsprite(plane, 23.0, expand=True)
    for _ in range(3):
        assert np.array_equal(tf.rotsprite(plane, 23.0, expand=True), first)


def test_the_turn_invents_no_colour():
    rng = np.random.default_rng(13)
    plane = rng.integers(0, 3, size=(10, 10, 4), dtype=np.uint8) * 127
    before = {tuple(px) for px in plane.reshape(-1, 4).tolist()}
    turned = tf.rotsprite(plane, 37.0, expand=True)
    after = {tuple(px) for px in turned.reshape(-1, 4).tolist()}
    # The fill the rotation puts in the corners is the one addition.
    assert after <= before | {(0, 0, 0, 0)}


def test_an_unexpanded_turn_comes_back_the_size_it_went_in():
    plane = np.zeros((10, 10, 4), dtype=np.uint8)
    plane[3:7, 3:7] = A
    assert tf.rotsprite(plane, 30.0).shape == (10, 10, 4)


# --- the way in, and the ceiling ---------------------------------------------


def test_rotate_routes_rotsprite_through_the_filter():
    rng = np.random.default_rng(17)
    plane = rng.integers(0, 3, size=(8, 8, 4), dtype=np.uint8) * 127
    through = tf.rotate(plane, 45.0, expand=True, resample="rotsprite")
    direct = tf.rotsprite(plane, 45.0, expand=True)
    assert np.array_equal(through, direct)


def test_it_is_offered_as_a_resample_mode():
    assert "rotsprite" in tf.RESAMPLES


def test_a_scale_asked_for_rotsprite_gets_nearest():
    """The algorithm has nothing to say about a scale, and quietly smoothing
    one because the user picked the pixel-art *rotation* would be backwards."""
    plane = _plane([[A, B], [B, A]])
    assert np.array_equal(
        tf.scale(plane, (8, 8), resample="rotsprite"),
        tf.scale(plane, (8, 8), resample="nearest"),
    )


def test_a_shear_asked_for_rotsprite_gets_nearest():
    plane = np.zeros((8, 8, 4), dtype=np.uint8)
    plane[2:6, 2:6] = A
    assert np.array_equal(
        tf.shear(plane, (20.0, 0.0), resample="rotsprite"),
        tf.shear(plane, (20.0, 0.0), resample="nearest"),
    )


def test_a_plane_over_the_ceiling_falls_back_to_nearest():
    """64x in area is the cost, and a rotate drag pays it on every mouse-move.
    Above the ceiling the honest answer is the cheaper filter."""
    assert tf.rotsprite_fits((512, 512))
    assert not tf.rotsprite_fits((1024, 1024))

    side = int(tf.ROTSPRITE_MAX_PIXELS**0.5) + 8
    big = np.zeros((side, side, 4), dtype=np.uint8)
    big[:, :] = A
    assert not tf.rotsprite_fits((big.shape[1], big.shape[0]))
    assert np.array_equal(
        tf.rotate(big, 15.0, expand=True, resample="rotsprite"),
        tf.rotate(big, 15.0, expand=True, resample="nearest"),
    )
