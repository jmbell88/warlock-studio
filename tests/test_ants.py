"""Where the marching-ants dashes land.

The drawing is frame-thread imgui code with no headless test, which is exactly
why the arithmetic was pulled out of it: everything that can be *wrong* about
the ants -- a dash starting in the wrong place, the pattern restarting at every
corner, a dash cutting across a staircase instead of following it, the light and
dark halves swapping -- is in here and is a pure function of a loop, a zoom and
a phase.
"""

from __future__ import annotations

import numpy as np

from warlock.studio import ants

# A 3x3 square, which is four vertices and a perimeter of twelve: long enough
# to hold two whole six-pixel dashes and to make one cross a corner.
SQUARE = [(0, 0), (3, 0), (3, 3), (0, 3)]


def _one(loops):
    prepared = ants.prepare(loops)
    assert len(prepared) == 1
    return prepared[0]


def _lattice_rect(width: int, height: int) -> list[tuple[int, int]]:
    """Every lattice vertex of a rectangle, which is what ``contours()`` emits:
    unit steps with no collinear simplification."""
    top = [(x, 0) for x in range(width)]
    right = [(width, y) for y in range(height)]
    bottom = [(x, height) for x in range(width, 0, -1)]
    left = [(0, y) for y in range(height, 0, -1)]
    return top + right + bottom + left


def _staircase() -> list[tuple[int, int]]:
    """A closed loop with a direction change at every step -- the shape a lasso
    actually traces, and the one a per-dash chord would straighten out."""
    out: list[tuple[int, int]] = []
    for i in range(8):
        out += [(i, i), (i + 1, i)]
    for i in range(8, 0, -1):
        out += [(i, i), (i - 1, i)]
    return out


# --- preparing a loop -------------------------------------------------------


def test_a_prepared_loop_is_closed_and_carries_its_arc_length():
    verts, cum = _one([SQUARE])
    assert verts.shape == (5, 2)
    assert tuple(verts[-1]) == tuple(verts[0])
    assert np.array_equal(cum, [0.0, 3.0, 6.0, 9.0, 12.0])


def test_a_loop_with_no_length_is_not_a_loop():
    """Both of the degenerate shapes ``contours()`` will not produce and the
    pane must survive anyway: a single point, and a repeated one."""
    assert ants.prepare([[(4, 4)]]) == []
    assert ants.prepare([[(4, 4), (4, 4)]]) == []


# --- where the dashes fall --------------------------------------------------


def test_the_dash_pattern_runs_along_the_loop_and_not_along_each_segment():
    """The property the old walk did not have.

    Its inner ``while`` restarted at every vertex, so on a lattice of unit steps
    the boundaries fell every one pixel rather than every six. Here the first
    dash runs from 0 to 6, which is over the corner at 3 and a whole segment
    into the next side -- two lines, both lit, no boundary between them.
    """
    verts, cum = _one([SQUARE])
    starts, ends, lit = ants.dash_segments(
        verts, cum, zoom=1.0, offset=(0.0, 0.0), phase=0.0
    )

    assert np.array_equal(lit, [True, True, False, False])
    assert np.allclose(starts, [(0, 0), (3, 0), (3, 3), (0, 3)])
    assert np.allclose(ends, [(3, 0), (3, 3), (0, 3), (0, 0)])


def test_the_phase_moves_the_boundaries_and_clips_the_dash_that_hangs_off_the_start():
    """``phase`` is what makes them march. The dash straddling the loop's start
    is drawn from zero rather than wrapped -- which is what the old walk did,
    since ``walked`` simply began at ``-phase``."""
    verts, cum = _one([SQUARE])
    starts, ends, lit = ants.dash_segments(
        verts, cum, zoom=1.0, offset=(0.0, 0.0), phase=3.0
    )

    # The unlit tail of the dash that began before the loop did, then a whole
    # lit dash across the corner at 6, then the head of the next unlit one.
    assert np.array_equal(lit, [False, True, True, False])
    assert np.allclose(starts, [(0, 0), (3, 0), (3, 3), (0, 3)])
    assert np.allclose(ends, [(3, 0), (3, 3), (0, 3), (0, 0)])


def test_the_lit_half_is_the_one_the_old_walk_lit():
    """``int((walked + travelled) // DASH) % 2 == 0`` with ``walked`` starting
    at ``-phase``, evaluated inside each piece. Floor division and Python's
    modulo on a negative index are both load-bearing: the dash before the loop's
    start has index -1, and -1 % 2 is 1, which is off."""
    verts, cum = _one([SQUARE])
    for phase in (0.0, 1.0, 3.0, 5.5, 6.0, 11.9):
        starts, ends, lit = ants.dash_segments(
            verts, cum, zoom=1.0, offset=(0.0, 0.0), phase=phase
        )
        heads = _arc_positions(verts, cum, starts)
        tails = _arc_positions(verts, cum, ends)
        for index, (head, tail) in enumerate(zip(heads, tails, strict=True)):
            middle = (head + tail) / 2.0
            expected = int((middle - phase) // ants.DASH) % 2 == 0
            assert bool(lit[index]) == expected, (phase, head, tail)


def _arc_positions(verts, cum, points):
    """Arc length of each point, recovered by walking the loop.

    Deliberately not the inverse of the code under test: it measures distance
    from the loop's start along the polyline, which is the definition the dash
    pattern is written against. The last point of a closed loop is the first
    one, so a tail at the very end reads as 0 and is corrected to the total.
    """
    out = []
    for point in points:
        for index in range(len(cum) - 1):
            a, b = verts[index], verts[index + 1]
            span = cum[index + 1] - cum[index]
            along = float(np.dot(point - a, (b - a) / span))
            if -1e-9 <= along <= span + 1e-9 and np.allclose(
                a + (b - a) * (along / span), point, atol=1e-6
            ):
                out.append(cum[index] + along)
                break
        else:  # pragma: no cover - a point off the loop is the bug this catches
            raise AssertionError(f"{point} is not on the loop")
    for index in range(1, len(out)):
        if out[index] <= out[index - 1]:
            out[index] = cum[-1]
    return out


# --- following the outline --------------------------------------------------


def test_a_dash_never_cuts_the_corner_off_a_staircase():
    """The reason a dash is not one line from its start to its end.

    A contour is a staircase, and a chord across six steps of one both misses
    the steps and, on a one-pixel spike, skips it entirely -- the ants would
    stop sitting on the boundary the fill used, which is the whole contract.
    """
    verts, cum = _one([_staircase()])
    starts, ends, _ = ants.dash_segments(
        verts, cum, zoom=1.0, offset=(0.0, 0.0), phase=2.0
    )
    # Every step changes direction, so nothing merges and no line is longer
    # than the one unit step it lies on.
    assert float(np.hypot(*(ends - starts).T).max()) <= 1.0 + 1e-9
    _assert_on_the_outline(verts, cum, starts)
    _assert_on_the_outline(verts, cum, ends)


def _assert_on_the_outline(verts, cum, points):
    _arc_positions(verts, cum, points)  # raises if a point is off the polyline


def test_a_straight_run_of_lattice_steps_is_one_line_not_one_per_pixel():
    """Where the speed comes from. The contour of a marquee is one vertex per
    pixel; what gets drawn is one line per dash per side."""
    verts, cum = _one([_lattice_rect(200, 120)])
    starts, ends, lit = ants.dash_segments(
        verts, cum, zoom=1.0, offset=(0.0, 0.0), phase=0.0
    )
    perimeter = 2 * (200 + 120)
    assert len(cum) == perimeter + 1
    # A line per dash, plus one more wherever a corner splits one.
    assert len(starts) <= perimeter / ants.DASH + 8
    assert np.isclose(np.hypot(*(ends - starts).T).sum(), perimeter)
    assert 0.4 < lit.mean() < 0.6


def test_the_lines_tile_the_loop_exactly_once_leaving_no_gap():
    """Each line starts where the last one ended: a gap is a dotted outline and
    an overlap is a double-drawn one, and both look like a rendering bug."""
    verts, cum = _one([SQUARE])
    starts, ends, _ = ants.dash_segments(
        verts, cum, zoom=2.0, offset=(17.0, -4.0), phase=4.25
    )
    assert np.allclose(starts[1:], ends[:-1])
    assert np.allclose(starts[0], (17.0, -4.0))
    assert np.allclose(ends[-1], (17.0, -4.0))


# --- the view ---------------------------------------------------------------


def test_a_dash_is_six_screen_pixels_at_every_zoom():
    """``DASH`` is a screen constant, so a zoomed-in loop gets *more* dashes
    over the same pixels rather than longer ones."""
    verts, cum = _one([_lattice_rect(40, 40)])
    for zoom in (0.5, 1.0, 4.0):
        starts, ends, lit = ants.dash_segments(
            verts, cum, zoom=zoom, offset=(0.0, 0.0), phase=0.0
        )
        drawn = np.hypot(*(ends - starts).T)
        assert np.isclose(drawn.sum(), 160.0 * zoom)
        assert drawn.max() <= ants.DASH + 1e-9
        # Every whole dash is DASH long, so the count tracks the screen size.
        assert abs(len(starts) - 160.0 * zoom / ants.DASH) <= 8


def test_the_screen_transform_is_a_uniform_scale_about_the_offset():
    """The same affine ``inker_state.to_screen`` applies -- the pane passes it
    canvas (0, 0) *through* that function, so this is the half that has to
    match it."""
    verts, cum = _one([SQUARE])
    plain, _, _ = ants.dash_segments(verts, cum, zoom=1.0, offset=(0.0, 0.0), phase=0.0)
    moved, _, _ = ants.dash_segments(verts, cum, zoom=1.0, offset=(10.0, 20.0), phase=0.0)
    assert np.allclose(moved - plain, (10.0, 20.0))
    bigger, _, _ = ants.dash_segments(verts, cum, zoom=3.0, offset=(0.0, 0.0), phase=0.0)
    assert np.allclose(bigger.min(axis=0), (0.0, 0.0))
    assert np.allclose(bigger.max(axis=0), (9.0, 9.0))


def test_a_boundary_landing_exactly_on_a_vertex_lands_on_the_vertex():
    """``searchsorted(side="right") - 1`` is what puts it there rather than at
    the far end of the previous segment -- the same place the old walk put it,
    since it reset ``travelled`` at every vertex. The splice leaves a duplicate
    bound behind, and a zero-length line is dropped rather than drawn."""
    verts, cum = _one([[(0, 0), (6, 0), (6, 6), (0, 6)]])
    starts, ends, lit = ants.dash_segments(
        verts, cum, zoom=1.0, offset=(0.0, 0.0), phase=0.0
    )
    assert np.array_equal(lit, [True, False, True, False])
    assert np.allclose(ends[0], (6.0, 0.0))
    assert np.allclose(starts[1], (6.0, 0.0))


def test_a_zoomed_out_loop_smaller_than_one_dash_still_draws_all_of_itself():
    """A tiny selection at a low zoom: the whole perimeter is less than a dash,
    so it is one lit dash -- but still four lines, because it is a square."""
    verts, cum = _one([[(0, 0), (1, 0), (1, 1), (0, 1)]])
    starts, ends, lit = ants.dash_segments(
        verts, cum, zoom=0.05, offset=(0.0, 0.0), phase=0.0
    )
    assert lit.all() and len(starts) == 4
    assert np.allclose(starts[0], (0.0, 0.0))
    assert np.allclose(ends[-1], (0.0, 0.0))


def test_a_degenerate_zoom_draws_nothing_rather_than_dividing_by_it():
    verts, cum = _one([SQUARE])
    starts, _, lit = ants.dash_segments(verts, cum, zoom=0.0, offset=(0.0, 0.0), phase=0.0)
    assert len(starts) == 0 and len(lit) == 0


# --- viewport culling (B23) --------------------------------------------------


def test_cull_keeps_only_runs_touching_the_rect_and_never_reorders():
    verts, cum = _one([SQUARE])
    starts, ends, lit = ants.dash_segments(verts, cum, 10.0, (0.0, 0.0), 0.0)
    # A rectangle covering everything keeps every run, identically.
    all_kept = ants.cull(starts, ends, lit, (-100.0, -100.0, 1000.0, 1000.0))
    assert np.array_equal(all_kept[0], starts)
    assert np.array_equal(all_kept[1], ends)
    assert np.array_equal(all_kept[2], lit)
    # A rectangle over the left half drops every run wholly to its right,
    # keeps every run touching it, and preserves order among survivors.
    left = (-1.0, -1.0, 15.0, 100.0)
    ks, ke, kl = ants.cull(starts, ends, lit, left)
    assert 0 < len(ks) < len(starts)
    for a, b in zip(ks, ke, strict=True):
        assert min(a[0], b[0]) <= 15.0
    # Nothing visible: everything is culled.
    es, ee, el = ants.cull(starts, ends, lit, (500.0, 500.0, 600.0, 600.0))
    assert len(es) == len(ee) == len(el) == 0


def test_cull_is_conservative_about_touching_runs():
    """A run crossing the rect boundary is kept -- dropping it would clip the
    visible outline short."""
    starts = np.array([[0.0, 0.0], [20.0, 0.0]])
    ends = np.array([[10.0, 0.0], [30.0, 0.0]])
    lit = np.array([True, False])
    ks, _ke, kl = ants.cull(starts, ends, lit, (5.0, -1.0, 6.0, 1.0))
    assert len(ks) == 1 and bool(kl[0]) is True


def test_loop_box_is_the_canvas_space_aabb():
    verts, _cum = _one([SQUARE])
    assert ants.loop_box(verts) == (0.0, 0.0, 3.0, 3.0)


# --- the view's orientation (Ink9) -------------------------------------------


def test_a_basis_turns_the_dashes_without_touching_the_dash_pattern():
    """The whole reason the orientation is handed in separately from the zoom:
    every distance in ``dash_segments`` is an arc length in canvas space, so a
    transform that only turns leaves all of it alone."""
    import numpy as np

    from warlock.studio import ants

    verts = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 6.0], [0.0, 6.0], [0.0, 0.0]])
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(verts, axis=0), axis=1))])

    plain = ants.dash_segments(verts, cum, 1.0, (0.0, 0.0), 0.0)
    quarter = np.array([[0.0, -1.0], [1.0, 0.0]])
    turned = ants.dash_segments(verts, cum, 1.0, (0.0, 0.0), 0.0, basis=quarter)

    # Same number of runs and the same lit pattern -- only the coordinates move.
    assert len(plain[0]) == len(turned[0])
    assert plain[2].tolist() == turned[2].tolist()
    assert np.allclose(turned[0], plain[0] @ quarter.T)
    assert np.allclose(turned[1], plain[1] @ quarter.T)


def test_no_basis_is_exactly_what_the_upright_view_always_produced():
    import numpy as np

    from warlock.studio import ants

    verts = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0], [0.0, 0.0]])
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(verts, axis=0), axis=1))])
    a = ants.dash_segments(verts, cum, 2.0, (3.0, 5.0), 1.0)
    b = ants.dash_segments(verts, cum, 2.0, (3.0, 5.0), 1.0, basis=np.eye(2))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
