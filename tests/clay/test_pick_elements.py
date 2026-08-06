"""Element picking: projection, the occlusion rule, and the marquee masks.

Every one of these runs headless -- there is no GL context anywhere in the
file, which is the whole reason the picking rules live in numpy rather than in
an id-buffer readback.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock.studio.clay import adjacency as adj
from warlock.studio.clay import pick as bp
from warlock.studio.clay import primitives as prim
from warlock.studio.viewer import math3d as m3

WIDTH, HEIGHT = 400, 300


def _camera(distance: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """``(view_proj, eye)`` for a camera on +Z looking at the origin."""
    eye = m3.vec3(0.0, 0.0, distance)
    view = m3.look_at(eye, m3.vec3(0.0, 0.0, 0.0), m3.vec3(0.0, 1.0, 0.0))
    proj = m3.perspective(45.0, WIDTH / HEIGHT, 0.01, 100.0)
    return proj @ view, eye


def _screen(mesh, matrix=None):
    view_proj, eye = _camera()
    return bp.project(
        mesh.positions,
        m3.identity() if matrix is None else matrix,
        view_proj,
        eye,
        WIDTH,
        HEIGHT,
    )


# --- projection -------------------------------------------------------------


def test_the_origin_projects_to_the_middle_of_the_viewport() -> None:
    view_proj, eye = _camera()
    screen = bp.project(
        np.zeros((1, 3), dtype="f4"), m3.identity(), view_proj, eye, WIDTH, HEIGHT
    )
    assert screen.xy[0] == pytest.approx([WIDTH / 2, HEIGHT / 2])
    assert screen.depth[0] == pytest.approx(4.0)
    assert bool(screen.front[0])


def test_a_vertex_behind_the_camera_is_flagged_and_parked_off_screen() -> None:
    view_proj, eye = _camera()
    screen = bp.project(
        np.array([[0.0, 0.0, 10.0]], dtype="f4"),
        m3.identity(),
        view_proj,
        eye,
        WIDTH,
        HEIGHT,
    )
    assert not bool(screen.front[0])
    # Not mirrored through the origin into a plausible-looking pixel.
    assert screen.xy[0, 0] < -1000.0


def test_the_object_transform_is_applied_before_the_camera() -> None:
    view_proj, eye = _camera()
    moved = m3.compose(m3.vec3(1.0, 0.0, 0.0), m3.quat_identity(), m3.vec3(1, 1, 1))
    screen = bp.project(
        np.zeros((1, 3), dtype="f4"), moved, view_proj, eye, WIDTH, HEIGHT
    )
    assert screen.xy[0, 0] > WIDTH / 2
    assert screen.depth[0] > 4.0


def test_projecting_nothing_gives_empty_arrays_rather_than_raising() -> None:
    view_proj, eye = _camera()
    screen = bp.project(
        np.zeros((0, 3), dtype="f4"), m3.identity(), view_proj, eye, WIDTH, HEIGHT
    )
    assert screen.xy.shape == (0, 2)
    assert bp.nearest_vertex(screen, (0.0, 0.0)) is None
    assert bp.nearest_edge(screen, np.zeros((0, 2)), (0.0, 0.0)) is None


# --- nearest vertex and edge ------------------------------------------------


def test_clicking_a_vertex_picks_it_and_clicking_the_gap_picks_nothing() -> None:
    m = prim.box()
    screen = _screen(m)
    target = 6  # +X +Y +Z, the corner nearest the camera's upper right
    point = tuple(screen.xy[target])
    assert bp.nearest_vertex(screen, point) == target
    # Twenty pixels off, well outside the eight-pixel radius.
    assert bp.nearest_vertex(screen, (point[0] + 20.0, point[1])) is None


def test_a_click_between_two_vertices_takes_the_nearer_one() -> None:
    m = prim.box()
    screen = _screen(m)
    a, b = 6, 7  # the two top corners of the near face
    mid = (screen.xy[a] + screen.xy[b]) * 0.5
    nudged = mid + (screen.xy[a] - mid) * 0.9
    assert bp.nearest_vertex(screen, tuple(nudged), radius=50.0) == a


def test_the_far_side_of_a_closed_mesh_is_occluded() -> None:
    m = prim.box()
    screen = _screen(m)
    # The two corners at +X +Y differ only in Z, so they project to nearly the
    # same pixel; the surface is the near face at z = 0.5, distance 3.5.
    near = int(np.argmin(screen.depth))
    point = tuple(screen.xy[near])
    assert bp.nearest_vertex(screen, point, surface_depth=None) == near

    far = np.flatnonzero(
        (np.linalg.norm(screen.xy - screen.xy[near], axis=1) < 1.0)
        & (screen.depth > screen.depth[near] + 0.5)
    )
    if len(far):
        picked = bp.nearest_vertex(screen, point, surface_depth=float(screen.depth[near]))
        assert picked == near, "the near vertex wins, and the far one is rejected"

    lone = int(np.argmax(screen.depth))
    assert (
        bp.nearest_vertex(
            screen,
            tuple(screen.xy[lone]),
            surface_depth=float(screen.depth[near]),
            radius=1.0,
        )
        != lone
    )


def test_a_silhouette_click_with_no_surface_hit_still_picks_the_rim() -> None:
    m = prim.box()
    screen = _screen(m)
    rim = int(np.argmax(screen.depth))
    # surface_depth None is "the ray missed everything", so nothing occludes.
    assert bp.nearest_vertex(screen, tuple(screen.xy[rim]), surface_depth=None) == rim


def test_an_open_sheet_picks_on_both_sides_because_it_has_one_surface() -> None:
    # The plane lies in XZ, so it is edge-on to a camera on +Z; stand it up.
    m = prim.plane(size=(2.0, 2.0))
    upright = m3.compose(
        m3.vec3(),
        m3.quat_from_axis_angle(m3.vec3(1.0, 0.0, 0.0), np.pi / 2),
        m3.vec3(1.0, 1.0, 1.0),
    )
    screen = _screen(m, upright)
    depth = float(screen.depth.mean())
    for v in range(len(m.positions)):
        assert bp.nearest_vertex(screen, tuple(screen.xy[v]), surface_depth=depth) == v


def test_clicking_along_an_edge_picks_it_anywhere_on_its_length() -> None:
    m = prim.box()
    a = adj.adjacency(m)
    screen = _screen(m)
    edge = 0
    ends = a.edge_verts[edge]
    for t in (0.1, 0.5, 0.9):
        point = screen.xy[ends[0]] * (1 - t) + screen.xy[ends[1]] * t
        assert bp.nearest_edge(screen, a.edge_verts, tuple(point)) == edge


def test_an_edges_depth_is_interpolated_where_the_cursor_is() -> None:
    # An edge running diagonally away from the camera. Grabbing its near half
    # must work against a surface at 4.0 and its far half must not -- which is
    # only true if the depth is read where the cursor is rather than off an
    # endpoint.
    positions = np.array([[-1.0, 0.0, 1.0], [1.0, 0.0, -1.0]], dtype="f4")
    view_proj, eye = _camera()
    screen = bp.project(positions, m3.identity(), view_proj, eye, WIDTH, HEIGHT)
    edges = np.array([[0, 1]])
    assert screen.depth[0] < 4.0 < screen.depth[1]

    near_point = tuple(screen.xy[0] * 0.95 + screen.xy[1] * 0.05)
    assert bp.nearest_edge(screen, edges, near_point, surface_depth=4.0) == 0
    far_point = tuple(screen.xy[0] * 0.05 + screen.xy[1] * 0.95)
    assert bp.nearest_edge(screen, edges, far_point, surface_depth=4.0) is None


# --- marquee ----------------------------------------------------------------


def test_a_marquee_over_everything_takes_everything_including_the_far_side() -> None:
    m = prim.box()
    a = adj.adjacency(m)
    screen = _screen(m)
    whole = (0.0, 0.0, float(WIDTH), float(HEIGHT))
    assert len(bp.marquee_verts(screen, whole)) == len(m.positions)
    assert len(bp.marquee_edges(screen, a.edge_verts, whole)) == 12
    assert len(bp.marquee_faces(screen, m.loops, m.starts, whole)) == 6


def test_a_marquee_takes_an_edge_only_when_both_ends_are_in_it() -> None:
    m = prim.box()
    a = adj.adjacency(m)
    screen = _screen(m)
    # The left half of the viewport: some edges straddle the boundary.
    half = (0.0, 0.0, WIDTH / 2.0, float(HEIGHT))
    inside = set(bp.marquee_verts(screen, half).tolist())
    for lo, hi in bp.marquee_edges(screen, a.edge_verts, half).tolist():
        assert lo in inside and hi in inside
    straddling = [
        e
        for e in a.edge_verts.tolist()
        if (e[0] in inside) != (e[1] in inside)
    ]
    assert straddling, "the fixture has to contain a straddling edge to prove it"


def test_a_marquee_takes_a_face_only_when_all_its_corners_are_in_it() -> None:
    m = prim.box()
    screen = _screen(m)
    half = (0.0, 0.0, WIDTH / 2.0, float(HEIGHT))
    inside = set(bp.marquee_verts(screen, half).tolist())
    for f in bp.marquee_faces(screen, m.loops, m.starts, half).tolist():
        corners = m.loops[m.starts[f] : m.starts[f + 1]].tolist()
        assert set(corners) <= inside


def test_a_zero_area_marquee_takes_nothing() -> None:
    m = prim.box()
    screen = _screen(m)
    point = (float(screen.xy[0, 0]), float(screen.xy[0, 1]))
    empty = (point[0], point[1], point[0], point[1])
    assert len(bp.marquee_verts(screen, empty)) <= 1
    assert len(bp.marquee_faces(screen, m.loops, m.starts, empty)) == 0


def test_a_marquee_dragged_backwards_covers_the_same_region() -> None:
    m = prim.box()
    screen = _screen(m)
    forward = (50.0, 40.0, 350.0, 260.0)
    backward = (350.0, 260.0, 50.0, 40.0)
    assert np.array_equal(
        bp.marquee_verts(screen, forward), bp.marquee_verts(screen, backward)
    )


def test_marquee_masks_on_an_empty_mesh_come_back_empty() -> None:
    view_proj, eye = _camera()
    screen = bp.project(
        np.zeros((0, 3), dtype="f4"), m3.identity(), view_proj, eye, WIDTH, HEIGHT
    )
    rect = (0.0, 0.0, 10.0, 10.0)
    assert len(bp.marquee_verts(screen, rect)) == 0
    assert len(bp.marquee_edges(screen, np.zeros((0, 2)), rect)) == 0
    assert (
        len(bp.marquee_faces(screen, np.zeros(0, dtype="i4"), np.zeros(1, dtype="i4"), rect))
        == 0
    )
