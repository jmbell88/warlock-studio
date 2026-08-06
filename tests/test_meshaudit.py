"""hole_fraction must find see-through gaps without inventing them on solid meshes.

The false-positive direction is the one that matters. This measurement exists to
decide whether a generation-flag change actually helped, so a harness that
reports holes on a watertight sphere would send the whole investigation the
wrong way -- which is exactly what happened with the checks it replaces.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from warlock.meshaudit import DEFAULT_VIEWS, hole_fraction


@pytest.fixture
def sphere_glb(tmp_path):
    """A watertight icosphere -- the no-false-positives control."""
    path = tmp_path / "sphere.glb"
    trimesh.creation.icosphere(subdivisions=4, radius=0.5).export(path)
    return path


def test_watertight_sphere_reports_no_holes(sphere_glb):
    result = hole_fraction(sphere_glb, resolution=512)
    assert result["worst"] == pytest.approx(0.0, abs=1e-4)
    assert all(view["blobs"] == 0 for view in result["views"])


def test_watertight_cube_reports_no_holes(tmp_path):
    path = tmp_path / "cube.glb"
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(path)
    assert hole_fraction(path, resolution=512)["worst"] == pytest.approx(0.0, abs=1e-4)


def test_a_hole_punched_through_both_walls_is_found(tmp_path):
    """Both caps removed, so a ray down +z passes through the mesh entirely.

    Removing only the near cap would not be see-through: with no backface
    culling the far wall still covers those pixels, and correctly so.
    """
    path = tmp_path / "pierced.glb"
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=0.5)
    keep = np.abs(sphere.face_normals[:, 2]) < 0.9
    trimesh.Trimesh(vertices=sphere.vertices, faces=sphere.faces[keep]).export(path)

    result = hole_fraction(path, views=((0.0, 0.0, 1.0),), resolution=512)
    assert result["worst"] > 0.01
    assert result["views"][0]["blobs"] >= 1


def test_a_solid_mesh_stays_solid_from_every_default_view(sphere_glb):
    result = hole_fraction(sphere_glb, resolution=256)
    assert len(result["views"]) == len(DEFAULT_VIEWS)
    for view in result["views"]:
        assert view["hole_px"] == 0
        assert view["silhouette_px"] > 0


def test_higher_resolution_does_not_manufacture_holes(sphere_glb):
    """Sub-pixel splatting is the one approximation here; it must not leak in."""
    for resolution in (256, 512, 1024):
        assert hole_fraction(sphere_glb, resolution=resolution)["worst"] == pytest.approx(
            0.0, abs=1e-4
        )


def test_reports_face_count_and_both_summary_statistics(sphere_glb):
    result = hole_fraction(sphere_glb, resolution=256)
    assert result["faces"] == 5120  # icosphere(subdivisions=4)
    assert result["mean"] <= result["worst"]


# --- the reachability half ---------------------------------------------------
#
# ``_enclosed_gaps`` used to be two iterative fixpoints, each O(image diameter)
# full-array passes. Connected components answer the same question in one linear
# pass, and "the same question" is the claim worth pinning: the fixpoints are
# kept here as the oracle so the replacement is checked against the code it
# replaced rather than against an idea of what it did.


def _fixpoint_gaps(covered: np.ndarray) -> np.ndarray:
    """The original flood fill: grow from the border until it stops changing."""
    free = ~covered
    reached = np.zeros_like(free)
    reached[0] |= free[0]
    reached[-1] |= free[-1]
    reached[:, 0] |= free[:, 0]
    reached[:, -1] |= free[:, -1]
    while True:
        grown = reached.copy()
        grown[1:] |= reached[:-1]
        grown[:-1] |= reached[1:]
        grown[:, 1:] |= reached[:, :-1]
        grown[:, :-1] |= reached[:, 1:]
        grown &= free
        if grown.sum() == reached.sum():
            break
        reached = grown
    return free & ~reached


def _fixpoint_blobs(mask: np.ndarray) -> int:
    """The original blob count: iterative label propagation to a fixpoint."""
    if not mask.any():
        return 0
    labels = np.zeros(mask.shape, dtype=np.int64)
    labels[mask] = np.arange(1, int(mask.sum()) + 1)
    while True:
        grown = labels.copy()
        grown[1:] = np.maximum(grown[1:], labels[:-1])
        grown[:-1] = np.maximum(grown[:-1], labels[1:])
        grown[:, 1:] = np.maximum(grown[:, 1:], labels[:, :-1])
        grown[:, :-1] = np.maximum(grown[:, :-1], labels[:, 1:])
        grown[~mask] = 0
        if np.array_equal(grown, labels):
            break
        labels = grown
    return len(np.unique(labels)) - 1


def _assert_matches_the_fixpoints(covered: np.ndarray) -> None:
    from warlock.meshaudit import _enclosed_gaps

    holes, blobs = _enclosed_gaps(covered)
    assert np.array_equal(holes, _fixpoint_gaps(covered))
    assert blobs == _fixpoint_blobs(_fixpoint_gaps(covered))


@pytest.mark.parametrize("seed", range(8))
def test_connected_components_match_the_fixpoint_oracle_on_noise(seed):
    rng = np.random.default_rng(seed)
    # Dense enough to enclose gaps, sparse enough to leave background reachable.
    _assert_matches_the_fixpoints(rng.random((64, 64)) < 0.55)


def test_a_ring_encloses_exactly_one_hole():
    covered = np.zeros((32, 32), dtype=bool)
    covered[8:24, 8:24] = True
    covered[12:20, 12:20] = False
    _assert_matches_the_fixpoints(covered)
    from warlock.meshaudit import _enclosed_gaps

    holes, blobs = _enclosed_gaps(covered)
    assert blobs == 1
    assert int(holes.sum()) == 64


def test_two_holes_touching_only_at_a_corner_stay_two():
    """Four-connectivity is the contract the fixpoints had -- eight would merge
    these into one blob and silently halve every count in the corpus."""
    covered = np.ones((16, 16), dtype=bool)
    covered[0, :] = covered[-1, :] = covered[:, 0] = covered[:, -1] = False
    covered[6, 6] = covered[7, 7] = False
    _assert_matches_the_fixpoints(covered)
    from warlock.meshaudit import _enclosed_gaps

    assert _enclosed_gaps(covered)[1] == 2


def test_a_fully_covered_view_has_no_gaps_and_no_blobs():
    _assert_matches_the_fixpoints(np.ones((16, 16), dtype=bool))


def test_a_fully_empty_view_is_all_background():
    _assert_matches_the_fixpoints(np.zeros((16, 16), dtype=bool))


def test_a_gap_touching_the_border_is_background_not_a_hole():
    covered = np.ones((16, 16), dtype=bool)
    covered[0:4, 7] = False  # a slot cut in from the top edge
    _assert_matches_the_fixpoints(covered)
    from warlock.meshaudit import _enclosed_gaps

    holes, blobs = _enclosed_gaps(covered)
    assert blobs == 0
    assert not holes.any()


def test_chunking_the_rasteriser_is_bit_identical_to_one_pass(sphere_glb, monkeypatch):
    """The (n, k, k) working set is chunked to bound a commit spike on a
    500k-triangle mesh. `covered` is written in place, so the chunk size must
    be invisible in the result -- forcing it down to one triangle per pass is
    the strongest version of that claim."""
    import warlock.meshaudit as meshaudit

    unchunked = hole_fraction(sphere_glb, resolution=256)
    monkeypatch.setattr(meshaudit, "_BATCH_MAX_CELLS", 1)
    chunked = hole_fraction(sphere_glb, resolution=256)
    assert chunked == unchunked
