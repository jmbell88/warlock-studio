"""Combining per-view projections into one atlas.

The arithmetic half of a re-texture, and the three rules in it that are the
difference between a texture and a mess: a texel no view could see keeps its
original colour, the result is grown past every island edge, and nothing here
pretends there was an occlusion test.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from warlock.pipelines import retexture


def _write(path, arr, mode="RGB"):
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode).save(path)


# -- the weighted mean -------------------------------------------------------


def test_the_only_view_that_saw_a_texel_decides_it():
    colours = np.zeros((2, 4, 4, 3), dtype=np.float32)
    colours[0] = 1.0  # white
    weights = np.zeros((2, 4, 4), dtype=np.float32)
    weights[0] = 1.0
    base = np.zeros((4, 4, 3), dtype=np.float32)
    assert np.allclose(retexture.combine(colours, weights, base), 1.0)


def test_two_views_are_averaged_by_their_weights():
    colours = np.zeros((2, 2, 2, 3), dtype=np.float32)
    colours[0], colours[1] = 1.0, 0.0
    weights = np.stack(
        [np.full((2, 2), 0.75, np.float32), np.full((2, 2), 0.25, np.float32)]
    )
    out = retexture.combine(colours, weights, np.zeros((2, 2, 3), np.float32))
    assert np.allclose(out, 0.75)


def test_a_texel_no_view_saw_keeps_its_original_colour():
    """Zero weight means "nothing looked at this", not "this is black".

    It happens inside every fold and under every overhang, and filling those
    with the sum turns the inside of a barrel into a void.
    """
    colours = np.ones((2, 2, 2, 3), dtype=np.float32)
    weights = np.zeros((2, 2, 2), dtype=np.float32)
    base = np.full((2, 2, 3), 0.3, dtype=np.float32)
    assert np.allclose(retexture.combine(colours, weights, base), 0.3)


def test_an_edge_on_view_is_dropped_rather_than_scaled_down():
    """A facing ratio near zero is one pixel smeared along a strip of surface.

    Scaling it down is not enough: the texels it covers are exactly the ones no
    other view covers either, so at any non-zero weight the smear wins by
    default wherever it is the only contributor. Here the base must show
    through instead.
    """
    colours = np.ones((1, 2, 2, 3), dtype=np.float32)
    weights = np.full((1, 2, 2), retexture.MIN_FACING / 2, dtype=np.float32)
    base = np.zeros((2, 2, 3), dtype=np.float32)
    assert np.allclose(retexture.combine(colours, weights, base), 0.0)


def test_a_weight_exactly_at_the_floor_still_counts():
    colours = np.ones((1, 2, 2, 3), dtype=np.float32)
    weights = np.full((1, 2, 2), retexture.MIN_FACING, dtype=np.float32)
    base = np.zeros((2, 2, 3), dtype=np.float32)
    assert np.allclose(retexture.combine(colours, weights, base), 1.0)


# -- the dilation ------------------------------------------------------------


def test_an_island_grows_into_the_texels_around_it():
    image = np.zeros((7, 7, 3), dtype=np.float32)
    mask = np.zeros((7, 7), dtype=bool)
    image[3, 3] = 1.0
    mask[3, 3] = True
    out = retexture.dilate(image, mask, passes=1)
    # The four neighbours take the island's colour; the diagonals do not yet.
    assert np.allclose(out[2, 3], 1.0)
    assert np.allclose(out[4, 3], 1.0)
    assert np.allclose(out[3, 2], 1.0)
    assert np.allclose(out[3, 4], 1.0)
    assert np.allclose(out[2, 2], 0.0)


def test_dilation_never_overwrites_a_written_texel():
    image = np.zeros((5, 5, 3), dtype=np.float32)
    image[:, :2] = 1.0
    mask = np.zeros((5, 5), dtype=bool)
    mask[:, :2] = True
    out = retexture.dilate(image, mask, passes=3)
    assert np.allclose(out[:, :2], 1.0)


def test_dilation_does_not_wrap():
    """`material.py` wraps everything and this must not.

    A UV atlas is not periodic: rolling here would carry the left edge's
    islands onto the right, which is a bleed between unrelated parts of one
    mesh -- and it would look like a texture bug rather than a filter bug.
    """
    image = np.zeros((4, 4, 3), dtype=np.float32)
    mask = np.zeros((4, 4), dtype=bool)
    image[:, 0] = 1.0
    mask[:, 0] = True
    out = retexture.dilate(image, mask, passes=2)
    assert np.allclose(out[:, -1], 0.0), "the last column took the first column's colour"


def test_dilation_of_a_fully_written_image_changes_nothing():
    rng = np.random.default_rng(0)
    image = rng.random((6, 6, 3)).astype(np.float32)
    out = retexture.dilate(image, np.ones((6, 6), dtype=bool), passes=4)
    assert np.array_equal(out, image)


# -- the view basis ----------------------------------------------------------


def test_yaw_zero_looks_from_in_front_of_the_subject():
    """The templates put forward at -Y, which is the sentence `sheet.py`
    carries about column 0. If this drifts, a re-texture's "front" is the
    mesh's back and every restyle is applied to the wrong side."""
    x, y, z = retexture.view_matrix(0.0, 0.0)
    assert (round(x, 6), round(y, 6), round(z, 6)) == (0.0, -1.0, 0.0)


def test_the_six_views_are_distinct_directions():
    dirs = [retexture.view_matrix(*v) for v in retexture.VIEWS]
    for i, a in enumerate(dirs):
        for b in dirs[i + 1 :]:
            assert not np.allclose(a, b, atol=1e-3)


def test_every_view_direction_is_a_unit_vector():
    for view in retexture.VIEWS:
        assert np.isclose(np.linalg.norm(retexture.view_matrix(*view)), 1.0)


# -- assemble ----------------------------------------------------------------


def _bakes(tmp_path, n=2, size=8):
    for i in range(n):
        colour = np.zeros((size, size, 3), dtype=np.float32)
        colour[..., i % 3] = 1.0
        weight = np.zeros((size, size), dtype=np.float32)
        weight[:, i * 2 : i * 2 + 2] = 1.0
        _write(tmp_path / f"bake_{i:02d}.png", colour)
        _write(tmp_path / f"weight_{i:02d}.png", weight, "L")
    return n


def test_assemble_writes_an_atlas_and_reports_its_coverage(tmp_path):
    n = _bakes(tmp_path)
    base = tmp_path / "base.png"
    _write(base, np.full((8, 8, 3), 0.25, np.float32))

    report = retexture.assemble(tmp_path, base, tmp_path / "out.png", count=n)

    assert report is not None
    assert report["views"] == n
    assert report["size"] == [8, 8]
    # Four of eight columns were covered by some view.
    assert report["coverage"] == pytest.approx(0.5)
    assert (tmp_path / "out.png").exists()


def test_assemble_says_out_loud_that_nothing_was_occlusion_tested(tmp_path):
    """The limitation rides in the record, not only in a docstring: the record
    is what a user reading the job's params sees, and it is the finding that
    decides whether Tier 3 is needed."""
    n = _bakes(tmp_path)
    report = retexture.assemble(tmp_path, None, tmp_path / "out.png", count=n)
    assert report["occlusion_tested"] is False


def test_assemble_with_no_bakes_at_all_is_a_none(tmp_path):
    assert retexture.assemble(tmp_path, None, tmp_path / "out.png", count=3) is None


def test_assemble_skips_a_view_whose_pair_is_incomplete(tmp_path):
    n = _bakes(tmp_path, n=2)
    (tmp_path / "weight_01.png").unlink()
    report = retexture.assemble(tmp_path, None, tmp_path / "out.png", count=n)
    assert report["views"] == 1


def test_a_bake_of_the_wrong_size_refuses_rather_than_broadcasting(tmp_path):
    """A mismatched bake is a Blender-side bug, and averaging whatever happens
    to broadcast would hide it inside a texture nobody can read back."""
    _bakes(tmp_path, n=1, size=8)
    _write(tmp_path / "bake_01.png", np.zeros((4, 4, 3), np.float32))
    _write(tmp_path / "weight_01.png", np.ones((4, 4), np.float32), "L")
    assert retexture.assemble(tmp_path, None, tmp_path / "out.png", count=2) is None


def test_a_base_of_another_size_is_ignored_rather_than_resized(tmp_path):
    n = _bakes(tmp_path, size=8)
    base = tmp_path / "base.png"
    _write(base, np.zeros((16, 16, 3), np.float32))
    assert retexture.assemble(tmp_path, base, tmp_path / "out.png", count=n) is not None


def test_an_uncovered_texel_with_no_base_reads_as_grey_rather_than_a_hole(tmp_path):
    # Wide enough that the last column is further from the covered ones than
    # DILATE can reach -- the point is what an *unreachable* texel holds, and on
    # a narrow canvas the dilation legitimately fills the whole thing.
    size = 4 * retexture.DILATE + 8
    n = _bakes(tmp_path, size=size)
    retexture.assemble(tmp_path, None, tmp_path / "out.png", count=n)
    with Image.open(tmp_path / "out.png") as im:
        arr = np.asarray(im.convert("RGB"))
    assert arr[0, -1].tolist() == [128, 128, 128]


def test_the_module_stays_on_the_host_side_of_the_bpy_split():
    import pathlib

    source = pathlib.Path(retexture.__file__).read_text(encoding="utf-8")
    for banned in ("import bpy", "from ..service", "from ..queue", "from ..studio"):
        assert banned not in source


# -- the two copies of the view basis ----------------------------------------


def test_the_workers_view_direction_matches_this_modules():
    """`blender_worker` keeps its own copy and must not drift from this one.

    The worker runs in a bpy interpreter and imports nothing from the host half
    by design, which is the same split `rigging.fit_template` sits on -- and it
    gets the same treatment: the duplicate is pinned by a test rather than by a
    comment asking the two to stay identical. A drift here would rotate every
    weight map relative to the colours it weights, which looks like a bad
    restyle rather than like a bug.
    """
    import ast
    import pathlib

    from warlock.pipelines import blender_worker

    source = pathlib.Path(blender_worker.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_view_direction"
    )
    namespace: dict = {}
    exec(compile(ast.Module([fn], []), "<worker>", "exec"), namespace)  # noqa: S102

    for yaw, pitch in retexture.VIEWS:
        assert namespace["_view_direction"](yaw, pitch) == pytest.approx(
            retexture.view_matrix(yaw, pitch)
        )


def test_the_specs_carry_the_views_they_are_given():
    from warlock import rigging
    from warlock.pipelines import blender_worker

    views = list(retexture.VIEWS)
    v = rigging.views_spec(
        __import__("pathlib").Path("m.glb"),
        __import__("pathlib").Path("d"),
        views,
        size=512,
    )
    p = rigging.project_spec(
        __import__("pathlib").Path("m.glb"),
        __import__("pathlib").Path("d"),
        __import__("pathlib").Path("o"),
        views,
        size=512,
        texture_size=1024,
    )
    assert v["op"] in blender_worker.OPS and p["op"] in blender_worker.OPS
    # Lists, not tuples: the spec is JSON and a tuple would come back a list on
    # the far side anyway, so it is written as one here.
    assert v["views"] == [list(x) for x in views]
    assert p["views"] == v["views"]
    # Both ops frame from the same render size, or the projection lands
    # somewhere the colours are not.
    assert p["size"] == v["size"]
