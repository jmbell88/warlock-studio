"""Combining per-view projections into one atlas.

The arithmetic half of a re-texture, and the three rules in it that are the
difference between a texture and a mess: a texel no view could see keeps its
original colour, the result is grown past every island edge, and nothing here
pretends there was an occlusion test.
"""

from __future__ import annotations

from pathlib import Path

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


def test_the_views_are_distinct_directions():
    dirs = [retexture.view_matrix(*v) for v in retexture.VIEWS]
    for i, a in enumerate(dirs):
        for b in dirs[i + 1 :]:
            assert not np.allclose(a, b, atol=1e-3)


def test_the_basis_is_six_axes_and_four_upper_diagonals():
    """Ten views: the axis six a person means by "front, back, sides, top,
    bottom", plus the four upper 3/4 diagonals -- the directions that look
    *through* a perforated shell onto the interior walls the axis set either
    misses or, before the occlusion test, smeared."""
    assert len(retexture.VIEWS) == 10
    for diagonal in ((45.0, 35.0), (135.0, 35.0), (225.0, 35.0), (315.0, 35.0)):
        assert diagonal in retexture.VIEWS


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


def test_a_missing_pair_refuses_the_whole_assembly(tmp_path):
    """It used to average whatever it could read. A partial set looks like a
    finished re-texture and ships a mesh restyled on some sides only, with
    nothing anywhere saying so -- and the sides that failed are exactly the
    ones that keep their old skin, which is indistinguishable from the stated
    "a texel no view saw keeps its colour" behaviour."""
    n = _bakes(tmp_path, n=2)
    (tmp_path / "weight_01.png").unlink()
    assert retexture.assemble(tmp_path, None, tmp_path / "out.png", count=n) is None


def test_more_views_asked_for_than_were_baked_is_a_refusal(tmp_path):
    _bakes(tmp_path, n=2)
    assert retexture.assemble(tmp_path, None, tmp_path / "out.png", count=3) is None


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


# -- putting the atlas back into the GLB -------------------------------------


def _png(colour, size=8) -> bytes:
    import io

    buf = io.BytesIO()
    Image.new("RGB", (size, size), colour).save(buf, "PNG")
    return buf.getvalue()


def _bin(body: bytes) -> bytes:
    import struct

    from warlock import glbio

    return struct.pack("<II", len(body), glbio.CHUNK_BIN) + body


def _textured_glb(path, *, shared_image=False, materials=1) -> bytes:
    """A minimal GLB whose one image is the base colour of every material.

    Hand-built rather than exported: the swap is a container-format operation,
    so the fixture has to be a container rather than whatever an exporter felt
    like emitting.
    """
    import struct

    from warlock import glbio

    png = _png((10, 20, 30))
    body = png + b"\0" * (-len(png) % 4)
    textures = [{"source": 0}]
    if shared_image:
        # Two textures over one image, which is the case an in-place overwrite
        # gets wrong.
        textures.append({"source": 0})
    gltf = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(body)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": len(png)}],
        "images": [{"bufferView": 0, "mimeType": "image/png"}],
        "textures": textures,
        "materials": [
            {"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
            for _ in range(materials)
        ],
    }
    if shared_image:
        gltf["materials"][0].setdefault("normalTexture", {"index": 1})
    header = struct.pack("<III", glbio.GLB_MAGIC, 2, 0)
    rest = struct.pack("<II", len(body), glbio.CHUNK_BIN) + body
    data = glbio.rebuild_glb(header, gltf, rest)
    path.write_bytes(data)
    return png


def test_the_current_albedo_comes_back_out_at_the_bake_resolution(tmp_path):
    """`assemble`'s base has to be addressable at the size the projections were
    baked at, or "keep the old colour" cannot be expressed for a single texel."""
    glb = tmp_path / "model.glb"
    _textured_glb(glb)
    out = tmp_path / "base.png"
    assert retexture.extract_base_colour(glb, out, size=32) is True
    with Image.open(out) as im:
        assert im.size == (32, 32)
        assert im.convert("RGB").getpixel((0, 0)) == (10, 20, 30)


def test_a_mesh_with_no_albedo_is_a_false_rather_than_a_raise(tmp_path):
    """An untextured mesh is a legitimate input, and `assemble` already has an
    honest stand-in for a missing base."""
    import struct

    from warlock import glbio

    glb = tmp_path / "bare.glb"
    header = struct.pack("<III", glbio.GLB_MAGIC, 2, 0)
    glb.write_bytes(glbio.rebuild_glb(header, {"asset": {"version": "2.0"}}, b""))
    assert retexture.extract_base_colour(glb, tmp_path / "b.png", size=8) is False


def test_a_swapped_atlas_reads_back_as_the_new_albedo(tmp_path):
    glb = tmp_path / "model.glb"
    _textured_glb(glb)
    atlas = tmp_path / "atlas.png"
    atlas.write_bytes(_png((200, 100, 50), size=16))

    dest = tmp_path / "new.glb"
    assert retexture.swap_base_colour(glb, atlas, dest) is True
    out = tmp_path / "read.png"
    assert retexture.extract_base_colour(dest, out, size=16) is True
    with Image.open(out) as im:
        assert im.convert("RGB").getpixel((0, 0)) == (200, 100, 50)


def test_the_swap_appends_rather_than_overwriting_a_shared_image(tmp_path):
    """The short version -- rewrite the image's bytes in place -- is wrong the
    moment anything else references the same image: a mesh whose normal map
    happens to be the same PNG would have it replaced by an albedo, silently,
    and only some meshes would show it."""
    glb = tmp_path / "model.glb"
    original = _textured_glb(glb, shared_image=True)
    atlas = tmp_path / "atlas.png"
    atlas.write_bytes(_png((200, 100, 50), size=16))
    dest = tmp_path / "new.glb"
    assert retexture.swap_base_colour(glb, atlas, dest) is True

    from warlock import glbio

    gltf, buffer = glbio.read_glb(dest)
    # The other slot still points at the original image, byte for byte.
    normal = gltf["materials"][0]["normalTexture"]["index"]
    assert normal == 1
    kept = gltf["textures"][normal]["source"]
    view = gltf["bufferViews"][gltf["images"][kept]["bufferView"]]
    start = view.get("byteOffset", 0)
    assert buffer[start : start + view["byteLength"]] == original


def test_every_material_sharing_the_atlas_is_repointed(tmp_path):
    """_first_base_colour finds *a* slot; a multi-material mesh over one atlas
    would otherwise come back half restyled."""
    glb = tmp_path / "model.glb"
    _textured_glb(glb, materials=3)
    atlas = tmp_path / "atlas.png"
    atlas.write_bytes(_png((200, 100, 50), size=16))
    dest = tmp_path / "new.glb"
    assert retexture.swap_base_colour(glb, atlas, dest) is True

    from warlock import glbio

    gltf, _ = glbio.read_glb(dest)
    indices = {
        m["pbrMetallicRoughness"]["baseColorTexture"]["index"]
        for m in gltf["materials"]
    }
    assert indices == {len(gltf["textures"]) - 1}


def test_the_swap_leaves_the_rest_of_the_document_alone(tmp_path):
    """Only the JSON chunk and the tail of the BIN chunk move. Everything the
    grounding transform put in the node graph has to survive, which is the same
    argument normalize_glb makes about not re-exporting."""
    glb = tmp_path / "model.glb"
    _textured_glb(glb)

    from warlock import glbio

    before, buffer_before = glbio.read_glb(glb)
    before["nodes"] = [{"name": "grounding", "scale": [2.0, 2.0, 2.0]}]
    header, doc, rest = glbio.split_glb(glb.read_bytes())
    doc["nodes"] = before["nodes"]
    glb.write_bytes(glbio.rebuild_glb(header, doc, rest))

    atlas = tmp_path / "atlas.png"
    atlas.write_bytes(_png((1, 2, 3), size=4))
    dest = tmp_path / "new.glb"
    assert retexture.swap_base_colour(glb, atlas, dest) is True
    after, buffer_after = glbio.read_glb(dest)
    assert after["nodes"] == before["nodes"]
    # Every original byte still at its original offset -- the new PNG is
    # appended, so nothing that pointed into the buffer had to be adjusted.
    assert buffer_after[: len(buffer_before)] == buffer_before
    assert after["buffers"][0]["byteLength"] == len(buffer_after)


# -- what a re-texture makes stale, and what it deliberately does not ---------


def test_the_surface_exports_are_a_stated_subset_of_the_derived_ones():
    """One authority for the whole set, one stated subset of it, and the
    containment asserted -- so a new export cannot join one list and quietly
    miss the other."""
    from warlock.service import files

    assert set(retexture.SURFACE_DERIVED) < set(files.DERIVED)
    # The two left out, named rather than merely absent: an STL is geometry and
    # a convex collision hull has no material at all, so a re-texture -- which
    # changes no geometry -- leaves both byte-identical.
    assert set(files.DERIVED) - set(retexture.SURFACE_DERIVED) == {
        "model.stl",
        "collision.glb",
    }


def test_a_mesh_with_no_uvs_cannot_be_given_a_skin(tmp_path):
    """Attaching an atlas anyway renders one texel's colour over everything,
    which reads as a failed restyle rather than as an unwrapped mesh."""
    import struct

    from warlock import glbio

    glb = tmp_path / "bare.glb"
    header = struct.pack("<III", glbio.GLB_MAGIC, 2, 0)
    doc = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": 4}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
    }
    glb.write_bytes(glbio.rebuild_glb(header, doc, _bin(bytes(4))))
    atlas = tmp_path / "atlas.png"
    atlas.write_bytes(_png((1, 2, 3), size=4))
    assert retexture.swap_base_colour(glb, atlas, tmp_path / "o.glb") is False
    assert not (tmp_path / "o.glb").exists()


def test_an_untextured_unwrapped_mesh_gains_the_slot(tmp_path):
    """The Clay case, and the interesting one: a hand-modelled box carries a
    real box unwrap where a reconstruction carries roughly one texel per
    triangle. Refusing it would put the only meshes with usable UVs out of
    reach of the one feature that needs them.
    """
    import struct

    from warlock import glbio

    glb = tmp_path / "clay.glb"
    header = struct.pack("<III", glbio.GLB_MAGIC, 2, 0)
    doc = {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": 4}],
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}}]}
        ],
    }
    glb.write_bytes(glbio.rebuild_glb(header, doc, _bin(bytes(4))))
    atlas = tmp_path / "atlas.png"
    atlas.write_bytes(_png((200, 100, 50), size=16))
    dest = tmp_path / "out.glb"
    assert retexture.swap_base_colour(glb, atlas, dest) is True

    gltf, _ = glbio.read_glb(dest)
    # A material was minted and every primitive points at it.
    assert gltf["meshes"][0]["primitives"][0]["material"] == 0
    slot = gltf["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]
    assert slot["index"] == 0
    out = tmp_path / "read.png"
    assert retexture.extract_base_colour(dest, out, size=16) is True
    with Image.open(out) as im:
        assert im.convert("RGB").getpixel((0, 0)) == (200, 100, 50)


def test_the_default_atlas_size_is_the_mesh_s_own(tmp_path):
    """Measured, not assumed. A fixed size changed nothing a view covered and
    quietly halved the resolution of the ~63% no view did -- and that is the
    half a re-texture is supposed to leave exactly as it found it.
    """
    glb = tmp_path / "model.glb"
    _textured_glb(glb)  # its atlas is 8 px square
    assert retexture.atlas_size(glb) == 8


def test_an_unreadable_or_absent_atlas_has_no_size(tmp_path):
    import struct

    from warlock import glbio

    glb = tmp_path / "bare.glb"
    header = struct.pack("<III", glbio.GLB_MAGIC, 2, 0)
    glb.write_bytes(glbio.rebuild_glb(header, {"asset": {"version": "2.0"}}, b""))
    assert retexture.atlas_size(glb) is None
    assert retexture.atlas_size(tmp_path / "nothing.glb") is None


def test_a_swap_that_cannot_read_the_mesh_raises_rather_than_saying_no_albedo(tmp_path):
    """An I/O failure is not "this mesh has no albedo slot". Returning False
    for it made the caller raise that message and send the user to look at a
    mesh that was fine."""
    with pytest.raises(OSError):
        retexture.swap_base_colour(
            tmp_path / "missing.glb", tmp_path / "atlas.png", tmp_path / "out.glb"
        )
    assert not (tmp_path / "out.glb").exists()


# -- the occlusion test --------------------------------------------------------
#
# The 2026-08-08 measurement's verdict was "recolour, not re-texture", and its
# named cause was coverage: the weight was facing x render alpha, so a surface
# pointing at a camera got that camera's pixels whether or not anything sat in
# the way, and everything behind a perforated shell was masked by the shell's
# own holes. The fix is a per-texel depth compare -- zread is what the camera
# actually saw at this texel's pixel, zsurf is the texel's own camera depth,
# both in the inverted near=1/far=0 encoding -- done on the host, where it can
# be tested without a Blender run.


def test_depth_encode_puts_near_at_one_and_far_at_zero():
    # extent 1, distance 2: the subject spans camera depths 1..3.
    assert retexture.depth_encode(1.0, 1.0, 2.0) == pytest.approx(1.0)
    assert retexture.depth_encode(3.0, 1.0, 2.0) == pytest.approx(0.0)
    assert retexture.depth_encode(2.0, 1.0, 2.0) == pytest.approx(0.5)


def test_depth_encode_clamps_outside_the_framed_range():
    assert retexture.depth_encode(0.0, 1.0, 2.0) == pytest.approx(1.0)
    assert retexture.depth_encode(9.0, 1.0, 2.0) == pytest.approx(0.0)


def test_the_workers_depth_terms_match_this_modules():
    """`blender_worker` encodes depth in shader nodes as (offset - d) * scale,
    and this module decodes with `depth_encode`. The two are pinned the way
    `_view_direction` is: a drift here reads every visibility compare against
    the wrong plane, which looks like random dropout rather than like a bug."""
    import ast
    import pathlib

    from warlock.pipelines import blender_worker

    source = pathlib.Path(blender_worker.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_depth_terms"
    )
    namespace: dict = {}
    exec(compile(ast.Module([fn], []), "<worker>", "exec"), namespace)  # noqa: S102

    for extent, distance in ((1.0, 2.0), (3.5, 7.0), (0.25, 0.5)):
        offset, scale = namespace["_depth_terms"](extent, distance)
        for d in (distance - extent, distance, distance + extent, 0.0):
            assert min(max((offset - d) * scale, 0.0), 1.0) == pytest.approx(
                retexture.depth_encode(d, extent, distance)
            )


def test_a_texel_the_camera_actually_saw_is_fully_visible():
    z = np.full((2, 2), 0.6, dtype=np.float32)
    assert np.allclose(retexture.visibility(z, z), 1.0)


def test_a_texel_behind_a_nearer_surface_is_invisible():
    zsurf = np.full((2, 2), 0.4, dtype=np.float32)
    zread = np.full((2, 2), 0.5, dtype=np.float32)  # something nearer in the way
    assert np.allclose(retexture.visibility(zread, zsurf), 0.0)


def test_background_depth_of_zero_reads_as_visible():
    """No occluder rendered at this pixel means zread is the encoding's far
    plane -- which the inverted near=1 encoding makes 0, so "nothing in the
    way" needs no special case."""
    zread = np.zeros((2, 2), dtype=np.float32)
    zsurf = np.full((2, 2), 0.4, dtype=np.float32)
    assert np.allclose(retexture.visibility(zread, zsurf), 1.0)


def test_quantisation_jitter_within_the_bias_stays_visible():
    """8-bit channels quantise the depth range at 1/255; a texel must not lose
    its own visibility to that."""
    zsurf = np.full((2, 2), 0.5, dtype=np.float32)
    zread = zsurf + 1.0 / 255.0
    assert np.allclose(retexture.visibility(zread, zsurf), 1.0)


def test_visibility_fades_rather_than_stepping_inside_the_bias_band():
    zsurf = np.zeros((1, 1), dtype=np.float32)
    inside = retexture.visibility(np.full((1, 1), 4.0 / 255.0, np.float32), zsurf)
    assert 0.0 < float(inside[0, 0]) < 1.0
    nearer = retexture.visibility(np.full((1, 1), 3.0 / 255.0, np.float32), zsurf)
    further = retexture.visibility(np.full((1, 1), 5.0 / 255.0, np.float32), zsurf)
    assert float(nearer[0, 0]) > float(inside[0, 0]) > float(further[0, 0])


def test_combine_drops_an_occluded_view_even_at_full_facing():
    """The whole point: facing said "this camera sees the texel square on" and
    the depth compare says "through a wall". The wall wins."""
    colours = np.ones((1, 2, 2, 3), dtype=np.float32)
    weights = np.ones((1, 2, 2), dtype=np.float32)
    vis = np.zeros((1, 2, 2), dtype=np.float32)
    base = np.full((2, 2, 3), 0.3, dtype=np.float32)
    out = retexture.combine(colours, weights, base, vis=vis)
    assert np.allclose(out, 0.3)


def test_combine_with_full_visibility_is_exactly_the_legacy_result():
    rng = np.random.default_rng(7)
    colours = rng.random((3, 4, 4, 3)).astype(np.float32)
    weights = rng.random((3, 4, 4)).astype(np.float32)
    base = rng.random((4, 4, 3)).astype(np.float32)
    legacy = retexture.combine(colours, weights, base)
    seen = retexture.combine(
        colours, weights, base, vis=np.ones((3, 4, 4), np.float32)
    )
    assert np.array_equal(legacy, seen)


def test_visibility_multiplies_after_the_facing_floor():
    """A half-visible texel of a well-facing view still speaks at half weight;
    a sub-floor facing stays dropped no matter how visible it is."""
    colours = np.stack(
        [np.ones((2, 2, 3), np.float32), np.zeros((2, 2, 3), np.float32)]
    )
    weights = np.stack(
        [np.full((2, 2), 0.8, np.float32), np.full((2, 2), 0.8, np.float32)]
    )
    vis = np.stack(
        [np.full((2, 2), 0.5, np.float32), np.ones((2, 2), np.float32)]
    )
    base = np.zeros((2, 2, 3), dtype=np.float32)
    out = retexture.combine(colours, weights, base, vis=vis)
    # 0.4 of white against 0.8 of black.
    assert np.allclose(out, 0.4 / 1.2, atol=1e-6)

    floored = np.stack(
        [np.full((2, 2), retexture.MIN_FACING / 2, np.float32),
         np.full((2, 2), 0.8, np.float32)]
    )
    out = retexture.combine(colours, floored, base, vis=vis)
    assert np.allclose(out, 0.0)


# -- the feathered frontier ----------------------------------------------------


def test_feather_is_nothing_where_nothing_was_seen_and_whole_where_weight_is_solid():
    total = np.zeros((6, 6), dtype=np.float32)
    total[:, :3] = 1.0
    blend = retexture.feather_blend(total)
    assert blend[0, 0] == pytest.approx(1.0)
    assert blend[0, -1] == pytest.approx(0.0)


def test_feather_softens_the_frontier_rather_than_stepping():
    size = 16
    total = np.zeros((size, size), dtype=np.float32)
    total[:, : size // 2] = 1.0
    blend = retexture.feather_blend(total)
    row = blend[size // 2]
    assert np.all(np.diff(row) <= 1e-6), "blend must fall monotonically"
    boundary = row[size // 2 - 1 : size // 2 + 1]
    assert np.any((boundary > 0.0) & (boundary < 1.0))


def test_feather_does_not_wrap():
    total = np.zeros((8, 8), dtype=np.float32)
    total[:, 0] = 1.0
    blend = retexture.feather_blend(total)
    assert np.allclose(blend[:, -1], 0.0)


# -- assemble, with its eyes open ------------------------------------------------


def _depth_pairs(tmp_path, n=2, size=8, occluded=False):
    """R = what the camera saw, G = the texel's own depth. Equal means seen."""
    for i in range(n):
        pair = np.zeros((size, size, 3), dtype=np.float32)
        pair[..., 1] = 0.5
        pair[..., 0] = 0.75 if occluded else 0.5
        _write(tmp_path / f"depthpair_{i:02d}.png", pair)


def test_assemble_with_occlusion_consumes_the_pairs_and_says_so(tmp_path):
    n = _bakes(tmp_path)
    _depth_pairs(tmp_path, n=n)
    report = retexture.assemble(
        tmp_path, None, tmp_path / "out.png", count=n, occlusion=True
    )
    assert report is not None
    assert report["occlusion_tested"] is True
    assert "coverage_effective" in report


def test_assemble_with_occlusion_refuses_on_a_missing_pair(tmp_path):
    """All or nothing, the same rule as the bakes: a view whose depth pair
    vanished is a view whose smear went back to being invisible."""
    n = _bakes(tmp_path)
    _depth_pairs(tmp_path, n=n)
    (tmp_path / "depthpair_01.png").unlink()
    assert (
        retexture.assemble(tmp_path, None, tmp_path / "out.png", count=n, occlusion=True)
        is None
    )


def test_assemble_without_occlusion_ignores_the_pairs_and_stays_legacy(tmp_path):
    """The old path must stay byte-identical -- it is the behaviour the
    2026-08-08 measurement numbers describe, and the A/B in the probe depends
    on it being reachable."""
    n = _bakes(tmp_path)
    plain = retexture.assemble(tmp_path, None, tmp_path / "plain.png", count=n)
    _depth_pairs(tmp_path, n=n, occluded=True)
    unaware = retexture.assemble(tmp_path, None, tmp_path / "unaware.png", count=n)
    assert plain is not None and unaware is not None
    assert plain["occlusion_tested"] is False
    assert (tmp_path / "plain.png").read_bytes() == (tmp_path / "unaware.png").read_bytes()


def test_an_occluded_region_keeps_its_base_colour_through_assemble(tmp_path):
    """One view, square-on facing, and a wall in front of the right half: the
    left half is repainted and the right half keeps the old skin. This is the
    overhang-smear case from the measurement doc, in miniature."""
    size = 32
    colour = np.ones((size, size, 3), dtype=np.float32)
    weight = np.ones((size, size), dtype=np.float32)
    _write(tmp_path / "bake_00.png", colour)
    _write(tmp_path / "weight_00.png", weight, "L")
    pair = np.zeros((size, size, 3), dtype=np.float32)
    pair[..., 1] = 0.4
    pair[..., 0] = 0.4
    pair[:, size // 2 :, 0] = 0.8  # something nearer, over the right half
    _write(tmp_path / "depthpair_00.png", pair)
    base = tmp_path / "base.png"
    _write(base, np.zeros((size, size, 3), np.float32))

    report = retexture.assemble(
        tmp_path, base, tmp_path / "out.png", count=1, occlusion=True
    )
    assert report is not None
    with Image.open(tmp_path / "out.png") as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    # Far from the frontier on both sides: repainted white, and kept black --
    # past the reach of both the feather blur and the dilation.
    assert np.allclose(arr[:, : size // 2 - 8], 1.0, atol=0.02)
    assert np.allclose(arr[:, size // 2 + 8 :], 0.0, atol=0.02)
    assert report["coverage"] == pytest.approx(0.5, abs=0.1)


# -- the ControlNet hint ---------------------------------------------------------


def test_depth_hint_rescales_the_subject_to_the_full_range(tmp_path):
    """controlnet-depth-sdxl was trained on inverted relative depth: nearest
    point white, furthest point black, background black. A hint that only used
    the middle of the range would whisper where it should speak."""
    depth = np.zeros((8, 8), dtype=np.float32)
    depth[:, :4] = 0.3
    depth[:, 4:] = 0.5
    _write(tmp_path / "depth.png", depth, "L")
    view = np.zeros((8, 8, 4), dtype=np.float32)
    view[..., 3] = 1.0  # everything is subject
    Image.fromarray((view * 255).astype(np.uint8), "RGBA").save(tmp_path / "view.png")

    assert retexture.depth_hint(
        tmp_path / "depth.png", tmp_path / "view.png", tmp_path / "hint.png"
    ) is True
    with Image.open(tmp_path / "hint.png") as im:
        assert im.mode == "RGB"
        arr = np.asarray(im)
    assert arr[0, 0].tolist() == [0, 0, 0]
    assert arr[0, -1].tolist() == [255, 255, 255]


def test_depth_hint_paints_the_background_black(tmp_path):
    depth = np.full((8, 8), 0.6, dtype=np.float32)
    _write(tmp_path / "depth.png", depth, "L")
    view = np.zeros((8, 8, 4), dtype=np.float32)
    view[:, :4, 3] = 1.0  # subject on the left, background on the right
    Image.fromarray((view * 255).astype(np.uint8), "RGBA").save(tmp_path / "view.png")

    assert retexture.depth_hint(
        tmp_path / "depth.png", tmp_path / "view.png", tmp_path / "hint.png"
    ) is True
    with Image.open(tmp_path / "hint.png") as im:
        arr = np.asarray(im.convert("RGB"))
    assert np.all(arr[:, 4:] == 0)
    assert np.all(arr[:, :4] > 0)


def test_depth_hint_with_an_unreadable_input_is_a_false(tmp_path):
    assert retexture.depth_hint(
        tmp_path / "missing.png", tmp_path / "also_missing.png", tmp_path / "h.png"
    ) is False


# -- the specs carry the depth switch ---------------------------------------------


def test_the_specs_default_to_no_depth_and_carry_the_ask():
    """Off unless asked, so every other caller of the two ops keeps meaning
    exactly what it meant before the depth pass existed."""
    from warlock import rigging

    path = Path("m.glb")
    views = list(retexture.VIEWS)
    v = rigging.views_spec(path, Path("d"), views, size=512)
    p = rigging.project_spec(path, Path("d"), Path("o"), views, size=512, texture_size=1024)
    assert v["depth"] is False and p["depth"] is False
    v = rigging.views_spec(path, Path("d"), views, size=512, depth=True)
    p = rigging.project_spec(
        path, Path("d"), Path("o"), views, size=512, texture_size=1024, depth=True
    )
    assert v["depth"] is True and p["depth"] is True


def test_a_torn_write_leaves_nothing_behind(tmp_path):
    """The destination is model.glb's replacement. A half-written one that
    survived the failure would be renamed over the served mesh by the caller's
    next line."""
    glb = tmp_path / "model.glb"
    _textured_glb(glb)
    atlas = tmp_path / "atlas.png"
    atlas.write_bytes(_png((200, 100, 50), size=16))
    dest = tmp_path / "new.glb"

    real = Path.write_bytes

    def tamper(self, data):
        if self == dest:
            real(self, data[: len(data) // 2])
            raise OSError("the disk filled up")
        return real(self, data)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(Path, "write_bytes", tamper)
        with pytest.raises(OSError, match="disk filled up"):
            retexture.swap_base_colour(glb, atlas, dest)
    assert not dest.exists()
