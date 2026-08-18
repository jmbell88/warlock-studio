"""scale_glb must resize the mesh without re-encoding the file's binary payload."""

from __future__ import annotations

import os
import struct
import threading
import zipfile
from pathlib import Path

import numpy as np
import pytest
import trimesh

from warlock.pipelines.postprocess import (
    _split_glb,
    glb_to_obj_zip,
    glb_to_stl,
    scale_glb,
)


@pytest.fixture
def cube_glb(tmp_path):
    """A 1 m cube exported as GLB."""
    path = tmp_path / "model.glb"
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(path)
    return path


def _extent(path):
    return float(trimesh.load(path).extents.max())


def test_scales_longest_axis_to_the_target(cube_glb):
    factor = scale_glb(cube_glb, 2.5)
    assert factor == pytest.approx(2.5)
    assert _extent(cube_glb) == pytest.approx(2.5, rel=1e-4)


def test_scales_down_too(tmp_path):
    path = tmp_path / "model.glb"
    trimesh.creation.box(extents=(4.0, 1.0, 2.0)).export(path)
    scale_glb(path, 1.0)
    mesh = trimesh.load(path)
    assert mesh.extents.max() == pytest.approx(1.0, rel=1e-4)
    # Proportions are preserved -- it is a uniform scale, not a fit-to-box.
    assert sorted(mesh.extents) == pytest.approx([0.25, 0.5, 1.0], rel=1e-4)


def test_binary_payload_is_untouched(cube_glb):
    _, _, before = _split_glb(cube_glb.read_bytes())
    scale_glb(cube_glb, 3.0)
    _, _, after = _split_glb(cube_glb.read_bytes())
    assert before == after


def test_rewritten_container_stays_well_formed(cube_glb):
    scale_glb(cube_glb, 3.0)
    data = cube_glb.read_bytes()
    _magic, _version, length = struct.unpack_from("<III", data, 0)
    assert length == len(data)
    json_len = struct.unpack_from("<I", data, 12)[0]
    assert json_len % 4 == 0  # the spec requires 4-byte-aligned chunks


def test_scaling_is_idempotent_when_already_at_target(cube_glb):
    assert scale_glb(cube_glb, 1.0) == 1.0
    assert _extent(cube_glb) == pytest.approx(1.0, rel=1e-4)


def test_repeated_scaling_composes(cube_glb):
    scale_glb(cube_glb, 2.0)
    scale_glb(cube_glb, 0.5)
    assert _extent(cube_glb) == pytest.approx(0.5, rel=1e-4)


def test_degenerate_target_is_a_no_op(cube_glb):
    assert scale_glb(cube_glb, 0.0) == 1.0
    assert _extent(cube_glb) == pytest.approx(1.0, rel=1e-4)


def test_flat_mesh_with_zero_extent_axis_still_scales(tmp_path):
    """A plane has a zero-thickness axis; only the *longest* axis matters."""
    path = tmp_path / "model.glb"
    plane = trimesh.Trimesh(
        vertices=np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0]], dtype=float),
        faces=np.array([[0, 1, 2]]),
    )
    plane.export(path)
    scale_glb(path, 1.0)
    assert trimesh.load(path).extents.max() == pytest.approx(1.0, rel=1e-4)


def test_derived_exports_see_the_scale(cube_glb, tmp_path):
    """The STL/OBJ paths load the GLB with trimesh, so they must agree with it."""
    scale_glb(cube_glb, 3.0)
    stl = glb_to_stl(cube_glb, tmp_path / "model.stl")
    assert trimesh.load(stl).extents.max() == pytest.approx(3.0, rel=1e-4)


def test_rejects_a_file_that_is_not_a_glb(tmp_path):
    path = tmp_path / "model.glb"
    path.write_bytes(b"not a glb at all, really")
    with pytest.raises(ValueError):
        scale_glb(path, 2.0)


def test_write_is_atomic_rename_not_in_place_truncation(cube_glb, monkeypatch):
    """A concurrent reader must never observe a 0-byte/partial file mid-scale.

    write_bytes() opens "wb", which truncates immediately; scale_glb must
    instead write to a temp file and os.replace() it into place.
    """
    seen_sizes = []
    real_replace = os.replace

    def spying_replace(src, dst):
        # If the rewrite ever truncated glb_path in place before this call,
        # dst would already be 0 bytes by now.
        seen_sizes.append(Path(dst).stat().st_size if Path(dst).exists() else None)
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spying_replace)
    original_size = cube_glb.stat().st_size

    scale_glb(cube_glb, 3.0)

    assert seen_sizes == [original_size]
    assert not cube_glb.with_suffix(".glb.tmp").exists()


# --- on-demand exports ------------------------------------------------------
#
# STL and OBJ are produced by the file-serving route the first time they are
# requested, so two requests for the same artifact (one double-clicked download
# link) can run concurrently over the same destination path.


def test_obj_zip_contains_the_model(cube_glb, tmp_path):
    zip_path = glb_to_obj_zip(cube_glb, tmp_path / "model_obj.zip")
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert "model.obj" in names


def test_obj_zip_leaves_no_working_directory_behind(cube_glb, tmp_path):
    """The export workdir used to be a fixed 'obj_export' next to the zip, kept
    forever -- a second copy of the model and every texture in each job dir."""
    out = tmp_path / "out"
    out.mkdir()
    glb_to_obj_zip(cube_glb, out / "model_obj.zip")
    assert [p.name for p in out.iterdir()] == ["model_obj.zip"]


def test_obj_zip_includes_nested_texture_files(cube_glb, tmp_path, monkeypatch):
    """trimesh may write textures into a subdirectory; the old non-recursive
    iterdir() would hand zf.write a directory and blow up."""
    import warlock.pipelines.postprocess as pp

    real_load = pp.trimesh.load

    def load_with_nested_export(path, *args, **kwargs):
        scene = real_load(path, *args, **kwargs)
        real_export = scene.export

        def export(target, *a, **k):
            result = real_export(target, *a, **k)
            nested = Path(target).parent / "textures"
            nested.mkdir(exist_ok=True)
            (nested / "albedo.png").write_bytes(b"fake-texture")
            return result

        scene.export = export
        return scene

    monkeypatch.setattr(pp.trimesh, "load", load_with_nested_export)
    zip_path = glb_to_obj_zip(cube_glb, tmp_path / "model_obj.zip")
    with zipfile.ZipFile(zip_path) as zf:
        assert "textures/albedo.png" in zf.namelist()


@pytest.mark.parametrize(
    "export, name",
    [(glb_to_stl, "model.stl"), (glb_to_obj_zip, "model_obj.zip")],
)
def test_exports_land_atomically(cube_glb, tmp_path, monkeypatch, export, name):
    """The destination must not exist at all until the file is complete."""
    dest = tmp_path / name
    real_replace = os.replace
    observed = []

    def spying_replace(src, dst):
        if Path(dst) == dest:
            observed.append(dest.exists())
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spying_replace)
    export(cube_glb, dest)

    assert observed == [False], "the export wrote to the destination before finishing"
    assert dest.stat().st_size > 0
    # No temp files left over.
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".")] == []


@pytest.mark.parametrize(
    "export, name",
    [(glb_to_stl, "model.stl"), (glb_to_obj_zip, "model_obj.zip")],
)
def test_concurrent_exports_produce_a_valid_file(cube_glb, tmp_path, export, name):
    dest = tmp_path / name
    errors: list[BaseException] = []
    start = threading.Barrier(4)

    def run():
        start.wait()
        try:
            export(cube_glb, dest)
        except BaseException as exc:  # noqa: BLE001 - any failure is the finding
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert errors == []
    # Whichever writer won, the result is a complete file, not a torn one.
    if name.endswith(".zip"):
        with zipfile.ZipFile(dest) as zf:
            assert zf.testzip() is None
            assert "model.obj" in zf.namelist()
    else:
        assert trimesh.load(dest).extents.max() == pytest.approx(1.0, rel=1e-4)


def test_normalize_grounds_and_centres(tmp_path):
    import trimesh

    from warlock.pipelines import postprocess

    box = trimesh.creation.box(extents=(1.0, 2.0, 1.0))
    box.apply_translation((5.0, 7.0, -3.0))
    path = tmp_path / "m.glb"
    trimesh.Scene(box).export(path)

    result = postprocess.normalize_glb(path, 4.0)

    loaded = trimesh.load(path)
    mesh = loaded.to_mesh()
    lo, hi = mesh.bounds
    assert abs(float(max(mesh.extents)) - 4.0) < 1e-4     # scaled
    assert abs(float(lo[1])) < 1e-6                        # grounded: min Y == 0
    assert abs(float(lo[0] + hi[0])) < 1e-6                # centred in X
    assert abs(float(lo[2] + hi[2])) < 1e-6                # centred in Z
    assert abs(result["achieved_size_m"] - 4.0) < 1e-4


def test_normalize_without_a_target_still_grounds(tmp_path):
    import trimesh

    from warlock.pipelines import postprocess

    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    box.apply_translation((0.0, 9.0, 0.0))
    path = tmp_path / "m.glb"
    trimesh.Scene(box).export(path)

    postprocess.normalize_glb(path, None)

    assert abs(float(trimesh.load(path).to_mesh().bounds[0][1])) < 1e-6


# --- grounding against every root shape a GLB can arrive in ------------------
#
# trimesh's own exporter writes one identity root, which is the only shape the
# tests above ever saw -- and the only shape the old grounding was right for.
# Blender-authored uploads and Clay exports both put real TRS on root nodes,
# and Clay puts *one root per object*: `import_mesh` grounded those wrong and
# `meshreport` then filed a pivot complaint about a mesh the user had just
# authored. These build the roots by hand for that reason.


def _rooted(path, roots, extents=(1.0, 2.0, 3.0)):
    """A GLB whose scene roots are the mesh nodes themselves, one per entry in
    ``roots``, each updated with that entry's own transform keys."""
    import trimesh

    from warlock.pipelines import postprocess

    scene = trimesh.Scene()
    for i in range(len(roots)):
        scene.add_geometry(trimesh.creation.box(extents=extents), node_name=f"box{i}")
    path.write_bytes(scene.export(file_type="glb"))

    header, gltf, rest = postprocess._split_glb(path.read_bytes())
    gltf_scene = gltf["scenes"][gltf.get("scene", 0)]
    mesh_nodes = [i for i, node in enumerate(gltf["nodes"]) if "mesh" in node]
    gltf_scene["nodes"] = mesh_nodes
    for node in gltf["nodes"]:
        node.pop("children", None)
    for index, extra in zip(mesh_nodes, roots, strict=True):
        gltf["nodes"][index].update(extra)
    path.write_bytes(postprocess._rebuild_glb(header, gltf, rest))
    return path


def _grounded(path):
    import trimesh

    lo, hi = trimesh.load(path).bounds
    return (
        abs(float(lo[1])) < 1e-4
        and abs(float(lo[0] + hi[0])) < 1e-4
        and abs(float(lo[2] + hi[2])) < 1e-4
    )


# A quarter turn about X, which maps Y onto Z -- the case a Y-axis rotation
# hides, because that one commutes with a Y-only grounding offset.
_QUARTER_X = [0.7071067811865476, 0.0, 0.0, 0.7071067811865476]


@pytest.mark.parametrize(
    "label, roots",
    [
        ("identity", [{}]),
        ("rotated", [{"rotation": _QUARTER_X}]),
        ("rotated and translated", [{"rotation": _QUARTER_X, "translation": [1.0, 5.0, -2.0]}]),
        ("non-uniformly scaled", [{"scale": [2.0, 0.5, 1.0], "translation": [1.0, 5.0, -2.0]}]),
        # glTF allows a node to carry a matrix instead of TRS, and a general
        # one need not decompose into TRS at all.
        ("matrix", [{"matrix": [1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0, 0, 2, 5, -1, 1]}]),
        ("multi-root", [{}, {"translation": [4.0, 0.0, 0.0]}, {"translation": [-4.0, 0.0, 0.0]}]),
        (
            "multi-root, each transformed",
            [{"rotation": _QUARTER_X}, {"translation": [4.0, 1.0, 0.0], "scale": [1.5] * 3}],
        ),
    ],
)
def test_grounding_holds_whatever_the_roots_carry(tmp_path, label, roots):
    """The regression: bounds are measured in world space but the transform was
    applied *under* each root, composing as ``M_root . T . S`` instead of
    ``T . S . M_root`` -- so a rotated root rotated the grounding offset."""
    from warlock.pipelines import postprocess

    path = _rooted(tmp_path / f"{label.replace(' ', '_').replace(',', '')}.glb", roots)

    postprocess.normalize_glb(path, None)

    assert _grounded(path), f"{label}: not grounded"


@pytest.mark.parametrize(
    "label, roots",
    [
        ("one root", [{"rotation": _QUARTER_X, "translation": [1.0, 5.0, -2.0]}]),
        ("three roots", [{}, {"translation": [4.0, 0.0, 0.0]}, {"translation": [-4.0, 0.0, 0.0]}]),
    ],
)
def test_a_size_target_is_met_however_many_roots_there_are(tmp_path, label, roots):
    """The second half of the same bug, and the louder one: the transform was
    inserted once *per root* carrying the same world-space translation, while
    each root's own offset stayed unscaled above it. A three-object Clay export
    asked for 2 m came out 8.2 m across -- and reported 2."""
    import trimesh

    from warlock.pipelines import postprocess

    path = _rooted(tmp_path / f"{label.replace(' ', '_')}.glb", roots)

    result = postprocess.normalize_glb(path, 2.0)

    assert float(max(trimesh.load(path).extents)) == pytest.approx(2.0, abs=1e-3)
    assert result["achieved_size_m"] == pytest.approx(2.0, abs=1e-3)
    assert _grounded(path)


def test_the_scene_roots_are_left_free_of_transforms(tmp_path):
    """Why the composition happens here rather than in a node wrapped above the
    roots: trimesh treats a scene root as the graph's base frame and discards
    its transform, so a root that kept one would leave the GLB grounded and
    every derived STL and OBJ where it started. Emptying the roots is what lets
    one rule serve every root at once."""
    from warlock.pipelines import postprocess

    path = _rooted(tmp_path / "m.glb", [{"rotation": _QUARTER_X, "translation": [1.0, 5.0, 0.0]}])

    postprocess.normalize_glb(path, None)

    _header, gltf, _rest = postprocess._split_glb(path.read_bytes())
    for root in gltf["scenes"][gltf.get("scene", 0)]["nodes"]:
        node = gltf["nodes"][root]
        assert not {"translation", "rotation", "scale", "matrix"} & set(node)
        assert node["children"], "the root must still parent its content"


def test_normalize_stages_through_a_dotfile(tmp_path, monkeypatch):
    """The staged-writes rule: the staging file is a dotfile beside the served
    name. This one spent a while as a visible ``m.glb.tmp`` sibling, outside
    both the dotfile convention and any finally."""
    from warlock.pipelines import postprocess

    path = tmp_path / "m.glb"
    trimesh.Scene(trimesh.creation.box(extents=(1.0, 1.0, 1.0))).export(path)

    staged: list[str] = []
    real_replace = os.replace

    def spy(src, dst):
        staged.append(Path(src).name)
        return real_replace(src, dst)

    monkeypatch.setattr(postprocess.os, "replace", spy)
    postprocess.normalize_glb(path, None)

    assert staged and staged[0].startswith(".m.glb.")
    assert [p.name for p in tmp_path.iterdir()] == ["m.glb"]


def test_a_failed_normalize_sweeps_its_staging_file(tmp_path, monkeypatch):
    """A rebuild that raises must leave the served file untouched and no
    staging litter -- a stranded dotfile sits in the job directory for its
    whole life, because nothing ever sweeps one."""
    from warlock.pipelines import postprocess

    path = tmp_path / "m.glb"
    trimesh.Scene(trimesh.creation.box(extents=(1.0, 1.0, 1.0))).export(path)
    before = path.read_bytes()

    def boom(*_args):
        raise RuntimeError("encode failed")

    monkeypatch.setattr(postprocess, "_rebuild_glb", boom)
    with pytest.raises(RuntimeError, match="encode failed"):
        postprocess.normalize_glb(path, None)

    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["m.glb"]


def test_collision_hull_is_convex_and_small(tmp_path):
    import trimesh

    from warlock.pipelines import postprocess

    # A sphere is the worst case for face count and the easiest convexity check.
    src = tmp_path / "m.glb"
    trimesh.Scene(trimesh.creation.icosphere(subdivisions=4)).export(src)

    source_faces = len(trimesh.load(src).to_mesh().faces)
    out = postprocess.glb_to_collision(src, tmp_path / "collision.glb")

    hull = trimesh.load(out).to_mesh()
    # Convexity and watertightness are the contract -- a collider that is not
    # convex is silently wrong rather than loudly broken. The face count is
    # best-effort: trimesh's decimation backend is optional and not a
    # dependency here, so an unsimplified hull is an accepted outcome.
    assert hull.is_convex
    assert hull.is_watertight
    assert len(hull.faces) <= source_faces


def test_textures_zip_contains_the_pbr_maps(tmp_path):
    import zipfile

    import numpy as np
    import trimesh
    from PIL import Image

    from warlock.pipelines import postprocess

    mesh = trimesh.creation.box(extents=(1, 1, 1))
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)),
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new("RGB", (8, 8), (255, 0, 0))
        ),
    )
    src = tmp_path / "m.glb"
    trimesh.Scene(mesh).export(src)

    out = postprocess.glb_to_textures_zip(src, tmp_path / "textures.zip")

    with zipfile.ZipFile(out) as zf:
        assert "base_color.png" in zf.namelist()


def test_textures_zip_on_an_untextured_mesh_raises(tmp_path):
    import pytest
    import trimesh

    from warlock.pipelines import postprocess

    src = tmp_path / "m.glb"
    trimesh.Scene(trimesh.creation.box()).export(src)
    with pytest.raises(ValueError):
        postprocess.glb_to_textures_zip(src, tmp_path / "textures.zip")
