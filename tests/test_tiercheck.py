"""What a gltfpack tier has to preserve, measured off the GLB itself.

Tier qualification. A named tier has been a live code path since the binary
landed on
2026-08-07, and the *only* thing keeping it out of the generate forms is that
nobody has shown it keeps UVs, both PBR maps and material assignment. That is
not a matter of opinion, so it is a function: two GLBs in, a verdict out.

Why the JSON chunk and not trimesh. trimesh merges a scene into one mesh with
one material, which destroys the exact thing under test -- whether a
*multi-material* mesh still assigns each primitive its own material after a
simplify pass. ``glbio.read_glb`` is stdlib and answers the question directly,
which also keeps ``tiercheck`` pure in the ``vram.py``/``meshreport`` sense:
importable with no GPU, no Blender and no gltfpack.

The rules are asymmetric on purpose: a tier may legitimately *improve* nothing
and must never lose anything. So every check is "not worse than the source",
never "identical to it".
"""

from __future__ import annotations

import json
import struct

import pytest

from warlock import tiercheck

# --- fixtures ----------------------------------------------------------------


def _glb(gltf: dict) -> bytes:
    """A GLB carrying just this JSON chunk. No BIN chunk: every check here reads
    the JSON, and a buffer would be bytes nothing looks at."""
    payload = json.dumps(gltf).encode()
    payload += b" " * (-len(payload) % 4)
    header = struct.pack("<III", tiercheck.glbio.GLB_MAGIC, 2, 12 + 8 + len(payload))
    return header + struct.pack("<II", len(payload), tiercheck.glbio.CHUNK_JSON) + payload


def _textured(*, uv: bool = True, base: bool = True, mr: bool = True, normal: bool = True,
              materials: int = 2, assigned: bool = True) -> bytes:
    """A two-material mesh with textures, the shape trellis output has."""
    mats = []
    for i in range(materials):
        pbr: dict = {}
        if base:
            pbr["baseColorTexture"] = {"index": i}
        if mr:
            pbr["metallicRoughnessTexture"] = {"index": i}
        mat: dict = {"pbrMetallicRoughness": pbr}
        if normal:
            mat["normalTexture"] = {"index": i}
        mats.append(mat)
    prims = []
    for i in range(materials):
        attrs = {"POSITION": 0}
        if uv:
            attrs["TEXCOORD_0"] = 1
        prim: dict = {"attributes": attrs}
        if assigned:
            prim["material"] = i
        prims.append(prim)
    return _glb(
        {
            "asset": {"version": "2.0"},
            "materials": mats,
            "meshes": [{"primitives": prims}],
            "images": [{"mimeType": "image/png"} for _ in range(materials)],
        }
    )


def _write(tmp_path, name: str, data: bytes):
    path = tmp_path / name
    path.write_bytes(data)
    return path


# --- surveying ---------------------------------------------------------------


def test_a_survey_counts_what_a_tier_could_lose(tmp_path):
    survey = tiercheck.survey(_write(tmp_path, "a.glb", _textured()))

    assert survey.primitives == 2
    assert survey.uv_primitives == 2
    assert survey.materials == 2
    assert survey.unassigned == 0
    assert survey.base_color == 2
    assert survey.metallic_roughness == 2
    assert survey.normal_map == 2
    assert survey.extensions_required == ()


def test_a_survey_counts_only_the_materials_a_primitive_actually_uses(tmp_path):
    """An unreferenced material is not preservation -- gltfpack drops those on
    purpose, and counting them would make a real loss look like a wash."""
    data = _glb(
        {
            "asset": {"version": "2.0"},
            "materials": [{"pbrMetallicRoughness": {}}, {"pbrMetallicRoughness": {}}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "material": 0}]}],
        }
    )
    survey = tiercheck.survey(_write(tmp_path, "a.glb", data))

    assert survey.materials == 1
    assert survey.declared_materials == 2


def test_a_survey_notices_a_primitive_with_no_material_at_all(tmp_path):
    survey = tiercheck.survey(_write(tmp_path, "a.glb", _textured(assigned=False)))

    assert survey.unassigned == 2
    assert survey.materials == 0


def test_a_survey_reports_what_the_file_demands_of_a_reader(tmp_path):
    """``-noq`` exists so gltfpack does not write KHR_mesh_quantization, which
    the viewer's loader *refuses* -- it decodes as silently wrong geometry
    otherwise. A tier that reintroduced it would produce a file the app cannot
    open, which no UV or texture count would catch."""
    data = _glb(
        {
            "asset": {"version": "2.0"},
            "extensionsRequired": ["KHR_mesh_quantization"],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        }
    )
    survey = tiercheck.survey(_write(tmp_path, "a.glb", data))

    assert survey.extensions_required == ("KHR_mesh_quantization",)


# --- comparing ---------------------------------------------------------------


def test_a_tier_that_preserves_everything_passes(tmp_path):
    before = tiercheck.survey(_write(tmp_path, "a.glb", _textured()))
    after = tiercheck.survey(_write(tmp_path, "b.glb", _textured()))

    verdict = tiercheck.compare(before, after)
    assert verdict.ok is True
    assert verdict.failures == ()


def test_losing_the_uvs_fails(tmp_path):
    before = tiercheck.survey(_write(tmp_path, "a.glb", _textured()))
    after = tiercheck.survey(_write(tmp_path, "b.glb", _textured(uv=False)))

    verdict = tiercheck.compare(before, after)
    assert verdict.ok is False
    assert any("UV" in f for f in verdict.failures)


@pytest.mark.parametrize(
    "kwargs,word",
    [
        ({"base": False}, "base colour"),
        ({"mr": False}, "metallic"),
    ],
)
def test_losing_either_pbr_map_fails(tmp_path, kwargs, word):
    """Both, named separately: "a texture went missing" does not say which pass
    of a bake to look at."""
    before = tiercheck.survey(_write(tmp_path, "a.glb", _textured()))
    after = tiercheck.survey(_write(tmp_path, "b.glb", _textured(**kwargs)))

    verdict = tiercheck.compare(before, after)
    assert verdict.ok is False
    assert any(word in f for f in verdict.failures)


def test_merging_two_materials_into_one_fails(tmp_path):
    """``-km`` is passed for exactly this: without it the material assignment,
    and therefore both PBR textures, can be dropped on merge."""
    before = tiercheck.survey(_write(tmp_path, "a.glb", _textured(materials=2)))
    after = tiercheck.survey(_write(tmp_path, "b.glb", _textured(materials=1)))

    verdict = tiercheck.compare(before, after)
    assert verdict.ok is False
    assert any("material" in f for f in verdict.failures)


def test_dropping_a_primitives_material_assignment_fails(tmp_path):
    before = tiercheck.survey(_write(tmp_path, "a.glb", _textured()))
    after = tiercheck.survey(_write(tmp_path, "b.glb", _textured(assigned=False)))

    verdict = tiercheck.compare(before, after)
    assert verdict.ok is False
    assert any("unassigned" in f for f in verdict.failures)


def test_requiring_an_extension_the_source_did_not_fails(tmp_path):
    before = tiercheck.survey(_write(tmp_path, "a.glb", _textured()))
    after_data = _glb(
        {
            "asset": {"version": "2.0"},
            "extensionsRequired": ["KHR_mesh_quantization"],
            "materials": [{"pbrMetallicRoughness": {}}],
            "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        }
    )
    after = tiercheck.survey(_write(tmp_path, "b.glb", after_data))

    verdict = tiercheck.compare(before, after)
    assert verdict.ok is False
    assert any("KHR_mesh_quantization" in f for f in verdict.failures)


def test_a_source_with_no_textures_cannot_be_failed_for_losing_them(tmp_path):
    """"Not worse than the source" and never "identical to an ideal": a rock
    with one untextured material is a legitimate input, and a tier must not be
    marked failing for something that was never there."""
    plain = _textured(base=False, mr=False, normal=False, uv=False, materials=1)
    before = tiercheck.survey(_write(tmp_path, "a.glb", plain))
    after = tiercheck.survey(_write(tmp_path, "b.glb", plain))

    verdict = tiercheck.compare(before, after)
    assert verdict.ok is True


def test_a_source_that_was_already_missing_something_says_so(tmp_path):
    """A note, not a failure -- and it must be visible, or a whole qualification
    could pass by testing preservation of nothing."""
    plain = _textured(base=False, mr=False, normal=False, uv=False, materials=1)
    before = tiercheck.survey(_write(tmp_path, "a.glb", plain))
    after = tiercheck.survey(_write(tmp_path, "b.glb", plain))

    verdict = tiercheck.compare(before, after)
    assert verdict.notes
    assert any("no UVs" in note for note in verdict.notes)


# --- the qualification run ---------------------------------------------------


def _fake_optimizer(**degrade):
    """Stands in for gltfpack: writes an output degraded however the test says."""

    def run(source, dest, *, target_triangles):
        dest.write_bytes(_textured(**degrade))
        return {"requested": target_triangles, "achieved": target_triangles}

    return run


def test_a_qualification_reports_one_row_per_subject_and_tier(tmp_path):
    chest = _write(tmp_path, "chest.glb", _textured())
    sword = _write(tmp_path, "sword.glb", _textured())

    report = tiercheck.qualify(
        [chest, sword], ("draft", "standard"), tmp_path / "out",
        optimizer=_fake_optimizer(),
    )

    assert [(r.subject, r.tier) for r in report.rows] == [
        ("chest", "draft"), ("chest", "standard"),
        ("sword", "draft"), ("sword", "standard"),
    ]
    assert report.ok is True
    assert report.qualified == ("draft", "standard")


def test_a_tier_qualifies_only_when_it_passed_on_every_subject(tmp_path):
    """One chest is not a qualification. The bar names three subjects because a
    simplifier's failure modes are shape-dependent -- a sword is thin, a rock is
    a blob, and a chest has flat panels and a lid."""
    good = _write(tmp_path, "chest.glb", _textured())
    report = tiercheck.qualify(
        [good], ("draft",), tmp_path / "out", optimizer=_fake_optimizer(uv=False)
    )

    assert report.ok is False
    assert report.qualified == ()
    assert report.rows[0].verdict.ok is False


def test_one_failing_tier_does_not_end_the_run(tmp_path):
    """Nine combinations, unattended: the point of a harness is the whole table."""
    chest = _write(tmp_path, "chest.glb", _textured())

    def optimizer(source, dest, *, target_triangles):
        if target_triangles == tiercheck.optimize_profiles()["draft"]:
            raise RuntimeError("gltfpack exited 1")
        dest.write_bytes(_textured())
        return {"requested": target_triangles, "achieved": target_triangles}

    report = tiercheck.qualify(
        [chest], ("draft", "standard"), tmp_path / "out", optimizer=optimizer
    )

    assert len(report.rows) == 2
    assert report.rows[0].error == "gltfpack exited 1"
    assert report.rows[1].verdict.ok is True
    assert report.qualified == ("standard",)


def test_the_raw_tier_is_not_a_tier_to_qualify(tmp_path):
    """It is the absence of one -- ``PROFILES["raw"]`` is None and the run would
    be comparing a file with itself."""
    chest = _write(tmp_path, "chest.glb", _textured())

    with pytest.raises(ValueError):
        tiercheck.qualify([chest], ("raw",), tmp_path / "out", optimizer=_fake_optimizer())


def test_a_qualification_never_writes_inside_the_corpus_it_reads(tmp_path):
    """Non-negotiable: ``optimize_job`` rewrites ``model.glb`` in place and
    deletes the derived artifacts and rig sheets that describe the old mesh. The
    corpus here is the handful of *accepted* meshes -- the only ones on which
    preservation can be judged at all -- so a harness that consumed them would
    destroy its own inputs and could be run exactly once."""
    corpus = tmp_path / "assets" / "abc123def456"
    corpus.mkdir(parents=True)
    source = corpus / "source.glb"
    source.write_bytes(_textured())
    before = sorted(p.name for p in corpus.iterdir())

    tiercheck.qualify(
        [source], ("draft",), tmp_path / "out", optimizer=_fake_optimizer()
    )

    assert sorted(p.name for p in corpus.iterdir()) == before
    assert source.read_bytes() == _textured()


def test_a_wholly_unreadable_file_is_a_failure_not_an_exception(tmp_path):
    """The harness runs nine of these unattended; one bad output must report
    itself rather than ending the run."""
    path = _write(tmp_path, "b.glb", b"not a glb at all")

    survey = tiercheck.survey(path)
    assert survey is None
