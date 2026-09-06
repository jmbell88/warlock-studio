"""The humanoid archetype: the bake, the skeleton it fits, and the clips.

The load-bearing claim of the whole increment is that **a species is a shape you
can pose**, and it is made here three times over: every fitted joint of every
shipped species lies inside solid geometry, it stays there at both ends of every
appearance slider, and the shipped clips move it somewhere worth rendering.

Regeneration is compared to the checked-in asset **to a tolerance, not byte for
byte**. ``manifold3d`` does not promise a bit-identical arrangement across
versions and Catmull-Clark is a float sum over an adjacency; what has to hold is
that the mesh is the same mesh -- the same counts, the same vertices to 1e-5 of
unit height, the same channels. A byte comparison would fail on a dependency
bump that changed nothing anybody could see, and the correct response to that
failure would be to re-bake, which is exactly what the loose comparison lets
happen without a red test in between.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from warlock import clips as clipslib
from warlock import rigging
from warlock.characters import DEFAULT_RECIPE, CharacterError, Recipe, families
from warlock.characters import family as familylib
from warlock.characters.humanoid import generate
from warlock.characters.instantiate import instantiate
from warlock.pipelines import charsheet
from warlock.studio.viewer import gltf

SILHOUETTES = sorted(familylib.silhouettes("humanoid"))
SPECIES = sorted(familylib.families_of("humanoid"))

#: 1e-5 of unit height is thirty microns on a three-metre troll -- far below
#: anything a 64px sprite can show, and far above float noise.
TOL = 1e-5


@pytest.fixture(scope="module")
def rebuilt() -> dict[str, generate.Baked]:
    """Every silhouette, generated from scratch once for the whole module.

    Module-scoped because a union plus a subdivision is about half a second and
    eight tests want the same four meshes; ``--dist loadfile`` keeps the file on
    one worker, so the cache is a cache and not four of them.
    """
    return {key: generate.build(key) for key in SILHOUETTES}


@pytest.fixture(scope="module")
def shipped() -> dict[str, tuple[gltf.Model, dict[str, np.ndarray]]]:
    out = {}
    for key in SILHOUETTES:
        fam = next(f for f in families().values() if f.silhouette == key)
        with np.load(fam.masks_npz, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        out[key] = (gltf.load(fam.base_glb), arrays)
    return out


def _stacked(model: gltf.Model) -> np.ndarray:
    return np.concatenate([p.positions for mesh in model.meshes for p in mesh]).astype("f8")


def _mesh_of(baked: generate.Baked):
    from warlock.studio.clay import mesh as bm

    return bm.Mesh(
        positions=baked.positions.astype("f4"),
        loops=baked.loops,
        starts=baked.starts,
        material=baked.regions.astype("i4"),
        smooth=np.ones(baked.face_count, dtype=bool),
        uv=None,
    )


def _triangles(baked: generate.Baked) -> np.ndarray:
    from warlock.studio.clay import mesh as bm

    tris, _face = bm.triangulate(_mesh_of(baked))
    return np.asarray(tris, dtype="i8")


def _inside(positions: np.ndarray, tris: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Which of *points* are inside the closed surface. Moller-Trumbore, +X ray.

    A real containment test rather than a bounding-box one: "the joint is
    somewhere in the box" is true of a joint hanging in the air beside the hip,
    and a bone hanging in the air is precisely what sends Blender's heat solve
    into the envelope fallback the inspector calls a degraded outcome.
    """
    a = positions[tris[:, 0]]
    b = positions[tris[:, 1]]
    c = positions[tris[:, 2]]
    e1, e2 = b - a, c - a
    ray = np.array([1.0, 0.0, 0.0])
    h = np.cross(ray, e2)
    det = np.einsum("ij,ij->i", e1, h)
    usable = np.abs(det) > 1e-14
    safe = np.where(usable, det, 1.0)
    out = []
    for point in points:
        s = point - a
        u = np.einsum("ij,ij->i", s, h) / safe
        q = np.cross(s, e1)
        v = (q @ ray) / safe
        t = np.einsum("ij,ij->i", e2, q) / safe
        hit = usable & (u >= 0) & (u <= 1) & (v >= 0) & (u + v <= 1) & (t > 1e-9)
        out.append(bool(hit.sum() % 2))
    return np.array(out)


def _joint_points(baked: generate.Baked) -> np.ndarray:
    """``(J, 2, 3)`` head/tail in glTF axes."""
    return np.stack(
        [
            generate._to_gltf(np.array([b["head"] for b in baked.joints], dtype="f8")),
            generate._to_gltf(np.array([b["tail"] for b in baked.joints], dtype="f8")),
        ],
        axis=1,
    )


def _pulled_in(positions: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Every point nudged 1% toward the centroid before a containment test.

    Two landmarks sit **on** the surface by construction and not by accident:
    the head's tail is at z = 1.0 normalized, which is the top of the bounding
    box and therefore the crown of the head sphere, and each foot's tail is at
    z = 0, which is the sole. Both are exactly on the boundary, where a
    crossing count is a coin toss. The nudge is the smallest honest way to ask
    the question the test means -- "is this joint in the body" -- of a template
    that deliberately puts two of them on its skin.
    """
    centre = positions.mean(axis=0)
    return centre + 0.99 * (points - centre)


# --- the bake ---------------------------------------------------------------


def test_every_silhouette_group_has_features_and_an_asset():
    assert set(generate.FEATURES) == set(SILHOUETTES)
    for key in SILHOUETTES:
        fam = next(f for f in families().values() if f.silhouette == key)
        assert fam.base_glb.is_file(), key
        assert fam.masks_npz.is_file(), key


@pytest.mark.parametrize("silhouette", SILHOUETTES)
def test_regenerating_reproduces_the_checked_in_asset(silhouette, rebuilt, shipped):
    """The generator is the record of *how* and the asset is the record of
    *what*, and this is the only thing keeping the two honest. To a tolerance
    rather than byte for byte -- see the module docstring."""
    baked = rebuilt[silhouette]
    model, arrays = shipped[silhouette]
    fresh_data, fresh_arrays = generate.bake(baked)
    del fresh_data

    ours = _stacked(model)
    theirs = np.concatenate([p for _r, p, _i in generate.primitives_of(baked)]).astype("f8")
    assert ours.shape == theirs.shape, "the checked-in mesh has a different vertex count"
    assert np.abs(ours - theirs).max() < TOL

    assert set(arrays) == set(fresh_arrays), "the mask file's keys drifted"
    assert np.array_equal(arrays["prim_offsets"], fresh_arrays["prim_offsets"])
    assert np.array_equal(arrays["prim_regions"], fresh_arrays["prim_regions"])
    for key in sorted(k for k in arrays if k.startswith(("disp/", "jdisp/"))):
        assert np.abs(arrays[key].astype("f8") - fresh_arrays[key].astype("f8")).max() < TOL, key


@pytest.mark.parametrize("silhouette", SILHOUETTES)
def test_the_mask_file_is_pinned_to_the_mesh_it_was_baked_against(silhouette, shipped):
    """One artifact stored as two files. A ``.masks.npz`` baked against a
    different ``.glb`` would displace the wrong vertices -- silently, and worst
    on the ones that move most -- so the digest is checked at every load rather
    than trusted."""
    import hashlib

    model, arrays = shipped[silhouette]
    digest = hashlib.blake2b(_stacked(model).astype("f4").tobytes(), digest_size=16).digest()
    assert bytes(arrays["positions_digest"].tobytes()) == digest


@pytest.mark.parametrize("silhouette", SILHOUETTES)
def test_the_checked_in_glb_loads_back_grounded_and_unit_tall(silhouette, shipped):
    model, _arrays = shipped[silhouette]
    positions = _stacked(model)
    lo, hi = positions.min(axis=0), positions.max(axis=0)
    assert lo[1] == pytest.approx(0.0, abs=1e-6), "not grounded"
    assert hi[1] - lo[1] == pytest.approx(1.0, abs=1e-6), "not unit height"
    assert (lo[0] + hi[0]) / 2 == pytest.approx(0.0, abs=1e-6), "not centred in x"
    assert (lo[2] + hi[2]) / 2 == pytest.approx(0.0, abs=1e-6), "not centred in z"


@pytest.mark.parametrize("silhouette", SILHOUETTES)
def test_every_baked_mesh_is_one_closed_solid(silhouette, rebuilt):
    """The union, the smoothing and the weld all have to leave it closed:
    Blender's bone-heat solve refuses non-manifold input, and the fallback is
    envelope weighting, which the inspector reports as needing review."""
    from warlock.studio.clay import adjacency

    report = adjacency.check_manifold(_mesh_of(rebuilt[silhouette]))
    assert report.clean, (
        f"{silhouette}: {len(report.boundary_edges)} boundary, "
        f"{len(report.nonmanifold_edges)} non-manifold, "
        f"{len(report.flipped_edges)} flipped edges"
    )


@pytest.mark.parametrize("silhouette", SILHOUETTES)
def test_a_baked_mesh_stays_inside_its_face_budget(silhouette, rebuilt):
    assert rebuilt[silhouette].face_count <= generate.MAX_FACES


@pytest.mark.parametrize("silhouette", SILHOUETTES)
def test_every_region_a_theme_paints_actually_exists_on_the_mesh(silhouette, rebuilt):
    """A region with no faces is a colour nobody can see, and -- worse -- a
    theme key that looks configured and does nothing."""
    baked = rebuilt[silhouette]
    names = familylib.get_archetype("humanoid").regions
    present = {names[i] for i in np.unique(baked.regions)}
    assert "skin" in present and "eye" in present
    assert ("tooth" in present) == generate.FEATURES[silhouette].tusks


# --- the skeleton -----------------------------------------------------------


@pytest.mark.parametrize("silhouette", SILHOUETTES)
def test_the_generated_mesh_fits_the_humanoid_template_exactly(silhouette, rebuilt):
    """The mesh is grown *around* the shipped template's landmarks, so the
    landmarks fitted to the finished mesh's own bounding box have to come back
    where the body was built. Anything else means the fixed-point loop in
    ``build`` closed on the wrong number and every joint is off by that much."""
    baked = rebuilt[silhouette]
    lo, hi = baked.bounds
    fitted = rigging.fit_template(
        rigging.get_template("humanoid"),
        [float(lo[0]), -float(hi[2]), 0.0],
        [float(hi[0]), -float(lo[2]), 1.0],
    )
    assert [b["name"] for b in fitted] == [b["name"] for b in baked.joints]
    for ours, theirs in zip(baked.joints, fitted, strict=True):
        assert ours["head"] == pytest.approx(theirs["head"], abs=TOL), ours["name"]
        assert ours["tail"] == pytest.approx(theirs["tail"], abs=TOL), ours["name"]


@pytest.mark.parametrize("silhouette", SILHOUETTES)
def test_every_joint_of_a_baked_silhouette_is_inside_the_body(silhouette, rebuilt):
    baked = rebuilt[silhouette]
    tris = _triangles(baked)
    points = _pulled_in(baked.positions, _joint_points(baked).reshape(-1, 3))
    inside = _inside(baked.positions, tris, points)
    names = [b["name"] for b in baked.joints]
    outside = [
        f"{names[i // 2]}.{('head', 'tail')[i % 2]}" for i, ok in enumerate(inside) if not ok
    ]
    assert not outside, f"{silhouette}: {outside} are outside the mesh"


@pytest.mark.parametrize("channel", [c.key for c in familylib.get_archetype("humanoid").channels])
@pytest.mark.parametrize("silhouette", SILHOUETTES)
def test_a_channel_at_either_bound_keeps_the_skeleton_in_the_body(silhouette, channel, rebuilt):
    """The claim that makes the sliders safe to expose. Both fields -- the one
    over the vertices and the one over the joints -- are the *same function*
    evaluated on two point sets, and this is what says so: if they could
    disagree, a slider at its bound is where they would.

    ``shoulder_width`` is the reason the test names both bounds. It was a
    constant push outward, which at -1 moved a narrow species' shoulder bones
    further than their own distance from the midline and left them crossed over
    and outside the chest.
    """
    baked = rebuilt[silhouette]
    tris = _triangles(baked)
    joints = _joint_points(baked)
    names = [b["name"] for b in baked.joints]
    for value in (-1.0, 1.0):
        positions = baked.positions + value * baked.displacements[channel]
        moved = joints + value * baked.joint_displacements[channel]
        inside = _inside(positions, tris, _pulled_in(positions, moved.reshape(-1, 3)))
        outside = [
            f"{names[i // 2]}.{('head', 'tail')[i % 2]}" for i, ok in enumerate(inside) if not ok
        ]
        assert not outside, f"{silhouette} {channel}={value:+}: {outside} left the mesh"


@pytest.mark.parametrize("silhouette", SILHOUETTES)
def test_a_displaced_mesh_is_still_the_same_closed_solid(silhouette, rebuilt):
    """A displacement field moves vertices and touches no index, so this cannot
    fail by arithmetic -- it can only fail if a channel ever stops being a
    displacement and starts being an edit, which is exactly the change that
    would need noticing."""
    from warlock.studio.clay import adjacency
    from warlock.studio.clay import mesh as bm

    baked = rebuilt[silhouette]
    for channel, field in baked.displacements.items():
        moved = bm.Mesh(
            positions=(baked.positions + field).astype("f4"),
            loops=baked.loops,
            starts=baked.starts,
            material=baked.regions.astype("i4"),
            smooth=np.ones(baked.face_count, dtype=bool),
            uv=None,
        )
        assert adjacency.check_manifold(moved).clean, f"{silhouette}/{channel}"


@pytest.mark.parametrize("species", SPECIES)
def test_every_shipped_species_is_a_body_its_own_skeleton_sits_inside(species, rebuilt):
    """Per species, not per silhouette: a species is its channel defaults, and
    six of them stack at once. The ogre carries +0.9 bulk, +0.75 shoulders and
    +0.6 hunch together, and no single-channel test would catch the pair that
    only fails in combination."""
    fam = familylib.get_family(species)
    baked = rebuilt[fam.silhouette]
    positions = baked.positions.copy()
    joints = _joint_points(baked)
    for channel, value in sorted(fam.appearance_defaults().items()):
        positions = positions + value * baked.displacements[channel]
        joints = joints + value * baked.joint_displacements[channel]
    inside = _inside(positions, _triangles(baked), _pulled_in(positions, joints.reshape(-1, 3)))
    names = [b["name"] for b in baked.joints]
    outside = [
        f"{names[i // 2]}.{('head', 'tail')[i % 2]}" for i, ok in enumerate(inside) if not ok
    ]
    assert not outside, f"{species}: {outside}"


def test_the_species_are_visibly_different_shapes():
    """The point of parameterising rather than modelling twelve meshes is that
    the parameters actually separate the species. A dwarf is squat, a skeleton is
    thin and an ogre is broad, measured as width over height off the displaced
    vertices -- if this ever collapses, the registry is twelve palettes."""
    ratios = {}
    cache: dict[str, generate.Baked] = {}
    for key in ("dwarf", "skeleton", "ogre", "human"):
        fam = familylib.get_family(key)
        baked = cache.setdefault(fam.silhouette, generate.build(fam.silhouette))
        positions = baked.positions.copy()
        for channel, value in sorted(fam.appearance_defaults().items()):
            positions = positions + value * baked.displacements[channel]
        span = positions.max(axis=0) - positions.min(axis=0)
        ratios[key] = float(span[0] / span[1])
    assert ratios["skeleton"] < ratios["human"] < ratios["dwarf"] < ratios["ogre"]


# --- the clips --------------------------------------------------------------


def test_the_humanoid_archetype_uses_the_shipped_template_and_clip_library():
    """No ``ogre.json`` and no ``clips/ogre.json``. A species that wanted lower
    arms says so with its own rest offset, never with a second copy of the walk
    cycle -- two libraries that start identical are two libraries that drift."""
    arch = familylib.get_archetype("humanoid")
    assert (arch.template, arch.clip_library) == ("humanoid", "humanoid")
    assert "ogre" not in rigging.templates()
    library = rigging.clip_library("humanoid")
    assert {c["name"] for c in library["clips"]} >= {"idle", "walk", "attack"}
    assert library["space"] == "delta"


def test_every_animation_the_default_recipe_asks_for_expands_to_its_frames():
    """The seam between a recipe and a render: ``expand_clips`` joins the
    shipped library to the resolved frame table, and a clip that expanded to the
    wrong count would lay one animation's cell into another animation's run."""
    layout = charsheet.resolve_layout(DEFAULT_RECIPE.layout_payload())
    records = clipslib.expand_clips("humanoid", layout)
    assert set(records) == set(DEFAULT_RECIPE.animations)
    for name, frames in DEFAULT_RECIPE.animations.items():
        assert len(records[name]) == frames, name


def test_an_animation_with_no_clip_is_a_broken_build_not_a_user_mistake():
    layout = charsheet.resolve_layout(
        Recipe.from_dict({"family": "ogre", "animations": {"idle": 4}}).layout_payload()
    )
    assert set(clipslib.expand_clips("humanoid", layout)) == {"idle"}


def _bone_basis(direction: np.ndarray) -> np.ndarray:
    """Blender's roll-0 bone matrix: the shortest arc taking +Y onto the bone.

    The clip library is authored in ``delta`` space -- each value is a rotation
    from the bone's *own rest orientation* -- so reading a clip back means
    knowing what that orientation is. This is the construction Blender uses, and
    the check that it is the right one is in the file itself: the library's
    header says "local X is the swing axis for both thigh and upper_arm", and
    under this basis the thigh's local X comes out as world X exactly.
    """
    d = np.asarray(direction, dtype="f8")
    d = d / np.linalg.norm(d)
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(y, d)
    c = float(np.dot(y, d))
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx / (1.0 + c)


def _quat_matrix(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _forward_kinematics(pose: dict, template) -> dict[str, np.ndarray]:
    """``bone -> world tail``, for one delta-space key pose on the template."""
    bones = {b["name"]: b for b in template.bones}
    rotations: dict[str, np.ndarray] = {}
    heads: dict[str, np.ndarray] = {}
    root = np.array(pose.get("root_translation") or [0.0, 0.0, 0.0], dtype="f8")
    for bone in template.bones:
        name = bone["name"]
        head = np.array(bone["head"], dtype="f8")
        basis = _bone_basis(np.array(bone["tail"], dtype="f8") - head)
        local = basis @ _quat_matrix(pose["bones"].get(name, [0.0, 0.0, 0.0, 1.0])) @ basis.T
        parent = bone["parent"]
        if parent is None:
            rotations[name] = local
            heads[name] = head + root
        else:
            rotations[name] = rotations[parent] @ local
            heads[name] = heads[parent] + rotations[parent] @ (
                head - np.array(bones[parent]["head"], dtype="f8")
            )
    return {
        b["name"]: heads[b["name"]]
        + rotations[b["name"]]
        @ (np.array(b["tail"], dtype="f8") - np.array(b["head"], dtype="f8"))
        for b in template.bones
    }


def test_the_attack_wind_raises_a_hand_above_the_rest_crown():
    """What the union framing in the sheet renderer exists for.

    The template's crown is z = 1.0 by definition, so a swing whose highest hand
    never passes it would fit inside the rest bounding box -- and a sheet framed
    on the rest pose would be a sheet where nothing ever leaves the cell. The
    shipped humanoid ``attack`` is a right-handed overhead swing and its wind key
    takes the right hand to 1.06, six per cent of a body length clear of the
    crown. This is the measurement, not an aspiration: if the clip is ever
    re-authored flatter, the framing has to be re-argued.
    """
    template = rigging.get_template("humanoid")
    poses = {p["name"]: p for p in rigging.clip_library("humanoid")["poses"].values()}
    keys = rigging.clip_keys("humanoid", "attack")
    highest = max(
        max(_forward_kinematics(key, template)[f"hand.{side}"][2] for side in ("L", "R"))
        for key in keys
    )
    assert highest > 1.0, f"the attack never leaves the rest bbox (peak {highest:.3f})"
    rest = _forward_kinematics(poses[keys[0]["name"]], template)
    assert highest > max(rest[f"hand.{s}"][2] for s in ("L", "R"))


def test_the_bone_basis_agrees_with_what_the_clip_library_says_about_itself():
    """The library's header claims local X is the swing axis for the thigh. If
    that were not this basis, every angle read back above would be about the
    wrong axis and the measurement would be meaningless."""
    template = rigging.get_template("humanoid")
    thigh = next(b for b in template.bones if b["name"] == "thigh.L")
    basis = _bone_basis(np.array(thigh["tail"]) - np.array(thigh["head"]))
    assert basis[:, 0] == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)


# --- instantiation ----------------------------------------------------------


def test_instantiating_writes_the_three_files_a_job_directory_serves(tmp_path):
    inst = instantiate(DEFAULT_RECIPE, tmp_path)
    for name in ("source.glb", "model.glb", "character.json"):
        assert (tmp_path / name).is_file(), name
    # No temp names left behind: every one of the three is staged and replaced.
    assert not list(tmp_path.glob(".*.tmp"))
    assert inst.family == "ogre"
    assert inst.bone_names == [b["name"] for b in rigging.get_template("humanoid").bones]
    assert set(inst.materials) == set(familylib.get_archetype("humanoid").regions)


def test_instantiating_twice_produces_byte_identical_files(tmp_path):
    """What makes a character cacheable and a rerun meaningful. Nothing in the
    path may consult a clock, a random number generator or a set iteration
    order."""
    a, b = tmp_path / "a", tmp_path / "b"
    instantiate(DEFAULT_RECIPE, a)
    instantiate(DEFAULT_RECIPE, b)
    for name in ("source.glb", "model.glb", "character.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


@pytest.mark.parametrize("species", SPECIES)
def test_every_species_instantiates_grounded_at_its_own_height(species, tmp_path):
    fam = familylib.get_family(species)
    recipe = Recipe.from_dict({"family": species, "theme": fam.themes[0].key})
    instantiate(recipe, tmp_path)
    model = gltf.load(tmp_path / "model.glb")
    positions = _stacked(model)
    lo, hi = positions.min(axis=0), positions.max(axis=0)
    assert lo[1] == pytest.approx(0.0, abs=1e-5), "not grounded"
    assert hi[1] - lo[1] == pytest.approx(fam.height_m, rel=1e-5)


def test_the_sidecar_says_what_was_built_and_what_was_asked_for(tmp_path):
    instantiate(DEFAULT_RECIPE, tmp_path)
    sidecar = json.loads((tmp_path / "character.json").read_text(encoding="utf-8"))
    assert sidecar["family"] == "ogre"
    assert sidecar["archetype"] == "humanoid"
    assert sidecar["template"] == "humanoid"
    assert sidecar["height_m"] == pytest.approx(2.6)
    assert sidecar["recipe"] == DEFAULT_RECIPE.as_dict()
    # The joints in the sidecar are the ones a rig job would be handed, so they
    # go through the same door a hand-corrected skeleton does.
    rigging.validate_joints(
        {"bones": [{k: b[k] for k in ("name", "head", "tail")} for b in sidecar["joints"]]},
        rigging.get_template("humanoid"),
    )


def test_a_mask_file_from_a_different_bake_is_refused_rather_than_applied(tmp_path, monkeypatch):
    """The two files are one artifact. A mismatched pair would displace the
    wrong vertices, worst on the ones that move most, and produce a character
    that is subtly and inexplicably wrong rather than obviously broken -- so the
    digest is checked at every load rather than trusted because the pair shipped
    together."""
    from pathlib import Path

    fam = familylib.get_family("ogre")
    with np.load(fam.masks_npz, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["positions_digest"] = np.zeros_like(arrays["positions_digest"])
    forged = tmp_path / "forged.npz"
    np.savez_compressed(forged, **arrays)

    monkeypatch.setattr(
        familylib.Family, "masks_npz", property(lambda self: Path(forged))
    )
    with pytest.raises(CharacterError) as excinfo:
        instantiate(DEFAULT_RECIPE, tmp_path / "out")
    assert excinfo.value.field == "family"
    assert "baked against a different" in str(excinfo.value)


def test_transformed_joints_uses_the_same_convention_clay_does(tmp_path):
    """``M @ v``, column vectors -- so one matrix moves a mesh through
    ``clay.mesh.transformed`` and its skeleton through this, and nobody has to
    remember which of the two wanted the transpose."""
    from warlock.characters.instantiate import transformed_joints

    inst = instantiate(DEFAULT_RECIPE, tmp_path)
    matrix = np.array(
        [[2.0, 0.0, 0.0, 1.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, -3.0], [0, 0, 0, 1.0]]
    )
    moved = transformed_joints(inst.joints, matrix)
    for before, after in zip(inst.joints, moved, strict=True):
        expected = [2 * before["head"][0] + 1, 2 * before["head"][1], 2 * before["head"][2] - 3]
        assert after["head"] == pytest.approx(expected)
        assert after["parent"] == before["parent"]


def test_a_species_with_no_baked_mesh_says_which_script_makes_one(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(
        familylib.Family, "base_glb", property(lambda self: Path(tmp_path / "missing.glb"))
    )
    with pytest.raises(CharacterError) as excinfo:
        instantiate(DEFAULT_RECIPE, tmp_path)
    assert "author_humanoid" in str(excinfo.value)
