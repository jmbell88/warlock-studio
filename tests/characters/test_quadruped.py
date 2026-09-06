"""The quadruped archetype: the bake, the skeleton it fits, and the clips.

``tests/characters/test_humanoid.py``'s claims, made again for a body plan that
runs along its own depth: every shipped species is a shape you can pose, it
stays one at both ends of every appearance slider, and the shipped clips move it
somewhere worth rendering.

Two things here are *not* copies of the humanoid file, and both are measured
rather than stylistic.

**The containment ray is not axis-aligned and the nudge is absolute.** A
crossing count along +X is unreliable on a body with a flat surface lying in the
ray's own plane, and a nudge of one per cent toward the centroid -- fine on a
humanoid, whose thinnest landmark-bearing feature is a forearm -- is larger than
the whole thickness of a smoothed wingtip or a tail tip. So the ray is oblique
and the nudge is :data:`NUDGE`, sized between the two numbers that bracket it:
larger than the thousandth of a body height a smoothed extremity retracts from
its own bounding plane, smaller than the thinnest feature any landmark sits in.

**Regeneration is compared to a tolerance, not byte for byte.** ``manifold3d``
does not promise a bit-identical arrangement across versions and Catmull-Clark
is a float sum over an adjacency. What has to hold is that it is the same mesh:
the same counts, the same vertices to 1e-5 of unit height, the same channels.
"""

from __future__ import annotations

import numpy as np
import pytest

from warlock import clips as clipslib
from warlock import rigging
from warlock.characters import Recipe
from warlock.characters import family as familylib
from warlock.characters.instantiate import instantiate
from warlock.characters.quadruped import generate
from warlock.pipelines import charsheet
from warlock.studio.viewer import gltf

ARCHETYPE = "quadruped"
SILHOUETTES = sorted(familylib.silhouettes(ARCHETYPE))
SPECIES = sorted(familylib.families_of(ARCHETYPE))
CHANNELS = [c.key for c in familylib.get_archetype(ARCHETYPE).channels]

#: 1e-5 of unit height is seventeen microns on a 1.7 m horse -- far below what a
#: 64px sprite can show and far above float noise.
TOL = 1e-5

#: How far a landmark is nudged toward the centroid before the crossing count.
#: See the module docstring: two and a half thousandths of a body height is
#: above the retraction of a smoothed extremity from its own bounding plane and
#: below the half-thickness of the thinnest thing any landmark sits inside.
NUDGE = 0.0025

#: An oblique ray. Every axis-aligned direction is parallel to some flat face of
#: some silhouette here -- the soles, the vanes -- and a crossing count along a
#: surface it lies in is a coin toss.
RAY = np.array([0.3117, 0.7071, 0.4231]) / np.linalg.norm([0.3117, 0.7071, 0.4231])


@pytest.fixture(scope="module")
def rebuilt() -> dict[str, generate.Baked]:
    """Every silhouette, generated from scratch once for the whole module."""
    return {key: generate.build(key) for key in SILHOUETTES}


@pytest.fixture(scope="module")
def shipped() -> dict[str, tuple[gltf.Model, dict[str, np.ndarray]]]:
    out = {}
    for key in SILHOUETTES:
        # Scoped to the archetype, not to ``families()``: a silhouette key only
        # has to be unique inside its own body plan, and the humanoid's
        # ``tusked`` and this archetype's tusker were one letter from picking
        # each other's asset out of a global search.
        fam = next(f for f in familylib.families_of(ARCHETYPE).values() if f.silhouette == key)
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
    """Which of *points* are inside the closed surface. Moller-Trumbore.

    A real containment test rather than a bounding-box one: "the joint is
    somewhere in the box" is true of a joint hanging in the air beside the hip,
    and a bone hanging in the air is what sends Blender's heat solve into the
    envelope fallback the inspector calls a degraded outcome.
    """
    a = positions[tris[:, 0]]
    b = positions[tris[:, 1]]
    c = positions[tris[:, 2]]
    e1, e2 = b - a, c - a
    h = np.cross(RAY, e2)
    det = np.einsum("ij,ij->i", e1, h)
    usable = np.abs(det) > 1e-14
    safe = np.where(usable, det, 1.0)
    out = []
    for point in points:
        s = point - a
        u = np.einsum("ij,ij->i", s, h) / safe
        q = np.cross(s, e1)
        v = (q @ RAY) / safe
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
    """Every point nudged :data:`NUDGE` toward the centroid.

    Some landmarks sit **on** the surface by construction and not by accident:
    the nose tip is the front-top corner of the box, the soles are the floor and
    the tail's tip is the back. All of them are exactly on the boundary, where a
    crossing count is a coin toss, so the question the test means -- "is this
    joint in the body" -- is asked one nudge inside.
    """
    centre = positions.mean(axis=0)
    delta = centre - points
    norm = np.linalg.norm(delta, axis=1)
    norm[norm == 0] = 1.0
    return points + NUDGE * delta / norm[:, None]


def _outside(baked: generate.Baked, positions: np.ndarray, points: np.ndarray) -> list[str]:
    inside = _inside(positions, _triangles(baked), _pulled_in(positions, points.reshape(-1, 3)))
    names = [b["name"] for b in baked.joints]
    return [
        f"{names[i // 2]}.{('head', 'tail')[i % 2]}" for i, ok in enumerate(inside) if not ok
    ]


# --- the bake ---------------------------------------------------------------


def test_every_silhouette_group_has_an_asset():
    assert set(generate.GROUPS) == set(SILHOUETTES)
    for key in SILHOUETTES:
        # Scoped to the archetype, not to ``families()``: a silhouette key only
        # has to be unique inside its own body plan, and the humanoid's
        # ``tusked`` and this archetype's tusker were one letter from picking
        # each other's asset out of a global search.
        fam = next(f for f in familylib.families_of(ARCHETYPE).values() if f.silhouette == key)
        assert fam.base_glb.is_file(), key
        assert fam.masks_npz.is_file(), key


@pytest.mark.parametrize("silhouette", SILHOUETTES)
def test_regenerating_reproduces_the_checked_in_asset(silhouette, rebuilt, shipped):
    """The generator is the record of *how* and the asset is the record of
    *what*, and this is the only thing keeping the two honest."""
    baked = rebuilt[silhouette]
    model, arrays = shipped[silhouette]
    _fresh_data, fresh_arrays = generate.bake(baked)

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
    on the ones that move most."""
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
    Blender's bone-heat solve refuses non-manifold input and the fallback is
    envelope weighting, which the inspector reports as needing review.

    This is also the test that pinned ``SMOOTH_EPS``. At the humanoid's 1e-5 the
    paw group came back with ten non-manifold edges on the muzzle, where the
    second weld was collapsing distinct vertices rather than tidying what
    Catmull-Clark had recreated.
    """
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
def test_a_quadruped_is_longer_than_it_is_tall(silhouette, rebuilt):
    """The one claim that makes this archetype not a humanoid on all fours: the
    body runs along the box's depth. If a group ever came out taller than it is
    long, every clip in ``clips/quadruped.json`` would be animating a shape the
    stride lengths were not authored for."""
    lo, hi = rebuilt[silhouette].bounds
    assert (hi[2] - lo[2]) > 1.05 * (hi[1] - lo[1]), silhouette


@pytest.mark.parametrize("silhouette", SILHOUETTES)
def test_the_regions_on_a_mesh_are_the_ones_its_features_call_for(silhouette, rebuilt):
    """A region with no faces is a colour nobody can see and -- worse -- a theme
    key that looks configured and does nothing. ``mane`` is the one that varies,
    and it varies with the group's own flag rather than with a guess."""
    baked = rebuilt[silhouette]
    names = familylib.get_archetype(ARCHETYPE).regions
    present = {names[i] for i in np.unique(baked.regions)}
    assert {"hide", "underbelly", "horn", "eye", "accent"} <= present
    assert ("mane" in present) == generate.GROUPS[silhouette].mane


# --- the skeleton -----------------------------------------------------------


@pytest.mark.parametrize("silhouette", SILHOUETTES)
def test_the_generated_mesh_fits_the_quadruped_template_exactly(silhouette, rebuilt):
    """The mesh is grown *around* the shipped template's landmarks, so the
    landmarks fitted to the finished mesh's own bounding box have to come back
    where the body was built. Anything else means the fixed-point loop in
    ``build`` closed on the wrong number and every joint is off by that much."""
    baked = rebuilt[silhouette]
    lo, hi = baked.bounds
    fitted = rigging.fit_template(
        rigging.get_template("quadruped"),
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
    outside = _outside(baked, baked.positions, _joint_points(baked))
    assert not outside, f"{silhouette}: {outside} are outside the mesh"


@pytest.mark.parametrize("channel", CHANNELS)
@pytest.mark.parametrize("silhouette", SILHOUETTES)
def test_a_channel_at_either_bound_keeps_the_skeleton_in_the_body(silhouette, channel, rebuilt):
    """The claim that makes the sliders safe to expose. Both fields -- the one
    over the vertices and the one over the joints -- are the *same function*
    evaluated on two point sets, and this is what says so.

    ``tail_length`` is the reason the test names both bounds. Its mask started
    as a projection onto the tail's axis with no distance term, and a hind foot
    sits forward of the tail's root but still projects onto it: at -1 the
    shortening reached down the legs and took two ankle joints out through the
    sole.
    """
    baked = rebuilt[silhouette]
    joints = _joint_points(baked)
    for value in (-1.0, 1.0):
        positions = baked.positions + value * baked.displacements[channel]
        moved = joints + value * baked.joint_displacements[channel]
        outside = _outside(baked, positions, moved)
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
    six of them stack at once. The bear carries +0.85 bulk with -0.35 neck and
    -0.6 tail together, and no single-channel test would catch the pair that
    only fails in combination."""
    fam = familylib.get_family(species)
    baked = rebuilt[fam.silhouette]
    positions = baked.positions.copy()
    joints = _joint_points(baked)
    for channel, value in sorted(fam.appearance_defaults().items()):
        positions = positions + value * baked.displacements[channel]
        joints = joints + value * baked.joint_displacements[channel]
    assert not _outside(baked, positions, joints), species


def test_the_species_are_visibly_different_animals():
    """The point of parameterising rather than modelling eight meshes is that
    the parameters actually separate the species. A bear is bulkier than a big
    cat, a deer is slighter than either, and a lizard is longer than all of
    them -- measured off the displaced vertices. If this ever collapses, the
    registry is eight palettes."""
    cache: dict[str, generate.Baked] = {}
    width: dict[str, float] = {}
    length: dict[str, float] = {}
    for key in SPECIES:
        fam = familylib.get_family(key)
        baked = cache.setdefault(fam.silhouette, generate.build(fam.silhouette))
        positions = baked.positions.copy()
        for channel, value in sorted(fam.appearance_defaults().items()):
            positions = positions + value * baked.displacements[channel]
        span = positions.max(axis=0) - positions.min(axis=0)
        width[key] = float(span[0] / span[1])
        length[key] = float(span[2] / span[1])
    assert width["deer"] < width["big_cat"] < width["bear"] < width["boar"]
    assert length["horse"] < length["wolf"] < length["big_cat"] < length["lizard"]
    # Every one of them is an animal on four legs, not a person on two.
    assert min(length.values()) > 1.0


# --- the clips --------------------------------------------------------------


def test_the_quadruped_archetype_uses_the_shipped_template_and_clip_library():
    """No ``wolf.json`` and no ``clips/wolf.json``. A species that wanted a
    shorter stride says so with its own channel defaults, never with a second
    copy of the walk cycle -- two libraries that start identical are two
    libraries that drift."""
    arch = familylib.get_archetype(ARCHETYPE)
    assert (arch.template, arch.clip_library) == ("quadruped", "quadruped")
    assert "wolf" not in rigging.templates()
    library = rigging.clip_library("quadruped")
    assert {c["name"] for c in library["clips"]} >= {"idle", "walk", "run", "attack", "jump"}
    assert library["space"] == "delta"


def test_every_animation_a_quadruped_recipe_asks_for_expands_to_its_frames():
    """The seam between a recipe and a render: ``expand_clips`` joins the
    shipped library to the resolved frame table, and a clip that expanded to the
    wrong count would lay one animation's cell into another animation's run."""
    recipe = Recipe.from_dict(
        {"family": "wolf", "animations": {"idle": 4, "walk": 8, "run": 8, "attack": 6, "jump": 6}}
    )
    layout = charsheet.resolve_layout(recipe.layout_payload())
    records = clipslib.expand_clips("quadruped", layout)
    assert set(records) == set(recipe.animations)
    for name, frames in recipe.animations.items():
        assert len(records[name]) == frames, name


# --- instantiation ----------------------------------------------------------


@pytest.mark.parametrize("species", SPECIES)
def test_every_species_instantiates_grounded_at_its_own_height(species, tmp_path):
    fam = familylib.get_family(species)
    recipe = Recipe.from_dict({"family": species, "theme": fam.themes[0].key})
    inst = instantiate(recipe, tmp_path)
    model = gltf.load(tmp_path / "model.glb")
    positions = _stacked(model)
    lo, hi = positions.min(axis=0), positions.max(axis=0)
    assert lo[1] == pytest.approx(0.0, abs=1e-5), "not grounded"
    assert hi[1] - lo[1] == pytest.approx(fam.height_m, rel=1e-5)
    assert inst.bone_names == [b["name"] for b in rigging.get_template("quadruped").bones]
    assert set(inst.materials) == set(familylib.get_archetype(ARCHETYPE).regions)
