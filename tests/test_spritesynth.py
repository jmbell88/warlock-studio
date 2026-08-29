"""The pure half of sprite synthesis: geometry, guides, mattes, arithmetic.

Everything here runs with a Pillow-drawn fixture and no GPU, which is the whole
point of the module being pure -- the parts of this feature that can be wrong
silently (a cell rectangle off by one, a palette that is not shared, a baseline
that clips a head off) are all decided here.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from warlock.pipelines import control, reference
from warlock.pipelines import spritesynth as ss

BG = (200, 200, 200)


def _atlas(kind, *, sizes=None, offset=(0, 0), bg=BG):
    """An atlas with one dark blob per cell, standing on the cell's floor."""
    geom = ss.geometry(kind)
    im = Image.new("RGB", (ss.ATLAS_PX, ss.ATLAS_PX), bg)
    draw = ImageDraw.Draw(im)
    for index, cell in enumerate(geom.cells):
        h = (sizes or {}).get(index, cell.h // 2)
        w = cell.w // 4
        cx = cell.x + cell.w // 2 + offset[0]
        bottom = cell.y + cell.h - cell.h // 10 + offset[1]
        draw.rectangle(
            (cx - w // 2, bottom - h, cx + w // 2, bottom), fill=(20, 30, 40)
        )
    return geom, im


# --- geometry ---------------------------------------------------------------


@pytest.mark.parametrize("kind", ss.SHEET_TYPES)
def test_the_cells_tile_the_atlas_exactly(kind):
    geom = ss.geometry(kind)
    assert geom.columns * geom.cell_w == ss.ATLAS_PX
    assert geom.rows * geom.cell_h == ss.ATLAS_PX
    covered = np.zeros((ss.ATLAS_PX, ss.ATLAS_PX), dtype=int)
    for cell in geom.cells:
        covered[cell.y : cell.y + cell.h, cell.x : cell.x + cell.w] += 1
    assert covered.min() == 1 and covered.max() == 1


def test_the_turnaround_order_is_front_left_right_back():
    cells = ss.geometry("turnaround").cells
    assert [c.name for c in cells] == list(ss.DIRECTION_ORDER)
    assert [(c.row, c.col) for c in cells] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert all(c.frame == 0 for c in cells)


def test_the_walk_rows_are_one_direction_each_left_to_right():
    geom = ss.geometry("walk")
    assert geom.frames_per_direction == 4
    for row, name in enumerate(ss.DIRECTION_ORDER):
        row_cells = [c for c in geom.cells if c.row == row]
        assert [c.name for c in row_cells] == [name] * 4
        assert [c.frame for c in row_cells] == [0, 1, 2, 3]
        assert [c.col for c in row_cells] == [0, 1, 2, 3]


@pytest.mark.parametrize("kind", ss.SHEET_TYPES)
def test_every_cell_carries_its_directions_yaw(kind):
    for cell in ss.geometry(kind).cells:
        assert cell.yaw == ss.DIRECTION_YAWS[cell.name]


def test_an_unknown_sheet_type_raises_rather_than_defaulting():
    with pytest.raises(ValueError, match="unknown sprite sheet type"):
        ss.geometry("isometric")


def test_a_planned_kind_is_not_a_fixed_atlas():
    """``geometry`` is the *legacy atlas* table and stays it. A planned kind
    reaching it would hand back a KeyError-shaped refusal, so the message names
    the door that does serve it."""
    with pytest.raises(ValueError, match="plan_sheet"):
        ss.geometry("walk8")


def test_the_legacy_geometries_carry_no_bands():
    """Empty ``bands`` is the signal for 'one 1024px atlas, laid out the way
    every draft on disk is' -- and the legacy cell rectangles stay generation
    pixels, because split/matte/preserve_front address the atlas through them."""
    for kind in ss.SHEET_TYPES:
        geom = ss.geometry(kind)
        assert geom.bands == ()
        assert geom.cell_w == ss.ATLAS_PX // geom.columns
        assert geom.cell_h == ss.ATLAS_PX // geom.rows


# --- planned sheets ---------------------------------------------------------

PLANNED = sorted(ss.PLANNED_KINDS)


def test_the_four_direction_order_is_the_legacy_one_not_the_clockwise_preset():
    """The trap: ``charsheet.DIRECTION_PRESETS[4]`` sweeps clockwise and every
    sprite draft on disk is front/left/right/back. Same set, different order --
    and taking the preset's order would relabel the back and right rows of
    every stored draft."""
    from warlock.pipelines import charsheet

    preset = tuple(name for name, _yaw in charsheet.DIRECTION_PRESETS[4])
    assert set(preset) == set(ss.SPRITE_DIRECTIONS[4])
    assert preset != ss.SPRITE_DIRECTIONS[4]
    assert ss.SPRITE_DIRECTIONS[4] == ss.DIRECTION_ORDER


def test_the_eight_directions_are_charsheets_own_and_not_a_third_copy():
    from warlock.pipelines import charsheet

    assert ss.SPRITE_DIRECTIONS[8] == tuple(
        name for name, _yaw in charsheet.DIRECTION_PRESETS[8]
    )
    yaws = {name: int(yaw) for name, yaw in charsheet.DIRECTIONS}
    assert yaws == ss.DIRECTION_YAWS_8
    # The four legacy yaws are the same numbers, or the two eras disagree about
    # where "back" is.
    for name, yaw in ss.DIRECTION_YAWS.items():
        assert ss.DIRECTION_YAWS_8[name] == yaw


def test_the_shared_actions_agree_with_troupes_frame_table():
    """Two tables on purpose -- ``hurt`` and ``cast`` have no Blender clip, so
    adding them to ``charsheet.ANIMATIONS`` would raise a KeyError an hour into
    a job -- and this is the overlap. A walk that is eight frames here and six
    there is a sheet whose two halves disagree about what a cycle is."""
    from warlock.pipelines import charsheet

    troupe = {name: frames for name, frames, _loop, _ms in charsheet.ANIMATIONS}
    shared = set(troupe) & set(ss.ACTION_FRAMES)
    assert shared == {"idle", "walk", "run", "attack", "jump"}
    for name in sorted(shared):
        assert ss.ACTION_FRAMES[name] == troupe[name]


def test_the_two_extra_actions_are_deliberately_not_troupes():
    from warlock.pipelines import charsheet

    troupe = {name for name, *_rest in charsheet.ANIMATIONS}
    assert set(ss.ACTION_FRAMES) - troupe == {"cast", "hurt"}


@pytest.mark.parametrize("kind", PLANNED)
def test_a_planned_sheet_is_one_direction_per_row_and_one_frame_per_column(kind):
    geom = ss.plan_kind(kind)
    assert geom.rows == len(geom.directions)
    assert geom.columns == geom.frames_per_direction
    assert len(geom.cells) == geom.rows * geom.columns
    for index, cell in enumerate(geom.cells):
        assert (cell.row, cell.col) == (index // geom.columns, index % geom.columns)
        assert cell.name == geom.directions[cell.row]
        assert cell.frame == cell.col
        assert cell.yaw == ss.DIRECTION_YAWS_8[cell.name]


@pytest.mark.parametrize("kind", PLANNED)
def test_a_planned_sheets_published_cells_are_logical_pixels(kind):
    geom = ss.plan_kind(kind, 24)
    assert (geom.cell_w, geom.cell_h) == (24, 24)
    for cell in geom.cells:
        assert (cell.w, cell.h) == (24, 24)
        assert (cell.x, cell.y) == (cell.col * 24, cell.row * 24)


@pytest.mark.parametrize("kind", PLANNED)
def test_every_cell_is_in_exactly_one_band(kind):
    geom = ss.plan_kind(kind)
    seen = [(c.name, c.frame) for band in geom.bands for c in band.cells]
    assert sorted(seen) == sorted((c.name, c.frame) for c in geom.cells)
    assert len(seen) == len(set(seen))


@pytest.mark.parametrize("kind", PLANNED)
def test_no_direction_is_split_across_bands(kind):
    """The whole argument for the band: drift between two frames of one
    direction plays at 10fps and reads as flicker, so they share a latent."""
    geom = ss.plan_kind(kind)
    assert len(geom.bands) == len(geom.directions)
    for band, name in zip(geom.bands, geom.directions, strict=True):
        assert band.direction == name
        assert {c.name for c in band.cells} == {name}
        assert [c.frame for c in band.cells] == list(range(geom.frames_per_direction))


@pytest.mark.parametrize("kind", PLANNED)
@pytest.mark.parametrize("logical", (16, 24, 32))
def test_no_band_exceeds_one_sdxl_frame_in_either_axis(kind, logical):
    geom = ss.plan_kind(kind, logical)
    for band in geom.bands:
        width, height = band.size
        assert 0 < width <= ss.ATLAS_PX
        assert 0 < height <= ss.ATLAS_PX
        assert (width, height) == (band.columns * band.cell_px, band.rows * band.cell_px)


@pytest.mark.parametrize("kind", PLANNED)
@pytest.mark.parametrize("logical", (16, 24, 32))
def test_no_cell_is_drawn_below_the_loras_art_resolution(kind, logical):
    """``tilesheet``'s measurement: the pixel-art LoRA spends ~8 generation
    pixels on one authored pixel at 1024, so a cell below that asks for detail
    it cannot resolve and comes back as mush."""
    geom = ss.plan_kind(kind, logical)
    for band in geom.bands:
        assert band.cell_px >= ss.PX_PER_ART_PIXEL * logical
        for cell in band.cells:
            assert cell.w == cell.h == band.cell_px


@pytest.mark.parametrize("kind", PLANNED)
def test_a_bands_cells_tile_it_exactly_and_start_at_its_own_origin(kind):
    geom = ss.plan_kind(kind)
    for band in geom.bands:
        width, height = band.size
        covered = np.zeros((height, width), dtype=int)
        for cell in band.cells:
            covered[cell.y : cell.y + cell.h, cell.x : cell.x + cell.w] += 1
        # A band whose frame count does not fill its grid leaves empty cells,
        # which is legal; what is not legal is two cells over one pixel.
        assert covered.max() == 1
        assert covered.sum() == len(band.cells) * band.cell_px**2


@pytest.mark.parametrize(
    ("frames", "expected"), ((1, (1, 1)), (4, (2, 2)), (6, (3, 2)), (8, (4, 2)))
)
def test_the_band_grids_are_the_ones_the_arithmetic_names(frames, expected):
    assert ss.band_grid(frames) == expected


def test_the_band_sizes_at_the_default_logical_size():
    """The table from the argument: 32px logical is a 256px cell, so four
    frames is 512x512, six is 768x512 and eight is 1024x512."""
    assert ss.plan_sheet("idle", 8).bands[0].size == (512, 512)
    assert ss.plan_sheet("attack", 8).bands[0].size == (768, 512)
    assert ss.plan_sheet("walk", 8).bands[0].size == (1024, 512)


def test_a_direction_too_big_for_one_band_is_refused_with_both_numbers():
    """Not a limitation, arithmetic. A 64px sprite is an honest 512px cell, and
    four of those fill a band exactly -- so an eight-frame walk at that size
    would have to split a direction across two denoises, which is the flicker
    the whole band rule exists to refuse."""
    assert ss.plan_sheet("idle", 8, None, 64).bands[0].size == (1024, 1024)
    with pytest.raises(ValueError, match="1024x1024"):
        ss.plan_sheet("walk", 8, None, 64)
    with pytest.raises(ValueError, match="never split"):
        ss.plan_sheet("attack", 8, None, 64)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        (("fly", 8, None, 32), "unknown sprite action"),
        (("walk", 6, None, 32), "4 or 8 directions"),
        (("walk", 8, 0, 32), "at least one frame"),
        (("walk", 8, None, 0), "at least 1"),
    ),
)
def test_plan_sheet_refuses_rather_than_defaulting(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ss.plan_sheet(*kwargs)


def test_an_unknown_kind_names_the_ones_that_exist():
    with pytest.raises(ValueError, match="unknown sprite sheet kind"):
        ss.plan_kind("gallop12")


def test_sheet_geometry_serves_both_eras_through_one_door():
    assert ss.sheet_geometry("walk") is ss.GEOMETRY["walk"]
    assert ss.sheet_geometry("walk8").kind == "walk8"


# --- action subjects --------------------------------------------------------


def test_a_subject_carries_the_prompt_the_action_and_the_view():
    text = ss.action_subject("a tin knight", "walk", "back_left")
    assert text.startswith("a tin knight, ")
    assert "walking" in text
    assert "rear three-quarter" in text


def test_every_action_and_every_direction_has_a_clause():
    for action in ss.ACTION_FRAMES:
        for direction in ss.SPRITE_DIRECTIONS[8]:
            assert ss.action_subject("x", action, direction).count(",") >= 2


def test_an_unknown_action_raises_rather_than_defaulting():
    """``tilesheet.sheet_subject``'s rule and its reason: a fallback here is
    invisible by construction, because it produces a plausible sheet described
    by the wrong sentence."""
    with pytest.raises(ValueError, match="unknown sprite action"):
        ss.action_subject("x", "dance", "front")


def test_an_unknown_direction_raises_too():
    with pytest.raises(ValueError, match="unknown sprite direction"):
        ss.action_subject("x", "walk", "upward")


def test_an_empty_prompt_leaves_no_leading_comma():
    assert not ss.action_subject("   ", "idle", "front").startswith(",")


def test_the_subjects_do_not_touch_the_shared_prompt_template():
    """The split ``tilesheet`` draws: the template serves the prompt preview and
    versions under ``PROMPT_VERSION``, these clauses are this module's own and
    version under ``SPRITE_DRAFT_VERSION``. Nothing here bumps that number."""
    from warlock.pipelines import prompt

    assert "walking at an even pace" not in prompt.SHEET_TEMPLATE


# --- guide templates --------------------------------------------------------


@pytest.mark.parametrize("kind", ss.SHEET_TYPES)
def test_the_shipped_template_covers_every_cell_in_order(kind):
    geom = ss.geometry(kind)
    template = ss.load_guide_template(kind)
    assert [(p.name, p.frame) for p in template.poses] == [
        (c.name, c.frame) for c in geom.cells
    ]


@pytest.mark.parametrize("kind", ss.SHEET_TYPES)
def test_the_shipped_template_has_a_comment_explaining_the_convention(kind):
    raw = json.loads((ss.TEMPLATE_DIR / f"{kind}.json").read_text(encoding="utf-8"))
    assert len(raw["comment"]) > 100


def _raw(kind):
    return json.loads((ss.TEMPLATE_DIR / f"{kind}.json").read_text(encoding="utf-8"))


def test_a_segment_naming_an_unknown_joint_is_refused_at_load():
    raw = _raw("turnaround")
    raw["segments"].append(["neck", "tentacle"])
    with pytest.raises(ValueError, match="missing a point"):
        ss._parse_template(raw, ss.geometry("turnaround"))


def test_a_template_with_the_wrong_cell_count_is_refused_at_load():
    raw = _raw("turnaround")
    raw["poses"] = raw["poses"][:3]
    with pytest.raises(ValueError, match="3 poses"):
        ss._parse_template(raw, ss.geometry("turnaround"))


def test_a_template_in_the_wrong_order_is_refused_at_load():
    raw = _raw("turnaround")
    raw["poses"][0], raw["poses"][1] = raw["poses"][1], raw["poses"][0]
    with pytest.raises(ValueError, match="wrong order"):
        ss._parse_template(raw, ss.geometry("turnaround"))


def test_a_point_outside_its_cell_is_refused_at_load():
    raw = _raw("turnaround")
    raw["poses"][0]["points"]["head"] = [1.4, 0.2]
    with pytest.raises(ValueError, match="outside its cell"):
        ss._parse_template(raw, ss.geometry("turnaround"))


def test_a_pose_with_no_head_is_refused_at_load():
    raw = _raw("turnaround")
    del raw["poses"][0]["points"]["head"]
    with pytest.raises(ValueError, match="no 'head' point"):
        ss._parse_template(raw, ss.geometry("turnaround"))


# --- mirror-derived templates -----------------------------------------------


def test_the_legacy_templates_carry_no_mirror_key_and_author_every_cell():
    """``turnaround.json`` and ``walk.json`` are untouched by the mirror, which
    is what keeps them byte for byte what they were.

    Asserted as a property rather than as a hash of the file, deliberately:
    ``turnaround.json``'s own comment says "This is DATA: the pose quality here
    is iterable without touching any code", and a byte pin would forbid exactly
    that iteration. What must not change is the *shape* -- no mirror key, every
    cell authored -- and git shows the rest.
    """
    for kind in ss.SHEET_TYPES:
        raw = _raw(kind)
        assert "mirror" not in raw
        geom = ss.geometry(kind)
        authored = {(p["name"], p["frame"]) for p in raw["poses"]}
        assert authored == {(c.name, c.frame) for c in geom.cells}


def test_the_shipped_right_view_is_already_the_exact_mirror_of_the_left():
    """The evidence the mirror rule is not an invention: the hand-authored
    ``turnaround.json`` was already following it, to float rounding."""
    raw = _raw("turnaround")
    by_name = {p["name"]: p["points"] for p in raw["poses"]}
    mirrored = ss._mirrored_points(by_name["left"])
    for joint, (x, y) in by_name["right"].items():
        assert mirrored[joint] == pytest.approx((x, y))


def _mirrored_raw(kind="idle8"):
    return json.loads((ss.TEMPLATE_DIR / f"{kind}.json").read_text(encoding="utf-8"))


def test_the_shipped_mirrored_template_expands_to_every_cell_in_order():
    geom = ss.plan_kind("idle8")
    raw = _mirrored_raw()
    assert len(raw["poses"]) == 20  # five authored directions of four frames
    template = ss.load_guide_template("idle8")
    assert [(p.name, p.frame) for p in template.poses] == [
        (c.name, c.frame) for c in geom.cells
    ]


def test_the_derived_directions_are_exact_reflections_of_their_sources():
    template = ss.load_guide_template("idle8")
    poses = {(p.name, p.frame): p for p in template.poses}
    for derived, source in ss._MIRROR_OF.items():
        for frame in range(4):
            assert poses[(derived, frame)].points == pytest.approx(
                ss._mirrored_points(poses[(source, frame)].points)
            )


def test_the_shipped_mirrored_template_has_a_comment_explaining_the_convention():
    assert len(_mirrored_raw()["comment"]) > 100


def test_mirroring_swaps_handedness_and_reflects_x():
    points = {"head": (0.5, 0.1), "hand.L": (0.2, 0.5), "hand.R": (0.9, 0.5)}
    assert ss._mirrored_points(points) == {
        "head": (0.5, 0.1),
        "hand.R": (0.8, 0.5),
        "hand.L": (pytest.approx(0.1), 0.5),
    }


def _tiny(poses, **extra):
    """A one-frame four-direction template over a ``hurt4``-shaped grid."""
    raw = {
        "kind": "idle4",
        "head_point": "head",
        "head_radius": 0.05,
        "segments": [["head", "hip"], ["hip", "foot.L"], ["hip", "foot.R"]],
        "mirror": {"axis": "x", "pairs": "suffix"},
        "poses": poses,
    }
    raw.update(extra)
    return raw


def _pose(name, frame, **points):
    base = {"head": [0.5, 0.1], "hip": [0.5, 0.5], "foot.L": [0.4, 0.9],
            "foot.R": [0.6, 0.9]}
    base.update(points)
    return {"name": name, "frame": frame, "points": base}


def _idle4_authored():
    """front, left and back at every frame -- ``right`` is the derivable one."""
    return [
        _pose(name, frame)
        for name in ("front", "left", "back")
        for frame in range(4)
    ]


def test_a_mirrored_template_need_only_author_the_left_hand_directions():
    template = ss._parse_template(_tiny(_idle4_authored()), ss.plan_kind("idle4"))
    assert len(template.poses) == 16
    assert [p.name for p in template.poses[8:12]] == ["right"] * 4


def test_an_authored_pose_beats_the_derived_one():
    """The escape hatch for an asymmetric action -- a sword swing that must not
    change hands -- without forcing every action to be hand-authored twice."""
    poses = _idle4_authored()
    # Grid order is front, left, right, back, so the authored right sits third.
    poses[8:8] = [_pose("right", frame, head=[0.7, 0.1]) for frame in range(4)]
    template = ss._parse_template(_tiny(poses), ss.plan_kind("idle4"))
    right = [p for p in template.poses if p.name == "right"]
    assert [p.points["head"] for p in right] == [(0.7, 0.1)] * 4


def test_a_joint_that_is_neither_paired_nor_central_is_refused_at_load():
    poses = _idle4_authored()
    poses[0]["points"]["tentacle"] = [0.5, 0.5]
    with pytest.raises(ValueError, match="neither a '.L'/'.R' pair"):
        ss._parse_template(_tiny(poses), ss.plan_kind("idle4"))


def test_a_half_of_a_pair_with_no_partner_is_refused_at_load():
    poses = _idle4_authored()
    del poses[0]["points"]["foot.R"]
    with pytest.raises(ValueError, match="cannot be mirrored"):
        ss._parse_template(_tiny(poses), ss.plan_kind("idle4"))


@pytest.mark.parametrize("missing", ("left", "back"))
def test_a_direction_with_nothing_to_mirror_names_the_half_to_author(missing):
    """One message for both cases, because from the outside they are one case:
    expansion walks the grid in order and every mirror source precedes what is
    derived from it, so a missing ``left`` is reported at ``left`` and never as
    ``right`` having nothing to reflect."""
    poses = [p for p in _idle4_authored() if p["name"] != missing]
    with pytest.raises(ValueError, match=f"no pose for '{missing}'/0"):
        ss._parse_template(_tiny(poses), ss.plan_kind("idle4"))
    with pytest.raises(ValueError, match="front, front_left, left, back_left, back"):
        ss._parse_template(_tiny(poses), ss.plan_kind("idle4"))


def test_the_authored_half_is_derived_from_the_mirror_table():
    assert ss.AUTHORED_DIRECTIONS == (
        "front", "front_left", "left", "back_left", "back",
    )
    assert set(ss.AUTHORED_DIRECTIONS) | set(ss._MIRROR_OF) == set(
        ss.SPRITE_DIRECTIONS[8]
    )


def test_two_poses_for_one_cell_are_refused_at_load():
    poses = _idle4_authored()
    poses.append(_pose("front", 0))
    with pytest.raises(ValueError, match="two poses for 'front'/0"):
        ss._parse_template(_tiny(poses), ss.plan_kind("idle4"))


def test_a_pose_naming_a_cell_the_grid_does_not_have_is_refused_at_load():
    poses = _idle4_authored() + [_pose("front", 9)]
    with pytest.raises(ValueError, match="not a cell of the 'idle4' grid"):
        ss._parse_template(_tiny(poses), ss.plan_kind("idle4"))


def test_authored_poses_out_of_grid_order_are_refused_at_load():
    """The positional order check cannot fire on a mirrored template, because
    the expansion builds the list *by* the grid -- so the order refusal is made
    against the authored poses instead, and keeps its teeth."""
    poses = _idle4_authored()
    poses[0], poses[4] = poses[4], poses[0]
    with pytest.raises(ValueError, match="wrong order"):
        ss._parse_template(_tiny(poses), ss.plan_kind("idle4"))


def test_a_mirror_axis_this_build_cannot_draw_is_refused_rather_than_ignored():
    raw = _tiny(_idle4_authored(), mirror={"axis": "y", "pairs": "suffix"})
    with pytest.raises(ValueError, match="one mirror and it is 'x'"):
        ss._parse_template(raw, ss.plan_kind("idle4"))


def test_a_pairing_scheme_this_build_cannot_draw_is_refused_too():
    raw = _tiny(_idle4_authored(), mirror={"axis": "x", "pairs": "prefix"})
    with pytest.raises(ValueError, match="suffix"):
        ss._parse_template(raw, ss.plan_kind("idle4"))


def test_the_expanded_list_is_what_every_later_check_sees():
    """Expansion runs first, so the count and the cell-for-cell order check are
    made against all sixteen poses and not against the five directions on disk.

    The per-*point* checks are a belt rather than a live catch here, and it is
    worth knowing why: an x-reflection maps [0, 1] onto [0, 1] and renames
    joints bijectively, so a derived pose cannot be out of its cell or missing a
    segment point unless its source already was. What expansion-before-
    validation actually buys is that the grid checks see a whole grid.
    """
    geom = ss.plan_kind("idle4")
    raw = _tiny(_idle4_authored())
    assert len(ss._expand_mirrored(raw, geom)) == len(geom.cells)
    raw["poses"][0]["points"]["head"] = [0.5, 1.4]
    with pytest.raises(ValueError, match="outside its cell"):
        ss._parse_template(raw, geom)


# --- the guide --------------------------------------------------------------


@pytest.mark.parametrize("kind", ss.SHEET_TYPES)
def test_the_guide_draws_structure_in_every_cell(kind):
    geom = ss.geometry(kind)
    guide = ss.render_guide(geom, ss.load_guide_template(kind))
    assert guide.size == (ss.ATLAS_PX, ss.ATLAS_PX)
    assert guide.mode == "RGB"
    for cell in geom.cells:
        assert control.edge_fraction(guide.crop(cell.box)) > 0.001


def test_the_guide_render_is_deterministic():
    geom = ss.geometry("walk")
    template = ss.load_guide_template("walk")
    first = ss.render_guide(geom, template).tobytes()
    second = ss.render_guide(geom, template).tobytes()
    assert first == second


def test_the_guide_never_draws_outside_a_cell():
    # A stroke that crossed a cell boundary would ask the ControlNet for a limb
    # belonging to the neighbouring direction.
    geom = ss.geometry("walk")
    arr = np.asarray(ss.render_guide(geom, ss.load_guide_template("walk")).convert("L"))
    for col in range(1, geom.columns):
        assert not arr[:, col * geom.cell_w - 1 : col * geom.cell_w + 1].any()


def test_a_band_guide_is_band_sized_and_draws_in_every_frame():
    geom = ss.plan_kind("idle8")
    template = ss.load_guide_template("idle8")
    for band in geom.bands:
        guide = ss.render_band_guide(band, template)
        assert guide.size == band.size
        assert guide.mode == "RGB"
        for cell in band.cells:
            assert control.edge_fraction(guide.crop(cell.box)) > 0.001


def test_a_band_guide_never_draws_outside_a_frame():
    geom = ss.plan_kind("idle8")
    band = geom.bands[0]
    arr = np.asarray(
        ss.render_band_guide(band, ss.load_guide_template("idle8")).convert("L")
    )
    for col in range(1, band.columns):
        assert not arr[:, col * band.cell_px - 1 : col * band.cell_px + 1].any()


def test_the_mirrored_bands_are_reflections_of_the_ones_they_came_from():
    """The property the whole mirror buys, checked in pixels rather than in
    points: the ``right`` band is the ``left`` band flipped."""
    geom = ss.plan_kind("idle8")
    template = ss.load_guide_template("idle8")
    bands = {band.direction: band for band in geom.bands}
    left = np.asarray(ss.render_band_guide(bands["left"], template).convert("L"))
    right = np.asarray(ss.render_band_guide(bands["right"], template).convert("L"))
    for index in range(bands["left"].columns * bands["left"].rows):
        col, row = index % bands["left"].columns, index // bands["left"].columns
        x, y = col * bands["left"].cell_px, row * bands["left"].cell_px
        size = bands["left"].cell_px
        a = left[y : y + size, x : x + size]
        b = right[y : y + size, x : x + size]
        # Within a pixel of an exact flip: the strokes are rasterised at integer
        # coordinates, so a reflected line can land one pixel across.
        assert abs(int(a.sum()) - int(b.sum())) < a.sum() * 0.05


def test_a_band_guide_refuses_a_template_missing_one_of_its_frames():
    geom = ss.plan_kind("idle8")
    template = ss.load_guide_template("idle8")
    trimmed = ss.GuideTemplate(
        kind=template.kind,
        head_radius=template.head_radius,
        head_point=template.head_point,
        segments=template.segments,
        poses=tuple(p for p in template.poses if p.frame != 2),
    )
    with pytest.raises(ValueError, match="no pose for 'front'/2"):
        ss.render_band_guide(geom.bands[0], trimmed)


def test_a_band_splits_on_its_own_rectangles():
    geom = ss.plan_kind("walk8")
    band = geom.bands[0]
    image = Image.new("RGB", band.size, BG)
    parts = ss.split(image, band)
    assert len(parts) == len(band.cells)
    assert all(p.size == (band.cell_px, band.cell_px) for p in parts)


# --- splitting --------------------------------------------------------------


def test_split_uses_the_predetermined_rectangles():
    geom, atlas = _atlas("turnaround")
    parts = ss.split(atlas, geom)
    assert len(parts) == len(geom.cells)
    assert all(p.size == (geom.cell_w, geom.cell_h) for p in parts)


def test_a_misregistered_atlas_still_splits_on_the_grid():
    geom, atlas = _atlas("walk", offset=(7, -5))
    parts = ss.split(atlas, geom)
    assert [p.size for p in parts] == [(geom.cell_w, geom.cell_h)] * len(geom.cells)


# --- matting ----------------------------------------------------------------


def test_the_flood_fill_mattes_every_cell():
    geom, atlas = _atlas("turnaround")
    matted_atlas, took = ss.matte_cells(atlas, geom)
    assert took == (True,) * 4
    alpha = np.asarray(matted_atlas)[:, :, 3]
    for cell in geom.cells:
        sub = alpha[cell.y : cell.y + cell.h, cell.x : cell.x + cell.w]
        assert sub.max() == 255 and sub.min() == 0


def test_a_cell_the_fill_ate_is_left_opaque_and_reported():
    geom = ss.geometry("turnaround")
    # A uniform cell has no subject at all: the fill takes the whole rectangle
    # and the honest answer is "not matted", not a fully transparent cell.
    atlas = Image.new("RGB", (ss.ATLAS_PX, ss.ATLAS_PX), BG)
    matted_atlas, took = ss.matte_cells(atlas, geom)
    assert took == (False,) * 4
    assert np.asarray(matted_atlas)[:, :, 3].min() == 255


def test_an_unmatted_atlas_still_quantizes():
    from warlock.pipelines import pixelsheet

    geom = ss.geometry("turnaround")
    atlas = Image.new("RGB", (ss.ATLAS_PX, ss.ATLAS_PX), BG)
    matted_atlas, _ = ss.matte_cells(atlas, geom)
    out, palette = pixelsheet.quantize_shared(matted_atlas, 8)
    assert out.size == matted_atlas.size and palette


# --- the baseline -----------------------------------------------------------


def _bottoms(atlas, geom):
    alpha = np.asarray(atlas.convert("RGBA"))[:, :, 3]
    return [ss._cell_bounds(alpha, c) for c in geom.cells]


def test_the_baseline_is_the_median_of_the_cells():
    geom = ss.geometry("turnaround")
    atlas = Image.new("RGBA", (ss.ATLAS_PX, ss.ATLAS_PX), (0, 0, 0, 0))
    draw = ImageDraw.Draw(atlas)
    for cell, bottom in zip(geom.cells, (300, 400, 400, 460), strict=True):
        draw.rectangle(
            (cell.x + 100, cell.y + bottom - 200, cell.x + 200, cell.y + bottom),
            fill=(10, 20, 30, 255),
        )
    aligned = ss.baseline_align(atlas, geom)
    bottoms = [b[1] for b in _bottoms(aligned, geom)]
    assert bottoms == [400, 400, 400, 400]


def test_alignment_never_pushes_a_subject_off_its_cell():
    geom = ss.geometry("turnaround")
    atlas = Image.new("RGBA", (ss.ATLAS_PX, ss.ATLAS_PX), (0, 0, 0, 0))
    draw = ImageDraw.Draw(atlas)
    # Three short subjects near the floor and one that fills its cell: the tall
    # one cannot be moved to the median without losing its head.
    for cell, (top, bottom) in zip(
        geom.cells, ((480, 500), (480, 500), (480, 500), (0, 511)), strict=True
    ):
        draw.rectangle(
            (cell.x + 100, cell.y + top, cell.x + 200, cell.y + bottom),
            fill=(10, 20, 30, 255),
        )
    aligned = ss.baseline_align(atlas, geom)
    for cell, bound in zip(geom.cells, _bottoms(aligned, geom), strict=True):
        assert bound is not None
        assert 0 <= bound[0] <= bound[1] <= cell.h - 1
    # And the pixel count is preserved, i.e. nothing was shifted into oblivion.
    before = int((np.asarray(atlas)[:, :, 3] > 0).sum())
    after = int((np.asarray(aligned)[:, :, 3] > 0).sum())
    assert before == after


def test_an_empty_atlas_aligns_to_itself():
    geom = ss.geometry("walk")
    atlas = Image.new("RGBA", (ss.ATLAS_PX, ss.ATLAS_PX), (0, 0, 0, 0))
    assert ss.baseline_align(atlas, geom).tobytes() == atlas.tobytes()


# --- reduction --------------------------------------------------------------


@pytest.mark.parametrize("kind", ss.SHEET_TYPES)
@pytest.mark.parametrize("logical", (32, 48, 64))
def test_reduction_lands_every_cell_boundary_on_a_pixel(kind, logical):
    geom, atlas = _atlas(kind)
    matted, _ = ss.matte_cells(atlas, geom)
    out = ss.reduce_atlas(matted, geom, logical)
    assert out.size == (geom.columns * logical, geom.rows * logical)
    # 48 does not divide 512 or 256, which is exactly why this is one resize of
    # the whole atlas: the *columns* divide, so each output cell is square and
    # sits on an integer boundary regardless.
    for index, cell in enumerate(geom.cells):
        assert (index % geom.columns) * logical == cell.col * logical


def test_reduction_refuses_a_nonsense_size():
    geom, atlas = _atlas("turnaround")
    with pytest.raises(ValueError, match="at least 1"):
        ss.reduce_atlas(atlas, geom, 0)


def test_one_palette_is_shared_across_the_whole_atlas():
    from warlock.pipelines import pixelsheet

    geom = ss.geometry("turnaround")
    atlas = Image.new("RGB", (ss.ATLAS_PX, ss.ATLAS_PX), BG)
    draw = ImageDraw.Draw(atlas)
    for index, cell in enumerate(geom.cells):
        draw.rectangle(
            (cell.x + 100, cell.y + 100, cell.x + 400, cell.y + 400),
            fill=(20 + index * 40, 60, 90),
        )
    matted, _ = ss.matte_cells(atlas, geom)
    out, palette = pixelsheet.quantize_shared(ss.reduce_atlas(matted, geom, 64), 8)
    assert len(palette) <= 8
    rgba = np.asarray(out)
    colours = {tuple(int(c) for c in p) for p in rgba[rgba[:, :, 3] > 0][:, :3]}
    assert len(colours) <= 8


# --- warnings ---------------------------------------------------------------


def _codes(warnings):
    return {w["code"] for w in warnings}


def test_an_empty_cell_is_reported():
    geom, atlas = _atlas("turnaround")
    draw = ImageDraw.Draw(atlas)
    back = geom.cells[3]
    draw.rectangle(back.box, fill=BG)
    matted, took = ss.matte_cells(atlas, geom)
    warnings = ss.structural_warnings(matted, geom, took)
    assert "empty" in _codes(warnings) or "unmatted" in _codes(warnings)


def test_a_cell_running_off_its_edge_is_reported():
    geom = ss.geometry("turnaround")
    atlas = Image.new("RGBA", (ss.ATLAS_PX, ss.ATLAS_PX), (0, 0, 0, 0))
    draw = ImageDraw.Draw(atlas)
    for index, cell in enumerate(geom.cells):
        top = cell.y if index == 2 else cell.y + 100
        draw.rectangle(
            (cell.x + 100, top, cell.x + 300, cell.y + 400), fill=(1, 2, 3, 255)
        )
    warnings = ss.structural_warnings(atlas, geom, (True,) * 4)
    clipped = [w for w in warnings if w["code"] == "clipped"]
    assert [w["cell"] for w in clipped] == ["right"]
    assert "top" in clipped[0]["detail"]


def test_a_cell_far_off_the_sheets_size_is_reported():
    geom = ss.geometry("turnaround")
    atlas = Image.new("RGBA", (ss.ATLAS_PX, ss.ATLAS_PX), (0, 0, 0, 0))
    draw = ImageDraw.Draw(atlas)
    for index, cell in enumerate(geom.cells):
        side = 40 if index == 1 else 300
        draw.rectangle(
            (cell.x + 50, cell.y + 50, cell.x + 50 + side, cell.y + 50 + side),
            fill=(1, 2, 3, 255),
        )
    warnings = ss.structural_warnings(atlas, geom, (True,) * 4)
    assert [w["cell"] for w in warnings if w["code"] == "occupancy"] == ["left"]


def test_an_unmatted_cell_is_reported():
    geom, atlas = _atlas("turnaround")
    matted, _ = ss.matte_cells(atlas, geom)
    warnings = ss.structural_warnings(matted, geom, (True, True, False, True))
    unmatted = [w for w in warnings if w["code"] == "unmatted"]
    assert [w["cell"] for w in unmatted] == ["right"]


def test_every_warning_carries_a_known_code_and_a_sentence():
    geom, atlas = _atlas("turnaround", sizes={0: 40})
    matted, took = ss.matte_cells(atlas, geom)
    for warning in ss.structural_warnings(matted, geom, took):
        assert warning["code"] in ss.WARNING_CODES
        assert warning["detail"].endswith(".")
        assert warning["cell"] in ss.DIRECTION_ORDER


def test_an_unknown_warning_code_cannot_be_minted():
    with pytest.raises(ValueError, match="unknown sprite warning code"):
        ss._warning(ss.geometry("turnaround").cells[0], "vibes", "hm.")


# --- front preservation -----------------------------------------------------


def _report(**kw):
    base = dict(ok=True, bbox=(0, 0, 100, 200), touches=(), components=1)
    base.update(kw)
    return reference.Report(**base)


def test_a_clean_reference_of_matching_proportions_is_pasted_in():
    ok, why = ss.front_fits(_report(), _report(bbox=(0, 0, 105, 200)))
    assert ok is True
    assert "reference drawing itself" in why


@pytest.mark.parametrize(
    ("source", "expected"),
    (
        ({"ok": False}, "refused"),
        ({"touches": ("left",)}, "touches the edge"),
        ({"components": 2}, "more than one object"),
        ({"bbox": None}, "could not be measured"),
    ),
)
def test_a_reference_that_does_not_fit_is_refused_with_a_reason(source, expected):
    ok, why = ss.front_fits(_report(**source), _report())
    assert ok is False
    assert expected in why


def test_a_reference_of_the_wrong_shape_is_refused_with_the_ratio():
    ok, why = ss.front_fits(_report(bbox=(0, 0, 400, 200)), _report())
    assert ok is False
    assert "proportions" in why


def test_the_pasted_front_shares_the_sheets_palette():
    from warlock.pipelines import pixelsheet

    geom, atlas = _atlas("turnaround")
    matted, _ = ss.matte_cells(atlas, geom)
    subject = Image.new("RGBA", (120, 240), (0, 0, 0, 0))
    ImageDraw.Draw(subject).rectangle((0, 0, 119, 239), fill=(250, 10, 10, 255))
    pasted = ss.preserve_front(matted, geom, subject)
    # Before quantization, so the reference's red is one of the colours median
    # cut is choosing between rather than a stranger pasted in afterwards.
    out, palette = pixelsheet.quantize_shared(ss.reduce_atlas(pasted, geom, 64), 8)
    front = geom.cells[0]
    cell = np.asarray(out)[0:64, 0:64]
    assert front.name == "front"
    assert (cell[:, :, 3] > 0).any()
    colours = {f"#{r:02x}{g:02x}{b:02x}" for r, g, b in cell[cell[:, :, 3] > 0][:, :3]}
    assert colours <= set(palette)


def test_the_pasted_front_stands_on_the_shared_baseline():
    geom = ss.geometry("turnaround")
    atlas = Image.new("RGBA", (ss.ATLAS_PX, ss.ATLAS_PX), (0, 0, 0, 0))
    draw = ImageDraw.Draw(atlas)
    for cell in geom.cells:
        draw.rectangle(
            (cell.x + 200, cell.y + 200, cell.x + 300, cell.y + 460),
            fill=(1, 2, 3, 255),
        )
    subject = Image.new("RGBA", (50, 50), (9, 9, 9, 255))
    pasted = ss.preserve_front(atlas, geom, subject)
    bounds = _bottoms(pasted, geom)
    assert bounds[0][1] == bounds[1][1] == 460


def test_a_wide_reference_never_bleeds_into_the_neighbouring_cell():
    """front_fits gates on bbox *aspect*, which still admits a source up to a
    third wider than the generated front's -- so a height-matched paste can come
    out wider than the cell, and the overflow would land in the "left" view."""
    geom = ss.geometry("turnaround")
    atlas = Image.new("RGBA", (ss.ATLAS_PX, ss.ATLAS_PX), (0, 0, 0, 0))
    draw = ImageDraw.Draw(atlas)
    for cell in geom.cells:
        draw.rectangle(
            (cell.x + 200, cell.y + 200, cell.x + 300, cell.y + 460),
            fill=(1, 2, 3, 255),
        )
    before = np.asarray(atlas).copy()
    subject = Image.new("RGBA", (500, 50), (9, 9, 9, 255))
    pasted = np.asarray(ss.preserve_front(atlas, geom, subject))
    front = geom.cells[0]
    outside = np.ones(pasted.shape[:2], dtype=bool)
    outside[front.y : front.y + front.h, front.x : front.x + front.w] = False
    assert (pasted[outside] == before[outside]).all()
    # Trimmed, not vanished: the paste still spans the whole cell width.
    cell_alpha = pasted[front.y : front.y + front.h, front.x : front.x + front.w, 3]
    assert cell_alpha[:, 0].any() and cell_alpha[:, -1].any()


def test_front_preservation_is_a_turnaround_only_step():
    geom = ss.geometry("walk")
    with pytest.raises(ValueError, match="turnaround-only"):
        ss.preserve_front(
            Image.new("RGBA", (ss.ATLAS_PX, ss.ATLAS_PX)), geom, Image.new("RGBA", (4, 4))
        )


# --- the sidecar ------------------------------------------------------------


def test_the_sidecar_slices_the_reduced_png_standalone():
    geom, atlas = _atlas("walk")
    matted, _ = ss.matte_cells(atlas, geom)
    reduced = ss.reduce_atlas(matted, geom, 48)
    sidecar = ss.draft_sidecar(
        draft_id="abc123",
        source_job="job1",
        created=1.0,
        geom=geom,
        logical_size=48,
        colors=32,
        candidates=[],
        recipe={},
    )
    assert sidecar["version"] == ss.SPRITE_DRAFT_VERSION
    assert sidecar["cell_w"] == sidecar["cell_h"] == 48
    covered = np.zeros((reduced.height, reduced.width), dtype=int)
    for cell in sidecar["cells"]:
        assert cell["x"] + cell["w"] <= reduced.width
        assert cell["y"] + cell["h"] <= reduced.height
        covered[
            cell["y"] : cell["y"] + cell["h"], cell["x"] : cell["x"] + cell["w"]
        ] += 1
    assert covered.min() == 1 and covered.max() == 1


def test_the_sidecar_cells_are_in_timeline_order_with_their_yaws():
    geom = ss.geometry("walk")
    sidecar = ss.draft_sidecar(
        draft_id="abc123",
        source_job="job1",
        created=1.0,
        geom=geom,
        logical_size=64,
        colors=16,
        candidates=[],
        recipe={"base_model": "sdxl_cfg"},
    )
    assert [(c["name"], c["frame"]) for c in sidecar["cells"]] == [
        (c.name, c.frame) for c in geom.cells
    ]
    assert all(c["yaw"] == ss.DIRECTION_YAWS[c["name"]] for c in sidecar["cells"])
    assert sidecar["recipe"]["base_model"] == "sdxl_cfg"


# --- the assembler's palette, dither and outline -------------------------------
#
# ``queue._sprite_assemble`` rather than a ``spritesynth`` function, and here
# rather than in a worker test, because it is the *pure* half of this feature by
# every test in this file's standard: a Pillow-drawn atlas in, an atlas out, no
# GPU, no filesystem, no service. It is also where the three new options are
# actually spent -- the worker only reads params and hands them over -- so a
# wrong colour, a grown silhouette or a per-cell quantisation is decided here
# and nowhere else.

#: A designed ramp with four unmistakable entries. Deliberately nothing the
#: fixture atlas contains, so "these are the palette's colours" cannot pass by
#: the source having happened to be that colour already.
RAMP = ((16, 16, 32), (90, 40, 120), (200, 90, 60), (245, 240, 210))


def _colour_atlas(kind="walk"):
    """One atlas whose cells are deliberately *different* colours.

    Which is the fixture the per-cell-quantisation assertion needs: with every
    cell the same colour, a per-cell median cut and a shared palette are
    indistinguishable.
    """
    geom = ss.geometry(kind)
    im = Image.new("RGB", (ss.ATLAS_PX, ss.ATLAS_PX), BG)
    draw = ImageDraw.Draw(im)
    for index, cell in enumerate(geom.cells):
        pad = cell.w // 6
        draw.rectangle(
            (cell.x + pad, cell.y + pad, cell.x + cell.w - pad, cell.y + cell.h - pad),
            fill=(20 + index * 12, 60 + index * 9, 140 - index * 6),
        )
    return geom, im


def _assemble(geom, atlas, *, logical=32, colors=8, **options):
    from warlock import queue as queue_mod

    # ``source_rgba``/``source_report`` are read only by the turnaround's
    # front-cell paste, so the walk grid may pass None for both -- which keeps
    # this fixture about the palette rather than about front preservation.
    return queue_mod._sprite_assemble(
        atlas, geom, logical, colors, None, None, **options
    )


def _cell_colours(image, geom, logical):
    """Every occupied cell's colour set, keyed by cell index."""
    arr = np.asarray(image)
    out = {}
    for index, cell in enumerate(geom.cells):
        y, x = cell.row * logical, cell.col * logical
        block = arr[y : y + logical, x : x + logical]
        out[index] = {
            tuple(int(c) for c in p) for p in block[block[:, :, 3] > 0][:, :3]
        }
    return out


def test_a_designed_palette_is_the_only_colours_in_every_cell():
    """The assertion that catches per-cell quantisation, which is the artefact
    this whole path exists to avoid: not "the atlas has few colours" but "every
    cell drew from the same set", checked cell by cell."""
    geom, atlas = _colour_atlas()
    out, record = _assemble(geom, atlas, designed=RAMP, outline="none")

    allowed = set(RAMP)
    per_cell = _cell_colours(out, geom, 32)
    assert per_cell, "the fixture produced no occupied cells"
    for index, colours in per_cell.items():
        assert colours, f"cell {index} came out empty"
        assert colours <= allowed, f"cell {index} invented {colours - allowed}"
    assert record["palette_source"] == "designed"
    # And more than one entry is genuinely in use, or the subset assertion above
    # would pass on a sheet that had collapsed to a single colour.
    assert len(set().union(*per_cell.values())) > 1


def test_a_named_palette_does_not_cost_the_shared_across_cells_property():
    """``resolve_palette``'s claim, made concrete. A colour drawn in every cell
    must come out the same colour in every cell -- which is what a per-cell
    median cut breaks, and what naming a palette is often assumed to give up.
    """
    geom = ss.geometry("walk")
    atlas = Image.new("RGB", (ss.ATLAS_PX, ss.ATLAS_PX), BG)
    draw = ImageDraw.Draw(atlas)
    for index, cell in enumerate(geom.cells):
        pad = cell.w // 6
        # The same shirt colour in every cell, and one distinguishing stripe so
        # the cells are not literally identical images.
        draw.rectangle(
            (cell.x + pad, cell.y + pad, cell.x + cell.w - pad, cell.y + cell.h - pad),
            fill=(40, 110, 170),
        )
        draw.rectangle(
            (
                cell.x + pad,
                cell.y + pad,
                cell.x + pad + 24,
                cell.y + pad + 24 + index * 4,
            ),
            fill=(230, 40, 40),
        )
    for designed in (RAMP, ()):
        out, _record = _assemble(geom, atlas, designed=designed, outline="none")
        per_cell = _cell_colours(out, geom, 32)
        shared = set.intersection(*per_cell.values())
        assert shared, "no colour survived into every cell"
        for index, colours in per_cell.items():
            assert shared <= colours, f"cell {index} lost a shared colour"


def test_a_derived_palette_stays_within_the_colour_budget():
    geom, atlas = _colour_atlas()
    out, record = _assemble(geom, atlas, colors=8, outline="none")

    assert record["palette_source"] == "derived"
    arr = np.asarray(out)
    colours = {tuple(int(c) for c in p) for p in arr[arr[:, :, 3] > 0][:, :3]}
    assert 0 < len(colours) <= 8
    assert record["palette"] == sorted(f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colours)


def test_an_inner_outline_cannot_change_the_silhouette_and_is_the_default():
    """``pixelize.OUTLINE_MODES``' own distinction, and the reason this path
    defaults to ``inner`` where Troupe defaults to ``outer``: a synthesised cell
    has no guaranteed margin, and ``outer`` grows into whatever is at the edge.
    """
    geom, atlas = _colour_atlas()
    plain, _ = _assemble(geom, atlas, designed=RAMP, outline="none")
    inner, _ = _assemble(geom, atlas, designed=RAMP, outline="inner")
    outer, _ = _assemble(geom, atlas, designed=RAMP, outline="outer")
    default, _ = _assemble(geom, atlas, designed=RAMP)

    def alpha(image):
        return np.asarray(image)[:, :, 3]

    assert inner.size == plain.size == outer.size
    # Silhouette, size and trim are all one fact about the alpha plane, and
    # ``inner`` only ever recolours pixels that were already opaque.
    assert np.array_equal(alpha(inner), alpha(plain))
    assert not np.array_equal(alpha(outer), alpha(plain))
    assert int(alpha(outer).sum()) > int(alpha(plain).sum())
    # It did something -- an outline that changed no pixel would pass the
    # assertion above trivially.
    assert inner.tobytes() != plain.tobytes()
    # The default is ``inner``, stated twice: once as the constant a door reads
    # and once as the bytes the assembler actually draws with no option given.
    assert ss.DEFAULT_SPRITE_OUTLINE == "inner"
    assert default.tobytes() == inner.tobytes()
    assert default.tobytes() != outer.tobytes()


def test_dither_is_off_unless_asked_for():
    geom, atlas = _colour_atlas()
    plain, _ = _assemble(geom, atlas, designed=RAMP, outline="none")
    dithered, _ = _assemble(geom, atlas, designed=RAMP, outline="none", dither=True)

    assert dithered.tobytes() != plain.tobytes()
    # Still only the palette's colours: a dither trades resolution for shades it
    # already has, it does not mix new ones.
    arr = np.asarray(dithered)
    colours = {tuple(int(c) for c in p) for p in arr[arr[:, :, 3] > 0][:, :3]}
    assert colours <= set(RAMP)


def test_dither_with_no_designed_palette_still_changes_the_sheet():
    """The claim the 2D form's Dither checkbox rests on, pinned on this arm.

    ``asset2d`` is the one pixel path where dither genuinely needs a palette --
    it calls ``map_palette`` only when there is one, and records
    ``bool(opts.dither and opts.palette)`` for exactly that reason. This path
    does not work that way: ``resolve_palette`` always returns a table, derived
    by median cut when nothing was named, and ``pixelize_atlas`` maps through it
    either way. So gating the checkbox on a palette here -- copying the rule
    from the pane that owns the other path -- would hide a setting that does
    something. ``tilesheet.quantize_tiles`` has the same property and its own
    test beside it.
    """
    geom, atlas = _colour_atlas()
    plain, plain_record = _assemble(geom, atlas, outline="none")
    dithered, record = _assemble(geom, atlas, outline="none", dither=True)

    assert plain_record["palette_source"] == record["palette_source"] == "derived"
    assert dithered.tobytes() != plain.tobytes()


@pytest.mark.parametrize("options", ({}, {"designed": RAMP, "dither": True}))
def test_the_same_inputs_twice_are_byte_identical(options):
    """No RNG anywhere on this path -- the bar ``tests/troupe/test_pixelize.py``
    sets for the pixeliser, held one caller up."""
    geom, atlas = _colour_atlas()
    first, first_record = _assemble(geom, atlas, **options)
    second, second_record = _assemble(geom, atlas, **options)

    assert first.tobytes() == second.tobytes()
    assert first_record == second_record


@pytest.mark.parametrize(
    ("logical", "exact"),
    # 1024 / (4 * 32) = 8 and 1024 / (4 * 64) = 4, so the walk grid supersamples
    # at 32 and 64 -- which is the whole reason SPRITE_DRAFT_VERSION moved to 2.
    # Nothing divides 1024 into 4 * 48, so 48 keeps the NEAREST fallback that
    # ``reduce_atlas`` always was.
    [(32, True), (48, False), (64, True)],
)
def test_the_report_names_the_reduction_that_actually_happened(logical, exact):
    """``pixelize_atlas`` sees an atlas already at the target and answers
    ``True`` unconditionally, so this field has to be measured against the real
    reduction -- the same correction ``_q_troupe`` makes one path along."""
    geom, atlas = _colour_atlas()
    out, record = _assemble(geom, atlas, logical=logical, designed=RAMP)

    assert out.size == (geom.columns * logical, geom.rows * logical)
    assert record["exact_stride"] is exact


def test_an_atlas_with_nothing_in_it_is_named_rather_than_published(monkeypatch):
    """Asked of both palette branches. ``quantize_shared`` used to be the only
    thing standing here, and it does not run when a palette was named -- so a
    designed request would have published two blank PNGs as a finished pair.

    Reached by replacing ``matte_cells``, because today it cannot be reached any
    other way and that is worth writing down: a cell whose fill covers less than
    ``MIN_MATTE_FRACTION`` is left **opaque** rather than published with a hole,
    and a cell that clears the floor has opaque pixels by definition -- so the
    two branches of that floor between them guarantee an opaque atlas. The guard
    is therefore about the day that floor changes, and this is the only way to
    ask it a question now.
    """
    geom = ss.geometry("walk")
    monkeypatch.setattr(
        ss,
        "matte_cells",
        lambda atlas, geom: (
            Image.new("RGBA", (ss.ATLAS_PX, ss.ATLAS_PX), (0, 0, 0, 0)),
            (False,) * len(geom.cells),
        ),
    )
    blank = Image.new("RGB", (ss.ATLAS_PX, ss.ATLAS_PX), (0, 0, 0))
    for designed in ((), RAMP):
        with pytest.raises(RuntimeError, match="empty in every cell"):
            _assemble(geom, blank, designed=designed)
