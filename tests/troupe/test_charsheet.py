"""Clip expansion, the character-sheet plan, and the sidecar's timing block."""

from __future__ import annotations

from pathlib import Path

import pytest

from warlock import rigging
from warlock.pipelines import charsheet as cs
from warlock.pipelines import sheet as sheetlib

IDENT = [0.0, 0.0, 0.0, 1.0]


def _pose(name, z=0.0, root=None):
    row = {"id": name, "name": name, "bones": {"hips": [0.0, 0.0, z, (1 - z * z) ** 0.5]}}
    if root:
        row["root_translation"] = root
    return row


# --- multi-key clips ---------------------------------------------------------


def test_a_cyclic_clip_never_lands_on_a_key_twice():
    """The seam rule, generalised: each segment stops short of its far key
    because that key is the next segment's frame 0."""
    keys = [_pose("a"), _pose("b", 0.2), _pose("c", 0.4), _pose("d", 0.2)]
    out = sheetlib.interpolate_clip(keys, [2, 2, 2, 2], closed=True)
    assert len(out) == 8
    assert [r["frame"] for r in out] == list(range(8))
    assert out[0]["bones"]["hips"] == keys[0]["bones"]["hips"]
    assert out[2]["bones"]["hips"] == keys[1]["bones"]["hips"]


def test_a_one_shot_lands_exactly_on_its_last_key():
    keys = [_pose("a"), _pose("b", 0.3), _pose("c", 0.5)]
    out = sheetlib.interpolate_clip(keys, [2, 2], closed=False)
    assert len(out) == 5
    assert out[-1]["bones"]["hips"] == pytest.approx(keys[-1]["bones"]["hips"])


def test_the_old_two_key_call_is_the_new_one_with_a_single_segment():
    a, b = _pose("a"), _pose("b", 0.4)
    assert sheetlib.interpolate(a, b, 6) == sheetlib.interpolate_clip(
        [a, b], [6, 6], closed=True
    )[:6]


def test_root_translation_is_interpolated_rather_than_refused():
    """Gap 1's other half: a bob is root translation, and it used to be a
    refusal by name."""
    keys = [_pose("down", root=[0.0, 0.0, -0.02]), _pose("up", root=[0.0, 0.0, 0.02])]
    out = sheetlib.interpolate_clip(keys, [2, 2], closed=True)
    assert [round(r["root_translation"][2], 4) for r in out] == [-0.02, 0.0, 0.02, 0.0]


def test_a_clip_of_offsetless_poses_records_no_offset_at_all():
    """Byte-identical to what this produced before offsets were interpolated,
    which is what keeps ``_sheet_root_offsets`` short-circuiting on them."""
    out = sheetlib.interpolate_clip([_pose("a"), _pose("b", 0.2)], [2, 2], closed=True)
    assert all("root_translation" not in r for r in out)


def test_easing_reshapes_the_spacing_without_moving_the_keys():
    keys = [_pose("a"), _pose("b", 0.5)]
    linear = sheetlib.interpolate_clip(keys, [4, 4], closed=True, easing="linear")
    eased = sheetlib.interpolate_clip(keys, [4, 4], closed=True, easing="ease")
    assert linear[0]["bones"] == eased[0]["bones"]
    assert linear[1]["bones"] != eased[1]["bones"]


@pytest.mark.parametrize(
    "args, kwargs, message",
    [
        (([_pose("a")], [2]), {}, "at least two keyframes"),
        (([_pose("a"), _pose("b"), _pose("c")], [2]), {}, "takes 2 segment"),
        (([_pose("a"), _pose("b")], [2, 2, 2]), {"closed": True}, "takes 2 segment"),
        (([_pose("a"), _pose("b")], [0, 1]), {"closed": True}, "at least one frame"),
        (([_pose("a"), _pose("b")], [30, 30]), {"closed": True}, "must be 1-"),
    ],
)
def test_a_clip_that_does_not_add_up_is_refused_by_name(args, kwargs, message):
    with pytest.raises(ValueError, match=message):
        sheetlib.interpolate_clip(*args, **kwargs)


def test_an_unknown_easing_is_refused():
    with pytest.raises(ValueError, match="easing must be one of"):
        sheetlib.interpolate_clip([_pose("a"), _pose("b")], [2], easing="bounce")


# --- the shipped clip library ------------------------------------------------


def test_the_humanoid_library_carries_the_twenty_two_keyframes():
    library = rigging.clip_library("humanoid")
    assert len(library["poses"]) == 22
    assert [c["name"] for c in library["clips"]] == [a[0] for a in cs.ANIMATIONS]


def test_every_shipped_clip_expands_to_exactly_its_frame_count():
    """The load-bearing one: a clip that expands to seven frames would lay a
    frame of some other animation into the walk's eighth cell."""
    library = rigging.clip_library("humanoid")
    wanted = {name: frames for name, frames, _l, _ms in cs.ANIMATIONS}
    for clip in library["clips"]:
        keys = rigging.clip_keys("humanoid", clip["name"])
        out = sheetlib.interpolate_clip(
            keys, clip["segments"], closed=clip["closed"], easing=clip["easing"]
        )
        assert len(out) == wanted[clip["name"]], clip["name"]


def test_the_cyclic_clips_are_the_looping_ones_and_the_one_shots_are_not():
    library = rigging.clip_library("humanoid")
    closed = {c["name"]: c["closed"] for c in library["clips"]}
    assert closed == {name: loop for name, _f, loop, _ms in cs.ANIMATIONS}


def test_the_walk_and_run_carry_a_vertical_bob():
    """The thing the root-translation refusal was costing."""
    for name in ("walk", "run"):
        keys = rigging.clip_keys("humanoid", name)
        heights = [k.get("root_translation", [0, 0, 0])[2] for k in keys]
        assert max(heights) > 0 > min(heights), name


def test_a_clip_naming_a_pose_the_file_lacks_costs_that_library(tmp_path, caplog):
    (tmp_path / "broken.json").write_text(
        '{"poses": [{"name": "a", "bones": {}}], '
        '"clips": [{"name": "c", "keys": ["a", "ghost"], "segments": [1, 1]}]}',
        encoding="utf-8",
    )
    assert rigging._load_clip_library(tmp_path) == {}


def test_a_template_with_no_clips_gets_an_empty_library():
    assert rigging.clip_library("fish") == {"poses": {}, "clips": [], "space": "node"}


# --- the plan ----------------------------------------------------------------


def _records():
    return {
        name: sheetlib.interpolate_clip(
            rigging.clip_keys("humanoid", name),
            next(c for c in rigging.clip_library("humanoid")["clips"] if c["name"] == name)[
                "segments"
            ],
            closed=loop,
            easing="linear",
        )
        for name, _frames, loop, _ms in cs.ANIMATIONS
    }


def test_the_plan_is_the_atlas_the_program_committed_to():
    layout = cs.plan(_records(), frame_size=128)
    assert (layout.columns, layout.rows) == (8, 32)
    assert (layout.width, layout.height) == (1024, 4096)
    assert len(layout.cells) == 256


def test_every_cell_lands_on_its_own_rectangle():
    layout = cs.plan(_records(), frame_size=64)
    placed = {(c.x, c.y) for c in layout.cells}
    assert len(placed) == len(layout.cells)
    for cell in layout.cells:
        assert cell.x == cell.column * 64 and cell.y == cell.row * 64


def test_a_cell_carries_the_pose_of_its_own_frame():
    layout = cs.plan(_records(), frame_size=64)
    records = _records()
    for cell, table in zip(layout.cells, cs.frame_table(), strict=True):
        assert cell.pose_name == table.animation
        assert cell.yaw == table.yaw
        assert cell.frame == table.frame
        assert cell.pose == records[table.animation][table.frame]["id"]


def test_a_clip_that_does_not_fill_the_table_is_refused_by_name():
    records = _records()
    records["walk"] = records["walk"][:7]
    with pytest.raises(ValueError, match="walk clip expands to 7 frames"):
        cs.plan(records)


def test_an_animation_the_table_does_not_have_is_refused():
    records = _records()
    records["cartwheel"] = []
    with pytest.raises(ValueError, match="not Troupe animations"):
        cs.plan(records)


def test_an_atlas_over_the_texture_limit_is_refused_before_anything_renders():
    with pytest.raises(ValueError, match="the limit is 8192"):
        cs.plan(_records(), frame_size=512)


# --- the sidecar's animation block -------------------------------------------


def test_every_cell_gets_a_duration():
    block = cs.animation_block()
    assert len(block["frames"]) == 256
    assert [f["cell_index"] for f in block["frames"]] == list(range(256))
    assert all(f["duration_ms"] > 0 for f in block["frames"])


def test_a_frame_carries_its_own_animation_speed():
    block = cs.animation_block()
    by_cell = {f["cell_index"]: f["duration_ms"] for f in block["frames"]}
    for animation, _direction, start, _end, _loop in cs.spans():
        wanted = next(a[3] for a in cs.ANIMATIONS if a[0] == animation)
        assert by_cell[start] == wanted


def test_there_is_one_tag_per_animation_and_direction():
    tags = cs.animation_block()["tags"]
    assert len(tags) == len(cs.ANIMATIONS) * len(cs.DIRECTIONS)
    assert tags[0]["name"] == "idle_front"
    assert {t["name"] for t in tags} == {
        f"{a[0]}_{d[0]}" for a in cs.ANIMATIONS for d in cs.DIRECTIONS
    }


def test_a_one_shot_tag_plays_once_and_a_cycle_does_not_say_so():
    tags = {t["name"]: t for t in cs.animation_block()["tags"]}
    assert tags["attack_front"]["loop"] is False
    assert tags["attack_front"]["repeat"] == 1
    assert tags["walk_front"]["loop"] is True
    assert "repeat" not in tags["walk_front"]


def test_the_block_goes_into_a_sidecar_unchanged():
    layout = cs.plan(_records(), frame_size=64)
    meta = sheetlib.sidecar(
        layout,
        sheet_id="a" * 12,
        source_job="b" * 12,
        image="sheet.png",
        created=0.0,
        animation=cs.animation_block(),
    )
    assert meta["animation"]["tags"][0]["name"] == "idle_front"
    assert len(meta["animation"]["frames"]) == len(meta["cells"])


# --- the handoff into Inker --------------------------------------------------


def test_a_rendered_sheet_opens_in_inker_with_its_tags_and_timing():
    """Gap 4's other end: the sidecar's ``animation`` block goes straight into
    a document, so a character arrives in the editor already tagged per
    direction and already playing at the speed it was rendered for."""
    import numpy as np

    from warlock.studio.inker import sheetin

    layout = cs.plan(_records(), frame_size=16)
    block = cs.animation_block()
    atlas = np.zeros((layout.height, layout.width, 4), dtype=np.uint8)
    doc = sheetin.document_from_sheet(
        atlas, [c.as_dict(16) for c in layout.cells], block
    )
    assert len(doc.anim.frames) == 256
    # No DirectionalLayout: these cells are not the four named directions of a
    # fixed grid, and the tags carry the directions instead.
    assert doc.anim.layout is None
    tags = {t.name: t for t in doc.anim.tags}
    assert tags["walk_left"].loop is True
    assert (tags["walk_left"].start, tags["walk_left"].end) == (
        cs.spans()[10][2],
        cs.spans()[10][3],
    )
    assert tags["jump_back"].loop is False and tags["jump_back"].repeat == 1
    idle_ms = next(a[3] for a in cs.ANIMATIONS if a[0] == "idle")
    run_start = next(s[2] for s in cs.spans() if s[0] == "run")
    run_ms = next(a[3] for a in cs.ANIMATIONS if a[0] == "run")
    assert doc.anim.frames[0].duration_ms == idle_ms
    assert doc.anim.frames[run_start].duration_ms == run_ms


def test_the_general_tag_builder_still_produces_the_walk_sheet_tags():
    from warlock.studio.inker import sheetin

    tags = sheetin.walk_tags()
    assert [t.name for t in tags] == [
        "walk_front", "walk_left", "walk_right", "walk_back"
    ]
    assert [(t.start, t.end) for t in tags] == [(0, 3), (4, 7), (8, 11), (12, 15)]


def test_a_tag_that_runs_backwards_is_refused():
    from warlock.studio.inker import sheetin

    with pytest.raises(ValueError, match="covers frames 5-2"):
        sheetin.span_tags([{"name": "bad", "start": 5, "end": 2}])


# --- pose space --------------------------------------------------------------


def test_a_clip_records_the_frame_its_rotations_are_in():
    keys = [_pose("a"), _pose("b", 0.3)]
    node = sheetlib.interpolate_clip(keys, [2, 2], closed=True)
    delta = sheetlib.interpolate_clip(keys, [2, 2], closed=True, space="delta")
    # Absent on the default, so a record from the two-key door is byte-identical
    # to the one it always was.
    assert all("space" not in r for r in node)
    assert all(r["space"] == "delta" for r in delta)
    assert [r["bones"] for r in node] == [r["bones"] for r in delta]


def test_an_unknown_pose_space_is_refused():
    with pytest.raises(ValueError, match="space must be one of"):
        sheetlib.interpolate_clip([_pose("a"), _pose("b")], [2], space="world")


def test_the_two_modules_name_the_same_pose_spaces():
    """``sheet`` decides what a clip record says and ``blender_worker`` decides
    what it means; a spelling in one and not the other is a clip that silently
    applies in the wrong frame."""
    from warlock.pipelines import blender_worker

    assert sheetlib.POSE_SPACES == blender_worker.POSE_SPACES


def test_the_shipped_clips_are_authored_as_deltas_from_rest():
    """The finding that made them work at all. A node-local value bakes in the
    rest orientation of the skeleton it was authored against, so the same
    numbers on a rig whose joints were measured rather than fitted produce a
    different -- and in practice broken -- pose."""
    library = rigging.clip_library("humanoid")
    assert library["space"] == "delta"
    assert {c["space"] for c in library["clips"]} == {"delta"}


def test_a_clip_library_that_says_nothing_is_read_as_node_local(tmp_path):
    (tmp_path / "old.json").write_text(
        '{"poses": [{"name": "a", "bones": {}}, {"name": "b", "bones": {}}], '
        '"clips": [{"name": "c", "keys": ["a", "b"], "segments": [1, 1], '
        '"closed": true}]}',
        encoding="utf-8",
    )
    library = rigging._load_clip_library(tmp_path)["old"]
    assert library["space"] == "node"
    assert library["clips"][0]["space"] == "node"


def test_a_delta_clip_reaches_the_worker_as_a_cell_that_says_so():
    """The end of the thread: ``_q_rig`` puts ``pose_space`` on the cell only
    where the record carries one, so every pose row is the cell it always was."""
    import warlock._q_rig as q_rig

    source = Path(q_rig.__file__).read_text(encoding="utf-8")
    assert '"pose_space"] = space' in source
    assert 'r["space"] for r in records if r.get("space")' in source


# --- the pivot's unit --------------------------------------------------------


def test_a_pivot_projected_at_the_render_size_lands_inside_the_cell():
    """The sidecar's pivot is cell-relative, and Troupe renders at 512 while
    packing at the logical size -- so the projected number has to be converted
    or it names a point far outside the cell it belongs to."""
    # Centre-bottom of the render, which is where a grounded subject's origin
    # projects: it must come back as centre-bottom of the *cell*.
    half = cs.RENDER_SIZE / 2.0
    assert cs.pivot_in_cell((half, float(cs.RENDER_SIZE)), 32) == (16.0, 32.0)
    assert cs.pivot_in_cell((half, float(cs.RENDER_SIZE)), 64) == (32.0, 64.0)


def test_every_supported_size_keeps_the_pivot_within_its_cell():
    """The failure in its own terms: unconverted, a 32px cell recorded a pivot
    near (256, 470) -- sixteen cells below the sprite."""
    for size in cs.SIZES:
        x, y = cs.pivot_in_cell((256.0, 470.0), size)
        assert 0.0 <= x <= size, size
        assert 0.0 <= y <= size, size


def test_no_pivot_stays_no_pivot():
    """``sidecar`` falls back to centre-bottom when there is none, so None has
    to survive the conversion rather than becoming (0, 0)."""
    assert cs.pivot_in_cell(None, 32) is None
