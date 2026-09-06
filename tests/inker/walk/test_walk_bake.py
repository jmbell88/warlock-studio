"""The cycle landed as a document: what a user gets when they press Bake.

The claims here are about *editability*, which is the brief's promise for the
bake and the thing that separates it from an exported sheet: real cels on real
tracks, none of them shared, in a document that opens the way an imported sheet
opens -- clean, unsaved, and with nothing to undo.
"""

from __future__ import annotations

import numpy as np
import pytest
from _figure import SIZE, figure

from warlock.studio.inker import ora, sheetout
from warlock.studio.inker.walk import bake, gait
from warlock.studio.inker.walk import rig as R


def _settings(rig: R.Rig, **changes: float) -> gait.WalkSettings:
    return gait.defaults_for(rig).replaced(**changes)


def _baked(**changes: float):
    rig = figure()
    return bake.document(rig, _settings(rig, **changes), SIZE)


def test_the_baked_document_has_one_track_per_assigned_part():
    doc = _baked()
    assert len(doc.anim.tracks) == len(R.PART_NAMES)
    assert [track.name for track in doc.anim.tracks] == [
        R.label(name) for name in R.PART_NAMES
    ]


def test_an_unassigned_limb_gets_no_track_at_all():
    """A figure drawn with one arm hidden behind it is a legitimate drawing, and
    an empty track for the arm it does not have is clutter in the timeline."""
    rig = figure(far=False)
    for limb in ("arm", "leg"):
        rig = R.copy_near_to_far(rig, limb)
    for name in ("far_upper_arm", "far_lower_arm", "far_hand"):
        rig = R.set_part(rig, name, R.Part())
    doc = bake.document(rig, _settings(rig), SIZE)
    names = {track.name for track in doc.anim.tracks}
    assert R.label("far_thigh") in names
    assert R.label("far_upper_arm") not in names


def test_the_tracks_are_laid_down_in_draw_order():
    """Bottom-first, far limbs behind the body and near ones in front, which for
    a side view is the whole of the depth problem."""
    doc = _baked()
    names = [track.name for track in doc.anim.tracks]
    assert names.index(R.label("far_thigh")) < names.index(R.label("torso"))
    assert names.index(R.label("torso")) < names.index(R.label("near_thigh"))


def test_there_are_eight_frames_at_the_duration_that_was_asked_for():
    doc = _baked(duration_ms=80)
    assert len(doc.anim.frames) == gait.WALK_FRAMES
    assert {frame.duration_ms for frame in doc.anim.frames} == {80}


def test_every_cel_is_its_own_layer():
    """**Never linked.** Two slots holding one object is how Inker expresses a
    link, so a shared cel would make a stroke on frame three appear on frame
    five -- the opposite of the independently editable cels the bake promises."""
    doc = _baked()
    cels = list(doc.anim.cels.values())
    assert len({id(cel) for cel in cels}) == len(cels)
    for track in doc.anim.tracks:
        for frame in doc.anim.frames:
            assert not doc.anim.is_linked(track.uid, frame.uid)


def test_the_walk_tag_covers_the_cycle_and_loops():
    doc = _baked()
    assert len(doc.anim.tags) == 1
    tag = doc.anim.tags[0]
    assert (tag.name, tag.start, tag.end, tag.loop) == (
        bake.TAG_NAME,
        0,
        gait.WALK_FRAMES - 1,
        True,
    )


def test_the_baked_document_opens_clean():
    """``sheetin``'s rule: a document a generator hands you has nothing behind
    it, so the first Ctrl+Z does nothing rather than dismantling a half-built
    walk. Building it directly instead of through undoable ops is what buys
    this, and it is the reason the bake does not go through ``add_frame``."""
    doc = _baked()
    assert doc.history.head == 0
    assert not doc.history.can_undo
    assert doc.path is None
    assert doc.file_format == "ora"


def test_the_cels_are_canvas_sized():
    doc = _baked()
    for cel in doc.anim.cels.values():
        assert cel.pixels.shape == (SIZE[1], SIZE[0], 4)


def test_the_baked_document_survives_an_ora_round_trip(tmp_path):
    """The bake writes an ordinary document and nothing else, so the ordinary
    writer has to be able to hold all of it -- tracks, frames, durations, the
    tag, and the pixels."""
    doc = _baked(duration_ms=120)
    path = tmp_path / "walk.ora"
    ora.write_ora(doc, path)
    reopened = ora.read_ora(path)
    assert len(reopened.anim.tracks) == len(doc.anim.tracks)
    assert len(reopened.anim.frames) == len(doc.anim.frames)
    assert [f.duration_ms for f in reopened.anim.frames] == [
        f.duration_ms for f in doc.anim.frames
    ]
    assert [(t.name, t.start, t.end, t.loop) for t in reopened.anim.tags] == [
        (t.name, t.start, t.end, t.loop) for t in doc.anim.tags
    ]
    for index in range(len(doc.anim.frames)):
        doc.set_current_frame(index)
        reopened.set_current_frame(index)
        assert np.array_equal(doc.flatten(), reopened.flatten())


def test_the_sheet_export_carries_the_frame_durations_and_the_tag():
    doc = _baked(duration_ms=90)
    _image, _plan, sidecar = sheetout.from_document(doc)
    animation = sidecar["animation"]
    assert [entry["duration_ms"] for entry in animation["frames"]] == [90] * gait.WALK_FRAMES
    assert [tag["name"] for tag in animation["tags"]] == [bake.TAG_NAME]
    assert animation["tags"][0]["loop"] is True


def test_baking_an_incomplete_rig_is_refused_by_name():
    rig = figure()
    rig = R.set_part(rig, "torso", R.Part())
    with pytest.raises(ValueError, match="torso"):
        bake.document(rig, _settings(rig), SIZE)


def test_baking_a_canvas_that_is_too_large_is_refused():
    """Fourteen tracks times eight frames is the document's inherent weight, not
    something the bake could optimise away, so the refusal is the honest answer
    rather than a slow one."""
    rig = figure()
    huge = (2048, 2048)
    assert bake.too_large(huge)
    with pytest.raises(ValueError, match="too large"):
        bake.document(rig, _settings(rig), huge)
    assert not bake.too_large(SIZE)


def test_the_preview_and_the_bake_show_the_same_picture():
    """They are folded off the same placed planes on purpose. A preview that
    could disagree with the bake is a preview nobody can trust."""
    rig = figure()
    settings = _settings(rig)
    doc = bake.document(rig, settings, SIZE)
    for index, preview in enumerate(bake.composite_frames(rig, settings, SIZE)):
        doc.set_current_frame(index)
        assert np.array_equal(doc.flatten(), preview)
