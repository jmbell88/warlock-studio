"""An effect on a document: insert, regenerate, the painted-cel rule, undo as
one step, and what survives a save to each format."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from warlock.studio import inker
from warlock.studio.inker import _doc_flourish, flourish, ora
from warlock.studio.inker.flourish import bake as B
from warlock.studio.inker.flourish import presets

_PUFF = presets.load("smoke_puff")


def _recipe(name: str = "smoke_puff", **over):
    """One loaded preset, so every variant shares its layer uids -- the
    situation a regenerate is in: the recipe edited, not reloaded."""
    rec = _PUFF if name == "smoke_puff" else presets.load(name)
    return dataclasses.replace(rec, width=32, height=32, supersample=2, **over)


def _small_layer_recipe():
    """A two-layer painterly recipe whose geometry fits a 32px canvas."""
    rec = presets.load("sword_impact")
    return dataclasses.replace(rec, width=32, height=32, supersample=2)


def test_insert_makes_a_group_of_tracks_with_a_tag_per_phase_in_one_step():
    doc = inker.Document.blank(32, 32)
    baked = B.bake(_small_layer_recipe())
    group = doc.insert_flourish(baked)
    assert doc.anim is not None
    assert len(doc.anim.frames) == baked.frame_count
    assert [t.name for t in doc.anim.tags] == [p.name for p in baked.recipe.phases]
    state = doc.flourish_state(group)
    assert state is not None
    members = [uid for uid, g in doc.group_of.items() if g == group]
    assert set(members) == set(state.tracks.values())
    assert len(members) == len(baked.recipe.layers)
    assert doc.groups[group].name == baked.recipe.name
    # One undo step back to a still document with no effect.
    doc.history.undo(doc)
    assert doc.anim is None
    assert doc.flourish == {} and doc.groups == {}
    doc.history.redo(doc)
    assert doc.flourish_state(group) is not None
    assert len(doc.anim.tracks) == 1 + len(baked.recipe.layers)


def test_the_effect_lands_centred_on_a_larger_document():
    doc = inker.Document.blank(64, 48)
    baked = B.bake(_recipe())
    group = doc.insert_flourish(baked)
    state = doc.flourish_state(group)
    assert state.offset == (16, 8)
    track_uid = next(iter(state.tracks.values()))
    frame = doc.anim.frames[6]
    cel = doc.anim.cels[(track_uid, frame.uid)]
    assert cel.pixels.shape == (48, 64, 4)
    # Nothing outside the recipe's canvas.
    assert not cel.pixels[:8].any() and not cel.pixels[:, :16].any()


def test_a_pixel_bake_is_one_track():
    doc = inker.Document.blank(32, 32)
    baked = B.bake(_recipe(mode="pixel", colors=6))
    group = doc.insert_flourish(baked)
    state = doc.flourish_state(group)
    assert list(state.tracks) == [_doc_flourish.COMPOSITE]


def test_the_active_layer_finds_its_effect():
    doc = inker.Document.blank(32, 32)
    baked = B.bake(_small_layer_recipe())
    group = doc.insert_flourish(baked)
    assert doc.flourish_group_of_active() == group
    doc.set_active_layer(0)  # the original background
    assert doc.flourish_group_of_active() is None


def test_regenerate_with_the_same_recipe_changes_nothing_but_is_one_step():
    doc = inker.Document.blank(32, 32)
    baked = B.bake(_recipe())
    group = doc.insert_flourish(baked)
    head = doc.history.head
    counts = doc.apply_flourish(group, baked)
    assert counts.taken == 0 and counts.conflicts == 0
    assert counts.agreed == baked.frame_count
    assert doc.history.head == head + 1
    doc.history.undo(doc)
    assert doc.history.head == head


def test_regenerate_takes_untouched_cels_and_keeps_painted_ones():
    doc = inker.Document.blank(32, 32)
    first = B.bake(_recipe(seed=1))
    group = doc.insert_flourish(first)
    state = doc.flourish_state(group)
    track_uid = next(iter(state.tracks.values()))
    # Paint on frame 3.
    frame = doc.anim.frames[3]
    painted = doc.anim.cels[(track_uid, frame.uid)]
    painted.pixels[0, 0] = (255, 0, 255, 255)
    second = B.bake(_recipe(seed=2))
    counts = doc.apply_flourish(group, second)
    assert counts.conflicts == 1 and counts.kept == 1
    assert counts.taken >= 1
    assert doc.flourish_conflicts(group) == [3]
    # The paint stands.
    assert tuple(doc.anim.cels[(track_uid, frame.uid)].pixels[0, 0]) == (255, 0, 255, 255)
    # Another untouched frame took the new render.
    other = doc.anim.frames[6]
    got = doc.anim.cels[(track_uid, other.uid)].pixels
    expected = _doc_flourish._place(second.flat()[6], (32, 32), (0, 0))
    assert np.array_equal(got, expected)
    # One Ctrl+Z puts the pixels and the record back together.
    doc.history.undo(doc)
    assert doc.flourish_conflicts(group) == []
    assert np.array_equal(
        doc.anim.cels[(track_uid, other.uid)].pixels,
        _doc_flourish._place(first.flat()[6], (32, 32), (0, 0)),
    )


def test_force_replaces_the_painted_cel():
    doc = inker.Document.blank(32, 32)
    group = doc.insert_flourish(B.bake(_recipe(seed=1)))
    state = doc.flourish_state(group)
    track_uid = next(iter(state.tracks.values()))
    frame = doc.anim.frames[3]
    doc.anim.cels[(track_uid, frame.uid)].pixels[0, 0] = (255, 0, 255, 255)
    counts = doc.apply_flourish(group, B.bake(_recipe(seed=2)), force=True)
    assert counts.conflicts == 0
    assert tuple(doc.anim.cels[(track_uid, frame.uid)].pixels[0, 0]) != (255, 0, 255, 255)


def test_resolve_clears_flags_as_its_own_step():
    doc = inker.Document.blank(32, 32)
    group = doc.insert_flourish(B.bake(_recipe(seed=1)))
    track_uid = next(iter(doc.flourish_state(group).tracks.values()))
    doc.anim.cels[(track_uid, doc.anim.frames[2].uid)].pixels[0, 0] = (1, 2, 3, 255)
    doc.apply_flourish(group, B.bake(_recipe(seed=2)))
    assert doc.flourish_conflicts(group) == [2]
    assert doc.resolve_flourish(group, [2])
    assert doc.flourish_conflicts(group) == []
    assert not doc.resolve_flourish(group, [2])
    doc.history.undo(doc)
    assert doc.flourish_conflicts(group) == [2]


def test_a_new_recipe_layer_gets_a_track_inside_the_group():
    doc = inker.Document.blank(32, 32)
    rec = _recipe()
    group = doc.insert_flourish(B.bake(rec))
    before = len(doc.anim.tracks)
    extra = dataclasses.replace(
        rec, layers=(*rec.layers, flourish.Layer(uid=flourish.new_uid(), kind="flash"))
    )
    counts = doc.apply_flourish(group, B.bake(flourish.clamp(extra)))
    assert counts.added == 1
    assert len(doc.anim.tracks) == before + 1
    state = doc.flourish_state(group)
    for track_uid in state.tracks.values():
        assert doc.group_of.get(track_uid) == group


def test_more_frames_are_appended_when_a_phase_grows():
    doc = inker.Document.blank(32, 32)
    rec = _recipe()
    group = doc.insert_flourish(B.bake(rec))
    longer = dataclasses.replace(rec, phases=(dataclasses.replace(rec.phases[0], frames=20),))
    doc.apply_flourish(group, B.bake(longer))
    assert len(doc.anim.frames) == 20


def test_detach_forgets_the_recipe_and_keeps_the_layers():
    doc = inker.Document.blank(32, 32)
    group = doc.insert_flourish(B.bake(_recipe()))
    tracks = len(doc.anim.tracks)
    assert doc.detach_flourish(group)
    assert doc.flourish_state(group) is None
    assert len(doc.anim.tracks) == tracks
    assert group in doc.groups
    doc.history.undo(doc)
    assert doc.flourish_state(group) is not None


def test_set_recipe_without_rendering_is_a_step_and_a_noop_when_equal():
    doc = inker.Document.blank(32, 32)
    rec = _recipe()
    group = doc.insert_flourish(B.bake(rec))
    assert not doc.set_flourish_recipe(group, rec)
    renamed = dataclasses.replace(rec, name="Puff 2")
    assert doc.set_flourish_recipe(group, renamed)
    assert doc.flourish_state(group).recipe.name == "Puff 2"
    doc.history.undo(doc)
    assert doc.flourish_state(group).recipe.name == rec.name


def test_apply_refuses_a_group_that_is_not_an_effect():
    doc = inker.Document.blank(32, 32)
    with pytest.raises(ValueError):
        doc.apply_flourish(999, B.bake(_recipe()))


def test_regenerate_with_a_linked_cel_raises_before_mutating_anything():
    """Finding #1. A linked cel used to be found mid-loop, after earlier
    frames of the same track were already rewritten and no undo step was
    pushed to cover them -- the raise that reached ``land`` left the document
    half-rendered. Every target must be checked before any of them is
    touched."""
    doc = inker.Document.blank(32, 32)
    first = B.bake(_recipe(seed=1))
    group = doc.insert_flourish(first)
    state = doc.flourish_state(group)
    track_uid = next(iter(state.tracks.values()))
    track_index = next(i for i, t in enumerate(doc.anim.tracks) if t.uid == track_uid)
    # Link frame 5's cel to frame 0's, so the two share one object -- the
    # regenerate must refuse it. Frame 0 is earlier in the render loop than
    # frame 5, so mutation of the earlier frames is exactly what a mid-loop
    # raise would have already done.
    assert doc.link_cel(0, track_index=track_index, frame_index=5)
    assert doc.anim.is_linked(track_uid, doc.anim.frames[5].uid)
    head = doc.history.head
    before_pixels = {
        i: doc.anim.cels[(track_uid, f.uid)].pixels.copy()
        for i, f in enumerate(doc.anim.frames)
        if (track_uid, f.uid) in doc.anim.cels
    }
    second = B.bake(_recipe(seed=2))
    with pytest.raises(ValueError, match="unlink"):
        doc.apply_flourish(group, second)
    # No undo step, and nothing -- including the frames the loop would have
    # reached before the linked one -- was rewritten.
    assert doc.history.head == head
    for i, pixels in before_pixels.items():
        assert np.array_equal(doc.anim.cels[(track_uid, doc.anim.frames[i].uid)].pixels, pixels)


def test_the_recipe_survives_an_ora_round_trip(tmp_path):
    doc = inker.Document.blank(40, 40)
    rec = _recipe(seed=5)
    group = doc.insert_flourish(B.bake(rec))
    track_uid = next(iter(doc.flourish_state(group).tracks.values()))
    doc.anim.cels[(track_uid, doc.anim.frames[1].uid)].pixels[2, 2] = (9, 9, 9, 255)
    doc.apply_flourish(group, B.bake(_recipe(seed=6)))
    assert doc.flourish_conflicts(group) == [1]
    path = tmp_path / "puff.ora"
    ora.write_ora(doc, path)
    again = inker.Document.load(path)
    assert len(again.flourish) == 1
    (guid, state), = again.flourish.items()
    assert guid in again.groups
    assert state.recipe == doc.flourish_state(group).recipe
    assert state.offset == (4, 4)
    assert len(state.tracks) == 1
    assert again.flourish_conflicts(guid) == [1]
    # And it still regenerates: the digests came back, so untouched cels take.
    counts = again.apply_flourish(guid, B.bake(_recipe(seed=7)))
    assert counts.conflicts == 1 and counts.taken >= 1


def test_an_ordinary_document_writes_no_flourish_key(tmp_path):
    doc = inker.Document.blank(8, 8)
    doc.add_frame()
    path = tmp_path / "plain.ora"
    ora.write_ora(doc, path)
    import zipfile

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        grid = next(n for n in names if n.endswith("animation.json"))
        assert b"flourish" not in zf.read(grid)


def test_aseprite_keeps_the_layers_and_drops_the_recipe(tmp_path):
    from warlock.studio.inker import asein, aseout

    doc = inker.Document.blank(32, 32)
    group = doc.insert_flourish(B.bake(_small_layer_recipe()))
    tracks = len(doc.anim.tracks)
    path = tmp_path / "fx.aseprite"
    aseout.write_aseprite(doc, path)
    again, _warnings = asein.document_from_aseprite(path.read_bytes())
    assert again.flourish == {}
    assert len(again.anim.tracks) == tracks
    assert [t.name for t in again.anim.tags] == [t.name for t in doc.anim.tags]
    assert group not in again.flourish
