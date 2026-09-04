"""Restyled keyframes: the interpolator, the snapshot track, and the door."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from warlock.studio import inker, inker_flourish, inker_mode, inker_ops, inker_state
from warlock.studio.inker.flourish import bake as B
from warlock.studio.inker.flourish import keyframes, presets
from warlock.studio.tasks import Done

# -- the interpolator -------------------------------------------------------------------------


def _plane(colour, size=(8, 8)):
    out = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    out[2:6, 2:6] = colour
    return out


def test_anchor_frames_are_spread_over_the_span_ends_included():
    assert keyframes.anchor_frames(0, 10, 3) == [0, 5, 10]
    assert keyframes.anchor_frames(4, 4, 3) == [4]
    assert keyframes.anchor_frames(0, 1, 6) == [0, 1]
    assert keyframes.anchor_frames(3, 12, 2) == [3, 12]


def test_anchors_come_back_verbatim_and_ends_hold():
    a = _plane((255, 0, 0, 255))
    b = _plane((0, 0, 255, 255))
    frames = keyframes.interpolate({2: a, 5: b}, 0, 7)
    assert len(frames) == 8
    assert np.array_equal(frames[2], a) and np.array_equal(frames[5], b)
    assert np.array_equal(frames[0], a) and np.array_equal(frames[7], b)
    mid = frames[3]
    assert 0 < int(mid[4, 4, 0]) < 255 and 0 < int(mid[4, 4, 2]) < 255
    assert mid[4, 4, 3] == 255


def test_the_blend_is_deterministic_and_the_field_moves_it():
    a = _plane((255, 255, 255, 255), (16, 16))
    b = np.zeros_like(a)
    b[8:12, 8:12] = (255, 255, 255, 255)
    plain = keyframes.interpolate({0: a, 4: b}, 0, 4)
    again = keyframes.interpolate({0: a, 4: b}, 0, 4)
    for x, y in zip(plain, again, strict=True):
        assert np.array_equal(x, y)

    def field(frame):
        return (np.full((16, 16), 1.0, dtype=np.float32), np.zeros((16, 16), dtype=np.float32))

    moved = keyframes.interpolate({0: a, 4: b}, 0, 4, field)
    assert not np.array_equal(plain[2], moved[2])
    assert np.array_equal(moved[0], a) and np.array_equal(moved[4], b)


def test_interpolate_refuses_nothing_and_mixed_sizes():
    with pytest.raises(ValueError):
        keyframes.interpolate({}, 0, 3)
    with pytest.raises(ValueError):
        keyframes.interpolate(
            {0: _plane((1, 1, 1, 255)), 1: _plane((1, 1, 1, 255), (4, 4))}, 0, 1
        )


def test_the_recipe_field_is_logical_size_and_bounded():
    rec = dataclasses.replace(presets.load("smoke_puff"), width=24, height=20, supersample=2)
    field = keyframes.field_from_recipe(rec)
    dx, dy = field(3)
    assert dx.shape == (20, 24) and dy.shape == (20, 24)
    assert float(np.abs(dx).max()) <= 1.0 and float(np.abs(dy).max()) <= 1.0
    assert np.array_equal(dx, field(3)[0])


def test_a_sized_field_matches_the_anchor_array_not_the_recipe():
    """Finding #12. ``decode_restyle`` resizes every anchor to the document's
    canvas, not the recipe's -- an effect lands centred and offset inside a
    document that can be a different size. The field ``interpolate`` shifts
    those anchors along has to share that shape, or ``_shift`` broadcasts a
    (recipe h, recipe w) displacement against a (doc h, doc w) plane.
    """
    rec = dataclasses.replace(presets.load("smoke_puff"), width=24, height=20, supersample=2)
    doc_size = (40, 32)  # bigger than the recipe, the ordinary case
    field = keyframes.field_from_recipe(rec, size=doc_size)
    dx, dy = field(3)
    assert dx.shape == (doc_size[1], doc_size[0])
    assert dy.shape == (doc_size[1], doc_size[0])

    a = _plane((255, 255, 255, 255), doc_size)
    b = np.zeros_like(a)
    planes = keyframes.interpolate({0: a, 4: b}, 0, 4, field)
    assert all(p.shape == a.shape for p in planes)


def test_a_recipe_sized_field_cannot_shift_a_differently_sized_anchor():
    """The bug #12 guards against: a field left at the recipe's own logical
    size cannot be broadcast against an anchor the restyle door has already
    resized to the document's canvas."""
    rec = dataclasses.replace(presets.load("smoke_puff"), width=24, height=20, supersample=2)
    field = keyframes.field_from_recipe(rec)  # no ``size`` -- the recipe's own
    a = _plane((255, 255, 255, 255), (40, 32))
    b = np.zeros_like(a)
    with pytest.raises(ValueError):
        keyframes.interpolate({0: a, 4: b}, 0, 4, field)


# -- the snapshot track ----------------------------------------------------------------------


def test_a_snapshot_track_joins_the_group_and_survives_a_regenerate():
    doc = inker.Document.blank(32, 32)
    rec = dataclasses.replace(presets.load("smoke_puff"), width=32, height=32, supersample=2)
    group = doc.insert_flourish(B.bake(rec))
    tracks = len(doc.anim.tracks)
    cels = {i: _plane((9, 9, 9, 255), (32, 32)) for i in range(3, 8)}
    head = doc.history.head
    track_uid = doc.insert_flourish_track(group, "Restyled puff", cels)
    assert doc.history.head == head + 1
    assert len(doc.anim.tracks) == tracks + 1
    assert doc.group_of[track_uid] == group
    state = doc.flourish_state(group)
    assert state.tracks[-1] == track_uid
    assert doc.anim.cels[(track_uid, doc.anim.frames[3].uid)].pixels[3, 3, 0] == 9
    # A regenerate walks the recipe's layers and leaves the snapshot alone.
    doc.apply_flourish(group, B.bake(dataclasses.replace(rec, seed=3)))
    assert doc.anim.cels[(track_uid, doc.anim.frames[3].uid)].pixels[3, 3, 0] == 9
    doc.history.undo(doc)
    doc.history.undo(doc)
    assert len(doc.anim.tracks) == tracks


# -- the door ------------------------------------------------------------------------------------


class _Store:
    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}

    def get(self, job_id):
        return self.jobs.get(job_id)


class _Ctx:
    def __init__(self, root: Path) -> None:
        self.state = SimpleNamespace(inker=inker_state.InkerState())
        self.svc = SimpleNamespace(store=_Store(), job_dir=lambda j: root / j, config=None)
        self.toasts: list = []
        self.tasks = SimpleNamespace(set_progress=lambda *a, **k: None)
        self.pending: list[Done] = []
        self._busy: set[str] = set()

    def toast(self, text, level="info", **_):
        self.toasts.append((text, level))

    def busy(self, key):
        return key in self._busy

    def progress(self, key):
        return None

    def submit(self, key, fn, *args, **kwargs):
        if key in self._busy:
            return False
        try:
            done = Done(key=key, result=fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001
            done = Done(key=key, error=exc)
        self._busy.add(key)
        self.pending.append(done)
        return True

    def land_all(self):
        while self.pending:
            done = self.pending.pop(0)
            self._busy.discard(done.key)
            inker_mode.on_task_done(self, done)


def _scene(tmp_path):
    ctx = _Ctx(tmp_path)
    tab = inker_state.InkerDoc(doc=inker.Document.blank(32, 32))
    ctx.state.inker.docs.append(tab)
    ctx.state.inker.active_uid = tab.uid
    rec = dataclasses.replace(presets.load("sword_impact"), width=32, height=32, supersample=2)
    group = tab.doc.insert_flourish(B.bake(rec))
    return ctx, tab, group


def test_the_restyle_op_is_greyed_without_an_effect(tmp_path):
    ctx, tab, group = _scene(tmp_path)
    op = inker_ops.get("flourish_restyle")
    assert op.enabled(ctx.state.inker, tab)
    tab.doc.set_active_layer(0)
    assert not op.enabled(ctx.state.inker, tab)
    assert inker_ops.reason_for(op, ctx.state.inker, tab) == inker_flourish.NO_EFFECT
    assert inker_flourish.phase_names(ctx.state.inker, tab) == []


def test_the_restyle_door_queues_polls_interpolates_and_lands(tmp_path, monkeypatch):
    ctx, tab, group = _scene(tmp_path)
    state = ctx.state.inker
    calls: list = []

    def fake_create_job(svc, **kwargs):
        calls.append(kwargs)
        return {"id": f"job{len(calls)}"}

    from warlock.service import jobs as svc_jobs

    monkeypatch.setattr(svc_jobs, "create_job", fake_create_job)
    assert inker_ops.run(ctx, inker_ops.get("flourish_restyle"))
    assert state.pending_dialog == inker_flourish.RESTYLE_POPUP
    assert inker_mode.flourish_restyle(
        ctx, tab, phase="sparks", subject="painted sparks", strength=0.6, anchors=3
    )
    assert not inker_mode.flourish_restyle(ctx, tab, phase="sparks")  # one at a time
    ctx.land_all()  # the jobs were queued
    pending = state.flourish_restyle_pending
    assert pending is not None and len(pending["jobs"]) == 3
    assert all(c["init_image"] and c["init_strength"] == 0.6 for c in calls)
    assert "painted sparks" in calls[0]["prompt"]
    span = pending["span"]
    assert pending["frames"] == keyframes.anchor_frames(span[0], span[1], 3)
    # Two of three done: nothing lands.
    for i, job_id in enumerate(pending["jobs"].values()):
        ctx.svc.store.jobs[job_id] = {"status": "done" if i < 2 else "running"}
    inker_flourish.poll_restyle(ctx, state, now=10.0)
    assert state.flourish_restyle_pending is not None
    # All done: the anchors are read, keyed out, interpolated and landed.
    colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    for (_index, job_id), colour in zip(pending["jobs"].items(), colours, strict=True):
        ctx.svc.store.jobs[job_id] = {"status": "done"}
        job_dir = tmp_path / job_id
        job_dir.mkdir()
        picture = Image.new("RGB", (64, 64), (0, 0, 0))
        for x in range(24, 40):
            for y in range(24, 40):
                picture.putpixel((x, y), colour)
        picture.save(job_dir / "input.png")
    tracks = len(tab.doc.anim.tracks)
    inker_flourish.poll_restyle(ctx, state, now=20.0)
    assert state.flourish_restyle_pending is None
    ctx.land_all()
    assert len(tab.doc.anim.tracks) == tracks + 1
    held = tab.doc.flourish_state(group)
    track_uid = held.tracks[-1]
    first, last = span
    for index in range(first, last + 1):
        cel = tab.doc.anim.cels[(track_uid, tab.doc.anim.frames[index].uid)]
        assert cel.pixels.shape == (32, 32, 4)
    # An anchor frame is the model's picture, keyed out and resized: red centre, clear corner.
    anchor = tab.doc.anim.cels[(track_uid, tab.doc.anim.frames[first].uid)].pixels
    assert anchor[16, 16, 0] > 200 and anchor[0, 0, 3] == 0
    assert ctx.toasts[-1][1] == "success"


def test_a_failed_job_ends_the_restyle_with_a_warning(tmp_path):
    ctx, tab, group = _scene(tmp_path)
    state = ctx.state.inker
    state.flourish_restyle_pending = {
        "tab_uid": tab.uid, "group": group, "phase": "hit", "span": [0, 2],
        "jobs": {0: "a", 2: "b"}, "frames": [0, 2], "next_poll": 0.0, "subject": "",
    }
    ctx.svc.store.jobs["a"] = {"status": "done"}
    ctx.svc.store.jobs["b"] = {"status": "failed", "error": "no VRAM"}
    inker_flourish.poll_restyle(ctx, state, now=1.0)
    assert state.flourish_restyle_pending is None
    assert ctx.toasts[-1][1] == "warn"
