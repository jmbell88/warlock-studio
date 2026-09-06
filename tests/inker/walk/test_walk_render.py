"""Poses to pixels: determinism, framing, and one golden digest per frame.

The digest table is the same device ``tests/inker/flourish`` uses and for the
same reason: what a render must not do is quietly change, and a hash over the
bytes says that in a way a description of the picture cannot. Regenerate it
deliberately with ``uv run python tests/inker/walk/_digests.py`` when a change to
the motion is intended, and never to make a red test go green.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest
from _figure import SIZE, figure

from warlock.studio.inker.walk import bake, gait, render
from warlock.studio.inker.walk import rig as R

DIGESTS = Path(__file__).with_name("digests.json")


def _settings(rig: R.Rig, **changes: float) -> gait.WalkSettings:
    return gait.defaults_for(rig).replaced(**changes)


def _digest(planes) -> str:
    sha = hashlib.sha256()
    for plane in planes:
        sha.update(np.ascontiguousarray(plane).tobytes())
    return sha.hexdigest()


@lru_cache(maxsize=1)
def _composites() -> tuple[np.ndarray, ...]:
    """Cached: every test below wants the same eight frames, and a RotSprite
    turn of fourteen parts is not cheap enough to repeat a dozen times in the
    default lane."""
    rig = figure()
    return tuple(bake.composite_frames(rig, _settings(rig), SIZE))


def test_the_same_rig_renders_the_same_bytes_twice():
    """Determinism is not a nicety here: the whole turn is RotSprite, whose
    every step is an integer copy, and a render that wobbled would mean
    something below it had started interpolating."""
    rig = figure()
    settings = _settings(rig)
    first = bake.composite_frames(rig, settings, SIZE)
    second = bake.composite_frames(rig, settings, SIZE)
    assert _digest(first) == _digest(second)


def test_two_rigs_built_the_same_way_render_the_same_bytes():
    """Nothing in a rig is identity-dependent -- no uid reaches the pixels -- so
    two figures assembled by the same steps are the same walk."""
    left = figure()
    right = figure()
    assert _digest(bake.composite_frames(left, _settings(left), SIZE)) == _digest(
        bake.composite_frames(right, _settings(right), SIZE)
    )


def test_the_digests_match():
    if not DIGESTS.exists():  # pragma: no cover -- the regenerator's message
        pytest.fail("digests.json is missing; run tests/inker/walk/_digests.py")
    stored = json.loads(DIGESTS.read_text(encoding="utf-8"))
    assert stored["figure"] == _digest(_composites())


def test_every_frame_is_the_same_size_as_the_canvas():
    """Fixed framing across the cycle, which is what makes the frames a sprite
    sheet rather than a pile of differently sized pictures."""
    for plane in _composites():
        assert plane.shape == (SIZE[1], SIZE[0], 4)
        assert plane.dtype == np.uint8


def test_every_frame_draws_something():
    for index, plane in enumerate(_composites()):
        assert plane[:, :, 3].any(), index


def test_a_part_composites_over_transparent_black():
    """Nothing is drawn onto an opaque backdrop on the way through, so an
    untouched pixel is (0, 0, 0, 0) and not a black one -- which is the
    difference between a sprite and a sprite on a black card."""
    plane = _composites()[0]
    empty = plane[plane[:, :, 3] == 0]
    assert empty.size
    assert not empty.any()


def test_the_cycle_reports_the_box_it_needs():
    rig = figure()
    rendered = render.frames(rig, _settings(rig))
    box = render.bounds(rendered)
    assert box is not None
    x0, y0, x1, y1 = box
    assert x0 < x1 and y0 < y1
    for frame in rendered:
        for pixels, (left, top) in frame.values():
            assert x0 <= left and top >= y0
            assert left + pixels.shape[1] <= x1
            assert top + pixels.shape[0] <= y1


def test_the_bounds_are_taken_over_the_whole_cycle_and_not_one_frame():
    """A foot that only leaves the canvas on frame six still clips the
    animation, so a per-frame box would report "it fits" for most of them."""
    rig = figure()
    rendered = render.frames(rig, _settings(rig))
    whole = render.bounds(rendered)
    assert whole is not None
    for frame in rendered:
        one = render.bounds([frame])
        assert one is not None
        assert one[0] >= whole[0] and one[2] <= whole[2]


def test_a_walk_that_fits_reports_no_clipping():
    rig = figure()
    assert render.clipping(render.frames(rig, _settings(rig)), SIZE) == (0, 0, 0, 0)


def test_a_walk_that_runs_off_the_canvas_says_by_how_much():
    """Shown before the bake rather than discovered after it: the bake crops
    silently, and a foot lost to the canvas edge reads as a rig error."""
    rig = figure()
    dropped = R.set_ground(rig, SIZE[1] + 20.0)
    over = render.clipping(render.frames(dropped, _settings(dropped)), SIZE)
    assert over[3] > 0


def test_a_part_lands_where_its_joint_went():
    """The overlay draws joints and the renderer turns parts, and the two agree
    only if the placement really is "the pivot, moved to the posed pivot". A foot
    is the case that shows it: it is flat through stance, so its pixels must sit
    exactly where the rest drawing had them relative to the ankle."""
    rig = figure()
    settings = _settings(rig)
    spec = R.BY_NAME["near_foot"]
    part = rig.parts["near_foot"]
    for pose in gait.cycle(rig, settings):
        if not pose.grounded["near"]:
            continue
        placed = render.part_frame(part, spec, rig, pose)
        assert placed is not None
        _pixels, (left, top) = placed
        rest_ankle = rig.joints["near_ankle"]
        posed_ankle = pose.joints["near_ankle"]
        assert left == pytest.approx(
            part.origin[0] + posed_ankle[0] - rest_ankle[0], abs=1.0
        )
        assert top == pytest.approx(part.origin[1] + posed_ankle[1] - rest_ankle[1], abs=1.0)


def test_an_unassigned_part_draws_nothing():
    rig = figure(far=False)
    for frame in render.frames(rig, _settings(rig)):
        assert not any(name.startswith("far_") for name in frame)


def test_the_far_copy_is_darker_but_the_same_shape():
    """The one adjustment a copied far limb gets. It multiplies colour and never
    alpha, so a silhouette is untouched -- otherwise "start from the near limb"
    would quietly also mean "and lose a pixel off its edge"."""
    rig = figure(brightness=0.6)
    near = rig.parts["near_thigh"].pixels
    far = rig.parts["far_thigh"].pixels
    assert near is not None and far is not None
    assert np.array_equal(near[:, :, 3], far[:, :, 3])
    lit = near[:, :, 3] > 0
    assert (far[:, :, :3][lit] < near[:, :, :3][lit]).any()


def test_an_unshaded_copy_is_pixel_identical():
    rig = figure(brightness=1.0)
    near = rig.parts["near_shin"].pixels
    far = rig.parts["far_shin"].pixels
    assert near is not None and far is not None
    assert np.array_equal(near, far)
