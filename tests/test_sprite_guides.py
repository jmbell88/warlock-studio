"""The shipped pose guides, as data.

``tests/test_spritesynth.py`` owns the guide *machinery* -- what
``_parse_template`` refuses and what ``render_band_guide`` draws.  This file
owns the guides themselves: every ``templates/sprite_guides/*.json`` that
:func:`spritesynth.available_kinds` offers a user, checked for the handful of
authoring slips that a stick figure cannot show you until twenty seconds of
generation have already run against it.

The cheap ones catch a transposed coordinate (a head below the hip) or a
duplicated frame (an animation that stutters, and the easiest slip of the lot,
because the way a pose is authored is by copying the one before it).  The
expensive one is the last: the eight-direction guides author five directions and
derive three, so the mirror is a load-bearing piece of arithmetic and this is
what says it still runs both ways.
"""

from __future__ import annotations

import json

import pytest

from warlock.pipelines import spritesynth as ss

#: The kinds this file was written for.  Named rather than discovered as well,
#: so deleting a template is a failure here and not a silently shorter run.
SHIPPED = ("idle8", "walk8", "run8", "attack8", "cast8", "jump8", "hurt8")

#: Where the eight-direction sheets carry their travel, and how much less of it
#: the rows facing the camera are allowed to show.  ``walk.json``'s comment is
#: the source: "a stride drawn towards the camera reads as a stumble, so those
#: rows carry the cycle in the body bob and the lifted foot instead".  A ratio
#: rather than an absolute, because the number that matters is how much smaller
#: the front row is than the profile -- and a floor rather than a window,
#: because more foreshortening is never the bug.
LOCOMOTION = ("walk8", "run8")
MIN_STRIDE_RATIO = 4.0


def _raw(kind):
    path = ss.TEMPLATE_DIR / f"{kind}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _by_direction(template):
    out: dict[str, dict[int, dict]] = {}
    for pose in template.poses:
        out.setdefault(pose.name, {})[pose.frame] = pose.points
    return out


@pytest.fixture(scope="module")
def templates():
    return {kind: ss.load_guide_template(kind) for kind in SHIPPED}


def test_every_shipped_kind_is_offered():
    """The files on disk and the kinds the form offers are the same set."""
    assert set(SHIPPED) <= set(ss.available_kinds())


@pytest.mark.parametrize("kind", SHIPPED)
def test_expands_to_the_whole_grid(templates, kind):
    """Five authored directions become eight, at the action's own frame count."""
    template = templates[kind]
    action, directions = ss._KIND_SPEC[kind]
    assert directions == 8
    assert len(template.poses) == ss.ACTION_FRAMES[action] * directions

    authored = {(p["name"], p["frame"]) for p in _raw(kind)["poses"]}
    assert {name for name, _frame in authored} == set(ss.AUTHORED_DIRECTIONS)


@pytest.mark.parametrize("kind", SHIPPED)
def test_every_pose_is_complete_and_inside_its_cell(templates, kind):
    template = templates[kind]
    joints = {name for pair in template.segments for name in pair}
    assert template.head_point in joints
    for pose in template.poses:
        missing = joints - set(pose.points)
        assert not missing, f"{kind} {pose.name}/{pose.frame} is missing {missing}"
        for joint, (x, y) in pose.points.items():
            where = f"{kind} {pose.name}/{pose.frame} {joint}"
            assert 0.0 <= x <= 1.0, f"{where} x={x}"
            assert 0.0 <= y <= 1.0, f"{where} y={y}"


@pytest.mark.parametrize("kind", SHIPPED)
def test_head_is_above_the_hip(templates, kind):
    """y is down, so a head below the hip is a transposed pair of coordinates."""
    for pose in templates[kind].poses:
        head = pose.points[templates[kind].head_point][1]
        hip = pose.points["hip"][1]
        assert head < hip, f"{kind} {pose.name}/{pose.frame}: head {head} hip {hip}"


@pytest.mark.parametrize("kind", SHIPPED)
def test_no_direction_repeats_a_frame(templates, kind):
    """A duplicated frame is an animation that stutters at that one frame."""
    for name, frames in _by_direction(templates[kind]).items():
        for f in range(1, max(frames) + 1):
            assert frames[f] != frames[f - 1], (
                f"{kind} {name} frames {f - 1} and {f} are identical"
            )


@pytest.mark.parametrize("kind", SHIPPED)
def test_mirror_round_trip(templates, kind):
    """The derived ``right`` row is the authored ``left`` row, reflected.

    Exact rather than approximate, because the loader derives it by ``x -> 1-x``
    with the ``.L``/``.R`` suffixes swapped and nothing else: any slack here
    would hide the case this is guarding, which is the mirror quietly not
    running at all.
    """
    rows = _by_direction(templates[kind])
    assert "right" in rows and "left" in rows
    for frame, left in rows["left"].items():
        right = rows["right"][frame]
        assert set(right) == set(left)
        for joint, (x, y) in left.items():
            partner = ss._mirror_joint(joint)
            assert right[partner] == (1.0 - x, y), (
                f"{kind} frame {frame}: {joint} did not mirror onto {partner}"
            )


@pytest.mark.parametrize("kind", LOCOMOTION)
def test_the_camera_facing_rows_shorten_the_stride(templates, kind):
    """walk.json's rule, held to by every eight-direction locomotion guide.

    The profile row swings a foot across a third of the cell; the front row must
    not, and pays for it in the two cues that survive a head-on view -- the body
    bob and the lifted foot, which are asserted here to be no *smaller* than the
    profile's.
    """
    rows = _by_direction(templates[kind])

    def travel(name, joint):
        xs = [points[joint][0] for points in rows[name].values()]
        return max(xs) - min(xs)

    def lift(name):
        return max(0.92 - points["foot.L"][1] for points in rows[name].values())

    def bob(name):
        ys = [points["hip"][1] for points in rows[name].values()]
        return max(ys) - min(ys)

    for facing in ("front", "back"):
        ratio = travel("left", "foot.L") / max(travel(facing, "foot.L"), 1e-6)
        assert ratio >= MIN_STRIDE_RATIO, (
            f"{kind}: the {facing} row's foot travels {1 / ratio:.0%} of the "
            "profile's, which is a stride drawn towards the camera"
        )
        assert lift(facing) >= lift("left") - 1e-9
        assert bob(facing) >= bob("left") - 1e-9
        # ...and the arms still have to move, or the row is a figure standing
        # still with its feet twitching.
        assert travel(facing, "hand.L") > 0.05
