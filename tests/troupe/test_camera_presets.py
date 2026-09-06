"""The camera a character sheet is framed from: one table, one default.

A preset is a *name for an elevation*, and the whole risk in offering names is
that the name and the number stop agreeing -- a combo saying "Isometric" over a
sheet rendered at 35 degrees is worse than no combo at all. So the table has one
home, ``pipelines.charsheet.CAMERA_PRESETS``: the door reads it, the form reads
the door, and the worker matches against it when it stamps the sidecar.
"""

from __future__ import annotations

import inspect
from pathlib import Path

from warlock import _q_troupe
from warlock.pipelines import charsheet
from warlock.pipelines import sheet as sheetlib
from warlock.service import troupe as svc_troupe
from warlock.studio.panes import troupe_settings


def test_the_form_and_the_door_read_one_preset_table(svc):
    """The combo's choices are ``troupe_options``' answer and nothing else, and
    ``troupe_options`` is ``charsheet.CAMERA_PRESETS`` reshaped.

    Pinned by scanning as well as by comparing, because the failure this is
    written against is not a wrong answer today -- it is a copy of the angles
    landing in the pane or the door tomorrow and drifting from the table the
    renderer frames from.
    """
    options = svc_troupe.troupe_options(svc)
    presets = options["camera_presets"]
    assert list(presets) == [key for key, _label, _angle in charsheet.CAMERA_PRESETS]
    assert presets == {
        key: {"label": label, "elevation": angle}
        for key, label, angle in charsheet.CAMERA_PRESETS
    }

    # The pane builds its choices out of ``options["camera_presets"]`` -- it
    # never names a preset or an angle itself.
    pane = inspect.getsource(troupe_settings)
    assert '"camera_presets"' in pane
    for key, label, angle in charsheet.CAMERA_PRESETS:
        # As a *literal*: "side" is a substring of "beside", which this file's
        # prose says several times about controls that sit next to each other.
        assert f'"{key}"' not in pane, f"the form names the preset {key!r} itself"
        assert f"'{key}'" not in pane, f"the form names the preset {key!r} itself"
        assert repr(angle) not in pane, f"the form names {angle} itself"
        assert f'"{label}"' not in pane, f"the form names the label {label!r} itself"

    # And across the whole package, two modules spell the default key -- and
    # only two.
    #
    # ``charsheet.py`` owns it. ``resolve.py`` names it as the *target of a
    # vocabulary*: the resolver's job is to turn "3/4 top down" into a preset
    # key, so the key has to be written down on the right-hand side of that
    # table, and there is no arithmetic there to derive it from. It is allowed
    # here on one condition, which the next assertion enforces -- **the
    # resolver may spell a preset's key and may never spell its angle.** A key
    # that drifts is caught the moment it drifts, by the two-way pin in
    # ``tests/characters/test_resolve.py`` (every key the vocabulary emits is a
    # real preset, *and* every preset is askable in words). An angle that
    # drifted would be silent, which is the failure this whole test exists for.
    root = Path(charsheet.__file__).resolve().parents[1]
    homes = sorted(
        path.name
        for path in root.rglob("*.py")
        if charsheet.DEFAULT_CAMERA_PRESET in path.read_text(encoding="utf-8")
    )
    assert homes == ["charsheet.py", "resolve.py"], homes

    resolver = (root / "characters" / "resolve.py").read_text(encoding="utf-8")
    for _key, _label, angle in charsheet.CAMERA_PRESETS:
        assert repr(angle) not in resolver, f"the resolver names the angle {angle}"


def test_the_default_preset_is_three_quarter_top_down_at_35_degrees():
    """The angle nearly every 2D game with depth is drawn at, and the reason it
    is the default rather than ``isometric``: 30 degrees is what every sheet
    this program has rendered so far used, and it was a renderer default that
    nobody chose."""
    assert charsheet.DEFAULT_CAMERA_PRESET == "three_quarter_top_down"
    angles = {key: angle for key, _label, angle in charsheet.CAMERA_PRESETS}
    assert angles[charsheet.DEFAULT_CAMERA_PRESET] == 35.0
    # The one preset that is a number the program already had, named.
    assert angles["isometric"] == sheetlib.DEFAULT_ELEVATION


def test_a_custom_elevation_writes_a_null_preset():
    """A sheet framed at an angle off the ladder records the angle and no name.

    The sidecar must not claim a framing the render was not made at, so the
    preset is matched exactly rather than snapped to the nearest -- 34 degrees
    is not "3/4 top-down", it is 34 degrees.
    """
    custom = _q_troupe._camera_meta(34.0, pixel_size=32, margin=sheetlib.FRAME_MARGIN)
    assert custom["preset"] is None
    assert custom["elevation"] == 34.0
    assert custom["pixel_size"] == 32
    assert custom["render_size"] == charsheet.RENDER_SIZE
    assert custom["projection"] == "orthographic"
    assert custom["frame_margin"] == sheetlib.FRAME_MARGIN

    # And a sheet framed *on* the ladder names the preset it was framed from.
    for key, _label, angle in charsheet.CAMERA_PRESETS:
        named = _q_troupe._camera_meta(
            angle, pixel_size=32, margin=sheetlib.FRAME_MARGIN
        )
        assert named["preset"] == key
        assert named["elevation"] == angle

    # The worker stamps it on every sheet, beside the layout snapshot.
    source = inspect.getsource(_q_troupe.TroupeOps._charsheet)
    assert 'meta["camera"] = _camera_meta(' in source
