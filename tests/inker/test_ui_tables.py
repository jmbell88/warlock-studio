"""What the panes offer against what the engine implements.

Every one of these is a table in a pane beside a table in the engine, and each
pair is a place the two have already drifted: the symmetry combo stopped at
``xy`` while ``brush.SYMMETRY`` carried ``radial``, so the mode the engine
implements, the "Ways" slider next to the combo and the manual chapter all
described a setting no user could select. Asserting membership in *both*
directions is the point -- a pane that offers something the engine does not have
fails on the first click, and one that offers less simply hides the feature.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from warlock.studio.inker import animation, brush, selection, transform
from warlock.studio.panes import inker_canvas, inker_timeline, inker_tools


def test_the_symmetry_combo_offers_every_mode_the_brush_implements():
    assert tuple(key for key, _label in inker_tools.SYMMETRY_LABELS) == brush.SYMMETRY


def test_the_nib_combo_offers_every_nib_the_brush_implements():
    assert tuple(key for key, _label in inker_tools.NIB_LABELS) == brush.NIBS


def test_the_tag_menu_names_every_direction_playback_understands():
    assert tuple(inker_timeline.DIRECTION_NOTES) == animation.DIRECTIONS


def test_the_default_direction_is_written_as_nothing_at_all():
    """A forward loop is what a tag has always been; labelling it would put a
    word on every tag in the band to distinguish the ordinary case from itself.
    """
    assert inker_timeline.DIRECTION_NOTES["forward"] == ""
    assert inker_timeline._tag_note(animation.Tag(name="x")) == ""
    assert inker_timeline._tag_note(animation.Tag(name="x", loop=False)) == " (once)"
    assert (
        inker_timeline._tag_note(animation.Tag(name="x", direction="pingpong"))
        == " (ping-pong)"
    )


@pytest.mark.parametrize(
    ("shift", "alt", "expected"),
    [(False, False, "replace"), (True, False, "add"), (False, True, "subtract"),
     (True, True, "intersect")],
)
def test_every_combine_op_is_reachable_from_the_keyboard(monkeypatch, shift, alt, expected):
    """The fourth was not: the Shift branch answered the pair, so ``intersect``
    existed in the engine and in no gesture."""
    monkeypatch.setattr(
        inker_canvas.imgui,
        "get_io",
        lambda: SimpleNamespace(key_shift=shift, key_alt=alt),
    )
    assert inker_canvas._combine_op() == expected


def test_the_modifiers_cover_the_engines_whole_vocabulary():
    reachable = set()
    for shift in (False, True):
        for alt in (False, True):
            reachable.add(
                "intersect" if shift and alt else "add" if shift else "subtract" if alt
                else "replace"
            )
    assert reachable == set(selection.COMBINE_OPS)


def test_the_resize_popup_offers_both_resamples():
    """Derived from ``transform.RESAMPLES`` at the call site rather than written
    out, so this asserts the tuple is the one thing there is to offer."""
    assert transform.RESAMPLES == ("smooth", "nearest")
