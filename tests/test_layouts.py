"""Saved workspace layouts: the arithmetic and the data, with no imgui.

Three properties carry this feature, and all three are testable as numbers.

**Reconciliation only ever reorders and hides.** A layout cannot delete a pane,
so there is no reachable state in which one cannot be got back -- which is what
makes the whole thing safe to ship without an escape hatch nobody can find.

**A newer blob is kept verbatim.** An older build rewriting a layout it does
not understand is the one way this can destroy something.

**Heights never go negative.** A negative child height silently kills a canvas:
it draws nothing, uploads no textures, and reads as a hang.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from warlock.studio import layout_skeleton as skeleton
from warlock.studio import layouts


class _Settings:
    def __init__(self, data=None):
        self.data = dict(data or {})
        self.writes = 0

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value
        self.writes += 1


def _slot(name, sizing=skeleton.FILL, height=0.0, share="", floor=0.0):
    return skeleton.Slot(
        id=name,
        label=name,
        draw=lambda ctx: None,
        sizing=sizing,
        height=height,
        share_key=share,
        floor=floor,
    )


# --- the arithmetic ---------------------------------------------------------


def test_a_column_of_one_fill_takes_everything():
    assert skeleton.heights([_slot("a")], 800.0, {}) == [800.0]


def test_a_fixed_slot_is_taken_out_before_the_rest_is_divided():
    slots = [_slot("preview", skeleton.FIXED, height=180.0), _slot("canvas")]
    assert skeleton.heights(slots, 800.0, {}) == [180.0, 620.0]


def test_a_fixed_slot_scales():
    slots = [_slot("preview", skeleton.FIXED, height=180.0), _slot("canvas")]
    assert skeleton.heights(slots, 800.0, {}, 1.5) == [270.0, 530.0]


def test_two_shares_divide_the_room_between_them():
    slots = [
        _slot("top", skeleton.SHARE, share="k1"),
        _slot("bottom", skeleton.SHARE, share="k2"),
    ]
    tall = skeleton.heights(slots, 600.0, {"k1": 0.25, "k2": 0.75})
    assert tall == [150.0, 450.0]


def test_a_share_never_takes_more_room_than_is_left():
    slots = [
        _slot("top", skeleton.SHARE, share="k1"),
        _slot("bottom", skeleton.SHARE, share="k2"),
    ]
    tall = skeleton.heights(slots, 600.0, {"k1": 0.9, "k2": 0.9})
    assert sum(tall) <= 600.0 + 1e-6


def test_a_floor_wins_over_a_mean_share():
    slots = [_slot("panel", skeleton.SHARE, share="k", floor=210.0), _slot("rest")]
    tall = skeleton.heights(slots, 600.0, {"k": 0.05})
    assert tall[0] == 210.0


def test_a_fill_never_goes_negative():
    """The failure this exists to stop: a canvas that draws nothing, uploads
    no textures and looks exactly like a hang."""

    slots = [_slot("huge", skeleton.FIXED, height=900.0), _slot("canvas")]
    tall = skeleton.heights(slots, 400.0, {})
    assert tall[1] == 0.0
    assert min(tall) >= 0.0


def test_the_heights_sum_to_the_room():
    slots = [
        _slot("preview", skeleton.FIXED, height=100.0),
        _slot("panel", skeleton.SHARE, share="k"),
        _slot("rest"),
    ]
    tall = skeleton.heights(slots, 700.0, {"k": 0.4})
    assert sum(tall) == pytest.approx(700.0)


# --- reconciliation ---------------------------------------------------------


def test_a_stored_order_is_honoured():
    assert skeleton.reconcile(["a", "b", "c"], ["c", "b", "a"]) == ["c", "b", "a"]


def test_a_retired_pane_is_dropped():
    assert skeleton.reconcile(["a", "b"], ["a", "gone", "b"]) == ["a", "b"]


def test_a_new_pane_lands_after_its_last_placed_predecessor():
    """Appending would put every pane added after a user saved their layout at
    the bottom of a column -- so a designer's second pane arrives last for
    everyone who ever dragged anything."""

    assert skeleton.reconcile(["a", "b", "c"], ["a", "c"]) == ["a", "b", "c"]
    assert skeleton.reconcile(["a", "b", "c"], ["c", "a"]) == ["c", "a", "b"]


def test_reconciliation_can_only_reorder_and_never_delete():
    builtin = ["a", "b", "c", "d"]
    for stored in ([], ["d"], ["d", "c", "b", "a"], ["x", "y"]):
        assert sorted(skeleton.reconcile(builtin, stored)) == sorted(builtin)


# --- the library ------------------------------------------------------------


def test_a_fresh_profile_has_the_two_built_ins_and_is_on_the_default():
    library = layouts.Library(_Settings())
    assert set(library.layouts) == set(layouts.BUILT_IN)
    assert library.active == "default"


def test_reading_a_layout_writes_nothing():
    """A launch that changes nothing must not rewrite the file."""

    settings = _Settings()
    library = layouts.Library(settings)
    library.order("inker", "left", ["inker-tools"])
    library.hidden("inker")
    assert settings.writes == 0


def test_an_arrangement_round_trips():
    settings = _Settings()
    library = layouts.Library(settings)
    library.record("inker", {"left": ["b", "a"]}, {"a"})
    again = layouts.Library(settings)
    assert again.arrangement("inker").columns["left"] == ["b", "a"]
    assert again.hidden("inker") == {"a"}


def test_a_newer_blob_is_kept_verbatim_and_not_applied():
    settings = _Settings(
        {
            layouts.LAYOUTS_KEY: {"future": {"v": 99, "workspaces": {"inker": {}}}},
            layouts.ACTIVE_KEY: "future",
        }
    )
    library = layouts.Library(settings)
    assert library.current().readable is False
    # It cannot change what is drawn...
    assert library.order("inker", "left", ["a", "b"]) == ["a", "b"]
    assert library.hidden("inker") == set()
    # ...and it survives a save of the others untouched.
    library.record("inker", {"left": ["b"]}, set())
    library.save()
    assert settings.data[layouts.LAYOUTS_KEY]["future"] == {
        "v": 99,
        "workspaces": {"inker": {}},
    }


def test_a_built_in_is_reset_rather_than_deleted():
    """There is no reachable state in which a pane cannot be got back."""

    settings = _Settings()
    library = layouts.Library(settings)
    library.record("inker", {"left": ["b", "a"]}, {"a"})
    assert library.delete("default") is True
    assert "default" in library.layouts
    assert library.arrangement("inker").columns == {}


def test_a_custom_layout_is_deleted_and_the_active_one_falls_back():
    settings = _Settings()
    library = layouts.Library(settings)
    library.duplicate("default", "mine")
    library.set_active("mine")
    library.delete("mine")
    assert library.active == "default"


def test_a_built_in_cannot_be_renamed():
    library = layouts.Library(_Settings())
    assert library.rename("default", "something") is False
    library.duplicate("default", "mine")
    assert library.rename("mine", "yours") is True
    assert "yours" in library.layouts


def test_mirrored_is_the_two_sidebars_swapped():
    columns = {"left": ["tools"], "centre": ["canvas"], "right": ["colours"]}
    assert layouts.mirrored(columns) == {
        "left": ["colours"],
        "centre": ["canvas"],
        "right": ["tools"],
    }


def test_the_keys_are_top_level_and_not_inside_the_layout_dict():
    """``Settings.set`` replaces a whole dict, and a test asserts the exact key
    set of ``settings["layout"]`` -- a fifth key in it would be a preference
    silently dropped the next time anything else saved."""

    settings = _Settings()
    library = layouts.Library(settings)
    library.record("inker", {"left": ["a"]}, set())
    assert layouts.LAYOUTS_KEY in settings.data
    assert "layout" not in settings.data


def test_a_layout_captures_arrangement_and_not_the_chrome():
    """A workspace switch that collapsed your navigation is the same class of
    surprise as the eighteen-failing-tests incident."""

    settings = _Settings()
    library = layouts.Library(settings)
    library.record("inker", {"left": ["a"]}, set())
    blob = settings.data[layouts.LAYOUTS_KEY]["default"]
    text = repr(blob)
    for forbidden in ("sidebar", "rail", "scale", "theme"):
        assert forbidden not in text


def test_the_active_layout_survives_a_reload():
    settings = _Settings()
    library = layouts.Library(settings)
    library.duplicate("default", "mine")
    library.set_active("mine")
    assert layouts.Library(settings).active == "mine"


def test_an_unknown_active_name_falls_back_rather_than_failing():
    settings = _Settings({layouts.ACTIVE_KEY: "gone"})
    assert layouts.Library(settings).active == "default"


def test_a_ctx_predicate_decides_whether_a_slot_is_live():
    ctx = SimpleNamespace(shown=False)
    slot = skeleton.Slot(
        id="tiles", label="Tiles", draw=lambda c: None, when=lambda c: c.shown
    )
    assert slot.applies(ctx) is False
    ctx.shown = True
    assert slot.applies(ctx) is True


# --- the editor's arithmetic ------------------------------------------------


def _rect(y, h=100.0):
    return (0.0, y, 300.0, h)


def test_a_drop_above_a_panes_middle_lands_before_it():
    from warlock.studio import layout_edit

    rects = [("a", _rect(0.0)), ("b", _rect(100.0)), ("c", _rect(200.0))]
    assert layout_edit.drop_index(rects, 10.0) == 0
    assert layout_edit.drop_index(rects, 120.0) == 1
    assert layout_edit.drop_index(rects, 260.0) == 3


def test_a_drop_into_an_empty_column_is_the_first_place():
    from warlock.studio import layout_edit

    assert layout_edit.drop_index([], 400.0) == 0


def test_moving_a_pane_onto_its_own_place_changes_nothing():
    """The index came from a list that still contained it, which is what makes
    remove-then-insert the whole rule rather than a special case."""

    from warlock.studio import layout_edit

    order = ["a", "b", "c"]
    assert layout_edit.moved(order, "b", 1) == order
    assert layout_edit.moved(order, "b", 2) == order


def test_moving_a_pane_up_and_down():
    from warlock.studio import layout_edit

    assert layout_edit.moved(["a", "b", "c"], "c", 0) == ["c", "a", "b"]
    assert layout_edit.moved(["a", "b", "c"], "a", 3) == ["b", "c", "a"]


def test_a_pane_arriving_from_another_column_is_inserted():
    from warlock.studio import layout_edit

    assert layout_edit.moved(["a", "b"], "x", 1) == ["a", "x", "b"]


def test_the_editor_is_a_toggle_that_drops_what_it_was_holding():
    from types import SimpleNamespace

    from warlock.studio import layout_edit

    state = SimpleNamespace()
    layout_edit.toggle(state)
    assert state.layout_edit.open is True
    state.layout_edit.dragging = "inker-colors"
    layout_edit.toggle(state)
    assert state.layout_edit.open is False
    assert state.layout_edit.dragging == ""


def test_the_splitters_are_suppressed_while_editing():
    """A resize handle and a drag target on the same two pixels is a gesture
    nobody can aim."""

    from warlock.studio import layout

    layout.begin_frame(True)
    try:
        assert layout.splitter("anything") == 0.0
    finally:
        layout.begin_frame(False)


def test_the_skeleton_declares_every_inker_pane():
    from warlock.studio import skeletons

    columns = skeletons.inker(None)
    ids = {slot.id for column in columns.values() for slot in column.slots}
    assert ids == {"inker-tools", "inker-colors", "inker-preview", "inker-tiles"}


def test_the_toolbox_is_neither_movable_nor_hideable():
    """A layout that could hide the toolbox is one that can leave a user with a
    column of nothing and no tool in hand."""

    from warlock.studio import skeletons

    rail = skeletons.inker(None)["left"].slots[0]
    assert rail.movable is False and rail.hideable is False
