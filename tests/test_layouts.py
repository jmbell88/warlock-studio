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


def test_v2_arrangement_round_trips_widths_and_vertical_shares():
    settings = _Settings()
    library = layouts.Library(settings)
    library.record(
        "plotter",
        {"left": ["tools"], "right": ["layers"]},
        set(),
        widths={"left": 272.0, "right": 418.0},
        shares={"plotter-tools": 0.37, "plotter-layers": 0.61},
    )
    again = layouts.Library(settings)
    assert settings.data[layouts.LAYOUTS_KEY]["default"]["v"] == 2
    assert again.width("plotter", "left") == 272.0
    assert again.width("plotter", "right") == 418.0
    assert again.share("plotter", "plotter-tools") == 0.37


def test_v1_uses_legacy_seeds_without_writing_until_an_edit():
    settings = _Settings(
        {
            "layout": {
                "sidebar": "wide",
                "settings_shares": {"clay-tools": 0.42},
            },
            layouts.LAYOUTS_KEY: {"default": {"v": 1, "workspaces": {"clay": {"columns": {}}}}},
        }
    )
    library = layouts.Library(settings)
    assert library.width("clay", "left") == 360.0
    assert library.share("clay", "clay-tools") == 0.42
    assert settings.writes == 0
    library.set_width("clay", "right", 410.0)
    assert settings.data[layouts.LAYOUTS_KEY]["default"]["v"] == 2
    assert settings.writes == 1


def test_independent_width_fit_compresses_without_losing_the_centre_floor():
    from warlock.studio import layout

    left, right, centre = layout.fit_widths(1100.0, 300.0, 420.0, 8.0, scale=1.5)
    assert left < right
    assert left + right + centre + 16.0 == pytest.approx(1100.0)
    assert centre == pytest.approx(220.0 * 1.5)


def test_right_boundary_drag_has_the_opposite_sign_to_the_left():
    settings = _Settings()
    library = layouts.Library(settings)
    from warlock.studio import layout

    assert layout.resize_side(library, "clay", "left", 20.0)
    assert layout.resize_side(library, "clay", "right", -20.0)
    assert library.width("clay", "left") == 320.0
    assert library.width("clay", "right") == 320.0


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
    slot = skeleton.Slot(id="tiles", label="Tiles", draw=lambda c: None, when=lambda c: c.shown)
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
    assert ids == {
        "inker-colors",
        "inker-picker",
        "inker-preview",
        "inker-tools",
        "inker-tiles",
        "inker-generate",
    }


def test_the_toolbox_is_an_ordinary_movable_pane():
    """The inverse of what this pinned while the toolbox was a 90 px rail.

    It was neither movable nor hideable because the left column held nothing
    else: hiding it left a column of nothing, and moving it was the one edit
    the mirrored arrangement existed to forbid. Both reasons went with the
    rail. The column it now sits in has three other panes, so a layout that
    hides it is a layout with a narrower toolbox column, not an empty one --
    and the tools are still reachable by their letters and from the Window
    menu.
    """

    from warlock.studio import skeletons

    slots = {slot.id: slot for slot in skeletons.inker(None)["right"].slots}
    tools = slots["inker-tools"]
    assert tools.movable is True and tools.hideable is True
