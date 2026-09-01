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


def test_a_share_gives_way_to_the_floor_of_the_fill_under_it():
    """A share slot could squeeze a fill slot under its own stated floor.

    Only the *share* slot's floor was ever read here, so a fill slot's floor
    was a number nothing consulted. Measured in Inker on 2026-08-23 at the
    app's default 1600x950: the Colour pane took its 0.55 of an 833 px column
    and left the Picker 363 px against a 400 px floor, so the Picker drew its
    hex field past its own bottom edge and imgui clipped it away. Same rule as
    ``layout.give_way``, one rung down and applied to the whole column.
    """
    slots = [
        _slot("colours", skeleton.SHARE, share="k", floor=210.0),
        _slot("picker", floor=400.0),
    ]
    tall = skeleton.heights(slots, 833.0, {"k": 0.55})
    assert tall[0] == 433.0
    assert tall[1] == 400.0


def test_a_share_keeps_its_own_floor_when_both_cannot_be_met():
    """Neither pane may be given a heading and nothing under it."""
    slots = [
        _slot("colours", skeleton.SHARE, share="k", floor=210.0),
        _slot("picker", floor=400.0),
    ]
    tall = skeleton.heights(slots, 500.0, {"k": 0.55})
    assert tall[0] == 210.0
    assert sum(tall) <= 500.0 + 1e-6


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


def test_a_saved_wave4_plotter_layout_reconciles_into_the_tiled_default():
    """The arrangement change needs no ``VERSION`` bump and no migration.

    Plotter's panes moved sides on 2026-09-01 -- Properties and the map file to
    the left, the layer stack over the tileset palette on the right -- and the
    tools pane stopped being a slot at all. Anybody who had ever dragged a
    Plotter pane has the *old* two lists in their settings file, and this is
    the assertion that says what happens to them.

    ``reconcile`` is per column against ``set(builtin)``, so both halves fall
    out of rules that already existed: a slot the saved column names but the
    new one does not is dropped there, and a slot the new column has but the
    saved one does not is inserted after its last placed predecessor. Nothing
    is rewritten, so a user who then drags nothing keeps their file untouched.
    """

    settings = _Settings()
    library = layouts.Library(settings)
    # Exactly what a v2 file written before the change holds.
    library.record(
        "plotter",
        {
            "left": ["plotter-tools", "plotter-tileset"],
            "right": ["plotter-layers", "plotter-properties", "plotter-bridge"],
        },
        set(),
        shares={"plotter-tools": 0.4, "plotter-layers": 0.6},
    )
    again = layouts.Library(settings)

    left = again.order("plotter", "left", ["plotter-properties", "plotter-bridge"])
    right = again.order("plotter", "right", ["plotter-layers", "plotter-tileset"])

    assert left == ["plotter-properties", "plotter-bridge"]
    # The layer stack was saved first on the right and stays first; the tileset
    # palette arrives from the other column and lands under it rather than
    # displacing it.
    assert right == ["plotter-layers", "plotter-tileset"]
    # And reading it wrote nothing: one ``record`` above, and no more.
    assert settings.writes == 1
    # The orphaned share is still in the file and is simply never consulted --
    # inert, not wrong. Deleting it would be a migration, which is the thing
    # this test exists to say is unnecessary.
    assert again.share("plotter", "plotter-tools") == 0.4


def test_a_plotter_layout_that_hid_a_moved_pane_still_hides_it():
    """A hidden entry names a slot, not a column, so it survives the move.

    ``skeletons.ordered`` drops a hidden slot wherever it now lives; the
    failure this guards against is the opposite one -- a pane a user had put
    away reappearing on the other side of the window after an upgrade.
    """

    settings = _Settings()
    library = layouts.Library(settings)
    library.record(
        "plotter",
        {"left": ["plotter-tools", "plotter-tileset"], "right": ["plotter-layers"]},
        {"plotter-tileset"},
    )
    again = layouts.Library(settings)

    assert again.hidden("plotter") == {"plotter-tileset"}
    assert "plotter-tileset" in again.order(
        "plotter", "right", ["plotter-layers", "plotter-tileset"]
    )


def test_a_shares_default_is_an_even_division_rather_than_a_flat_half():
    """The defect a screenshot caught, and its root cause.

    A share defaulted to 0.5 whatever the column held. That is exactly right
    for the one shape it was written against -- one share over one fill -- and
    wrong for every other: two shares at 0.5 each leave the fill *zero* pixels.
    """
    one = [_slot("a", skeleton.SHARE, share="a"), _slot("fill", skeleton.FILL)]
    assert skeleton.heights(one, 900.0, {}) == [450.0, 450.0], "unchanged"

    two = [
        _slot("a", skeleton.SHARE, share="a"),
        _slot("b", skeleton.SHARE, share="b"),
        _slot("fill", skeleton.FILL),
    ]
    assert skeleton.heights(two, 900.0, {}) == [300.0, 300.0, 300.0]

    three = [*two[:2], _slot("c", skeleton.SHARE, share="c"), two[2]]
    assert skeleton.heights(three, 900.0, {}) == [225.0, 225.0, 225.0, 225.0]


def test_a_saved_drag_still_wins_over_the_even_division():
    """The default is only what a column does before anybody has dragged
    anything; a recorded proportion is a decision and outranks it."""
    slots = [
        _slot("a", skeleton.SHARE, share="a"),
        _slot("b", skeleton.SHARE, share="b"),
        _slot("fill", skeleton.FILL),
    ]
    got = skeleton.heights(slots, 900.0, {"a": 0.5})
    assert got[0] == 450.0
    assert sum(got) == 900.0


@pytest.mark.parametrize("workspace", ["clay", "inker", "plotter", "sirens"])
def test_no_pane_of_any_workspace_is_allocated_nothing(workspace):
    """The property the flat default broke, asserted where it can be seen.

    Three panes were being drawn at zero height before 2026-09-01 -- Plotter's
    Map file, Clay's Document, and Sirens' Sound effects and Song file, the last
    two for far longer. Every one of them still ran, still passed every test
    that called its ``draw``, and was simply given no room by its column. Only a
    screenshot could show it, which is why this now asks the arithmetic
    directly.

    900 px is about what a sidebar column gets at the app's default 1600x950
    once the menu strip and the status row are taken off.
    """
    from types import SimpleNamespace

    from warlock.studio import skeletons

    ctx = SimpleNamespace(
        state=SimpleNamespace(clay=None, inker=None, plotter=None, sirens=None)
    )
    for column in skeletons.BUILDERS[workspace](ctx).values():
        slots = list(column.slots)
        if not slots:
            continue
        got = skeleton.heights(slots, 900.0, {})
        empty = [
            slot.id
            for slot, height in zip(slots, got, strict=True)
            if slot.sizing != skeleton.FIXED and height < 1.0
        ]
        assert not empty, f"{workspace}/{column.id}: {empty} drawn at no height"
        assert sum(got) == pytest.approx(900.0)


def test_a_fill_floor_reserves_room_out_of_the_shares_above_it():
    """The second half of the fix, and it does a different job from the even
    division: the division is what a column does with nobody's drag recorded,
    and the floor is what protects the fill once somebody has dragged a share
    wide. Both were added on 2026-09-01 and neither makes the other redundant.
    """
    slots = [
        _slot("a", skeleton.SHARE, share="a", floor=160.0),
        _slot("b", skeleton.SHARE, share="b", floor=120.0),
        _slot("fill", skeleton.FILL, floor=150.0),
    ]
    # 0.95 rather than 0.8: at 0.8 the share asks for 720 and the headroom is
    # 750, so nothing is clamped and the test would pass without the floor
    # doing anything. The clamp is what is being asserted.
    got = skeleton.heights(slots, 900.0, {"a": 0.95})

    assert sum(got) == 900.0
    assert got[0] == 750.0, "the share gave way to what is under it"
    assert got[0] < 900.0 * 0.95
