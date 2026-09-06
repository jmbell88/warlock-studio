"""The walk-cycle session in the shell: refusals, the source, and the bake.

The claim this file exists for is the one the whole feature is designed around
and the one nothing in the engine can check: **setting a walk up never edits the
drawing it was set up from.** Everything else here is the surface a user meets --
which menu rows are live, what a greyed one says, and where the baked document
lands.

A real imgui context with the control census on (``_ui_context``), so the panel
assertions are about controls a user would find rather than about functions
being callable.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from _ui_context import imgui_context

from warlock.studio import inker, inker_ops, inker_state, inker_walk, probe
from warlock.studio.inker import walk
from warlock.studio.inker.walk import rig as R
from warlock.studio.panes import inker_walk as pane

SIZE = (64, 64)


@pytest.fixture
def ui(monkeypatch):
    with imgui_context(monkeypatch) as imgui:
        yield imgui


class _Settings:
    def __init__(self) -> None:
        self.data: dict = {}

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value) -> None:
        self.data[key] = value


class _Ctx:
    def __init__(self) -> None:
        self.state = SimpleNamespace(inker=inker_state.InkerState(), manual=None, preview={})
        self.viewer = None
        self.toasts: list = []
        # ``_adopt`` records the opened path and persists the tool block on its
        # way past. ``test_inker_mode._RecentCtx``'s two methods and nothing
        # else -- a baked document has no path, so nothing is actually stored.
        self.settings = _Settings()

    def toast(self, text, level="info", **_):
        self.toasts.append((text, level))

    def busy(self, key):
        return False

    def progress(self, key):
        return None


def _scene():
    ctx = _Ctx()
    doc = inker.Document.blank(*SIZE)
    doc.stack.active.pixels[18:38, 28:36] = (200, 80, 80, 255)
    doc.stack.active.name = "Body"
    tab = inker_state.InkerDoc(doc=doc)
    ctx.state.inker.docs.append(tab)
    ctx.state.inker.active_uid = tab.uid
    return ctx, tab


#: A near-side half figure in flat blocks, and the rest pose its joints sit in.
#: Its own copy rather than ``tests/inker/walk/_figure``, which lives beside the
#: engine tests and is not importable from here -- and what this file needs is a
#: rig that is *complete*, not one that is byte-identical to the digest fixture.
BLOCKS = {
    "torso": (28, 18, 36, 38, (200, 80, 80, 255)),
    "head": (27, 8, 38, 19, (220, 180, 140, 255)),
    "near_upper_arm": (31, 19, 35, 29, (180, 60, 60, 255)),
    "near_lower_arm": (31, 28, 35, 38, (170, 55, 55, 255)),
    "near_hand": (30, 37, 36, 42, (220, 180, 140, 255)),
    "near_thigh": (30, 37, 35, 48, (70, 70, 160, 255)),
    "near_shin": (30, 47, 35, 57, (60, 60, 150, 255)),
    "near_foot": (30, 56, 40, 60, (40, 40, 40, 255)),
}
JOINTS = {
    "neck": (32, 18),
    "near_shoulder": (32, 20),
    "near_elbow": (32, 29),
    "near_wrist": (32, 38),
    "near_hip": (32, 38),
    "near_knee": (32, 48),
    "near_ankle": (32, 57),
    "near_toe": (39, 58),
}


def _plane(box, colour):
    out = np.zeros((SIZE[1], SIZE[0], 4), dtype=np.uint8)
    out[box[1] : box[3], box[0] : box[2]] = colour
    return out


def _rigged(ctx, tab):
    """A complete rig on the open session, built the way a user would."""
    inker_walk.open_session(ctx, tab)
    session = ctx.state.inker.walk
    for name, (x0, y0, x1, y1, colour) in BLOCKS.items():
        session.rig = walk.set_part(
            session.rig, name, walk.part_from_plane(_plane((x0, y0, x1, y1), colour))
        )
    for name, point in JOINTS.items():
        inker_walk.set_joint(ctx, tab, name, point)
    inker_walk.copy_near_to_far(ctx, tab, "arm")
    inker_walk.copy_near_to_far(ctx, tab, "leg")
    inker_walk.set_ground(ctx, tab, walk.default_ground(session.rig, SIZE[1]))
    return session


# -- the promise ---------------------------------------------------------------


def test_setting_a_walk_up_and_cancelling_leaves_the_drawing_exactly_as_it_was():
    """The property the whole design is arranged to have. Parts are lifted with
    ``selection_cutout``, which pushes no edit; the joints live on the session;
    Bake writes a new document. So there is nothing for Cancel to undo, and this
    is checking that rather than checking that a rollback works."""
    ctx, tab = _scene()
    before_rev = tab.doc.rev
    before_head = tab.doc.history.head
    before_pixels = tab.doc.flatten().copy()
    before_layers = len(tab.doc.stack.layers)

    _rigged(ctx, tab)
    inker_walk.cancel(ctx, tab)

    assert tab.doc.rev == before_rev
    assert tab.doc.history.head == before_head
    assert not tab.doc.history.can_undo
    assert len(tab.doc.stack.layers) == before_layers
    assert np.array_equal(tab.doc.flatten(), before_pixels)


def test_assigning_from_a_selection_does_not_add_a_layer_to_the_drawing():
    """``layer_from_selection`` would; ``selection_cutout`` does not. The whole
    difference between the two doors, asserted where it matters."""
    ctx, tab = _scene()
    inker_walk.open_session(ctx, tab)
    from warlock.studio.inker.selection import SelectionMask

    tab.doc.select(SelectionMask.from_rect(SIZE, (28, 18, 36, 38)))
    head = tab.doc.history.head
    layers = len(tab.doc.stack.layers)

    assert inker_walk.assign_selection(ctx, tab, "torso")
    assert ctx.state.inker.walk.rig.parts["torso"].assigned
    assert tab.doc.history.head == head
    assert len(tab.doc.stack.layers) == layers


def test_a_part_taken_from_a_selection_lands_where_the_selection_was():
    """The cutout arrives cropped, so its offset has to compose with the trim --
    otherwise every joint on that part is out by the marquee's corner."""
    ctx, tab = _scene()
    inker_walk.open_session(ctx, tab)
    from warlock.studio.inker.selection import SelectionMask

    tab.doc.select(SelectionMask.from_rect(SIZE, (28, 18, 36, 38)))
    inker_walk.assign_selection(ctx, tab, "torso")
    assert ctx.state.inker.walk.rig.parts["torso"].origin == (28, 18)


def test_a_layer_assignment_copies_rather_than_referring():
    """A stroke on the drawing halfway through setting a walk up must not
    restyle the rig -- a surprise with no undo behind it."""
    ctx, tab = _scene()
    inker_walk.open_session(ctx, tab)
    uid = tab.doc.stack.active.uid
    assert inker_walk.assign_layer(ctx, tab, "torso", uid)
    held = ctx.state.inker.walk.rig.parts["torso"].pixels.copy()
    tab.doc.stack.active.pixels[:, :] = (0, 255, 0, 255)
    assert np.array_equal(ctx.state.inker.walk.rig.parts["torso"].pixels, held)


# -- the session -----------------------------------------------------------------


def test_the_session_is_the_field_and_there_is_no_flag_beside_it():
    ctx, tab = _scene()
    assert ctx.state.inker.walk is None
    assert not inker_walk.is_open(ctx.state.inker, tab)
    assert inker_walk.open_session(ctx, tab)
    assert inker_walk.is_open(ctx.state.inker, tab)
    assert inker_walk.cancel(ctx, tab)
    assert ctx.state.inker.walk is None


def test_a_session_belongs_to_the_tab_that_opened_it():
    """``transform_uid``'s bug shape: the state object is shared by every tab and
    the panes draw whichever is in front, so a session with no owner would follow
    a tab switch and point the overlay at somebody else's drawing."""
    ctx, tab = _scene()
    other = inker_state.InkerDoc(doc=inker.Document.blank(*SIZE))
    ctx.state.inker.docs.append(other)
    inker_walk.open_session(ctx, tab)
    assert inker_walk.session(ctx.state.inker, tab) is not None
    assert inker_walk.session(ctx.state.inker, other) is None


def test_a_second_session_is_refused_by_name():
    ctx, tab = _scene()
    inker_walk.open_session(ctx, tab)
    assert not inker_walk.can_open(ctx.state.inker, tab)
    assert inker_walk.open_reason(ctx.state.inker, tab) == inker_walk.ALREADY_OPEN


def test_a_canvas_too_large_to_bake_is_refused_before_the_session_opens():
    """Rather than after fourteen parts have been assigned to it."""
    ctx = _Ctx()
    tab = inker_state.InkerDoc(doc=inker.Document.blank(2048, 2048))
    ctx.state.inker.docs.append(tab)
    ctx.state.inker.active_uid = tab.uid
    assert "too large" in inker_walk.open_reason(ctx.state.inker, tab)
    assert not inker_walk.open_session(ctx, tab)


def test_a_busy_document_refuses_the_tool_and_says_why():
    ctx, tab = _scene()
    tab.saving = True
    assert inker_walk.open_reason(ctx.state.inker, tab) == inker_walk.BUSY


# -- refusals the menu shows ------------------------------------------------------


def _op(name):
    return inker_ops.get(name)


def test_the_three_ops_are_registered_under_sprite():
    for name in ("walk_open", "walk_bake", "walk_cancel"):
        op = _op(name)
        assert op is not None, name
        assert op.menu == "Sprite"
        assert op.hint


def test_bake_is_refused_until_the_rig_is_finished_and_names_what_is_missing():
    ctx, tab = _scene()
    inker_walk.open_session(ctx, tab)
    state = ctx.state.inker
    assert not inker_walk.can_bake(state, tab)
    assert "torso" in inker_walk.bake_reason(state, tab)
    _rigged(ctx, tab)
    assert inker_walk.can_bake(state, tab)
    assert inker_walk.bake_reason(state, tab) == ""


def test_bake_and_cancel_are_refused_with_no_session_open():
    ctx, tab = _scene()
    state = ctx.state.inker
    assert inker_walk.bake_reason(state, tab) == inker_walk.NOT_OPEN
    assert inker_walk.cancel_reason(state, tab) == inker_walk.NOT_OPEN
    assert not _op("walk_bake").enabled(state, tab)
    assert not _op("walk_cancel").enabled(state, tab)


def test_cutting_from_a_selection_is_refused_with_no_selection():
    ctx, tab = _scene()
    inker_walk.open_session(ctx, tab)
    assert inker_walk.selection_reason(ctx.state.inker, tab) == inker_walk.NO_SELECTION
    assert not inker_walk.assign_selection(ctx, tab, "torso")


# -- placing joints ---------------------------------------------------------------


def test_a_click_on_empty_canvas_places_the_joint_the_row_is_naming():
    ctx, tab = _scene()
    inker_walk.open_session(ctx, tab)
    session = ctx.state.inker.walk
    session.joint = "near_hip"
    assert inker_walk.press(ctx, tab, (30.0, 40.0), 4.0)
    assert session.rig.joints["near_hip"] == (30.0, 40.0)


def test_a_click_near_a_placed_joint_grabs_it_rather_than_placing_another():
    ctx, tab = _scene()
    inker_walk.open_session(ctx, tab)
    session = ctx.state.inker.walk
    inker_walk.set_joint(ctx, tab, "near_hip", (30.0, 40.0))
    session.joint = "near_knee"
    inker_walk.press(ctx, tab, (31.0, 41.0), 4.0)
    assert session.grab == "near_hip"
    assert "near_knee" not in session.rig.joints


def test_dragging_a_joint_moves_it_and_releasing_ends_the_drag():
    ctx, tab = _scene()
    inker_walk.open_session(ctx, tab)
    session = ctx.state.inker.walk
    inker_walk.set_joint(ctx, tab, "near_hip", (30.0, 40.0))
    inker_walk.press(ctx, tab, (30.0, 40.0), 4.0)
    inker_walk.drag(ctx, tab, (35.0, 44.0))
    assert session.rig.joints["near_hip"] == (35.0, 44.0)
    inker_walk.release(ctx, tab)
    assert session.grab == ""


def test_the_ground_line_is_a_handle_matched_on_its_height_alone():
    """It spans the canvas, so an ``x`` test would mean hunting for the one
    place it can be grabbed."""
    ctx, tab = _scene()
    inker_walk.open_session(ctx, tab)
    session = ctx.state.inker.walk
    inker_walk.set_ground(ctx, tab, 50.0)
    assert inker_walk.nearest(session, (5.0, 51.0), 3.0) == "ground"
    assert inker_walk.nearest(session, (60.0, 51.0), 3.0) == "ground"
    inker_walk.press(ctx, tab, (5.0, 51.0), 3.0)
    inker_walk.drag(ctx, tab, (5.0, 44.0))
    assert session.rig.ground_y == 44.0


def test_placing_a_joint_walks_the_panel_on_to_the_next_one_missing():
    ctx, tab = _scene()
    session = _rigged(ctx, tab)
    session.rig = walk.set_joint(session.rig, "near_hip", (30.0, 40.0))
    del session.rig.joints["near_knee"]
    session.joint = "near_hip"
    inker_walk.press(ctx, tab, (30.0, 40.0), 4.0)
    inker_walk.release(ctx, tab)
    assert session.joint == "near_knee"


def test_adjusting_a_joint_after_the_rig_is_finished_does_not_move_the_selection():
    ctx, tab = _scene()
    session = _rigged(ctx, tab)
    session.joint = "near_hip"
    inker_walk.press(ctx, tab, session.rig.joints["near_hip"], 4.0)
    inker_walk.drag(ctx, tab, (33.0, 39.0))
    inker_walk.release(ctx, tab)
    assert session.joint == "near_hip"


# -- the settings the panel drives ------------------------------------------------


def test_the_defaults_follow_the_rig_until_a_slider_is_touched():
    """A moved joint changes the leg length, so an untouched stride follows it;
    a stride the user has set is theirs and stops being re-derived."""
    ctx, tab = _scene()
    session = _rigged(ctx, tab)
    derived = session.settings.stride
    assert derived > 0.0
    inker_walk.set_joint(ctx, tab, "near_ankle", (32.0, 50.0))
    assert session.settings.stride != derived
    inker_walk.set_setting(ctx, tab, "stride", 3.0)
    inker_walk.set_joint(ctx, tab, "near_ankle", (32.0, 57.0))
    assert session.settings.stride == 3.0


def test_a_stride_above_the_new_bound_is_brought_back_down_when_a_joint_moves():
    """The bound is geometry, so dragging the hip moves it -- and a stride left
    sitting above it would be silently shortened by the clamp on every frame
    instead of showing the user a number they can act on."""
    ctx, tab = _scene()
    session = _rigged(ctx, tab)
    from warlock.studio.inker.walk import gait

    inker_walk.set_setting(ctx, tab, "stride", gait.reachable_stride(session.rig))
    inker_walk.set_joint(ctx, tab, "near_hip", (32.0, 46.0))
    assert session.settings.stride <= gait.reachable_stride(session.rig) + 1e-6


def test_the_frame_duration_is_an_integer_of_at_least_one():
    ctx, tab = _scene()
    session = _rigged(ctx, tab)
    inker_walk.set_setting(ctx, tab, "duration_ms", 0)
    assert session.settings.duration_ms == 1


# -- preview ------------------------------------------------------------------


def test_the_preview_re_renders_only_when_the_rig_or_the_settings_move():
    ctx, tab = _scene()
    session = _rigged(ctx, tab)
    first = inker_walk.frames(session)
    assert inker_walk.frames(session) is first
    inker_walk.set_setting(ctx, tab, "arm_swing", 40.0)
    assert inker_walk.frames(session) is not first


def test_an_unfinished_rig_previews_nothing_rather_than_half_a_figure():
    ctx, tab = _scene()
    inker_walk.open_session(ctx, tab)
    assert inker_walk.frames(ctx.state.inker.walk) == []


def test_playback_advances_off_the_clock_and_wraps():
    ctx, tab = _scene()
    session = _rigged(ctx, tab)
    session.playing = True
    session.play_at = 0.0
    step = session.settings.duration_ms / 1000.0
    assert inker_walk.tick(session, now=0.0) == 0
    assert inker_walk.tick(session, now=step * 3.5) == 3
    assert inker_walk.tick(session, now=step * walk.WALK_FRAMES) == 0


def test_a_stalled_frame_catches_up_in_one_step_rather_than_drifting():
    """The whole reason playback is off the clock and not off a per-frame
    accumulator: a two-second hitch costs the cycle its place, not its pace."""
    ctx, tab = _scene()
    session = _rigged(ctx, tab)
    session.playing = True
    session.play_at = 0.0
    step = session.settings.duration_ms / 1000.0
    assert inker_walk.tick(session, now=step * 21) == 21 % walk.WALK_FRAMES


def test_stepping_a_frame_stops_playback():
    ctx, tab = _scene()
    session = _rigged(ctx, tab)
    inker_walk.step_frame(session, 1)
    assert not session.playing
    assert session.play_index == 1
    assert inker_walk.step_frame(session, -1) == 0
    assert inker_walk.step_frame(session, -1) == walk.WALK_FRAMES - 1


def test_resuming_playback_carries_on_from_the_frame_on_screen():
    ctx, tab = _scene()
    session = _rigged(ctx, tab)
    inker_walk.step_frame(session, 3)
    inker_walk.toggle_play(session, now=100.0)
    assert session.playing
    assert inker_walk.tick(session, now=100.0) == 3


# -- baking -------------------------------------------------------------------


def test_baking_opens_a_new_tab_and_leaves_the_source_open_and_unedited():
    ctx, tab = _scene()
    _rigged(ctx, tab)
    before = tab.doc.flatten().copy()
    baked = inker_walk.bake(ctx, tab)
    assert baked is not None
    assert baked is not tab
    assert tab in ctx.state.inker.docs
    assert np.array_equal(tab.doc.flatten(), before)
    assert len(baked.doc.anim.frames) == walk.WALK_FRAMES
    assert [t.name for t in baked.doc.anim.tags] == [walk.TAG_NAME]


def test_baking_closes_the_session():
    ctx, tab = _scene()
    _rigged(ctx, tab)
    inker_walk.bake(ctx, tab)
    assert ctx.state.inker.walk is None


def test_the_baked_tab_is_named_after_the_drawing_it_came_from():
    ctx, tab = _scene()
    tab.title = "ogre"
    _rigged(ctx, tab)
    assert inker_walk.bake(ctx, tab).title == "ogre walk"


# -- the panel ----------------------------------------------------------------


def _draw(ui, ctx):
    probe.begin_frame()
    ui.new_frame()
    ui.begin("host")
    try:
        pane.draw(ctx)
    finally:
        ui.end()
        ui.end_frame()
    return probe.census()


def test_the_panel_draws_its_controls_only_while_a_session_is_open(ui):
    ctx, tab = _scene()
    assert _draw(ui, ctx) == []
    _rigged(ctx, tab)
    labels = {control.label for control in _draw(ui, ctx)}
    assert any("Bake" in label for label in labels)
    assert any("Cancel" in label for label in labels)
    assert any("Stride" in label for label in labels)


def test_the_slot_is_drawn_only_while_a_session_is_open():
    ctx, tab = _scene()
    assert not pane.active(ctx)
    inker_walk.open_session(ctx, tab)
    assert pane.active(ctx)


def test_the_panel_says_what_is_missing_rather_than_only_greying_the_button(ui):
    """A greyed Bake with no reason is the defect this project has a rule about,
    and the panel says it at the top as well, because the setup is a sequence."""
    ctx, tab = _scene()
    inker_walk.open_session(ctx, tab)
    assert "torso" in walk.refusal(ctx.state.inker.walk.rig)
    bake = [c for c in _draw(ui, ctx) if "Bake" in c.label]
    assert bake and not bake[0].enabled


def test_every_part_and_limb_the_panel_groups_covers_the_whole_table():
    """A part missing from ``GROUPS`` is a part with no row and no way to assign
    it, which nothing else here would catch."""
    grouped: list[str] = []
    for limb, _heading in pane.GROUPS:
        grouped += [spec.name for spec in R.PARTS if spec.limb == limb] or (
            ["torso", "head"] if limb == "body" else []
        )
    assert sorted(grouped) == sorted(R.PART_NAMES)


# -- the key context ----------------------------------------------------------


def test_an_open_session_is_its_own_key_context():
    """Modal in the transform's sense: it owns the canvas until it is baked or
    cancelled, so Enter and Escape mean its two exits and reach nothing else."""
    ctx, tab = _scene()
    assert inker_state.key_context(ctx.state.inker, tab) != "WalkCycle"
    inker_walk.open_session(ctx, tab)
    assert inker_state.key_context(ctx.state.inker, tab) == "WalkCycle"


def test_the_walk_context_beats_a_selection_and_a_float():
    """A user lifts a body part out of a selection while setting one up, and
    Escape then has to mean "close the setup", not "drop the marquee"."""
    ctx, tab = _scene()
    from warlock.studio.inker.selection import SelectionMask

    inker_walk.open_session(ctx, tab)
    tab.doc.select(SelectionMask.from_rect(SIZE, (2, 2, 8, 8)))
    assert inker_state.key_context(ctx.state.inker, tab) == "WalkCycle"


def test_the_context_is_one_the_registry_knows_about():
    """Derived from the same table, so a name here that ``key_context`` can never
    return would be a binding that can never fire."""
    assert "WalkCycle" in inker_ops.CONTEXTS


# -- the workspace ------------------------------------------------------------


def _right(ctx):
    from warlock.studio import skeletons

    columns = skeletons.inker(ctx)
    return [slot.id for slot in columns["right"].slots if slot.applies(ctx)]


def test_the_toolbox_and_the_file_panel_stand_down_during_a_session():
    """Not tidying. The canvas is placing joints, so every tool in the toolbox is
    inert, and the file panel is a second spelling of File and Sheet menu rows --
    while between them they hold the column's largest floors, which is what the
    setup panel needs. Both come straight back on either exit."""
    ctx, tab = _scene()
    before = _right(ctx)
    assert "inker-tools" in before and "inker-generate" in before

    inker_walk.open_session(ctx, tab)
    during = _right(ctx)
    assert "inker-walk" in during
    assert "inker-tools" not in during
    assert "inker-generate" not in during

    inker_walk.cancel(ctx, tab)
    assert _right(ctx) == before


def test_the_preview_slot_appears_for_a_walk_on_a_still_drawing():
    """One Preview pane showing whatever there is to preview, rather than a
    second one appearing beside it. A still drawing has nothing, which is why the
    slot is normally absent; a session is exactly that thing."""
    ctx, tab = _scene()
    assert tab.doc.anim is None
    assert "inker-preview" not in _right(ctx)
    inker_walk.open_session(ctx, tab)
    assert "inker-preview" in _right(ctx)


def test_the_panel_is_gone_again_after_a_bake():
    ctx, tab = _scene()
    _rigged(ctx, tab)
    baked = inker_walk.bake(ctx, tab)
    assert baked is not None
    assert "inker-walk" not in _right(ctx)
    assert "inker-tools" in _right(ctx)


def test_the_parts_list_remembers_which_layer_was_chosen_rather_than_its_name():
    """Two layers routinely share a name -- a duplicate is called "Layer 2" twice
    as often as not -- so matching the combo's tick on ``Part.source`` would tick
    the wrong row while the right pixels sat in the rig."""
    ctx, tab = _scene()
    tab.doc.add_layer(name="Body")
    tab.doc.stack.active.pixels[10:14, 10:14] = (1, 2, 3, 255)
    second = tab.doc.stack.active.uid
    first = tab.doc.stack.layers[0].uid
    assert tab.doc.stack.layers[0].name == tab.doc.stack.layers[1].name

    inker_walk.open_session(ctx, tab)
    inker_walk.assign_layer(ctx, tab, "torso", second)
    assert ctx.state.inker.walk.assigned_from["torso"] == str(second)
    inker_walk.assign_layer(ctx, tab, "torso", first)
    assert ctx.state.inker.walk.assigned_from["torso"] == str(first)
    inker_walk.clear_part(ctx, tab, "torso")
    assert "torso" not in ctx.state.inker.walk.assigned_from


def test_copying_a_limb_carries_where_its_art_came_from():
    ctx, tab = _scene()
    session = _rigged(ctx, tab)
    session.assigned_from["near_thigh"] = "selection"
    inker_walk.copy_near_to_far(ctx, tab, "leg")
    assert session.assigned_from["far_thigh"] == "selection"
