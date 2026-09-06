"""The walk-cycle setup panel: parts, joints, four numbers and a preview.

A conditional slot in the right sidebar (``skeletons.inker``), drawn only while
a session is open on the tab in front -- the way ``inker-preview`` and
``inker-tiles`` already come and go. The joints themselves are dragged on the
*real* canvas rather than in a viewport of their own, so a shoulder is placed at
the zoom the drawing is being read at; ``inker_canvas._walk_overlay`` draws them
and ``inker_walk`` owns the arithmetic.

**Nothing here writes to the document.** Every control edits the session, and the
one that lands anything is Bake, which builds a new document. That is what makes
Cancel free: there is no edit to reverse.

The panel is ordered the way the job is done -- parts, then joints, then the
walk, then what it looks like -- and it says what to do next out loud, because
"assign fourteen layers and place fifteen points" is not a thing to leave a
reader to infer from a list of empty rows.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import icons, inker_walk, theme, widgets
from ..inker import walk
from ..inker.walk import rig as R
from ..manual import render as manual_render
from ..tokens import sp
from . import inker_textures

#: The pane's floor in design pixels, and it is a **floor** rather than a
#: request: the share above it is what actually sizes the pane, and at the app's
#: default window that comes out around four hundred.
#:
#: Small because the right column had no room for anything larger. Its other
#: four slots already floor at 804 of the 900 px a sidebar gets at 1600x950
#: (``tests/test_layouts.py`` asks the arithmetic directly, and counts every
#: declared slot whether or not its ``when`` would draw it), so 96 is what was
#: left. It buys the heading, the status line and a row or two, which is the
#: honest minimum below which this panel stops being usable at all -- and the
#: preview is not in here to need more, which is half of why it moved.
WALK_FLOOR = 96.0

#: The parts box's height in design pixels -- three rows or so, which is small
#: and is meant to be. The panel's share comes out near 300 at the app's default
#: window, and what has to survive that is the four **walk sliders** under this
#: box: assignment is done once and then never looked at again, while the
#: sliders are the loop the user actually lives in -- drag, watch the preview,
#: drag again. A list that scrolls inside three rows is still obviously a list;
#: a slider below the fold is a slider nobody finds.
PARTS_H = 96.0

#: Where a part row's combo starts, leaving room for the row's own name.
LABEL_W = 74.0

#: The Cut button's width. Measured rather than guessed: at ``sp(28)`` the label
#: came out as "Cu" -- imgui clips a button's text to its frame rather than
#: growing it, so a width that fits at 100 % silently truncates at 150 %.
CUT_W = 38.0

#: The limbs, in the order the panel groups them, with the heading each gets.
GROUPS: tuple[tuple[str, str], ...] = (
    ("body", "Body"),
    ("near_arm", "Near arm"),
    ("near_leg", "Near leg"),
    ("far_arm", "Far arm"),
    ("far_leg", "Far leg"),
)


def active(ctx: Any) -> bool:
    """Whether a walk is being set up on the tab in front."""
    state = getattr(ctx.state, "inker", None)
    tab = None if state is None else state.active
    return inker_walk.is_open(state, tab)


def draw(ctx: Any) -> None:
    state = ctx.state.inker
    tab = None if state is None else state.active
    widgets.section("Walk cycle")
    manual_render.help_button(ctx, "inker-walk")
    session = inker_walk.session(state, tab)
    if session is None:
        widgets.muted_wrapped("Sprite > Create walk cycle... starts one.")
        return
    _next_step(session)
    widgets.divider()
    _parts(ctx, tab, session)
    widgets.divider()
    _settings(ctx, tab, session)
    widgets.divider()
    _clipping_note(session)
    _exits(ctx, tab, state, session)


def _next_step(session: Any) -> None:
    """One line saying what the panel is waiting for.

    A refusal at the top rather than only under a greyed Bake button: the whole
    setup is a sequence, and a reader who has assigned six parts wants to know
    what the seventh is without hunting for the disabled control that knows.
    """
    refusal = walk.refusal(session.rig)
    if not refusal:
        widgets.muted("Ready to bake.")
        return
    imgui.push_style_color(imgui.Col_.text, theme.rgba(theme.WARN))
    widgets.muted_wrapped(refusal)
    imgui.pop_style_color()


def _parts(ctx: Any, tab: Any, session: Any) -> None:
    """The fourteen rows, in a box of their own that scrolls.

    Bounded rather than allowed to run the length of the panel: fourteen combos
    with their group headings are taller than any sidebar, and letting them push
    the sliders, the preview and Bake below the fold would make a panel whose
    first screen is entirely setup and whose second nobody finds. A box the
    reader scrolls *inside* keeps the whole job on one screen.
    """
    assigned = sum(1 for part in session.rig.parts.values() if part.assigned)
    widgets.field_label(f"Parts ({assigned} of {len(R.PART_NAMES)})")
    layers = _layer_options(tab)
    height = sp(PARTS_H)
    if imgui.begin_child("inker-walk-parts", (0.0, height), imgui.ChildFlags_.borders.value):
        for limb, heading in GROUPS:
            members = _members(limb)
            if not members:
                continue
            widgets.muted(heading)
            for name in members:
                _part_row(ctx, tab, session, name, layers)
            if limb.startswith("far_"):
                _copy_row(ctx, tab, session, limb.removeprefix("far_"))
        # Inside the box, under the two buttons it belongs to: it is what a
        # copy is shaded by, and it is read once beside them rather than
        # returned to the way the walk sliders below are.
        _brightness(ctx, tab, session)
    imgui.end_child()


def _members(limb: str) -> list[str]:
    if limb == "body":
        return ["torso", "head"]
    return [spec.name for spec in R.PARTS if spec.limb == limb]


def _layer_options(tab: Any) -> list[tuple[str, str]]:
    """The drawing's layers, top first -- the order the layers panel shows them.

    Bottom-first is the stack's own order and the wrong one to offer a reader,
    who is looking at a list that runs the other way.
    """
    out = [("", "Not assigned"), ("selection", "From selection")]
    for layer in reversed(tab.doc.stack.layers):
        out.append((str(layer.uid), layer.name))
    return out


def _row_label(name: str) -> str:
    """What a row calls itself, under its group's heading.

    The *kind* and not the full name: the heading above already said "Near arm",
    so a row reading "near upper arm" spends half its width repeating it. Rows
    with no visible name at all was the first thing wrong with this panel -- a
    combo showing a layer name says which layer, and nothing says which part.
    """
    if name in ("torso", "head"):
        return R.label(name)
    return R.label(name).removeprefix("near ").removeprefix("far ")


def _part_row(ctx: Any, tab: Any, session: Any, name: str, layers: list) -> None:
    part = session.rig.parts[name]
    imgui.push_id(f"walkpart{name}")
    # The session's record of which key was chosen, not a search for a layer
    # whose *name* matches: two layers routinely share one, and a search would
    # tick the wrong row while the right pixels sat in the rig.
    current = session.assigned_from.get(name, "")
    if current and current not in {key for key, _label in layers}:
        # The layer it came from has since been deleted or renamed away. The
        # pixels are still in the rig -- they were copied -- so the row says
        # where they came from rather than silently reading "Not assigned".
        layers = [*layers, (current, f"{part.source} (gone)")]
    widgets.muted(_row_label(name))
    imgui.same_line(sp(LABEL_W))
    chosen = widgets.combo(
        f"##{name}", current, layers, -1.0 - sp(CUT_W + 8.0), tooltip=R.label(name)
    )
    if chosen != current:
        if chosen == "":
            inker_walk.clear_part(ctx, tab, name)
        elif chosen.isdigit():
            inker_walk.assign_layer(ctx, tab, name, int(chosen))
    imgui.same_line()
    reason = inker_walk.selection_reason(ctx.state.inker, tab)
    if widgets.disabled_button(
        "Cut",
        not reason,
        (sp(CUT_W), 0),
        reason=reason,
        tooltip=f"Lift the {R.label(name)} out of the selection. Your drawing is not edited.",
    ):
        inker_walk.assign_selection(ctx, tab, name)
    imgui.pop_id()


def _copy_row(ctx: Any, tab: Any, session: Any, limb: str) -> None:
    del session
    if widgets.ghost_button(
        f"Copy near {limb} across",
        (-1, 0),
        tooltip=(
            f"Starts the far {limb} from the near one -- art and joints together "
            "-- shaded by the slider below. Adjust it from there."
        ),
    ):
        inker_walk.copy_near_to_far(ctx, tab, limb)


def _brightness(ctx: Any, tab: Any, session: Any) -> None:
    changed, value = widgets.labeled_slider_float(
        "Far-limb shading",
        session.far_brightness,
        0.3,
        1.0,
        fmt="%.2fx",
        help_text=(
            "How much darker a copied far limb is drawn. A far arm identical to "
            "the near one in front of it reads as one arm. Applied when you copy."
        ),
    )
    if changed:
        inker_walk.set_far_brightness(ctx, tab, value)


def _settings(ctx: Any, tab: Any, session: Any) -> None:
    widgets.field_label("Walk")
    ceiling = max(1.0, walk.reachable_stride(session.rig))
    leg = max(1.0, R.leg_length(session.rig))
    rows = (
        (
            "Stride",
            "stride",
            0.0,
            ceiling,
            "How far apart the feet are at a contact. Bounded by what the leg "
            "can reach with the hip where you put it.",
        ),
        ("Foot lift", "lift", 0.0, leg * 0.4, "How high the swinging foot clears the ground."),
        (
            "Body bob",
            "bob",
            0.0,
            leg * 0.3,
            "Extra vertical travel. The body already sinks at the contacts "
            "because the leg has to reach; this deepens it.",
        ),
        ("Arm swing", "arm_swing", 0.0, 70.0, "How far the arms swing, in degrees."),
    )
    for label, key, low, high, help_text in rows:
        changed, value = widgets.labeled_slider_float(
            label,
            float(getattr(session.settings, key)),
            low,
            high,
            percent=False,
            help_text=help_text,
        )
        if changed:
            inker_walk.set_setting(ctx, tab, key, value)
    _duration(ctx, tab, session)


def _duration(ctx: Any, tab: Any, session: Any) -> None:
    from .. import controls

    widgets.field_label(
        "Frame duration",
        "Milliseconds per frame. Eight frames at 100 ms is a step and a bit a second.",
    )
    imgui.set_next_item_width(-1)
    changed, value = controls.input_int(
        "##walkduration", int(session.settings.duration_ms), 10, 50, commit=True
    )
    if changed:
        inker_walk.set_setting(ctx, tab, "duration_ms", value)


def draw_preview(ctx: Any, tab: Any, session: Any) -> None:
    """The walk, playing, **in the workspace's own Preview pane**.

    Not in the setup panel below it, which is where this started and where it
    could not stay: the panel is a fourteen-row list over four sliders, so a
    preview at the end of it is permanently below the fold -- and the preview is
    the one thing a user looks at *continuously* while dragging a joint.

    The Preview pane is the right host rather than a second one. It exists to
    show what there is to preview; a still drawing has nothing, which is why the
    slot is normally absent, and during a session the walk is exactly that
    thing. ``inker_preview.draw`` hands over here when a session is open.
    """
    frames = inker_walk.frames(session)
    if not frames:
        widgets.muted_wrapped(walk.refusal(session.rig) or "Nothing to show yet.")
        return
    index = inker_walk.tick(session)
    _transport(session, index, len(frames))
    _image(ctx, tab, session, frames, index)


def _transport(session: Any, index: int, total: int) -> None:
    if widgets.transport("inker-walk", session.playing, size=(sp(72), 0), shortcut=""):
        inker_walk.toggle_play(session)
    imgui.same_line()
    if widgets.small_icon_button(icons.ARROW_LEFT, "Previous frame"):
        inker_walk.step_frame(session, -1)
    imgui.same_line()
    if widgets.small_icon_button(icons.ARROW_RIGHT, "Next frame"):
        inker_walk.step_frame(session, 1)
    imgui.same_line()
    widgets.frame_counter(index, total)


def _image(ctx: Any, tab: Any, session: Any, frames: list, index: int) -> None:
    """The shown frame at an **integer** scale, which is the point of it.

    A pixel-art walk judged through a fractional resample is a walk judged
    through a filter nobody will ship it with; the fit picks the largest whole
    multiple that fits and centres the result.
    """
    if ctx.viewer is None:
        return
    region = imgui.get_content_region_avail()
    height = max(sp(40.0), region.y)
    if region.x < 1.0 or height < 1.0:
        return
    plane = frames[index % len(frames)]
    texture = inker_textures.walk_texture(ctx, tab, index, plane)
    if texture is None:
        imgui.dummy((region.x, height))
        return
    width, tall = texture.size
    scale = max(1.0, float(int(min(region.x / max(width, 1), height / max(tall, 1)))))
    draw_w, draw_h = width * scale, tall * scale
    origin = imgui.get_cursor_screen_pos()
    x = origin.x + max(0.0, (region.x - draw_w) * 0.5)
    y = origin.y + max(0.0, (height - draw_h) * 0.5)
    draw_list = imgui.get_window_draw_list()
    widgets.checkerboard(draw_list, (x, y), (x + draw_w, y + draw_h))
    draw_list.add_image(widgets.texture_ref(texture), (x, y), (x + draw_w, y + draw_h))
    imgui.dummy((region.x, height))



def _clipping_note(session: Any) -> None:
    """Say it before the bake, not after: the bake crops silently."""
    over = inker_walk.clipping(session)
    if not any(over):
        return
    sides = [
        f"{amount} px {name}"
        for amount, name in zip(over, ("left", "top", "right", "bottom"), strict=True)
        if amount
    ]
    widgets.cost_note("The walk runs off the canvas by " + ", ".join(sides) + ".")


def _exits(ctx: Any, tab: Any, state: Any, session: Any) -> None:
    del session
    reason = inker_walk.bake_reason(state, tab)
    if widgets.primary_button(
        "Bake to new animation",
        (-1, 0),
        enabled=not reason,
        reason=reason,
        tooltip="Opens the walk in a new tab. This drawing is not touched.",
    ):
        inker_walk.bake(ctx, tab)
    if widgets.ghost_button("Cancel", (-1, 0), tooltip="Nothing was written, so nothing is lost."):
        inker_walk.cancel(ctx, tab)
