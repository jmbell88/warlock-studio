"""The walk-cycle session's half of the canvas: a command row and an overlay.

Split out of ``inker_canvas`` the way ``inker_slices`` and ``inker_gestures``
were, and for the same reason: the canvas pane is the largest file in the mode
and a session that owns the mouse for as long as it is open is a self-contained
piece of it. ``inker_canvas`` calls three functions from here -- ``row`` above
the canvas, ``overlay`` inside ``_paint``, and ``input`` before the paint tools
get a look at the press.

**The joints go on the real canvas, not in a viewport of their own.** A shoulder
placed in a thumbnail is a shoulder placed at whatever zoom the thumbnail
happened to be; placed here it is placed at the zoom the drawing is being read
at, with the drawing's own pixels under it, which is the whole argument for the
overlay costing an input branch.

Hit-testing is in **screen** pixels and converted back to document pixels for
the radius, ``plotter_canvas._handle_at``'s rule: a grab that shrank with the
zoom would be unusable at 8x and a fat target at 1x.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import icons, inker_state, inker_walk, theme, toolbar, widgets
from ..inker import walk
from ..inker.walk import rig as R
from ..tokens import sp

#: Radius the joint dots are drawn at, in design pixels.
JOINT_R = 4.0

#: The dash length of the ground line, in design pixels.
GROUND_DASH = 6.0


def row(ctx: Any, state: Any, tab: Any) -> None:
    """The bar above the canvas while a walk is being set up.

    ``_transform_row``'s shape, and its rule about the two ways out: Bake is the
    row's one primary and Cancel its ghost, both pinned, because this row exists
    for exactly as long as an uncommitted session does and the exits are the last
    things that should move house when the pane narrows.
    """
    session = inker_walk.session(state, tab)
    if session is None:
        return
    widgets.text_colored(theme.ACCENT, "Walk cycle")
    imgui.same_line()
    reason = inker_walk.bake_reason(state, tab)
    items = [
        toolbar.Item(
            "next",
            _next_label(session),
            icons.CROSSHAIR,
            tooltip="The joint a click on the drawing will place.",
            priority=1,
        ),
        toolbar.Item(
            "bake",
            "Bake",
            icons.CHECK,
            role=toolbar.ButtonRole.PRIMARY,
            pinned=True,
            enabled=not reason,
            tooltip=reason or "Open the walk as a new animation. This drawing is untouched.",
        ),
        toolbar.Item("cancel", "Cancel", icons.X, pinned=True),
    ]
    toolbar.toolbar("inker-walk", items, lambda key: _action(ctx, tab, session, key))
    widgets.divider()


def _next_label(session: Any) -> str:
    return R.label(session.joint) if session.joint else "Placed"


def _action(ctx: Any, tab: Any, session: Any, key: str) -> None:
    if key == "bake":
        inker_walk.bake(ctx, tab)
    elif key == "cancel":
        inker_walk.cancel(ctx, tab)
    elif key == "next":
        # Walk the reader on to whatever is still unplaced, which is what makes
        # fifteen points a sequence rather than a hunt.
        following = inker_walk.next_unplaced(session)
        inker_walk.select_joint(ctx, tab, following or session.joint)


# -- the overlay -----------------------------------------------------------------------


def overlay(ctx: Any, state: Any, tab: Any, draw_list: Any, origin) -> None:
    """Ground line, bones and joints, over the drawing they were placed on."""
    session = inker_walk.session(state, tab)
    if session is None:
        return
    del ctx
    view = tab.view
    _ground(session, tab, draw_list, view, origin)
    _bones(session, draw_list, view, origin)
    _joints(session, draw_list, view, origin)


def _screen(view: Any, origin, point) -> tuple[float, float]:
    return inker_state.to_screen(view, origin, point[0], point[1])


def _ground(session: Any, tab: Any, draw_list: Any, view: Any, origin) -> None:
    """A dashed line the width of the canvas, at the height the feet stand on.

    Dashed and drawn edge to edge because it is not part of the drawing: a solid
    rule across a sprite reads as something that will be exported.
    """
    from . import inker_canvas

    width = tab.doc.size[0]
    left = inker_canvas.crisp(_screen(view, origin, (0.0, session.rig.ground_y)))
    right = inker_canvas.crisp(_screen(view, origin, (float(width), session.rig.ground_y)))
    colour = inker_canvas._u32(theme.ACCENT, 0.75)
    span = right[0] - left[0]
    if span <= 0.0:
        return
    dash = max(2.0, sp(GROUND_DASH))
    steps = max(1, int(span // (dash * 2)))
    for step in range(steps):
        head = left[0] + step * span / steps
        tail = min(head + dash, right[0])
        draw_list.add_line((head, left[1]), (tail, right[1]), colour)


def _bones(session: Any, draw_list: Any, view: Any, origin) -> None:
    """A thin line along every segment whose two joints are placed.

    The skeleton is what tells a reader that the knee they just dragged belongs
    to the thigh above it; dots alone are fifteen unrelated points.
    """
    from . import inker_canvas

    colour = inker_canvas._u32(theme.MUTED, 0.7)
    for spec in R.PARTS:
        if spec.direction is None:
            continue
        a = session.rig.joints.get(spec.direction[0])
        b = session.rig.joints.get(spec.direction[1])
        if a is None or b is None:
            continue
        draw_list.add_line(_screen(view, origin, a), _screen(view, origin, b), colour)


def _joints(session: Any, draw_list: Any, view: Any, origin) -> None:
    """Every joint some assigned part needs, ringed when it is the one next up."""
    from . import inker_canvas

    radius = max(2.0, sp(JOINT_R))
    fill = inker_canvas._u32(theme.ACCENT, 0.9)
    edge = inker_canvas._u32(theme.EDGE, 1.0)
    for name, point in inker_walk.handles(session).items():
        centre = _screen(view, origin, point)
        draw_list.add_circle_filled(centre, radius, fill)
        draw_list.add_circle(centre, radius, edge)
        if name == session.joint:
            draw_list.add_circle(centre, radius + max(2.0, sp(3.0)), fill)


# -- input -----------------------------------------------------------------------------


def owns_mouse(state: Any, tab: Any) -> bool:
    return inker_walk.session(state, tab) is not None


def handle(ctx: Any, state: Any, tab: Any, point, *, active: bool) -> None:
    """Press, drag and release for the session. The paint tools never see these.

    A session takes the canvas the way a free transform does -- it is a *state*,
    not a tool -- so this returns before ``_press`` rather than beside it.
    """
    session = inker_walk.session(state, tab)
    if session is None:
        return
    radius = _radius(tab)
    if active and imgui.is_mouse_clicked(0):
        inker_walk.press(ctx, tab, point, radius)
        return
    if session.grab and imgui.is_mouse_down(0):
        inker_walk.drag(ctx, tab, point)
        return
    if session.grab:
        inker_walk.release(ctx, tab)


def _radius(tab: Any) -> float:
    """``inker_walk.GRAB_PX`` of screen, in document pixels at this zoom."""
    zoom = max(1e-6, float(getattr(tab.view, "zoom", 1.0)))
    return max(1.0, sp(inker_walk.GRAB_PX) / zoom)


def escape(ctx: Any, state: Any, tab: Any) -> bool:
    """Escape closes the session, ``state.transforming``'s answer to the key."""
    if inker_walk.session(state, tab) is None:
        return False
    inker_walk.cancel(ctx, tab)
    return True


def clipping_warning(state: Any, tab: Any) -> str:
    """The standing version of the toast a clipped bake would raise once.

    ``_transform_row``'s RotSprite note, applied here: a drag re-renders every
    frame and cannot say this every time, but the user is looking at the row and
    the panel the whole while.
    """
    session = inker_walk.session(state, tab)
    if session is None or not inker_walk.ready(session):
        return ""
    over = inker_walk.clipping(session)
    if not any(over):
        return ""
    return f"The walk runs off the canvas by {max(over)} px -- it will be cropped."


def refusal(state: Any, tab: Any) -> str:
    session = inker_walk.session(state, tab)
    return "" if session is None else walk.refusal(session.rig)
