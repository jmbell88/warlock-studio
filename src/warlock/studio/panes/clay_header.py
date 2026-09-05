"""The row across the top of Clay's viewport: what you are editing, and how.

Clay had no viewport toolbar at all. Which element mode you were in, which
transform tool you held, whether snapping was on and what the viewport was
showing were four separate blocks down a 300 px sidebar -- and the sidebar is
on the far side of the window from the model. Every one of them is a setting
changed *between* clicks in the viewport, which is the same argument that moved
Inker's tool options onto a context bar and Plotter's tools onto one.

**The mode and the tool are fields, not items.** ``toolbar`` collapses items
into an overflow menu before it collapses anything else, and a mode picker in a
menu is a mode picker nobody can see the state of -- which is the one thing a
mode picker is for. As fields they compete with the buttons by ``priority``, and
they are priority 0.

**Modes are lettered, not drawn.** ``icons.py`` is a transcription of
lucide-static 0.525.0 and its docstring forbids guessing a codepoint; the
vendored subset has nothing for a vertex, an edge or a face. ``inker_context``
made the same call for its four symmetry mirrors and said so.

What is *not* here yet, and deliberately: the pivot and orientation menus, and
X-mirror. Each is a control whose behaviour is a later wave, and a switch that
changes a setting nothing reads is worse than no switch -- it is the UI stating
what is not true, which is the defect this codebase has paid for before. They
arrive with the code that reads them.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import clay_mode, clay_state, controls, fonts, icons, toolbar, widgets
from ..manual import render as manual_render
from ..tokens import sp
from . import clay_tools

#: This bar's imgui id and the prefix every one of its controls is keyed from.
BAR = "clay-header"

#: How tall the header is, in design pixels, for the viewport's reservation.
#: A frame height plus the separator under it plus the row's own spacing --
#: measured rather than guessed would be better and is not possible: the
#: viewport has to subtract this *before* it draws, and the bar is drawn after.
HEADER_H = 34.0

SNAP_POPUP = "clay-snap"
PROPORTIONAL_POPUP = "clay-proportional"
OVERLAYS_POPUP = "clay-overlays"
VIEW_POPUP = "clay-view"

#: How the surface is drawn, in the order Blender lists them: the flat albedo,
#: the lit render, the edges alone. Glyph-free -- lucide has nothing for a
#: shading mode and ``icons.py`` forbids inventing a codepoint -- so the compact
#: tier is the first letter, which is unambiguous across three.
SHADING: tuple[tuple[str, str, str, str], ...] = (
    ("solid", "Solid", "S", "The albedo with no lighting: silhouette and topology, "
     "with no highlight sitting on the vertex you are dragging"),
    ("material", "Material", "M", "The lit render -- what the object will look like"),
    ("wireframe", "Wire", "W", "The edges alone, with no surface at all"),
)

#: The element modes, compact. ``clay_tools.MODE_BUTTONS`` carries the full
#: labels and the keys; this is only what a narrow window shows instead.
MODE_SHORT = {"object": "Obj", "vertex": "V", "edge": "E", "face": "F"}


def draw(ctx: Any, view: Any = None) -> None:
    """The bar. Called by the viewport, between the tab bar and the image."""

    state = clay_mode.ensure(ctx)
    tab = state.active
    if tab is None:
        return
    hit = toolbar.toolbar(
        BAR,
        _items(state),
        fields=[_mode_field(tab), _tool_field(state)],
        trailing=_trailing(ctx, state, view),
    )
    if hit == "snap":
        imgui.open_popup(SNAP_POPUP)
    elif hit == "proportional":
        imgui.open_popup(PROPORTIONAL_POPUP)
    _snap_popup(state)
    _proportional_popup(state)
    widgets.divider()


def _items(state: Any) -> list[Any]:
    """The two popovers that hold a group of numbers each.

    Behind a button rather than on the bar because both are a switch *and* the
    figures that switch governs -- a grid size, an angle, a radius -- and a 34 px
    row is the wrong shape for a number field. ``selected`` on the button is
    what keeps the state visible with the popover shut, which is the whole
    reason a popover is allowed to hold a switch at all.
    """

    return [
        toolbar.Item(
            "snap",
            "Snap",
            icons.MAGNET,
            tooltip="Snap a drag to the grid, to an angle, or onto the vertex "
            "under the cursor",
            selected=bool(state.snap or state.snap_vertex),
        ),
        toolbar.Item(
            "proportional",
            "Falloff",
            icons.CIRCLE,
            tooltip="Carry the geometry around the selection with the drag, "
            "fading out over a radius",
            selected=bool(state.proportional),
            priority=1,
        ),
    ]


def _mode_field(tab: Any) -> Any:
    """Object / Verts / Edges / Faces.

    The mode lives on the **document** rather than on ``ClayState``: it is the
    interpretation key for a selection, and an app-level mode would reinterpret
    every other tab's selection on a tab switch. So this reads ``tab.doc`` and
    the state is not involved.
    """

    def draw_it(compact: bool) -> None:
        doc = tab.doc
        options = [
            (mode, MODE_SHORT[mode] if compact else label)
            for mode, label, _key in clay_tools.MODE_BUTTONS
        ]
        tips = {
            mode: f"{label} mode  ({key})"
            for mode, label, key in clay_tools.MODE_BUTTONS
        }
        changed, picked = controls.segmented_choice(
            "clay-mode",
            options,
            doc.element_mode,
            tooltips=tips,
            compact=True,
            enabled=not tab.saving,
            reason=_SAVING,
        )
        if changed:
            doc.set_element_mode(picked)

    return toolbar.Field("mode", "Mode", draw_it, width=176.0, compact=104.0)


def _tool_field(state: Any) -> Any:
    """Select / Move / Rotate / Scale, as glyphs at both tiers.

    Four glyphs are already the smallest this can be, so its compact width is
    its full one -- which is what ``Field`` means by a control that gets no
    narrower. The letters are in the tooltips, where a compacted control's name
    always goes.
    """

    def draw_it(_compact: bool) -> None:
        options = [
            (key, clay_tools.TOOL_ICONS.get(key) or label[:1])
            for key, label, _shortcut in clay_state.TOOLS
        ]
        tips = {
            key: f"{label}  ({shortcut})" for key, label, shortcut in clay_state.TOOLS
        }
        changed, picked = controls.segmented_choice(
            "clay-tool", options, state.tool, tooltips=tips, compact=True
        )
        if changed:
            state.tool = picked

    return toolbar.Field("tool", "Tool", draw_it, width=124.0, compact=124.0)


_SAVING = "This document is being written; the controls come back when it lands."


def _snap_popup(state: Any) -> None:
    """The grid, the angle and the vertex switch. Lifted whole from the pane.

    Live while a save runs, on purpose: none of it touches the document.
    """

    with controls.menu_popup(SNAP_POPUP) as opened:
        if not opened:
            return
        changed, value = widgets.toggle(f"{icons.MAGNET} Snap", state.snap)
        if changed:
            state.snap = value
        imgui.begin_disabled(not state.snap)
        # "%.4f", because the grid steps by 1/16 m and imgui's default "%.3f"
        # drew that as 0.063 -- a field that disagrees with its own step button.
        _, state.snap_translate = controls.input_float(
            "grid (m)##snapt", state.snap_translate, 0.0625, 0.0, "%.4f"
        )
        _, state.snap_rotate = controls.input_float(
            "angle (deg)##snapr", state.snap_rotate, 5.0, 0.0
        )
        imgui.end_disabled()
        # Outside the disable, because it is a *separate* switch rather than a
        # mode of the grid: the two answer different questions -- "put it on
        # round numbers" and "put it exactly there" -- and a user aligning two
        # parts wants the second without giving up the first everywhere else.
        changed, value = widgets.toggle(
            f"{icons.MAGNET} Snap to vertex", state.snap_vertex
        )
        if changed:
            state.snap_vertex = value
        widgets.help_marker(
            "While moving, a drag lands on the vertex under the cursor rather "
            "than on the grid. The vertices being moved are never candidates, "
            "so a drag cannot snap onto itself. Typing a value or locking an "
            "axis (X/Y/Z during a drag) overrides it."
        )
        # Clamped rather than validated: zero is the off switch every snap
        # function already treats as the identity, and a negative grid is
        # meaningless.
        state.snap_translate = max(0.0, float(state.snap_translate))
        state.snap_rotate = max(0.0, float(state.snap_rotate))


def _proportional_popup(state: Any) -> None:
    with controls.menu_popup(PROPORTIONAL_POPUP) as opened:
        if not opened:
            return
        changed, value = widgets.toggle(
            f"{icons.CIRCLE} Soft falloff", state.proportional
        )
        if changed:
            state.proportional = value
        widgets.help_marker(
            "An element drag carries the geometry around the selection with it, "
            "fading out over the radius, so the surface bends instead of "
            "tearing. The radius is metres of world space, measured from the "
            "nearest selected vertex."
        )
        imgui.begin_disabled(not state.proportional)
        _, state.proportional_radius = controls.input_float(
            "radius (m)##propr", state.proportional_radius, 0.05, 0.0, "%.3f"
        )
        imgui.end_disabled()
        # Clamped rather than validated, the grid's rule: zero is the off switch
        # the falloff already treats as a hard selection.
        state.proportional_radius = max(0.0, float(state.proportional_radius))


#: What the viewport draws over the model, as a table. One row per flag so the
#: popover is a loop rather than a column of hand-written toggles, and so a
#: sixth overlay is one line here.
#:
#: ``grid`` is a field on ``ClayState`` and the rest live in ``state.overlays``,
#: which is not an inconsistency to tidy: the grid is wired straight to
#: ``ClayView.show_grid`` and has been since the viewport existed, and giving it
#: a second home in the dict would be two places that can disagree about one
#: switch. The table says which is which and the popover does not care.
OVERLAY_ROWS: tuple[tuple[str, str, str], ...] = (
    ("grid", "Grid", "The ground plane, at the snap size"),
    ("wire", "Wireframe", "Every edge, over the shaded surface"),
    (
        "stats",
        "Statistics",
        "Objects, vertices, edges, faces and triangles -- and how many are "
        "selected. Every one of these was unavailable anywhere in Clay before "
        "the overlay existed.",
    ),
)

#: What each row is when nothing has been changed, so the Overlays button can
#: say "something in here is not the default" without a second hand-written
#: list. It had one -- ``(("grid", True), ("wire", False))`` -- which left
#: ``stats`` out, so turning Statistics on and nothing else drew the button
#: unselected: the one overlay with no other sign it is on was the one the
#: button would not report.
OVERLAY_DEFAULTS: dict[str, bool] = {"grid": True, "wire": False, "stats": False}


def overlay_value(state: Any, key: str) -> bool:
    return bool(state.grid if key == "grid" else state.overlays.get(key, False))


def set_overlay(state: Any, key: str, value: bool) -> None:
    if key == "grid":
        state.grid = bool(value)
    else:
        state.overlays[key] = bool(value)


def _overlays_popup(ctx: Any, state: Any) -> None:
    with controls.menu_popup(OVERLAYS_POPUP) as opened:
        if not opened:
            return
        for key, label, tip in OVERLAY_ROWS:
            hit = controls.menu_item(
                f"{label}##{BAR}/overlay/{key}",
                "",
                overlay_value(state, key),
                tooltip=tip,
            )
            if bool(hit[0] if isinstance(hit, tuple) else hit):
                set_overlay(state, key, not overlay_value(state, key))


#: Front / Right / Top and their opposites, with the chord that reaches each.
#: The three that had buttons were labelled ``F``/``R``/``T`` in a sidebar and
#: their backs were reachable only by holding Shift, which nothing said.
AXIS_ROWS: tuple[tuple[str, str, str], ...] = (
    ("front", "Front", "Ctrl+1"),
    ("back", "Back", "Ctrl+Shift+1"),
    ("right", "Right", "Ctrl+3"),
    ("left", "Left", "Ctrl+Shift+3"),
    ("top", "Top", "Ctrl+7"),
    ("bottom", "Bottom", "Ctrl+Shift+7"),
)


def _view_popup(ctx: Any, state: Any, view: Any) -> None:
    """Where the camera looks from, and how it projects.

    ``view`` is the live viewport and is ``None`` before it exists -- the first
    frame, and exactly when a pane must not raise. The rows that need it are
    disabled rather than absent, which is the house pattern and also the honest
    answer: they will work in a moment.
    """

    with controls.menu_popup(VIEW_POPUP) as opened:
        if not opened:
            return
        camera = None if view is None else view.camera
        for name, label, chord in AXIS_ROWS:
            hit = controls.menu_item(
                f"{label}##{BAR}/axis/{name}",
                chord,
                False,
                camera is not None,
                reason="The viewport is still starting up.",
            )
            if bool(hit[0] if isinstance(hit, tuple) else hit):
                camera.look_along(name)
        controls.menu_separator()
        ortho = bool(camera is not None and camera.orthographic)
        hit = controls.menu_item(
            f"Orthographic##{BAR}/ortho",
            "Ctrl+5",
            ortho,
            camera is not None,
            reason="The viewport is still starting up.",
            tooltip="No perspective, so parallel edges stay parallel -- which "
            "is what makes two parts line up by eye.",
        )
        if bool(hit[0] if isinstance(hit, tuple) else hit):
            camera.orthographic = not ortho
        hit = controls.menu_item(
            f"Frame the selection##{BAR}/frame", "F", False, True
        )
        if bool(hit[0] if isinstance(hit, tuple) else hit):
            # A flag rather than a call, the house pattern: framing needs the
            # viewport, which is a thing ``main`` owns.
            state.frame_pending = True


def _trailing(ctx: Any, state: Any, view: Any) -> Any:
    """``Overlays``, ``View`` and the (?), at the end of the row.

    A trailing block rather than three more items, for ``inker_context``'s
    reason: ``toolbar`` draws its items before its fields, so as items these
    would land between Snap and the mode picker -- in the middle of what you are
    editing, which is the one place a setting about the *viewport* must not be.

    The tuple form rather than :class:`toolbar.Trailing`: both are already
    single words and a glyph, so there is no smaller tier for them to fall to,
    and a block that cannot collapse says so by having one width.
    """

    style = imgui.get_style()
    gap = style.item_spacing.x
    pad = style.frame_padding.x * 2.0
    shading = sum(
        imgui.calc_text_size(short).x + pad for _key, _label, short, _tip in SHADING
    )
    width = (
        shading
        + gap
        + imgui.calc_text_size("X-ray").x
        + pad
        + gap
        + imgui.calc_text_size("Overlays").x
        + pad
        + gap
        + imgui.calc_text_size("View").x
        + pad
        + gap
        + imgui.get_frame_height()
    )

    def draw_it() -> None:
        # The shading pill first, because it is the one control here that
        # changes what the render *is* rather than what is drawn over it.
        changed, picked = controls.segmented_choice(
            "clay-shading",
            [(key, short) for key, _label, short, _tip in SHADING],
            state.shading,
            tooltips={key: f"{label} -- {tip}" for key, label, _s, tip in SHADING},
            compact=True,
        )
        if changed:
            state.shading = picked
        imgui.same_line()
        if controls.button(
            f"X-ray##{BAR}/xray",
            role=controls.ButtonRole.GHOST,
            control_size=controls.ControlSize.COMPACT,
            selected=bool(state.xray),
            tooltip="See through the surface, so an element behind it can be "
            "picked (Alt+Z)",
        ):
            state.xray = not state.xray
        imgui.same_line()
        if controls.button(
            f"Overlays##{BAR}/overlays",
            role=controls.ButtonRole.GHOST,
            control_size=controls.ControlSize.COMPACT,
            selected=any(
                overlay_value(state, key) != OVERLAY_DEFAULTS[key]
                for key, _label, _tip in OVERLAY_ROWS
            ),
            tooltip="What the viewport draws over the model",
        ):
            imgui.open_popup(OVERLAYS_POPUP)
        _overlays_popup(ctx, state)
        imgui.same_line()
        if controls.button(
            f"View##{BAR}/view",
            role=controls.ButtonRole.GHOST,
            control_size=controls.ControlSize.COMPACT,
            tooltip="Where the camera looks from, and how it projects",
        ):
            imgui.open_popup(VIEW_POPUP)
        _view_popup(ctx, state, view)
        imgui.same_line()
        manual_render.help_button_inline(ctx, "clay-header")

    return (width, draw_it)


def measure(state: Any, tab: Any) -> float:
    """What the row needs at its **full** tier, in physical pixels.

    For ``tests/test_clay_header.py``, which asks the one question a header can
    silently get wrong: does it fit at the window the app opens at. A bar that
    plans ICON at 1600x950 is a bar whose every label is a hover away on the
    machine everybody uses.
    """

    style = imgui.get_style()
    gap = style.item_spacing.x
    items = _items(state)
    fields = [_mode_field(tab), _tool_field(state)]
    with fonts.label(imgui):
        widths = [widgets.button_width(item.label) for item in items]
    widths += [sp(field.width) for field in fields]
    return sum(widths) + gap * (len(widths) - 1)
