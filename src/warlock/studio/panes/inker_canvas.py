"""Paint mode's centre pane: the file row, the tab bar and the canvas itself.

Descended from the inline editor's canvas, which is why the view maths, the
press/drag/release split and the texture gating look familiar -- those were the
parts that worked. What is new is that the pane is now driving a document with
selections and layers, so the drawing order matters: checkerboard, composite,
floating pixels, grid, symmetry axes, marching ants, then whatever the current
drag is previewing, then the brush cursor. Everything after the composite is an
overlay and none of it touches pixels.

The one non-obvious rule is that a *drag* never commits anything. A press
decides what the drag means, the drag updates a preview or walks the brush, and
the release is the only thing that writes an undoable step -- which is what
makes every gesture exactly one Ctrl+Z.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any

import numpy as np
from imgui_bundle import imgui

from .. import (
    ants,
    controls,
    fonts,
    icons,
    imgui_backend,
    inker_mode,
    inker_state,
    theme,
    toolbar,
    widgets,
)
from ..inker import STAMP_MODES, textstamp
from ..inker.document import catmull_rom, curve_points, curve_spans
from ..inker.indexed import shade_ramp
from ..inker.slices import SliceKey, slice_props
from ..inker.tiling import SEAM_MAX, axes_of, canonical, seam_ratio, tile_offset
from ..inker_state import (
    BG_BUTTON_TOOLS,
    PAINT_TOOLS,
    PATH_SHAPE_TOOLS,
    SELECT_TOOLS,
    SHAPE_TOOLS,
)
from ..tokens import sp
from . import inker_bridge, inker_context, inker_menu, inker_textures

#: The four positions of the Tiled control, one per :data:`TILED_AXES` entry.
#: Prefixed, because in a row of icon buttons a bare "X" says nothing about
#: what it is an axis of.
TILED_LABELS = (
    ("off", "Tiled: off"),
    ("x", "Tiled: X"),
    ("y", "Tiled: Y"),
    ("both", "Tiled: X+Y"),
)

#: The zoom range this pane holds its view to, splatted into every
#: ``inker_state`` view call. Inker's own, narrower than the module globals that
#: Plotter and Packwright share -- see ``inker_state.INKER_MIN_ZOOM``. One dict
#: rather than a keyword pair per call site, because the four calls have to
#: agree or a fit and a wheel notch would clamp to different numbers.
_BOUNDS = {"lo": inker_state.INKER_MIN_ZOOM, "hi": inker_state.INKER_MAX_ZOOM}


def _u32(colour: int, alpha: float = 1.0) -> int:
    return imgui.get_color_u32(imgui.ImVec4(*theme.rgba(colour, alpha)))


# --- drawing through the view's orientation (Ink9) ---------------------------
#
# Every overlay below goes through one of these three, and that is the whole of
# what makes canvas rotation and the flipped view safe. The failure the feature
# invites is not a crash: it is one overlay left computing its own screen
# position from ``origin + x * zoom``, which is right at rotation 0 and silently
# a quarter turn out everywhere else -- a grid drawn across a canvas that is not
# there, ants beside the mask they describe. Quarter turns are what keeps these
# three enough (see ``inker_state.ROTATIONS``): an axis-aligned image rectangle
# comes out an axis-aligned *screen* rectangle, so a rect is still a rect and a
# grid is still two families of straight lines.


def _corners(view: Any, origin, x0: float, y0: float, x1: float, y1: float):
    """An image-space rectangle's four corners on screen, in image order:
    top-left, top-right, bottom-right, bottom-left."""
    to = inker_state.to_screen
    return (
        to(view, origin, x0, y0),
        to(view, origin, x1, y0),
        to(view, origin, x1, y1),
        to(view, origin, x0, y1),
    )


def _box(view: Any, origin, x0: float, y0: float, x1: float, y1: float):
    """An image-space rectangle as a screen AABB ``(top_left, bottom_right)``.

    Sound only because the orientation is a quarter turn: it maps the corner
    set onto itself, so the min and max of the four transformed corners *are*
    two opposite corners of the same rectangle rather than a box around a
    tilted one.
    """
    points = _corners(view, origin, x0, y0, x1, y1)
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys)), (max(xs), max(ys))


def _blit(draw_list: Any, texture: Any, view: Any, origin, x0, y0, x1, y1, **kwargs) -> None:
    """One image drawn over an image-space rectangle, however the view is turned.

    ``add_image_quad`` rather than ``add_image``, because the two-corner form
    can only draw an upright rectangle: the quad's four *positions* carry the
    turn and the four uvs stay in the texture's own order, which is exactly a
    rotation of the picture and not a resampling of it.

    ``uv`` overrides the corner coordinates for the one caller that tiles
    (the checkerboard), in the same top-left/top-right/bottom-right/bottom-left
    order the positions are in. ``uv0``/``uv1`` is the two-corner spelling of
    the same thing, for the tiled composite: values outside 0..1 with the
    sampler set to repeat are what draw the 3x3 neighbourhood in one call
    rather than as nine images positioned by hand.
    """
    uv0, uv1 = kwargs.pop("uv0", None), kwargs.pop("uv1", None)
    if uv0 is not None or uv1 is not None:
        (u0, v0), (u1, v1) = uv0 or (0.0, 0.0), uv1 or (1.0, 1.0)
        kwargs.setdefault("uv", ((u0, v0), (u1, v0), (u1, v1), (u0, v1)))
    uv = kwargs.pop("uv", ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    colour = kwargs.pop("colour", None)
    a, b, c, d = _corners(view, origin, x0, y0, x1, y1)
    ref = widgets.texture_ref(texture)
    if colour is None:
        draw_list.add_image_quad(ref, a, b, c, d, *uv)
    else:
        draw_list.add_image_quad(ref, a, b, c, d, *uv, colour)


def draw(ctx: Any) -> None:
    state = inker_mode.ensure(ctx)
    # The application owns the menu bar; this child still owns the parameter
    # and document popups requested by those registry-backed rows.
    inker_menu.draw_popups(ctx)
    if not state.docs:
        _empty(ctx, state)
        # Registered in this window whichever branch drew, because the empty
        # state's "New custom size..." opens it by name and a popup belongs to
        # the window that begins it.
        new_popup(ctx)
        inker_bridge.popups(ctx)
        return
    _tab_bar(ctx, state)
    tab = state.active
    if tab is not None:
        # Menu, tabs, context bar, canvas, status: Aseprite's order, and the
        # order the eye already knows. The view row -- what the *page* is
        # doing, as against what the tool is set to -- rides the bar's trailing
        # block; see ``_view_row``.
        _view_row(ctx, state, tab)
        inker_context.draw(ctx, state, tab)
        if state.transforming:
            _transform_row(ctx, state, tab)
        _canvas(ctx, state, tab)
    new_popup(ctx)
    # The four dialogs the retired bridge panel kept, hosted here for the same
    # reason the menu is: they are opened by name from a menu row.
    inker_bridge.popups(ctx)


# --- the canvas's own row ----------------------------------------------------
#
# New/Open/Save/Save as/Export PNG used to be here, and moved to
# ``inker_bridge`` in the UI redesign, wave 4.2. That row was the worst clipping case
# in the app -- eight labelled buttons plus a combo plus a status word, chained
# with ``same_line``, losing "Export PNG" off the right edge at 150 % -- and it
# was also the one document mode whose file actions did *not* live in its bridge
# panel, where Plotter's and Packwright's do. Moving them fixed both at once.
#
# What stays is what acts on the canvas you are looking at: undo/redo (the
# bridge panel's pair was three panels away from the stroke it reversed), the
# two view turns, and the status word. Through ``toolbar`` now, so the row
# degrades instead of clipping.


def _view_row(ctx: Any, state: Any, tab: Any) -> None:
    """What the *page* is doing: the turns, the centring, the tiling.

    Undo and Redo were the first two items here and are menu rows now -- the
    only two verbs on this row that changed the drawing rather than the view,
    and the pair every user reaches for by chord anyway. What is left is one
    honest group: nothing here writes a pixel, which is why **none of it is
    gated on ``busy``** -- an editor that refuses to let you *look* at your
    drawing while it writes a file is not protecting anything.
    """
    view = tab.view
    items = [
        toolbar.Item("rotate", "Rotate view", icons.ROTATE_CW, tooltip="Rotate the view (Ctrl+4)"),
        toolbar.Item("flip", "Flip view", icons.FLIP_HORIZONTAL, tooltip="Flip the view (Ctrl+5)"),
        # Distinct from *Fit view* (Ctrl+0, and a button in the bridge panel):
        # this keeps the zoom the user chose and only puts the page back under
        # the pane. Panning far enough to lose the canvas entirely is easy and
        # the only way back was to re-fit, which threw the zoom away. (The
        # citation here used to say "the View menu"; this app has never had
        # one, and a reader looking for it found nothing.)
        toolbar.Item("center", "Center view", icons.CROSSHAIR, tooltip="Center the page"),
    ]
    if view.rotation or view.flipped:
        # Said out loud, because the two are invisible once you have looked
        # away for a moment and a mirrored canvas silently teaches the wrong
        # hand. No glyph, so it keeps its words at every tier -- the label *is*
        # the state it is reporting.
        parts = ([f"{view.rotation} deg"] if view.rotation else []) + (
            ["flipped"] if view.flipped else []
        )
        items.append(
            toolbar.Item(
                "upright",
                " + ".join(parts),
                tooltip="The view only -- click to set it upright",
            )
        )
    toolbar.toolbar(
        "inker-canvas",
        items,
        lambda key: _view_action(ctx, tab, key),
        trailing=_view_trailing(ctx, tab),
    )


def _view_action(ctx: Any, tab: Any, key: str) -> None:
    view = tab.view
    if key == "rotate":
        inker_state.rotate_view(view, 1)
    elif key == "flip":
        inker_state.flip_view(view)
    elif key == "center":
        # ``pending_zoom`` and not ``fitted = False``, exactly as ``rotate_view``
        # and ``flip_view`` do it: clearing ``fitted`` re-*scales* as well as
        # re-centring, which is the one thing this button must not do.
        view.pending_zoom = view.zoom
    elif key == "upright":
        view.rotation, view.flipped = 0, False
        view.pending_zoom = view.zoom


def _status(tab: Any) -> tuple[int, str] | None:
    """The one word at the end of the row, or nothing. Pure, so the trailing
    block can measure it before it draws it."""
    if tab.saving:
        return (theme.MUTED, "saving...")
    if tab.playing:
        return (theme.MUTED, "playing...")
    if tab.dirty:
        return (theme.WARN, "unsaved")
    return None


def seam_text(ctx: Any, tab: Any) -> tuple[int, str] | None:
    """The worst wrap-seam ratio for this document, as a coloured word.

    ``None`` when the tab is not tiled -- there is no seam to report on a
    drawing nobody is wrapping.

    **Cached against the undo serial** (``UndoStack.head``, a monotone serial
    rather than a depth). The measurement is a full-image diff, and computing
    one per frame is exactly the frame-thread stall the whole task layer exists
    to avoid; the answer cannot change unless the document has, and the serial
    is what says so.
    """
    if tab.tiled == "off":
        return None
    key = f"inker_seam:{tab.uid}"
    serial = tab.doc.history.head
    cached = ctx.state.preview.get(key)
    if cached is None or cached[0] != serial:
        try:
            worst = max(seam_ratio(tab.doc.flatten(matte=False)))
        except ValueError:
            return None
        cached = (serial, worst)
        ctx.state.preview[key] = cached
    worst = cached[1]
    colour = theme.WARN if worst > SEAM_MAX else theme.MUTED
    return (colour, f"seam x{worst:.1f}")


def _view_trailing(ctx: Any, tab: Any) -> tuple[float, Any]:
    """The tiling combo and the status word, as one non-collapsible block.

    ``toolbar`` subtracts this before it chooses tiers, which is what stops the
    combo being measured after the row has already claimed the space.
    """
    status = _status(tab)
    seam = seam_text(ctx, tab)
    gap = imgui.get_style().item_spacing.x
    width = sp(104) + (imgui.calc_text_size(status[1]).x + gap if status else 0.0)
    if seam is not None:
        width += sp(74) + gap + imgui.calc_text_size(seam[1]).x + gap

    def draw_it() -> None:
        # One control driving the view *and* the writes, deliberately: a canvas
        # that showed its neighbours while the brush went on clamping at the
        # edge would be a picture of a seamless tile you cannot paint.
        tab.tiled = widgets.combo("##inkertiled", tab.tiled, list(TILED_LABELS), sp(104))
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Draws the eight neighbouring tiles around this one, and wraps "
                "every stroke, fill and shape across the seams it shows."
            )
        if seam is not None:
            imgui.same_line()
            # The classic put-the-seam-in-the-middle move, and the reason it is
            # half rather than any amount: pressing it twice on even dimensions
            # is the identity, so it is a look rather than an edit you undo.
            if widgets.disabled_button("Wrap 1/2##inkerwrap", not tab.busy, (sp(74), 0)):
                width_px, height_px = tab.doc.size
                tab.doc.offset_layer(width_px // 2, height_px // 2)
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Rolls this layer half a canvas in both directions, putting "
                    "the wrap seam in the middle where you can paint over it."
                )
            imgui.same_line()
            widgets.text_colored(seam[0], seam[1])
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "How hard the wrap join is against the picture's own grain. "
                    f"Above {SEAM_MAX:.1f} it reads as a visible edge."
                )
        if status is not None:
            imgui.same_line()
            widgets.text_colored(status[0], status[1])

    return (width, draw_it)


def _transform_row(ctx: Any, state: Any, tab: Any) -> None:
    """Numeric handles for the same operations the drag handles do.

    A drag cannot express "exactly 90 degrees", and a rotation that is nearly
    square is worse than either -- so the buttons are not a convenience, they
    are the only way to get an exact one.

    **Apply is the row's one primary and Cancel its ghost**, both pinned: this
    row exists for as long as a transform is uncommitted, and the two ways out
    of that state are the last things that should move house when the pane
    narrows. The sliders go in ``trailing`` -- they are the numbers the buttons
    are about, and measuring them first is what stops the link checkbox at the
    end of them being drawn past the edge at 150 %.
    """
    doc = tab.doc
    widgets.text_colored(theme.ACCENT, "Transform")
    imgui.same_line()
    items = [
        toolbar.Item("fliph", "Flip H", icons.FLIP_HORIZONTAL, priority=1),
        toolbar.Item("flipv", "Flip V", priority=1),
        toolbar.Item("ccw", "-90", priority=1),
        toolbar.Item("cw", "+90", icons.ROTATE_CW, priority=1),
        toolbar.Item(
            "apply",
            "Apply",
            icons.CHECK,
            role=toolbar.ButtonRole.PRIMARY,
            pinned=True,
        ),
        toolbar.Item("cancel", "Cancel", icons.X, pinned=True),
    ]
    toolbar.toolbar(
        "inker-transform",
        items,
        lambda key: _transform_action(ctx, doc, key),
        trailing=_transform_trailing(state, doc),
    )
    buf = doc.floating
    if buf is not None and state.resample == "rotsprite":
        from ..inker import transform

        if not transform.rotsprite_fits(buf.size):
            # The standing version of the toast Ctrl+T raised once: a drag
            # re-renders every frame and cannot say this every time, but the
            # user is looking at this row the whole while.
            widgets.muted("Too big for RotSprite -- turning with nearest neighbour.")
    imgui.separator()


def _transform_action(ctx: Any, doc: Any, key: str) -> None:
    if key == "fliph":
        doc.flip_floating("horizontal")
    elif key == "flipv":
        doc.flip_floating("vertical")
    elif key == "ccw":
        doc.rotate_floating(-90.0)
    elif key == "cw":
        doc.rotate_floating(90.0)
    elif key == "apply":
        inker_mode.end_transform(ctx, commit=True)
    elif key == "cancel":
        inker_mode.end_transform(ctx, commit=False)


def _transform_trailing(state: Any, doc: Any) -> tuple[float, Any] | None:
    """Angle, the two scale axes and the link, measured before the buttons.

    None while there is no floating buffer: the transform mode is entered a
    frame before the buffer exists, and three sliders describing nothing is
    worse than a row that grows.
    """
    buf = doc.floating
    if buf is None:
        return None
    gap = imgui.get_style().item_spacing.x
    # Each slider is its item width plus its own label, and imgui puts the
    # label *outside* the box -- which is exactly the part every hand-counted
    # reservation in this app has forgotten.
    width = (
        sp(160)
        + sp(110) * 2
        + imgui.calc_text_size("Angle").x
        + imgui.calc_text_size("X").x
        + imgui.calc_text_size("Y").x
        + widgets.button_width("Link")
        + sp(110) * 2
        + imgui.calc_text_size("H").x
        + imgui.calc_text_size("V").x
        + gap * 7
    )

    def draw_it() -> None:
        imgui.set_next_item_width(sp(160))
        changed, angle = controls.slider_float("Angle", buf.angle, -180.0, 180.0, "%.1f deg")
        if changed:
            doc.transform_floating(angle=angle, resample=state.resample)
        imgui.same_line()
        # Two sliders and a link, rather than the one that used to drive both
        # axes from ``scale[0]``: the engine has taken a per-axis scale all
        # along and the panel was the thing that could not express it.
        # "%.2fx", not imgui's default "%.3f": a scale factor is a multiplier
        # and reads as one with the unit attached, where a bare ``1.000``
        # beside an angle in degrees is a number with no stated dimension.
        imgui.set_next_item_width(sp(110))
        changed_x, fx = controls.slider_float("X##inkscalex", buf.scale[0], 0.05, 8.0, "%.2fx")
        imgui.same_line()
        imgui.set_next_item_width(sp(110))
        changed_y, fy = controls.slider_float("Y##inkscaley", buf.scale[1], 0.05, 8.0, "%.2fx")
        imgui.same_line()
        linked, value = controls.checkbox("Link##inkscalelink", state.transform_link)
        if linked:
            state.transform_link = value
        if imgui.is_item_hovered():
            imgui.set_tooltip("Scale both axes together. Shift does the same on a handle.")
        if changed_x or changed_y:
            if state.transform_link:
                fx = fy = fx if changed_x else fy
            doc.transform_floating(scale=(fx, fy), resample=state.resample)
        # **Skew, as two numbers** (6.4). The engine has taken a shear since
        # the free transform landed and nothing in the app could express one,
        # so a document could be sheared by a script and by nothing a user
        # could reach. Numbers rather than handles because the corner handles
        # are already scale and rotate, and a third meaning on the same eight
        # squares is a gesture nobody can aim -- the same argument the plan
        # makes for suppressing splitters while the layout editor is open.
        imgui.same_line()
        imgui.set_next_item_width(sp(110))
        changed_h, hx = controls.slider_float("H##inkshearx", buf.shear[0], -60.0, 60.0, "%.0f deg")
        imgui.same_line()
        imgui.set_next_item_width(sp(110))
        changed_v, hy = controls.slider_float("V##inksheary", buf.shear[1], -60.0, 60.0, "%.0f deg")
        if changed_h or changed_v:
            doc.transform_floating(shear=(hx, hy), resample=state.resample)

    return (width, draw_it)


def new_popup(ctx: Any) -> None:
    """The presets, and the custom size the presets could not express.

    The other half of the 3x3 anchor grid's popup: a resize has taken typed
    width and height since it was written, and creating one could only ever
    offer three squares -- so a user who wanted 1920x1080 had to make a square
    and then resize it, which is two undo steps and a guess about the anchor.

    The fields are remembered per session in ``state.preview`` rather than per
    document, because there is no document yet -- and because reopening the
    dialog after a mistake should offer the number that was nearly right.

    **Public, and drawn by two panes.** A popup belongs to the window that
    begins it, so the bridge panel's New button (the UI redesign, wave 4.2) cannot
    open one registered here: each pane registers its own and opens that one.
    The body is shared rather than copied, which is the whole reason this is
    not a private helper.
    """
    if not imgui.begin_popup("new-canvas"):
        return
    widgets.popup_chrome(_imgui=imgui)
    imgui.text("New canvas")
    for width, height in inker_mode.NEW_PRESETS:
        if controls.button(f"{width} x {height}", (sp(160), 0)):
            inker_mode.new_document(ctx, width, height)
            imgui.close_current_popup()

    imgui.separator()
    key = "inker_new_size"
    width, height = ctx.state.preview.get(key) or inker_mode.NEW_PRESETS[1]
    imgui.set_next_item_width(sp(72))
    changed_w, width = controls.input_int("W##newcanvas", int(width), 0)
    imgui.same_line()
    imgui.set_next_item_width(sp(72))
    changed_h, height = controls.input_int("H##newcanvas", int(height), 0)
    if changed_w or changed_h:
        # Clamped on the way *into* the field as well as on the way out, or the
        # box goes on showing a number that is not what the button would make.
        ctx.state.preview[key] = inker_mode.clamp_canvas(width, height)
    width, height = inker_mode.clamp_canvas(width, height)
    if controls.button(f"Create {width} x {height}", (sp(160), 0)):
        inker_mode.new_document(ctx, width, height)
        imgui.close_current_popup()
    widgets.muted(f"up to {inker_mode.NEW_MAX} px a side")
    imgui.end_popup()


def _empty(ctx: Any, state: Any) -> None:
    from pathlib import Path

    # The presets first, then the two escapes from them. The one *at the top*
    # is the primary and the rest are ghosts (``widgets.nothing_open``'s rule):
    # this screen used to draw five identical buttons, which is a menu claiming
    # every entry matters equally when four of them are variations on one.
    actions = [
        (
            f"New {width} x {height}",
            lambda w=width, h=height: inker_mode.new_document(ctx, w, h),
        )
        for width, height in inker_mode.NEW_PRESETS
    ]
    actions += [
        # The same popup the bridge panel's New button opens, rather than a
        # second set of fields. ``draw`` registers it in *this* window on the
        # empty branch too, which is what makes opening it by name from here
        # enough.
        ("New custom size...", lambda: imgui.open_popup("new-canvas")),
        ("Open a file...", lambda: inker_mode.ask_open(ctx)),
    ]
    widgets.nothing_open(
        "Start a canvas, open an image, or send one here from the library.",
        actions,
        recent_paths=inker_mode.recent_paths(ctx),
        on_open=lambda path: inker_mode.open_path(ctx, Path(path)),
    )


# --- tabs -------------------------------------------------------------------


def _tab_bar(ctx: Any, state: Any) -> None:
    flags = imgui.TabBarFlags_.reorderable.value | imgui.TabBarFlags_.auto_select_new_tabs.value
    if not imgui.begin_tab_bar("inker-tabs", flags):
        return
    for tab in list(state.docs):
        item_flags = 0
        if tab.dirty:
            # imgui's own dot, rather than a " *" in the title: the title is
            # also the tab's identity, and decorating it would move the tab.
            item_flags |= imgui.TabItemFlags_.unsaved_document.value
        opened, keep = imgui.begin_tab_item(tab.label, True, item_flags)
        if opened:
            state.activate(tab.uid)
            imgui.end_tab_item()
        if not keep:
            inker_mode.request_close(ctx, tab)
    imgui.end_tab_bar()


# --- the canvas -------------------------------------------------------------


def _canvas(ctx: Any, state: Any, tab: Any) -> None:
    flags = imgui.WindowFlags_.no_scroll_with_mouse.value | imgui.WindowFlags_.no_scrollbar.value
    hovered = False
    origin = None
    # A positive height, never a bottom offset: with little room left a "-26"
    # child collapses to nothing and the canvas (and its texture uploads)
    # silently stops being drawn.
    height = max(imgui.get_content_region_avail().y - sp(26), sp(16))
    # The canvas is the *document*, not chrome: the global pane radius
    # (``tokens.RADIUS_PANE``) would clip the artwork's four corners.
    imgui.push_style_var(imgui.StyleVar_.child_rounding.value, 0.0)
    if imgui.begin_child("inker-canvas", (0, height), imgui.ChildFlags_.borders.value, flags):
        origin = imgui.get_cursor_screen_pos()
        avail = imgui.get_content_region_avail()
        region = (max(avail.x, 16.0), max(avail.y, 16.0))
        view = tab.view
        if view.pending_zoom is not None:
            inker_state.centre(view, tab.doc.size, region, view.pending_zoom, **_BOUNDS)
            view.pending_zoom = None
        elif not view.fitted:
            inker_state.fit(view, tab.doc.size, region, **_BOUNDS)
        # The right button is taken as well as the left: it paints with the
        # background colour (C12d) and the canvas has no context menu to
        # conflict with. Without the flag imgui simply never reports the press.
        imgui.invisible_button(
            "##inker-surface",
            region,
            imgui.ButtonFlags_.mouse_button_left.value
            | imgui.ButtonFlags_.mouse_button_right.value,
        )
        active = imgui.is_item_active()
        hovered = imgui.is_item_hovered()
        if view.pending_zoom_rung:
            # Anchored on the cursor while it is over the canvas and on the
            # middle of the pane while it is not: a keyboard zoom must not
            # depend on where the mouse happens to be resting off-pane, and
            # zooming about a cursor two panes away throws the page off screen.
            mouse = imgui.get_mouse_pos()
            focus = (
                (mouse.x, mouse.y)
                if hovered
                else (origin.x + region[0] / 2.0, origin.y + region[1] / 2.0)
            )
            inker_state.zoom_ladder_step(
                view, (origin.x, origin.y), focus, view.pending_zoom_rung, **_BOUNDS
            )
            view.pending_zoom_rung = 0
        _input(ctx, state, tab, (origin.x, origin.y), active=active, hovered=hovered)
        _paint(ctx, state, tab, (origin.x, origin.y), hovered=hovered)
        # After everything ``_paint`` draws, because the bands sit on top of
        # the canvas -- and outside it, because ``_paint`` early-outs when
        # there is no composite yet and the rulers should not blink with it.
        if state.rulers:
            _rulers(tab, (origin.x, origin.y), region, hovered=hovered)
        # Inside the child, because that is where ``_press`` opened it from and
        # an imgui popup's id is computed off the id stack it was opened on.
        _text_popup(ctx, state, tab)
    else:
        # A keyboard zoom aimed at a canvas that did not draw this frame is
        # dropped, not banked: ``pending_zoom_rung`` is set by ``handle_key``
        # from outside the canvas, so left alone here it survives the frame
        # and fires a zoom rung later, unprompted. ``pending_zoom`` cannot do
        # this -- it is only ever set from inside a visible canvas.
        tab.view.pending_zoom_rung = 0
    imgui.end_child()
    imgui.pop_style_var()
    _status_bar(ctx, state, tab, origin, hovered)


#: The tools whose gesture is sized by the brush slider, and therefore the
#: ones whose size is worth a place in the status bar. The shapes are in it
#: because ``doc.shape`` takes the same width -- which is the thing that was
#: invisible: a line tool has a stroke width and nothing on screen said so.
_SIZED_TOOLS = PAINT_TOOLS | SHAPE_TOOLS


def _os_cursor(state: Any, tab: Any, *, hovered: bool) -> None:
    """What the pointer itself says about the gesture that is available.

    The pointer never changed shape over this canvas -- not while space-panning,
    not over the move tool, and not over a **locked layer**, which is the one
    that mattered: the refusal was a toast raised *after* a press had already
    been made, so the way to find out a layer was locked was to try to draw on
    it. The pointer is the only surface that can say so before the click.

    Set per frame while hovered, because imgui resets the cursor every frame;
    left alone otherwise, so the rest of the window keeps its own.
    """
    if not hovered:
        return
    if state.space_held or state.drag_kind == "pan":
        imgui.set_mouse_cursor(imgui.MouseCursor_.hand.value)
        return
    # The same question ``_locked_out`` asks at the press, and deliberately the
    # same exemptions: a pick or a marquee over a locked layer is not refused,
    # so it must not be shown as refused either.
    nudging_a_float = state.tool == "move" and tab.doc.floating is not None
    if state.tool not in _READ_ONLY_TOOLS and tab.doc.write_locked() and not nudging_a_float:
        imgui.set_mouse_cursor(imgui.MouseCursor_.not_allowed.value)
        return
    if state.tool == "move":
        imgui.set_mouse_cursor(imgui.MouseCursor_.resize_all.value)


def _cursor_pixel(state: Any, tab: Any, origin, hovered: bool):
    """The pixel under the cursor and its colour, or ``(None, None)``.

    ``origin`` is the pane's top-left as an ``(x, y)`` pair -- the shape every
    other view helper in this module takes, so the two callers cannot hand it
    two different things.

    In-bounds as well as hovered. The invisible button covers the whole child,
    so the old readout answered for the dead space *around* the page too, and
    with tiling off ``canonical`` folds nothing -- which is how a 256-wide
    document came to report "-37, 412" as a position.
    """
    if not hovered or origin is None:
        return None, None
    mouse = imgui.get_mouse_pos()
    px, py = inker_state.to_image(tab.view, origin, mouse.x, mouse.y)
    # Folded onto the canonical tile: over a neighbour in the 3x3 view the
    # raw number is off the canvas, and a readout saying "300, 40" on a
    # 256-wide document is a coordinate the user cannot use.
    px, py = canonical((px, py), tab.doc.size, axes_of(tab.tiled))
    point = (int(px), int(py))
    if not tab.doc.in_bounds(point):
        return None, None
    # Through the same ``sample_layer`` a pick would use, so the swatch is a
    # promise about what Alt+click is *going* to give rather than a second
    # opinion about the same pixel.
    return point, tab.doc.eyedrop(point, layer_only=state.sample_layer)


def _zoom_combo(tab: Any) -> None:
    """A zoom the user can *set*, rather than only nudge -- Plotter's control.

    Writes ``tab.view.pending_zoom``, which already exists and is already
    consumed by the canvas: only the canvas knows how big the pane is, so a
    combo cannot centre on its own.

    The width is explicit because ``widgets.combo`` defaults to -1, which is
    the whole remaining row -- the same bar-across-the-status-line the
    screenshot pass caught in Plotter. The tooltip is the accessible name for
    a ``##``-hidden combo, for the same reason.

    Keys come from ``inker_state.zoom_key`` rather than from an int round
    trip: this list holds 12.5%, and ``int(picked) / 100`` reads it as 12%.
    """
    imgui.same_line()
    current = inker_state.zoom_key(tab.view.zoom)
    options = [
        (inker_state.zoom_key(z), f"{inker_state.zoom_key(z)}%")
        for z in inker_state.ZOOM_PRESETS
    ]
    if current not in {key for key, _label in options}:
        # The wheel's 5% notches land between presets constantly, and a combo
        # showing a preset the view is not at would be the pane lying about
        # itself.
        options = [(current, f"{current}%"), *options]
        options.sort(key=lambda option: float(option[0]))
    picked = widgets.combo("##inker-zoom", current, options, sp(84), tooltip="Zoom")
    if picked != current:
        tab.view.pending_zoom = float(picked) / 100.0


def _status_bar(ctx: Any, state: Any, tab: Any, origin: Any, hovered: bool) -> None:
    """The numbers a paint program keeps under the canvas.

    It held four -- zoom, position, tool, document size -- and the four it did
    not hold were the ones a drawing session actually asks for: **the colour
    under the cursor** (the eyedropper's only feedback was a swatch in another
    pane), **the brush size** (``[`` and ``]`` changed a number nothing
    displayed), **the size of the selection** (``SelectionMask.bounds`` has
    always been there and no pane read it), and **which layer and frame are
    being drawn into** -- both of which were a highlight in a panel that can be
    scrolled away from, on the far side of the window from the cursor.

    In Aseprite's order, because that is the order the eye already knows:
    what is under the cursor, then what is selected, then the tool, then the
    document, then the zoom.
    """
    doc = tab.doc
    # The status bar is drawn outside the child, so it holds imgui's own cursor
    # object rather than the pair every view helper takes.
    at = None if origin is None else (origin.x, origin.y)
    point, picked = _cursor_pixel(state, tab, at, hovered)
    if picked is not None:
        imgui.color_button(
            "##inkerpick",
            imgui.ImVec4(*(channel / 255.0 for channel in picked)),
            imgui.ColorEditFlags_.alpha_preview_half.value,
            (sp(11), sp(11)),
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "The colour under the cursor -- what Alt+click would pick up.\n"
                f"#{picked[0]:02X}{picked[1]:02X}{picked[2]:02X}"
                + ("" if picked[3] == 255 else f" at {picked[3] * 100 // 255}% alpha")
            )
        imgui.same_line()
    parts = []
    if point is not None:
        parts.append(f"{point[0]}, {point[1]}")
    mask = doc.mask
    box = mask.bounds if mask is not None else None
    if box is not None:
        # The bounding box rather than a pixel count: ``bounds`` is cached
        # against the mask and a count is a full-canvas walk, which is not a
        # thing to do sixty times a second for a readout.
        parts.append(f"sel {box[2] - box[0]} x {box[3] - box[1]}")
    tool = inker_state.tool_label(state.tool)
    parts.append(f"{tool} {state.brush_size}" if state.tool in _SIZED_TOOLS else tool)
    active = doc.stack.active
    if active is not None and active.name:
        parts.append(active.name)
    if doc.anim is not None and doc.anim.frames:
        parts.append(f"frame {doc.anim.current + 1}/{len(doc.anim.frames)}")
    parts.append(f"{doc.size[0]} x {doc.size[1]}")
    widgets.muted("   ".join(parts))
    # The zoom is split out of the joined string rather than appended to it,
    # because it is the one number here the user can *set*. It stays last, so
    # the documented Aseprite reading order -- cursor, selection, tool,
    # document, zoom -- is unchanged.
    _zoom_combo(tab)
    tip = state.tip
    if tip is not None and not tip.alive():
        # Cleared on the frame it expires rather than on a timer: this is the
        # only code that knows the tip is on screen, and a tip nobody drew has
        # nothing to expire.
        state.tip = None
    elif tip is not None:
        # After the readouts, not instead of them. The tip is the loud thing
        # for six seconds; the numbers under the cursor are what the session
        # is actually made of, and a tip that hid them would trade one missing
        # readout for four.
        imgui.same_line()
        widgets.secondary(tip.text)
        if tip.remedy:
            imgui.same_line()
            _remedy(ctx, state, tip)


def _remedy(ctx: Any, state: Any, tip: Any) -> None:
    """The one button a tip is allowed to offer: an op, by name.

    Looked up rather than held, and dropped silently if the name has gone --
    a tip is transient and an op is not, so a stale name means the remedy was
    retired between the press and the frame, which is not worth a second
    message about.
    """
    from .. import inker_ops

    try:
        op = inker_ops.get(tip.remedy)
    except KeyError:  # pragma: no cover - a retired op named by an old tip
        return
    if widgets.ghost_button(f"{tip.remedy_label or op.label}##inkertip"):
        inker_ops.run(ctx, op)
        state.tip = None


# --- input ------------------------------------------------------------------


def _input(ctx: Any, state: Any, tab: Any, origin, *, active: bool, hovered: bool) -> None:
    io = imgui.get_io()
    mouse = imgui.get_mouse_pos()
    point = inker_state.to_image(tab.view, origin, mouse.x, mouse.y)

    if hovered and io.mouse_wheel:
        # Back to physical notches before stepping: the backend halves every
        # wheel event so lists scroll at a sane rate, and this pane wants the
        # count the user's hand made, not a fraction of it.
        notches = io.mouse_wheel / imgui_backend.WHEEL_SCALE
        if state.tool == "rect" and imgui.is_key_down(imgui.Key.c):
            # **Hold C and roll while dragging a rectangle**: Aseprite's own
            # corner-radius gesture. On the wheel rather than on a drag of its
            # own because the hand is already holding the shape out, and the
            # only spare axis is the one the wheel turns.
            state.corner_radius = max(0, int(state.corner_radius) + int(notches))
        else:
            inker_state.zoom_step(tab.view, origin, (mouse.x, mouse.y), notches, **_BOUNDS)

    _os_cursor(state, tab, hovered=hovered)

    # Middle-drag always pans; space-drag pans with the left button, which is
    # what every paint program does and what makes a tablet usable.
    panning = imgui.is_mouse_dragging(2) or (state.space_held and imgui.is_mouse_dragging(0))
    if (hovered or state.drag_kind == "pan") and panning:
        button = 2 if imgui.is_mouse_dragging(2) else 0
        delta = imgui.get_mouse_drag_delta(button)
        imgui.reset_mouse_drag_delta(button)
        state.drag_kind = "pan"
        tab.view.pan = (tab.view.pan[0] + delta.x, tab.view.pan[1] + delta.y)
        return
    if state.drag_kind == "pan" and not (imgui.is_mouse_down(2) or imgui.is_mouse_down(0)):
        state.drag_kind = ""

    if tab.busy:
        # A save is encoding the live document; letting a stroke land in the
        # middle of it would put half a stroke in the file. Playback is the
        # other half of ``busy``: the canvas is showing a cached flatten of some
        # other frame, so a stroke would land invisibly on the frame underneath.
        #
        # A half-drawn multi-click gesture is dropped rather than merely
        # suspended: its next click cannot be delivered while this returns
        # early, so leaving the vertices up would draw a polygon over a document
        # that is being encoded or played and finish it whenever the tab came
        # back -- against whatever the document looked like by then.
        state.clear_gesture()
        return
    if state.transforming and tab.doc.floating is not None:
        _transform_input(state, tab, origin, point, active=active)
        return
    pressed = 0 if imgui.is_mouse_clicked(0) else 1 if imgui.is_mouse_clicked(1) else -1
    # **A press is refused while a gesture owns the mouse.** Two buttons is
    # exactly what C12d made possible and what nothing here could produce
    # before it: the right button coming down mid-left-drag used to reach
    # ``_press``, whose inert and Alt-pick arms clear ``drag_kind`` and return
    # -- abandoning the open gesture in place. A blur or spray stroke was left
    # with its pixels written and no step (pushed out of band by the *next*
    # stroke's ``end_stroke``), and a layer move was left previewed, undoable
    # by nothing and restorable from a snapshot the next ``begin_layer_move``
    # would have rolled the layer back to. One button owns a gesture from press
    # to release; the other one does nothing until it is over.
    holding = bool(state.drag_kind) and imgui.is_mouse_down(state.drag_button)
    if active and pressed >= 0 and not state.space_held and not holding:
        if state.drag_kind:
            # A gesture whose own button is already up but whose release has
            # not been dispatched yet -- press and release inside one frame.
            # Closed here, with the *previous* gesture's tile offset still in
            # ``state``, so it commits where it was drawn rather than being
            # orphaned by the press below.
            _release(ctx, state, tab, _snapped(state, _local(state, point)))
        state.drag_button = pressed
        # The tile the press landed in, fixed for the whole gesture. See
        # ``tiling.tile_offset``: folding per point would jump the brush a full
        # tile at the seam.
        state.tile_offset = tile_offset(point, tab.doc.size, axes_of(tab.tiled))
        # ``origin`` rides along for the slice tool alone: its handles are
        # hit-tested in screen space (``_slice_grab``), which the point on its
        # own cannot answer.
        _press(ctx, state, tab, _snapped(state, _local(state, point)), origin)
    elif state.drag_kind and imgui.is_mouse_down(state.drag_button):
        _drag(state, tab, _snapped(state, _local(state, point)))
    elif state.drag_kind and not imgui.is_mouse_down(state.drag_button):
        _release(ctx, state, tab, _snapped(state, _local(state, point)))


def _local(state: Any, point: tuple[float, float]) -> tuple[float, float]:
    """A cursor position in the *pressed* tile's coordinates.

    The whole of what the 3x3 view costs the input path, and it is a
    subtraction rather than a modulo for the reason above: within one gesture
    the offset never changes, so a stroke that crosses a seam carries on in a
    straight line and the engine wraps it.
    """
    return (point[0] - state.tile_offset[0], point[1] - state.tile_offset[1])


def _snapped(state: Any, point: tuple[float, float]) -> tuple[float, float]:
    """The cursor, on a grid intersection, for the tools that want one.

    Applied here rather than inside each tool so there is one answer to "where
    did the user click" -- a press that snapped and a release that did not
    would draw a rectangle whose far corner is off the grid.

    **Never for a freehand stroke.** Quantising a brush to a 16-pixel lattice
    is not a drawing aid, it is a different tool, and the paint tools are the
    ones a grid is most likely to be switched on around.
    """
    if not (state.grid and state.grid_snap):
        return point
    if state.tool in PAINT_TOOLS or state.tool in ("lasso", "wand", "eyedropper"):
        return point
    step = max(2, int(state.grid_size))
    return (round(point[0] / step) * step, round(point[1] / step) * step)


# Corner handles, in screen pixels.
HANDLE = 5.0
# How far above the box the rotate handle floats.
ROTATE_ARM = 28.0
# The radial symmetry pivot's crosshair, in screen pixels before ``sp``. A
# mirror shows as a line across the page; a rotation has no line to draw, only
# the point it turns about, so the guide is a small ring rather than nothing.
SYMMETRY_PIVOT_RADIUS = 7.0


#: Which axes each grab point scales. The corners take both, the four edge
#: handles take one -- which is the whole of what makes the scale non-uniform,
#: and it is a table rather than a chain of ``if``s so a handle drawn by
#: ``_transform_box`` cannot be one the drag code has no opinion about. The
#: names are image-space compass points, as ``_handles`` builds them.
HANDLE_AXES = {
    "nw": "xy",
    "ne": "xy",
    "sw": "xy",
    "se": "xy",
    "n": "y",
    "s": "y",
    "e": "x",
    "w": "x",
}


def _handles(tab: Any, origin) -> dict[str, tuple[float, float]]:
    """Where the transform box's grab points are, in screen space."""
    buf = tab.doc.floating
    x, y = buf.offset
    width, height = buf.size
    view = tab.view
    corners = {
        "nw": (x, y),
        "ne": (x + width, y),
        "sw": (x, y + height),
        "se": (x + width, y + height),
        # The edge midpoints, which is what a per-axis scale is grabbed by.
        "n": (x + width / 2.0, y),
        "s": (x + width / 2.0, y + height),
        "w": (x, y + height / 2.0),
        "e": (x + width, y + height / 2.0),
    }
    out = {k: inker_state.to_screen(view, origin, *p) for k, p in corners.items()}
    # The arm points away from the box's top edge *in the image*, carried on to
    # screen -- not straight up, which is only the same thing while the page is
    # upright and puts the handle inside the box at 180 degrees.
    top = inker_state.to_screen(view, origin, x + width / 2.0, y)
    above = inker_state.to_screen(view, origin, x + width / 2.0, y - 1.0)
    dx, dy = above[0] - top[0], above[1] - top[1]
    length = math.hypot(dx, dy) or 1.0
    out["rotate"] = (top[0] + dx / length * ROTATE_ARM, top[1] + dy / length * ROTATE_ARM)
    return out


def _transform_input(state: Any, tab: Any, origin, point, *, active: bool) -> None:
    """Drag a handle to scale, the arm to rotate, the middle to move.

    Every drag is measured against what was true at the *press*, not against
    the previous frame: accumulating per-frame deltas makes a slow drag and a
    fast one produce different results, and a scale that compounds frame by
    frame runs away.
    """
    doc = tab.doc
    buf = doc.floating
    mouse = imgui.get_mouse_pos()
    centre = inker_state.to_screen(tab.view, origin, *buf.centre)

    if active and imgui.is_mouse_clicked(0):
        handles = _handles(tab, origin)
        grab = min(handles, key=lambda k: math.dist(handles[k], (mouse.x, mouse.y)))
        near = math.dist(handles[grab], (mouse.x, mouse.y)) <= HANDLE * 2.5
        state.drag_anchor = point
        state.transform_grab = grab
        # Both axes' reference distances are in *image* space rather than
        # screen space, and that is what makes a per-axis scale correct under a
        # turned page: the view's rotation swaps which screen axis an image
        # axis lands on, so measuring "how far across" on screen would scale
        # the wrong one at 90 degrees. The screen distance stays beside them
        # for the two gestures that are genuinely about the screen -- the
        # uniform ratio and the rotate bearing.
        cx, cy = buf.centre
        state.transform_ref = (
            buf.scale[0],
            buf.scale[1],
            buf.angle,
            max(1.0, math.dist(centre, (mouse.x, mouse.y))),
            math.degrees(math.atan2(mouse.y - centre[1], mouse.x - centre[0])),
            max(1e-3, abs(point[0] - cx)),
            max(1e-3, abs(point[1] - cy)),
        )
        if near and grab == "rotate":
            state.drag_kind = "rotate"
        elif near:
            state.drag_kind = "scale"
        elif buf.contains((int(point[0]), int(point[1]))):
            state.drag_kind = "move"
        else:
            state.drag_kind = ""
        state.last_point = point
        return

    if not state.drag_kind or state.transform_ref is None:
        return
    if not imgui.is_mouse_down(0):
        state.drag_kind = ""
        state.transform_ref = None
        return

    scale0x, scale0y, angle0, dist0, bearing0, ref_x, ref_y = state.transform_ref
    if state.drag_kind == "scale":
        cx, cy = buf.centre
        axes = HANDLE_AXES.get(state.transform_grab, "xy")
        fx = abs(point[0] - cx) / ref_x if "x" in axes else 1.0
        fy = abs(point[1] - cy) / ref_y if "y" in axes else 1.0
        if imgui.get_io().key_shift:
            # Shift constrains to uniform. For a corner that is the screen
            # distance ratio (the same number the drag used to produce before
            # there were two axes); for an edge handle there is only one live
            # ratio, so it is simply applied to both.
            fx = fy = math.dist(centre, (mouse.x, mouse.y)) / dist0 if axes == "xy" else fx * fy
        doc.transform_floating(scale=(scale0x * fx, scale0y * fy), resample=state.resample)
    elif state.drag_kind == "rotate":
        bearing = math.degrees(math.atan2(mouse.y - centre[1], mouse.x - centre[0]))
        step = bearing0 - bearing  # screen y grows downward; the engine's does not
        if tab.view.flipped:
            # A rotation of the *view* cancels in a difference of bearings, but
            # a mirror does not: dragging the handle clockwise on a flipped page
            # is anticlockwise on the drawing, and without this the buffer turns
            # the opposite way to the cursor.
            step = -step
        if imgui.get_io().key_shift:
            step = round(step / 15.0) * 15.0
        doc.transform_floating(angle=angle0 + step, resample=state.resample)
    elif state.drag_kind == "move":
        last = state.last_point or point
        doc.move_floating(round(point[0] - last[0]), round(point[1] - last[1]))
        state.last_point = point


def _transform_box(state: Any, tab: Any, draw_list: Any, origin) -> None:
    buf = tab.doc.floating
    if buf is None:
        return
    handles = _handles(tab, origin)
    colour = _u32(theme.ACCENT)
    # The screen box of the four corners rather than the ``nw``/``se`` pair:
    # those names are image-space, and a turned page makes ``nw`` the bottom
    # right of what is on screen, which ``add_rect`` draws as nothing at all.
    a, b = _box(
        tab.view,
        origin,
        buf.offset[0],
        buf.offset[1],
        buf.offset[0] + buf.size[0],
        buf.offset[1] + buf.size[1],
    )
    draw_list.add_rect(a, b, colour)
    top = inker_state.to_screen(tab.view, origin, buf.offset[0] + buf.size[0] / 2.0, buf.offset[1])
    draw_list.add_line(top, handles["rotate"], colour)
    for name, point in handles.items():
        if name == "rotate":
            draw_list.add_circle_filled(point, HANDLE, colour)
        else:
            draw_list.add_rect_filled(
                (point[0] - HANDLE, point[1] - HANDLE),
                (point[0] + HANDLE, point[1] + HANDLE),
                colour,
            )


def _combine_op() -> str:
    """Shift adds, Alt subtracts, both intersect -- the universal convention.

    The pair is checked *first*, or the Shift branch answers it and the fourth
    of ``selection.COMBINE_OPS`` stays unreachable, which is what it was.
    """
    io = imgui.get_io()
    if io.key_shift and io.key_alt:
        return "intersect"
    if io.key_shift:
        return "add"
    if io.key_alt:
        return "subtract"
    return "replace"


# The tools that read the document rather than write to it, so a content lock
# has nothing to say about them: the eyedropper samples, and the four selection
# tools build a mask that lives on the document rather than in a layer.
_READ_ONLY_TOOLS = frozenset({"eyedropper"}) | SELECT_TOOLS


def _group_shown(doc: Any) -> bool:
    """Whether the groups above the active layer let it show at all.

    Read off the same fold the compositor folds in (``LayerStack._entries``):
    a layer inside a hidden group keeps its own eye on, so ``active.visible``
    alone said "shown" about a stroke the composite would never draw. Missing
    rows are treated as shown for ``_entries``' reason -- a fold caught
    mid-rebuild must cost a missing warning at worst, never a wrong one.
    """
    fold = doc.stack.group_fold
    if fold is None:
        return True
    index = doc.stack.active_index
    if index >= len(fold):
        return True
    return bool(fold[index][0])


def _locked_out(ctx: Any, state: Any, tab: Any) -> bool:
    """One toast per press when the active layer refuses tool-level writes.

    The toast lives here rather than at the engine's doors for the reason the
    engine refuses silently: ``write_colour`` is asked once per dab and once per
    preview frame, and this is the only place that knows a press is a press.
    """
    doc = tab.doc
    if state.tool in _READ_ONLY_TOOLS:
        return False
    if not doc.write_locked():
        # **Hidden is not refused, but it is no longer silent.** There is no
        # visibility check at any door: the stroke lands, the composite does
        # not show it, and the only thing on screen that could explain it is
        # an eye icon in a different pane. Painting under a hidden layer to
        # unhide it later is a real workflow, so the answer is a sentence
        # rather than a refusal -- and coalesced through ``toast_once`` rather
        # than raised per press: a multi-click shape tool presses once per
        # vertex and a stroke-per-second painter presses without limit, and
        # both were stacking one identical toast per press for as long as
        # they went on.
        if not doc.stack.active.visible:
            state.say(
                "That layer is hidden, so nothing you paint on it will show.",
                remedy="show_layer",
                remedy_label="Show it",
            )
        elif not _group_shown(doc):
            state.say(
                "That layer is inside a hidden group, so nothing you paint on "
                "it will show. The group's eye is on its timeline row."
            )
        return False
    # Nudging a buffer that is already floating writes to no layer -- the hole
    # it came out of was cut before the lock went on, and ``commit_floating``
    # is deliberately not refused either.
    if state.tool == "move" and doc.floating is not None:
        return False
    state.say(inker_mode.LOCKED_LAYER, remedy="layer_properties", remedy_label="Unlock")
    state.drag_kind = ""
    return True


def tile_cell(doc: Any, point) -> tuple[int, int] | None:
    """The grid cell of the active tilemap layer a canvas point falls in.

    ``None`` when the active row is not a tilemap, when it is bound to nothing,
    or when the point is off the canvas -- the three answers the cursor and the
    stamp both need, asked once so the outline drawn under the mouse and the
    cell a press writes to can never be two different cells.

    Floor division on a *floored* coordinate rather than on the float: a point
    at -0.5 belongs to no cell, and ``int(-0.5) // 4`` would put it in the last
    row of the grid above.
    """
    uid = doc.active_tilemap_uid()
    if uid is None:
        return None
    tileset_uid = doc.active_tileset_uid()
    if tileset_uid is None:
        return None
    try:
        tileset = doc.tileset_slot(tileset_uid).tileset
    except KeyError:  # pragma: no cover - a dangling binding
        return None
    x, y = int(math.floor(point[0])), int(math.floor(point[1]))
    width, height = doc.size
    if x < 0 or y < 0 or x >= width or y >= height:
        return None
    return (x // tileset.tile_w, y // tileset.tile_h)


def _tile_stamp(state: Any, tab: Any, point) -> bool:
    """Put the picked tile down in the cell under *point*. -> whether it wrote.

    **Once per cell entered**, which is what ``tile_cell`` is remembered for: a
    drag inside one cell sends a mouse-move per frame, and one history step per
    frame would make a single stamp forty steps deep. Straight through
    ``place_tiles`` -- the refs door -- so this tool never writes a pixel; the
    picture it produces is the materialization the engine derives.
    """
    doc = tab.doc
    cell = tile_cell(doc, point)
    if cell is None or cell == state.tile_cell:
        return False
    state.tile_cell = cell
    uid = doc.active_tilemap_uid()
    tileset_uid = doc.active_tileset_uid()
    if uid is None or tileset_uid is None:  # pragma: no cover - tile_cell said so
        return False
    # The same clamp the panel applies every frame, applied again at the write.
    # ``place_tiles`` refuses an id past the end of the tileset **by raising**,
    # and a raise out of a press is a window down rather than a refusal a user
    # can meet -- so the last line of defence is not allowed to be the first
    # one. The panel is drawn whenever a document has a tileset, so this is
    # belt to its braces rather than the only guard.
    state.clamp_tile_pick(tileset_uid, doc.tileset_slot(tileset_uid).tileset.tile_count)
    patch = np.array([[state.tile_gid()]], dtype=np.uint32)
    return bool(doc.place_tiles(uid, cell, patch))


def _manual_note(ctx: Any, state: Any, doc: Any) -> None:
    """The Manual-mode revert sentence, once per gesture that earned it.

    Derived at the pane from view state and the history serial, which is the
    standing ruling: ``_commit_tilemap_patch`` reverts *silently* -- it has no
    channel to report through and gains none, because "a gesture happened" is
    not something the document knows. What it does know is that a reverted
    commit pushes no step, so a head that has not moved across a paint gesture
    on a Manual tilemap cel is exactly the case, and the pane is the only place
    holding both ends of that comparison.

    Through ``toast_once`` for ``_locked_out``'s reason: a user painting on a
    Manual layer earns this sentence on every stroke, and one copy on screen is
    the message.
    """
    if state.tile_head is None:
        return
    head, state.tile_head = state.tile_head, None
    if doc.history.head == head:
        # Aseprite's own "Disable Snap to Grid" button, in this app's words: the
        # remedy is the *op*, so the tip cannot offer a switch the menus and
        # the keyboard do not have.
        state.say(
            inker_state.TILE_MANUAL_REVERTED,
            remedy="tile_auto",
            remedy_label="Switch to Auto",
        )


def _press(ctx: Any, state: Any, tab: Any, point, origin=(0.0, 0.0)) -> None:
    doc = tab.doc
    tool = state.tool
    # Handed over once, here, at the top of every gesture. The document's funnel
    # reads it to decide which of two identical swatches a stroke lands in (see
    # ``Document.paint_slot``), and here is the one place that is true of every
    # tool at once -- a brush, a fill, a shape, a gradient and a text stamp all
    # paint the foreground and all deserve the slot the user clicked. Setting it
    # per tool would be five places to forget.
    doc.paint_slot = getattr(state, "fg_slot", None)
    # Where the Manual-mode revert notice is measured from, banked at the same
    # moment and for the same reason ``paint_slot`` is: this is the one place
    # that runs at the top of every gesture, whatever tool it belongs to.
    state.tile_head = (
        doc.history.head
        if str(doc.tile_behavior) == "manual" and doc.active_tilemap_uid() is not None
        else None
    )
    state.drag_anchor = point
    state.last_point = point
    state.combine = _combine_op()
    ipoint = (int(math.floor(point[0])), int(math.floor(point[1])))

    # Before every paint branch, and it returns rather than falling through:
    # the slice tool must never leave a dab behind, which is exactly what a
    # missing early-out looks like the first time somebody drags on the canvas.
    if tool == "slice":
        if state.drag_button == 0:
            _slice_press(ctx, state, tab, origin, point)
        else:
            # Inert on the right button, and said here rather than by the
            # ``BG_BUTTON_TOOLS`` check below because this branch returns above
            # it: the slice tool is one of the tools that reserve the button
            # instead of spending it, so a right-press must not start a slice.
            state.drag_kind = ""
        return

    # The text tool, on the slice arm's shape and for its reasons: it returns
    # before every paint branch so a click can never leave a dab behind, and it
    # is inert on the right button rather than spending it. The lock is asked
    # here rather than at the OK button -- offering a popup, a font list and a
    # size, and only then saying the layer is locked, is a form the app knew
    # the answer to before it drew it.
    if tool == "text":
        if state.drag_button == 0 and not _locked_out(ctx, state, tab):
            state.text_at = ipoint
            _open_text(state, tab)
        state.drag_kind = ""
        return

    # Ctrl picks the *layer* under the cursor, which is the same gesture Alt
    # gives for colour. Here for the slice and text arms' reason -- it returns
    # rather than falling through, so a Ctrl+click can never leave a dab behind
    # whatever tool is held -- and before the Alt branch, so the two modifiers
    # never both fire on one press. Ctrl is genuinely free on this canvas:
    # panning is middle-drag or space-drag.
    if imgui.get_io().key_ctrl:
        hit = doc.layer_at(ipoint)
        if hit is not None:
            doc.set_active_layer(hit)
        # A miss is a silent no-op: a toast on every Ctrl+click over empty
        # canvas would be noise on a gesture people repeat.
        state.drag_kind = ""
        return

    # Alt over a paint tool picks the colour under the cursor, which is the one
    # convention a user coming from any other paint program reaches for without
    # thinking. Checked before the tool branches rather than inside the paint
    # one, so it reads as what it is: a modifier over the whole toolbox that the
    # paint tools happen to be the only ones with a free Alt for -- Alt already
    # subtracts on the selection tools and expands from centre on the shapes.
    if tool == "eyedropper" or (tool in PAINT_TOOLS and imgui.get_io().key_alt):
        picked = doc.eyedrop(ipoint, layer_only=state.sample_layer)
        if picked is not None:
            # Alt+right-click picks the *background* colour, which is the other
            # half of the right button painting with it: the two are one
            # gesture pair, and picking into fg from a right-click would make
            # the button mean two different things.
            if state.drag_button == 1:
                state.bg = picked
            else:
                state.set_fg(picked)
        state.drag_kind = ""
        return
    if state.drag_button == 1 and tool not in BG_BUTTON_TOOLS:
        # Inert rather than a second meaning; see ``BG_BUTTON_TOOLS``.
        state.drag_kind = ""
        return
    # After the right-button check above, so a press that is inert anyway does
    # not toast about a lock it was never going to reach.
    if _locked_out(ctx, state, tab):
        return
    if tool == "tile":
        # Before the paint branches and returning, the slice tool's shape: this
        # tool writes *refs* and must never leave a dab behind. The refusal is
        # the toolbox's own sentence, said here because a shortcut key selects
        # a tool without asking the panel anything.
        refusal = inker_state.tool_reason(tool, doc)
        if refusal:
            state.say(refusal)
            state.drag_kind = ""
            return
        doc.commit_floating()
        state.tile_cell = None
        state.drag_kind = "tile"
        _tile_stamp(state, tab, point)
        return
    colour = state.bg if state.drag_button == 1 else state.fg
    if tool == "fill":
        doc.commit_floating()
        doc.fill(
            ipoint,
            colour,
            thresh=state.wand_tolerance,
            contiguous=state.wand_contiguous,
            wrap=tab.tiled,
        )
        # The one paint gesture that never reaches ``_release``: a fill is a
        # click, so its Manual-mode notice is due here.
        _manual_note(ctx, state, doc)
        state.drag_kind = ""
        return
    if tool == "wand":
        doc.commit_floating()
        doc.select_wand(
            ipoint,
            tolerance=state.wand_tolerance,
            op=state.combine,
            contiguous=state.wand_contiguous,
            wrap=tab.tiled,
        )
        state.drag_kind = ""
        return
    if tool == "move":
        if doc.floating is not None and doc.floating.contains(ipoint):
            state.drag_kind = "move"
        elif doc.mask is not None and doc.mask.contains(ipoint):
            doc.lift()
            state.drag_kind = "move"
        elif doc.begin_layer_move():
            # The third arm (C10): no buffer and no selection means "move what
            # is on this layer", which is what the move tool does everywhere
            # else and what this one used to answer with nothing at all.
            state.drag_kind = "layer_move"
        else:
            state.drag_kind = ""
        return
    if tool == "lasso_poly":
        # Before the shared selection branch and returning, because the whole
        # point of this tool is that it does *not* start a ``drag_kind="lasso"``
        # drag: its vertices are clicked, not dragged out.
        _gesture_press(ctx, state, tab, point)
        return
    if tool in PATH_SHAPE_TOOLS:
        # One branch up's reason, from the other side of the toolbox (Q-c):
        # these are shape tools whose points are *clicked*, so they have to
        # reach the gesture rather than the ``drag_kind="shape"`` drag below.
        # They share the branch above's position and not its exemptions: these
        # paint, so ``_locked_out`` has already refused the click on a locked
        # layer and the right button has already gone inert.
        _gesture_press(ctx, state, tab, point)
        return
    if tool in SELECT_TOOLS:
        doc.commit_floating()
        # An unmodified drag starting *inside* the selection moves its edges
        # rather than replacing them -- the marquee's own version of grabbing
        # what you can see. The modifier check comes first, so Shift and Alt
        # still start the combine drags they have always started: a user
        # Shift-dragging to add a second region routinely starts inside the
        # first one, and reading the modifiers second would have taken that
        # gesture away.
        if state.combine == "replace" and doc.mask is not None and doc.mask.contains(ipoint):
            state.drag_kind = "mask-move"
            return
        state.drag_kind = "lasso" if tool == "lasso" else "marquee"
        state.lasso = [point]
        return
    if tool in SHAPE_TOOLS:
        doc.commit_floating()
        state.drag_kind = "shape"
        return
    if tool == "gradient":
        doc.commit_floating()
        state.drag_kind = "gradient"
        return
    if tool in PAINT_TOOLS:
        # Said here rather than only in the tools panel: the panel greys the
        # button, but a shortcut key selects a tool without asking the panel
        # anything, so this is the door a shading press on a document with no
        # palette actually arrives at.
        refusal = inker_state.tool_reason(tool, doc)
        if refusal:
            state.say(refusal)
            state.drag_kind = ""
            return
        doc.commit_floating()
        spraying = tool == "spray"
        # **Shift begins the stroke where the last one ended**, which is
        # Aseprite's line-from-last-point and the one gesture in the box that
        # is faster than the line tool for what it does: click, Shift-click,
        # Shift-click walks a polyline in the brush you are already holding.
        #
        # Not a separate code path -- the stroke opens at the remembered point
        # and is immediately walked to the click, so everything the brush does
        # (nib, symmetry, pixel-perfect, the tip, the wrap) comes with it, and
        # a Shift-*drag* carries on freehand from the end of the line. Never
        # for the spray: its advance is ``spray_at`` on a timer rather than a
        # walk down a segment, so there is no line for this to draw.
        from_last = imgui.get_io().key_shift and not spraying and tab.view.last_paint is not None
        opening = tab.view.last_paint if from_last else point
        # Asked once, and it decides two arguments rather than one; see
        # ``_press_mode`` for why the ink cannot be read independently of it.
        tip = state.tip_for(tool)
        doc.begin_stroke(
            opening,
            colour,
            size=_dab_size(state, spraying),
            hardness=state.hardness,
            opacity=state.opacity,
            spacing=state.spacing,
            mode=_press_mode(state, tool, tip),
            strength=state.strength,
            nib=state.nib,
            angle=float(state.brush_angle),
            # Both forced off for the spray: the corner filter is about a
            # *line* and there is no line here, and a lag on a stationary
            # airbrush would move the cloud away from the cursor.
            pixel_perfect=False if spraying else state.pixel_perfect,
            axis=state.symmetry_axis,
            radial=state.radial_count,
            stabilise=0.0 if spraying else state.stabilise,
            speed_taper=state.speed_taper,
            symmetry=state.symmetry,
            wrap=tab.tiled,
            scatter=state.brush_size / 2.0 if spraying else 0.0,
            # The determinism seam: the engine is a pure function of the seed
            # and the call sequence, and this is the one place entropy enters.
            seed=random.getrandbits(32) if spraying else 0,
            # Read at the press and carried for the whole stroke, so selecting
            # different slots mid-drag cannot change what the ramp is halfway
            # along -- and so nothing about the ramp reaches the document.
            ramp=shade_ramp(doc.palette, state.palette_slots) if tool == "shade" else (),
            shade_dir=state.shade_dir,
            # The captured tip, when this tool is set to use one. Asked through
            # ``tip_for`` rather than read off the two fields here, so the press
            # and the cursor outline drawn a frame earlier cannot disagree about
            # whether the picture is what is about to land.
            stamp=tip,
            stamp_align=state.stamp_align,
            # Or-ed with the layer's own flag inside the engine (6.1): the ink
            # says "for this stroke", the layer says "always", and neither may
            # switch the other off.
            lock_alpha=inker_state.ink_locks_alpha(state.ink),
        )
        if from_last:
            doc.stroke_to(point)
        state.spray_carry = 0.0
        state.drag_kind = "spray" if spraying else "paint"


def _press_mode(state: Any, tool: str, tip: Any) -> str:
    """Which brush mode this press opens with. **The tip outranks the ink.**

    One line of policy, in a function of its own because it is the seam a real
    bug slipped through and because a test has to be able to pin it against
    ``tip_for`` directly.

    The ink is the *only* thing that can send a mode other than
    ``BRUSH_MODES[tool]`` (6.1: five inks, ``inker_state.INKS``). But an image
    tip is honoured for two modes only -- a captured tip's alpha is both its
    shape and its transparency, so a copy ink has nothing left to say about one
    -- and ``StrokeState`` drops a tip handed to any other. Reading the ink
    independently of the tip made that a silent, unrecoverable failure: the
    panel hid the ink control while a tip was loaded, the cursor drew the tip's
    box and the setting stayed set, so the stroke stamped a round replace dab
    with nothing on screen to say why and no control left to turn the ink off.
    Two ordinary gestures reached it -- set the ink and then capture (Ctrl+B
    does not touch it), and apply a preset, since the ink is one of the options
    a preset carries.

    So ``tip_for`` is the single source of truth: whatever it advertises is
    what the press lays down, and a stale ink is ignored rather than obeyed. It
    is ignored rather than *cleared* because the tip is the transient thing --
    forget the tip and the ink the user chose is still theirs.
    """
    base = inker_state.BRUSH_MODES[tool]
    if tool not in inker_state.PAINT_TOOLS:
        return base
    mode = inker_state.ink_mode(state.ink, base)
    if tip is not None and mode not in STAMP_MODES:
        return base
    # The eraser, the blur and the smudge keep their own arithmetic: an ink
    # says what a *colour* does to a pixel, and none of those three writes one.
    return mode if base == "paint" else base


def _dab_size(state: Any, spraying: bool) -> int:
    """The brush diameter one dab is stamped at.

    For every tool but the spray that is the size slider. For the spray the
    slider is the width of the *cloud* -- what the brush cursor draws and what
    "spray width" means in every other editor -- so the dab is a fraction of it
    (``inker_state.SPRAY_DAB_FRACTION``); a spray whose dabs are as wide as its
    own disc is a blob.
    """
    if not spraying:
        return state.brush_size
    return max(1, round(state.brush_size * inker_state.SPRAY_DAB_FRACTION))


# --- the polygonal lasso, and the multi-click gesture under it (C4) -----------
#
# The pane's second kind of gesture. A drag is press / move / release and
# ``drag_kind`` names it while the button is down; this one is a run of separate
# clicks with the button *up* in between, so it cannot be a ``drag_kind`` at all
# -- and must not be, because ``_input`` refuses a press while a gesture owns the
# mouse (the C12d guard) and a click sequence would be eaten by its own state.
# Each click is therefore complete at the press: nothing drags, nothing releases,
# and ``InkerState.gesture_pts`` is the whole of what survives between clicks.
#
# Built as the shared thing rather than the poly lasso's own, because the curve
# and polygon shape tools are this gesture with a different landing -- and Q-c
# is that landing: ``_gesture_press`` collects for all four tools, and
# ``inker_mode.commit_gesture`` decides whether the vertices become a selection
# or paint. The only thing they differ in on the way there is whether a click
# back on the first point closes them (``CLOSES_ON_FIRST``).

#: How near the first vertex a click has to land to close the polygon, in
#: **screen** pixels before ``sp`` -- the slice handles' rule, for the slice
#: handles' reason: an image-space radius is a hundredth of a pixel at 100x zoom
#: and half the canvas at 5%, so the target would be unhittable at one end of the
#: zoom range and unavoidable at the other.
POLY_CLOSE = 7.0

#: The gesture tools whose path is **closed**, and which therefore end on a
#: click back at the first vertex. The poly lasso (a polygon is what a selection
#: is) and the polygon shape tool; the polyline and the curve are open paths, so
#: a click near their first point is a click like any other and placing a vertex
#: there is the only thing it can honestly mean. They end on a double-click or
#: Enter instead, which every gesture tool answers.
CLOSES_ON_FIRST = frozenset({"lasso_poly", "polygon"})


def closes_gesture(points, point, zoom: float, radius: float) -> bool:
    """Whether a click at ``point`` closes the polygon on its first vertex.

    Image-space distance times the zoom *is* the screen distance, because the
    view is a uniform scale after a quarter turn and a turn preserves length
    (``inker_state.basis`` is orthonormal). So this is quarter-turn and flip
    invariant without ever building a screen coordinate -- which is what lets it
    be a pure function of three numbers rather than of the view.

    Below three vertices there is nothing to close: two points are a line, and a
    click back on the first would otherwise end the gesture with a selection the
    rasteriser has to refuse anyway.
    """
    if len(points) < 3:
        return False
    return math.dist(points[0], point) * zoom <= radius


def _gesture_press(ctx: Any, state: Any, tab: Any, point) -> None:
    """One click of a multi-click gesture: open, extend, or close the path.

    Shared by the polygonal lasso and by the three clicked shape tools (Q-c),
    which differ only in what ``commit_gesture`` does with the vertices and in
    whether a click back on the first one closes them. A second copy of this
    arithmetic is exactly where the two would come to disagree about what a
    double-click means.

    ``drag_kind`` is left empty on every arm, which is the load-bearing part:
    it keeps the next click out of the C12d guard, keeps ``_drag`` and
    ``_release`` out of the gesture entirely -- so this tool can never take the
    freehand ``drag_kind="lasso"`` path -- and keeps ``clear_drag`` free to mean
    "cancel the gesture" everywhere it is already called from.
    """
    doc = tab.doc
    if not state.gesture_pts:
        # The float lands on the first click, as it does for every other
        # selection tool's press and every paint tool's -- the user has moved on
        # from it.
        doc.commit_floating()
        # Captured once, here. ``state.combine`` is re-read at every press, and
        # letting go of Shift before the closing click would otherwise turn an
        # add into a replace that throws the selection away. Recorded for the
        # painting shapes too, which have no use for it: one gesture has one
        # opening arm, and a field left stale is a field the next tool reads.
        state.gesture_combine = state.combine
        state.gesture_pts = [point]
        state.drag_kind = ""
        return
    if imgui.is_mouse_double_clicked(state.drag_button) or (
        state.tool in CLOSES_ON_FIRST
        and closes_gesture(state.gesture_pts, point, tab.view.zoom, sp(POLY_CLOSE))
    ):
        # The closing click places no vertex of its own: a double-click's second
        # press lands on top of the first, and a click near vertex 0 *is*
        # vertex 0. Either would add a degenerate edge to the polygon.
        inker_mode.commit_gesture(state, tab)
    else:
        state.gesture_pts.append(point)
    state.drag_kind = ""


#: The settled part of the previewed curve: ``(vertices, samples)``, or None.
#:
#: The frame loop never blocks, and this overlay runs on it: a fifty-vertex
#: curve resampled from scratch every frame is milliseconds of pure arithmetic
#: per frame for a picture that changed in two segments. Module-level rather
#: than per-tab state because it is a *memo* of a pure function keyed on its own
#: argument -- a stale entry cannot be read, only missed, so nothing has to
#: invalidate it when the tab, the tool or the document changes.
_curve_settled: tuple[tuple[tuple[float, float], ...], list[tuple[float, float]]] | None
_curve_settled = None


def _curve_path(points, cursor) -> list[tuple[float, float]]:
    """The previewed curve through ``points`` and the cursor, resampling only
    what the cursor can move.

    A Catmull-Rom segment is a function of four consecutive control points, so
    the provisional cursor point reaches exactly the last two segments -- the
    one that ends at it and the one before, which has it as its far neighbour.
    Everything earlier is settled, and settled is what is memoised.

    The result is **identical** to ``catmull_rom([*points, cursor])``, sample
    for sample and bit for bit, which it has to be: the same four control points
    in the same order through the same arithmetic, and the tail is recomputed
    from a four-point slice whose middle two segments have the same neighbours
    they do in the whole path. That equality is the test.
    """
    global _curve_settled

    pts = curve_points(points)  # collapsed exactly as ``catmull_rom`` does
    if len(pts) < 3:
        # One or two control points make no settled segment to keep; the whole
        # path is at most the two segments this would recompute anyway.
        return catmull_rom([*pts, cursor])
    key = tuple(pts)
    if _curve_settled is None or _curve_settled[0] != key:
        _, spans = curve_spans(pts)
        settled = [pts[0]]
        # Segments 0 .. len(pts) - 3, which end at the second-to-last vertex.
        for span in spans[: len(pts) - 2]:
            settled.extend(span)
        _curve_settled = (key, settled)
    out = list(_curve_settled[1])
    # The last three vertices plus the cursor: its middle two segments have
    # exactly the neighbours they have in the whole path, so their samples are
    # the whole path's. The first one is the settled tail already in ``out``.
    for span in curve_spans([*pts[-3:], cursor])[1][1:]:
        out.extend(span)
    return out


def _gesture_preview(state: Any, tab: Any, draw_list: Any, origin) -> None:
    """The open path, its rubber band, and where the closing click goes.

    Through ``to_screen`` like every other overlay (Ink9), so the polygon is
    drawn on the turned or mirrored page rather than a quarter turn away from
    it. The rubber band follows the *snapped* cursor rather than the raw one,
    which is the rule ``_preview``'s shape branch already states: a band drawn
    somewhere the next vertex will not go is worse than no band at all.

    The curve is previewed **segment by segment through the same arithmetic the
    commit rasterises** (``_curve_path``, which is ``catmull_rom`` with the
    settled segments memoised), rather than as the straight chords between its
    points: a spline preview that shows a polygon is a preview of a different
    tool, and one that samples the curve its own way is a promise the rasteriser
    has not made. The cursor rides along as a provisional last point, so what is
    on screen is the curve *if you click here* -- exactly what the polygon's
    rubber-band edge already means.
    """
    points = state.gesture_pts
    if not points:
        return
    view = tab.view
    colour = _u32(theme.ACCENT)
    mouse = imgui.get_mouse_pos()
    cursor = _snapped(state, _local(state, inker_state.to_image(view, origin, mouse.x, mouse.y)))
    if state.tool == "curve":
        curve = [inker_state.to_screen(view, origin, x, y) for x, y in _curve_path(points, cursor)]
        for a, b in zip(curve, curve[1:], strict=False):
            draw_list.add_line(a, b, colour)
        return
    screen = [inker_state.to_screen(view, origin, x, y) for x, y in points]
    for a, b in zip(screen, screen[1:], strict=False):
        draw_list.add_line(a, b, colour)
    tip = inker_state.to_screen(view, origin, *cursor)
    draw_list.add_line(screen[-1], tip, colour)
    if state.tool in CLOSES_ON_FIRST and len(screen) >= 3:
        # The edge a commit would close with, and the target that closes it.
        # Fainter than the placed edges: it is what *would* happen, not what has.
        draw_list.add_line(tip, screen[0], _u32(theme.ACCENT, 0.4))
        draw_list.add_circle(screen[0], sp(POLY_CLOSE), colour)


# --- the text tool ------------------------------------------------------------
#
# A popup on the canvas rather than a section in the tools panel, because the
# gesture is "put a word *here*": the click chooses the spot and the popup is
# the only thing between it and a floating buffer. Both halves live in this
# module and neither may move to another pane -- an imgui popup is matched by an
# id computed off the id stack, so an ``open_popup`` in the canvas child and a
# ``begin_popup`` in a sidebar would never meet (the same trap
# ``inker_bridge.CONVERT_POPUP`` is written up under).
#
# There is no live preview. A stamp *is* a floating buffer, which the canvas
# already draws and the user can already drag, so previewing it before the OK
# button would be a second, worse copy of what the next click gives them.

TEXT_POPUP = "inker-text"

#: How tall the typing box is, in design pixels: four lines and a bit, which is
#: what a caption or a label takes and enough for a longer one to scroll in.
TEXT_BOX_HEIGHT = 88.0

#: The cap on what one stamp may hold. Generous rather than meaningful -- it is
#: there so a paste of a whole file into the box cannot ask FreeType to lay out
#: a megabyte on the frame thread.
TEXT_MAX = 4000


def _open_text(state: Any, tab: Any) -> None:
    """Start a text stamp at the press that just landed.

    The font scan happens here rather than at import: reading several hundred
    directory entries is a frame's worth of work that a session which never
    touches the tool must not pay, and ``font_choices`` caches it from the
    first open onwards.

    **Antialiasing follows the document until the user says otherwise**: off on
    an indexed one, on everywhere else, decided here on every open. A palette
    is a promise that the file holds exactly those colours, and an antialiased
    edge is a rim of blends that each snap to the nearest slot -- so the
    default that keeps the promise is the monochrome rasteriser.

    It is a *default* and not a rule: an indexed document with a soft-edged
    palette is a real thing, so ticking the box in the popup sets
    ``text_aa_touched`` and this stops deciding anything. That flag is the
    whole fix for what the first version got wrong -- it asked whether the text
    tool had a stored options *entry*, which ``options_for`` creates on the
    first read of any of the three, so one popup on an RGB document made every
    indexed document for the rest of the session open with AA on. The manual
    says "on an indexed document it starts off", with no session in the
    sentence, and a promise that holds until you have used the tool once is not
    the promise.
    """
    inker_mode.font_choices()
    if not state.text_aa_touched:
        state.aa = not tab.doc.palette
    imgui.open_popup(TEXT_POPUP)


def _text_popup(ctx: Any, state: Any, tab: Any) -> None:
    """The typing box, the font, the size and the AA toggle.

    Nothing to clean up when imgui closes it on a click outside -- unlike the
    filter and conversion sessions this is cloned from, a text stamp previews
    nothing and holds nothing on the document, so an unanswered popup is
    simply a stamp that was never made. That is the whole of why there is no
    ``text_open`` flag beside ``filter_uid``.
    """
    if not imgui.begin_popup(TEXT_POPUP):
        return
    widgets.popup_chrome(_imgui=imgui)
    state.text_buffer = widgets.multiline(
        "##inkertext", state.text_buffer, sp(TEXT_BOX_HEIGHT), TEXT_MAX
    )
    choices = inker_mode.font_choices()
    if choices:
        state.font = widgets.combo("##inkerfont", state.font, choices, sp(240))
    else:
        # Only reachable with the vendored face missing *and* no system font
        # directory, i.e. a broken install. Said out loud rather than drawn as
        # an empty combo the user would click at.
        widgets.muted("No fonts found.")
    changed, value = widgets.labeled_slider_int(
        "Size", int(state.text_size), textstamp.MIN_SIZE, textstamp.MAX_SIZE
    )
    if changed:
        state.text_size = int(value)
    changed, value = controls.checkbox("Antialias", bool(state.aa), _imgui=imgui)
    if changed:
        state.aa = bool(value)
        # The user has an opinion now, so ``_open_text`` stops forming one --
        # in *both* directions. Ticking it on an indexed document keeps it
        # ticked there, and unticking it on an RGB one keeps it unticked; a
        # default that reasserted itself on the next click would be a checkbox
        # the user cannot operate.
        state.text_aa_touched = True
    widgets.help_marker(
        "Off renders the glyphs as whole pixels -- no partial coverage "
        "anywhere -- which is what pixel art and an indexed palette want."
    )
    imgui.color_button(
        "##inkertextfg", imgui.ImVec4(*[c / 255.0 for c in state.fg]), 0, (sp(16), sp(16))
    )
    imgui.same_line()
    widgets.muted("the foreground colour")

    imgui.dummy((0, 4))
    if controls.button("OK##inkertext", (sp(90), 0), _imgui=imgui):
        # The tool becomes Move on success (``stamp_text``), so the popup must
        # close either way: a refusal has already toasted why.
        inker_mode.stamp_text(ctx, state, tab)
        imgui.close_current_popup()
    imgui.same_line()
    if controls.button("Cancel##inkertext", (sp(90), 0), _imgui=imgui):
        imgui.close_current_popup()
    imgui.end_popup()


# --- slices -------------------------------------------------------------------
#
# A tool rather than a pane, so there is no new help anchor and no fourth
# sidebar: the overlay is on the canvas where the rectangles are, and the list,
# the toggles and the delete button ride the tools panel like every other tool's
# options do.
#
# The whole surface obeys the pane's two existing rules. Every screen position
# goes through ``_corners``/``_box`` rather than ``origin + x * zoom``, which is
# what keeps it correct on a turned or mirrored page. And a drag commits
# nothing: the press records what the slice looked like, the drag mutates the
# live object so the overlay follows the cursor, and the release pushes exactly
# one ``set_slice`` -- one gesture, one Ctrl+Z.

#: A slice's corner grab squares and the pivot's ring, in **design** pixels --
#: everything below runs both through ``sp``, drawing and hit-testing alike, so
#: the grab radius cannot come out smaller than the handle a user can see on a
#: scaled display.
SLICE_HANDLE = 4.0
SLICE_PIVOT_RADIUS = 5.0
#: How much bigger the grab radius is than the thing drawn. Generous, because
#: the alternative to grabbing a corner is starting a new slice on top of the
#: one you were aiming at.
SLICE_GRAB = 2.5
#: The smallest rectangle a drag will make, in **image** pixels. Two rather than
#: one: a stationary cursor at a fractional position rounds outward to a 1x1
#: rectangle, so a one-pixel floor would put a slice down on every stray click.
#: A genuinely one-pixel slice is still reachable by dragging a corner in.
SLICE_MIN = 2
#: How long a dash of the nine-slice centre is, in screen pixels. Dashed rather
#: than solid because the centre sits inside the slice's own outline and two
#: solid rectangles a few pixels apart read as one thick border.
SLICE_DASH = 4.0

#: Which corner of the bounds a resize drag is holding, and the pair of indices
#: into ``(x0, y0, x1, y1)`` it writes. Named so the press can record one string
#: and the drag can stay arithmetic.
SLICE_CORNERS = {"nw": (0, 1), "ne": (2, 1), "sw": (0, 3), "se": (2, 3)}


def slices_visible(state: Any) -> bool:
    """Whether the overlay draws. Derived rather than a second flag to keep in
    step: the slice tool forces it on, and ``show_slices`` is only ever the
    answer to "keep showing them while I paint"."""
    return bool(state.show_slices) or state.tool == "slice"


def _near(a, b, radius: float) -> bool:
    return math.dist(a, b) <= radius


def _slice_grab(state: Any, tab: Any, origin, point) -> tuple[str, str]:
    """What a press at ``point`` is holding: ``(kind, corner)``.

    Handles are hit-tested in **screen** space so a grab square is the same size
    to the hand at every zoom -- an image-space radius is a hundredth of a pixel
    at 100x and half the canvas at 5%. The rest (is the cursor inside a slice)
    is image space, where the rectangle actually is.

    The order is outermost first: the bounds corners, then the pivot, then the
    centre's corners, then the body, then empty canvas. Two grabs can genuinely
    coincide -- a pivot parked on a corner -- and resizing is the gesture a user
    reaches for at a corner, so it wins there.
    """
    doc = tab.doc
    entry = doc.slice_by_uid(state.slice_uid)
    mouse = imgui.get_mouse_pos()
    at = (mouse.x, mouse.y)
    frame_uid = tab.frame_uid
    if entry is not None:
        key = entry.at(frame_uid)
        x0, y0, x1, y1 = key.bounds
        corners = {
            "nw": (x0, y0),
            "ne": (x1, y0),
            "sw": (x0, y1),
            "se": (x1, y1),
        }
        grab = sp(SLICE_HANDLE) * SLICE_GRAB
        for name, (cx, cy) in corners.items():
            if _near(inker_state.to_screen(tab.view, origin, cx, cy), at, grab):
                return "slice-resize", name
        if key.pivot is not None:
            pivot = inker_state.to_screen(tab.view, origin, x0 + key.pivot[0], y0 + key.pivot[1])
            if _near(pivot, at, sp(SLICE_PIVOT_RADIUS) * SLICE_GRAB):
                return "slice-pivot", ""
        if key.center is not None:
            cx0, cy0, cx1, cy1 = key.center
            inner = {
                "nw": (x0 + cx0, y0 + cy0),
                "ne": (x0 + cx1, y0 + cy0),
                "sw": (x0 + cx0, y0 + cy1),
                "se": (x0 + cx1, y0 + cy1),
            }
            for name, (cx, cy) in inner.items():
                if _near(inker_state.to_screen(tab.view, origin, cx, cy), at, grab):
                    return "slice-center", name
    # The body of *any* slice, last one first: the list is drawn in order, so
    # the last is the one on top and the one the click visibly landed on.
    for candidate in reversed(doc.slices):
        x0, y0, x1, y1 = candidate.at(frame_uid).bounds
        if x0 <= point[0] < x1 and y0 <= point[1] < y1:
            state.slice_uid = candidate.uid
            return "slice-move", ""
    return "slice-new", ""


def _slice_press(ctx: Any, state: Any, tab: Any, origin, point) -> None:
    """Decide what this gesture is, and record what it started from."""
    if state.transforming:
        # A free transform owns the canvas until it is applied or cancelled;
        # letting a slice drag start underneath it would commit a step against
        # a document that is mid-operation.
        state.drag_kind = ""
        return
    kind, corner = _slice_grab(state, tab, origin, point)
    entry = tab.doc.slice_by_uid(state.slice_uid)
    state.drag_kind = kind
    state.slice_drag = (
        None if entry is None or kind == "slice-new" else slice_props(entry),
        corner,
    )


def _nudged(rect, corner: str, dx: float, dy: float):
    """One corner of a rectangle moved, the other three left alone.

    The corner is allowed to cross the far one, which is what every editor's
    resize does; the ordering happens once, at release, inside
    ``clamp_rect``. Ordering *during* the drag would swap which pair of indices
    the corner's name points at, and the rectangle would stick at the crossing.
    """
    out = list(rect)
    ix, iy = SLICE_CORNERS[corner]
    out[ix] += dx
    out[iy] += dy
    return tuple(out)


def _slice_drag(state: Any, tab: Any, point) -> None:
    """Move the live slice so the overlay follows the cursor. Pushes nothing.

    Measured against what was true at the **press**, never against the previous
    frame -- the rule ``_transform_input`` states, and here it is the difference
    between a drag that works and one that does not: the geometry is integer, so
    a per-frame delta of a third of a pixel truncates to nothing every frame and
    a slow drag at a high zoom moves the rectangle not at all.

    **Whatever the overlay is drawing is what moves.** On a frame with a key of
    its own that is the key, not the slice's own rectangle -- they are the same
    thing only on an unkeyed frame, and dragging the base while the canvas draws
    the key is a gesture that visibly does nothing.
    """
    entry = tab.doc.slice_by_uid(state.slice_uid)
    if entry is None or state.slice_drag is None or state.drag_kind == "slice-new":
        return
    before, corner = state.slice_drag
    if before is None:
        return
    anchor = state.drag_anchor or point
    dx, dy = point[0] - anchor[0], point[1] - anchor[1]

    frame_uid = tab.frame_uid
    keyed = frame_uid is not None and frame_uid in before["keys"]
    start = (
        before["keys"][frame_uid]
        if keyed
        else SliceKey(before["bounds"], before["pivot"], before["center"])
    )
    bounds, pivot, center = start.bounds, start.pivot, start.center

    if state.drag_kind == "slice-move":
        x0, y0, x1, y1 = bounds
        bounds = (x0 + dx, y0 + dy, x1 + dx, y1 + dy)
    elif state.drag_kind == "slice-resize":
        bounds = _nudged(bounds, corner, dx, dy)
    elif state.drag_kind == "slice-pivot" and pivot is not None:
        pivot = (pivot[0] + dx, pivot[1] + dy)
    elif state.drag_kind == "slice-center" and center is not None:
        center = _nudged(center, corner, dx, dy)

    if keyed:
        # A fresh frozen key in a fresh dictionary, never a write into the live
        # one: the step's "before" is holding that dictionary's contents and
        # must not move with the drag.
        entry.keys = {**entry.keys, frame_uid: SliceKey(bounds, pivot, center)}
        return
    entry.bounds, entry.pivot, entry.center = bounds, pivot, center
    entry.normalise()


def _slice_release(ctx: Any, state: Any, tab: Any, point) -> None:
    """One step for the whole gesture, or nothing at all."""
    doc = tab.doc
    if state.drag_kind == "slice-new":
        anchor = state.drag_anchor or point
        rect = marquee_rect(anchor, point)
        if rect[2] - rect[0] >= SLICE_MIN and rect[3] - rect[1] >= SLICE_MIN:
            state.slice_uid = doc.add_slice(rect).uid
            return
        # A click with no drag on empty canvas deselects, which is what the
        # marquee tool does with the same gesture -- and it is not "make a
        # one-pixel slice", which is what rounding a stationary cursor outward
        # would otherwise produce on every stray click.
        state.slice_uid = 0
        return
    if state.slice_drag is None:
        return
    before, _corner = state.slice_drag
    if before is not None:
        # ``was``, because the live object has already been dragged: reading the
        # before here would compare the slice against itself and push nothing.
        doc.set_slice(state.slice_uid, was=before)


def _dashed_rect(draw_list: Any, a, b, colour: int) -> None:
    """A rectangle in dashes, so it reads as *inside* the slice's own outline
    rather than as a second border a few pixels in from it."""
    corners = ((a[0], a[1]), (b[0], a[1]), (b[0], b[1]), (a[0], b[1]))
    for start, end in zip(corners, (*corners[1:], corners[0]), strict=True):
        length = math.dist(start, end)
        if length <= 0.0:
            continue
        steps = max(1, int(length // (SLICE_DASH * 2)))
        ux, uy = (end[0] - start[0]) / length, (end[1] - start[1]) / length
        for step in range(steps):
            head = step * length / steps
            tail = min(head + SLICE_DASH, length)
            draw_list.add_line(
                (start[0] + ux * head, start[1] + uy * head),
                (start[0] + ux * tail, start[1] + uy * tail),
                colour,
            )


def _slices(state: Any, tab: Any, draw_list: Any, origin) -> None:
    """Every slice, and the handles of the selected one.

    Everything here is placed through ``_box``/``to_screen``, never through
    ``origin + x * zoom``: a quarter turn maps an axis-aligned image rectangle
    onto an axis-aligned *screen* rectangle, and that is only true of a position
    that has been through the view's basis.
    """
    frame_uid = tab.frame_uid
    outline = _u32(theme.ACCENT, 0.75)
    inner = _u32(theme.ACCENT, 0.45)
    hot = _u32(theme.ACCENT)
    for entry in tab.doc.slices:
        key = entry.at(frame_uid)
        x0, y0, x1, y1 = key.bounds
        selected = entry.uid == state.slice_uid
        a, b = _box(tab.view, origin, x0, y0, x1, y1)
        draw_list.add_rect(a, b, hot if selected else outline)
        if key.center is not None:
            cx0, cy0, cx1, cy1 = key.center
            ca, cb = _box(tab.view, origin, x0 + cx0, y0 + cy0, x0 + cx1, y0 + cy1)
            _dashed_rect(draw_list, ca, cb, inner)
        if key.pivot is not None:
            px, py = inker_state.to_screen(tab.view, origin, x0 + key.pivot[0], y0 + key.pivot[1])
            radius = sp(SLICE_PIVOT_RADIUS)
            draw_list.add_circle((px, py), radius, hot if selected else outline)
            draw_list.add_line((px - radius, py), (px + radius, py), hot)
            draw_list.add_line((px, py - radius), (px, py + radius), hot)
        if not selected:
            continue
        size = sp(SLICE_HANDLE)
        for cx, cy in _corners(tab.view, origin, x0, y0, x1, y1):
            draw_list.add_rect_filled((cx - size, cy - size), (cx + size, cy + size), hot)


def _drag(state: Any, tab: Any, point) -> None:
    doc = tab.doc
    if state.drag_kind.startswith("slice-"):
        _slice_drag(state, tab, point)
        state.last_point = point
        return
    if state.drag_kind == "tile":
        _tile_stamp(state, tab, point)
        state.last_point = point
        return
    if state.drag_kind == "paint":
        doc.stroke_to(point)
    elif state.drag_kind == "spray":
        # Every held frame, moving or not: an airbrush keeps emitting while the
        # button is down, which is the whole tool. A rate times a delta rather
        # than a count per frame, so the cloud is the same on a slow machine as
        # on a fast one -- with the fraction carried, or a rate under one dab a
        # frame would round to nothing sixty times a second.
        state.spray_carry += state.spray_rate * imgui.get_io().delta_time
        emit = int(state.spray_carry)
        if emit > 0:
            state.spray_carry -= emit
            doc.spray_at(point, emit)
    elif state.drag_kind == "layer_move":
        # From the anchor, not from the last point: the session re-renders from
        # its snapshot, so what it wants is the total offset.
        anchor = state.drag_anchor or point
        doc.preview_layer_move(round(point[0] - anchor[0]), round(point[1] - anchor[1]))
    elif state.drag_kind == "move" and doc.floating is not None:
        last = state.last_point or point
        doc.move_floating(round(point[0] - last[0]), round(point[1] - last[1]))
    elif state.drag_kind == "lasso" and (
        not state.lasso or math.dist(state.lasso[-1], point) >= 2.0
    ):
        # Sampled rather than every pixel: a lasso is a polygon, and a vertex
        # per mouse-move makes a thousand-point one out of a slow drag.
        state.lasso.append(point)
    state.last_point = point


def _shape_drag(state: Any, anchor, point):
    """A shape drag's two endpoints, with the modifiers read *now*.

    Live rather than sampled at press, unlike ``combine``: a user decides a
    rectangle should have been a square halfway through drawing it, and the
    preview has to be able to change its mind with them.
    """
    io = imgui.get_io()
    return inker_state.shape_endpoints(
        state.tool, anchor, point, constrain=io.key_shift, from_centre=io.key_alt
    )


def marquee_rect(anchor, point) -> tuple[int, int, int, int]:
    """The pixel rectangle a marquee drag covers, whichever way it was drawn.

    Ordering the corners has to come *before* rounding them. Flooring the
    anchor and ceiling the release point rounds outward only while the drag
    runs down and to the right; reverse the drag and the same two rules round
    inward on both corners, so an identical gesture selects a smaller rectangle
    depending on the direction it was made in.
    """
    x0, x1 = sorted((float(anchor[0]), float(point[0])))
    y0, y1 = sorted((float(anchor[1]), float(point[1])))
    return (
        int(math.floor(x0)),
        int(math.floor(y0)),
        int(math.ceil(x1)),
        int(math.ceil(y1)),
    )


def _release(ctx: Any, state: Any, tab: Any, point) -> None:
    from ..inker import SelectionMask

    doc = tab.doc
    anchor = state.drag_anchor or point
    kind = state.drag_kind

    if kind.startswith("slice-"):
        _slice_release(ctx, state, tab, point)
        state.clear_drag()
        return
    if kind in ("paint", "spray"):
        doc.end_stroke()
        # Where the next Shift-click draws from. After ``end_stroke``, so a
        # refused or empty stroke still moves it -- the user's hand was here
        # either way, and a line back to some earlier point they have forgotten
        # about is more surprising than a line from where they just clicked.
        tab.view.last_paint = point
    elif kind == "layer_move":
        doc.commit_layer_move()
    elif kind == "shape":
        p0, p1 = _shape_drag(state, anchor, point)
        doc.shape(
            state.tool,
            (int(p0[0]), int(p0[1])),
            (int(p1[0]), int(p1[1])),
            # Never the background colour: a right-press on a shape tool is
            # inert and never reaches here (see ``BG_BUTTON_TOOLS``).
            state.fg,
            state.brush_size,
            filled=state.shape_filled,
            wrap=tab.tiled,
            # The rectangle's rounded corners (6.4). Read at the commit rather
            # than at the press, so ``C`` held mid-drag changes the shape under
            # the cursor -- which is what Aseprite's own gesture does.
            radius=int(state.corner_radius) if state.tool == "rect" else 0,
        )
    elif kind == "marquee":
        rect = marquee_rect(anchor, point)
        if rect[2] > rect[0] and rect[3] > rect[1]:
            build = (
                SelectionMask.from_ellipse
                if state.tool == "select_ellipse"
                else SelectionMask.from_rect
            )
            doc.select(build(doc.size, rect), state.combine)
        else:
            # A click with no drag inside a select tool means "deselect", which
            # is what every other editor does and what stops a stray click
            # leaving a one-pixel selection nothing can be painted outside.
            doc.deselect()
    elif kind == "mask-move":
        # One step for the whole drag: the live offset was drawn by shifting
        # the ants and touched nothing, so this is the first and only thing the
        # gesture pushes.
        dx, dy = _mask_shift(state, point)
        if (dx or dy) and doc.mask is not None:
            doc.select(doc.mask.translated(dx, dy))
    elif kind == "lasso":
        # The same landing the polygonal lasso commits through (C4), so the two
        # cannot come to disagree about what a run of vertices selects. A drag
        # too short to be a polygon is the marquee's stray click: deselect.
        if not inker_mode.polygon_select(doc, state.lasso, state.combine):
            doc.deselect()
    elif kind == "gradient":
        # To transparent means the *foreground* colour at zero alpha, not the
        # background one: fading to a transparent black leaves a dark fringe
        # wherever the two are blended.
        end = (*state.fg[:3], 0) if state.gradient_to_transparent else state.bg
        doc.gradient(
            anchor,
            point,
            state.fg,
            end,
            kind=state.gradient_kind,
            # Empty means the foreground-to-background preset, which is a live
            # reading of the two colours rather than a copy of them -- so
            # swapping with X changes the next gradient, as it always has.
            stops=state.gradient_stops or None,
            # "none" is the tool option's spelling and None is the engine's:
            # the engine's is a *path*, not a value, and keeping the two apart
            # is what makes the undithered arithmetic byte-identical.
            dither=None if state.gradient_dither == "none" else state.gradient_dither,
        )
    # Only the kinds that actually paint. A marquee drag pushes no history step
    # either, and reading a still head after one would announce a revert that
    # never happened; ``clear_drag`` below drops the banked head for every other
    # gesture. The tile stamp is excluded by construction -- it writes refs and
    # never reaches the tileset at all.
    if kind in ("paint", "spray", "shape", "gradient"):
        _manual_note(ctx, state, doc)
    state.clear_drag()


# --- drawing ----------------------------------------------------------------

#: Onion tints. Red behind, green ahead -- the convention every 2D animation
#: tool has used for decades, so picking differently would be a novelty the user
#: has to learn in exchange for nothing.
#: The tints an onion ghost is drawn in, as *defaults* -- the live values are
#: ``InkerState.onion_tint_back``/``_forward`` (6.7), because which colours
#: read as "before" and "after" depends on the drawing they are ghosting.
ONION_BACK = 0xE05050
ONION_FORWARD = 0x50C060


def _onion_span(state: Any, tab: Any, anim: Any, current: int) -> tuple[int, int]:
    """The range of frames the ghosts may come from, inclusive.

    The whole document, unless *wrap in tag* is on and the playhead is inside
    one (6.7) -- in which case it is that tag's own span, because what an
    animator working inside a walk cycle means by "the frame before this one"
    is the tag's last frame and not the previous clip's. The tag lookup is
    ``advance``'s own span logic, read through the same accessor, so the two
    cannot disagree about which frames a tag holds.
    """
    last = len(anim.frames) - 1
    if not getattr(state, "onion_wrap_tag", False):
        return (0, last)
    for tag in getattr(anim, "tags", ()) or ():
        low, high = int(getattr(tag, "start", 0)), int(getattr(tag, "end", 0))
        if low <= current <= high:
            return (max(0, low), min(last, high))
    return (0, last)


def _onion_index(current: int, delta: int, span: tuple[int, int]) -> int | None:
    """Which frame a ghost ``delta`` away is, or None if there is not one.

    Wrapping inside the span rather than clamping: a clamp draws the same
    ghost twice at the ends of a cycle, which reads as one ghost that stopped
    moving -- the failure the whole feature exists to avoid.
    """
    low, high = span
    width = high - low + 1
    if width <= 0:
        return None
    index = current + delta
    if low <= index <= high:
        return index
    if width == 1:
        return None
    return low + (index - low) % width


def _onion(ctx: Any, state: Any, tab: Any, draw_list, view, origin, size) -> None:
    """Neighbouring frames, tinted and faded, beneath the live one.

    Furthest first so the nearest neighbour ends up on top, and each step
    fainter than the last: the point is to see where a drawing *came from*, and
    three equally solid ghosts are just a mess.
    """
    anim = tab.doc.anim
    if anim is None:
        return
    current = anim.current
    # Resolved once, before the loop: the active track's uid does not change
    # between two ghosts of the same draw, and asking per neighbour would be
    # the same answer four times.
    track_uid = None
    if state.onion_current_layer:
        active = tab.doc.stack.active_index
        if 0 <= active < len(anim.tracks):
            track_uid = anim.tracks[active].uid
    span = _onion_span(state, tab, anim, current)
    tints = (
        int(getattr(state, "onion_tint_back", ONION_BACK)),
        int(getattr(state, "onion_tint_forward", ONION_FORWARD)),
    )
    for offset in range(max(state.onion_before, state.onion_after), 0, -1):
        for delta, colour in ((-offset, tints[0]), (offset, tints[1])):
            limit = state.onion_before if delta < 0 else state.onion_after
            index = _onion_index(current, delta, span)
            if offset > limit or index is None:
                continue
            texture = inker_textures.frame_texture(
                ctx, tab, anim.frames[index].uid, track_uid=track_uid
            )
            if texture is None:
                continue
            fade = state.onion_alpha / (offset ** max(0.0, float(state.onion_falloff)))
            _blit(
                draw_list,
                texture,
                view,
                origin,
                0,
                0,
                size[0],
                size[1],
                colour=_u32(colour, fade),
            )


def _playback_frame(ctx: Any, tab: Any, draw_list, view, origin, size) -> None:
    anim = tab.doc.anim
    if anim is None or not anim.frames:
        return
    index = max(0, min(tab.play_index, len(anim.frames) - 1))
    texture = inker_textures.frame_texture(ctx, tab, anim.frames[index].uid)
    if texture is not None:
        _blit(draw_list, texture, view, origin, 0, 0, size[0], size[1])


def _paint(ctx: Any, state: Any, tab: Any, origin, *, hovered: bool) -> None:
    doc = tab.doc
    view = tab.view
    draw_list = imgui.get_window_draw_list()
    width, height = doc.size
    # The canvas's screen AABB, which under a quarter turn is still exactly the
    # canvas rather than a box around it -- see ``_box``.
    top_left, bottom_right = _box(view, origin, 0, 0, width, height)
    # And the neighbourhood the tiled view shows: one tile out along each
    # wrapped axis, the canvas itself along the others. Every overlay below
    # stays on the canonical box -- overlays are canonical-only in v1 -- but the
    # checkerboard is the surface the page sits on, so it extends with it.
    tiled = _tiled_extent(tab, doc.size)
    tiled_tl, tiled_br = _box(view, origin, *tiled)

    _backdrop(ctx, tab, draw_list, view, origin, tiled, tiled_tl, tiled_br)
    # Before the composite and before its ``None`` early-out, so the strip is
    # genuinely beneath the live drawing rather than sometimes instead of it.
    if state.onion and not tab.playing and not state.onion_in_front:
        _onion(ctx, state, tab, draw_list, view, origin, doc.size)

    if tab.playing:
        # The cached flatten of the frame being played, not the document's
        # composite: the playhead on the document deliberately does not move
        # during playback, so the composite is still frame one.
        _playback_frame(ctx, tab, draw_list, view, origin, doc.size)
        draw_list.add_rect(top_left, bottom_right, _u32(theme.EDGE))
        return

    texture = inker_textures.composite(ctx, tab, nearest=view.zoom >= 1.0)
    if texture is None:
        return
    axes = axes_of(tab.tiled)
    # Set on **both** branches, every frame. Turning tiling on used to be a
    # one-way door in the reference viewer for exactly this reason: the sampler
    # was switched to GL_REPEAT and never put back, so the single-tile view that
    # followed sampled a wrapped texture at its own edges -- which is the one
    # place a seamless tile is not seamless, since LINEAR filtering there blends
    # the far edge in. moderngl skips the GL call when the value already
    # matches, so the idempotent write costs nothing.
    texture.repeat_x, texture.repeat_y = axes[0], axes[1]
    x0, y0, x1, y1 = tiled
    _blit(
        draw_list,
        texture,
        view,
        origin,
        x0,
        y0,
        x1,
        y1,
        uv0=(x0 / width, y0 / height),
        uv1=(x1 / width, y1 / height),
    )
    _floating(ctx, tab, draw_list, origin)
    if state.onion and not tab.playing and state.onion_in_front:
        # **Over the drawing** (6.7). After the composite and the floating
        # buffer, so what is checked against the ghosts is what is actually on
        # the canvas -- including the pixels that have not landed yet.
        _onion(ctx, state, tab, draw_list, view, origin, doc.size)
    draw_list.add_rect(top_left, bottom_right, _u32(theme.EDGE))

    if state.grid:
        _grid(state, draw_list, view, origin, doc.size, top_left, bottom_right)
    if state.pixel_grid:
        # Fainter than the tile grid and drawn after it, so where both are on
        # the tile lines still read as the stronger of the two.
        _grid(
            state,
            draw_list,
            view,
            origin,
            doc.size,
            top_left,
            bottom_right,
            step=1,
            alpha=0.25,
        )
    if state.layer_edges:
        _layer_edges(tab, draw_list, view, origin)
    if state.tile_numbers:
        _tile_numbers(state, tab, draw_list, view, origin)
    if state.symmetry != "none":
        _symmetry(state, draw_list, view, origin, doc.size)
    _ants(ctx, tab, draw_list, origin, state)
    if slices_visible(state):
        _slices(state, tab, draw_list, origin)
    if state.transforming:
        _transform_box(state, tab, draw_list, origin)
    _preview(state, tab, draw_list, origin)
    # Beside the drag preview rather than inside it: a multi-click gesture holds
    # no ``drag_kind``, which is the first thing ``_preview`` returns on.
    _gesture_preview(state, tab, draw_list, origin)
    if hovered:
        # **The shapes get the ring too.** ``doc.shape`` takes the same
        # ``brush_size`` the brush does, so a line has a stroke width -- and
        # nothing on screen said what it was, at any zoom, until it was
        # committed. The ring is the only place a variable width is legible.
        if state.tool in _RINGED_TOOLS:
            _cursor(state, draw_list, view)
        _pixel_cell(state, tab, draw_list, origin)
        if state.tool == "tile":
            _tile_cursor(state, tab, draw_list, origin)


def _tiled_extent(tab: Any, size) -> tuple[float, float, float, float]:
    """The image-space rectangle the canvas draws, in canvas coordinates.

    The canvas itself with tiling off, and one tile out along each *wrapped*
    axis with it on -- so X-only tiling shows a horizontal strip of three
    rather than a 3x3 block, which is the honest picture of what will actually
    wrap when you paint on it.
    """
    width, height = size
    wrap_x, wrap_y = axes_of(tab.tiled)
    x0, x1 = (-width, 2 * width) if wrap_x else (0, width)
    y0, y1 = (-height, 2 * height) if wrap_y else (0, height)
    return (x0, y0, x1, y1)


def _backdrop(ctx, tab, draw_list, view, origin, tiled, top_left, bottom_right) -> None:
    """What the composite is drawn over: the matte colour, or the checkerboard.

    This is the whole of "the preview tells the truth". ``Document.flatten``
    composites ``doc.matte`` *under* the stack where there is no background
    layer, so every flat-PNG export of a matted document fills erased areas
    with it -- while the canvas went on showing a checkerboard there, and the
    user only found out by reopening the file they saved.

    A solid quad here is arithmetically the same picture as compositing the
    matte under the stack, for one draw call and no pixels: the composite is
    already drawn over this with source-over blending. The alternative --
    matting inside ``inker_textures.composite`` -- costs a second full-canvas
    buffer per tab, invalidated per dab, on the frame thread.

    The four rotated corners rather than the screen AABB that
    ``_checkerboard`` is content with: a checker under a quarter-turned page
    may fill the box around it, but a white quad filling that box would paint
    over the desk outside the page.
    """
    matte = tab.doc.matte
    if matte is None or tab.doc.has_background:
        _checkerboard(ctx, draw_list, top_left, bottom_right)
        return
    red, green, blue, alpha = (list(matte) + [255, 255, 255, 255])[:4]
    colour = imgui.get_color_u32(
        imgui.ImVec4(red / 255.0, green / 255.0, blue / 255.0, alpha / 255.0)
    )
    a, b, c, d = _corners(view, origin, *tiled)
    draw_list.add_quad_filled(a, b, c, d, colour)


def _checkerboard(ctx: Any, draw_list: Any, top_left, bottom_right) -> None:
    """The transparency backdrop, drawn in **screen** space and deliberately not
    put through the view's turn.

    It is the surface the page sits on rather than part of the page: rotating it
    with the canvas would make the squares spin, which says nothing and looks
    like a bug. Under a quarter turn the canvas's screen AABB *is* the canvas,
    so an upright ``add_image`` over that box covers exactly what it should.
    """
    texture = inker_textures.checker(ctx)
    if texture is None:
        return
    # UVs in tile units, so one draw call covers the canvas at any zoom and the
    # squares stay a constant size on screen rather than in image space.
    span = (bottom_right[0] - top_left[0], bottom_right[1] - top_left[1])
    tile = texture.size[0]
    draw_list.add_image(
        widgets.texture_ref(texture),
        top_left,
        bottom_right,
        (0, 0),
        (span[0] / tile, span[1] / tile),
    )


def _floating(ctx: Any, tab: Any, draw_list: Any, origin) -> None:
    buf = tab.doc.floating
    if buf is None:
        return
    texture = inker_textures.floating(ctx, tab, nearest=tab.view.zoom >= 1.0)
    if texture is None:
        return
    x, y = buf.offset
    fw, fh = buf.size
    _blit(draw_list, texture, tab.view, origin, x, y, x + fw, y + fh)
    a, b = _box(tab.view, origin, x, y, x + fw, y + fh)
    draw_list.add_rect(a, b, _u32(theme.ACCENT))


def _layer_edges(tab: Any, draw_list: Any, view: Any, origin) -> None:
    """Outline what the active layer actually holds (6.8).

    The *content* bounds rather than the canvas: every layer is canvas-sized in
    this engine, so a border round the whole page would say nothing. Nothing is
    drawn for an empty layer, which is the honest answer -- an outline of
    nothing at the origin would read as a one-pixel drawing.
    """
    import numpy as np

    layer = tab.doc.stack.active
    opaque = layer.pixels[..., 3] > 0
    if not opaque.any():
        return
    rows = np.flatnonzero(opaque.any(axis=1))
    cols = np.flatnonzero(opaque.any(axis=0))
    x0, y0 = int(cols[0]), int(rows[0])
    x1, y1 = int(cols[-1]) + 1, int(rows[-1]) + 1
    corners = [
        inker_state.to_screen(view, origin, x, y)
        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    ]
    colour = _u32(theme.ACCENT, 0.8)
    for index in range(4):
        draw_list.add_line(corners[index], corners[(index + 1) % 4], colour)


def _tile_numbers(state: Any, tab: Any, draw_list: Any, view: Any, origin) -> None:
    """Each cell's local tile id, on a tilemap layer (6.8).

    A reading aid for authoring a tileset, and gated on the zoom for the pixel
    grid's reason: numbers smaller than the digits they are made of are a grey
    haze over the art rather than information.
    """
    doc = tab.doc
    layer = doc.stack.active
    refs = getattr(layer, "refs", None)
    if refs is None:
        return
    slot = doc.tileset_slot(doc.active_tileset_uid())
    if slot is None:
        return
    tile_w, tile_h = slot.tileset.tile_w, slot.tileset.tile_h
    if tile_w * view.zoom < 24.0:
        return
    colour = _u32(theme.TEXT, 0.75)
    height, width = refs.shape[:2]
    for row in range(height):
        for column in range(width):
            local = int(refs[row, column]) & 0x1FFFFFFF
            if not local:
                continue
            at = inker_state.to_screen(view, origin, column * tile_w + 2, row * tile_h + 2)
            draw_list.add_text(at, colour, str(local))


def _grid(
    state: Any,
    draw_list: Any,
    view: Any,
    origin,
    size,
    top_left,
    bottom_right,
    *,
    step: int = 0,
    alpha: float = 0.55,
) -> None:
    """Clipped to the visible rectangle: a 16-pixel grid over a 4096 canvas is
    a quarter of a million lines if it is drawn in full.

    ``step`` overrides the tile grid's size, which is how the **pixel grid**
    (6.8) is drawn: it is the same clipped walk at a step of one, not a second
    implementation -- and the six-screen-pixel floor below is what keeps it
    from being most of what is on screen at a low zoom.
    """
    colour = _u32(theme.EDGE, alpha)
    step = max(1, int(step or state.grid_size))
    # The line below the step is the real floor: six screen pixels between
    # grid lines. (There was a ``GRID_MIN_ZOOM`` branch here that computed
    # ``min(step, step)`` -- a no-op guarding a per-pixel grid that was never
    # implemented.)
    if step * view.zoom < 6.0:
        return
    width, height = size
    # Only the visible index range (B22): the canvas span intersected with
    # the window, turned back into grid indices. A 16 px grid over a zoomed
    # 4096 canvas used to walk all 257 columns to draw the dozen on screen.
    #
    # The range is derived by inverse-transforming the visible rectangle's
    # corners rather than by dividing a screen offset by the zoom, which is the
    # form that only worked while ``top_left`` was the image's origin: under a
    # quarter turn it is a different corner of the canvas, so the old
    # subtraction produced a range that was correct at rotation 0 and empty
    # (or reversed) at every other.
    win = imgui.get_window_pos()
    wsz = imgui.get_window_size()
    left = max(top_left[0], win.x)
    right = min(bottom_right[0], win.x + wsz.x)
    top = max(top_left[1], win.y)
    bottom = min(bottom_right[1], win.y + wsz.y)
    if left > right or top > bottom:
        return
    seen = [
        inker_state.to_image(view, origin, sx, sy) for sx in (left, right) for sy in (top, bottom)
    ]
    lo_x = max(0, int(min(p[0] for p in seen) / step) * step)
    hi_x = min(width, int(max(p[0] for p in seen)) + step)
    lo_y = max(0, int(min(p[1] for p in seen) / step) * step)
    hi_y = min(height, int(max(p[1] for p in seen)) + step)
    # An image-space line still lands as an axis-aligned screen line, because
    # the orientation is a quarter turn -- but which screen axis it lands on
    # swaps, so both endpoints are transformed rather than one coordinate being
    # borrowed from the canvas box.
    for x in range(lo_x, hi_x + 1, step):
        a = inker_state.to_screen(view, origin, x, 0)
        b = inker_state.to_screen(view, origin, x, height)
        draw_list.add_line(a, b, colour)
    for y in range(lo_y, hi_y + 1, step):
        a = inker_state.to_screen(view, origin, 0, y)
        b = inker_state.to_screen(view, origin, width, y)
        draw_list.add_line(a, b, colour)


# --- rulers ------------------------------------------------------------------

#: The bands' thickness in design pixels (through ``sp`` at draw time). Small
#: enough that four-digit labels overhang the vertical band a little, which is
#: the GIMP answer and better than a band wide enough for the worst label.
RULER_THICKNESS = 18.0

#: Labelled ticks want at least this many screen pixels between them.
RULER_LABEL_PX = 48.0


def ruler_step(zoom: float, minimum_px: float = RULER_LABEL_PX) -> int:
    """The smallest 1/2/5-ladder step whose labels sit ``minimum_px`` apart.

    Metric in the decimal sense -- the ladder is 1, 2, 5, 10, 20, 50, ... -- so
    every number a user reads off the band is a round one, at every zoom the
    pane allows. Pure, so the test can walk the whole zoom range.
    """
    if zoom <= 0.0:
        return 1
    magnitude = 1
    while True:
        for base in (1, 2, 5):
            step = base * magnitude
            if step * zoom >= minimum_px:
                return int(step)
        magnitude *= 10


def ruler_minor(step: int) -> int:
    """The minor-tick spacing under ``step``: fifths when they divide whole,
    halves otherwise, nothing when the step cannot split (step 1)."""
    if step % 5 == 0:
        return step // 5
    return step // 2


def _rulers(tab: Any, origin, region, *, hovered: bool) -> None:
    """Pixel rulers along the canvas child's top and left edges.

    An overlay, drawn after everything on the canvas: it owns no layout and
    swallows no clicks. Each band measures whichever *image* axis currently
    runs along it -- under a quarter turn the top ruler is reading image Y --
    found by sampling the view transform at the band's two ends; interpolating
    between those samples is exact because the transform is affine.
    """
    view = tab.view
    draw_list = imgui.get_window_draw_list()
    thickness = sp(RULER_THICKNESS)
    left, top = origin
    right, bottom = left + region[0], top + region[1]
    back = _u32(theme.ELEV_1, 0.95)
    ink = _u32(theme.MUTED)
    draw_list.add_rect_filled((left, top), (right, top + thickness), back)
    draw_list.add_rect_filled((left, top), (left + thickness, bottom), back)
    with fonts.small(imgui):
        _ruler_band(draw_list, view, origin, (left, right), top, thickness, ink, horizontal=True)
        _ruler_band(draw_list, view, origin, (top, bottom), left, thickness, ink, horizontal=False)
    # The corner square after both bands, so neither's ticks show through it,
    # and the two edge lines after the corner so they run unbroken.
    draw_list.add_rect_filled((left, top), (left + thickness, top + thickness), back)
    edge = _u32(theme.EDGE)
    draw_list.add_line((left, top + thickness), (right, top + thickness), edge)
    draw_list.add_line((left + thickness, top), (left + thickness, bottom), edge)
    if hovered:
        # The cursor's shadow on each band -- the part of a ruler that answers
        # "how big is what I am about to draw" while a drag is in flight.
        mouse = imgui.get_mouse_pos()
        accent = _u32(theme.ACCENT)
        if left <= mouse.x <= right:
            draw_list.add_line((mouse.x, top), (mouse.x, top + thickness), accent)
        if top <= mouse.y <= bottom:
            draw_list.add_line((left, mouse.y), (left + thickness, mouse.y), accent)


def _ruler_band(
    draw_list: Any,
    view: Any,
    origin,
    span: tuple[float, float],
    cross: float,
    thickness: float,
    ink: int,
    *,
    horizontal: bool,
) -> None:
    """One band's ticks and labels, between screen positions ``span`` at the
    fixed other coordinate ``cross``."""
    a0, a1 = span
    if horizontal:
        p0 = inker_state.to_image(view, origin, a0, cross)
        p1 = inker_state.to_image(view, origin, a1, cross)
    else:
        p0 = inker_state.to_image(view, origin, cross, a0)
        p1 = inker_state.to_image(view, origin, cross, a1)
    # Which image axis runs along this band: under a quarter turn exactly one
    # coordinate varies, and the flipped view only flips its sign.
    axis = 0 if abs(p1[0] - p0[0]) >= abs(p1[1] - p0[1]) else 1
    v0, v1 = p0[axis], p1[axis]
    if abs(v1 - v0) < 1e-9:
        return
    scale = (a1 - a0) / (v1 - v0)
    step = ruler_step(abs(scale))
    minor = ruler_minor(step)
    if minor and minor * abs(scale) < sp(5.0):
        minor = 0
    tick = minor or step
    lo, hi = min(v0, v1), max(v0, v1)
    start = int(math.floor(lo / tick)) * tick
    for value in range(start, int(math.ceil(hi)) + tick, tick):
        position = a0 + (value - v0) * scale
        major = value % step == 0
        length = thickness * (0.45 if major else 0.25)
        if horizontal:
            draw_list.add_line(
                (position, cross + thickness - length), (position, cross + thickness), ink
            )
            if major:
                draw_list.add_text((position + 3.0, cross + 1.0), ink, str(value))
        else:
            draw_list.add_line(
                (cross + thickness - length, position), (cross + thickness, position), ink
            )
            if major:
                draw_list.add_text((cross + 2.0, position + 1.0), ink, str(value))


def _symmetry(state: Any, draw_list: Any, view: Any, origin, size) -> None:
    """The guide, drawn where the engine actually reflects.

    It used to draw at ``width / 2`` and ``height / 2`` unconditionally, which
    was wrong twice over: ``brush._mirror`` reflects about ``(width - 1) / 2``
    by default, and it honours ``state.symmetry_axis`` when the user has moved
    it -- so a moved axis left the line pointing at the middle of the page while
    the strokes came out somewhere else. ``brush.axis_or_default`` is now the
    one answer both read. Radial had no guide at all and now gets the pivot,
    which is the only thing there is to show for it: its reflections are turns
    rather than lines.
    """
    from ..inker import brush

    width, height = size
    colour = _u32(theme.ACCENT, 0.6)
    ax, ay = brush.axis_or_default((int(width), int(height)), state.symmetry_axis)
    # Both endpoints through ``to_screen``, for ``_grid``'s reason: the axis a
    # mirror line lands on swaps with the page, so borrowing one coordinate
    # from a corner would draw it across the canvas the wrong way.
    if state.symmetry in ("x", "xy"):
        a = inker_state.to_screen(view, origin, ax, 0)
        b = inker_state.to_screen(view, origin, ax, height)
        draw_list.add_line(a, b, colour)
    if state.symmetry in ("y", "xy"):
        a = inker_state.to_screen(view, origin, 0, ay)
        b = inker_state.to_screen(view, origin, width, ay)
        draw_list.add_line(a, b, colour)
    if state.symmetry == "radial":
        centre = inker_state.to_screen(view, origin, ax, ay)
        radius = sp(SYMMETRY_PIVOT_RADIUS)
        draw_list.add_circle(centre, radius, colour)
        draw_list.add_line((centre[0] - radius, centre[1]), (centre[0] + radius, centre[1]), colour)
        draw_list.add_line((centre[0], centre[1] - radius), (centre[0], centre[1] + radius), colour)


def _mask_shift(state: Any, point: Any = None) -> tuple[int, int]:
    """Whole pixels a mask-move drag has travelled, or ``(0, 0)``.

    One function for the preview and for the release, so what the ants show
    while the drag runs and what ``select`` is handed when it ends cannot
    disagree by a pixel. Measured against the *press* rather than accumulated
    per frame, which is the rule the transform handles already follow.
    """
    if state.drag_kind != "mask-move" or state.drag_anchor is None:
        return (0, 0)
    tip = point if point is not None else state.last_point
    if tip is None:
        return (0, 0)
    return (
        int(round(tip[0] - state.drag_anchor[0])),
        int(round(tip[1] - state.drag_anchor[1])),
    )


def _contours(ctx: Any, tab: Any):
    """The selection outline, recomputed only when the mask object changes.

    ``select()`` always builds a new mask, so identity is a sound cache key --
    and it has to be one, because the document's revision ticks on every dab
    and tracing the boundary per stroke frame would be the most expensive thing
    on screen.

    What is cached is what :mod:`~warlock.studio.ants` prepared rather than the
    raw lattice points: the vertex arrays and the cumulative arc lengths are a
    pure function of the mask too, so measuring the perimeter belongs on the
    same side of this cache as tracing the boundary does.
    """
    key = f"paint_ants:{tab.uid}"
    cached = ctx.state.preview.get(key)
    mask = tab.doc.mask
    if cached is not None and cached[0] is mask:
        return cached[1]
    prepared = ants.prepare(mask.contours() if mask is not None else [])
    # The canvas-space AABB rides beside each loop (B23): a pure function of
    # the mask too, so it belongs on this side of the cache.
    loops = [(verts, cum, ants.loop_box(verts)) for verts, cum in prepared]
    ctx.state.preview[key] = (mask, loops)
    return loops


def _ants(ctx: Any, tab: Any, draw_list: Any, origin, state: Any = None) -> None:
    loops = _contours(ctx, tab)
    if not loops:
        return
    view = tab.view
    # A mask-move drag previews by sliding the *drawn* outline and nothing
    # else: the mask itself is untouched until the release, so the preview
    # costs no recompute of the contours and pushes no history. Expressed as an
    # image-space origin rather than a screen delta, so it goes through the
    # view's turn with everything else.
    shift = (0, 0) if state is None else _mask_shift(state)
    # Canvas (0, 0) on screen, from the same function every other overlay uses:
    # ``to_screen`` is a uniform scale plus this offset, and a second spelling
    # of it is how the ants end up one pixel off the mask they describe.
    offset = inker_state.to_screen(view, origin, float(shift[0]), float(shift[1]))
    phase = (time.monotonic() * ants.ANT_SPEED) % (ants.DASH * 2)
    light, dark = _u32(theme.TEXT), _u32(theme.BG)
    # The visible window, for the two culls below (B23): a whole loop whose
    # box misses it is skipped before any dash arithmetic, and the runs of a
    # loop that straddles it are masked before the per-run Python loop.
    win = imgui.get_window_pos()
    wsz = imgui.get_window_size()
    clip = (win.x, win.y, win.x + wsz.x, win.y + wsz.y)
    rows = inker_state.basis(view)
    matrix = np.asarray(rows, dtype=np.float64)
    for verts, cum, box in loops:
        # The loop's canvas box, put on screen through the *same* orientation
        # the dashes go through. Under a quarter turn the transformed corners
        # are still two opposite corners of an axis-aligned box, so a min/max
        # over the four is exact rather than conservative -- but the old form,
        # which scaled the box's own coordinates, silently culled every loop the
        # moment the page was turned.
        corners = [
            inker_state.to_screen(view, origin, x + shift[0], y + shift[1])
            for x in (box[0], box[2])
            for y in (box[1], box[3])
        ]
        xs = [p[0] for p in corners]
        ys = [p[1] for p in corners]
        if min(xs) > clip[2] or max(xs) < clip[0] or min(ys) > clip[3] or max(ys) < clip[1]:
            continue
        starts, ends, on = ants.dash_segments(verts, cum, view.zoom, offset, phase, basis=matrix)
        starts, ends, on = ants.cull(starts, ends, on, clip)
        # One call per dash. imgui has no batched per-segment-colour API, so
        # this loop is irreducible -- but it is over dashes now rather than over
        # every unit lattice step, which at zoom 1 is six times fewer.
        for (ax, ay), (bx, by), lit in zip(
            starts.tolist(), ends.tolist(), on.tolist(), strict=True
        ):
            draw_list.add_line((ax, ay), (bx, by), light if lit else dark)


def _preview(state: Any, tab: Any, draw_list: Any, origin) -> None:
    """What the current drag would produce, before it produces it."""
    if state.drag_anchor is None or not state.drag_kind:
        return
    view = tab.view
    mouse = imgui.get_mouse_pos()
    anchor = inker_state.to_screen(view, origin, *state.drag_anchor)

    # **The point the release will be handed, not the cursor.** The dispatcher
    # commits through ``_release(..., _snapped(_local(point)))``, so a preview
    # drawn from the raw mouse position is a picture of a gesture that is not
    # about to happen. Derived once, here, because it was derived in the shape
    # branch alone and the other four drew the cursor: with *Snap to grid* on,
    # the marquee's rubber band tracked the cursor and the release landed on
    # the lattice; in tiled view a band begun over a neighbouring tile was
    # drawn from the canonical-tile anchor to a cursor two tiles away, i.e.
    # stretched across the whole 3x3 view. The shape branch had been fixed for
    # exactly this and carried the reasoning; the fix belongs above the split.
    landing = _snapped(state, _local(state, inker_state.to_image(view, origin, mouse.x, mouse.y)))
    tip = inker_state.to_screen(view, origin, *landing)
    colour = _u32(theme.ACCENT)
    kind, tool = state.drag_kind, state.tool

    if kind == "slice-new":
        # The rectangle a release would add. The other four slice drags need no
        # preview at all: they move the live slice, so ``_slices`` is already
        # drawing exactly what the release will keep.
        draw_list.add_rect(anchor, tip, colour)
        return
    if kind.startswith("slice-"):
        return

    if kind == "shape":
        # Through the same function the release goes through, and back to
        # screen: a preview drawn from the raw cursor while the commit applies a
        # constraint is a picture of a shape the user is not about to get.
        p0, p1 = _shape_drag(state, state.drag_anchor, landing)
        anchor = inker_state.to_screen(view, origin, *p0)
        tip = inker_state.to_screen(view, origin, *p1)

    if kind == "lasso" and len(state.lasso) > 1:
        points = [inker_state.to_screen(view, origin, x, y) for x, y in state.lasso]
        for a, b in zip(points, points[1:], strict=False):
            draw_list.add_line(a, b, colour)
        draw_list.add_line(points[-1], tip, colour)
    elif kind == "gradient":
        draw_list.add_line(anchor, tip, colour, 2.0)
    elif kind == "marquee" and tool == "select_ellipse":
        _ellipse(draw_list, anchor, tip, colour)
    elif kind == "marquee" or (kind == "shape" and tool == "rect"):
        draw_list.add_rect(anchor, tip, colour)
    elif kind == "shape" and tool == "line":
        draw_list.add_line(anchor, tip, colour)
    elif kind == "shape" and tool == "ellipse":
        _ellipse(draw_list, anchor, tip, colour)


def _ellipse(draw_list: Any, a, b, colour: int) -> None:
    centre = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)
    radii = (abs(b[0] - a[0]) * 0.5, abs(b[1] - a[1]) * 0.5)
    if radii[0] > 0.5 and radii[1] > 0.5:
        draw_list.add_ellipse(centre, radii, colour)


#: Everything drawn with a ring under the cursor: the tools whose gesture is
#: sized by the brush slider. ``PAINT_TOOLS`` alone left the six shape tools
#: without one, and they take the same width.
_RINGED_TOOLS = PAINT_TOOLS | SHAPE_TOOLS

#: Screen pixels per image pixel below which the single-cell outline is not
#: drawn. Under about this the cell is smaller than the line tracing it, so the
#: outline stops being a cell and becomes a smudge that hides the art under it.
PIXEL_CELL_MIN_ZOOM = 4.0


def _pixel_cell(state: Any, tab: Any, draw_list: Any, origin) -> None:
    """Outline the one pixel the next dab lands on, once it is big enough to see.

    The ring says how *wide* the brush is and says nothing about *which* pixel
    it is centred on, which at 800% is the only question being asked -- the ring
    is drawn at the raw cursor and floats between cells rather than around one.
    This is the half that was missing, and it is the half pixel art is made of.

    Quarter turns keep the cell axis-aligned, so two corners describe it; a free
    rotation would not, and the view deliberately does not have one.
    """
    view = tab.view
    if view.zoom < PIXEL_CELL_MIN_ZOOM or origin is None:
        return
    point, _picked = _cursor_pixel(state, tab, origin, True)
    if point is None:
        return
    a = inker_state.to_screen(view, origin, float(point[0]), float(point[1]))
    b = inker_state.to_screen(view, origin, float(point[0] + 1), float(point[1] + 1))
    lo = (min(a[0], b[0]), min(a[1], b[1]))
    hi = (max(a[0], b[0]), max(a[1], b[1]))
    draw_list.add_rect(lo, hi, _u32(theme.TEXT, 0.85))


def _tile_cursor(state: Any, tab: Any, draw_list: Any, origin) -> None:
    """Outline the whole cell the tile stamp would land in.

    ``_pixel_cell``'s shape at the grid's scale, and drawn for the same reason:
    the stamp writes a *cell*, so a one-pixel cursor would be a picture of the
    wrong unit. Nothing at all on a layer the tool refuses -- ``tile_cell``
    answers ``None`` there, which is the same question the press asks, so the
    outline is on screen exactly when a click would write.

    Two corners describe it because the view's turns are quarter turns and keep
    the cell axis-aligned; a free rotation would not, and the view deliberately
    does not have one.
    """
    if origin is None:
        return
    mouse = imgui.get_mouse_pos()
    point = inker_state.to_image(tab.view, origin, mouse.x, mouse.y)
    cell = tile_cell(tab.doc, point)
    if cell is None:
        return
    tileset = tab.doc.tileset_slot(tab.doc.active_tileset_uid()).tileset
    x0, y0 = cell[0] * tileset.tile_w, cell[1] * tileset.tile_h
    a = inker_state.to_screen(tab.view, origin, float(x0), float(y0))
    b = inker_state.to_screen(
        tab.view, origin, float(x0 + tileset.tile_w), float(y0 + tileset.tile_h)
    )
    lo = (min(a[0], b[0]), min(a[1], b[1]))
    hi = (max(a[0], b[0]), max(a[1], b[1]))
    # Thickness is the **fifth** positional in this binding, not the sixth --
    # ``plotter_tileset``'s selection outline is the same call.
    draw_list.add_rect(lo, hi, _u32(theme.ACCENT, 0.9), 0.0, sp(2))


def _cursor(state: Any, draw_list: Any, view: Any) -> None:
    """A circle the size of the brush. The one piece of feedback that makes a
    variable-size brush usable at all."""
    mouse = imgui.get_mouse_pos()
    colour = _u32(theme.TEXT, 0.7)
    tip = state.tip_for(state.tool)
    if tip is not None:
        # The tip's own box, not the size slider's circle: an image brush is not
        # round and is not the slider's size, so the ring would be a picture of
        # a brush that is not the one in hand. Drawn at the *variant's* size, so
        # a quarter turn of a tall stamp shows a wide box -- which is what will
        # land.
        width, height = tip.size
        half_w, half_h = width * 0.5 * view.zoom, height * 0.5 * view.zoom
        draw_list.add_rect(
            (mouse.x - half_w, mouse.y - half_h),
            (mouse.x + half_w, mouse.y + half_h),
            colour,
        )
        return
    radius = max(2.0, state.brush_size * 0.5 * view.zoom)
    if state.nib == "square":
        draw_list.add_rect(
            (mouse.x - radius, mouse.y - radius), (mouse.x + radius, mouse.y + radius), colour
        )
        return
    draw_list.add_circle((mouse.x, mouse.y), radius, colour)
    # A pixel nib has no falloff, so it has no inner ring to draw -- and the
    # ring is read as "this is where it is solid", which for a hard-edged nib
    # would be a picture of a softness it does not have.
    if state.nib == "soft" and state.hardness < 0.99:
        inner = radius * max(state.hardness, 0.05)
        draw_list.add_circle((mouse.x, mouse.y), inner, _u32(theme.TEXT, 0.25))
