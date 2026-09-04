"""Clay's workspace: the viewport, its tabs, its marquee and its drag HUD.

A **mixin on** :class:`~.main.App`, which is this repository's idiom for a body
of drawing that belongs to the shell -- ``ClayView`` is assembled the same way
from five ops classes and ``Document`` from six. So ``self`` here is the App and
every method's body is unchanged.

Lifted out of ``studio/main`` on 2026-09-04 (T7 of the 2026-09-02 review), after
the behavioural findings that touch it were closed, so the move is code motion
over tested behaviour.

The viewport itself is ``ClayView``'s; what is here is the *pane* around it --
the layout skeleton, the invisible button that takes the mouse, the tab bar,
the empty state and the two overlays drawn on top.
"""

from __future__ import annotations

from typing import Any


class ClayViewport:
    """Clay's pane drawing, mixed into :class:`~.main.App`.

    The shell names it reaches are imported *inside* the methods that use them:
    ``main`` imports this module to build the class, so a module-scope import
    back would be a cycle. Same shape as ``review_panes``.
    """

    def _clay_workspace(self) -> None:
        """The same sidebar / centre / sidebar skeleton every other mode uses:

            [ clay-tools ]  the header    [ clay-outliner ]
            [            ]  the viewport  [ clay-props    ]
            [            ]  the hint      [ clay-bridge   ]

        Both sidebars are ``skeletons.clay``, which is where the argument for
        that arrangement is written down. It was the last sidebar-shaped
        workspace composed by hand here, which is to say the last one a saved
        layout could not permute.
        """
        from imgui_bundle import imgui

        from . import clay_mode, skeletons, widgets
        from . import layout as layout_mod
        from .main import _column_boundary

        ctx = self.app_ctx
        lay = self.layout
        left_w = layout_mod.sidebar_width("left")
        right_w = layout_mod.sidebar_width("right")
        columns = skeletons.for_mode(ctx, "clay")

        layout_mod.column(
            ctx,
            lay,
            skeletons.ordered(ctx, self.layouts, "clay", columns["left"]),
            width=left_w,
            handle_length=left_w,
        )

        _column_boundary(self.layouts, "clay", "left")
        width = layout_mod.centre_width()
        flags = imgui.WindowFlags_.no_scroll_with_mouse.value
        with layout_mod.pane(
            "clay-centre",
            (width, 0),
            layout_mod.PaneRole.CONTENT,
            window_flags=flags,
        ) as visible:
            if visible:
                self._clay_viewport(ctx, clay_mode, widgets)

        _column_boundary(self.layouts, "clay", "right")
        layout_mod.column(
            ctx,
            lay,
            skeletons.ordered(ctx, self.layouts, "clay", columns["right"]),
            width=right_w,
            handle_length=right_w,
        )

    def _clay_viewport(self, ctx: Any, clay_mode: Any, widgets: Any) -> None:
        from imgui_bundle import imgui

        from . import tokens
        from .main import TARGET_FPS
        from .panes import clay_header, clay_hud, clay_menu

        self._clay_tabs(ctx, clay_mode)
        tab = clay_mode.active(ctx)
        if tab is None:
            self._clay_empty(ctx, clay_mode)
            return
        # The header, between the tabs and the image. Before the content region
        # is read, exactly as Inker's context bar and Plotter's toolbar are: the
        # viewport sizes itself from what is left, so a strip drawn after the
        # measurement is a strip drawn over the render.
        clay_header.draw(ctx, getattr(ctx, "clay_view", None))
        avail = imgui.get_content_region_avail()
        # And the hint line's own row, reserved rather than drawn over: a line
        # the viewport has already claimed the height for is a line clipped away
        # at the bottom of the pane, which is where every status row in this app
        # has gone wrong at least once.
        hint_h = float(tokens.sp(clay_hud.HINT_H))
        rect = (
            imgui.get_cursor_screen_pos().x,
            imgui.get_cursor_screen_pos().y,
            max(avail.x, 1.0),
            max(avail.y - hint_h, 1.0),
        )
        state = clay_mode.ensure(ctx)
        if state.frame_pending:
            # The other half of ``F``: the mode recorded the intent and this is
            # the only place that has the viewport to act on it (B6).
            state.frame_pending = False
            self._frame_clay_selection()
        view = self._ensure_build_view()
        # One viewport, many tabs: the camera belongs to the *document*, so it
        # is snapshotted off the live one on the way out of a tab and put back
        # on the way in. Done here rather than in ``ClayState.activate`` because
        # this is the only place that has the viewport -- and it is keyed on
        # what is being drawn rather than on the switch, so a tab restored from
        # a ``.wblk`` or closed out from under the pointer lands correctly too.
        if self._clay_camera_tab != tab.uid:
            clay_mode.remember_camera(ctx, state.get(self._clay_camera_tab))
            clay_mode.apply_camera(ctx, tab)
            self._clay_camera_tab = tab.uid
        # The shading mode decides two of the renderer's three switches and the
        # overlay decides the third: *Wire* is the surface replaced by its
        # edges, *Solid* is the surface drawn unlit, and the Wireframe overlay
        # is edges drawn over whichever of those is showing.
        view.wireframe = state.shading == "wireframe"
        view.flat = state.shading == "solid"
        view.wire_overlay = bool(state.overlays.get("wire", False))
        view.xray = bool(state.xray)
        view.show_grid = state.grid
        texture = view.draw(tab.doc, rect, 1.0 / TARGET_FPS)
        imgui.image(widgets.texture_ref(texture), (rect[2], rect[3]), (0, 1), (1, 0))
        self._build_hovered = imgui.is_item_hovered()
        self._clay_marquee(imgui, view, rect)
        self._clay_drag_hud(imgui, widgets, view, rect)
        # Over the render and inside the same clip: the widget is a control you
        # reach for without looking away from what you are turning, which is the
        # whole of why it is in the corner rather than in a pane.
        # Clears ``_build_hovered``: the flag was recorded off the render image
        # above, which cannot know a control has since been drawn over it, and
        # ``_clay_event`` routes the pygame press on it -- so a click on a ball
        # turned the camera and picked the mesh behind it in one gesture.
        if clay_hud.axis_widget(ctx, view, rect):
            self._build_hovered = False
        # Opposite corner from the widget: two readouts in one corner is one of
        # them unreadable.
        clay_hud.stats_overlay(ctx, rect)
        clay_menu.draw(ctx, view)
        # Last, and under the image: read when you are stuck, and a line over
        # the model covers the thing you are stuck on.
        imgui.set_cursor_screen_pos((rect[0], rect[1] + rect[3]))
        clay_hud.hint_line(ctx)

    def _clay_tabs(self, ctx: Any, clay_mode: Any) -> None:
        """Clay's open documents, which nothing has ever drawn.

        The document model was all there -- ``docs``, ``active``, ``activate``,
        ``cycle``, ``close`` -- and the only thing missing was the bar: Ctrl+Tab
        switched between documents with nothing on screen to say there was more
        than one, and ``close`` had no caller at all.

        Drawn above ``_clay_empty`` as well as above the viewport, because the
        last tab closing is exactly when the bar disappears and the empty state
        has to be what is underneath it.

        ``unsaved_document`` rather than a ``"* "`` prefix, which is Inker's
        rule and the right one: the title is half of the tab's identity.
        """
        from imgui_bundle import imgui

        state = clay_mode.ensure(ctx)
        if not state.docs:
            return
        # ``auto_select_new_tabs`` for ``inker_canvas``'s reason: without it, a
        # second opened document lands behind the first and "Open" looks inert.
        flags = imgui.TabBarFlags_.reorderable.value | imgui.TabBarFlags_.auto_select_new_tabs.value
        if not imgui.begin_tab_bar("clay-tabs", flags):
            return
        for tab in list(state.docs):
            item_flags = imgui.TabItemFlags_.unsaved_document.value if tab.dirty else 0
            opened, keep = imgui.begin_tab_item(tab.label, True, item_flags)
            if opened:
                state.activate(tab.uid)
                imgui.end_tab_item()
            if not keep:
                clay_mode.close_tab(ctx, tab.uid)
        imgui.end_tab_bar()

    def _clay_empty(self, ctx: Any, clay_mode: Any) -> None:
        """What Clay shows with nothing open, mirroring the raster editor's.

        Buttons rather than a sentence: ``new_document`` was reachable only
        through Ctrl+N, so the empty state told the user to "start a document"
        and offered no way to.
        """

        from . import widgets

        # This was written as a copy of ``inker_canvas._empty`` and the copy
        # dropped the ``sp()`` scaling, so at 150 % the raster editor's empty
        # state grew with the text while Clay's kept 240-*physical*-pixel
        # buttons under 1.5x labels -- which is where a label stops fitting its
        # button. Both are one function now (the UI redesign, wave 2), which is the
        # only fix that also holds for the next copy.
        widgets.nothing_open(
            "Start a model, open a document, or drop a .wblk on the window.",
            [
                ("New model", lambda: clay_mode.new_document(ctx)),
                ("Open a file...", lambda: clay_mode.ask_open(ctx)),
            ],
            # No recent list here: it is the bridge panel's, on both of its
            # branches, which is where Plotter and Packwright keep theirs (B5).
        )

    def _clay_marquee(self, imgui: Any, view: Any, rect: Any) -> None:
        """The selection rectangle, drawn in imgui rather than in GL.

        It is a two-dimensional screen decoration with no depth and no place in
        the scene, so putting it through the renderer would mean a vertex
        buffer rebuilt every mouse-move for four corners. The draw list is
        already there and already clipped to this window.
        """
        box = getattr(view, "marquee", None)
        if box is None:
            return
        draw = imgui.get_window_draw_list()
        x0, y0 = rect[0] + min(box[0], box[2]), rect[1] + min(box[1], box[3])
        x1, y1 = rect[0] + max(box[0], box[2]), rect[1] + max(box[1], box[3])
        draw.add_rect_filled((x0, y0), (x1, y1), imgui.get_color_u32((1, 1, 1, 0.08)))
        draw.add_rect((x0, y0), (x1, y1), imgui.get_color_u32((1, 1, 1, 0.55)))

    def _clay_drag_hud(self, imgui: Any, widgets: Any, view: Any, rect: Any) -> None:
        """What the live drag currently amounts to, above the cursor.

        In the draw list for ``_clay_marquee``'s reason, and *near the cursor*
        rather than in a corner: the number answers a question the user is
        asking with their hand, and a readout they have to look away to find is
        one they stop looking at. It draws only while a drag is live, so an idle
        viewport is unchanged.
        """
        text = getattr(view, "drag_hud", "")
        if not text or not getattr(view, "dragging", False):
            return
        from . import theme
        from .tokens import sp

        mouse = imgui.get_mouse_pos()
        x, y = mouse.x + sp(18), mouse.y - sp(28)
        draw = imgui.get_window_draw_list()
        size = imgui.calc_text_size(text)
        pad = sp(6)
        draw.add_rect_filled(
            (x - pad, y - pad),
            (x + size.x + pad, y + size.y + pad),
            imgui.get_color_u32(theme.rgba(theme.ELEV_2, 0.92)),
            sp(4),
        )
        draw.add_text((x, y), imgui.get_color_u32(theme.rgba(theme.TEXT)), text)
