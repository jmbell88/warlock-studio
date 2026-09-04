"""Poser's workspace: the viewport, the joint menu and the viewer it owns.

A **mixin on** :class:`~.main.App`, which is this repository's idiom for a body
of drawing that belongs to the shell -- ``ClayView`` is assembled the same way
from five ops classes and ``Document`` from six. So ``self`` here is the App and
every method's body is unchanged.

Lifted out of ``studio/main`` on 2026-09-04 (T7 of the 2026-09-02 review), after
the behavioural findings that touch it were closed, so the move is code motion
over tested behaviour.

Poser has its **own** ``Viewer`` instance rather than the shared one, which is
why ``_ensure_poser_viewer`` is here and why the router must never let the
shared viewer see its events -- one drag would orbit both cameras.
"""

from __future__ import annotations

from typing import Any


class PoserViewport:
    """Poser's pane drawing, mixed into :class:`~.main.App`.

    ``review_panes``' lazy-import rule, for its reason.
    """

    def _poser_workspace(self) -> None:
        """The sidebar / centre / sidebar skeleton, Poser's way:

            [ poser_library + poser_clips ]  viewport  [ poser_controls ]

        One pane per side rather than Clay's stacked pairs: the library and the
        controls are each one scroller, and an empty half-pane would be chrome.
        The clip editor is a *section* inside the left scroller for that same
        reason -- a skeleton with no clips shows one collapsed heading, where a
        split pane would show an empty half.
        """
        from imgui_bundle import imgui

        from . import layout as layout_mod
        from .main import _column_boundary
        from .panes import poser_clips, poser_controls, poser_library

        ctx = self.app_ctx
        left_w = layout_mod.sidebar_width("left")
        right_w = layout_mod.sidebar_width("right")
        with layout_mod.pane(
            "poser-library",
            (left_w, 0),
            layout_mod.PaneRole.SIDEBAR,
            edge=layout_mod.PaneEdge.RIGHT,
        ) as visible:
            if visible:
                poser_library.draw(ctx)
                # Under the pose library rather than in a mode of its own: a
                # clip is a *library* of the same kind of thing, and the right
                # sidebar has to stay free for the joint controls, which are
                # the actual editing surface for a key.
                poser_clips.draw(ctx)

        _column_boundary(self.layouts, "poser", "left")
        width = layout_mod.centre_width()
        flags = imgui.WindowFlags_.no_scroll_with_mouse.value
        with layout_mod.pane(
            "poser-centre",
            (width, 0),
            layout_mod.PaneRole.CONTENT,
            window_flags=flags,
        ) as visible:
            if visible:
                self._poser_viewport(ctx)

        _column_boundary(self.layouts, "poser", "right")
        with layout_mod.pane(
            "poser-controls",
            (right_w, 0),
            layout_mod.PaneRole.INSPECTOR,
            edge=layout_mod.PaneEdge.LEFT,
        ) as visible:
            if visible:
                poser_controls.draw(ctx)

    def _poser_viewport(self, ctx: Any) -> None:
        from imgui_bundle import imgui

        from . import icons, poser_mode, widgets
        from .main import TARGET_FPS
        from .panes import overlay

        self._poser_hovered = False
        state = poser_mode.ensure(ctx)
        if not ctx.rigging_available:
            overlay.placeholder(ctx)
            return
        viewer = self._ensure_poser_viewer()
        showing = poser_mode.sync_preview(ctx, viewer)
        if not showing:
            # Both branches through ``centred_empty``, the shape every other
            # workspace's empty viewport takes. These two were the app's one
            # pair of top-left muted lines where nine centred cards go.
            if state.building:
                overlay.centred_empty(
                    icons.PERSON_STANDING,
                    "Building the skeleton preview",
                    "The armature is built by Blender once per skeleton and "
                    "cached; the first open of a template takes a moment.",
                )
            elif state.error:
                overlay.centred_empty(
                    icons.TRIANGLE_ALERT, "The skeleton did not build", state.error
                )
            else:
                overlay.placeholder(ctx)
            return
        avail = imgui.get_content_region_avail()
        rect = (
            imgui.get_cursor_screen_pos().x,
            imgui.get_cursor_screen_pos().y,
            max(avail.x, 1.0),
            max(avail.y, 1.0),
        )
        texture = viewer.render(rect, 1.0 / TARGET_FPS)
        imgui.image(widgets.texture_ref(texture), (rect[2], rect[3]), (0, 1), (1, 0))
        self._poser_hovered = imgui.is_item_hovered()
        self._poser_menu(ctx, viewer)

    def _poser_menu(self, ctx: Any, viewer: Any) -> None:
        """The joint's right-click menu (B7).

        Drawn here rather than in a pane because a popup belongs to the window
        that begins it and this is that window -- the same reason
        ``clay_menu`` is called from Clay's viewport. The viewer records
        ``menu_request`` and knows nothing about imgui.
        """
        from imgui_bundle import imgui

        from . import controls, widgets

        popup = "poser-joint-menu"
        if viewer.menu_request is not None:
            viewer.menu_request = None
            imgui.open_popup(popup)
        if not imgui.begin_popup(popup):
            return
        widgets.popup_chrome(_imgui=imgui)
        selected = viewer.editor.selected
        if selected is None:
            widgets.secondary("No joint selected")
        else:
            widgets.secondary(str(selected))
            imgui.separator()
            # Through the *viewer*, never the editor: every one of these has a
            # ``_after_pose_change`` behind it that re-skins the preview, which
            # is exactly the step a direct editor call would skip.
            if controls.menu_item_simple("Clear this joint's rotation"):
                viewer.reset_bone()
            if controls.menu_item_simple("Deselect"):
                viewer.editor.selected = None
        imgui.separator()
        if controls.menu_item_simple("Reset the whole pose"):
            # Through the same guard the *button* has
            # (``poser_controls``/``pose_panel``), which this bypassed: "reset
            # every joint" throws away an unsaved pose, and a right-click menu
            # is the easiest of the three doors to hit by accident.
            from . import poser_mode

            poser_mode.guard(ctx, "reset every joint", viewer.reset_all)
        imgui.end_popup()

    def _ensure_poser_viewer(self) -> Any:
        """Poser's own Viewer, built on first use for ClayView's reason -- and
        mirrored onto the ctx so poser_mode's guard can reach the editor."""
        from .viewer_embed import Viewer

        if self.poser_viewer is None:
            self.poser_viewer = Viewer(self.ctx)
            self.app_ctx.poser_viewer = self.poser_viewer
        return self.poser_viewer
