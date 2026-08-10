"""Poser's right sidebar: the selected joint, the root, saving.

The mirror of ``pose_panel``'s pose-mode half, over the Poser viewer instead
of the shared one -- which is the whole reason it can exist beside an open
inspector pose session without either discarding the other.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import poser_mode, theme, widgets
from ..manual import render as manual_render


def draw(ctx: Any) -> None:
    state = poser_mode.ensure(ctx)
    if not widgets.header("Pose"):
        return
    manual_render.help_button(ctx, "poser-controls")
    if not ctx.rigging_available:
        widgets.muted("Posing needs Blender, which is not installed.")
        return
    viewer = poser_mode.viewer_of(ctx)
    if viewer is None or not viewer.pose_mode:
        widgets.muted("The skeleton preview is still loading.")
        return

    _banner(state, viewer)
    _joint(viewer)
    _root(viewer)
    _save(ctx, viewer)


def _banner(state: Any, viewer: Any) -> None:
    record = next(
        (p for p in state.poses if p.get("id") == viewer.editor.current), None
    )
    label = str(record.get("name")) if record else "New pose"
    if viewer.editor.has_unsaved_edits():
        label += " - unsaved changes"
    widgets.text_colored(theme.ACCENT, label)


def _joint(viewer: Any) -> None:
    selected = viewer.selected_bone
    widgets.muted(selected or "Click a joint to rotate it.")
    if widgets.disabled_button("Reset joint", selected is not None):
        viewer.reset_bone()
    imgui.same_line()
    if imgui.button("Reset all"):
        viewer.reset_all()
    if viewer.editor.mirror_pairs:
        # Hidden for a serpent or a fish, the pose_panel rule: a skeleton with
        # no mirror pairs has nothing to mirror.
        imgui.same_line()
        if imgui.button("Mirror"):
            viewer.mirror()


def _root(viewer: Any) -> None:
    editor = viewer.editor
    if editor.root is None:
        return
    offset = editor.root_translation()
    if editor.selected == editor.root:
        # Shown only while the root is selected: the toggle changes what the
        # gizmo on that joint does, and drawing it against any other joint
        # would claim a capability the selection does not have.
        changed, value = imgui.checkbox("Move root", editor.root_translate)
        if changed:
            editor.root_translate = bool(value)
        if editor.root_translate:
            widgets.muted_wrapped(
                "Drag the arrows to offset the whole pose. Units are character "
                "heights; the bake scales them onto each asset's own rig."
            )
    if any(offset):
        widgets.muted(
            f"root offset  x {offset[0]:+.2f}  y {offset[1]:+.2f}  z {offset[2]:+.2f}"
        )


def _save(ctx: Any, viewer: Any) -> None:
    imgui.dummy((0, 8))
    busy = ctx.busy(poser_mode.SAVE_KEY)
    editing = viewer.editor.current is not None
    if editing and widgets.disabled_button("Save", not busy, (-1, 0)):
        poser_mode.save(ctx)
    if widgets.disabled_button("Save as...", not busy, (-1, 0)):
        poser_mode.save_as(ctx)
