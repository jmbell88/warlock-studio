"""The app's own settings: how it looks, where its panes sit, what it loaded.

Deliberately not the generation sidebar, which is also called "settings" and
owns a job's parameters. Nothing here belongs to a job; everything here
survives a restart, and the child id says ``app-settings`` for exactly that
reason -- ``settings`` is already taken.

The model list is read-only on purpose. Weights are one-time manual
``hf download``s (the app is offline by contract), so there is nothing here to
click; what the pane is for is answering "does it know about the model I
downloaded", which used to require reading the log.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import icons, theme, tokens, widgets
from ..manual import render as manual_render
from ..tokens import sp


def draw(ctx: Any) -> None:
    # always_use_window_padding, because a *borderless* child gets zero window
    # padding by default -- so this pane's content sat flush against the host
    # window's left edge while every bordered sidebar got the theme's gutter.
    if imgui.begin_child(
        "app-settings", (0, 0), imgui.ChildFlags_.always_use_window_padding.value
    ):
        _interface(ctx)
        _layout(ctx)
        _models(ctx)
    imgui.end_child()


# --- interface --------------------------------------------------------------


def _interface(ctx: Any) -> None:
    widgets.section("Interface")
    # After the section heading, never before begin_child: help_button is a
    # same_line, and same_line returns to the *previous* row unconditionally.
    # Called first in draw() it landed on the mode switch, on top of the health
    # dot -- which every other pane avoids only because a header precedes it.
    manual_render.help_button(ctx, "app-settings")
    lo, hi = tokens.ui_scale_bounds(_base(ctx))
    stored = _scale_of(ctx)
    imgui.set_next_item_width(sp(260))
    changed, value = imgui.slider_float("UI scale", stored, lo, hi, "%.2fx")
    if changed:
        # Live, so dragging shows what it will look like -- but only committed
        # on release: every intermediate value would otherwise be a settings
        # write and a full style rebuild per mouse-move.
        _apply_scale(ctx, value)
    if imgui.is_item_deactivated_after_edit():
        ctx.settings.set("ui_scale", round(float(value), 2))
    if hi < tokens.UI_SCALE_RANGE[1]:
        widgets.muted(
            f"This display already scales by {_base(ctx):.2f}x, which leaves room for {hi:.2f}x."
        )
    widgets.muted("Text sharpens fully after a restart.")

    show_fps = bool(ctx.state.show_fps)
    changed, show_fps = imgui.checkbox("Show frame rate (F10)", show_fps)
    if changed:
        ctx.state.show_fps = show_fps
        ctx.settings.set("show_fps", show_fps)


def _base(ctx: Any) -> float:
    """The monitor's own scale, sampled at startup and never folded back in."""
    return float(getattr(ctx, "dpi_scale", 1.0)) or 1.0


def _scale_of(ctx: Any) -> float:
    base = _base(ctx)
    lo, hi = tokens.ui_scale_bounds(base)
    return min(max(tokens.SCALE / base, lo), hi)


def _apply_scale(ctx: Any, value: float) -> None:
    """Rescale everything drawn from tokens, then rebuild the style from it.

    ``theme.apply`` reads ``tokens.SCALE`` at call time and is idempotent, so
    calling it again is how a new scale reaches padding and rounding. The font
    atlas is not rebuilt -- it is baked at startup, which is why the glyphs
    only get crisper after a restart.
    """
    from .. import theme as theme_mod

    base = _base(ctx)
    lo, hi = tokens.ui_scale_bounds(base)
    tokens.set_scale(base * min(max(float(value), lo), hi))
    theme_mod.apply(imgui)


# --- layout -----------------------------------------------------------------


def _layout(ctx: Any) -> None:
    from .. import layout as layout_mod

    widgets.section("Layout")
    lay = getattr(ctx, "layout", None)
    if lay is None:
        widgets.muted("No layout to reset.")
        return
    if imgui.button("Reset pane sizes"):
        lay.sidebar_w = 340.0
        lay.inspector_w = 340.0
        lay.settings_share = 0.55
        lay.save()
        ctx.toast("Pane sizes reset.")
    imgui.same_line()
    if imgui.button("Reset collapsed sections"):
        # The map, not the individual keys: every section falls back to its own
        # default-open when it finds nothing stored.
        ctx.settings.set("panels_open", {})
        ctx.toast("Section states reset.")
    widgets.muted(
        f"Sidebars are clamped to {int(layout_mod.SIDEBAR_MIN)}-{int(layout_mod.SIDEBAR_MAX)}px."
    )


# --- models -----------------------------------------------------------------


def _models(ctx: Any) -> None:
    widgets.section("Models")
    base = list(getattr(ctx, "base_models", None) or [])
    loras = list(getattr(ctx, "style_loras", None) or [])
    if not base:
        widgets.muted("No image models registered.")
    for _key, label in base:
        _row(label)
    if loras:
        imgui.dummy((0, sp(tokens.SP_1)))
        for key, label in loras:
            if key:
                _row(label)
    imgui.dummy((0, sp(tokens.SP_1)))
    if getattr(ctx, "rigging_available", False):
        _row("Rigging (bpy) available")
    else:
        widgets.muted("Rigging (bpy) not installed.")
    imgui.dummy((0, sp(tokens.SP_1)))
    # What this list is, rather than what it is not yet. A promise about a
    # future release is the one thing a UI string cannot keep, and this pane
    # was the only place in the app making one.
    widgets.muted(
        "Read-only. Weights are one-time manual downloads; the startup "
        "diagnostics give the exact command for a missing one."
    )


def _row(label: str) -> None:
    missing = "missing" in label
    widgets.text_colored(theme.WARN if missing else theme.MUTED, f"{icons.CIRCLE} {label}")
