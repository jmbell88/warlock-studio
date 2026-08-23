"""The application menu model and its single host-level renderer.

Rows are adapters over the command palette and workspace operation registries;
the menu owns placement, never a second implementation of an action.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MenuSpec:
    """One command wherever it is presented in the menu tree."""

    identity: str
    path: tuple[str, ...]
    order: int
    label: str
    enabled: bool
    checked: bool
    shortcut: str
    disabled_reason: str
    callback: Callable[[], None]
    separator_before: bool = False


ROOTS = ("File", "Edit", "View", "Workspace", "Window", "Help")

_COMMAND_PATHS: dict[str, tuple[str, ...]] = {
    "new-drawing": ("File",),
    "new-clay": ("File",),
    "new-map": ("File",),
    "new-atlas": ("File",),
    "save": ("File",),
    "save-as": ("File",),
    "export": ("File",),
    "quit": ("File",),
    "undo": ("Edit",),
    "redo": ("Edit",),
    "delete": ("Edit",),
    "frame": ("View",),
    "wireframe": ("View",),
    "turntable": ("View",),
    "clear-viewport": ("View",),
    "fps": ("View",),
    "workspace-layout": ("Window",),
    "profiles": ("Window",),
    "show-trash": ("Window",),
    "empty-trash": ("Window",),
    "manual": ("Help",),
    "shortcuts": ("Help",),
    "diagnostics": ("Help",),
    "open-log": ("Help",),
}


def _checked(ctx: Any, key: str) -> bool:
    if key.startswith("go:"):
        return ctx.state.mode == key.partition(":")[2]
    return {
        "wireframe": bool(getattr(ctx.state, "wireframe", False)),
        "turntable": bool(getattr(ctx.state, "turntable", False)),
        "fps": bool(getattr(ctx.state, "show_fps", False)),
        "show-trash": bool(getattr(getattr(ctx.state, "filters", None), "trash", False)),
    }.get(key, False)


def _command_specs(ctx: Any) -> list[MenuSpec]:
    from . import modes, palette

    out: list[MenuSpec] = []
    commands = palette.commands(ctx)
    for index, command in enumerate(commands):
        path = ("Workspace",) if command.key.startswith("go:") else _COMMAND_PATHS.get(command.key)
        if path is None:
            # Mode-specific actions form a contextual menu instead of making
            # File/Edit into miscellaneous command dumps.
            if (
                command.group in ("Actions", "Viewport")
                and ctx.state.mode not in ("home", "settings", "library")
                and ctx.state.mode != "inker"
            ):
                label = next(
                    (name for key, name, _icon in modes.MODES if key == ctx.state.mode),
                    "Actions",
                )
                path = (label,)
            else:
                continue
        enabled = bool(command.enabled(ctx))
        out.append(
            MenuSpec(
                identity=f"command:{command.key}",
                path=path,
                order=index,
                label=command.label,
                enabled=enabled,
                checked=_checked(ctx, command.key),
                shortcut=command.hint,
                disabled_reason=command.why,
                callback=lambda command=command: command.run(ctx),
            )
        )
    return out


def _inker_specs(ctx: Any) -> list[MenuSpec]:
    if ctx.state.mode != "inker":
        return []
    from . import inker_mode, inker_ops
    from .panes import inker_menu

    state = inker_mode.ensure(ctx)
    tab = state.active
    out = []
    for index, op in enumerate(inker_ops.OPS):
        if not op.menu:
            continue
        out.append(
            MenuSpec(
                identity=f"inker:{op.name}",
                path=(op.menu,),
                order=index,
                label=op.label,
                enabled=bool(op.enabled(state, tab)),
                checked=False,
                shortcut=op.key,
                disabled_reason=op.reason,
                callback=lambda op=op: inker_menu.activate(ctx, op),
                separator_before=bool(op.separator_before),
            )
        )
    return out


def specs(ctx: Any, layout: Any = None) -> list[MenuSpec]:
    """The current menu tree as data, rebuilt so state never goes stale."""

    rows = _command_specs(ctx) + _inker_specs(ctx)
    if layout is not None:
        rows.append(
            MenuSpec(
                identity="window:navigation-labels",
                path=("Window",),
                order=10_000,
                label="Navigation labels",
                enabled=True,
                checked=layout.rail == "labels",
                shortcut="",
                disabled_reason="",
                callback=lambda: layout.set_rail("icons" if layout.rail == "labels" else "labels"),
            )
        )
    return rows


def roots(rows: list[MenuSpec]) -> list[str]:
    """Stable root order, with contextual workspace menus before View."""

    present = {row.path[0] for row in rows if row.path}
    contextual: list[str] = []
    for row in rows:
        root = row.path[0] if row.path else ""
        if root and root not in ROOTS and root not in contextual:
            contextual.append(root)
    ordered = ["File", "Edit", *contextual, "View", "Workspace", "Window", "Help"]
    return [name for name in ordered if name in present or name in ROOTS]


def draw(ctx: Any, layout: Any = None) -> None:
    """Render the 26 dp global menu bar in the host window."""

    from imgui_bundle import imgui

    from . import controls, tokens

    rows = specs(ctx, layout)
    imgui.push_style_var(imgui.StyleVar_.frame_padding.value, (tokens.sp(8), tokens.sp(6)))
    opened = imgui.begin_menu_bar()
    imgui.pop_style_var()
    if not opened:
        return
    try:
        for root in roots(rows):
            with controls.menu(root) as menu_open:
                if not menu_open:
                    continue
                for row in sorted(
                    (one for one in rows if one.path and one.path[0] == root),
                    key=lambda one: one.order,
                ):
                    if row.separator_before:
                        controls.menu_separator()
                    hit = controls.menu_item(
                        f"{row.label}##menu/{row.identity}",
                        row.shortcut,
                        row.checked,
                        row.enabled,
                        reason=row.disabled_reason,
                    )
                    clicked = hit[0] if isinstance(hit, tuple) else hit
                    if clicked and row.enabled:
                        row.callback()
    finally:
        imgui.end_menu_bar()
