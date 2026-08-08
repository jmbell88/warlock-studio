"""The app's own settings: how it looks, where its panes sit, what it loaded.

Deliberately not the generation sidebar, which is also called "settings" and
owns a job's parameters. Nothing here belongs to a job; everything here
survives a restart, and the child id says ``app-settings`` for exactly that
reason -- ``settings`` is already taken.

The model list answers "does it know about the model I downloaded", which used
to require reading the log -- and now also downloads it. It was read-only by
design for as long as the app had no way to fetch anything; the app still has
none, and that is the point of how the button works: it plans the fetch here
(``warlock.fetch``, pure) and hands it to a child process
(``pipelines/fetch_worker``) that sets ``HF_HUB_OFFLINE=0`` in its own
environment and dies. Nothing in this process ever becomes online-capable, and
nothing in the generation pipeline can reach any of it.

Everything else on the pane is still the app's own settings rather than a
job's, and still read-only where the answer is not the user's to change.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from .. import app_ctx, icons, theme, tokens, widgets
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
        _config(ctx)
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
    widgets.help_marker(
        "On top of what the monitor already scales by, so 1.00x is the size "
        "Windows asked for rather than 96 dpi."
    )
    if changed:
        # Live, so dragging shows what it will look like -- but only committed
        # on release: every intermediate value would otherwise be a settings
        # write and a full style rebuild per mouse-move.
        _apply_scale(ctx, value)
    if imgui.is_item_deactivated_after_edit():
        ctx.settings.set("ui_scale", round(float(value), 2))
        # On release only (K99): re-baking the atlas per mouse-move would be a
        # font rebuild sixty times a second, and the flag is consumed between
        # frames rather than here for the reason ``fonts.reload`` gives.
        ctx.state.fonts_dirty = True
    if hi < tokens.UI_SCALE_RANGE[1]:
        widgets.muted(
            f"This display already scales by {_base(ctx):.2f}x, which leaves room for {hi:.2f}x."
        )
    widgets.muted("Icons and text re-bake when you let go of the slider.")

    # M105. The palette is a table of names in ``tokens`` and every pane reads
    # ``theme.NAME``, so switching is this plus a re-``apply`` -- imgui's style
    # holds *copies* of the numbers, which is the one thing the live lookup
    # cannot do for it.
    chosen = widgets.labeled_combo(
        "Theme",
        tokens.THEME,
        [(name, name) for name in tokens.PALETTES],
    )
    widgets.help_marker(
        "The whole palette, including the viewport background. It takes effect "
        "at once and is remembered."
    )
    if chosen != tokens.THEME:
        _apply_theme(ctx, chosen)

    show_fps = bool(ctx.state.show_fps)
    changed, show_fps = imgui.checkbox("Show frame rate (F10)", show_fps)
    if changed:
        ctx.state.show_fps = show_fps
        ctx.settings.set("show_fps", show_fps)


def _apply_theme(ctx: Any, name: str) -> None:
    from .. import theme as theme_mod

    applied = tokens.set_theme(name)
    theme_mod.apply(imgui)
    ctx.settings.set("theme", applied)


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
    atlas is *not* rebuilt here: that has to happen between frames, so the
    caller raises ``state.fonts_dirty`` on release and the frame loop consumes
    it (K99).
    """
    from .. import theme as theme_mod

    base = _base(ctx)
    lo, hi = tokens.ui_scale_bounds(base)
    tokens.set_scale(base * min(max(float(value), lo), hi))
    theme_mod.apply(imgui)


# --- effective configuration ------------------------------------------------


def _config(ctx: Any) -> None:
    """K100: what this process is actually running on, in the Settings pane.

    Collapsed, because thirty rows is a wall of text -- and read-only, because
    every one of these is an environment variable the app process consumed at
    import time. An editable version would have to say "restart to apply" under
    every field, which is a settings pane that cannot change a setting.
    """
    widgets.section("Configuration")
    if not imgui.collapsing_header("Effective configuration##app-settings"):
        return
    config_table(ctx)
    if imgui.small_button("Copy as text"):
        from ...config import effective

        imgui.set_clipboard_text(
            "\n".join(
                f"{s.env if s.from_env else s.name} = {s.value}"
                for s in effective(ctx.runtime.config)
            )
        )


def config_table(ctx: Any) -> None:
    """The rows themselves, shared with the diagnostics popup.

    Overridden rows first (S140). The two or three a host has actually changed
    are the whole diagnostic value, so they are what is visible when the
    section is opened, and the rest is there to confirm a suspicion rather than
    to be read through. It shows the *variable name* for an overridden row and
    the setting's own name otherwise, because an install whose behaviour
    disagrees with the manual almost always disagrees because something in its
    environment says so.

    Shares ``config.effective`` with ``warlock doctor``, which is the point of
    building the data source once: the copy a user pastes into an issue and the
    list they read on screen are the same answer.
    """
    from ...config import effective
    from .. import theme

    settings = effective(ctx.runtime.config)
    overridden = [s for s in settings if s.from_env]
    widgets.muted(
        "Everything at its default."
        if not overridden
        else f"{len(overridden)} of {len(settings)} set by the environment."
    )
    for setting in sorted(settings, key=lambda s: (not s.from_env, s.name)):
        if setting.from_env:
            widgets.text_colored(theme.ACCENT, setting.env)
        else:
            widgets.muted(setting.name)
        imgui.same_line()
        imgui.text_wrapped(setting.value)


# --- layout -----------------------------------------------------------------


def _layout(ctx: Any) -> None:
    from .. import layout as layout_mod

    widgets.section("Layout")
    lay = getattr(ctx, "layout", None)
    if lay is None:
        widgets.muted("No layout to reset.")
        return
    if imgui.button("Reset pane sizes"):
        lay.settings_share = 0.55
        lay.save()
        ctx.toast("Pane sizes reset.")
    imgui.same_line()
    if imgui.button("Reset collapsed sections"):
        # The map, not the individual keys: every section falls back to its own
        # default-open when it finds nothing stored.
        ctx.settings.set("panels_open", {})
        ctx.toast("Section states reset.")
    # M106. Named sizes rather than a drag: the module docstring's argument
    # against dragging a form's width stands, and what it did not answer is
    # that one number cannot suit a 1600-wide window and a 5120 one.
    chosen = widgets.labeled_combo(
        "Sidebar width",
        getattr(lay, "sidebar", "default"),
        [(key, f"{key} ({int(width)} px)") for key, width in layout_mod.SIDEBAR_WIDTHS.items()],
    )
    if chosen != getattr(lay, "sidebar", "default"):
        lay.set_sidebar_width(chosen)


# --- models -----------------------------------------------------------------


# The order rows are grouped in, and the heading each group gets. Written out
# rather than taken from whatever order the rows arrive in, so a registry
# gaining a table cannot silently append an unlabelled block.
_GROUPS = (
    ("base", "Image models"),
    ("lora", "Style LoRAs"),
    ("adapter", "Conditioning"),
    ("control", "Conditioning"),
    ("metric", "Measurement"),
    ("pose", "Measurement"),
    ("matting", "Measurement"),
)


def _models(ctx: Any) -> None:
    widgets.section("Models")
    rows = list(getattr(ctx, "model_rows", None) or [])
    if not rows:
        widgets.muted("No image models registered.")
        return
    # One fetch at a time across the whole pane. Two concurrent children can
    # legitimately want the same destination -- sdxl and sdxl_cfg are one
    # download -- and the staging-directory rule in the worker protects the
    # *destination*, not two writers of one staging tree.
    busy = ctx.tasks.any_busy("download:")
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_kind.setdefault(str(row.get("kind")), []).append(row)

    heading = ""
    for kind, label in _GROUPS:
        group = by_kind.get(kind) or []
        if not group:
            continue
        if label != heading:
            heading = label
            imgui.dummy((0, sp(tokens.SP_1)))
            widgets.muted(label)
        for row in group:
            _row(ctx, row, busy)

    imgui.dummy((0, sp(tokens.SP_1)))
    picks = {r["row_key"] for r in rows if r["row_key"] in ctx.model_picks and not r["present"]}
    if widgets.disabled_button(
        f"Download selected ({len(picks)})", bool(picks) and not busy
    ):
        _start(ctx, sorted(picks), key="download:selection")
    _selection_progress(ctx)

    imgui.dummy((0, sp(tokens.SP_1)))
    if getattr(ctx, "rigging_available", False):
        widgets.text_colored(theme.MUTED, f"{icons.CIRCLE} Rigging (bpy) available")
    else:
        widgets.muted("Rigging (bpy) not installed.")
    imgui.dummy((0, sp(tokens.SP_1)))
    # What the button does and does not do. The offline contract is the app's
    # single most load-bearing property, so the one place that breaks it says so
    # rather than leaving the user to infer it.
    widgets.muted(
        "A download runs in a separate process; this one stays offline. The "
        "startup diagnostics still give the exact command for a missing model."
    )


def _row(ctx: Any, row: dict[str, Any], busy: bool) -> None:
    row_key = str(row["row_key"])
    present = bool(row.get("present"))
    label = str(row.get("label") or row_key)
    if present:
        widgets.text_colored(theme.MUTED, f"{icons.CIRCLE_CHECK} {label}")
        return
    if not row.get("downloadable"):
        widgets.text_colored(theme.WARN, f"{icons.CIRCLE} {label} - weights missing")
        return

    ticked = row_key in ctx.model_picks
    changed, ticked = imgui.checkbox(f"##pick-{row_key}", ticked)
    if changed:
        # Only a missing row can be ticked, and the set is pruned as rows
        # arrive present, so it can never name something already on disk.
        if ticked:
            ctx.model_picks.add(row_key)
        else:
            ctx.model_picks.discard(row_key)
    imgui.same_line()
    size = float(row.get("size_gib") or 0.0)
    suffix = f" ({size:.1f} GB)" if size else ""
    widgets.text_colored(theme.WARN, f"{icons.CIRCLE} {label}{suffix}")

    key = app_ctx.download_key(row_key)
    running = ctx.tasks.is_busy(key)
    if running:
        found = ctx.progress(key)
        widgets.progress_bar(float(found.get("percent") or 0.0) if found else 0.0)
        if found and found.get("label"):
            widgets.muted(str(found["label"]))
        return
    # Its own line rather than same_line after full-width text: the label is
    # long and the sidebar is 300 px, and a button drawn past the edge is gone
    # rather than squeezed.
    if widgets.disabled_button(f"{icons.DOWNLOAD} Download##{row_key}", not busy):
        _start(ctx, [row_key], key=key)


def _selection_progress(ctx: Any) -> None:
    key = "download:selection"
    if not ctx.tasks.is_busy(key):
        return
    found = ctx.progress(key)
    widgets.progress_bar(float(found.get("percent") or 0.0) if found else 0.0)
    if found and found.get("label"):
        widgets.muted(str(found["label"]))


def _start(ctx: Any, row_keys: list[str], *, key: str) -> None:
    """Submit the fetch. The progress callback is bound to the task key here.

    Closed over on the frame thread and called from the task thread --
    ``TaskRunner.set_progress`` is the thread-safe half, and it drops a report
    for a key that is no longer in flight, so a child's last line landing after
    collection cannot resurrect a bar.
    """
    from ...service import downloads as svc_downloads

    def run() -> Any:
        return svc_downloads.download(
            ctx.svc,
            row_keys,
            on_progress=lambda percent, label: ctx.tasks.set_progress(key, percent, label),
        )

    if ctx.submit(key, run):
        ctx.toast("Downloading...")
