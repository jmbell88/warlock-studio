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

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from imgui_bundle import imgui

from ... import fetch, models, vram
from ...service import library as svc_library
from .. import app_ctx, controls, dialogs, forms, icons, theme, tokens, widgets
from .. import layouts as layouts_mod
from ..manual import render as manual_render
from ..tokens import sp

#: How wide the settings column is allowed to grow, in design pixels.
#:
#: A settings pane is a *form*, and a form on a 5120-wide monitor was one
#: checkbox on the left and four metres of nothing to its right -- the label and
#: its control so far apart that reading a row meant tracking across the screen.
#: Everything here is one column of short rows, so the column gets a measure and
#: the window gets the rest (the UI redesign, wave 4.1).
CONTENT_W = 640
#: The Models category alone, because the reason ``CONTENT_W`` is 640 does not
#: hold there: "everything here is one column of short rows" stopped being
#: true when those rows became a four-column table with a description in it.
MODELS_CONTENT_W = 980
CATEGORY_W = 184

#: How wide a labelled control in this pane gets. The column is 640 and the
#: values in it are a scale, a palette name and a width -- all short. A combo
#: stretched to the column is a dropdown four times wider than the longest thing
#: it can contain, and it pushes its own ``help_marker`` onto the next line.
FIELD_W = 260

#: The five things this pane is, and the order they are offered in.
#:
#: Appearance first because it is the one most people came for; Advanced last
#: because it is read-only diagnostics. The keys are stable ids -- they are in
#: the segmented control's imgui ids and in ``state.preview`` -- and the labels
#: are what changes when the wording does.
# What each palette is called in the combo. The key is a settings-file value and
# the label is copy, and conflating the two is how a Settings pane comes to say
# "dark" in lower case beside a sentence-cased everything else. Anything absent
# falls back to a title-cased key, so a new palette is readable before it is
# named.
_THEME_LABELS = {
    "dark": "Dark",
    "light": "Light",
    "pixel": "Pixel",
}

CATEGORIES = [
    ("appearance", f"{icons.PALETTE} Appearance"),
    ("models", f"{icons.BOX} Models"),
    ("storage", f"{icons.FOLDER_OPEN} Storage"),
    ("health", f"{icons.ACTIVITY} Health"),
    ("advanced", f"{icons.SETTINGS} Advanced"),
]

#: The all-or-nothing fallback labelling, per ``segmented_control``'s rule: the
#: glyph alone, with the full label restored as a tooltip. Derived from
#: ``CATEGORIES`` so a renamed category cannot leave a stale abbreviation.
CATEGORIES_COMPACT = [(key, label.split(" ", 1)[0]) for key, label in CATEGORIES]

#: Where the chosen category lives. ``state.preview`` and not ``settings``,
#: deliberately: which tab of a settings pane you last had open is not a
#: preference, it is where you were -- and a user who opens Settings to change
#: the theme should not land on a disk-usage figure because they pruned a
#: library a fortnight ago.
CATEGORY_SLOT = "settings_category"


def draw(ctx: Any) -> None:
    # always_use_window_padding, because a *borderless* child gets zero window
    # padding by default -- so this pane's content sat flush against the host
    # window's left edge while every bordered sidebar got the theme's gutter.
    if imgui.begin_child("app-settings", (0, 0), imgui.ChildFlags_.always_use_window_padding.value):
        category = _category_rail(ctx)
        imgui.same_line()
        measure = MODELS_CONTENT_W if category == "models" else CONTENT_W
        width = min(sp(measure), imgui.get_content_region_avail().x)
        if imgui.begin_child("app-settings-body", (width, 0)):
            # Settings draws into ``##content`` rather than through
            # ``layout.pane``, so it asks for its own section blocks. Named here
            # rather than made automatic: a scope splits *this child's* draw
            # list, and the right bracket is the child, which only the code that
            # opened it knows.
            with widgets.section_blocks():
                category_label = dict(CATEGORIES)[category].split(" ", 1)[-1]
                widgets.pane_header(category_label)
                # No outer ``forms.Form`` around the dispatch: no category body
                # ever read it -- ``_interface`` opens its own nested Form for
                # the adaptive layout, and the rest are tables and read-only
                # rows. All a Form contributes besides layout is an id scope,
                # and nothing persisted keys on imgui id paths.
                _category_body(ctx, category)
        imgui.end_child()
    imgui.end_child()


def _category_rail(ctx: Any) -> str:
    """Persistent Settings navigation, independent of content scrolling."""

    current = str(ctx.state.preview.get(CATEGORY_SLOT) or CATEGORIES[0][0])
    if current not in dict(CATEGORIES):
        current = CATEGORIES[0][0]
    if imgui.begin_child("app-settings-categories", (sp(CATEGORY_W), 0)):
        widgets.pane_header("Settings")
        manual_render.help_button(ctx, "app-settings")
        for key, label in CATEGORIES:
            if controls.selectable(f"{label}##settings-category/{key}", key == current)[0]:
                current = key
                ctx.state.preview[CATEGORY_SLOT] = key
    imgui.end_child()
    return current


def _centre(width: float) -> float:
    """Indent the cursor so a ``width``-wide column sits centred. -> its width.

    Returns the width actually available, which is ``width`` only when the
    window is wide enough to hold it -- a narrow window gets the whole region
    and no indent, because centring something that already fills its container
    is a no-op with a rounding error in it.
    """
    avail = imgui.get_content_region_avail().x
    if avail <= width:
        return avail
    imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + (avail - width) * 0.5)
    return width


def _categories(ctx: Any, width: float) -> str:
    """The category switch, and the chosen key."""
    current = str(ctx.state.preview.get(CATEGORY_SLOT) or CATEGORIES[0][0])
    if current not in dict(CATEGORIES):
        current = CATEGORIES[0][0]
    chosen = widgets.segmented_control(
        "settings-cat",
        CATEGORIES,
        current,
        compact=CATEGORIES_COMPACT,
        max_width=width,
    )
    if chosen != current:
        ctx.state.preview[CATEGORY_SLOT] = chosen
    # On the switch's line, not the title's. ``help_button`` is a right-aligned
    # ``same_line``, and ``pane_title`` ends in a spacer -- so calling it after
    # the title put the (?) alone on an otherwise empty row, floating between
    # the heading and the categories with nothing to belong to.
    manual_render.help_button(ctx, "app-settings")
    imgui.dummy((0, sp(tokens.SP_2)))
    return chosen


def _category_body(ctx: Any, category: str) -> None:
    if category == "models":
        _models(ctx)
    elif category == "storage":
        _storage(ctx)
    elif category == "health":
        _health(ctx)
    elif category == "advanced":
        _layout(ctx)
        _layouts(ctx)
        _config(ctx)
    else:
        _interface(ctx)


# --- interface --------------------------------------------------------------


def _interface(ctx: Any, form_ui: forms.Form | None = None) -> None:
    # Optional keeps the small source-level/unit probes able to call this piece
    # directly while the real pane supplies the adaptive form context.
    if form_ui is None:
        with forms.Form("application-settings/interface") as nested:
            _interface(ctx, nested)
        return
    # No section heading: the lit segment above already says "Appearance", and a
    # heading repeating it is a second answer to a question nobody asked. The
    # categories that hold *more than one* group keep theirs.
    lo, hi = tokens.ui_scale_bounds(_base(ctx))
    stored = _scale_of(ctx)
    imgui.set_next_item_width(sp(FIELD_W))
    changed, value = controls.slider_float("UI scale", stored, lo, hi, "%.2fx")
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
    # cannot do for it. The options come from the table rather than from a list
    # here, so a palette added there appears without an edit in this pane; only
    # the *label* is spelled below, because a dict key is not a sentence.
    _changed, chosen = form_ui.combo(
        "theme",
        "Theme",
        tokens.THEME,
        [(name, _THEME_LABELS.get(name, name.title())) for name in tokens.PALETTES],
        help_text="The whole palette: chrome, canvas surround and every hand-drawn edge.",
        helper="It takes effect at once and is remembered.",
    )
    if chosen != tokens.THEME:
        _apply_theme(ctx, chosen)

    show_fps = bool(ctx.state.show_fps)
    changed, show_fps = form_ui.switch("show_fps", "Show frame rate (F10)", show_fps)
    if changed:
        ctx.state.show_fps = show_fps
        ctx.settings.set("show_fps", show_fps)

    show_resources = bool(ctx.state.show_resources)
    changed, show_resources = form_ui.switch(
        "show_resources",
        "System resources",
        show_resources,
        help_text=(
            "VRAM, RAM and CPU at the right end of the status bar -- a live "
            "reading of what the machine has left. VRAM is the figure a "
            "refused generation is about."
        ),
    )
    if changed:
        ctx.state.show_resources = show_resources
        ctx.settings.set("show_resources", show_resources)

    reduced = bool(ctx.state.reduce_motion)
    changed, reduced = form_ui.switch(
        "reduce_motion",
        "Reduce motion",
        reduced,
        help_text="Turns off transitions, hover motion, and sliding switches.",
        helper="Everything still changes, just without moving.",
    )
    if changed:
        _apply_reduce_motion(ctx, reduced)
    # There was an "Effects" section here: the four Phase 5 GPU-tier switches
    # (soft shadows, translucent panels, spring motion, continuous corners).
    # The phase shipped them behind "a config flag per item while it
    # stabilizes, folded away once trusted", and on 2026-08-12 they were folded
    # away. Each effect already degrades to the pre-Phase-5 drawing on its own
    # when the GL side is missing, which is what the flags were insuring
    # against; "Reduce motion" above remains the accessibility switch, and it
    # is a different question from a rendering tier.


def _apply_reduce_motion(ctx: Any, reduced: bool) -> None:
    """The state, the module flag and the settings file, in one place.

    Honoured *centrally* in ``motion``: every animated surface in the app reads
    its number from that module, so this is one switch rather than one per
    widget. Nothing has to be repainted or rebuilt -- unlike the theme and the
    scale, motion is read per frame and the next one is already the new answer.
    """
    from .. import motion

    ctx.state.reduce_motion = reduced
    motion.set_reduced(reduced)
    ctx.settings.set("reduce_motion", reduced)


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


def _layouts(ctx: Any) -> None:
    """Saved workspace layouts: the administration, and **the canonical path**.

    Here rather than in the switcher popup because Settings is reachable from
    the rail in every mode and no workspace layout can touch its single-column
    composition -- so this is the one screen that cannot be lost by a layout
    going wrong, which is exactly where the thing that fixes a layout belongs.
    """
    library = getattr(ctx, "layouts", None)
    if library is None:
        return
    widgets.section("Workspace layouts")
    names = sorted(library.layouts)
    chosen = widgets.labeled_combo(
        "Layout",
        library.active,
        [
            (
                name,
                name if library.layouts[name].readable else f"{name} (a newer version)",
            )
            for name in names
        ],
        sp(FIELD_W),
        help_text=(
            "Which panes are in which column, and how tall. It does not carry "
            "the sidebar width, the rail, the UI scale or the theme -- those "
            "are the app's, not a workspace's."
        ),
    )
    if chosen != library.active and library.layouts[chosen].readable:
        library.set_active(chosen)
    width = widgets.grid_width(3)
    if controls.button("Duplicate", (width, 0)):
        base = f"{library.active} copy"
        name, index = base, 2
        while name in library.layouts:
            name, index = f"{base} {index}", index + 1
        library.duplicate(library.active, name)
        library.set_active(name)
        ctx.toast(f"Copied to {name}.")
    imgui.same_line()
    if widgets.disabled_button(
        "Rename...",
        library.active not in layouts_mod.BUILT_IN,
        (width, 0),
        reason="A built-in layout keeps its name -- the reset commands say it.",
    ):
        ctx.prompts.ask(
            dialogs.Prompt(
                title="Rename this layout",
                label="Name",
                value=library.active,
                on_accept=lambda text: library.rename(library.active, text.strip()[:40]),
            )
        )
    imgui.same_line()
    if controls.button("Reset", (width, 0)):
        library.reset()
        ctx.toast(f"{library.active} is back to the built-in arrangement.")
    if widgets.disabled_button(
        "Delete this layout",
        library.active not in layouts_mod.BUILT_IN,
        (-1, 0),
        reason=(
            "A built-in is reset rather than deleted: there is no state in "
            "which a pane cannot be got back."
        ),
    ):
        library.delete(library.active)
    widgets.muted_wrapped(
        "A layout can only reorder and hide panes -- never delete one -- and a "
        "hidden pane is always listed here with one click to bring it back."
    )


def _config(ctx: Any) -> None:
    """K100: what this process is actually running on, in the Settings pane.

    Collapsed, because thirty rows is a wall of text -- and read-only, because
    every one of these is an environment variable the app process consumed at
    import time. An editable version would have to say "restart to apply" under
    every field, which is a settings pane that cannot change a setting.
    """
    widgets.section("Configuration")
    if not controls.collapsing_header("Effective configuration##app-settings"):
        return
    config_table(ctx)
    if controls.small_button("Copy as text"):
        from ...config import effective

        imgui.set_clipboard_text(
            "\n".join(
                f"{s.env if s.from_env else s.name} = {s.value}"
                for s in effective(ctx.runtime.config)
            )
        )


def config_table(ctx: Any) -> None:
    """The rows themselves, shared with the health/reporting surfaces.

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


# --- health -----------------------------------------------------------------


@dataclass(frozen=True)
class HealthRow:
    """One check, resolved to what is drawn for it."""

    name: str
    detail: str
    glyph: str
    colour: int
    ok: bool
    #: What happens *if* it fails, so it is set on passing rows too. Only
    #: meaningful alongside ``ok``; see ``health_rows``' band.
    fatal: bool = False


def health_rows(checks: Iterable[Any]) -> list[HealthRow]:
    """Every check as a row.

    Pure, so the wording is assertable without a window -- which is the half of
    the old diagnostics popup that was never covered, and the half that had to
    be read off a screenshot to know whether a failing check said anything at
    all. Attributes are read defensively because this also renders the static
    checks a partially-started runtime carries.

    **Failures first, fatal before warning** -- the same rule and the same
    reason as ``config_table``'s overridden rows (S140). A healthy install runs
    past thirty checks, so a reader who arrived by clicking "2 things need
    attention" was being asked to find those two in a wall of green ticks. The
    order within each band is doctor's own, which is roughly the order things
    are needed in.
    """
    rows: list[HealthRow] = []
    for check in checks:
        ok = bool(getattr(check, "ok", False))
        fatal = bool(getattr(check, "fatal", False))
        rows.append(
            HealthRow(
                name=str(getattr(check, "name", "")),
                detail=str(getattr(check, "detail", "")),
                # Lucide, as the status pills are: "o" and "x" were the last
                # hand-spelled state glyphs in the app, and a lowercase o at
                # 11 px beside a red x is two letters rather than two shapes.
                glyph=icons.CHECK if ok else icons.CIRCLE_X,
                colour=theme.OK if ok else (theme.ERR if fatal else theme.WARN),
                ok=ok,
                fatal=fatal,
            )
        )

    def band(row: HealthRow) -> int:
        # Not keyed on the colour: a *passing* check still carries the fatal
        # flag -- it says what happens if it ever fails -- so anything reading
        # "fatal" without "not ok" first would reorder the green band too.
        if row.ok:
            return 2
        return 0 if row.fatal else 1

    # ``sorted`` is stable, so doctor's own order survives inside each band.
    return sorted(rows, key=band)


def health_summary(rows: list[HealthRow]) -> str:
    """The line above the list, which is the number Home's health row shows."""
    if not rows:
        # Not an error state: the first poll lands a moment after the window
        # opens, so an empty list means "not yet" and never "nothing to say".
        return "No checks have run yet."
    failing = [row for row in rows if not row.ok]
    if not failing:
        return "Everything checks out."
    return f"{len(failing)} of {len(rows)} need attention."


def health_report(rows: list[HealthRow]) -> str:
    """What *Copy details* puts on the clipboard: the list, as pasteable text."""
    return "\n".join(f"{'ok' if row.ok else 'FAIL'} {row.name}: {row.detail}" for row in rows)


def _health(ctx: Any) -> None:
    """What doctor found, named -- a category rather than the old popup.

    The checks were readable in one place: a popup opened from the rail's
    Issues badge and from the status bar. When that popup went, so did the only
    surface that said *which* check failed and why. A fatal one still reaches
    the error banner, but a non-fatal one -- "Blender (rigging)", a style LoRA
    whose file is missing -- became a number on Home's health row and a
    tooltip, and `warlock doctor` in a terminal was the only way to read it.

    A category and not a popup because Home's health row already points here,
    because it is the same question as the Models table beside it, and because
    a modal that has to be closed before the Settings it names can be changed
    was the popup's own worst habit.
    """
    rows = health_rows(getattr(ctx.runtime, "checks", []) or [])
    widgets.section("Checks")
    widgets.muted(health_summary(rows))
    for row in rows:
        widgets.text_colored(row.colour, row.glyph)
        imgui.same_line()
        imgui.text(row.name)
        # The detail under the name rather than chained onto it. The popup
        # these rows come from was 480 px of its own and still ran a glyph, a
        # name, a dash and a sentence across one line; in a settings column
        # beside a category rail there is no room for the fourth, and
        # ``same_line`` past the content edge clips rather than wraps.
        if row.detail:
            imgui.indent()
            widgets.muted_wrapped(row.detail)
            imgui.unindent()
    _dismissed(ctx)
    _health_actions(ctx, rows)


def _dismissed(ctx: Any) -> None:
    """What Dismiss took off the error banner (F59).

    Every writer of ``state.errors`` fires once, so clearing the list is the
    only copy gone: a worker that died is reported through that list and
    through no doctor row at all. ``dismiss_errors`` moves rather than deletes
    for this reader's sake, and without one it was deleting.
    """
    messages = list(getattr(ctx.state, "dismissed_errors", []) or [])
    if not messages:
        return
    widgets.section("Dismissed")
    for message in messages:
        widgets.text_colored(theme.ERR, icons.TRIANGLE_ALERT)
        imgui.same_line()
        imgui.text_wrapped(str(message))


def _health_actions(ctx: Any, rows: list[HealthRow]) -> None:
    """The three things a reader of a failing row wants next.

    Laid out through ``same_line_or_wrap`` rather than a bare ``same_line``:
    four buttons is more than a settings column holds beside a category rail,
    and the popup these came from was a window of its own. The helper asks the
    layout whether the next one fits and starts a row when it does not, which
    is the one exemption the overflow walk allows.
    """
    # These act on the whole page rather than on the "Dismissed" block that may
    # precede them, and inventing a heading to say so would be labelling a
    # thing to fix a rectangle -- so the block ends and they belong to nothing.
    widgets.end_section()
    if controls.button("Copy details", role=controls.ButtonRole.GHOST):
        imgui.set_clipboard_text(health_report(rows))
    # Re-ask rather than wait out the poll (N111). The static half is only
    # recomputed on ``force``, which is what makes this worth having at all:
    # having just installed the weights a row names, nothing short of a restart
    # would otherwise change its mind.
    widgets.same_line_or_wrap(sp(160))
    if controls.button("Run checks again", role=controls.ButtonRole.GHOST):
        from ...service import system as svc_system

        ctx.submit("health", svc_system.current_checks, ctx.svc, force=True)
    # Chapter 12. The rows name the failure and its remedy; what they cannot
    # hold is what to do when the remedy does not take.
    widgets.same_line_or_wrap(sp(160))
    manual_render.troubleshooting_button(ctx)
    from .. import component_gallery

    if component_gallery.enabled():
        widgets.same_line_or_wrap(sp(180))
        if controls.button("Component gallery", role=controls.ButtonRole.GHOST):
            component_gallery.request()


# --- layout -----------------------------------------------------------------


def _layout(ctx: Any) -> None:
    from .. import layout as layout_mod

    widgets.section("Layout")
    lay = getattr(ctx, "layout", None)
    if lay is None:
        widgets.muted("No layout to reset.")
        return
    if controls.button("Reset pane sizes"):
        lay.settings_share = 0.55
        # Every keyed split too, or "reset" would leave each workspace on
        # whatever it had drifted to and only move the ones that never did.
        lay.shares.clear()
        lay.save()
        ctx.toast("Pane sizes reset.")
    imgui.same_line()
    if controls.button("Reset collapsed sections"):
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
        sp(FIELD_W),
    )
    if chosen != getattr(lay, "sidebar", "default"):
        lay.set_sidebar_width(chosen)


# --- storage ----------------------------------------------------------------


_MEASURED = False


def _reset_measure() -> None:
    """For tests, for ``_reset_sweep``'s reason."""
    global _MEASURED
    _MEASURED = False


def _model_storage(ctx: Any) -> None:
    """The measured size of the model store, asked for once per session."""
    global _MEASURED
    from ..state import format_bytes

    if not _MEASURED:
        _MEASURED = True
        from ...service import downloads as svc_downloads

        ctx.submit("model-storage", svc_downloads.disk_usage, ctx.svc)
    found = getattr(ctx, "model_storage", None)
    if found:
        noun = "file" if found["files"] == 1 else "files"
        widgets.muted(f"{found['files']} model {noun} - {format_bytes(int(found['bytes']))}")
    else:
        widgets.muted("Measuring the model store...")


def _storage(ctx: Any) -> None:
    """What the library is holding on disk, and the two ways to hold less.

    Both figures are the library's own -- ``ctx.cache.storage`` for the
    workshop and ``library.measure_trash`` for the trash -- because a second
    measurement of the same directories would be a second answer to one
    question, and the two would disagree for as long as one of them was stale.

    The **buttons** moved here from the library's footer (the UI redesign, wave
    4.1). Under a scrolling list of assets, "Clean library..." reads as an
    action on the assets you can see rather than on all of them; and that
    footer is the narrowest row in the app's narrowest column, which is how it
    put Prune past the panel's right edge twice in one file's history. The
    *confirms* stay in ``library.py`` -- the wording of a destructive question
    is the feature, and it is asserted where it lives.
    """
    from ..state import format_bytes
    from . import library

    # No heading: the lit segment says "Storage". Maintenance below keeps
    # its own, because it is a second group inside this one category.
    if ctx.cache.storage_error:
        # Before the figure rather than after it (E45): a stale total beside a
        # warning reads as current, and this line is the only thing saying the
        # number below is the last one that worked.
        widgets.text_colored(theme.WARN, "Could not measure disk use.")
        if imgui.is_item_hovered():
            imgui.set_tooltip(ctx.cache.storage_error)
    storage = ctx.cache.storage
    if storage:
        widgets.muted(f"{storage['job_dirs']} jobs - {format_bytes(int(storage['bytes']))}")
    else:
        widgets.muted("Measuring what is on disk...")
    # The weights, which appeared in no storage view at all: every figure in
    # Models is the registry's *declared* size, so on a full install the
    # largest thing on the disk was the one thing this pane never mentioned.
    _model_storage(ctx)
    # An empty page rather than the loaded one: this pane has no list of
    # trashed jobs to hand over, and the argument is only the stamp that
    # decides *when* to walk the disk again -- so Settings asks under its own
    # stamp and the library goes on asking under the page it is showing.
    summary = library.trash_summary(library.measure_trash(ctx, []))
    widgets.muted(f"In the trash: {summary}" if summary else "Measuring the trash...")

    widgets.section("Maintenance")
    # The two non-destructive ones first, and deliberately above Prune: this
    # group is otherwise entirely made of buttons that remove things, and the
    # answer to "is my library alright" should not sit below the two ways to
    # make it less alright.
    if controls.button("Check library"):
        ctx.submit("library-verify", svc_library.verify, ctx.svc)
    widgets.muted(
        "Looks for assets whose files are gone, folders no asset claims, and "
        "reviews whose asset was deleted. Changes nothing."
    )
    imgui.dummy((0, sp(tokens.SP_1)))
    if controls.button("Back up the index"):
        # No folder dialog. A backup nobody can be bothered to take is worth
        # nothing, and the destination is not a decision worth a modal -- it
        # goes to a stamped folder beside the library, which is also where a
        # user looking for one would think to look. ``warlock library backup
        # --to`` is the surface for putting it somewhere else.
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        dest = Path(ctx.svc.config.home) / "backups" / stamp
        ctx.submit("library-backup", svc_library.backup, ctx.svc, dest)
    widgets.muted(
        "Copies the database holding every prompt, seed, name, tag and review "
        "-- the part that cannot be regenerated from the files -- into a "
        "stamped folder under your library."
    )
    imgui.dummy((0, sp(tokens.SP_1)))
    if controls.button("Prune..."):
        library.ask_prune(ctx)
    widgets.muted(
        "Deletes everything but the newest few assets from disk. Running jobs, "
        "and anything you accepted or labelled, are kept."
    )
    imgui.dummy((0, sp(tokens.SP_1)))
    # Its own row, deliberately not a ``same_line`` after Prune: these two are
    # not a pair of equals, and a destructive button beside an ordinary one
    # invites the wrong click of the two.
    if widgets.destructive_button("Clean library...", (0, 0)):
        library.ask_clean(ctx)
    widgets.muted(
        "Deletes every asset, trashed or not -- including the accepted ones and "
        "the labelled images the quality judge is measured against."
    )


# --- models -----------------------------------------------------------------


# The order rows are grouped in, and the heading each group gets -- taken from
# the vocabulary that owns it (``warlock.fetch.KINDS``) rather than hand-copied
# here. The copy claimed a registry gaining a table could not silently append an
# unlabelled block, and did the opposite: this loop only draws the kinds it
# lists, so an unlisted one vanished from the pane entirely with nothing to say
# it existed. Derived, plus the fallback heading below, is what makes that true.
_GROUPS = fetch.GROUPS

# Where a kind this build has never heard of lands. It should be unreachable --
# the rows come from the same table these groups do -- but "unreachable" is what
# the hand-copy assumed too, and a labelled block a user can read beats a
# download that is simply not offered.
_UNGROUPED = "Other"


# One sweep per session, on the first frame the Models category is drawn. A
# module flag rather than ctx state: it is a fact about this process's disk,
# not a preference, and re-sweeping on every visit would walk ~17 directories
# for a case that only arises after a cancelled fetch.
_SWEPT = False


def _reset_sweep() -> None:
    """For tests, which get a fresh service per case and not a fresh module."""
    global _SWEPT
    _SWEPT = False


def _sweep_staging(ctx: Any) -> None:
    """Reclaim what a cancelled download stranded, and say that it happened.

    ``downloads._sweep_staging`` runs at the start of the next fetch, so a
    user who cancels once and never downloads again keeps the staging tree
    forever -- invisible to every presence probe and to Storage. Off the frame
    thread, because it removes directories.
    """
    global _SWEPT
    if _SWEPT or ctx.tasks.any_busy("download:"):
        return
    _SWEPT = True
    from ...service import downloads as svc_downloads

    ctx.submit("sweep-staging", svc_downloads.sweep_staging, ctx.svc)


def _models(ctx: Any) -> None:
    # No heading: the lit segment says "Models". See ``_interface``.
    _sweep_staging(ctx)
    rows = list(getattr(ctx, "model_rows", None) or [])
    if not rows:
        widgets.muted("No image models registered.")
        return
    # One fetch at a time across the whole pane. Two concurrent children can
    # legitimately want the same destination -- sdxl and sdxl_cfg are one
    # download -- and the staging-directory rule in the worker protects the
    # *destination*, not two writers of one staging tree.
    # ...and a removal counts as one: the two operate on the same directories
    # (uninstalling ``sdxl`` while the shared checkpoint is being fetched is a
    # race with no good outcome), so one mutation at a time across the pane.
    busy = ctx.tasks.any_busy("download:") or ctx.tasks.any_busy("remove:")
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_kind.setdefault(str(row.get("kind")), []).append(row)

    ordered = [*_GROUPS, *((k, _UNGROUPED) for k in by_kind if k not in dict(_GROUPS))]
    # Sorted by *heading*, first-appearance order, so kinds that share one are
    # adjacent however the registry lists them. Without it the prompt expander
    # -- which is an image model and is declared after the ControlNets --
    # drew a second "Image models" caption and a second header row further
    # down the pane, which reads as two different sections of the same name.
    order: dict[str, int] = {}
    for index, (_kind, label) in enumerate(ordered):
        # ``setdefault``, not a comprehension: a later kind sharing a heading
        # must not move the heading to its own position, which is how the
        # first attempt at this put Style LoRAs above Image models.
        order.setdefault(label, index)
    ordered.sort(key=lambda pair: order.get(pair[1], len(order)))
    heading = ""
    pending: list[dict[str, Any]] = []
    for kind, label in ordered:
        group = by_kind.get(kind) or []
        if not group:
            continue
        if label != heading:
            if pending:
                _table(ctx, heading, pending, busy)
                pending = []
            heading = label
            imgui.dummy((0, sp(tokens.SP_1)))
            widgets.muted(label)
        # Accumulated across kinds, because a *heading* can cover more than
        # one: Conditioning is IP-Adapter and ControlNet, and drawing a table
        # each gave one heading two header rows.
        pending.extend(group)
    if pending:
        _table(ctx, heading, pending, busy)

    imgui.dummy((0, sp(tokens.SP_1)))
    recommended = str(getattr(ctx, "recommended_base_label", "") or "")
    if recommended:
        widgets.muted(f"Recommended for this GPU: {recommended}")
    picks = {r["row_key"] for r in rows if r["row_key"] in ctx.model_picks and not r["present"]}
    if widgets.disabled_button(
        f"Download selected ({len(picks)})",
        bool(picks) and not busy,
        reason=(
            "A download is already running."
            if busy
            else "Tick a model above that is not on disk yet."
        ),
    ):
        _start(ctx, sorted(picks), key="download:selection")
    _selection_progress(ctx)

    _loras(ctx)

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


# --- style LoRAs the user brings ---------------------------------------------------

_LORA_FAMILIES = (("sdxl", "SDXL"), ("flux2klein", "FLUX.2 klein"))


def _loras(ctx: Any) -> None:
    """Import a LoRA file, train one from a folder, remove one.

    Under the model tables because that is where a style LoRA is otherwise
    agreed to; its own block because none of these rows comes from the
    download registry. Everything a row *is* -- family, trigger, weight -- is
    what the picker and the loader read, so the form asks for exactly those.
    """
    from ... import generation
    from ...pipelines import lora_train
    from ...service import loras as svc_loras

    imgui.dummy((0, sp(tokens.SP_1)))
    widgets.muted("Your style LoRAs")
    manual_render.help_button(ctx, "loras")
    rows = generation.load_lora_manifests(ctx.svc.config)
    if not rows:
        widgets.muted("None yet. Import a .safetensors adapter, or train one from your own art.")
    busy = ctx.tasks.any_busy("lora:")
    for row in rows:
        weight = f"{row.tuned_weight:.2f}"
        source = "trained here" if row.source.startswith("trained:") else row.source
        widgets.muted(f"{row.label}  -  {row.family}, weight {weight}, {source}")
        if row.trigger_text:
            imgui.same_line()
            widgets.muted(f'  trigger "{row.trigger_text}"')
        imgui.same_line()
        if controls.small_button(f"Remove##lora-{row.key}") and not busy:
            ctx.submit(f"lora:remove:{row.key}", svc_loras.remove_lora, ctx.svc, row.key)
            ctx.toast(f"Removed {row.label}.")

    imgui.dummy((0, sp(tokens.SP_1)))
    if widgets.disabled_button("Import a LoRA file...", not busy):
        picked = dialogs.open_file("Import a style LoRA", ["*.safetensors"])
        if picked is not None:
            ctx.state.preview["lora_import"] = {
                "source": str(picked),
                "label": picked.stem[: lora_train.MAX_LABEL],
                "trigger_text": "",
                "tuned_weight": models.DEFAULT_LORA_WEIGHT,
                "family": models.FAMILY_SDXL,
                "commercial": False,
            }
    imgui.same_line()
    if widgets.disabled_button("Train from a folder...", not busy):
        folder = dialogs.select_folder("Pick a folder of images in the style")
        if folder is not None:
            ctx.state.preview["lora_train"] = {
                "folder": str(folder),
                "label": folder.name[: lora_train.MAX_LABEL],
                "trigger": f"{folder.name} style"[: lora_train.MAX_TRIGGER],
                "steps": lora_train.DEFAULT_STEPS,
            }
    _lora_import_form(ctx)
    _lora_train_form(ctx)


def lora_import_kwargs(form: dict[str, Any]) -> dict[str, Any]:
    """The keyword arguments ``service.loras.import_lora`` takes, from the form."""
    return {
        "label": str(form.get("label") or ""),
        "family": str(form.get("family") or models.FAMILY_SDXL),
        "trigger_text": str(form.get("trigger_text") or ""),
        "tuned_weight": float(form.get("tuned_weight") or models.DEFAULT_LORA_WEIGHT),
        "commercial": bool(form.get("commercial")),
    }


def training_images(folder: Path) -> list[Path]:
    """Every image the trainer would take from ``folder``, sorted, not recursive."""
    from ...service import loras as svc_loras

    try:
        return sorted(
            p for p in Path(folder).iterdir()
            if p.is_file() and p.suffix.lower() in svc_loras.IMAGE_SUFFIXES
        )
    except OSError:
        return []


def _lora_import_form(ctx: Any) -> None:
    from ...service import loras as svc_loras

    form = ctx.state.preview.get("lora_import")
    if not form:
        return
    widgets.muted(f"Importing {Path(form['source']).name}")
    with forms.Form("lora-import", errors=ctx.state.field_errors) as form_ui:
        _c, form["label"] = form_ui.text("label", "Name", form["label"], max_length=64)
        _c, form["trigger_text"] = form_ui.text(
            "trigger_text", "Trigger words", form["trigger_text"], max_length=64,
            help_text="Prepended to every prompt that selects this style.",
        )
        changed, value = form_ui.number(
            "tuned_weight", "Weight", float(form["tuned_weight"]),
            helper=f"0 to {models.LORA_WEIGHT_MAX}",
        )
        if changed:
            form["tuned_weight"] = value
        _c, form["family"] = form_ui.combo("family", "Family", form["family"], _LORA_FAMILIES)
        _c, form["commercial"] = controls.checkbox(
            "Licensed for commercial use", bool(form["commercial"])
        )
    if widgets.disabled_button("Add style", not ctx.busy("lora:import")) and ctx.submit(
        "lora:import", svc_loras.import_lora, ctx.svc, form["source"],
        **lora_import_kwargs(form),
    ):
        ctx.state.preview.pop("lora_import", None)
        ctx.toast("Style added.")
    imgui.same_line()
    if controls.small_button("Cancel##lora-import"):
        ctx.state.preview.pop("lora_import", None)


def _lora_train_form(ctx: Any) -> None:
    from ...pipelines import lora_train
    from ...service import loras as svc_loras

    form = ctx.state.preview.get("lora_train")
    if not form:
        return
    # Scanned when the folder changes, not every frame this form is drawn.
    # ``training_images`` is an ``iterdir`` plus a filter and a sort, and this
    # form stays on screen for as long as it takes to type a name -- on a
    # network path or a large training set that was a per-frame disk hit on the
    # frame thread, for a count that cannot change while the dialog is up.
    if form.get("scanned_folder") != form["folder"]:
        form["images"] = training_images(Path(form["folder"]))
        form["scanned_folder"] = form["folder"]
    images = form["images"]
    widgets.muted(
        f"Training from {Path(form['folder']).name}: {len(images)} images "
        f"({lora_train.MIN_IMAGES} to {lora_train.MAX_IMAGES})"
    )
    with forms.Form("lora-train", errors=ctx.state.field_errors) as form_ui:
        _c, form["label"] = form_ui.text("label", "Name", form["label"], max_length=64)
        _c, form["trigger"] = form_ui.text(
            "trigger", "Trigger words", form["trigger"], max_length=64,
            help_text="The phrase that summons the style in a prompt.",
        )
        changed, value = form_ui.number(
            "steps", "Steps", int(form["steps"]),
            helper=f"{lora_train.MIN_STEPS} to {lora_train.MAX_STEPS}; "
            f"{lora_train.DEFAULT_STEPS} is about half an hour on a fast card",
        )
        if changed:
            form["steps"] = value
    ok = lora_train.MIN_IMAGES <= len(images) <= lora_train.MAX_IMAGES
    pressed = widgets.disabled_button(
        "Train style",
        ok and not ctx.busy("lora:train"),
        reason="" if ok else "The folder needs between 3 and 100 images.",
    )
    if pressed and ctx.submit(
        "lora:train", svc_loras.train_lora, ctx.svc, images,
        label=form["label"], trigger=form["trigger"], steps=int(form["steps"]),
    ):
        ctx.state.preview.pop("lora_train", None)
        ctx.toast("Training queued. The card is yours again when it finishes.")
    imgui.same_line()
    if controls.small_button("Cancel##lora-train"):
        ctx.state.preview.pop("lora_train", None)


#: The blank line that separates a description's short form from its long
#: one. Named because three call sites split or join on it.
_PARA = "\n\n"

#: The four columns, and the three fixed widths. ``Description`` takes what is
#: left, because it is the only column whose content has no natural size.
_COLUMNS = (("Model", 300.0), ("Size", 84.0), ("Description", 0.0), ("Actions", 132.0))


def _table(ctx: Any, heading: str, group: list[dict[str, Any]], busy: bool) -> None:
    """One heading's rows as a table. Per heading, not per kind and not one
    for the pane.

    Per *heading* so the group captions survive between tables, which is what
    keeps ``_GROUPS == fetch.GROUPS`` meaningful -- a single table would have
    to fake them as spanning rows -- and so a heading that covers two kinds
    (Conditioning is IP-Adapter and ControlNet) draws one header row rather
    than two.

    ``row_bg | borders_inner_h`` and **not** full borders: a fully ruled table
    in a pane that removed every other border is exactly the register the
    redesign undid. The horizontal rules are what a four-column row needs to
    be scannable; the vertical ones are decoration.
    """
    flags = (
        imgui.TableFlags_.row_bg.value
        | imgui.TableFlags_.borders_inner_h.value
        | imgui.TableFlags_.sizing_stretch_prop.value
    )
    if not imgui.begin_table(f"##models-{heading}", len(_COLUMNS), flags):
        return
    try:
        for name, width in _COLUMNS:
            if width:
                imgui.table_setup_column(
                    name, imgui.TableColumnFlags_.width_fixed.value, sp(width)
                )
            else:
                imgui.table_setup_column(name, imgui.TableColumnFlags_.width_stretch.value)
        imgui.table_headers_row()
        for row in group:
            imgui.table_next_row()
            _row(ctx, row, busy)
    finally:
        imgui.end_table()


def _fit_badge(row: dict[str, Any]) -> None:
    """The one-line "and can this card hold it" note, under a base row.

    Read with ``.get`` and skipped when absent, never defaulted to a verdict:
    the key is only present when the service had a resolved VRAM plan, and a
    pane that turned "unknown" into "won't fit" would refuse a card nobody
    measured. Drawn for present rows too -- a downloaded model that cannot run
    here is exactly the thing worth saying.
    """
    verdict = row.get("vram")
    # The measured figure beside the word. "tight fit" says a verdict and not
    # its evidence, and the number is what makes it checkable against a card
    # the user knows the size of.
    if verdict == vram.FIT_TIGHT:
        widgets.text_colored(theme.WARN, "tight fit")
    elif verdict == vram.FIT_NO:
        widgets.text_colored(theme.ERR, "won't fit")
    else:
        # The plain cost is not a refusal, so it rides the Model tooltip with
        # the other reference facts -- in a 84 px column it clipped, and a
        # figure cut off mid-word is worse than one a hover away.
        return
    if imgui.is_item_hovered():
        imgui.set_tooltip(_vram_note(row))


def _remove_control(ctx: Any, row: dict[str, Any], busy: bool) -> None:
    """The trash button under a downloaded row, and the figure beside it.

    Drawn only when ``removable`` -- a recipe whose every file is shared with
    another model that would still need it has nothing to offer, and a button
    that refused on click would be worse than no button.

    The freed figure is ``removal_plan``'s, not the download size, and that is
    the whole reason it is shown: uninstalling one of the four SDXL 1.0 recipes
    frees 0.8 GB, and a row that implied 7 would be lying about a delete. It
    is on this button's tooltip since the rows became a table, and the confirm
    dialog repeats it before anything is removed.
    """
    row_key = str(row["row_key"])
    if not row.get("removable"):
        return
    key = app_ctx.remove_key(row_key)
    if ctx.tasks.is_busy(key):
        found = ctx.progress(key)
        widgets.progress_bar(float(found.get("percent") or 0.0) if found else 0.0)
        return
    if widgets.disabled_button(f"{icons.TRASH} Remove##{row_key}", not busy):
        _confirm_removal(ctx, row)
    freed = float(row.get("freed_gib") or 0.0)
    if freed and imgui.is_item_hovered():
        # Demoted from a line to the tooltip when the rows became a table, and
        # safe *because* ``_confirm_removal`` states the same figure in the
        # dialog: the number is still in front of the user before anything is
        # deleted, which is the moment it has to be.
        imgui.set_tooltip(f"Frees about {freed:.1f} GB.")


def _confirm_removal(ctx: Any, row: dict[str, Any]) -> None:
    """Ask first. The consequence sentence differs for a base model, because
    that is the one removal that can stop generation working entirely."""
    row_key = str(row["row_key"])
    label = str(row.get("label") or row_key)
    freed = float(row.get("freed_gib") or 0.0)
    message = f"{label} will be deleted from disk, freeing about {freed:.1f} GB."
    if row.get("kind") == "base":
        message += (
            "\n\nGeneration will refuse to run until a base model is "
            "reinstalled. You can download it again from this pane."
        )
    dialogs.ask_delete(
        ctx,
        title="Remove model",
        message=message,
        on_confirm=lambda: _start_removal(ctx, row_key),
    )


def _start_removal(ctx: Any, row_key: str) -> None:
    """Submit the uninstall, with the same progress binding ``_start`` uses.

    Extracted from the confirm's lambda so it can be called without a frame,
    and so the closure over ``key`` is written once for both mutations.
    """
    from ...service import downloads as svc_downloads

    key = app_ctx.remove_key(row_key)

    def run() -> Any:
        return svc_downloads.uninstall(
            ctx.svc,
            [row_key],
            on_progress=lambda percent, label: ctx.tasks.set_progress(key, percent, label),
        )

    if ctx.submit(key, run):
        ctx.toast("Removing...")


def _detail(row: dict[str, Any]) -> None:
    """Where the weights come from, and what an adapter answers to.

    Both are on the registry entry and neither was on screen: the list showed
    a label, a declared size and a tick, so "which pixel-art LoRA is this"
    and "what do I have to type for it to do anything" were answerable only
    from MODELS.md, which is a file beside the app rather than in it.
    """
    trigger = str(row.get("trigger") or "")
    if trigger:
        widgets.muted(f"   triggers: {trigger}")
    repos = tuple(row.get("repos") or ())
    if repos:
        widgets.muted(f"   {', '.join(repos)}")


def _row(ctx: Any, row: dict[str, Any], busy: bool) -> None:
    """One model as four cells: what it is, how big, what for, and what to do.

    What stays *visible* and what moves to a tooltip is decided by whether the
    user has to act on it. The presence glyph, the label, the size figure and
    the fit badge stay -- the badge especially, because "won't fit this GPU"
    is a refusal the user otherwise meets at generation time. The repository
    list and an adapter's trigger words go to the Model tooltip: reference
    facts, looked up once. The freed figure goes to the Remove tooltip, which
    is safe because ``_confirm_removal`` states it again in the dialog before
    anything is deleted.
    """
    row_key = str(row["row_key"])
    present = bool(row.get("present"))
    label = str(row.get("label") or row_key)
    downloadable = bool(row.get("downloadable"))

    imgui.table_next_column()
    if present:
        widgets.text_colored(theme.MUTED, f"{icons.CIRCLE_CHECK} {label}")
    elif not downloadable:
        widgets.text_colored(theme.WARN, f"{icons.CIRCLE} {label}")
    else:
        ticked = row_key in ctx.model_picks
        changed, ticked = controls.checkbox(f"##pick-{row_key}", ticked)
        if changed:
            # Only a missing row can be ticked, and the set is pruned as rows
            # arrive present, so it can never name something already on disk.
            if ticked:
                ctx.model_picks.add(row_key)
            else:
                ctx.model_picks.discard(row_key)
        imgui.same_line()
        widgets.text_colored(theme.WARN, f"{icons.CIRCLE} {label}")
    _identity_tooltip(row)

    imgui.table_next_column()
    size = float(row.get("size_gib") or 0.0)
    widgets.muted(f"{size:.1f} GB" if size else "--")
    _fit_badge(row)

    imgui.table_next_column()
    _description(row)

    imgui.table_next_column()
    _actions(ctx, row, busy)


def _vram_note(row: dict[str, Any]) -> str:
    """What this model costs on the card, as a number. Empty when unmeasured.

    The figure beside the word: "tight fit" states a verdict and not its
    evidence, and the number is what makes it checkable against a card whose
    size the user knows.
    """
    need = row.get("vram_gib")
    if not isinstance(need, int | float):
        return ""
    return f"About {float(need):.0f} GB of VRAM while it is loaded."


def _identity_tooltip(row: dict[str, Any]) -> None:
    """Which weights this actually is, and what an adapter answers to.

    Both are on the registry entry and neither was ever on screen before the
    rows grew a detail line: the list showed a label, a size and a tick, so
    "which pixel-art LoRA is this" was answerable only from MODELS.md, which
    is a file beside the app rather than in it. They are a tooltip now rather
    than two more lines because they are looked up once and then known.
    """
    if not imgui.is_item_hovered():
        return
    parts = [str(row.get("label") or row.get("row_key"))]
    long = str(row.get("description") or "").split(_PARA)
    if len(long) > 1:
        parts.append(long[1])
    trigger = str(row.get("trigger") or "")
    if trigger:
        parts.append(f"Triggers: {trigger}")
    note = _vram_note(row)
    if note:
        parts.append(note)
    trigger_repos = tuple(row.get("repos") or ())
    if trigger_repos:
        parts.append(", ".join(trigger_repos))
    licence = licence_note(row)
    if licence:
        parts.append(licence)
    imgui.set_tooltip(_PARA.join(parts))


def licence_note(row: dict[str, Any]) -> str:
    """What this row's weights permit. Empty when the row carries no licence.

    A plain sentence rather than a badge, because the tooltip is where the
    other reference facts already live -- but the *restricted* case also gets a
    visible marker in the cell (see ``_description``), since a licence that
    forbids selling the output is not a reference fact, it is a decision the
    user has to make before spending 7 GB.
    """
    licence = str(row.get("license") or "")
    if not licence:
        return ""
    if not row.get("commercial", True):
        return (
            f"Licence: {licence}. Images from this model may NOT be used "
            f"commercially. {row.get('license_note') or ''}"
        ).strip()
    note = str(row.get("license_note") or "")
    return f"Licence: {licence}. {note}".strip() if note else f"Licence: {licence}."


def _description(row: dict[str, Any]) -> None:
    """The short form in the cell; the long form is in the Model tooltip.

    A row that has no prose yet says nothing rather than repeating its label:
    ``description`` is defaulted on the spec so a new entry is legal before it
    is written, and an empty cell is what says "not written yet".
    """
    text = str(row.get("description") or "").split(_PARA)[0]
    if text:
        widgets.muted(text)
    # In the cell and not only in the tooltip. A tooltip is something you find
    # after wondering; whether you are allowed to sell what this makes is not a
    # question a user of an *asset generator* knows to wonder about, and the
    # download button is right there in the same row.
    if row.get("license") and not row.get("commercial", True):
        widgets.text_colored(theme.WARN, "non-commercial")
        if imgui.is_item_hovered():
            imgui.set_tooltip(licence_note(row))


def downloadable_note(row: dict[str, Any]) -> str:
    """Why a row cannot be fetched, if it cannot. Empty when it can."""
    if row.get("present") or row.get("downloadable"):
        return ""
    return (
        "These weights have no download recipe in this build -- install them "
        "by hand; the startup diagnostics print the exact command."
    )


def _actions(ctx: Any, row: dict[str, Any], busy: bool) -> None:
    """Install, Remove, a progress bar, or an inert word saying why not.

    A row that cannot be downloaded stays *visible* and inert rather than
    disappearing: a vanished row is worse than one that says it is not
    available, because there is nothing left to ask about.
    """
    row_key = str(row["row_key"])
    if bool(row.get("present")):
        _remove_control(ctx, row, busy)
        return
    if not row.get("downloadable"):
        widgets.muted("Unavailable")
        if imgui.is_item_hovered():
            imgui.set_tooltip(downloadable_note(row))
        return

    key = app_ctx.download_key(row_key)
    if ctx.tasks.is_busy(key):
        found = ctx.progress(key)
        widgets.progress_bar(float(found.get("percent") or 0.0) if found else 0.0)
        if found and found.get("label"):
            widgets.muted(str(found["label"]))
        # The bulk bar has had this since MDL-14 and a single row had not,
        # which made "cancel" depend on which of two identical buttons started
        # the fetch. Same mechanism, same reasoning -- see ``_cancel``.
        _cancel(ctx, key)
        return
    if widgets.disabled_button(f"{icons.DOWNLOAD} Install##{row_key}", not busy):
        _start(ctx, [row_key], key=key)


def _selection_progress(ctx: Any) -> None:
    key = "download:selection"
    if not ctx.tasks.is_busy(key):
        return
    found = ctx.progress(key)
    widgets.progress_bar(float(found.get("percent") or 0.0) if found else 0.0)
    if found and found.get("label"):
        widgets.muted(str(found["label"]))
    _cancel(ctx, key)


def _cancel(ctx: Any, key: str) -> None:
    """Cancel, beside the bar. Every mechanism this needs already existed and
    only the button was missing: the fetch child is tracked (``winjob``, under
    a reason starting with "fetch") so it can be terminated *by that prefix* --
    killing the whole registry here would take a live Blender bake or the
    persistent matting worker with it. The kill-on-close job reaps it, and
    publication is staged -- so a cancelled download leaves no half-installed
    model, just the staging tree the sweep clears. Without it a mistaken 16 GB
    fetch on a slow line could be stopped only by quitting the app, because
    the timeout is four hours (MDL-14).

    One fetch runs at a time across the pane, so the prefix kill is
    unambiguous however this is reached -- which is why a row and the bulk bar
    can share it.
    """
    imgui.same_line()
    from ... import winjob

    if controls.small_button(f"Cancel##cancel-{key}"):
        stopped = winjob.terminate_tracked("fetch")
        ctx.toast("Stopping the download..." if stopped else "Nothing left to stop.")


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
