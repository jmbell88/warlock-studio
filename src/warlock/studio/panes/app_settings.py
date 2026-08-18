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

from ... import fetch, vram
from .. import app_ctx, controls, dialogs, forms, icons, theme, tokens, widgets
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

#: How wide a labelled control in this pane gets. The column is 640 and the
#: values in it are a scale, a palette name and a width -- all short. A combo
#: stretched to the column is a dropdown four times wider than the longest thing
#: it can contain, and it pushes its own ``help_marker`` onto the next line.
FIELD_W = 260

#: The four things this pane is, and the order they are offered in.
#:
#: Appearance first because it is the one most people came for; Advanced last
#: because it is read-only diagnostics. The keys are stable ids -- they are in
#: the segmented control's imgui ids and in ``state.preview`` -- and the labels
#: are what changes when the wording does.
CATEGORIES = [
    ("appearance", f"{icons.PALETTE} Appearance"),
    ("models", f"{icons.BOX} Models"),
    ("storage", f"{icons.FOLDER_OPEN} Storage"),
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
    if imgui.begin_child(
        "app-settings", (0, 0), imgui.ChildFlags_.always_use_window_padding.value
    ):
        width = _centre(sp(CONTENT_W))
        # An inner child rather than an indent, so every row inside it measures
        # its own ``content_region_avail`` against the column and not against
        # the window -- which is what makes the model rows, the config table and
        # the sliders bound themselves without being told the number.
        if imgui.begin_child("app-settings-body", (width, 0)):
            # Settings draws into ``##content`` rather than through
            # ``layout.pane``, so it asks for its own section blocks. Named here
            # rather than made automatic: a scope splits *this child's* draw
            # list, and the right bracket is the child, which only the code that
            # opened it knows.
            with widgets.section_blocks():
                # This mode is one pane filling the window, so it is the one
                # surface in the app that has to say what it is out loud (UX.md
                # Phase 2) -- the three-column modes are named by the rail item
                # that is lit.
                widgets.pane_header(
                    "Settings",
                    help_text="Application appearance, models, storage, and layout.",
                )
                with forms.Form("application-settings") as form_ui:
                    _category_body(ctx, _categories(ctx, width), form_ui)
        imgui.end_child()
    imgui.end_child()


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


def _category_body(ctx: Any, category: str, form_ui: forms.Form) -> None:
    if category == "models":
        _models(ctx)
    elif category == "storage":
        _storage(ctx)
    elif category == "advanced":
        _layout(ctx)
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
    # cannot do for it.
    _changed, chosen = form_ui.combo(
        "theme",
        "Theme",
        tokens.THEME,
        [(name, name) for name in tokens.PALETTES],
        help_text="The whole palette, including the viewport background.",
        helper="It takes effect at once and is remembered.",
    )
    if chosen != tokens.THEME:
        _apply_theme(ctx, chosen)

    show_fps = bool(ctx.state.show_fps)
    changed, show_fps = form_ui.switch(
        "show_fps", "Show frame rate (F10)", show_fps
    )
    if changed:
        ctx.state.show_fps = show_fps
        ctx.settings.set("show_fps", show_fps)

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
    if controls.button("Reset pane sizes"):
        lay.settings_share = 0.55
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
    # An empty page rather than the loaded one: this pane has no list of
    # trashed jobs to hand over, and the argument is only the stamp that
    # decides *when* to walk the disk again -- so Settings asks under its own
    # stamp and the library goes on asking under the page it is showing.
    summary = library.trash_summary(library.measure_trash(ctx, []))
    widgets.muted(f"In the trash: {summary}" if summary else "Measuring the trash...")

    widgets.section("Maintenance")
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


def _models(ctx: Any) -> None:
    # No heading: the lit segment says "Models". See ``_interface``.
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
    heading = ""
    for kind, label in ordered:
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


def _fit_badge(row: dict[str, Any]) -> None:
    """The one-line "and can this card hold it" note, under a base row.

    Read with ``.get`` and skipped when absent, never defaulted to a verdict:
    the key is only present when the service had a resolved VRAM plan, and a
    pane that turned "unknown" into "won't fit" would refuse a card nobody
    measured. Drawn for present rows too -- a downloaded model that cannot run
    here is exactly the thing worth saying.
    """
    verdict = row.get("vram")
    if verdict == vram.FIT_TIGHT:
        widgets.text_colored(theme.WARN, "    tight fit")
    elif verdict == vram.FIT_NO:
        widgets.text_colored(theme.ERR, "    won't fit this GPU")


def _remove_control(ctx: Any, row: dict[str, Any], busy: bool) -> None:
    """The trash button under a downloaded row, and the figure beside it.

    Drawn only when ``removable`` -- a recipe whose every file is shared with
    another model that would still need it has nothing to offer, and a button
    that refused on click would be worse than no button.

    The freed figure is ``removal_plan``'s, not the download size, and that is
    the whole reason it is shown: uninstalling one of the four SDXL 1.0 recipes
    frees 0.8 GB, and a row that implied 7 would be lying about a delete.
    """
    row_key = str(row["row_key"])
    if not row.get("removable"):
        return
    key = app_ctx.remove_key(row_key)
    if ctx.tasks.is_busy(key):
        found = ctx.progress(key)
        widgets.progress_bar(float(found.get("percent") or 0.0) if found else 0.0)
        return
    freed = float(row.get("freed_gib") or 0.0)
    if freed:
        widgets.muted(f"    frees {freed:.1f} GB")
    if widgets.disabled_button(f"{icons.TRASH} Remove##{row_key}", not busy):
        _confirm_removal(ctx, row)


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


def _row(ctx: Any, row: dict[str, Any], busy: bool) -> None:
    row_key = str(row["row_key"])
    present = bool(row.get("present"))
    label = str(row.get("label") or row_key)
    if present:
        widgets.text_colored(theme.MUTED, f"{icons.CIRCLE_CHECK} {label}")
        _fit_badge(row)
        _remove_control(ctx, row, busy)
        return
    if not row.get("downloadable"):
        widgets.text_colored(theme.WARN, f"{icons.CIRCLE} {label} - weights missing")
        return

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
    size = float(row.get("size_gib") or 0.0)
    suffix = f" ({size:.1f} GB)" if size else ""
    widgets.text_colored(theme.WARN, f"{icons.CIRCLE} {label}{suffix}")
    _fit_badge(row)

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
    # Cancel, beside the bar. Every mechanism this needs already existed and
    # only the button was missing: the fetch child is tracked (``winjob``,
    # under a reason starting with "fetch") so it can be terminated *by that
    # prefix* -- killing the whole registry here would take a live Blender bake
    # or the persistent matting worker with it. The kill-on-close job reaps
    # it, and publication is staged -- so a cancelled download leaves no
    # half-installed model, just the staging tree that the next download
    # sweeps. Without it a mistaken 16 GB fetch on a slow line could be
    # stopped only by quitting the app, because the timeout is four hours
    # (MDL-14).
    imgui.same_line()
    from ... import winjob

    if controls.small_button(f"Cancel##cancel-{key}"):
        stopped = winjob.terminate_tracked("fetch")
        ctx.toast(
            "Stopping the download..." if stopped else "Nothing left to stop."
        )


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
