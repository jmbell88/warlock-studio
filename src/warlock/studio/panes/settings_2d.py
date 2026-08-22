"""The 2D pane: everything that composes the SDXL prompt, and Generate.

This pane owns the prompt and every field that reaches the text encoder; the
3D pane owns nothing that does. Since the 2026-08-17 taxonomy retirement the
form is flat -- no folds, no guidance groups -- and every section draws as a
full-width tinted block, matching Plotter's tools pane: the block scope is
opened *inside* the ``2d-form`` child so the fills land on the child's own
draw list rather than under its opaque background.

The composed-prompt preview is debounced and computed on a task thread: it
loads CLIP's tokenizers to count tokens, which is far too slow for a keystroke.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from imgui_bundle import imgui

from ... import guidance as guidancelib
from ... import models as modelslib
from ... import vectors
from ...bench import findings as findings_lib
from ...pipelines import tilesheet as tilesheetlib
from ...service import jobs as svc_jobs
from ...service import sprites as svc_sprites
from ...service import system as svc_system
from ...service import tilesheets as svc_tilesheets
from ...service.errors import Invalid
from ...service.validation import (
    MAX_PROMPT,
    MAX_REFERENCE_COUNT,
    MAX_UPLOAD_BYTES,
    random_seed,
)
from .. import (
    anchors,
    controls,
    create_assets,
    dialogs,
    focus,
    forms,
    profiles,
    theme,
    tokens,
    widgets,
)
from ..manual import render as manual_render
from ..tokens import sp
from ..widgets import field_options as _options
from . import model_gate

PREVIEW_DEBOUNCE = 0.3

# This pane's key in the focus ring (UX.md Phase 3). The controls on the common
# path take a place in it: the ring exists so a first job can be composed and
# submitted without the mouse.
FOCUS_PANE = "2d"


# What the submit block took last frame, in design pixels (K92). The same
# measure-last-frame idiom the library's footer uses, and for the same reason:
# the block's height is a function of the theme, the UI scale and how many
# problems it is currently reporting, so no constant is right for all of them.
# Seeded at roughly one button plus its cost note, so the first frame reserves
# something sane rather than nothing.
_submit_px = [96.0]
_LOAD_FINDINGS = object()


def draw(ctx: Any) -> None:
    state = ctx.state
    form = state.form_2d
    findings_doc = findings_lib.load(Path(ctx.svc.config.bench_dir) / "findings.json")
    # Compatibility for live callers from the pre-registry UI that still set
    # ``output`` directly. Settings loaded from disk have already migrated and
    # been synchronised, so only an in-memory non-reference override reaches
    # this bridge.
    if "asset_type" not in form:
        form["asset_type"] = create_assets.legacy_asset_type(form)
        create_assets.sync_legacy_fields(form)
    # Form.errors now places the rings and copy beneath the owning controls;
    # these are the routes it replaces and keeps wired by the same field keys:
    # field_error(ctx.state, "prompt")
    # field_error(ctx.state, "base_model")
    # field_error(ctx.state, "style_lora")
    # field_error(ctx.state, "count")
    with forms.Form("create-2d", errors=ctx.state.field_errors) as form_ui:
        # The form scrolls; Generate does not (K92): the one control every
        # visit ends with must never sit at the bottom of a scrolled column.
        focus.pump(state, FOCUS_PANE)
        focus.begin(state, FOCUS_PANE)
        if imgui.begin_child("2d-form", (0, -sp(_submit_px[0]))):
            # The block scope opens *inside* the child: section() fills go to
            # the current window's draw list, and a scope opened outside would
            # paint onto the parent pane's list, where this child's opaque
            # PANEL background covers all but the 8dp left overhang. The
            # ``with`` closes before end_child -- an unbalanced splitter
            # corrupts the next frame (widgets.py, _BlockScope).
            with widgets.section_blocks():
                widgets.section("Asset type")
                _asset_type(ctx, form)
                widgets.section("Prompt")
                manual_render.help_button(ctx, "settings-2d")
                _prompt(ctx, form, form_ui)
                _history(ctx, form)
                widgets.section("Image model")
                if create_assets.selected(form).intent == "tileset":
                    _locked_sheet_recipe(ctx, "Tile-set recipe", part="model")
                else:
                    _model(ctx, form, findings_doc)
                widgets.section("Style LoRA")
                if create_assets.selected(form).intent == "tileset":
                    _locked_sheet_recipe(ctx, "Locked for coherent pixel tiles", part="lora")
                else:
                    _lora(ctx, form, show_strength=False, findings_doc=findings_doc)
                if create_assets.selected(form).intent == "sprite":
                    _locked_sheet_recipe(ctx, "Final sheet recipe", sprite=True)

                # Disclosure state belongs to this running workspace, not the
                # recipe, so it is kept only in AppState and never settings.
                imgui.set_next_item_open(
                    bool(getattr(state, "create_advanced", False)),
                    imgui.Cond_.always.value,
                )
                opened = controls.collapsing_header("Advanced##create")
                state.create_advanced = bool(opened)
                if opened:
                    widgets.section("References & conditioning")
                    _references(ctx, form)
                    widgets.section("Seed & count")
                    _run_controls(ctx, form, form_ui)
                    if form.get("output") == "sheet":
                        widgets.section("Dimensions")
                        manual_render.help_button(ctx, "settings-sheet")
                        # The asset type fixes sheet kind/view/layout; this
                        # section exposes only the resulting pixel dimensions.
                        if form.get("sheet_type") == "sprite":
                            _sprite_size(ctx, form, form_ui)
                        else:
                            _tile_size(ctx, form, form_ui)
                    widgets.section("Negative prompt")
                    _negative(ctx, form)
                    widgets.section("Profiles")
                    _profiles(ctx, form)
                    widgets.section("Prompt enrichment")
                    _expand(ctx, form)
                    _preview(ctx)
                    if form.get("style_lora") and create_assets.selected(form).intent != "tileset":
                        widgets.section("Style strength")
                        _lora_strength(ctx, form, findings_doc)
        imgui.end_child()
        top = imgui.get_cursor_pos_y()
        _submit(ctx, form)
        height = imgui.get_cursor_pos_y() - top
        if height > 0:
            _submit_px[0] = height / max(tokens.SCALE, 0.01)


# What a field is *called* on screen, where that is not its key with the
# underscores taken out. Empty since the taxonomy retirement (it carried
# ``art_style``); the helper stays because the sweep-axis forms and Review's
# base-capture labels still route every field name through it.
FIELD_LABELS: dict[str, str] = {}


def field_label(field: str) -> str:
    return FIELD_LABELS.get(field, field.replace("_", " "))


def _asset_type(ctx: Any, form: dict[str, Any]) -> None:
    """The one top-level choice; legacy service switches follow it."""
    before = create_assets.selected(form).key
    picked = widgets.combo(
        "##asset_type", before, list(create_assets.ASSET_TYPE_OPTIONS)
    )
    form["asset_type"] = picked if picked in create_assets.ASSET_TYPES else before
    spec = create_assets.sync_legacy_fields(form)
    if spec.key != before:
        ctx.state.preview_dirty_at = time.monotonic()
        ctx.state.clear_field_error("asset_type")


def _locked_sheet_recipe(
    ctx: Any, note: str, *, part: str = "both", sprite: bool = False
) -> None:
    """Say what the pinned sheet stage really loads; never draw fake pickers."""
    base_key = (
        svc_sprites.SPRITE_BASE_MODEL if sprite else svc_tilesheets.TILE_SHEET_BASE_MODEL
    )
    lora_key = modelslib.PIXEL_SHEET_LORA
    if part in ("model", "both"):
        if part == "both":
            widgets.field_label("Image model")
        imgui.text_wrapped(modelslib.BASE_MODELS[base_key].label)
    if part in ("lora", "both"):
        if part == "both":
            widgets.field_label("Style LoRA")
        imgui.text_wrapped(modelslib.STYLE_LORAS[lora_key].label)
    widgets.muted_wrapped(note)


def _tile_size(ctx: Any, form: dict[str, Any], form_ui: forms.Form) -> None:
    """Only the editable dimension of a tileset asset type."""
    options = _tile_options()
    changed, picked = form_ui.segmented_choice(
        "tile_size", "Tile size", str(form.get("tile_size", "32")),
        tuple((str(size), str(size)) for size in options["tile_sizes"]),
        help_text="How many pixels across one tile is.", compact=True,
    )
    if changed:
        form["tile_size"] = picked
        ctx.state.clear_field_error("tile_size")


def _sprite_size(ctx: Any, form: dict[str, Any], form_ui: forms.Form) -> None:
    """Only the editable dimension of a sprite asset type."""
    options = _sprite_options()
    changed, picked = form_ui.segmented_choice(
        "cell_size", "Cell size", str(form.get("cell_size", "64")),
        tuple((str(size), str(size)) for size in options["logical_sizes"]),
        help_text="How many pixels across one frame is.", compact=True,
    )
    if changed:
        form["cell_size"] = picked


def _findings_hint(
    ctx: Any,
    param: str,
    value: Any,
    doc: Any = _LOAD_FINDINGS,
) -> str | None:
    """The sweep's own verdict on this field's current value, or None.

    Read fresh every frame -- ``findings.load`` is mtime-cached, so the common
    case (no bench dir, or an unchanged file) costs one ``stat()`` and never
    blocks the frame loop.

    Scoped to what the user is currently asking for. This pane owns the prompt,
    so it always knows its subject: the hash of the prompt in the form is what
    ``vectors.prompt_hash`` recorded on every verdict and observation, so
    ``hint`` can prefer the evidence about *this* subject and say when it fell
    back to the pooled corpus. Hashing a short string once per control per
    frame is a sha1 over a few dozen bytes, which is nothing beside the
    ``stat()`` above it.
    """
    if doc is _LOAD_FINDINGS:
        doc = findings_lib.load(Path(ctx.svc.config.bench_dir) / "findings.json")
    return findings_lib.hint(
        doc,
        param,
        value,
        prompt_hash=vectors.prompt_hash(ctx.state.form_2d.get("prompt")),
    )


# --- pieces -----------------------------------------------------------------


# The output kind is **not** a control on this pane. ``_asset_type`` is what
# the user picks, and ``create_assets.sync_legacy_fields`` sets ``form["output"]``
# from the chosen spec -- so the pre-registry segmented control that used to be
# here, its ``OUTPUTS`` table, its left/right keyboard arms and its
# ``OUTPUT_NOTES`` prose were all superseded rather than lost. They were deleted
# on 2026-08-22 with zero callers between them; ``git show`` has them if the
# asset registry is ever unwound.


# The two things a Sheet can be, stepped through by index so a third added to
# the control and missed by the arrows cannot become a segment the keyboard
# cannot reach.
SHEET_TYPES: tuple[tuple[str, str], ...] = (
    ("tile", "Tile grid"),
    ("sprite", "Sprite sheet"),
)

# ``tile_sheet_options()`` and ``sprite_options()`` are pure functions of module
# constants, and this pane calls them from the frame loop. The tile-sheet one
# builds a geometry per size per view -- sixty-four Cell objects each --
# which is nothing once and a thousand short-lived dataclasses a frame at 60fps.
# Cached in a one-slot list, the ``_submit_px`` idiom, because there is nothing
# for them to go stale against: neither reads config, disk or state.
_sheet_options: list[Any] = [None, None]


def _tile_options() -> dict[str, Any]:
    if _sheet_options[0] is None:
        _sheet_options[0] = svc_tilesheets.tile_sheet_options()
    return _sheet_options[0]


def _sprite_options() -> dict[str, Any]:
    if _sheet_options[1] is None:
        _sheet_options[1] = svc_sprites.sprite_options()
    return _sheet_options[1]


def sheet_rows(form: dict[str, Any]) -> tuple[str, ...]:
    """Which registry rows the Sheet output currently needs.

    A function of the form rather than a constant, because the two arms load
    different things and the tile arm's own list depends on whether a reference
    is attached: its IP-Adapter is optional, and a gate that demanded one would
    tell a user with everything the common request uses that they are missing a
    download. Shared with :func:`weights_problem` so the note above the button
    and the gate inside the section cannot disagree.
    """
    if form.get("sheet_type") == "sprite":
        return svc_sprites.SPRITE_ROWS
    if form.get("ref_path"):
        return svc_tilesheets.TILE_SHEET_REFERENCE_ROWS
    return svc_tilesheets.TILE_SHEET_ROWS


def _sheet(ctx: Any, form: dict[str, Any], form_ui: forms.Form) -> None:
    """The Sheet output's own fields, drawn only while it is selected.

    Three controls at most, which is the feature: a sheet is composed from the
    prompt every other output uses, plus the two facts the prompt cannot carry
    -- how big a tile is and which way the grid runs. Everything else (the
    palette, the grid, the negative prompt) is a measured default rather than a
    control, because a form with nine fields is one nobody reaches the end of.
    """
    before = form.get("sheet_type", "tile")
    keys = [key for key, _label in SHEET_TYPES]
    with focus.item(ctx.state, FOCUS_PANE, "sheet_type") as focused:
        form["sheet_type"] = widgets.segmented_control(
            "sheet_type", list(SHEET_TYPES), before
        )
        if focused:
            here = keys.index(form["sheet_type"]) if form["sheet_type"] in keys else 0
            if imgui.is_key_pressed(imgui.Key.left_arrow):
                form["sheet_type"] = keys[(here - 1) % len(keys)]
            if imgui.is_key_pressed(imgui.Key.right_arrow):
                form["sheet_type"] = keys[(here + 1) % len(keys)]
    if form["sheet_type"] != before:
        ctx.state.preview_dirty_at = time.monotonic()

    # Drawn before the fields rather than after, so a user on a host missing the
    # ControlNet reads why nothing will happen before they fill anything in.
    model_gate.draw(ctx, sheet_rows(form), what="A sheet")

    if form["sheet_type"] == "sprite":
        _sprite_fields(ctx, form, form_ui)
    else:
        _tile_fields(ctx, form, form_ui)


def _view_of(form: dict[str, Any]) -> str:
    """The form's view, in today's spelling.

    The form field is still ``projection`` -- it is a persisted key and a
    control the user has a name for -- and a profile saved before the
    vocabulary widened carries ``"orthogonal"``. Read through the service's
    alias table rather than by comparing strings here, so the pane never holds
    a second opinion about what an old value means.
    """
    stored = str(form.get("projection") or svc_tilesheets.DEFAULT_VIEW)
    return svc_tilesheets.LEGACY_VIEWS.get(stored, stored)


def _tile_fields(ctx: Any, form: dict[str, Any], form_ui: forms.Form) -> None:
    options = _tile_options()
    with focus.item(ctx.state, FOCUS_PANE, "tile_size"):
        changed, picked = form_ui.segmented_choice(
            "tile_size",
            "Tile size",
            str(form.get("tile_size", "32")),
            tuple((str(size), str(size)) for size in options["tile_sizes"]),
            help_text="How many pixels across one tile is.",
            compact=True,
        )
    if changed:
        form["tile_size"] = picked
        ctx.state.clear_field_error("tile_size")
    with focus.item(ctx.state, FOCUS_PANE, "projection"):
        # Both the keys and the words come off the service reply. They were
        # written out here as a literal pair, which is why a third view could
        # not be added without editing a pane that knows nothing about what a
        # view is -- and why the refusal below used to spell its own list too.
        labels = options["view_labels"]
        changed, picked = form_ui.segmented_choice(
            "projection",
            "View",
            _view_of(form),
            tuple((key, labels.get(key, key)) for key in options["views"]),
            help_text=(
                "Where the camera is. Top-down is flat; 3/4 tilts it so tiles "
                "show a front face; isometric is the 2:1 diamond lattice."
            ),
            compact=True,
        )
    if changed:
        form["projection"] = picked
        ctx.state.clear_field_error("projection")
    # The finished size, said rather than left to be worked out. The arithmetic
    # is the service's (``tile_sheet_options`` returns it per size per view)
    # precisely so this line cannot drift from what lands on disk.
    entry = next(
        (row for row in options["sizes"] if str(row["key"]) == str(form.get("tile_size"))),
        None,
    )
    if entry is not None:
        shape = entry["views"].get(_view_of(form)) or {}
        if shape:
            widgets.muted(
                f"{options['tiles']} tiles of {shape['tile_w']}x{shape['tile_h']} "
                f"- a {shape['sheet_w']}x{shape['sheet_h']} sheet"
            )


def _sprite_fields(ctx: Any, form: dict[str, Any], form_ui: forms.Form) -> None:
    """The sprite arm, which is two steps and says so.

    The character is drawn first and kept as a row of its own; the sheet is a
    follow-up job against it. Stated in the pane because the library will show
    two rows for one button press, and a user who was not told that has watched
    the app do something it did not offer to do.
    """
    options = _sprite_options()
    types = tuple(
        (row["key"], f"{row['key'].title()} ({row['columns']}x{row['rows']})")
        for row in options["sheet_types"]
    )
    with focus.item(ctx.state, FOCUS_PANE, "sheet_layout"):
        changed, picked = form_ui.segmented_choice(
            "sheet_layout",
            "Layout",
            str(form.get("sheet_layout") or options["defaults"]["sheet_type"]),
            types,
            help_text="Four facings, or a four-frame walk cycle in each.",
            compact=True,
        )
    if changed:
        form["sheet_layout"] = picked
    with focus.item(ctx.state, FOCUS_PANE, "cell_size"):
        changed, picked = form_ui.segmented_choice(
            "cell_size",
            "Cell size",
            str(form.get("cell_size") or options["defaults"]["logical_size"]),
            tuple((str(size), str(size)) for size in options["logical_sizes"]),
            help_text="How many pixels across one frame is.",
            compact=True,
        )
    if changed:
        form["cell_size"] = picked
    widgets.muted_wrapped(
        "Two steps: the character is drawn from the prompt and kept as its own "
        "asset, then two candidate sheets are imagined from it. A reference "
        "image, if you attach one, shapes the character."
    )


def _profiles(ctx: Any, form: dict[str, Any]) -> None:
    """The saved-style picker, next to the shipped presets.

    Same "fills the fields, stays editable" contract a preset has -- the
    difference is only who wrote it. Saving goes through the ordinary Prompt
    modal rather than an inline text field so the name is asked for once,
    rather than every frame the section is open.
    """
    saved = profiles.list_profiles(ctx.settings)
    if saved:
        # Shows the active profile while the form still matches it, and
        # "Custom" once it has been edited past it.
        active = profiles.get_active(ctx.settings)
        current = (
            active
            if active in saved and all(form.get(k) == v for k, v in saved[active].items())
            else ""
        )
        options = [("", "Custom")] + [(name, name) for name in sorted(saved)]
        chosen = widgets.combo("##profile", current, options, width=-84)
        if chosen and chosen != current and chosen in saved:
            profiles.apply(form, saved[chosen])
            profiles.set_active(ctx.settings, chosen)
            ctx.state.preview_dirty_at = time.monotonic()
        # Wrap-aware, all three: with the picker taking the row's width, the
        # buttons continued past the pane edge at 1.5 scale and were clipped.
        widgets.same_line_or_wrap(widgets.button_width("Save as..."))
    if controls.button("Save as..."):
        ctx.prompts.ask(
            dialogs.Prompt(
                title="Save profile",
                label="Name",
                value=profiles.get_active(ctx.settings) or "",
                on_accept=lambda name: _save_profile(ctx, form, name),
            )
        )
    widgets.same_line_or_wrap(widgets.button_width("Manage..."))
    # The manager, from the picker it is about (the UI redesign, wave 3). It was a
    # top-level mode, which put a shelf of saved settings in the navigation
    # beside the six creative workspaces and made "manage my styles" somewhere
    # you travel to rather than something you do to the form in front of you.
    if widgets.ghost_button("Manage..."):
        from . import profiles_panel

        profiles_panel.open_sheet(ctx)
    widgets.same_line_or_wrap(widgets.button_width("Reset..."))
    if controls.button("Reset..."):
        ctx.confirms.ask(
            dialogs.Confirm(
                title="Reset the image settings?",
                message=(
                    "The prompt, the negative prompt, the model, the LoRA, "
                    "the reference and the run controls go back to their "
                    "defaults. Saved profiles are kept, and the 3D form is "
                    "untouched."
                ),
                confirm_label="Reset",
                cancel_label="Cancel",
                on_confirm=lambda: _reset(ctx),
            )
        )


def _reset(ctx: Any) -> None:
    """The 2D form back to first-launch defaults.

    A fresh ``default_form_2d`` rather than a field-by-field clear, so a field
    added later is reset by having been added rather than by somebody
    remembering this function -- and so the seed is *rerolled* rather than
    zeroed, which is what that default does and why it is a function.
    """
    from ..state import default_form_2d

    ctx.state.form_2d = default_form_2d()
    ctx.state.preview = {}
    ctx.state.preview_dirty_at = time.monotonic()
    ctx.toast("The image settings are back to their defaults.")


def _save_profile(ctx: Any, form: dict[str, Any], name: str) -> None:
    profiles.save_profile(ctx.settings, name, profiles.capture(form))
    profiles.set_active(ctx.settings, name)
    ctx.toast(f"Saved the profile {name}.")


def _prompt(ctx: Any, form: dict[str, Any], form_ui: forms.Form) -> None:
    before = form["prompt"]
    with focus.item(ctx.state, FOCUS_PANE, "prompt"):
        _changed, form["prompt"] = form_ui.multiline_text(
            "prompt",
            "Description",
            before,
            height=90,
            max_length=MAX_PROMPT,
            help_text="Describe the subject and the result you want to generate.",
            helper=f"{len(before)}/{MAX_PROMPT} characters",
        )
        anchors.mark("create/prompt")
    if form["prompt"] != before:
        ctx.state.preview_dirty_at = time.monotonic()
        ctx.state.clear_field_error("prompt")


def _history(ctx: Any, form: dict[str, Any]) -> None:
    if not ctx.state.history:
        return
    imgui.same_line()
    if controls.button("Recent"):
        imgui.open_popup("prompt-history")
    if imgui.begin_popup("prompt-history"):
        widgets.popup_chrome(_imgui=imgui)
        for entry in ctx.state.history:
            label = entry if len(entry) <= 60 else entry[:57] + "..."
            if controls.menu_item(f"{label}##{hash(entry)}", "", False)[0]:
                form["prompt"] = entry
                ctx.state.preview_dirty_at = time.monotonic()
        imgui.end_popup()


EXPAND_NOTES = {
    "asset": (
        "A local model enriches short prompts with detail that suits a 3D "
        "reference: the single-subject framing stays."
    ),
    "scene": (
        "A local model enriches short prompts and the single-subject framing "
        "is dropped -- for 2D pictures, not for meshes."
    ),
}


def _expand(ctx: Any, form: dict[str, Any]) -> None:
    """The prompt-expansion mode, under the prompt it acts on.

    Options come from the catalog like every other select; unlike them the
    combo has no blank entry, because "off" is a real mode rather than "say
    nothing about this". Detailed prompts skip expansion on their own (the
    worker's gate), which the note under the live modes says.
    """
    widgets.field_label("enrich")
    entries = (ctx.guidance.get("fields") or {}).get("expand") or []
    options = [(e["key"], e["label"]) for e in entries] or [("off", "Off")]
    before = form.get("expand") or "off"
    form["expand"] = widgets.combo("##expand", before, options)
    widgets.field_error(ctx.state, "expand")
    if form["expand"] != before:
        ctx.state.preview_dirty_at = time.monotonic()
        ctx.state.clear_field_error("expand")
    note = EXPAND_NOTES.get(form["expand"])
    if note is not None:
        widgets.muted_wrapped(
            note + " Prompts that are already detailed are left as written."
        )
        if form.get("output") == "tile":
            widgets.muted_wrapped(
                "A seamless tile is never expanded: the enrichment describes "
                "subjects, and a tile has none."
            )
        if form.get("output") == "sheet" and form.get("sheet_type") != "sprite":
            # The sprite arm is exempt: its first step is an ordinary reference
            # of one character, which is exactly what the enrichment describes.
            widgets.muted_wrapped(
                "A tile grid is never expanded: the enrichment describes one "
                "subject, and a sheet is sixty-four."
            )


def _preview(ctx: Any) -> None:
    """The composed prompt, recomputed off-thread after a typing pause."""
    state = ctx.state
    if state.preview_dirty_at and time.monotonic() - state.preview_dirty_at > PREVIEW_DEBOUNCE:
        raw = {k: v for k, v in state.form_2d.items() if v not in ("", None)}
        form = state.form_2d
        grid = form.get("output") == "sheet" and form.get("sheet_type") != "sprite"
        # The *subject*, not the raw prompt, for a grid: the projection and
        # detail clauses are the pipeline's and are appended before the
        # template, so a preview built from the bare prompt would be missing
        # the half of the composition that this output kind adds.
        subject = (
            tilesheetlib.sheet_subject(
                form["prompt"], _view_of(form)
            )
            if grid
            else form["prompt"]
        )
        # Only stop asking once the request was actually taken. ``submit``
        # refuses a key that is already in flight, and the first preview runs a
        # CLIP tokenizer -- so clearing the flag unconditionally dropped the
        # edit that arrived during that first run, and the composition on
        # screen stayed stale until the user typed again.
        if ctx.submit(
            "preview",
            svc_system.prompt_preview,
            ctx.svc,
            {**raw, "prompt": None},
            subject,
            # Threaded rather than inferred: the output kind is not a guidance
            # field, so without it a tile would be previewed through the
            # single-centred-object framing its job will never use.
            tile=form.get("output") == "tile",
            tilesheet=grid,
        ):
            state.preview_dirty_at = 0.0
    preview = state.preview
    if not preview:
        return
    if imgui.tree_node("Prompt actually sent"):
        imgui.text_wrapped(preview.get("prompt") or "")
        # ``token_count``, not ``tokens``: that name is the design-token module
        # this file imports, and shadowing it inside a draw function is a
        # NameError waiting for the next person to reach for ``tokens.SP_2``.
        token_count, chunks = preview.get("tokens"), preview.get("chunks")
        if token_count is not None:
            # Chunks, not a truncation warning: the composed prompt is split on
            # comma boundaries and each chunk encoded separately, so a long one
            # costs attention rather than being cut off.
            widgets.muted(f"{token_count} tokens - {chunks} chunk(s)")
        imgui.tree_pop()


def _references(ctx: Any, form: dict[str, Any]) -> None:
    """Conditioning: an image to steer appearance and/or structure.

    Every control below the picker is hidden until there is a reference, and
    the Structure group is hidden again unless the chosen base can run a
    ControlNet. That is this pane's existing rule -- the same one that hides
    the LoRA strength slider without a LoRA: a control with nothing to act on
    is a control that cannot do anything.
    """
    # The block is grouped so a dropped file can outline exactly what it landed
    # in (H70).
    imgui.begin_group()
    origin = imgui.get_cursor_screen_pos()
    try:
        _reference_body(ctx, form)
    finally:
        imgui.end_group()
        widgets.ring(
            origin,
            imgui.get_item_rect_max(),
            theme.ACCENT,
            widgets.drop_flash(ctx.state, "2d-ref"),
        )


def _reference_body(ctx: Any, form: dict[str, Any]) -> None:
    path = form["ref_path"]
    if not path:
        found = profiles.active_anchor(ctx.settings, ctx.svc.config)
        if found is not None:
            active = profiles.get_active(ctx.settings)
            widgets.muted_wrapped(
                f"The profile {active} has a style anchor; every generation "
                "under it is conditioned on that image. Attaching one here "
                "replaces it for this asset."
            )
    if path:
        imgui.text_wrapped(Path(path).name)
        if controls.button("Clear##ref"):
            form["ref_path"] = ""
            # The selections go with it: they cannot be submitted without an
            # image, and leaving them set would disable Generate with a
            # message about a picker the user just emptied.
            form["ip_adapter"] = ""
            form["control"] = ""
            return
        imgui.same_line()
    busy = ctx.busy("ref-upload")
    if widgets.disabled_button(
        "Choose an image..." if not path else "Replace...",
        not busy,
        reason="A file picker is already open.",
    ):
        ctx.submit(
            "ref-upload", dialogs.open_file, "Choose a reference image", dialogs.IMAGE_FILTER
        )
    if not path:
        widgets.muted("...or drop an image on the window.")
        return

    # Field labels rather than nested section() calls: inside the block scope
    # a section would open a new block, and these are two halves of the one
    # References block.
    widgets.field_label("appearance")
    form["ip_adapter"] = widgets.combo(
        "##ip_adapter", form["ip_adapter"], _options(ctx, "ip_adapter")
    )
    if form["ip_adapter"]:
        changed, value = controls.slider_float(
            "Strength##ip", float(form["ip_scale"]), *_range(ctx, "ip_scale_range", 0.0, 1.5)
        )
        if changed:
            form["ip_scale"] = value

    widgets.field_label("structure")
    note = structure_note(ctx, form)
    if note is not None:
        widgets.muted_wrapped(note)
        return
    form["control"] = widgets.combo("##control", form["control"], _options(ctx, "control"))
    if form["control"]:
        changed, value = controls.slider_float(
            "Strength##cn",
            float(form["control_scale"]),
            *_range(ctx, "control_scale_range", 0.0, 2.0),
        )
        if changed:
            form["control_scale"] = value
        changed, value = controls.slider_float(
            "Until##cn", float(form["control_end"]), *_range(ctx, "control_end_range", 0.0, 1.0)
        )
        if changed:
            form["control_end"] = value
        widgets.help_marker(
            "How far into the drawing the structure keeps acting. Ending early "
            "lets the last steps add detail the reference never had; 1.0 holds "
            "the shape to the end and tends to look traced."
        )


def _range(ctx: Any, key: str, low: float, high: float) -> tuple[float, float]:
    """The bounds the service will actually enforce, so a slider can never
    produce a value the submit rejects."""
    bounds = ctx.guidance.get(key)
    if isinstance(bounds, list) and len(bounds) == 2:
        return (float(bounds[0]), float(bounds[1]))
    return (low, high)


def _base_labels(ctx: Any, keys: list[str]) -> str:
    """The picker's own labels for a set of base-model keys.

    Labels rather than keys: "sdxl_cfg" is not what the combo shows, and a
    message naming something the user cannot find in the list is worse than no
    message at all.
    """
    labels = [label for key, label in (ctx.base_models or []) if key in keys]
    return ", ".join(labels or keys)


def negative_prompt_note(ctx: Any, form: dict[str, Any]) -> str | None:
    """Why the negative prompt is inert here, or None when it is live.

    A distilled base runs at guidance 0, and text2image encodes the negative
    branch only above 1.0 -- so on turbo the field accepted text, stored it in
    params and changed nothing about the image. That silence is the bug; this
    is the sentence that ends it.
    """
    bases = ctx.guidance.get("cfg_bases") or []
    if (form.get("base_model") or "") in bases:
        return None
    return (
        "This model runs at guidance 0, so the negative prompt has no effect. "
        f"It does on: {_base_labels(ctx, bases)}."
    )


def _lora_labels(ctx: Any, keys: list[str]) -> str:
    """The picker's own labels for a set of style-LoRA keys.

    _base_labels' argument applied to the other combo: a message naming
    "pixelklein" points at something the user cannot find in the list.
    """
    labels = [label for key, label in (ctx.style_loras or []) if key in keys]
    return ", ".join(labels or keys)


def lora_note(ctx: Any, form: dict[str, Any]) -> str | None:
    """Why the style LoRA picker is inert here, or None when it is live.

    The narrow question -- whether *any* adapter in the registry is fitted to
    this architecture -- which is the only case where the control has nothing
    at all to do. An adapter names one architecture's modules, so a mismatch is
    not a weak effect but a refusal: the service rejects the submit outright
    rather than generating without it.
    """
    bases = ctx.guidance.get("lora_bases") or []
    if (form.get("base_model") or "") in bases:
        return None
    return (
        "No style LoRA in the registry is fitted to this model's architecture. "
        f"These models can use one: {_base_labels(ctx, bases)}."
    )


def lora_options(ctx: Any, form: dict[str, Any]) -> list[tuple[str, str]]:
    """The style-LoRA combo's entries for the chosen base.

    Those fitted to it, plus whatever the form already holds, marked. Keeping a
    stale selection listed is load-bearing rather than tidy: widgets.combo
    falls back to index 0 for a value it cannot find, so dropping it would draw
    "no style LoRA" over a selection ``validate`` is refusing -- making the
    value that keeps Generate off the one control the user cannot see. That is
    exactly the dead end ``clear_unusable`` exists to prevent, arriving by
    another door. The marking mirrors what main.py puts on a base whose weights
    are missing.
    """
    fitting = (ctx.guidance.get("loras_by_base") or {}).get(form.get("base_model") or "") or []
    options: list[tuple[str, str]] = []
    for key, label in ctx.style_loras or []:
        if key in fitting:
            options.append((key, label))
        elif key and key == (form.get("style_lora") or ""):
            options.append((key, f"{label} - not fitted to this model"))
    return options


def lora_filter_note(ctx: Any, form: dict[str, Any]) -> str | None:
    """Why the picker lists fewer styles than the registry holds, or None.

    A second function rather than a branch inside ``lora_note`` for
    ``tile_bases``' reason: that one explains a control that cannot act at all,
    this one a control acting on less than the whole list. Folded together,
    one sentence comes to say both things under a disabled combo.
    """
    by_base = ctx.guidance.get("loras_by_base") or {}
    fitting = by_base.get(form.get("base_model") or "") or []
    if not fitting:
        # lora_note owns this case; saying it twice is the fold above.
        return None
    everything = [key for key, _ in (ctx.style_loras or [])]
    if len(fitting) >= len(everything):
        return None
    return (
        "A style LoRA is fitted to one architecture, so this model is offered "
        f"only: {_lora_labels(ctx, fitting)}."
    )


def structure_note(ctx: Any, form: dict[str, Any]) -> str | None:
    """Which bases could run the ControlNet this one cannot, or None."""
    bases = ctx.guidance.get("controlnet_bases") or []
    if (form.get("base_model") or "") in bases:
        return None
    return (
        "Structure control needs a full-CFG model -- pick one of "
        f"{_base_labels(ctx, bases)} under Advanced."
    )


#: Where the sentences explaining a base-model change's clears are kept between
#: frames. ``state.preview`` is this app's frame-scratch namespace and is not
#: persisted, which is right: the notice belongs to the change the user just
#: made, not to the form that outlives the session.
CLEARED_KEY = "base_model_cleared"


def clear_unusable(ctx: Any, form: dict[str, Any]) -> list[str]:
    """Drop the selections the newly chosen base cannot run.

    -> one sentence per selection cleared, for the pane to show.

    Called *only* when the base model changes, never per frame: a form restored
    with a style picked under another base must keep it until the user changes
    the base, or opening the pane silently rewrites a selection nobody touched
    and does it before the note explaining it can be read.

    Clearing rather than only disabling, because a disabled control that
    ``validate`` refuses is a dead end -- the value keeping Generate off is the
    one thing the user cannot reach, and the only recovery was to guess which
    earlier choice to undo. It applies to exactly the two gates ``validate``
    refuses: the style LoRA, whose picker goes disabled, and the structure
    control, whose whole group ``structure_note`` hides. The negative prompt is
    deliberately absent -- nothing refuses a submit over it, so its own note is
    the whole remedy, and it holds text the user typed.
    """
    cleared: list[str] = []
    base = form.get("base_model") or ""
    # The *pair*, not the base. Asking "is this base in lora_bases()" was right
    # only while one architecture had adapters and the others had none: with
    # both families covered that test is never true, and the clear would
    # silently stop happening. An unknown stored base resolves to [] and
    # therefore clears, the pane's standing rule for a settings-file value.
    fitting = (ctx.guidance.get("loras_by_base") or {}).get(base) or []
    if form.get("style_lora") and form["style_lora"] not in fitting:
        form["style_lora"] = ""
        # The weight goes back to the default with it: it scales a selection
        # that no longer exists, and a strength left at 0.2 would silently
        # apply to whatever style is picked next.
        form["lora_weight"] = modelslib.DEFAULT_LORA_WEIGHT
        cleared.append(
            "The style LoRA was cleared: it is not fitted to this model's "
            "architecture."
            if fitting
            else "The style LoRA was cleared: this model cannot use one."
        )
    if form.get("control") and base not in (ctx.guidance.get("controlnet_bases") or []):
        # Only the selection, exactly as the Clear-reference button does: the
        # strengths are hidden with it and never submitted without it.
        form["control"] = ""
        cleared.append(
            "The structure control was cleared: this model cannot run a ControlNet."
        )
    return cleared


def _model(ctx: Any, form: dict[str, Any], findings_doc: Any = _LOAD_FINDINGS) -> None:
    # The section heading is the label; a labeled_combo here would say
    # "Model" twice in two type styles.
    before = form["base_model"]
    form["base_model"] = widgets.combo("##base_model", form["base_model"], ctx.base_models)
    # The refusal this most often carries is ``check_weights``' -- a model that
    # is selected and not downloaded, with the ``hf download`` line in it -- and
    # the control it names is three collapsed sections away from the button that
    # was pressed. See ``widgets.field_error``.
    widgets.field_error(ctx.state, "base_model")
    if form["base_model"] != before:
        ctx.state.preview[CLEARED_KEY] = clear_unusable(ctx, form)
        ctx.state.clear_field_error("base_model")
    # Under the Model combo rather than beside each control it emptied: this is
    # about the change just made, and the structure control's own section is
    # behind a header the user may never open.
    for note in ctx.state.preview.get(CLEARED_KEY) or ():
        widgets.muted_wrapped(note)
    # On its own line, not same_line: the combo above takes the full width, so
    # a continuation would start on the right edge and clip entirely.
    hint = _findings_hint(ctx, "base_model", form["base_model"], findings_doc)
    if hint is not None:
        widgets.hint_text(hint)


def lora_default_weight(key: str) -> float:
    """The measured strength for one style LoRA, or the flat default.

    ``ctx.style_loras`` is pinned to 2-tuples by the smoke tests and
    ``guidance.catalog()`` does not carry the weight, so the pane reads the
    registry it already imports.
    """
    spec = modelslib.STYLE_LORAS.get(key or "")
    return spec.default_weight if spec is not None else modelslib.DEFAULT_LORA_WEIGHT


def reseed_lora_weight(form: dict[str, Any], was_lora: str) -> None:
    """Put the newly-picked adapter's tuned strength into ``form``.

    Shared by this pane and the profile editor rather than written twice.
    Each adapter carries its own measured strength -- pixel-art-klein restores
    an rslora scale of 16, so the flat ``DEFAULT_LORA_WEIGHT`` is ~14x its
    usable band and returns black frames -- and ``guidance.normalize`` only
    applies ``default_weight`` when the caller *omits* the field. Both of these
    forms always send a number, so the seed has to happen at the widget. It
    lives here because the profile editor already had this bug once by holding
    a second copy of the rule, and a third copy would find it again.
    """
    if form["style_lora"] != was_lora:
        form["lora_weight"] = lora_default_weight(form["style_lora"])


def _lora(
    ctx: Any,
    form: dict[str, Any],
    *,
    show_strength: bool = True,
    findings_doc: Any = _LOAD_FINDINGS,
) -> None:
    no_lora = lora_note(ctx, form)
    if no_lora is not None:
        # Disabled rather than hidden, this pane's stated rule: the form holds
        # a style the user picked under another base, and hiding the control
        # would make that selection vanish with no explanation of why the
        # submit is now refused.
        imgui.begin_disabled()
    was_lora = form["style_lora"]
    form["style_lora"] = widgets.combo(
        "##style_lora", form["style_lora"], lora_options(ctx, form)
    )
    widgets.field_error(ctx.state, "style_lora")
    if form["style_lora"] != was_lora:
        ctx.state.clear_field_error("style_lora")
    reseed_lora_weight(form, was_lora)
    hint = _findings_hint(ctx, "style_lora", form["style_lora"], findings_doc)
    if hint is not None:
        widgets.hint_text(hint)
    if form["style_lora"] and show_strength:
        _lora_strength(ctx, form, findings_doc)
    if no_lora is not None:
        imgui.end_disabled()
        widgets.muted_wrapped(no_lora)
    else:
        # One sentence at a time: lora_note explains a control that cannot act,
        # lora_filter_note one acting on less than the whole list, and both
        # under a disabled combo would be one control saying two things.
        narrowed = lora_filter_note(ctx, form)
        if narrowed is not None:
            widgets.muted_wrapped(narrowed)


def _lora_strength(
    ctx: Any, form: dict[str, Any], findings_doc: Any = _LOAD_FINDINGS
) -> None:
    """The advanced half of the style choice."""
    if not form.get("style_lora"):
        return
    changed, value = controls.slider_float("Strength", form["lora_weight"], 0.0, 1.5)
    if changed:
        form["lora_weight"] = value
    widgets.muted_wrapped(f"tuned default: {lora_default_weight(form['style_lora']):g}")
    hint = _findings_hint(ctx, "lora_weight", form["lora_weight"], findings_doc)
    if hint is not None:
        widgets.hint_text(hint)


def _negative(ctx: Any, form: dict[str, Any]) -> None:
    inert = negative_prompt_note(ctx, form)
    if inert is not None:
        # Disabled rather than hidden, and with the reason underneath: the
        # field holds text the user typed under another base, and hiding it
        # would make that text vanish without saying why.
        imgui.begin_disabled()
    # The section heading above is the label; imgui would draw a multiline's
    # own label to the *right* of a -1-wide field, clipped off the panel.
    before = form["negative_prompt"]
    form["negative_prompt"] = widgets.multiline("##negative", before, 54, MAX_PROMPT)
    if form["negative_prompt"] != before:
        ctx.state.preview_dirty_at = time.monotonic()
    if inert is not None:
        imgui.end_disabled()
        widgets.muted_wrapped(inert)


def _run_controls(ctx: Any, form: dict[str, Any], form_ui: forms.Form) -> None:
    """Count and seed, beside the button that uses them.

    These lived under Advanced, which meant "roll again" and "how many"
    required re-expanding a collapsed section every session while the submit
    footer talked about the count as if it were visible.
    """
    if form.get("output") == "sheet":
        # Pinned to one, and the control is not drawn at all. Both doors refuse
        # a batch and say why -- a tile sheet is one generation by construction,
        # and N characters each spawning two more sheets is 3N passes from one
        # button -- so a row of radios here would be four choices of which three
        # are refusals. Written into the form as well as skipped, because the
        # value is persisted and a 4 left over from the Object output would
        # reach the door.
        form["count"] = 1
        with focus.item(ctx.state, FOCUS_PANE, "seed"):
            changed, seed = form_ui.number("seed", "Seed", int(form["seed"]))
        if changed:
            form["seed"] = max(0, seed)
        _seed_row(ctx, form, form_ui)
        return
    with focus.item(ctx.state, FOCUS_PANE, "count") as focused:
        changed, picked = form_ui.segmented_choice(
            "count",
            # Not "References": the References *section* 150 dp above this
            # one is the input images, and one word cannot be both the pictures
            # going in and the count coming out.
            "Tiles" if form.get("output") == "tile" else "How many",
            str(form["count"]),
            tuple((str(count), str(count)) for count in (1, 2, 4, 8)),
            help_text="How many alternatives this run should create.",
            compact=True,
        )
        if changed:
            form["count"] = int(picked)
            ctx.state.clear_field_error("count")
        # Hand-answered, as the output switch is: a row of radios is one
        # control to the keyboard even though it is four items to imgui.
        if focused:
            choices = (1, 2, 4, 8)
            here = choices.index(form["count"]) if form["count"] in choices else 0
            if imgui.is_key_pressed(imgui.Key.left_arrow):
                form["count"] = choices[(here - 1) % len(choices)]
            if imgui.is_key_pressed(imgui.Key.right_arrow):
                form["count"] = choices[(here + 1) % len(choices)]
    # After the whole row rather than inside it: the ring goes round the last
    # item drawn, and what was refused is "how many", not the fourth radio.
    with focus.item(ctx.state, FOCUS_PANE, "seed"):
        changed, seed = form_ui.number("seed", "Seed", int(form["seed"]))
    if changed:
        form["seed"] = max(0, seed)
    _seed_row(ctx, form, form_ui)


def _seed_row(ctx: Any, form: dict[str, Any], form_ui: forms.Form) -> None:
    """Reroll and Lock, beside the seed field both paths draw.

    Split out when the Sheet output stopped drawing the count: the two paths
    share everything from here down, and a second copy of a wrap-aware row is
    exactly the kind of duplicate that comes back as a control drawn nowhere at
    1.5 scale.
    """
    # No ring on the seed, deliberately: nothing in ``service`` raises a
    # refusal naming it (the range check is fieldless, and the widget already
    # clamps), and a call here would also have to sit after the Reroll and Lock
    # controls that share its line -- where the rect it rings is the help
    # marker's. A pointer at the wrong control is worse than no pointer.
    # Wrapped rather than clipped: at 1.5 scale the seed field, its label,
    # Reroll, Lock and the help marker come to more than the sidebar's content
    # region, and ``same_line`` past the edge draws a control nowhere -- the
    # bug that once hid seven of them. Found by the 1.5-scale half of the
    # screenshot pass, which is the half that keeps finding these.
    if controls.button("Reroll", role=controls.ButtonRole.GHOST):
        form["seed"] = random_seed()
    changed, locked = form_ui.switch(
        "seed_locked",
        "Lock seed",
        bool(form["seed_locked"]),
        help_text="Reuse this seed when the form is unchanged.",
        helper="Unlocked, every submit rerolls it.",
    )
    if changed:
        form["seed_locked"] = locked


def _submit(ctx: Any, form: dict[str, Any]) -> None:
    imgui.dummy((0, sp(8)))
    imgui.separator()
    problems = validate(form)
    for problem in problems:
        imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(*theme.rgba(theme.ERR)))
        imgui.text_wrapped(problem)
        imgui.pop_style_color()
    count = _safe_int(form.get("count"), 1)
    spec = create_assets.selected(form)
    if form.get("output") == "sheet":
        # Its own sentence rather than a third noun in the line below: a sheet
        # is one press and either one generation or three, which "One sheet - a
        # few seconds" would misreport in the sprite case by a factor of three.
        widgets.muted(
            "One sheet - about a minute"
            if form.get("sheet_type") != "sprite"
            else "A character and two candidate sheets - a few minutes"
        )
    else:
        noun = {
            "image_2d": "image",
            "model_3d": "reference",
            "seamless_tile": "tile",
        }.get(spec.key, "reference")
        widgets.muted(
            f"{count} {noun}s - a few seconds each"
            if count > 1
            else f"One {noun} - a few seconds"
        )
    busy = ctx.busy("submit")
    enabled = not problems and not busy
    with focus.item(ctx.state, FOCUS_PANE, "generate") as focused:
        pressed = widgets.primary_button(spec.create_label, (-1, sp(34)), enabled=enabled)
        anchors.mark("create/generate")
        # Enter on the last stop of the ring, which is what makes the whole
        # form keyboard-only: everything above it is a stock imgui control that
        # answers the keyboard once it has focus, and this is the one that
        # cannot be "typed into". Ctrl+Enter still works from anywhere.
        if focused and enabled and _enter_pressed():
            pressed = True
    if pressed:
        generate(ctx, form)
    if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
        imgui.set_tooltip("Ctrl+Enter")


def _enter_pressed() -> bool:
    return imgui.is_key_pressed(imgui.Key.enter) or imgui.is_key_pressed(imgui.Key.keypad_enter)


def validate(form: dict[str, Any]) -> list[widgets.Problem]:
    """What would be refused, said before the button is pressed.

    A summary rather than a refusal on submit: the API checks all of this too,
    but a disabled button with a reason beats a toast after a round trip.

    Each entry carries the control it is about (:class:`widgets.Problem`, a
    ``str`` subclass, so the aggregate block and every existing comparison are
    unchanged). The field is what lets the *keyboard* doors -- Ctrl+Enter and
    the palette, which call :func:`generate` directly and never draw that block
    -- put the ring on the control the button path would have pointed at.
    """
    problems: list[widgets.Problem] = []
    asset_key = form.get("asset_type")
    if asset_key is not None and asset_key not in create_assets.ASSET_TYPES:
        problems.append(widgets.Problem("Choose a recognised asset type.", "asset_type"))
    prompt = form.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        problems.append(widgets.Problem("A prompt is required.", "prompt"))
    if isinstance(prompt, str) and len(prompt) > MAX_PROMPT:
        problems.append(
            widgets.Problem(f"The prompt is over {MAX_PROMPT} characters.", "prompt")
        )
    count = _safe_int(form.get("count"), 0)
    if not 1 <= count <= MAX_REFERENCE_COUNT:
        problems.append(
            widgets.Problem(
                f"References must be between 1 and {MAX_REFERENCE_COUNT}.", "count"
            )
        )
    grid = form.get("output") == "sheet" and form.get("sheet_type") != "sprite"
    base = form.get("base_model")
    style = form.get("style_lora")
    # A tile set's fixed recipe does not read either selection. It validates
    # its pinned pair at its own service door.
    if not grid and (not isinstance(base, str) or base not in modelslib.BASE_MODELS):
        problems.append(widgets.Problem("Choose a recognised image model.", "base_model"))
    if not grid and (not isinstance(style, str) or (style and style not in modelslib.STYLE_LORAS)):
        problems.append(widgets.Problem("Choose a recognised style LoRA.", "style_lora"))
    if not grid and style:
        try:
            weight = float(form.get("lora_weight"))
        except (TypeError, ValueError, OverflowError):
            weight = float("nan")
        if not modelslib.LORA_WEIGHT_MIN <= weight <= modelslib.LORA_WEIGHT_MAX:
            problems.append(
                widgets.Problem(
                    f"Style strength must be between {modelslib.LORA_WEIGHT_MIN:g} "
                    f"and {modelslib.LORA_WEIGHT_MAX:g}.",
                    "style_lora",
                )
            )
    # The tile-grid arm is the one output that does not go through
    # ``create_job``: ``create_tile_sheet`` pins its own base, its own LoRA and
    # its own ControlNet and reads none of the four fields below. So the three
    # checks after this are skipped for it -- not as a tolerance, but because a
    # disabled Generate reading "Conditioning needs a reference image" over a
    # ``control`` the run will never load is a refusal about somebody else's
    # job. It is reachable rather than theoretical: ``control`` is persisted and
    # ``ref_path`` is VOLATILE, so any session that once conditioned an Object
    # reopens with the pair already split. The sprite arm is deliberately *not*
    # exempt -- its first step is an ordinary reference job and reads all four.
    # Both reachable from a restored form rather than from this frame's
    # controls, which is why they are checked here and not only where the
    # widgets are drawn: a persisted selection outlives the ref_path that
    # justified it (ref_path is VOLATILE), and the base model can be changed
    # under Advanced after a control was picked.
    if not grid and not form.get("ref_path") and (form.get("ip_adapter") or form.get("control")):
        problems.append(
            widgets.Problem("Conditioning needs a reference image.", "ref_path")
        )
    if (
        not grid
        and form.get("control")
        and base not in modelslib.controlnet_bases()
    ):
        problems.append(
            widgets.Problem("Structure control needs a full-CFG model.", "base_model")
        )
    # Reachable the same way: a style picked under one base survives a change
    # of base under Advanced, and the service refuses the submit outright
    # rather than generating without it.
    if not grid and form.get("style_lora") and form["style_lora"] not in (
        modelslib.loras_by_base().get(base) or []
    ):
        problems.append(
            widgets.Problem(
                "The style LoRA is not fitted to this model's architecture.",
                "style_lora",
            )
        )
    if form.get("output") == "tile" and base not in modelslib.tile_bases():
        problems.append(
            widgets.Problem("Seamless tiles need an SDXL model.", "base_model")
        )
    if form.get("output") == "sheet":
        # The tile arm's own two fields, and nothing about the model: what a
        # sheet is short of on this host is a different question, and
        # ``weights_problem`` asks it against the rows a sheet actually loads.
        size = str(form.get("tile_size") or "")
        if grid and size not in {str(s) for s in svc_tilesheets.TILE_SIZES}:
            # Reachable from a restored form rather than from this frame's
            # control: the value is persisted, and the menu it came from can
            # change between releases.
            problems.append(
                widgets.Problem(
                    f"Tile size must be one of {list(svc_tilesheets.TILE_SIZES)}.",
                    "tile_size",
                )
            )
        if grid and _view_of(form) not in svc_tilesheets.VIEWS:
            # Interpolated rather than spelled out. The sentence used to name
            # its two values, so the day a third arrived the form would have
            # refused it with a list that did not contain it.
            problems.append(
                widgets.Problem(
                    f"View must be one of {list(svc_tilesheets.VIEWS)}.", "projection"
                )
            )
    return problems


def _safe_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return fallback


def submit_kwargs(form: dict[str, Any]) -> dict[str, Any]:
    """The 2D form as create_job takes it.

    ``output`` is this pane's own switch and never "model": this pane is the
    first stage of a two-stage pipeline made visible, and going straight to a
    mesh from here would spend two minutes of GPU on an image nobody has
    approved. A tile has no second stage at all.
    """
    tile = form.get("output") == "tile"
    # Every surviving field is machinery (model identity, conditioning) rather
    # than subject taxonomy, so a tile submits the same set an object does.
    known = set(guidancelib.form_fields())
    fields = {k: v for k, v in form.items() if k in known and v not in ("", None)}
    sprite_sheet = None
    if form.get("output") == "sheet" and form.get("sheet_type") == "sprite":
        # The sprite arm is an ordinary reference job carrying a follow-up
        # request, which is the rig checkbox's shape: the character is a row in
        # its own right and the sheet is queued against it once it lands. So it
        # comes through this function rather than round a second door, and the
        # only thing that distinguishes it is this block.
        sprite_sheet = {
            "sheet_type": form.get("sheet_layout") or "turnaround",
            "logical_size": int(form.get("cell_size") or 64),
            "colors": svc_sprites.DEFAULT_SPRITE_COLORS,
        }
    return {
        "kind": "text",
        "prompt": form["prompt"].strip(),
        "output": "tile" if tile else "reference",
        "count": _safe_int(form.get("count"), 1),
        **({"sprite_sheet": sprite_sheet} if sprite_sheet is not None else {}),
        "seed": int(form["seed"]),
        "negative_prompt": form["negative_prompt"] or None,
        "lora_weight": float(form["lora_weight"]) if form.get("style_lora") else None,
        # Mirroring lora_weight: sent only alongside the selection it scales,
        # so an unused slider never reaches params as a live setting.
        "ip_scale": float(form["ip_scale"]) if form.get("ip_adapter") else None,
        "control_scale": float(form["control_scale"]) if form.get("control") else None,
        "control_end": float(form["control_end"]) if form.get("control") else None,
        "guidance_fields": fields,
        **create_assets.persisted_intent(form),
    }


def _generate_tile_sheet(ctx: Any, form: dict[str, Any]) -> None:
    """Submit the tile grid, on the shared ``submit`` key.

    The same key every other output uses, deliberately: it is one form and one
    Generate button, so two submits in flight from it is the thing the key
    exists to prevent -- and the busy state the button reads is keyed on that
    name.

    The form values are read here, on the frame thread, because they are UI
    state; the reference *file* is read in the task, because a large one would
    freeze the window for as long as the disk took. ``generate``'s own split,
    kept.
    """
    prompt = form["prompt"].strip()
    tile_size = int(form.get("tile_size") or 32)
    view = _view_of(form)
    seed = int(form["seed"])
    negative = form.get("negative_prompt") or None
    ref_path = form.get("ref_path") or ""

    def run():
        reference = None
        if ref_path:
            try:
                with Path(ref_path).open("rb") as fh:
                    reference = fh.read(MAX_UPLOAD_BYTES + 1)
            except OSError as exc:
                # ``field=`` so the ring lands on the file control rather than
                # the refusal arriving as a toast with no subject.
                raise Invalid(
                    f"could not read {Path(ref_path).name}: {exc}", field="ref_path"
                ) from exc
        return svc_tilesheets.create_tile_sheet(
            ctx.svc,
            prompt=prompt,
            tile_size=tile_size,
            view=view,
            seed=seed,
            negative_prompt=negative,
            reference=reference,
            **create_assets.persisted_intent(form),
        )

    ctx.submit("submit", run)


def anchor_kwargs(ctx: Any, form: dict[str, Any], kwargs: dict[str, Any]) -> str:
    """Fold the active profile's style anchor into a submit, in place.

    -> the path to read as the conditioning reference, or "" for none.

    A manual attachment wins and this returns "" for it: the anchor is what a
    whole set has in common, and the image the user just dropped is what this
    one asset needs. The path is *returned* rather than read here because this
    runs on the frame thread and generate() reads files in its task.
    """
    if form.get("ref_path"):
        return ""
    found = profiles.active_anchor(ctx.settings, ctx.svc.config)
    if found is None:
        return ""
    path, scale = found
    # setdefault, not assignment: a form that already names an adapter chose
    # it, and the anchor only supplies one where there was none.
    kwargs.setdefault("guidance_fields", {}).setdefault(
        "ip_adapter", profiles.ANCHOR_ADAPTER
    )
    kwargs["ip_scale"] = float(scale)
    return str(path)


# Which registry row each model-shaped form field would need, and the name the
# refusal puts the ring on. The service layer is the authority (``check_weights``
# raises the real refusal at the door); this is the courtesy in front of it, and
# it answers from ``ctx.model_rows`` rather than the disk for ``model_gate``'s
# reason -- this runs on the frame thread sixty times a second.
_WEIGHT_FIELDS = (
    ("base_model", "base", "The image model"),
    ("style_lora", "lora", "The style LoRA"),
    ("ip_adapter", "adapter", "The reference adapter"),
    ("control", "control", "The structure control"),
)


def weights_problem(ctx: Any, form: dict[str, Any]) -> widgets.Problem | None:
    """The first selected model this host has not downloaded, or None.

    Beside :func:`validate` rather than inside it, and the split is deliberate.
    ``validate`` is about the *form* -- a prompt that is empty, a count out of
    range -- and is true on any machine; this is about this **install**, and
    two forms identical in every field can disagree about it. Keeping them
    apart is also what stops the aggregate block above Generate from listing a
    download as a mistake the user made.

    ``model_gate``'s doctrine throughout: an empty ``model_rows`` says nothing
    rather than everything-is-missing (a headless ctx, or the first frame
    before the answers land, must not lock a fully-installed host), and a row
    the snapshot has never heard of is skipped. A stale snapshot costs a
    missing warning, never a wrong outcome -- ``service.validation.check_weights``
    is still the authority and still refuses at the door.
    """
    by_key = {str(row.get("row_key")): row for row in (getattr(ctx, "model_rows", None) or [])}
    if not by_key:
        return None
    grid = form.get("output") == "sheet" and form.get("sheet_type") != "sprite"
    if grid:
        # A tile set's door pins its own base and LoRA and ignores the form's,
        # so walking ``_WEIGHT_FIELDS`` here would point at a selection the run
        # never reads. Sprite sheets are different: their preliminary
        # reference does use those selected fields, then a pinned final recipe.
        for row_key in sheet_rows(form):
            row = by_key.get(row_key)
            if row is None or row.get("present"):
                continue
            label = row.get("label") or row_key
            return widgets.Problem(
                f"A sheet needs {label!r}, which is not downloaded. "
                f"Install it in Settings.",
                "output",
            )
        return None
    for field, kind, noun in _WEIGHT_FIELDS:
        chosen = str(form.get(field) or "")
        if not chosen:
            continue
        row = by_key.get(f"{kind}:{chosen}")
        if row is None or row.get("present"):
            continue
        label = row.get("label") or chosen
        return widgets.Problem(
            f"{noun} {label!r} is selected but not downloaded. "
            f"Install it in Settings, or pick another.",
            field,
        )
    # The expander, keyed differently: the form holds a *mode*, and every
    # mode runs the one registry entry -- so the row is looked up by that
    # entry's key rather than by the form value.
    if (form.get("expand") or "off") != "off":
        row = by_key.get(f"expander:{modelslib.DEFAULT_EXPANDER}")
        if row is not None and not row.get("present"):
            label = row.get("label") or modelslib.DEFAULT_EXPANDER
            return widgets.Problem(
                f"Prompt expansion needs {label!r}, which is not downloaded. "
                f"Install it in Settings, or turn expansion off.",
                "expand",
            )
    if form.get("output") == "sheet" and form.get("sheet_type") == "sprite":
        # Only after the visible preliminary recipe has passed: both stages
        # are real requirements, and the first problem should point at the
        # editable control before naming the locked follow-up recipe.
        for row_key in sheet_rows(form):
            row = by_key.get(row_key)
            if row is None or row.get("present"):
                continue
            label = row.get("label") or row_key
            return widgets.Problem(
                f"The final sheet needs {label!r}, which is not downloaded. "
                f"Install it in Settings.",
                "output",
            )
    return None


def refuse(ctx: Any, problems: list[widgets.Problem]) -> None:
    """Say no where the user can see it, whichever door they came through.

    Shared with ``settings_3d.promote`` because the two refusals are the same
    refusal: a form that would not submit, reached from a button that is
    disabled *and* from two keyboard doors where nothing is disabled at all.
    The button path shows the whole list as red text above itself and needs
    nothing from here; the keyboard paths get the first problem as a toast --
    first rather than all of them, because a stack of four toasts is a wall,
    and the ring below marks the rest.
    """
    ctx.state.clear_field_errors()
    for problem in problems:
        ctx.state.note_field_error(getattr(problem, "field", ""), str(problem))
    if problems:
        ctx.toast(str(problems[0]), "warn")


def generate(ctx: Any, form: dict[str, Any]) -> None:
    problems = validate(form)
    # The install-shaped refusal, folded in behind the form-shaped ones: it is
    # the same "this will not submit" from the user's side, and putting it here
    # rather than at the service door turns a two-minute queue-and-fail into an
    # immediate sentence naming the control. The service still refuses at the
    # door; this only means the user rarely reaches it.
    weights = weights_problem(ctx, form)
    if weights is not None and not problems:
        problems = [weights]
    if problems:
        # Said out loud, because this is not only the button's path. Ctrl+Enter
        # and the palette's Generate call straight in here, and the only
        # feedback a refusal had was the red block above the button -- which
        # the keyboard user is by definition not looking at, and which the
        # palette covers. So the first problem becomes a toast and every
        # problem that names a control gets its ring, which is exactly what the
        # button path shows without being asked.
        refuse(ctx, problems)
        return
    # A new submit is judged on its own: the rings from the last one describe a
    # request that no longer exists, and leaving them up would have the app
    # pointing at a control while it works on the value in it.
    ctx.state.clear_field_errors()
    if not form["seed_locked"]:
        # Generation is deterministic in the seed, so an unchanged form would
        # otherwise produce the identical image twice and read as a no-op.
        form["seed"] = random_seed()
    ctx.state.remember_prompt(form["prompt"])
    if form.get("output") == "sheet" and form.get("sheet_type") != "sprite":
        # The one output that does not go through ``create_job``: a tile grid is
        # its own job kind, with its own door and its own admission. The sprite
        # arm deliberately *does* go through it -- see ``submit_kwargs`` -- so
        # this is the only branch here.
        _generate_tile_sheet(ctx, form)
        return
    kwargs = submit_kwargs(form)
    ref_path = form.get("ref_path") or anchor_kwargs(ctx, form, kwargs)

    # The form values are read here, on the frame thread, because they are UI
    # state; the *file* is read in the task, because a large one would freeze
    # the window for as long as the disk took. Copied from settings_3d.upload,
    # including the MAX_UPLOAD_BYTES + 1 read that is create_job's contract.
    def run():
        if ref_path:
            try:
                with Path(ref_path).open("rb") as fh:
                    kwargs["reference"] = fh.read(MAX_UPLOAD_BYTES + 1)
            except OSError as exc:
                # ``field=`` so the ring lands on the file control rather
                # than the refusal arriving as a toast with no subject: the
                # caller cannot know which of the form's inputs was at fault,
                # and this one does.
                raise Invalid(
                    f"could not read {Path(ref_path).name}: {exc}", field="ref_path"
                ) from exc
        return svc_jobs.create_job(ctx.svc, **kwargs)

    ctx.submit("submit", run)
