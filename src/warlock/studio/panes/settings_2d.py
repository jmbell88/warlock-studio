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

from ... import generation, vectors
from ... import guidance as guidancelib
from ... import models as modelslib
from ...bench import findings as findings_lib
from ...pipelines import tileatlas as tileatlaslib
from ...pipelines import tilesheet as tilesheetlib
from ...service import jobs as svc_jobs
from ...service import palettes as svc_palettes
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
    if form.get("generation_type") not in generation.GENERATION_TYPES:
        form["generation_type"] = create_assets.legacy_asset_type(form)
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
                if _negative_supported(ctx, form):
                    widgets.section("Negative prompt / Avoid")
                    _negative(ctx, form)
                _history(ctx, form)
                if _is_tile_arm(form):
                    # Above the model and not under Advanced: it decides what
                    # the sheet is a sheet *of*, and in two of its three
                    # layouts the words that are actually generated are typed
                    # in this section rather than in the Prompt one above it.
                    widgets.section("Tile layout")
                    _tile_layout(ctx, form, form_ui)
                elif form.get("output") == "sheet":
                    # The sprite arm's own layout, in the same place and for the
                    # same reason: which action and how many directions is what
                    # the sheet depicts, and it is also what decides whether the
                    # press is one generation or sixteen.
                    widgets.section("Sprite layout")
                    _sprite_layout(ctx, form, form_ui)
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
                        sprite_arm = form.get("sheet_type") == "sprite"
                        if sprite_arm:
                            _sprite_size(ctx, form, form_ui)
                        else:
                            _tile_size(ctx, form, form_ui)
                        _target_cell(ctx, form, form_ui)
                        _pixel_look(ctx, form, form_ui, sprite=sprite_arm)
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
        "##generation_type", before, list(generation.GENERATION_TYPE_OPTIONS)
    )
    form["asset_type"] = picked if picked in generation.GENERATION_TYPES else before
    form["generation_type"] = form["asset_type"]
    spec = create_assets.sync_legacy_fields(form)
    if spec.key != before:
        ctx.state.preview_dirty_at = time.monotonic()
        ctx.state.clear_field_error("asset_type")


#: Where the sentences explaining a *tier* change's clears are kept between
#: frames -- :data:`CLEARED_KEY`'s sibling, and separate from it because the two
#: are drawn under different combos and a change of one must not wipe the
#: other's explanation.
QUALITY_CLEARED_KEY = "quality_cleared"


def _quality(ctx: Any, form: dict[str, Any]) -> None:
    """Fast/Quality is a recipe choice, not a checkpoint choice."""
    if create_assets.selected(form).intent == "tileset":
        return
    before = str(form.get("quality") or "quality")
    picked = widgets.combo("##quality", before, [("fast", "Fast"), ("quality", "Quality")])
    form["quality"] = picked if picked in generation.QUALITY_TIERS else before
    widgets.field_error(ctx.state, "quality")
    if form["quality"] != before:
        ctx.state.preview_dirty_at = time.monotonic()
        ctx.state.clear_field_error("quality")
        ctx.state.preview[QUALITY_CLEARED_KEY] = clear_for_tier(ctx, form)
    for note in ctx.state.preview.get(QUALITY_CLEARED_KEY) or ():
        widgets.muted_wrapped(note)
    _recipe_note(ctx, form)


def _recipe_note(ctx: Any, form: dict[str, Any]) -> None:
    """What the chosen tier trades, under the combo that chose it.

    Fast is genuinely worse -- four steps instead of thirty, no ControlNet, no
    negative prompt -- and that is fine; a tier that is honestly worse and says
    so is a choice the user can make. What is not fine is the silence this
    replaces: until 2026-08-29 both tiers named ``sdxl_cfg`` and the control
    changed nothing at all, and merely pointing it at a different checkpoint
    without saying what changed would only move the silence one step.
    """
    if str(form.get("model_mode") or "auto") == "advanced":
        # Advanced names its own checkpoint and prints that model's own
        # description; a tier note there would describe a routing that is not
        # happening.
        return
    resolved = _resolved_recipe(ctx, form)
    if resolved is None or not resolved.recipe.note:
        return
    widgets.muted_wrapped(resolved.recipe.note)


def _resolved_recipe(ctx: Any, form: dict[str, Any]) -> Any:
    """What automatic routing would load for this form, or None.

    Wrapped because it runs on the frame thread from three note helpers: a
    partially restored form must make the pane say nothing rather than raise
    inside the draw, which is ``_negative_supported``'s standing rule here.
    """
    try:
        request = generation.request_from_legacy(form)
        return generation.resolve_recipe(request, ctx.svc.config)
    except Exception:
        # Silent on purpose, and the same choice ``_negative_supported`` makes
        # for the same reason: this runs sixty times a second inside the draw,
        # so a partially restored form must make the pane say *nothing* rather
        # than log a line per frame or raise through the frame loop. The
        # service remains the final compatibility gate, and it is not silent.
        return None


def clear_for_tier(ctx: Any, form: dict[str, Any]) -> list[str]:
    """Drop the selections the newly chosen tier cannot run.

    -> one sentence per selection cleared, for the pane to show.

    :func:`clear_unusable`'s argument applied to the other end of the same
    routing. Under automatic routing the *tier* picks the checkpoint, so
    switching to Fast strands a ControlNet and an Avoid text exactly the way
    switching the base model under Advanced does -- and both of those controls
    are hidden rather than merely disabled once the tier cannot use them, which
    would leave Generate refusing on a field that is off screen. Clearing is
    the same choice ``clear_unusable`` makes and for the same reason: a
    refusal the user cannot act on is a dead end.

    Called only on a change of tier, never per frame -- see ``clear_unusable``.
    """
    cleared: list[str] = []
    if str(form.get("model_mode") or "auto") == "advanced":
        # The tier does not choose the checkpoint here, so it cannot strand
        # anything; ``clear_unusable`` owns that half.
        return cleared
    resolved = _resolved_recipe(ctx, form)
    if resolved is None:
        # Either the form does not compile or this host qualifies no recipe;
        # the Recipe combo already says so, and clearing selections on the
        # strength of an answer nobody has would be the silent rewrite
        # ``clear_unusable`` refuses to do.
        return cleared
    # Cannot raise: ``_resolved_recipe`` just built the same request and got a
    # recipe out of it.
    caps = generation.capability_controls(generation.request_from_legacy(form), resolved)
    if form.get("control") and not caps["controlnet"]:
        form["control"] = ""
        cleared.append(
            "The structure control was cleared: this recipe runs at guidance 0 "
            "and cannot run a ControlNet."
        )
    if str(form.get("negative_prompt") or "").strip() and not caps["negative_prompt"]:
        form["negative_prompt"] = ""
        cleared.append(
            "The Avoid text was cleared: this recipe runs at guidance 0, where "
            "a negative prompt has no effect."
        )
    return cleared


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
    sizes = tile_sizes_for(form)
    changed, picked = form_ui.segmented_choice(
        "tile_size", "Tile size", str(form.get("tile_size", "32")),
        tuple((str(size), str(size)) for size in sizes),
        help_text="How many pixels across one tile is.",
        # Why the menu is shorter here than it is for the grid layout. Said
        # rather than left as an absence: 48 px is offered for a grid sheet and
        # is missing from this row, and an unexplained gap reads as a bug.
        helper=(
            f"A seamless material is drawn at {tileatlaslib.MATERIAL_PX} px and "
            f"reduced, so its tile size has to divide that exactly."
            if is_seamless(form)
            else ""
        ),
        compact=True,
    )
    if changed:
        form["tile_size"] = picked
        ctx.state.clear_field_error("tile_size")


#: Where the sentences explaining a layout change's clears are kept between
#: frames. :data:`CLEARED_KEY`'s sibling and for its reason -- the notice belongs
#: to the change the user just made, not to the form that outlives the session.
TILE_MODE_CLEARED_KEY = "tile_mode_cleared"


def clear_for_layout(form: dict[str, Any]) -> list[str]:
    """Drop the geometry the newly chosen layout cannot draw.

    -> one sentence per value moved, for the pane to show.

    :func:`clear_unusable`'s rule applied to the tile arm, and for its reason: a
    control that ``validate`` refuses while offering only legal values is a dead
    end unless the illegal value is cleared, and the two things a seamless layout
    cannot keep -- a 48 px tile and a view that does not wrap -- are both
    persisted, so both survive a switch of layout.

    Called only when the layout changes, never per frame. A form *restored* with
    a size the layout refuses keeps it: ``validate`` names it above Generate, the
    control offers the sizes that work, and rewriting a stored value on the way
    in would change a request nobody touched.
    """
    if not is_seamless(form):
        return []
    options = _tile_options()
    cleared: list[str] = []
    sizes = tile_sizes_for(form)
    if str(form.get("tile_size") or "") not in {str(size) for size in sizes}:
        form["tile_size"] = str(options["defaults"]["tile_size"])
        cleared.append(
            f"The tile size moved to {form['tile_size']} px: a seamless material "
            f"is reduced from one {tileatlaslib.MATERIAL_PX} px frame, and only "
            f"{sizes} divide it exactly."
        )
    views = views_for(form)
    if _view_of(form) not in views:
        form["projection"] = views[0]
        label = options["view_labels"].get(views[0], views[0])
        cleared.append(
            f"The view moved to {label}: a seamless material wraps a square, and "
            f"neither an isometric diamond nor a 3/4 tile's visible front face is "
            f"one."
        )
    return cleared


def _tile_layout(ctx: Any, form: dict[str, Any], form_ui: forms.Form) -> None:
    """What this sheet is a sheet *of*, and therefore which request it compiles.

    Three layouts, and the pane never holds a second opinion about any of their
    ceilings: the list of them, their labels, the material and cell limits and
    the two geometry menus all come from ``svc_tilesheets.tile_sheet_options``,
    which is the door that enforces them.
    """
    options = _tile_options()
    before = tile_mode_of(form)
    changed, picked = form_ui.combo(
        "mode",
        "Layout",
        before,
        tuple((key, options["mode_labels"].get(key, key)) for key in options["modes"]),
        help_text=(
            "Materials and Terrain set draw each surface on its own, seamlessly, "
            "and lay the results out. Grid paints one frame through a guide and "
            "cuts it into sixty-four cells."
        ),
    )
    if changed and picked != before:
        form["tile_mode"] = picked
        # The geometry a seamless layout cannot keep, dropped with a sentence --
        # and the preview recomposed, because the words that will be sent are a
        # different set of words now.
        ctx.state.preview[TILE_MODE_CLEARED_KEY] = clear_for_layout(form)
        ctx.state.preview_dirty_at = time.monotonic()
        for field in _TILE_FIELDS:
            ctx.state.clear_field_error(field)
    for note in ctx.state.preview.get(TILE_MODE_CLEARED_KEY) or ():
        widgets.muted_wrapped(note)
    mode = tile_mode_of(form)
    if mode == svc_tilesheets.MODE_MATERIALS:
        _tile_materials(ctx, form, form_ui, options)
    elif mode == svc_tilesheets.MODE_TERRAIN:
        _tile_terrain(ctx, form, form_ui)
    else:
        _tile_grid(ctx, form, form_ui, options)


#: Every field this section owns, for the one thing that has to name them
#: together: clearing last submit's rings when the layout changes, since a
#: refusal about a material list is not about the request the user is now
#: composing.
_TILE_FIELDS = (
    "mode",
    "prompt_items",
    "variants",
    "inner_terrain",
    "outer_terrain",
    "boundary",
    "tile_size",
    "projection",
)


def _tile_materials(
    ctx: Any, form: dict[str, Any], form_ui: forms.Form, options: dict[str, Any]
) -> None:
    """The list of surfaces, and how many draws of each.

    One generation per cell, which is why the count is said out loud beside the
    field rather than left to be discovered when the queue takes four minutes.
    """
    lines = material_lines(form)
    variants = _safe_int(form.get("variants"), 1)
    cells = len(lines) * max(variants, 1)
    before = str(form.get("materials") or "")
    changed, text = form_ui.multiline_text(
        "prompt_items",
        "Materials",
        before,
        height=90,
        max_length=MAX_PROMPT * int(options["max_materials"]),
        help_text=(
            "One surface per line. Each line is generated on its own as a "
            "seamless tile, so this list is where the variety comes from."
        ),
        helper=(
            f"{len(lines)}/{options['max_materials']} materials - "
            f"{len(lines)} x {variants} = {cells} cells, "
            f"{options['max_cells']} at most"
        ),
    )
    if changed:
        form["materials"] = text
        ctx.state.preview_dirty_at = time.monotonic()
        ctx.state.clear_field_error("prompt_items")
    changed, picked = form_ui.segmented_choice(
        "variants",
        "Draws of each",
        str(variants),
        tuple((str(count), str(count)) for count in range(1, int(options["max_variants"]) + 1)),
        help_text=(
            "How many times each line is drawn. Every draw is its own full "
            "generation, on its own seed."
        ),
        compact=True,
    )
    if changed:
        form["variants"] = picked
        ctx.state.clear_field_error("variants")
    # Reachable at last: the service and the worker have carried ``style_lock``
    # since the materials mode landed, and no pane set it. The cost is stated
    # beside it because it is the one thing the checkbox changes about the
    # budget -- the first material becomes the IP-Adapter reference for every
    # one after it, which loads the encoder.
    changed, locked = controls.checkbox(
        "Keep one style across the list", bool(form.get("style_lock"))
    )
    widgets.help_marker(
        "The first material is generated on its own, then used as the appearance "
        "reference for every material after it, so the list reads as one artist's "
        "set. Loads the IP-Adapter (about 1.2 GB more on the card) and makes "
        "materials 2..N depend on the first one's roll."
    )
    if changed:
        form["style_lock"] = locked
    changed, erase = controls.checkbox("Erase the seam", bool(form.get("seam_erase")))
    widgets.help_marker(
        "After each material is drawn, roll it so the wrap seam runs through the "
        "middle and redraw a band around it in place. One more pass per material; "
        "use it when the wrap preview shows a join."
    )
    if changed:
        form["seam_erase"] = erase
    _tile_description_note()


def _tile_terrain(ctx: Any, form: dict[str, Any], form_ui: forms.Form) -> None:
    """Two surfaces and the world they share.

    The forty-seven cases are composited from the pair by a computed coverage
    field, so neither field here describes an edge -- see :func:`_tile_terrain`'s
    boundary helper and ``pipelines.tilemask``.
    """
    for key, label, help_text in (
        (
            "inner_terrain",
            "Inside",
            "The surface the forty-seven cases are pictures of: the islands, "
            "coastlines and peninsulas a stroke paints.",
        ),
        (
            "outer_terrain",
            "Outside",
            "What surrounds it. Generated too, so it is described too.",
        ),
    ):
        changed, text = form_ui.text(
            key, label, str(form.get(key) or ""), help_text=help_text, max_length=MAX_PROMPT
        )
        if changed:
            form[key] = text
            ctx.state.preview_dirty_at = time.monotonic()
            ctx.state.clear_field_error(key)
    changed, text = form_ui.text(
        "boundary",
        "Shared setting",
        str(form.get("boundary") or ""),
        help_text=(
            "Words added to both surfaces so two separate generations come back "
            "sharing a world and a palette -- 'a temperate coastline'. Optional."
        ),
        # Named for the *place*, and the helper says why: the boundary itself is
        # a computed field, and a drawn edge inside a tile is the one defect this
        # layout exists to make impossible.
        helper=(
            "Not a description of the join. The join is computed, and a drawn "
            "edge would be cut across by it."
        ),
        max_length=MAX_PROMPT,
    )
    if changed:
        form["boundary"] = text
        ctx.state.preview_dirty_at = time.monotonic()
        ctx.state.clear_field_error("boundary")
    _tile_description_note()


def _tile_grid(
    ctx: Any, form: dict[str, Any], form_ui: forms.Form, options: dict[str, Any]
) -> None:
    """The original layout: one frame, one guide, sixty-four cells.

    Kept reachable and described honestly rather than hidden. It is the only
    layout that draws a 3/4 or an isometric tile, which is why the view lives
    here -- the other two accept one view and would draw a picker with nothing
    to pick.
    """
    before = _view_of(form)
    changed, picked = form_ui.combo(
        "projection",
        "View",
        before,
        tuple((key, options["view_labels"].get(key, key)) for key in options["views"]),
        help_text="Where the camera is. Only this layout draws the other two.",
    )
    if changed and picked != before:
        form["projection"] = picked
        ctx.state.preview_dirty_at = time.monotonic()
        ctx.state.clear_field_error("projection")
    widgets.muted_wrapped(
        "One 1024 px frame is painted through a grid guide and cut into "
        f"{options['tiles']} cells. Every cell of the guide is identical, so the "
        "cells tend to come back as one scene cut up or as one tile repeated "
        "(docs/measurements/2026-08-18-tile-sheet-grid.md). Materials and Terrain "
        "set were built to replace it; it stays for 3/4 and isometric, and for "
        "rerunning a sheet made under it."
    )


def _tile_description_note() -> None:
    """What the Description above actually does in the two seamless layouts.

    It names the sheet and is recorded with it; the words that reach the model
    are the ones typed in this section. Said out loud because the alternative is
    a form with two prompt-shaped fields, one of which silently does nothing to
    the picture -- which is the failure a preview of the wrong template would
    also produce.
    """
    widgets.muted_wrapped(
        "The Description above names this sheet in the library. What each tile is "
        "painted from is what you type here."
    )


#: The Action combo's own key space, and why it is not simply the action name.
#:
#: **A legacy kind and an action can be spelled the same and mean different
#: sheets.** Legacy ``walk`` is a four-frame cycle over four directions; the
#: action ``walk`` is eight frames, and picking it composes ``walk8``. One combo
#: cannot carry both under the key ``"walk"`` -- a stored legacy walk would show
#: as the action, and selecting it would silently double the user's cycle. So
#: every entry that names a *kind* is prefixed, every entry that names an action
#: is bare, and the two are converted at exactly one place each.
LEGACY_KEY_PREFIX = "legacy:"

#: What a legacy sheet kind is called in that combo. The ``walk`` is labelled
#: with its frame count because that is the whole trap; the turnaround is the
#: first entry and needs no disambiguation.
LEGACY_LAYOUT_LABELS: dict[str, str] = {
    "turnaround": "Turnaround (still views)",
    "walk": "Walk (legacy, 4 frames)",
}


def sprite_action_key(layout: str) -> str:
    """The Action combo's key for a stored ``sheet_layout``. See
    :data:`LEGACY_KEY_PREFIX`."""
    mode, action, _directions = generation.sprite_from_layout(layout)
    if mode in generation.SPRITE_LEGACY_MODES:
        return f"{LEGACY_KEY_PREFIX}{mode}"
    if layout not in generation.SPRITE_SHEET_KINDS:
        # A kind from some other build: named as itself, so the combo can show
        # what the form is actually set to rather than moving it.
        return layout
    return action


def sprite_action_options(
    options: dict[str, Any], current: str
) -> tuple[tuple[str, str], ...]:
    """The Action combo's entries: the turnaround, then what has a guide.

    An action is offered **only if its pose guide is on this disk**, which is
    ``sprite_options``' own filter and the whole reason that key exists: the
    guide is what decides where the limbs go, so an action offered without one
    is a control whose result is eight bands of an unposed character -- or, once
    the doors refuse it, a control whose only outcome is that refusal.

    A stored layout the menu does not carry is appended rather than dropped,
    which is :func:`palette_options`' rule and its reason: silently moving a
    form off the thing it says it is set to is how a user comes to submit
    something they did not choose. That covers both the legacy ``walk`` -- a
    real sheet this build still draws -- and a kind from some other build, which
    is not.

    ``current`` is a :func:`sprite_action_key`, not a layout.
    """
    out = [(f"{LEGACY_KEY_PREFIX}turnaround", LEGACY_LAYOUT_LABELS["turnaround"])]
    out.extend((entry["key"], entry["label"]) for entry in options["actions"])
    if current not in {key for key, _label in out}:
        bare = current.removeprefix(LEGACY_KEY_PREFIX)
        out.append((current, LEGACY_LAYOUT_LABELS.get(bare, f"{bare} (unavailable)")))
    return tuple(out)


def sprite_action_entry(options: dict[str, Any], action: str) -> dict[str, Any] | None:
    """``sprite_options()['actions']``' row for ``action``, or None for a
    turnaround or a legacy kind -- neither of which is one."""
    for entry in options["actions"]:
        if entry["key"] == action:
            return entry
    return None


def sprite_layout_for(
    options: dict[str, Any], action: str, directions: int
) -> str:
    """The ``sheet_layout`` an Action/Directions pair names.

    Takes a :func:`sprite_action_key`, so a prefixed legacy entry resolves to the
    kind it names and the Directions control has nothing to say about it.

    Falls back to the action's *first available* direction count rather than to
    the asked-for one, because the two controls move independently: picking an
    action that has no eight-direction guide while the Directions control still
    says eight must land on a sheet that exists.
    """
    if action.startswith(LEGACY_KEY_PREFIX):
        return action.removeprefix(LEGACY_KEY_PREFIX)
    entry = sprite_action_entry(options, action)
    if entry is None:
        return action
    counts = [row["count"] for row in entry["directions"]]
    if not counts:
        return action
    return f"{action}{directions if directions in counts else counts[0]}"


def _sprite_logical(form: dict[str, Any], sizes: tuple[int, ...]) -> int:
    """The cell size this form will actually be submitted at.

    Clamped **here** rather than only in the picker, and that is the difference
    between a gate and a decoration: the Action control is always on screen and
    the size picker is inside Advanced, so a user who picks an eight-frame walk
    without ever opening Advanced would otherwise compile a 64px request that
    the door refuses -- a press that does nothing, decided by a section they
    never looked at.
    """
    if not sizes:
        return _safe_int(form.get("cell_size"), 64)
    asked = _safe_int(form.get("cell_size"), max(sizes))
    return asked if asked in sizes else max(sizes)


def sprite_plan(form: dict[str, Any]) -> dict[str, Any]:
    """What this form's sprite arm will actually draw, arithmetic included.

    One function, read by the Dimensions section's summary line, by the size
    picker's ladder and by :func:`sprite_sheet_kwargs`, so what the user is told
    and what is submitted are the same numbers rather than two calculations of
    them. Every one of them comes from ``sprite_options()`` -- the door's own --
    for the reason the tile arm's do: a pane that recomputes a cell count is a
    label that goes stale the first time a frame count moves.
    """
    options = _sprite_options()
    layout = str(form.get("sheet_layout") or "turnaround")
    _mode, action, directions = generation.sprite_from_layout(layout)
    entry = sprite_action_entry(options, action)
    row = None
    if entry is not None and layout not in generation.SPRITE_LEGACY_MODES:
        row = next(
            (r for r in entry["directions"] if r["count"] == directions), None
        )
    if row is None:
        # A turnaround or a legacy walk: one generation of one fixed atlas, and
        # its grid is the ``sheet_types`` table's rather than an action's.
        fixed = next(
            (t for t in options["sheet_types"] if t["key"] == layout),
            options["sheet_types"][0],
        )
        sizes = tuple(fixed["logical_sizes"])
        return {
            "layout": layout,
            "action": "",
            "directions": len(fixed["directions"]),
            "frames": int(fixed["frames_per_direction"]),
            "cells": int(fixed["cells"]),
            "bands": 1,
            "candidates": 2,
            "generations": 2,
            "sizes": sizes,
            "logical_size": _sprite_logical(form, sizes),
        }
    candidates = int(row["candidates"])
    sizes = tuple(entry["logical_sizes"])
    return {
        "layout": layout,
        "action": action,
        "directions": int(row["count"]),
        "frames": int(entry["frames"]),
        "cells": int(row["cells"]),
        "bands": int(row["bands"]),
        "candidates": candidates,
        "generations": int(row["bands"]) * candidates,
        "sizes": sizes,
        "logical_size": _sprite_logical(form, sizes),
    }


def _sprite_cost(plan: dict[str, Any]) -> str:
    """The one sentence under the sprite controls, from :func:`sprite_plan`."""
    # The wait comes from the door, not from a second multiplication of
    # ``seconds_per_generation`` here: the sprite panel draws the same sentence
    # about the same press, and two copies of the arithmetic is two promises.
    when = svc_sprites.generation_time_phrase(plan["generations"])
    draft = "one draft" if plan["candidates"] == 1 else f"{plan['candidates']} drafts"
    return (
        f"{plan['directions']} directions x {plan['frames']} frames = "
        f"{plan['cells']} cells, {plan['generations']} generations for "
        f"{draft}, {when}."
    )


def _sprite_layout(ctx: Any, form: dict[str, Any], form_ui: forms.Form) -> None:
    """What the sheet depicts: an action, and how many ways it is drawn.

    Above the model and not under Advanced, for the reason the tile arm's layout
    is: it decides what the sheet is a sheet *of*, and it is the choice that
    decides how long the press will take -- eight directions is eight
    generations, which is a fact a user is owed before pressing rather than
    after.
    """
    options = _sprite_options()
    layout = str(form.get("sheet_layout") or "turnaround")
    _mode, action, directions = generation.sprite_from_layout(layout)
    current = sprite_action_key(layout)
    changed, picked = form_ui.combo(
        "sprite_action",
        "Action",
        current,
        sprite_action_options(options, current),
        help_text=(
            "What the character is doing. Only the actions this install has a "
            "pose guide for are offered -- the guide is what puts the limbs "
            "where they belong."
        ),
    )
    if changed:
        form["sheet_layout"] = sprite_layout_for(options, picked, directions)
        ctx.state.clear_field_error("sheet_type")
        ctx.state.preview_dirty_at = time.monotonic()
        layout = str(form["sheet_layout"])
        _mode, action, directions = generation.sprite_from_layout(layout)
    entry = sprite_action_entry(options, action)
    if entry is not None and layout not in generation.SPRITE_LEGACY_MODES:
        counts = [row["count"] for row in entry["directions"]]
        changed, count = form_ui.segmented_choice(
            "sprite_directions",
            "Directions",
            str(directions),
            tuple((str(c), f"{c} ways") for c in counts),
            help_text=(
                "How many ways the character is drawn facing. One direction is "
                "one generation, so eight of them is eight."
            ),
            compact=True,
        )
        if changed:
            form["sheet_layout"] = sprite_layout_for(options, action, int(count))
    widgets.muted_wrapped(_sprite_cost(sprite_plan(form)))


def _sprite_size(ctx: Any, form: dict[str, Any], form_ui: forms.Form) -> None:
    """Only the editable dimension of a sprite asset type.

    **The ladder is gated on the action**, which is the difference between a
    refusal the user can act on and one they meet after pressing. One direction
    of an eight-frame walk is eight cells of ``PX_PER_ART_PIXEL`` times the
    logical size, and above 32px that band is past one SDXL frame -- so
    ``spritesynth.plan_sheet`` refuses it, naming both numbers, and both service
    doors re-raise that sentence. A picker still offering 48 and 64 there would
    be three sizes of which two are a refusal.
    """
    options = _sprite_options()
    plan = sprite_plan(form)
    sizes = plan["sizes"] or tuple(options["logical_sizes"])
    current = str(plan["logical_size"])
    if str(form.get("cell_size", "")) != current:
        # Written back so the control shows what the submit will send. The clamp
        # itself is ``_sprite_logical``'s, above -- a picker that was the only
        # thing holding the line would not hold it for a user who never opened
        # this section.
        form["cell_size"] = current
    changed, picked = form_ui.segmented_choice(
        "cell_size", "Cell size", current,
        tuple((str(size), str(size)) for size in sizes),
        help_text="How many pixels across one frame is.", compact=True,
    )
    if changed:
        form["cell_size"] = picked
    if len(sizes) < len(options["logical_sizes"]):
        # "An 8-frame", "A 4-frame". The frame counts a plan can carry are 4, 6,
        # 8 and 16, and 8 is the only one spoken with a leading vowel -- so this
        # is the whole rule rather than a general article function, which would
        # be a paragraph of English for three numbers that never change.
        article = "An" if plan["frames"] == 8 else "A"
        widgets.muted_wrapped(
            f"{article} {plan['frames']}-frame {plan['action']} is drawn one "
            f"whole direction at a time, and only {max(sizes)}px and below fit "
            "one generation."
        )


def _pixel_look(
    ctx: Any, form: dict[str, Any], form_ui: forms.Form, *, sprite: bool
) -> None:
    """An authored palette, dithering, and -- on the sprite arm only -- outlines.

    The three settings both sheet doors have taken since they started sharing
    ``service.pixelopts`` and that no pane offered, which made the whole
    capability unreachable: an authored ramp is the single highest-leverage art
    input in the program (``pipelines.pixelize``' own words) and it could not be
    named from the one form that composes a sheet.

    **No outline control on the tile arm, and that is not an omission.**
    ``pixelize._edge_mask`` pads with ``constant_values=False``, so on a cell
    that is opaque edge to edge -- which every tile is -- every border pixel has
    a "transparent" neighbour and ``inner`` returns the outer ring of *each*
    cell: a grid line around every tile rather than an outline of anything in
    one. ``create_tile_sheet`` refuses it by name; a form offering it would be a
    control whose only outcome is that refusal.

    **The dither box is not hidden behind the palette**, which is the opposite
    of what ``inspector`` does two panes over, and the difference is in the
    pipelines rather than in taste. ``asset2d`` applies dither *inside*
    ``map_palette`` and takes its own quantize branch otherwise, so there it
    genuinely does nothing without a palette -- and that file records
    ``bool(opts.dither and opts.palette)`` for exactly that reason. Both sheet
    paths route through ``map_palette`` either way:
    ``tilesheet.quantize_tiles`` branches on ``not entries and not dither``, and
    ``queue`` hands the sprite atlas to ``pixelize.pixelize_atlas`` with
    whatever ``resolve_palette`` returned. So on this form a dither with no
    palette dithers against the derived table, which is a real and different
    picture -- and hiding the box would make *that* the unreachable capability.
    Troupe's pane, the one that shipped these controls first, draws it
    unconditionally for the same reason.

    The palette list comes from the arm's own door and never from
    ``tile_sheet_options`` / ``sprite_options``: those are pure functions of
    module constants and this pane caches them for the life of the process, so
    a directory listing inside one would mean a palette dropped in five minutes
    ago never appears. ``inspector.palette_names`` is the one stat-per-frame
    guard over that listing and is shared rather than copied.
    """
    from . import inspector

    door = svc_sprites.sprite_palettes if sprite else svc_tilesheets.tile_sheet_palettes
    installed = inspector.palette_names(ctx, door)
    chosen = str(form.get("palette") or "")
    if installed or chosen:
        # Only when there is something to pick. A combo whose one entry is
        # "derive one" is a picker with nothing in it, and palettes are opt-in
        # -- the honest rendering of "none installed" is no control, which is
        # ``palettes.available``'s own stated rule and ``inspector``'s. A form
        # that *names* one is the exception: see :func:`palette_options`.
        changed, picked = form_ui.combo(
            "palette",
            "Palette",
            chosen,
            palette_options(installed, chosen),
            help_text=(
                "Map every pixel to the nearest colour of a palette you "
                "authored, instead of to the colours this render happened to "
                "contain."
            ),
            # From the loader's own tuple, never restated. This line named
            # ``.pal`` and ``.txt`` until 2026-08-29 and ``palettes.SUFFIXES``
            # has never carried either, so a user who dropped one in the folder
            # was told it would work and then watched it not appear -- no error,
            # no row, nothing to see.
            helper=svc_palettes.SUFFIX_HELP,
        )
        if changed:
            form["palette"] = picked
            ctx.state.clear_field_error("palette")
    changed, dithered = form_ui.switch(
        "dither",
        "Dither",
        bool(form.get("dither")),
        help_text=(
            "Add an ordered 4x4 offset before each pixel picks its colour, so "
            "a gradient reads as a texture rather than as a band."
        ),
        helper=(
            "Against the chosen palette."
            if form.get("palette")
            else "Against the palette this sheet derives for itself."
        ),
    )
    if changed:
        form["dither"] = dithered
    if not sprite:
        return
    options = _sprite_options()
    changed, picked = form_ui.segmented_choice(
        "outline",
        "Outline",
        str(form.get("outline") or options["defaults"]["outline"]),
        tuple((mode, OUTLINE_LABELS.get(mode, mode)) for mode in options["outlines"]),
        help_text=(
            "Darken the edge of each frame. Inside recolours the character's "
            "own edge pixels; Around grows the silhouette by one pixel, which "
            "a frame already touching its cell edge will have clipped."
        ),
        compact=True,
    )
    if changed:
        form["outline"] = picked
        ctx.state.clear_field_error("outline")


def palette_options(installed: list[str], chosen: str) -> tuple[tuple[str, str], ...]:
    """The palette combo's entries: "derive one", what is installed, and a
    selection that is not.

    ``lora_options``' rule, and for its reason. A palette is a file, so a stem
    the form holds can stop existing between two launches -- an external drive,
    a folder tidied -- and the door refuses the submit by that name. Dropping it
    from the list would leave the combo showing its bare stem with no
    explanation, or, in any control that falls back to entry zero, silently
    rewrite the user's choice to "derive one" and change what the sheet looks
    like without saying so. Listed and marked, the thing keeping Generate off is
    the one thing on screen.

    Shared with the profile editor, which draws the same picker over the same
    directory against a draft rather than the live form.
    """
    options = [("", "Derived from the render"), *((name, name) for name in installed)]
    if chosen and chosen not in installed:
        options.append((chosen, f"{chosen} - not in the palette folder"))
    return tuple(options)


#: What each ``pixelize.OUTLINE_MODES`` entry is called on screen. The keys are
#: the pipeline's words and these are sentences about what happens, which is
#: this pane's rule for every other segmented control: "outer" is a direction
#: only to somebody who already knows where the outline goes.
OUTLINE_LABELS = {"none": "None", "inner": "Inside", "outer": "Around"}


def _target_cell(ctx: Any, form: dict[str, Any], form_ui: forms.Form) -> None:
    """Optional final reduction; blank means keep the high-resolution cell."""
    values = [("", "Keep working resolution")]
    values.extend((str(size), f"{size}px") for size in generation.TARGET_CELL_PRESETS)
    values.append(("custom", "Custom (8–256px)"))
    current = str(form.get("target_cell_px") or "")
    known = current if current in {x[0] for x in values} else "custom"
    selected = widgets.combo("##target_cell_px", known, values)
    if selected != "custom":
        form["target_cell_px"] = selected
        return
    raw = form.get("target_cell_px")
    try:
        number = int(raw)
    except (TypeError, ValueError):
        number = generation.TARGET_CELL_PRESETS[-1]
    changed, number = form_ui.number("target_cell_px_custom", "Custom cell size", number)
    if changed:
        form["target_cell_px"] = str(number)
    widgets.muted("Blank preserves the 256px/512px working cell; reduction never upscales.")


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


def _is_tile_arm(form: dict[str, Any]) -> bool:
    """Whether this form is the Sheet output's *tile* arm.

    The expression six places in this file were spelling out, given a name once
    the tile arm grew three layouts of its own: "not a sprite sheet" and "the
    grid layout" stopped being the same sentence, and a local called ``grid``
    that meant the first is exactly how the submit came to compile a materials
    request with no materials in it.
    """
    return form.get("output") == "sheet" and form.get("sheet_type") != "sprite"


def sheet_rows(form: dict[str, Any]) -> tuple[str, ...]:
    """Which registry rows the Sheet output currently needs.

    A function of the form rather than a constant, because the two arms load
    different things and the tile arm's own list depends on whether a reference
    is attached: its IP-Adapter is optional, and a gate that demanded one would
    tell a user with everything the common request uses that they are missing a
    download. Shared with :func:`weights_problem` so the note above the button
    and the gate inside the section cannot disagree.

    **And on the layout**, since the tile arm grew three of them: the grid guide
    *is* a ControlNet and the two seamless layouts never open one, so asking for
    the grid's rows under a materials sheet told a user with everything that
    request uses to download canny weights it will never load. The per-mode maps
    are ``tile_sheet_options``' own, which is where :func:`rows_needed` publishes
    them.
    """
    if form.get("sheet_type") == "sprite":
        return svc_sprites.SPRITE_ROWS
    key = "mode_reference_rows_needed" if form.get("ref_path") else "mode_rows_needed"
    return tuple(_tile_options()[key][tile_mode_of(form)])


def tile_mode_of(form: dict[str, Any]) -> str:
    """The tile layout this form is asking for, in the service's own spelling.

    An unrecognised stored value reads as the default rather than as a refusal,
    which is this pane's standing rule for a settings-file value: the field is
    persisted, the menu can change between releases, and a form that resolved to
    nothing would disable Generate over a control whose value the user cannot
    see.

    **The default is not ``grid``.** The door refuses the grid layout unless the
    request explicitly asks for it -- see ``create_tile_sheet``'s ``allow_grid``
    -- precisely so that a default nobody chose can never land on the one layout
    a measurement says does not work.
    """
    stored = str(form.get("tile_mode") or svc_tilesheets.DEFAULT_MODE)
    return stored if stored in svc_tilesheets.TILE_MODES else svc_tilesheets.DEFAULT_MODE


def is_seamless(form: dict[str, Any]) -> bool:
    """Whether this form draws each tile as its own seamless material.

    The two layouts ``pipelines.tileatlas`` builds, asked as one question,
    because everything that differs between them and the grid -- the tile sizes
    that divide a 1024px material, the one view that wraps, what the preview
    composes -- differs the same way for both.
    """
    return tile_mode_of(form) != svc_tilesheets.MODE_GRID


def tile_sizes_for(form: dict[str, Any]) -> list[int]:
    """Which tile sizes this layout can publish.

    Sourced from the service rather than filtered here: a seamless material is
    reduced from one 1024px frame on an exact partition, so 48 is on the grid's
    menu and not on this one -- and the day that frame size changes, this list
    changes with it because ``tile_sheet_options`` derives it by asking
    ``pipelines.tileatlas``.
    """
    options = _tile_options()
    return list(options["seamless_tile_sizes" if is_seamless(form) else "tile_sizes"])


def views_for(form: dict[str, Any]) -> list[str]:
    """Which views this layout can draw. One, for the seamless pair.

    ``tileatlas``' own list, for :func:`tile_sizes_for`'s reason: an isometric
    tile is a diamond and a 3/4 tile has a visible front face, so neither wraps,
    and the sentences explaining that live in the pipeline that refuses them.
    """
    options = _tile_options()
    return list(options["seamless_views" if is_seamless(form) else "views"])


def material_lines(form: dict[str, Any]) -> tuple[str, ...]:
    """The materials field as the door takes it: one surface per non-blank line.

    Blank lines are dropped rather than counted, which is what makes a trailing
    newline harmless -- the door drops them too, and a form that counted them
    would report a cell total the request will not produce.
    """
    return tuple(
        line
        for line in (raw.strip() for raw in str(form.get("materials") or "").splitlines())
        if line
    )


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


def seamless_subject(form: dict[str, Any]) -> str | None:
    """The subject the *first* cell of a seamless layout will be generated from.

    ``None`` when the request does not describe one yet, which is a real answer
    rather than a failure: a materials sheet with no lines and a terrain set
    with no inner surface have no first material, and the honest preview of a
    request that names nothing is no preview at all.

    Composed by ``pipelines.tileatlas`` rather than here -- the style clause both
    layouts append and the context a terrain set shares between its two halves
    are that module's, and a second copy of either would be a preview of a
    sentence nothing sends.
    """
    mode = tile_mode_of(form)
    try:
        if mode == svc_tilesheets.MODE_TERRAIN:
            return tileatlaslib.terrain_subjects(
                str(form.get("inner_terrain") or ""),
                str(form.get("outer_terrain") or ""),
                str(form.get("boundary") or ""),
            )[0]
        lines = material_lines(form)
        return tileatlaslib.material_subject(lines[0], index=0, total=len(lines))
    except (IndexError, ValueError):
        return None


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
        if _is_tile_arm(form):
            # The sprite arm is exempt: its first step is an ordinary reference
            # of one character, which is exactly what the enrichment describes.
            widgets.muted_wrapped(
                "A tileset is never expanded: each tile is generated from the "
                "line typed for it, and the enrichment describes one subject."
                if is_seamless(form)
                else "A tile grid is never expanded: the enrichment describes "
                "one subject, and a sheet is sixty-four."
            )


def _preview_nothing(state: Any) -> None:
    """Drop the composition on screen and stop asking for a new one.

    For the one case where "what would be sent" has no answer: a seamless layout
    that names no material yet. Leaving the last preview up would show a
    composition of words that are no longer in the form, which is the same
    untruth as previewing the wrong template.

    The two *cleared* notices survive it. They are not the preview -- they
    belong to a change the user just made and are only kept in this dict because
    it is the frame-scratch namespace (see :data:`CLEARED_KEY`).
    """
    state.preview = {
        key: value
        for key, value in (state.preview or {}).items()
        if key in (CLEARED_KEY, TILE_MODE_CLEARED_KEY)
    }
    state.preview_dirty_at = 0.0


def _preview(ctx: Any) -> None:
    """The composed prompt, recomputed off-thread after a typing pause."""
    state = ctx.state
    if state.preview_dirty_at and time.monotonic() - state.preview_dirty_at > PREVIEW_DEBOUNCE:
        raw = {k: v for k, v in state.form_2d.items() if v not in ("", None)}
        form = state.form_2d
        grid = _is_tile_arm(form) and not is_seamless(form)
        seamless = _is_tile_arm(form) and is_seamless(form)
        # The *subject*, not the raw prompt, for a grid: the projection and
        # detail clauses are the pipeline's and are appended before the
        # template, so a preview built from the bare prompt would be missing
        # the half of the composition that this output kind adds.
        #
        # And for the two seamless layouts, the *first material* through the
        # tile template -- which is the sentence that will actually run. The
        # grid's own subject previewed here would be a composition no generation
        # on this path ever sees: they compose one material at a time and the
        # Description is not part of any of them.
        subject = form["prompt"]
        if grid:
            subject = tilesheetlib.sheet_subject(form["prompt"], _view_of(form))
        elif seamless:
            first = seamless_subject(form)
            if first is None:
                # Nothing describable yet, so nothing is previewed: a request
                # with no material has no first material, and showing the
                # Description here would show a sentence this layout never
                # sends.
                _preview_nothing(state)
                return
            subject = first
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
            # single-centred-object framing its job will never use. A seamless
            # layout's cells go through ``prompt.TILE_TEMPLATE`` -- the very
            # template ``text2image.generate(tile=True)`` selects -- which is why
            # it previews as a tile and not as a sheet.
            tile=form.get("output") == "tile" or seamless,
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
            form["init_image"] = False
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

    widgets.field_label("start image")
    changed, on = controls.checkbox(
        "Start from this image (img2img)", bool(form.get("init_image"))
    )
    widgets.help_marker(
        "The reference is the picture the drawing starts from rather than only "
        "what it looks at. Low strength keeps its layout and repaints the surface; "
        "high strength keeps only the gist."
    )
    if changed:
        form["init_image"] = on
    if form.get("init_image"):
        changed, value = controls.slider_float(
            "Strength##init",
            float(form.get("init_strength") or 0.45),
            *_range(ctx, "init_strength_range", 0.3, 0.65),
        )
        if changed:
            form["init_strength"] = value

    widgets.field_label("structure")
    note = recipe_structure_note(ctx, form) or structure_note(ctx, form)
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


def recipe_structure_note(ctx: Any, form: dict[str, Any]) -> str | None:
    """Why *automatic* routing's recipe cannot run a ControlNet, or None.

    :func:`structure_note`'s sibling for the other half of the Recipe control.
    That one answers for the checkpoint the user picked under Advanced; this
    one answers for the checkpoint the tier picks on their behalf, which is not
    in ``form["base_model"]`` at all. Without it the Structure picker was drawn
    under Fast, the selection was submitted, and the refusal came back from
    ``guidance.normalize`` naming ``base_model`` -- a combo automatic routing
    does not display.
    """
    if str(form.get("model_mode") or "auto") == "advanced":
        return None
    resolved = _resolved_recipe(ctx, form)
    if resolved is None:
        # The pane already says "no compatible installed recipe" under the
        # Recipe combo; saying it again here names the wrong subject.
        return None
    spec = modelslib.BASE_MODELS.get(resolved.base_model)
    if spec is None or spec.controlnet:
        return None
    return (
        f"{resolved.recipe.label} runs at guidance 0 and cannot run a ControlNet. "
        "Switch the Recipe to Quality, or pick a full-CFG model under Advanced."
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
    widgets.field_label("Recipe")
    _quality(ctx, form)
    mode_before = str(form.get("model_mode") or "auto")
    mode = widgets.combo(
        "##model_mode", mode_before, [("auto", "Automatic"), ("advanced", "Advanced")]
    )
    form["model_mode"] = mode if mode in generation.MODEL_MODES else mode_before
    if form["model_mode"] == "auto":
        request = generation.request_from_legacy(form)
        resolved = generation.resolve_recipe(request, ctx.svc.config)
        if resolved is None:
            widgets.muted_wrapped(
                "No compatible installed recipe is available. "
                "Install a model in Settings or open Advanced."
            )
        else:
            imgui.text_wrapped(f"{resolved.recipe.label} · {resolved.base_model}")
            if resolved.warning:
                widgets.wrapped(theme.WARN, resolved.warning)
        return
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
        form["model_override"] = form["base_model"]
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
    _licence_note(form["base_model"])


def _licence_note(key: str) -> None:
    """What this checkpoint's weights permit, under the picker that chose them.

    **The users of a game-asset generator will sell the output**, and this app
    shipped two checkpoints that restrict exactly that while saying so nowhere
    -- not here, not at download time, not in ``docs/MODELS.md``. Telling them
    nothing is the posture most likely to hurt somebody who trusted the tool.

    Drawn only when there is something to say: a permissive licence gets one
    muted line, and eleven identical "commercially permitted" rows would train
    the eye to skip the one row that is not.
    """
    spec = modelslib.BASE_MODELS.get(key or "")
    if spec is None or not spec.license:
        return
    if not spec.commercial:
        # ``wrapped`` in WARN rather than ``muted_wrapped``: this is the one
        # line on the pane that can cost the user money, and a muted sentence
        # among muted sentences is a sentence nobody reads.
        widgets.wrapped(
            theme.WARN,
            f"Licence: {spec.license}. Images from this model may NOT be used "
            f"commercially. {spec.license_note}".strip(),
        )
        return
    if spec.license_note:
        widgets.hint_text(f"Licence: {spec.license}. {spec.license_note}")
        return
    widgets.muted_wrapped(f"Licence: {spec.license} — commercial use permitted.")


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


def _negative_supported(ctx: Any, form: dict[str, Any]) -> bool:
    """Show Avoid only when the resolved recipe will actually consume it."""
    try:
        request = generation.request_from_legacy(form)
        resolved = generation.resolve_recipe(request, ctx.svc.config)
        return generation.capability_controls(request, resolved)["negative_prompt"]
    except Exception:
        # The service remains the final compatibility gate.  During a partially
        # restored form, hiding an unresolved control is safer than presenting
        # an active field whose text would be silently discarded.
        return False


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
        if form.get("sheet_type") != "sprite":
            widgets.muted("One sheet - about a minute")
        else:
            # The arithmetic rather than "a few minutes", which was written when
            # a sprite sheet was always one generation twice and is a factor of
            # eight out for an eight-direction one. Same numbers as the line
            # under the layout controls, from the same function.
            plan = sprite_plan(form)
            widgets.muted(
                f"A character, then {plan['generations']} generations of sheet"
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
    # The *tile arm*, which is not the same question as the *grid layout* -- see
    # :func:`_is_tile_arm`. Every check below that was written when the two were
    # one thing is about the arm, because what makes a tile set exempt from them
    # is that its door pins its own recipe, which all three layouts do.
    tileset = _is_tile_arm(form)
    base = form.get("base_model")
    style = form.get("style_lora")
    # A tile set's fixed recipe does not read either selection. It validates
    # its pinned pair at its own service door.
    if not tileset and (not isinstance(base, str) or base not in modelslib.BASE_MODELS):
        problems.append(widgets.Problem("Choose a recognised image model.", "base_model"))
    if not tileset and (
        not isinstance(style, str) or (style and style not in modelslib.STYLE_LORAS)
    ):
        problems.append(widgets.Problem("Choose a recognised style LoRA.", "style_lora"))
    if not tileset and style:
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
    # The tile arm is the one output that does not go through
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
    if (
        not tileset
        and not form.get("ref_path")
        and (form.get("ip_adapter") or form.get("control"))
    ):
        problems.append(
            widgets.Problem("Conditioning needs a reference image.", "ref_path")
        )
    if (
        not tileset
        and form.get("control")
        and base not in modelslib.controlnet_bases()
    ):
        problems.append(
            widgets.Problem("Structure control needs a full-CFG model.", "base_model")
        )
    # Reachable the same way: a style picked under one base survives a change
    # of base under Advanced, and the service refuses the submit outright
    # rather than generating without it.
    if not tileset and form.get("style_lora") and form["style_lora"] not in (
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
        # Both menus are asked for *this layout*: a seamless material is reduced
        # from one 1024px frame on an exact partition and wraps a square, so 48
        # px and two of the three views are on the grid's menu and not on
        # theirs. The lists come from the service so the pane holds no second
        # opinion about either ceiling.
        sizes = tile_sizes_for(form)
        views = views_for(form)
        if tileset and size not in {str(s) for s in sizes}:
            # Reachable from a restored form rather than from this frame's
            # control: the value is persisted, and the menu it came from can
            # change between releases -- or between layouts.
            problems.append(
                widgets.Problem(f"Tile size must be one of {sizes}.", "tile_size")
            )
        if tileset and _view_of(form) not in views:
            # Interpolated rather than spelled out. The sentence used to name
            # its two values, so the day a third arrived the form would have
            # refused it with a list that did not contain it.
            problems.append(
                widgets.Problem(f"View must be one of {views}.", "projection")
            )
        if tileset:
            problems.extend(_layout_problems(form))
        if tileset or form.get("sheet_type") == "sprite":
            raw_target = form.get("target_cell_px") or ""
            target = None if raw_target == "" else _safe_int(raw_target, -1)
            for issue in generation.validate_target_cell(
                target, isometric=tileset and _view_of(form) == "isometric"
            ):
                problems.append(widgets.Problem(issue.message, issue.field))
    return problems


def _layout_problems(form: dict[str, Any]) -> list[widgets.Problem]:
    """What the chosen tile layout is still short of.

    The door's own refusals, asked before the request exists. Each one names the
    control it is about, and each ceiling is read from
    ``tile_sheet_options`` rather than written here -- the door enforces them
    against ``asset_workflows.collection_cells``, and a second set of numbers in
    a pane is a form that accepts what the door then refuses.

    The grid layout contributes nothing: everything it needs is the prompt and
    the geometry, both checked above.
    """
    options = _tile_options()
    mode = tile_mode_of(form)
    problems: list[widgets.Problem] = []
    if mode == svc_tilesheets.MODE_MATERIALS:
        lines = material_lines(form)
        variants = _safe_int(form.get("variants"), 0)
        if not lines:
            problems.append(
                widgets.Problem(
                    "A materials sheet is the list of surfaces you type; describe "
                    "at least one, one per line.",
                    "prompt_items",
                )
            )
        if len(lines) > int(options["max_materials"]):
            problems.append(
                widgets.Problem(
                    f"{len(lines)} materials is past the {options['max_materials']} "
                    f"one sheet can name.",
                    "prompt_items",
                )
            )
        if any(len(line) > MAX_PROMPT for line in lines):
            problems.append(
                widgets.Problem(
                    f"One material is over {MAX_PROMPT} characters.", "prompt_items"
                )
            )
        if not 1 <= variants <= int(options["max_variants"]):
            problems.append(
                widgets.Problem(
                    f"Draws of each material must be between 1 and "
                    f"{options['max_variants']}.",
                    "variants",
                )
            )
        elif len(lines) * variants > int(options["max_cells"]):
            problems.append(
                widgets.Problem(
                    f"{len(lines)} materials by {variants} draws is "
                    f"{len(lines) * variants} cells, past the "
                    f"{options['max_cells']} one sheet can hold; each cell is its "
                    f"own full generation.",
                    "variants",
                )
            )
    elif mode == svc_tilesheets.MODE_TERRAIN:
        # One sentence per empty field, and each names its own control.
        #
        # This loop used to append one *identical* sentence per empty field, so
        # a fresh terrain form -- where both are empty -- stacked the same
        # words twice above Generate, and neither copy said which of the two it
        # was about. A refusal here is meant to be a sentence the user can act
        # on, and "one of the two fields" is not one. The list stays per-field
        # rather than collapsing to a single line because ``refuse`` rings the
        # control each problem names: a merged line could ring only one of them
        # and would leave the other looking accepted.
        for field, name in (("inner_terrain", "Inside"), ("outer_terrain", "Outside")):
            if not str(form.get(field) or "").strip():
                problems.append(
                    widgets.Problem(
                        f"{name} is empty. A terrain set is two surfaces and both "
                        f"are generated, so both have to be described.",
                        field,
                    )
                )
        for field in ("inner_terrain", "outer_terrain", "boundary"):
            if len(str(form.get(field) or "")) > MAX_PROMPT:
                problems.append(
                    widgets.Problem(f"That is over {MAX_PROMPT} characters.", field)
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
    sprite_sheet = (
        sprite_sheet_kwargs(form)
        if form.get("output") == "sheet" and form.get("sheet_type") == "sprite"
        else None
    )
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
        "init_image": bool(form.get("init_image")) and bool(form.get("ref_path")),
        "init_strength": (
            float(form.get("init_strength") or 0.45)
            if form.get("init_image") and form.get("ref_path")
            else None
        ),
        "control_scale": float(form["control_scale"]) if form.get("control") else None,
        "control_end": float(form["control_end"]) if form.get("control") else None,
        "guidance_fields": fields,
        **create_assets.persisted_intent(form),
    }


def sprite_sheet_kwargs(form: dict[str, Any]) -> dict[str, Any]:
    """The 2D form as ``create_job``'s ``sprite_sheet=`` block takes it.

    :func:`tile_sheet_kwargs`' opposite number on the other arm, and split out
    of :func:`submit_kwargs` for that function's reason: the sprite arm is an
    ordinary reference job *carrying a follow-up request* -- the rig checkbox's
    shape, so the character is a row in its own right and the sheet is queued
    against it once it lands -- and the compilation of that follow-up is the one
    part of the press no test could name while it lived inside a literal.

    ``_jobs_create._check_sprite_sheet`` validates every key here at the
    *reference* door rather than when the follow-up is minted, so a palette
    that has been deleted since the form listed it costs the request instead of
    an SDXL generation and an hour.

    ``candidates`` is sent rather than left to the door's own default, and that
    is not belt-and-braces: the Dimensions section has already told the user how
    many generations this press costs, and a block that let the worker decide
    the number separately is how a form comes to promise eight and spend
    sixteen. It is :func:`sprite_plan`'s number, which is the line's number.
    """
    plan = sprite_plan(form)
    return {
        "sheet_type": plan["layout"],
        "candidates": plan["candidates"],
        "logical_size": plan["logical_size"],
        "colors": svc_sprites.DEFAULT_SPRITE_COLORS,
        "target_cell_px": (
            None if form.get("target_cell_px") in (None, "")
            else _safe_int(form.get("target_cell_px"), 0)
        ),
        # The three the form draws under Dimensions. Sent always rather than
        # only when set: the door's own defaults are these values, and a block
        # that omitted them would make "no palette" and "the form was never
        # asked" the same request -- which is how a setting comes to be recorded
        # as something nobody chose.
        "palette": str(form.get("palette") or ""),
        "dither": bool(form.get("dither")),
        "outline": str(form.get("outline") or svc_sprites.DEFAULT_SPRITE_OUTLINE),
    }


def tile_sheet_kwargs(form: dict[str, Any]) -> dict[str, Any]:
    """The 2D form as ``create_tile_sheet`` takes it, minus the reference bytes.

    :func:`submit_kwargs`' opposite number, and it exists for the reason that
    one does: the tile arm is the only output that does not go through
    ``create_job``, so the compilation of its request had no name and lived
    inside a closure -- where no test could reach it. The submit then went on
    sending a request with no layout in it, against a door whose default layout
    is ``materials``, and every press was refused at ``field="prompt_items"``
    with nothing in the form saying why.

    **``allow_grid`` is sent when, and only when, the user picked the grid.**
    That flag is the door's escape hatch on a refusal about a measurement rather
    than about an impossibility, so an explicit choice is exactly what it is for
    -- and a default that carried it would put every unconsidered press back on
    the layout the measurement is about.
    """
    mode = tile_mode_of(form)
    kwargs: dict[str, Any] = {
        "prompt": str(form.get("prompt") or "").strip(),
        "tile_size": _safe_int(form.get("tile_size"), 32),
        "view": _view_of(form),
        "seed": int(form["seed"]),
        "negative_prompt": form.get("negative_prompt") or None,
        "mode": mode,
        # The pixel look, from the two controls under Dimensions. No ``outline``
        # key at all -- and the absence is load-bearing rather than tidy: the
        # door refuses one by name (a tile is opaque edge to edge, so an outline
        # is a grid line around every cell), and a form that sent even
        # ``"none"`` here would be naming a setting this kind does not have.
        "palette": str(form.get("palette") or ""),
        "dither": bool(form.get("dither")),
        **create_assets.persisted_intent(form),
    }
    if mode == svc_tilesheets.MODE_MATERIALS:
        kwargs["prompt_items"] = list(material_lines(form))
        kwargs["variants"] = _safe_int(form.get("variants"), 1)
    elif mode == svc_tilesheets.MODE_TERRAIN:
        kwargs["inner_terrain"] = str(form.get("inner_terrain") or "").strip()
        kwargs["outer_terrain"] = str(form.get("outer_terrain") or "").strip()
        kwargs["boundary"] = str(form.get("boundary") or "").strip()
    else:
        kwargs["allow_grid"] = True
    return kwargs


def _generate_tile_sheet(ctx: Any, form: dict[str, Any]) -> None:
    """Submit the tile set, on the shared ``submit`` key.

    The same key every other output uses, deliberately: it is one form and one
    Generate button, so two submits in flight from it is the thing the key
    exists to prevent -- and the busy state the button reads is keyed on that
    name.

    The form values are read here, on the frame thread, because they are UI
    state; the reference *file* is read in the task, because a large one would
    freeze the window for as long as the disk took. ``generate``'s own split,
    kept.
    """
    kwargs = tile_sheet_kwargs(form)
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
        return svc_tilesheets.create_tile_sheet(ctx.svc, reference=reference, **kwargs)

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
    if _is_tile_arm(form):
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
    if _is_tile_arm(form):
        # The one output that does not go through ``create_job``: a tile set is
        # its own job kind, with its own door and its own admission. The sprite
        # arm deliberately *does* go through it -- see ``submit_kwargs`` -- so
        # this is the only branch here.
        _generate_tile_sheet(ctx, form)
        return
    resolved = None
    if create_assets.selected(form).key in {"image", "model_3d", "seamless_material"}:
        request = generation.request_from_legacy(form)
        resolved = generation.resolve_recipe(request, ctx.svc.config)
        recipe_issues = generation.validate_request(request, resolved)
        if recipe_issues:
            refuse(ctx, [widgets.Problem(item.message, item.field) for item in recipe_issues])
            return
    kwargs = submit_kwargs(form)
    if resolved is not None:
        # Automatic routing is resolved at submit time. The selected recipe is
        # copied after the legacy door accepts the request so reruns retain the
        # exact model/checksum even if the registry changes later.
        kwargs["guidance_fields"]["base_model"] = resolved.base_model
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
        result = svc_jobs.create_job(ctx.svc, **kwargs)
        if resolved is not None:
            payload = {"version": generation.RECIPE_REGISTRY_VERSION, **resolved.to_dict()}
            for job_id in result.get("ids", [result["id"]]):
                ctx.svc.store.merge_params(
                    job_id,
                    {"generation_request": request.to_dict(), "resolved_recipe": payload},
                )
        return result

    ctx.submit("submit", run)
