"""The 2D pane: everything that composes the SDXL prompt, and Generate.

This pane owns the prompt and every guidance select; the 3D pane owns nothing
that reaches the text encoder. The one field both need is ``platform``, which
is why it is deliberately *two* controls -- here it is a prompt fragment, there
it is the geometry resolution, and one control cannot be owned by two panes.

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
from ...pipelines import prompt as prompt_lib
from ...service import jobs as svc_jobs
from ...service import system as svc_system
from ...service.errors import Invalid
from ...service.validation import (
    MAX_PROMPT,
    MAX_REFERENCE_COUNT,
    MAX_UPLOAD_BYTES,
    random_seed,
)
from .. import controls, dialogs, focus, forms, profiles, theme, tokens, vector_presets, widgets
from ..manual import render as manual_render
from ..tokens import sp
from ..widgets import field_options as _options

PREVIEW_DEBOUNCE = 0.3

# This pane's key in the focus ring (UX.md Phase 3). The controls on the common
# path take a place in it and the ones behind the fold do not: the ring exists
# so a first job can be composed and submitted without the mouse, and Tab
# through forty controls is not that. Anything behind "More options" is reached
# by opening it, which is one Tab and one Enter away.
FOCUS_PANE = "2d"


# What the submit block took last frame, in design pixels (K92). The same
# measure-last-frame idiom the library's footer uses, and for the same reason:
# the block's height is a function of the theme, the UI scale and how many
# problems it is currently reporting, so no constant is right for all of them.
# Seeded at roughly one button plus its cost note, so the first frame reserves
# something sane rather than nothing.
_submit_px = [96.0]


def draw(ctx: Any) -> None:
    state = ctx.state
    form = state.form_2d
    # Form.errors now places the rings and copy beneath the owning controls;
    # these are the routes it replaces and keeps wired by the same field keys:
    # field_error(ctx.state, "prompt")
    # field_error(ctx.state, "base_model")
    # field_error(ctx.state, "style_lora")
    # field_error(ctx.state, "count")
    with forms.Form("create-2d", errors=ctx.state.field_errors) as form_ui:
        # The form scrolls; Generate does not (K92). This pane is twelve
        # collapsible sections tall, and the one control every visit ends with
        # sat at the bottom of all of them.
        focus.pump(state, FOCUS_PANE)
        focus.begin(state, FOCUS_PANE)
        if imgui.begin_child("2d-form", (0, -sp(_submit_px[0]))):
            _presets(ctx, form)
            _vector_presets(ctx)
            _profiles(ctx, form)
            _output(ctx, form)
            widgets.section("Prompt")
            manual_render.help_button(ctx, "settings-2d")
            _prompt(ctx, form, form_ui)
            _history(ctx, form)
            _preview(ctx)
            _run_controls(ctx, form, form_ui)
            _more(ctx, form)
        imgui.end_child()
        top = imgui.get_cursor_pos_y()
        _submit(ctx, form)
        height = imgui.get_cursor_pos_y() - top
        if height > 0:
            _submit_px[0] = height / max(tokens.SCALE, 0.01)


# The twelve optional taxonomies, grouped by what they describe. Grouping and
# per-field labels are the fix for the old column of identical unlabelled
# combos, where a chosen value ("worn", "brass") no longer said which question
# it answered.
GUIDANCE_GROUPS = (
    # ``framing`` sits here rather than in the 3D pane: it is a clause of the
    # SDXL prompt, so the one-owner rule puts it with the pane that owns the
    # prompt. It is not in TILE_FIELDS, so guidance_groups drops it for a tile,
    # which is right -- TILE_TEMPLATE has a framing of its own.
    ("Subject", ("category", "genre", "setting", "silhouette", "framing")),
    ("Style", ("art_style", "palette", "mood", "rarity")),
    ("Surface", ("material", "condition", "emissive")),
)

# What a field is *called* on screen, where that is not its key with the
# underscores taken out. The key stays ``art_style`` deliberately: it is what
# every job on disk recorded, what ``guidance._lookup`` re-normalizes a rerun
# through, and what the findings and verdict buckets are keyed on -- renaming
# it would need a ``_LEGACY_ALIASES`` entry and would still split the corpus in
# two, because a vector recorded under the old spelling is a different string.
FIELD_LABELS = {"art_style": "era style"}


def field_label(field: str) -> str:
    return FIELD_LABELS.get(field, field.replace("_", " "))


def _field_options(ctx: Any, field: str) -> list[tuple[str, str]]:
    """``widgets.field_options`` with this pane's own name for the blank entry.

    The empty option is the one that names the *question* -- it is what the
    combo shows until something is chosen -- so a renamed label that stopped at
    the heading would leave the control itself still saying "art style...".
    """
    options = widgets.field_options(ctx, field)
    if options and options[0][0] == "":
        options[0] = ("", f"{field_label(field)}...")
    return options


def guidance_groups(form: dict[str, Any]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """The taxonomy groups this form should draw.

    A tile has no subject, so category, silhouette, rarity, mood and emissive
    describe something that is deliberately not in the picture -- drawing them
    would offer controls the prompt compiler then throws away. The fields are
    not *cleared*, only hidden and unsubmitted, so switching back brings back
    what was typed.
    """
    if form.get("output") != "tile":
        return GUIDANCE_GROUPS
    allowed = set(prompt_lib.TILE_FIELDS)
    out = []
    for title, fields in GUIDANCE_GROUPS:
        kept = tuple(f for f in fields if f in allowed)
        if kept:
            out.append((title, kept))
    return tuple(out)


# The one reveal (UX.md Phase 3), and what is behind it. A key rather than a
# literal at three call sites because ``request_open`` names it too.
MORE_KEY = "2d/more"

# The header nested one level deeper inside the fold, and the fields it draws
# refusals against. Named beside MORE_KEY because a refusal reaching one of
# these must open *both* headers: opening the outer fold alone rings a control
# the inner, still-collapsed "Advanced" header never draws.
ADVANCED_KEY = "2d/advanced"
ADVANCED_FIELDS = ("base_model", "style_lora", "negative_prompt")


def folded_fields(form: dict[str, Any]) -> tuple[str, ...]:
    """Every form key that lives behind "More options".

    Derived from the taxonomy groups rather than written out, so a field added
    to ``GUIDANCE_GROUPS`` is folded by having been added -- and the three that
    are not in a group are named here because they are genuinely a different
    list, not because anybody chose to restate one.
    """
    grouped = tuple(f for _title, fields in GUIDANCE_GROUPS for f in fields)
    return grouped + (
        "platform",
        "ref_path",
        "ip_adapter",
        "control",
        "base_model",
        "style_lora",
        "negative_prompt",
    )


def more_summary(form: dict[str, Any]) -> str:
    """What the fold is hiding, said while it is closed.

    Disclosure, not deletion: the point of a reveal is that the user can see
    there is something behind it and roughly what. The *count* is the half that
    matters and the reason this is not a static caption -- a form restored with
    a style, a genre and a conditioning image set looks identical to an empty
    one with the fold shut, which is how somebody comes to spend two minutes of
    GPU on settings they had forgotten were there.
    """
    from ..state import default_form_2d

    defaults = default_form_2d()
    set_count = sum(
        1
        for key in folded_fields(form)
        if form.get(key) not in ("", None) and form.get(key) != defaults.get(key)
    )
    what = "Subject, Style, Surface, Reference and the model"
    if not set_count:
        return f"{what} - nothing set."
    return f"{what} - {set_count} set."


def _more(ctx: Any, form: dict[str, Any]) -> None:
    """The twelve taxonomy selects, the reference and Advanced, behind one fold.

    This pane was "twelve collapsible sections tall" by its own comment, and the
    disclosure it had -- ``default_open=False`` on two of them -- did not help,
    because the eight that were open are the ones a first job does not need. The
    common path above is what the first screen shows: what to draw, what kind of
    output, how many. Everything here refines that.

    Nothing is removed and nothing is hidden that could refuse a submit
    silently: the aggregate block above Generate still lists every problem, and
    a refusal that names a control *in here* opens the fold on the next frame
    rather than ringing a control nobody can see.
    """
    for key in folds_to_open(set(ctx.state.field_errors), form):
        widgets.request_open(key)
    if not widgets.header("More options", default_open=False, persist_key=MORE_KEY):
        widgets.muted_wrapped(more_summary(form))
        return
    _guidance(ctx, form)
    _reference(ctx, form)
    _advanced(ctx, form)


def folds_to_open(named: set[str], form: dict[str, Any]) -> tuple[str, ...]:
    """Which persist keys a refusal naming ``named`` fields must reveal.

    A pure function of the error set and the form, so the safety property --
    a refusal never rings a control nothing draws -- is assertable without a
    GL context. Outer fold first: ``request_open`` is order-insensitive, but
    the tuple reads as the path the user will watch open.
    """
    if not named & set(folded_fields(form)):
        return ()
    if named & set(ADVANCED_FIELDS):
        return (MORE_KEY, ADVANCED_KEY)
    return (MORE_KEY,)


def _findings_hint(ctx: Any, param: str, value: Any) -> str | None:
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
    doc = findings_lib.load(Path(ctx.svc.config.bench_dir) / "findings.json")
    return findings_lib.hint(
        doc,
        param,
        value,
        prompt_hash=vectors.prompt_hash(ctx.state.form_2d.get("prompt")),
    )


def _guidance(ctx: Any, form: dict[str, Any]) -> None:
    for title, fields in guidance_groups(form):
        widgets.section(title)
        if imgui.begin_table(f"guidance/{title}", 2):
            for field in fields:
                imgui.table_next_column()
                widgets.field_label(field_label(field))
                imgui.set_next_item_width(-1)
                options = _field_options(ctx, field)
                before = form[field]
                form[field] = widgets.combo(f"##{field}", form[field], options, 0)
                # One call site for twelve controls: the refusal is looked up by
                # the field's own name, so the loop that draws the taxonomy is
                # also the loop that points at whichever of them was refused.
                widgets.field_error(ctx.state, field)
                if form[field] != before:
                    ctx.state.clear_field_error(field)
                hint = _findings_hint(ctx, field, form[field])
                if hint is not None:
                    widgets.hint_text(hint)
            imgui.end_table()
    if form.get("output") == "tile":
        # platform is a hint about how much detail to draw an *object* with,
        # and is not in TILE_FIELDS -- so on a tile it is a control whose value
        # the prompt compiler discards.
        return
    # Narrowed to leave room for the marker: a full-width combo pushes it off
    # the panel.
    #
    # "detail brief", not "platform detail" (UX.md Phase 3). This and the 3D
    # pane's "Detail" were two near-identical names for a prompt fragment and a
    # geometry resolution, kept apart by nothing but a tooltip on each
    # apologising for the other -- and both were called after the *key*
    # (``platform``), which is a storage name rather than a description of what
    # the control does. It says "brief" because that is what it is: an
    # instruction in the prompt about how much detail to draw, which the sampler
    # may or may not honour. The key stays ``platform``: it is what every job on
    # disk recorded and what the findings buckets are keyed on, and this is
    # ``FIELD_LABELS``' own argument for keeping ``art_style``.
    widgets.field_label("detail brief")
    before = form["platform"]
    form["platform"] = widgets.combo(
        "##g_platform", form["platform"], _options(ctx, "platform"), width=-30
    )
    widgets.field_error(ctx.state, "platform")
    if form["platform"] != before:
        ctx.state.clear_field_error("platform")
    widgets.help_marker(
        "A prompt hint about how much detail to draw. How much geometry the "
        "mesh gets is the 3D pane's Mesh resolution, and separate."
    )


# --- pieces -----------------------------------------------------------------


def _preset_matches(form: dict[str, Any], preset: dict[str, Any]) -> bool:
    fields = {k: v for k, v in (preset.get("fields") or {}).items() if k in form}
    if any(form.get(k) != v for k, v in fields.items()):
        return False
    return not preset.get("prompt") or form.get("prompt") == preset["prompt"]


def _presets(ctx: Any, form: dict[str, Any]) -> None:
    presets = ctx.guidance.get("presets") or []
    if not presets:
        return
    # The combo *shows* the preset the form currently equals, or "Custom": the
    # old write-only picker showed "preset..." forever, so applying one left no
    # trace of which one the form was wearing.
    current = next((p["key"] for p in presets if _preset_matches(form, p)), "")
    options = [("", "Custom")] + [(p["key"], p["label"]) for p in presets]
    widgets.field_label("preset")
    with focus.item(ctx.state, FOCUS_PANE, "preset"):
        chosen = widgets.combo("##preset", current, options)
    if not chosen or chosen == current:
        return
    preset = next((p for p in presets if p["key"] == chosen), None)
    if preset is None:
        return
    # A preset fills the fields rather than becoming a hidden mode: everything
    # it set stays visible and editable, which is what makes it a starting
    # point rather than a black box.
    form.update({k: v for k, v in (preset.get("fields") or {}).items() if k in form})
    if preset.get("prompt"):
        form["prompt"] = preset["prompt"]
    ctx.state.preview_dirty_at = time.monotonic()


def _vector_presets(ctx: Any) -> None:
    """The settings vectors Review found and the user saved.

    A second picker rather than more entries in the shipped-preset combo: those
    are starting points somebody wrote, these are configurations the recorded
    verdicts say worked. Hidden entirely until one is saved, so a user who never
    reviews anything never sees an empty control.

    It fills *both* forms -- a vector carries the mesh-side settings too -- which
    is why it takes ctx rather than the 2D form alone.

    The last-applied name is remembered only so there is something for Forget
    to name. Presets could be saved (from Review) and applied but never
    removed, and nothing capped the list, so ``studio_settings.json`` grew
    monotonically with no way back.
    """
    saved = vector_presets.list_presets(ctx.settings)
    if not saved:
        return
    widgets.field_label("found settings")
    options = [("", "-")] + [(name, name) for name in sorted(saved)]
    chosen = widgets.combo("##vector-preset", "", options)
    if chosen and chosen in saved:
        vector_presets.apply(ctx.state, saved[chosen])
        ctx.state.preview_dirty_at = time.monotonic()
        ctx.state.preview["vector_preset"] = chosen
        ctx.toast(f"Applied {chosen} to the 2D and 3D forms.")
    last = ctx.state.preview.get("vector_preset")
    if last in saved:
        imgui.same_line()
        if widgets.disabled_button(f"Forget {last}", True):
            ctx.confirms.ask(
                dialogs.Confirm(
                    title="Forget this preset?",
                    message=f"{last} will be removed from your saved settings.",
                    on_confirm=lambda name=last: _forget_vector_preset(ctx, name),
                )
            )


def _forget_vector_preset(ctx: Any, name: str) -> None:
    vector_presets.delete_preset(ctx.settings, name)
    if ctx.state.preview.get("vector_preset") == name:
        ctx.state.preview.pop("vector_preset", None)
    ctx.toast(f"Forgot the preset {name}.")


def _output(ctx: Any, form: dict[str, Any]) -> None:
    """Object or tile -- the one thing that changes what this pane submits.

    A segmented control rather than a combo: there are exactly two, and the
    choice changes which guidance groups are on screen, so it has to read as a
    mode and not as one more select in a column of selects.
    """
    before = form.get("output", "reference")
    with focus.item(ctx.state, FOCUS_PANE, "output") as focused:
        form["output"] = widgets.segmented_control(
            "output",
            [("reference", "Object"), ("tile", "Seamless tile")],
            before,
        )
        # A hand-drawn control, so imgui's focus does nothing for it and the
        # keys are answered here. Left/Right rather than Enter, because it is a
        # switch between two states rather than a thing to press.
        if focused and imgui.is_key_pressed(imgui.Key.left_arrow):
            form["output"] = "reference"
        if focused and imgui.is_key_pressed(imgui.Key.right_arrow):
            form["output"] = "tile"
    if form["output"] != before:
        ctx.state.preview_dirty_at = time.monotonic()
    if form["output"] == "tile":
        widgets.muted(
            "A tile is drawn with wrapping convolutions, so its edges match "
            "when repeated. It has no subject, so the object taxonomy is "
            "hidden and it cannot be made into a mesh."
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
        widgets.field_label("profile")
        chosen = widgets.combo("##profile", current, options, width=-84)
        if chosen and chosen != current and chosen in saved:
            profiles.apply(form, saved[chosen])
            profiles.set_active(ctx.settings, chosen)
            ctx.state.preview_dirty_at = time.monotonic()
        imgui.same_line()
    if controls.button("Save as..."):
        ctx.prompts.ask(
            dialogs.Prompt(
                title="Save profile",
                label="Name",
                value=profiles.get_active(ctx.settings) or "",
                on_accept=lambda name: _save_profile(ctx, form, name),
            )
        )
    imgui.same_line()
    # The manager, from the picker it is about (the UI redesign, wave 3). It was a
    # top-level mode, which put a shelf of saved settings in the navigation
    # beside the six creative workspaces and made "manage my styles" somewhere
    # you travel to rather than something you do to the form in front of you.
    if widgets.ghost_button("Manage..."):
        from . import profiles_panel

        profiles_panel.open_sheet(ctx)
    imgui.same_line()
    if controls.button("Reset..."):
        ctx.confirms.ask(
            dialogs.Confirm(
                title="Reset the image settings?",
                message=(
                    "The prompt, the negative prompt, every guidance select, "
                    "the model, the LoRA, the reference and the run controls go "
                    "back to their defaults. Saved profiles and presets are "
                    "kept, and the 3D form is untouched."
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


def _preview(ctx: Any) -> None:
    """The composed prompt, recomputed off-thread after a typing pause."""
    state = ctx.state
    if state.preview_dirty_at and time.monotonic() - state.preview_dirty_at > PREVIEW_DEBOUNCE:
        raw = {k: v for k, v in state.form_2d.items() if v not in ("", None)}
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
            state.form_2d["prompt"],
            # Threaded rather than inferred: the output kind is not a guidance
            # field, so without it a tile would be previewed through the
            # single-centred-object framing its job will never use.
            tile=state.form_2d.get("output") == "tile",
        ):
            state.preview_dirty_at = 0.0
    preview = state.preview
    if not preview:
        return
    if imgui.tree_node("Prompt actually sent"):
        imgui.text_wrapped(preview.get("prompt") or "")
        # Advisory, never a refusal (P124): a deliberate clash is a legitimate
        # thing to ask a sampler for. What was wrong was composing it silently
        # -- ``art_style=snes`` contributes "vivid saturated colours" over a
        # brief that named black and silver and blue, and the only way to
        # notice was to read the sentence above and know which words were the
        # user's.
        for conflict in preview.get("conflicts") or []:
            widgets.text_colored(theme.WARN, conflict)
        tokens, chunks = preview.get("tokens"), preview.get("chunks")
        if tokens is not None:
            # Chunks, not a truncation warning: the composed prompt is split on
            # comma boundaries and each chunk encoded separately, so a long one
            # costs attention rather than being cut off.
            widgets.muted(f"{tokens} tokens - {chunks} chunk(s)")
        imgui.tree_pop()


def _reference(ctx: Any, form: dict[str, Any]) -> None:
    """Conditioning: an image to steer appearance and/or structure.

    Every control below the picker is hidden until there is a reference, and
    the Structure group is hidden again unless the chosen base can run a
    ControlNet. That is this pane's existing rule -- the same one that hides
    the LoRA strength slider without a LoRA: a control with nothing to act on
    is a control that cannot do anything.
    """
    if not widgets.header("Reference", default_open=False, persist_key="2d/reference"):
        return

    # The block is grouped so a dropped file can outline exactly what it landed
    # in (H70) -- this section is collapsed by default, so a drop used to be
    # accepted with nothing on screen moving.
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
            widgets.muted(
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

    widgets.section("Appearance")
    form["ip_adapter"] = widgets.combo(
        "##ip_adapter", form["ip_adapter"], _options(ctx, "ip_adapter")
    )
    if form["ip_adapter"]:
        changed, value = controls.slider_float(
            "Strength##ip", float(form["ip_scale"]), *_range(ctx, "ip_scale_range", 0.0, 1.5)
        )
        if changed:
            form["ip_scale"] = value

    widgets.section("Structure")
    note = structure_note(ctx, form)
    if note is not None:
        widgets.muted(note)
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


def _advanced(ctx: Any, form: dict[str, Any]) -> None:
    if not widgets.header("Advanced", default_open=False, persist_key=ADVANCED_KEY):
        return
    before = form["base_model"]
    form["base_model"] = widgets.labeled_combo("Model", form["base_model"], ctx.base_models)
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
        widgets.muted(note)
    hint = _findings_hint(ctx, "base_model", form["base_model"])
    if hint is not None:
        imgui.same_line()
        widgets.secondary(hint)
    no_lora = lora_note(ctx, form)
    if no_lora is not None:
        # Disabled rather than hidden, this pane's stated rule: the form holds
        # a style the user picked under another base, and hiding the control
        # would make that selection vanish with no explanation of why the
        # submit is now refused.
        imgui.begin_disabled()
    was_lora = form["style_lora"]
    form["style_lora"] = widgets.labeled_combo(
        "Style LoRA", form["style_lora"], lora_options(ctx, form)
    )
    widgets.field_error(ctx.state, "style_lora")
    if form["style_lora"] != was_lora:
        ctx.state.clear_field_error("style_lora")
    hint = _findings_hint(ctx, "style_lora", form["style_lora"])
    if hint is not None:
        imgui.same_line()
        widgets.secondary(hint)
    if form["style_lora"]:
        # Hidden without a LoRA rather than disabled: a weight slider with
        # nothing to weight is a control that cannot do anything.
        changed, value = controls.slider_float("Strength", form["lora_weight"], 0.0, 1.5)
        if changed:
            form["lora_weight"] = value
        # The only shipped sweep is lora-weight-v1, so this is the one
        # slider in the form with a findings bucket behind it -- the
        # feedback loop the sweep exists for. findings.hint absorbs the
        # float32 rounding a slider hands back (see its docstring).
        hint = _findings_hint(ctx, "lora_weight", form["lora_weight"])
        if hint is not None:
            imgui.same_line()
            widgets.secondary(hint)
    if no_lora is not None:
        imgui.end_disabled()
        widgets.muted(no_lora)
    else:
        # One sentence at a time: lora_note explains a control that cannot act,
        # lora_filter_note one acting on less than the whole list, and both
        # under a disabled combo would be one control saying two things.
        narrowed = lora_filter_note(ctx, form)
        if narrowed is not None:
            widgets.muted(narrowed)
    inert = negative_prompt_note(ctx, form)
    if inert is not None:
        # Disabled rather than hidden, and with the reason underneath: the
        # field holds text the user typed under another base, and hiding it
        # would make that text vanish without saying why.
        imgui.begin_disabled()
    # Above the box, not beside it: imgui draws a multiline's label to the
    # *right* of the field, and this one is -1 wide, so the word "Negative" was
    # clipped off the edge of the panel and the box was unlabelled.
    widgets.field_label("negative prompt")
    before = form["negative_prompt"]
    form["negative_prompt"] = widgets.multiline("##negative", before, 54, MAX_PROMPT)
    if form["negative_prompt"] != before:
        ctx.state.preview_dirty_at = time.monotonic()
    if inert is not None:
        imgui.end_disabled()
        widgets.muted(inert)


def _run_controls(ctx: Any, form: dict[str, Any], form_ui: forms.Form) -> None:
    """Count and seed, beside the button that uses them.

    These lived under Advanced, which meant "roll again" and "how many"
    required re-expanding a collapsed section every session while the submit
    footer talked about the count as if it were visible.
    """
    widgets.section("Run")
    with focus.item(ctx.state, FOCUS_PANE, "count") as focused:
        changed, picked = form_ui.segmented_choice(
            "count",
            "Tiles" if form.get("output") == "tile" else "References",
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
    count = int(form["count"])
    noun = "tile" if form.get("output") == "tile" else "reference"
    widgets.muted(
        f"{count} {noun}s - a few seconds each" if count > 1 else f"One {noun} - a few seconds"
    )
    busy = ctx.busy("submit")
    enabled = not problems and not busy
    with focus.item(ctx.state, FOCUS_PANE, "generate") as focused:
        pressed = widgets.primary_button("Generate", (-1, sp(34)), enabled=enabled)
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
    if not form["prompt"].strip():
        problems.append(widgets.Problem("A prompt is required.", "prompt"))
    if len(form["prompt"]) > MAX_PROMPT:
        problems.append(
            widgets.Problem(f"The prompt is over {MAX_PROMPT} characters.", "prompt")
        )
    if not 1 <= int(form["count"]) <= MAX_REFERENCE_COUNT:
        problems.append(
            widgets.Problem(
                f"References must be between 1 and {MAX_REFERENCE_COUNT}.", "count"
            )
        )
    # Both reachable from a restored form rather than from this frame's
    # controls, which is why they are checked here and not only where the
    # widgets are drawn: a persisted selection outlives the ref_path that
    # justified it (ref_path is VOLATILE), and the base model can be changed
    # under Advanced after a control was picked.
    if not form.get("ref_path") and (form.get("ip_adapter") or form.get("control")):
        problems.append(
            widgets.Problem("Conditioning needs a reference image.", "ref_path")
        )
    if form.get("control") and form["base_model"] not in modelslib.controlnet_bases():
        problems.append(
            widgets.Problem("Structure control needs a full-CFG model.", "base_model")
        )
    # Reachable the same way: a style picked under one base survives a change
    # of base under Advanced, and the service refuses the submit outright
    # rather than generating without it.
    if form.get("style_lora") and form["style_lora"] not in (
        modelslib.loras_by_base().get(form["base_model"]) or []
    ):
        problems.append(
            widgets.Problem(
                "The style LoRA is not fitted to this model's architecture.",
                "style_lora",
            )
        )
    if form.get("output") == "tile" and form["base_model"] not in modelslib.tile_bases():
        problems.append(
            widgets.Problem("Seamless tiles need an SDXL model.", "base_model")
        )
    return problems


def submit_kwargs(form: dict[str, Any]) -> dict[str, Any]:
    """The 2D form as create_job takes it.

    ``output`` is this pane's own switch and never "model": this pane is the
    first stage of a two-stage pipeline made visible, and going straight to a
    mesh from here would spend two minutes of GPU on an image nobody has
    approved. A tile has no second stage at all.
    """
    tile = form.get("output") == "tile"
    known = set(guidancelib.form_fields())
    if tile:
        # The hidden groups must not reach the submit either: a row claiming a
        # category the prompt compiler discarded is a lie about what produced
        # the image. base_model and style_lora are re-added because they are
        # model identity rather than subject taxonomy -- a tile still needs a
        # checkpoint, and the trigger word of a style LoRA is prepended to a
        # tile prompt exactly as it is to an object's.
        known &= set(prompt_lib.TILE_FIELDS) | {"base_model", "style_lora"}
    fields = {k: v for k, v in form.items() if k in known and v not in ("", None)}
    return {
        "kind": "text",
        "prompt": form["prompt"].strip(),
        "output": "tile" if tile else "reference",
        "count": int(form["count"]),
        "seed": int(form["seed"]),
        "negative_prompt": form["negative_prompt"] or None,
        "lora_weight": float(form["lora_weight"]) if form.get("style_lora") else None,
        # Mirroring lora_weight: sent only alongside the selection it scales,
        # so an unused slider never reaches params as a live setting.
        "ip_scale": float(form["ip_scale"]) if form.get("ip_adapter") else None,
        "control_scale": float(form["control_scale"]) if form.get("control") else None,
        "control_end": float(form["control_end"]) if form.get("control") else None,
        "guidance_fields": fields,
    }


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
