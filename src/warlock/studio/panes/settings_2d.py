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
from .. import dialogs, profiles, theme, widgets
from ..manual import render as manual_render
from ..widgets import field_options as _options

PREVIEW_DEBOUNCE = 0.3


def draw(ctx: Any) -> None:
    state = ctx.state
    form = state.form_2d

    _presets(ctx, form)
    _profiles(ctx, form)
    _output(ctx, form)
    widgets.section("Prompt")
    manual_render.help_button(ctx, "settings-2d")
    _prompt(ctx, form)
    _history(ctx, form)
    _preview(ctx)

    _guidance(ctx, form)
    _reference(ctx, form)
    _advanced(ctx, form)
    _run_controls(ctx, form)
    _submit(ctx, form)


# The eleven optional taxonomies, grouped by what they describe. Grouping and
# per-field labels are the fix for the old column of identical unlabelled
# combos, where a chosen value ("worn", "brass") no longer said which question
# it answered.
GUIDANCE_GROUPS = (
    ("Subject", ("category", "genre", "setting", "silhouette")),
    ("Style", ("art_style", "palette", "mood", "rarity")),
    ("Surface", ("material", "condition", "emissive")),
)


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


def _guidance(ctx: Any, form: dict[str, Any]) -> None:
    for title, fields in guidance_groups(form):
        widgets.section(title)
        if imgui.begin_table(f"guidance/{title}", 2):
            for field in fields:
                imgui.table_next_column()
                widgets.field_label(field.replace("_", " "))
                imgui.set_next_item_width(-1)
                form[field] = widgets.combo(f"##{field}", form[field], _options(ctx, field), 0)
            imgui.end_table()
    if form.get("output") == "tile":
        # platform is a hint about how much detail to draw an *object* with,
        # and is not in TILE_FIELDS -- so on a tile it is a control whose value
        # the prompt compiler discards.
        return
    # Narrowed to leave room for the marker: a full-width combo pushes it off
    # the panel, and the note is the whole reason this control is not the 3D
    # pane's platform.
    widgets.field_label("platform detail")
    form["platform"] = widgets.combo(
        "##g_platform", form["platform"], _options(ctx, "platform"), width=-30
    )
    widgets.help_marker(
        "A prompt hint about how much detail to draw. The mesh resolution is "
        "the 3D pane's own platform control."
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


def _output(ctx: Any, form: dict[str, Any]) -> None:
    """Object or tile -- the one thing that changes what this pane submits.

    A segmented control rather than a combo: there are exactly two, and the
    choice changes which guidance groups are on screen, so it has to read as a
    mode and not as one more select in a column of selects.
    """
    before = form.get("output", "reference")
    form["output"] = widgets.segmented_control(
        "output",
        [("reference", "Object"), ("tile", "Seamless tile")],
        before,
    )
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
    if imgui.button("Save as..."):
        ctx.prompts.ask(
            dialogs.Prompt(
                title="Save profile",
                label="Name",
                value=profiles.get_active(ctx.settings) or "",
                on_accept=lambda name: _save_profile(ctx, form, name),
            )
        )


def _save_profile(ctx: Any, form: dict[str, Any], name: str) -> None:
    profiles.save_profile(ctx.settings, name, profiles.capture(form))
    profiles.set_active(ctx.settings, name)
    ctx.toast(f"Saved the profile {name}.")


def _prompt(ctx: Any, form: dict[str, Any]) -> None:
    before = form["prompt"]
    form["prompt"] = widgets.multiline("##prompt", before, 90, MAX_PROMPT)
    if form["prompt"] != before:
        ctx.state.preview_dirty_at = time.monotonic()
    remaining = MAX_PROMPT - len(form["prompt"])
    widgets.text_colored(
        theme.WARN if remaining < 100 else theme.MUTED,
        f"{len(form['prompt'])}/{MAX_PROMPT}",
    )


def _history(ctx: Any, form: dict[str, Any]) -> None:
    if not ctx.state.history:
        return
    imgui.same_line()
    if imgui.button("Recent"):
        imgui.open_popup("prompt-history")
    if imgui.begin_popup("prompt-history"):
        for entry in ctx.state.history:
            label = entry if len(entry) <= 60 else entry[:57] + "..."
            if imgui.menu_item(f"{label}##{hash(entry)}", "", False)[0]:
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
        if imgui.button("Clear##ref"):
            form["ref_path"] = ""
            # The selections go with it: they cannot be submitted without an
            # image, and leaving them set would disable Generate with a
            # message about a picker the user just emptied.
            form["ip_adapter"] = ""
            form["control"] = ""
            return
        imgui.same_line()
    busy = ctx.busy("ref-upload")
    if widgets.disabled_button("Choose an image..." if not path else "Replace...", not busy):
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
        changed, value = imgui.slider_float(
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
        changed, value = imgui.slider_float(
            "Strength##cn",
            float(form["control_scale"]),
            *_range(ctx, "control_scale_range", 0.0, 2.0),
        )
        if changed:
            form["control_scale"] = value
        changed, value = imgui.slider_float(
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


def structure_note(ctx: Any, form: dict[str, Any]) -> str | None:
    """Which bases could run the ControlNet this one cannot, or None."""
    bases = ctx.guidance.get("controlnet_bases") or []
    if (form.get("base_model") or "") in bases:
        return None
    return (
        "Structure control needs a full-CFG model -- pick one of "
        f"{_base_labels(ctx, bases)} under Advanced."
    )


def _advanced(ctx: Any, form: dict[str, Any]) -> None:
    if not widgets.header("Advanced", default_open=False, persist_key="2d/advanced"):
        return
    form["base_model"] = widgets.combo("Model", form["base_model"], ctx.base_models)
    form["style_lora"] = widgets.combo("Style LoRA", form["style_lora"], ctx.style_loras)
    if form["style_lora"]:
        # Hidden without a LoRA rather than disabled: a weight slider with
        # nothing to weight is a control that cannot do anything.
        changed, value = imgui.slider_float("Strength", form["lora_weight"], 0.0, 1.5)
        if changed:
            form["lora_weight"] = value
    inert = negative_prompt_note(ctx, form)
    if inert is not None:
        # Disabled rather than hidden, and with the reason underneath: the
        # field holds text the user typed under another base, and hiding it
        # would make that text vanish without saying why.
        imgui.begin_disabled()
    before = form["negative_prompt"]
    form["negative_prompt"] = widgets.multiline("Negative", before, 54, MAX_PROMPT)
    if form["negative_prompt"] != before:
        ctx.state.preview_dirty_at = time.monotonic()
    if inert is not None:
        imgui.end_disabled()
        widgets.muted(inert)


def _run_controls(ctx: Any, form: dict[str, Any]) -> None:
    """Count and seed, beside the button that uses them.

    These lived under Advanced, which meant "roll again" and "how many"
    required re-expanding a collapsed section every session while the submit
    footer talked about the count as if it were visible.
    """
    widgets.section("Run")
    imgui.text("Tiles" if form.get("output") == "tile" else "References")
    for count in (1, 2, 4, 8):
        imgui.same_line()
        if imgui.radio_button(f"{count}##count", form["count"] == count):
            form["count"] = count

    imgui.set_next_item_width(120)
    changed, seed = imgui.input_int("Seed", int(form["seed"]), 0, 0)
    if changed:
        form["seed"] = max(0, seed)
    imgui.same_line()
    if imgui.button("Reroll"):
        form["seed"] = random_seed()
    imgui.same_line()
    changed, locked = imgui.checkbox("Lock", bool(form["seed_locked"]))
    if changed:
        form["seed_locked"] = locked
    widgets.help_marker(
        "Locked, Generate reuses this seed so an unchanged form reproduces "
        "exactly. Unlocked, every submit rerolls it."
    )


def _submit(ctx: Any, form: dict[str, Any]) -> None:
    imgui.dummy((0, 8))
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
    if widgets.primary_button("Generate", (-1, 34), enabled=not problems and not busy):
        generate(ctx, form)
    if imgui.is_item_hovered(imgui.HoveredFlags_.allow_when_disabled.value):
        imgui.set_tooltip("Ctrl+Enter")


def validate(form: dict[str, Any]) -> list[str]:
    """What would be refused, said before the button is pressed.

    A summary rather than a refusal on submit: the API checks all of this too,
    but a disabled button with a reason beats a toast after a round trip.
    """
    problems: list[str] = []
    if not form["prompt"].strip():
        problems.append("A prompt is required.")
    if len(form["prompt"]) > MAX_PROMPT:
        problems.append(f"The prompt is over {MAX_PROMPT} characters.")
    if not 1 <= int(form["count"]) <= MAX_REFERENCE_COUNT:
        problems.append(f"References must be between 1 and {MAX_REFERENCE_COUNT}.")
    # Both reachable from a restored form rather than from this frame's
    # controls, which is why they are checked here and not only where the
    # widgets are drawn: a persisted selection outlives the ref_path that
    # justified it (ref_path is VOLATILE), and the base model can be changed
    # under Advanced after a control was picked.
    if not form.get("ref_path") and (form.get("ip_adapter") or form.get("control")):
        problems.append("Conditioning needs a reference image.")
    if form.get("control") and form["base_model"] not in modelslib.controlnet_bases():
        problems.append("Structure control needs a full-CFG model.")
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


def generate(ctx: Any, form: dict[str, Any]) -> None:
    if validate(form):
        return
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
                raise Invalid(f"could not read {Path(ref_path).name}: {exc}") from exc
        return svc_jobs.create_job(ctx.svc, **kwargs)

    ctx.submit("submit", run)
