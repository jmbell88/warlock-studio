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
from typing import Any

from imgui_bundle import imgui

from ... import guidance as guidancelib
from ...service import jobs as svc_jobs
from ...service import system as svc_system
from ...service.validation import MAX_PROMPT, MAX_REFERENCE_COUNT, random_seed
from .. import theme, widgets

PREVIEW_DEBOUNCE = 0.3


def _options(ctx: Any, field: str) -> list[tuple[str, str]]:
    """(key, label) pairs for one taxonomy field, with a blank first entry.

    Blank because every guidance field is optional: an empty select means "say
    nothing about this", which is a different prompt from any of the choices.
    """
    entries = (ctx.guidance.get("fields") or {}).get(field) or []
    return [("", f"{field.replace('_', ' ')}...")] + [
        (e["key"], e["label"]) for e in entries
    ]


def draw(ctx: Any) -> None:
    state = ctx.state
    form = state.form_2d

    _presets(ctx, form)
    widgets.section("Prompt")
    _prompt(ctx, form)
    _history(ctx, form)
    _preview(ctx)

    widgets.section("Guidance")
    for field in ("category", "genre", "setting", "art_style"):
        form[field] = widgets.combo(f"##{field}", form[field], _options(ctx, field))
    for field in ("material", "condition", "palette", "mood", "silhouette", "emissive", "rarity"):
        form[field] = widgets.combo(f"##{field}", form[field], _options(ctx, field))
    # Narrowed to leave room for the marker: a full-width combo pushes it off
    # the panel, and the note is the whole reason this control is not the 3D
    # pane's platform.
    form["platform"] = widgets.combo(
        "##g_platform", form["platform"], _options(ctx, "platform"), width=-30
    )
    widgets.help_marker(
        "A prompt hint about how much detail to draw. The mesh resolution is "
        "the 3D pane's own platform control."
    )

    _advanced(ctx, form)
    _submit(ctx, form)


# --- pieces -----------------------------------------------------------------


def _presets(ctx: Any, form: dict[str, Any]) -> None:
    presets = ctx.guidance.get("presets") or []
    if not presets:
        return
    options = [("", "preset...")] + [(p["key"], p["label"]) for p in presets]
    chosen = widgets.combo("##preset", "", options)
    if not chosen:
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
        state.preview_dirty_at = 0.0
        raw = {k: v for k, v in state.form_2d.items() if v not in ("", None)}
        ctx.submit(
            "preview",
            svc_system.prompt_preview,
            ctx.svc,
            {**raw, "prompt": None},
            state.form_2d["prompt"],
        )
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


def _advanced(ctx: Any, form: dict[str, Any]) -> None:
    if not widgets.header("Advanced", default_open=False):
        return
    form["base_model"] = widgets.combo("Model", form["base_model"], ctx.base_models)
    form["style_lora"] = widgets.combo("Style LoRA", form["style_lora"], ctx.style_loras)
    if form["style_lora"]:
        # Hidden without a LoRA rather than disabled: a weight slider with
        # nothing to weight is a control that cannot do anything.
        changed, value = imgui.slider_float("Strength", form["lora_weight"], 0.0, 1.5)
        if changed:
            form["lora_weight"] = value
    before = form["negative_prompt"]
    form["negative_prompt"] = widgets.multiline("Negative", before, 54, MAX_PROMPT)
    if form["negative_prompt"] != before:
        ctx.state.preview_dirty_at = time.monotonic()

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

    imgui.text("References")
    for count in (1, 2, 4, 8):
        imgui.same_line()
        if imgui.radio_button(f"{count}##count", form["count"] == count):
            form["count"] = count


def _submit(ctx: Any, form: dict[str, Any]) -> None:
    imgui.dummy((0, 8))
    imgui.separator()
    problems = validate(form)
    for problem in problems:
        widgets.text_colored(theme.ERR, problem)
    count = int(form["count"])
    widgets.muted(
        f"{count} reference{'s' if count > 1 else ''} - a few seconds each"
        if count > 1
        else "One reference - a few seconds"
    )
    busy = ctx.busy("submit")
    if widgets.disabled_button("Generate", not problems and not busy, (-1, 34)):
        generate(ctx, form)


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
    return problems


def submit_kwargs(form: dict[str, Any]) -> dict[str, Any]:
    """The 2D form as create_job takes it.

    Always ``output="reference"``: this pane is the first stage of a two-stage
    pipeline made visible, and going straight to a mesh from here would spend
    two minutes of GPU on an image nobody has approved.
    """
    known = set(guidancelib.form_fields())
    fields = {k: v for k, v in form.items() if k in known and v not in ("", None)}
    return {
        "kind": "text",
        "prompt": form["prompt"].strip(),
        "output": "reference",
        "count": int(form["count"]),
        "seed": int(form["seed"]),
        "negative_prompt": form["negative_prompt"] or None,
        "lora_weight": float(form["lora_weight"]) if form.get("style_lora") else None,
        "guidance_fields": fields,
    }


def generate(ctx: Any, form: dict[str, Any]) -> None:
    if validate(form):
        return
    if not form["seed_locked"]:
        # Generation is deterministic in the seed, so an unchanged form would
        # otherwise produce the identical image twice and read as a no-op.
        form["seed"] = random_seed()
    ctx.state.remember_prompt(form["prompt"])
    ctx.submit("submit", svc_jobs.create_job, ctx.svc, **submit_kwargs(form))
