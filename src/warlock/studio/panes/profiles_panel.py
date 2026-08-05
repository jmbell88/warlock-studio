"""The profile manager: list, create, edit, delete, set active.

Draws the same controls the 2D pane's Advanced section and Guidance block do,
against a *draft* dict rather than against the live form -- editing a profile
must not change what the next Generate would send, and creating one from the
landing screen happens when there is no form on screen at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from imgui_bundle import imgui

from ...service import validation
from ...service.validation import MAX_UPLOAD_BYTES
from .. import dialogs, profiles, theme, widgets
from ..manual import render as manual_render
from . import settings_2d


def draw(ctx: Any) -> None:
    manual_render.help_button(ctx, "profiles")
    if ctx.state.profile_draft is not None:
        _editor(ctx)
        return
    _list(ctx)


# --- the list ---------------------------------------------------------------


def _list(ctx: Any) -> None:
    saved = profiles.list_profiles(ctx.settings)
    active = profiles.get_active(ctx.settings)
    if imgui.button("New profile"):
        _open_draft(ctx, "", profiles.capture(ctx.state.form_2d))
    if not saved:
        widgets.muted(
            "A profile remembers the model, the LoRA, the negative prompt and "
            "the core style choices under a name -- the prompt, the seed and "
            "the per-asset guidance stay per-generation."
        )
        return
    imgui.separator()
    for name in sorted(saved):
        imgui.push_id(name)
        if name == active:
            widgets.text_colored(theme.ACCENT, f"{name} (active)")
        else:
            imgui.text(name)
        widgets.muted(_summary(ctx, saved[name]))
        if name != active and imgui.small_button("Set active"):
            profiles.set_active(ctx.settings, name)
        if name != active:
            imgui.same_line()
        if imgui.small_button("Edit"):
            _open_draft(ctx, name, saved[name])
        imgui.same_line()
        if imgui.small_button("Apply to form"):
            profiles.apply(ctx.state.form_2d, saved[name])
            profiles.set_active(ctx.settings, name)
            ctx.toast(f"Applied {name}.")
        imgui.same_line()
        if imgui.small_button("Delete"):
            ctx.confirms.ask(
                dialogs.Confirm(
                    title="Delete this profile?",
                    message=f"{name} is removed. Nothing already generated changes.",
                    confirm_label="Delete",
                    cancel_label="Keep",
                    on_confirm=lambda n=name: profiles.delete_profile(
                        ctx.settings, n, ctx.svc.config
                    ),
                )
            )
        imgui.separator()
        imgui.pop_id()


def _summary(ctx: Any, fields: dict[str, Any]) -> str:
    """What the profile is, in the labels the pickers use rather than in keys."""
    parts = [
        _label(ctx.base_models, fields.get("base_model")),
        _label(ctx.style_loras, fields.get("style_lora")),
    ]
    chosen = [f for f in profiles.TAXONOMY if fields.get(f)]
    if chosen:
        parts.append(f"{len(chosen)} style field{'s' if len(chosen) > 1 else ''}")
    return " - ".join(p for p in parts if p) or "nothing set"


def _label(options: list[tuple[str, str]], key: Any) -> str:
    if not key:
        return ""
    return next((label for k, label in options if k == key), str(key))


# --- the editor -------------------------------------------------------------


def _open_draft(ctx: Any, name: str, fields: dict[str, Any]) -> None:
    # Started from the profile's own fields laid over a blank capture, so a
    # profile saved before a field existed still opens with every control.
    draft = profiles.capture(profiles.apply(_blank(), fields))
    ctx.state.profile_draft = draft
    ctx.state.profile_draft_name = name
    ctx.state.profile_draft_origin = name


def _blank() -> dict[str, Any]:
    from ..state import default_form_2d

    return default_form_2d()


def _editor(ctx: Any) -> None:
    draft = ctx.state.profile_draft
    name = ctx.state.profile_draft_name
    imgui.text("New profile" if not name else f"Editing {name}")
    ctx.state.profile_draft_name = widgets.input_text(
        "Name", ctx.state.profile_draft_name, max_length=60
    )

    widgets.section("Model")
    draft["base_model"] = widgets.combo("Model", draft.get("base_model", ""), ctx.base_models)
    draft["style_lora"] = widgets.combo(
        "Style LoRA", draft.get("style_lora", ""), ctx.style_loras
    )
    if draft["style_lora"]:
        changed, value = imgui.slider_float("Strength", float(draft["lora_weight"]), 0.0, 1.5)
        if changed:
            draft["lora_weight"] = value
    # The same cap the service enforces. Accepting twice as much here only
    # meant the refusal arrived at submit time, against a profile the user had
    # already saved.
    inert = settings_2d.negative_prompt_note(ctx, draft)
    if inert is not None:
        imgui.begin_disabled()
    draft["negative_prompt"] = widgets.multiline(
        "Negative", draft.get("negative_prompt", ""), 54, validation.MAX_PROMPT
    )
    if inert is not None:
        imgui.end_disabled()
        widgets.muted(inert)

    widgets.section("Style")
    for field in profiles.TAXONOMY:
        draft[field] = widgets.combo(
            f"##p_{field}", draft.get(field, ""), widgets.field_options(ctx, field)
        )
    draft["platform"] = widgets.combo(
        "##p_platform", draft.get("platform", ""), widgets.field_options(ctx, "platform")
    )

    _anchor(ctx, name)

    imgui.dummy((0, 8))
    imgui.separator()
    saveable = bool(ctx.state.profile_draft_name.strip())
    if widgets.disabled_button("Save", saveable, (150, 0)):
        _save(ctx)
    imgui.same_line()
    if imgui.button("Cancel", (150, 0)):
        _close(ctx)


def _anchor(ctx: Any, name: str) -> None:
    """The style anchor: one image every asset in this set is conditioned on.

    Only offered on a profile that has been saved once, because the anchor is
    stored against the name -- there is nowhere to put it while the editor is
    still holding an unnamed draft.
    """
    widgets.section("Style anchor")
    saved = profiles.list_profiles(ctx.settings)
    if name not in saved:
        widgets.muted("Save the profile once, then attach an anchor image to it.")
        return
    fields = saved[name]
    path = profiles.anchor_path(ctx.svc.config, fields)
    if path is not None:
        if ctx.textures is not None:
            texture = ctx.textures.get(f"anchor:{name}", path)
            if texture is not None:
                imgui.image(widgets.texture_ref(texture), (96, 96))
        changed, value = imgui.slider_float(
            "Strength##anchor", float(fields.get("anchor_scale") or 0.6), 0.0, 1.5
        )
        if changed:
            profiles.save_profile(
                ctx.settings, name, {**fields, "anchor_scale": float(value)}
            )
        if imgui.small_button("Remove anchor"):
            profiles.clear_anchor(ctx.settings, ctx.svc.config, name)
        imgui.same_line()
    busy = ctx.busy("anchor-pick")
    if widgets.disabled_button(
        "Choose an image..." if path is None else "Replace...", not busy
    ):
        ctx.submit("anchor-pick", _pick_anchor, ctx, name)
    if path is None:
        widgets.muted(
            "Every generation under this profile is conditioned on the anchor, "
            "which is what keeps a set of assets looking like one set."
        )


def _pick_anchor(ctx: Any, name: str) -> None:
    """Runs on a task thread: both the dialog and the read block."""
    chosen = dialogs.open_file("Choose a style anchor", dialogs.IMAGE_FILTER)
    if chosen is None:
        return
    with Path(chosen).open("rb") as fh:
        data = fh.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        ctx.toast("That image is over 20 MB.", "error")
        return
    profiles.set_anchor(ctx.settings, ctx.svc.config, name, data)
    ctx.toast(f"Anchor set for {name}.")


def _save(ctx: Any) -> None:
    name = ctx.state.profile_draft_name.strip()
    origin = ctx.state.profile_draft_origin
    # capture(), not the raw draft: the draft is a whole blank form with the
    # profile laid over it, and saving it wholesale would store every field the
    # profile is not supposed to carry.
    fields = profiles.capture(ctx.state.profile_draft)
    if origin and origin != name:
        # A rename moves the anchor with the profile: save_profile preserves
        # anchor fields under the *same* name, and this is the one path where
        # the name changes underneath them.
        carried = profiles.list_profiles(ctx.settings).get(origin) or {}
        fields.update({k: carried[k] for k in profiles.ANCHOR_FIELDS if k in carried})
    profiles.save_profile(ctx.settings, name, fields)
    if origin and origin != name:
        # A rename moves the profile rather than forking it: the editor was
        # opened on one entry, and leaving the old name behind would make a
        # typo correction look like it silently duplicated everything.
        profiles.delete_profile(ctx.settings, origin, ctx.svc.config)
    profiles.set_active(ctx.settings, name)
    ctx.toast(f"Saved the profile {name}.")
    _close(ctx)


def _close(ctx: Any) -> None:
    ctx.state.profile_draft = None
    ctx.state.profile_draft_name = ""
    ctx.state.profile_draft_origin = ""
