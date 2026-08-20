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
from .. import controls, create_stages, dialogs, forms, icons, journal, profiles, theme, widgets
from ..manual import render as manual_render
from ..tokens import sp
from . import settings_2d

# How wide the sheet is. Narrow on purpose: the panel is a list of names with
# a summary line each, and an editor of single-column fields -- a manager
# stretched across a 1600 px window would be one column of text and a thousand
# pixels of nothing.
SHEET_W = 560.0
SHEET_HEIGHT = 0.7

# Whether the sheet was up last frame, so ``popover_enter`` can be told which
# frame is its first. Module state for ``manual.render``'s reason: there is one
# manager and it is drawn from one place.
_was_open = [False]


def open_sheet(ctx: Any) -> None:
    ctx.state.profiles_open = True


def close_sheet(ctx: Any) -> None:
    """Put the manager away, asking first if a draft would be lost.

    Through :func:`guard` rather than by clearing the flag, which is the whole
    reason closing is a function: the sheet can be dismissed by Esc or by a
    close button, and both of those are ways to lose typing that the mode this
    replaced could only lose by switching away -- where the same guard already
    stood.
    """

    def proceed() -> None:
        _close(ctx)
        ctx.state.profiles_open = False

    guard(ctx, "close the profile manager", proceed)


def draw_sheet(ctx: Any) -> None:
    """The manager over whatever is on screen. Drawn from ``App._overlays``.

    The Manual overlay's recipe exactly (see ``manual.render.draw_overlay``):
    a plain frosted window rather than a modal, so it neither takes the one
    popup slot a frame has -- this panel raises confirms of its own, for Delete
    and for the dirty-draft guard -- nor dims the pane it is about.
    """
    if not ctx.state.profiles_open:
        _was_open[0] = False
        return
    appearing = not _was_open[0]
    _was_open[0] = True

    viewport = imgui.get_main_viewport()
    alpha, rise = widgets.popover_enter("profiles", appearing)
    width = min(viewport.work_size.x - sp(80), sp(SHEET_W))
    imgui.set_next_window_pos(
        (
            viewport.work_pos.x + viewport.work_size.x * 0.5,
            viewport.work_pos.y + viewport.work_size.y * 0.5 + rise,
        ),
        imgui.Cond_.always.value,
        (0.5, 0.5),
    )
    imgui.set_next_window_size((width, viewport.work_size.y * SHEET_HEIGHT))
    frosted = widgets.frosted()
    if frosted:
        imgui.set_next_window_bg_alpha(0.0)
    imgui.push_style_var(imgui.StyleVar_.alpha.value, alpha)
    radius = widgets.push_surface_rounding()
    # ``begin`` returns a (visible, open) tuple in imgui_bundle, and a tuple is
    # always truthy -- the guard needs the first element or it never skips.
    # ``end`` stays unconditional either way; that is begin/end's contract.
    opened = imgui.begin(
        "##profiles-sheet",
        None,
        imgui.WindowFlags_.no_title_bar.value
        | imgui.WindowFlags_.no_move.value
        | imgui.WindowFlags_.no_resize.value
        | imgui.WindowFlags_.no_collapse.value
        | imgui.WindowFlags_.no_saved_settings.value,
    )[0]
    widgets.pop_surface_rounding()
    if opened:
        widgets.window_shadow("overlay", radius=radius)
        if frosted:
            widgets.window_backdrop(radius=radius)
        close_w = imgui.get_frame_height()
        imgui.set_cursor_pos_x(
            max(imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x - close_w, 0.0)
        )
        if widgets.icon_button(
            f"{icons.CIRCLE_X}##profiles-close", "Close (Esc)", borderless=True
        ):
            close_sheet(ctx)
        draw(ctx)
    imgui.end()
    imgui.pop_style_var()


def draw(ctx: Any) -> None:
    # The heading comes first, and that is not cosmetic. ``help_button`` is a
    # ``same_line``, and ``same_line`` returns to the *previous row*
    # unconditionally -- with nothing drawn before it, the (?) attached to
    # whatever the last pane had left on screen, which here is the global mode
    # switch, and overlapped the health dot on it. ``app_settings._interface``
    # carries the same comment for the same reason; this pane called it first
    # in ``draw`` and got exactly the defect that one describes (UX-15).
    widgets.section("Style profiles")
    manual_render.help_button(ctx, "profiles")
    if ctx.state.profile_draft is not None:
        _editor(ctx)
        return
    _list(ctx)


# --- the list ---------------------------------------------------------------


def _list(ctx: Any) -> None:
    saved = profiles.list_profiles(ctx.settings)
    active = profiles.get_active(ctx.settings)
    # *After* the button, not before it. ``help_marker`` is a right-aligned
    # ``same_line``, so called first in this function it had no control to
    # attach to and drew a (?) alone on an empty row between the heading and
    # the button -- the defect ``app_settings._categories`` and
    # ``stage_rig.draw`` both carry a comment about, found here by the wave 6
    # screenshot pass. On the button it also reads better: it says what the
    # thing you are about to make remembers.
    if controls.button("New profile"):
        _open_draft(ctx, "", profiles.capture(ctx.state.form_2d))
    widgets.help_marker(
        "A profile remembers the model, the LoRA, the negative prompt and the "
        "core style choices under a name. The prompt, the seed and the "
        "per-asset guidance are never part of one."
    )
    if not saved:
        # H73: the same sentence, but as the empty state the rest of the app
        # uses -- an icon and a title say "there is nothing here yet" before
        # the paragraph explaining what would be.
        widgets.empty_state(
            icons.PALETTE,
            "No profiles yet",
            "A profile remembers the model, the LoRA, the negative prompt and "
            "the core style choices under a name -- the prompt, the seed and "
            "the per-asset guidance stay per-generation.",
        )
        return
    imgui.separator()
    # J86. A style library grows one profile at a time and is never pruned, so
    # it is the panel list most likely to outgrow a scroll.
    needle = widgets.list_filter(ctx, "profiles", len(saved))
    shown = 0
    for name in sorted(saved):
        if needle and needle not in name.lower():
            continue
        shown += 1
        imgui.push_id(name)
        if name == active:
            widgets.text_colored(theme.ACCENT, f"{name} (active)")
        else:
            imgui.text(name)
        widgets.muted(_summary(ctx, saved[name]))
        if name != active and controls.small_button("Set active"):
            profiles.set_active(ctx.settings, name)
        if name != active:
            imgui.same_line()
        if controls.small_button("Edit"):
            _open_draft(ctx, name, saved[name])
        imgui.same_line()
        if controls.small_button("Apply to form"):
            profiles.apply(ctx.state.form_2d, saved[name])
            profiles.set_active(ctx.settings, name)
            ctx.toast(f"Applied {name}.")
        imgui.same_line()
        if controls.small_button("Delete"):
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
    widgets.no_matches(needle, shown)


def _summary(ctx: Any, fields: dict[str, Any]) -> str:
    """What the profile is, in the labels the pickers use rather than in keys."""
    parts = [
        _label(ctx.base_models, fields.get("base_model")),
        _label(ctx.style_loras, fields.get("style_lora")),
    ]
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
    with forms.Form("profile-editor") as form_ui:
        def negative_prompt(*, enabled: bool, reason: str) -> tuple[bool, str]:
            return form_ui.multiline(
                "negative_prompt",
                "Negative prompt",
                ctx.state.profile_draft.get("negative_prompt", ""),
                validation.MAX_PROMPT,
                height=54,
                enabled=enabled,
                reason=reason,
            )

        _editor_form(ctx, form_ui, negative_prompt)


def _editor_form(ctx: Any, form_ui: forms.Form, negative_prompt: Any) -> None:
    draft = ctx.state.profile_draft
    name = ctx.state.profile_draft_name
    imgui.text("New profile" if not name else f"Editing {name}")
    _changed, ctx.state.profile_draft_name = form_ui.text(
        "name", "Name", ctx.state.profile_draft_name, max_length=60
    )

    widgets.section("Model")
    _changed, draft["base_model"] = form_ui.combo(
        "base_model", "Model", draft.get("base_model", ""), ctx.base_models
    )
    # The same pairing question the generate pane asks, and the same answer:
    # only the adapters fitted to the drafted base, plus a stale selection kept
    # visible and marked so it cannot become the value the user cannot see.
    # The *summary* above deliberately still reads the full ctx.style_loras --
    # it must be able to name a selection the draft's base does not take.
    _changed, draft["style_lora"] = form_ui.combo(
        "style_lora",
        "Style LoRA",
        draft.get("style_lora", ""),
        settings_2d.lora_options(ctx, draft),
    )
    if draft["style_lora"]:
        changed, value = form_ui.slider(
            "lora_weight", "Strength", float(draft["lora_weight"]), 0.0, 1.5
        )
        if changed:
            draft["lora_weight"] = value
    # The same cap the service enforces. Accepting twice as much here only
    # meant the refusal arrived at submit time, against a profile the user had
    # already saved.
    inert = settings_2d.negative_prompt_note(ctx, draft)
    if inert is not None:
        imgui.begin_disabled()
    _changed, draft["negative_prompt"] = negative_prompt(
        enabled=inert is None,
        reason=inert or "",
    )
    if inert is not None:
        imgui.end_disabled()
        widgets.muted(inert)

    _anchor(ctx, name)

    saveable = bool(ctx.state.profile_draft_name.strip())
    # The former imgui.button("Cancel" route is the cancel slot below; it still
    # goes through the same dirty-draft guard, now with the shared footer owning
    # placement and visual role.
    form_ui.footer(
        ("Save", lambda: _save(ctx)),
        cancel=(
            "Cancel",
            lambda: guard(ctx, "cancel", lambda: _close(ctx)),
        ),
        enabled=saveable,
        reason="Enter a profile name before saving.",
    )


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
                imgui.image(widgets.texture_ref(texture), (sp(96), sp(96)))
        changed, value = controls.slider_float(
            "Strength##anchor", float(fields.get("anchor_scale") or 0.6), 0.0, 1.5
        )
        if changed:
            profiles.save_profile(
                ctx.settings, name, {**fields, "anchor_scale": float(value)}
            )
        if controls.small_button("Remove anchor"):
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


def _pick_anchor(ctx: Any, name: str) -> dict[str, Any] | None:
    """Runs on a task thread: both the dialog and the read block.

    Only those two. The settings write and the toast are handed back for
    ``adopt_anchor`` to do on the frame thread -- ``Settings`` is frame-thread
    state and a toast is UI, and this was the one place doing both from a
    worker.
    """
    chosen = dialogs.open_file("Choose a style anchor", dialogs.IMAGE_FILTER)
    if chosen is None:
        return None
    with Path(chosen).open("rb") as fh:
        data = fh.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        return {"name": name, "too_big": True}
    return {"name": name, "png": data}


def adopt_anchor(ctx: Any, result: Any) -> None:
    """The frame-thread half of ``_pick_anchor``. Called from the task pump."""
    if not isinstance(result, dict):
        return
    if result.get("too_big"):
        ctx.toast("That image is over 20 MB.", "error")
        return
    name = str(result.get("name") or "")
    png = result.get("png")
    if not name or not png:
        return
    profiles.set_anchor(ctx.settings, ctx.svc.config, name, png)
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


def is_dirty(ctx: Any) -> bool:
    """Whether the open profile draft differs from what it was opened on.

    A draft is a document like any other -- nine fields and an anchor image --
    and it was the only one no guard knew about: Cancel discarded it outright
    and Quit did not ask (UX-17). Compared against the origin rather than
    tracked with a flag, because the editor writes every field on every frame
    and a flag would be true the moment the pane was opened.
    """
    # ``getattr`` rather than attribute access, which is ``docmodes.guard``'s
    # own rule and for its reason: asking whether there is unsaved work must
    # not require the state that says there is none to exist yet. The quit
    # chain runs against panes that have never been opened.
    draft = getattr(ctx.state, "profile_draft", None)
    if draft is None:
        return False
    origin = getattr(ctx.state, "profile_draft_origin", "")
    if not origin:
        # A new profile: dirty as soon as it has a name or any field set.
        name = getattr(ctx.state, "profile_draft_name", "")
        return bool(name.strip()) or any(draft.values())
    saved = profiles.list_profiles(ctx.settings).get(origin) or {}
    if getattr(ctx.state, "profile_draft_name", "").strip() != origin:
        return True
    return any(draft.get(k, "") != saved.get(k, "") for k in set(draft) | set(saved))


def guard(ctx: Any, action: str, proceed: Any) -> bool:
    """Ask before losing an unsaved profile draft. -> whether it went ahead now.

    The same shape as every other document mode's guard, so the quit chain can
    walk it beside them.
    """
    from .. import dialogs

    if not is_dirty(ctx):
        proceed()
        return True
    ctx.confirms.ask(
        dialogs.Confirm(
            title="Discard unsaved work?",
            message=(
                f"This style profile has unsaved changes, which will be lost if "
                f"you {action}."
            ),
            confirm_label="Discard",
            cancel_label="Keep editing",
            on_confirm=proceed,
        )
    )
    return False


def _close(ctx: Any) -> None:
    ctx.state.profile_draft = None
    ctx.state.profile_draft_name = ""
    ctx.state.profile_draft_origin = ""


# --- crash recovery (UX-05) ---------------------------------------------------
#
# A profile draft is the smallest journalled thing in the app and the one most
# clearly *worth* journalling: it is a form somebody has filled in and not
# pressed Save on, held nowhere but ``AppState``, and losing it costs exactly
# the typing. There is one at a time by construction, which is why the slot's
# uid is a constant.
#
# Payload equality is the head, for the pose provider's reason: a dict of a
# dozen strings is cheaper to compare than to re-encode.


class _DraftSlot:
    """The open draft as the journal sees it, with its marks on ``AppState``.

    The marks live on the state rather than on the draft dict because the dict
    is replaced wholesale every time a different profile is opened, and a mark
    that went with it would mint a new filename per open.
    """

    def __init__(self, state: Any) -> None:
        self.state = state

    @property
    def journal_name(self) -> str:
        return getattr(self.state, "profile_journal_name", "") or ""

    @journal_name.setter
    def journal_name(self, value: str) -> None:
        self.state.profile_journal_name = value

    @property
    def journal_head(self) -> Any:
        return getattr(self.state, "profile_journal_head", None)

    @journal_head.setter
    def journal_head(self, value: Any) -> None:
        self.state.profile_journal_head = value

    @property
    def journal_at(self) -> float:
        return float(getattr(self.state, "profile_journal_at", 0.0) or 0.0)

    @journal_at.setter
    def journal_at(self, value: float) -> None:
        self.state.profile_journal_at = value


def _draft_payload(slot: Any) -> bytes:
    import json as _json

    state = slot.state
    return _json.dumps(
        {
            "name": state.profile_draft_name,
            "origin": state.profile_draft_origin,
            "fields": state.profile_draft or {},
        },
        sort_keys=True,
    ).encode("utf-8")


def _journal_slots(ctx: Any) -> list[Any]:
    """The open draft, if there is one and it differs from what it came from.

    An untouched draft is not unsaved work: opening a saved profile to look at
    it and closing the panel must not leave a crash copy that gets offered back
    as though something had been typed.
    """
    state = ctx.state
    draft = getattr(state, "profile_draft", None)
    if draft is None:
        return []
    origin = getattr(state, "profile_draft_origin", "")
    from .. import profiles

    # ``profiles.list_profiles``, not ``settings.get("profiles")``. Profiles are
    # stored under ``profiles.KEY``, which is ``"user_profiles"`` -- named that
    # deliberately, per that module's own docstring, "because the app already
    # has another thing by that name". Nothing writes ``"profiles"``, so this
    # read was always ``{}``: ``origin in saved`` was never true and the guard
    # this function's docstring describes never ran. Opening a saved profile to
    # look at it and closing the panel left a crash copy, and ``journal.py`` has
    # no age-out, so the next launch offered to recover a draft nobody edited.
    # ``is_dirty``, two functions up, had it right all along.
    saved = profiles.list_profiles(ctx.settings) if hasattr(ctx, "settings") else {}
    if (
        origin
        and origin in saved
        and profiles.capture(profiles.apply(dict(draft), saved[origin])) == draft
    ):
        # Byte-identical to the profile it was opened from: nothing typed.
        return []
    return [_DraftSlot(state)]


def _journal_adopt(ctx: Any, path: Path, meta: dict[str, Any]) -> bool:
    """Reopen a recovered draft into the profiles panel.

    It replaces whatever is open, and that is safe in a way the document modes
    are not: this only ever runs at startup, before anything can have been
    typed, and a draft is not a file on disk that could be overwritten.
    """
    import json as _json

    try:
        data = _json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    ctx.state.profile_draft = dict(data.get("fields") or {})
    ctx.state.profile_draft_name = str(data.get("name") or "")
    ctx.state.profile_draft_origin = str(data.get("origin") or "")
    ctx.state.profile_journal_name = Path(path).name
    # The Reference stage with the manager over it, which is where the draft
    # was being typed: the sheet is not a destination, so "put the reader back"
    # means putting back both halves.
    create_stages.go(ctx, "reference")
    ctx.state.profiles_open = True
    return True


JOURNAL = journal.register(
    journal.Provider(
        kind="profile",
        ext=".profile.json",
        label="profile draft",
        slots=_journal_slots,
        # One at a time by construction: the panel holds a single draft.
        uid_of=lambda slot: "draft",
        title_of=lambda slot: slot.state.profile_draft_name or "Untitled profile",
        head_of=_draft_payload,
        encode=_draft_payload,
        adopt=_journal_adopt,
    )
)
