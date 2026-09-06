"""Create's command bar: the whole brief, across the top of the Reference stage.

**What a press needs, on one row, never scrolled.** The four controls here --
what to make, the words, how many, and the button -- were the top and the
bottom of a 316 dp column with six sections between them, so the prompt sat
fourth behind two dropdowns and Generate sat under a scroll. They are the only
four a common visit touches, and they are the four that never fit together.

The recipe stays in the settings column (``panes/settings_2d``), which is the
other half of the split: this bar is *what to make*, and that column is *how*.
A control belongs to exactly one of them, the same one-owner rule the two
generation panes already keep.

**The Reference stage only.** Mesh, Rig, Pose and Export draw no bar and their
columns simply start higher; nothing reserves an empty strip. That is
``create_stages``' own rule about the rail applied to the bar under it --
shipping a row with one live control and three dead ones is not honest, and a
bar that is present but inert is worse than a bar that is absent. The Mesh
stage can grow its own when it has four controls worth the width.

Drawn through :func:`layout.pane` rather than bare, unlike the stage rail above
it. That is what puts it in ``layout.FRAME_PANES``, which is what gives it the
role fill, the divider, ``guard``'s error isolation, and a pane slot for
``probe._pane_at`` -- without it ``/exercise-mode create`` reports the bar's
controls against the empty-string pane, which reads downstream as four controls
nobody owns.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from . import anchors, controls, create_assets, focus, theme, widgets
from .tokens import sp

#: The pane's height in design pixels. Enough for the prompt field plus the
#: padding ``layout.pane`` pushes, and fixed rather than measured: the row's
#: tallest control is the prompt, whose height this file chooses, so there is
#: nothing here for a measure-last-frame idiom to discover.
BAR_H = 62.0

#: The prompt field's own height. Two lines rather than one, because
#: ``MAX_PROMPT`` is a thousand characters and a single-line input for a
#: paragraph shows the tail and hides the subject. It scrolls past two.
PROMPT_H = 40.0

#: Fixed widths for the three controls that flank the prompt, which takes what
#: is left. Design pixels; ``sp`` scales them.
#:
#: ``PROMPT_MIN_W`` is where the prompt stops shrinking and the count is
#: dropped instead -- see :func:`_row_widths`.
PROMPT_MIN_W = 150.0
TYPE_W = 138.0
COUNT_W = 124.0
GENERATE_W = 158.0

#: This bar's key in the focus ring. Its own rather than ``settings_2d``'s
#: ``"2d"``: the ring is walked per pane, and the two are now two panes.
FOCUS_PANE = "brief"


def shows(ctx: Any) -> bool:
    """Whether the bar has anything true to say. -> only on the Reference stage."""
    from . import create_stages

    return create_stages.at(ctx.state, "reference")


def draw(ctx: Any) -> None:
    """The bar. Called from ``main._build_ui`` between the rail and the columns."""
    from .panes import settings_2d

    state = ctx.state
    form = state.form_2d
    # The same synchronisation the settings column runs, and for the same
    # reason: the five derived door fields are a function of the type, and the
    # type is edited *here* now. Running it before anything is drawn keeps the
    # count control's own visibility honest on the frame the type changes.
    if "asset_type" not in form:
        form["asset_type"] = create_assets.legacy_asset_type(form)
    spec = create_assets.sync_legacy_fields(form)

    focus.pump(state, FOCUS_PANE)
    focus.begin(state, FOCUS_PANE)

    # **Both sheet doors and the character door make exactly one thing per
    # press.** ``sync_legacy_fields`` has already written ``count = 1`` for all
    # three, so four radios of which three are refusals would be a control
    # offering what the thing behind it will not do.
    hide_count = form.get("output") in ("sheet", "character")
    problems = settings_2d.problems_for(ctx, form)
    busy = ctx.busy("submit")

    _type(ctx, form)
    imgui.same_line()
    prompt_w, show_count = _row_widths(hide_count)
    _prompt(ctx, form, prompt_w)
    imgui.same_line()
    if show_count:
        _count(ctx, form)
        imgui.same_line()
    _generate(ctx, form, spec, enabled=not problems and not busy, problems=problems)


def _row_widths(sheet: bool) -> tuple[float, bool]:
    """-> (prompt width, whether to draw the count). **The row's give-way order.**

    Measured *after* the type combo and its ``same_line``, so
    ``get_content_region_avail`` has already taken the combo off -- subtracting
    ``TYPE_W`` here as well double-counted it, and the floor underneath then
    turned the shortfall into a row wider than the pane. ``same_line`` past the
    pane edge draws a control nowhere, so what that produced at the resize
    floor was a clipped Generate: the one control the bar exists to keep
    visible.

    So the row gives way in a stated order, the way the rail does. The prompt
    shrinks first, to ``PROMPT_MIN_W``. Then the **count** is dropped -- it is
    the only one of the four with a sane default and the only one whose value
    is restated elsewhere, in the plan block's "N candidates". The type and
    Generate never give way, because they are what the bar is for.
    """
    gap = imgui.get_style().item_spacing.x
    avail = imgui.get_content_region_avail().x
    need = sp(GENERATE_W) + gap
    count = 0.0 if sheet else sp(COUNT_W) + gap
    prompt = avail - need - count
    if count and prompt < sp(PROMPT_MIN_W):
        count = 0.0
        prompt = avail - need
    return max(sp(PROMPT_MIN_W), prompt), count > 0


def _type(ctx: Any, form: dict[str, Any]) -> None:
    """What to make. The one choice that decides what everything else means."""
    before = create_assets.selected(form).key
    with focus.item(ctx.state, FOCUS_PANE, "asset_type"):
        picked = widgets.combo(
            "##generation-type",
            before,
            list(create_assets.ASSET_TYPE_OPTIONS),
            width=sp(TYPE_W),
            tooltip=_TYPE_HINTS.get(before, ""),
        )
    form["asset_type"] = picked if picked in create_assets.ASSET_TYPES else before
    form["generation_type"] = form["asset_type"]
    if create_assets.sync_legacy_fields(form).key != before:
        ctx.state.clear_field_error("asset_type")
    _ring(ctx, "asset_type")


def _prompt(ctx: Any, form: dict[str, Any], width: float) -> None:
    """The words. The field this whole rearrangement is about.

    An explicit ``width``, which is the one thing that matters here:
    ``widgets.multiline`` defaults to -1, meaning *fill the row*. In a column
    that is what you want; on a row it takes the width the two controls after
    it were going to use, and ``same_line`` past the pane edge draws a control
    nowhere -- so the count and Generate vanished off the right-hand side
    entirely.
    """
    before = form["prompt"]
    with focus.item(ctx.state, FOCUS_PANE, "prompt"):
        form["prompt"] = widgets.multiline(
            "##brief-prompt", before, sp(PROMPT_H), _max_prompt(), width=width
        )
        anchors.mark("create/prompt")
        widgets.char_count(form["prompt"], _max_prompt())
    if form["prompt"] != before:
        ctx.state.clear_field_error("prompt")
    if imgui.is_item_hovered() and not str(form["prompt"]).strip():
        imgui.set_tooltip("Describe one subject -- a prop, a character, a surface.")
    _ring(ctx, "prompt")


def _count(ctx: Any, form: dict[str, Any]) -> None:
    """How many alternatives one press should draw.

    Not drawn at all for a sheet: both sheet doors refuse a batch and say why,
    so four radios of which three are refusals would be a control offering
    what the thing behind it will not do. ``sync_legacy_fields`` has already
    written ``count = 1`` for those.
    """
    imgui.begin_group()
    imgui.push_item_width(sp(COUNT_W))
    with focus.item(ctx.state, FOCUS_PANE, "count") as focused:
        changed, picked = controls.segmented_choice(
            "brief-count",
            tuple((str(n), str(n)) for n in _COUNTS),
            str(form["count"]),
            compact=True,
            tooltips=_COUNT_HINTS,
        )
        if changed:
            form["count"] = int(picked)
            ctx.state.clear_field_error("count")
        # Hand-answered, as it was in the column: a row of radios is one
        # control to the keyboard even though it is four items to imgui.
        if focused:
            here = _COUNTS.index(form["count"]) if form["count"] in _COUNTS else 0
            before_arrow = form["count"]
            if imgui.is_key_pressed(imgui.Key.left_arrow):
                form["count"] = _COUNTS[(here - 1) % len(_COUNTS)]
            if imgui.is_key_pressed(imgui.Key.right_arrow):
                form["count"] = _COUNTS[(here + 1) % len(_COUNTS)]
            # The click branch above clears the ring on a change; this
            # hand-rolled branch edits ``form["count"]`` the same way and must
            # clear the same error, or a user who fixes an invalid count with
            # the keyboard instead of a click keeps a ring pointing at a value
            # that is no longer wrong. The 2026-09-05 audit, finding create-07.
            if form["count"] != before_arrow:
                ctx.state.clear_field_error("count")
    imgui.pop_item_width()
    _ring(ctx, "count")
    imgui.end_group()


def _generate(
    ctx: Any, form: dict[str, Any], spec: Any, *, enabled: bool, problems: list[Any]
) -> None:
    """The press. Always visible, which is the point of the bar.

    The *reason* it is disabled stays in the settings column's plan block,
    which lists every problem and offers the one-click repairs. Here it is a
    tooltip: a bar has no room for a list, and a button that says nothing about
    why it is dead is the complaint this redesign started from.
    """
    from .panes import settings_2d

    with focus.item(ctx.state, FOCUS_PANE, "generate") as focused:
        pressed = widgets.primary_button(
            spec.create_label,
            (sp(GENERATE_W), sp(PROMPT_H)),
            enabled=enabled,
            # ``Problem`` is a str subclass -- the message *is* the object.
            reason=str(problems[0]) if problems else "",
            tooltip="Ctrl+Enter",
        )
        anchors.mark("create/generate")
        if focused and enabled and _enter_pressed():
            pressed = True
    if pressed:
        settings_2d.generate(ctx, form)


def _ring(ctx: Any, field: str) -> bool:
    """Mark the control just drawn if a refusal named it. -> whether it did.

    ``widgets.field_error`` is the column's version and draws the message and
    any install offer *below* the control, which in a one-row bar would push
    the row apart. The ring alone here; the words are in the plan block.
    """
    if not (getattr(ctx.state, "field_errors", None) or {}).get(field):
        return False
    widgets.ring(
        imgui.get_item_rect_min(), imgui.get_item_rect_max(), theme.ERR, 0.9, thick=1.5
    )
    return True


def _max_prompt() -> int:
    """``MAX_PROMPT``, imported lazily so this module stays cheap to import."""
    from ..service.validation import MAX_PROMPT

    return MAX_PROMPT


def _enter_pressed() -> bool:
    return imgui.is_key_pressed(imgui.Key.enter) or imgui.is_key_pressed(
        imgui.Key.keypad_enter
    )


#: The count control's four values. 8 is ``validation.MAX_REFERENCE_COUNT``.
_COUNTS: tuple[int, ...] = (1, 2, 4, 8)

#: Per-pill hover text for the count control, the same shape as
#: ``_TYPE_HINTS`` beside it. Four bare pills ("1 2 4 8") carried no label and
#: no tooltip -- unlike the Type combo, which names each value on hover -- so
#: a first-time user had nothing on screen or on hover saying they choose how
#: many candidates one press draws. The 2026-09-05 audit, finding create-11.
_COUNT_HINTS: dict[str, str] = {
    "1": "Draw one candidate.",
    "2": "Draw two candidates to compare.",
    "4": "Draw four candidates to compare.",
    "8": "Draw eight, the most one press can generate.",
}

def _species_count() -> int:
    """How many species the character registry ships. Counted, never typed.

    ``characters.family`` imports nothing but the standard library, so reading
    it at import time here costs a dict copy and drags nothing in behind it.
    """
    from ..characters.family import families

    return len(families())


#: One line per type, on the combo's tooltip. The five-row descriptive block
#: that used to sit under the selector is gone with the column it sat in; this
#: is what survives of it, which is the orientation and not the prose.
_TYPE_HINTS: dict[str, str] = {
    "image": "A standalone 2D image.",
    "3d_model": "A reference image you can turn into a 3D model.",
    "seamless_material": "A seamless surface texture.",
    "tileset": "A coherent pixel-art tile sheet.",
    "sprite_sheet": "A character in several frames and directions.",
    # The real scope, in one line, because every other hint here describes an
    # SDXL request and this one describes the opposite: a body built from the
    # species registry, rigged and rendered on the CPU. "No GPU needed" is the
    # half a user with a small card most needs to read.
    #
    # The species count is *counted*, never typed: a sibling adding a row to
    # ``characters.family._FAMILIES`` must not leave a tooltip promising the
    # number the registry held last month.
    "character": (
        f"{_species_count()} species across four body plans, built and animated "
        f"into a sprite sheet. No GPU needed."
    ),
}

__all__ = ["BAR_H", "FOCUS_PANE", "draw", "shows"]
