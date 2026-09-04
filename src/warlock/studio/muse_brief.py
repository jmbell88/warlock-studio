"""Muse's command bar: the whole brief, across the top of the mode.

``create_brief``'s shape, and deliberately so -- the two are the same claim
about the same kind of screen: *what a press needs, on one row, never scrolled.*
The bar is **what to make** (the style tags, the lyrics, how long, how many, and
the button) and the column beside it (``panes/muse_recipe``) is **how**. A
control belongs to exactly one of them, which is the one-owner rule Create's
two panes already keep.

**Unconditional, unlike Create's.** ``create_brief.shows`` gates on the
Reference stage because the other four stages have no brief to press; Muse has
no stages, so the bar is simply always there. That is the only structural
difference between the two files, and it is why this one has no ``shows``.

Drawn through :func:`layout.pane` rather than bare, for ``create_brief``'s
reason: that is what puts it in ``layout.FRAME_PANES``, which is what gives it
the role fill, the divider, ``guard``'s error isolation and a pane slot for
``probe._pane_at`` -- without it ``/exercise-mode muse`` reports the bar's
controls against the empty-string pane, which reads downstream as controls
nobody owns.
"""

from __future__ import annotations

from typing import Any

from imgui_bundle import imgui

from . import anchors, controls, focus, muse_mode, theme, widgets
from .tokens import sp

#: The pane's height in design pixels. Taller than Create's 62: this bar
#: carries a *second* multi-line field (the lyrics), because a lyric block is
#: the model's other real input and burying it in the recipe column would say
#: it was a setting.
BAR_H = 118.0

#: The two text fields' heights. The tags field is two lines for
#: ``create_brief``'s reason -- a single line for a paragraph shows the tail and
#: hides the subject -- and the lyrics field is four, because a verse is four
#: lines and a field that cannot show one is a field you write blind in.
TAGS_H = 40.0
LYRICS_H = 62.0

#: Fixed widths for the controls that flank the text fields, which take what is
#: left. ``TEXT_MIN_W`` is where they stop shrinking and the count is dropped
#: instead -- see :func:`_row_widths`.
TEXT_MIN_W = 180.0
DURATION_W = 150.0
COUNT_W = 124.0
GENERATE_W = 158.0

#: This bar's key in the focus ring. Its own, as ``create_brief``'s is: the ring
#: is walked per pane and the bar and the recipe column are two panes.
FOCUS_PANE = "muse-brief"

#: The count control's values, and the top one is ``_jobs_music.MAX_COUNT``.
_COUNTS: tuple[int, ...] = (1, 2, 4)

#: The duration presets, in seconds. A segmented choice rather than a free
#: number because these are the four lengths game music is actually asked for,
#: and a drag that can land on 143 seconds offers precision nobody wants.
#: Every value is inside ``_jobs_music``'s bounds, which is the only thing
#: about them the door cares about.
_DURATIONS: tuple[int, ...] = (30, 60, 120, 240)


def draw(ctx: Any) -> None:
    """The bar. Called from ``main._muse_workspace`` above the columns."""
    state = muse_mode.ensure(ctx)
    form = state.form

    focus.pump(ctx.state, FOCUS_PANE)
    focus.begin(ctx.state, FOCUS_PANE)

    busy = ctx.busy("submit")
    text_w, show_count = _row_widths()

    _tags(ctx, form, text_w)
    imgui.same_line()
    _duration(ctx, form)
    if show_count:
        imgui.same_line()
        _count(ctx, form)
    imgui.same_line()
    _generate(ctx, enabled=not busy)
    _lyrics(ctx, form, text_w)


def _row_widths() -> tuple[float, bool]:
    """-> (text width, whether to draw the count). **The row's give-way order.**

    Stated, the way ``create_brief._row_widths`` states it, and for the reason
    that file learned the hard way: ``same_line`` past the pane edge draws a
    control *nowhere*, so an unstated order does not produce a cramped row, it
    produces a missing Generate.

    The text fields shrink first, to ``TEXT_MIN_W``. Then the **count** is
    dropped -- it is the one control with a sane default whose value is also
    visible in the tray, as the number of cards a press produced. Duration and
    Generate never give way: the first is the parameter that decides what the
    press costs, and the second is what the bar is for.
    """
    gap = imgui.get_style().item_spacing.x
    avail = imgui.get_content_region_avail().x
    fixed = sp(DURATION_W) + gap + sp(GENERATE_W) + gap
    count = sp(COUNT_W) + gap
    text = avail - fixed - count
    if text < sp(TEXT_MIN_W):
        count = 0.0
        text = avail - fixed
    return max(sp(TEXT_MIN_W), text), count > 0


def _tags(ctx: Any, form: dict[str, Any], width: float) -> None:
    """The style tags. The model's first input, under the model's own name.

    Comma-separated, because that is literally what ACE-Step's text encoder was
    trained on -- not a sentence. The placeholder says so rather than a label
    above it saying "Style", which would take a row this bar does not have.
    """
    before = form["prompt"]
    with focus.item(ctx.state, FOCUS_PANE, "prompt"):
        form["prompt"] = widgets.multiline(
            "##muse-tags", before, sp(TAGS_H), _max_prompt(), width=width
        )
        anchors.mark("muse/tags")
        widgets.char_count(form["prompt"], _max_prompt())
    if form["prompt"] != before:
        ctx.state.clear_field_error("prompt")
    if imgui.is_item_hovered() and not str(form["prompt"]).strip():
        imgui.set_tooltip(
            "Style tags, comma separated -- 'dark ambient, dungeon, low strings, "
            "slow'. Not a sentence."
        )
    _ring(ctx, "prompt")


def _lyrics(ctx: Any, form: dict[str, Any], width: float) -> None:
    """The lyric block. The model's second input, and optional.

    On its own row under the tags rather than beside them: two multi-line
    fields sharing a row would each be half a field. It is still *the bar* and
    not the recipe column, because it is part of what to make -- an instrumental
    and a song with a chorus are different requests, not the same request at a
    different setting.
    """
    before = form["lyrics"]
    with focus.item(ctx.state, FOCUS_PANE, "lyrics"):
        form["lyrics"] = widgets.multiline(
            "##muse-lyrics", before, sp(LYRICS_H), _max_lyrics(), width=width
        )
        anchors.mark("muse/lyrics")
    if form["lyrics"] != before:
        ctx.state.clear_field_error("lyrics")
    if imgui.is_item_hovered() and not str(form["lyrics"]).strip():
        imgui.set_tooltip(
            "Lyrics, with [verse] and [chorus] markers. Leave it empty for an "
            "instrumental."
        )
    _ring(ctx, "lyrics")


def _duration(ctx: Any, form: dict[str, Any]) -> None:
    """How long. The one parameter that decides what the press costs."""
    imgui.begin_group()
    imgui.push_item_width(sp(DURATION_W))
    with focus.item(ctx.state, FOCUS_PANE, "duration") as focused:
        current = int(form["duration"])
        changed, picked = controls.segmented_choice(
            "muse-duration",
            tuple((str(n), f"{n}s") for n in _DURATIONS),
            str(current if current in _DURATIONS else _DURATIONS[1]),
            compact=True,
        )
        if changed:
            form["duration"] = float(picked)
            ctx.state.clear_field_error("duration")
        # Hand-answered, as Create's count is: a row of radios is one control
        # to the keyboard even though it is four items to imgui.
        if focused:
            _step(form, "duration", _DURATIONS, cast=float)
    imgui.pop_item_width()
    _ring(ctx, "duration")
    imgui.end_group()


def _count(ctx: Any, form: dict[str, Any]) -> None:
    """How many takes one press should draw."""
    imgui.begin_group()
    imgui.push_item_width(sp(COUNT_W))
    with focus.item(ctx.state, FOCUS_PANE, "count") as focused:
        changed, picked = controls.segmented_choice(
            "muse-count",
            tuple((str(n), str(n)) for n in _COUNTS),
            str(form["count"]),
            compact=True,
        )
        if changed:
            form["count"] = int(picked)
            ctx.state.clear_field_error("count")
        if focused:
            _step(form, "count", _COUNTS, cast=int)
    imgui.pop_item_width()
    _ring(ctx, "count")
    imgui.end_group()


def _step(form: dict[str, Any], field: str, values: tuple[int, ...], *, cast: Any) -> None:
    """Left/Right through a segmented choice. One spelling for the two of them."""
    current = int(form[field])
    here = values.index(current) if current in values else 0
    if imgui.is_key_pressed(imgui.Key.left_arrow):
        form[field] = cast(values[(here - 1) % len(values)])
    if imgui.is_key_pressed(imgui.Key.right_arrow):
        form[field] = cast(values[(here + 1) % len(values)])


def _generate(ctx: Any, *, enabled: bool) -> None:
    """The press. Always visible, which is the point of the bar.

    Disabled only while a submit is in flight. It is deliberately *not* also
    disabled on an empty prompt or on missing weights: the service refuses both
    with a sentence naming the control, which is more use than a dead button --
    and unlike Create there is no plan block below to carry the list, so a
    refusal that never arrives is a refusal nobody reads.
    """
    with focus.item(ctx.state, FOCUS_PANE, "generate") as focused:
        pressed = widgets.primary_button(
            "Generate",
            (sp(GENERATE_W), sp(TAGS_H)),
            enabled=enabled,
            tooltip="Ctrl+Enter",
        )
        anchors.mark("muse/generate")
        if focused and enabled and _enter_pressed():
            pressed = True
    if pressed:
        muse_mode.generate(ctx)


def _ring(ctx: Any, field: str) -> bool:
    """Mark the control just drawn if a refusal named it. -> whether it did.

    The ring alone, no message: ``widgets.field_error`` draws its text *below*
    the control, which in a bar would push the row apart. The words arrive as
    the toast ``ctx.toast`` already raised.
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


def _max_lyrics() -> int:
    from ..service._jobs_music import MAX_LYRICS

    return MAX_LYRICS


def _enter_pressed() -> bool:
    return imgui.is_key_pressed(imgui.Key.enter) or imgui.is_key_pressed(
        imgui.Key.keypad_enter
    )


__all__ = ["BAR_H", "FOCUS_PANE", "draw"]
