"""Adaptive, validation-aware form composition for studio panes."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from imgui_bundle import imgui

from . import controls, fonts, theme, tokens
from .tokens import sp

FORM_BREAKPOINT = 480.0
LABEL_WIDTH = 120.0


@dataclass(frozen=True)
class FormLayout:
    """The layout decision for one available width, in physical pixels."""

    stacked: bool
    label_width: float
    control_width: float


def adaptive_layout(available_width: float, *, scale: float | None = None) -> FormLayout:
    """Choose stacked/column layout from a physical available width.

    The threshold is expressed in design pixels, so a 720 px pane at 150 %
    makes the same decision as a 480 px pane at 100 %.
    """

    factor = max(float(tokens.SCALE if scale is None else scale), 0.01)
    design_width = max(float(available_width), 0.0) / factor
    stacked = design_width < FORM_BREAKPOINT
    label = 0.0 if stacked else LABEL_WIDTH * factor
    gap = tokens.SP_2 * factor if not stacked else 0.0
    return FormLayout(stacked, label, max(float(available_width) - label - gap, 0.0))


def sentence_case(label: str) -> str:
    """Normalise a UI label without damaging acronyms embedded in it."""

    text = " ".join(str(label).replace("_", " ").strip().split())
    if not text:
        return text
    if text.isupper() and len(text) > 1:
        text = text.lower()
    return text[:1].upper() + text[1:]


class Form:
    """Context that owns label alignment, help, helper, and validation copy."""

    def __init__(
        self,
        form_id: str,
        *,
        errors: Mapping[str, str] | None = None,
        on_edit: Callable[[str], None] | None = None,
        available_width: float | None = None,
    ) -> None:
        self.form_id = form_id
        # A snapshot, so a clear partway down the form cannot change what the
        # rest of this frame is showing.
        self.errors = dict(errors or {})
        self.on_edit = on_edit
        self.available_width = available_width
        self.layout = FormLayout(True, 0.0, 0.0)
        self._entered = False

    def __enter__(self) -> Form:
        width = (
            float(self.available_width)
            if self.available_width is not None
            else float(imgui.get_content_region_avail().x)
        )
        self.layout = adaptive_layout(width)
        imgui.push_id(self.form_id)
        self._entered = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._entered:
            imgui.pop_id()
            self._entered = False

    def _error(self, field: str, explicit: str = "") -> str:
        return explicit or str(self.errors.get(field, "") or "")

    def _answer(self, field: str, answer: Any) -> Any:
        """Report an edit to the owner, and hand the control's answer back.

        Editing the control a refusal named is the clearest possible "I have
        dealt with that", and a ring that outlived it would be an app arguing
        about a value that is no longer there -- ``State.clear_field_error``'s
        own sentence. ``settings_2d`` says it by hand at some thirty call
        sites; every form that takes ``errors`` gets it here instead, which is
        why the three panes that already passed ``errors`` and cleared nothing
        showed rings that only a *successful* submit could dismiss.
        """
        if self.on_edit is not None and isinstance(answer, tuple) and answer and answer[0]:
            self.on_edit(field)
        return answer

    def _label(self, label: str, help_text: str) -> None:
        # ``widgets.field_label``'s register -- small caps, muted, the info
        # glyph beside the name -- and not a second one. Until 2026-09-05 a
        # ``Form`` field was labelled in the body face in sentence case while
        # the workspaces' ``labeled_*`` controls used small caps, so
        # "Steps" in Poser and "STEPS" in Muse were the same kind of thing
        # drawn as two. What this class owns is the placement (stacked below
        # the breakpoint, a label column above it) and the validation copy;
        # the face is the app's one field face.
        from . import widgets

        widgets.field_label(sentence_case(label), help_text or None)

    @contextmanager
    def field(
        self,
        field: str,
        label: str,
        *,
        help_text: str = "",
        helper: str = "",
        error: str = "",
    ) -> Iterator[str]:
        """Lay out one labelled control and yield its resolved error string."""

        resolved = self._error(field, error)
        start_x = imgui.get_cursor_pos_x()
        if self.layout.stacked:
            self._label(label, help_text)
            imgui.set_next_item_width(-1.0)
            control_x = start_x
        else:
            imgui.begin_group()
            self._label(label, help_text)
            imgui.end_group()
            imgui.same_line(start_x + self.layout.label_width)
            imgui.set_next_item_width(self.layout.control_width)
            control_x = start_x + self.layout.label_width
        yield resolved
        note = resolved or helper
        if note:
            imgui.set_cursor_pos_x(control_x)
            imgui.push_text_wrap_pos(0.0)
            with fonts.small(imgui):
                imgui.text_colored(
                    imgui.ImVec4(*theme.rgba(theme.ERR if resolved else theme.MUTED)),
                    note,
                )
            imgui.pop_text_wrap_pos()
        imgui.dummy((0, sp(tokens.SP_1)))

    def text(
        self,
        field: str,
        label: str,
        value: str,
        *,
        help_text: str = "",
        helper: str = "",
        error: str = "",
        max_length: int = 1000,
        hint: str = "",
        enabled: bool = True,
        reason: str = "",
    ) -> tuple[bool, str]:
        with self.field(field, label, help_text=help_text, helper=helper, error=error) as problem:
            if hint and hasattr(imgui, "input_text_with_hint"):
                result = controls._field_call(
                    "input_text_with_hint",
                    f"##{field}",
                    hint,
                    value,
                    enabled=enabled,
                    reason=reason,
                    error=problem,
                )
            else:
                result = controls.input_text(
                    f"##{field}",
                    value,
                    enabled=enabled,
                    reason=reason,
                    error=problem,
                )
        changed, out = self._answer(field, result)
        return changed, out[:max_length] if changed else value

    def multiline_text(
        self,
        field: str,
        label: str,
        value: str,
        *,
        height: float = 96.0,
        max_length: int = 4000,
        help_text: str = "",
        helper: str = "",
        error: str = "",
        enabled: bool = True,
        reason: str = "",
    ) -> tuple[bool, str]:
        with self.field(field, label, help_text=help_text, helper=helper, error=error) as problem:
            result = controls.input_text_multiline(
                f"##{field}",
                value,
                (-1, sp(height)),
                imgui.InputTextFlags_.word_wrap.value,
                enabled=enabled,
                reason=reason,
                error=problem,
            )
        changed, out = self._answer(field, result)
        return changed, out[:max_length] if changed else value

    def multiline(
        self,
        field: str,
        label: str,
        value: str,
        max_length: int = 4000,
        **kwargs: Any,
    ) -> tuple[bool, str]:
        """Sentence-form alias with the service cap allowed positionally."""

        return self.multiline_text(field, label, value, max_length=max_length, **kwargs)

    def number(
        self,
        field: str,
        label: str,
        value: int | float,
        *,
        help_text: str = "",
        helper: str = "",
        error: str = "",
        enabled: bool = True,
        reason: str = "",
        fmt: str = "%.3f",
    ) -> tuple[bool, int | float]:
        with self.field(field, label, help_text=help_text, helper=helper, error=error) as problem:
            if isinstance(value, int) and not isinstance(value, bool):
                result = controls.input_int(
                    f"##{field}",
                    value,
                    enabled=enabled,
                    reason=reason,
                    error=problem,
                )
            else:
                result = controls.input_float(
                    f"##{field}",
                    float(value),
                    0.0,
                    0.0,
                    fmt,
                    enabled=enabled,
                    reason=reason,
                    error=problem,
                )
        return self._answer(field, result)

    def combo(
        self,
        field: str,
        label: str,
        current: str,
        options: Sequence[tuple[str, str]],
        *,
        help_text: str = "",
        helper: str = "",
        error: str = "",
        enabled: bool = True,
        reason: str = "",
    ) -> tuple[bool, str]:
        with self.field(field, label, help_text=help_text, helper=helper, error=error) as problem:
            result = controls.combo(
                f"##{field}",
                current,
                options,
                enabled=enabled,
                reason=reason,
                error=problem,
            )
        return self._answer(field, result)

    def slider(
        self,
        field: str,
        label: str,
        value: int | float,
        low: int | float,
        high: int | float,
        *,
        help_text: str = "",
        helper: str = "",
        error: str = "",
        enabled: bool = True,
        reason: str = "",
        fmt: str | None = None,
    ) -> tuple[bool, int | float]:
        with self.field(field, label, help_text=help_text, helper=helper, error=error) as problem:
            if isinstance(value, int) and isinstance(low, int) and isinstance(high, int):
                result = controls.slider_int(
                    f"##{field}",
                    value,
                    low,
                    high,
                    enabled=enabled,
                    reason=reason,
                    error=problem,
                )
            else:
                result = controls.slider_float(
                    f"##{field}",
                    float(value),
                    float(low),
                    float(high),
                    fmt or "%.3f",
                    enabled=enabled,
                    reason=reason,
                    error=problem,
                )
        return self._answer(field, result)

    def switch(
        self,
        field: str,
        label: str,
        value: bool,
        *,
        help_text: str = "",
        helper: str = "",
        enabled: bool = True,
        reason: str = "",
    ) -> tuple[bool, bool]:
        with self.field(field, label, help_text=help_text, helper=helper):
            result = controls.switch(
                "",
                value,
                control_id=field,
                enabled=enabled,
                reason=reason,
            )
        return self._answer(field, result)

    # **There is no ``checkbox`` on this form, on purpose.** There was, and it
    # drew nothing at all when off: the field grid puts the label in the left
    # column and submits the control as ``##field``, so what remained was an
    # unlabelled imgui checkbox -- whose frame background is ``ELEV_1``, the
    # same value the section panel it sits on is filled with, in dark, light
    # *and* pixel alike. Off it was an empty gap the height of a control; on it
    # was a tick that appeared out of nowhere. Create's Dither box shipped like
    # that, and so did Troupe's movement rows and the pose rows in Sheet.
    #
    # Every one of those four call sites is now :meth:`switch`, which was
    # already what every other Boolean on every one of these forms used, and
    # which draws a track and a knob rather than relying on a frame edge to be
    # visible. A Boolean in a form grid is not a checkbox here, and a second
    # spelling that is invisible in all three shipped palettes is the thing to
    # delete rather than to route around. ``controls.checkbox`` stays: its
    # other callers pass a real label, so there is always something on screen.

    def segmented_choice(
        self,
        field: str,
        label: str,
        current: str,
        options: Sequence[tuple[str, str]],
        *,
        help_text: str = "",
        helper: str = "",
        enabled: bool = True,
        reason: str = "",
        compact: bool = False,
    ) -> tuple[bool, str]:
        with self.field(field, label, help_text=help_text, helper=helper):
            result = controls.segmented_choice(
                field,
                options,
                current,
                enabled=enabled,
                reason=reason,
                compact=compact,
            )
        return self._answer(field, result)

    def note(self, field: str) -> bool:
        """Draw a recorded refusal that belongs to no single control. -> shown.

        Some addresses name a *composition* rather than a field: Troupe's
        ``layout`` is the whole movement table, and ringing an arbitrary row of
        it would point at the wrong switch. The message still has to appear on
        the pane the refusal came from, so it goes above the block it is about,
        in the same colour and the same words the field version uses.

        No ring, deliberately -- ``widgets.field_error`` draws one around the
        item just submitted, and here there is no such item.
        """
        message = self._error(field)
        if not message:
            return False
        imgui.push_text_wrap_pos(0.0)
        imgui.text_colored(imgui.ImVec4(*theme.rgba(theme.ERR)), message)
        imgui.pop_text_wrap_pos()
        imgui.dummy((0, sp(tokens.SP_1)))
        return True

    def readonly(
        self,
        field: str,
        label: str,
        value: Any,
        *,
        help_text: str = "",
        helper: str = "",
    ) -> None:
        """A value in the form grid that is intentionally not editable."""

        with self.field(field, label, help_text=help_text, helper=helper):
            imgui.text(str(value))

    def footer(
        self,
        primary: tuple[str, Callable[[], Any]],
        *,
        cancel: tuple[str, Callable[[], Any]] | None = None,
        reset: tuple[str, Callable[[], Any]] | None = None,
        enabled: bool = True,
        reason: str = "",
    ) -> str | None:
        """Stable action footer with exactly one accent-filled action."""

        imgui.separator()
        clicked: str | None = None
        if reset is not None and controls.button(reset[0], role=controls.ButtonRole.GHOST):
            reset[1]()
            clicked = "reset"
        if cancel is not None:
            if reset is not None:
                imgui.same_line()
            if controls.button(cancel[0], role=controls.ButtonRole.GHOST):
                cancel[1]()
                clicked = "cancel"
        if reset is not None or cancel is not None:
            imgui.same_line()
        if controls.button(
            primary[0],
            role=controls.ButtonRole.PRIMARY,
            enabled=enabled,
            reason=reason,
        ):
            primary[1]()
            clicked = "primary"
        return clicked
