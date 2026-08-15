"""Adaptive, validation-aware form composition for studio panes."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from imgui_bundle import imgui

from . import controls, fonts, icons, theme, tokens
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
        available_width: float | None = None,
    ) -> None:
        self.form_id = form_id
        self.errors = dict(errors or {})
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

    def _label(self, label: str, help_text: str) -> None:
        with fonts.label(imgui):
            imgui.text(sentence_case(label))
        if help_text:
            imgui.same_line()
            imgui.text_colored(imgui.ImVec4(*theme.rgba(theme.MUTED)), icons.INFO)
            if imgui.is_item_hovered():
                imgui.set_tooltip(help_text)

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
        with self.field(
            field, label, help_text=help_text, helper=helper, error=error
        ) as problem:
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
        changed, out = result
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
        with self.field(
            field, label, help_text=help_text, helper=helper, error=error
        ) as problem:
            result = controls.input_text_multiline(
                f"##{field}",
                value,
                (-1, sp(height)),
                imgui.InputTextFlags_.word_wrap.value,
                enabled=enabled,
                reason=reason,
                error=problem,
            )
        changed, out = result
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

        return self.multiline_text(
            field, label, value, max_length=max_length, **kwargs
        )

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
        with self.field(
            field, label, help_text=help_text, helper=helper, error=error
        ) as problem:
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
        return result

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
        with self.field(
            field, label, help_text=help_text, helper=helper, error=error
        ) as problem:
            result = controls.combo(
                f"##{field}",
                current,
                options,
                enabled=enabled,
                reason=reason,
                error=problem,
            )
        return result

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
        with self.field(
            field, label, help_text=help_text, helper=helper, error=error
        ) as problem:
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
        return result

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
        return result

    def checkbox(
        self,
        field: str,
        label: str,
        value: bool,
        *,
        help_text: str = "",
        helper: str = "",
        error: str = "",
        enabled: bool = True,
        reason: str = "",
    ) -> tuple[bool, bool]:
        with self.field(
            field, label, help_text=help_text, helper=helper, error=error
        ) as problem:
            result = controls.checkbox(
                f"##{field}",
                value,
                enabled=enabled,
                reason=reason,
                error=problem,
            )
        return result

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
        return result

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
