"""The one-line Prompt modal -- naming a pose, renaming a job.

Driven against a stub imgui rather than a real context: what is worth pinning
here is which *string* the accept path hands on, and that is a property of the
twenty lines in ``PromptQueue.draw``, not of the renderer. Nothing tested this
module at all, which is how a Save button that acted on stale text survived.
"""

from __future__ import annotations

from typing import Any

import pytest

from warlock.studio import dialogs


class _Enum:
    value = 0


class _Flags:
    appearing = _Enum()
    always_auto_resize = _Enum()
    enter_returns_true = _Enum()
    # The modal fades and rises as it appears (UX.md Phase 1), which is one
    # style var pushed around ``begin`` and a ``dummy`` inside it.
    alpha = _Enum()


class _Key:
    """Named keys, as opaque tokens -- the fake never presses one."""

    enter = object()
    keypad_enter = object()
    escape = object()


class _FakeImgui:
    """Enough imgui to run ``PromptQueue.draw`` once.

    ``input_text`` returns the live buffer and a flag that -- under
    ``enter_returns_true`` -- means *Enter was pressed*, which is exactly the
    contract the real widget has and the one the bug misread.
    """

    Cond_ = _Flags
    WindowFlags_ = _Flags
    InputTextFlags_ = _Flags
    StyleVar_ = _Flags
    Key = _Key

    def __init__(self, *, typed: str, enter: bool = False, press: str | None = None) -> None:
        self.typed = typed
        self.enter = enter
        self.press = press
        self.closed = False

    def open_popup(self, _title: str) -> None: ...
    def get_main_viewport(self) -> Any:
        return type("V", (), {"get_center": staticmethod(lambda: (0.0, 0.0))})()

    def set_next_window_pos(self, *_a: Any) -> None: ...
    def push_style_var(self, *_a: Any) -> None: ...
    def pop_style_var(self, *_a: Any) -> None: ...
    def dummy(self, *_a: Any) -> None: ...
    def begin_popup_modal(self, *_a: Any) -> tuple[bool, Any]:
        return (True, None)

    def set_next_item_width(self, _w: float) -> None: ...
    def is_any_item_active(self) -> bool:
        return True

    def set_keyboard_focus_here(self) -> None: ...
    def input_text(self, _label: str, _value: str, _flags: int) -> tuple[bool, str]:
        return (self.enter, self.typed)

    def button(self, label: str, _size: Any = None) -> bool:
        return label == self.press

    def is_key_pressed(self, _key: Any) -> bool:
        # Esc and Enter are read from imgui because the frame loop stops
        # dispatching shortcuts while a modal is up. Nothing here presses one.
        return False

    def same_line(self) -> None: ...
    def close_current_popup(self) -> None:
        self.closed = True

    def end_popup(self) -> None: ...


def _run(monkeypatch: pytest.MonkeyPatch, prompt: dialogs.Prompt, fake: _FakeImgui) -> Any:
    monkeypatch.setattr(dialogs, "imgui", fake)
    queue = dialogs.PromptQueue()
    queue.ask(prompt)
    queue.draw()
    return queue


def test_clicking_save_uses_the_text_that_was_typed(monkeypatch) -> None:
    """``enter_returns_true`` makes the returned flag mean *Enter*, not
    *changed* -- so storing the buffer only when it was set left ``value`` at
    whatever seeded the prompt, and Save renamed a thing to its old name."""
    got: list[str] = []
    prompt = dialogs.Prompt(title="Rename", label="Name", value="old", on_accept=got.append)
    queue = _run(monkeypatch, prompt, _FakeImgui(typed="new", press="Save"))

    assert got == ["new"]
    assert queue.pending is None


def test_a_fresh_prompt_saves_rather_than_silently_refusing(monkeypatch) -> None:
    """The empty-seed case: the guard is ``prompt.value.strip()``, so a Save
    that never saw the typed text refused the click with no feedback at all."""
    got: list[str] = []
    prompt = dialogs.Prompt(title="Name it", label="Name", on_accept=got.append)
    _run(monkeypatch, prompt, _FakeImgui(typed="  walk  ", press="Save"))

    assert got == ["walk"]


def test_enter_still_accepts(monkeypatch) -> None:
    got: list[str] = []
    prompt = dialogs.Prompt(title="Name it", label="Name", on_accept=got.append)
    _run(monkeypatch, prompt, _FakeImgui(typed="idle", enter=True))

    assert got == ["idle"]


def test_cancel_accepts_nothing_and_closes(monkeypatch) -> None:
    got: list[str] = []
    prompt = dialogs.Prompt(title="Name it", label="Name", on_accept=got.append)
    queue = _run(monkeypatch, prompt, _FakeImgui(typed="idle", press="Cancel"))

    assert got == []
    assert queue.pending is None


def test_saving_nothing_at_all_neither_accepts_nor_closes(monkeypatch) -> None:
    """Whitespace is not a name, and the modal has to stay up to say so."""
    got: list[str] = []
    prompt = dialogs.Prompt(title="Name it", label="Name", on_accept=got.append)
    queue = _run(monkeypatch, prompt, _FakeImgui(typed="   ", press="Save"))

    assert got == []
    assert queue.pending is prompt


def test_a_second_question_does_not_displace_the_one_on_screen(monkeypatch) -> None:
    queue = dialogs.PromptQueue()
    first = dialogs.Prompt(title="A", label="Name")
    queue.ask(first)
    queue.ask(dialogs.Prompt(title="B", label="Name"))
    assert queue.pending is first
