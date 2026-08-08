"""Native file dialogs, and the confirmations that used to be ``window.confirm``.

The file dialogs come from imgui-bundle's bundled portable-file-dialogs, which
is the OS's own picker -- an in-app file browser would be a second, worse
Explorer. They are *modal to the OS*, so every one of them is opened from a
task thread and polled, never called on the frame thread: a blocking picker on
the frame thread freezes the progress bar behind it, which is the exact thing
the browser's ``confirm()`` did that made the old cancel button feel broken.

The confirmations are ordinary imgui modals for the same reason.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from imgui_bundle import imgui
from imgui_bundle import portable_file_dialogs as pfd

from ..service.errors import Failed
from . import widgets
from .tokens import sp

log = logging.getLogger(__name__)

# Filters, as portable-file-dialogs wants them: name, then patterns.
GLB_FILTER = ["glTF binary (*.glb)", "*.glb"]
IMAGE_FILTER = ["Images", "*.png *.jpg *.jpeg *.webp *.bmp"]
ZIP_FILTER = ["Zip archive (*.zip)", "*.zip"]
PNG_FILTER = ["PNG image (*.png)", "*.png"]
JSON_FILTER = ["JSON (*.json)", "*.json"]

# The modal's two fixed measurements, in design pixels (K97). Both used to be
# raw numbers, so at 200% scale a 150 px button held text drawn at 300% and the
# label ran off the end of it.
BUTTON_W = 150.0
FIELD_W = 320.0

ARTIFACT_FILTERS = {
    ".json": JSON_FILTER,
    ".glb": GLB_FILTER,
    ".stl": ["Stereolithography (*.stl)", "*.stl"],
    ".zip": ZIP_FILTER,
    ".fbx": ["Filmbox (*.fbx)", "*.fbx"],
    ".png": PNG_FILTER,
    ".log": ["Text (*.txt *.log)", "*.txt *.log"],
}


def open_file(title: str, filters: list[str] | None = None) -> Path | None:
    """Blocking; call from a task thread.

    ``None`` means **the user cancelled**, and nothing else (E44). A picker that
    failed to open raises instead: both used to return ``None``, so a portable-
    file-dialogs failure -- no zenity on the host, a destroyed parent window --
    was indistinguishable from Escape, and the caller correctly did nothing
    about it. Every call site is inside a task closure, so the raise arrives as
    the ordinary failure toast rather than as a crash.
    """
    try:
        picked = pfd.open_file(title, "", filters or ["All files", "*"]).result()
    except Exception as exc:
        log.exception("the file picker failed")
        raise Failed("The file picker did not open. See the log for details.") from exc
    return Path(picked[0]) if picked else None


def save_file(title: str, default_name: str, filters: list[str] | None = None) -> Path | None:
    """Blocking; call from a task thread. ``None`` is a cancel; see
    :func:`open_file` for why a failure raises rather than joining it."""
    try:
        picked = pfd.save_file(title, default_name, filters or ["All files", "*"]).result()
    except Exception as exc:
        log.exception("the save picker failed")
        raise Failed("The save dialog did not open. See the log for details.") from exc
    return Path(picked) if picked else None


def filters_for(name: str) -> list[str]:
    return ARTIFACT_FILTERS.get(Path(name).suffix.lower(), ["All files", "*"])


@dataclass
class Confirm:
    """A pending yes/no question.

    Held as data rather than asked inline so the frame that raises it and the
    frame that answers it are different frames -- which is what keeps the
    viewport rendering underneath.
    """

    title: str
    message: str
    confirm_label: str = "Discard"
    cancel_label: str = "Keep editing"
    on_confirm: Any = None
    # An optional widget drawn between the message and the buttons, for a
    # question that has a *parameter* rather than only an answer -- the prune
    # keep-count (O116). Deliberately a callable rather than a declared field
    # type: a confirm is not a form builder, and the two or three questions
    # that want one already know how to draw it.
    body: Any = None
    _open: bool = field(default=False, repr=False)
    _focused: bool = field(default=False, repr=False)


def _enter_pressed() -> bool:
    """Enter or numpad Enter, as a *press* rather than a repeat.

    Read from imgui rather than from pygame because a modal is imgui's: the
    frame loop stops dispatching shortcuts entirely while one is up (otherwise
    Esc both cancels the dialog and leaves the mode behind it), so this is the
    only layer that still sees the key.
    """
    return imgui.is_key_pressed(imgui.Key.enter) or imgui.is_key_pressed(
        imgui.Key.keypad_enter
    )


def _escape_pressed() -> bool:
    return imgui.is_key_pressed(imgui.Key.escape)


class ConfirmQueue:
    """A queue of yes/no questions, drawn one modal at a time.

    It really is a queue (I78). It used to keep a single slot and drop anything
    that arrived while one was up, on the reasoning that questions come from
    user actions and the user cannot act twice between frames -- which is true
    of *clicks* and false of everything else that asks. A quit asks three
    questions in a row (painted pixels, built geometry, an unsaved pose) and
    had to hand-nest them through callbacks to survive; a finished task and a
    click landing in one frame are two questions with no user double-action
    involved at all. Dropping the second was silent, and what it dropped was
    the offer to save someone's work.
    """

    def __init__(self) -> None:
        self._queue: deque[Confirm] = deque()

    @property
    def pending(self) -> Confirm | None:
        """The question on screen, or ``None``. Read by the frame loop to know
        a modal is up; it is the head of the queue, never a second field."""
        return self._queue[0] if self._queue else None

    @property
    def waiting(self) -> int:
        """How many questions are behind the one on screen."""
        return max(0, len(self._queue) - 1)

    def ask(self, confirm: Confirm) -> None:
        self._queue.append(confirm)

    def dismiss(self) -> None:
        """Drop the question on screen without answering it, outside a draw.

        ``pending`` is the head of the queue and so is deliberately read-only:
        assigning ``None`` to it used to be how a caller said this, and on a
        queue that would silently throw away everything behind it too.
        """
        if self._queue:
            self._queue.popleft()

    def _answered(self) -> None:
        imgui.close_current_popup()
        self._queue.popleft()

    def draw(self) -> None:
        confirm = self.pending
        if confirm is None:
            return
        if not confirm._open:
            imgui.open_popup(confirm.title)
            confirm._open = True
        centre = imgui.get_main_viewport().get_center()
        imgui.set_next_window_pos(centre, imgui.Cond_.appearing.value, (0.5, 0.5))
        opened, _ = imgui.begin_popup_modal(
            confirm.title, None, imgui.WindowFlags_.always_auto_resize.value
        )
        if not opened:
            return
        imgui.text_wrapped(confirm.message)
        if self.waiting:
            widgets.muted(f"{self.waiting} more to answer")
        if confirm.body is not None:
            imgui.dummy((0, 6))
            confirm.body()
        imgui.dummy((0, 6))
        # The action is red, the escape is neutral: two identical buttons make
        # a destructive question a coin toss.
        confirmed = widgets.destructive_button(confirm.confirm_label, (sp(BUTTON_W), 0))
        # Focus lands on the confirming button (I77) so Enter is unambiguous
        # and so a keyboard user can see where they are. Only on the frame the
        # modal appears: re-focusing every frame would fight the Tab key.
        if not confirm._focused:
            imgui.set_item_default_focus()
            confirm._focused = True
        imgui.same_line()
        cancelled = imgui.button(confirm.cancel_label, (sp(BUTTON_W), 0))
        # Esc cancels, Enter confirms. Enter is read here rather than left to
        # imgui's own nav activation because keyboard nav is not enabled: with
        # it off, a focused button is drawn as focused and does nothing.
        if _escape_pressed():
            cancelled = True
        elif _enter_pressed():
            confirmed = True
        if confirmed:
            self._answered()
            if confirm.on_confirm is not None:
                confirm.on_confirm()
        elif cancelled:
            self._answered()
        imgui.end_popup()


@dataclass
class Prompt:
    """A one-line text question -- naming a pose, renaming a job."""

    title: str
    label: str
    value: str = ""
    on_accept: Any = None
    _open: bool = field(default=False, repr=False)


class PromptQueue:
    """A queue, for the reason :class:`ConfirmQueue` is one."""

    def __init__(self) -> None:
        self._queue: deque[Prompt] = deque()

    @property
    def pending(self) -> Prompt | None:
        return self._queue[0] if self._queue else None

    @property
    def waiting(self) -> int:
        return max(0, len(self._queue) - 1)

    def ask(self, prompt: Prompt) -> None:
        self._queue.append(prompt)

    def dismiss(self) -> None:
        if self._queue:
            self._queue.popleft()

    def _answered(self) -> None:
        imgui.close_current_popup()
        self._queue.popleft()

    def draw(self) -> None:
        prompt = self.pending
        if prompt is None:
            return
        if not prompt._open:
            imgui.open_popup(prompt.title)
            prompt._open = True
        centre = imgui.get_main_viewport().get_center()
        imgui.set_next_window_pos(centre, imgui.Cond_.appearing.value, (0.5, 0.5))
        opened, _ = imgui.begin_popup_modal(
            prompt.title, None, imgui.WindowFlags_.always_auto_resize.value
        )
        if not opened:
            return
        imgui.set_next_item_width(sp(FIELD_W))
        if not imgui.is_any_item_active():
            imgui.set_keyboard_focus_here()
        # ``enter_returns_true`` makes the returned flag mean *Enter was
        # pressed*, not *the text changed* -- but the returned string is the
        # live buffer either way. Storing it only when the flag was set left
        # ``value`` at whatever it was seeded with for the whole modal, so
        # typing a name and clicking Save saved the old one (or, for a fresh
        # prompt, silently refused an empty string). Only Enter ever worked.
        entered, value = imgui.input_text(
            prompt.label, prompt.value, imgui.InputTextFlags_.enter_returns_true.value
        )
        prompt.value = value
        if self.waiting:
            widgets.muted(f"{self.waiting} more to answer")
        accepted = entered or imgui.button("Save", (sp(BUTTON_W), 0))
        imgui.same_line()
        cancelled = imgui.button("Cancel", (sp(BUTTON_W), 0)) or _escape_pressed()
        if accepted and prompt.value.strip():
            self._answered()
            if prompt.on_accept is not None:
                prompt.on_accept(prompt.value.strip())
        elif cancelled:
            self._answered()
        imgui.end_popup()
