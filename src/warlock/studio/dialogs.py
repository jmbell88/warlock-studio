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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from imgui_bundle import imgui
from imgui_bundle import portable_file_dialogs as pfd

from . import widgets

log = logging.getLogger(__name__)

# Filters, as portable-file-dialogs wants them: name, then patterns.
GLB_FILTER = ["glTF binary (*.glb)", "*.glb"]
IMAGE_FILTER = ["Images", "*.png *.jpg *.jpeg *.webp *.bmp"]
ZIP_FILTER = ["Zip archive (*.zip)", "*.zip"]
PNG_FILTER = ["PNG image (*.png)", "*.png"]
JSON_FILTER = ["JSON (*.json)", "*.json"]

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
    """Blocking; call from a task thread."""
    try:
        picked = pfd.open_file(title, "", filters or ["All files", "*"]).result()
    except Exception:
        log.exception("the file picker failed")
        return None
    return Path(picked[0]) if picked else None


def save_file(title: str, default_name: str, filters: list[str] | None = None) -> Path | None:
    """Blocking; call from a task thread."""
    try:
        picked = pfd.save_file(title, default_name, filters or ["All files", "*"]).result()
    except Exception:
        log.exception("the save picker failed")
        return None
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
    _open: bool = field(default=False, repr=False)


class ConfirmQueue:
    """At most one question at a time, drawn as a modal."""

    def __init__(self) -> None:
        self.pending: Confirm | None = None

    def ask(self, confirm: Confirm) -> None:
        # A second question while one is open is dropped rather than stacked:
        # they only ever come from a user action, and the user cannot have
        # taken two actions between frames.
        if self.pending is None:
            self.pending = confirm

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
        imgui.dummy((0, 6))
        # The action is red, the escape is neutral: two identical buttons make
        # a destructive question a coin toss.
        if widgets.destructive_button(confirm.confirm_label, (150, 0)):
            imgui.close_current_popup()
            self.pending = None
            if confirm.on_confirm is not None:
                confirm.on_confirm()
        imgui.same_line()
        if imgui.button(confirm.cancel_label, (150, 0)):
            imgui.close_current_popup()
            self.pending = None
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
    def __init__(self) -> None:
        self.pending: Prompt | None = None

    def ask(self, prompt: Prompt) -> None:
        if self.pending is None:
            self.pending = prompt

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
        imgui.set_next_item_width(320)
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
        accepted = entered or imgui.button("Save", (150, 0))
        imgui.same_line()
        cancelled = imgui.button("Cancel", (150, 0))
        if accepted and prompt.value.strip():
            imgui.close_current_popup()
            self.pending = None
            if prompt.on_accept is not None:
                prompt.on_accept(prompt.value.strip())
        elif cancelled:
            imgui.close_current_popup()
            self.pending = None
        imgui.end_popup()
