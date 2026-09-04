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
from . import controls, filetypes, theme, widgets
from .tokens import sp

log = logging.getLogger(__name__)

#: The most questions either queue will hold. See ``ConfirmQueue.ask`` for why
#: this is a backstop against a runaway caller rather than a policy about how
#: many questions a user should be asked.
MAX_QUEUED = 64

# Filters, as portable-file-dialogs wants them: name, then patterns. The image
# suffixes come from ``filetypes`` -- the same tuple the drop router refuses
# against -- because a picker that offers what a drop refuses (or the reverse)
# is one list maintained in two places.
GLB_FILTER = ["glTF binary (*.glb)", "*.glb"]
IMAGE_FILTER = ["Images", filetypes.pattern()]
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


def select_folder(title: str, default_path: str = "") -> Path | None:
    """A directory the user picked. Blocking; call from a task thread.

    The third picker, and it exists because one export in the app writes a
    *family* of files under names it chooses rather than one file under a name
    the user chooses: Sirens' export lays down ``song.wav`` beside ``stems/``
    and ``sfx/``, and asking for that through :func:`save_file` would put the
    user's typed filename on one of the four and silently ignore it for the
    rest. What is actually being chosen there is the destination folder, so
    that is what is asked for.

    ``None`` means **the user cancelled** and nothing else, and a picker that
    failed to open raises -- :func:`open_file`'s rule, for its reason.
    """
    try:
        picked = pfd.select_folder(title, default_path).result()
    except Exception as exc:
        log.exception("the folder picker failed")
        raise Failed("The folder picker did not open. See the log for details.") from exc
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
    # *type* -- a confirm is not a form builder, and the two or three questions
    # that want one already know how to draw it. (The em dash is load-bearing:
    # a line starting ``# type:`` is a type comment, and this one made mypy
    # refuse to parse the file, which stopped the check on the whole tree.)
    body: Any = None
    _open: bool = field(default=False, repr=False)
    _focused: bool = field(default=False, repr=False)


#: What the *cancel* button is called, by what the question is. Three words for
#: three questions, and the point is that there are three rather than one per
#: call site: a reader answers on the buttons, and a label that varies by author
#: carries no information. The library disagreed with itself -- "Keep" on two
#: purges and "Cancel" on the three *larger* ones beside them.
#:
#: * **"Keep editing"** -- unsaved work is about to be abandoned. What is kept
#:   is the session you are in. :func:`ask_close_unsaved`, and ``Confirm``'s own
#:   default.
#: * **"Keep"** -- something already saved is about to be destroyed. What is
#:   kept is the thing. :func:`ask_delete`, and the library's five purges.
#: * **"Cancel"** -- the action is reversible, or is a reset, and declining
#:   costs nothing to name.
#:
#: A few callers say it in more words for one screen -- "Keep drawing", "Keep
#: them", "Stay" -- which is the same verb with the object named, not a fourth
#: answer.
CANCEL_LABELS = ("Keep editing", "Keep", "Cancel")


def ask_close_unsaved(ctx: Any, title: str, on_confirm: Any) -> None:
    """Ask before closing a document with unsaved changes, in the one wording.

    Five modes ask this and it was spelled two ways: four said "*Name* has
    unsaved changes." over ``Confirm``'s own [Discard] [Keep editing], and
    Inker said "The changes to *Name* will be lost." over [Close] [Keep
    editing]. Same question, same stakes, different sentence and a different
    button -- and the pair is what a reader answers on, so the one that is worth
    keeping identical is the buttons. :func:`ask_delete`'s argument, applied to
    the *other* of ``Confirm``'s two questions.
    """
    ctx.confirms.ask(
        Confirm(
            title="Close without saving?",
            message=f"{title} has unsaved changes.",
            on_confirm=on_confirm,
        )
    )


def ask_delete(
    ctx: Any, *, title: str, message: str, on_confirm: Any, body: Any = None
) -> None:
    """Ask before something permanent goes, in the one wording.

    ``Confirm``'s own defaults are the *unsaved-work* pair -- "Discard" and
    "Keep editing" -- which is a different question from this one: a delete is
    not about abandoning an edit, it is about destroying a thing that is
    already saved. The three panes that ask it (a pose, a rendered sheet, a
    sweep) each spelled out the same two labels beside their own message, which
    is exactly the shape a fourth spelling grows out of. What stays per caller
    is the message, because the only interesting part of a delete confirm is
    what *else* goes with it -- the baked GLB, the restyled copies, the meshes
    -- and that is never the same sentence twice.

    ``body`` is ``Confirm.body`` passed through, for the one delete that has a
    *parameter* as well as an answer: removing a sweep can optionally take the
    units retention would otherwise keep, and that belongs as a tick the reader
    meets after the sentence explaining what it costs -- not as a second red
    button they have to choose between before reading either.
    """
    ctx.confirms.ask(
        Confirm(
            title=title,
            message=message,
            confirm_label="Delete",
            cancel_label="Keep",
            on_confirm=on_confirm,
            body=body,
        )
    )


def _has_context() -> bool:
    """Whether there is a live ImGui context for ``widgets`` to draw into.

    The precondition for the renderer-only typography in :func:`draw_prompt`:
    ``widgets.field_label`` reaches for ``widgets.imgui`` whatever this module's
    own binding was replaced with, so what has to be true is that a context
    exists -- not that two modules hold the same object. Written defensively
    because the headless interaction tests' backend is a deliberately tiny
    stand-in and need not carry the getter at all.
    """
    getter = getattr(widgets.imgui, "get_current_context", None)
    if getter is None:
        return False
    try:
        return getter() is not None
    except Exception:  # noqa: BLE001 - a stand-in backend may raise instead
        return False


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
        """Queue one question. Refuses to queue past :data:`MAX_QUEUED`.

        **A cap and not a smaller queue**, which is the distinction the class
        docstring above is about: dropping the *second* question was the bug,
        and dropping the sixty-fifth is a backstop against a caller that has
        started asking from somewhere it should not. Sixty-four is two orders
        above the longest legitimate run -- a quit asks three -- so reaching it
        means a per-frame code path is calling this, which at 60 fps is an
        unbounded deque and a modal the user can never finish answering.
        Refused loudly, because a question silently dropped is exactly what
        this queue exists not to do.
        """
        if len(self._queue) >= MAX_QUEUED:
            log.warning(
                "refusing to queue a %d-th confirm (%r) -- something is asking "
                "from a per-frame path",
                len(self._queue) + 1,
                confirm.title,
            )
            return
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
        appearing = not confirm._open
        if appearing:
            imgui.open_popup(confirm.title)
            confirm._open = True
        centre = imgui.get_main_viewport().get_center()
        imgui.set_next_window_pos(centre, imgui.Cond_.appearing.value, (0.5, 0.5))
        # Fade and rise on the way in (UX.md Phase 1). Pushed around ``begin``
        # rather than inside it: the modal's own background is painted there,
        # and a dialog whose panel cuts in while its text fades up is worse
        # than the hard cut it replaced.
        alpha, rise = widgets.popover_enter(f"confirm/{confirm.title}", appearing)
        # Translucent over the app it is blocking (UX.md Phase 5): the
        # background is cleared before ``begin`` paints it and drawn back
        # below, blurred where there is a capture to blur and solid where
        # there is not.
        frosted = widgets.frosted()
        if frosted:
            imgui.set_next_window_bg_alpha(0.0)
        imgui.push_style_var(imgui.StyleVar_.alpha.value, alpha)
        radius = widgets.push_surface_rounding()
        opened, _ = imgui.begin_popup_modal(
            confirm.title, None, imgui.WindowFlags_.always_auto_resize.value
        )
        widgets.pop_surface_rounding()
        if not opened:
            imgui.pop_style_var()
            return
        # The deepest step of the ramp: a modal is the one surface that stops
        # the app underneath it, and the shadow is what says so before the text
        # is read.
        widgets.window_shadow("overlay", radius=radius)
        if frosted:
            widgets.window_backdrop(radius=radius)
        if rise > 0.0:
            imgui.dummy((0, rise))
        imgui.text_wrapped(confirm.message)
        if self.waiting:
            widgets.muted(f"{self.waiting} more to answer")
        if confirm.body is not None:
            imgui.dummy((0, sp(6)))
            confirm.body()
        imgui.dummy((0, sp(6)))
        # The action is red, the escape is neutral: two identical buttons make
        # a destructive question a coin toss.
        confirmed = widgets.destructive_button(confirm.confirm_label, (sp(BUTTON_W), 0))
        imgui.same_line()
        cancelled = controls.button(
            confirm.cancel_label,
            (sp(BUTTON_W), 0),
            role=controls.ButtonRole.SECONDARY,
            _imgui=imgui,
        )
        # Focus lands on the *safe* button, and Enter activates that one.
        #
        # It used to land on the destructive one, with Enter confirming it --
        # so a user dismissing the previous dialog with Enter, or simply
        # pressing it out of habit, deleted something (UX-07). "Where am I"
        # deserves an answer and Enter deserves a meaning; neither requires
        # that the answer be Delete. Destroying now takes an explicit act: a
        # click, or Tab to the red button and Space.
        #
        # Only on the frame the modal appears: re-focusing every frame would
        # fight the Tab key.
        if not confirm._focused:
            imgui.set_item_default_focus()
            confirm._focused = True
        # Esc cancels, Enter takes the safe way out. Both are read here rather
        # than left to imgui's own nav activation: ``setup_window`` sets
        # ``nav_enable_keyboard`` (for the focus drawing), but the app's
        # keyboard focus is ``focus.py``'s ring and not imgui's, so which
        # widget nav thinks is focused must not decide a modal's answer.
        if _escape_pressed() or _enter_pressed():
            cancelled = True
        if confirmed:
            self._answered()
            if confirm.on_confirm is not None:
                confirm.on_confirm()
        elif cancelled:
            self._answered()
        imgui.end_popup()
        imgui.pop_style_var()


@dataclass
class Prompt:
    """A one-line text question -- naming a pose, renaming a job."""

    title: str
    label: str
    value: str = ""
    on_accept: Any = None
    _open: bool = field(default=False, repr=False)
    #: :class:`Confirm`'s one-shot, for its reason. See the focus call.
    _focused: bool = field(default=False, repr=False)


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
        """:meth:`ConfirmQueue.ask`'s cap, for its reason."""
        if len(self._queue) >= MAX_QUEUED:
            log.warning(
                "refusing to queue a %d-th prompt (%r) -- something is asking "
                "from a per-frame path",
                len(self._queue) + 1,
                prompt.title,
            )
            return
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
        appearing = not prompt._open
        if appearing:
            imgui.open_popup(prompt.title)
            prompt._open = True
        centre = imgui.get_main_viewport().get_center()
        imgui.set_next_window_pos(centre, imgui.Cond_.appearing.value, (0.5, 0.5))
        alpha, rise = widgets.popover_enter(f"prompt/{prompt.title}", appearing)
        # Translucent over the app it is blocking (UX.md Phase 5): the
        # background is cleared before ``begin`` paints it and drawn back
        # below, blurred where there is a capture to blur and solid where
        # there is not.
        frosted = widgets.frosted()
        if frosted:
            imgui.set_next_window_bg_alpha(0.0)
        imgui.push_style_var(imgui.StyleVar_.alpha.value, alpha)
        radius = widgets.push_surface_rounding()
        opened, _ = imgui.begin_popup_modal(
            prompt.title, None, imgui.WindowFlags_.always_auto_resize.value
        )
        widgets.pop_surface_rounding()
        if not opened:
            imgui.pop_style_var()
            return
        widgets.window_shadow("overlay", radius=radius)
        if frosted:
            widgets.window_backdrop(radius=radius)
        if rise > 0.0:
            imgui.dummy((0, rise))
        # A prompt is a one-field form, so it follows the same label-above-
        # field pattern as every pane.  The headless interaction tests replace
        # only this module's ImGui binding; renderer-only typography is skipped
        # for that deliberately tiny backend.
        #
        # Asked as "is there a live context for ``widgets.field_label`` to draw
        # into", which is the actual precondition -- that function reaches for
        # ``widgets.imgui`` regardless of what this module was rebound to. It
        # used to be ``imgui is widgets.imgui``, an object-identity test, which
        # makes the *production* label conditional on two modules happening to
        # hold the same object: any tool or test that rebinds either one, for a
        # reason having nothing to do with this label, ships a form field with
        # no label at all, and no test exercises the production shape.
        if _has_context():
            widgets.field_label(prompt.label)
        imgui.set_next_item_width(sp(FIELD_W))
        # Only on the frame the modal appears -- ``Confirm``'s one-shot and its
        # reason. ``is_any_item_active`` is true of the *field*, so the old
        # spelling re-grabbed the keyboard on every frame the field was not
        # active, which is every frame after a Tab: the focus snapped straight
        # back and the buttons could not be reached from the keyboard at all.
        if not prompt._focused:
            imgui.set_keyboard_focus_here()
            prompt._focused = True
        # ``enter_returns_true`` makes the returned flag mean *Enter was
        # pressed*, not *the text changed* -- but the returned string is the
        # live buffer either way. Storing it only when the flag was set left
        # ``value`` at whatever it was seeded with for the whole modal, so
        # typing a name and clicking Save saved the old one (or, for a fresh
        # prompt, silently refused an empty string). Only Enter ever worked.
        entered, value = controls.input_text(
            "##prompt-value",
            prompt.value,
            imgui.InputTextFlags_.enter_returns_true.value,
            _imgui=imgui,
        )
        prompt.value = value
        if self.waiting:
            widgets.muted(f"{self.waiting} more to answer")
        # A blank name has always been refused -- silently. Save looked
        # enabled, clicking it closed nothing and did nothing, and the modal sat
        # there with no indication that the field was the problem (UX-24). The
        # refusal is the same; what is added is saying so and disabling the
        # control that cannot work.
        blank = not prompt.value.strip()
        if blank:
            # Through the module-local ``imgui`` rather than ``widgets.muted``,
            # deliberately: this file's tests replace ``dialogs.imgui`` with a
            # fake, and ``widgets`` reaches for the real module -- which, with
            # no imgui context in a headless test, is an access violation
            # rather than an error. The existing ``waiting`` note gets away
            # with it only because that branch is never taken in those tests.
            #
            # Opaque rather than ``text_disabled`` (UX-18): this is the one
            # sentence in the modal that says why Save will not work, so it is
            # the last copy in the app that should be drawn at 3.20:1.
            imgui.text_colored(imgui.ImVec4(*theme.rgba(theme.MUTED)), "Name required.")
        saved = controls.button(
            "Save",
            (sp(BUTTON_W), 0),
            role=controls.ButtonRole.PRIMARY,
            enabled=not blank,
            reason="Enter a name first.",
            _imgui=imgui,
        )
        accepted = (entered or saved) and not blank
        imgui.same_line()
        cancelled = (
            controls.button(
                "Cancel",
                (sp(BUTTON_W), 0),
                role=controls.ButtonRole.GHOST,
                _imgui=imgui,
            )
            or _escape_pressed()
        )
        if accepted and prompt.value.strip():
            self._answered()
            if prompt.on_accept is not None:
                prompt.on_accept(prompt.value.strip())
        elif cancelled:
            self._answered()
        imgui.end_popup()
        imgui.pop_style_var()


def modal_open(ctx: Any) -> bool:
    """Whether *any* modal is on screen and owns the keyboard.

    It used to know about exactly two: the confirm queue and the prompt queue.
    The matte preview is a third -- a real modal, drawn in front of the
    promotion, with its own Accept and Cancel -- and every global shortcut
    leaked straight through it (UX-08). Ctrl+K opened the palette behind it;
    Ctrl+Enter submitted the form the modal was a *question about*; a mode key
    left the app somewhere else with the modal still up. Ownership is a
    property of "a modal is up", not of which queue happens to hold it, so the
    predicate asks all four.

    A module function with ``App._modal_open`` delegating to it, because the
    guided tour needs the same question and is deliberately *not* one of the
    answers: ``panes/tour.py`` suspends its scrim while a modal is up, and a
    second copy of this list would be a copy that stopped agreeing the next
    time a modal was added.

    **Here rather than in the frame loop**, which is where it used to live: the
    tour is a pane, and importing ``main`` for one predicate made a leaf depend
    on the shell. This module already owns two of the four answers.
    """
    from . import matte_preview
    from .panes import first_run

    return (
        ctx.confirms.pending is not None
        or ctx.prompts.pending is not None
        or matte_preview.is_open(ctx)
        or first_run.is_open(ctx)
    )
