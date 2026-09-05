"""Where a drawing leaves the Inker: the four service-backed verbs, as buttons.

They have always existed -- ``Make 3D``, ``Save as reference``, ``Add to
Packwright`` and ``Revert to original`` are registered ops in
:mod:`~warlock.studio.inker_ops` -- but they were rows in the File menu and
nowhere else. A user who has drawn something and wants a mesh out of it looks
at the panel column, not at a menu, and every *other* workspace answers that
look: Clay has its bridge pane, Plotter has its files pane, Packwright has its
export pane. This is the Inker's.

**No new pipeline work.** Each button dispatches the same op the menu row does,
through :func:`inker_menu.activate`, so a parameterised op still opens its
sheet and ``pending_dialog`` still works.

Generating *into* a layer was called "a separate programme, deliberately not
started here" when this pane was written. It shipped on 2026-08-30 and it is
not one of these buttons: masked regeneration lives on the Edit menu and in
:mod:`~warlock.studio.inker.inpaint`, because it acts on a *selection* inside
the open document rather than sending the document somewhere. The four verbs
below are still the ones that hand a drawing to another workspace, which is
what this pane is for.

Every button is a :func:`widgets.disabled_button` carrying
``inker_ops.reason_for``: this pane's whole reason for existing is that a verb
should be visible before it is available, and a greyed button with no sentence
is worse than a menu row.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from imgui_bundle import imgui

from .. import anchors, inker_mode, inker_ops, tokens, widgets
from ..manual import render as manual_render
from ..tokens import sp
from . import inker_menu

#: The least this pane may be squeezed to, in design px: the file header's
#: two rows and status line, the heading, four full-width buttons and the
#: link line under them.
GENERATE_FLOOR = 270.0

#: The ops, in the order the File menu registers them, with the one-line note
#: that says what each is *for*. The names are checked against the registry at
#: import time by :func:`ops`, so a rename cannot leave a dead button here.
OPS: tuple[tuple[str, str], ...] = (
    ("send_to_3d", "Reconstruct a mesh from this drawing."),
    ("save_as_reference", "Put it in the library as a reference image."),
    ("add_to_packwright", "Send it to the atlas packer as a source."),
    ("revert", "Throw away the edits and reopen what was there before."),
)


def ops() -> list[Any]:
    """The four ops, resolved. ``KeyError`` here rather than a dead button."""

    return [inker_ops.get(name) for name, _note in OPS]


def draw(ctx: Any) -> None:
    anchors.mark_window("inker/generate")
    state = inker_mode.ensure(ctx)
    tab = state.active
    if tab is not None:
        # The shared document header every workspace carries (2026-09-05).
        # Inker was the one mode with no file block in any pane -- its verbs
        # were menu rows only -- so a user who had just met Save in Clay's
        # pane looked here and found nothing.
        widgets.section("Drawing file")
        widgets.document_header(
            tab,
            new=lambda: inker_mode.new_document(ctx, 1024, 1024),
            open_=lambda: inker_mode.ask_open(ctx),
            save=lambda: inker_mode.save(ctx, tab),
            save_as=lambda: inker_mode.save_as(ctx, tab),
            saving=tab.busy,
        )
        imgui.dummy((0, sp(tokens.SP_2)))
        # The Undo/Redo pair and the history popover every other bridge
        # draws. Four bridges' comments said "while Inker drew the same pair
        # twice"; it drew it nowhere (2026-09-05).
        widgets.history_block(
            ctx,
            tab,
            key="inker",
            undo=lambda: tab.doc.undo(),
            redo=lambda: tab.doc.redo(),
            step=lambda index: inker_mode.step_history(ctx, tab, index),
        )
        imgui.dummy((0, sp(tokens.SP_2)))
    # Inker's exits -- Make 3D, Save as reference, Add to Packwright -- under
    # the heading every workspace puts over the same verbs.
    widgets.exits()
    # After the heading, never before it: ``help_button`` is a ``same_line``.
    manual_render.help_button(ctx, "inker-generate")

    for op, (_name, note) in zip(ops(), OPS, strict=True):
        enabled = bool(op.enabled(state, tab))
        if widgets.disabled_button(
            f"{op.label}##inkgen/{op.name}",
            enabled,
            (-1, 0),
            reason=inker_ops.reason_for(op, state, tab),
            tooltip=note,
        ):
            inker_menu.activate(ctx, op)
        widgets.muted(note)
        imgui.dummy((0, sp(4)))

    _link(tab)
    if tab is None:
        # The recent list, which every other bridge offers and this one did
        # not; ``inker_mode.recent_paths`` existed with no pane reading it.
        widgets.recent_files(
            inker_mode.recent_paths(ctx),
            lambda path: inker_mode.open_path(ctx, Path(path)),
        )


def link_line(tab: Any) -> str:
    """What this document is attached to, in one sentence.

    Pure and public because it is the *explanation* of a greyed Revert, and an
    explanation is worth an assertion rather than a screenshot.
    """
    if tab is None:
        return "No drawing is open."
    if not tab.linked:
        return "Not linked to a library asset -- Save as reference makes one."
    kind = tab.link_kind or "job"
    original = "with the original kept" if tab.has_original else "no original kept yet"
    return f"Linked to {kind} {tab.job_id} ({original})."


def _link(tab: Any) -> None:
    widgets.muted(link_line(tab))
