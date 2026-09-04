"""The rules every document mode shares, in one place.

Inker, Clay, Plotter and Packwright are four editors over four document types,
and they are the same *program* four times: a tab list, a save that is a state
rather than a call, a texture cache keyed by tab, a question asked before
unsaved work is lost. Each of those grew its own copy, and near-identical copies
of a rule are not merely untidy -- they are four places for the rule to change
in three of them. This module holds the ones that are genuinely one rule; the
ones that differ on purpose stay where they are, and say why there.

:func:`viewer_guard` is the same question asked over a *viewer* instead of a
document list -- what the two pose editors ask, where the unsaved work lives in
a ``PoseEditor`` rather than in tabs.

**Import discipline is what makes this importable from anywhere.** stdlib,
typing and numpy at module scope and nothing else: ``dialogs`` is imported
inside :func:`guard` and :func:`viewer_guard`, ``imgui_backend`` inside
:func:`forget_texture`, and PIL inside :func:`decode_rgba` -- so a ``*_state``
module can reach for :func:`title_for` without dragging a window in, and the
lazy-Pillow rule the engines follow holds here too.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def start_save(ctx: Any, tab: Any, key: str, run: Any) -> None:
    """Submit a save, and unlock the tab if the runner refused it.

    The runner refuses a key that is already in flight. Leaving the flag set
    there is what makes a tab read-only forever after a double press.
    """
    tab.saving = True
    if not ctx.submit(key, run):
        tab.saving = False


def title_for(path: Path | None) -> str:
    """A tab's title from its file. ``clay_state`` keeps its own, on ``stem``:
    a Clay tab is named for the document rather than for the file."""
    return path.name if path is not None else "Untitled"


def decode_rgba(path: Path) -> np.ndarray:
    """One image file as RGBA. Task thread only.

    Through :mod:`.pixelguard` rather than a bare ``Image.open`` because the
    ceiling has to be asked *before* ``convert``, which is the call that
    allocates -- ``packwright/wpack.py`` states that rule verbatim at the one
    door that already had it. This is the door every other mode reaches an
    image through, so it is the one place the rule buys the most.
    """
    from . import pixelguard

    return pixelguard.decode_rgba(path, f"{Path(path).name}")


def forget_texture(texture: Any) -> None:
    """Unregister a texture with the imgui backend, then release it.

    Both halves, in this order. An id the renderer does not know maps to no
    moderngl object and the image comes out as the font atlas; releasing first
    leaves the backend holding a dead object under a GL name the driver will
    reuse.
    """
    from . import imgui_backend

    renderer = imgui_backend.current()
    # Headless/editor callers can hold texture-shaped objects with release()
    # but no GL name.  Still give a renderer the first chance to forget them
    # (recording/test renderers deliberately support those objects), while an
    # ambient real GL renderer must not prevent their release.
    if renderer is not None and (
        hasattr(texture, "glo") or not isinstance(renderer, imgui_backend.ImguiRenderer)
    ):
        renderer.forget_texture(texture)
    texture.release()


def release_prefix(ctx: Any, prefix: str) -> None:
    """Drop every ``state.preview`` entry under ``prefix``, releasing textures.

    The list is materialised before the loop because the pop mutates the dict,
    and ``hasattr(value, "release")`` is what separates a texture from the
    plain stamps stored alongside it under the same prefix.
    """
    for key in [k for k in list(ctx.state.preview) if k.startswith(prefix)]:
        value = ctx.state.preview.pop(key, None)
        if value is not None and hasattr(value, "release"):
            forget_texture(value)


#: The five document modes, by the ``AppState`` attribute each keeps its tabs
#: on. The quit chain walks these in order; :func:`any_unsaved` asks all five
#: the one question the window caption is about.
DOC_MODES: tuple[str, ...] = ("inker", "clay", "plotter", "packwright", "sirens")


def any_unsaved(ctx: Any) -> bool:
    """Whether *any* document in *any* mode has unsaved changes.

    ``getattr`` rather than each mode's ``ensure``, for :func:`guard`'s reason:
    asking must not create the state that says no.
    """

    for attr in DOC_MODES:
        state = getattr(ctx.state, attr, None)
        if state is not None and state.any_dirty:
            return True
    return False


def guard(ctx: Any, attr: str, singular: str, plural: str, verb: str, proceed: Any) -> bool:
    """Ask before losing unsaved work. -> whether it went ahead now.

    One question for all of them: ``ConfirmQueue`` really queues (I78), so
    asking per dirty document would not lose any -- it would park the user in
    front of a run of modals they answer the same way each time.

    ``getattr`` rather than the mode's ``ensure``: asking which documents are
    unsaved must not create the state that says none is, which is what the quit
    chain relies on before a mode has ever been opened.
    """
    from . import dialogs

    state = getattr(ctx.state, attr)
    if state is None or not state.any_dirty:
        proceed()
        return True
    count = sum(1 for doc in state.docs if doc.dirty)
    what = f"one {singular} has" if count == 1 else f"{count} {plural} have"
    ctx.confirms.ask(
        dialogs.Confirm(
            title="Discard unsaved work?",
            message=f"{what[0].upper()}{what[1:]} unsaved changes, which will be lost"
            f" if you {verb}.",
            on_confirm=proceed,
        )
    )
    return False


def pose_undo_key(viewer: Any, event: Any) -> bool:
    """Ctrl+Z / Ctrl+Shift+Z / Ctrl+Y over a pose editor. -> consumed?

    Here for ``viewer_guard``'s reason: the two entry points into pose editing
    -- Poser's authoring session and the inspector's asset pose mode -- are
    different viewers and different keyboard doors, and the binding is the same
    one over the same ``PoseEditor``. Written twice it would drift, and the
    half that drifted would be the inspector's, which is the one nobody opens
    on purpose.

    Both spellings of redo, because both are in the app already: Inker binds
    Ctrl+Shift+Z and Clay binds Ctrl+Y, and a user who learned either in one
    mode should not discover the other is the one that works here.

    Returns False for everything else, so the caller falls through to the
    bindings that were there before -- this is added *in front of* the 2D/3D
    keyboard, not in place of it.
    """
    import pygame

    if event.type != pygame.KEYDOWN or not event.mod & pygame.KMOD_CTRL:
        return False
    if viewer is None or not viewer.pose_mode or not viewer.editor.bound:
        return False
    editor = viewer.editor
    if event.key == pygame.K_z and not event.mod & pygame.KMOD_SHIFT:
        moved = editor.undo()
    elif event.key == pygame.K_y or (event.key == pygame.K_z and event.mod & pygame.KMOD_SHIFT):
        moved = editor.redo()
    else:
        return False
    if moved:
        # Everything a pose change owes the rest of the app: the skinning
        # palettes hold the old matrices, the dirty indicator is derived, and
        # the frame is stale. ``_after_pose_change`` is the one call that does
        # all three, and the gizmo drag already goes through it.
        after = getattr(viewer, "_after_pose_change", None)
        if after is not None:
            after()
    # Consumed either way. A Ctrl+Z with an empty stack must not fall through
    # to whatever Z means in the mode underneath.
    return True


def viewer_guard(ctx: Any, viewer: Any, noun: str, verb: str, proceed: Any) -> bool:
    """:func:`guard`'s rule over a pose viewer. -> whether it went ahead now.

    The two pose editors keep their unsaved work in a ``PoseEditor`` rather
    than in a tab list, but they ask the same question in the same words. The
    caller passes the viewer *and* the noun because which viewer is the right
    one is the whole distinction between them -- the Poser reads its own
    instance, the inspector reads the shared one, and no edit can live in both
    -- and the inspector's noun changes with the editor's mode.
    """
    from . import dialogs

    if viewer is None or not viewer.pose_mode or not viewer.editor.has_unsaved_edits():
        proceed()
        return True
    ctx.confirms.ask(
        dialogs.Confirm(
            title="Discard unsaved changes?",
            message=f"Unsaved {noun} changes will be lost if you {verb}.",
            on_confirm=proceed,
        )
    )
    return False
