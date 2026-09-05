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

import os
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


def find_path(docs: Any, path: Path) -> Any:
    """The already-open tab over ``path``, so opening twice focuses rather
    than forking -- two tabs over one file would race on save.

    **Resolved and case-folded, because on Windows one file has many
    spellings.** ``Level.WMAP`` and ``level.wmap`` are the same file and
    ``Path.__eq__`` says they are not, so the same document opened from the
    recents list and from a drop forked into two tabs that then raced on save
    -- exactly what the method exists to prevent. ``resolve`` folds the other
    spellings (a relative path, a symlink, an 8.3 short name); ``normcase`` is
    what makes the comparison case-insensitive *where the filesystem is*, and
    a no-op where it is not.

    Plotter fixed this on its own copy and wrote, in its docstring, that Clay
    and Packwright carried the same bug and fixing them "is deliberately not
    this change". Inker and Sirens carried it too and were not named. One
    body now, and every ``*State.find_path`` delegates here (2026-09-05).
    """
    probe = os.path.normcase(str(Path(path).resolve()))
    for doc in docs:
        if doc.path is None:
            continue
        if os.path.normcase(str(Path(doc.path).resolve())) == probe:
            return doc
    return None


#: The Ctrl chords **every** document mode refuses while its save is writing:
#: the two that move the history head the save captured, and the export, which
#: flattens or serialises the live document -- the same read a save makes and
#: just as wrong to take while one is in flight. Inker alone guarded ``e``;
#: each mode adds its own restructuring chords on top (2026-09-05).
WRITE_CHORDS = frozenset({"z", "y", "e"})


def blocked_while_writing(tab: Any, name: str, chords: frozenset[str] = WRITE_CHORDS) -> bool:
    """Whether Ctrl+``name`` waits for the tab's save. ``busy`` where the tab
    has a second reason (Inker's playback), ``saving`` otherwise."""
    if name not in chords:
        return False
    return bool(getattr(tab, "busy", getattr(tab, "saving", False)))


def refuse(
    ctx: Any, text: str, *, action: str | None = None, action_arg: str | None = None
) -> None:
    """A gesture's refusal: one sentence, coalesced, with its remedy if one exists.

    Inker has a tip strip under its canvas (``InkerState.say``) and a remedy
    button on it; the other workspaces have toasts. What they did not have was
    one *rule*: Clay wrapped ``ctx.toast(msg, "error")`` in ``_toast``, Sirens
    called it bare, Plotter and Packwright called it bare and unconditionally
    -- so a shape tool refused once per vertex stacked a copy per click, which
    is the stacking ``toast_once`` was written to stop. Every refusal that is
    not Inker's goes through here (2026-09-05): error level, coalesced, and an
    ``action`` where the refusal has an undo (``plotter_mode._locked_toast``'s
    Unlock).
    """
    once = getattr(ctx, "toast_once", None)
    if once is not None:
        once(text, "error", action, action_arg)
    elif action is not None:
        ctx.toast(text, "error", action=action, action_arg=action_arg)
    else:
        ctx.toast(text, "error")


#: What a tab's x says while its save is still writing. One sentence for
#: every mode: Clay refused silently and the other four did not refuse at all.
CLOSE_WHILE_SAVING = "Still saving -- close it once the save lands."


def close_tab(ctx: Any, state: Any, uid: str, release: Any) -> None:
    """Close one document, asking first if it has unsaved work.

    The one shape every mode had copied: forget the crash copy, release what
    the document owns in the single GL context (``release``, the per-mode
    part), take it out of the tab list, and ask ``ask_close_unsaved`` first if
    it is dirty. Its own question rather than :func:`guard`, because ``guard``
    asks about every dirty document in the workspace and the answer sought
    here is about *this* one.

    **Refused while the tab is saving, out loud.** The serialise task reads
    the live document on a task thread, and closing it out from under that
    read is the one way to lose the file being written rather than merely the
    edits since. Clay alone had the refusal (silently -- the x did nothing);
    Inker, Plotter, Packwright and Sirens encode the live document the same
    way and let the tab go (2026-09-05 consistency pass).

    ``journal.drop`` rather than leaving the copy: the document is on disk
    under a name the user chose, or is gone from the session -- either way the
    crash copy describes work that is no longer at risk, and one left behind
    is exactly the file that gets offered back after a clean session and
    confuses somebody (UX-05).
    """
    tab = state.get(uid)
    if tab is None:
        return
    if getattr(tab, "saving", False):
        ctx.toast(CLOSE_WHILE_SAVING, "info")
        return

    def drop() -> None:
        from . import journal

        journal.drop(ctx, tab)
        release(tab)
        state.close(uid)

    if not getattr(tab, "dirty", False):
        drop()
        return
    from . import dialogs

    dialogs.ask_close_unsaved(ctx, tab.title, drop)


def tab_label(tab: Any) -> str:
    """What imgui draws on a tab. The id after ``###`` is what it *matches*
    on, so the visible part is free to change without moving the tab. Five
    ``label`` properties carried this line, three with this docstring."""
    return f"{tab.title}###{tab.uid}"


def recents_for(mode: str) -> tuple[Any, Any, Any]:
    """``(remember_path, forget_path, recent_paths)`` for one mode.

    Through :mod:`.recents` rather than onto a field of the mode's own state:
    the document modes kept independent ``recent`` lists, and Home's single
    Resume list cannot be built from them at all -- bare path lists carry no
    ordering *between* them. There is one list, and five modes each wrote the
    same three wrappers over it, keyed on their own name (2026-09-05). A
    caller that turned out not to open a path forgets it, :mod:`.recents`' own
    rule, without having to know the mode's kind string.
    """
    from . import recents

    def remember_path(ctx: Any, path: Any) -> None:
        recents.remember(ctx.settings, mode, path)

    def forget_path(ctx: Any, path: Any) -> None:
        recents.forget(ctx.settings, mode, path)

    def recent_paths(ctx: Any) -> list[str]:
        return recents.paths(ctx.settings, mode)

    return remember_path, forget_path, recent_paths


def save(tab: Any, *, save_as: Any, save_to: Any) -> None:
    """Ctrl+S: Save As when the document has never been written anywhere,
    and nothing at all while a save is already in flight -- the runner
    refuses a key already running, and the second press would otherwise
    leave ``saving`` set forever (``start_save``'s rule). Four modes carried
    this body verbatim; Inker adds a pre-step and then delegates here."""
    if tab is None or getattr(tab, "saving", False):
        return
    if tab.path is None:
        save_as()
        return
    save_to()


def tab_bar(ctx: Any, state: Any, bar_id: str, close: Any) -> None:
    """The document tabs above a workspace's centre pane.

    Five panes drew this byte for byte but for the bar id and the close call.
    ``reorderable`` and ``auto_select_new_tabs``: without the second, a second
    opened document lands behind the first and "Open" looks inert. The dirty
    mark is imgui's own dot (``unsaved_document``) rather than a ``"* "`` in
    the title, because the title is half of the tab's identity and decorating
    it would move the tab. ``close`` takes the tab; every mode's close door
    goes through :func:`close_tab`.
    """
    from imgui_bundle import imgui

    if not state.docs:
        return
    flags = imgui.TabBarFlags_.reorderable.value | imgui.TabBarFlags_.auto_select_new_tabs.value
    if not imgui.begin_tab_bar(bar_id, flags):
        return
    for tab in list(state.docs):
        item_flags = imgui.TabItemFlags_.unsaved_document.value if tab.dirty else 0
        opened, keep = imgui.begin_tab_item(tab.label, True, item_flags)
        if opened:
            state.activate(tab.uid)
            imgui.end_tab_item()
        if not keep:
            close(tab)
    imgui.end_tab_bar()


def mark_recovered(tab: Any, path: Any, doc: Any = None) -> None:
    """A recovered document reads dirty until the user saves it somewhere.

    The reader hands back a document already marked saved, and a clean
    recovered tab closes without a confirm -- taking the journal copy, the
    only surviving copy of the work, with it. Where ``dirty`` delegates to
    the document (Plotter, Packwright, Sirens) the never-matching head goes on
    the *document* and the tab's mirror follows so ``mark_saved`` keeps them in
    step; Clay's and Inker's tabs hold the head themselves. The copy's name is
    recorded so saving or closing the tab is what clears it. Written out five
    times before 2026-09-05, three of them with this paragraph.
    """
    if doc is not None:
        doc.saved_head = -1
    tab.saved_head = -1
    tab.journal_name = Path(path).name


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
