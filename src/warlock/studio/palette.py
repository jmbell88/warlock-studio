"""The command palette's data half: what the commands are, and what matches.

Pure in the way :mod:`~warlock.vram` and :mod:`~warlock.memlog` are -- stdlib
only, no imgui, no moderngl, no pygame -- so every rule about *which* command a
query finds is assertable without a window. The drawing half is
:mod:`.panes.palette`, and it owns no list of its own.

Two things here are deliberate and easy to undo.

**The mode commands are derived from** :data:`.modes.MODES`, not written out.
A palette is a second index of everything the app can do, and a hand-written
one is a second index that drifts -- a thirteenth mode would gain a switch
segment and be missing from the one surface whose entire job is telling the
user what exists. It is also, since the positional Alt+digit bindings went
away, the *only* keyboard route to a mode, which is what turns the derivation
from tidiness into a requirement.
For the same reason what a mode command *does* is :func:`.state.set_mode` and
not four lines of its own: that copy existed, and it had already drifted.

**Every command carries ``enabled``, and a disabled command is *listed*.** The
tempting version filters the list to what can run right now, which makes the
palette a worse menu: a user searching for "wireframe" from Home learns
nothing from an empty result, and learns where to go from a greyed row that
says which mode it belongs to. The one thing a disabled row may not do is run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import modes, state

# How many assets the quick-open section offers at once. A palette is a
# shortlist: past a handful the eye is scanning rather than recognising, and
# the list is the newest N of M anyway, so a longer one would still not be
# "every asset".
MAX_ASSETS = 8


@dataclass(frozen=True)
class Command:
    """One thing the palette can do.

    ``key`` is stable and is what a test names; ``label`` is what the user
    reads and may be reworded freely. ``group`` is a heading, ``hint`` is the
    keyboard shortcut.

    ``why`` is the reason the row is greyed, and it is the half of the module
    docstring's argument that was never built: "learns where to go from a
    greyed row that says which mode it belongs to" was the whole case for
    listing disabled commands, and the row said nothing -- it greyed out and
    Enter on it returned in silence. A command with ``enabled`` and no ``why``
    is the gap, not the norm.
    """

    key: str
    label: str
    group: str
    run: Callable[[Any], None]
    hint: str = ""
    enabled: Callable[[Any], bool] = field(default=lambda _ctx: True)
    why: str = ""


# --- matching ----------------------------------------------------------------


def match(query: str, text: str) -> int | None:
    """Score ``text`` against ``query``, or ``None`` for no match.

    A subsequence match, case-insensitive: "wf" finds "Toggle wireframe" and
    "gtc" finds "Go to Clay". Higher is better. The scoring is deliberately
    crude and deliberately *stated*, because the alternative -- ranking by
    whatever order the commands happen to be built in -- makes the first row a
    property of this file's layout rather than of what was typed, and the first
    row is the one Enter runs.

    Three things earn points, in the order they matter: a prefix match on the
    whole string, a match that starts at a word boundary, and adjacency between
    consecutive matched characters. A shorter haystack breaks ties, so "Clay"
    outranks "Edit in Clay" for the query "clay".
    """
    if not query:
        return 0
    needle = query.casefold()
    hay = text.casefold()
    if needle in hay:
        # A contiguous hit always beats a scattered one, and a prefix beats
        # anything inside the string.
        base = 1000 if hay.startswith(needle) else 700
        if hay.startswith(needle) or hay[hay.index(needle) - 1] in " -/:":
            base += 100
        return base - len(hay)
    score = 0
    position = 0
    previous = -2
    for char in needle:
        found = hay.find(char, position)
        if found < 0:
            return None
        if found == 0 or hay[found - 1] in " -/:":
            score += 20
        if found == previous + 1:
            score += 10
        previous = found
        position = found + 1
        score += 1
    return score - len(hay)


def rank(query: str, items: list[Any], text_of: Callable[[Any], str]) -> list[Any]:
    """Everything that matches, best first, stable within a score.

    Stable because the input order is meaningful -- commands arrive grouped,
    assets arrive newest first -- and an unstable sort would reshuffle equal
    rows as the user types a character that changes nothing.
    """
    scored = []
    for index, item in enumerate(items):
        value = match(query, text_of(item))
        if value is not None:
            scored.append((-value, index, item))
    scored.sort(key=lambda row: (row[0], row[1]))
    return [item for _score, _index, item in scored]


# --- the commands ------------------------------------------------------------


def _go(key: str) -> Callable[[Any], None]:
    """A mode command, through the app's one mode switch.

    It used to be a second copy of ``App._set_mode``'s four lines, and the two
    had already drifted: this one had no early return, so choosing the mode you
    were already in recorded ``previous_mode == mode`` and the next Esc from a
    pass-through mode fell back to Home instead of to the work you left.
    """

    def run(ctx: Any) -> None:
        state.set_mode(ctx.state, key)

    return run


def _generate(ctx: Any) -> None:
    from . import create_stages
    from .panes import settings_2d, settings_3d

    if create_stages.at(ctx.state, "reference"):
        settings_2d.generate(ctx, ctx.state.form_2d)
    else:
        settings_3d.promote(ctx, ctx.cache.get(ctx.state.source_job), ctx.state.form_3d)


def _in_generate_mode(ctx: Any) -> bool:
    from . import create_stages

    return create_stages.in_create(ctx.state)


def _viewport(ctx: Any) -> bool:
    return ctx.state.mode in modes.VIEWPORT_MODES


# Shared by the three viewport toggles: one sentence, so they cannot drift into
# three different accounts of the same gate.
_VIEWPORT_WHY = "Only in the 2D and 3D panes, which are where the viewport is."


def _selected(ctx: Any) -> Any:
    return ctx.cache.get(ctx.state.selected)


# --- the document modes, as one table ----------------------------------------
#
# Save, Save as, Export and Undo/Redo are four commands whose whole content is
# "whichever document mode is in front". Written out per mode they would be
# sixteen commands, four of which would be forgotten the next time a mode is
# added; written as a dispatch they are four, and the dispatch is one line per
# mode.
#
# The entry points are the modes' *own*: ``inker_mode.save`` and the rest are
# what the bridge buttons and the Ctrl+S handlers already call, so the palette
# cannot drift into a second way of saving. Undo goes to the document's own
# ``undo()``, which is uid-addressed inside -- the palette never touches a
# history directly.
_DOC_MODES: dict[str, tuple[str, str]] = {
    # mode -> (module name, export command label)
    "inker": ("inker_mode", "Export PNG"),
    "clay": ("clay_mode", "Export to the library"),
    "plotter": ("plotter_mode", "Export .tmx"),
    "packwright": ("packwright_mode", "Export atlas + JSON"),
}

_DOC_WHY = "Open a drawing, model, map or atlas first."


def _doc_mode(ctx: Any):
    """``(module, tab)`` for the document mode in front, or ``(None, None)``."""
    from importlib import import_module

    entry = _DOC_MODES.get(ctx.state.mode)
    if entry is None:
        return None, None
    module = import_module(f".{entry[0]}", __package__)
    return module, module.active(ctx)


def _has_doc(ctx: Any) -> bool:
    _module, tab = _doc_mode(ctx)
    return tab is not None


def _doc_save(ctx: Any) -> None:
    module, tab = _doc_mode(ctx)
    if tab is not None:
        module.save(ctx, tab)


def _doc_save_as(ctx: Any) -> None:
    module, tab = _doc_mode(ctx)
    if tab is not None:
        module.save_as(ctx, tab)


def _doc_export(ctx: Any) -> None:
    module, tab = _doc_mode(ctx)
    if tab is None:
        return
    # Each mode's *primary* export, which is the one its bridge puts first and
    # the one Ctrl+E is bound to. The others stay panel-only: a palette listing
    # four exports per mode is a list nobody reads.
    if ctx.state.mode == "inker":
        module.export_png(ctx, tab)
    elif ctx.state.mode == "clay":
        module.export_asset(ctx, tab)
    elif ctx.state.mode == "plotter":
        module.export_map(ctx, "tmx", tab)
    else:
        module.export_files(ctx, tab)


def _doc_export_label(ctx: Any) -> str:
    entry = _DOC_MODES.get(ctx.state.mode)
    return entry[1] if entry else "Export"


def _doc_history(ctx: Any):
    """The active document's undo stack, or None."""
    _module, tab = _doc_mode(ctx)
    doc = getattr(tab, "doc", None)
    return getattr(doc, "history", None)


def _can_undo(ctx: Any) -> bool:
    history = _doc_history(ctx)
    return bool(history is not None and history.can_undo)


def _can_redo(ctx: Any) -> bool:
    history = _doc_history(ctx)
    return bool(history is not None and history.can_redo)


def _doc_undo(ctx: Any) -> None:
    _module, tab = _doc_mode(ctx)
    if tab is not None and getattr(tab, "doc", None) is not None:
        tab.doc.undo()


def _doc_redo(ctx: Any) -> None:
    _module, tab = _doc_mode(ctx)
    if tab is not None and getattr(tab, "doc", None) is not None:
        tab.doc.redo()


def _mode_commands() -> list[Command]:
    """One per switch segment, in the switch's order.

    Derived rather than listed -- see the module docstring. No hint: there is
    no per-mode key to advertise any more, and a hint naming Ctrl+K would be
    the palette telling you how to open the palette you are looking at.
    """
    return [
        Command(key=f"go:{key}", label=f"Go to {label}", group="Go to", run=_go(key))
        for key, label, _icon in modes.MODES
    ]


def commands(ctx: Any) -> list[Command]:
    """Every command, in a sensible reading order.

    Takes ``ctx`` so a future command can be present only where it exists at
    all (as the rig actions are, on a host with no bpy) -- as opposed to being
    merely *disabled*, which is the default and is what almost everything here
    wants.
    """
    from .panes import library

    def reroll(ctx: Any) -> None:
        from ..service import jobs as svc_jobs

        job = _selected(ctx)
        if job is not None:
            ctx.submit(
                f"rerun:{job['id']}", svc_jobs.rerun_job, ctx.svc, job["id"], mode="reroll"
            )

    def rerollable(ctx: Any) -> bool:
        job = _selected(ctx)
        return bool(
            job
            and job["status"] in ("done", "error", "cancelled")
            and not (job["kind"] == "image" and job.get("stage") == "reference")
        )

    def delete(ctx: Any) -> None:
        from . import dialogs

        job = _selected(ctx)
        if job is None:
            return
        ctx.confirms.ask(
            dialogs.Confirm(
                title="Delete this asset?",
                message="The job and everything derived from it are removed from disk.",
                confirm_label="Delete",
                cancel_label="Keep",
                on_confirm=lambda: library.delete_asset(ctx, job["id"]),
            )
        )

    def new_drawing(ctx: Any) -> None:
        from . import inker_mode

        inker_mode.new_document(ctx, 1024, 1024)

    def new_clay(ctx: Any) -> None:
        from . import clay_mode

        clay_mode.new_document(ctx)

    def new_atlas(ctx: Any) -> None:
        from . import packwright_mode

        packwright_mode.new_document(ctx)

    def wireframe(ctx: Any) -> None:
        ctx.state.wireframe = not ctx.state.wireframe
        if ctx.viewer is not None:
            ctx.viewer.set_wireframe(ctx.state.wireframe)

    def turntable(ctx: Any) -> None:
        ctx.state.turntable = not ctx.state.turntable
        if ctx.viewer is not None:
            ctx.viewer.set_turntable(ctx.state.turntable)

    def frame(ctx: Any) -> None:
        if ctx.viewer is not None:
            ctx.viewer.frame()

    def clear_viewport(ctx: Any) -> None:
        # Through the App's helper rather than ``viewer.clear()``, because
        # emptying the viewer is only half of it -- the pin against
        # ``_sync_viewer`` is the other half, and a palette command that cleared
        # the canvas for one frame would look broken rather than useful.
        if ctx.clear_viewport is not None:
            ctx.clear_viewport()

    def fps(ctx: Any) -> None:
        ctx.state.show_fps = not ctx.state.show_fps

    def new_map(ctx: Any) -> None:
        from . import plotter_mode

        plotter_mode.new_document(ctx)

    def quit_app(ctx: Any) -> None:
        # Through the app's own guard, never straight to the exit: it is the
        # chain that asks about unsaved pixels, geometry and a pose in turn,
        # and a palette entry that bypassed it would be the one way out of the
        # app that loses work.
        ask = getattr(ctx, "ask_quit", None)
        if ask is not None:
            ask()

    def open_log(ctx: Any) -> None:
        ctx.open_log()

    def show_trash(ctx: Any) -> None:
        ctx.state.filters.trash = not ctx.state.filters.trash
        state.set_mode(ctx.state, "library")

    def empty_trash(ctx: Any) -> None:
        from ..service import jobs as svc_jobs
        from . import dialogs

        ctx.confirms.ask(
            dialogs.Confirm(
                title="Empty the trash?",
                message="Every trashed asset and everything derived from it is "
                "removed from disk. This cannot be undone.",
                confirm_label="Empty",
                cancel_label="Cancel",
                on_confirm=lambda: ctx.submit("empty-trash", svc_jobs.empty_trash, ctx.svc),
            )
        )

    def manual(ctx: Any) -> None:
        # The Manual stopped being a mode in the UI redesign, wave 3, so
        # ``_mode_commands`` no longer derives an entry for it -- and a
        # reference reachable only by a function key is a reference most people
        # do not know exists. This is that door, kept in the palette where
        # every other whole-app action is.
        from .manual import render as manual_render

        manual_render.open_at(ctx, (ctx.state.manual.chapter, None))

    def profiles(ctx: Any) -> None:
        # Replaces the derived ``go:profiles``: the manager is a sheet over
        # the Reference stage's pane now rather than a mode, and it is *about*
        # that pane's form, so opening it from anywhere else means going there
        # first.
        from . import create_stages
        from .panes import profiles_panel

        create_stages.go(ctx, "reference")
        profiles_panel.open_sheet(ctx)

    def diagnostics(ctx: Any) -> None:
        # The rail's badge is drawn only when a check is failing, so on a
        # healthy install there is nothing to click -- and "everything is fine"
        # is exactly the claim somebody occasionally wants the evidence for.
        # A one-shot for ``shortcuts``' reason: a palette command is not inside
        # the window the popup is registered in.
        from . import rail

        rail.request("diagnostics")

    def shortcuts(ctx: Any) -> None:
        # A flag rather than a call, because a palette command cannot open an
        # imgui popup: the palette's own window is closing on this frame, and
        # ``open_popup`` names a popup registered inside whatever window is
        # current. The draw site consumes it, which is also what lets Ctrl+/
        # and this command be the same door.
        ctx.state.shortcuts_requested = True

    return [
        *_mode_commands(),
        Command(
            key="generate",
            label="Generate / Make 3D",
            group="Actions",
            run=_generate,
            hint="Ctrl+Enter",
            enabled=_in_generate_mode,
            why="Open the 2D or 3D generate pane first.",
        ),
        Command(key="new-drawing", label="New drawing", group="Actions", run=new_drawing),
        Command(key="new-clay", label="New Clay document", group="Actions", run=new_clay),
        Command(key="new-atlas", label="New atlas", group="Actions", run=new_atlas),
        # The fourth of the four, missing since Plotter shipped: three of the
        # document modes could be started from here and the map editor could
        # not, which reads as Plotter not having a New at all.
        Command(key="new-map", label="New map", group="Actions", run=new_map),
        Command(
            key="reroll",
            label="Reroll the selected asset",
            group="Actions",
            run=reroll,
            enabled=rerollable,
            why="Select a finished asset in the library first.",
        ),
        Command(
            key="delete",
            label="Delete the selected asset...",
            group="Actions",
            run=delete,
            enabled=lambda ctx: _selected(ctx) is not None,
            why="Select an asset in the library first.",
        ),
        Command(
            key="wireframe",
            label="Toggle wireframe",
            group="Viewport",
            run=wireframe,
            hint="W",
            enabled=_viewport,
            why=_VIEWPORT_WHY,
        ),
        Command(
            key="turntable",
            label="Toggle turntable",
            group="Viewport",
            run=turntable,
            hint="S",
            enabled=_viewport,
            why=_VIEWPORT_WHY,
        ),
        Command(
            key="frame",
            label="Frame the model",
            group="Viewport",
            run=frame,
            hint="F",
            enabled=_viewport,
            why=_VIEWPORT_WHY,
        ),
        Command(
            key="clear-viewport",
            label="Clear the viewport",
            group="Viewport",
            run=clear_viewport,
            enabled=_viewport,
            why=_VIEWPORT_WHY,
        ),
        Command(
            key="fps",
            label="Toggle the frame-rate readout",
            group="Viewport",
            run=fps,
            hint="F10",
        ),
        # --- the document commands, whichever document mode is in front -----
        Command(
            key="save",
            label="Save",
            group="Document",
            run=_doc_save,
            hint="Ctrl+S",
            enabled=_has_doc,
            why=_DOC_WHY,
        ),
        Command(
            key="save-as",
            label="Save as...",
            group="Document",
            run=_doc_save_as,
            hint="Ctrl+Shift+S",
            enabled=_has_doc,
            why=_DOC_WHY,
        ),
        Command(
            key="export",
            # Labelled for the mode it will act on, because "Export" alone in a
            # searchable list is a command whose result the user cannot predict.
            label=_doc_export_label(ctx),
            group="Document",
            run=_doc_export,
            hint="Ctrl+E",
            enabled=_has_doc,
            why=_DOC_WHY,
        ),
        Command(
            key="undo",
            label="Undo",
            group="Document",
            run=_doc_undo,
            hint="Ctrl+Z",
            enabled=_can_undo,
            why="Nothing to undo in the document in front.",
        ),
        Command(
            key="redo",
            label="Redo",
            group="Document",
            run=_doc_redo,
            hint="Ctrl+Shift+Z",
            enabled=_can_redo,
            why="Nothing to redo: this is the newest step.",
        ),
        # --- the application commands ---------------------------------------
        Command(
            key="manual",
            label="Open the manual",
            group="Application",
            run=manual,
            hint="F1",
        ),
        Command(
            key="profiles",
            label="Manage style profiles",
            group="Application",
            run=profiles,
        ),
        Command(
            key="shortcuts",
            label="Keyboard shortcuts",
            group="Application",
            run=shortcuts,
            hint="Ctrl+/",
        ),
        Command(
            key="diagnostics",
            label="Issues",
            group="Application",
            run=diagnostics,
        ),
        Command(
            key="show-trash",
            label="Show the trash",
            group="Application",
            run=show_trash,
        ),
        Command(
            key="empty-trash",
            label="Empty the trash...",
            group="Application",
            run=empty_trash,
            enabled=lambda ctx: any(job.get("deleted_at") for job in ctx.cache.jobs),
            why="The trash is empty.",
        ),
        Command(key="open-log", label="Open the log", group="Application", run=open_log),
        Command(key="quit", label="Quit", group="Application", run=quit_app),
    ]


def asset_label(job: Any) -> str:
    """What quick-open matches against and shows. The id is in there because
    it is what a log line, a toast and a bug report all name."""
    name = job.get("name") or job.get("prompt") or ""
    return f"{name} {job['id']}".strip()


def assets(ctx: Any, query: str) -> list[Any]:
    """The quick-open half: assets whose name, prompt or id matches.

    Empty for an empty query -- a palette that lists the newest eight assets
    before a key is pressed pushes the commands off the screen, and the library
    is right there.
    """
    if not query.strip():
        return []
    # Trashed assets are excluded here rather than by ``Filters``: quick-open
    # does not go through the library's filter bar at all, so without this the
    # one surface that searches by name would be the one place a deleted asset
    # still turns up.
    live = [job for job in ctx.cache.jobs if not job.get("deleted_at")]
    return rank(query, live, asset_label)[:MAX_ASSETS]
