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
    keyboard shortcut or the reason the row is greyed.
    """

    key: str
    label: str
    group: str
    run: Callable[[Any], None]
    hint: str = ""
    enabled: Callable[[Any], bool] = field(default=lambda _ctx: True)


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
    from .panes import settings_2d, settings_3d

    if ctx.state.mode == "2d":
        settings_2d.generate(ctx, ctx.state.form_2d)
    else:
        settings_3d.promote(ctx, ctx.cache.get(ctx.state.source_job), ctx.state.form_3d)


def _in_generate_mode(ctx: Any) -> bool:
    return ctx.state.mode in ("2d", "3d")


def _viewport(ctx: Any) -> bool:
    return ctx.state.mode in modes.VIEWPORT_MODES


def _selected(ctx: Any) -> Any:
    return ctx.cache.get(ctx.state.selected)


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

    def fps(ctx: Any) -> None:
        ctx.state.show_fps = not ctx.state.show_fps

    return [
        *_mode_commands(),
        Command(
            key="generate",
            label="Generate / Make 3D",
            group="Actions",
            run=_generate,
            hint="Ctrl+Enter",
            enabled=_in_generate_mode,
        ),
        Command(key="new-drawing", label="New drawing", group="Actions", run=new_drawing),
        Command(key="new-clay", label="New Clay document", group="Actions", run=new_clay),
        Command(
            key="reroll",
            label="Reroll the selected asset",
            group="Actions",
            run=reroll,
            enabled=rerollable,
        ),
        Command(
            key="delete",
            label="Delete the selected asset...",
            group="Actions",
            run=delete,
            enabled=lambda ctx: _selected(ctx) is not None,
        ),
        Command(
            key="wireframe",
            label="Toggle wireframe",
            group="Viewport",
            run=wireframe,
            hint="W",
            enabled=_viewport,
        ),
        Command(
            key="turntable",
            label="Toggle turntable",
            group="Viewport",
            run=turntable,
            hint="S",
            enabled=_viewport,
        ),
        Command(
            key="frame",
            label="Frame the model",
            group="Viewport",
            run=frame,
            hint="F",
            enabled=_viewport,
        ),
        Command(
            key="fps",
            label="Toggle the frame-rate readout",
            group="Viewport",
            run=fps,
            hint="F10",
        ),
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
