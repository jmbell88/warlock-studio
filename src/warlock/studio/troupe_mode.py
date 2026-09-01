"""Troupe's controller: characters, sheets, the clock, and the way out.

The layer that knows about jobs and task threads; the engine under
``studio/troupe/`` knows about neither, and ``pipelines/charsheet.py`` knows
about neither *and* about no filesystem. The panes draw, this decides.

**A character is a mesh with a rig and at least one character sheet**, and the
sidebar lists them by walking the ``charsheet`` rows rather than the asset
library. That is deliberate: the library's question is "what have I made", and
this mode's is "what can I animate" -- a hundred barrels are the right answer to
the first and noise in the second.

**Nothing here is a document.** Troupe holds a selection over files a worker
published, so there is no journal provider, no Save and no undo stack -- see
:mod:`.troupe_state`. What the mode *can* lose in a crash is which frame the
preview was on.

**The preview is a clock, not a frame counter.** Durations are milliseconds per
animation and the window runs at whatever rate it runs at, so stepping one cell
per draw would play a 60 ms run cycle at monitor speed on one machine and half
that on another. :func:`advance` takes the frame's delta and is called from the
pane that draws -- there is no per-mode update hook, which is the arrangement
``packwright_mode`` already lives with.

Every task key carries the ``troupe-`` prefix, because the app claims results by
prefix: a key without one is a result delivered nowhere.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from . import icons
from .state import set_mode
from .troupe import spec as troupe_spec
from .troupe_state import (  # noqa: F401  -- re-exported; every caller says troupe_mode.X
    TroupeState,
    ensure,
)

log = logging.getLogger(__name__)

#: How many jobs deep the character walk looks. The same page size the library
#: pages by, and for the same reason: an unbounded walk over a corpus of
#: thousands is a frame-thread cost that grows with how long the user has owned
#: the app.
SCAN_LIMIT = 400

#: How often the cast is re-read while something is on the chain, and while
#: nothing is. ``jobs_cache``'s two numbers, for its reason -- the library
#: settled on 0.5s live and 3.0s idle over the same store, and a second cadence
#: over the same rows would be a second answer to a question already answered.
CAST_REFRESH_LIVE = 0.5
CAST_REFRESH_IDLE = 3.0

#: How often :func:`sheets` and :func:`active_sheet` go back to disk. Between
#: the two of them the Troupe panes asked three to four times *per frame*; this
#: is ``db.list``'s "twice a second" applied to a directory instead of a table.
SHEETS_REFRESH = 0.5


# --- characters -------------------------------------------------------------


def characters(ctx: Any) -> list[dict[str, Any]]:
    """Every mesh that has at least one finished character sheet.

    Built from the ``charsheet`` rows rather than from the sheets on disk: the
    rows are one indexed query and the sheets are a directory walk per job, and
    the two agree by construction -- a sheet exists because a row produced it.

    **The title comes off the charsheet row, not the library page.** Every
    charsheet door mints its row with ``source["prompt"]``, so the row already
    carries the character's description; the cache is consulted only to prefer
    a name the user has since typed. This used to ``continue`` on a cache miss,
    justified as "a row whose source job has since been pruned" -- but
    ``ctx.cache.get`` answers about the *library's currently loaded page*, and
    a pruned job is a different question with a different answer
    (``store.get`` returning None). So a character silently vanished from the
    cast as soon as its mesh scrolled off that page, which has nothing to do
    with this mode. A genuinely pruned character is still handled, one layer
    down and honestly: ``sheets()`` finds no sidecars and ``open_sheet``
    refuses.
    """
    seen: dict[str, dict[str, Any]] = {}
    for row in ctx.svc.store.list(limit=SCAN_LIMIT, kind="charsheet"):
        if row["status"] != "done" or row.get("deleted_at"):
            continue
        source = str((row.get("params") or {}).get("source_job") or "")
        if not source or source in seen:
            continue
        # ``store.list`` does not filter the trash, and trashing a mesh does
        # not cascade to the sheets made from it -- so a character thrown away
        # keeps its live charsheet rows. The cache used to hide that by
        # accident (the loaded page is filtered by ``state.trash``), and
        # dropping the cache dependency above dropped the accident with it.
        # Skipped on a *known* trashed source, kept on a cache miss: the two
        # are different answers and only one of them is about the trash.
        job = ctx.cache.get(source) or {}
        if job.get("deleted_at"):
            continue
        seen[source] = {
            "id": source,
            "prompt": job.get("name") or job.get("prompt") or row.get("prompt") or "",
            "created_at": row["created_at"],
        }
    return list(seen.values())


def cast_and_pending(ctx: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """``(characters, in_progress)``, throttled. What the cast pane draws from.

    **Both halves are SQL, and the pane that wants them draws every frame.**
    ``characters`` is a ``kind``-filtered page and ``in_progress`` an unfiltered
    one, each up to :data:`SCAN_LIMIT` rows with per-row JSON parsing and a sort
    on top -- and both used to run per draw, serialised behind ``JobStore``'s
    one lock, contending with the worker thread that is updating those very
    rows at the moment a user is watching this mode.

    ``db.list``'s own docstring sizes it for "twice a second"; ``jobs_cache``
    holds the library to exactly that. This is the same rule for the same
    store. :func:`invalidate_cast` closes the gap the interval would otherwise
    leave, so a submission still shows up on the next frame rather than up to
    three seconds later.
    """
    state = ensure(ctx)
    now = time.monotonic()
    if state.cast_cache is None or now >= state.cast_next:
        pending = in_progress(ctx)
        state.cast_cache = (characters(ctx), pending)
        # Faster while the chain is moving: a row that is going to change soon
        # is worth re-reading soon, and one that is not, is not.
        state.cast_next = now + (CAST_REFRESH_LIVE if pending else CAST_REFRESH_IDLE)
    return state.cast_cache


def invalidate_cast(ctx: Any) -> None:
    """Drop the throttled cast so the next draw re-reads it."""
    ensure(ctx).cast_cache = None


def sheets(ctx: Any, job_id: str) -> list[dict[str, Any]]:
    """The character sheets in one job's directory, newest first.

    Only the character sheets: a mesh can also hold ordinary pose sheets, and
    they have no ``animation`` block, no direction runs and nothing this mode
    can play. Filtered on the block rather than on the row that made it,
    because the *artifact* is what the preview reads.

    **Throttled, for ``cast_and_pending``'s reason one directory over.** This is
    a glob plus a stat plus a read plus a JSON parse *per sheet*, and the cast
    pane calls it from its draw -- so a mode left open on a character with ten
    sheets hit the disk ten times a frame for a directory that changes only when
    a sheet is built. Keyed on ``job_id`` as well as timed, so switching
    character reads immediately; :func:`invalidate_sheets` closes the same gap a
    build would otherwise leave.
    """
    state = ensure(ctx)
    if not job_id:
        return []
    now = time.monotonic()
    if state.sheets_cache is None or state.sheets_key != job_id or now >= state.sheets_next:
        state.sheets_cache = _read_sheets(ctx, job_id)
        state.sheets_key = job_id
        state.sheets_next = now + SHEETS_REFRESH
    return state.sheets_cache


def invalidate_sheets(ctx: Any) -> None:
    """Drop the throttled sheet reads so the next draw re-reads the directory."""
    state = ensure(ctx)
    state.sheets_cache = None
    state.sheet_cache = None


def _read_sheets(ctx: Any, job_id: str) -> list[dict[str, Any]]:
    """The uncached read :func:`sheets` throttles."""
    from .. import rigging

    out = [
        record
        for record in rigging.list_sheets(ctx.job_dir(job_id))
        if (record.get("animation") or {}).get("tags")
    ]
    out.sort(key=lambda r: float(r.get("created") or 0.0), reverse=True)
    return out


#: The phase sentences an unfinished character can be at, in chain order. The
#: gate is the only one the *user* can be blocking on, which is why it is the
#: only one that offers a button.
_WAITING = "Waiting for you. Approve the drawing in Create."

#: How many unfinished characters the cast lists. A reference that was never
#: approved stays unapproved forever, so without a cap a user who abandons
#: drafts accumulates a sidebar of them above the characters they finished --
#: the section would grow into the noise it exists to prevent.
MAX_IN_PROGRESS = 5


#: What each reference pose is called in prose. The keys are the door's names
#: and are what travel in ``params``; these are only what is read. One table,
#: here rather than in a pane, because the sidebar, the toast and the form all
#: name the same thing and three spellings of it is three chances to drift.
POSE_LABELS = {"tpose": "T-pose", "apose": "A-pose"}


def _pose_label(row: Any) -> str:
    """What to call the pose a job row was drawn against.

    Falls back to the T-pose, and deliberately not to the door's default: a row
    queued before the pose was a choice was drawn against that guide, and
    ``_q_generate`` redraws it against the same one. Saying "A-pose" over it
    would describe a character nobody queued.
    """
    params = (row or {}).get("params") or {}
    return POSE_LABELS.get(str(params.get("guide_pose") or "tpose"), "reference")


def in_progress(ctx: Any) -> list[dict[str, Any]]:
    """Characters on the chain that cannot be played yet, newest first.

    The chain is reference -> a human gate in Create -> mesh -> rig -> sheet,
    and until the last link lands the character exists nowhere :func:`characters`
    looks. Submitting one from this mode hands off to Create (see
    :func:`start_character`), so a user who came back before approving the
    drawing was shown "No character sheets yet" -- the mode's own empty state,
    telling them nothing had been started, about the thing they had just
    started. This is the row that was missing.

    One unfiltered page rather than a query per link: the classification needs
    the references, their promoted children and the charsheet rows together,
    and asking three times would be three times the cost to answer one
    question. ``status`` is read off the rows, so a character that is mid-mesh
    says so without this walking the queue.
    """
    rows = ctx.svc.store.list(limit=SCAN_LIMIT)
    refs: list[dict[str, Any]] = []
    children: dict[str, list[dict[str, Any]]] = {}
    sheeted: set[str] = set()
    # The mesh-sourced chain, keyed on the mesh the rows name as their source.
    # Same page, no extra query: the rig and charsheet rows are already on the
    # one unfiltered walk below.
    sourced: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        # ``store.list`` does not filter the trash. A character the user threw
        # away must not come back as work they still owe -- and a trashed mesh
        # must not count as the promotion that makes its reference "building",
        # which is why this is dropped before the walk rather than after it.
        if row.get("deleted_at"):
            continue
        params = row.get("params") or {}
        if row.get("kind") == "charsheet" and row.get("status") == "done":
            sheeted.add(str(params.get("source_job") or ""))
        parent = str(row.get("parent_id") or "")
        if parent:
            children.setdefault(parent, []).append(row)
        if params.get("troupe") and row.get("stage") == "reference":
            refs.append(row)
        if row.get("kind") in ("rig", "charsheet"):
            source = str(params.get("source_job") or "")
            if source:
                sourced.setdefault(source, []).append(row)

    out: list[dict[str, Any]] = []
    for row in refs:
        status = str(row.get("status") or "")
        kids = children.get(str(row["id"]), [])
        if any(str(kid["id"]) in sheeted for kid in kids):
            # Finished: ``characters`` already holds it, and listing it twice
            # would say the thing it is about to draw is not ready.
            continue
        if status in ("error", "cancelled"):
            # The library's failure card owns a failed row, with the error text
            # and the reroll. Repeating it here would be a second account of
            # one failure, in a sidebar that cannot act on it.
            continue
        if status != "done":
            phase, waiting = f"Drawing the {_pose_label(row)} reference...", False
        elif not kids:
            phase, waiting = _WAITING, True
        else:
            phase, waiting = "Building the mesh, rig and sheet...", False
        out.append(
            {
                "id": str(row["id"]),
                "prompt": row.get("name") or row.get("prompt") or "",
                "phase": phase,
                "waiting": waiting,
                "created_at": row.get("created_at"),
            }
        )
    out.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    # The cap applies to the reference-sourced entries *only*, and the reason
    # is the cap's own reason: a reference that was never approved stays
    # unapproved forever, so without a limit abandoned drafts pile up above
    # the finished cast. A rig or a charsheet row is claimed by a serial queue
    # and terminates -- capping those would hide work that is genuinely
    # running, which is the opposite failure. Do not "simplify" this into one
    # cap.
    return out[:MAX_IN_PROGRESS] + _sent_in_progress(ctx, sourced, sheeted, seen=out)


def _sent_in_progress(
    ctx: Any,
    sourced: dict[str, list[dict[str, Any]]],
    sheeted: set[str],
    *,
    seen: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """The characters that came in through "Send to Troupe", mid-chain.

    Without this the door is a multi-minute silent CPU spend: a mesh-sourced
    character matches no ``troupe`` reference, so the user watches a rig and
    256 rendered cells in front of the exact "No character sheets yet" empty
    state :func:`in_progress` exists to eliminate.

    Errored rows are dropped for the reference pass's reason: the library's
    failure card owns a failed row, with the error text and the reroll, and a
    second account of one failure in a sidebar that cannot act on it is worse
    than none.
    """
    already = {str(item.get("id") or "") for item in seen}
    out: list[dict[str, Any]] = []
    for source, rows in sourced.items():
        if source in sheeted or source in already:
            continue
        live = [row for row in rows if str(row.get("status") or "") not in ("error", "cancelled")]
        if not live:
            continue
        kinds = {str(row.get("kind") or "") for row in live}
        # A mesh-sourced character has no human gate, so ``waiting`` is False
        # throughout -- which is what keeps ``_WAITING``'s claim true by
        # construction: the gate is the only phase the user can be blocking on
        # and therefore the only one that offers a button.
        phase = (
            "Rendering the character sheet..."
            if "charsheet" in kinds
            else "Rigging the mesh..."
        )
        job = ctx.cache.get(source) or {}
        if job.get("deleted_at"):
            continue
        newest = max(str(row.get("created_at") or "") for row in live)
        out.append(
            {
                "id": source,
                "prompt": job.get("name") or job.get("prompt") or live[0].get("prompt") or "",
                "phase": phase,
                "waiting": False,
                "created_at": newest,
            }
        )
    out.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return out


def can_send_to_troupe(ctx: Any, job: Any) -> bool:
    """Whether "Send to Troupe" belongs on this job's row.

    From the cached row alone -- no filesystem call, because a toolbar asks
    this every frame. ``inker_mode.can_edit_job``'s shape and its argument.

    Two things are deliberately *not* in this predicate. **A rig**: an
    unrigged mesh is exactly what the door is for, and it mints the rig
    itself. **The rig template**: it lives only in the ``rig.json`` sidecar,
    which is a disk read -- so the button is offered optimistically and the
    service refuses, instantly and before anything is minted, in a sentence
    that already reads well ("a character sheet is animated from the humanoid
    clip library, and this mesh is rigged as quadruped"). Recording the
    template on the mesh row instead would mean a worker writing a fact onto
    its source's params, which then has to join ``DERIVED_PARAMS`` or a reroll
    wears a stale template -- all to save one toast.

    ``model.glb`` and never ``source.glb``: the reconstruction is not the
    asset, and every other export is a function of the derived mesh.
    """
    del ctx
    return bool(
        job
        and job.get("stage") == "model"
        and job.get("status") == "done"
        and not job.get("deleted_at")
        and "model.glb" in (job.get("files") or [])
    )


def sendable_meshes(ctx: Any) -> list[dict[str, Any]]:
    """Every mesh the picker inside Troupe may offer, newest first.

    ``characters``' page cap and its argument: an unbounded walk over a corpus
    of thousands is a frame-thread cost that grows with how long the user has
    owned the app.
    """
    from ..service import jobs as svc_jobs

    # Through the service's listing rather than ``store.list``: the predicate
    # reads ``files``, which is ``attach_files``' doing and not a column.
    out = [
        {
            "id": str(row["id"]),
            "prompt": row.get("name") or row.get("prompt") or "",
            "created_at": row.get("created_at"),
        }
        for row in svc_jobs.list_jobs(ctx.svc, limit=SCAN_LIMIT)
        if can_send_to_troupe(ctx, row)
    ]
    out.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return out


def send_to_troupe(ctx: Any, job: Any, form: dict[str, Any] | None = None) -> bool:
    """Take an existing mesh into Troupe. -> whether the request was submitted.

    **It does not switch modes**, and that is the difference from
    :func:`start_character`, which does: there the gate lives in Create and
    the user has something to do next. Here there is nothing to do, and
    yanking somebody out of the library mid-review to watch a spinner is the
    opposite of the affordance. The toast says where to watch instead.
    """
    from ..service import troupe as svc_troupe

    job_id = str((job or {}).get("id") or "")
    if not job_id:
        return False
    settings = _layout_request(form or {}) if form else {}
    request = dict(form or {})

    def run() -> Any:
        return svc_troupe.send_to_troupe(
            ctx.svc,
            job_id,
            logical_size=request.get("logical_size"),
            colors=request.get("colors"),
            outline=request.get("outline"),
            reduce_mode=request.get("reduce_mode"),
            dither=bool(request.get("dither")),
            palette=request.get("palette") or None,
            elevation=request.get("elevation"),
            lighting=request.get("lighting"),
            name=str(request.get("name") or ""),
            layout=settings or None,
        )

    return bool(ctx.submit(f"troupe-send:{job_id}", run))


def open_sheet(ctx: Any, job_id: str, sheet_id: str = "") -> bool:
    """Enter Troupe pointed at one sheet. **The one door in from elsewhere.**

    Two callers -- the library's overflow item and a finished charsheet's
    "Show" toast -- and each used to be its own copy of ``select`` plus a
    ``set_mode``, which is one copy away from disagreeing about which sheet.

    Returns False, and does *not* switch the mode, when the character has no
    playable sheet: the preview would draw "No character on screen", and
    arriving at an empty room is the failure this door exists to stop. The
    caller says so in its own words rather than being switched into silence.
    """
    state = ensure(ctx)
    if not job_id or not sheets(ctx, job_id):
        return False
    select(ctx, job_id, sheet_id)
    if not state.sheet_id:
        return False
    set_mode(ctx.state, "troupe")
    return True


def select(ctx: Any, job_id: str, sheet_id: str = "") -> None:
    """Point the mode at a character, and at one of its sheets.

    The clock is reset here rather than left to run: carried across a selection
    it would show the new character mid-stride at whatever frame the old one
    happened to be on, which reads as a rendering fault rather than as a
    preview that kept playing.
    """
    state = ensure(ctx)
    state.job_id = job_id
    # The throttled directory read is dropped rather than waited out, for the
    # reason ``on_task_done`` drops the cast: the interval exists to stop idle
    # polling, not to delay news the user has just asked for by name.
    invalidate_sheets(ctx)
    available = sheets(ctx, job_id)
    if sheet_id and any(r["id"] == sheet_id for r in available):
        state.sheet_id = sheet_id
    else:
        state.sheet_id = available[0]["id"] if available else ""
    state.clock = 0.0
    state.frame = 0
    _reconcile_preview(ctx)
    release_texture(ctx)


def active_sheet(ctx: Any) -> dict[str, Any] | None:
    """The selected sheet's sidecar, or None. Throttled like :func:`sheets`.

    Three panes ask for this in their draw -- the preview twice and the sheet
    list once -- so it was three JSON reads a frame of a file that changes only
    when a sheet is rebuilt.
    """
    from .. import rigging

    state = ensure(ctx)
    if not (state.job_id and state.sheet_id):
        return None
    key = (state.job_id, state.sheet_id)
    now = time.monotonic()
    if (
        state.sheet_cache is None
        or state.sheet_cache_key != key
        or now >= state.sheet_cache_next
    ):
        state.sheet_cache = rigging.read_sheet(ctx.job_dir(key[0]), key[1])
        state.sheet_cache_key = key
        state.sheet_cache_next = now + SHEETS_REFRESH
    return state.sheet_cache


def preview_layout(ctx: Any) -> dict[str, Any]:
    """The active sheet's immutable layout, with a pre-v2 legacy fallback."""

    record = active_sheet(ctx) or {}
    snapshot = record.get("troupe")
    if isinstance(snapshot, dict):
        try:
            movements = snapshot.get("movements") or ()
            runs = snapshot.get("runs") or ()
            valid = bool(movements and runs) and all(
                str(m.get("key") or "")
                and int(m.get("frames") or 0) > 0
                and bool(m.get("directions"))
                for m in movements
            ) and all(
                str(run.get("movement") or "")
                and str(run.get("direction") or "")
                and 0 <= int(run.get("start")) <= int(run.get("end"))
                for run in runs
            )
        except (AttributeError, TypeError, ValueError):
            valid = False
        if valid:
            return snapshot
        return {"version": 2, "movements": [], "runs": [], "cell_count": 0}
    table = troupe_spec.load()
    movements = [
        {
            "key": animation.name,
            "label": animation.name.title(),
            "frames": animation.frames,
            "loop": animation.loop,
            "duration_ms": animation.duration_ms,
            "directions": [
                {"key": direction.name, "yaw": direction.yaw}
                for direction in table.directions
            ],
        }
        for animation in table.animations
    ]
    runs = [
        {
            "movement": animation,
            "direction": direction,
            "start": start,
            "end": end,
        }
        for animation, direction, start, end, _loop in table.spans()
    ]
    return {"version": 1, "movements": movements, "runs": runs, "cell_count": 256}


def preview_movement(ctx: Any, name: str | None = None) -> dict[str, Any] | None:
    wanted = name or ensure(ctx).animation
    return next(
        (m for m in preview_layout(ctx).get("movements") or () if m.get("key") == wanted),
        None,
    )


def _reconcile_preview(ctx: Any) -> None:
    state = ensure(ctx)
    movements = preview_layout(ctx).get("movements") or ()
    if not movements:
        return
    movement = next((m for m in movements if m.get("key") == state.animation), movements[0])
    state.animation = str(movement.get("key") or "")
    directions = movement.get("directions") or ()
    keys = [str(d.get("key") or "") for d in directions]
    if keys and state.direction not in keys:
        state.direction = keys[0]
    state.frame = min(state.frame, max(int(movement.get("frames") or 1) - 1, 0))


# --- the clock --------------------------------------------------------------


def advance(ctx: Any, dt: float) -> None:
    """Move the preview on by ``dt`` seconds of wall clock.

    A ``while`` rather than an ``if``: a frame that took longer than one sprite
    frame -- a job finishing, a texture upload, the window being dragged --
    must skip cells rather than fall behind and never catch up. Bounded by the
    run's own length, so a pathological stall costs at most one lap.
    """
    state = ensure(ctx)
    if not state.playing:
        return
    animation = preview_movement(ctx)
    if animation is None:
        return
    # Named ``interval`` rather than ``step`` because ``step`` is this
    # module's one-frame nudge, and a local shadowing it here would read as a
    # call to it.
    duration = int(animation.get("duration_ms") or 100)
    frames = int(animation.get("frames") or 1)
    interval = max(duration, 1) / 1000.0 / max(state.speed, 0.01)
    state.clock += max(dt, 0.0)
    laps = 0
    while state.clock >= interval and laps <= frames:
        state.clock -= interval
        state.frame += 1
        laps += 1
    if animation.get("loop"):
        state.frame %= frames
    else:
        # A one-shot holds its last frame, which is what the extra landing
        # frame in ``sheet.interpolate_clip`` exists for. Held rather than
        # looped *and* rather than stopped: a preview that stops needs a
        # control to start it again, and the point of the mode is that a bad
        # frame is obvious without pressing anything.
        state.frame = min(state.frame, frames - 1)


def cell_index(ctx: Any) -> int | None:
    """Which cell of the atlas the preview is showing.

    Through ``spec.cells()`` rather than arithmetic over the animation lengths:
    that table is the studio's copy of the frame table and
    ``tests/troupe/test_troupe_geometry_agreement.py`` is the sole owner of its
    agreement with the pipeline's. A second piece of arithmetic here would be a
    third copy nothing owns.
    """
    state = ensure(ctx)
    for run in preview_layout(ctx).get("runs") or ():
        if run.get("movement") == state.animation and run.get("direction") == state.direction:
            start, end = int(run.get("start") or 0), int(run.get("end") or 0)
            index = start + state.frame
            return index if start <= index <= end else None
    return None


def set_animation(ctx: Any, name: str) -> None:
    state = ensure(ctx)
    if name == state.animation:
        return
    state.animation = name
    state.clock = 0.0
    state.frame = 0
    _reconcile_preview(ctx)


def set_direction(ctx: Any, name: str) -> None:
    state = ensure(ctx)
    state.direction = name
    movement = preview_movement(ctx)
    state.frame = min(state.frame, max(int((movement or {}).get("frames") or 1) - 1, 0))


def step(ctx: Any, delta: int) -> None:
    """Nudge one frame, and stop playing -- stepping implies looking."""
    state = ensure(ctx)
    state.playing = False
    frames = int((preview_movement(ctx) or {}).get("frames") or 1)
    state.frame = (state.frame + delta) % frames
    state.clock = 0.0


# --- the texture ------------------------------------------------------------
#
# One texture for the whole atlas, uploaded once per sheet and drawn as a
# sub-rectangle per frame -- ``sheet_panel``'s arrangement, and the same
# forget-then-release rule: ``widgets.texture_ref`` registers with the imgui
# backend, so the driver may not reuse the GL name until the backend has been
# told. The filter is NEAREST, which is the whole point: a linear-filtered
# sprite is the one thing a pixel-art preview must never show.


def release_texture(ctx: Any) -> None:
    """Forget-then-release the cached atlas texture. Also called at teardown."""
    ctx.state.preview.pop("troupe_texture:key", None)
    cached = ctx.state.preview.pop("troupe_texture", None)
    if cached is None:
        return
    from . import imgui_backend

    renderer = imgui_backend.current()
    if renderer is not None:
        renderer.forget_texture(cached)
    cached.release()


def atlas_texture(ctx: Any) -> Any:
    """The selected sheet's atlas, uploaded once. ``None`` when there is none.

    Keyed on ``(job, sheet)`` rather than on the file's mtime: a published
    sheet is write-once under a fresh id -- the worker stages the pixel-art
    atlas onto the served name and the sidecar is what marks it complete -- so
    a stale texture cannot exist for a key that has not changed.
    """
    from PIL import Image

    from .. import rigging

    state = ensure(ctx)
    if ctx.viewer is None or not (state.job_id and state.sheet_id):
        return None
    key = (state.job_id, state.sheet_id)
    if ctx.state.preview.get("troupe_texture:key") == key:
        return ctx.state.preview.get("troupe_texture")

    path = rigging.sheet_png_path(ctx.job_dir(state.job_id), state.sheet_id)
    if not path.exists():
        return None
    release_texture(ctx)
    try:
        with Image.open(path) as opened:
            opened.load()
            atlas = opened.convert("RGBA")
    except Exception:
        # Logged as well as swallowed: an unreadable atlas leaves nothing in
        # warlock.log otherwise, and a blank preview is not a diagnosis.
        log.exception("could not read the character sheet %s", path)
        return None
    texture = ctx.viewer.ctx.texture(atlas.size, 4, atlas.tobytes())
    texture.filter = (ctx.viewer.ctx.NEAREST, ctx.viewer.ctx.NEAREST)
    ctx.state.preview["troupe_texture"] = texture
    ctx.state.preview["troupe_texture:key"] = key
    return texture


# --- the two doors ----------------------------------------------------------


def _layout_request(form: dict[str, Any]) -> dict[str, Any]:
    """Strip presentation-only flags from the editable Troupe form."""

    source = form.get("layout") or {}
    return {
        "version": 2,
        "columns": int(source.get("columns") or 8),
        "movements": [
            {
                "key": row.get("key"),
                "frames": row.get("frames"),
                "directions": row.get("directions"),
            }
            for row in source.get("movements") or ()
            if row.get("enabled", True)
        ],
    }


def start_character(ctx: Any, form: dict[str, Any]) -> bool:
    """Submit the pose reference that starts a character.

    The *first* link only. The gate is the point of the shape: this queues one
    cheap image, the user approves it in Create, and only then is the
    reconstruction spent. Which is also why this hands off to Create rather
    than staying here -- the approval lives where every other reference's
    approval lives, and a second promote button would be a second gate.
    """
    from ..service import jobs as svc_jobs

    key = "troupe-start"
    if ctx.busy(key):
        return False
    # Last time's rings first: a new submit is judged on its own, and this
    # request's refusals name controls on ``panes/troupe_settings``'s form.
    ctx.state.clear_field_errors()
    return ctx.submit(
        key,
        svc_jobs.create_job,
        ctx.svc,
        kind="text",
        prompt=str(form.get("prompt") or ""),
        output="reference",
        troupe={
            "variant": form.get("variant"),
            "pose": form.get("pose"),
            "logical_size": form.get("logical_size"),
            "colors": form.get("colors"),
            "outline": form.get("outline"),
            "reduce_mode": form.get("reduce_mode"),
            "dither": bool(form.get("dither")),
            "palette": form.get("palette") or "",
            "layout": _layout_request(form),
        },
    )


def build_sheet(ctx: Any, job_id: str, form: dict[str, Any]) -> bool:
    """Queue another character sheet for a mesh that is already rigged.

    The direct door -- for a supplied base mesh, or for a second sheet at a
    different size from the same character. It is not the chain's normal route
    and is not meant to be: the chain's route is the block on the reference.
    """
    from ..service import troupe as svc_troupe

    key = f"troupe-sheet:{job_id}"
    if ctx.busy(key):
        return False
    ctx.state.clear_field_errors()
    return ctx.submit(
        key,
        svc_troupe.create_charsheet,
        ctx.svc,
        job_id,
        logical_size=form.get("logical_size"),
        colors=form.get("colors"),
        outline=form.get("outline"),
        reduce_mode=form.get("reduce_mode"),
        dither=bool(form.get("dither")),
        palette=form.get("palette") or "",
        layout=_layout_request(form),
    )


def open_in_inker(ctx: Any) -> bool:
    """Hand the selected sheet to Inker as an animated document.

    Through the bridge Packwright and the sheet inspector already use, rather
    than a second importer: a character sheet is an ordinary sheet plus an
    ``animation`` block, and Inker's sheet import already reads that block into
    tags. A second path would be a second dialect of one format.
    """
    from . import inker_mode

    state = ensure(ctx)
    if not (state.job_id and state.sheet_id):
        return False
    inker_mode.open_rendered_sheet(ctx, state.job_id, state.sheet_id)
    return True


def add_to_packwright(ctx: Any) -> bool:
    """The other way out, and the other existing bridge -- one sheet's cells
    into an atlas beside everything else being packed."""
    from . import packwright_mode

    state = ensure(ctx)
    if not (state.job_id and state.sheet_id):
        return False
    packwright_mode.add_rendered_sheet(ctx, state.job_id, state.sheet_id)
    return True


def on_task_done(ctx: Any, done: Any) -> None:
    """Adopt what a ``troupe-`` task returned.

    Three of them, and none produces anything to adopt beyond a job id: the
    work itself is queued behind the GPU and lands as rows the sidebar reads
    (``in_progress`` finds them by kind and source, so the id is not kept).
    """
    state = ensure(ctx)
    # Every one of these queues or finishes a row the cast is built from, so
    # the throttled copy is dropped rather than waited out: the interval is
    # there to stop idle polling, not to delay news this pane already has.
    invalidate_cast(ctx)
    invalidate_sheets(ctx)
    result = getattr(done, "result", None)
    if done.key == "troupe-start":
        # The form's own choice: this fires on the way out of the submit, and
        # the row it queued is not readable here yet.
        pose = str((state.form or {}).get("pose") or "")
        ctx.toast(
            f"Drawing the {POSE_LABELS.get(pose, 'pose')} reference. "
            "Approve it in Create to build the mesh.",
            "success",
        )
        set_mode(ctx.state, "create")
    elif done.key.startswith("troupe-send:"):
        # Two shapes behind one press, and the toast says which happened: an
        # unrigged mesh is minutes of CPU behind a button that is not called
        # "Rig", and a user who is not told that will think it hung. No mode
        # switch -- see ``send_to_troupe``.
        rigged = isinstance(result, dict) and result.get("rigged") is not False
        ctx.toast(
            "Rendering the character sheet. Watch it in Troupe."
            if rigged
            else "Rigging the mesh, then rendering the character sheet. Watch it in Troupe.",
            "success",
        )
    else:
        ctx.toast("Queued the configured character sheet; give it a few minutes.", "info")


def on_task_failed(ctx: Any, done: Any) -> None:
    """A refused request, said once and in the pane's own words."""
    invalidate_cast(ctx)
    invalidate_sheets(ctx)
    ctx.toast(str(getattr(done, "error", "") or "That request was refused."), "error")


def handle_key(ctx: Any, event: Any) -> bool:
    """Space toggles playback; Left/Right step a frame.

    Which is why ``troupe`` is in ``modes.NAV_KEY_MODES``: one press must not
    also step imgui's focus ring.

    **Presses only.** This used to act on ``event.key`` without looking at
    ``event.type``, so every one of these ran twice per press -- Space toggled
    play and toggled it straight back, and a tap of Right stepped two frames.
    ``plotter_mode.handle_key`` states the same contract explicitly and for the
    same reason; a release is not consumed, because nothing downstream acts on
    a bare KEYUP.
    """
    import pygame

    if event.type != pygame.KEYDOWN:
        return False
    state = ensure(ctx)
    if event.key == pygame.K_SPACE:
        state.playing = not state.playing
        return True
    if event.key == pygame.K_LEFT:
        step(ctx, -1)
        return True
    if event.key == pygame.K_RIGHT:
        step(ctx, 1)
        return True
    return False


# Home's "New character" tile is ``panes.landing.start_troupe``, which is what
# Home actually wires. The duplicate that sat here was deleted on 2026-08-22
# with zero callers: two entry points to one mode is one of them going stale.


#: The glyph the rail, Home and the palette all use for this mode. One name, so
#: a screen the user has seen does not change its picture per surface.
ICON = icons.PERSON_STANDING
