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


# --- characters -------------------------------------------------------------


def characters(ctx: Any) -> list[dict[str, Any]]:
    """Every mesh that has at least one finished character sheet.

    Built from the ``charsheet`` rows rather than from the sheets on disk: the
    rows are one indexed query and the sheets are a directory walk per job, and
    the two agree by construction -- a sheet exists because a row produced it.

    A row whose source job has since been pruned is skipped rather than shown
    as a broken card: the sheet went with the directory.
    """
    seen: dict[str, dict[str, Any]] = {}
    for row in ctx.svc.store.list(limit=SCAN_LIMIT, kind="charsheet"):
        if row["status"] != "done":
            continue
        source = str((row.get("params") or {}).get("source_job") or "")
        if not source or source in seen:
            continue
        job = ctx.cache.get(source)
        if job is None:
            continue
        seen[source] = {
            "id": source,
            "prompt": job.get("prompt") or "",
            "created_at": row["created_at"],
        }
    return list(seen.values())


def sheets(ctx: Any, job_id: str) -> list[dict[str, Any]]:
    """The character sheets in one job's directory, newest first.

    Only the character sheets: a mesh can also hold ordinary pose sheets, and
    they have no ``animation`` block, no direction runs and nothing this mode
    can play. Filtered on the block rather than on the row that made it,
    because the *artifact* is what the preview reads.
    """
    from .. import rigging

    if not job_id:
        return []
    out = [
        record
        for record in rigging.list_sheets(ctx.job_dir(job_id))
        if (record.get("animation") or {}).get("tags")
    ]
    out.sort(key=lambda r: float(r.get("created") or 0.0), reverse=True)
    return out


def select(ctx: Any, job_id: str, sheet_id: str = "") -> None:
    """Point the mode at a character, and at one of its sheets.

    The clock is reset here rather than left to run: carried across a selection
    it would show the new character mid-stride at whatever frame the old one
    happened to be on, which reads as a rendering fault rather than as a
    preview that kept playing.
    """
    state = ensure(ctx)
    state.job_id = job_id
    available = sheets(ctx, job_id)
    if sheet_id and any(r["id"] == sheet_id for r in available):
        state.sheet_id = sheet_id
    else:
        state.sheet_id = available[0]["id"] if available else ""
    state.clock = 0.0
    state.frame = 0
    release_texture(ctx)


def active_sheet(ctx: Any) -> dict[str, Any] | None:
    """The selected sheet's sidecar, or None."""
    from .. import rigging

    state = ensure(ctx)
    if not (state.job_id and state.sheet_id):
        return None
    return rigging.read_sheet(ctx.job_dir(state.job_id), state.sheet_id)


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
    table = troupe_spec.load()
    try:
        animation = table.animation(state.animation)
    except KeyError:
        return
    # Named ``interval`` rather than ``step`` because ``step`` is this
    # module's one-frame nudge, and a local shadowing it here would read as a
    # call to it.
    interval = max(animation.duration_ms, 1) / 1000.0 / max(state.speed, 0.01)
    state.clock += max(dt, 0.0)
    laps = 0
    while state.clock >= interval and laps <= animation.frames:
        state.clock -= interval
        state.frame += 1
        laps += 1
    if animation.loop:
        state.frame %= animation.frames
    else:
        # A one-shot holds its last frame, which is what the extra landing
        # frame in ``sheet.interpolate_clip`` exists for. Held rather than
        # looped *and* rather than stopped: a preview that stops needs a
        # control to start it again, and the point of the mode is that a bad
        # frame is obvious without pressing anything.
        state.frame = min(state.frame, animation.frames - 1)


def cell_index(ctx: Any) -> int | None:
    """Which cell of the atlas the preview is showing.

    Through ``spec.cells()`` rather than arithmetic over the animation lengths:
    that table is the studio's copy of the frame table and
    ``tests/troupe/test_troupe_geometry_agreement.py`` is the sole owner of its
    agreement with the pipeline's. A second piece of arithmetic here would be a
    third copy nothing owns.
    """
    state = ensure(ctx)
    for cell in troupe_spec.load().cells():
        if (
            cell.animation == state.animation
            and cell.direction == state.direction
            and cell.frame == state.frame
        ):
            return cell.index
    return None


def set_animation(ctx: Any, name: str) -> None:
    state = ensure(ctx)
    if name == state.animation:
        return
    state.animation = name
    state.clock = 0.0
    state.frame = 0


def set_direction(ctx: Any, name: str) -> None:
    ensure(ctx).direction = name


def step(ctx: Any, delta: int) -> None:
    """Nudge one frame, and stop playing -- stepping implies looking."""
    state = ensure(ctx)
    state.playing = False
    frames = troupe_spec.load().animation(state.animation).frames
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


def start_character(ctx: Any, form: dict[str, Any]) -> bool:
    """Submit the T-pose reference that starts a character.

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
    return ctx.submit(
        key,
        svc_jobs.create_job,
        ctx.svc,
        kind="text",
        prompt=str(form.get("prompt") or ""),
        output="reference",
        troupe={
            "variant": form.get("variant"),
            "logical_size": form.get("logical_size"),
            "colors": form.get("colors"),
            "outline": form.get("outline"),
            "reduce_mode": form.get("reduce_mode"),
            "dither": bool(form.get("dither")),
            "palette": form.get("palette") or "",
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
    return ctx.submit(
        key,
        svc_troupe.create_charsheet,
        ctx.svc,
        job_id,
        logical_size=form.get("logical_size"),
        colors=form.get("colors"),
        outline=form.get("outline"),
        dither=bool(form.get("dither")),
        palette=form.get("palette") or "",
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

    Two of them, and neither produces anything to adopt beyond a job id: the
    work itself is queued behind the GPU and lands as rows the sidebar reads.
    What this does is remember the id, so the sidebar can show *this* submit's
    progress rather than whatever the queue happens to be running.
    """
    state = ensure(ctx)
    result = getattr(done, "result", None)
    if isinstance(result, dict) and result.get("id"):
        state.pending = str(result["id"])
    if done.key == "troupe-start":
        ctx.toast(
            "Drawing the T-pose reference. Approve it in Create to build the mesh.",
            "success",
        )
        set_mode(ctx.state, "create")
    else:
        ctx.toast("Queued a character sheet. 256 frames; give it a few minutes.", "info")


def on_task_failed(ctx: Any, done: Any) -> None:
    """A refused request, said once and in the pane's own words."""
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


def start_from_home(ctx: Any) -> None:
    """Home's "New character" tile."""
    set_mode(ctx.state, "troupe")
    ensure(ctx)


#: The glyph the rail, Home and the palette all use for this mode. One name, so
#: a screen the user has seen does not change its picture per surface.
ICON = icons.PERSON_STANDING
