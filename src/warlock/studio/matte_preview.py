"""The promote flow's matte preview: what is on screen, and what asked for it.

The shape ``clay_mode``/``review_mode`` set: no imgui here at all, so every
rule about *when* a preview is computed, cached, dropped or shown is assertable
headlessly. ``panes/settings_3d`` draws it.

The one hard rule is the one the whole app runs on: BiRefNet is seconds of host
compute, so nothing here ever calls ``service.matte.preview`` -- it submits it,
and the frame thread only adopts the answer. Which is also where the cache is
written from, and that ordering is not incidental: ``service.matte.remember``
reads the clock to decide whether a stamp is safely in the past, and adopting
on the frame thread means that read is unambiguously later than the read of the
pixels the stamp describes.

Nothing here is persisted. A cached cutout outliving the session that measured
it would be a claim about a file that has had a whole session to change.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..service import matte as svc_matte

log = logging.getLogger(__name__)


@dataclass
class MatteState:
    """The open preview, if any, and what pressing Accept would submit."""

    # The reference being previewed. Empty means the modal is closed -- one
    # field rather than a separate flag, because "open, about nothing" is not a
    # state anything could draw.
    job_id: str = ""
    # ``settings_3d.promote_kwargs`` as it stood when the user pressed the
    # button. Captured then rather than re-read on Accept: the form is live UI
    # state and the preview the user is looking at is of the settings they
    # pressed with.
    kwargs: dict[str, Any] = field(default_factory=dict)
    # Whether the composition gate was already answered with a confirm, i.e.
    # whether the promotion goes through with ``force``.
    force: bool = False
    # ``input.png``'s mtime the last time the frame looked, so the pane can
    # notice a re-save landing behind it (a Fix-matte round trip) and ask for a
    # fresh cutout instead of showing the one it took before the edit.
    stamp: int | None = None
    preview: Any = None
    # Whether ``imgui.open_popup`` has been called for the current question --
    # the ``dialogs.Confirm._open`` idiom, and here for the same reason: the
    # call must happen exactly once, and the pane redraws every frame.
    _open: bool = False
    # The stamp whose cutout *failed*, so ``pump`` does not ask again for the
    # same bytes. Without it a refused preview -- ``service.matte.preview``
    # raises before any compute when ``input.png`` is missing -- left
    # ``preview`` None with ``job_id`` still set, and ``matte_modal`` is drawn
    # unconditionally every frame: the cache missed, the submit was accepted
    # because the previous task had already failed, and the modal sat on
    # "Cutting the subject out..." behind one new red toast per frame until the
    # 100-entry history had rolled over.
    #
    # A stamp rather than a flag, so the *remedy* still works: Fix matte
    # rewrites ``input.png``, the mtime moves, and the next frame asks again on
    # its own.
    failed_stamp: int | None = None
    # Previews by job id, cached under the racily-clean rule. Kept on the state
    # rather than module-global so a second Ctx (a test, a compare view) does
    # not inherit another's answers.
    cache: dict[str, Any] = field(default_factory=dict)


def ensure(ctx: Any) -> MatteState:
    state = ctx.state.matte
    if state is None:
        state = MatteState()
        ctx.state.matte = state
    return state


def key(job_id: str) -> str:
    return f"matte-preview:{job_id}"


def open_for(ctx: Any, job_id: str, kwargs: dict[str, Any], *, force: bool = False) -> MatteState:
    """Put the preview in front of the promotion. Draws nothing itself."""
    state = ensure(ctx)
    state.job_id = job_id
    state.kwargs = dict(kwargs)
    state.force = force
    state.stamp = None
    state.preview = None
    state.failed_stamp = None
    state._open = False
    return state


def close(ctx: Any) -> None:
    state = ctx.state.matte
    if state is not None:
        state.job_id = ""
        state.preview = None
        state.stamp = None
        state.failed_stamp = None
        state._open = False


def is_open(ctx: Any) -> bool:
    """Whether the preview modal is up. Tolerant of a partial ctx.

    ``getattr`` rather than attribute access, which is ``docmodes.guard``'s rule
    and here for the same reason: ``App._modal_open`` asks this on every key
    press to decide whether a shortcut is suppressed (UX-08), and it must not
    require a state object that a caller with no matte preview has never built.
    """
    state = getattr(getattr(ctx, "state", None), "matte", None)
    return bool(state is not None and state.job_id)


def pump(ctx: Any) -> MatteState | None:
    """Keep the open preview in step with the file. Frame thread; cheap.

    One ``stat`` per frame, which is the price of the stamp being the file's
    own mtime and is what makes a Fix-matte round trip show the edited cutout
    rather than the one that sent the user to the editor. The submit is
    refused while one is already in flight (``TaskRunner.submit``'s contract),
    so calling this every frame costs nothing extra.
    """
    state = ctx.state.matte
    if state is None or not state.job_id:
        return None
    stamp = svc_matte.stamp_for(ctx.svc.job_dir(state.job_id) / "input.png")
    if state.preview is not None and state.stamp == stamp:
        return state
    state.stamp = stamp
    hit = svc_matte.cached(state.cache, state.job_id, stamp)
    if hit is not None:
        state.preview = hit
        return state
    state.preview = None
    if state.failed_stamp == stamp:
        # Already tried these exact bytes and it failed. The toast has been
        # shown once; asking again would only show it again.
        return state
    ctx.submit(key(state.job_id), svc_matte.preview, ctx.svc, state.job_id)
    return state


def busy(ctx: Any) -> bool:
    state = ctx.state.matte
    return bool(state is not None and state.job_id and ctx.busy(key(state.job_id)))


def on_task_failed(ctx: Any, done: Any) -> None:
    """A cutout that could not be computed. Frame thread.

    The seventh arm of ``App._collect_tasks``' failure dispatch, and the one
    that was missing. Every mode prefix routed its failures somewhere and
    ``matte-`` routed nowhere, so the failure fell through to the shared toast
    and ``continue`` -- leaving ``pump`` to re-submit the identical task on the
    very next frame, forever.

    Latching the stamp rather than closing the modal: the user asked to see a
    cutout and closing it under them answers a different question. The toast
    (raised by ``_collect_tasks`` before this runs, with the refusal's own
    ``field``) says what went wrong; the modal stays up with its Cancel, and
    Fix matte still clears the latch by moving the file's mtime.
    """
    state = ctx.state.matte
    if state is None or not state.job_id:
        return
    if not str(getattr(done, "key", "")).endswith(state.job_id):
        # A cutout for a reference the user has since moved off. The current
        # one is untouched -- latching here would suppress a submit that has
        # never been tried.
        return
    state.failed_stamp = state.stamp


def on_task_done(ctx: Any, done: Any) -> None:
    """Adopt a finished cutout. Frame thread -- see the module docstring."""
    state = ensure(ctx)
    preview = done.result
    if preview is None or not hasattr(preview, "job_id"):
        return
    svc_matte.remember(state.cache, preview)
    if preview.job_id != state.job_id:
        # The user closed the modal, or moved to another reference, while the
        # cutout was being computed. Cached (it is still true about that file)
        # and not shown.
        return
    state.preview = preview
    state.stamp = preview.stamp


def accept(ctx: Any, submit: Any) -> None:
    """Run ``submit(kwargs, force)`` for the previewed job and close.

    The promotion itself belongs to the pane -- it owns the task key and the
    filter nudge -- so this hands the captured settings back rather than
    reaching for ``service.jobs``.
    """
    state = ctx.state.matte
    if state is None or not state.job_id:
        return
    kwargs, force = dict(state.kwargs), state.force
    close(ctx)
    submit(kwargs, force)


def fix(ctx: Any) -> None:
    """"Fix matte": the reference, in Inker, with this cutout as its alpha."""
    from . import inker_mode

    state = ctx.state.matte
    if state is None or not state.job_id:
        return
    job = ctx.cache.get(state.job_id) or {"id": state.job_id}
    close(ctx)
    inker_mode.open_job_reference(ctx, job, matte=True)
