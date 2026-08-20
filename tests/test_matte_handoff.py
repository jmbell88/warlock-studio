"""The promote preview's wiring, and the Inker hand-off it opens.

No window: everything asserted here is about *when* work is dispatched, what a
cutout does to a document's alpha and history, and whether the tab that comes
back knows it holds an unsaved edit. The drawing itself is one imgui call over
a texture the ThumbnailCache already owns the rules for.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
from PIL import Image

from warlock.studio import inker, inker_mode, matte_preview
from warlock.studio.panes import settings_3d
from warlock.studio.state import DEFAULT_FORM_3D, AppState


class _Ctx:
    """Enough of ``Ctx`` for the frame-thread half."""

    def __init__(self, svc=None) -> None:
        self.state = AppState()
        self.state.form_3d = dict(DEFAULT_FORM_3D)
        self.svc = svc if svc is not None else SimpleNamespace(job_dir=lambda _id: _MISSING)
        self.submitted: list = []
        self.cache = SimpleNamespace(get=lambda _id: None, invalidate=lambda: None)
        self.textures = None
        self.toasts: list = []

    def submit(self, key, fn, *args, **kwargs):
        self.submitted.append((key, fn, args, kwargs))
        return True

    def busy(self, key):
        return any(k == key for k, *_ in self.submitted)

    def toast(self, message, level="info", **extra):
        # ``promote`` says why it refused now, rather than returning in
        # silence, so the frame-thread half needs somewhere to say it.
        self.toasts.append((message, level))


class _MissingDir:
    def __truediv__(self, _name):
        from pathlib import Path

        return Path("nowhere") / _name


_MISSING = _MissingDir()


def _png(pixels: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(pixels, "RGBA" if pixels.shape[2] == 4 else "RGB").save(buf, "PNG")
    return buf.getvalue()


def _subject(size: int = 64) -> np.ndarray:
    arr = np.full((size, size, 3), 230, dtype=np.uint8)
    arr[16:48, 16:48] = (20, 30, 40)
    return arr


# --- the promote flow -------------------------------------------------------


def test_make_3d_opens_the_preview_instead_of_submitting():
    """The whole point: the two minutes of GPU wait behind a look at the matte."""
    ctx = _Ctx()
    source = {"id": "abc123", "status": "done", "files": ["input.png"], "params": {}}

    settings_3d.promote(ctx, source, ctx.state.form_3d)

    assert ctx.submitted == []
    assert matte_preview.is_open(ctx)
    assert ctx.state.matte.job_id == "abc123"


def test_the_form_is_captured_when_the_button_is_pressed():
    """The preview is of the settings the user pressed with, so a form edited
    while the modal is open must not change what Accept queues."""
    ctx = _Ctx()
    ctx.state.form_3d["size_m"] = 2.0
    source = {"id": "abc123", "status": "done", "files": ["input.png"], "params": {}}

    settings_3d.promote(ctx, source, ctx.state.form_3d)
    ctx.state.form_3d["size_m"] = 9.0

    assert ctx.state.matte.kwargs["size_m"] == 2.0


def test_accept_hands_back_the_captured_settings_and_closes():
    ctx = _Ctx()
    source = {"id": "abc123", "status": "done", "files": ["input.png"], "params": {}}
    settings_3d.promote(ctx, source, ctx.state.form_3d)
    seen: list = []

    matte_preview.accept(ctx, lambda kwargs, force: seen.append((kwargs, force)))

    assert seen and seen[0][0]["rig"] is False
    assert not matte_preview.is_open(ctx)


def test_an_unfinished_reference_never_opens_the_preview():
    ctx = _Ctx()
    settings_3d.promote(ctx, {"id": "abc", "status": "running", "files": []}, ctx.state.form_3d)
    assert not matte_preview.is_open(ctx)


def test_pump_submits_the_cutout_to_a_task_and_never_computes_it(svc, monkeypatch):
    """BiRefNet is seconds of host compute; the frame thread only asks."""
    from warlock.service import jobs as svc_jobs

    job_id = svc_jobs.import_reference(svc, _png(_subject()))["id"]
    ctx = _Ctx(svc)
    called = []
    monkeypatch.setattr(
        matte_preview.svc_matte, "preview", lambda *a, **k: called.append(a) or None
    )
    matte_preview.open_for(ctx, job_id, {})

    state = matte_preview.pump(ctx)

    assert called == []  # submitted, not run
    assert [k for k, *_ in ctx.submitted] == [matte_preview.key(job_id)]
    assert state.preview is None


def test_a_result_for_a_job_the_user_left_is_cached_but_not_shown(svc):
    from warlock.service import jobs as svc_jobs
    from warlock.service import matte as svc_matte

    job_id = svc_jobs.import_reference(svc, _png(_subject()))["id"]
    ctx = _Ctx(svc)
    matte_preview.open_for(ctx, "someone-else", {})
    preview = svc_matte.replace_stamp(svc_matte.preview(svc, job_id), 1)

    matte_preview.on_task_done(ctx, SimpleNamespace(result=preview, key="matte-preview:x"))

    assert ctx.state.matte.preview is None
    assert svc_matte.cached(ctx.state.matte.cache, job_id, 1) is preview


def test_fix_opens_the_reference_in_inker_with_the_matte(svc, monkeypatch):
    from warlock.service import jobs as svc_jobs

    job_id = svc_jobs.import_reference(svc, _png(_subject()))["id"]
    ctx = _Ctx(svc)
    matte_preview.open_for(ctx, job_id, {})
    seen = {}
    monkeypatch.setattr(
        inker_mode,
        "open_job_reference",
        lambda c, job, **kw: seen.update({"job": job, **kw}),
    )

    matte_preview.fix(ctx)

    assert seen["matte"] is True
    assert seen["job"]["id"] == job_id
    assert not matte_preview.is_open(ctx)


# --- the document the hand-off produces -------------------------------------


def test_apply_matte_cuts_alpha_as_one_undoable_step():
    doc = inker.Document.blank(8, 8)
    doc.stack.active.pixels[:, :] = (10, 20, 30, 255)
    doc.invalidate_all()
    doc.add_layer()
    doc.stack.active.pixels[:, :] = (200, 0, 0, 255)
    doc.invalidate_all()
    head = doc.history.head
    alpha = np.zeros((8, 8), dtype=np.uint8)
    alpha[2:6, 2:6] = 255

    assert doc.apply_matte(alpha) is True
    for layer in doc.stack:
        assert int(layer.pixels[0, 0, 3]) == 0
        assert int(layer.pixels[3, 3, 3]) == 255
    # One Ctrl+Z, not one per layer.
    assert doc.undo() is True
    assert doc.history.head == head
    assert int(doc.stack.active.pixels[0, 0, 3]) == 255


def test_apply_matte_clears_the_flatten_matte_so_the_cut_survives_a_save():
    """``matte_for`` gives an opaque photo white to flatten onto, which would
    put every cut pixel straight back on save -- the exact file this avoids."""
    pixels = np.dstack([_subject(8), np.full((8, 8), 255, np.uint8)])
    doc = inker.Document.from_pixels(pixels)
    assert doc.matte is not None

    doc.apply_matte(np.zeros((8, 8), dtype=np.uint8))

    assert doc.matte is None
    flat = doc.flatten()
    assert int(flat[..., 3].max()) == 0


def test_apply_matte_refuses_a_plane_that_is_not_the_canvas():
    doc = inker.Document.blank(8, 8)
    assert doc.apply_matte(np.zeros((4, 4), dtype=np.uint8)) is False
    assert doc.history.head == 0


def test_the_matte_handoff_opens_a_dirty_tab(svc):
    """The cutout is on screen and in no file, so closing must ask first."""
    from warlock.service import jobs as svc_jobs
    from warlock.studio.inker_state import InkerDoc

    job_id = svc_jobs.import_reference(svc, _png(_subject()))["id"]

    result = inker_mode._load_job(svc, job_id, matte=True)

    tab = InkerDoc(doc=result["doc"], saved_head=result["saved_head"])
    assert tab.dirty
    assert int(result["doc"].composite[..., 3].min()) == 0


def test_without_the_matte_the_tab_opens_clean(svc):
    from warlock.service import jobs as svc_jobs
    from warlock.studio.inker_state import InkerDoc

    job_id = svc_jobs.import_reference(svc, _png(_subject()))["id"]

    result = inker_mode._load_job(svc, job_id)

    assert "saved_head" not in result
    assert not InkerDoc(doc=result["doc"], saved_head=result["doc"].history.head).dirty


def test_a_failed_cut_still_opens_the_reference(svc, monkeypatch):
    """The user asked for the editor; the matte is the improvement they did
    not ask for, so it log-and-swallows like every other one."""
    from warlock.service import jobs as svc_jobs
    from warlock.service import matte as svc_matte

    job_id = svc_jobs.import_reference(svc, _png(_subject()))["id"]
    monkeypatch.setattr(
        svc_matte, "alpha_plane", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )

    result = inker_mode._load_job(svc, job_id, matte=True)

    assert result["doc"] is not None
    assert result["job_id"] == job_id


# --- a cutout that fails ------------------------------------------------------


class _Failed:
    """The shape ``TaskRunner.poll`` hands ``_collect_tasks`` for a failure."""

    ok = False

    def __init__(self, key: str) -> None:
        self.key = key
        self.error = ValueError("this job has no reference image")


def test_a_failed_cutout_is_asked_for_once_and_not_once_per_frame(svc, monkeypatch):
    """``matte_modal`` runs ``pump`` every frame, and a failure left no note
    behind: the cache missed, the submit was accepted because the previous task
    had already failed, and the modal sat on "Cutting the subject out..."
    behind one new red toast per frame.

    The failure to model is the cheap one -- ``service.matte.preview`` refuses
    a reference with no ``input.png`` *before* any compute -- because that is
    the one that spins at the full frame rate.
    """
    from warlock.service import jobs as svc_jobs

    job_id = svc_jobs.import_reference(svc, _png(_subject()))["id"]
    ctx = _Ctx(svc)
    monkeypatch.setattr(matte_preview.svc_matte, "preview", lambda *a, **k: None)
    matte_preview.open_for(ctx, job_id, {})

    matte_preview.pump(ctx)
    assert len(ctx.submitted) == 1

    matte_preview.on_task_failed(ctx, _Failed(matte_preview.key(job_id)))
    for _frame in range(5):
        matte_preview.pump(ctx)
    assert len(ctx.submitted) == 1, "the failed cutout was asked for again"


def test_the_modal_stays_open_after_a_failure(svc, monkeypatch):
    """The user asked to see a cutout; closing the modal under them answers a
    different question. The toast says what went wrong and Cancel still works."""
    from warlock.service import jobs as svc_jobs

    job_id = svc_jobs.import_reference(svc, _png(_subject()))["id"]
    ctx = _Ctx(svc)
    monkeypatch.setattr(matte_preview.svc_matte, "preview", lambda *a, **k: None)
    matte_preview.open_for(ctx, job_id, {})
    matte_preview.pump(ctx)

    matte_preview.on_task_failed(ctx, _Failed(matte_preview.key(job_id)))

    assert matte_preview.is_open(ctx)


def test_a_failure_for_another_reference_does_not_latch_this_one(svc, monkeypatch):
    """The latch is per-stamp on the *open* preview. A result arriving for a
    reference the user has moved off must not suppress a submit that has never
    been tried."""
    from warlock.service import jobs as svc_jobs

    job_id = svc_jobs.import_reference(svc, _png(_subject()))["id"]
    ctx = _Ctx(svc)
    monkeypatch.setattr(matte_preview.svc_matte, "preview", lambda *a, **k: None)
    matte_preview.open_for(ctx, job_id, {})
    matte_preview.pump(ctx)

    matte_preview.on_task_failed(ctx, _Failed(matte_preview.key("someone-else")))

    assert ctx.state.matte.failed_stamp is None
