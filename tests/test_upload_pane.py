"""The 3D pane's upload path reads the file off the frame thread.

Both callers (the picker's completion and the drop handler) run in the frame
loop, so a synchronous read_bytes of a large file froze the window -- and
allocated the whole file before create_job's byte cap ever saw it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from warlock.service.errors import Invalid, TooLarge
from warlock.service.validation import MAX_UPLOAD_BYTES
from warlock.studio.panes import settings_3d
from warlock.studio.state import DEFAULT_FORM_3D


class _Ctx:
    def __init__(self) -> None:
        self.state = SimpleNamespace(form_3d=dict(DEFAULT_FORM_3D))
        self.svc = object()
        self.submitted: list = []

    def submit(self, key, fn, *args, **kwargs):
        self.submitted.append((key, fn, args, kwargs))
        return True


def test_upload_hands_the_file_read_to_a_task(tmp_path, monkeypatch):
    """Nothing is read on the calling thread: the path need not even exist
    until the closure runs."""
    seen = {}
    monkeypatch.setattr(
        settings_3d.svc_jobs, "create_job", lambda svc, **kw: seen.update(kw) or "id"
    )
    ctx = _Ctx()
    path = tmp_path / "ref.png"
    settings_3d.upload(ctx, path)

    assert [key for key, *_ in ctx.submitted] == ["submit"]
    assert seen == {}  # the file has not been touched yet

    path.write_bytes(b"png-bytes")
    _key, fn, args, kwargs = ctx.submitted[0]
    assert fn(*args, **kwargs) == "id"
    assert seen["image"] == b"png-bytes"
    assert seen["kind"] == "image"


def test_upload_reads_at_most_one_byte_past_the_cap(tmp_path, monkeypatch):
    """create_job's contract: it only needs MAX_UPLOAD_BYTES + 1 bytes to know
    the upload is too large, so an oversized file is never fully allocated."""

    def fake_create_job(svc, *, image, **kw):
        assert len(image) == MAX_UPLOAD_BYTES + 1
        raise TooLarge("Reference image is too large (max 20 MB).")

    monkeypatch.setattr(settings_3d.svc_jobs, "create_job", fake_create_job)
    ctx = _Ctx()
    path = tmp_path / "huge.png"
    path.write_bytes(b"\x00" * (MAX_UPLOAD_BYTES + 10))
    settings_3d.upload(ctx, path)

    _key, fn, args, kwargs = ctx.submitted[0]
    with pytest.raises(TooLarge):
        fn(*args, **kwargs)


def test_an_unreadable_file_becomes_a_readable_toast(tmp_path):
    """OSError in the task would surface as a bare class name otherwise."""
    ctx = _Ctx()
    settings_3d.upload(ctx, tmp_path / "gone.png")
    _key, fn, args, kwargs = ctx.submitted[0]
    with pytest.raises(Invalid, match="could not read gone.png"):
        fn(*args, **kwargs)


# --- the 2D pane's conditioning reference ------------------------------------


class _Ctx2D:
    def __init__(self, **form) -> None:
        from warlock.studio.state import AppState, default_form_2d

        # A real ``AppState`` for the parts that are state rather than stubs:
        # ``generate`` clears the field-error rings before it submits (UX.md
        # Phase 3), and a namespace that merely happens to carry ``form_2d``
        # would have to grow a no-op for every method the pane learns to call.
        state = AppState()
        state.form_2d = {**default_form_2d(), **form}
        self.state = state
        # No active profile is ever set, so anchor_kwargs's consultation of
        # ctx.settings / ctx.svc.config short-circuits before either is read
        # for real -- these only need to exist, the same as on the live
        # AppContext.
        self.settings = SimpleNamespace(get=lambda key, default=None: default)
        self.svc = SimpleNamespace(config=SimpleNamespace(data_dir=None))
        self.submitted: list = []

    def submit(self, key, fn, *args, **kwargs):
        self.submitted.append((key, fn, args, kwargs))
        return True


def test_the_reference_is_read_off_the_frame_thread(tmp_path, monkeypatch):
    """Same rule as the 3D upload: picking a 20 MB reference must not freeze
    the window for as long as the disk takes."""
    from warlock.studio.panes import settings_2d

    seen = {}
    monkeypatch.setattr(
        settings_2d.svc_jobs, "create_job", lambda svc, **kw: seen.update(kw) or "id"
    )
    path = tmp_path / "ref.png"
    ctx = _Ctx2D(prompt="a barrel", ref_path=str(path), ip_adapter="plus")
    settings_2d.generate(ctx, ctx.state.form_2d)

    assert seen == {}  # not touched yet
    path.write_bytes(b"png-bytes")
    _key, fn, args, kwargs = ctx.submitted[0]
    assert fn(*args, **kwargs) == "id"
    assert seen["reference"] == b"png-bytes"
    assert seen["guidance_fields"]["ip_adapter"] == "plus"


def test_an_unreadable_reference_becomes_a_readable_error(tmp_path):
    from warlock.studio.panes import settings_2d

    ctx = _Ctx2D(prompt="a barrel", ref_path=str(tmp_path / "gone.png"))
    settings_2d.generate(ctx, ctx.state.form_2d)
    _key, fn, args, kwargs = ctx.submitted[0]
    with pytest.raises(Invalid, match="gone.png"):
        fn(*args, **kwargs)


def test_no_reference_means_no_reference_kwarg(tmp_path, monkeypatch):
    """The unconditioned submit must be exactly what it was before any of this
    existed -- create_job is never handed reference=None to interpret."""
    from warlock.studio.panes import settings_2d

    seen = {}
    monkeypatch.setattr(
        settings_2d.svc_jobs, "create_job", lambda svc, **kw: seen.update(kw) or "id"
    )
    ctx = _Ctx2D(prompt="a barrel")
    settings_2d.generate(ctx, ctx.state.form_2d)
    _key, fn, args, kwargs = ctx.submitted[0]
    fn(*args, **kwargs)
    assert "reference" not in seen


def test_a_scale_is_sent_only_with_the_selection_it_scales():
    """Mirrors lora_weight: an unused slider must not reach params as a live
    setting the next run would inherit."""
    from warlock.studio.panes import settings_2d
    from warlock.studio.state import default_form_2d

    bare = settings_2d.submit_kwargs({**default_form_2d(), "prompt": "a barrel"})
    assert bare["ip_scale"] is None
    assert bare["control_scale"] is None

    picked = settings_2d.submit_kwargs(
        {
            **default_form_2d(),
            "prompt": "a barrel",
            "ip_adapter": "plus",
            "ip_scale": 0.9,
            "control": "canny",
            "control_scale": 0.4,
        }
    )
    assert picked["ip_scale"] == 0.9
    assert picked["control_scale"] == 0.4


def test_validate_catches_what_a_restored_form_can_still_be():
    """ref_path is VOLATILE, so a persisted selection outlives the image that
    justified it -- and the base model can be changed after a control is
    picked."""
    from warlock.studio.panes import settings_2d
    from warlock.studio.state import default_form_2d

    orphan = {**default_form_2d(), "prompt": "a barrel", "ip_adapter": "plus"}
    assert any("reference image" in p for p in settings_2d.validate(orphan))

    wrong_base = {
        **default_form_2d(),
        "prompt": "a barrel",
        "ref_path": "D:/x.png",
        "control": "canny",
        "base_model": "turbo",
    }
    assert any("full-CFG" in p for p in settings_2d.validate(wrong_base))

    ok = {**wrong_base, "base_model": "sdxl_cfg"}
    assert settings_2d.validate(ok) == []
