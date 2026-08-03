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
