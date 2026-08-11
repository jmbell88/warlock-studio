"""Packwright's file layer: the facade, the staged writes and the framed opens.

``packwright_mode`` is a facade over this module, and the pins below say so by
identity rather than by behaviour: every pane, every key binding and the whole
of ``tests/test_packwright_mode.py`` says ``packwright_mode.save(ctx)``, so a
wrapper here would be a second object where all of them reach for one.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from test_packwright_mode import FakeCtx, _Done, _pack, _tab

from warlock.studio import docmodes, packwright_io, packwright_mode, packwright_state
from warlock.studio.packwright import wpack


def test_the_mode_re_exports_the_io_layer():
    for name in (
        "WPACK_FILTER",
        "PNG_FILTER",
        "IMAGE_FILTER",
        "_load",
        "_decode",
        "_start",
        "ask_open",
        "open_path",
        "save",
        "save_as",
        "save_to",
        "export_files",
        "export_library",
        "edit_asset_in_packwright",
    ):
        assert getattr(packwright_mode, name) is getattr(packwright_io, name), name


def test_the_mode_re_exports_the_state_layer():
    """``ensure`` and ``active`` touch nothing but ``ctx.state.packwright``,
    which is ``packwright_state``'s whole charter."""
    for name in ("PackTab", "PackwrightState", "ensure", "active"):
        assert getattr(packwright_mode, name) is getattr(packwright_state, name), name


def test_the_shared_helpers_are_the_shared_objects():
    assert packwright_io._start is docmodes.start_save
    assert packwright_io._decode is docmodes.decode_rgba
    assert packwright_state.title_for is docmodes.title_for


# --- staged writes ------------------------------------------------------------


def _temps(directory: Path) -> list[str]:
    return [p.name for p in directory.iterdir() if p.name.endswith(".tmp")]


def test_a_save_leaves_no_temporary_behind(tmp_path):
    """The staging name is a dotfile *and* the cleanup is a ``finally``: the
    temporary used to be ``atlas.wpack.tmp``, sorted right beside the file it is
    a fragment of, and abandoned there by any failing write."""
    ctx = FakeCtx()
    tab = _tab(ctx)
    path = tmp_path / "atlas.wpack"
    packwright_mode.save_to(ctx, tab, path)
    assert path.exists()
    assert _temps(tmp_path) == []


def test_a_failed_replace_leaves_the_previous_file_intact(tmp_path, monkeypatch):
    ctx = FakeCtx()
    tab = _tab(ctx)
    path = tmp_path / "atlas.wpack"
    path.write_bytes(b"the previous save")

    def refuse(src, dst):
        raise OSError("the disk said no")

    monkeypatch.setattr(packwright_io.os, "replace", refuse)
    with pytest.raises(OSError, match="the disk said no"):
        packwright_mode.save_to(ctx, tab, path)
    assert path.read_bytes() == b"the previous save"
    assert _temps(tmp_path) == []


def test_an_export_that_cannot_encode_writes_nothing(tmp_path, monkeypatch):
    """Every file is staged before any is replaced, which is the whole reason
    this passes: the PNG used to be on disk on top of the previous export by the
    time the sidecar raised."""
    from warlock.studio import dialogs
    from warlock.studio.packwright import texturepacker

    ctx = FakeCtx()
    tab = _tab(ctx)
    packwright_mode.set_settings(ctx, tab, mode="grid")
    _pack(ctx, tab)
    (tmp_path / "out.png").write_bytes(b"the previous export")
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: tmp_path / "out.png")

    def boom(*a, **k):
        raise ValueError("no sidecar today")

    monkeypatch.setattr(texturepacker, "tp_bytes", boom)
    with pytest.raises(ValueError, match="no sidecar"):
        packwright_mode.export_files(ctx, tab)
    assert (tmp_path / "out.png").read_bytes() == b"the previous export"
    assert not (tmp_path / "out.json").exists()
    assert _temps(tmp_path) == []


def test_a_renamed_sprite_reaches_the_exported_sidecar(tmp_path, monkeypatch):
    """The end of the chain the pane bypass broke: rename -> re-arm -> repack ->
    the layout the TexturePacker ``filename`` is written from."""
    import json

    from warlock.studio import dialogs

    ctx = FakeCtx()
    tab = _tab(ctx)
    packwright_mode.rename_source(ctx, tab, tab.doc.sources[0].uid, "hero")
    _pack(ctx, tab)
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: tmp_path / "out.png")
    packwright_mode.export_files(ctx, tab)

    sidecar = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert "hero" in json.dumps(sidecar)


def test_a_whole_set_lands_together(tmp_path):
    written = {
        tmp_path / "a.png": b"one",
        tmp_path / "nested" / "b.json": b"two",
        tmp_path / "c.tsx": b"three",
    }
    packwright_io._write(written)
    for path, blob in written.items():
        assert path.read_bytes() == blob
    assert _temps(tmp_path) == []


def test_a_staged_write_uses_a_dotfile(tmp_path, monkeypatch):
    seen: list[str] = []
    real = os.replace

    def note(src, dst):
        seen.append(Path(src).name)
        real(src, dst)

    monkeypatch.setattr(packwright_io.os, "replace", note)
    packwright_io._write({tmp_path / "atlas.wpack": b"x"})
    assert seen == [".atlas.wpack.tmp"]


# --- opening ------------------------------------------------------------------


def test_a_file_past_the_read_ceiling_is_refused_before_it_is_read(tmp_path, monkeypatch):
    from warlock.service import files as svc_files
    from warlock.service.errors import TooLarge

    path = tmp_path / "atlas.wpack"
    path.write_bytes(wpack.wpack_bytes(_tab(FakeCtx()).doc))
    monkeypatch.setattr(svc_files, "MAX_PACK_SOURCE_BYTES", 8)
    with pytest.raises(TooLarge) as caught:
        packwright_io._load(path)
    assert caught.value.field == "file"


def test_a_corrupt_file_is_refused_with_the_reason_rather_than_the_log(tmp_path):
    """Only a ``ServiceError``'s text passes the task classifier, so a bare
    ``ValueError`` here reached the user as "see the log for details" -- which
    names neither the file nor what is wrong with it."""
    from warlock.service.errors import ServiceError

    path = tmp_path / "atlas.wpack"
    path.write_bytes(b"not a zip at all")
    with pytest.raises(ServiceError) as caught:
        packwright_io._load(path)
    assert caught.value.message.startswith("This atlas could not be opened")
    assert "not a Warlock atlas" in caught.value.message


def test_an_open_that_fails_drops_the_file_off_the_resume_list(tmp_path):
    """A recent list that keeps offering a file that does not open is worse
    than a short one. The key carries the path rather than a hash of it
    precisely so this can be done."""
    ctx = FakeCtx()
    packwright_mode.ensure(ctx)
    path = tmp_path / "gone.wpack"
    packwright_mode.remember_path(ctx, path)
    assert packwright_mode.recent_paths(ctx) == [str(path)]

    packwright_mode.on_task_failed(ctx, _Done(f"{packwright_io.OPEN_PREFIX}{path}"))
    assert packwright_mode.recent_paths(ctx) == []


def test_an_open_key_keeps_a_drive_letter_intact():
    """Split on the first colon only: ``packwright-open:D:/atlases/one.wpack``
    is one key naming one path, not a prefix and a drive."""
    ctx = FakeCtx(accept=False)
    packwright_mode.ensure(ctx)
    packwright_io.open_path(ctx, Path("D:/atlases/one.wpack"))
    key = ctx.submitted[-1]
    assert key.startswith(packwright_io.OPEN_PREFIX)
    assert Path(key.split(":", 1)[1]) == Path("D:/atlases/one.wpack")


def test_an_asset_with_no_atlas_beside_it_says_so(tmp_path):
    from warlock.service.errors import ServiceError

    class _Svc:
        pass

    ctx = FakeCtx(svc=_Svc())
    from warlock.service import files as svc_files

    original = svc_files.packwright_source_path
    try:
        svc_files.packwright_source_path = lambda svc, job_id: tmp_path / "nothing.wpack"
        with pytest.raises(ServiceError, match="no atlas document"):
            packwright_io.edit_asset_in_packwright(ctx, {"id": "j1"})
    finally:
        svc_files.packwright_source_path = original
