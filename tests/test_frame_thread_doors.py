"""Nothing expensive runs on the frame thread -- the doors the review named.

The 2026-09-02 review's theme T2 listed nine places where a decode, an encode
or a batch of inserts ran on the pygame frame thread: the Review sweep launch
and its mesh loads, the reference PNG on job completion, the Clay crash
recovery, the Troupe atlas, the Inker revert reload, the Packwright save
encodes, the Sirens sample re-encode per snapshot, and the settings flush
during a splitter drag. Each had a sibling in the tree that already did it
right (the GLB parse/adopt split, the ``inker-recover`` task), and each is
now that shape.

The guard here is behavioural where it can be: a ctx whose ``submit`` runs
the task on a *real worker thread* and joins, with the expensive function
spied to record which thread it ran on. A regression that moves the decode
back into the frame-thread half shows up as the test thread's name. The
Review launch and mesh load are pinned in ``tests/test_review_mode.py``.
"""

from __future__ import annotations

import inspect
import threading
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image
from test_clay_mode import FakeCtx as ClayCtx
from test_clay_mode import _tab as clay_tab
from test_inker_mode import _PaletteCtx
from test_sirens_mode import FakeCtx as SirensCtx
from test_sirens_mode import _tab as sirens_tab

from warlock.studio import clay_mode, inker, inker_mode, packwright_io, troupe_mode
from warlock.studio.clay import document as clay_document
from warlock.studio.clay import serialize as clay_serialize
from warlock.studio.inker_state import InkerDoc
from warlock.studio.packwright import wpack
from warlock.studio.packwright.document import PackDoc
from warlock.studio.packwright.sources import Sprite
from warlock.studio.sirens import wsng
from warlock.studio.viewer_embed import Viewer

WORKER = "warlock-task-test"


@dataclass
class _Done:
    key: str
    result: Any = None
    tag: Any = None
    error: Any = None
    message: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None


class _Threaded:
    """``submit`` on a real worker thread, joined -- so the test sees the task
    half run where it would run, and the recorder below can name the thread."""

    submitted: list[str]
    tags: list[Any]
    result: Any

    def submit(self, key: str, fn: Any, *args: Any, tag: Any = None, **kwargs: Any) -> bool:
        self.submitted.append(key)
        self.tags.append(tag)
        box: dict[str, Any] = {}

        def go() -> None:
            try:
                box["result"] = fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - re-raised on the caller
                box["error"] = exc

        worker = threading.Thread(target=go, name=WORKER)
        worker.start()
        worker.join()
        if "error" in box:
            raise box["error"]
        self.result = box["result"]
        return True


def _spy(monkeypatch: pytest.MonkeyPatch, owner: Any, name: str) -> list[str]:
    """Wrap ``owner.name`` to record the thread each call ran on."""
    real = getattr(owner, name)
    threads: list[str] = []

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        threads.append(threading.current_thread().name)
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, name, wrapped)
    return threads


class _GL:
    """A texture factory: the one GL call the frame-thread half makes."""

    NEAREST = 0
    LINEAR = 1

    def __init__(self) -> None:
        self.uploads: list[tuple[tuple[int, int], int]] = []

    def texture(self, size: Any, components: int, data: bytes) -> Any:
        assert len(data) == size[0] * size[1] * components
        self.uploads.append((tuple(size), len(data)))
        return SimpleNamespace(size=tuple(size), filter=None, release=lambda: None)


def _png(path: Path, size: tuple[int, int] = (8, 4)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", size, (10, 20, 30, 255)).save(path)
    return path


# --- 1. the reference picture on job completion -------------------------------


def test_the_reference_picture_is_decoded_apart_from_its_upload(tmp_path):
    parsed = Viewer.parse_reference(_png(tmp_path / "input.png"))
    assert parsed[0] == (8, 4) and len(parsed[1]) == 8 * 4 * 4

    viewer = object.__new__(Viewer)
    viewer.ctx = _GL()
    viewer.reference = None
    viewer._forget = lambda texture: None
    viewer.adopt_reference(parsed)
    assert viewer.reference.size == (8, 4)
    assert viewer.reference.filter == (_GL.LINEAR, _GL.LINEAR)


def test_the_viewer_sync_submits_the_decode_and_adopts_it_on_landing():
    """The timer path never calls the blocking ``load_reference``."""
    from warlock.studio import main

    sync = inspect.getsource(main.App._sync_viewer)
    assert "parse_reference" in sync and "load_reference(" not in sync
    adopt = inspect.getsource(main.App._adopt_model)
    assert "adopt_reference" in adopt


# --- 2. the Troupe atlas ------------------------------------------------------


class _TroupeCtx(_Threaded):
    def __init__(self, root: Path) -> None:
        self.root = root
        self.state = SimpleNamespace(troupe=None, preview={}, mode="troupe")
        self.viewer = SimpleNamespace(ctx=_GL())
        self.submitted, self.tags, self.result = [], [], None
        self.toasts: list[tuple[str, str]] = []

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def busy(self, key: str) -> bool:
        return False

    def toast(self, text: str, level: str = "info", *a: Any, **k: Any) -> None:
        self.toasts.append((text, level))


def _sheet(ctx: _TroupeCtx) -> tuple[str, str]:
    from warlock import rigging

    state = troupe_mode.ensure(ctx)
    state.job_id, state.sheet_id = "job1", rigging.new_id()
    _png(rigging.sheet_png_path(ctx.job_dir("job1"), state.sheet_id), (16, 8))
    return "job1", state.sheet_id


def test_the_troupe_atlas_is_decoded_on_a_task_and_uploaded_when_it_lands(tmp_path, monkeypatch):
    ctx = _TroupeCtx(tmp_path)
    key = _sheet(ctx)
    threads = _spy(monkeypatch, troupe_mode, "_decode_atlas")

    assert troupe_mode.atlas_texture(ctx) is None, "not yet: the decode is in flight"
    assert ctx.submitted == [troupe_mode.atlas_key(*key)]
    assert threads == [WORKER]
    assert ctx.viewer.ctx.uploads == [], "nothing touched GL on the frame thread"

    troupe_mode.on_task_done(ctx, _Done(ctx.submitted[-1], ctx.result, tag=key))
    texture = troupe_mode.atlas_texture(ctx)
    assert texture is not None and texture.size == (16, 8)
    assert ctx.viewer.ctx.uploads == [((16, 8), 16 * 8 * 4)]
    assert len(ctx.submitted) == 1, "cached: the second ask decodes nothing"


def test_a_decoded_atlas_for_a_sheet_no_longer_on_screen_is_dropped(tmp_path):
    from warlock import rigging

    ctx = _TroupeCtx(tmp_path)
    key = _sheet(ctx)
    troupe_mode.atlas_texture(ctx)
    troupe_mode.ensure(ctx).sheet_id = rigging.new_id()
    troupe_mode.on_task_done(ctx, _Done(ctx.submitted[-1], ctx.result, tag=key))
    assert ctx.viewer.ctx.uploads == []
    assert "troupe_texture" not in ctx.state.preview


def test_an_unreadable_atlas_is_tried_once(tmp_path):
    ctx = _TroupeCtx(tmp_path)
    key = _sheet(ctx)
    troupe_mode.atlas_texture(ctx)
    troupe_mode.on_task_failed(ctx, _Done(ctx.submitted[-1], tag=key, error=OSError("bad png")))
    for _ in range(3):
        assert troupe_mode.atlas_texture(ctx) is None
    assert len(ctx.submitted) == 1


# --- 3. the Clay crash recovery -----------------------------------------------


class _ClayCtx(_Threaded, ClayCtx):
    def __init__(self) -> None:
        ClayCtx.__init__(self)
        self.tags: list[Any] = []


def test_a_recovered_clay_model_is_read_on_a_task_and_adopted_dirty(tmp_path, monkeypatch):
    ctx = _ClayCtx()
    authored = clay_tab(ctx).doc
    path = tmp_path / "crate.wblk"
    path.write_bytes(clay_serialize.wblk_bytes(authored))
    threads = _spy(monkeypatch, clay_serialize, "read_wblk")

    assert clay_mode._journal_adopt(ctx, path, {"title": "Crate"}) is True
    assert ctx.submitted[-1].startswith("clay-recover:")
    assert threads == [WORKER]

    clay_mode.on_task_done(ctx, _Done(ctx.submitted[-1], ctx.result))
    tab = next(t for t in ctx.state.clay.docs if t.title == "Crate (recovered)")
    assert tab.path is None
    assert tab.saved_head == -1 and tab.dirty
    assert tab.journal_name == path.name


def test_a_recovered_clay_model_that_will_not_parse_says_so(tmp_path):
    """Through ``journal.adopt_failed`` since 2026-09-05 -- a warning with the
    log behind it, the sentence every provider says -- where a raise here
    arrived as an *error* toast no other mode's copy raised."""
    path = tmp_path / "bad.wblk"
    path.write_bytes(b"not a zip")
    assert clay_mode._load_recovery(path, {}) is None


def test_the_clay_uid_counter_survives_a_reserve_from_a_task_thread():
    """``reserve_uid`` swaps the counter out from under ``new_uid``; both hold
    the lock now, and a burst from both sides mints no duplicate."""
    minted: list[int] = []
    stop = threading.Event()

    def reserve_loop() -> None:
        base = clay_document.new_uid()
        while not stop.is_set():
            clay_document.reserve_uid(base + 5)

    worker = threading.Thread(target=reserve_loop, name=WORKER)
    worker.start()
    try:
        for _ in range(2000):
            minted.append(clay_document.new_uid())
    finally:
        stop.set()
        worker.join()
    assert len(set(minted)) == len(minted)
    assert minted == sorted(minted), "monotonic through every reserve"


# --- 4. the Inker revert reload ----------------------------------------------


class _InkerCtx(_Threaded, _PaletteCtx):
    def __init__(self) -> None:
        _PaletteCtx.__init__(self)
        self.tags: list[Any] = []
        self.svc = None
        self.viewer = None
        self.cache = SimpleNamespace(invalidate=lambda: None)
        self.confirms = SimpleNamespace(pending=None)
        self.confirms.ask = lambda confirm: setattr(self.confirms, "pending", confirm)


def test_a_revert_decodes_the_restored_image_on_the_task(tmp_path, monkeypatch):
    from warlock.service import files as svc_files

    monkeypatch.setattr(svc_files, "revert_reference", lambda svc, job_id: None)
    monkeypatch.setattr(svc_files, "discard_inker_working", lambda svc, job_id: None)
    ctx = _InkerCtx()
    path = _png(tmp_path / "input.png", (6, 5))
    doc = inker.Document.blank(4, 4)
    tab = InkerDoc(doc=doc, title="input.png", path=path, saved_head=doc.history.head)
    tab.job_id, tab.has_original = "job7", True
    inker_mode.ensure(ctx).add(tab)
    threads = _spy(monkeypatch, inker.Document, "load")

    inker_mode.revert(ctx, tab)
    ctx.confirms.pending.on_confirm()
    assert ctx.submitted[-1] == f"inker-revert:{tab.uid}"
    assert threads == [WORKER]
    assert ctx.result["doc"] is not None and tab.doc is doc, "adopted on landing, not here"

    inker_mode.on_task_done(ctx, _Done(ctx.submitted[-1], ctx.result))
    assert tab.doc is not doc
    assert tab.doc.size == (6, 5)
    assert tab.has_original is False
    assert ctx.toasts[-1][0] == "Back to the original image."


def test_a_revert_whose_image_will_not_reopen_still_reports_the_revert(tmp_path, monkeypatch):
    from warlock.service import files as svc_files

    monkeypatch.setattr(svc_files, "revert_reference", lambda svc, job_id: None)
    monkeypatch.setattr(svc_files, "discard_inker_working", lambda svc, job_id: None)
    ctx = _InkerCtx()
    path = tmp_path / "input.png"
    path.write_bytes(b"not a png")
    doc = inker.Document.blank(4, 4)
    tab = InkerDoc(doc=doc, title="input.png", path=path, saved_head=doc.history.head)
    tab.job_id, tab.has_original = "job7", True
    inker_mode.ensure(ctx).add(tab)

    inker_mode.revert(ctx, tab)
    ctx.confirms.pending.on_confirm()
    assert ctx.result["reverted"] is True and ctx.result["doc"] is None
    inker_mode.on_task_done(ctx, _Done(ctx.submitted[-1], ctx.result))
    assert tab.doc is doc
    assert "could not be reopened" in ctx.toasts[-1][0]
    assert ctx.toasts[-1][1] == "error"


# --- 5. the Packwright save --------------------------------------------------


def _sprite(key: str) -> Sprite:
    pixels = np.zeros((6, 8, 4), dtype=np.uint8)
    pixels[1:-1, 1:-1] = (int(key[-1]) * 20, 40, 60, 255)
    return Sprite(key=key, name=f"name-{key}", pixels=pixels)


def _pack_doc() -> PackDoc:
    doc = PackDoc()
    for i in range(3):
        doc.add_source(_sprite(f"s{i}"))
    doc.mark_saved()
    return doc


def test_a_snapshot_is_what_the_document_was_when_the_save_was_pressed():
    """The document goes on being edited while the task encodes."""
    doc = _pack_doc()
    before = wpack.wpack_bytes(doc)
    snap = wpack.snapshot(doc)
    doc.add_source(_sprite("s9"))
    doc.rename_source(doc.sources[0].uid, "renamed")
    assert wpack.snapshot_bytes(snap) == before
    assert len(wpack.read_wpack(wpack.snapshot_bytes(snap)).sources) == 3


class _PackCtx(_Threaded):
    def __init__(self) -> None:
        self.submitted, self.tags, self.result = [], [], None


def test_a_packwright_save_encodes_its_pngs_on_the_task(tmp_path, monkeypatch):
    threads = _spy(monkeypatch, wpack, "png_bytes")
    ctx = _PackCtx()
    tab = SimpleNamespace(doc=_pack_doc(), uid="t1", saving=False, title="atlas")
    out = tmp_path / "atlas.wpack"

    packwright_io.save_to(ctx, tab, out)
    assert ctx.submitted == ["packwright-save:t1"]
    assert threads and set(threads) == {WORKER}
    assert wpack.read_wpack(out.read_bytes()).sources[1].key == "s1"


# --- 6. the Sirens sample re-encode ------------------------------------------


def test_a_sample_is_encoded_to_wav_once_across_snapshots(monkeypatch):
    doc = sirens_tab(SirensCtx()).doc
    doc.set_sample("kick", np.linspace(-1.0, 1.0, 256, dtype=np.float32))
    reference = wsng.wsng_bytes(doc)
    encodes = _spy(monkeypatch, wsng.wavout, "wav_bytes")

    first = wsng.wsng_bytes(doc)
    second = wsng.wsng_bytes(doc)
    assert encodes == [], "already cached from the first snapshot"
    assert first == second == reference

    doc.set_sample("kick", np.zeros(64, dtype=np.float32))
    third = wsng.wsng_bytes(doc)
    assert len(encodes) == 1, "a replaced array is a new encode"
    assert third != first
    assert len(wsng.read_wsng(third).samples["kick"]) == 64


# --- 7. the settings flush under a drag ---------------------------------------


def test_the_settings_flush_waits_for_the_mouse_button_to_come_up():
    from warlock.studio import main

    source = inspect.getsource(main)
    guarded = "if not imgui.is_any_mouse_down():\n            self.app_ctx.settings.tick()"
    assert guarded in source
    assert source.count("settings.tick()") == 1
