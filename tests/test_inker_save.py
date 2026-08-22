"""Saving a linked paint document, and what the save must not race with.

A linked save writes two files -- ``input.png`` and the ``paint.ora`` behind it
-- and their *order* is the only thing deciding whether reopening the reference
brings the layers back or flattens them away. That is worth pinning, because
the ordering that reads as safest is the one that breaks it.
"""

from __future__ import annotations

import io
import zipfile
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from warlock.service import files as svc_files
from warlock.service import jobs as svc_jobs
from warlock.studio import inker, inker_mode
from warlock.studio.inker_state import InkerDoc


def _png(size=(32, 32), colour=(200, 30, 30, 255)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", size, colour).save(buf, "PNG")
    return buf.getvalue()


def _reference(svc) -> str:
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x", output="reference")["id"]
    svc.job_dir(job_id).mkdir(parents=True, exist_ok=True)
    (svc.job_dir(job_id) / "input.png").write_bytes(_png())
    svc.store.set_status(job_id, "done")
    return job_id


class FakeCtx:
    """Runs the submitted callable inline, so the test sees what the task
    thread would have done without needing one.

    Carries a ``state`` because saving settles the canvas first
    (``inker_mode._settle``), and settling reads ``ctx.state.inker`` to find an
    open palette-conversion session. ``None`` is the honest value here: these
    tabs are built by hand and never went through ``inker_mode.ensure``.
    """

    def __init__(self, svc: Any) -> None:
        self.svc = svc
        self.state = SimpleNamespace(inker=None)
        self.submitted: list[str] = []

    def submit(self, key: str, run: Any) -> bool:
        self.submitted.append(key)
        self.result = run()
        return True


def _tab(job_id: str) -> InkerDoc:
    doc = inker.Document.blank(32, 32)
    doc.stack.active.pixels[..., :] = 128
    doc.add_layer()
    return InkerDoc(
        doc=doc,
        title="ref",
        job_id=job_id,
        link_kind="reference-edit",
        saved_head=0,
    )


# --- the ordering -----------------------------------------------------------


def test_a_linked_save_writes_the_flat_half_first(monkeypatch, svc):
    """The defect: the sidecar was written first, so ``input.png`` always ended
    up newer and every save marked its own layers stale. Reopening the
    reference then silently flattened a layered document to one layer.

    Asserted as call *order* rather than as the resulting mtimes, because two
    consecutive small writes land on the same ``st_mtime`` most of the time on
    a typical filesystem -- and ``inker_working_status`` compares with ``>=``,
    so an mtime assertion passes under the bug roughly two runs in three.
    """
    job_id = _reference(svc)
    order: list[str] = []

    real_flat = svc_files.save_edited_image
    real_layers = svc_files.save_inker_working

    def flat(*args, **kwargs):
        order.append("input.png")
        return real_flat(*args, **kwargs)

    def layers(*args, **kwargs):
        order.append("paint.ora")
        return real_layers(*args, **kwargs)

    monkeypatch.setattr(svc_files, "save_edited_image", flat)
    monkeypatch.setattr(svc_files, "save_inker_working", layers)

    inker_mode._save_linked(FakeCtx(svc), _tab(job_id))

    assert order == ["input.png", "paint.ora"]


def test_a_linked_save_leaves_its_layers_fresh(svc):
    """The property the order buys, stated the way a reopen sees it.

    Weaker than the ordering test above on purpose: with equal mtimes ``>=``
    reports fresh either way. It is here to catch the reverse failure -- a save
    that somehow leaves the sidecar genuinely older.
    """
    job_id = _reference(svc)
    ctx = FakeCtx(svc)

    inker_mode._save_linked(ctx, _tab(job_id))

    status = svc_files.inker_working_status(svc, job_id)
    assert status == {"exists": True, "fresh": True}


def test_a_linked_save_writes_both_halves(svc):
    job_id = _reference(svc)
    ctx = FakeCtx(svc)
    tab = _tab(job_id)

    inker_mode._save_linked(ctx, tab)

    job_dir = svc.job_dir(job_id)
    assert (job_dir / "paint.ora").exists()
    assert (job_dir / svc_files.ORIGINAL).exists()  # the flat half ran too
    with zipfile.ZipFile(job_dir / "paint.ora") as zf:
        assert "mergedimage.png" in zf.namelist()


def test_a_linked_save_still_records_the_hand_edit(svc):
    job_id = _reference(svc)
    inker_mode._save_linked(FakeCtx(svc), _tab(job_id))
    assert svc.store.get(job_id)["params"]["hand_edited"] is True


def test_the_layers_survive_a_round_trip_through_a_save(svc):
    """The property the ordering exists for, stated end to end."""
    job_id = _reference(svc)
    tab = _tab(job_id)
    inker_mode._save_linked(FakeCtx(svc), tab)

    status = svc_files.inker_working_status(svc, job_id)
    reopened = inker.Document.load(svc_files.inker_working_path(svc, job_id))
    assert status["fresh"]
    assert len(reopened.stack) == len(tab.doc.stack) == 2


def test_a_reference_rewritten_after_a_save_makes_the_layers_stale_again(svc):
    """The other half of the rule, unchanged: a revert or a regenerate must
    still win over layers describing pixels that are gone."""
    import os
    import time

    job_id = _reference(svc)
    inker_mode._save_linked(FakeCtx(svc), _tab(job_id))

    later = time.time() + 10
    os.utime(svc.job_dir(job_id) / "input.png", (later, later))
    assert not svc_files.inker_working_status(svc, job_id)["fresh"]


# --- the floating buffer, which lives in no layer -----------------------------


def _floating(tab: InkerDoc) -> None:
    """Put pixels on the canvas that a layer encode would not see."""
    import numpy as np

    tab.doc.clipboard.put(
        np.full((8, 8, 4), 255, np.uint8), np.full((8, 8), 255, np.uint8)
    )
    assert tab.doc.paste((4, 4)) is True
    assert tab.doc.floating is not None


def test_a_linked_save_is_not_dirty_the_instant_it_finishes(svc):
    """``rev`` was read before ``commit_floating``, which pushes a step -- so
    the head recorded as saved was one behind the document and every linked
    save left the tab dirty forever: the quit guard nagged on every exit and
    send_to_3d refused with "Save first" no matter how often it was saved."""
    from types import SimpleNamespace

    from warlock.studio.inker_state import InkerState

    job_id = _reference(svc)
    tab = _tab(job_id)
    _floating(tab)

    ctx = FakeCtx(svc)
    state = InkerState()
    state.add(tab)
    ctx.state = SimpleNamespace(inker=state, mode="inker")
    ctx.cache = SimpleNamespace(invalidate=lambda: None)
    ctx.toast = lambda *a, **k: None
    ctx.viewer = SimpleNamespace(path=None)

    inker_mode._save_linked(ctx, tab)
    inker_mode.on_task_done(
        ctx, SimpleNamespace(key=f"inker-save:{tab.uid}", result=ctx.result)
    )

    assert tab.dirty is False


def test_an_unlinked_save_writes_the_floating_pixels(svc, tmp_path):
    """``_write`` encodes the layer stack, and a pasted buffer is in no layer.
    The file omitted it, the canvas still showed it, and ``mark_saved`` then
    called the document clean -- so closing the tab discarded the paste with
    no prompt."""
    tab = _tab("")
    tab.link_kind = ""
    tab.job_id = ""
    tab.path = tmp_path / "doc.ora"
    tab.file_format = "ora"
    _floating(tab)

    inker_mode.save(FakeCtx(svc), tab)

    reopened = inker.Document.load(tab.path)
    assert tab.doc.floating is None, "the buffer must have landed on a layer"
    assert int(reopened.flatten()[8, 8, 3]) == 255


def _wired(svc, tab) -> Any:
    """A ctx with the pieces ``on_task_done`` reaches for."""
    from types import SimpleNamespace

    from warlock.studio.inker_state import InkerState

    ctx = FakeCtx(svc)
    state = InkerState()
    state.add(tab)
    ctx.state = SimpleNamespace(inker=state, mode="inker")
    ctx.cache = SimpleNamespace(invalidate=lambda: None)
    ctx.toast = lambda *a, **k: None
    ctx.viewer = SimpleNamespace(path=None)
    return ctx


def test_saving_a_drawing_as_a_new_reference_offers_no_revert(svc):
    """"Save as reference" mints the reference from the drawn pixels, so there
    is no ``input.orig.png`` behind it. ``has_original = False`` was set and
    then immediately overwritten by an unconditional True in the ``tab.linked``
    branch below it -- offering a Revert button that could only fail."""
    from types import SimpleNamespace

    tab = _tab("")
    tab.link_kind = ""
    ctx = _wired(svc, tab)
    job_id = _reference(svc)

    inker_mode.on_task_done(
        ctx,
        SimpleNamespace(
            key=f"inker-save:{tab.uid}", result={"link": True, "job_id": job_id, "rev": 0}
        ),
    )

    assert tab.linked
    assert tab.has_original is False


def test_an_ordinary_linked_save_still_has_an_original_to_revert_to(svc):
    from types import SimpleNamespace

    job_id = _reference(svc)
    tab = _tab(job_id)
    ctx = _wired(svc, tab)

    inker_mode.on_task_done(
        ctx, SimpleNamespace(key=f"inker-save:{tab.uid}", result={"rev": 0})
    )

    assert tab.has_original is True


def test_sending_an_unlinked_drawing_to_3d_locks_the_document_while_it_encodes(svc):
    """``png_bytes`` walks the layer stack on a task thread, so an undo, a crop
    or a rotate landing mid-walk restructures it underneath. Every other encode
    path sets ``saving``; this one did not."""
    from types import SimpleNamespace

    tab = _tab("")
    tab.link_kind = ""
    tab.job_id = ""
    ctx = _wired(svc, tab)
    locked: list[bool] = []
    ctx.submit = lambda key, run: (
        ctx.submitted.append(key) or locked.append(tab.saving) or True
    )

    inker_mode.send_to_3d(ctx, tab)

    assert ctx.submitted == [f"inker-send:{tab.uid}"]
    assert locked == [True], "the encode ran with the document unlocked"

    inker_mode.on_task_done(
        ctx, SimpleNamespace(key=f"inker-send:{tab.uid}", result={"id": "a" * 12})
    )
    assert tab.saving is False


# --- the gate ---------------------------------------------------------------


def test_the_mutating_shortcuts_do_nothing_while_a_save_is_running():
    """``_write`` encodes the live document on a task thread. That tolerates a
    stroke -- pixels are written in place -- but not the stack changing shape,
    so every shortcut that restructures it has to be held off.

    Driven through ``_ctrl_key`` rather than asserted against the constant: a
    set nothing consults gates nothing.
    """
    from types import SimpleNamespace

    import pygame

    from warlock.studio.inker_state import InkerState

    tab = _tab("")
    doc = tab.doc
    doc.select_all()  # something for undo to have to reverse
    before_head, before_layers = doc.history.head, len(doc.stack)

    state = InkerState()
    state.add(tab)
    ctx = SimpleNamespace(state=SimpleNamespace(inker=state), svc=None)

    def press(letter: str) -> None:
        """Through ``handle_key``, which is where the gate is since W2.7.

        The bindings are the op registry's now and the busy check sits in front
        of it, so driving ``_ctrl_key`` alone would test the half that no
        longer holds most of them.
        """
        key = getattr(pygame, f"K_{letter}", None)
        if key is None:
            return
        inker_mode.handle_key(
            ctx, pygame.event.Event(pygame.KEYDOWN, key=key, mod=pygame.KMOD_LCTRL)
        )

    tab.saving = True
    for name in sorted(inker_mode._MUTATING_CTRL):
        press(name)

    assert (doc.history.head, len(doc.stack)) == (before_head, before_layers)

    # ...and the same keys do land once the save is over, so the gate is a gate
    # and not a permanently dead branch.
    tab.saving = False
    press("z")
    assert doc.history.head == before_head - 1


def test_every_document_mutating_panel_is_gated_on_the_saving_flag():
    """The keyboard path, the canvas and the layers panel each gate on
    ``tab.saving``; the side panels did not, so Flip / Rotate / Undo / Scale /
    Crop stayed live while ``write_ora`` was walking the stack -- an archive
    whose ``stack.xml``, merged image and layer PNGs disagree about the canvas
    size.

    ``busy`` counts as the gate as well as ``saving``, and is now what these
    panels actually use: it is ``saving or playing``, so it is strictly the
    stronger claim. Accepting both is what stops this test forcing the weaker
    spelling back in the day a third reason is added.

    Asserted structurally: driving these needs a live imgui frame, but what
    went wrong is the *absence* of a call, which a frame cannot show either.
    """
    import ast
    import inspect

    from warlock.studio.panes import inker_bridge, inker_timeline, inker_tools

    targets = (
        # ``_canvas_ops`` was the bridge panel's row of flips and rotates; the
        # gate is ``inker_ops.ready`` now, and every op that reads it is
        # checked by ``tests/inker/test_inker_ops.py``.
        inker_bridge._resize_popup,
        inker_tools._options,
        inker_timeline._frame_menu,
        inker_timeline._cell_menu,
    )
    for func in targets:
        tree = ast.parse(inspect.getsource(func).lstrip())
        gates = [
            call
            for call in ast.walk(tree)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "begin_disabled"
            and any(
                isinstance(node, ast.Attribute) and node.attr in ("saving", "busy")
                for arg in call.args
                for node in ast.walk(arg)
            )
        ]
        assert gates, f"{func.__qualname__} mutates the document with no save gate"


def test_busy_is_saving_or_playing():
    """The gate every panel above reads. Both halves make restructuring the
    document unsafe -- a save is encoding the layer stack off-thread, and
    playback is showing a cached flatten of some other frame -- so they are one
    question, asked once."""
    tab = _tab("")
    assert not tab.busy
    tab.saving = True
    assert tab.busy
    tab.saving, tab.playing = False, True
    assert tab.busy


def test_a_failed_save_clears_the_saving_flag():
    """Otherwise the tab is read-only forever: every gate added above keys on
    this flag, so a stuck one disables the whole editor."""
    from types import SimpleNamespace

    from warlock.studio.inker_state import InkerState

    tab = _tab("")
    tab.saving = True
    state = InkerState()
    state.add(tab)
    ctx = SimpleNamespace(state=SimpleNamespace(inker=state))

    inker_mode.on_task_failed(ctx, SimpleNamespace(key=f"inker-save:{tab.uid}"))
    assert not tab.saving


# --- what an in-place save is allowed to write ------------------------------


class _SaveCtx:
    """A ctx that records toasts and runs submits inline, like ``FakeCtx``, but
    carrying a real ``InkerState`` so ``_settle`` and the save lock behave."""

    def __init__(self) -> None:
        from warlock.studio.inker_state import InkerState

        self.state = SimpleNamespace(inker=InkerState())
        self.submitted: list[str] = []
        self.toasts: list[tuple[str, str]] = []

    def submit(self, key: str, run: Any, *args: Any) -> bool:
        self.submitted.append(key)
        self.result = run(*args)
        return True

    def toast(self, text: str, level: str = "info", *_: Any) -> None:
        self.toasts.append((text, level))


def _file_tab(ctx: _SaveCtx, path) -> InkerDoc:
    doc = inker.Document.load(path)
    tab = InkerDoc(doc=doc, title=path.name, path=path, file_format=doc.file_format)
    ctx.state.inker.add(tab)
    return tab


def test_saving_a_jpg_routes_to_save_as_instead_of_writing_png_bytes(tmp_path, monkeypatch):
    """The defect: ``Document.load`` stamps every non-ORA input
    ``file_format="png"`` and ``_write`` dispatches on that format, never on the
    path -- so Ctrl+S over an opened ``.jpg`` wrote PNG bytes into a file still
    named ``.jpg``, silently, over the user's original.

    The fix refuses the in-place write and offers Save As. Asserted on the
    original bytes surviving as well as on the routing, because the routing is
    only interesting for what it prevents.
    """
    src = tmp_path / "photo.jpg"
    Image.new("RGB", (16, 16), (10, 200, 90)).save(src, "JPEG")
    before = src.read_bytes()

    ctx = _SaveCtx()
    tab = _file_tab(ctx, src)
    dest = tmp_path / "photo.ora"
    monkeypatch.setattr(inker_mode.dialogs, "save_file", lambda *a, **k: dest)

    inker_mode.save(ctx, tab)

    assert any(key.startswith("inker-saveas:") for key in ctx.submitted)
    assert not any(key.startswith("inker-save:") for key in ctx.submitted)
    assert src.read_bytes() == before, "the JPG was overwritten"
    assert dest.exists()
    assert ctx.toasts and "JPG" in ctx.toasts[0][0]


def test_every_openable_non_writable_suffix_is_gated(tmp_path, monkeypatch):
    """The gate keys on the suffix rather than on JPEG specifically, so every
    format ``filetypes`` advertises as openable but the app cannot write back
    is covered by construction. WebP is skipped where Pillow lacks the codec.
    """
    from warlock.studio import filetypes

    formats = {".jpg": "JPEG", ".jpeg": "JPEG", ".webp": "WEBP", ".bmp": "BMP"}
    assert set(filetypes.IMAGE_SUFFIXES) - set(inker_mode.WRITABLE_SUFFIXES) == set(formats)

    for suffix, encoder in formats.items():
        src = tmp_path / f"pic{suffix}"
        try:
            Image.new("RGB", (8, 8), (1, 2, 3)).save(src, encoder)
        except (OSError, KeyError, ValueError):  # pragma: no cover - codec absent
            continue
        ctx = _SaveCtx()
        tab = _file_tab(ctx, src)
        dest = tmp_path / f"out{suffix}.ora"
        monkeypatch.setattr(inker_mode.dialogs, "save_file", lambda *a, _d=dest, **k: _d)
        inker_mode.save(ctx, tab)
        assert any(k.startswith("inker-saveas:") for k in ctx.submitted), suffix


def test_a_png_still_saves_in_place(tmp_path):
    """The negative control. The gate must not touch the two suffixes an
    in-place save has always been allowed to write."""
    src = tmp_path / "art.png"
    src.write_bytes(_png(size=(8, 8), colour=(4, 5, 6, 255)))

    ctx = _SaveCtx()
    tab = _file_tab(ctx, src)
    inker_mode.save(ctx, tab)

    assert ctx.submitted == [f"inker-save:{tab.uid}"]
    assert not ctx.toasts
    assert Image.open(src).size == (8, 8)


def test_an_ora_still_saves_in_place(tmp_path):
    """The other half of the negative control."""
    src = tmp_path / "art.ora"
    inker.write_ora(inker.Document.blank(8, 8), src)

    ctx = _SaveCtx()
    tab = _file_tab(ctx, src)
    inker_mode.save(ctx, tab)

    assert ctx.submitted == [f"inker-save:{tab.uid}"]
    assert not ctx.toasts


# --- Save As can now write .aseprite ----------------------------------------


def _untitled_tab() -> InkerDoc:
    doc = inker.Document.blank(16, 16)
    doc.stack.active.pixels[..., :] = (10, 200, 90, 255)
    return InkerDoc(doc=doc, title="Untitled")


def test_save_as_writes_an_aseprite_file_asein_can_read_back(tmp_path, monkeypatch):
    """The round trip the retirement is for: Save As can now put an edited
    drawing into ``.aseprite``, and what it writes must be the real format --
    readable by the same parser an import uses, not a renamed ORA."""
    from warlock.studio.inker import asein

    ctx = _SaveCtx()
    tab = _untitled_tab()
    ctx.state.inker.add(tab)
    dest = tmp_path / "sprite.aseprite"
    monkeypatch.setattr(inker_mode.dialogs, "save_file", lambda *a, **k: dest)

    inker_mode.save_as(ctx, tab)

    assert ctx.result == {"path": dest, "rev": 0, "format": "aseprite", "retitle": True}
    reopened, warnings = asein.document_from_aseprite(dest.read_bytes())
    assert not warnings
    assert reopened.stack.size == (16, 16)


def test_an_ase_suffix_also_routes_to_aseprite(tmp_path, monkeypatch):
    """Aseprite's older releases wrote ``.ase``; both suffixes name the one
    format, so both must route the same way."""
    from warlock.studio.inker import asein

    ctx = _SaveCtx()
    tab = _untitled_tab()
    ctx.state.inker.add(tab)
    dest = tmp_path / "sprite.ase"
    monkeypatch.setattr(inker_mode.dialogs, "save_file", lambda *a, **k: dest)

    inker_mode.save_as(ctx, tab)

    assert ctx.result["format"] == "aseprite"
    asein.document_from_aseprite(dest.read_bytes())  # does not raise


def test_a_reachable_writer_refusal_names_itself_not_the_generic_toast(
    tmp_path, monkeypatch
):
    """The refusal a real user can reach: an indexed document that picked up a
    tileset strip holding a colour off its own palette (imported from a
    ``.tsx``, or from Plotter) refuses to write, by name, at
    ``aseout._resolved_indices``. Before this fix that ``ValueError`` reached
    ``tasks.py``'s generic branch as a bare exception and the toast the user
    actually saw was "Something went wrong; see the log for details." --
    the named sentence landing only in the log.

    ``_write``'s aseprite branch now wraps the writer in ``invalid_from``, the
    same idiom ``import_tileset`` already uses on the read side of this same
    format (``inker_mode.py``), so the refusal surfaces as a ``ServiceError``
    -- which is what ``tasks.py.poll`` and ``main.py``'s toast already know how
    to carry verbatim. ``_SaveCtx.submit`` runs the task inline, the same seam
    ``test_an_unsupported_tsx_is_refused_by_name`` (``tests/inker/
    test_inker_tsx.py``) asserts the import side through, so the raised
    ``Invalid`` is asserted directly here rather than through a poll loop.
    """
    import numpy as np

    from warlock.service.errors import Invalid
    from warlock.studio.inker.document import Document
    from warlock.studio.inker.tiles import strip

    def _tile(colour: tuple[int, int, int, int]) -> np.ndarray:
        return np.full((4, 4, 4), colour, dtype=np.uint8)

    def _tileset(*colours, name="tiles"):
        blank = np.zeros((4, 4, 4), dtype=np.uint8)
        built = strip(np.stack([blank, *[_tile(c) for c in colours]], axis=0))
        from dataclasses import replace

        return replace(built, name=name)

    doc = Document.blank(8, 8)
    doc.stack[0].name = "Background"
    doc.stack[0].pixels[:, :] = (200, 30, 30, 255)
    doc.invalidate_all()
    doc.convert_to_indexed(
        [(0, 0, 0, 0), (200, 30, 30, 255), (30, 200, 30, 255)], transparent=0
    )
    slot = doc.add_tileset(_tileset((200, 30, 30, 255), (30, 200, 30, 255)))
    doc.add_tilemap_layer(slot.uid, name="Tiles")
    # A tileset the document never bound is enough to reach the writer's
    # refusal, and is simpler to build than a bound one: any tileset the
    # writer walks and finds an off-palette pixel in triggers the same path.
    doc.add_tileset(_tileset((10, 10, 10, 255), name="stray"))

    ctx = _SaveCtx()
    tab = InkerDoc(doc=doc, title="Untitled")
    ctx.state.inker.add(tab)
    dest = tmp_path / "stray.aseprite"
    monkeypatch.setattr(inker_mode.dialogs, "save_file", lambda *a, **k: dest)

    with pytest.raises(Invalid, match="stray"):
        inker_mode.save_as(ctx, tab)

    assert not dest.exists()


def test_save_as_still_defaults_to_ora_for_an_unrelated_flow(tmp_path, monkeypatch):
    """The default filter row and the suggested name are unchanged: a save
    dialog cancelled with no suffix chosen, or one given a plain name, still
    lands on ``.ora`` byte-identically with the pre-Wave-5 writer."""
    ctx = _SaveCtx()
    tab = _untitled_tab()
    ctx.state.inker.add(tab)
    calls: list[Any] = []

    def fake_save_file(title, default_name, filters):
        calls.append((title, default_name, filters))
        return tmp_path / "sprite"  # no suffix at all

    monkeypatch.setattr(inker_mode.dialogs, "save_file", fake_save_file)

    inker_mode.save_as(ctx, tab)

    assert calls[0][1] == "untitled.ora"
    assert calls[0][2][:2] == inker_mode.ORA_FILTER
    dest = tmp_path / "sprite.ora"
    assert ctx.result == {"path": dest, "rev": 0, "format": "ora", "retitle": True}
    assert dest.read_bytes() == inker.ora_bytes(tab.doc)


def test_ctrl_s_on_an_aseprite_backed_tab_forces_save_as_with_its_own_toast(
    tmp_path, monkeypatch
):
    """An aseprite-format tab's *next* Ctrl+S still cannot write in place --
    ``WRITABLE_SUFFIXES`` never grew this suffix -- so it is routed back
    through Save As. Its toast is its own, not the generic "came from a
    {SUFFIX} file" sentence every other excluded format reuses: this file was
    *written* by Save As a moment ago, not imported, so "came from" would be
    backwards and ungrammatical to boot ("a ASEPRITE file")."""
    src = tmp_path / "sprite.aseprite"
    doc = inker.Document.blank(8, 8)
    inker.write_aseprite(doc, src)

    ctx = _SaveCtx()
    tab = InkerDoc(doc=doc, title=src.name, path=src, file_format="aseprite")
    ctx.state.inker.add(tab)
    dest = tmp_path / "sprite2.aseprite"
    monkeypatch.setattr(inker_mode.dialogs, "save_file", lambda *a, **k: dest)

    inker_mode.save(ctx, tab)

    assert any(key.startswith("inker-saveas:") for key in ctx.submitted)
    assert not any(key.startswith("inker-save:") for key in ctx.submitted)
    assert ctx.toasts
    message = ctx.toasts[0][0]
    assert "came from a" not in message.lower()
    assert "save as" in message.lower()
    assert ".aseprite" in message.lower()
    assert dest.exists()


def test_a_still_plain_png_tabs_save_flow_is_untouched(tmp_path):
    """Negative control for the new branch: an ordinary PNG tab must not be
    caught by anything added for Aseprite."""
    src = tmp_path / "art.png"
    src.write_bytes(_png(size=(8, 8), colour=(4, 5, 6, 255)))

    ctx = _SaveCtx()
    tab = _file_tab(ctx, src)
    inker_mode.save(ctx, tab)

    assert ctx.submitted == [f"inker-save:{tab.uid}"]
    assert not ctx.toasts


def test_a_dirty_aseprite_tabs_journal_payload_is_still_ora(tmp_path):
    """The journal is format-agnostic ORA by doctrine (see
    ``inker_mode._journal_encode``): a tab whose ``file_format`` is
    ``"aseprite"`` must still hand the crash-recovery loop bytes that
    ``ora.read_ora`` can open, not an Aseprite encode."""
    from warlock.studio.inker import ora

    src = tmp_path / "sprite.aseprite"
    doc = inker.Document.blank(8, 8)
    tab = InkerDoc(doc=doc, title=src.name, path=src, file_format="aseprite")

    payload = inker_mode._journal_encode(tab)

    copy = tmp_path / "journal.ora"
    copy.write_bytes(payload)
    reopened = ora.read_ora(copy)
    assert reopened.stack.size == (8, 8)
