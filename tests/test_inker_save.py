"""Saving a linked paint document, and what the save must not race with.

A linked save writes two files -- ``input.png`` and the ``paint.ora`` behind it
-- and their *order* is the only thing deciding whether reopening the reference
brings the layers back or flattens them away. That is worth pinning, because
the ordering that reads as safest is the one that breaks it.
"""

from __future__ import annotations

import io
import zipfile
from typing import Any

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
    thread would have done without needing one."""

    def __init__(self, svc: Any) -> None:
        self.svc = svc
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
    a typical filesystem -- and ``paint_working_status`` compares with ``>=``,
    so an mtime assertion passes under the bug roughly two runs in three.
    """
    job_id = _reference(svc)
    order: list[str] = []

    real_flat = svc_files.save_edited_image
    real_layers = svc_files.save_paint_working

    def flat(*args, **kwargs):
        order.append("input.png")
        return real_flat(*args, **kwargs)

    def layers(*args, **kwargs):
        order.append("paint.ora")
        return real_layers(*args, **kwargs)

    monkeypatch.setattr(svc_files, "save_edited_image", flat)
    monkeypatch.setattr(svc_files, "save_paint_working", layers)

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

    status = svc_files.paint_working_status(svc, job_id)
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

    status = svc_files.paint_working_status(svc, job_id)
    reopened = inker.Document.load(svc_files.paint_working_path(svc, job_id))
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
    assert not svc_files.paint_working_status(svc, job_id)["fresh"]


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
    ctx.state = SimpleNamespace(paint=state, mode="paint")
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


# --- the gate ---------------------------------------------------------------


def test_the_mutating_shortcuts_do_nothing_while_a_save_is_running():
    """``_write`` encodes the live document on a task thread. That tolerates a
    stroke -- pixels are written in place -- but not the stack changing shape,
    so every shortcut that restructures it has to be held off.

    Driven through ``_ctrl_key`` rather than asserted against the constant: a
    set nothing consults gates nothing.
    """
    from types import SimpleNamespace

    from warlock.studio.inker_state import InkerState

    tab = _tab("")
    doc = tab.doc
    doc.select_all()  # something for undo to have to reverse
    before_head, before_layers = doc.history.head, len(doc.stack)

    state = InkerState()
    state.add(tab)
    ctx = SimpleNamespace(state=SimpleNamespace(paint=state), svc=None)
    event = SimpleNamespace(key=0, unicode="")

    tab.saving = True
    for name in sorted(inker_mode._MUTATING_CTRL):
        inker_mode._ctrl_key(ctx, state, tab, doc, name, event, shift=False)

    assert (doc.history.head, len(doc.stack)) == (before_head, before_layers)

    # ...and the same keys do land once the save is over, so the gate is a gate
    # and not a permanently dead branch.
    tab.saving = False
    inker_mode._ctrl_key(ctx, state, tab, doc, "z", event, shift=False)
    assert doc.history.head == before_head - 1


def test_every_document_mutating_panel_is_gated_on_the_saving_flag():
    """The keyboard path, the canvas and the layers panel each gate on
    ``tab.saving``; the side panels did not, so Flip / Rotate / Undo / Scale /
    Crop stayed live while ``write_ora`` was walking the stack -- an archive
    whose ``stack.xml``, merged image and layer PNGs disagree about the canvas
    size.

    Asserted structurally: driving these needs a live imgui frame, but what
    went wrong is the *absence* of a call, which a frame cannot show either.
    """
    import ast
    import inspect

    from warlock.studio.panes import inker_bridge, inker_tools

    for func in (inker_bridge._canvas_ops, inker_bridge._resize_popup, inker_tools._options):
        tree = ast.parse(inspect.getsource(func).lstrip())
        gates = [
            call
            for call in ast.walk(tree)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "begin_disabled"
            and any(
                isinstance(node, ast.Attribute) and node.attr == "saving"
                for arg in call.args
                for node in ast.walk(arg)
            )
        ]
        assert gates, f"{func.__qualname__} mutates the document with no save gate"


def test_a_failed_save_clears_the_saving_flag():
    """Otherwise the tab is read-only forever: every gate added above keys on
    this flag, so a stuck one disables the whole editor."""
    from types import SimpleNamespace

    from warlock.studio.inker_state import InkerState

    tab = _tab("")
    tab.saving = True
    state = InkerState()
    state.add(tab)
    ctx = SimpleNamespace(state=SimpleNamespace(paint=state))

    inker_mode.on_task_failed(ctx, SimpleNamespace(key=f"inker-save:{tab.uid}"))
    assert not tab.saving
