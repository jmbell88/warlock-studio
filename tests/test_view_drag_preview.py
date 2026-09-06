"""The GPU preview buffer a live element drag writes, across a kind switch.

The 2026-09-06 audit (create2-09) found ``_restart_keyboard_drag`` restoring
the selection overlay to the pre-drag vertex positions on ``G``/``R``/``S``
mid-drag, but never resetting the GPU vertex buffer ``_preview_element_drag``
wrote for the abandoned transform: ``_restart_keyboard_drag`` only rebinds
``obj.translation``/``rotation``/``scale``, which an element drag never
touches. The drawn mesh kept showing the abandoned transform's preview until
the next mouse-move or typed digit called ``_preview_element_drag`` again --
for one or more frames the mesh and its own overlay disagreed about where the
geometry was.

This module builds the smallest fake cache/overlay that lets the real
``DragOps._preview_positions`` run -- ``clay.document.preview_primitives`` is
pure numpy, no GL context needed -- so the assertion is against the actual
code path, not a re-implementation of it.
"""

from __future__ import annotations

import functools
from types import SimpleNamespace
from typing import Any

import numpy as np

from warlock.studio._view_drag import DragOps, _ElementDrag
from warlock.studio.clay import document as bd
from warlock.studio.clay import elements as el
from warlock.studio.clay import primitives as bp


class _FakeGPU:
    def __init__(self) -> None:
        self.calls: list[Any] = []

    def update_vertices(self, primitive: Any) -> None:
        self.calls.append(primitive)


class _FakeOverlay:
    def __init__(self) -> None:
        self.written: list[np.ndarray] = []

    def write_positions(self, positions: np.ndarray) -> None:
        self.written.append(np.array(positions, copy=True))


def _view(doc: bd.ClayDoc, uid: int) -> SimpleNamespace:
    obj = doc.by_uid(uid)
    original = np.array(obj.mesh.positions, dtype="f8")
    drag = _ElementDrag(
        before=obj.mesh,
        verts=np.array([0, 1]),
        local=original[[0, 1]],
        matrix=np.eye(4),
        inverse=np.eye(4),
        # What a live "move" drag would have written to the GPU already --
        # the abandoned transform's preview, distinct from the untouched mesh.
        preview=original + 5.0,
    )
    gpu = _FakeGPU()
    overlay = _FakeOverlay()
    entry = SimpleNamespace(gpu=SimpleNamespace(draws=[(None, gpu)]))
    view = SimpleNamespace(
        _key_kind="move",
        _drag_start={},
        _element_drags={uid: drag},
        _cache={uid: entry},
        _overlays={uid: overlay},
        _last_mouse=(0.0, 0.0),
        _drag_origin=(0.0, 0.0, 0.0),
        _key_anchor=None,
        _restore_overlays=lambda doc, uids: None,
        _view_plane_point=lambda local, centre: None,
        _clear_drag_input=lambda: None,
    )
    view._preview_positions = functools.partial(DragOps._preview_positions, view)
    return view, gpu, overlay, drag, original


def _doc() -> tuple[bd.ClayDoc, int]:
    doc = bd.ClayDoc()
    uid = doc.add_object(bd.Obj(uid=bd.new_uid(), name="Box", mesh=bp.box())).uid
    doc.element_mode = "vertex"
    doc.set_element_sel(uid, el.ElementSel(verts=[0, 1]))
    return doc, uid


def test_switching_keyboard_drag_kind_resets_the_element_gpu_buffer():
    doc, uid = _doc()
    view, gpu, overlay, drag, original = _view(doc, uid)

    DragOps._restart_keyboard_drag(view, doc, "rotate")

    assert drag.preview is None, "the abandoned transform's preview is gone"
    assert gpu.calls, "the GPU buffer was rewritten this frame"
    assert overlay.written, "the overlay was rewritten this frame"
    # The buffer must match the mesh's own (pre-drag) positions, not the
    # abandoned move's preview the overlay was restored to.
    written_positions = overlay.written[-1]
    assert np.array_equal(written_positions, np.array(original, dtype="f4"))
