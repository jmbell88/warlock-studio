"""Fixtures shared by the Plotter tests that need a mode rather than a map."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from warlock.studio import plotter_mode, plotter_state
from warlock.studio.plotter.tilemap import MapDoc


@pytest.fixture
def plotter_ctx():
    """A context with Plotter open on one map. -> ``(ctx, state)``.

    Enough of the app for ``handle_key``, which is the *real* input path for
    every view toggle in this mode -- the sidebar's switches and these chords
    write the same fields, and a chord is the half a test can press.
    """
    ctx = SimpleNamespace(
        state=SimpleNamespace(plotter=None, mode="plotter", preview={}),
        settings=SimpleNamespace(get=lambda _k: None, set=lambda _k, _v: None),
        toast=lambda *_a, **_k: None,
        viewer=None,
    )
    state = plotter_state.ensure(ctx)
    doc = MapDoc(8, 8, 16, 16)
    doc.add_tile_layer("Tiles")
    tab = plotter_state.PlotterDoc(doc=doc, title="Map", saved_head=doc.history.head)
    state.add(tab)
    assert plotter_mode.ensure(ctx) is state
    return ctx, state
