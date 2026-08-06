"""The asset viewer's imgui registration rule.

``Viewport.resize`` releases its texture and makes a new one, and the imgui
backend maps GL names to moderngl objects: releasing without forgetting leaves
the backend holding a dead object under a name the driver is free to reissue,
which is how an unrelated image starts rendering as this one. ``ClayView``
already implements and tests both halves; these pin the same contract on the
asset ``Viewer``.
"""

from __future__ import annotations

from typing import Any

import pytest

from warlock.studio.viewer_embed import Viewer


@pytest.fixture
def viewer(gl):
    v = Viewer(gl)
    yield v
    v.release()


def test_the_outgoing_texture_is_forgotten_before_a_resize(viewer, monkeypatch) -> None:
    forgotten: list[Any] = []
    monkeypatch.setattr(viewer, "_forget", forgotten.append)

    viewer.render((0.0, 0.0, 64.0, 64.0), 0.0)
    first = viewer.viewport.texture
    viewer.render((0.0, 0.0, 96.0, 64.0), 0.0)

    assert first in forgotten
    assert viewer.viewport.texture is not first


def test_a_render_at_the_same_size_forgets_nothing(viewer, monkeypatch) -> None:
    viewer.render((0.0, 0.0, 64.0, 64.0), 0.0)

    forgotten: list[Any] = []
    monkeypatch.setattr(viewer, "_forget", forgotten.append)
    viewer.render((0.0, 0.0, 64.0, 64.0), 0.0)
    viewer.render((0.0, 0.0, 64.0, 64.0), 0.0)

    assert forgotten == []


def test_release_forgets_the_texture_before_freeing_it(gl, monkeypatch) -> None:
    v = Viewer(gl)
    order: list[str] = []
    monkeypatch.setattr(v, "_forget", lambda tex: order.append("forget"))
    real_release = v.viewport.release

    def release() -> None:
        order.append("release")
        real_release()

    monkeypatch.setattr(v.viewport, "release", release)
    v.release()
    assert order == ["forget", "release"]
