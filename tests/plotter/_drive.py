"""A fake mouse over the *real* ``plotter_canvas._object_input``.

The gesture is a state machine spread over three frames -- press, one or more
held frames, release -- and every interesting rule in it (which handle wins,
what a modifier means, when an undo step is pushed) lives in the dispatch rather
than in a helper a unit test could call. So these tests drive the real function
with a synthetic pointer, one call per frame, which is ``tests/inker``'s
``_Mouse`` idiom for exactly the same reason.

``_object_input`` reads imgui through a *function-local* ``from imgui_bundle
import imgui``, so there is no module global to swap: the attributes are
monkeypatched onto the real module and put back by the fixture. Nothing here
opens a context -- the pane reads five mouse functions and nothing else.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from warlock.studio import inker_state, plotter_state
from warlock.studio.plotter.tilemap import MapDoc, MapObject, new_uid


class Mouse:
    """imgui's mouse, as much of it as the object gestures read."""

    def __init__(self) -> None:
        self.at = (0.0, 0.0)
        self.down = {0: False, 1: False, 2: False}
        self.clicked = {0: False, 1: False, 2: False}
        self.released = {0: False, 1: False, 2: False}
        self.ctrl = False
        self.alt = False
        self.shift = False

    def install(self, monkeypatch: Any) -> None:
        from imgui_bundle import imgui

        monkeypatch.setattr(
            imgui, "get_mouse_pos", lambda: SimpleNamespace(x=self.at[0], y=self.at[1])
        )
        monkeypatch.setattr(imgui, "is_mouse_clicked", lambda button: self.clicked[button])
        monkeypatch.setattr(imgui, "is_mouse_down", lambda button: self.down[button])
        monkeypatch.setattr(imgui, "is_mouse_released", lambda button: self.released[button])
        monkeypatch.setattr(
            imgui,
            "get_io",
            lambda: SimpleNamespace(
                key_ctrl=self.ctrl, key_alt=self.alt, key_shift=self.shift
            ),
        )


class Scene:
    """A map with one object layer, its document, mode state and a fake mouse.

    The view is identity -- zoom 1, no pan, origin at (0, 0) -- so a screen
    coordinate *is* a map pixel and every position in a test can be read
    straight off the object it is aiming at.
    """

    def __init__(
        self,
        monkeypatch: Any,
        *,
        tile: int = 16,
        size: int = 16,
        tool: str = "object",
    ) -> None:
        self.doc = MapDoc(size, size, tile, tile)
        self.doc.add_tile_layer("Tiles")
        self.layer = self.doc.add_object_layer("Objects")
        self.doc.set_active_layer(self.layer.uid)
        self.state = plotter_state.PlotterState()
        # The Select tool, because that is what an object layer opens on -- and
        # what empty space means depends on it: Select sweeps a marquee, every
        # insert tool draws its shape. Pass ``tool="object_rect"`` for the
        # other half.
        self.state.tool = tool
        self.tab = SimpleNamespace(
            doc=self.doc,
            uid="tab-1",
            busy=False,
            view=inker_state.PaintView(zoom=1.0, pan=(0.0, 0.0), fitted=True),
        )
        self.ctx = SimpleNamespace(toast=lambda *_a, **_k: None)
        self.mouse = Mouse()
        self.mouse.install(monkeypatch)

    def add(self, **kwargs: Any) -> MapObject:
        obj = self.doc.add_object(self.layer.uid, MapObject(uid=new_uid(), **kwargs))
        self.state.select_object(obj.uid)
        return obj

    def object(self, uid: int) -> MapObject:
        return next(o for o in self.doc.layer(self.layer.uid).objects if o.uid == uid)

    def frame(
        self,
        at: tuple[float, float],
        *,
        click: bool = False,
        down: bool = False,
        release: bool = False,
        ctrl: bool = False,
        shift: bool = False,
        hovered: bool = True,
    ) -> None:
        from warlock.studio.panes import plotter_canvas

        self.mouse.at = (float(at[0]), float(at[1]))
        self.mouse.clicked = {0: click, 1: False, 2: False}
        self.mouse.down = {0: down or click, 1: False, 2: False}
        self.mouse.released = {0: release, 1: False, 2: False}
        self.mouse.ctrl = ctrl
        self.mouse.shift = shift
        plotter_canvas._object_input(self.ctx, self.state, self.tab, (0.0, 0.0), hovered)
