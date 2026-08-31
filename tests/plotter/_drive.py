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


class TileScene:
    """The tileset editor's tile view, over one tile, with a fake mouse.

    The second scene in this file rather than a second harness: it reuses
    :class:`Mouse` verbatim and drives ``plotter_tileset_editor``'s real
    per-tile dispatch one call per frame, exactly as :class:`Scene` drives
    ``plotter_canvas._object_input``.

    Two tabs share it, because they are the same gesture over the same square:
    ``tileset_tab="Collision"`` drives ``_collision_input`` and
    ``tileset_tab="Terrain"`` drives ``_terrain_input``. They take the same
    arguments, so :meth:`frame` differs by one lookup rather than by a second
    harness -- which is the whole reason the click-region machinery was
    factored out in the first place.

    The view is the production one -- ``COLLISION_VIEW`` design px over the
    tile, so 16x for a 16 px tile -- because the grab radius is in *screen*
    pixels and at 1:1 a single radius would cover half the tile. Tests still
    aim in tile pixels and :meth:`screen` converts, which is the same
    "positions are read, never computed" rule the object tests follow.
    """

    def __init__(
        self,
        monkeypatch: Any,
        *,
        tile: int = 16,
        tiles: int = 4,
        side: float | None = None,
        tileset_tab: str = "Collision",
    ) -> None:
        import numpy as np

        from warlock.studio.panes.plotter_tileset_editor import COLLISION_VIEW
        from warlock.studio.tilegrid.picking import TileView
        from warlock.studio.tilegrid.tileset import Tileset

        pixels = np.zeros((tile, tile * tiles, 4), dtype=np.uint8)
        pixels[..., 3] = 255
        self.doc = MapDoc(4, 4, tile, tile)
        self.doc.add_tile_layer("Tiles")
        self.ref = self.doc.add_tileset(
            Tileset(name="Set", tile_w=tile, tile_h=tile, pixels=pixels)
        )
        self.local = 0
        self.state = plotter_state.PlotterState()
        self.state.editing_tileset = 0
        self.state.tileset_tab = str(tileset_tab)
        self.tab = SimpleNamespace(
            doc=self.doc,
            uid="tab-1",
            busy=False,
            view=inker_state.PaintView(zoom=1.0, pan=(0.0, 0.0), fitted=True),
        )
        self.toasts: list[tuple[str, str]] = []
        self.ctx = SimpleNamespace(
            toast=lambda text, kind="info": self.toasts.append((str(text), str(kind)))
        )
        self.view = TileView(
            origin=(0.0, 0.0),
            side=float(COLLISION_VIEW if side is None else side),
            tile_w=tile,
            tile_h=tile,
        )
        self.mouse = Mouse()
        self.mouse.install(monkeypatch)

    # -- reading back ---------------------------------------------------------

    @property
    def meta(self) -> Any:
        return self.doc.tilesets[0].tileset.meta_of(self.local)

    @property
    def wangsets(self) -> tuple:
        """The tileset's Wang sets, re-read every time.

        Never cached and never off ``self.ref``: the Terrain tab writes through
        ``replace_tileset``, which swaps the whole ``TilesetRef``, so a held
        reference is the atlas as it was before the last click.
        """
        return tuple(self.doc.tilesets[0].tileset.wangsets)

    @property
    def wangset(self) -> Any:
        return self.wangsets[int(self.state.tileset_wangset)]

    @property
    def shapes(self) -> tuple:
        return tuple(self.meta.collision)

    def selected(self) -> Any:
        at = self.state.tileset_shape
        return None if at is None else self.shapes[int(at)]

    def screen(self, x: float, y: float) -> tuple[float, float]:
        """A tile-pixel point, where the pointer has to be to hit it."""
        return self.view.to_screen(x, y)

    def add(self, kind: Any) -> Any:
        """Press the real *Add* button's handler. -> the shape it selected."""
        from warlock.studio.panes import plotter_tileset_editor

        plotter_tileset_editor._add_shape(
            self.state, self.tab, 0, self.local, self.meta, kind
        )
        return self.selected()

    def handle(self, name: str) -> tuple[float, float]:
        """Where a named handle of the selected shape is, in tile pixels.

        Read off the production function rather than recomputed: a test that
        aimed at its own idea of the position would pass while the drawn grip
        sat somewhere unclickable.
        """
        from warlock.studio.tilegrid import picking

        return picking.box_handles(self.selected())[name]

    # -- pressing -------------------------------------------------------------

    def frame(
        self,
        at: tuple[float, float],
        *,
        click: bool = False,
        down: bool = False,
        release: bool = False,
        ctrl: bool = False,
        shift: bool = False,
        alt: bool = False,
        hovered: bool = True,
    ) -> None:
        from warlock.studio.panes import plotter_tileset_editor

        self.mouse.at = self.screen(*at)
        self.mouse.clicked = {0: click, 1: False, 2: False}
        self.mouse.down = {0: down or click, 1: False, 2: False}
        self.mouse.released = {0: release, 1: False, 2: False}
        self.mouse.ctrl = ctrl
        self.mouse.shift = shift
        self.mouse.alt = alt
        dispatch = (
            plotter_tileset_editor._terrain_input
            if self.state.tileset_tab == "Terrain"
            else plotter_tileset_editor._collision_input
        )
        dispatch(self.ctx, self.state, self.tab, 0, self.local, self.view, hovered)

    def drag(
        self, start: tuple[float, float], end: tuple[float, float], **mods: Any
    ) -> None:
        """Press, one held frame, release -- the three-frame gesture."""
        self.frame(start, click=True, **mods)
        self.frame(end, down=True, **mods)
        self.frame(end, release=True, **mods)


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
