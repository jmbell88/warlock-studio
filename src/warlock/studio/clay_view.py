"""The Clay viewport: GL, camera, gizmos and picking over the existing stack.

Modelled on :mod:`~warlock.studio.viewer_embed` and reusing its parts wholesale
-- ``camera``, ``render``, ``glctx``, ``grid``, ``gizmo``, ``programs`` and
``capture`` are all shared unchanged. What is different is the *subject*: the
3D pane shows one loaded GLB, and this shows a live document of many objects,
each of which can change independently while the others do not.

That difference is the whole design of the GPU cache. An entry is keyed on
``(uid, id(obj.mesh), materials)``, and it is sound precisely because ``Mesh``
is a frozen dataclass and every op on it is ``Mesh -> Mesh``: a changed mesh is
a *different object*, so identity misses exactly when it should, and an
unchanged one is the same object however many times the document around it was
edited. An in-place mutation anywhere in ``build.mesh`` would break this and
would show up as the viewport drawing the old shape forever with nothing in the
data to say why -- which is stated in that module's own docstring as the reason
it is immutable.

Two rules travel across from the existing viewer unchanged, and both are the
sort that fail invisibly:

* **imgui draws through moderngl.** ``studio/imgui_backend.py`` reimplements
  imgui's GL3 backend on moderngl because moderngl caches GL state, so a raw
  ``glBindTexture`` behind its back leaves the viewport rendering with whatever
  the panels last bound. Nothing here touches raw GL, and the resolved texture
  reaches imgui through ``widgets.texture_ref`` -- which *registers* it, since
  an id the renderer does not know maps to no moderngl object.
* **The viewport background is deliberately not tone-mapped.** three sets the
  clear colour straight back to sRGB, so it is the literal hex, and the
  renderer owns that; nothing here re-grades it.

**The mouse map is Wings3D's, and RMB pan is gone.** Right-drag used to pan,
and the context menu needs the right button more: a menu that appears under the
cursor with the ops that apply to what is selected is the whole point of an
element-mode editor, and a modifier-plus-right-drag would be a worse pan than
the middle button already is. So button 2 pans, button 3 opens the menu on a
*release within four pixels of the press* -- a right-drag does nothing at all,
which means a user who grabs the wrong button mid-orbit loses nothing.

**Modifiers are read at the press, from ``pygame.key.get_mods()``.** Not from
the event: a ``MOUSEBUTTONDOWN`` carries no modifier state, and tracking KEYDOWN
and KEYUP to shadow it is a second copy of something the platform already knows
and gets wrong the first time the window loses focus with Shift held.

Registration has a matching half that ``Viewport`` makes easy to miss:
``resize`` releases and recreates its texture, so a resize frees a GL name the
imgui backend may still be holding. :meth:`ClayView.draw` forgets the outgoing
texture before that happens.

**``ClayView``'s concerns live in the ``_view_*.py`` mixins it inherits** -- the
GPU cache, the element overlay, the bounds and centres, picking, and the whole
of input and dragging -- while what stays here is the class itself and the
frame: the fields, the draw, the world-matrix memo, the composite the renderer
consumes, the imgui-texture bookkeeping, and the gizmo the app state chooses.
The split is code motion only; ``ClayView`` is one class with one surface, and
every rule stated above (including :meth:`ClayView._narrow` being the single
narrowing site, in ``_view_drag``) still holds of it.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ._view_bounds import BoundsOps
from ._view_cache import CacheOps

# Re-exported rather than left behind: these are the viewport's names for
# them, and ``_Entry`` in particular is what ``__init__`` annotates its
# cache with. ``scenelib`` comes back through here for the same reason --
# ``clay_view.scenelib`` is the module object the GPU upload is patched on.
from ._view_cache import _Entry as _Entry
from ._view_cache import _materials_key as _materials_key
from ._view_cache import _object_key as _object_key
from ._view_cache import scenelib as scenelib
from ._view_drag import DragOps

# ``__init__`` annotates the live-drag map with it.
from ._view_drag import _ElementDrag as _ElementDrag
from ._view_overlay import OverlayOps

# ``__init__`` annotates the overlay cache with it.
from ._view_overlay import _SelOverlay as _SelOverlay

# ``Hit`` is constructed in the mixin, so it is defined there -- but the name
# a caller reaches for is the viewport's, so it is re-exported here.
from ._view_pick import Hit as Hit
from ._view_pick import PickOps
from .viewer import capture, glctx
from .viewer import math3d as m3
from .viewer.camera import Camera, screen_ray
from .viewer.gizmo import RotateGizmo, ScaleGizmo, TranslateGizmo
from .viewer.render import Renderer

log = logging.getLogger(__name__)


# Which gizmo each tool drives. Held as data so the dispatch is one lookup
# rather than a chain that a fifth tool would have to be threaded through.
GIZMO_FOR_TOOL = {"move": "translate", "rotate": "rotate", "scale": "scale"}


class _Composite:
    """The renderer's view of many cached objects at once.

    Not a ``GpuModel``: it owns nothing and releases nothing, and the two
    methods here are the entire surface ``Renderer._draw_model`` uses. Skinning
    is not part of it -- Clay has no skins, which is also why
    ``glbwrite`` refuses one.
    """

    __slots__ = ("draws",)

    def __init__(self, draws: list[Any]) -> None:
        self.draws = draws

    def palette(self, node: Any) -> None:
        return None


#: How opaque the surface is in X-ray. A third: enough to read the silhouette
#: and the shading, and little enough that an edge on the far side is pickable
#: through it -- which is the whole point of the mode.
XRAY_ALPHA = 0.33


class ClayView(CacheOps, BoundsOps, PickOps, OverlayOps, DragOps):
    """The Clay viewport, from the UI's point of view."""

    def __init__(self, ctx: Any, app_ctx: Any = None) -> None:
        """``ctx`` is the moderngl context, as ``Viewer``'s is.

        ``app_ctx`` is separate and optional because the two are genuinely
        different things and conflating them is the bug this signature exists
        to prevent: everything that draws needs the GL context, and the only
        thing that needs the app is reading which transform tool is selected --
        which is an *app* setting shared across documents, so the view reads it
        rather than holding a copy that could drift.
        """
        self.ctx = ctx
        self.app_ctx = app_ctx
        self.renderer = Renderer(ctx)
        self.viewport = glctx.Viewport(ctx, (16, 16))
        self.camera = Camera()
        self.wireframe = False
        # What the surface is drawn as, and what is drawn over it. Set by the
        # pane each frame beside ``wireframe`` and ``show_grid``, which is how
        # every other view setting reaches here.
        #
        # ``flat`` is Blender's *Solid*: the albedo with no lighting, which is
        # the shading a modeller works in because it shows silhouette and
        # topology without a specular highlight sitting on the vertex being
        # dragged. ``xray`` is the see-through pass -- see ``Renderer.draw``.
        self.flat = False
        self.wire_overlay = False
        self.xray = False
        # The grid toggle in the tools pane had no reader at all: the pane wrote
        # ``state.grid`` and the renderer was never told. One field, set by the
        # pane layer beside ``wireframe``, which already worked that way.
        self.show_grid = True
        self.radius = 1.0

        self.translate_gizmo = TranslateGizmo(ctx, self.renderer.programs)
        self.rotate_gizmo = RotateGizmo(ctx, self.renderer.programs)
        self.scale_gizmo = ScaleGizmo(ctx, self.renderer.programs)

        self._cache: dict[int, _Entry] = {}
        # Counted rather than inferred: "only what changed was rebuilt" is a
        # property worth asserting, and there is no other way to see it.
        self.rebuilds = 0

        self._rect = (0.0, 0.0, 1.0, 1.0)
        self._grab: str | None = None  # orbit | pan | gizmo | marquee
        self._last_mouse = (0.0, 0.0)
        self._drag_uids: list[int] = []
        self._drag_start: dict[int, tuple[Any, Any, Any]] = {}
        # The drag's *total* rotation, and the gizmo's origin at the press.
        # ``RotateGizmo.update`` hands back the increment since the last call
        # and ``TranslateGizmo.update`` the gizmo's new world position, so
        # neither is usable against a transform recorded at the press without
        # these two: the first has to be accumulated into a total, the second
        # turned into a displacement.
        self._drag_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self._drag_origin = np.zeros(3)
        # A live keyboard drag: which transform it is, and where on the view
        # plane it started. Empty and None between drags -- see
        # ``_view_drag.begin_keyboard_drag``.
        self._key_kind = ""
        self._key_anchor: Any = None

        # What the cursor is over in an element mode, as ``(uid, index)`` read
        # through the document's own mode. Updated only on motion with no grab
        # -- a hover that recomputed every frame would reproject every vertex
        # of every object while the scene sits perfectly still.
        self.hover_element: tuple[int, int] | None = None
        # ``(key, screen, mesh)`` -- the mesh pinned so the id in the key stays
        # sound; see ``screen_of``.
        self._screens: dict[int, tuple[Any, Any, Any]] = {}

        # Where the right button went down, and where the menu should open.
        # ``menu_request`` is read and cleared by the pane layer, which is the
        # only layer allowed to know imgui exists.
        self._rmb_at: tuple[float, float] | None = None
        self.menu_request: tuple[float, float] | None = None
        # The live marquee rectangle in viewport pixels, drawn by the pane.
        self.marquee: tuple[float, float, float, float] | None = None
        self._marquee_from: tuple[float, float] | None = None
        self._marquee_add = "replace"
        self._element_drags: dict[int, _ElementDrag] = {}
        self._overlays: dict[int, _SelOverlay] = {}
        self._element_centre = np.zeros(3)
        # Redraw bookkeeping (B13), the shape Viewer.render uses (B12).
        self._render_dirty = True
        self._last_render_key: Any = None
        # Per-object world matrices, keyed on the identity of the three
        # transform arrays -- sound because every transform write *rebinds*
        # them (the documented gizmo rule) rather than mutating in place (B26).
        # The arrays themselves ride in the entry (the ``_view_cache._Entry``
        # pin), because an id is only sound while its object is alive: a freed
        # array's address coming back on a different transform would otherwise
        # match a stale matrix.
        self._world_cache: dict[
            int, tuple[tuple[int, int, int], Any, tuple[Any, Any, Any]]
        ] = {}
        # element_centre / selection_centre / world_bounds memos (B25/B27),
        # each ``(key, answer, pins)`` -- the pins hold what the key's ids name.
        self._centre_memo: tuple[Any, Any, Any] | None = None
        self._bounds_memo: dict[bool, tuple[Any, tuple[Any, Any], Any]] = {}
        # One shared empty element-selection, so an unselected object stops
        # synthesising a fresh empty() per frame (B26).
        self._empty_sel: Any = None

        # What the keyboard has said about the drag under way, and what the HUD
        # should draw for it. Both are per drag: created at the press and
        # dropped at the release, because a lock that outlived one would
        # silently constrain the next.
        from .clay import drag as bdrag

        self.drag_input = bdrag.DragInput()
        self.drag_hud: str = ""
        # The world position a move has snapped onto, or None. Held rather than
        # recomputed by the consumer because it is found from the *cursor*, and
        # the transform is applied a layer down where the cursor is gone.
        self._snap_point: np.ndarray | None = None

    # -- drawing -----------------------------------------------------------

    def draw(self, doc: Any, rect: tuple[float, float, float, float], dt: float) -> Any:
        """Draw one frame into the viewport. -> the resolved texture.

        Skipped -- the last resolved texture returned as-is -- when nothing
        that feeds the draw moved (B13): the document's own ``rev`` covers
        every edit, selection and visibility change; ``handle_event`` marks a
        redraw for hover, marquee and drags; the camera answers for itself;
        and the tool decides which gizmo is on screen, so it is in the key.
        """
        self._rect = rect
        width, height = int(max(rect[2], 1)), int(max(rect[3], 1))
        key = (
            width, height, bool(self.wireframe), bool(self.show_grid),
            # Every view setting that changes the picture has to be in the key,
            # or the frame that turns one on is skipped as "nothing moved" and
            # the switch reads as broken until something else forces a redraw.
            bool(self.flat), bool(self.wire_overlay), bool(self.xray),
            id(doc), doc.rev, getattr(self.state, "tool", "select"),
        )
        if (
            not self._render_dirty
            and key == self._last_render_key
            and self.camera.settled()
            and self.viewport.texture is not None
        ):
            return self.viewport.texture
        self._last_render_key = key
        self._render_dirty = False
        self._resize(width, height)
        self.camera.update(dt)
        self.sync(doc)

        self.renderer.draw(
            self.viewport,
            self.camera,
            self._composite(doc),
            wireframe=self.wireframe,
            flat=self.flat,
            show_grid=self.show_grid,
            wire_overlay=self.wire_overlay,
            alpha=XRAY_ALPHA if self.xray else 1.0,
            overlays=self._element_overlays(doc) + self._gizmo_draws(doc, height),
        )
        return self.viewport.texture

    def _world(self, obj: Any) -> Any:
        """This object's world matrix, memoized on the transform arrays (B26).

        Sound because every transform write *rebinds* the three arrays -- the
        documented gizmo rule ("rebind rather than write through trs()'s live
        arrays") -- so the identity triple changes exactly when the transform
        does. One memo serves the composite, the overlays, the centres and the
        bounds, which is what "compute world matrices once" means here.
        """
        key = (id(obj.translation), id(obj.rotation), id(obj.scale))
        hit = self._world_cache.get(obj.uid)
        if hit is not None and hit[0] == key:
            return hit[1]
        world = m3.compose(obj.translation, obj.rotation, obj.scale)
        self._world_cache[obj.uid] = (
            key, world, (obj.translation, obj.rotation, obj.scale)
        )
        if len(self._world_cache) > 4096:
            self._world_cache.clear()
        return world

    def _composite(self, doc: Any) -> Any:
        """Every cached object as one thing the renderer can draw in one pass.

        ``Renderer.draw`` clears the target it is given, so a call per object
        would erase the one before it -- and a per-object call would also mean
        a grid pass and an overlay pass apiece. What it actually consumes from
        a ``GpuModel`` is ``draws`` and ``palette``, so the composite supplies
        exactly those over the cached entries, with each node's ``world`` set
        to that object's transform.

        The transform being carried on the node rather than in the cache key is
        what keeps a move from rebuilding a buffer: it is a uniform written per
        frame, which is what ``world`` already is for a glTF node.
        """
        draws = []
        for obj in doc.objects:
            entry = self._cache.get(obj.uid)
            if entry is None:
                continue
            world = self._world(obj)
            for node, primitive in entry.gpu.draws:
                node.world = world
                draws.append((node, primitive))
        return _Composite(draws) if draws else None

    def _resize(self, width: int, height: int) -> None:
        """Resize, forgetting the outgoing texture first.

        ``Viewport.resize`` releases its texture and makes a new one, and the
        imgui backend maps GL names to moderngl objects: releasing without
        forgetting leaves it holding a dead object under a name the driver is
        free to reissue, which is how an unrelated image starts rendering as
        this one.
        """
        if (width, height) == self.viewport.size:
            return
        self._forget(self.viewport.texture)
        self.viewport.resize((width, height))

    def _forget(self, texture: Any) -> None:
        if texture is None:
            return
        from . import imgui_backend

        renderer = imgui_backend.current()
        if renderer is not None:
            renderer.forget_texture(texture)

    def _gizmo_draws(self, doc: Any, height: int) -> list[Any]:
        gizmo = self.active_gizmo(doc)
        if gizmo is None:
            return []
        centre = self.selection_centre(doc)
        if centre is None:
            return []
        gizmo.place(centre, m3.identity(), self.camera, height)
        return gizmo.draws()

    # -- the gizmo, and the app state that chooses it -----------------------

    def active_gizmo(self, doc: Any) -> Any:
        """The gizmo for the current tool, or None.

        The tool lives on the mode's state rather than here, because it is an
        *app* setting shared across documents -- so this reads it rather than
        holding it. Q (select) draws no gizmo in any mode, which in an element
        mode is what frees the left button for the marquee.
        """
        kind = GIZMO_FOR_TOOL.get(getattr(self.state, "tool", "select"), "")
        if not kind or not doc.selection:
            return None
        if doc.element_mode != "object" and not doc.element_sel:
            return None
        return {
            "translate": self.translate_gizmo,
            "rotate": self.rotate_gizmo,
            "scale": self.scale_gizmo,
        }[kind]

    @property
    def state(self) -> Any:
        """Clay's state, or None when the view is driven headlessly."""
        app_ctx = self.app_ctx
        return None if app_ctx is None else getattr(app_ctx.state, "clay", None)

    def _ray(self, local: tuple[float, float]):
        return screen_ray(
            self.camera, local[0], local[1], int(self._rect[2]), int(self._rect[3])
        )

    # -- capture and teardown ----------------------------------------------

    def screenshot(self) -> Any:
        return capture.image(self.viewport)

    def release(self) -> None:
        self.clear()
        self._release_overlays()
        self.translate_gizmo.release()
        self.rotate_gizmo.release()
        self.scale_gizmo.release()
        # Forgotten before the GL object goes, for the reason ``_resize``
        # states -- the driver reissues the name and an unrelated image starts
        # rendering as this one.
        self._forget(self.viewport.texture)
        self.viewport.release()
        self.renderer.release()
