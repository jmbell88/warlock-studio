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

Registration has a matching half that ``Viewport`` makes easy to miss:
``resize`` releases and recreates its texture, so a resize frees a GL name the
imgui backend may still be holding. :meth:`ClayView.draw` forgets the outgoing
texture before that happens.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .clay import document as bd
from .viewer import capture, glctx, gltf, picking
from .viewer import math3d as m3
from .viewer import scene as scenelib
from .viewer.camera import Camera, screen_ray
from .viewer.gizmo import RotateGizmo, ScaleGizmo, TranslateGizmo
from .viewer.render import Renderer

log = logging.getLogger(__name__)

@dataclass(frozen=True)
class Hit:
    """What a viewport ray found: the object, the distance, and the face.

    The distance is what element picking's occlusion test compares against --
    "no further from the eye than the surface you can see there" -- so it has
    to come back with the hit rather than being re-derived from a second ray
    cast that could disagree with the first.
    """

    uid: int
    t: float
    face: int


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


class _Entry:
    """One object's GPU state, and the key that says whether it is still valid."""

    __slots__ = ("key", "gpu", "model")

    def __init__(self, key: Any, gpu: Any, model: Any) -> None:
        self.key = key
        self.gpu = gpu
        self.model = model


def _materials_key(doc: Any) -> tuple[int, ...]:
    """Identity of every palette entry, in order.

    Identity rather than value, matching ``to_model`` and ``GpuMaterial``:
    ``ClayDoc.set_material`` replaces the entry object, so an edit changes the
    key, and a palette that has not been touched keeps it. Comparing values
    would mean hashing five floats per entry per frame to learn the same thing.
    """
    return tuple(id(m) for m in doc.materials)


def _object_key(obj: Any, materials: tuple[int, ...]) -> tuple[Any, ...]:
    # The mesh by identity -- see the module docstring. The transform is
    # deliberately *not* in the key: it is a uniform, not a buffer, so moving
    # an object must not rebuild it.
    return (id(obj.mesh), materials, obj.material)


class ClayView:
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
        self.radius = 1.0

        self.translate_gizmo = TranslateGizmo(ctx, self.renderer.programs)
        self.rotate_gizmo = RotateGizmo(ctx, self.renderer.programs)
        self.scale_gizmo = ScaleGizmo(ctx, self.renderer.programs)

        self._cache: dict[int, _Entry] = {}
        # Counted rather than inferred: "only what changed was rebuilt" is a
        # property worth asserting, and there is no other way to see it.
        self.rebuilds = 0

        self._rect = (0.0, 0.0, 1.0, 1.0)
        self._grab: str | None = None  # orbit | pan | gizmo
        self._last_mouse = (0.0, 0.0)
        self._drag_uids: list[int] = []
        self._drag_start: dict[int, tuple[Any, Any, Any]] = {}

    # -- the GPU cache -----------------------------------------------------

    def sync(self, doc: Any) -> None:
        """Bring the GPU up to date with the document, rebuilding only what changed."""
        materials = _materials_key(doc)
        live: set[int] = set()
        for obj in doc.objects:
            if not obj.visible:
                continue
            live.add(obj.uid)
            key = _object_key(obj, materials)
            entry = self._cache.get(obj.uid)
            if entry is not None and entry.key == key:
                continue
            if entry is not None:
                entry.gpu.release()
            self._cache[obj.uid] = self._build(obj, doc, key)
            self.rebuilds += 1
        for uid in [u for u in self._cache if u not in live]:
            # A hidden object's buffers go too: ``visible=False`` means it does
            # not render, does not export and is not picked, and holding its
            # upload would make it the one of the three that is only half true.
            self._cache.pop(uid).gpu.release()

    def _build(self, obj: Any, doc: Any, key: Any) -> _Entry:
        prims = bd.to_primitives(obj, doc.materials)
        node = gltf.Node(name=obj.name, mesh=0)
        model = gltf.Model([node], [0], [prims], [])
        return _Entry(key, scenelib.GpuModel(self.ctx, model), model)

    def clear(self) -> None:
        for entry in self._cache.values():
            entry.gpu.release()
        self._cache.clear()

    # -- drawing -----------------------------------------------------------

    def draw(self, doc: Any, rect: tuple[float, float, float, float], dt: float) -> Any:
        """Draw one frame into the viewport. -> the resolved texture."""
        self._rect = rect
        width, height = int(max(rect[2], 1)), int(max(rect[3], 1))
        self._resize(width, height)
        self.camera.update(dt)
        self.sync(doc)

        self.renderer.draw(
            self.viewport,
            self.camera,
            self._composite(doc),
            wireframe=self.wireframe,
            overlays=self._gizmo_draws(doc, height),
        )
        return self.viewport.texture

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
            world = m3.compose(obj.translation, obj.rotation, obj.scale)
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

    # -- selection and framing ---------------------------------------------

    def active_gizmo(self, doc: Any) -> Any:
        """The gizmo for the current tool, or None.

        The tool lives on the mode's state rather than here, because it is an
        *app* setting shared across documents -- so this reads it rather than
        holding it.
        """
        kind = GIZMO_FOR_TOOL.get(getattr(self.state, "tool", "select"), "")
        if not kind or not doc.selection:
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

    def selection_centre(self, doc: Any) -> np.ndarray | None:
        lo, hi = self.world_bounds(doc, selected_only=True)
        return None if lo is None else (lo + hi) * 0.5

    def world_bounds(self, doc: Any, *, selected_only: bool = False) -> tuple[Any, Any]:
        """The world AABB over visible objects, or ``(None, None)`` if there are none."""
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        found = False
        for obj in doc.objects:
            if not obj.visible or (selected_only and obj.uid not in doc.selection):
                continue
            box = self._object_world_box(obj)
            if box is None:
                continue
            lo = np.minimum(lo, box[0])
            hi = np.maximum(hi, box[1])
            found = True
        return (lo, hi) if found else (None, None)

    def _object_world_box(self, obj: Any) -> tuple[Any, Any] | None:
        from .clay import mesh as bm

        if len(obj.mesh.positions) == 0:
            return None
        lo, hi = bm.bounds(obj.mesh)
        corners = np.array(
            [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        )
        matrix = m3.compose(obj.translation, obj.rotation, obj.scale)
        world = (matrix @ np.hstack([corners, np.ones((8, 1))]).T).T[:, :3]
        return world.min(axis=0), world.max(axis=0)

    def frame_selection(self, doc: Any) -> float:
        """Put the selection -- or the whole document -- on screen."""
        lo, hi = self.world_bounds(doc, selected_only=bool(doc.selection))
        if lo is None:
            lo, hi = self.world_bounds(doc)
        if lo is None:
            return 0.0
        self.radius = self.camera.frame(lo, hi)
        self.camera.set_target((lo + hi) * 0.5)
        self.renderer.fit_grid(lo, hi)
        return self.radius

    def pick_face(self, doc: Any, local: tuple[float, float]) -> Hit | None:
        """What a click lands on: which object, how far, and which *face*.

        Object space per object, with the AABB prefilter, so the ray goes
        through one inverse transform per object rather than every triangle
        going through the forward one.

        The triangulation comes from ``adjacency.cached_triangulation``, keyed
        weakly on the mesh -- not from the GPU cache entry, which is also keyed
        on the palette and would therefore be rebuilt by a material edit that
        cannot possibly have moved a triangle. That cache is finally what makes
        ``tri_face`` earn its place: it maps the triangle the ray hit back to
        the n-gon the user thinks they clicked.
        """
        from .clay.adjacency import cached_triangulation

        origin, direction = self._ray(local)
        best: Hit | None = None
        for obj in doc.objects:
            if not obj.visible:
                continue
            tris, tri_face = cached_triangulation(obj.mesh)
            hit = picking.ray_object(
                origin,
                direction,
                m3.compose(obj.translation, obj.rotation, obj.scale),
                obj.mesh.positions.astype("f8"),
                tris,
            )
            if hit is not None and (best is None or hit[0] < best.t):
                face = int(tri_face[hit[1]]) if len(tri_face) else -1
                best = Hit(uid=obj.uid, t=float(hit[0]), face=face)
        return best

    def pick(self, doc: Any, local: tuple[float, float]) -> int | None:
        """Which object a click lands on. -> its uid, or None.

        Kept as the object-mode entry point, and kept returning a bare uid:
        that is what selection in object mode is, and widening it would make
        every caller unpack a record to ignore two thirds of it.
        """
        hit = self.pick_face(doc, local)
        return None if hit is None else hit.uid

    def screen_of(self, doc: Any, uid: int) -> Any:
        """One object's vertices projected into the viewport, for element picking.

        Built here rather than in ``clay/pick.py`` because it is the only step
        that needs the camera; everything the picking rules actually decide is
        pure numpy over what this returns.
        """
        from .clay import pick as bp

        obj = doc.by_uid(uid)
        width, height = int(max(self._rect[2], 1)), int(max(self._rect[3], 1))
        self.camera.aspect = width / max(height, 1)
        return bp.project(
            obj.mesh.positions,
            m3.compose(obj.translation, obj.rotation, obj.scale),
            self.camera.projection() @ self.camera.view(),
            self.camera.position,
            width,
            height,
        )

    # -- input -------------------------------------------------------------

    def handle_event(self, doc: Any, event: Any, hovered: bool) -> bool:
        """Feed one pygame event. -> whether the viewport consumed it.

        ``hovered`` is whether the pointer is over the viewport image; a drag
        already in progress ignores it, so crossing onto a panel mid-orbit does
        not drop the drag.
        """
        import pygame

        local = self._local(event)
        if event.type == pygame.MOUSEBUTTONDOWN and hovered:
            return self._press(doc, event.button, local)
        if event.type == pygame.MOUSEBUTTONUP:
            return self._release_drag(doc)
        if event.type == pygame.MOUSEMOTION:
            return self._motion(doc, local)
        if event.type == pygame.MOUSEWHEEL and hovered:
            self.camera.dolly(event.y)
            return True
        return False

    def _local(self, event: Any) -> tuple[float, float]:
        pos = getattr(event, "pos", None)
        if pos is None:
            return self._last_mouse
        return (pos[0] - self._rect[0], pos[1] - self._rect[1])

    def _press(self, doc: Any, button: int, local: tuple[float, float]) -> bool:
        self._last_mouse = local
        if button not in (1, 2, 3):
            return False
        if button == 1:
            origin, direction = self._ray(local)
            gizmo = self.active_gizmo(doc)
            axis = gizmo.hit(origin, direction) if gizmo is not None else None
            if axis is not None and gizmo.begin(axis, origin, direction):
                # Every selected object's transform is recorded at the press,
                # not read per frame: that is what ``set_transform``'s ``was``
                # argument takes, and reading it live would compare a value
                # against itself and record an empty step.
                self._drag_uids = [o.uid for o in doc.objects if o.uid in doc.selection]
                self._drag_start = {
                    uid: tuple(np.array(v, copy=True) for v in doc.by_uid(uid).trs())
                    for uid in self._drag_uids
                }
                self._grab = "gizmo"
                return True
            hit = self.pick(doc, local)
            doc.select([hit] if hit is not None else [])
            self._grab = "orbit"
            return True
        self._grab = "pan"
        return True

    def _release_drag(self, doc: Any) -> bool:
        if self._grab is None:
            return False
        was, self._grab = self._grab, None
        if was == "gizmo":
            gizmo = self.active_gizmo(doc)
            if gizmo is not None:
                gizmo.end_drag()
            self._commit_drag(doc)
        return True

    def _commit_drag(self, doc: Any) -> None:
        """One history step per object, recorded against where the drag began."""
        for uid, was in self._drag_start.items():
            try:
                doc.set_transform(uid, was=was)
            except KeyError:
                continue  # deleted mid-drag
        self._drag_uids = []
        self._drag_start = {}

    def _motion(self, doc: Any, local: tuple[float, float]) -> bool:
        dx = local[0] - self._last_mouse[0]
        dy = local[1] - self._last_mouse[1]
        self._last_mouse = local
        height = int(max(self._rect[3], 1))
        if self._grab is None:
            gizmo = self.active_gizmo(doc)
            if gizmo is not None:
                origin, direction = self._ray(local)
                gizmo.hover = gizmo.hit(origin, direction)
            return False
        if self._grab == "orbit":
            self.camera.orbit(dx, dy, height)
        elif self._grab == "pan":
            self.camera.pan(dx, dy, height)
        elif self._grab == "gizmo":
            self._drag_gizmo(doc, local)
        return True

    def _drag_gizmo(self, doc: Any, local: tuple[float, float]) -> None:
        """Apply a gizmo delta to every selected object, in place.

        In place, and committed as one history step on release: a step per
        mouse-move would fill the undo stack with a hundred entries for one
        drag, and the intermediate positions are not states the user wants to
        step back through.
        """
        gizmo = self.active_gizmo(doc)
        if gizmo is None:
            return
        state = self.state
        origin, direction = self._ray(local)
        delta = gizmo.update(origin, direction)
        if delta is None:
            return
        for uid, was in self._drag_start.items():
            try:
                obj = doc.by_uid(uid)
            except KeyError:
                continue
            self._apply(obj, was, delta, state)
        doc.touch()

    def _apply(self, obj: Any, was: Any, delta: Any, state: Any) -> None:
        from .clay import ops

        snap = bool(getattr(state, "snap", False))
        if isinstance(delta, np.ndarray) and delta.shape == (3,) and self._is_scale(state):
            obj.scale = np.array(was[2], dtype="f8") * delta
        elif isinstance(delta, np.ndarray) and delta.shape == (4,):
            obj.rotation = m3.quat_normalize(m3.quat_mul(delta, np.array(was[1], dtype="f8")))
            if snap:
                obj.rotation = ops.snap_rotation(obj.rotation, state.snap_rotate)
        else:
            obj.translation = np.array(delta, dtype="f8")
            if snap:
                obj.translation = ops.snap_translation(obj.translation, state.snap_translate)

    def _is_scale(self, state: Any) -> bool:
        return getattr(state, "tool", "") == "scale"

    def _ray(self, local: tuple[float, float]):
        return screen_ray(
            self.camera, local[0], local[1], int(self._rect[2]), int(self._rect[3])
        )

    # -- capture and teardown ----------------------------------------------

    def screenshot(self) -> Any:
        return capture.image(self.viewport)

    def thumbnail_png(self) -> bytes:
        return capture.png_bytes(self.viewport)

    def release(self) -> None:
        self.clear()
        self.translate_gizmo.release()
        self.rotate_gizmo.release()
        self.scale_gizmo.release()
        # Forgotten before the GL object goes, for the reason ``_resize``
        # states -- the driver reissues the name and an unrelated image starts
        # rendering as this one.
        self._forget(self.viewport.texture)
        self.viewport.release()
        self.renderer.release()
