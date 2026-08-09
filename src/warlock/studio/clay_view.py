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
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import moderngl
import numpy as np

from .clay import document as bd
from .clay.topo import corner_spans as _corner_spans
from .clay.topo import flat_next as _flat_next
from .viewer import capture, glctx, gltf, picking
from .viewer import math3d as m3
from .viewer import scene as scenelib
from .viewer.camera import Camera, screen_ray
from .viewer.gizmo import RotateGizmo, ScaleGizmo, TranslateGizmo
from .viewer.render import DrawItem, Renderer

log = logging.getLogger(__name__)

@dataclass
class _ElementDrag:
    """One object's half of a live element drag.

    Everything is recorded at the press and nothing is re-read during the drag:
    ``before`` is the mesh the gizmo was grabbed over, ``verts`` the vertices it
    moves, ``local`` their object-space positions then, and ``matrix`` /
    ``inverse`` the object's placement. A drag that re-read the object every
    frame would compound its own output -- each frame applying the delta to the
    result of the last -- and drift away from the cursor.
    """

    before: Any
    verts: Any
    local: Any
    matrix: Any
    inverse: Any
    # The positions the last preview frame produced. The commit reads *this*
    # rather than the object's mesh, because the whole point of the preview is
    # that the object's mesh never moved -- reading it back would find the
    # press-time geometry and commit nothing at all.
    preview: Any = None
    # One weight per entry of ``verts``, or None for a hard selection. It is a
    # weight rather than a second vertex list because the falloff has to be a
    # *blend*: a vertex at 0.3 moves three tenths of the way, which no set
    # membership can express.
    weights: Any = None


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

# How close the cursor has to come to a vertex for a move to snap onto it, in
# pixels. Larger than ``pick.VERT_RADIUS`` on purpose: picking asks "did you
# mean to click this", where a generous radius selects the wrong thing, and
# snapping asks "are you near this", where a mean one makes the feature feel
# broken -- the user is already dragging, so there is nothing else the gesture
# could have meant.
SNAP_VERTEX_RADIUS = 14.0


def _about(centre: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """*matrix* conjugated to act about *centre* rather than about the origin."""
    to = m3.identity()
    to[:3, 3] = -np.asarray(centre, dtype="f8")
    back = m3.identity()
    back[:3, 3] = np.asarray(centre, dtype="f8")
    return back @ matrix @ to


def _apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homo = np.hstack([np.asarray(points, dtype="f8"), np.ones((len(points), 1))])
    return (np.asarray(matrix, dtype="f8") @ homo.T).T[:, :3]


# --- the element-selection overlay -------------------------------------------

# Colours, matching the rest of the app's accent language: red for what is
# selected, yellow for what the cursor is over, and a dim grey for the guides
# that show where the elements *are* before any of them is selected.
SEL_COLOR = (0.95, 0.25, 0.25, 1.0)
HOVER_COLOR = (1.0, 0.85, 0.2, 1.0)
GUIDE_COLOR = (0.55, 0.58, 0.62, 0.35)
FILL_COLOR = (0.95, 0.25, 0.25, 0.28)

# How far a selected face's translucent fill is pulled toward the eye, as a
# fraction of its distance. ``glPolygonOffset`` is the textbook answer and is
# deliberately not used: it is global GL state that moderngl caches, so setting
# it here would leak into the gizmo pass and the grid, and unsetting it is one
# more thing to get wrong on an early return. Scaling the model matrix about
# the eye is local to the draw and needs no cleanup.
FILL_BIAS = 0.0015


class _SelOverlay:
    """One object's selection overlay: guides, selection, hover, and the fill.

    **One position buffer, many index buffers.** Every overlay draw for an
    object reads the same vertices -- the wireframe guide, the selected edges,
    the hovered face -- so uploading the positions once and varying only the
    index buffer is both the small upload and the reason a *hover* change is
    nearly free: it rebuilds one tiny index buffer and touches nothing else.

    The whole thing is keyed on ``(id(mesh), id(sel), mode, hover)``. Identity
    on the first two because ``Mesh`` and ``ElementSel`` are both frozen and
    replaced whole rather than mutated -- which is exactly what makes identity
    a sound cache key, and is stated as such in both of their docstrings.

    ``specs`` is the draw list built from those buffers, cached here because it
    too is a pure function of the key: every index buffer is decided by the
    mesh, the selection, the mode and the hover, so it is built once per key
    and replayed each frame. Building it per frame -- which is what calling
    ``indexed`` per draw per frame amounts to -- minted a fresh IBO and VAO
    every time and released them only on a key change, a GL-object leak at
    frame rate for as long as the cursor held still.
    """

    __slots__ = ("ctx", "key", "pos_vbo", "count", "specs", "_ibos", "_vaos", "_program")

    def __init__(self, ctx: Any, program: Any, key: Any, positions: Any) -> None:
        self.ctx = ctx
        self._program = program
        self.key = key
        data = np.ascontiguousarray(positions, dtype="f4")
        self.count = len(data)
        self.pos_vbo = ctx.buffer(data.tobytes())
        self.specs: list[Any] | None = None
        self._ibos: list[Any] = []
        self._vaos: list[Any] = []

    def write_positions(self, positions: Any) -> None:
        """Rewrite the vertices in place, for a live drag. No VAO is rebuilt."""
        data = np.ascontiguousarray(positions, dtype="f4")
        if len(data) == self.count:
            self.pos_vbo.write(data.tobytes())

    def indexed(self, indices: Any, mode: int) -> Any:
        """A vertex array over the shared positions and a fresh index buffer."""
        flat = np.ascontiguousarray(indices, dtype="u4").reshape(-1)
        if len(flat) == 0:
            return None
        ibo = self.ctx.buffer(flat.tobytes())
        vao = self.ctx.vertex_array(
            self._program, [(self.pos_vbo, "3f", "a_position")], ibo
        )
        self._ibos.append(ibo)
        self._vaos.append(vao)
        return vao

    def release(self) -> None:
        self.specs = None
        for vao in self._vaos:
            vao.release()
        for ibo in self._ibos:
            ibo.release()
        self._vaos.clear()
        self._ibos.clear()
        self.pos_vbo.release()


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

        # What the cursor is over in an element mode, as ``(uid, index)`` read
        # through the document's own mode. Updated only on motion with no grab
        # -- a hover that recomputed every frame would reproject every vertex
        # of every object while the scene sits perfectly still.
        self.hover_element: tuple[int, int] | None = None
        self._screens: dict[int, tuple[Any, Any]] = {}

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
        self._world_cache: dict[int, tuple[tuple[int, int, int], Any]] = {}
        # element_centre / selection_centre / world_bounds memos (B25/B27).
        self._centre_memo: tuple[Any, Any] | None = None
        self._bounds_memo: dict[bool, tuple[Any, tuple[Any, Any]]] = {}
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
        self._screens.clear()

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
            show_grid=self.show_grid,
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
        self._world_cache[obj.uid] = (key, world)
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

    def _element_overlays(self, doc: Any) -> list[Any]:
        """The element-mode overlay draws for every visible object.

        Guides are depth-tested, so the far side of a closed mesh does not fog
        the near one with a grey haze; the *selection* is depth-off, so a
        selected vertex you have orbited behind the surface is still visible
        and still says it is selected. That split is the whole reason
        ``DrawItem`` grew a ``depth`` flag.
        """
        if doc.element_mode == "object":
            self._release_overlays()
            return []

        program = self.renderer.programs.get("solid")
        mode = doc.element_mode
        hover = self.hover_element
        items: list[Any] = []
        live: set[int] = set()
        for obj in doc.objects:
            if not obj.visible or len(obj.mesh.positions) == 0:
                continue
            live.add(obj.uid)
            # Keyed on what the *document* holds, not on what the accessor
            # hands back: ``element_sel_of`` synthesises a fresh ``empty()``
            # for an object with nothing selected, and nothing keeps it alive
            # -- so the key changed every frame (a full VBO re-upload per
            # unselected object) or, once the allocator reissued the address,
            # matched a stale overlay and left a new selection invisible.
            stored = doc.element_sel.get(obj.uid)
            if stored is not None:
                sel = doc.element_sel_of(obj.uid)
            else:
                # One shared empty selection for every unselected object
                # (B26): synthesising a fresh empty() per object per frame
                # allocated for nothing, and the overlay key deliberately uses
                # ``id(stored)`` -- which is stable at None -- not this.
                if self._empty_sel is None:
                    self._empty_sel = doc.element_sel_of(obj.uid)
                sel = self._empty_sel
            hover_index = hover[1] if hover is not None and hover[0] == obj.uid else -1
            key = (id(obj.mesh), id(stored), mode, hover_index)
            overlay = self._overlays.get(obj.uid)
            if overlay is None or overlay.key != key:
                if overlay is not None:
                    overlay.release()
                overlay = _SelOverlay(self.ctx, program, key, obj.mesh.positions)
                self._overlays[obj.uid] = overlay
            world = self._world(obj)
            items.extend(
                self._overlay_items(obj, overlay, world, mode, sel, hover_index)
            )
        for uid in [u for u in self._overlays if u not in live]:
            self._overlays.pop(uid).release()
        return items

    def _overlay_items(
        self, obj: Any, overlay: Any, world: Any, mode: str, sel: Any, hover: int
    ) -> list[Any]:
        if overlay.specs is None:
            overlay.specs = self._overlay_specs(obj, overlay, mode, sel, hover)
        items: list[Any] = []
        for vao, gl_mode, color, depth, size, biased in overlay.specs:
            items.append(
                DrawItem(
                    vao=vao,
                    color=color,
                    model=_toward_eye(self.camera.position) @ world if biased
                    else world,
                    mode=gl_mode,
                    depth=depth,
                    point_size=size,
                )
            )
        return items

    def _overlay_specs(
        self, obj: Any, overlay: Any, mode: str, sel: Any, hover: int
    ) -> list[Any]:
        """Build the overlay's draws, once per cache key.

        Only the matrices are per-frame -- the object can move under a drag and
        the fill's eye-ward bias follows the camera -- which is why a spec
        records *whether* to bias rather than a matrix, and why ``world`` is
        composed by the caller each frame.
        """
        from .clay.adjacency import adjacency, cached_triangulation

        adj = adjacency(obj.mesh)
        specs: list[Any] = []

        def add(
            indices: Any,
            gl_mode: int,
            color: Any,
            *,
            depth: bool,
            size: float = 0.0,
            biased: bool = False,
        ) -> None:
            vao = overlay.indexed(indices, gl_mode)
            if vao is not None:
                specs.append((vao, gl_mode, color, depth, size, biased))

        # The dim guides: where the elements are, before any is picked.
        add(adj.edge_verts, moderngl.LINES, GUIDE_COLOR, depth=True)
        if mode == "vertex":
            add(np.arange(len(obj.mesh.positions)), moderngl.POINTS, GUIDE_COLOR,
                depth=True, size=3.0)
            add(sel.verts, moderngl.POINTS, SEL_COLOR, depth=False, size=7.0)
            if hover >= 0:
                add([hover], moderngl.POINTS, HOVER_COLOR, depth=False, size=9.0)
        elif mode == "edge":
            ids = adj.edge_ids(sel.edges)
            add(adj.edge_verts[ids[ids >= 0]], moderngl.LINES, SEL_COLOR, depth=False)
            if hover >= 0:
                add(adj.edge_verts[hover], moderngl.LINES, HOVER_COLOR, depth=False)
        else:
            tris, tri_face = cached_triangulation(obj.mesh)
            if len(tris):
                chosen = tris[np.isin(tri_face, sel.faces)]
                # Depth-tested, so a fill on the far side of a closed mesh does
                # not bleed through it -- and biased toward the eye so it does
                # not z-fight the very face it is covering.
                add(chosen, moderngl.TRIANGLES, FILL_COLOR, depth=True, biased=True)
                add(_face_outline(obj.mesh, sel.faces), moderngl.LINES, SEL_COLOR,
                    depth=False)
                if hover >= 0:
                    add(tris[tri_face == hover], moderngl.TRIANGLES, HOVER_COLOR,
                        depth=False)
        return specs

    def _release_overlays(self) -> None:
        for overlay in self._overlays.values():
            overlay.release()
        self._overlays.clear()

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

    def selection_centre(self, doc: Any) -> np.ndarray | None:
        """Where the gizmo goes: the object box's centre, or the elements'.

        In an element mode the gizmo belongs to what is selected *inside* the
        objects, not to the objects: putting it at the object's centre would
        leave a user dragging a handle two metres from the four vertices it
        moves, with the rotate ring pivoting about a point they never chose.
        """
        if doc.element_mode != "object":
            return self.element_centre(doc)
        lo, hi = self.world_bounds(doc, selected_only=True)
        return None if lo is None else (lo + hi) * 0.5

    def element_centre(self, doc: Any) -> np.ndarray | None:
        """The world centroid of every selected element, across every object.

        Memoized on the identities of everything it reads (B25): the selection
        objects, the meshes and the transform arrays are all replaced whole
        rather than mutated, so identity misses exactly when the answer can
        change -- and this is asked every frame a gizmo is on screen.
        """
        from .clay import elements as el

        key = (id(doc), tuple(
            (uid, id(sel), *self._obj_key(doc, uid))
            for uid, sel in doc.element_sel.items()
        ))
        memo = self._centre_memo
        if memo is not None and memo[0] == key:
            return memo[1]

        total = np.zeros(3)
        count = 0
        for uid, sel in doc.element_sel.items():
            try:
                obj = doc.by_uid(uid)
            except KeyError:
                continue
            verts = el.affected_verts(obj.mesh, sel)
            if not len(verts):
                continue
            matrix = self._world(obj)
            local = obj.mesh.positions[verts].astype("f8")
            world = (matrix @ np.hstack([local, np.ones((len(local), 1))]).T).T[:, :3]
            total += world.sum(axis=0)
            count += len(world)
        centre = None if count == 0 else total / count
        self._centre_memo = (key, centre)
        return centre

    @staticmethod
    def _obj_key(doc: Any, uid: int) -> tuple:
        """The identity triple-plus-mesh that pins one object's geometry."""
        try:
            obj = doc.by_uid(uid)
        except KeyError:
            return ()
        return (id(obj.mesh), id(obj.translation), id(obj.rotation), id(obj.scale))

    def world_bounds(self, doc: Any, *, selected_only: bool = False) -> tuple[Any, Any]:
        """The world AABB over visible objects, or ``(None, None)`` if there are none.

        Memoized the way ``element_centre`` is (B27): the key names every
        object's visibility, selection membership, mesh identity and transform
        identities, so a miss happens exactly when the box can move.
        """
        key = (id(doc), tuple(
            (
                obj.uid, obj.visible, obj.uid in doc.selection,
                id(obj.mesh), id(obj.translation), id(obj.rotation), id(obj.scale),
            )
            for obj in doc.objects
        ))
        memo = self._bounds_memo.get(selected_only)
        if memo is not None and memo[0] == key:
            return memo[1]
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
        out = (lo, hi) if found else (None, None)
        self._bounds_memo[selected_only] = (key, out)
        return out

    def _object_world_box(self, obj: Any) -> tuple[Any, Any] | None:
        """``ops.world_box``, and deliberately nothing else.

        It used to be a second copy of the same eight corners, which is how the
        properties panel's dimensions row and the camera's framing would have
        come to disagree about the size of one object.
        """
        from .clay import ops as bops

        return bops.world_box(obj)

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
                self._world(obj),
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

        Cached per object on ``(id(mesh), transform, camera, rect)``, which is
        every input the projection has. Hover runs on every mouse move, and a
        200k-vertex import reprojected per move is the difference between a
        viewport and a slideshow -- while a camera that has not moved makes the
        key hit and the whole thing free.
        """
        from .clay import pick as bp

        obj = doc.by_uid(uid)
        width, height = int(max(self._rect[2], 1)), int(max(self._rect[3], 1))
        self.camera.aspect = width / max(height, 1)
        matrix = self._world(obj)
        key = (
            id(obj.mesh),
            matrix.tobytes(),
            (self.camera.theta, self.camera.phi, self.camera.distance),
            tuple(self.camera.target),
            (width, height),
        )
        cached = self._screens.get(uid)
        if cached is not None and cached[0] == key:
            return cached[1]
        screen = bp.project(
            obj.mesh.positions,
            matrix,
            self.camera.projection() @ self.camera.view(),
            self.camera.position,
            width,
            height,
        )
        self._screens[uid] = (key, screen)
        return screen

    def pick_element(
        self, doc: Any, local: tuple[float, float], hit: Any = None
    ) -> tuple[int, int] | None:
        """What element is under the cursor: ``(uid, index)``, read through the mode.

        The *index* is a vertex index, an index into ``adjacency.edge_verts`` or
        a face index, depending on ``doc.element_mode`` -- one shape for all
        three, because every caller here does the same thing with it.

        The surface hit is passed in rather than recast so the occlusion test
        compares against the same ray the object pick used; recasting would let
        the two disagree by an ulp and make a vertex on the near face flicker
        in and out of pickability.
        """
        from .clay import pick as bp
        from .clay.adjacency import adjacency

        mode = doc.element_mode
        if mode == "object":
            return None
        if hit is None:
            hit = self.pick_face(doc, local)
        depth = None if hit is None else hit.t

        best: tuple[float, int, int] | None = None
        for obj in doc.objects:
            if not obj.visible:
                continue
            screen = self.screen_of(doc, obj.uid)
            if mode == "vertex":
                index = bp.nearest_vertex(screen, local, surface_depth=depth)
            elif mode == "edge":
                index = bp.nearest_edge(
                    screen, adjacency(obj.mesh).edge_verts, local, surface_depth=depth
                )
            else:
                index = hit.face if hit is not None and hit.uid == obj.uid else None
                index = None if index is not None and index < 0 else index
            if index is None:
                continue
            key = float(screen.depth[index]) if mode == "vertex" else 0.0
            if best is None or key < best[0]:
                best = (key, obj.uid, int(index))
        return None if best is None else (best[1], best[2])

    def element_sel_for(self, doc: Any, uid: int, index: int) -> Any:
        """One picked element as an :class:`~.clay.elements.ElementSel`."""
        from .clay import elements as el
        from .clay.adjacency import adjacency

        mode = doc.element_mode
        if mode == "vertex":
            return el.ElementSel(verts=[index])
        if mode == "edge":
            return el.ElementSel(edges=[adjacency(doc.by_uid(uid).mesh).edge_verts[index]])
        return el.ElementSel(faces=[index])

    # -- input -------------------------------------------------------------

    def handle_event(self, doc: Any, event: Any, hovered: bool) -> bool:
        """Feed one pygame event. -> whether the viewport consumed it.

        ``hovered`` is whether the pointer is over the viewport image; a drag
        already in progress ignores it, so crossing onto a panel mid-orbit does
        not drop the drag.
        """
        import pygame

        # Any event that reaches the viewport can move the picture: a press
        # starts a drag or a marquee, motion re-hovers an element, a wheel
        # dollies. Cheaper to redraw one frame than to enumerate which.
        self._render_dirty = True
        local = self._local(event)
        if event.type == pygame.MOUSEBUTTONDOWN and hovered:
            return self._press(doc, event.button, local)
        if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 3:
                return self._rmb_release(local)
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

    def _mods(self) -> tuple[bool, bool, bool]:
        """``(shift, ctrl, alt)`` at this instant. See the module docstring."""
        try:
            import pygame

            mods = pygame.key.get_mods()
            return (
                bool(mods & pygame.KMOD_SHIFT),
                bool(mods & pygame.KMOD_CTRL),
                bool(mods & pygame.KMOD_ALT),
            )
        except Exception:  # pragma: no cover - headless pygame without a display
            return (False, False, False)

    def _press(self, doc: Any, button: int, local: tuple[float, float]) -> bool:
        self._last_mouse = local
        # A gizmo drag owns the mouse until its button comes up. Without this,
        # pressing the middle button mid-drag overwrote ``_grab`` with "pan",
        # so releasing the left button found nothing to commit and the drag was
        # stranded: the object stayed wherever the last motion put it, with no
        # history step and the gizmo still holding a live drag.
        if self._grab == "gizmo" and button != 1:
            return True
        if button == 3:
            self._rmb_at = local
            return True
        if button == 2:
            self._grab = "pan"
            return True
        if button != 1:
            return False

        shift, ctrl, alt = self._mods()
        if alt:
            # Alt+drag always orbits, in every mode: it is the one gesture that
            # must never be reinterpreted, because it is how a user looks at
            # what they are about to click.
            self._grab = "orbit"
            return True

        origin, direction = self._ray(local)
        gizmo = self.active_gizmo(doc)
        axis = gizmo.hit(origin, direction) if gizmo is not None else None
        if axis is not None and gizmo.begin(axis, origin, direction):
            self._begin_gizmo_drag(doc)
            return True

        if doc.element_mode != "object":
            return self._press_element(doc, local, shift=shift, ctrl=ctrl)

        hit = self.pick(doc, local)
        doc.select([hit] if hit is not None else [])
        self._grab = "orbit"
        return True

    def _begin_gizmo_drag(self, doc: Any) -> None:
        """Record every selected object's transform at the press.

        Not read per frame: that is what ``set_transform``'s ``was`` argument
        takes, and reading it live would compare a value against itself and
        record an empty step.
        """
        self._drag_uids = [o.uid for o in doc.objects if o.uid in doc.selection]
        self._drag_start = {
            uid: tuple(np.array(v, copy=True) for v in doc.by_uid(uid).trs())
            for uid in self._drag_uids
        }
        self._drag_quat = m3.quat_identity()
        gizmo = self.active_gizmo(doc)
        origin = None if gizmo is None else getattr(gizmo, "origin", None)
        self._drag_origin = np.zeros(3) if origin is None else np.array(origin, dtype="f8")
        self._grab = "gizmo"
        from .clay import drag as bdrag

        self.drag_input = bdrag.DragInput()
        self.drag_hud = ""
        self._snap_point = None
        if doc.element_mode != "object":
            self._begin_element_drag(doc)

    def _press_element(
        self, doc: Any, local: tuple[float, float], *, shift: bool, ctrl: bool
    ) -> bool:
        """LMB in an element mode: pick, clear, or start a marquee.

        The order is what makes it feel right. An element under the cursor
        wins; failing that, a *surface* under the cursor means the user clicked
        the object and missed everything on it, which clears that object rather
        than the whole selection; failing that, they clicked empty space, and
        what happens then is the tool's business -- Q starts a marquee, the
        transform tools orbit, because dragging a gizmo tool over the void is
        how a user looks around.
        """
        from .clay import elements as el

        how = "add" if shift else ("subtract" if ctrl else "replace")
        hit = self.pick_face(doc, local)
        picked = self.pick_element(doc, local, hit)
        if picked is not None:
            uid, index = picked
            one = self.element_sel_for(doc, uid, index)
            if how == "replace":
                doc.clear_element_sel()
            doc.set_element_sel(
                uid, el.combine(doc.element_sel_of(uid), one, how)
            )
            self._grab = "orbit"
            return True
        if hit is not None:
            doc.set_element_sel(hit.uid, el.empty())
            self._grab = "orbit"
            return True
        if getattr(self.state, "tool", "select") == "select":
            self._grab = "marquee"
            self._marquee_from = local
            self._marquee_add = how
            self.marquee = (local[0], local[1], local[0], local[1])
        else:
            self._grab = "orbit"
        return True

    def _rmb_release(self, local: tuple[float, float]) -> bool:
        """The context menu, on a release that did not travel.

        Four pixels rather than zero: a right-click on a trackpad routinely
        moves one or two, and a menu that refuses to open because the finger
        shifted reads as the app ignoring the click.
        """
        at, self._rmb_at = self._rmb_at, None
        if at is None:
            return False
        if abs(local[0] - at[0]) < 4.0 and abs(local[1] - at[1]) < 4.0:
            self.menu_request = local
        return True

    def _release_drag(self, doc: Any) -> bool:
        if self._grab is None:
            return False
        was, self._grab = self._grab, None
        if was == "gizmo":
            gizmo = self.active_gizmo(doc)
            if gizmo is not None:
                gizmo.end_drag()
            if doc.element_mode != "object":
                self._commit_element_drag(doc)
            else:
                self._commit_drag(doc)
            self._clear_drag_input()
        elif was == "marquee":
            self._commit_marquee(doc)
        return True

    # -- the keyboard's half of a drag (Clay18) ------------------------------

    @property
    def dragging(self) -> bool:
        """Whether a gizmo drag is live. What the key handler checks *first*.

        First because the number row is bound to the element modes: a ``1``
        typed into a drag has to be a digit, not a jump into vertex mode
        halfway through moving something.
        """
        return self._grab == "gizmo"

    def _clear_drag_input(self) -> None:
        from .clay import drag as bdrag

        self.drag_input = bdrag.DragInput()
        self.drag_hud = ""
        self._snap_point = None

    def drag_key(self, doc: Any, name: str) -> bool:
        """Feed one key name to the live drag. -> whether it was consumed.

        The transform is re-applied immediately rather than waiting for the next
        mouse move, because a typed value with the mouse held still is the whole
        point of typing one. Re-applying is safe because every gizmo's
        ``update`` is a function of the ray it is handed and the press it began
        from: ``RotateGizmo`` returns a zero increment for an unchanged ray,
        and the other two are measured from the press outright.
        """
        if not self.dragging or not self.drag_input.key(name):
            return False
        self._drag_gizmo(doc, self._last_mouse)
        return True

    def cancel_drag(self, doc: Any) -> bool:
        """Esc during a drag: put everything back and record nothing.

        Distinct from a release, which commits. Without it Esc cleared the
        pane's own drag bookkeeping and left the view still holding the grab,
        so the objects stayed wherever the last motion put them with no history
        step to take them back.
        """
        if not self.dragging:
            return False
        self._grab = None
        gizmo = self.active_gizmo(doc)
        if gizmo is not None:
            gizmo.end_drag()
        if doc.element_mode != "object":
            drags, self._element_drags = self._element_drags, {}
            for uid in drags:
                entry = self._cache.pop(uid, None)
                if entry is not None:
                    entry.gpu.release()
        else:
            for uid, was in self._drag_start.items():
                try:
                    obj = doc.by_uid(uid)
                except KeyError:
                    continue
                obj.translation, obj.rotation, obj.scale = (np.array(v, copy=True) for v in was)
            self._drag_uids = []
            self._drag_start = {}
        self._clear_drag_input()
        doc.touch()
        return True

    def _snap_vertex(self, doc: Any, local: tuple[float, float]) -> np.ndarray | None:
        """The world position of the vertex under the cursor, or ``None``.

        Screen-space rather than a world-space proximity search, which is the
        difference between "snap to what I am pointing at" and "snap to whatever
        happens to be near the thing I am moving" -- and only the first is a
        gesture the user can aim.

        The vertices being dragged are excluded per object. A drag that could
        snap onto its own moving geometry would track the cursor exactly and
        report a snap, which is the worst failure available: it looks like the
        feature working.
        """
        from .clay import pick as bp

        best: tuple[float, np.ndarray] | None = None
        for obj in doc.objects:
            if not obj.visible:
                continue
            drag = self._element_drags.get(obj.uid)
            allowed = None
            if drag is not None:
                allowed = np.ones(len(obj.mesh.positions), dtype=bool)
                allowed[np.asarray(drag.verts, dtype="i8")] = False
            screen = self.screen_of(doc, obj.uid)
            index = bp.nearest_vertex(
                screen, local, radius=SNAP_VERTEX_RADIUS, allowed=allowed
            )
            if index is None:
                continue
            depth = float(screen.depth[index])
            if best is not None and depth >= best[0]:
                continue
            local_pos = np.append(obj.mesh.positions[index].astype("f8"), 1.0)
            best = (depth, (self._world(obj) @ local_pos)[:3])
        return None if best is None else best[1]

    def _commit_marquee(self, doc: Any) -> None:
        """Apply the swept rectangle, or clear the selection if it has no area.

        A zero-area marquee is a click on empty space, and clicking empty space
        deselects -- so the same gesture does both, with the area deciding
        which, rather than the press having to guess in advance.
        """
        from .clay import elements as el
        from .clay import pick as bp
        from .clay.adjacency import adjacency

        rect, self.marquee, self._marquee_from = self.marquee, None, None
        if rect is None:
            return
        if abs(rect[2] - rect[0]) < 2.0 and abs(rect[3] - rect[1]) < 2.0:
            if self._marquee_add == "replace":
                doc.clear_element_sel()
            return

        mode = doc.element_mode
        if self._marquee_add == "replace":
            doc.clear_element_sel()
        for obj in doc.objects:
            if not obj.visible:
                continue
            screen = self.screen_of(doc, obj.uid)
            if mode == "vertex":
                swept = el.ElementSel(verts=bp.marquee_verts(screen, rect))
            elif mode == "edge":
                swept = el.ElementSel(
                    edges=bp.marquee_edges(screen, adjacency(obj.mesh).edge_verts, rect)
                )
            else:
                swept = el.ElementSel(
                    faces=bp.marquee_faces(screen, obj.mesh.loops, obj.mesh.starts, rect)
                )
            if el.is_empty(swept):
                continue
            how = "subtract" if self._marquee_add == "subtract" else "add"
            doc.set_element_sel(
                obj.uid, el.combine(doc.element_sel_of(obj.uid), swept, how)
            )

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
            self.hover_element = (
                None if doc.element_mode == "object" else self.pick_element(doc, local)
            )
            return False
        if self._grab == "marquee":
            start = self._marquee_from or local
            self.marquee = (start[0], start[1], local[0], local[1])
            return True
        if self._grab == "orbit":
            self.camera.orbit(dx, dy, height)
        elif self._grab == "pan":
            self.camera.pan(dx, dy, height)
        elif self._grab == "gizmo":
            self._drag_gizmo(doc, local)
        return True

    def _begin_element_drag(self, doc: Any) -> None:
        """Snapshot every selected object's affected vertices at the press."""
        from .clay import drag as bdrag
        from .clay import elements as el

        state = self.state
        radius = 0.0
        if bool(getattr(state, "proportional", False)):
            radius = float(getattr(state, "proportional_radius", 0.0))

        self._element_drags = {}
        centre = self.element_centre(doc)
        self._element_centre = np.zeros(3) if centre is None else centre
        for uid, sel in doc.element_sel.items():
            try:
                obj = doc.by_uid(uid)
            except KeyError:
                continue
            verts = el.affected_verts(obj.mesh, sel)
            if not len(verts):
                continue
            weights = None
            if radius > 0.0:
                # The radius is metres of *world* space -- that is what the
                # field says and what the user judges by eye -- and the search
                # is in the object's own frame, so an object at scale 2 would
                # otherwise reach twice as far as the number on the panel. Max
                # rather than a mean, the ``ops.join`` rule: it is the bound
                # that holds per axis.
                scale = float(np.max(np.abs(np.asarray(obj.scale, dtype="f8"))))
                verts, weights = bdrag.proportional_set(
                    obj.mesh.positions, verts, radius / scale if scale > 0.0 else radius
                )
            matrix = self._world(obj)
            try:
                inverse = np.linalg.inv(matrix)
            except np.linalg.LinAlgError:
                continue
            self._element_drags[uid] = _ElementDrag(
                before=obj.mesh,
                verts=verts,
                local=obj.mesh.positions[verts].astype("f8"),
                matrix=matrix,
                inverse=inverse,
                weights=weights,
            )

    def _element_world_transform(self, delta: Any, state: Any) -> Any:
        """The gizmo's delta as a world-space affine about the drag's centre.

        One 4x4 for all three tools, so the per-vertex step downstream is a
        single ``inverse @ W @ matrix`` and knows nothing about which gizmo the
        user grabbed.
        """
        from .clay import ops

        centre = self._element_centre
        # The grid stands down for anything more specific. A vertex snap and a
        # typed value are both statements about where the thing goes, and
        # re-quantising either onto the grid afterwards would move it off the
        # answer the user just gave -- which is the same ordering ``_narrow``
        # states and is repeated here because this is where it takes effect.
        snap = bool(getattr(state, "snap", False)) and not (
            self._snap_point is not None or self.drag_input.active
        )
        world = m3.identity()
        if isinstance(delta, np.ndarray) and delta.shape == (4,):
            if snap:
                delta = ops.snap_rotation(delta, getattr(state, "snap_rotate", 0.0))
            rotation = m3.compose(m3.vec3(), delta, m3.vec3(1.0, 1.0, 1.0))
            world = _about(centre, rotation)
        elif self._is_scale(state):
            factors = np.asarray(delta, dtype="f8").reshape(3)
            scale = np.diag(np.append(factors, 1.0))
            world = _about(centre, scale)
        else:
            target = np.asarray(delta, dtype="f8").reshape(3)
            if snap:
                target = ops.snap_translation(target, getattr(state, "snap_translate", 0.0))
            world = m3.identity()
            world[:3, 3] = target - centre
        return world

    def _preview_element_drag(self, doc: Any, delta: Any, state: Any) -> None:
        """Move the affected vertices on the GPU only, without touching the document.

        Nothing is pushed and no ``Mesh`` reaches the document: the drag writes
        the moved vertices straight into the cached buffers, so ``rebuilds``
        stays flat across a whole drag and there is exactly one history step at
        the end rather than one per mouse-move.
        """
        world = self._element_world_transform(delta, state)
        for uid, drag in self._element_drags.items():
            moved = _apply_affine(drag.inverse @ world @ drag.matrix, drag.local)
            if drag.weights is not None:
                # The blend is between where the vertex *was* and where the
                # whole transform would have put it, which is exact for a
                # translation and the standard reading for the other two: a
                # partially rotated vertex is one that has travelled part of the
                # way along the same path.
                moved = drag.local + drag.weights[:, None] * (moved - drag.local)
            positions = np.array(drag.before.positions, dtype="f4")
            positions[drag.verts] = moved
            drag.preview = positions
            self._preview_positions(doc, uid, positions)
        doc.touch()

    def _preview_positions(self, doc: Any, uid: int, positions: Any) -> None:
        """Rewrite one object's vertex buffers in place. No VAO is rebuilt."""
        from dataclasses import replace

        entry = self._cache.get(uid)
        if entry is None:
            return
        obj = doc.by_uid(uid)
        drag = self._element_drags.get(uid)
        base = obj.mesh if drag is None else drag.before
        preview = replace(obj, mesh=replace(base, positions=positions))
        prims = bd.to_primitives(preview, doc.materials)
        for (_node, gpu), primitive in zip(entry.gpu.draws, prims, strict=False):
            try:
                gpu.update_vertices(primitive)
            except ValueError:  # pragma: no cover - topology changed under a drag
                return
        overlay = getattr(self, "_overlays", {}).get(uid)
        if overlay is not None:
            overlay.write_positions(positions)

    def _commit_element_drag(self, doc: Any) -> None:
        """One ``MeshEdit`` per object, against the mesh the drag began on.

        A drag that moved nothing pushes nothing -- dirty is a comparison
        against the history head, and a no-op step would make a saved document
        ask to be saved again. It still has to evict the previewed entries: the
        buffers hold whatever the last preview frame wrote, and the mesh
        identity the cache keys on has not changed, so nothing else would ever
        rebuild them.
        """
        from dataclasses import replace

        drags, self._element_drags = self._element_drags, {}
        for uid, drag in drags.items():
            try:
                doc.by_uid(uid)
            except KeyError:
                continue
            final = drag.before.positions if drag.preview is None else drag.preview
            if np.array_equal(final, drag.before.positions):
                entry = self._cache.pop(uid, None)
                if entry is not None:
                    entry.gpu.release()
                continue
            doc.set_mesh(
                uid,
                replace(drag.before, positions=final),
                select=doc.element_sel_of(uid),
            )

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
        delta = self._accumulate(delta)
        delta = self._narrow(doc, delta, state, local)
        if doc.element_mode != "object":
            self._preview_element_drag(doc, delta, state)
            return
        for uid, was in self._drag_start.items():
            try:
                obj = doc.by_uid(uid)
            except KeyError:
                continue
            self._apply(obj, was, delta, state)
        doc.touch()

    def _narrow(self, doc: Any, delta: Any, state: Any, local: tuple[float, float]) -> Any:
        """Axis lock, typed value and vertex snap, applied to one drag delta.

        One place for all three, above both the object path and the element
        path, so neither has to learn what a lock is -- the same argument that
        keeps the constraint out of the three gizmos. It also fixes the order,
        which is a decision rather than an accident: **an explicit constraint
        beats a snap.** A user who has typed ``X 2`` has said exactly where the
        thing goes, and quietly moving it onto a nearby vertex instead would be
        the app overruling a number it was given.

        The anchor for a translation is the same point the consumer measures
        against -- the element centroid in an element mode, the gizmo's own
        origin otherwise -- so the constrained displacement is the one that is
        actually applied rather than one measured from somewhere else.
        """
        from .clay import drag as bdrag

        tool = getattr(state, "tool", "") if state is not None else ""
        entry = self.drag_input
        if isinstance(delta, np.ndarray) and delta.shape == (4,):
            self.drag_hud = _rotation_hud(delta, entry)
            return bdrag.constrain_rotation(delta, entry)
        if self._is_scale(state):
            out = bdrag.constrain_scale(delta, entry)
            self.drag_hud = bdrag.readout("scale", out, entry)
            return out

        anchor = self._element_centre if doc.element_mode != "object" else self._drag_origin
        target = np.asarray(delta, dtype="f8").reshape(3)
        self._snap_point = None
        if not entry.active and bool(getattr(state, "snap_vertex", False)):
            self._snap_point = self._snap_vertex(doc, local)
            if self._snap_point is not None:
                target = self._snap_point
        moved = bdrag.constrain_translation(target - anchor, entry)
        self.drag_hud = bdrag.readout(tool or "move", moved, entry)
        if self._snap_point is not None:
            self.drag_hud += "  snapped"
        return anchor + moved

    def _accumulate(self, delta: Any) -> Any:
        """Turn one ``update`` return into a quantity measured from the press.

        The two gizmos that need it need it for opposite reasons and both were
        wrong the same way. ``RotateGizmo.update`` documents its return as the
        *increment since the last update* -- composing that against the
        press-time rotation keeps only the last mouse-move, so a 90-degree
        sweep landed at whatever the final frame happened to travel. It is
        summed here instead, which is exact rather than approximate: every
        increment of one drag is about the same local axis, so the products
        commute and the total is the angle the cursor actually swept.

        ``TranslateGizmo.update`` is the mirror image -- it returns the
        gizmo's *new world position*, which is only an object's new position
        for an object whose origin happened to sit under the gizmo. Measured
        against the press-time gizmo origin it becomes a displacement, which
        is what every selected object can be moved by.

        ``ScaleGizmo`` already returns a factor relative to the drag's start
        and passes straight through.
        """
        if isinstance(delta, np.ndarray) and delta.shape == (4,):
            self._drag_quat = m3.quat_normalize(m3.quat_mul(delta, self._drag_quat))
            return self._drag_quat
        return delta

    def _apply(self, obj: Any, was: Any, delta: Any, state: Any) -> None:
        from .clay import ops

        # See ``_element_world_transform``: the grid stands down for a vertex
        # snap or a typed value, both of which are more specific answers to the
        # same question.
        snap = bool(getattr(state, "snap", False)) and not (
            self._snap_point is not None or self.drag_input.active
        )
        if isinstance(delta, np.ndarray) and delta.shape == (3,) and self._is_scale(state):
            obj.scale = np.array(was[2], dtype="f8") * delta
        elif isinstance(delta, np.ndarray) and delta.shape == (4,):
            obj.rotation = m3.quat_normalize(m3.quat_mul(delta, np.array(was[1], dtype="f8")))
            if snap:
                obj.rotation = ops.snap_rotation(obj.rotation, state.snap_rotate)
        else:
            moved = np.asarray(delta, dtype="f8").reshape(3) - self._drag_origin
            obj.translation = np.array(was[0], dtype="f8") + moved
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


def _rotation_hud(quat: Any, entry: Any) -> str:
    """The angle a rotation drag currently amounts to, in degrees.

    Read off the *constrained* quaternion rather than the raw one, so a locked
    axis reads as the angle about that axis and a typed value reads back as
    itself -- which is the whole reason the HUD is worth drawing.
    """
    from .clay import drag as bdrag

    out = bdrag.constrain_rotation(quat, entry)
    length = float(np.linalg.norm(np.asarray(out, dtype="f8")[:3]))
    angle = np.degrees(2.0 * float(np.arctan2(length, float(out[3]))))
    return bdrag.readout("rotate", np.array([angle]), entry)


def _toward_eye(eye: Any) -> np.ndarray:
    """A world matrix that shrinks everything a hair toward the eye.

    Which pulls a selected face's translucent fill in front of the face it
    covers. ``glPolygonOffset`` is the textbook answer and is deliberately not
    used: it is global GL state that moderngl caches, so setting it here would
    leak into the gizmo pass and the grid, and unsetting it is one more thing to
    get wrong on an early return. Scaling about the eye is local to the draw and
    needs no cleanup at all.
    """
    scale = 1.0 - FILL_BIAS
    matrix = m3.identity()
    matrix[:3, :3] = np.eye(3) * scale
    matrix[:3, 3] = np.asarray(eye, dtype="f8") * FILL_BIAS
    return matrix


def _face_outline(mesh: Any, faces: Any) -> np.ndarray:
    """The border of each face, as ``LINES`` pairs. Empty for no faces."""
    faces = np.asarray(faces, dtype="i8").reshape(-1)
    if len(faces) == 0:
        return np.zeros((0, 2), dtype="u4")
    starts = mesh.starts.astype("i8")
    counts = starts[faces + 1] - starts[faces]
    offsets, nxt, _ = _flat_next(counts)
    corners = _corner_spans(starts, faces)
    return np.stack([mesh.loops[corners], mesh.loops[corners[nxt]]], axis=1)
