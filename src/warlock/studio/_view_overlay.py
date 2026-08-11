"""The element-mode overlay: guides, selection, hover and the face fill.

Split out of :mod:`~warlock.studio.clay_view` as pure code motion. The overlay
is its own small GL cache -- one position buffer and many index buffers per
object, keyed on ``(id(mesh), id(sel), mode, hover)`` -- and it is the reason
``DrawItem`` carries a ``depth`` flag: guides are depth-tested so the far side
of a closed mesh does not fog the near one, and the *selection* is not, so a
vertex you have orbited behind the surface still says it is selected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import moderngl
import numpy as np

from .clay.topo import corner_spans as _corner_spans
from .clay.topo import flat_next as _flat_next
from .viewer import math3d as m3
from .viewer.render import DrawItem

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .clay_view import ClayView


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


class OverlayOps:
    """``ClayView``'s element overlay. See the module docstring."""

    # -- the element-selection overlay -------------------------------------

    def _element_overlays(self: ClayView, doc: Any) -> list[Any]:
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
        self: ClayView, obj: Any, overlay: Any, world: Any, mode: str, sel: Any, hover: int
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
        self: ClayView, obj: Any, overlay: Any, mode: str, sel: Any, hover: int
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

    def _release_overlays(self: ClayView) -> None:
        for overlay in self._overlays.values():
            overlay.release()
        self._overlays.clear()
