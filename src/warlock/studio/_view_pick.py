"""What the cursor is over: object picking, projection and element picking.

Split out of :mod:`~warlock.studio.clay_view` as pure code motion. :class:`Hit`
lives here rather than in the viewport module because these methods construct it
at runtime, and a mixin may not import ``clay_view`` outside ``TYPE_CHECKING``;
``clay_view`` re-exports the name, so every existing importer is unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .viewer import picking

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .clay_view import ClayView


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


class PickOps:
    """``ClayView``'s picking. See the module docstring."""

    # -- picking -----------------------------------------------------------

    def pick_face(self: ClayView, doc: Any, local: tuple[float, float]) -> Hit | None:
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
        from .clay.adjacency import cached_positions_f8, cached_triangulation

        origin, direction = self._ray(local)
        best: Hit | None = None
        for obj in doc.objects:
            if not obj.visible:
                continue
            tris, tri_face = cached_triangulation(obj.mesh)
            positions = cached_positions_f8(obj.mesh)
            hit = picking.ray_object(
                origin,
                direction,
                self._world(obj),
                positions,
                tris,
                # Positions and tree both come from the same frozen mesh, so
                # they cannot disagree about the geometry -- which is the whole
                # precondition the narrowed sweep rests on.
                bvh=picking.cached_bvh(obj.mesh, positions, tris),
            )
            if hit is not None and (best is None or hit[0] < best.t):
                face = int(tri_face[hit[1]]) if len(tri_face) else -1
                best = Hit(uid=obj.uid, t=float(hit[0]), face=face)
        return best

    def pick(self: ClayView, doc: Any, local: tuple[float, float]) -> int | None:
        """Which object a click lands on. -> its uid, or None.

        Kept as the object-mode entry point, and kept returning a bare uid:
        that is what selection in object mode is, and widening it would make
        every caller unpack a record to ignore two thirds of it.
        """
        hit = self.pick_face(doc, local)
        return None if hit is None else hit.uid

    def screen_of(self: ClayView, doc: Any, uid: int) -> Any:
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
            # The projection *kind* moves every projected point without moving
            # the camera, so leaving it out let a Ctrl+5 toggle serve stale
            # positions to pick, hover and the marquee until the camera moved.
            bool(self.camera.orthographic),
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
        # The mesh rides along as the ``_view_cache._Entry`` pin does: its id is
        # in the key, and an id is only sound while the object it named is
        # alive -- a freed mesh's address coming back on new geometry would
        # otherwise match a stale projection.
        self._screens[uid] = (key, screen, obj.mesh)
        return screen

    def pick_element(
        self: ClayView, doc: Any, local: tuple[float, float], hit: Any = None
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
        # X-ray is see-through for the pick as well as the draw: with no
        # surface depth the nearest element wins wherever it sits, which is
        # what ``clay_state.xray`` and the Clay chapter say it does.
        depth = None if hit is None or getattr(self, "xray", False) else hit.t

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
            if mode == "vertex":
                key = float(screen.depth[index])
            elif mode == "edge":
                # The mean of the endpoints' depths -- the same reading
                # ``nearest_edge`` compared against the surface. This was a
                # constant 0.0, so with two objects' edges under the cursor
                # the earlier one in ``doc.objects`` always won regardless of
                # which edge was nearer the camera.
                a, b = adjacency(obj.mesh).edge_verts[index]
                key = 0.5 * (float(screen.depth[a]) + float(screen.depth[b]))
            else:
                # A face hit names one object already; nothing to rank.
                key = 0.0
            if best is None or key < best[0]:
                best = (key, obj.uid, int(index))
        return None if best is None else (best[1], best[2])

    def element_sel_for(self: ClayView, doc: Any, uid: int, index: int) -> Any:
        """One picked element as an :class:`~.clay.elements.ElementSel`."""
        from .clay import elements as el
        from .clay.adjacency import adjacency

        mode = doc.element_mode
        if mode == "vertex":
            return el.ElementSel(verts=[index])
        if mode == "edge":
            return el.ElementSel(edges=[adjacency(doc.by_uid(uid).mesh).edge_verts[index]])
        return el.ElementSel(faces=[index])
