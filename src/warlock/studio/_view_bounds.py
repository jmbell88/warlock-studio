"""Where the selection is: centres, world boxes and framing.

Split out of :mod:`~warlock.studio.clay_view` as pure code motion. Everything
here is memoized on the identity of what it reads -- the meshes, the selections
and the three transform arrays -- which is sound for the same reason the GPU
cache's key is: each of those is replaced whole rather than mutated, so a memo
misses exactly when the answer can change. These are asked every frame a gizmo
is on screen.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .clay_view import ClayView


class BoundsOps:
    """``ClayView``'s centres, bounds and framing. See the module docstring."""

    # -- selection and framing ---------------------------------------------

    def selection_centre(self: ClayView, doc: Any) -> np.ndarray | None:
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

    def element_centre(self: ClayView, doc: Any) -> np.ndarray | None:
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
        # Everything whose id() the key carries, held so those ids cannot be
        # recycled while the memo lives -- the ``_view_cache._Entry`` pin. A
        # freed selection or mesh whose address came back on a new one would
        # otherwise match a stale centre.
        pins: list[Any] = [doc]
        for uid, sel in doc.element_sel.items():
            pins.append(sel)
            try:
                obj = doc.by_uid(uid)
            except KeyError:
                continue
            pins.extend((obj.mesh, obj.translation, obj.rotation, obj.scale))
            verts = el.affected_verts(obj.mesh, sel)
            if not len(verts):
                continue
            matrix = self._world(obj)
            local = obj.mesh.positions[verts].astype("f8")
            world = (matrix @ np.hstack([local, np.ones((len(local), 1))]).T).T[:, :3]
            total += world.sum(axis=0)
            count += len(world)
        centre = None if count == 0 else total / count
        self._centre_memo = (key, centre, tuple(pins))
        return centre

    @staticmethod
    def _obj_key(doc: Any, uid: int) -> tuple:
        """The identity triple-plus-mesh that pins one object's geometry."""
        try:
            obj = doc.by_uid(uid)
        except KeyError:
            return ()
        return (id(obj.mesh), id(obj.translation), id(obj.rotation), id(obj.scale))

    def world_bounds(self: ClayView, doc: Any, *, selected_only: bool = False) -> tuple[Any, Any]:
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
        # The pin: see ``element_centre``. Every object whose ids the key names
        # is held alive for as long as the memo can match on them.
        pins: list[Any] = [doc]
        for obj in doc.objects:
            pins.extend((obj.mesh, obj.translation, obj.rotation, obj.scale))
            if not obj.visible or (selected_only and obj.uid not in doc.selection):
                continue
            box = self._object_world_box(obj)
            if box is None:
                continue
            lo = np.minimum(lo, box[0])
            hi = np.maximum(hi, box[1])
            found = True
        out = (lo, hi) if found else (None, None)
        self._bounds_memo[selected_only] = (key, out, tuple(pins))
        return out

    def _object_world_box(self: ClayView, obj: Any) -> tuple[Any, Any] | None:
        """``ops.world_box``, and deliberately nothing else.

        It used to be a second copy of the same eight corners, which is how the
        properties panel's dimensions row and the camera's framing would have
        come to disagree about the size of one object.
        """
        from .clay import ops as bops

        return bops.world_box(obj)

    def frame_selection(self: ClayView, doc: Any) -> float:
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
