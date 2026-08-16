"""The mouse and the keyboard: every gesture the Clay viewport understands.

Split out of :mod:`~warlock.studio.clay_view` as pure code motion. The mouse map
and the modifier rule are stated in that module's docstring; what lives here is
their implementation, plus the two halves of a live drag -- the object path,
which moves transforms in place and commits one history step per object on
release, and the element path, which previews by writing vertex buffers on the
GPU and never touches the document until the release.

:meth:`DragOps._narrow` is deliberately the **single** narrowing site for axis
locks, typed values and vertex snapping, above both paths -- an invariant named
in ``docs/INVARIANTS.md`` as ``ClayView._narrow``, which it still is: the class
that carries this mixin is ``ClayView``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .clay import document as bd
from .viewer import math3d as m3

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .clay_view import ClayView


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


class DragOps:
    """``ClayView``'s input handling and drags. See the module docstring."""

    # -- input -------------------------------------------------------------

    def handle_event(self: ClayView, doc: Any, event: Any, hovered: bool) -> bool:
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

    def _local(self: ClayView, event: Any) -> tuple[float, float]:
        pos = getattr(event, "pos", None)
        if pos is None:
            return self._last_mouse
        return (pos[0] - self._rect[0], pos[1] - self._rect[1])

    def _mods(self: ClayView) -> tuple[bool, bool, bool]:
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

    def _press(self: ClayView, doc: Any, button: int, local: tuple[float, float]) -> bool:
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

    def _begin_gizmo_drag(self: ClayView, doc: Any) -> None:
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
        self: ClayView, doc: Any, local: tuple[float, float], *, shift: bool, ctrl: bool
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

    def _rmb_release(self: ClayView, local: tuple[float, float]) -> bool:
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

    def _release_drag(self: ClayView, doc: Any) -> bool:
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
    def dragging(self: ClayView) -> bool:
        """Whether a gizmo drag is live. What the key handler checks *first*.

        First because the number row is bound to the element modes: a ``1``
        typed into a drag has to be a digit, not a jump into vertex mode
        halfway through moving something.
        """
        return self._grab == "gizmo"

    def _clear_drag_input(self: ClayView) -> None:
        from .clay import drag as bdrag

        self.drag_input = bdrag.DragInput()
        self.drag_hud = ""
        self._snap_point = None

    def drag_key(self: ClayView, doc: Any, name: str) -> bool:
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

    def cancel_drag(self: ClayView, doc: Any) -> bool:
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

    def _snap_vertex(self: ClayView, doc: Any, local: tuple[float, float]) -> np.ndarray | None:
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

    def _commit_marquee(self: ClayView, doc: Any) -> None:
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

    def _commit_drag(self: ClayView, doc: Any) -> None:
        """One history step per object, recorded against where the drag began."""
        for uid, was in self._drag_start.items():
            try:
                doc.set_transform(uid, was=was)
            except KeyError:
                continue  # deleted mid-drag
        self._drag_uids = []
        self._drag_start = {}

    def _motion(self: ClayView, doc: Any, local: tuple[float, float]) -> bool:
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

    def _begin_element_drag(self: ClayView, doc: Any) -> None:
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

    def _element_world_transform(self: ClayView, delta: Any, state: Any) -> Any:
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

    def _preview_element_drag(self: ClayView, doc: Any, delta: Any, state: Any) -> None:
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

    def _preview_positions(self: ClayView, doc: Any, uid: int, positions: Any) -> None:
        """Rewrite one object's vertex buffers in place. No VAO is rebuilt.

        Through ``preview_primitives`` rather than ``to_primitives``: the mesh
        the drag began on is fixed for the drag's whole duration, so the
        material grouping, the corner gathers and the index buffer are built
        once and only the positions and normals are recomputed per frame. A
        200k-triangle import went 368 ms a frame to 92 -- see
        ``docs/measurements/2026-08-16-interactive-defects.md``.
        """
        entry = self._cache.get(uid)
        if entry is None:
            return
        obj = doc.by_uid(uid)
        drag = self._element_drags.get(uid)
        base = obj.mesh if drag is None else drag.before
        prims = bd.preview_primitives(base, positions, doc.materials)
        for (_node, gpu), primitive in zip(entry.gpu.draws, prims, strict=False):
            try:
                gpu.update_vertices(primitive)
            except ValueError:  # pragma: no cover - topology changed under a drag
                return
        overlay = getattr(self, "_overlays", {}).get(uid)
        if overlay is not None:
            overlay.write_positions(positions)

    def _commit_element_drag(self: ClayView, doc: Any) -> None:
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

    def _drag_gizmo(self: ClayView, doc: Any, local: tuple[float, float]) -> None:
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

    def _narrow(
        self: ClayView, doc: Any, delta: Any, state: Any, local: tuple[float, float]
    ) -> Any:
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

    def _accumulate(self: ClayView, delta: Any) -> Any:
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

    def _apply(self: ClayView, obj: Any, was: Any, delta: Any, state: Any) -> None:
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

    def _is_scale(self: ClayView, state: Any) -> bool:
        return getattr(state, "tool", "") == "scale"
