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


#: How far the pointer may travel between an Alt press and its release and
#: still count as a click rather than an orbit, in pixels (Manhattan).
#:
#: Four, and the number is doing real work: a mouse moves a pixel or two under
#: the click of a finger, so zero makes the gesture unreliable, and a generous
#: threshold eats the beginning of a deliberate orbit.
ALT_CLICK_SLOP = 4.0


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
            return self._release_drag(doc, event.button)
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
        # A keyboard drag has no button held, so a press is how it *ends*: the
        # left button commits it and the right cancels, which is Blender's
        # arrangement and the one a modeller's hand already knows.
        if self._grab == "keydrag":
            if button == 1:
                self._release_drag(doc, button=1)
            else:
                self.cancel_drag(doc)
            return True
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
            #
            # Alt+*click* is a loop select, and the two have to share the
            # button because both are Blender's. They are told apart on the
            # release by how far the pointer moved -- see ``_alt_click``:
            # anything that is a drag orbits, and only a press and a release in
            # the same place selects. Recorded here rather than decided here,
            # because at press time there is nothing yet to tell them apart.
            self._grab = "orbit"
            self._alt_at = local
            self._alt_ctrl = ctrl
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

    def begin_keyboard_drag(self: ClayView, doc: Any, kind: str) -> bool:
        """``G``/``R``/``S``: start a transform with no handle grabbed.

        -> whether one started. The modeller's gesture, and the one Clay had no
        route to at all: every transform went through grabbing a coloured arrow,
        which means finding it, which means the object is never moved without
        first looking at the gizmo rather than at the model.

        Measured on the plane through the pivot perpendicular to the camera,
        which is what "move it where the mouse goes" means when no axis has been
        chosen -- and the same plane all three kinds measure on, so ``X`` mid-
        drag narrows a translation, a rotation and a scale by the one rule in
        :meth:`_narrow` rather than three.

        It reuses ``_begin_gizmo_drag`` outright: the snapshot, the pivot and
        the fresh ``DragInput`` are the same three things a handle drag needs,
        and a second copy of them is a second place for a cancel to restore the
        wrong values from.
        """

        if self._grab is not None or not doc.selection:
            return False
        if doc.element_mode != "object" and not doc.element_sel:
            return False
        centre = self.selection_centre(doc)
        if centre is None:
            return False
        self._begin_gizmo_drag(doc)
        # ``_begin_gizmo_drag`` reads the *gizmo's* origin, and Select draws no
        # gizmo -- so the pivot is taken from the selection directly, which is
        # the same point the gizmo would have been placed at.
        self._drag_origin = np.asarray(centre, dtype="f8")
        anchor = self._view_plane_point(self._last_mouse, self._drag_origin)
        if anchor is None:
            self._grab = None
            return False
        self._grab = "keydrag"
        self._key_kind = kind
        self._key_anchor = anchor
        return True

    def _view_plane_point(
        self: ClayView, local: tuple[float, float], centre: Any
    ) -> Any:
        """Where the pointer's ray meets the plane through ``centre`` facing the
        camera. ``None`` when the ray runs parallel to it, which the caller
        treats as "no movement" rather than as an error."""
        from .viewer import picking

        origin, direction = self._ray(local)
        normal = np.asarray(self.camera.position, dtype="f8") - np.asarray(
            centre, dtype="f8"
        )
        length = float(np.linalg.norm(normal))
        if length < 1e-9:
            return None
        return picking.ray_plane(origin, direction, np.asarray(centre), normal / length)

    def _drag_keyboard(self: ClayView, doc: Any, local: tuple[float, float]) -> None:
        """One frame of a keyboard drag: the same delta shapes a gizmo makes.

        Deliberately the same three shapes -- a 3-vector for a move, a 4-vector
        quaternion for a rotate, a 3-vector of factors for a scale -- so
        everything downstream (``_narrow``, ``_accumulate``, ``_apply``, the
        element path) is the code that already exists rather than a parallel
        set that has to be kept agreeing with it.
        """
        from .clay import drag as bdrag

        point = self._view_plane_point(local, self._drag_origin)
        if point is None or self._key_anchor is None:
            return
        pivot = np.asarray(self._drag_origin, dtype="f8")
        anchor = np.asarray(self._key_anchor, dtype="f8")
        state = self.state
        if self._key_kind == "rotate":
            forward = np.asarray(self.camera.position, dtype="f8") - pivot
            norm = float(np.linalg.norm(forward))
            if norm < 1e-9:
                return
            angle = bdrag.screen_angle(pivot, forward / norm, anchor, point)
            delta = m3.quat_from_axis_angle(forward / norm, angle)
        elif self._key_kind == "scale":
            was = float(np.linalg.norm(anchor - pivot))
            now = float(np.linalg.norm(point - pivot))
            if was < 1e-9:
                return
            factor = now / was
            delta = np.array([factor, factor, factor], dtype="f8")
        else:
            # A translation is reported as the *destination*, not the
            # displacement: that is what the translate gizmo hands back and
            # what ``_apply`` subtracts ``_drag_origin`` from.
            delta = pivot + (point - anchor)
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

    def _release_drag(self: ClayView, doc: Any, button: int = 1) -> bool:
        if self._grab is None:
            return False
        # The press half's guard, mirrored: a grab is owned by the button that
        # began it -- pan by the middle button, everything else by the left --
        # and only that button's release ends it. Without this an MMB release
        # or a wheel tick (buttons 4/5) mid-LMB-drag committed the gizmo drag
        # early, and the real LMB-up then found no grab to commit.
        owner = 2 if self._grab == "pan" else 1
        if button != owner:
            return True
        was, self._grab = self._grab, None
        if was == "orbit" and self._alt_click(doc):
            return True
        if was in ("gizmo", "keydrag"):
            # A keyboard drag holds no handle, so there is no gizmo drag to end
            # -- but everything after that is identical, which is the whole
            # point of routing it through the same delta shapes.
            gizmo = self.active_gizmo(doc) if was == "gizmo" else None
            if gizmo is not None:
                gizmo.end_drag()
            if doc.element_mode != "object":
                self._commit_element_drag(doc)
            else:
                self._commit_drag(doc)
            self._clear_drag_input()
            self._end_keyboard_drag()
        elif was == "marquee":
            self._commit_marquee(doc)
        return True

    def _alt_click(self: ClayView, doc: Any) -> bool:
        """An Alt+click that never became a drag: select the loop under it.

        -> whether it selected something, which the caller takes as "this
        release is spoken for".

        **Four pixels** is the whole of what separates this from an orbit, and
        the number is doing real work: a mouse moves a pixel or two under the
        click of a finger, so zero would make the gesture unreliable, and a
        generous threshold would eat the beginning of a deliberate orbit. It is
        measured against the *press*, so a long orbit that happens to return to
        where it started still orbits -- the pointer having travelled is what
        makes it a drag, not where it ended up.

        Ctrl picks the ring rather than the loop, which is Blender's
        Ctrl+Alt+click and is why the modifier is captured at the press: by the
        release the user may have let go of it.
        """
        at, self._alt_at = self._alt_at, None
        ctrl, self._alt_ctrl = self._alt_ctrl, False
        if at is None or doc.element_mode == "object":
            return False
        moved = abs(self._last_mouse[0] - at[0]) + abs(self._last_mouse[1] - at[1])
        if moved > ALT_CLICK_SLOP:
            return False
        return self.select_loop_at(doc, at, ring=ctrl)

    def select_loop_at(
        self: ClayView, doc: Any, local: tuple[float, float], *, ring: bool = False
    ) -> bool:
        """Select the edge loop -- or ring -- through the element under ``local``.

        Public because the manual names the gesture and a test presses it; the
        pane never calls it.

        Works from whatever the mode is picking: an edge directly, and a vertex
        or a face through one of its edges, because "the loop through this" is a
        sentence a user means in every element mode and refusing two of the
        three would be an offer taken back.
        """
        from .clay import select as bsel

        found = self.pick_element(doc, local)
        if found is None:
            return False
        uid, index = found
        try:
            obj = doc.by_uid(uid)
        except KeyError:
            return False
        edge = self._edge_under(doc, obj.mesh, index)
        if edge is None:
            return False
        pairs = (
            bsel.edge_ring(obj.mesh, edge)
            if ring
            else bsel.edge_loop(obj.mesh, edge)
        )
        if not len(pairs):
            return False
        from .clay import elements as el

        doc.set_element_sel(uid, el.ElementSel(edges=pairs))
        doc.select([uid])
        return True

    def _edge_under(self: ClayView, doc: Any, mesh: Any, index: int) -> Any:
        """One edge of whatever the pick returned, as a vertex pair.

        In edge mode the pick *is* an edge. In the other two it is a vertex or a
        face, and any edge touching it is a seed the loop can run from -- which
        is an arbitrary choice among a few, and is the honest one: a loop
        through a vertex is not a single answer, and picking the first is what
        Blender does from a face too.
        """
        from .clay.adjacency import adjacency

        a = adjacency(mesh)
        mode = doc.element_mode
        if mode == "edge":
            if not (0 <= index < a.n_edges):
                return None
            return tuple(int(v) for v in a.edge_verts[index])
        if mode == "vertex":
            corners = a.vertex_corners(int(index))
            if not len(corners):
                return None
            return tuple(int(v) for v in a.edge_verts[int(a.corner_edge[corners[0]])])
        starts = mesh.starts
        if not (0 <= index < len(starts) - 1):
            return None
        return tuple(int(v) for v in a.edge_verts[int(a.corner_edge[int(starts[index])])])

    # -- the keyboard's half of a drag (Clay18) ------------------------------

    @property
    def dragging(self: ClayView) -> bool:
        """Whether a transform drag is live. What the key handler checks *first*.

        First because the number row is bound to the element modes: a ``1``
        typed into a drag has to be a digit, not a jump into vertex mode
        halfway through moving something.

        Both kinds count. A gizmo drag is a handle held with the mouse down; a
        **keyboard drag** is ``G``/``R``/``S`` with the mouse merely moving, and
        every rule that applies during one applies during the other -- the axis
        lock, the typed value, Esc to cancel. They differ in how they end, and
        nowhere else: a gizmo drag ends when its button comes up, and a keyboard
        drag has no button held, so it ends on a *press*.
        """
        return self._grab in ("gizmo", "keydrag")

    def _clear_drag_input(self: ClayView) -> None:
        from .clay import drag as bdrag

        self.drag_input = bdrag.DragInput()
        self.drag_hud = ""
        self._snap_point = None

    def _end_keyboard_drag(self: ClayView) -> None:
        self._key_kind = ""
        self._key_anchor = None

    def drag_key(self: ClayView, doc: Any, name: str) -> bool:
        """Feed one key name to the live drag. -> whether it was consumed.

        The transform is re-applied immediately rather than waiting for the next
        mouse move, because a typed value with the mouse held still is the whole
        point of typing one. Re-applying is safe because every gizmo's
        ``update`` is a function of the ray it is handed and the press it began
        from: ``RotateGizmo`` returns a zero increment for an unchanged ray,
        and the other two are measured from the press outright.
        """
        if not self.dragging:
            return False
        # ``G``/``R``/``S`` mid-drag switch which transform is running, which is
        # Blender's and is what makes "move it, no -- rotate it" one gesture
        # rather than a cancel and a restart. Only during a keyboard drag: a
        # handle drag is holding a specific arrow, and switching under it would
        # leave the gizmo's own drag state describing a transform nobody is
        # doing any more.
        switch = {"g": "move", "r": "rotate", "s": "scale"}.get(name)
        if switch is not None and self._grab == "keydrag":
            self._restart_keyboard_drag(doc, switch)
            return True
        if not self.drag_input.key(name):
            return False
        if self._grab == "keydrag":
            self._drag_keyboard(doc, self._last_mouse)
        else:
            self._drag_gizmo(doc, self._last_mouse)
        return True

    def _restart_keyboard_drag(self: ClayView, doc: Any, kind: str) -> None:
        """Switch a live keyboard drag to another transform.

        The objects are put back first, so the new transform is measured from
        where they *started* rather than from wherever the abandoned one left
        them -- a rotate that began from a half-finished move would carry that
        move into its result and there would be no way to undo one without the
        other.
        """
        if self._key_kind == kind:
            return
        for uid, was in self._drag_start.items():
            try:
                obj = doc.by_uid(uid)
            except KeyError:
                continue
            obj.translation, obj.rotation, obj.scale = (
                np.array(v, copy=True) for v in was
            )
        # And the overlays, for ``cancel_drag``'s reason: the objects have been
        # put back, and an overlay still holding the abandoned transform's
        # positions is the new transform measured against a picture of the old
        # one.
        self._restore_overlays(doc, list(self._drag_start))
        self._key_kind = kind
        self._key_anchor = self._view_plane_point(self._last_mouse, self._drag_origin)
        self._clear_drag_input()
        doc.touch()

    def _restore_overlays(self: ClayView, doc: Any, uids: Any) -> None:
        """Put the selection overlays back on the mesh's own positions.

        The preview writes moved vertices straight into each overlay's VBO
        (``_preview_element_drag``), and the overlay is keyed on ``id(mesh)`` --
        which does not change during a drag -- so an abandoned drag left the
        wireframe, the vertex dots and the edge lines at the previewed
        positions while the *mesh* was back where it started. It looked like
        the cancel had half worked, and stayed that way until something else
        rebuilt the overlay.
        """
        overlays = getattr(self, "_overlays", {})
        for uid in uids:
            overlay = overlays.get(uid)
            if overlay is None:
                continue
            try:
                obj = doc.by_uid(uid)
            except KeyError:
                continue
            overlay.write_positions(obj.mesh.positions)

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
            self._restore_overlays(doc, drags)
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
        self._end_keyboard_drag()
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

        object_mode = doc.element_mode == "object"
        best: tuple[float, np.ndarray] | None = None
        for obj in doc.objects:
            if not obj.visible:
                continue
            # In object mode the whole object rides the drag, so every one of
            # its vertices -- reprojected at the live transform -- would track
            # the cursor exactly and report a snap: the same self-snap the
            # ``allowed`` mask below exists to prevent on the element path.
            if object_mode and obj.uid in self._drag_start:
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
        elif self._grab == "keydrag":
            self._drag_keyboard(doc, local)
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
        # ``drag.verts`` is exactly the set written into ``positions`` above,
        # which is what makes the incremental normals safe; with no element drag
        # in hand the mover is a gizmo over the whole object and there is no
        # smaller set to name, so nothing is passed and every face is recomputed.
        prims = bd.preview_primitives(
            base, positions, doc.materials, moved=None if drag is None else drag.verts
        )
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
        """One object's transform under the live drag.

        **A rotate and a scale orbit the pivot, not the object's own origin**,
        and that is a fix rather than a feature: the gizmo is drawn at
        ``selection_centre`` -- the median of what is selected -- while this
        wrote only ``obj.rotation``, so dragging the ring with two objects
        selected span each of them in place around a ring drawn somewhere
        neither of them was. The picture said "these turn about here" and the
        document did something else.

        It changes what a *single* off-centre object does too, and that is the
        same correction rather than a side effect: the ring is drawn at the
        bounding box's centre, so an object whose origin is not there has always
        been rotating about a point other than the one on screen. Blender's
        Median Point pivot behaves exactly as this now does.

        ``_drag_origin`` is that pivot -- the gizmo's own origin, captured at
        the press -- which the translation arm has always subtracted and the
        other two never read.
        """
        from .clay import ops

        # See ``_element_world_transform``: the grid stands down for a vertex
        # snap or a typed value, both of which are more specific answers to the
        # same question.
        snap = bool(getattr(state, "snap", False)) and not (
            self._snap_point is not None or self.drag_input.active
        )
        pivot = np.asarray(self._drag_origin, dtype="f8")
        before = np.array(was[0], dtype="f8")
        if isinstance(delta, np.ndarray) and delta.shape == (3,) and self._is_scale(state):
            obj.scale = np.array(was[2], dtype="f8") * delta
            obj.translation = pivot + (before - pivot) * np.asarray(delta, dtype="f8")
        elif isinstance(delta, np.ndarray) and delta.shape == (4,):
            # The *delta* angle is what snaps, exactly as the element path does
            # in ``_element_world_transform`` and for ``ops.snap_rotation``'s
            # own stated reason: "rotate this by fifteen degrees about the ring
            # I grabbed" leaves an already-placed object where it was put,
            # where quantising the absolute orientation visibly swings it the
            # moment the drag starts.
            if snap:
                delta = ops.snap_rotation(delta, state.snap_rotate)
            obj.rotation = m3.quat_normalize(m3.quat_mul(delta, np.array(was[1], dtype="f8")))
            obj.translation = pivot + m3.quat_rotate(delta, before - pivot)
        else:
            moved = np.asarray(delta, dtype="f8").reshape(3) - self._drag_origin
            obj.translation = np.array(was[0], dtype="f8") + moved
            if snap:
                obj.translation = ops.snap_translation(obj.translation, state.snap_translate)

    def _is_scale(self: ClayView, state: Any) -> bool:
        return getattr(state, "tool", "") == "scale"
