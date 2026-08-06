"""The Viewer: everything the UI is allowed to know about the 3D pane.

One object, one contract. The panes never touch a framebuffer, a camera or a
GLB -- they call ``load_model``, ``enter_pose_mode``, ``get_pose`` and so on,
and get back plain Python. Which is what lets the renderer be tested against a
standalone context with no window, and the panes be tested with no GL at all.

Input arrives as pygame events, but only when the viewport image is hovered:
imgui owns the mouse otherwise, and a drag that started on the image keeps
ownership until the button comes up so a rotate does not stop halfway when the
pointer crosses onto a panel.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .viewer import capture, glctx, gltf, picking
from .viewer import markers as markerslib
from .viewer import math3d as m3
from .viewer import scene as scenelib
from .viewer import sheet as sheetlib
from .viewer.camera import Camera, screen_ray
from .viewer.gizmo import RotateGizmo, TranslateGizmo
from .viewer.pose import PoseEditor
from .viewer.render import Renderer

log = logging.getLogger(__name__)


class Viewer:
    """The 3D pane, from the UI's point of view."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self.renderer = Renderer(ctx)
        self.viewport = glctx.Viewport(ctx, (16, 16))
        self.compare_viewport: glctx.Viewport | None = None
        self.camera = Camera()
        self.compare_camera = Camera()

        self.model: gltf.Model | None = None
        self.gpu: scenelib.GpuModel | None = None
        self.placement = m3.identity()
        self.radius = 1.0
        self.path: Path | None = None

        self.compare_model: gltf.Model | None = None
        self.compare_gpu: scenelib.GpuModel | None = None
        self.compare_placement = m3.identity()

        self.wireframe = False
        self.pose_mode = False
        # Which job's rig the editor is bound to, so a write can never land on
        # whichever asset happens to be selected. See enter_pose_mode.
        self.pose_job_id: str | None = None
        self.editor = PoseEditor()
        self.markers = markerslib.JointMarkers(ctx, self.renderer.programs)
        self.rotate_gizmo = RotateGizmo(ctx, self.renderer.programs)
        self.translate_gizmo = TranslateGizmo(ctx, self.renderer.programs)
        self.on_pose_dirty: Any = None

        # Set every frame by the pane so input can be mapped into the image.
        self._rect = (0.0, 0.0, 1.0, 1.0)
        self._grab: str | None = None  # "orbit" | "pan" | "gizmo" | "marker"
        self._last_mouse = (0.0, 0.0)
        # A reference image is shown as a plain texture rather than as geometry.
        self.reference: Any = None
        # The GLB a parse has been dispatched for but not yet adopted. Held so
        # the timer that dispatched it does not dispatch it again on every tick
        # until the result lands, and so nothing else decides the viewer is
        # "showing the wrong thing" while a load is in flight.
        self.pending: Path | None = None
        # The direction strip being rendered a cell at a time, if any.
        self._strip: Any = None

    # -- loading -----------------------------------------------------------

    @staticmethod
    def parse_model(path: Path) -> Any:
        """The blocking half of a load: parse the glTF and decode its textures.

        Touches no GL and no viewer state, so it is safe on a task thread --
        which is the whole point. It is the expensive half by a wide margin (a
        full accessor decode plus a PNG per texture slot), and it used to run
        on the frame thread at the exact moment a job finished, which is when
        the file is largest and coldest.
        """
        return gltf.load(path)

    def adopt_model(self, model: Any, path: Path) -> None:
        """The frame-thread half: upload, and take the parsed model as current.

        Must stay on the frame thread -- ``GpuModel`` creates buffers and
        textures on the one GL context, and so does releasing the old one.
        """
        gpu = scenelib.GpuModel(self.ctx, model)
        self._release_model()
        # Whatever parse was in flight is no longer wanted: this *is* the
        # viewport's content now. Without this, the blocking path -- entering
        # the pose editor, loading a Review unit -- left ``pending`` naming a
        # parse dispatched a moment earlier, which still matched when it landed
        # and was adopted over the top. In the pose case that discarded the
        # editor with unsaved rotations in it, bypassing ``pose_panel.guard``,
        # the confirm every other way out of the editor goes through.
        self.pending = None
        self.model, self.gpu, self.path = model, gpu, Path(path)
        self.placement = scenelib.placement(model)
        self.clear_reference()
        self.exit_pose_mode()
        self.frame()

    def load_model(self, path: Path) -> None:
        """Show a GLB, both halves at once. Blocking -- it decodes textures.

        Kept for the callers where the wait is the point: entering the pose
        editor and loading a Review unit both happen because the user just
        pressed something, and a hand-over there would flash an empty viewport
        for a frame instead. ``_sync_viewer`` uses the split above, because it
        fires on a *timer*, on the frame a job finishes.
        """
        self.adopt_model(self.parse_model(path), path)

    def clear(self) -> None:
        self._release_model()
        self.exit_pose_mode()
        self.model = self.gpu = self.path = None
        # See ``adopt_model``: an in-flight parse must not land on an emptied
        # viewport either.
        self.pending = None

    def _release_model(self) -> None:
        if self.gpu is not None:
            self.gpu.release()
        self.gpu = None

    def load_reference(self, path: Path) -> None:
        """Show a 2D reference image instead of a mesh."""
        from PIL import Image

        self.clear_reference()
        with Image.open(path) as im:
            rgba = im.convert("RGBA")
            texture = self.ctx.texture((rgba.width, rgba.height), 4, rgba.tobytes())
        texture.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
        self.reference = texture

    def clear_reference(self) -> None:
        if self.reference is not None:
            # Registered via widgets.texture_ref in main._draw_reference, so
            # the backend must drop the name before the driver can reuse it.
            self._forget(self.reference)
            self.reference.release()
            self.reference = None

    @property
    def has_model(self) -> bool:
        return self.gpu is not None

    def stats(self) -> dict[str, Any]:
        """Counted off the loaded scene, not read from mesh_report: the report
        describes model.glb and the viewer may be showing rig.glb or a pose."""
        if self.model is None:
            return {}
        lo, hi = self.model.bounds()
        size = (hi - lo) * np.array(
            [self.placement[0, 0], self.placement[1, 1], self.placement[2, 2]]
        )
        return {
            "triangles": self.model.triangle_count,
            "vertices": sum(
                len(p.positions) for prims in self.model.meshes for p in prims
            ),
            "size": tuple(float(v) for v in size),
        }

    # -- camera ------------------------------------------------------------

    def frame(self) -> float:
        if self.model is None:
            return 0.0
        lo, hi = self._world_bounds()
        self.radius = self.camera.frame(lo, hi)
        self.renderer.fit_grid(lo, hi)
        return self.radius

    def _world_bounds(self):
        lo, hi = self.model.bounds()
        corners = np.array(
            [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        )
        world = (self.placement @ np.hstack([corners, np.ones((8, 1))]).T).T[:, :3]
        return world.min(axis=0), world.max(axis=0)

    def set_wireframe(self, on: bool) -> None:
        self.wireframe = bool(on)

    def set_turntable(self, on: bool) -> None:
        # OrbitControls' own autoRotate, not a rotation of the model: rotating
        # the model would move the thing the markers are positioned against.
        self.camera.auto_rotate = bool(on)

    # -- compare -----------------------------------------------------------

    def compare(self, path: Path) -> None:
        model = gltf.load(path)
        gpu = scenelib.GpuModel(self.ctx, model)
        self.exit_compare()
        self.compare_model, self.compare_gpu = model, gpu
        # Placed the way the left-hand mesh is, or the two would not line up.
        self.compare_placement = scenelib.placement(model)
        if self.compare_viewport is None:
            self.compare_viewport = glctx.Viewport(self.ctx, (16, 16))

    def exit_compare(self) -> None:
        if self.compare_gpu is not None:
            self.compare_gpu.release()
        self.compare_gpu = self.compare_model = None

    @property
    def comparing(self) -> bool:
        return self.compare_gpu is not None

    # -- pose --------------------------------------------------------------

    def enter_pose_mode(self, rig: dict[str, Any] | None = None, job_id: str | None = None) -> bool:
        """Bind the editor to the loaded rig. ``job_id`` is *whose* rig it is.

        Recorded because the editor outlives the selection: ``_sync_viewer``
        deliberately does not reload while posing, so clicking another asset
        leaves this rig on screen with the inspector pointing somewhere else.
        Without a bound id the panel used the selected job for both, and Save
        wrote one asset's rotations into another's pose directory while joints
        mode re-skinned it against the wrong skeleton -- silently, because rig
        validation checks bone *names* and two jobs on one template share them.
        """
        if self.model is None or not self.model.skins:
            return False
        self.editor.bind(self.model)
        self.editor.mirror_pairs = list((rig or {}).get("mirror_pairs") or [])
        self.editor.fitted = list((rig or {}).get("bones") or [])
        self.pose_mode = True
        self.pose_job_id = job_id
        return True

    def exit_pose_mode(self) -> None:
        self.pose_mode = False
        self.pose_job_id = None
        self.rotate_gizmo.end_drag()
        self.translate_gizmo.end_drag()
        self.editor.clear()

    def get_pose(self) -> dict[str, list[float]]:
        return self.editor.pose()

    def set_pose(self, bones: dict[str, Any], *, pose_id: str | None = None, dirty=True) -> None:
        self.editor.apply(bones, pose_id=pose_id, dirty=dirty)
        self._after_pose_change()

    def apply_preset(self, preset: dict[str, Any]) -> None:
        self.editor.apply_preset(preset)
        self._after_pose_change()

    def reset_bone(self) -> None:
        self.editor.reset_bone()
        self._after_pose_change()

    def reset_all(self, *, dirty: bool = True) -> None:
        self.editor.reset_all(dirty=dirty)
        self._after_pose_change()

    def mirror(self) -> None:
        self.editor.mirror()
        self._after_pose_change()

    def _after_pose_change(self) -> None:
        if self.gpu is not None:
            self.gpu.refresh_palettes()
        if self.on_pose_dirty is not None:
            self.on_pose_dirty(self.editor.has_unsaved_edits())

    @property
    def selected_bone(self) -> str | None:
        return self.editor.selected

    def enter_joints_mode(self) -> None:
        self.editor.enter_joints_mode()
        self._after_pose_change()

    def exit_joints_mode(self) -> None:
        self.editor.exit_joints_mode()
        self._after_pose_change()

    def joint_moves(self) -> dict[str, list[float]]:
        return dict(self.editor.moved)

    def corrected_bones(self) -> list[dict[str, Any]]:
        return self.editor.corrected_bones()

    # -- rendering ---------------------------------------------------------

    def render(self, rect: tuple[float, float, float, float], dt: float) -> Any:
        """Draw one frame into the viewport. -> the resolved texture."""
        self._rect = rect
        width, height = int(max(rect[2], 1)), int(max(rect[3], 1))
        self._resize(self.viewport, width, height)
        self.camera.update(dt)
        self.renderer.draw(
            self.viewport,
            self.camera,
            self.gpu,
            model_matrix=self.placement,
            wireframe=self.wireframe,
            overlays=self._overlays(height),
        )
        if self.comparing and self.compare_viewport is not None:
            # One camera state, two renders: the point of a comparison is that
            # both meshes are seen from the identical angle.
            self._resize(self.compare_viewport, width, height)
            self.compare_camera.copy_from(self.camera)
            self.renderer.draw(
                self.compare_viewport,
                self.compare_camera,
                self.compare_gpu,
                model_matrix=self.compare_placement,
                wireframe=self.wireframe,
            )
        return self.viewport.texture

    def _resize(self, viewport: glctx.Viewport, width: int, height: int) -> None:
        """Resize, forgetting the outgoing texture first.

        ``Viewport.resize`` releases its texture and makes a new one, and the
        imgui backend maps GL names to moderngl objects: releasing without
        forgetting leaves it holding a dead object under a name the driver is
        free to reissue, which is how an unrelated image starts rendering as
        this one. Same rule, same shape as ``ClayView._resize``.
        """
        if (width, height) == viewport.size:
            return
        self._forget(viewport.texture)
        viewport.resize((width, height))

    def _forget(self, texture: Any) -> None:
        if texture is None:
            return
        from . import imgui_backend

        renderer = imgui_backend.current()
        if renderer is not None:
            renderer.forget_texture(texture)

    def _overlays(self, height: int) -> list[Any]:
        if not self.pose_mode or not self.editor.bound:
            return []
        radius = picking.marker_radius(self.radius)
        items = self.markers.draws(
            self.editor.handles, radius, self.editor.selected, self.placement
        )
        gizmo = self._active_gizmo()
        if gizmo is not None and self.editor.selected is not None:
            origin = picking.to_world(self.placement, self.editor.handles[self.editor.selected])
            basis = (
                self.placement @ self.model.nodes[self.model.by_name[self.editor.selected]].world
                if self.editor.mode == "pose"
                else m3.identity()
            )
            gizmo.place(origin, basis, self.camera, height)
            items += gizmo.draws()
        return items

    def _active_gizmo(self):
        if not self.pose_mode:
            return None
        return self.translate_gizmo if self.editor.mode == "joints" else self.rotate_gizmo

    # -- capture -----------------------------------------------------------

    def screenshot(self) -> Any:
        return capture.image(self.viewport)

    def thumbnail_png(self) -> bytes:
        return capture.png_bytes(self.viewport)

    def render_sheet_strip(self, yaws: list[float], elevation: float, flat: bool) -> Any:
        """Every direction in one call. Blocking: a draw and a synchronous
        GPU-to-CPU readback per cell, so sixteen of them is a visible freeze.
        The pane drives ``begin_sheet_strip``/``step_sheet_strip`` instead."""
        if self.gpu is None or self.model is None:
            return None
        return sheetlib.strip(
            self.renderer,
            self.gpu,
            self.model,
            yaws,
            elevation=elevation,
            flat=flat,
            model_matrix=self.placement,
        )

    def begin_sheet_strip(self, yaws: list[float], elevation: float, flat: bool) -> bool:
        """Start an incremental direction strip. -> whether there was a mesh.

        Any strip already in progress is dropped: the user has changed what
        they want, and finishing the old one would spend frames on an image
        about to be replaced.
        """
        self.cancel_sheet_strip()
        if self.gpu is None or self.model is None:
            return False
        self._strip = sheetlib.StripRender(
            self.renderer,
            self.gpu,
            self.model,
            yaws,
            elevation=elevation,
            flat=flat,
            model_matrix=self.placement,
        )
        return True

    def step_sheet_strip(self) -> Any:
        """Render one more cell. -> the finished image, or None while it runs.

        Frame thread only, like every other GL call here -- the point of the
        split is *how much* of it happens per frame, not which thread it is on.
        """
        if self._strip is None:
            return None
        if not self._strip.step():
            return None
        image, self._strip = self._strip.image, None
        return image

    @property
    def stripping(self) -> bool:
        return self._strip is not None

    def cancel_sheet_strip(self) -> None:
        if self._strip is not None:
            self._strip.release()
            self._strip = None

    # -- input -------------------------------------------------------------

    def handle_event(self, event: Any, hovered: bool) -> bool:
        """Feed one pygame event. -> whether the viewer consumed it.

        ``hovered`` is whether the pointer is over the viewport image; a drag
        already in progress ignores it, so crossing onto a panel mid-rotate
        does not drop the drag.
        """
        import pygame

        local = self._local(event)
        if event.type == pygame.MOUSEBUTTONDOWN and hovered:
            return self._press(event.button, local)
        if event.type == pygame.MOUSEBUTTONUP:
            return self._release(event.button)
        if event.type == pygame.MOUSEMOTION:
            return self._motion(local)
        if event.type == pygame.MOUSEWHEEL and hovered:
            self.camera.dolly(event.y)
            return True
        return False

    def _local(self, event: Any) -> tuple[float, float]:
        pos = getattr(event, "pos", None)
        if pos is None:
            return self._last_mouse
        return (pos[0] - self._rect[0], pos[1] - self._rect[1])

    def _press(self, button: int, local: tuple[float, float]) -> bool:
        self._last_mouse = local
        if button not in (1, 2, 3):
            return False
        if button == 1 and self.pose_mode and self.editor.bound:
            origin, direction = self._ray(local)
            gizmo = self._active_gizmo()
            axis = gizmo.hit(origin, direction) if self.editor.selected else None
            if axis is not None and gizmo.begin(axis, origin, direction):
                self._grab = "gizmo"
                return True
            centres = {
                name: picking.to_world(self.placement, point)
                for name, point in self.editor.handles.items()
            }
            hit = picking.nearest_hit(
                origin, direction, centres, picking.marker_radius(self.radius)
            )
            if hit is not None:
                self.editor.selected = hit
                self._grab = "marker"
                return True
        self._grab = "pan" if button in (2, 3) else "orbit"
        return True

    def _release(self, button: int) -> bool:
        if self._grab is None:
            return False
        was = self._grab
        self._grab = None
        if was == "gizmo":
            gizmo = self._active_gizmo()
            if gizmo is not None:
                gizmo.end_drag()
            if self.on_pose_dirty is not None:
                self.on_pose_dirty(self.editor.has_unsaved_edits())
        del button
        return True

    def _motion(self, local: tuple[float, float]) -> bool:
        dx = local[0] - self._last_mouse[0]
        dy = local[1] - self._last_mouse[1]
        self._last_mouse = local
        height = int(max(self._rect[3], 1))
        if self._grab is None:
            if self.pose_mode and self.editor.selected is not None:
                origin, direction = self._ray(local)
                gizmo = self._active_gizmo()
                if gizmo is not None:
                    gizmo.hover = gizmo.hit(origin, direction)
            return False
        if self._grab == "orbit":
            self.camera.orbit(dx, dy, height)
        elif self._grab == "pan":
            self.camera.pan(dx, dy, height)
        elif self._grab == "gizmo":
            origin, direction = self._ray(local)
            gizmo = self._active_gizmo()
            if self.editor.mode == "joints":
                moved = gizmo.update(origin, direction)
                if moved is not None:
                    self.editor.move_handle(
                        self.editor.selected, picking.from_world(self.placement, moved)
                    )
            else:
                delta = gizmo.update(origin, direction)
                if delta is not None:
                    self.editor.rotate_selected(delta)
                    if self.gpu is not None:
                        self.gpu.refresh_palettes()
        return True

    def _ray(self, local: tuple[float, float]):
        return screen_ray(
            self.camera, local[0], local[1], int(self._rect[2]), int(self._rect[3])
        )

    # -- teardown ----------------------------------------------------------

    def release(self) -> None:
        self.clear_reference()
        self._release_model()
        self.cancel_sheet_strip()
        self.exit_compare()
        self.markers.release()
        self.rotate_gizmo.release()
        self.translate_gizmo.release()
        self._forget(self.viewport.texture)
        self.viewport.release()
        if self.compare_viewport is not None:
            self._forget(self.compare_viewport.texture)
            self.compare_viewport.release()
        self.renderer.release()
