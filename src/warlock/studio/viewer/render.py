"""Drawing a model into a viewport.

The renderer owns the things that outlive any one model -- the program cache,
the environment probe, the grid -- and nothing else. A frame is: clear, draw
the grid, draw every primitive with the lit (or unlit) program, then draw the
overlays with depth testing off, which is how the frontend's ``renderOrder =
999`` markers stayed visible through the mesh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import moderngl
import numpy as np

from . import math3d as m3
from .env import Environment
from .glctx import Viewport
from .grid import Grid, span_for
from .programs import ProgramCache
from .scene import GpuModel


@dataclass
class DrawItem:
    """One overlay draw: a vertex array, a colour and how to draw it.

    Markers, gizmo handles and Clay's element-selection overlays are the users.
    They are described rather than drawn directly so the renderer can order
    them without each of them knowing that it is an overlay.

    ``depth`` is the one thing they genuinely disagree about. A joint marker
    inside a mesh must draw through it -- one you cannot see is one you cannot
    grab -- but a selected vertex on the *far* side of a closed mesh must not,
    or the user is looking at a cloud of dots with no way to tell which are in
    front. So depth-tested items draw first, into the depth buffer the model
    left behind, and depth-off items draw over them, exactly as before.

    ``point_size`` is only read for ``POINTS``; zero leaves whatever the
    context had, which is what every existing caller wants.
    """

    vao: Any
    color: tuple[float, float, float, float]
    model: np.ndarray = field(default_factory=m3.identity)
    mode: int = moderngl.TRIANGLES
    vertices: int = -1
    depth: bool = False
    point_size: float = 0.0


class Renderer:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self.programs = ProgramCache(ctx)
        self.env = Environment(ctx)
        self.grid = Grid(ctx, self.programs)
        self.exposure = 1.0

    # -- frame -------------------------------------------------------------

    def draw(
        self,
        viewport: Viewport,
        camera: Any,
        gpu: GpuModel | None,
        *,
        model_matrix: np.ndarray | None = None,
        wireframe: bool = False,
        flat: bool = False,
        show_grid: bool = True,
        background: tuple[float, float, float, float] | None = None,
        overlays: list[DrawItem] | None = None,
    ) -> None:
        ctx = self.ctx
        viewport.use()
        clear = background if background is not None else (*self.env.background, 1.0)
        viewport.draw_target.clear(*clear, depth=1.0)

        camera.aspect = viewport.size[0] / max(viewport.size[1], 1)
        view = camera.view()
        proj = camera.projection()

        ctx.enable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        if show_grid:
            ctx.wireframe = False
            self.grid.render(view, proj)

        if gpu is not None:
            model_matrix = m3.identity() if model_matrix is None else model_matrix
            # Desktop-only: GLES has no glPolygonMode, so a GLES port would
            # need a barycentric shader instead. Left as-is deliberately --
            # this app ships a window, not a web page.
            ctx.wireframe = wireframe
            self._draw_model(gpu, camera, view, proj, model_matrix, flat)
            ctx.wireframe = False

        if overlays:
            # Depth-tested first, then depth-off over the top -- see
            # ``DrawItem.depth``. Everything goes through the one "solid"
            # program, the gizmo idiom: an overlay is a coloured vertex array,
            # and a second shader would be a second place to keep the tone
            # mapping in step.
            program = self.programs.get("solid")
            program["u_view"].write(m3.gl_bytes(view))
            program["u_proj"].write(m3.gl_bytes(proj))
            program["u_exposure"].value = self.exposure
            for tested in (True, False):
                items = [item for item in overlays if item.depth is tested]
                if not items:
                    continue
                if tested:
                    ctx.enable(moderngl.DEPTH_TEST)
                else:
                    ctx.disable(moderngl.DEPTH_TEST)
                for item in items:
                    if item.point_size:
                        ctx.point_size = item.point_size
                    program["u_model"].write(m3.gl_bytes(item.model))
                    program["u_color"].value = item.color
                    item.vao.render(mode=item.mode, vertices=item.vertices)
            ctx.enable(moderngl.DEPTH_TEST)

        viewport.resolve()

    def _draw_model(
        self,
        gpu: GpuModel,
        camera: Any,
        view: np.ndarray,
        proj: np.ndarray,
        model_matrix: np.ndarray,
        flat: bool,
    ) -> None:
        name = "unlit" if flat else "pbr"
        # The per-frame constants -- view, projection, exposure, camera, the
        # environment -- are written once per *program* rather than once per
        # primitive (B15), mirroring the overlay pass. A frame draws dozens of
        # primitives through a handful of programs.
        view_bytes = m3.gl_bytes(view)
        proj_bytes = m3.gl_bytes(proj)
        camera_pos = tuple(camera.position)
        seen: set[int] = set()
        # The normal-matrix cache lives on GpuModel; Clay's _Composite has
        # none, so those draws fall back to the inline computation.
        normal_bytes = getattr(gpu, "normal_matrix_bytes", None)
        for node, primitive in gpu.draws:
            program = self.programs.get(name, primitive.defines)
            world = model_matrix @ node.world
            program["u_model"].write(m3.gl_bytes(world))
            if id(program) not in seen:
                seen.add(id(program))
                program["u_view"].write(view_bytes)
                program["u_proj"].write(proj_bytes)
                program["u_exposure"].value = self.exposure
                if "u_camera_pos" in program:
                    program["u_camera_pos"].value = camera_pos
                    self.env.bind(program)
            if "u_normal_matrix" in program:
                # The inverse transpose, so a non-uniform scale does not tilt
                # the normals. Per node because the placement transform is
                # uniform but a glTF node's need not be; cached per node on
                # the model (B15) because a node's world only moves on a pose
                # change or a placement change.
                #
                # A zero on any scale axis makes the 3x3 singular, and an
                # unguarded inverse raised out of the *draw* -- one flattened
                # object took the whole viewport down. Its normals are
                # undefined either way, so the identity is as good an answer
                # as exists and the rest of the scene still renders.
                if normal_bytes is not None:
                    program["u_normal_matrix"].write(normal_bytes(node, world))
                else:
                    try:
                        normal_matrix = np.linalg.inv(world[:3, :3]).T
                    except np.linalg.LinAlgError:
                        normal_matrix = np.eye(3)
                    program["u_normal_matrix"].write(
                        np.ascontiguousarray(normal_matrix.T, dtype="f4").tobytes()
                    )
            primitive.material.bind(program)
            if len(primitive.material.textures) >= 5 and "u_camera_pos" in program:
                # A five-texture material walks its units up to 4, which is
                # the environment probe's slot -- rebind it, since the hoist
                # above only bound it once per program.
                self.env.bind(program)
            if primitive.skinned and "u_joints" in program:
                palette = gpu.palette(node)
                if palette:
                    program["u_joints"].write(palette)
            if primitive.material.material.double_sided:
                self.ctx.disable(moderngl.CULL_FACE)
            else:
                # Matches three: single-sided means back faces are culled, and
                # trellis output is genuinely closed often enough for it to
                # matter to fill rate.
                self.ctx.enable(moderngl.CULL_FACE)
            primitive.vao(program).render()
        self.ctx.disable(moderngl.CULL_FACE)

    # -- helpers -----------------------------------------------------------

    def fit_grid(self, lo: np.ndarray, hi: np.ndarray) -> None:
        size = np.asarray(hi) - np.asarray(lo)
        self.grid.set_span(span_for(float(size[0]), float(size[2])))

    def release(self) -> None:
        self.grid.release()
        self.env.release()
        self.programs.release()
