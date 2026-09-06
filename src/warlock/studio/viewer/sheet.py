"""The sprite-sheet direction preview.

One 64x64 orthographic cell per yaw, composited into a strip. It is a
*direction* preview, not a sheet: it cannot pose the mesh, so drawing one row
per pose would draw the same row N times -- the grid the worker will actually
produce is stated as a summary line instead, and :func:`summary` is the part
that has to agree with ``pipelines.sheet.plan``.

The camera math is the browser's, which is Blender's: yaw 0 sits on **+Z**,
because Blender's -Y front becomes +Z once the GLB is exported Y-up. Get that
wrong and every sheet is rotated by a quarter turn against the sidecar that
describes it.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ...pipelines import sheet as sheetlib
from . import math3d as m3
from .glctx import Viewport

CELL = 64
# The widest the subject can look from any yaw, with the same margin
# blender_worker.op_sheet frames with -- imported from the module both sides
# already go through, rather than typed out a second time here. A preview that
# frames differently from the renderer depicts a sheet nobody will get.
#
# What this preview depicts is the *unposed* model turning, which is exactly
# the case ``op_sheet``'s posed union collapses to: with no pose to measure the
# union is the rest box and the two agree to the bit. A sheet whose poses leave
# the rest box (a jump apex) is framed a little wider than this strip shows --
# correctly, since the alternative there is a clipped apex, and the strip has
# no poses to know it from.
MARGIN = sheetlib.FRAME_MARGIN


def extent_of(size: np.ndarray) -> float:
    """The ortho box half-height's basis: the subject's worst-case silhouette."""
    return max(math.hypot(float(size[0]), float(size[2])), float(size[1]), 1e-6) * MARGIN


def camera_position(
    centre: np.ndarray, yaw_degrees: float, elevation: float, distance: float
) -> np.ndarray:
    """Where the preview camera sits for one yaw.

    ``elevation`` is in radians, as the worker takes it. Yaw turns about +Y,
    starting at +Z, which is the direction the templates put the subject's
    front at once Blender's -Y has been exported Y-up.
    """
    a = math.radians(yaw_degrees)
    return np.asarray(centre, dtype="f8") + np.array(
        [
            distance * math.sin(a) * math.cos(elevation),
            distance * math.sin(elevation),
            distance * math.cos(a) * math.cos(elevation),
        ]
    )


class OrthoCamera:
    """The narrow camera interface Renderer.draw needs, orthographic.

    A separate small class rather than a mode on Camera: the orbit camera's
    whole state is spherical-about-a-target with damping, and none of that
    means anything for a fixed turntable frame.
    """

    def __init__(self) -> None:
        self.position = m3.vec3(0, 0, 1)
        self.target = m3.vec3()
        self.extent = 1.0
        self.near = 0.001
        self.far = 10.0
        self.aspect = 1.0
        self.fov = 45.0  # unused; present so screen_scale-style helpers work

    def view(self) -> np.ndarray:
        return m3.look_at(self.position, self.target, m3.vec3(0, 1, 0))

    def projection(self) -> np.ndarray:
        half = self.extent / 2.0
        return m3.orthographic(-half, half, -half, half, self.near, self.far)


class StripRender:
    """A direction strip, one cell per :meth:`step`.

    Incremental because every cell is a draw *and* a ``read_rgba`` -- a
    synchronous GPU-to-CPU readback, which stalls the pipeline. Sixteen of them
    in one frame is a visible freeze on the frame the user pressed the button;
    sixteen frames of one is a quarter of a second nobody notices. All of it
    has to stay on the frame thread regardless: it is the one GL context.

    Cleared to alpha zero and drawn with no grid: the sheet must contain the
    subject and nothing else, which is what the browser's beginPreviewScene
    arranged by hiding the grid, the markers and the background.

    The camera is framed once, from the bounds as they are now -- the same rule
    the sheet worker follows, and for the same reason: reframing per cell makes
    the subject jump between directions.
    """

    def __init__(
        self,
        renderer: Any,
        gpu: Any,
        model: Any,
        yaws: list[float],
        *,
        elevation: float = 0.0,
        flat: bool = True,
        model_matrix: np.ndarray | None = None,
        cell: int = CELL,
    ) -> None:
        from PIL import Image

        self._renderer = renderer
        self._gpu = gpu
        # Public: the pane draws "3 of 16" from it (L104), and a strip that
        # knows how many cells it has is the only thing that does -- the caller
        # computed the list from a form field that may since have changed.
        self.yaws = list(yaws)
        self._yaws = self.yaws
        self._flat = flat
        self._elevation = elevation
        self._cell = cell
        self.index = 0

        lo, hi = model.bounds()
        self._placement = m3.identity() if model_matrix is None else model_matrix
        lo = (self._placement @ np.append(lo, 1.0))[:3]
        hi = (self._placement @ np.append(hi, 1.0))[:3]
        size, self._centre = hi - lo, (lo + hi) * 0.5
        extent = extent_of(size)
        self._distance = extent * 2.0

        self._camera = OrthoCamera()
        self._camera.extent = extent
        self._camera.target = self._centre
        self._camera.near = 0.001
        self._camera.far = self._distance * 4.0

        self._viewport = Viewport(renderer.ctx, (cell, cell))
        self.image = Image.new("RGBA", (cell * max(len(self._yaws), 1), cell), (0, 0, 0, 0))

    @property
    def done(self) -> bool:
        return self.index >= len(self._yaws)

    def step(self) -> bool:
        """Render the next cell. -> whether the strip is now finished."""
        from PIL import Image

        if self.done:
            return True
        i, yaw = self.index, self._yaws[self.index]
        self._camera.position = camera_position(
            self._centre, yaw, self._elevation, self._distance
        )
        self._renderer.draw(
            self._viewport,
            self._camera,
            self._gpu,
            model_matrix=self._placement,
            flat=self._flat,
            show_grid=False,
            background=(0.0, 0.0, 0.0, 0.0),
        )
        self.image.paste(
            Image.fromarray(self._viewport.read_rgba(), "RGBA"), (i * self._cell, 0)
        )
        self.index += 1
        return self.done

    def release(self) -> None:
        self._viewport.release()


def strip(
    renderer: Any,
    gpu: Any,
    model: Any,
    yaws: list[float],
    *,
    elevation: float = 0.0,
    flat: bool = True,
    model_matrix: np.ndarray | None = None,
    cell: int = CELL,
) -> Any:
    """Every cell at once. -> PIL image.

    The whole-strip form, kept for callers with no frame to spread the work
    over -- the tests, and anything headless. The interactive preview drives
    :class:`StripRender` a cell at a time instead.
    """
    render = StripRender(
        renderer, gpu, model, yaws,
        elevation=elevation, flat=flat, model_matrix=model_matrix, cell=cell,
    )
    try:
        while not render.step():
            pass
        return render.image
    finally:
        render.release()


def summary(rows: int, yaws: int, frame_size: int, clip: bool = False) -> str:
    """The line under the preview. Must agree with ``pipelines.sheet.plan``."""
    rows = max(rows, 1)
    text = (
        f"{rows} x {yaws} = {rows * yaws} render cells - "
        f"{frame_size * yaws}x{frame_size * rows} px output"
    )
    return text + " - animated clip" if clip else text
