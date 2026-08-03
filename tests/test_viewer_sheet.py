"""The sheet preview's geometry and the capture path.

The load-bearing fact is where yaw 0 looks: Blender's -Y front becomes +Z once
the GLB is exported Y-up, so column 0 is the front view. A quarter turn out
here and every sheet disagrees with the sidecar that describes it.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import trimesh

from warlock.studio.viewer import capture, glctx, gltf
from warlock.studio.viewer import math3d as m3
from warlock.studio.viewer import scene as scenelib
from warlock.studio.viewer import sheet as sheetlib
from warlock.studio.viewer.render import Renderer


def test_yaw_zero_looks_along_plus_z():
    pos = sheetlib.camera_position(m3.vec3(0, 0, 0), 0.0, 0.0, 5.0)
    assert pos == pytest.approx([0, 0, 5], abs=1e-9)


def test_yaw_turns_the_same_way_the_worker_does():
    """90 degrees puts the camera on +X, which is the subject's left."""
    pos = sheetlib.camera_position(m3.vec3(0, 0, 0), 90.0, 0.0, 5.0)
    assert pos == pytest.approx([5, 0, 0], abs=1e-9)


def test_elevation_lifts_the_camera_without_changing_its_distance():
    centre = m3.vec3(1, 2, 3)
    pos = sheetlib.camera_position(centre, 37.0, math.radians(30.0), 5.0)
    assert float(np.linalg.norm(pos - centre)) == pytest.approx(5.0)
    assert pos[1] > centre[1]


def test_the_extent_is_the_worst_case_silhouette_with_the_workers_margin():
    """A subject is widest across its diagonal at 45 degrees, so framing on
    x or z alone crops it a quarter turn later."""
    size = m3.vec3(3.0, 1.0, 4.0)
    assert sheetlib.extent_of(size) == pytest.approx(5.0 * 1.12)
    # A tall thin subject is framed by its height instead.
    assert sheetlib.extent_of(m3.vec3(0.1, 9.0, 0.1)) == pytest.approx(9.0 * 1.12)


def test_a_zero_size_subject_does_not_divide_by_zero():
    assert sheetlib.extent_of(m3.vec3(0, 0, 0)) > 0


def test_the_summary_states_the_grid_the_worker_will_produce():
    assert sheetlib.summary(3, 8, 128) == "3 x 8 = 24 render cells - 1024x384 px output"
    assert sheetlib.summary(0, 4, 64).startswith("1 x 4 = 4 render cells")
    assert sheetlib.summary(2, 4, 64, clip=True).endswith("animated clip")


def test_an_ortho_camera_frames_its_extent():
    camera = sheetlib.OrthoCamera()
    camera.extent = 4.0
    camera.position = m3.vec3(0, 0, 8)
    camera.near, camera.far = 0.001, 32.0
    clip = camera.projection() @ camera.view() @ np.array([2.0, 0.0, 0.0, 1.0])
    assert clip[0] == pytest.approx(1.0)


# --- rendering --------------------------------------------------------------


@pytest.fixture
def box_model(gl, tmp_path):
    path = tmp_path / "box.glb"
    trimesh.creation.box(extents=(1.0, 2.0, 1.0)).export(path)
    model = gltf.load(path)
    return model, scenelib.GpuModel(gl, model)


def test_a_strip_is_one_transparent_cell_per_yaw(gl, box_model):
    model, gpu = box_model
    renderer = Renderer(gl)
    try:
        img = sheetlib.strip(
            renderer, gpu, model, [0.0, 90.0, 180.0, 270.0],
            model_matrix=scenelib.placement(model),
        )
    finally:
        renderer.release()
    assert img.size == (sheetlib.CELL * 4, sheetlib.CELL)
    alpha = np.array(img)[..., 3]
    # The subject is opaque; the surround is not. Both must be true, or the
    # sheet composites onto a black rectangle rather than onto nothing.
    assert (alpha == 255).any()
    assert (alpha == 0).any()
    assert alpha[0, 0] == 0


def test_every_cell_of_a_symmetric_subject_looks_the_same(gl, box_model):
    """A box is its own turnaround; a strip where the cells differ means the
    camera is not orbiting the centre it framed."""
    model, gpu = box_model
    renderer = Renderer(gl)
    try:
        img = sheetlib.strip(
            renderer, gpu, model, [0.0, 90.0, 180.0, 270.0],
            model_matrix=scenelib.placement(model), flat=True,
        )
    finally:
        renderer.release()
    cells = np.array(img).reshape(sheetlib.CELL, 4, sheetlib.CELL, 4)
    first = cells[:, 0]
    for i in range(1, 4):
        assert np.abs(cells[:, i].astype(int) - first.astype(int)).mean() < 2.0


def test_a_capture_is_opaque_unless_asked_otherwise(gl):
    from PIL import Image

    viewport = glctx.Viewport(gl, (32, 32))
    try:
        viewport.use()
        viewport.draw_target.clear(0.2, 0.4, 0.6, 0.0)
        viewport.resolve()
        opaque = Image.open(__import__("io").BytesIO(capture.png_bytes(viewport)))
        assert opaque.mode == "RGB"
        kept = Image.open(__import__("io").BytesIO(capture.png_bytes(viewport, opaque=False)))
        assert kept.mode == "RGBA"
        assert kept.getpixel((0, 0))[3] == 0
    finally:
        viewport.release()


def test_saving_a_capture_creates_its_directory(gl, tmp_path):
    viewport = glctx.Viewport(gl, (8, 8))
    try:
        viewport.use()
        viewport.draw_target.clear(0.0, 0.0, 0.0, 1.0)
        viewport.resolve()
        out = capture.save_png(viewport, tmp_path / "nested" / "shot.png")
    finally:
        viewport.release()
    assert out.exists() and out.read_bytes().startswith(b"\x89PNG")
