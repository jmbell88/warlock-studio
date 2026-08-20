"""What the GPU actually draws.

Three kinds of pin here, in increasing order of how much they'd cost to debug
without them:

* **Numeric** -- the colour pipeline, rendered and compared against a CPU
  reference of the same formulas. A tone-mapping port that is subtly wrong
  looks fine until you put it next to the old one.
* **Structural** -- the background is the exact hex, a rest-pose skinned mesh
  matches the unskinned one, a pose moves pixels.
* **Environmental** -- the SH coefficients as literals, because the
  ambient-from-below they carry is the whole reason the probe exists.

All of them skip without a GL 3.3 context.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from warlock.studio.viewer import env as envlib
from warlock.studio.viewer import glctx, gltf
from warlock.studio.viewer import math3d as m3
from warlock.studio.viewer import scene as scenelib
from warlock.studio.viewer.camera import Camera
from warlock.studio.viewer.render import DrawItem, Renderer


@pytest.fixture(scope="session")
def renderer(gl):
    r = Renderer(gl)
    yield r
    r.release()


@pytest.fixture
def viewport(gl):
    vp = glctx.Viewport(gl, (128, 128))
    yield vp
    vp.release()


@pytest.fixture
def box_glb(tmp_path):
    path = tmp_path / "box.glb"
    trimesh.creation.box(extents=(1.0, 1.0, 1.0)).export(path)
    return path


def _shown(gl, renderer, viewport, path, **kwargs):
    """Load, frame and draw a GLB. -> (rgba array, gpu model, camera)."""
    model = gltf.load(path)
    gpu = scenelib.GpuModel(gl, model)
    camera = Camera()
    lo, hi = model.bounds()
    camera.frame(lo, hi)
    renderer.fit_grid(lo, hi)
    renderer.draw(viewport, camera, gpu, model_matrix=scenelib.placement(model), **kwargs)
    return viewport.read_rgba(), gpu, camera


# --- colour pipeline --------------------------------------------------------


def _cpu_aces_srgb(linear: np.ndarray, exposure: float = 1.0) -> np.ndarray:
    """The reference: Stephen Hill's ACES fit, then the exact sRGB OETF."""
    inp = np.array(
        [[0.59719, 0.35458, 0.04823],
         [0.07600, 0.90834, 0.01566],
         [0.02840, 0.13383, 0.83777]]
    )
    out = np.array(
        [[1.60475, -0.53108, -0.07367],
         [-0.10208, 1.10813, -0.00605],
         [-0.00327, -0.07276, 1.07602]]
    )
    v = inp @ (linear * (exposure / 0.6))
    v = (v * (v + 0.0245786) - 0.000090537) / (v * (0.983729 * v + 0.4329510) + 0.238081)
    v = np.clip(out @ v, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, np.power(v, 0.41666) * 1.055 - 0.055)


@pytest.mark.parametrize("hexcolor", [0x7C6CF0, 0x4CC38A, 0xFFFFFF, 0x101010])
def test_the_shader_tone_maps_exactly_like_the_cpu_reference(
    gl, renderer, viewport, hexcolor
):
    """The 1/0.6 pre-scale is the one that gets dropped, and dropping it makes
    every render darker in a way that is only obvious side by side."""
    srgb = envlib.hex_to_srgb(hexcolor)
    triangle = gl.buffer(np.array([-3, -3, 0, 3, -3, 0, 0, 3, 0], dtype="f4").tobytes())
    program = renderer.programs.get("solid")
    vao = gl.vertex_array(program, [(triangle, "3f", "a_position")])

    camera = Camera()
    # An identity view/projection, so the clip-space triangle covers the frame.
    camera.view = lambda: m3.identity()
    camera.projection = lambda: m3.identity()
    renderer.draw(
        viewport, camera, None, show_grid=False,
        overlays=[DrawItem(vao=vao, color=(*srgb, 1.0))],
    )
    pixel = viewport.read_rgba()[64, 64, :3] / 255.0

    linear = np.array([envlib.srgb_to_linear(c) for c in srgb])
    assert pixel == pytest.approx(_cpu_aces_srgb(linear), abs=2 / 255)
    vao.release()
    triangle.release()


def test_the_background_is_the_literal_hex_and_not_tone_mapped(
    gl, renderer, viewport, box_glb
):
    """three sets the clear colour straight back to sRGB; running it through
    ACES would make the panel and the viewport two different greys."""
    pixels, _gpu, _camera = _shown(gl, renderer, viewport, box_glb)
    expected = [
        (envlib.BACKGROUND_HEX >> 16) & 0xFF,
        (envlib.BACKGROUND_HEX >> 8) & 0xFF,
        envlib.BACKGROUND_HEX & 0xFF,
    ]
    assert list(pixels[0, 0, :3]) == expected


# --- environment ------------------------------------------------------------


def test_the_probe_lights_a_downward_normal(gl):
    """The load-bearing property: without it a quarter of every model renders
    darker than the background and reads as holes in the mesh."""
    sh = envlib.sh9_irradiance(envlib.gradient_equirect())

    def irradiance(n):
        x, y, z = n
        basis = np.array([1, y, z, x, x * y, y * z, 3 * z * z - 1, x * z, x * x - y * y])
        return (sh * basis[:, None]).sum(axis=0)

    down = irradiance(np.array([0.0, -1.0, 0.0]))
    up = irradiance(np.array([0.0, 1.0, 0.0]))
    assert (down > 0.5).all(), "ambient from below is what the probe exists for"
    assert (up > down).all(), "and the sky is still brighter than the ground"


def test_the_sh_projection_of_a_constant_sky_is_pi_times_its_radiance():
    """The textbook value. If the basis normalisation or the cosine
    convolution is off, this is where it shows as a plain number."""
    flat = np.full((16, 32, 3), 0.5, dtype="f4")
    sh = envlib.sh9_irradiance(flat)
    n = np.array([0.0, 1.0, 0.0])
    basis = np.array([1, n[1], n[2], n[0], 0, 0, 3 * n[2] ** 2 - 1, 0, n[0] ** 2 - n[1] ** 2])
    # Half a percent, not exact: the integral is a 32x16 Riemann sum, and the
    # residual band-2 terms it leaves are the whole of the difference.
    assert (sh * basis[:, None]).sum(axis=0) == pytest.approx([np.pi * 0.5] * 3, rel=5e-3)


def test_the_gradient_is_brightest_at_the_horizon():
    equirect = envlib.gradient_equirect()
    ground, horizon, sky = equirect[0, 0], equirect[8, 0], equirect[15, 0]
    assert horizon.mean() > ground.mean()
    assert sky.mean() > horizon.mean()
    assert ground == pytest.approx(envlib.GROUND, abs=0.02)


def test_prefiltering_at_zero_roughness_reproduces_its_input():
    """A mirror reflection has to be the environment, not a blur of it."""
    source = envlib.gradient_equirect()
    out = envlib.prefilter(source, envlib.EQUIRECT_SIZE, 0.0)
    assert out == pytest.approx(source, abs=1e-5)


def test_prefiltering_at_full_roughness_flattens_towards_the_average():
    """A rough surface reflects a blur of the environment, not the environment
    -- and the blur keeps its energy rather than darkening towards it."""
    source = envlib.gradient_equirect()
    rough = envlib.prefilter(source, (8, 4), 1.0)
    assert rough.std() < source.std() * 0.75
    assert rough.mean() == pytest.approx(source.mean(), rel=0.1)


# --- geometry ---------------------------------------------------------------


def test_a_model_actually_covers_pixels(gl, renderer, viewport, box_glb):
    pixels, _gpu, _camera = _shown(gl, renderer, viewport, box_glb)
    background = np.array([0x14, 0x15, 0x1A])
    lit = (np.abs(pixels[..., :3].astype(int) - background) > 12).any(axis=-1)
    # A framed unit cube fills a good chunk of the frame; a few percent would
    # mean the camera or the placement transform is wrong rather than absent.
    assert 0.1 < lit.mean() < 0.7


def test_wireframe_draws_strictly_less_than_solid(gl, renderer, viewport, box_glb):
    solid, _gpu, _camera = _shown(gl, renderer, viewport, box_glb)
    wire, _gpu, _camera = _shown(gl, renderer, viewport, box_glb, wireframe=True)
    background = np.array([0x14, 0x15, 0x1A])

    def coverage(px):
        return (np.abs(px[..., :3].astype(int) - background) > 12).any(axis=-1).mean()

    assert coverage(wire) < coverage(solid)


def test_the_viewport_reads_back_top_row_first(gl, renderer, viewport):
    """GL reads bottom-up and every consumer counts from the top; getting this
    backwards flips every sheet cell and every thumbnail."""
    camera = Camera()
    viewport.use()
    viewport.draw_target.clear(0.0, 0.0, 0.0, 1.0)
    gl.scissor = (0, 0, 128, 32)  # the *bottom* 32 rows in GL's own terms
    viewport.draw_target.clear(1.0, 1.0, 1.0, 1.0)
    gl.scissor = None
    pixels = viewport.read_rgba()
    assert pixels[0, 0, 0] == 0, "the top row must be the one GL called last"
    assert pixels[-1, 0, 0] == 255
    del camera


def test_resizing_a_viewport_is_a_no_op_when_the_size_is_unchanged(gl):
    vp = glctx.Viewport(gl, (64, 48))
    texture = vp.texture
    assert vp.resize((64, 48)) is False
    assert vp.texture is texture
    assert vp.resize((80, 48)) is True
    assert vp.texture is not texture
    vp.release()


# --- skinning ---------------------------------------------------------------

# One particular finished, rigged job. `assets/` is gitignored and a job is
# routinely pruned, so the skip names *this asset* rather than the directory:
# a prune degrades to an honest skip instead of a test that never runs again
# while claiming it only wants the main checkout.
REAL_JOB = "44593039ccee"
REAL_RIG = f"assets/{REAL_JOB}/rig.glb"
REAL_MESH = f"assets/{REAL_JOB}/model.glb"
needs_real = pytest.mark.skipif(
    not (
        __import__("pathlib").Path(REAL_RIG).exists()
        and __import__("pathlib").Path(REAL_MESH).exists()
    ),
    reason=f"needs the rigged asset assets/{REAL_JOB}/ (pruned, or not this checkout)",
)


@needs_real
def test_a_rig_at_rest_renders_as_its_unskinned_mesh(gl, renderer, viewport):
    """The palette is the identity at rest by construction, so any difference
    here is a skinning bug rather than a modelling one."""
    rigged, _gpu, _camera = _shown(gl, renderer, viewport, REAL_RIG)
    plain, _gpu, _camera = _shown(gl, renderer, viewport, REAL_MESH)
    delta = np.abs(rigged[..., :3].astype(int) - plain[..., :3].astype(int))
    assert delta.mean() < 2.0
    assert delta.max() < 40


@needs_real
def test_posing_a_joint_moves_pixels(gl, renderer, viewport):
    model = gltf.load(REAL_RIG)
    gpu = scenelib.GpuModel(gl, model)
    camera = Camera()
    lo, hi = model.bounds()
    camera.frame(lo, hi)
    placement = scenelib.placement(model)

    renderer.draw(viewport, camera, gpu, model_matrix=placement)
    before = viewport.read_rgba()

    bone = next(b for b in model.by_name if b not in ("Node_0", "neutral_bone"))
    model.set_rotation(bone, m3.quat_from_axis_angle(m3.vec3(0, 0, 1), 0.8))
    model.update_world()
    gpu.refresh_palettes()
    renderer.draw(viewport, camera, gpu, model_matrix=placement)
    after = viewport.read_rgba()

    # One joint of twenty on a mesh that does not fill the frame: the bar is
    # "the palette reached the vertex shader", not "the silhouette changed".
    changed = (np.abs(before.astype(int) - after.astype(int)) > 8).any(axis=-1)
    assert changed.mean() > 0.005
    gpu.release()


# --- programs ---------------------------------------------------------------


def test_a_program_is_built_once_per_define_set(gl, renderer):
    first = renderer.programs.get("pbr", ("HAS_BASE_COLOR_MAP",))
    again = renderer.programs.get("pbr", ("HAS_BASE_COLOR_MAP",))
    other = renderer.programs.get("pbr", ())
    assert first is again
    assert first is not other


def test_every_program_compiles(gl, renderer):
    """Cheap, and it is the difference between a shader typo failing at import
    and failing the first time someone opens a mesh with a normal map."""
    from warlock.studio.viewer.programs import SOURCES

    for name in SOURCES:
        assert renderer.programs.get(name) is not None
    for defines in (
        ("SKINNED",),
        ("HAS_BASE_COLOR_MAP", "HAS_MR_MAP"),
        ("SKINNED", "HAS_BASE_COLOR_MAP", "HAS_MR_MAP", "HAS_NORMAL_MAP"),
        ("HAS_EMISSIVE_MAP", "HAS_AO_MAP"),
    ):
        assert renderer.programs.get("pbr", defines) is not None


# --- release hygiene ----------------------------------------------------------


def test_a_viewport_never_orphans_a_renderbuffer():
    """Releasing a moderngl framebuffer does not release its attachments, and
    the app sets no gc_mode -- so the MSAA colour and depth renderbuffers must
    be held by name and released on every reallocation. Built inline, they
    leaked a 4xMSAA pair per resize for the life of the context, and a splitter
    drag resizes on nearly every frame.

    A recording fake rather than the ``gl`` fixture: what is under test is the
    bookkeeping, and a real context cannot say what was *not* freed."""

    class _Obj:
        def __init__(self, released):
            self._released = released
            self.filter = None
            self.repeat_x = self.repeat_y = False

        def release(self):
            self._released.append(self)

    class _Ctx:
        def __init__(self):
            self.made: list[_Obj] = []
            self.released: list[_Obj] = []

        def _new(self):
            obj = _Obj(self.released)
            self.made.append(obj)
            return obj

        def texture(self, size, components):
            return self._new()

        def renderbuffer(self, size, components, samples=0):
            return self._new()

        def depth_renderbuffer(self, size, samples=0):
            return self._new()

        def framebuffer(self, color_attachments=None, depth_attachment=None):
            return self._new()

    for samples in (4, 1):
        ctx = _Ctx()
        vp = glctx.Viewport(ctx, (16, 16), samples=samples)
        vp.resize((32, 32))
        vp.release()
        leaked = [o for o in ctx.made if o not in ctx.released]
        assert not leaked, f"{len(leaked)} GL object(s) leaked at samples={samples}"
