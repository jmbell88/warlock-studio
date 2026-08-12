"""The procedural environment: a sky/horizon/ground gradient, and what it lights.

A generated gradient rather than an imported studio HDRI, for the same reason
the browser build had one: nothing is fetched at runtime and no HDR file is
vendored. Its job is not to be seen -- the background is a flat colour -- but to
supply ambient light *from below*. Without it a downward-facing surface gets
only the hemisphere light's ground colour, which is nearly black, so roughly a
quarter of a model's surface renders darker than the background and reads as a
hole in the mesh.

Two products come out of the same 32x16 gradient:

* **Diffuse** -- nine spherical-harmonic coefficients, integrated on the CPU.
  Exact for a source this smooth (it has no energy above the second band), and
  it carries the ambient-from-below that the whole environment exists for.
* **Specular** -- a small GGX-prefiltered cubemap, seven mips of roughness.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# The gradient, in linear light. Read off the frontend's gradientEnvironment().
SKY = (0.42, 0.48, 0.60)
HORIZON = (0.34, 0.34, 0.36)
GROUND = (0.16, 0.155, 0.15)

EQUIRECT_SIZE = (32, 16)
# The specular probe: a prefiltered equirect with one mip per roughness step.
# Equirect rather than a cubemap because moderngl cannot write a cube face at a
# given mip level (and cannot attach one to a framebuffer either) -- and
# because three's own PMREM output is a 2D texture too, so this is the closer
# analogue rather than a compromise.
PROBE_SIZE = (64, 32)
PROBE_MIPS = 6

# Hemisphere light: white sky, 0x000223 ground, intensity 0.3. The ground
# colour is nearly black on purpose -- the environment probe is what fills in
# from below, and this only tints.
HEMI_SKY_HEX = 0xFFFFFF
HEMI_GROUND_HEX = 0x000223
HEMI_INTENSITY = 0.3

# Key light: white, intensity 1.5, from (3, 4, 2). Purely for shaping.
KEY_DIRECTION = (3.0, 4.0, 2.0)
KEY_COLOR_HEX = 0xFFFFFF
KEY_INTENSITY = 1.5

# The viewport background. Deliberately *not* tone-mapped: three sets the clear
# colour through getUnlitUniformColorSpace, which converts the working-space
# value straight back to sRGB -- so what reaches the framebuffer is the literal
# hex. Running it through ACES here would make the panel and the viewport two
# different greys.
# Matches the dark palette's tokens.BG so the viewport and the panels around it
# read as one surface; kept literal here (not imported) because the viewer
# package stays free of studio-UI imports. It is unconditional, so under the
# light palette the viewport stays dark and that agreement is lost -- an open
# design call rather than an oversight (LEFTOVERS.md section 15 -- deleted; git
# history), because the
# background doubles as the render-skip key.
BACKGROUND_HEX = 0x0F1014


def srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def hex_to_linear(value: int) -> tuple[float, float, float]:
    """A three.js colour literal: sRGB hex in, linear working space out."""
    return tuple(
        srgb_to_linear(((value >> shift) & 0xFF) / 255.0) for shift in (16, 8, 0)
    )


def hex_to_srgb(value: int) -> tuple[float, float, float]:
    return tuple(((value >> shift) & 0xFF) / 255.0 for shift in (16, 8, 0))


def gradient_equirect(size: tuple[int, int] = EQUIRECT_SIZE) -> np.ndarray:
    """-> (h, w, 3) float32 linear radiance.

    Row 0 is v=0, which equirectangular mapping puts at -Y: the ground. The
    interpolation is horizon-to-edge in both directions rather than a single
    top-to-bottom ramp, which is what puts the brightest band at the horizon.
    """
    width, height = size
    sky = np.array(SKY, dtype="f4")
    horizon = np.array(HORIZON, dtype="f4")
    ground = np.array(GROUND, dtype="f4")
    v = (np.arange(height, dtype="f4") + 0.5) / height
    t = np.abs(v - 0.5) * 2.0
    edge = np.where(v[:, None] > 0.5, sky[None, :], ground[None, :])
    rows = horizon[None, :] + (edge - horizon[None, :]) * t[:, None]
    return np.repeat(rows[:, None, :], width, axis=1).astype("f4")


def sh9_irradiance(equirect: np.ndarray) -> np.ndarray:
    """-> (9, 3) float32 coefficients ready for ``shIrradiance`` in GLSL.

    Ramamoorthi & Hanrahan: project onto the first nine real SH basis
    functions, then convolve with the clamped-cosine kernel (A0 = pi,
    A1 = 2pi/3, A2 = pi/4). Both the basis normalisation and that factor are
    folded in here, so the shader evaluates a bare polynomial.
    """
    height, width, _ = equirect.shape
    # Latitude at the row centre; the same mapping the shader samples with.
    lat = (np.arange(height) + 0.5) / height
    lat = np.pi * (lat - 0.5)
    lon = 2.0 * np.pi * ((np.arange(width) + 0.5) / width - 0.5)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")

    y = np.sin(lat_grid)
    x = np.cos(lat_grid) * np.cos(lon_grid)
    z = np.cos(lat_grid) * np.sin(lon_grid)
    # dw = cos(lat) dlat dlon -- the cos is why the poles contribute less than
    # their pixel count suggests.
    dw = np.cos(lat_grid) * (np.pi / height) * (2.0 * np.pi / width)

    k = [0.282095, 0.488603, 0.488603, 0.488603,
         1.092548, 1.092548, 0.315392, 1.092548, 0.546274]
    basis = [
        np.ones_like(x), y, z, x,
        x * y, y * z, 3.0 * z * z - 1.0, x * z, x * x - y * y,
    ]
    a = [math.pi, 2.0 * math.pi / 3.0, math.pi / 4.0]
    band = [0, 1, 1, 1, 2, 2, 2, 2, 2]

    out = np.zeros((9, 3), dtype="f8")
    for i in range(9):
        # L_i = integral of L * Y_i; C_i = A_l * k_i * L_i, so k appears twice.
        moment = (equirect * (k[i] * basis[i] * dw)[:, :, None]).sum(axis=(0, 1))
        out[i] = a[band[i]] * k[i] * moment
    return out.astype("f4")


def equirect_directions(size: tuple[int, int]) -> np.ndarray:
    """-> (h, w, 3) unit directions, one per texel of an equirect image.

    The exact inverse of :func:`sample_equirect`'s mapping, which is what makes
    a zero-roughness prefilter reproduce its input rather than rotate it.
    """
    width, height = size
    lat = np.pi * ((np.arange(height) + 0.5) / height - 0.5)
    lon = 2.0 * np.pi * ((np.arange(width) + 0.5) / width - 0.5)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    return np.stack(
        [
            np.cos(lat_grid) * np.cos(lon_grid),
            np.sin(lat_grid),
            np.cos(lat_grid) * np.sin(lon_grid),
        ],
        axis=-1,
    )


def sample_equirect(equirect: np.ndarray, dirs: np.ndarray) -> np.ndarray:
    """Bilinear equirectangular lookup. Wraps in longitude, clamps in latitude."""
    height, width, _ = equirect.shape
    x = np.arctan2(dirs[..., 2], dirs[..., 0]) / (2.0 * np.pi) + 0.5
    y = np.arcsin(np.clip(dirs[..., 1], -1.0, 1.0)) / np.pi + 0.5
    fx = x * width - 0.5
    fy = np.clip(y * height - 0.5, 0.0, height - 1.0)
    x0 = np.floor(fx).astype(int)
    y0 = np.floor(fy).astype(int)
    tx = (fx - x0)[..., None]
    ty = (fy - y0)[..., None]
    x0m, x1m = x0 % width, (x0 + 1) % width
    y0c, y1c = np.clip(y0, 0, height - 1), np.clip(y0 + 1, 0, height - 1)
    top = equirect[y0c, x0m] * (1 - tx) + equirect[y0c, x1m] * tx
    bottom = equirect[y1c, x0m] * (1 - tx) + equirect[y1c, x1m] * tx
    return top * (1 - ty) + bottom * ty


def _ggx_samples(count: int = 64) -> np.ndarray:
    """Hammersley points, as (count, 2). Deterministic, so a golden image is."""
    i = np.arange(count, dtype="f8")
    bits = np.arange(count, dtype=np.uint32)
    bits = (bits << np.uint32(16)) | (bits >> np.uint32(16))
    bits = ((bits & np.uint32(0x55555555)) << np.uint32(1)) | (
        (bits & np.uint32(0xAAAAAAAA)) >> np.uint32(1)
    )
    bits = ((bits & np.uint32(0x33333333)) << np.uint32(2)) | (
        (bits & np.uint32(0xCCCCCCCC)) >> np.uint32(2)
    )
    bits = ((bits & np.uint32(0x0F0F0F0F)) << np.uint32(4)) | (
        (bits & np.uint32(0xF0F0F0F0)) >> np.uint32(4)
    )
    bits = ((bits & np.uint32(0x00FF00FF)) << np.uint32(8)) | (
        (bits & np.uint32(0xFF00FF00)) >> np.uint32(8)
    )
    return np.stack([i / count, bits.astype("f8") * 2.3283064365386963e-10], axis=1)


def prefilter(
    equirect: np.ndarray,
    size: tuple[int, int],
    roughness: float,
    samples: int = 64,
) -> np.ndarray:
    """-> (h, w, 3) float32 radiance, GGX-prefiltered at one roughness.

    The split-sum approximation's standard simplification: the view direction
    is assumed to equal the normal, which is what makes a single prefiltered
    map serve every viewing angle.
    """
    n = equirect_directions(size)
    if roughness <= 0.0:
        return sample_equirect(equirect, n).astype("f4")
    alpha = roughness * roughness
    # An orthonormal frame per texel, avoiding the degenerate cross product at
    # the poles by picking the up vector the normal is least aligned with.
    up = np.where(
        np.abs(n[..., 2:3]) < 0.999,
        np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
    )
    tangent = np.cross(up, n)
    tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True)
    bitangent = np.cross(n, tangent)

    total = np.zeros(n.shape, dtype="f8")
    weight = np.zeros(n.shape[:2] + (1,), dtype="f8")
    for xi in _ggx_samples(samples):
        phi = 2.0 * np.pi * xi[0]
        cos_theta = np.sqrt((1.0 - xi[1]) / (1.0 + (alpha * alpha - 1.0) * xi[1]))
        sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
        h = (
            tangent * (sin_theta * np.cos(phi))
            + bitangent * (sin_theta * np.sin(phi))
            + n * cos_theta
        )
        light = 2.0 * (n * h).sum(axis=-1, keepdims=True) * h - n
        light /= np.linalg.norm(light, axis=-1, keepdims=True)
        dot_nl = np.clip((n * light).sum(axis=-1, keepdims=True), 0.0, None)
        total += sample_equirect(equirect, light) * dot_nl
        weight += dot_nl
    filtered = total / np.maximum(weight, 1e-9)
    # A texel no sample reached (only possible at a degenerate frame) falls
    # back to its own direction rather than to black.
    return np.where(weight > 0, filtered, sample_equirect(equirect, n)).astype("f4")


class Environment:
    """The GPU-side environment: SH coefficients and a prefiltered cubemap."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        equirect = gradient_equirect()
        self.sh = sh9_irradiance(equirect)
        self.probe = self._prefilter(equirect)
        self.max_mip = float(PROBE_MIPS - 1)
        self.hemi_sky = tuple(
            c * HEMI_INTENSITY for c in hex_to_linear(HEMI_SKY_HEX)
        )
        self.hemi_ground = tuple(
            c * HEMI_INTENSITY for c in hex_to_linear(HEMI_GROUND_HEX)
        )
        self.key_color = tuple(
            c * KEY_INTENSITY for c in hex_to_linear(KEY_COLOR_HEX)
        )
        direction = np.array(KEY_DIRECTION, dtype="f8")
        self.key_direction = tuple(direction / np.linalg.norm(direction))
        self.background = hex_to_srgb(BACKGROUND_HEX)

    def _prefilter(self, equirect: np.ndarray) -> Any:
        """Build the roughness-mipped specular probe and upload it, mip by mip.

        On the CPU, in numpy: the source is 32x16 and the target's largest mip
        is 64x32, so the whole thing is well under a million samples and needs
        neither a shader nor a render target. A full PMREM port was considered
        and rejected for the same reason it would be anywhere -- it exists to
        preserve detail a real HDRI has, and this gradient has none.
        """
        probe = self.ctx.texture(PROBE_SIZE, 3, dtype="f4")
        probe.filter = (self.ctx.LINEAR_MIPMAP_LINEAR, self.ctx.LINEAR)
        # Allocates the mip chain; every level is then overwritten with a
        # properly prefiltered one rather than a box-downsampled one.
        probe.build_mipmaps(0, PROBE_MIPS - 1)
        # Wrapping in longitude, clamped in latitude: the seam behind the
        # camera is a real sample boundary, the poles are not.
        probe.repeat_x, probe.repeat_y = True, False
        for mip in range(PROBE_MIPS):
            size = (max(1, PROBE_SIZE[0] >> mip), max(1, PROBE_SIZE[1] >> mip))
            pixels = prefilter(equirect, size, mip / (PROBE_MIPS - 1))
            probe.write(pixels.tobytes(), level=mip)
        return probe

    def bind(self, program: Any, unit: int = 4) -> None:
        """Set every environment uniform a lit program declares."""
        if "u_sh" in program:
            program["u_sh"].write(np.ascontiguousarray(self.sh).tobytes())
        if "u_env" in program:
            self.probe.use(unit)
            program["u_env"].value = unit
            program["u_env_max_mip"].value = self.max_mip
        if "u_hemi_sky" in program:
            program["u_hemi_sky"].value = self.hemi_sky
            program["u_hemi_ground"].value = self.hemi_ground
        if "u_light_dir" in program:
            program["u_light_dir"].value = self.key_direction
            program["u_light_color"].value = self.key_color

    def release(self) -> None:
        self.probe.release()
