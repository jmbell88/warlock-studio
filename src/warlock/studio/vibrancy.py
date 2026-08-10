"""What floats blurs what it floats over (UX.md Phase 5, item 2).

The command palette, the modals and the header popups are surfaces the app puts
*over* itself, and Phase 2 gave them a shadow and a solid fill. This gives them
the other half of the material: a blurred copy of what is behind them, tinted,
so a floating surface reads as translucent rather than as an opaque rectangle
with a soft edge.

**The copy is of the last frame with nothing floating on it, and that is the
design rather than a shortcut.** The phase's own sketch was "an offscreen copy
of the composed frame *before* the overlay layer", which needs the frame split
in two at a point imgui does not name: draw data arrives as a flat list of
command lists with nothing on them saying which window each came from, so
"before the overlays" is a guess about list ordering that a tooltip or a popup
invalidates. Capturing the *clean* frame instead is exact where that is
approximate, and it has the property the naive alternative does not: a surface
never samples a blur of itself, which is what makes the region under an open
palette converge to smeared mush over a few frames.

What it costs is staleness -- the backdrop is the app as it was when the
surface appeared. That is invisible for the surfaces this is applied to (a
modal blocks the app behind it; the palette is opened, typed in and closed)
and it is *stated* rather than hidden, which is why toasts deliberately do not
use it: a toast appears while the thing it reports on is still moving.

The capture is a quarter-resolution blit, throttled, and the blur runs only on
frames where something actually asked for the backdrop. On an idle app with no
palette open the whole module costs one blit every 60-odd milliseconds.
"""

from __future__ import annotations

import logging
import time
from typing import Any

log = logging.getLogger(__name__)

# How much smaller the captured copy is than the window. A blur this heavy
# throws away the detail a full-resolution capture would carry anyway, and the
# quarter costs a sixteenth of the fill.
DOWNSCALE = 4

# The floor on the captured size, so a minimised or tiny window still produces
# a texture rather than a zero-sized one.
MIN_SIDE = 8

# How often the clean copy is refreshed, in seconds. The backdrop is heavily
# blurred and only ever seen under a surface that has *just* appeared, so a
# sixteenth of a second of staleness is not visible -- and an unthrottled
# capture is a full-screen blit on every frame of an app that is otherwise
# idle at twelve.
CAPTURE_INTERVAL = 1.0 / 15.0

# How many texels apart the taps are. Two rather than one, measured against a
# capture with readable UI text in it: at one texel the backdrop under the
# palette was still legible, which is a panel you can read *through* rather
# than a material -- and legibility of the panel's own text is what the tint
# above it is for.
TAP_TEXELS = 2.0

_VERTEX = """
#version 330 core
in vec2 in_pos;
out vec2 uv;
void main() {
    uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_FRAGMENT = """
#version 330 core
uniform sampler2D Source;
uniform vec2 Step;
in vec2 uv;
out vec4 Out_Color;
void main() {
    // A nine-tap Gaussian, five distinct weights (sigma ~= 2 texels), written
    // out rather than computed: the loop bound has to be a constant anyway.
    // At quarter resolution and with the bilinear upscale counted this is a
    // ~13 px blur in window pixels, which is where a backdrop stops being
    // readable and starts being a material.
    float w[5] = float[](0.2270270, 0.1945946, 0.1216216, 0.0540541, 0.0162162);
    vec4 total = texture(Source, uv) * w[0];
    for (int i = 1; i < 5; ++i) {
        total += texture(Source, uv + Step * float(i)) * w[i];
        total += texture(Source, uv - Step * float(i)) * w[i];
    }
    Out_Color = total;
}
"""


class _Pipeline:
    """The textures, framebuffers and program the blur needs, for one size."""

    def __init__(self, gl: Any, size: tuple[int, int]) -> None:
        import numpy as np

        self.gl = gl
        self.size = size
        self.clean = gl.texture(size, 4)
        self.ping = gl.texture(size, 4)
        self.pong = gl.texture(size, 4)
        for texture in (self.clean, self.ping, self.pong):
            import moderngl

            texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
            # Clamped: a blur samples past the edge by construction, and
            # wrapping puts the top of the window into the bottom of it.
            texture.repeat_x = texture.repeat_y = False
        self.clean_fbo = gl.framebuffer(color_attachments=[self.clean])
        self.ping_fbo = gl.framebuffer(color_attachments=[self.ping])
        self.pong_fbo = gl.framebuffer(color_attachments=[self.pong])
        self.program = gl.program(vertex_shader=_VERTEX, fragment_shader=_FRAGMENT)
        quad = np.array([-1.0, -1.0, 3.0, -1.0, -1.0, 3.0], dtype="f4")
        self.vbo = gl.buffer(quad.tobytes())
        self.vao = gl.vertex_array(self.program, [(self.vbo, "2f", "in_pos")])
        self.ref: Any = None

    def release(self) -> None:
        for item in (
            self.vao,
            self.vbo,
            self.program,
            self.clean_fbo,
            self.ping_fbo,
            self.pong_fbo,
            self.clean,
            self.ping,
            self.pong,
        ):
            try:
                item.release()
            except Exception:  # noqa: BLE001 - teardown continues past one bad handle
                log.debug("could not release a vibrancy resource", exc_info=True)


_PIPELINE: _Pipeline | None = None
_BROKEN = False
# Whether the blurred texture matches the current clean copy. The blur is run
# lazily on the first surface that asks for it, so a capture only marks it out
# of date rather than paying for a blur nothing may look at.
_STALE = True
# Whether anything sampled the backdrop this frame. The capture at the end of
# the frame is skipped when something did, which is the whole of how a surface
# is kept out of its own backdrop.
_USED = False
_LAST_CAPTURE = 0.0


def available() -> bool:
    """Whether the effect is switched on and has not failed."""
    from . import effects

    return bool(effects.VIBRANCY) and not _BROKEN


def _target_size(window: tuple[float, float]) -> tuple[int, int]:
    return (
        max(int(window[0]) // DOWNSCALE, MIN_SIDE),
        max(int(window[1]) // DOWNSCALE, MIN_SIDE),
    )


def capture(gl: Any, window: tuple[float, float]) -> None:
    """Copy the composed frame, if nothing floating was drawn over it.

    Called at the very end of the frame, after imgui has been rendered: what is
    on the default framebuffer at that point *is* the composed frame, so there
    is nothing to intercept and no second render pass.
    """
    global _PIPELINE, _BROKEN, _STALE, _USED, _LAST_CAPTURE

    used, _USED = _USED, False
    if not available() or used:
        # The switch going off takes the pipeline with it, here rather than in
        # the settings pane: this is the one call the frame loop already makes
        # every frame, so the resources follow the flag with nobody having to
        # remember to release them -- and it is a no-op once they are gone.
        if not available():
            _release_pipeline()
        return
    now = time.monotonic()
    if now - _LAST_CAPTURE < CAPTURE_INTERVAL:
        return
    size = _target_size(window)
    try:
        if _PIPELINE is None or _PIPELINE.size != size:
            # Through the paired release, never ``_PIPELINE.release()`` bare:
            # the pong is registered with the imgui renderer, and releasing a
            # registered texture leaves the backend holding a dead object
            # under a GL name the driver will reuse.
            _release_pipeline()
            _PIPELINE = _Pipeline(gl, size)
        gl.copy_framebuffer(_PIPELINE.clean_fbo, gl.screen)
    except Exception:
        log.warning("translucent panels are unavailable; using solid fills", exc_info=True)
        _BROKEN = True
        _release_pipeline()
        return
    _LAST_CAPTURE = now
    _STALE = True


def backdrop() -> Any:
    """The blurred copy as an imgui texture ref, or ``None``.

    Blurring here rather than at capture time is what keeps the cost off the
    frames nobody is looking at a floating surface on: the two passes run once
    per *capture that something actually samples*, not once per capture.
    """
    global _BROKEN, _STALE, _USED

    if not available() or _PIPELINE is None:
        return None
    _USED = True
    if _STALE:
        try:
            _blur(_PIPELINE)
        except Exception:
            log.warning("translucent panels are unavailable; using solid fills", exc_info=True)
            _BROKEN = True
            _release_pipeline()
            return None
        _STALE = False
    return _PIPELINE.ref


def _blur(pipeline: _Pipeline) -> None:
    """Two separable passes: clean -> ping (across), ping -> pong (down)."""
    import moderngl

    from . import imgui_backend

    gl = pipeline.gl
    width, height = pipeline.size
    # Every GL flag off for the two passes. Blending is the one that matters:
    # this writes an opaque copy over whatever the target held last, and
    # blending it with the previous capture is a ghost that never quite clears.
    # The imgui renderer sets its own state at the top of every render, so
    # nothing downstream inherits this.
    gl.enable_only(moderngl.NOTHING)
    for source, target, step in (
        (pipeline.clean, pipeline.ping_fbo, (TAP_TEXELS / width, 0.0)),
        (pipeline.ping, pipeline.pong_fbo, (0.0, TAP_TEXELS / height)),
    ):
        target.use()
        gl.viewport = (0, 0, width, height)
        source.use(0)
        pipeline.program["Source"].value = 0
        pipeline.program["Step"].value = step
        pipeline.vao.render(mode=4, vertices=3)
    # Back to the screen, which is what every other offscreen pass in the app
    # leaves bound: the imgui renderer binds it itself, but a viewport left
    # pointing at a quarter-size framebuffer would be inherited by whatever
    # draws next inside the same frame.
    gl.screen.use()
    renderer = imgui_backend.current()
    if renderer is not None:
        renderer.register_texture(pipeline.pong)
    from imgui_bundle import imgui

    if pipeline.ref is None:
        pipeline.ref = imgui.ImTextureRef(pipeline.pong.glo)


def uv_for(low: Any, high: Any, window: tuple[float, float]) -> tuple[Any, Any]:
    """The captured frame's UVs under a screen-space rectangle.

    The Y range comes back inverted on purpose: the capture is a blit of the
    default framebuffer, whose origin is bottom-left, and imgui's screen
    coordinates start at the top. Flipping here rather than in the shader keeps
    the pipeline a plain copy of the frame, which is what makes it debuggable
    by looking at it.
    """
    width = max(float(window[0]), 1.0)
    height = max(float(window[1]), 1.0)
    return (
        (low[0] / width, 1.0 - low[1] / height),
        (high[0] / width, 1.0 - high[1] / height),
    )


def _release_pipeline() -> None:
    global _PIPELINE
    from . import imgui_backend

    if _PIPELINE is None:
        return
    renderer = imgui_backend.current()
    if renderer is not None:
        renderer.forget_texture(_PIPELINE.pong)
    _PIPELINE.release()
    _PIPELINE = None


def release_all() -> None:
    """Drop everything. Frame thread only, at teardown."""
    global _STALE, _USED
    _release_pipeline()
    _STALE = True
    _USED = False
