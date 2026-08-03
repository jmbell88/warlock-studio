"""imgui rendering and input, on moderngl and pygame.

imgui-bundle ships a pygame backend, but it draws through PyOpenGL. Every raw
``glBindTexture``/``glUseProgram`` it makes goes behind moderngl's back, and
moderngl caches state -- so the viewport would render with whatever the UI
happened to leave bound. This backend issues the same draw calls through the
same moderngl context the viewer uses, which is what makes "one context, panels
and 3D together" actually true rather than merely intended.

It is a direct port of ``ImGui_ImplOpenGL3_RenderDrawData`` plus the bundle's
own pygame event mapping; the interesting parts are the texture bridge (imgui
hands out GL ids, moderngl hands out objects, so we keep both) and the
deliberate refusal to advertise ``renderer_has_vtx_offset`` -- moderngl has no
base-vertex draw, and without the flag imgui never emits a non-zero one.
"""

from __future__ import annotations

import ctypes
from typing import Any

import moderngl
import numpy as np
import pygame
from imgui_bundle import imgui

VERTEX_SHADER = """
#version 330 core
uniform mat4 ProjMtx;
in vec2 Position;
in vec2 UV;
in vec4 Color;
out vec2 Frag_UV;
out vec4 Frag_Color;
void main() {
    Frag_UV = UV;
    Frag_Color = Color;
    gl_Position = ProjMtx * vec4(Position.xy, 0, 1);
}
"""

FRAGMENT_SHADER = """
#version 330 core
uniform sampler2D Texture;
in vec2 Frag_UV;
in vec4 Frag_Color;
out vec4 Out_Color;
void main() {
    Out_Color = Frag_Color * texture(Texture, Frag_UV.st);
}
"""


# The renderer in use, so ``widgets.texture_ref`` can register a texture from
# anywhere in the UI. A module-level current-thing mirrors imgui's own current
# context, and for the same reason: the alternative is threading the renderer
# through every pane that shows a picture.
_current: ImguiRenderer | None = None


def current() -> ImguiRenderer | None:
    return _current


class ImguiRenderer:
    """Draws imgui's vertex soup through a moderngl context."""

    def __init__(self, ctx: moderngl.Context) -> None:
        global _current
        if imgui.get_current_context() is None:
            raise RuntimeError("create an imgui context before the renderer")
        _current = self
        self.ctx = ctx
        self.io = imgui.get_io()
        self.io.delta_time = 1.0 / 60.0
        self.prog = ctx.program(
            vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER
        )
        # Reserved once and grown as needed: imgui's vertex count swings with
        # how much is on screen, and reallocating a buffer per frame is the
        # one avoidable per-frame allocation in the whole loop.
        self._vbo = ctx.buffer(reserve=64 * 1024, dynamic=True)
        self._ibo = ctx.buffer(reserve=32 * 1024, dynamic=True)
        self._vao = ctx.vertex_array(
            self.prog,
            [(self._vbo, "2f 2f 4f1", "Position", "UV", "Color")],
            index_buffer=self._ibo,
            index_element_size=4,
        )
        # imgui identifies textures by GL id; moderngl identifies them by
        # object. Both directions are needed: imgui tells us an id to bind, and
        # only the object can bind itself.
        self._textures: dict[int, moderngl.Texture] = {}

        self.io.backend_flags |= imgui.BackendFlags_.renderer_has_textures.value
        limit = ctx.info["GL_MAX_TEXTURE_SIZE"]
        imgui.get_platform_io().renderer_texture_max_width = limit
        imgui.get_platform_io().renderer_texture_max_height = limit

    # -- textures ----------------------------------------------------------

    def _update_textures(self) -> None:
        for tex in imgui.get_platform_io().textures:
            if tex.status != imgui.ImTextureStatus.ok:
                self._update_texture(tex)

    def _update_texture(self, tex: Any) -> None:
        if tex.status == imgui.ImTextureStatus.want_create:
            pixels = np.asarray(tex.get_pixels_array(), dtype=np.uint8)
            gl_tex = self.ctx.texture((tex.width, tex.height), 4, pixels.tobytes())
            # Bilinear is required by default: imgui bakes anti-aliased line
            # geometry into the atlas and point sampling shreds it.
            gl_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            gl_tex.repeat_x = gl_tex.repeat_y = False
            self._textures[gl_tex.glo] = gl_tex
            tex.set_tex_id(gl_tex.glo)
            tex.status = imgui.ImTextureStatus.ok
        elif tex.status == imgui.ImTextureStatus.want_updates:
            gl_tex = self._textures.get(tex.tex_id)
            if gl_tex is None:
                return
            full = np.asarray(tex.get_pixels_array(), dtype=np.uint8).reshape(
                tex.height, tex.width, tex.bytes_per_pixel
            )
            for r in tex.updates:
                # Copied rather than sliced in place: moderngl writes a flat
                # buffer with no row stride, so the sub-rect has to be
                # contiguous before it goes anywhere.
                block = np.ascontiguousarray(full[r.y : r.y + r.h, r.x : r.x + r.w])
                gl_tex.write(block.tobytes(), viewport=(r.x, r.y, r.w, r.h))
            tex.status = imgui.ImTextureStatus.ok
        elif tex.status == imgui.ImTextureStatus.want_destroy:
            gl_tex = self._textures.pop(tex.tex_id, None)
            if gl_tex is not None:
                gl_tex.release()
            tex.set_tex_id(0)
            tex.status = imgui.ImTextureStatus.destroyed

    def register_texture(self, texture: moderngl.Texture) -> int:
        """Make a viewer-owned texture showable with ``imgui.image``.

        The id is the GL name, which is what ``ImTextureID`` is here -- but the
        renderer binds through moderngl, which has no "bind this raw id", so it
        needs the object as well. Overwritten on every call rather than
        inserted once: a texture that was released and whose name the driver
        handed to a different one would otherwise be bound from a stale entry.
        """
        self._textures[texture.glo] = texture
        return texture.glo

    def forget_texture(self, texture: moderngl.Texture) -> None:
        self._textures.pop(texture.glo, None)

    # -- drawing -----------------------------------------------------------

    def render(self, draw_data: Any) -> None:
        io = self.io
        display_w, display_h = io.display_size
        scale_x, scale_y = io.display_framebuffer_scale
        fb_w, fb_h = int(display_w * scale_x), int(display_h * scale_y)

        self._update_textures()
        if fb_w == 0 or fb_h == 0:
            return
        draw_data.scale_clip_rects(io.display_framebuffer_scale)

        self.ctx.screen.use()
        # Scissor is not in this set: moderngl turns GL_SCISSOR_TEST on and off
        # with the ``scissor`` property itself, and listing it here would be a
        # flag it does not define.
        self.ctx.enable_only(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self.ctx.viewport = (0, 0, fb_w, fb_h)
        self.ctx.wireframe = False

        # Column-major, y down: imgui's coordinates put the origin top-left.
        self.prog["ProjMtx"].write(
            np.array(
                [
                    [2.0 / display_w, 0.0, 0.0, 0.0],
                    [0.0, 2.0 / -display_h, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [-1.0, 1.0, 0.0, 1.0],
                ],
                dtype="f4",
            ).tobytes()
        )
        self.prog["Texture"].value = 0

        for commands in draw_data.cmd_lists:
            n_vtx = commands.vtx_buffer.size()
            n_idx = commands.idx_buffer.size()
            vtx_bytes = n_vtx * imgui.VERTEX_SIZE
            idx_bytes = n_idx * imgui.INDEX_SIZE
            if vtx_bytes > self._vbo.size:
                self._vbo.orphan(max(vtx_bytes, self._vbo.size * 2))
            if idx_bytes > self._ibo.size:
                self._ibo.orphan(max(idx_bytes, self._ibo.size * 2))
            self._vbo.write(_as_bytes(commands.vtx_buffer.data_address(), vtx_bytes))
            self._ibo.write(_as_bytes(commands.idx_buffer.data_address(), idx_bytes))

            for command in commands.cmd_buffer:
                gl_tex = self._textures.get(command.tex_ref.get_tex_id())
                if gl_tex is not None:
                    gl_tex.use(0)
                x, y, z, w = command.clip_rect
                self.ctx.scissor = (int(x), int(fb_h - w), int(z - x), int(w - y))
                self._vao.render(
                    moderngl.TRIANGLES,
                    vertices=command.elem_count,
                    first=command.idx_offset,
                )
        self.ctx.scissor = None

    def shutdown(self) -> None:
        global _current
        if _current is self:
            _current = None
        for tex in imgui.get_platform_io().textures:
            if tex.ref_count == 1:
                tex.status = imgui.ImTextureStatus.want_destroy
                self._update_texture(tex)
        self.io.backend_flags &= ~imgui.BackendFlags_.renderer_has_textures.value
        self._vao.release()
        self._vbo.release()
        self._ibo.release()
        self.prog.release()


def _as_bytes(address: int, size: int) -> bytes:
    """imgui hands out a raw pointer; moderngl wants a buffer."""
    return ctypes.string_at(address, size)


# --- input ------------------------------------------------------------------

_KEY_MAP = {
    pygame.K_LEFT: imgui.Key.left_arrow,
    pygame.K_RIGHT: imgui.Key.right_arrow,
    pygame.K_UP: imgui.Key.up_arrow,
    pygame.K_DOWN: imgui.Key.down_arrow,
    pygame.K_PAGEUP: imgui.Key.page_up,
    pygame.K_PAGEDOWN: imgui.Key.page_down,
    pygame.K_HOME: imgui.Key.home,
    pygame.K_END: imgui.Key.end,
    pygame.K_INSERT: imgui.Key.insert,
    pygame.K_DELETE: imgui.Key.delete,
    pygame.K_BACKSPACE: imgui.Key.backspace,
    pygame.K_SPACE: imgui.Key.space,
    pygame.K_RETURN: imgui.Key.enter,
    pygame.K_ESCAPE: imgui.Key.escape,
    pygame.K_KP_ENTER: imgui.Key.keypad_enter,
    pygame.K_TAB: imgui.Key.tab,
    pygame.K_a: imgui.Key.a,
    pygame.K_c: imgui.Key.c,
    pygame.K_v: imgui.Key.v,
    pygame.K_x: imgui.Key.x,
    pygame.K_y: imgui.Key.y,
    pygame.K_z: imgui.Key.z,
    pygame.K_LCTRL: imgui.Key.left_ctrl,
    pygame.K_RCTRL: imgui.Key.right_ctrl,
    pygame.K_LALT: imgui.Key.left_alt,
    pygame.K_RALT: imgui.Key.right_alt,
    pygame.K_LSHIFT: imgui.Key.left_shift,
    pygame.K_RSHIFT: imgui.Key.right_shift,
}

_MODIFIER_MAP = {
    pygame.K_LCTRL: imgui.Key.mod_ctrl,
    pygame.K_RCTRL: imgui.Key.mod_ctrl,
    pygame.K_LSHIFT: imgui.Key.mod_shift,
    pygame.K_RSHIFT: imgui.Key.mod_shift,
    pygame.K_LALT: imgui.Key.mod_alt,
    pygame.K_RALT: imgui.Key.mod_alt,
}

_MOUSE_MAP = {2: imgui.MouseButton_.middle, 3: imgui.MouseButton_.right}


def process_event(event: Any) -> bool:
    """Feed one pygame event to imgui. -> whether imgui consumed it.

    "Consumed" here means "imgui knows about it", not "nobody else may see
    it": the caller still decides, from ``io.want_capture_mouse`` and friends,
    whether the viewport gets a look. A key event is deliberately reported as
    *not* consumed so the shortcut layer can see it alongside imgui.
    """
    io = imgui.get_io()
    if event.type == pygame.MOUSEMOTION:
        io.add_mouse_pos_event(event.pos[0], event.pos[1])
        return True
    if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP):
        down = event.type == pygame.MOUSEBUTTONDOWN
        io.add_mouse_button_event(_MOUSE_MAP.get(event.button, event.button - 1), down)
        return True
    if event.type == pygame.MOUSEWHEEL:
        io.add_mouse_wheel_event(event.x * 0.5, event.y * 0.5)
        return True
    if event.type in (pygame.KEYDOWN, pygame.KEYUP):
        down = event.type == pygame.KEYDOWN
        if event.key in _KEY_MAP:
            io.add_key_event(_KEY_MAP[event.key], down)
        if event.key in _MODIFIER_MAP:
            io.add_key_event(_MODIFIER_MAP[event.key], down)
        if down:
            # The key event and the character are two separate facts, and a
            # letter is both: Ctrl+V needs Key.v, and typing "v" needs the
            # character. Filtering on isprintable() rather than on "was this
            # key mapped" is what lets both be true -- Return and Backspace
            # carry a control character in event.unicode and must not reach
            # the text field as one.
            for char in getattr(event, "unicode", ""):
                if char.isprintable() and 0 < ord(char) < 0x10000:
                    io.add_input_character(ord(char))
        return False
    return False
