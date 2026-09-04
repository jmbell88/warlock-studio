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
import logging
from typing import Any

import moderngl
import numpy as np
import pygame
from imgui_bundle import imgui

log = logging.getLogger(__name__)

#: Where ``_note_registry`` starts complaining. The font atlas is one, the two
#: bounded caches account for a few hundred between them, and every other
#: registration is paired with a ``forget_texture`` -- so anything past this is
#: a pairing that has come undone rather than a busy session.
TEXTURE_WARN_AT = 512

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
        # The point at which the registry's size is worth a log line. See
        # ``_note_registry``; it starts above anything the two bounded caches
        # plus the font atlas can account for.
        self._warn_at = TEXTURE_WARN_AT

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

    def _note_registry(self) -> None:
        """Say so, once, when the texture registry outgrows what is plausible.

        There is no sweep here and there deliberately is not one: an entry is a
        GL name mapped to a live moderngl object, and the renderer has no way
        to ask whether the *owner* still wants it -- guessing would free a
        texture a pane is about to draw. What was missing was any signal at
        all. One missed ``forget_texture`` is a permanent leak of one texture
        with nothing anywhere saying so, and the failure is silent right up to
        the point where the driver refuses an allocation.

        The threshold doubles each time it fires, so a genuinely large session
        says this three or four times over its life rather than once a frame.
        Every caller of ``register_texture`` pairs it with ``forget_texture``
        through ``docmodes.forget_texture``, and both texture caches are
        bounded, so the steady state is a few hundred at most.
        """
        if len(self._textures) <= self._warn_at:
            return
        log.warning(
            "imgui backend holds %d registered textures -- something is "
            "registering without forgetting",
            len(self._textures),
        )
        # Diagnostic-only: this raises the bar rather than resetting it, so a
        # steady leak is mentioned at _warn_at, 2x, 4x, 8x... and then, once
        # the doublings outrun a realistic session, effectively never again.
        # That is intentional -- the point of this method is a log line, not
        # a mitigation, and there is no sweep to re-arm (see the docstring
        # above: guessing which entries are dead would free a texture a pane
        # is about to draw). Do not mistake the doubling for a rate limiter
        # that keeps re-triggering, and do not add a reset-on-shrink here --
        # that would need a notion of "shrink" this registry does not have.
        self._warn_at = len(self._textures) * 2

    def register_texture(self, texture: moderngl.Texture) -> int:
        """Make a viewer-owned texture showable with ``imgui.image``.

        The id is the GL name, which is what ``ImTextureID`` is here -- but the
        renderer binds through moderngl, which has no "bind this raw id", so it
        needs the object as well. Overwritten on every call rather than
        inserted once: a texture that was released and whose name the driver
        handed to a different one would otherwise be bound from a stale entry.
        """
        self._textures[texture.glo] = texture
        self._note_registry()
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

        # The three Phase 5 caches hold textures made from *this* context and
        # registered against *this* renderer, so they die with it -- here
        # rather than in App.teardown, because the smoke suite builds and drops
        # renderers without ever building an App, and a sprite left behind
        # would be handed to the next renderer as a live GL name it does not own.
        from . import shadows, surfaces, vibrancy

        shadows.release_all()
        surfaces.release_all()
        vibrancy.release_all()
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


def _as_bytes(address: int, size: int) -> Any:
    """imgui hands out a raw pointer; moderngl wants a buffer.

    ``from_address`` rather than ``ctypes.string_at``: the latter *copies* the
    whole vertex or index buffer into a fresh ``bytes`` before moderngl copies
    it again into the VBO, once per command list per frame, for nothing. A
    ctypes array over the same address supports the buffer protocol, so the
    upload reads imgui's own memory. Nothing outlives the ``write`` below, so
    there is no lifetime to keep: imgui owns the storage until the next frame.
    """
    return (ctypes.c_ubyte * size).from_address(address)


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
    pygame.K_CAPSLOCK: imgui.Key.caps_lock,
    pygame.K_MENU: imgui.Key.menu,
    pygame.K_PRINTSCREEN: imgui.Key.print_screen,
    pygame.K_PAUSE: imgui.Key.pause,
    pygame.K_SCROLLLOCK: imgui.Key.scroll_lock,
    pygame.K_NUMLOCKCLEAR: imgui.Key.num_lock,
    # The whole alphabet, the digit row, the function row, the keypad and the
    # punctuation, rather than the six letters the clipboard and undo chords
    # happened to need (UX-02). A key imgui is never told about is one no
    # shortcut inside a text field can use, no nav step can read, and no
    # ``is_key_pressed`` in a pane can answer for -- and which of the 104 keys
    # were mapped was, before this, a record of which features were built
    # first rather than of anything about the keyboard.
    **{
        getattr(pygame, f"K_{c}"): getattr(imgui.Key, c)
        for c in "abcdefghijklmnopqrstuvwxyz"
    },
    **{getattr(pygame, f"K_{d}"): getattr(imgui.Key, f"_{d}") for d in "0123456789"},
    **{getattr(pygame, f"K_F{n}"): getattr(imgui.Key, f"f{n}") for n in range(1, 13)},
    **{
        getattr(pygame, f"K_KP{d}"): getattr(imgui.Key, f"keypad{d}")
        for d in "0123456789"
    },
    pygame.K_KP_PLUS: imgui.Key.keypad_add,
    pygame.K_KP_MINUS: imgui.Key.keypad_subtract,
    pygame.K_KP_MULTIPLY: imgui.Key.keypad_multiply,
    pygame.K_KP_DIVIDE: imgui.Key.keypad_divide,
    pygame.K_KP_PERIOD: imgui.Key.keypad_decimal,
    pygame.K_MINUS: imgui.Key.minus,
    pygame.K_EQUALS: imgui.Key.equal,
    pygame.K_LEFTBRACKET: imgui.Key.left_bracket,
    pygame.K_RIGHTBRACKET: imgui.Key.right_bracket,
    pygame.K_BACKSLASH: imgui.Key.backslash,
    pygame.K_SEMICOLON: imgui.Key.semicolon,
    pygame.K_QUOTE: imgui.Key.apostrophe,
    pygame.K_BACKQUOTE: imgui.Key.grave_accent,
    pygame.K_COMMA: imgui.Key.comma,
    pygame.K_PERIOD: imgui.Key.period,
    pygame.K_SLASH: imgui.Key.slash,
    pygame.K_LCTRL: imgui.Key.left_ctrl,
    pygame.K_RCTRL: imgui.Key.right_ctrl,
    pygame.K_LALT: imgui.Key.left_alt,
    pygame.K_RALT: imgui.Key.right_alt,
    pygame.K_LSHIFT: imgui.Key.left_shift,
    pygame.K_RSHIFT: imgui.Key.right_shift,
    # The Windows/Command pair and the menu key. imgui has names for all three
    # and the table simply did not carry them, so a chord holding Super was a
    # chord imgui could not see -- and ``K_APPLICATION``, which is what a
    # keyboard's context-menu key sends, arrived as nothing at all. That key
    # is ``K_MENU`` here and not ``K_APPLICATION``: pygame-ce has no such
    # name, which the review's entry assumed it did.
    pygame.K_LGUI: imgui.Key.left_super,
    pygame.K_RGUI: imgui.Key.right_super,
    pygame.K_MENU: imgui.Key.menu,
}

_MODIFIER_MAP = {
    pygame.K_LCTRL: imgui.Key.mod_ctrl,
    pygame.K_RCTRL: imgui.Key.mod_ctrl,
    pygame.K_LSHIFT: imgui.Key.mod_shift,
    pygame.K_RSHIFT: imgui.Key.mod_shift,
    pygame.K_LALT: imgui.Key.mod_alt,
    pygame.K_RALT: imgui.Key.mod_alt,
    pygame.K_LGUI: imgui.Key.mod_super,
    pygame.K_RGUI: imgui.Key.mod_super,
}

# pygame's button numbers, and *only* the three imgui names. It is a complete
# whitelist rather than a two-entry patch over ``button - 1`` because pygame
# numbers more buttons than imgui has: 4 and 5 are the legacy wheel notches and
# 6/7 are a mouse's side buttons (SDL's X1/X2, the back/forward pair on most
# gaming mice). ``button - 1`` sent those in as 5 and 6, and imgui asserts on
# anything past ImGuiMouseButton_COUNT (5) -- a stray thumb click on the side
# of the mouse took the whole frame loop down with a RuntimeError.
_MOUSE_MAP = {
    1: imgui.MouseButton_.left,
    2: imgui.MouseButton_.middle,
    3: imgui.MouseButton_.right,
}

# What a physical wheel notch becomes in ``io.mouse_wheel``. Halved because one
# SDL notch scrolled a list two rows at a time, and it is applied here so that
# every scroller in the app inherits it from one place.
#
# It is a *named* constant rather than a literal because it is not private to
# this module in one respect: a pane that wants notches back (Inker's canvas
# steps the zoom by a fixed 5% per notch, not by a fraction of one) has to
# divide it out, and doing that against a bare ``0.5`` would be a second copy
# of this decision that nothing would update if the scroll speed ever changed.
WHEEL_SCALE = 0.5

# The keys imgui's navigation moves and activates on.
#
# **Tab is deliberately not in the set.** That is the whole rule keyboard
# access settles on here (UX-02): *Tab traverses, and the arrows belong to
# whatever surface binds them.* Home's tiles, the library list and Review all
# bind Up/Down or Left/Right, and Inker and Plotter hold Space to pan -- so
# with imgui nav simply switched on, one press both moved the app's own
# selection and stepped a focus ring nobody asked for. Suppressing those keys
# at the one door they come through is what lets nav be on everywhere without
# a second owner for any key, and it beats flagging ``no_nav_inputs`` on each
# window because the surfaces that bind arrows are not all child windows.
_NAV_KEYS = frozenset(
    {
        pygame.K_LEFT,
        pygame.K_RIGHT,
        pygame.K_UP,
        pygame.K_DOWN,
        pygame.K_SPACE,
        pygame.K_PAGEUP,
        pygame.K_PAGEDOWN,
        pygame.K_HOME,
        pygame.K_END,
    }
)

# Set once a frame by the app, from whether the surface on screen binds them.
_nav_keys_reserved = False


def reserve_nav_keys(reserved: bool) -> None:
    """Declare that the surface on screen owns the arrows and Space."""
    global _nav_keys_reserved
    _nav_keys_reserved = bool(reserved)


def _forwards(event: Any, io: Any) -> bool:
    """Whether one key event should reach imgui at all.

    A reservation never applies while a text field has the keyboard: Home, End
    and the arrows are caret movement there, and a mode that binds Up/Down for
    its list has no claim on them while somebody is typing into a prompt.
    """
    if not _nav_keys_reserved or event.key not in _NAV_KEYS:
        return True
    return bool(io.want_text_input)


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
        button = _MOUSE_MAP.get(event.button)
        if button is None:
            # A button imgui has no name for. Dropped rather than clamped: the
            # app's own handlers still see the event below, and folding a side
            # button onto left would make it click whatever is under the cursor.
            return False
        io.add_mouse_button_event(button, event.type == pygame.MOUSEBUTTONDOWN)
        return True
    if event.type == pygame.MOUSEWHEEL:
        io.add_mouse_wheel_event(event.x * WHEEL_SCALE, event.y * WHEEL_SCALE)
        return True
    if event.type == pygame.TEXTINPUT:
        # SDL's *committed* text, and since UX-19 the only source of characters
        # in the app. It was ``KEYDOWN.unicode`` before, which is a different
        # and worse fact: a keypress carries the character that key produces on
        # its own, so an IME's composed result never arrived (the keys that
        # composed it did, one raw letter at a time, into the field the user
        # was composing *for*), a dead-key accent double-typed, and anything
        # needing more than one keystroke to name one character could not be
        # written at all.
        io.add_input_characters_utf8(event.text)
        return True
    if event.type == pygame.TEXTEDITING:
        # The IME's pre-edit buffer -- what is being composed but not yet
        # committed. Swallowed rather than inserted: imgui has no pre-edit
        # rendering, so adding these characters would put the half-finished
        # composition into the field and then leave it there when the real
        # TEXTINPUT arrived. Consumed rather than ignored so the shortcut layer
        # does not see a composition in progress as a chord.
        return True
    if event.type in (pygame.KEYDOWN, pygame.KEYUP):
        down = event.type == pygame.KEYDOWN
        # Modifiers are forwarded even when the nav keys are not: a mode owning
        # Up/Down has no claim on Shift, and imgui reading a chord as unmodified
        # is how Shift+Tab stops going backwards.
        if event.key in _MODIFIER_MAP:
            io.add_key_event(_MODIFIER_MAP[event.key], down)
        if event.key in _KEY_MAP and _forwards(event, io):
            io.add_key_event(_KEY_MAP[event.key], down)
        return False
    return False


# Whether SDL is currently generating TEXTINPUT events. Tracked rather than
# queried because pygame exposes no getter, and calling start/stop every frame
# restarts the IME composition on some backends.
_text_input_on: bool | None = None


def sync_text_input(pygame_module: Any = pygame) -> bool:
    """Turn SDL's text input on exactly while imgui wants it. -> the new state.

    Called once a frame, after ``imgui.render()``. Two things follow from it,
    and the second is the reason it exists rather than the app simply leaving
    text input on forever: SDL only emits ``TEXTINPUT`` while it is started, so
    this is what makes typing work at all; and an IME's candidate window pops up
    whenever text input is active, so leaving it on would put a Japanese
    composition bar over the viewport every time somebody pressed a key to
    orbit the camera.
    """
    global _text_input_on
    want = bool(imgui.get_io().want_text_input)
    if want != _text_input_on:
        if want:
            pygame_module.key.start_text_input()
        else:
            pygame_module.key.stop_text_input()
        _text_input_on = want
    return want


def set_ime_rect(
    pos: tuple[float, float], size: tuple[float, float], pygame_module: Any = pygame
) -> None:
    """Tell SDL where the text being composed is, so the IME can sit beside it.

    Opt-in, per call site, and that is a limitation rather than a design: Dear
    ImGui knows exactly where its caret is and hands the position to a backend
    through ``Platform_SetImeDataFn``, but imgui_bundle exposes neither that
    callback nor the context's ``PlatformImeData``, so the caret is unreachable
    from Python. What *is* reachable is the rect of the item just drawn, which
    is the field rather than the caret inside it -- close enough that the
    candidate window lands on the right control, and the best available until
    the binding grows the hook.
    """
    rect = pygame_module.Rect(int(pos[0]), int(pos[1]), int(size[0]), int(size[1]))
    pygame_module.key.set_text_input_rect(rect)
