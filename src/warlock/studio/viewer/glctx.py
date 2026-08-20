"""Framebuffers: a multisampled target, its resolve, and how to read it back.

Everything the viewer draws goes into an offscreen pair rather than the window:
a 4x multisampled colour+depth buffer to draw into, and a plain texture to
resolve into, which is what imgui then shows as an image. That indirection is
what lets the same code path serve the viewport, a compare pane, a sheet
preview cell and a golden-image test -- none of which is the screen.
"""

from __future__ import annotations

import moderngl
import numpy as np

# 4x is where the cost/benefit turns over on a wireframe grid, which is the
# thinnest thing drawn and the reason MSAA is on at all: three's own antialias
# flag is what the browser build used, and its lines have no other AA either.
DEFAULT_SAMPLES = 4


class Viewport:
    """A resizable multisampled render target with a resolved texture.

    Reallocated only when the requested size actually changes -- a docked panel
    reports its size every frame, and rebuilding four attachments per frame
    would dominate the frame budget for no visible difference.
    """

    def __init__(self, ctx: moderngl.Context, size: tuple[int, int], samples: int | None = None):
        self.ctx = ctx
        self.samples = DEFAULT_SAMPLES if samples is None else samples
        self.size = (0, 0)
        self._msaa: moderngl.Framebuffer | None = None
        self._resolve: moderngl.Framebuffer | None = None
        self.texture: moderngl.Texture | None = None
        # The renderbuffers behind ``_msaa``, held by name because releasing a
        # framebuffer does not release its attachments (and the app sets no
        # gc_mode, so a dropped reference frees nothing either). Built inline,
        # they leaked a colour + depth pair per reallocation -- and a splitter
        # drag reallocates on nearly every frame.
        self._color_rb: moderngl.Renderbuffer | None = None
        self._depth_rb: moderngl.Renderbuffer | None = None
        self.resize(size)

    def resize(self, size: tuple[int, int]) -> bool:
        """-> whether anything was reallocated."""
        width = max(1, int(size[0]))
        height = max(1, int(size[1]))
        if (width, height) == self.size:
            return False
        self.release()
        self.size = (width, height)
        self.texture = self.ctx.texture((width, height), 4)
        self.texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.texture.repeat_x = self.texture.repeat_y = False
        self._resolve = self.ctx.framebuffer(color_attachments=[self.texture])
        if self.samples > 1:
            # A multisample texture cannot be sampled by an ordinary sampler2D,
            # which is why there are two framebuffers and a blit rather than
            # one target used both ways.
            self._color_rb = self.ctx.renderbuffer(
                (width, height), 4, samples=self.samples
            )
            self._depth_rb = self.ctx.depth_renderbuffer(
                (width, height), samples=self.samples
            )
            self._msaa = self.ctx.framebuffer(
                color_attachments=[self._color_rb],
                depth_attachment=self._depth_rb,
            )
        else:
            self._depth_rb = self.ctx.depth_renderbuffer((width, height))
            self._msaa = self.ctx.framebuffer(
                color_attachments=[self.texture],
                depth_attachment=self._depth_rb,
            )
        return True

    @property
    def draw_target(self) -> moderngl.Framebuffer:
        assert self._msaa is not None
        return self._msaa

    def use(self) -> None:
        self.draw_target.use()
        self.ctx.viewport = (0, 0, *self.size)

    def resolve(self) -> moderngl.Texture:
        """Flatten the samples down into the texture imgui can show."""
        assert self._resolve is not None and self.texture is not None
        if self.samples > 1:
            self.ctx.copy_framebuffer(self._resolve, self._msaa)
        return self.texture

    def read_raw(self) -> bytes:
        """The resolved pixels as GL hands them back: RGBA8, bottom row first.

        For consumers that can fold the row flip into their own decode
        (Pillow's raw loader takes a negative orientation), so the numpy
        reshape-flip-copy in :meth:`read_rgba` is not paid twice (D41).
        """
        assert self._resolve is not None
        self.resolve()
        return self._resolve.read(components=4, alignment=1)

    def read_rgba(self) -> np.ndarray:
        """-> (h, w, 4) uint8, top row first.

        GL reads bottom-up; every consumer here (Pillow, a golden PNG, a sheet
        cell) counts rows from the top, so the flip happens once, here, rather
        than in each of them.
        """
        assert self._resolve is not None
        self.resolve()
        raw = self._resolve.read(components=4, alignment=1)
        image = np.frombuffer(raw, dtype=np.uint8).reshape(self.size[1], self.size[0], 4)
        return image[::-1].copy()

    def release(self) -> None:
        # Attachments after their framebuffers: the renderbuffers must be
        # released by name, because they are exactly what a framebuffer's
        # ``release`` leaves behind.
        for obj in (self._msaa, self._resolve, self.texture, self._color_rb, self._depth_rb):
            if obj is not None:
                obj.release()
        self._msaa = self._resolve = self.texture = None
        self._color_rb = self._depth_rb = None
        self.size = (0, 0)

