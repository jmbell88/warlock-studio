"""A bounded cache of thumbnail textures.

Keyed on (job id, mtime) so a re-rendered thumbnail replaces itself without
anyone having to invalidate anything. Bounded because a workshop accumulates
hundreds of jobs and a 256x256 RGBA texture each is real VRAM -- the library
only ever shows a screenful, so the eviction is on total count and the least
recently drawn goes first.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

MAX_TEXTURES = 120
# Thumbnails are drawn at card width; anything larger is memory spent on
# detail the card cannot show.
MAX_SIDE = 256


class ThumbnailCache:
    def __init__(self, ctx: Any, limit: int = MAX_TEXTURES) -> None:
        self.ctx = ctx
        self.limit = limit
        self._entries: OrderedDict[tuple[str, float], Any] = OrderedDict()
        self._missing: set[tuple[str, float]] = set()
        # Which frame each entry was last handed out on, and the textures whose
        # release is waiting for the frame that drew them to finish. Both exist
        # for the same reason: a card asks for its texture during the UI build
        # and the pixels are not fetched until the backend draws, so releasing
        # inside that window frees something the draw list still points at.
        self._touched: dict[tuple[str, float], int] = {}
        self._retired: list[Any] = []
        self._frame = 0

    def begin_frame(self) -> None:
        """Start a frame, releasing whatever last frame retired.

        The previous frame's draw list has been submitted and consumed by now,
        so anything evicted during it is finally safe to free.
        """
        self._frame += 1
        retired, self._retired = self._retired, []
        for texture in retired:
            self._release_one(texture)

    def get(self, job_id: str, path: Path) -> Any | None:
        """-> a texture for the job's thumb.png, or None if there isn't one.

        Decoding happens on the frame thread on purpose: it is one small PNG,
        it happens once per thumbnail for the life of the process, and the
        alternative is a placeholder that flickers for a frame.
        """
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return None
        key = (job_id, mtime)
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
            self._touched[key] = self._frame
            return entry
        if key in self._missing:
            return None
        texture = self._load(path)
        if texture is None:
            self._missing.add(key)
            return None
        self._entries[key] = texture
        self._touched[key] = self._frame
        self._evict()
        return texture

    def _load(self, path: Path) -> Any | None:
        from PIL import Image

        try:
            with Image.open(path) as im:
                im = im.convert("RGBA")
                im.thumbnail((MAX_SIDE, MAX_SIDE))
                texture = self.ctx.texture(im.size, 4, im.tobytes())
        except Exception:
            log.debug("could not decode %s", path, exc_info=True)
            return None
        texture.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
        texture.repeat_x = texture.repeat_y = False
        return texture

    def _evict(self) -> None:
        """Retire least-recently-used entries, but never one drawn this frame.

        Skipping the current frame is what stops the cache thrashing when more
        thumbnails are on screen than it holds: evicting card 1 to make room for
        card 121 would free a texture card 1 has already queued a draw for, and
        then re-decode it next frame to do the same thing again. Overshooting
        the limit for one frame is the cheaper answer -- the overshoot is
        bounded by what fits on screen, and it drains as soon as the list
        scrolls.
        """
        for key in list(self._entries):
            if len(self._entries) <= self.limit:
                return
            if self._touched.get(key) == self._frame:
                continue
            self._retired.append(self._entries.pop(key))
            self._touched.pop(key, None)

    def _release_one(self, texture: Any) -> None:
        """Drop the imgui backend's registration before freeing the texture.

        The backend maps GL names to moderngl objects; releasing without
        forgetting leaves it holding a dead object under a name the driver is
        free to hand to the next texture created, which is how an unrelated
        image starts rendering as this one.
        """
        from . import imgui_backend

        renderer = imgui_backend.current()
        if renderer is not None:
            renderer.forget_texture(texture)
        texture.release()

    def release(self) -> None:
        for texture in self._entries.values():
            self._release_one(texture)
        for texture in self._retired:
            self._release_one(texture)
        self._entries.clear()
        self._retired.clear()
        self._touched.clear()
        self._missing.clear()
