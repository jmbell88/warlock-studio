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
            return entry
        if key in self._missing:
            return None
        texture = self._load(path)
        if texture is None:
            self._missing.add(key)
            return None
        self._entries[key] = texture
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
        while len(self._entries) > self.limit:
            _key, texture = self._entries.popitem(last=False)
            texture.release()

    def release(self) -> None:
        for texture in self._entries.values():
            texture.release()
        self._entries.clear()
        self._missing.clear()
