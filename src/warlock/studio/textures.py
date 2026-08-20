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
        # Keyed (key string, mtime, unfiltered): the file's version, and the
        # sampling baked into the decoded texture.
        self._entries: OrderedDict[tuple[str, float, bool], Any] = OrderedDict()
        self._missing: set[tuple[str, float, bool]] = set()
        # Which frame each entry was last handed out on, and the textures whose
        # release is waiting for the frame that drew them to finish. Both exist
        # for the same reason: a card asks for its texture during the UI build
        # and the pixels are not fetched until the backend draws, so releasing
        # inside that window frees something the draw list still points at.
        self._touched: dict[tuple[str, float, bool], int] = {}
        self._retired: list[Any] = []
        self._frame = 0
        # Per-key-string index over ``_entries``/``_missing`` (L101):
        # ``_supersede`` used to scan every key in both on every insert, which
        # made inserting into a big cache O(size) instead of O(versions of one
        # key). Maintained beside every insert and removal below.
        self._by_key: dict[str, set[tuple[str, float, bool]]] = {}
        self._missing_by_key: dict[str, set[tuple[str, float, bool]]] = {}
        # Frame-scoped stat() memo (B17): several panes ask for one thumbnail
        # in one frame, and the file cannot change mid-frame meaningfully.
        self._stats: dict[str, float | None] = {}

    def begin_frame(self) -> None:
        """Start a frame, releasing whatever last frame retired.

        The previous frame's draw list has been submitted and consumed by now,
        so anything evicted during it is finally safe to free.
        """
        self._frame += 1
        self._stats.clear()
        retired, self._retired = self._retired, []
        for texture in retired:
            self._release_one(texture)

    def get(
        self,
        job_id: str,
        path: Path,
        *,
        nearest: bool = False,
        max_side: int = MAX_SIDE,
    ) -> Any | None:
        """-> a texture for the job's thumb.png, or None if there isn't one.

        Decoding happens on the frame thread on purpose: it is one small PNG,
        it happens once per thumbnail for the life of the process, and the
        alternative is a placeholder that flickers for a frame.

        ``nearest`` samples the texture unfiltered -- what keeps a 32 px pixel
        artifact crisp when drawn at an integer multiple. It is part of the key
        because the filter is baked into the texture object at decode time: it
        held only by convention that a given key string was requested with one
        filter, and the failure -- a blurry pixel preview, or a crisp thumbnail
        somewhere else -- is silent and depends on which was decoded first.

        ``max_side`` is in the key for exactly that reason. This class was a
        *thumbnail* cache and 256 px was hardcoded; the manual's screenshots
        come through it too and a screenshot reduced to 256 px is unreadable,
        so the cap is a parameter -- and a cap that was not part of the key
        would hand back whichever size happened to be decoded first.
        """
        stat_key = str(path)
        if stat_key in self._stats:
            mtime = self._stats[stat_key]
        else:
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = None
            self._stats[stat_key] = mtime
        if mtime is None:
            return None
        key = (job_id, mtime, nearest, max_side)
        entry = self._entries.get(key)
        if entry is not None:
            self._entries.move_to_end(key)
            self._touched[key] = self._frame
            return entry
        if key in self._missing:
            return None
        texture = self._load(path, nearest, max_side)
        if texture is None:
            self._missing.add(key)
            self._missing_by_key.setdefault(job_id, set()).add(key)
            return None
        self._supersede(job_id, mtime)
        self._insert(key, texture)
        return texture

    def from_pixels(
        self, key: str, version: float, size: tuple[int, int], data: bytes, components: int = 3
    ) -> Any | None:
        """A texture for pixels that never were a file. -> it, or None.

        The matte preview is computed on a task thread and never touches the
        disk, so it has no path and no mtime -- but it wants every one of this
        cache's rules: the same deferred release (a texture handed out during a
        frame is still in that frame's draw list when the backend draws),
        ``_supersede`` so a recomputed preview retires the one it replaces
        rather than leaving it to LRU pressure, and the same registration and
        ``forget_texture`` pairing. ``version`` plays the mtime's part and is
        the caller's stamp -- for the preview, ``input.png``'s own mtime, which
        is exactly what "these pixels are out of date" means.
        """
        entry_key = (key, version, False)
        entry = self._entries.get(entry_key)
        if entry is not None:
            self._entries.move_to_end(entry_key)
            self._touched[entry_key] = self._frame
            return entry
        try:
            texture = self.ctx.texture(size, components, data)
            texture.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
            texture.repeat_x = texture.repeat_y = False
        except Exception:
            log.debug("could not upload %s", key, exc_info=True)
            return None
        self._supersede(key, version)
        self._insert(entry_key, texture)
        return texture

    def _insert(self, key: tuple[str, float, bool], texture: Any) -> None:
        self._entries[key] = texture
        self._touched[key] = self._frame
        self._by_key.setdefault(key[0], set()).add(key)
        self._evict()

    def _drop_entry(self, key: tuple[str, float, bool]) -> Any:
        texture = self._entries.pop(key)
        self._touched.pop(key, None)
        keys = self._by_key.get(key[0])
        if keys is not None:
            keys.discard(key)
            if not keys:
                del self._by_key[key[0]]
        return texture

    def _supersede(self, job_id: str, mtime: float) -> None:
        """Retire the *older versions* of one key string.

        The docstring's promise -- "a re-rendered thumbnail replaces itself
        without anyone having to invalidate anything" -- was only half true:
        the new mtime minted a new entry and the old one sat in ``_entries``
        until ordinary LRU pressure reached it. Harmless for a thumbnail
        rendered once, and not harmless for the pixel preview, where every turn
        of the palette knob left another dead 128x128 behind and evicted
        library thumbnails to make room for them.

        Version, not filter: two entries differing only in ``nearest`` describe
        the same bytes and both are live, so retiring by key string alone would
        make two panes drawing one file at two samplings evict each other every
        frame.
        """
        stale = [k for k in self._by_key.get(job_id, ()) if k[1] != mtime]
        for key in stale:
            self._retired.append(self._drop_entry(key))
        missing = self._missing_by_key.get(job_id)
        if missing:
            gone = {k for k in missing if k[1] != mtime}
            self._missing.difference_update(gone)
            missing.difference_update(gone)
            if not missing:
                del self._missing_by_key[job_id]

    def _load(
        self, path: Path, nearest: bool = False, max_side: int = MAX_SIDE
    ) -> Any | None:
        from PIL import Image

        try:
            with Image.open(path) as im:
                im = im.convert("RGBA")
                im.thumbnail((max_side, max_side))
                texture = self.ctx.texture(im.size, 4, im.tobytes())
        except Exception:
            log.debug("could not decode %s", path, exc_info=True)
            return None
        mode = self.ctx.NEAREST if nearest else self.ctx.LINEAR
        texture.filter = (mode, mode)
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
            self._retired.append(self._drop_entry(key))

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
        self._by_key.clear()
        self._missing_by_key.clear()
        self._stats.clear()
