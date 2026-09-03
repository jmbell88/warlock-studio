"""Multi-document state for Packwright, without imgui.

The ``inker_state`` / ``plotter_state`` split, third instance, and the view type
is imported for the same reason it is there: a zoomable, pannable 2D viewport is
not about pixels, and a second copy would be a second set of clamping rules and
a second Ctrl+0.

**The atlas is a task result, not a document field.** ``PackDoc`` holds sources
and settings and derives its layout on demand -- that is what makes a re-export
reproducible -- but *composing* the atlas is numpy work over every sprite, which
is not frame-thread work at a hundred of them. So the composed pixels and the
layout they came from live here, on the tab, adopted when the task lands.

**``pack_generation`` is what the texture cache keys on.** A repack replaces the
pixels wholesale, and an upload gated on anything else -- a rev, a hash, the
settings -- either re-uploads a megapixel every frame or misses a change. A
counter bumped in exactly one place is the only version of this that is both.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import docmodes
from .inker_state import PaintView

WPACK_SUFFIX = ".wpack"

_uids = itertools.count(1)


@dataclass
class PackTab:
    """One tab."""

    doc: Any
    title: str = "Untitled"
    path: Path | None = None
    uid: str = field(default_factory=lambda: f"pw{next(_uids)}")
    view: PaintView = field(default_factory=PaintView)
    saved_head: int = 0
    saving: bool = False

    # Crash-safety, owned by :mod:`studio.journal` (UX-05). Inker's three
    # fields, verbatim, because they are the same three questions: which file
    # this tab owns under the autosave directory (minted on the first copy, so
    # an untouched tab litters nothing), the history position that copy
    # captured (an undo back to it is not a new edit), and the debounce.
    journal_name: str = ""
    journal_head: int | None = None
    journal_at: float = 0.0

    # The last successful pack. ``layout`` is what the items pane lists and the
    # preview outlines; ``atlas`` is what it draws.
    layout: Any = None
    atlas: np.ndarray | None = None
    # Bumped once, when a pack is adopted. The texture cache keys on it.
    pack_generation: int = 0
    # A pack is in flight. Separate from ``saving`` because a repack does not
    # stop the user editing -- it only means the preview is a frame behind.
    packing: bool = False
    # Something changed that the current layout does not describe. A *flag*
    # pumped from the preview pane's draw rather than a direct submit, for the
    # reason ``findings_dirty`` is one: ``TaskRunner.submit`` refuses a key
    # already in flight and nothing re-arms it, so a burst of edits would
    # otherwise leave the last one unpacked for good.
    pack_dirty: bool = True
    # What the last pack could not do, if anything. Kept so the items pane can
    # say so rather than showing an empty list that looks like success.
    pack_error: str = ""

    @property
    def busy(self) -> bool:
        return self.saving

    @property
    def pack_stale_why(self) -> str:
        """Why the atlas on screen is not the one this document describes.

        Empty when it is. A *failed* pack clears ``packing`` and records
        ``pack_error``, but leaves ``layout``/``atlas`` at the last pack that
        worked -- deliberately, because a picture the user can still look at
        beats a blank pane. What was missing is that nothing said so: the
        preview drew the old atlas with no mark on it and both exports wrote
        it, so a pack that could not fit produced a file describing sprites the
        document no longer holds (the 2026-09-02 review, section 7).
        """
        if not self.pack_error or self.atlas is None:
            return ""
        return (
            "The last pack failed, so this is the atlas from before it: "
            f"{self.pack_error}"
        )

    @property
    def dirty(self) -> bool:
        return self.doc.dirty

    @property
    def label(self) -> str:
        return f"{self.title}###{self.uid}"

    def mark_saved(self, head: int | None = None) -> None:
        self.doc.mark_saved(head)
        self.saved_head = self.doc.saved_head
        self.saving = False

    def adopt_pack(self, layout: Any, atlas: np.ndarray) -> None:
        """Take a finished pack. The one place ``pack_generation`` moves.

        **``pack_dirty`` is deliberately not touched here.** An edit made while
        a pack was in flight set it, and that edit is not in the layout landing
        now -- clearing it would drop the edit for good, because ``request_pack``
        clears the flag only on an accepted submit and nothing else re-arms it.
        The flag is cleared at the submit, never at the adoption.
        """
        self.layout = layout
        self.atlas = atlas
        self.pack_generation += 1
        self.packing = False
        self.pack_error = ""
        self.view.fitted = False


@dataclass
class PackwrightState:
    docs: list[PackTab] = field(default_factory=list)
    active_uid: str = ""

    # Which source row is selected, by uid. View state, shared between the
    # sources list, the items list and the preview's highlight -- one answer to
    # "which sprite are we talking about" rather than three.
    selected: int | None = None

    # A decoded tile sheet waiting for its cell size: ``(path, display name,
    # pixels)``, set by the decode task and consumed (or dropped, on cancel)
    # by the sources pane's popup. Inker's ``sheet_import`` trio, verbatim,
    # because a sheet cannot say its own tile size and the user answers in a
    # popup either way.
    tileset_import: tuple[str, str, np.ndarray] | None = None
    tileset_import_open: bool = False
    tileset_cell: tuple[int, int] = (32, 32)
    # Whether to drop tiles whose content another tile already carries, and
    # whether a flipped or turned copy counts as the same content.
    #
    # **Default off, both.** A repack is a faithful repack unless somebody asks
    # otherwise: an atlas quietly missing the four tiles that happened to be
    # rotations of each other is a bug report nobody can reproduce.
    tileset_dedup: bool = False
    tileset_dedup_flips: bool = False
    # The popup's counts, and the inputs they were computed from. Cached
    # because the dedup pass behind them is measured in hundreds of
    # milliseconds on a full sheet and the popup redraws every frame; see
    # ``panes.packwright_sources._tileset_popup``. Cleared wherever
    # ``tileset_import`` is, so a stale preview cannot outlive its sheet.
    tileset_preview_key: tuple[Any, ...] | None = None
    tileset_preview: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)

    @property
    def active(self) -> PackTab | None:
        for doc in self.docs:
            if doc.uid == self.active_uid:
                return doc
        return self.docs[-1] if self.docs else None

    @property
    def any_dirty(self) -> bool:
        return any(doc.dirty for doc in self.docs)

    def add(self, doc: PackTab) -> PackTab:
        self.docs.append(doc)
        self.active_uid = doc.uid
        self.selected = None
        return doc

    def get(self, uid: str) -> PackTab | None:
        for doc in self.docs:
            if doc.uid == uid:
                return doc
        return None

    def close(self, uid: str) -> bool:
        doc = self.get(uid)
        if doc is None:
            return False
        index = self.docs.index(doc)
        self.docs.remove(doc)
        if self.active_uid == uid:
            self.active_uid = self.docs[min(index, len(self.docs) - 1)].uid if self.docs else ""
        self.selected = None
        return True

    def activate(self, uid: str) -> None:
        if uid != self.active_uid:
            self.active_uid = uid
            # The selection names a source of the *previous* document.
            self.selected = None

    def cycle(self, step: int = 1) -> None:
        if len(self.docs) < 2:
            return
        current = self.active
        index = self.docs.index(current) if current in self.docs else 0
        self.activate(self.docs[(index + step) % len(self.docs)].uid)

    def find_path(self, path: Path) -> PackTab | None:
        for doc in self.docs:
            if doc.path is not None and doc.path == path:
                return doc
        return None



def ensure(ctx: Any) -> PackwrightState:
    """The mode's state, built on first use.

    Here rather than in ``packwright_mode`` because this and :func:`active`
    touch exactly one thing -- ``ctx.state.packwright`` -- which is this
    module's whole charter, and neither knows a job or a task thread exists.
    The ``plotter_state`` shape.
    """
    state = ctx.state.packwright
    if state is None:
        state = PackwrightState()
        ctx.state.packwright = state
    return state


def active(ctx: Any) -> PackTab | None:
    """The focused tab, or ``None``. Deliberately *not* through :func:`ensure`:
    asking which atlas is open must not create the state that says none is."""
    state = ctx.state.packwright
    return state.active if state is not None else None


# The same answer in three of the four modes; Clay's is on ``stem`` on purpose.
title_for = docmodes.title_for
