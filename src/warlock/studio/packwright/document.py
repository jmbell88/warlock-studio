"""The pack document: sources, settings, and the history over them.

**The layout is not in here.** It is derived from the sources and the settings
every time it is wanted, because the packer is deterministic -- so re-deriving
is free of the one real cost a cache has, which is being wrong. It is also what
makes a re-export of an unchanged document byte-identical without anything
having to remember to invalidate.

The usual two rules apply and are not restated in each edit: a step is addressed
by uid so that reordering the source list cannot retarget it, and a call that
changes nothing pushes nothing -- renaming a source to its own name, or
re-applying the settings the document already has, is a real user action and is
not a change.

**A key may appear once.** Adding a source whose key is already present is
refused rather than deduplicated silently: two sprites under one key would pack
into one slot, and the one that lost would simply be missing from the atlas with
nothing anywhere to say so.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, replace
from typing import Any

from ..undo import Edit, UndoStack
from .layout import Layout, PackSettings
from .layout import layout as build_layout
from .sources import Sprite

_uids = itertools.count(1)

# What every "no such source" refusal says. A ``ValueError`` rather than the
# bare ``KeyError`` these used to raise: the mode frames a ``ValueError`` into a
# toast and lets anything else through to the log, so a uid that went stale --
# a selection surviving an undo, say -- reached the key dispatch as a crash.
MISSING_SOURCE = "that source is not in this pack"

# The pane's own ``max_length``, mirrored here because this is where it has to
# be true: a name typed into the field is bounded by the widget, and a name that
# arrived in a hand-edited manifest is not bounded by anything.
MAX_NAME_LEN = 64


def new_uid() -> int:
    return next(_uids)


@dataclass
class Source:
    """One sprite in the document, plus whatever the user renamed it to.

    The override is stored beside the sprite rather than written into it: a
    sprite is frozen and its ``key`` is derived from where it came from, so
    renaming must not be able to change identity. Renaming the third layer of a
    document does not make it a different layer.
    """

    uid: int
    sprite: Sprite
    name_override: str = ""

    @property
    def key(self) -> str:
        return self.sprite.key

    @property
    def name(self) -> str:
        return self.name_override or self.sprite.name


# --- edits --------------------------------------------------------------------


@dataclass
class SourceAddEdit(Edit):
    source: Source
    index: int

    def __post_init__(self) -> None:
        # An undone add holds the only reference to a full sprite's pixels, so
        # a zero cost would hide megabytes from the byte budget entirely.
        self.cost = int(self.source.sprite.pixels.nbytes)

    def undo(self, doc: Any) -> None:
        doc._detach(self.source)

    def redo(self, doc: Any) -> None:
        doc._attach(self.source, self.index)


@dataclass
class SourceRemoveEdit(Edit):
    source: Source
    index: int

    def __post_init__(self) -> None:
        self.cost = int(self.source.sprite.pixels.nbytes)

    def undo(self, doc: Any) -> None:
        doc._attach(self.source, self.index)

    def redo(self, doc: Any) -> None:
        doc._detach(self.source)


@dataclass
class SourceRenameEdit(Edit):
    uid: int
    before: str
    after: str

    def undo(self, doc: Any) -> None:
        doc._apply_name(self.uid, self.before)

    def redo(self, doc: Any) -> None:
        doc._apply_name(self.uid, self.after)


@dataclass
class SettingsEdit(Edit):
    """The whole settings object, before and after.

    A frozen dataclass of six scalars, so a field-by-field edit type would buy
    nothing but six places for the two halves to disagree.
    """

    before: PackSettings
    after: PackSettings

    def undo(self, doc: Any) -> None:
        doc._apply_settings(self.before)

    def redo(self, doc: Any) -> None:
        doc._apply_settings(self.after)


# --- the document -------------------------------------------------------------


class PackDoc:
    def __init__(
        self,
        sources: list[Source] | None = None,
        settings: PackSettings | None = None,
    ) -> None:
        self.sources: list[Source] = list(sources or [])
        self.settings = settings or PackSettings()
        self.history = UndoStack()
        self.saved_head = 0

    # -- identity ------------------------------------------------------------

    @property
    def dirty(self) -> bool:
        return self.history.head != self.saved_head

    def mark_saved(self, head: int | None = None) -> None:
        """``head`` is the value captured *before* the encode was handed to a
        task thread; the document may have moved on since."""
        self.saved_head = self.history.head if head is None else int(head)

    # -- lookup --------------------------------------------------------------

    def source(self, uid: int) -> Source | None:
        for entry in self.sources:
            if entry.uid == uid:
                return entry
        return None

    def index_of(self, uid: int) -> int:
        for index, entry in enumerate(self.sources):
            if entry.uid == uid:
                return index
        raise ValueError(MISSING_SOURCE)

    def has_key(self, key: str) -> bool:
        return any(entry.key == key for entry in self.sources)

    def sprites(self) -> list[Sprite]:
        """Every sprite, in canonical key order, wearing its display name.

        The rename is applied *here* rather than stored on the sprite, which is
        what keeps ``key`` -- and therefore the pack order and the whole
        determinism contract -- independent of what anything is called.

        ``meta`` is copied through, and this is the one place it could quietly
        not be: the rebuild names its fields, so a sprite that had a pivot lost
        it the moment the user renamed it -- a bug with no symptom until the
        export. A test pins it.
        """
        return [
            entry.sprite
            if not entry.name_override
            else Sprite(
                key=entry.sprite.key,
                name=entry.name_override,
                pixels=entry.sprite.pixels,
                meta=entry.sprite.meta,
            )
            for entry in sorted(self.sources, key=lambda e: e.key)
        ]

    def layout(self) -> Layout:
        """Derived, every time. See the module docstring for why."""
        return build_layout(self.sprites(), self.settings)

    # -- mutation ------------------------------------------------------------

    def add_source(self, sprite: Sprite, *, index: int | None = None) -> Source:
        if self.has_key(sprite.key):
            raise ValueError(
                f"this pack already holds {sprite.key!r} -- two sprites under one "
                "key would pack into one slot"
            )
        source = Source(uid=new_uid(), sprite=sprite)
        at = len(self.sources) if index is None else max(0, min(int(index), len(self.sources)))
        self.history.push(SourceAddEdit(source=source, index=at))
        self._attach(source, at)
        return source

    def remove_source(self, uid: int) -> None:
        index = self.index_of(uid)
        source = self.sources[index]
        self.history.push(SourceRemoveEdit(source=source, index=index))
        self._detach(source)

    def rename_source(self, uid: int, name: str) -> None:
        """Rename one source, or clear the override with an empty string.

        The name is a *door*: it lands verbatim in the TexturePacker sidecar's
        ``filename`` and in a ``.tsx``, which are read by other programs, so a
        path separator or a control character in it is a name that means
        something else once it leaves here. The empty string stays legal --
        that is how the override is cleared, and what the sprite is then called
        is what it came in as.
        """
        source = self.source(uid)
        if source is None:
            raise ValueError(MISSING_SOURCE)
        after = str(name)
        if len(after) > MAX_NAME_LEN:
            raise ValueError(f"a sprite name is at most {MAX_NAME_LEN} characters")
        if any(ch in after for ch in "/\\") or any(ch < " " for ch in after):
            raise ValueError(
                "a sprite name cannot hold a path separator or a control character"
            )
        if after == source.name_override:
            return
        self.history.push(SourceRenameEdit(uid=int(uid), before=source.name_override, after=after))
        self._apply_name(uid, after)

    def set_settings(self, **values: Any) -> None:
        """Apply a settings change, with one adjustment: a mode change that
        does not also name ``power_of_two`` re-baselines it to the new
        mode's default rather than carrying the old mode's resolved value
        across.

        ``PackSettings.power_of_two``'s ``None`` sentinel only resolves in
        ``__post_init__``, at construction -- and by the time a document
        holds a settings object, the field is already a concrete bool.
        ``dataclasses.replace`` copies that concrete value onto the new mode
        verbatim, which is how a grid pack's resolved ``False`` used to
        survive a switch to MaxRects and contradict MaxRects's own
        documented ``True`` default. Passing ``None`` back through
        ``replace`` re-triggers the resolution the same way a fresh
        ``PackSettings()`` does. This does drop an earlier explicit choice
        across the switch -- the honest outcome, since the settings pane's
        checkbox only ever shows the *resolved* value and there is nothing
        left to say "the user picked this one on purpose" once ``__post_init__``
        has already turned it into a plain bool. An explicit ``power_of_two``
        passed in the *same* call always wins; this only fires when the
        caller left it unsaid.
        """
        if (
            "mode" in values
            and values["mode"] != self.settings.mode
            and "power_of_two" not in values
        ):
            values = {**values, "power_of_two": None}
        after = replace(self.settings, **values)
        if after == self.settings:
            return
        self.history.push(SettingsEdit(before=self.settings, after=after))
        self._apply_settings(after)

    def _attach(self, source: Source, index: int) -> None:
        self.sources.insert(index, source)

    def _detach(self, source: Source) -> None:
        for index, entry in enumerate(self.sources):
            if entry is source:
                del self.sources[index]
                return
        raise ValueError(MISSING_SOURCE)

    def _apply_name(self, uid: int, name: str) -> None:
        source = self.source(uid)
        if source is None:
            raise ValueError(MISSING_SOURCE)
        source.name_override = name

    def _apply_settings(self, settings: PackSettings) -> None:
        self.settings = settings

    # -- history -------------------------------------------------------------

    def undo(self) -> bool:
        return self.history.undo(self)

    def redo(self) -> bool:
        return self.history.redo(self)
