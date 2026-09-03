"""What a re-rendered cell should do to the cell a person has been painting.

``sheetscope`` says *where* a correction goes on a character sheet; this says
*whether* a fresh render may land on one. Pure, verdict-returning, and it
writes nothing -- the frames it judges are handed to ``_doc_sheet.SheetOps``,
which is the one door onto the document.

**The problem it exists for.** A Troupe sheet opens in Inker, the user cleans a
hand here and a foot there, and then wants one animation re-rendered because
the clip was wrong. Re-rendering produces a whole new sheet; opening it throws
away every cleanup, and merging it blindly overwrites them. Neither is what
anybody wants, and the difference between "this cell is untouched" and "this
cell is somebody's afternoon" is not visible in the pixels alone. It is visible
against a *third* picture: what the renderer gave us last time.

So the merge is three-way, and this module is the comparison. Three digests per
cell -- the recorded base, the pixels on screen now, and the incoming render --
decide one of five verdicts, and only ``take`` writes anything.

**The rule that outranks the table**: nothing painted is ever overwritten
silently. Where the user's work and the renderer's disagree, the hand edit
stands and the cell is flagged for a person to look at. That is the default and
it is not configurable, because the cost of the two mistakes is not symmetric:
a cell wrongly kept is one the user re-takes in a click, and a cell wrongly
taken is work that is gone.

No outward imports. This package may not reach for ``warlock.pipelines`` (see
``tests/inker/test_inker_imports.py``), and nothing here needs to: a digest is
``hashlib`` over an array, and a verdict is three string comparisons.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DIGEST_ALGORITHM",
    "SHEET_BLOCK_VERSION",
    "VERDICTS",
    "MergeCounts",
    "SheetBase",
    "base_from_payload",
    "cell_digest",
    "classify",
    "counts_sentence",
]

#: Named in the payload and checked on read, never assumed. A stored corpus of
#: ``.ora`` files becomes keyed on this the moment one is written, so a future
#: change is a read-side branch on this string rather than a silent mismatch --
#: the same rule ``docs/measurements/`` states for a constant the corpus is
#: keyed on, applied to a hash instead of a threshold.
DIGEST_ALGORITHM = "blake2b-16"

#: The ``sheet`` block's own version, inside the block. The animation payload's
#: version deliberately does *not* move for this: every reader is ``.get``-based
#: and a build that does not know the key opens the file as the document it was
#: before the key existed, which is ``groups``' rule one key over.
SHEET_BLOCK_VERSION = 1

#: The five answers, and what each writes.
#:
#: ``take``     -- the user did not touch it and the render changed. Take it.
#: ``keep``     -- the user painted it and the render did not change. Keep it.
#: ``agreed``   -- nothing to do: either nobody changed anything, or the user
#:                 painted exactly what the renderer now produces.
#: ``conflict`` -- both changed, and differently. **Keep the edit, flag it.**
#: ``unknown``  -- no base was recorded, so there is no way to tell an edit from
#:                 a render. Never writes; the op refuses rather than guesses.
VERDICTS = ("take", "keep", "agreed", "conflict", "unknown")


def cell_digest(pixels: Any) -> str:
    """A stable fingerprint of one cell's pixels.

    ``blake2b`` at 16 bytes: 8 KB across a 256-cell sheet, which is free beside
    the atlas it describes, and far past any accidental collision.

    **The shape is hashed with the bytes**, and that is not decoration. A canvas
    resized between the import and the merge produces a *different* digest
    rather than one that might coincide, which is what lets :func:`classify` be
    honest instead of confidently wrong about a document whose geometry moved.

    ``ascontiguousarray`` because a cel's plane can be a view after a geometry
    op, and a view's ``tobytes`` is its base's buffer.
    """
    import hashlib

    import numpy as np

    array = np.ascontiguousarray(pixels, dtype=np.uint8)
    shaped = "x".join(str(n) for n in array.shape)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(f"{shaped}|".encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def classify(base: str | None, current: str, incoming: str) -> str:
    """One cell's verdict. See :data:`VERDICTS` for what each one means.

    Order matters here only in that ``unknown`` comes first: without a base
    there is no third point to triangulate from, and every other branch would
    be reading a two-way comparison as if it were three-way.

    The ``current == incoming`` case is worth its own line rather than falling
    into ``conflict``: the user painted what the renderer has now caught up to,
    and calling that a conflict would be asking somebody to arbitrate between
    two identical pictures.
    """
    if not base:
        return "unknown"
    edited = current != base
    rerendered = incoming != base
    if not edited:
        return "take" if rerendered else "agreed"
    if not rerendered:
        return "keep"
    return "agreed" if current == incoming else "conflict"


@dataclass(frozen=True)
class MergeCounts:
    """What one merge did, by verdict. Returned so the toast can say it."""

    taken: int = 0
    kept: int = 0
    agreed: int = 0
    conflicts: int = 0
    unknown: int = 0

    @property
    def total(self) -> int:
        return self.taken + self.kept + self.agreed + self.conflicts + self.unknown

    @property
    def wrote(self) -> bool:
        return self.taken > 0


def counts_sentence(counts: MergeCounts) -> str:
    """The toast. Names what happened, and never hides a conflict in a total.

    Built by omission rather than by listing zeroes: "Took 48 cells." is the
    whole truth about a clean merge, and burying it in three zeroes makes the
    one number that matters harder to find.
    """
    parts: list[str] = []
    if counts.taken:
        parts.append(f"Took {counts.taken} {'cell' if counts.taken == 1 else 'cells'}")
    if counts.kept:
        parts.append(f"kept {counts.kept} {'edit' if counts.kept == 1 else 'edits'}")
    if counts.conflicts:
        word = "conflict" if counts.conflicts == 1 else "conflicts"
        parts.append(f"flagged {counts.conflicts} {word}")
    if not parts:
        return "Nothing to merge -- that render matches what is here."
    return ", ".join(parts) + "."


@dataclass
class SheetBase:
    """What the renderer last gave us, per cell, plus what is still unresolved.

    **Keyed by frame uid in memory and by index in the file.** The uid is the
    undo stack's rule -- every edit addresses its layer by uid, never by index
    -- so an inserted or reordered frame cannot slide a digest onto the wrong
    cell mid-session. The index is ``_animation_json``'s rule, because a uid is
    minted per process and means nothing in a file. ``cel_notes`` and
    ``frame_palettes`` already live at exactly this address.
    """

    digests: dict[int, str] = field(default_factory=dict)
    conflicts: set[int] = field(default_factory=set)
    source: dict[str, str] = field(default_factory=dict)
    algorithm: str = DIGEST_ALGORITHM

    def copy(self) -> SheetBase:
        """A snapshot an undo step can hold without aliasing the live one."""
        return SheetBase(
            digests=dict(self.digests),
            conflicts=set(self.conflicts),
            source=dict(self.source),
            algorithm=self.algorithm,
        )

    def payload(self, index_of: Mapping[int, int]) -> dict[str, Any] | None:
        """The ``sheet`` block, or ``None`` when there is nothing to write.

        ``index_of`` maps frame uid to frame index. Sorted lists throughout,
        because ``.ora`` writing is pinned byte-identical and a set's iteration
        order is not a promise.
        """
        cells = sorted(
            (index_of[uid], digest)
            for uid, digest in self.digests.items()
            if uid in index_of
        )
        if not cells:
            return None
        return {
            "version": SHEET_BLOCK_VERSION,
            "algorithm": self.algorithm,
            "source": dict(sorted(self.source.items())),
            "cells": [{"frame": index, "digest": digest} for index, digest in cells],
            "conflicts": sorted(
                index_of[uid] for uid in self.conflicts if uid in index_of
            ),
        }


def base_from_payload(payload: Any, uid_at: Sequence[int]) -> SheetBase | None:
    """Read a ``sheet`` block back, or ``None`` if it is not one we can use.

    ``uid_at`` is the frame uids in index order, which is what turns the file's
    indices back into the addresses the rest of the merge uses.

    **Guarded rather than validating**, the way ``_read_groups`` and ``layout``
    are: a base digest is metadata *about* a picture that is already fully and
    correctly built, so anything malformed leaves the document with no base and
    the merge op refusing by name. An unknown ``algorithm`` is in that class on
    purpose -- digests we cannot recompute are worse than none, because they
    would classify every cell as edited.
    """
    if not isinstance(payload, Mapping):
        return None
    if payload.get("algorithm") != DIGEST_ALGORITHM:
        return None
    raw = payload.get("cells")
    if not isinstance(raw, list):
        return None
    digests: dict[int, str] = {}
    for entry in raw:
        if not isinstance(entry, Mapping):
            return None
        index, digest = entry.get("frame"), entry.get("digest")
        if not isinstance(index, int) or not isinstance(digest, str) or not digest:
            return None
        if not 0 <= index < len(uid_at):
            return None
        digests[uid_at[index]] = digest
    if not digests:
        return None
    conflicts = {
        uid_at[index]
        for index in payload.get("conflicts") or ()
        if isinstance(index, int) and 0 <= index < len(uid_at)
    }
    source = payload.get("source")
    return SheetBase(
        digests=digests,
        conflicts=conflicts,
        source={
            str(k): str(v) for k, v in (source or {}).items() if isinstance(source, Mapping)
        },
    )
