"""One merged most-recently-used list, across every document mode.

Inker, Clay, Plotter and Packwright each kept a ``recent: list[str]`` on their
own state, with near-identical ``remember``/``forget`` methods and the same
bound -- four spellings of one fact. Merging them at draw time is not merely
untidy, it is *impossible*: four bare path lists carry no ordering between
them, so "the six things you touched most recently" cannot be answered from
them at all. So the list moved here and gained a timestamp.

Pure: stdlib only, and it talks to the settings object through ``get``/``set``
and nothing else, so every rule about ordering is assertable without a window.

Assets are deliberately *not* written here. A generated asset is a row in the
job store with a ``created`` of its own, so Home derives its Resume rows from
the job cache at draw time; a second copy of that ordering in the settings file
would be a record of the library that the library could contradict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# The settings key this module owns, and the four legacy per-mode keys it folds
# in on first read.
SETTING = "recents"

# The document modes that have files to remember. Not ``modes.WORK_MODES``:
# 2D, 3D and Review own no document, and importing ``modes`` here would be a
# dependency this module does not otherwise need.
KINDS: tuple[str, ...] = ("inker", "clay", "plotter", "packwright")

# One screenful of a menu, per kind. The merged list is bounded by the sum
# rather than by this, because a session spent entirely in Clay should not push
# every other mode's history out.
MAX_RECENT = 10


@dataclass(frozen=True)
class Entry:
    kind: str
    path: str
    # Epoch seconds, or ``None`` for a row folded in from a legacy per-mode
    # list -- those carry an order but no clock. ``None`` sorts last and is not
    # an error, the rule ``Filters.order`` applies to every unanswerable row.
    when: float | None = None


def _stored(settings: Any) -> list[dict[str, Any]] | None:
    raw = settings.get(SETTING)
    if not isinstance(raw, list):
        return None
    return [row for row in raw if isinstance(row, dict)]


def _decode(rows: list[dict[str, Any]]) -> list[Entry]:
    out: list[Entry] = []
    for row in rows:
        kind = row.get("kind")
        path = row.get("path")
        if kind not in KINDS or not isinstance(path, str) or not path:
            continue
        when = row.get("when")
        out.append(Entry(kind, path, float(when) if isinstance(when, int | float) else None))
    return out


def _encode(entries: list[Entry]) -> list[dict[str, Any]]:
    return [
        {"kind": e.kind, "path": e.path, **({} if e.when is None else {"when": e.when})}
        for e in entries
    ]


def _order(entries: list[Entry]) -> list[Entry]:
    """Newest first, with unstamped rows last in their stored order.

    Stable, so the migration's per-kind order survives among the rows that
    share a ``None`` stamp -- which is the only ordering those rows have.
    """
    return [
        entry
        for _index, entry in sorted(
            enumerate(entries),
            key=lambda pair: (pair[1].when is None, -(pair[1].when or 0.0), pair[0]),
        )
    ]


def _migrate(settings: Any) -> list[Entry]:
    """Fold each mode's own ``recent`` list in, in its stored MRU order.

    Runs once: the migrated list is written under :data:`SETTING`, and from
    then on ``_stored`` answers. The four per-mode keys are left where they are
    rather than deleted -- an older build opening the same settings file still
    reads them, and they cost a few hundred bytes.
    """
    entries: list[Entry] = []
    for kind in KINDS:
        block = settings.get(kind)
        if not isinstance(block, dict):
            continue
        for path in block.get("recent") or []:
            if isinstance(path, str) and path:
                entries.append(Entry(kind, path, None))
    _write(settings, entries)
    return entries


def _write(settings: Any, entries: list[Entry]) -> None:
    settings.set(SETTING, _encode(entries))


def _load(settings: Any) -> list[Entry]:
    rows = _stored(settings)
    if rows is None:
        return _migrate(settings)
    return _decode(rows)


def entries(settings: Any, kind: str | None = None) -> list[Entry]:
    """The merged list newest-first, or one kind's slice of it."""
    found = _order(_load(settings))
    if kind is None:
        return found
    return [e for e in found if e.kind == kind]


def paths(settings: Any, kind: str) -> list[str]:
    """One kind's recent paths, newest first -- what a mode's own panel draws."""
    return [e.path for e in entries(settings, kind)]


def remember(settings: Any, kind: str, path: Any, *, when: float | None = None) -> None:
    """Put ``path`` at the front of ``kind``'s history. ``None`` is a no-op.

    Deduplicated on ``(kind, path)`` -- the same file opened in two modes is two
    rows, because activating them does two different things -- and bounded per
    kind, so a busy session in one mode cannot evict every other mode's history.
    """
    if path is None or kind not in KINDS:
        return
    import time

    text = str(path)
    if not text:
        return
    stamp = time.time() if when is None else when
    found = [e for e in _load(settings) if not (e.kind == kind and e.path == text)]
    found.insert(0, Entry(kind, text, stamp))
    kept: list[Entry] = []
    counts = dict.fromkeys(KINDS, 0)
    for entry in _order(found):
        if counts[entry.kind] >= MAX_RECENT:
            continue
        counts[entry.kind] += 1
        kept.append(entry)
    _write(settings, kept)


def forget(settings: Any, kind: str, path: Any) -> None:
    """Drop a path that turned out not to open.

    A recent list that keeps offering a moved file is worse than a short one --
    the rule each of the four per-mode lists already followed, kept.
    """
    if path is None:
        return
    text = str(path)
    found = _load(settings)
    kept = [e for e in found if not (e.kind == kind and e.path == text)]
    if len(kept) != len(found):
        _write(settings, kept)
