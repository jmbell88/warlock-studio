"""Crash recovery for authored documents, for every mode rather than one.

UX-05. Inker has had a crash-safe autosave since it shipped and nothing else
did: a Clay model, a Plotter map, a Packwright atlas, a pose being authored and
a profile draft were all one power cut away from gone. The mechanism Inker
proved is right, and the whole of what was wrong was that it was written into
one mode.

**Not in an engine package.** ``studio/inker/``, ``clay/``, ``plotter/`` and
``packwright/`` import no imgui, no moderngl and no ``service``, and their
import sets are pinned by tests. This knows about ``ctx``.

**Not in ``service/``.** That layer is about *jobs* -- rows in a store, work in
a queue, artifacts on disk under a job id. A document open in a tab is none of
those things, and putting it there would give the service layer an opinion
about editor state it has managed not to have.

So: beside ``docmodes``, and following its import discipline -- stdlib only at
module scope, everything else lazy, headlessly testable. A discipline test pins
that, because the reason it holds is the reason ``docmodes`` holds: this is
called from the frame loop in every mode, including the ones that have loaded
none of the editors.

## The storage

Files in ``data_dir/autosave/`` -- the directory Inker already writes to, kept
rather than renamed, because a rename would strand every crash copy sitting in
it right now behind a version that no longer looks there.

Two files per slot: the **payload** (``.ora``, ``.wblk``, ``.wmap``, ``.wpack``,
``.pose.json``, ``.profile.json`` -- each mode's own format, so a recovered file
is openable by hand and by the mode's ordinary reader) and a ``<stem>.meta.json``
sidecar naming the kind, the title and when it was taken.

**The sidecar is written last and is the completion gate.** A payload with no
sidecar is a copy that was interrupted mid-write, and it is not offered. Both go
through a temp name and ``os.replace``, which is the staged-write rule this repo
applies to everything it serves -- but the ordering is the part that matters:
tmp+replace makes each file whole or absent, and the sidecar makes the *pair*
atomic.

No sqlite. The store is one connection behind an RLock on the asyncio thread,
this runs on the frame thread, and a crash copy that needed a lock the crashing
thread was holding is a crash copy that is not written.

## The loop

Inker's, verbatim in shape, because every part of it was load-bearing:

- **Frame-pumped**, in every mode. A crash while the user is looking at the
  library still loses the painting.
- **Debounced** at ``JOURNAL_SECONDS``. Long enough that the encode is
  invisible, short enough that what is lost is one idea.
- **Gated on the document's head**, not on a flag: an undo back to the copied
  position is not a new edit, and re-encoding an idle document every two
  minutes is work nobody asked for.
- **Encoded on the frame thread**, written on a task. The encoders read the
  live document, and encoding after a hand-off would capture whatever the user
  drew in between.
- **Through ``ctx.submit``**, whose key-dedupe is the backpressure: a slow
  encode skips a beat rather than queuing.
- **Declining keeps the files.** "Not now" is not "delete my work".

## The rule the whole feature turns on

**A journal entry is never a save.** It does not clear ``dirty``, does not move
``saved_head``, does not retitle the tab and does not touch a linked job. All it
promises is that a crash loses minutes rather than an afternoon, and every one
of those omissions is what stops it silently answering a question the user has
not been asked -- "where should this go".
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: How long a document may go unjournalled before a copy is taken. Two minutes
#: is the interval every editor with this feature has converged on.
JOURNAL_SECONDS = 120.0

#: How many recovered slots the offer lists **as rows at once**. A crash with
#: more open than this is one where the count is the useful part, so the rest
#: are named by number and surface as the listed ones are dealt with.
#:
#: A property of the offer rather than of the scan, and it was the other way
#: round: ``recoverable`` returned ``out[:MAX_RECOVERY]`` and nothing pruned
#: the remainder, so a crash with twelve documents open buried four of them --
#: never listed, never offered, never swept -- and ``status_line``, the
#: sentence the crash dialog shows, counted the truncated list and told the
#: user there were eight.
MAX_RECOVERY = 8

#: The sidecar's suffix. A full second suffix rather than a replacement, so a
#: payload's own extension survives in the stem and ``foo.pose.json`` and
#: ``foo.ora`` cannot collide on one sidecar.
META_SUFFIX = ".meta.json"

#: Sidecar schema. Bumped only for a change a reader cannot ignore; a version
#: it has never heard of is skipped rather than guessed at.
VERSION = 1


@dataclass(frozen=True)
class Provider:
    """One mode's answers. Everything the journal knows about a document kind.

    Callables rather than a subclass, matching how the rest of ``studio``
    composes: a mode is a module of functions with no instance to hang methods
    on, which is the same reason ``AppState`` holds the panes' state.

    ``head_of`` returns anything comparable with ``==``. An undo stack's serial
    is the good answer where there is one; a small document with no history --
    a pose, a profile draft -- passes its encoded payload, so "has it changed"
    is payload equality. Both are the same question and neither needs a flag.
    """

    kind: str
    #: The payload's suffix, including the dot. ``.pose.json`` is legal.
    ext: str
    #: What one of these is called in a sentence: "two drawings have...".
    label: str
    #: ``ctx -> [slot]``. Only slots worth journalling: a mode returns the
    #: dirty ones and skips the rest, because it knows what dirty means.
    slots: Callable[[Any], list[Any]]
    uid_of: Callable[[Any], str]
    title_of: Callable[[Any], str]
    head_of: Callable[[Any], Any]
    #: ``slot -> bytes``. **Frame thread.** See the module docstring.
    encode: Callable[[Any], bytes]
    #: ``(ctx, path, meta) -> bool``. Reopen one recovered payload; False means
    #: "could not, and has said so".
    adopt: Callable[[Any, Path, dict[str, Any]], bool]


_PROVIDERS: dict[str, Provider] = {}


def register(provider: Provider) -> Provider:
    """Add a kind. Idempotent, so a module re-imported under a test is fine."""
    _PROVIDERS[provider.kind] = provider
    return provider


#: The modules that register a provider. Imported on demand rather than at
#: module scope, which keeps this file stdlib-only and keeps a session that
#: never opens Clay from paying for the mesh engine.
_PROVIDER_MODULES = (
    "inker_mode",
    "clay_mode",
    "plotter_mode",
    "packwright_mode",
    "poser_mode",
    # A pane rather than a mode: the profiles panel owns the draft, and there
    # is no ``profiles_mode`` for it to live in.
    "panes.profiles_panel",
)


def ensure_providers() -> None:
    """Import every mode that owns a kind, so recovery can see all of them.

    The registration is a side effect of importing a mode, which is right for
    the *pump* -- a Clay document cannot exist before ``clay_mode`` has been
    imported -- and wrong for the startup offer, which runs before any mode
    has been opened and would otherwise find nothing able to adopt a Clay copy
    and log that as a mystery.

    Failures are swallowed per module: a mode that cannot import on this host
    is a kind that cannot be recovered, which is worth a log line and is not
    worth taking the other three down for.
    """
    from importlib import import_module

    for name in _PROVIDER_MODULES:
        try:
            module = import_module(f".{name}", __package__)
        except Exception:
            log.exception("journal: %s could not be imported to register its kind", name)
            continue
        # Re-registered from the module's own ``JOURNAL`` rather than relying on
        # the import's side effect: the second call to ``import_module`` is a
        # cache hit and runs no module body, so a registry that was cleared --
        # by a test's teardown, or by a future reload -- would stay cleared and
        # every crash copy would read as a kind nothing can adopt.
        provider = getattr(module, "JOURNAL", None)
        if isinstance(provider, Provider):
            register(provider)


def providers() -> list[Provider]:
    """Every registered kind, in registration order."""
    return list(_PROVIDERS.values())


def provider_for(kind: str) -> Provider | None:
    return _PROVIDERS.get(kind)


# --- bookkeeping, carried on the slot -----------------------------------------
#
# Three fields per state class, which is Inker's ``autosave_name``/``_head``/
# ``_at`` generalised. On the slot rather than in a dict here because the slot
# is what has a lifetime: a tab closed while the journal held a dict entry for
# it would leak one, and a tab reopened would inherit a stranger's.


@dataclass
class Mark:
    """What the journal remembers about one slot. Defaults mean "never taken"."""

    name: str = ""
    head: Any = None
    at: float = 0.0


def mark_of(slot: Any) -> Mark:
    return Mark(
        name=getattr(slot, "journal_name", "") or "",
        head=getattr(slot, "journal_head", None),
        at=float(getattr(slot, "journal_at", 0.0) or 0.0),
    )


def set_mark(slot: Any, mark: Mark) -> None:
    slot.journal_name = mark.name
    slot.journal_head = mark.head
    slot.journal_at = mark.at


# --- where it lives -----------------------------------------------------------


def directory(ctx: Any) -> Path:
    """``data_dir/autosave/``. The name is Inker's and is deliberately kept."""
    return Path(ctx.svc.config.autosave_dir)


def _safe(text: str, limit: int = 40) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in text)[:limit]


def payload_name(provider: Provider, slot: Any) -> str:
    """A stable, safe filename for one slot.

    From the title and the slot's uid rather than from its path: a path can be
    absent, can hold anything the filesystem allows, and can change under a
    Save As while a copy is already on disk under the old name.
    """
    stem = _safe(provider.title_of(slot)) or "untitled"
    return f"{stem}-{provider.uid_of(slot)}{provider.ext}"


def meta_path(payload: Path) -> Path:
    return Path(payload).with_name(Path(payload).name + META_SUFFIX)


# --- writing ------------------------------------------------------------------


def pump(ctx: Any, now: float | None = None) -> int:
    """One frame's worth. -> how many copies were submitted.

    Called once a frame by the app, in every mode. Every provider, every slot,
    and the same three gates Inker's loop applied: not busy, changed since the
    last copy, and past the debounce.
    """
    stamp = time.monotonic() if now is None else now
    written = 0
    for provider in providers():
        try:
            slots = provider.slots(ctx)
        except Exception:
            # A provider that raises must not stop the others: this runs every
            # frame in every mode, and one broken kind taking the whole journal
            # down is the failure the journal exists to prevent.
            log.exception("journal: %s could not list its slots", provider.kind)
            continue
        for slot in slots:
            if _write_if_due(ctx, provider, slot, stamp):
                written += 1
    return written


def _write_if_due(ctx: Any, provider: Provider, slot: Any, stamp: float) -> bool:
    mark = mark_of(slot)
    head = provider.head_of(slot)
    if mark.head is not None and _same_head(mark.head, head):
        # Nothing has changed since the last copy. Compared against the head
        # rather than tracked with a flag, for ``dirty``'s reason: an undo back
        # to the journalled position is not a new edit.
        return False
    if mark.at == 0.0:
        # Never seen (Mark's default, and what ``drop`` resets to): arm the
        # debounce at this pump rather than comparing against zero. The
        # production clock is ``time.monotonic()``, which is *boot*-relative
        # and so virtually always past ``JOURNAL_SECONDS`` already -- compared
        # against a ``0.0`` default, the wait branch below was unreachable
        # outside a test with a synthetic session clock, and the very first
        # pump after an edit wrote: a copy taken while the user was still
        # typing the first stroke. Armed here, the copy comes JOURNAL_SECONDS
        # after the pump that first saw the document dirty, whether the
        # session is two seconds or two days old -- and a dropped slot (saved,
        # closed) re-arms the same way instead of re-firing early. A stamp of
        # exactly 0.0 cannot arm (it is the sentinel); the clock is monotonic,
        # so the next pump arms instead.
        set_mark(slot, Mark(name=mark.name, head=mark.head, at=stamp))
        return False
    if stamp - mark.at < JOURNAL_SECONDS:
        # A document dirtied moments ago waits like every other one.
        return False
    return write(ctx, provider, slot, stamp)


def _same_head(a: Any, b: Any) -> bool:
    """``==``, but never raising and never returning an array.

    A payload-equality provider compares ``bytes``; a history-backed one
    compares ints. Both are fine. What is not fine is a comparison that raises
    or answers with something whose truth value is ambiguous, in a function
    called sixty times a second.
    """
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001 - see the docstring: unequal is the safe answer
        # Silent by design and the only one of these that is: this runs sixty
        # times a second, and a head whose ``==`` raises is a head that has
        # changed as far as the debounce is concerned -- which costs one extra
        # journal write and never a missed one.
        return False


# Drop tombstones. ``drop`` unlinks on the frame thread while the write it may
# be racing sits *queued* on a multi-worker pool: ``ctx.submit``'s key-dedupe
# stops two writes overlapping but sequences nothing against a drop, so a copy
# submitted a moment before save-and-close would be re-created after drop had
# deleted it -- and with the mark already reset, the orphaned pair is offered
# back on the next launch for a properly-saved document. A generation per
# payload name (the one identity both sides know), bumped by drop under the
# same lock the task checks it under: a pending write whose generation is
# stale skips rather than resurrecting the pair.
#
# The lock the write task holds across its disk I/O is **per payload name**
# (``_write_locks``), never module-wide: ``drop`` runs on the frame thread and
# must wait for an in-flight write *of its own document* so its unlinks remove
# what that write just put down -- but one global lock made it wait on any
# document's write, so dropping a just-saved sketch parked the frame loop
# behind an unrelated map's multi-MB encode landing on disk. ``_drop_lock``
# now guards only the two registries and is never held across I/O. Entries
# are never removed from either dict -- a name's queued write can drain
# arbitrarily late -- and each is a short string, an int and a lock object,
# so a session's worth of drops is bytes.
_drop_lock = threading.Lock()
_drop_gen: dict[str, int] = {}
_write_locks: dict[str, threading.Lock] = {}


def _write_lock(name: str) -> threading.Lock:
    with _drop_lock:
        lock = _write_locks.get(name)
        if lock is None:
            lock = _write_locks[name] = threading.Lock()
        return lock


def write(ctx: Any, provider: Provider, slot: Any, stamp: float | None = None) -> bool:
    """Take one copy now, ignoring the debounce. -> whether it was submitted.

    The encode happens **here**, on the frame thread, and only the write goes
    to a task -- see the module docstring.

    **The two halves of the mark move at two different moments**, and that
    split is the whole of what this function is careful about.

    ``name`` and ``at`` are recorded here, on a successful submit: they are the
    debounce, and a refused key -- ``ctx.submit``'s key-dedupe, which is the
    backpressure -- has to be retried on the next pump rather than recorded as
    done.

    ``head`` is recorded by the task, after ``_write_pair`` returns, because
    until then it is not true. It is the field that means "a copy of *this*
    state is on disk", and advancing it here recorded a crash copy that a full
    disk, a yanked directory or a permission change had prevented: the slot
    read as journalled at the current head, ``_write_if_due``'s first gate then
    saw nothing changed, and no retry ever came. The app believed a copy
    existed when none did -- the one thing the journal cannot be wrong about,
    because every other promise it makes is conditional on that one.

    The price of the split is that a failed write is retried
    ``JOURNAL_SECONDS`` later rather than on the next frame, because ``at`` did
    move. That is deliberate: a disk that has just refused a write is not a
    disk to re-encode a document against sixty times a second, and the failure
    is toasted by name in the meantime (``main._collect_tasks``).
    """
    stamp = time.monotonic() if stamp is None else stamp
    mark = mark_of(slot)
    name = mark.name or payload_name(provider, slot)
    root = directory(ctx)
    payload = root / name
    head = provider.head_of(slot)
    try:
        data = provider.encode(slot)
    except Exception:
        log.exception("journal: could not encode %s/%s", provider.kind, name)
        return False
    meta = {
        "version": VERSION,
        "kind": provider.kind,
        "title": provider.title_of(slot),
        "uid": provider.uid_of(slot),
        "at": time.time(),
    }

    hold = _write_lock(name)
    with _drop_lock:
        gen = _drop_gen.get(name, 0)

    def run() -> None:
        # This name's lock is held across the whole write, so a ``drop`` for
        # this document waits for the pair to be on disk before it unlinks --
        # and only a drop for *this* document does; see the tombstone comment.
        with hold:
            with _drop_lock:
                if _drop_gen.get(name, 0) != gen:
                    # Dropped between submit and execution: the document was
                    # saved or closed, and writing now would resurrect the
                    # deleted pair.
                    return
            _write_pair(payload, data, meta)
            with _drop_lock:
                # The head, and only now that the pair is really on disk. The
                # generation is re-read under the same lock ``drop`` resets the
                # mark under, so a drop that ran while this write was queued
                # cannot have its reset overwritten from here.
                if _drop_gen.get(name, 0) == gen:
                    set_mark(slot, Mark(name=name, head=head, at=stamp))

    if not ctx.submit(f"journal:{provider.kind}:{provider.uid_of(slot)}", run):
        return False
    # The debounce only -- ``head`` stays where it was until the write lands.
    # See the docstring: this is what makes a failed write retry instead of
    # reading as a copy that exists.
    set_mark(slot, Mark(name=name, head=mark.head, at=stamp))
    return True


def _write_pair(payload: Path, data: bytes, meta: dict[str, Any]) -> None:
    """Both halves staged, then both renamed in, the sidecar last. Task thread.

    The rename order is the completion gate: tmp+replace makes each file whole
    or absent, and renaming the sidecar second makes the *pair* atomic. A crash
    between them leaves a payload nothing offers, which is the right answer --
    it is a copy that was interrupted mid-write.

    **Both stagings happen before either rename**, and that is not the obvious
    order. Written the obvious way -- stage the payload, rename it, stage the
    sidecar, rename it -- a disk that fills between the two puts a *whole*
    first-ever copy on disk with no sidecar to offer it. ``recoverable`` is
    sidecar-driven by design, so that work is present, permanently invisible
    and never swept. Staging both first moves the two writes that can run out
    of room to before anything is published, and leaves between the renames
    only two metadata operations on one directory.

    A sidecar rename that fails anyway leaves the payload **in place** rather
    than unlinking it. It is the user's work, written in the mode's own format,
    and this module's standing rule is that deleting somebody's work is the one
    outcome worse than not offering it -- an unoffered file in ``autosave/``
    can still be opened by hand, and a deleted one cannot. The failure
    propagates, and the task that ran this now says so by name rather than
    through the generic "something went wrong" (``main._collect_tasks``).

    Each staging name is unlinked in a ``finally`` -- ``service/derive.py``'s
    ``_staged`` rule, for its reason: a staging file is a dotfile and nothing
    ever sweeps one, so a ``write_bytes`` or ``os.replace`` that raises would
    otherwise strand it beside the journal for good. The suppress is for the
    unlink only; the write's own failure still propagates.
    """
    payload.parent.mkdir(parents=True, exist_ok=True)
    side = meta_path(payload)
    tmp = payload.with_name(f".{payload.name}.tmp")
    tmp_meta = side.with_name(f".{side.name}.tmp")
    try:
        tmp.write_bytes(data)
        tmp_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        os.replace(tmp, payload)
        os.replace(tmp_meta, side)
    finally:
        for staging in (tmp, tmp_meta):
            with contextlib.suppress(OSError):
                staging.unlink(missing_ok=True)


def drop(ctx: Any, slot: Any) -> None:
    """Forget one slot's copy. Never raises -- it is cleanup, not an edit.

    Called when the document is saved or closed: at that moment the work is
    somewhere the user chose, and a crash copy of it is clutter that would be
    offered back on the next launch.

    **Quitting is deliberately not one of those moments**, and the asymmetry is
    checked rather than assumed. Closing a dirty tab confirms and then drops
    the copy; confirming "Discard unsaved work?" on the way *out* of the app
    leaves it, so the next launch offers it back. That is a mild surprise and
    the safe direction of one: "discard" at quit is answered under time
    pressure about possibly several documents at once, and the cost of being
    wrong is asymmetric -- an unwanted offer is one click, and a discarded
    afternoon is an afternoon. It is also what makes the journal's promise
    unconditional, which is the only form of that promise anybody can rely on.
    """
    mark = mark_of(slot)
    if not mark.name:
        return
    payload = directory(ctx) / mark.name
    # Before the unlinks, and under *this name's* write lock: a queued write
    # for this name that has not started yet sees the new generation and
    # skips; one already inside ``_write_pair`` holds the name's lock, so the
    # bump waits for it and the unlinks below remove what it just wrote. A
    # write for a different document holds a different lock and is not waited
    # on -- this runs on the frame thread, and the frame loop never blocks on
    # work that is not its own.
    with _write_lock(mark.name), _drop_lock:
        _drop_gen[mark.name] = _drop_gen.get(mark.name, 0) + 1
        # The reset is *inside* the hold rather than above it, which is the
        # half that had to change when the mark started being advanced by the
        # write task. A reset taken before the lock is a reset an in-flight
        # write lands on top of: the task finishes ``_write_pair``, records a
        # mark for a document that has just been saved and closed, and the
        # unlinks below then leave that mark naming a payload that is gone.
        set_mark(slot, Mark())
    for path in (meta_path(payload), payload):
        # Sidecar first: it is the completion gate, so removing it first means
        # a crash between the two unlinks leaves an unoffered payload rather
        # than a sidecar naming a file that has gone.
        try:
            path.unlink(missing_ok=True)
        except OSError:
            log.warning("journal: could not remove %s", path, exc_info=True)


# --- reading ------------------------------------------------------------------


@dataclass(frozen=True)
class Recovered:
    """One offerable crash copy: a payload, and what its sidecar said."""

    path: Path
    kind: str
    title: str
    at: float
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def adoptable(self) -> bool:
        return provider_for(self.kind) is not None


def _read_meta(side: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("version") != VERSION:
        # A version this build has never heard of is skipped rather than
        # guessed at: a half-understood recovery is worse than none.
        return None
    return data


def _at_of(meta: dict[str, Any]) -> float:
    """When the sidecar says the copy was taken, or 0.0.

    ``_read_meta`` checks that the file parses, that it is a dict and that the
    version is one this build knows -- and then every *field* was trusted.
    ``float(meta.get("at") or 0.0)`` on a string, a list or a dict raises, and
    the caller is ``snapshot``, which runs on the **first frame after a crash**:
    a malformed timestamp took the app down at precisely the moment the user
    needed it to open, and it would do so on every launch after, because
    nothing rewrites the file on that path. ``status_line`` already wraps its
    whole call for this reason (it runs in a process on its way out); the
    startup offer had no such wrapper and needed one less.

    0.0 rather than a refusal: ``at`` decides the sort order and the "2 hours
    ago" label, and neither is worth withholding somebody's work over. A copy
    with no readable timestamp sorts last and says nothing about its age.
    """
    try:
        value = float(meta.get("at") or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def recoverable(ctx: Any) -> list[Recovered]:
    """What a previous session left, newest first. **All of it.**

    Driven by the **sidecars**, which is what makes the completion gate real: a
    payload whose sidecar is missing was interrupted mid-write and is not here.

    Not capped. See :data:`MAX_RECOVERY` for the cap this used to apply and why
    it belongs to the offer instead: a list truncated here is work that nothing
    lists, nothing offers and nothing sweeps, and a count taken off it is a
    count that under-reports the user's own unsaved documents to them.
    """
    root = directory(ctx)
    try:
        sides = [p for p in root.glob(f"*{META_SUFFIX}") if p.is_file()]
    except OSError:
        return []
    out: list[Recovered] = []
    for side in sides:
        meta = _read_meta(side)
        if meta is None:
            continue
        payload = side.with_name(side.name[: -len(META_SUFFIX)])
        if not payload.is_file():
            continue
        out.append(
            Recovered(
                path=payload,
                kind=str(meta.get("kind") or ""),
                title=str(meta.get("title") or payload.stem),
                at=_at_of(meta),
                meta=meta,
            )
        )
    out.sort(key=lambda r: r.at, reverse=True)
    return out


def adopt(ctx: Any, found: list[Recovered]) -> int:
    """Reopen each. -> how many were handed to a provider.

    A kind with no provider is skipped and logged rather than dropped from
    disk: the mode may simply not be built into this run, and deleting
    somebody's work because this build does not understand it is the one
    outcome worse than not offering it.
    """
    taken = 0
    for one in found:
        provider = provider_for(one.kind)
        if provider is None:
            log.warning("journal: nothing can adopt a %r copy at %s", one.kind, one.path)
            continue
        try:
            if provider.adopt(ctx, one.path, one.meta):
                taken += 1
        except Exception:
            log.exception("journal: %s could not adopt %s", one.kind, one.path)
    return taken


def snapshot(ctx: Any) -> list[Recovered]:
    """What a previous session left, read **once** and remembered on the state.

    This function is where the offer's correctness lives, and it is the one
    thing that did not change when the offer moved off a startup modal and onto
    the home screen. The autosave directory is also where *this* session writes
    its own copies, so anything that re-read it later would begin listing the
    user's open documents back at them as though they had been crashed out of.
    The modal avoided that by being one-shot on the first frame; a panel that is
    on screen for as long as the user is on the home screen cannot be, so the
    *scan* is one-shot instead and the panel renders a snapshot of it.

    ``None`` and ``[]`` are different answers, deliberately: "not looked yet"
    and "looked, found nothing". A falsy test here would rescan every frame,
    which is the bug above with extra steps.
    """
    if ctx.state.recovery is None:
        # Providers before the directory, not after: a copy whose kind nothing
        # has registered would be listed and then found unadoptable, which
        # reads as a corrupt journal rather than as an unimported module.
        ensure_providers()
        ctx.state.recovery = recoverable(ctx)
    return ctx.state.recovery


def _unlist(ctx: Any, found: Recovered) -> None:
    """Drop one row from the snapshot, by path."""
    rows = ctx.state.recovery
    if rows is not None:
        ctx.state.recovery = [row for row in rows if row.path != found.path]


def take(ctx: Any, found: Recovered) -> bool:
    """Reopen one crash copy and drop its row. -> whether a provider took it.

    The files are **left on disk**. :func:`drop` owns deletion and fires when
    the recovered document is saved or closed; removing them here would take
    the copy away for the window between reopening it and saving it, which is
    precisely the window a second crash would land in.
    """
    taken = adopt(ctx, [found]) > 0
    _unlist(ctx, found)
    return taken


def discard(ctx: Any, found: Recovered) -> None:
    """Delete one crash copy and drop its row. Never raises -- it is cleanup.

    The unlink order is :func:`drop`'s and so is the reason: the sidecar is the
    completion gate, so removing it first means a crash between the two leaves
    an unoffered payload rather than a sidecar naming a file that has gone.
    """
    for path in (meta_path(found.path), found.path):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            log.warning("journal: could not remove %s", path, exc_info=True)
    _unlist(ctx, found)


def status_line(ctx: Any) -> str:
    """One sentence about the journal, for the crash dialog (UX-06).

    Computed from the directory rather than from memory, because the caller is
    a process that is on its way out and may have nothing left to ask.
    """
    try:
        found = recoverable(ctx)
    except Exception:  # noqa: BLE001 - the sentence *is* the surfacing
        # Not swallowed: the refusal is the return value. This runs inside the
        # crash dialog, in a process on its way out, so the honest thing to
        # say when the scan itself fails is that we do not know.
        return "Unsaved work may not have been kept."
    if not found:
        return "No unsaved work was waiting to be recovered."
    what = "document" if len(found) == 1 else "documents"
    return f"{len(found)} unsaved {what} will be offered back on the next launch."
