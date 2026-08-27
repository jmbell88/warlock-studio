"""Sirens' file layer: what a song document is read from and written to.

Split out of ``sirens_mode`` for the reason ``packwright_io`` was: it is the
half that touches a disk, and the rules that shape it are all about that. **No
file dialog and no encode ever runs on the frame thread** -- a native picker is
modal to the OS and blocks until dismissed, and a song is a zip of numpy arrays
to build -- so both go through ``ctx.submit``, which is why saving is a *state*
(``SongTab.saving``) rather than a call that returns. **The head a save records
is captured before the submit**: a head read after an unbounded modal dialog
describes whatever the user did while it was open.

**Every write is staged**, through :func:`_write`, so a write that fails leaves
the previous file where it was rather than truncating it.

**The engine's refusals are framed rather than forwarded.** Only a
``ServiceError``'s text survives ``tasks.py``'s classifier; a bare
``ValueError`` out of ``read_wsng`` reaches the user as "Something went wrong;
see the log for details", which names neither the file nor the reason.

Every task key carries the ``sirens-`` prefix, because the app claims results
by prefix: a key without one is a result delivered nowhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import atomic, dialogs, docmodes, filetypes, sirens_state, sizeguard
from .sirens_state import SongTab, active, ensure

#: The picker's one row. Label *and* pattern through :mod:`.filetypes`, which
#: is the rule ``packwright_io`` states at length: portable-file-dialogs pairs a
#: filter list off two at a time -- a label, then that label's space-separated
#: patterns -- so a label followed by one entry per glob makes the second glob
#: the row's only pattern and the third the *next row's label*. A scan test
#: holds every ``*_FILTER`` in the app to this shape.
SONG_FILTER = [filetypes.describe("Warlock song", (".wsng",)), filetypes.pattern((".wsng",))]

#: What a ``.wav`` drop is offered through, and what sample import will use in
#: Phase 3. Here rather than in the pane so the drop router and the picker
#: cannot advertise different formats.
WAV_FILTER = [filetypes.describe("WAV audio", (".wav",)), filetypes.pattern((".wav",))]

#: A song document is small -- patterns are int16 and samples are the only bulk
#: -- so the ceiling is the one the untrusted-zip door already uses rather than
#: a second number invented here. Read at call time so a test can lower it.
_start = docmodes.start_save


# --- opening ------------------------------------------------------------------


def _within_ceiling(path: Path) -> Path:
    """Refuse a file too big to open, before a byte of it is read.

    ``wsng.MAX_DECOMPRESSED_BYTES`` is the engine's door on what the archive
    *claims*; this is the door on what it *weighs*, and the two are different
    questions -- ``packwright_io`` states the pair. The number is the engine's
    own, rather than a second one here, because "how big may a song be" has one
    answer and two would drift the first time either moved.

    A ``ServiceError`` rather than a ``ValueError``: this is the *mode's*
    refusal about a file the user picked, and its text reaches the user
    verbatim through the task classifier.
    """
    from .sirens import wsng

    return sizeguard.within_ceiling(path, wsng.MAX_DECOMPRESSED_BYTES)


def _load(path: Path) -> dict[str, Any]:
    """Blocking; task thread only. Raises rather than returning a broken tab."""
    from ..service.errors import invalid_from
    from .sirens import wsng

    path = _within_ceiling(Path(path))
    try:
        doc = wsng.read_wsng(path.read_bytes())
    except ValueError as exc:
        raise invalid_from(exc, "This song could not be opened", field="file") from exc
    return {"doc": doc, "path": str(path), "title": sirens_state.title_for(path)}


def ask_open(ctx: Any) -> None:
    ensure(ctx)

    def run() -> dict[str, Any] | None:
        path = dialogs.open_file("Open a song", SONG_FILTER)
        return None if path is None else _load(path)

    ctx.submit("sirens-open", run)


#: The key prefix an open-by-path carries, and the claim on everything after
#: the first colon. The path itself is the rest of the key rather than a hash of
#: it, so a *failure* can name the file that did not open and drop it off Home's
#: Resume list -- which a hash cannot. Split on the first colon only, which
#: leaves a drive letter intact.
OPEN_PREFIX = "sirens-open:"


def open_path(ctx: Any, path: Path) -> None:
    state = ensure(ctx)
    path = Path(path)
    existing = state.find_path(path)
    if existing is not None:
        # Focus rather than fork: two tabs over one path would race on save.
        state.activate(existing.uid)
        return
    ctx.submit(f"{OPEN_PREFIX}{path}", _load, path)


# --- writing ------------------------------------------------------------------


def _write(files: dict[Path, bytes]) -> None:
    """Write a whole set of files: stage all of them, then replace each.

    One file today and more in Phase 4 (``song.wav`` beside ``stems/``), and
    ``atomic.staged_set`` is what makes that a set that lands together rather
    than an export whose third file leaves the first two on top of an older
    one.
    """
    atomic.staged_set(files)


# --- saving -------------------------------------------------------------------


def save_to(ctx: Any, tab: SongTab, path: Path) -> None:
    from .sirens import wsng

    path = Path(path)
    head = tab.doc.history.head
    data = wsng.wsng_bytes(tab.doc)

    def run() -> dict[str, Any]:
        _write({path: data})
        return {"head": head, "path": str(path), "retitle": True}

    _start(ctx, tab, f"sirens-save:{tab.uid}", run)


def save(ctx: Any, tab: SongTab | None = None) -> None:
    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    if tab.path is None:
        save_as(ctx, tab)
        return
    save_to(ctx, tab, tab.path)


def save_as(ctx: Any, tab: SongTab | None = None) -> None:
    from .sirens import wsng

    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    head = tab.doc.history.head
    data = wsng.wsng_bytes(tab.doc)
    stem = Path(tab.title).stem or "song"

    def run() -> dict[str, Any] | None:
        path = dialogs.save_file(
            "Save the song", f"{stem}{sirens_state.WSNG_SUFFIX}", SONG_FILTER
        )
        if path is None:
            return None
        path = path.with_suffix(sirens_state.WSNG_SUFFIX)
        _write({path: data})
        return {"head": head, "path": str(path), "retitle": True}

    _start(ctx, tab, f"sirens-saveas:{tab.uid}", run)
