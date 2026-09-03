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

#: What a ``.wav`` drop is offered through, and what the instrument pane's
#: sample picker opens. Here rather than in the pane so the drop router and the
#: picker cannot advertise different formats.
SAMPLE_FILTER = [filetypes.describe("WAV audio", (".wav",)), filetypes.pattern((".wav",))]

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


# --- samples ------------------------------------------------------------------


#: The key prefix a sample decode carries. The rest of the key is the *tab*'s
#: uid rather than the file's, because that is what ``on_task_done`` looks a tab
#: up by -- and because one tab decoding two samples at once is a race over the
#: same sample table rather than a feature.
SAMPLE_PREFIX = "sirens-sample:"


def _sample_ceiling(path: Path) -> Path:
    """Refuse a file too big to be a sample, before a byte of it is read.

    ``wavout.MAX_SAMPLE_FRAMES`` is the engine's door on how many frames it will
    decode; this is the door on what the file *weighs*, which is a different
    question and has to be answered first -- the frame count is in a header the
    file has to be read to reach. The number is the engine's own, times the
    widest frame this build decodes (stereo 32-bit), rather than a second
    figure invented here.
    """
    from .sirens import wavout

    return sizeguard.within_ceiling(path, wavout.MAX_SAMPLE_FRAMES * 8)


def _decode_sample(path: Path, instrument: int | None) -> dict[str, Any]:
    """Blocking; task thread only. A ``.wav`` as the engine's own float mono.

    ``wavout.read_wav`` is the whole conversion -- mono, ``float32`` in
    ``[-1, 1]``, resampled to the render rate, and refused past
    ``MAX_SAMPLE_FRAMES`` -- so there is no second decoder here to disagree with
    the one that reads a sample back out of a ``.wsng``.

    **It returns the name rather than the key.** Which key a sample lands under
    depends on what the document already holds, and the document is the frame
    thread's; deciding here would mean reading it from a task.
    """
    from ..service.errors import invalid_from
    from .sirens import synth, wavout

    path = _sample_ceiling(Path(path))
    try:
        pcm = wavout.read_wav(path.read_bytes(), synth.SAMPLE_RATE)
    except ValueError as exc:
        raise invalid_from(exc, "This sample could not be loaded", field="file") from exc
    if not pcm.size:
        raise invalid_from(
            ValueError("it has no audio in it"), "This sample could not be loaded", field="file"
        )
    return {"pcm": pcm, "name": path.stem, "instrument": instrument}


def import_sample(ctx: Any, tab: SongTab, path: Path, instrument: int | None = None) -> None:
    """Decode a ``.wav`` into the tab's sample table. What a drop does.

    Decoding is task work and not frame work: a minute of 48 kHz stereo is a
    resample over three million frames, which is not something to do between
    two draws.
    """
    if tab is None or tab.busy:
        return
    ctx.submit(f"{SAMPLE_PREFIX}{tab.uid}", _decode_sample, Path(path), instrument)


def ask_sample(ctx: Any, tab: SongTab, instrument: int | None = None) -> None:
    """The picker, for the instrument pane's ``sample`` field.

    The dialog runs *inside* the task for the reason every other picker here
    does: a native file dialog is modal to the OS and blocks until it is
    dismissed, which on the frame thread is a frozen window.
    """
    if tab is None or tab.busy:
        return

    def run() -> dict[str, Any] | None:
        path = dialogs.open_file("Add a sample", SAMPLE_FILTER)
        return None if path is None else _decode_sample(path, instrument)

    ctx.submit(f"{SAMPLE_PREFIX}{tab.uid}", run)


def free_sample_key(doc: Any, name: str) -> str:
    """A sample-table key based on ``name`` that nothing in ``doc`` holds yet.

    Two files called ``kick.wav`` from two folders are two samples, and landing
    the second on the first's key would silently retune every note that used it.
    Suffixed rather than refused, because the user's answer to "that name is
    taken" is always "then use another one".
    """
    from .sirens import instruments as inst

    stem = (name or "sample").strip()[: inst.MAX_NAME_LEN] or "sample"
    if stem not in doc.samples:
        return stem
    for index in range(2, len(doc.samples) + 3):
        candidate = f"{stem[: inst.MAX_NAME_LEN - 4]} {index}"
        if candidate not in doc.samples:
            return candidate
    return stem


# --- writing ------------------------------------------------------------------


def _write(files: dict[Path, bytes]) -> None:
    """Write a whole set of files: stage all of them, then replace each.

    ``atomic.staged_set``: **every file is staged before any of them is
    replaced.** A song document saves as one file and exports as many --
    ``song.wav`` beside a ``stems/`` and an ``sfx/`` directory -- and an encode
    that raised on the ninth would otherwise leave eight new WAVs on top of a
    previous export's other twelve, under names that say they belong together.
    The leaf carries the rest of the argument.
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


# --- exporting ----------------------------------------------------------------
#
# **The ``.wsng`` is the composition and every WAV is a pure function of it**
# (``docs/INVARIANTS.md``). That is what :func:`export_plan` is: a document and
# a destination in, a complete ``{path: bytes}`` map out, with no clock, no
# randomness and no filesystem read anywhere in it -- so re-exporting a document
# nobody has touched writes the same bytes it wrote last time, and a test can
# say so without a disk. ``wavout`` already holds up its half (no timestamps, no
# writer string); this holds up the half above it by deciding every filename
# from the document alone.
#
# **Nothing is written until every file has been encoded.** The plan is built
# whole and handed to :func:`_write` in one call, which is what makes a refusal
# -- a hostile name, a song past the render ceiling -- leave no half-populated
# ``stems/`` behind for the user to wonder about.


#: What the export picker offers. Its own list rather than :data:`SAMPLE_FILTER`
#: even though the patterns are identical: that one is the *import* door and is
#: what a ``.wav`` drop is matched against, and one list serving both would tie
#: what this build can read to what it can write.
WAV_FILTER = [filetypes.describe("WAV audio", (".wav",)), filetypes.pattern((".wav",))]

#: The whole mix, at the root of the chosen directory. A fixed name rather than
#: the document's title: the title is a ``.wsng`` filename that may hold
#: anything, the stems and effects beside it are named after their own parts of
#: the document, and a folder whose four names come from four different places
#: is a folder nobody can write a build script against.
SONG_NAME = "song.wav"

#: The two subdirectories. Named here because :func:`export_plan` and the pane
#: that reports what landed both spell them, and two spellings of a directory
#: name is how a report comes to describe a folder that does not exist.
STEM_DIR = "stems"
SFX_DIR = "sfx"

#: The key prefix an export carries. The rest of it is the tab's uid, which is
#: what ``on_task_done`` looks a tab up by -- and one tab exporting twice at
#: once is two encodes racing onto one directory rather than a feature.
EXPORT_PREFIX = "sirens-export:"


def safe_stem(name: str, fallback: str) -> str:
    """``name`` as a filename stem, or ``fallback`` when it cannot be one.

    Channel names and sound-effect names are **user-supplied text that becomes a
    path**, which is the one shape ``wsng.py`` refused outright: a ``.wsng``
    numbers its archive members and keeps the names in the manifest, precisely
    so that ``../`` and ``CON`` cannot reach a filesystem through them. An
    export cannot take that way out -- ``sfx/coin.wav`` is the whole point of
    the directory, and a folder of ``sfx/1.wav`` with a sidecar mapping is a
    format nobody's build script reads -- so the names come through, sanitised.

    ``inker.sheetout``'s two rules rather than a third copy of them: its
    character class (letters, digits, dot, dash, underscore) and its list of the
    device names Windows still reserves. Both are exported from there for
    exactly this reason, and a second regex here would be the one that failed to
    learn about ``CONIN$``.

    **Falls back rather than refusing.** ``sheetout`` raises because an Inker
    sheet split is one export per tag and a name that cannot be a file means the
    user has to go and fix a tag; here the name is one of forty in a document
    where nothing else is wrong, and taking the whole export down over an effect
    somebody called ``...`` would be a refusal about the wrong thing. The
    fallback is positional (``effect3``), so the file is still findable.
    """
    from .inker import sheetout

    # ``strip(" .-")`` on top of the character class. A dot is legal *inside* a
    # filename and illegal at either end of one on Windows, which silently
    # strips it -- so ``fx.`` and ``fx`` would land on one file; ``sheetout``
    # refuses that case, and here it is simply trimmed off first. The dash goes
    # with it because ``../evil`` sanitises to ``..-evil``, and a file whose
    # name starts with a dash is one every shell reads as a flag.
    stem = sheetout.sanitize_stem(str(name)).strip(" .-")
    if not stem:
        return fallback
    try:
        sheetout.reserved_check(stem)
    except ValueError:
        return fallback
    return stem


def _unique(stems: list[str]) -> list[str]:
    """Suffix any stem that is already taken, in order. -> the same length.

    Two channels called ``Pulse 1`` -- or a ``Pulse/1`` and a ``Pulse-1``, which
    sanitise to one name -- are two files, and letting the second land on the
    first's name is one export silently overwriting another. ``sheetout``
    refuses a collision because a template that collides is a template the user
    should fix; a channel list is not a template, and the fix here is simply a
    number.
    """
    seen: dict[str, int] = {}
    out: list[str] = []
    for stem in stems:
        count = seen.get(stem, 0) + 1
        seen[stem] = count
        out.append(stem if count == 1 else f"{stem}-{count}")
    return out


def _under(root: Path, *parts: str) -> Path:
    """``root`` joined with ``parts``, refusing anything that escapes it.

    Belt and braces over :func:`safe_stem`, and deliberately so: that function
    decides what a name *becomes* and this one checks where the result *lands*,
    which are two different mistakes. A sanitiser is one regex away from letting
    a separator through and the consequence would be a file written outside the
    directory the user picked -- so the answer is checked rather than argued
    from, once, at the only place a path is built.
    """
    path = root.joinpath(*parts)
    if root.resolve() not in path.resolve().parents:
        raise ValueError(f"{'/'.join(parts)!r} is not a name inside the export folder")
    return path


def channel_stems(doc: Any) -> list[str]:
    """One filename stem per channel, in the document's channel order."""
    return _unique(
        [
            safe_stem(one.name, f"channel{index + 1}")
            for index, one in enumerate(doc.channels)
        ]
    )


def oneshot_stems(doc: Any) -> list[str]:
    """One filename stem per sound effect, in the document's own order."""
    return _unique(
        [
            safe_stem(one.name, f"effect{index + 1}")
            for index, one in enumerate(doc.oneshots)
        ]
    )


def _stem_render(doc: Any, index: int) -> tuple[Any, tuple[int, int] | None]:
    """The mix with every channel but ``index`` silenced. -> ``synth.render``'s pair.

    One line, because the masked render is the engine's now
    (:func:`~.sirens.synth.render_only`) -- a stem and a muted channel are the
    same operation asked for by two surfaces, and the argument for how it works
    (the effect column survives, so a stem stays sample-aligned with the mix)
    lives there.
    """
    from .sirens import synth

    samples, loop, _marks = synth.render_only(doc, {index})
    return samples, loop


def export_plan(doc: Any, directory: Path) -> dict[Path, bytes]:
    """Every file an export writes, encoded. Pure; blocking; task thread only.

    Separated from the task and the picker because it is the half worth
    asserting: byte-identity across two exports, a stem holding one channel and
    a hostile name landing inside the folder are all statements about this
    function, and none of them needs a dialog or a ``ctx`` to be true.

    The loop points go into ``song.wav``'s ``smpl`` chunk and into every stem's,
    which is why ``wavout`` was written by hand -- a soundtrack whose loop is in
    a sidecar the engine does not read is a soundtrack that does not loop.
    """
    from .sirens import synth, wavout

    directory = Path(directory)
    rate = synth.SAMPLE_RATE
    pcm, loop = synth.render(doc)
    files: dict[Path, bytes] = {
        _under(directory, SONG_NAME): wavout.wav_bytes(pcm, rate, loop=loop)
    }
    for index, stem in enumerate(channel_stems(doc)):
        samples, stem_loop = _stem_render(doc, index)
        files[_under(directory, STEM_DIR, f"{stem}.wav")] = wavout.wav_bytes(
            samples, rate, loop=stem_loop
        )
    for one, stem in zip(doc.oneshots, oneshot_stems(doc), strict=True):
        files[_under(directory, SFX_DIR, f"{stem}.wav")] = wavout.wav_bytes(
            synth.render_oneshot(doc, one.uid), rate
        )
    return files


def _export(data: bytes, directory: Path) -> dict[str, Any]:
    """Blocking; task thread only. The snapshot in, the report out.

    ``read_wsng`` rather than the live document, which is ``request_render``'s
    rule and its reason: a numpy view handed to a thread is a view of an array
    the caret is writing into, and rendering a song is seconds. The zip round
    trip is the price of an export that cannot tear.
    """
    from ..service.errors import invalid_from
    from .sirens import wsng

    directory = Path(directory)
    try:
        doc = wsng.read_wsng(data)
        files = export_plan(doc, directory)
    except ValueError as exc:
        # Framed: only a ``ServiceError``'s text survives the task classifier,
        # and the engine's own sentence -- a song past the render ceiling names
        # the ceiling -- is the half that says what to do about it.
        raise invalid_from(exc, "That song did not export", field="file") from exc
    _write(files)
    return {"directory": str(directory), "files": len(files)}


def export_to(ctx: Any, tab: SongTab, directory: Path) -> None:
    """Export into a directory already chosen. The door a test comes through.

    The snapshot is taken here, on the frame thread, for the reason
    :func:`~.sirens_mode.request_render` takes its own here: this is where the
    document is safe to read.
    """
    from .sirens import wsng

    if tab is None or tab.saving:
        return
    data = wsng.wsng_bytes(tab.doc)
    _start(ctx, tab, f"{EXPORT_PREFIX}{tab.uid}", lambda: _export(data, Path(directory)))


def export_files(ctx: Any, tab: SongTab | None = None) -> None:
    """``song.wav``, ``stems/`` and ``sfx/`` into a folder the user picks.

    A **folder** picker rather than a save dialog, alone among this app's
    exports, because alone among them this one writes a family of files under
    names it chooses: a typed filename would land on ``song.wav`` and be
    ignored by the twelve files beside it. See ``dialogs.select_folder``.

    The picker runs *inside* the task, which is every picker in this module: a
    native dialog is modal to the OS and blocks until dismissed, and on the
    frame thread that is a frozen window.
    """
    from .sirens import wsng

    tab = tab or active(ctx)
    if tab is None or tab.saving:
        return
    if not tab.doc.order and not tab.doc.oneshots:
        # Refused at the door rather than exporting a folder of empty WAVs. An
        # order list with nothing in it is what a brand-new document has, and
        # ``request_render`` already refuses it for the same reason.
        ctx.toast("There is nothing in the order list to export yet.", "error")
        return
    data = wsng.wsng_bytes(tab.doc)

    def run() -> dict[str, Any] | None:
        directory = dialogs.select_folder("Export the song")
        return None if directory is None else _export(data, directory)

    _start(ctx, tab, f"{EXPORT_PREFIX}{tab.uid}", run)
