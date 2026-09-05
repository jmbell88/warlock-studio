"""The Home screen: what changed, what the machine is doing, what you were on.

It used to be a grid of nine tiles, seven of which were the mode switch again
-- the same names under the same glyphs, one click either way -- while the
switch itself is drawn unconditionally at the top of every frame. So Home spent
its whole screen re-offering navigation that is permanently on screen, and the
two tiles that were *not* modes (Open Existing, Profiles) were real destinations
wearing a tile because there was nowhere else to put them. They are modes now,
and the grid is gone.

What is left is the three questions nothing in the app answered: what changed,
what is the machine doing, and what was I working on -- in one column, in that
order, with the answers weighted by how often they are the reason somebody is
here. "What changed" is a card you dismiss once per release; "what is the
machine doing" is one quiet line; "what was I working on" is the rest of the
screen, as pictures (the UI redesign, wave 4.3).

Nothing here is persisted: ``AppState.mode`` defaults to ``"home"``, which is
what makes this appear on every launch rather than only the first ever.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from imgui_bundle import imgui

from ... import changelog
from .. import (
    asset_open,
    controls,
    create_stages,
    fonts,
    icons,
    modes,
    recents,
    theme,
    tokens,
    widgets,
)
from ..manual import render as manual_render
from ..state import DEFAULT_FORM_3D, default_form_2d, format_bytes, set_mode
from ..tokens import sp
from . import thumbs

log = logging.getLogger(__name__)

# How many Resume rows are offered. A shortlist rather than a history: past a
# dozen the eye is scanning rather than recognising, and the Library mode is one
# click away for everything else.
MAX_RESUME = 12

# How stale the unreviewed-mesh count may get before Home re-asks. It is a table
# scan behind the one serialized connection, so it can never run on the frame
# thread -- ``pump`` submits it and the block draws the last answer, which is the
# same bargain the storage walk already makes.
UNREVIEWED_TTL = 30.0

# Enough to answer "is there a review pass waiting" without scanning the whole
# table. At the cap the figure is rendered with a "+", because a count that
# silently saturates is a wrong number rather than a rounded one.
UNREVIEWED_LIMIT = 200

# The mode each document kind opens in, and the glyph its Resume row wears.
# Off ``modes.MODES`` rather than a second table: a row that opened Clay under
# Plotter's icon is exactly the drift a hand-copied glyph produces.
_MODE_ICONS = {key: icon for key, _label, icon in modes.MODES}


@dataclass(frozen=True)
class Row:
    """One Resume row.

    ``kind`` is a document mode for a file and ``"asset"`` for a library row,
    which is the whole of what tells :func:`activate` how to open it. ``when``
    is epoch seconds, or ``None`` for a legacy recent-files entry that carries
    an order but no clock -- those sort last, the rule ``Filters.order`` applies
    to every unanswerable row.
    """

    kind: str
    key: str
    icon: str
    name: str
    when: float | None = None


# --- the resume list --------------------------------------------------------


def _asset_rows(ctx: Any) -> list[Row]:
    """Finished assets, newest first, derived from the job cache.

    Deliberately *not* written into :mod:`.recents`: an asset already has a
    ``created_at`` in the job store, and a second copy in the settings file
    would be a record of the library the library could contradict -- a row that
    is trashed or pruned simply stops being an answer here, with nothing to
    clean up.
    """
    cache = getattr(ctx, "cache", None)
    if cache is None:
        return []
    out: list[Row] = []
    for job in cache.jobs:
        # The cache page is the workshop and the trash together (that is what
        # lets ``Filters.trash`` be a view), so the trash is excluded here on
        # the row's own ``deleted_at`` -- a column already on the row, costing
        # no DB scan, and independent of whatever the library's filter bar is
        # set to. Home is not the library.
        if job.get("deleted_at"):
            continue
        if job.get("status") != "done":
            continue
        name = str(job.get("name") or job.get("prompt") or job.get("id") or "asset")
        # The same reference/tile/model split the library filter uses: a job
        # that stops at an image belongs to the stage that made it -- asked
        # through ``asset_open`` so the picture is the picture of the place the
        # row actually opens, follow-ups included.
        stage = asset_open.route(job).stage or "reference"
        when = job.get("created_at")
        out.append(
            Row(
                kind="asset",
                key=str(job.get("id") or ""),
                icon=create_stages.ICONS.get(stage, icons.IMAGE),
                name=name,
                when=float(when) if isinstance(when, int | float) else None,
            )
        )
    return out


def _document_rows(ctx: Any) -> list[Row]:
    settings = getattr(ctx, "settings", None)
    if settings is None:
        return []
    return [
        Row(
            kind=entry.kind,
            key=entry.path,
            icon=_MODE_ICONS.get(entry.kind, icons.FILE_IMAGE),
            name=Path(entry.path).name or entry.path,
            when=entry.when,
        )
        for entry in recents.entries(settings)
    ]


#: The last answer :func:`rows` gave, and the state it was an answer *to*.
#: Home asks for this list three times a frame -- the count, the grid and the
#: keyboard -- and the asset half walks the whole job cache page and sorts it
#: each time. Memoised on the cache generation the way ``jobs_cache.visible``
#: is (B19), plus the recent-document list itself, which is settings data and
#: cheap to read: a row cannot go stale without one of the two moving.
_ROWS_CACHE: tuple[Any, list[Row]] | None = None


def _rows_key(ctx: Any, documents: list[Row]) -> Any:
    """The state this list is an answer to, or None if it cannot be one.

    ``None`` when the cache does not count its own generations -- a bare
    ``SimpleNamespace`` in a test, or a headless context -- because a key that
    cannot tell two job pages apart would serve one of them the other's rows.
    """

    cache = getattr(ctx, "cache", None)
    generation = getattr(cache, "_generation", None)
    if generation is None:
        return None
    return (
        id(cache),
        generation,
        tuple((row.kind, row.key, row.when) for row in documents),
    )


def rows(ctx: Any) -> list[Row]:
    """Every Resume row this frame, newest first.

    One merged list rather than a section per mode: the question is "what was I
    working on", and the answer to that does not sort by which editor it was in.
    Three things index this list -- the click, the arrow keys and Enter -- so
    there is exactly one function answering "what is the nth row".
    """
    global _ROWS_CACHE

    documents = _document_rows(ctx)
    key = _rows_key(ctx, documents)
    if key is not None and _ROWS_CACHE is not None and _ROWS_CACHE[0] == key:
        return _ROWS_CACHE[1]
    found = documents + _asset_rows(ctx)
    found.sort(key=lambda row: (row.when is None, -(row.when or 0.0)))
    found = found[:MAX_RESUME]
    if key is not None:
        _ROWS_CACHE = (key, found)
    return found


def activate(ctx: Any, index: int) -> None:
    """Open the ``index``-th row. Shared by the click and by Enter."""
    drawn = rows(ctx)
    if not drawn:
        return
    open_row(ctx, drawn[index % len(drawn)])


def open_row(ctx: Any, row: Row) -> None:
    if row.kind == "asset":
        # ``state.select`` plus a mode, never ``viewer.load_model``:
        # ``_sync_viewer`` owns the pose guard, ``viewer.pending`` and the
        # off-thread parse, and a shortcut past it reproduces two of those three
        # and gets the last one wrong.
        asset_open.open_asset(ctx, row.key)
        return
    path = Path(row.key)
    if not path.exists():
        # A recent list that keeps offering a moved file is worse than a short
        # one, and a click that silently does nothing is worse than either.
        recents.forget(ctx.settings, row.kind, row.key)
        ctx.toast(f"{path.name} is no longer there.", "warn")
        return
    module = KIND_OPENERS.get(row.kind)
    if module is None:
        ctx.toast(f"Nothing here can open a {row.kind} document.", "warn")
        return
    set_mode(ctx.state, _KIND_MODES.get(row.kind, row.kind))
    _open_with(module, ctx, path)


#: Which module's ``open_path`` opens each recent-document kind. One table
#: beside ``_KIND_MODES`` rather than a dict literal inside ``open_row``: the
#: two drifted, and a ``.wsng`` row did nothing on click with no toast.
KIND_OPENERS = {
    "inker": "inker_mode",
    "clay": "clay_mode",
    "plotter": "plotter_mode",
    "packwright": "packwright_mode",
    "sirens": "sirens_io",
}


def _open_with(module: str, ctx: Any, path: Path) -> None:
    from importlib import import_module

    import_module(f"..{module}", __package__).open_path(ctx, path)


def move(ctx: Any, delta: int) -> None:
    """Up/Down between the Resume rows. Wraps: it is a short ring, not a list."""
    count = len(rows(ctx))
    if count:
        ctx.state.home_index = (ctx.state.home_index + delta) % count


# Kept as a name here for its callers and tests; the one clock lives in
# ``widgets`` since 2026-09-05 so the inspector and Home agree on "yesterday".
ago = widgets.ago


# --- the status block -------------------------------------------------------
#
# Four rows, each a sentence about this machine rather than about the work, and
# each clickable where it has somewhere to go. Pure, so the wording and the
# destinations are assertable without a window -- which is the half that went
# wrong the last time a Home row counted something.


@dataclass(frozen=True)
class Status:
    key: str
    icon: str
    text: str
    colour: int
    target: str = ""  # a mode to switch to, or "" for a row that only reports
    #: A ``Filters.status`` value to set on the way, or "" to leave the filter
    #: alone. Only the queue row uses one: "what ran" is reconstructible
    #: nowhere else in the app -- there is no job list -- and the Library's
    #: status filter is the nearest thing to one.
    status_filter: str = ""
    #: A Settings category to open on the way, or "" to leave it alone. Only
    #: the health row uses one: it counts checks that are listed on exactly one
    #: page, and landing on Appearance because that is the remembered tab would
    #: send the reader somewhere the number they clicked is not explained.
    settings_category: str = ""


def _health_status(ctx: Any) -> Status:
    checks = list(getattr(ctx.runtime, "checks", []) or [])
    # Downloads the user has not made yet are **not** "things needing
    # attention": on a fresh install every model row is failing, and counting
    # them here greeted a new user with "17 things need attention" about an
    # app with nothing wrong with it. They have their own row below, which
    # says what they are and offers the button.
    failing = [c for c in checks if not c.ok and not getattr(c, "pending_install", False)]
    fatal = [c for c in failing if c.fatal]
    if not checks:
        return Status("health", icons.LOADER, "Issues - still checking", theme.MUTED)
    if not failing:
        return Status(
            "health",
            icons.CIRCLE_CHECK,
            "Everything checks out",
            theme.MUTED,
            "settings",
            settings_category="health",
        )
    what = "1 thing needs" if len(failing) == 1 else f"{len(failing)} things need"
    return Status(
        "health",
        icons.TRIANGLE_ALERT if not fatal else icons.CIRCLE_ALERT,
        f"{what} attention - see Health",
        theme.ERR if fatal else theme.WARN,
        "settings",
        settings_category="health",
    )


def _queue_status(ctx: Any) -> Status:
    job = getattr(getattr(ctx, "cache", None), "active", None)
    if job is None:
        return Status("queue", icons.CIRCLE, "Queue idle", theme.MUTED)
    name = str(job.get("name") or job.get("prompt") or job.get("id") or "a job")
    if len(name) > 30:
        name = name[:29] + "-"
    percent = ""
    reporter = getattr(ctx, "progress", None)
    found = reporter(str(job.get("id") or "")) if callable(reporter) else None
    if isinstance(found, dict) and isinstance(found.get("percent"), int | float):
        percent = f" {int(found['percent'])}%"
    word = "running" if job.get("status") == "running" else "queued"
    # The one status row that went nowhere, and the only route to a job list
    # there is: nothing in the app lists what has run, so "what happened to
    # that" is otherwise reconstructible only by finding the Library's status
    # filter yourself. Idle stays a pure report -- a filtered list with nothing
    # in it is a worse answer than the sentence already on screen.
    return Status(
        "queue",
        icons.PLAY,
        f"{word}: {name}{percent}",
        theme.ACCENT,
        "library",
        "running",
    )


def _library_status(ctx: Any) -> Status:
    cache = getattr(ctx, "cache", None)
    if cache is None:
        return Status("library", icons.FOLDER_OPEN, "Library", theme.MUTED, "library")
    total = getattr(cache, "total", 0) or 0
    parts = [f"{total} assets"]
    try:
        failed = cache.failures(ctx.state.filters)
    except Exception:
        # Logged as well as swallowed, the rule every comparable site here
        # follows (``pose_panel._open_rig``). Silent, a broken filter or a
        # cache that cannot answer showed up as "0 failed" -- which is not a
        # missing number but a *wrong* one, and the one it is most reassuring
        # to be wrong about. The count is cosmetic, so it still must not take
        # the status line down with it.
        log.exception("could not count failed jobs for the landing status")
        failed = 0
    if failed:
        parts.append(f"{failed} failed")
    storage = getattr(cache, "storage", None)
    if isinstance(storage, dict) and storage.get("bytes"):
        parts.append(format_bytes(storage["bytes"]))
    return Status("library", icons.FOLDER_OPEN, " - ".join(parts), theme.MUTED, "library")


def _review_status(ctx: Any) -> Status | None:
    count = getattr(ctx.state, "home_unreviewed", None)
    if not count:
        # ``None`` is "not asked yet" and 0 is "nothing waiting"; neither is a
        # sentence worth a row, and they are deliberately not distinguished on
        # screen -- the difference matters to the cache, not to the reader.
        return None
    figure = f"{UNREVIEWED_LIMIT}+" if count >= UNREVIEWED_LIMIT else str(count)
    noun = "mesh" if count == 1 else "meshes"
    return Status(
        "review", icons.CIRCLE_CHECK, f"{figure} {noun} unreviewed", theme.MUTED, "review"
    )


def _setup_status(ctx: Any) -> Status | None:
    """"Generation is not set up yet", or None once it is.

    The calm, persistent counterpart to the first-run panel. That panel can be
    dismissed -- deliberately, because an offer that must be answered before
    the app can be used is a toll rather than an offer -- and before this there
    was nothing left saying what the dismissed offer had been about.

    Answered from ``ctx.model_rows`` rather than the disk, ``model_gate``'s
    doctrine and for its reason: this runs on the frame thread. An empty
    snapshot says nothing, so a headless ctx and the first frame before the
    answers land both stay quiet rather than claiming everything is missing.
    """
    from . import model_gate
    from .first_run import GENERATION_ROWS

    rows = model_gate.missing(ctx, GENERATION_ROWS)
    if not rows:
        return None
    total = sum(float(row.get("size_gib") or 0.0) for row in rows)
    noun = "download" if len(rows) == 1 else "downloads"
    return Status(
        "setup",
        icons.DOWNLOAD,
        f"Generation is not set up yet - {len(rows)} {noun}, about {total:.0f} GB",
        theme.MUTED,
        "settings",
        settings_category="models",
    )


def status_rows(ctx: Any) -> list[Status]:
    found = [_health_status(ctx), _queue_status(ctx), _library_status(ctx)]
    setup = _setup_status(ctx)
    if setup is not None:
        found.append(setup)
    review = _review_status(ctx)
    if review is not None:
        found.append(review)
    return found


def visible_home_rows(rows: list[Status]) -> list[Status]:
    """Home's quiet status line.

    The health row is not one of Home's: the rail's badge and the doctor banner
    already say it, and a third rendering of one fact is what the keyword this
    used to take was half-admitting.
    """

    return [row for row in rows if row.key in HOME_STATUS]


def _count_unreviewed(svc: Any) -> int:
    from ..review_mode import SOURCE

    return len(svc.store.unverdicted_models(source=SOURCE, limit=UNREVIEWED_LIMIT))


def pump(ctx: Any) -> None:
    """Re-ask for the unreviewed count when the last answer has gone stale.

    A submit refused by the runner (a key already in flight) leaves the stamp
    alone, so the next frame simply asks again -- the stamp is moved only when
    the request was *accepted*, which is the ``findings_dirty`` rule and the
    reason a burst of requests cannot strand the row on a stale figure.
    """
    now = time.monotonic()
    if now - ctx.state.home_unreviewed_at < UNREVIEWED_TTL:
        return
    if ctx.submit("home-unreviewed", _count_unreviewed, ctx.svc):
        ctx.state.home_unreviewed_at = now


# --- drawing ----------------------------------------------------------------
#
# Two working columns: quick starts and machine state stay stable on the left;
# the larger right side is resumable work.  Release notes remain dismissible,
# so the utility column never turns into a permanent changelog.


#: Which status rows Home draws. ``status_rows`` still computes all of them --
#: it is the shared source the rail's health badge reads, and forking it would
#: be two answers to "is this install healthy" -- but the *library* row is not
#: one of Home's questions any more: it counted assets on a screen whose whole
#: lower half is now assets, and its destination is one click away in the rail.
#: The health row went with it, for the same reason: it was the *third*
#: rendering of one fact, after the rail's badge and the doctor banner -- and
#: ``visible_home_rows`` already suppressed it whenever the banner was up,
#: which is half of this argument made and then stopped.
HOME_STATUS = ("setup", "queue", "review")

#: How many bullets the What's New card shows before "all release notes".
#: Three is a summary; the eighth bullet of a release is a changelog, and a
#: changelog is what the popup is for. Each is drawn as its lead sentence only
#: (``changelog.lead``) -- three whole bullets was the summary in name and the
#: changelog in fact.
NEWS_BULLETS = 3

#: Where the dismissal is remembered. Keyed on the *version* rather than on a
#: boolean, so the next release brings the card back on its own -- a "seen"
#: flag would have to be cleared by whatever wrote the release, and nothing
#: does.
NEWS_SEEN_KEY = "news_seen_version"

#: The Resume grid's thumbnail, in design pixels. The picture is the cell:
#: everything else is one line of name and one of "4m ago" under it.
#: ``tokens.THUMB_CELL``, which Review's labelling grid draws at too.
RESUME_THUMB = tokens.THUMB_CELL


def news_should_show(release: Any, seen: str) -> bool:
    """Whether the What's New card is drawn this frame.

    Pure, and taking the two values rather than a ``Ctx``, because the whole
    question is an interaction between a file and a settings key: a card that
    reappears after being dismissed, and one that never returns for the next
    release, are both bugs with no pixels in them.

    A release with no bullets is not shown. ``changelog.current`` falls back to
    the newest entry when the running version has no section of its own, which
    is right for "what changed recently" and would be wrong for a card that
    names a version -- but the fallback still carries bullets, so the empty
    case here is the genuinely empty one.
    """
    if release is None or not release.bullets:
        return False
    return seen != release.version


def draw(ctx: Any) -> None:
    pump(ctx)
    _header(ctx)
    # Once per draw: ``_status`` is the only consumer now, but the rule stands
    # for the reason it was written -- recomputing is two walks over the
    # runtime checks and the queue for one line of sentences.
    status = status_rows(ctx)
    # A scroller, because the Resume grid is as tall as it needs to be and the
    # window is not. Borderless -- this is the mode's whole surface, and a frame
    # drawn around the only thing on screen is a frame around nothing -- but
    # ``always_use_window_padding`` with it: a *borderless* child gets zero
    # window padding by default, which put the heading's first letter and the
    # first card's left edge hard against the window's.
    if imgui.begin_child("landing/body", (0, 0), imgui.ChildFlags_.always_use_window_padding.value):
        # Home draws into ``##content`` rather than through ``layout.pane``, so
        # it asks for its own section blocks; the bracket is this child, which
        # is the only thing that knows where it ends.
        avail = imgui.get_content_region_avail().x
        gap = imgui.get_style().item_spacing.x
        left_w = min(sp(390), max(avail * 0.34, sp(260)))
        left_w = min(left_w, max(avail - sp(300) - gap, avail * 0.48))
        if imgui.begin_child("landing/quick", (left_w, 0)):
            with widgets.section_blocks():
                _recovery(ctx)
                _tour_offer(ctx)
                _news(ctx)
                _start(ctx)
                _status(ctx, status)
                widgets.end_section()
                _news_footer(ctx)
        imgui.end_child()
        imgui.same_line()
        if imgui.begin_child("landing/resume", (0, 0)):
            with widgets.section_blocks():
                _resume(ctx)
        imgui.end_child()
    imgui.end_child()


TOUR_DISMISSED_KEY = "tour_offer_dismissed"


def _tour_offer(ctx: Any) -> None:
    """The way in for someone who has just installed the app.

    Shown until it is taken or dismissed, and *not* as a modal on the first
    frame: the recovery prompt above was a modal once and the argument that
    retired it applies here too -- an offer that has to be answered before the
    app can be used is a toll, not an offer.

    It stands down once the tour it offers has been finished, so the reader is
    not invited to a thing they have already done.

    **The dismissal is per tour.** "Not now" wrote one global flag, so a single
    press retired the card for all four tours permanently -- including the two
    Sirens ones, which sit behind the other two in the list and which most
    readers would therefore never be offered at all. It is now the set of tours
    that have been declined, and a legacy ``"1"`` still means all of them, so an
    installed user who already pressed it does not get the card back.
    """
    from ..tour import scripts as tour_scripts

    state = getattr(ctx.state, "tour", None)
    if state is None or state.running:
        return
    dismissed = _dismissed(ctx)
    offer = next(
        (
            one
            for one in tour_scripts.TOURS
            if one.key not in state.finished and one.key not in dismissed
        ),
        None,
    )
    if offer is None:
        return
    widgets.section("New here?")
    imgui.text_wrapped(offer.blurb)
    if widgets.primary_button(f"Start: {offer.title}##tour-offer"):
        from . import tour as tour_pane

        tour_pane.start(ctx, offer.key)
    imgui.same_line()
    if controls.button("Not now##tour-offer", role=controls.ButtonRole.GHOST):
        ctx.settings.set(TOUR_DISMISSED_KEY, sorted(dismissed | {offer.key}))


def _dismissed(ctx: Any) -> set[str]:
    """Which tours have been declined. A list-valued setting -- ``tours_finished``
    is already one -- with the old ``"1"`` read as "all of them"."""
    from ..settings import as_list
    from ..tour import scripts as tour_scripts

    stored = ctx.settings.get(TOUR_DISMISSED_KEY, None)
    if isinstance(stored, str):
        return {one.key for one in tour_scripts.TOURS} if stored else set()
    return {str(one) for one in as_list(stored)}


def _header(ctx: Any) -> None:
    with fonts.display(imgui):
        imgui.text("Warlock Studio")
    imgui.same_line()
    with fonts.small(imgui):
        # The app's version, on screen for the first time: ``main._version``
        # existed only for crash provenance, and "which build is this" was a
        # question the UI could not answer at all.
        widgets.muted(_version())
    manual_render.help_button(ctx, "home")
    # One line saying what the app is. The rail names the workspaces and the
    # manual explains them, but nothing on screen said what any of it was for
    # -- and Ctrl+K is the only keyboard route to a mode, advertised nowhere.
    widgets.muted_wrapped(
        "Local AI asset generation: prompts and drawings into references, "
        "sprites and 3D models. Press Ctrl+K for the command palette."
    )
    imgui.dummy((0, sp(tokens.SP_3)))


# The installed version, once: ``main._version`` is an importlib.metadata
# distribution walk, the header and the news block both ask, and an installed
# version cannot change under a running process -- the changelog cache's own
# argument, applied to the other half of "what build is this".
_VERSION: str | None = None


def _version() -> str:
    global _VERSION
    if _VERSION is None:
        from ... import installed_version

        _VERSION = installed_version()
    return _VERSION


# --- what a crashed session left ---------------------------------------------
#
# This used to be a modal on the first frame: "Recover unsaved work?", Recover /
# Not now, all of them or none of them, asked before the user had seen the app.
# Three things were wrong with that and only one of them was the modality.
#
# It was **all-or-nothing**, so a session that crashed with one document worth
# keeping and two worth abandoning had no answer that was right. It was
# **once**, so "Not now" was indistinguishable from "never" until the next
# launch. And it was **in front of everything**, which is the wrong weight for a
# question whose most common answer arrives before the user has any context for
# it -- a dialog on a screen you have not read yet is a dialog you dismiss.
#
# As a section it is per-row, it stays until it is dealt with, and it is exactly
# as loud as it needs to be. What did *not* move is the scan: see
# ``journal.snapshot``, which is called on the first frame and never again.

#: Journal kind -> the mode recovering it should switch to.
#:
#: Only the *mode* is written down; the glyph comes off ``_MODE_ICONS`` as every
#: other row's does, for that table's own reason -- a hand-copied glyph is how a
#: row ends up opening Clay under Plotter's icon.
#:
#: Two of the six do not name their own mode, and they differ in why. A pose is
#: authored in Poser, which is a plain alias. A profile draft is not a mode at
#: all: it is a sheet over Create, and its provider's ``adopt`` already puts the
#: app where it needs to be. The empty string means exactly that -- recovering
#: it navigates itself, and this pane must not second-guess where to.
_KIND_MODES = {
    "inker": "inker",
    "clay": "clay",
    "plotter": "plotter",
    "packwright": "packwright",
    "sirens": "sirens",
    "pose": "poser",
}

#: How wide a recovery row's button is, in design pixels. Fixed rather than
#: sized to its label so a column of them shares one right edge -- the titles
#: beside them are document names and will not.
RECOVER_BUTTON = 96.0


def _recovery(ctx: Any) -> None:
    """One row per document a previous session left behind.

    Rendered from :func:`journal.snapshot`, which is a snapshot in the strict
    sense -- see its docstring for why re-reading the directory here would start
    offering the user their own open documents back.
    """
    from .. import journal

    found = list(journal.snapshot(ctx))
    if not found:
        return
    widgets.section("Unsaved work")
    widgets.muted_wrapped(
        "Warlock closed with these still open. Recovering gives you the document "
        "as it was; you still choose where to save it."
    )
    imgui.dummy((0, sp(tokens.SP_2)))
    # ``MAX_RECOVERY`` rows at a time, and the cap lives here rather than in
    # the scan -- see that constant. The rest are counted rather than hidden:
    # a row leaves the snapshot the moment it is recovered or discarded, so
    # the next one takes its place on the following frame and every document
    # is eventually reachable. Truncating the scan instead simply lost them.
    shown = found[: journal.MAX_RECOVERY]
    for entry in shown:
        _recovery_row(ctx, journal, entry)
    waiting = len(found) - len(shown)
    if waiting:
        with fonts.small(imgui):
            widgets.muted(
                f"and {waiting} more, shown as these are dealt with. "
                f"Discard all removes all {len(found)}."
            )
    imgui.dummy((0, sp(tokens.SP_2)))
    # Plural regardless of the count, and last: discarding is the destructive
    # answer, so it does not sit above the rows it would delete, and it is not
    # per-row because a per-row bin next to a per-row Recover is two small
    # targets one pixel apart with opposite meanings.
    #
    # It discards ``found``, not ``shown``: "all" has to mean all, which is
    # why the line above says how many that is whenever the two differ.
    if controls.button(f"{icons.TRASH} Discard all##recovery-discard"):
        for entry in found:
            journal.discard(ctx, entry)
        ctx.toast("The recovered copies were discarded.", "info")
    # Everything below this on Home -- the release note, the New button, the
    # status line -- has no heading of its own, so the block has to be closed
    # here or it grows to cover all of it. See ``widgets.end_section``.
    widgets.end_section()


def _recovery_row(ctx: Any, journal: Any, entry: Any) -> None:
    """One document: what it is, when it was copied, and the button.

    A row rather than a ``widgets.card``: the Resume grid below is pictures
    because recognising your own work is a visual act, and there is no picture
    for a document that was never opened in this session -- only a name. A grid
    of six identical glyph placeholders would say less than six lines do.
    """
    mode = _KIND_MODES.get(entry.kind, "")
    imgui.push_id(f"recovery-{entry.path.name}")
    imgui.text(_MODE_ICONS.get(mode, icons.FILE_IMAGE))
    imgui.same_line()
    provider = journal.provider_for(entry.kind)
    stamp = ago(entry.at)
    what = provider.label if provider is not None else entry.kind
    note = f"{what}, {stamp}" if stamp else what
    width = sp(RECOVER_BUTTON)
    # Trimmed to what the row can actually hold -- the note beside it and the
    # right-aligned button are both fixed, so an untrimmed title simply ran
    # under the Recover button. Measured, not counted, like the Resume grid's
    # cells (``widgets.fit_text`` says why a character budget is wrong).
    with fonts.small(imgui):
        note_w = imgui.calc_text_size(note).x
    spacing = imgui.get_style().item_spacing.x
    room = imgui.get_content_region_avail().x - note_w - width - spacing * 3
    imgui.text(widgets.fit_text(entry.title, max(room, sp(48))))
    imgui.same_line()
    with fonts.small(imgui):
        widgets.muted(note)
    imgui.same_line()
    # Right-aligned, so a column of them lines up whatever the titles do.
    imgui.same_line(imgui.get_content_region_avail().x + imgui.get_cursor_pos_x() - width)
    if entry.adoptable:
        if controls.button(f"Recover##{entry.path.name}", (width, 0)):
            if not journal.take(ctx, entry):
                ctx.toast(f"{entry.title} could not be reopened.", "error")
            elif mode:
                # Straight to the editor holding it. The old modal did not
                # navigate and did not need to -- it fired before the user had
                # chosen anywhere to be -- but a button on the home screen that
                # reopens a document somewhere you cannot see has, as far as the
                # user can tell, done nothing at all. An empty mode is a
                # provider that navigates itself; see ``_KIND_MODES``.
                set_mode(ctx.state, mode)
    else:
        # A kind this build does not have a provider for. Greyed rather than
        # hidden, and the files are left alone: the mode may simply not be
        # compiled into this run, and silently dropping the row would look like
        # the work was never saved.
        widgets.muted("unavailable")
    imgui.pop_id()


def _news(ctx: Any) -> None:
    """The dismissible What's New card, or nothing at all."""
    release = changelog.current(_version())
    seen = str(ctx.settings.get(NEWS_SEEN_KEY, "") or "")
    if not news_should_show(release, seen):
        return
    # Leads, not the whole bullets. Every bullet in the shipped file opens with
    # a sentence naming what changed and then argues for it at ~150 words, so
    # three of them took roughly a quarter of Home above the fold. The full
    # text is one ghost button away, at the foot of this same screen.
    bullets = tuple(changelog.lead(text) for text in release.bullets[:NEWS_BULLETS])
    style = imgui.get_style()
    pad = style.window_padding
    avail = imgui.get_content_region_avail().x
    # The width ``text_wrapped`` will actually use inside the card, because
    # ``widgets.card`` has no scrollbar by design: a bullet measured against a
    # different number than it is drawn at is a bullet clipped off the bottom.
    # The dismiss button shortens only the *headline*'s line, which is measured
    # unwrapped anyway.
    wrap = max(avail - pad.x * 2, sp(120))
    with fonts.label(imgui):
        height = imgui.get_text_line_height() + style.item_spacing.y
    with fonts.small(imgui):
        for bullet in bullets:
            # The second positional is ``text_end``, not the hide-after-##
            # flag: the wrap width is the fourth argument, and passing it third
            # measures an unwrapped line.
            height += (
                imgui.calc_text_size(f"- {bullet}", None, False, wrap).y + style.item_spacing.y
            )
    height += pad.y * 2
    with widgets.card("landing/news", (0, height)) as visible:
        if visible:
            _card_margin(pad)
            with fonts.label(imgui):
                imgui.text(f"What's new in {release.version}")
            offset = imgui.get_cursor_pos_x() + imgui.get_content_region_avail().x
            imgui.same_line(max(offset - sp(24), 0.0))
            if widgets.small_icon_button(
                f"{icons.X}##landing-news-dismiss", "Dismiss", borderless=True
            ):
                ctx.settings.set(NEWS_SEEN_KEY, release.version)
            for bullet in bullets:
                with fonts.small(imgui):
                    widgets.muted_wrapped(f"- {bullet}")
            imgui.unindent(pad.x)
    imgui.dummy((0, sp(tokens.SP_3)))


def _news_footer(ctx: Any) -> None:
    """The way back to every release, whether or not the card is showing.

    A ghost at the foot of the screen rather than a section of its own: the
    full changelog is a thing you go and look up, and it spent the whole of
    this pane's first design taking half the width to be looked up in.
    """
    imgui.dummy((0, sp(tokens.SP_4)))
    if widgets.ghost_button("All release notes..."):
        imgui.open_popup("landing-news-all")
    _news_popup()


def _news_popup() -> None:
    if not imgui.begin_popup("landing-news-all"):
        return
    widgets.popup_chrome(_imgui=imgui)
    releases = changelog.entries()
    if not releases:
        widgets.muted("No changelog shipped with this build.")
        imgui.end_popup()
        return
    current = changelog.current(_version())
    if imgui.begin_child(
        "landing/news-scroll", (sp(tokens.SURFACE_W_SHEET), sp(420))
    ):
        # The current release open and the rest collapsed: "what changed" is
        # about this build, and the older ones are there so the answer has a
        # context rather than so it has a scrollback.
        for index, release in enumerate(releases):
            headline = release.version + (f" - {release.date}" if release.date else "")
            opened = current is not None and release.version == current.version
            flags = imgui.TreeNodeFlags_.default_open.value if opened else 0
            if controls.collapsing_header(f"{headline}##news{index}", flags):
                for bullet in release.bullets:
                    with fonts.small(imgui):
                        # Full text for *this* build, leads for the history --
                        # ``lead``'s own rule, applied one surface further out.
                        #
                        # ``CHANGELOG.md`` is written for the person maintaining
                        # this, and its candour is an asset: it records internal
                        # review scores, the measurement that made a default
                        # wrong, and what a crash actually was. That is exactly
                        # the wrong first impression, though, and this popup put
                        # every word of it in front of anyone who clicked "All
                        # release notes..." -- forty releases of debugging
                        # narrative reading as forty releases of instability.
                        #
                        # Truncating rather than softening the file: the fix for
                        # tone here is a shorter surface, not less honesty in the
                        # record.
                        widgets.muted_wrapped(
                            f"- {bullet if opened else changelog.lead(bullet)}"
                        )
    imgui.end_child()
    imgui.end_popup()


def _start(ctx: Any) -> None:
    """One primary and a menu, where six equal buttons used to be.

    The six were a 3-across grid of identically-weighted buttons -- a menu
    claiming all of them matter equally, drawn at the loudest weight the pane
    had, above the thing the user actually came back for. The menu says the
    same six words; the button says there is one place to start.
    """
    if widgets.primary_button(f"{icons.PLUS} New...", (sp(160), 0)):
        imgui.open_popup("landing-new")
    if imgui.begin_popup("landing-new"):
        widgets.popup_chrome(_imgui=imgui)
        for key, label, icon, action in NEW_ITEMS:
            if controls.menu_item(f"{icon}  {label}##landing-new-{key}", "", False)[0]:
                action(ctx)
        imgui.end_popup()
    # Beside ``New...`` and quieter than it, deliberately on both counts.
    # Importing is not one of ``NEW_ITEMS``: those are "the things this app can
    # start **from nothing**", and a file the user already has is the opposite
    # errand -- folding it in would make that sentence false. But it is the
    # errand somebody with a mesh arrives *with*, and until 2026-08-30 there
    # was no way to do it at all: geometry entered the library only by being
    # reconstructed or built in Clay, and a dropped ``.glb`` reached Clay
    # alone, which refuses a rigged one because it has no skinning.
    imgui.same_line()
    if widgets.ghost_button(f"{icons.BOX} Import mesh..."):
        from . import library

        library.pick_and_import_mesh(ctx)
    imgui.dummy((0, sp(tokens.SP_2)))


def _status(ctx: Any, status: list[Status]) -> None:
    """One quiet line: what is wrong, what is running, what is waiting.

    Four stacked rows became one line because three of the four say "nothing to
    report" on a healthy idle install -- a block of chrome saying so at the top
    of every launch. What is left is only the rows that are *actionable*, and
    each still carries its own destination.
    """
    gap = imgui.get_style().item_spacing.x
    first = True
    for row in visible_home_rows(status):
        width = imgui.calc_text_size(row.icon).x + imgui.calc_text_size(row.text).x + gap * 3
        if not first:
            widgets.same_line_or_wrap(width)
        first = False
        widgets.text_colored(row.colour, row.icon)
        imgui.same_line()
        if row.target:
            # A button with no fill until it is pointed at: this is a *line*,
            # and three filled pills across the top of Home is three calls to
            # action for "everything checks out". The ghost register, at
            # small_button height so it sits on the line rather than raising it.
            imgui.push_style_color(imgui.Col_.text.value, imgui.ImVec4(*theme.rgba(row.colour)))
            imgui.push_style_color(imgui.Col_.button.value, imgui.ImVec4(0.0, 0.0, 0.0, 0.0))
            imgui.push_style_color(
                imgui.Col_.button_hovered.value,
                imgui.ImVec4(*theme.rgba(theme.ELEV_2)),
            )
            clicked = controls.small_button(f"{row.text}##landing-status-{row.key}")
            imgui.pop_style_color(3)
            if imgui.is_item_hovered():
                imgui.set_mouse_cursor(imgui.MouseCursor_.hand.value)
            if clicked:
                if row.status_filter:
                    ctx.state.filters.status = row.status_filter
                if row.settings_category:
                    from .app_settings import CATEGORY_SLOT

                    ctx.state.preview[CATEGORY_SLOT] = row.settings_category
                set_mode(ctx.state, row.target)
        else:
            widgets.text_colored(row.colour, row.text)


def _resume(ctx: Any) -> None:
    """What you were working on, as pictures.

    A grid rather than a list, and the dominant thing on the screen rather than
    half of one column. Recognising your own work is a visual act: the row that
    said ``chest_of_drawers_v3.wmap - 2h ago`` was a filename and a clock, and
    the picture that would have identified it in one glance was sitting in the
    job directory the whole time.
    """
    widgets.section("Resume")
    drawn = rows(ctx)
    if not drawn:
        widgets.muted("Nothing yet. Start something above, or press Ctrl+K.")
        return
    style = imgui.get_style()
    pad = style.window_padding
    gap = style.item_spacing.x
    thumb = sp(RESUME_THUMB)
    cell_w = thumb + pad.x * 2
    cell_h = thumb + pad.y * 2 + imgui.get_text_line_height_with_spacing() * 2
    avail = imgui.get_content_region_avail().x
    columns = max(int((avail + gap) // (cell_w + gap)), 1)
    focus = ctx.state.home_index % len(drawn)
    for index, row in enumerate(drawn):
        if index % columns:
            imgui.same_line()
        _resume_cell(ctx, row, index, (cell_w, cell_h), thumb, pad, focused=index == focus)


def _card_margin(pad: Any) -> None:
    """Open a gutter inside a ``widgets.card``.

    A card is a *borderless* child as far as imgui is concerned -- it draws its
    own fill and its own border on the parent's list -- so it gets zero window
    padding, and everything in one sat flush against its left edge. Indented
    rather than pushed as a style var, because the push would have to reach
    inside ``widgets.card``'s context manager to land before ``begin_child``.

    The caller unindents; the pair is deliberately not a context manager,
    because the cards here already sit inside one.
    """
    imgui.dummy((0, pad.y - imgui.get_style().item_spacing.y))
    imgui.indent(pad.x)


def _resume_cell(
    ctx: Any,
    row: Row,
    index: int,
    size: tuple[float, float],
    thumb: float,
    pad: Any,
    *,
    focused: bool,
) -> None:
    """One cell: the picture, the name, and how long ago.

    The whole card is the affordance, as a library card's is -- the click test
    follows ``end_child`` because the child window *is* the item, which is what
    makes the hover lift work there too.

    ``focused`` is drawn as a ring rather than as a fill, because the fill is
    already spoken for by hover: a keyboard cursor that looked exactly like the
    pointer being somewhere else is a cursor you cannot find.
    """
    imgui.push_id(f"landing-cell-{row.kind}-{row.key}")
    origin = imgui.get_cursor_screen_pos()
    with widgets.card(f"landing/cell/{index}", size) as visible:
        if visible:
            _card_margin(pad)
            job = ctx.cache.get(row.key) if row.kind == "asset" else None
            if job is not None:
                thumbs.job_thumb(ctx, job, thumb)
            else:
                # A document has no rendered picture yet -- the mode glyph in a
                # frame is what a library card draws for a job with no
                # thumbnail, and the same placeholder here says "this is a
                # drawing" rather than "this is broken".
                widgets.thumb_placeholder(thumb, row.icon)
            name = widgets.fit_text(row.name, thumb)
            imgui.text(name)
            if name != row.name and imgui.is_item_hovered():
                imgui.set_tooltip(row.name)
            stamp = ago(row.when)
            if stamp:
                with fonts.small(imgui):
                    widgets.muted(stamp)
            imgui.unindent(pad.x)
    if imgui.is_item_hovered():
        imgui.set_mouse_cursor(imgui.MouseCursor_.hand.value)
        # Hovering moves the keyboard cursor too, so the two never disagree
        # about which cell Enter would take.
        ctx.state.home_index = index
        if row.kind != "asset":
            imgui.set_tooltip(row.key)
    if imgui.is_item_clicked():
        open_row(ctx, row)
    if focused:
        # ImVec2 and not a tuple: ``ring`` reads ``.x``/``.y``, which is the
        # shape every other caller hands it (a rect min/max straight off imgui).
        widgets.ring(
            imgui.ImVec2(origin.x, origin.y),
            imgui.ImVec2(origin.x + size[0], origin.y + size[1]),
            theme.ACCENT,
            1.0,
            2.0,
        )
    imgui.pop_id()


# --- the actions ------------------------------------------------------------


def start_2d(ctx: Any) -> None:
    """A clean prompt form at the Reference stage.

    ``default_form_2d`` rolls its own seed, so this is genuinely a fresh start
    rather than last session's form with the prompt cleared.
    """
    ctx.state.form_2d = default_form_2d()
    ctx.state.select(None)
    create_stages.go(ctx, "reference")


def start_3d(ctx: Any) -> None:
    """A clean mesh form at the Mesh stage.

    Still a separate entry on the New... menu even though the two stages are
    one mode: "I have an image and I want a mesh of it" is a different errand
    from "I want a picture of a barrel", and the menu is a list of errands.
    """
    ctx.state.form_3d = dict(DEFAULT_FORM_3D)
    ctx.state.select(None)
    create_stages.go(ctx, "mesh")


def start_inker(ctx: Any) -> None:
    """Inker keeps whatever was open: unlike the two generate panes, there is
    no "fresh form" here -- the documents *are* the work, and its own empty
    state offers the canvas sizes."""
    set_mode(ctx.state, "inker")


def start_clay(ctx: Any) -> None:
    """Clay keeps whatever was open, as Inker does.

    The one addition is an empty Clay: the button says "new model", so arriving
    at a mode with nothing in it and no obvious way to begin is a dead end. A
    document is minted only when there are none -- opening one over existing
    work would break the keeps-whatever-was-open contract.
    """
    from .. import clay_mode

    set_mode(ctx.state, "clay")
    if not clay_mode.ensure(ctx).docs:
        clay_mode.new_document(ctx)


def start_plotter(ctx: Any) -> None:
    from .. import plotter_mode

    set_mode(ctx.state, "plotter")
    if not plotter_mode.ensure(ctx).docs:
        # Asks rather than invents. Entering Plotter from Home *was* the act of
        # creating a map, silently and at whatever the default happened to be.
        plotter_mode.ask_new_document(ctx)


def start_packwright(ctx: Any) -> None:
    from .. import packwright_mode

    set_mode(ctx.state, "packwright")
    if not packwright_mode.ensure(ctx).docs:
        packwright_mode.new_document(ctx)


def start_sirens(ctx: Any) -> None:
    """Sirens from Home, and a song to type into when it arrives.

    The four document modes' shape rather than Troupe's: a tracker with no
    document is a grid of nothing, and ``new_song`` is deliberately not an
    empty document -- five channels, a pattern and an order that points at it,
    so the first note typed makes a sound.
    """
    from .. import sirens_mode

    set_mode(ctx.state, "sirens")
    if not sirens_mode.ensure(ctx).docs:
        sirens_mode.new_document(ctx)


def start_troupe(ctx: Any) -> None:
    """Troupe from Home. It asks for nothing and invents nothing.

    Unlike the four document modes above it, entering Troupe does not create
    anything: the mode is a selection over character sheets a worker published,
    and what a user does first here is either watch one or describe a new
    character in the form. Creating something on entry would be the "entering
    Plotter *was* the act of creating a map" mistake the function above it
    exists to have stopped making.
    """
    from .. import troupe_mode

    set_mode(ctx.state, "troupe")
    troupe_mode.ensure(ctx)


#: The eight things this app can start from nothing, in the order the menu
#: offers them. "New 3D model" and "New Model" used to sit side by side and the second
#: one meant *Clay* -- two buttons whose labels differ by a word neither of them
#: defines, one of which generates a mesh from a prompt and the other opens a
#: modelling workspace (UX-23). Named for what they do instead: the mode is the
#: noun a user can act on.
#:
#: Below the functions rather than above them, because it names them. The keys
#: are the menu's own ids and nothing else -- the first two stopped being mode
#: keys in wave 5, and are spelled for the stages they open so that a reader
#: cannot take them for modes that no longer exist.
NEW_ITEMS: tuple[tuple[str, str, str, object], ...] = (
    ("create-reference", "New 2D image", icons.IMAGE, start_2d),
    ("create-mesh", "New 3D model", icons.BOX, start_3d),
    ("inker", "New drawing", icons.PEN_TOOL, start_inker),
    ("clay", "New Clay model", icons.RULER, start_clay),
    ("plotter", "New tile map", icons.GRID, start_plotter),
    ("packwright", "New sprite atlas", icons.LAYERS, start_packwright),
    ("sirens", "New song", icons.AUDIO_WAVEFORM, start_sirens),
    ("troupe", "New character", icons.PERSON_STANDING, start_troupe),
)
