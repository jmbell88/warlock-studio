"""Everything the UI remembers between frames.

Plain dataclasses with no imgui in sight, so the interesting parts -- which
action a card offers, how the ETA is smoothed, what a filter matches -- are
testable as ordinary Python. imgui's own widget state (a text buffer's cursor,
a combo's scroll) stays inside imgui; this is the application's state, and the
line between the two is that anything here would still make sense if the app
were driven by a script.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any

# The prompt history the 2D pane offers. Twenty is what the browser kept: long
# enough to find yesterday's phrasing, short enough to scan.
MAX_HISTORY = 20

def guidance_fields() -> tuple[str, ...]:
    """The taxonomy selects the 2D pane owns.

    Read off ``guidance.form_fields()`` rather than listed here, so a new table
    in guidance.py appears in the pane without an edit -- the same reason the
    HTTP layer had one ``_pick_guidance``. The exceptions are fields the pane
    renders as something other than a plain combo: the two model pickers sit
    under Advanced, the two conditioning pickers live in the Reference section
    and are hidden until there is an image to condition on, and ``platform``
    is deliberately split in two.
    """
    from .. import guidance

    return tuple(
        f
        for f in guidance.form_fields()
        if f not in ("base_model", "style_lora", "platform", "ip_adapter", "control")
    )


def default_form_2d() -> dict[str, Any]:
    """What a submit is composed from.

    Split across the two panes deliberately: a control belongs to exactly one
    of them, and ``platform`` is *two* fields because one control cannot be
    owned by two panes -- the 2D pane's is a prompt fragment, the 3D pane's is
    the geometry resolution.

    The seed is rolled here rather than being a constant: it is deliberately
    not persisted (settings.VOLATILE), so a literal default meant every launch
    opened on the same seed and a first Generate reproduced last week's image.
    """
    from ..service.validation import random_seed

    form: dict[str, Any] = {
        "prompt": "",
        "negative_prompt": "",
        "base_model": "",
        "style_lora": "",
        "lora_weight": 0.9,
        "seed": random_seed(),
        "seed_locked": False,
        "count": 1,
        "platform": "",
        # reference | tile. What this pane submits. A tile is the same pipeline
        # with circular padding and a different framing template, so it belongs
        # to the pane that owns the prompt rather than to a mode of its own --
        # and it is persisted, unlike the seed, because someone making a
        # texture set is making several.
        "output": "reference",
        # Conditioning. Every number is a float literal on purpose:
        # restore_form gates on `type(value) is type(default)`, so an int here
        # would make a persisted 0.6 fail to restore.
        "ref_path": "",
        "ip_adapter": "",
        "ip_scale": 0.6,
        "control": "",
        "control_scale": 0.65,
        "control_end": 0.8,
    }
    form.update(dict.fromkeys(guidance_fields(), ""))
    return form

def form_from_params(params: dict[str, Any]) -> dict[str, Any]:
    """A 2D form filled from a finished job's params -- "another like this".

    Only keys the form already has, which is what keeps a derived value the
    worker recorded (``composed_prompt``, ``reference_report``, a mesh verdict)
    from ever becoming a submitted field: the form is the allowlist.

    Numbers are coerced rather than type-matched. Params arrive from JSON,
    where a strength saved as 0.6 can come back as an int 0 or 1, and the
    persisted-settings merge's ``type(value) is type(default)`` rule would
    silently drop exactly those fields.
    """
    form = default_form_2d()
    for key, default in list(form.items()):
        value = params.get(key)
        if value is None:
            continue
        try:
            if isinstance(default, bool):
                form[key] = bool(value)
            elif isinstance(default, float):
                form[key] = float(value)
            elif isinstance(default, int):
                form[key] = int(value)
            elif isinstance(default, str):
                form[key] = str(value)
        except (TypeError, ValueError):
            continue
    return form


DEFAULT_FORM_3D: dict[str, Any] = {
    "platform": "",
    "profile": "raw",
    # Deliberately without a widget. A triangle budget only means anything for
    # profile "custom", and the generate form offers ``raw`` alone until a tier
    # has been qualified -- so a control here would offer a number that "raw"
    # ignores. The plumbing through _payload is kept because it is correct the
    # moment a tier is exposed; the retarget control on a finished mesh is where
    # a budget is actually chosen today.
    "custom_triangles": 0,
    "size_m": 0.0,
    "bg_removal": "",
    "mesh_seed": 0,
    "mesh_seed_locked": False,
    # How many meshes one Make 3D queues. 1 is the old behaviour exactly.
    # Reliability rather than variety: trellis is deterministic in its seed and
    # its failure mode is a lottery, so three attempts and a picker is the
    # cheapest answer to a hole through the shoulder. Bounded at
    # validation.MAX_MESH_CANDIDATES, because each one is two minutes of a
    # serial worker.
    "candidates": 1,
    "rig": False,
    "rig_template": "",
    # Whether the host recentres and rescales the subject before the trellis
    # upload. False, matching queue.DEFAULT_REFERENCE_PREP: whether
    # normalising here helps or fights the exe's own preprocessing is
    # unmeasured, so the pane offers the switch rather than turning it on for
    # everyone. Flip both together once the occupancy sweep says which way.
    "reference_prep": False,
}


@dataclass
class Filters:
    """The library's filter bar. Persisted, because a workshop is filtered the
    same way every session."""

    text: str = ""
    status: str = "all"  # all | done | running | error
    kind: str = "all"  # all | reference | model | rig | sheet
    favorites_only: bool = False
    # newest | best. Persisted with the rest of the filter bar, because a
    # workshop is browsed the same way every session.
    sort: str = "newest"

    def matches(self, job: dict[str, Any]) -> bool:
        if job.get("sweep_id"):
            # A sweep's units are dozens of near-identical rows whose whole
            # purpose is to be compared against each other in Review. Left in,
            # one launched sweep buries a workshop's actual assets. They are
            # reachable by their sweep, and deleting the sweep deletes them.
            return False
        if job.get("candidate_group"):
            # And the same rule for a mesh candidate nobody has picked yet:
            # three attempts at one asset are three near-identical cards, and
            # the choice between them belongs in the picker rather than in a
            # list of finished work. The column goes NULL the moment the user
            # keeps one (``service.jobs.keep_candidate`` dissolves the whole
            # group), so nothing stays hidden without a picker to reach it by.
            return False
        if self.favorites_only and not job.get("favorite"):
            return False
        if self.status != "all" and job.get("status") != self.status:
            return False
        if self.kind != "all" and card_kind(job) != self.kind:
            return False
        if self.text:
            needle = self.text.strip().lower()
            haystack = " ".join(
                str(job.get(k) or "") for k in ("name", "prompt", "tags", "id")
            ).lower()
            if needle not in haystack:
                return False
        return True

    def failures(self, jobs: list[dict[str, Any]]) -> int:
        """How many rows switching the status filter to "error" would reveal.

        Derived by re-running :meth:`matches` under that one substitution
        rather than by counting ``status == "error"``, so the number is exactly
        what the click produces: the kind, text and favourites filters still
        apply afterwards, and a count that ignored them would offer a jump to
        an empty list. Sweep units stay excluded for the same reason they are
        excluded from the list -- one failed sweep is dozens of rows nobody
        can see.
        """
        probe = replace(self, status="error")
        return sum(1 for job in jobs if probe.matches(job))

    def order(self, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The list in the order the bar asks for.

        Stable and non-destructive: "newest" returns the query's own order
        untouched, and "best" is a stable sort, so candidates that share a
        score stay in submission order rather than shuffling every refresh.
        """
        if self.sort != "best":
            return list(jobs)

        def key(job: dict[str, Any]) -> tuple[int, float]:
            rank = (job.get("params") or {}).get("rank")
            if not isinstance(rank, dict):
                # Unranked sorts into its own bucket *below* every scored job,
                # a refused one included. Most of a workshop predates the
                # score, so treating "no score" as a middling one would scatter
                # hundreds of old assets through the ranked candidates the
                # sort exists to surface -- and treating it as zero would tie
                # it with the refused, which the bucket keeps distinct.
                return (1, 0.0)
            try:
                return (0, -float(rank.get("score") or 0.0))
            except (TypeError, ValueError):
                return (1, 0.0)

        return sorted(jobs, key=key)


def card_kind(job: dict[str, Any]) -> str:
    """What the filter calls this row.

    Not simply job["kind"]: a text job that stops at a reference, one that
    stops at a tile and one that goes on to a mesh are all the same kind and
    three different things to look for.
    """
    if job.get("kind") in ("rig", "sheet"):
        return job["kind"]
    stage = job.get("stage")
    if stage in ("reference", "tile"):
        return stage
    return "model"


# How long each level of toast stays up, in seconds (H68). Four levels, and
# the ladder is the *reading time* each one asks for rather than a severity
# score: a confirmation is read at a glance, a warning is a sentence, and an
# error usually says what to do next and four seconds is not long enough to
# read a sentence and act on it.
#
# ``success`` and ``warn`` are new, and the gap they fill is real: everything
# that finished, and everything that half-worked, has been arriving in the same
# neutral grey as "settings copied to the form". A level with no entry here
# falls back to ``info`` -- the same rule ``Toast.action`` follows, and what
# lets a level be introduced from the calling side.
TOAST_LEVELS: dict[str, float] = {
    "info": 4.0,
    "success": 4.0,
    "warn": 6.0,
    "error": 8.0,
}

# The levels that take mouse input and offer a close button: a message the user
# may need to act on must not be un-clickable, and must not vanish under the
# cursor. See ``widgets.toasts``.
TOAST_STICKY = frozenset({"warn", "error"})

# How many past toasts the history keeps. A session's worth of notices, not a
# log file: the log file is the log file, and this is the thing you check
# because something flashed past while you were looking at the viewport.
TOAST_LOG_MAX = 100


@dataclass
class Toast:
    text: str
    level: str = "info"  # one of TOAST_LEVELS
    born: float = field(default_factory=time.monotonic)
    # A route the toast offers alongside its text. Only "log" today: an
    # unexpected exception's toast says "see the log for details" and the one
    # button that opens it lives inside the diagnostics popup, which is further
    # away than eight seconds. Named rather than a bool so the widget's label
    # is a function of the toast; an unrecognised name simply draws nothing.
    action: str | None = None
    # What the action acts *on* -- a job id for "show", a sweep id for
    # "review". A field rather than a name like ``show:abc123``: the label
    # comes from ``TOAST_ACTIONS[action]``, so an id spliced into the name
    # would make every toast's action unrecognised and silently unlabelled.
    action_arg: str | None = None

    @property
    def ttl(self) -> float:
        # A level the UI has not learned is an ordinary notice rather than an
        # exception: the same rule ``action`` follows, and the reason a new
        # level can be introduced from one side.
        return TOAST_LEVELS.get(self.level, TOAST_LEVELS["info"])

    def expired(self, now: float | None = None) -> bool:
        return (time.monotonic() if now is None else now) - self.born > self.ttl


# How often the status strip re-reads the machine. Every reading behind it is
# a syscall -- two Win32 calls and a ``cudaMemGetInfo`` -- and the frame loop
# must never block, so they are sampled at about 2 Hz and the frames in between
# redraw the same numbers. Two per second is also as fast as any of them is
# worth reading: a memory figure that changes 60 times a second is unreadable.
RESOURCE_PERIOD = 0.5


def _pair(used: float | None, total: float | None) -> str:
    """``21.4/32``, or ``--`` when either half is unavailable.

    Never ``0``: ``vram.device_memory`` returns None whenever torch has not
    been imported into this process yet, which is most of a session, and a zero
    there would read as "the card is empty" rather than "nobody has looked".
    """
    if used is None or total is None:
        return "--"
    return f"{used:.1f}/{total:.0f}"


@dataclass(frozen=True)
class Resources:
    """One sample of what the machine is doing, all four fields None-able.

    Taken straight from ``memlog`` and ``vram``, which are the two pure modules
    that answer with None rather than raising -- so a reading this cannot get
    is simply absent, and every renderer here has to say so.

    The host pair is deliberately *this process* over the *system* commit
    limit: those are the two halves of the 2026-08-03 crash ("are we the leak"
    and "how close is the wall"), and they are the two numbers memlog exists to
    record. The device pair is device-wide rather than ours, because
    ``cudaMemGetInfo`` reports the card -- which is the point, since
    trellis-server's ~16 GB lives in another process entirely.
    """

    private_gib: float | None = None
    commit_gib: float | None = None
    commit_limit_gib: float | None = None
    vram_used_gib: float | None = None
    vram_total_gib: float | None = None
    vram_free_gib: float | None = None

    @classmethod
    def sample(cls) -> Resources:
        """Read the machine once. Never raises; every source is None-safe."""
        from .. import memlog, vram

        proc = memlog.process_memory()
        sysmem = memlog.system_memory()
        # device_memory() and never probe(): probe imports torch, which costs
        # seconds, and this runs on the frame thread.
        device = vram.device_memory()
        used = None if device is None else device.total_gib - device.free_gib
        return cls(
            private_gib=None if proc is None else proc.private,
            commit_gib=None if sysmem is None else sysmem.commit_total,
            commit_limit_gib=None if sysmem is None else sysmem.commit_limit,
            vram_used_gib=used,
            vram_total_gib=None if device is None else device.total_gib,
            vram_free_gib=None if device is None else device.free_gib,
        )

    def line(self, fps: float, frames: int) -> str:
        """The compact strip: ``58 fps · 1.9/32 GB · VRAM 21.4/32``.

        Latin-1 throughout -- the separator is U+00B7, which imgui's default
        atlas carries; anything above it draws as the missing-glyph box.
        """
        rate = f"{fps:.0f}" if frames else "--"
        host = _pair(self.private_gib, self.commit_limit_gib)
        return f"{rate} fps · {host} GB · VRAM {_pair(self.vram_used_gib, self.vram_total_gib)}"

    def detail(self, fps: float, frame_ms: float, worst_ms: float, frames: int) -> str:
        """The tooltip: the same readings said in full, one per line."""
        lines = []
        if frames:
            lines.append(f"Frames: {fps:.1f} fps, {frame_ms:.1f} ms mean, {worst_ms:.1f} ms peak")
        else:
            lines.append("Frames: not measured yet")
        if self.private_gib is None:
            lines.append("Host memory: unavailable")
        else:
            lines.append(f"This process: {self.private_gib:.2f} GiB private commit")
        if self.commit_gib is None or self.commit_limit_gib is None:
            lines.append("System commit: unavailable")
        else:
            fraction = self.commit_gib / self.commit_limit_gib if self.commit_limit_gib else 0.0
            lines.append(
                f"System commit: {self.commit_gib:.1f}/{self.commit_limit_gib:.1f} GiB"
                f" ({fraction * 100:.0f}%)"
            )
        if self.vram_total_gib is None:
            # The honest sentence, and the common one: nothing has imported
            # torch yet, so there is no reading rather than a reading of zero.
            lines.append("Device VRAM: unavailable until a job loads torch")
        else:
            lines.append(
                f"Device VRAM: {self.vram_used_gib:.1f}/{self.vram_total_gib:.1f} GiB used,"
                f" {self.vram_free_gib:.1f} GiB free (whole card)"
            )
        return "\n".join(lines)


@dataclass
class ManualState:
    """The Manual mode: which chapter it shows and where in it.

    Whether it is on screen is ``AppState.mode == "manual"`` and nothing else --
    the Manual is a mode like any other, so a second "am I visible" flag here
    would be a way for the two to disagree.
    """

    chapter: str = "00-index"
    # Set alongside a navigation, consumed by the renderer on the frame the
    # heading is drawn -- scrolling needs a cursor position that only exists
    # mid-draw.
    pending_anchor: str | None = None
    search: str = ""

    def open_at(self, chapter: str, anchor: str | None = None) -> None:
        self.chapter = chapter
        self.pending_anchor = anchor


@dataclass
class AppState:
    """The whole UI's mutable state."""

    # The one thing that decides what a pane shows, one of ``modes.KEYS``:
    # home | manual | 2d | 3d | inker | clay | review | settings. It defaults
    # to the Home screen, which is what makes the chooser appear on every
    # launch rather than only the first ever; no mode is ever persisted
    # (``test_no_mode_is_persisted_anywhere`` pins that).
    mode: str = "home"
    # Where Esc goes from a mode you only pass through (Home, the Manual, app
    # Settings): back to the work you left, not to the chooser. ``mode_observed``
    # is the other half and exists because ``mode`` is assigned from a dozen
    # places (landing tiles, library cards, Inker's own open path) -- rather
    # than funnel all of them through a setter, ``App._note_mode`` samples the
    # pair once per key event, which is both the only place ``previous_mode``
    # is read and late enough to have seen a change made by a click, by a drop
    # or by F1 earlier in the very same frame. Neither is persisted: no mode is
    # (``test_no_mode_is_persisted_anywhere``), so a remembered one would be a
    # write with no reader across launches.
    previous_mode: str = "home"
    mode_observed: str = "home"
    selected: str | None = None
    # The job the library list should scroll into view on the next frame it is
    # drawn, set by arrow-key navigation and cleared by the card that answers
    # it. A one-shot rather than a standing position: a sticky value would drag
    # the list back every frame while the user is dragging the scrollbar.
    library_scroll_to: str | None = None
    # The asset currently being dragged, if any. imgui's Python drag-and-drop
    # payload is an integer, so the job id travels here instead; one drag is in
    # flight at a time by construction. See ``library.DRAG_JOB``.
    dragging_job: str | None = None
    # The command palette (I80). Three plain fields rather than a state object:
    # it holds a query, a cursor and whether it is up, and nothing about it
    # survives being closed -- reopening on the last query would make Ctrl+K
    # act on a search the user has forgotten typing.
    # Which slot last received a drop, and when (H70). A dropped file used to
    # be acknowledged only by a toast in the far corner of the window, while
    # the control that actually changed -- in 2D, inside a section that is
    # collapsed by default -- showed nothing at all. See ``widgets.drop_flash``.
    drop_flash_slot: str = ""
    drop_flash_at: float = 0.0
    palette_open: bool = False
    palette_query: str = ""
    palette_index: int = 0
    comparing: str | None = None
    # Which asset the inspector's Reject button is armed for, waiting on a
    # reason. Keyed by job id rather than being a bare flag, so selecting a
    # different asset disarms it -- an armed state belongs to the thing that was
    # on screen when it was armed, exactly as Review's ``pending_reject`` does.
    inspector_reject_armed: str | None = None
    form_2d: dict[str, Any] = field(default_factory=default_form_2d)
    form_3d: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_FORM_3D))
    filters: Filters = field(default_factory=Filters)
    history: list[str] = field(default_factory=list)
    checked: set[str] = field(default_factory=set)
    toasts: list[Toast] = field(default_factory=list)
    # Every toast raised this session, newest first, capped (H67). A toast is
    # the app's only channel for "that worked" and "that did not", and it is
    # gone in four seconds -- which for anything that happens while the user is
    # looking elsewhere means it never happened at all. Not persisted: it is a
    # record of *this* run, and one restored from disk would be a list of
    # things that are no longer true.
    toast_log: list[Toast] = field(default_factory=list)
    # The composed-prompt preview, refreshed off-thread as the prompt is typed.
    preview: dict[str, Any] = field(default_factory=dict)
    preview_dirty_at: float = 0.0
    # The frame-rate overlay (F10). Persisted, because someone watching for a
    # stall wants it to survive the restart they are about to do. It is the
    # *detailed* view -- mean and worst frame -- and the always-on strip beside
    # the mode switch is the summary; see ``overlay.fps_meter``.
    show_fps: bool = False
    # The status strip's last sample and when it was taken, throttled to
    # RESOURCE_PERIOD. A timestamp rather than a frame counter because the
    # thing being bounded is syscalls per second, not per frame.
    resources: Resources = field(default_factory=Resources)
    resources_at: float = 0.0
    wireframe: bool = False
    turntable: bool = False
    source_job: str | None = None  # the 2D asset the 3D pane starts from
    # Every outstanding failure the banner is showing, oldest first. A list
    # rather than the single ``last_error`` slot it replaces: three writers
    # (a failed doctor check, a dead worker, a worker that never started) all
    # assigned to that slot, so a launch that failed two checks reported
    # whichever wrote last and lost the other with nothing to say it had.
    errors: list[str] = field(default_factory=list)
    # What Dismiss took off screen, kept rather than dropped (F59). Every writer
    # of ``errors`` fires exactly once -- the startup doctor sweep, and the two
    # one-shot worker checks -- so clearing the list destroyed the only copy of
    # the text, and the sole remaining trace of a launch that failed two checks
    # was a coloured dot. The diagnostics popup shows these, which is what makes
    # Dismiss "put it away" rather than "forget it".
    dismissed_errors: list[str] = field(default_factory=list)
    # The Home screen's sub-view. Not persisted: Home always opens on the
    # chooser rather than on whichever list was last being browsed.
    landing_view: str = "choose"  # choose | open | profiles
    profile_draft: dict[str, Any] | None = None
    profile_draft_name: str = ""
    # The name the draft was opened under, so renaming one in the editor moves
    # it rather than leaving the old name behind as a duplicate.
    profile_draft_origin: str = ""
    # Inker mode's open documents and tool settings, built on first use.
    # Typed Any so state.py keeps no import of the editor or of Pillow, and
    # lazy so a session that never draws pays nothing for it.
    inker: Any = None
    # Clay's own multi-document state, built on first use by
    # ``clay_mode.ensure``. Untyped and None here for the reason ``inker`` is:
    # AppState is the shared frame state and deliberately knows nothing about
    # what a mode keeps, so a session that never opens Clay pays nothing.
    clay: Any = None
    # Review mode's sweep runs and where in one it is, built on first use by
    # ``review_mode.ensure`` and untyped here for the reason ``clay`` is.
    # Nothing in it is persisted: a stored run directory would outlive the
    # sweep it names.
    review: Any = None
    # Whether ``findings.json`` is behind the evidence in the DB. A flag rather
    # than a submit, because ``TaskRunner.submit`` *refuses* a key already in
    # flight and nothing re-arms it: five verdicts in a second used to run one
    # recompute over the set as it stood at the first press and silently drop
    # the rest until the next verdict. It is also what lets the worker's
    # observations reach the file at all -- they are appended off the frame
    # thread by a component that cannot submit anything, so the frame loop
    # notices the job finishing and marks this instead.
    findings_dirty: bool = False
    # Which probe question needs retraining, or None. The ``findings_dirty``
    # pattern exactly, and for the identical reason -- ``TaskRunner.submit``
    # refuses a key already in flight and nothing re-arms it, so a burst of
    # labels would train once on the set as it stood at the first press and drop
    # the rest. A stage string rather than a bool because a labelling pass is
    # about one question at a time, and the training run needs to know which.
    judge_dirty: str | None = None
    # Whether the open review's units need scoring by the judge. The same flag
    # pattern for the third time, and the third time for the same reason: a
    # score request follows every scan and every retrain, and a direct submit
    # would drop whichever one arrived while the previous run was still going.
    review_scores_dirty: bool = False
    # The promote flow's matte preview and its cutout cache, built on first use
    # by ``matte_preview.ensure``. Untyped and None here for the reason
    # ``clay``/``review`` are, and never persisted: a stored cutout would be a
    # claim about a file that has had a whole session to change.
    matte: Any = None
    manual: ManualState = field(default_factory=ManualState)
    # The selected asset's parsed manifest.json, held as ((job id, mtime), data)
    # so the Export tab reads and parses it once per version of the file rather
    # than once per frame. The same (id, mtime) idiom ThumbnailCache uses, for
    # the same reason: the stat that decides whether to re-read is the cheap
    # half, and a derivation rewrites the manifest under a tab that is open.
    # One slot, because one asset is inspected at a time.
    manifest: Any = None
    # The palette directory's listing, held as (mtime, [names]) for exactly the
    # reason above: the combo is drawn every frame and a directory walk per
    # frame is a syscall storm for a folder that changes when a user drops a
    # file in it. A palette added while the app runs appears on the next frame,
    # because dropping a file moves the directory's own mtime.
    palettes: Any = None
    # The selected asset's ``rig.json``, as ``{job id: (mtime stamp, data)}``.
    # A rig is recorded on the *rig* job's row but written into the **source**
    # job's directory, and the Rig & Pose tab opens on that source job -- so the
    # file is the only place the selection on screen can learn how its own mesh
    # was bound. Read on the frame thread and therefore cached, under the same
    # racily-clean rule ``files.attach_files`` documents: a re-rig lands inside
    # a directory whose mtime may not move, so a stamp is only remembered once
    # it is safely in the past. Keyed by job rather than one slot, because the
    # inspector is not the only reader and a one-slot cache thrashes the moment
    # two are alive. Never persisted: it describes a file, not a preference.
    rig_meta_cache: dict[str, Any] = field(default_factory=dict)

    # -- the status strip ---------------------------------------------------

    def pump_resources(self, now: float | None = None) -> Resources:
        """Re-sample at most every RESOURCE_PERIOD; return what to draw.

        Called from the frame loop, so the throttle is the whole point: the
        first sample is taken on the first frame (``resources_at`` starts at 0)
        and every frame in between redraws the numbers already held.
        """
        now = time.monotonic() if now is None else now
        if now - self.resources_at >= RESOURCE_PERIOD:
            self.resources = Resources.sample()
            self.resources_at = now
        return self.resources

    # -- toasts ------------------------------------------------------------

    def toast(
        self,
        text: str,
        level: str = "info",
        action: str | None = None,
        action_arg: str | None = None,
    ) -> None:
        entry = Toast(text=text, level=level, action=action, action_arg=action_arg)
        self.toasts.append(entry)
        # And into the history, which is the same object rather than a copy of
        # its text (H67): the history has to be able to say what *level* a
        # message was and when, and a list of strings cannot. Newest first, so
        # the popup that draws it needs no reversal and no scroll to the end.
        self.toast_log.insert(0, entry)
        del self.toast_log[TOAST_LOG_MAX:]

    def expire_toasts(self) -> None:
        now = time.monotonic()
        self.toasts = [t for t in self.toasts if not t.expired(now)]

    # -- the error banner --------------------------------------------------

    def note_error(self, text: str | None) -> None:
        """Add a failure to the banner, once.

        Deduplicated because the conditions are re-derived rather than edged:
        a doctor check that is still failing is reported again on the next
        refresh, and a banner that grew a line each time would bury the first.
        """
        text = (text or "").strip()
        if text and text not in self.errors:
            self.errors.append(text)

    def dismiss_errors(self) -> None:
        """Off the banner, into the popup -- never out of existence (F59)."""
        self.dismissed_errors.extend(m for m in self.errors if m not in self.dismissed_errors)
        self.errors.clear()

    @property
    def error_text(self) -> str:
        """Every outstanding failure as one block -- what Copy details copies."""
        return "\n".join(self.errors)

    # -- history -----------------------------------------------------------

    def remember_prompt(self, prompt: str) -> None:
        """Most recent first, deduplicated, bounded."""
        prompt = (prompt or "").strip()
        if not prompt:
            return
        self.history = [prompt] + [p for p in self.history if p != prompt]
        del self.history[MAX_HISTORY:]

    # -- selection ---------------------------------------------------------

    def select(self, job_id: str | None) -> None:
        if job_id != self.selected:
            self.selected = job_id
            # A comparison is between the selection and something else; keeping
            # it across a selection change would compare two jobs neither of
            # which the user just clicked.
            self.comparing = None

    def toggle_check(self, job_id: str) -> None:
        self.checked.symmetric_difference_update({job_id})


# --- the primary-action ladder ----------------------------------------------
#
# Ported from the browser's card renderer. One button per card, and which one
# is a function of what the job *is*, in a fixed order of precedence -- the
# point being that the obvious next step is always the one on offer.

ACTIONS = {
    "cancel": "Cancel",
    "retry": "Try again",
    "promote": "Make 3D",
    "rig": "Rig",
    "open": "Open",
}


def primary_action(job: dict[str, Any], *, rigging_available: bool = True) -> str | None:
    """The one action a job's card offers, or None."""
    status = job.get("status")
    if status in ("queued", "running"):
        return "cancel"
    if status == "error":
        return "retry"
    if status != "done":
        return None
    files = job.get("files") or []
    if job.get("stage") == "tile":
        # No mesh, no rig: a tile's next step is to be exported, which the
        # inspector's Export tab is. "Open" selects it and shows that tab.
        return "open" if "input.png" in files else None
    if job.get("stage") == "reference":
        # A finished reference's next step is the mesh it exists for.
        return "promote" if "input.png" in files else None
    if "model.glb" not in files:
        return None
    if rigging_available and job.get("kind") != "rig" and "rig.glb" not in files:
        return "rig"
    return "open"


# --- progress ---------------------------------------------------------------


class Eta:
    """A smoothed estimate of how much longer the running job has.

    Exponentially weighted (0.7 old, 0.3 new) because the raw estimate jumps
    every time a stage boundary changes the rate, and a number that flickers
    between "2 min" and "20 s" is worse than none. Suppressed until the job is
    warm and meaningfully underway, for the same reason: an estimate from 3%
    of a cold start is a guess about the weights loading.
    """

    ALPHA = 0.3

    def __init__(self) -> None:
        self.value: float | None = None
        self._job: str | None = None

    def update(
        self, job_id: str | None, percent: float, elapsed: float, cold: bool
    ) -> float | None:
        if job_id != self._job:
            self._job, self.value = job_id, None
        if job_id is None or cold or not (10.0 <= percent < 100.0) or elapsed <= 5.0:
            return None
        remaining = elapsed * (100.0 - percent) / percent
        self.value = (
            remaining
            if self.value is None
            else (1 - self.ALPHA) * self.value + self.ALPHA * remaining
        )
        return self.value


def format_duration(seconds: float | None) -> str:
    """A clock a person reads at a glance: 45s, 2m 10s, 1h 04m."""
    if seconds is None or seconds < 0:
        return ""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


def format_bytes(count: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(count) < 1024 or unit == "GB":
            return f"{count:.0f} {unit}" if unit == "B" else f"{count:.1f} {unit}"
        count /= 1024
    return f"{count:.1f} GB"
