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
    # profile "custom", and every decimating tier needs vendor/gltfpack, which
    # is not present -- so a control here would offer a number that "raw"
    # ignores. The plumbing through _payload is kept because it is correct the
    # moment a tier is qualified and exposed; the retarget control on a
    # finished mesh is where a budget is actually chosen today.
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
        if self.kind != "all" and _kind_of(job) != self.kind:
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


def _kind_of(job: dict[str, Any]) -> str:
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


@dataclass
class Toast:
    text: str
    level: str = "info"  # info | error
    born: float = field(default_factory=time.monotonic)
    # A route the toast offers alongside its text. Only "log" today: an
    # unexpected exception's toast says "see the log for details" and the one
    # button that opens it lives inside the diagnostics popup, which is further
    # away than eight seconds. Named rather than a bool so the widget's label
    # is a function of the toast; an unrecognised name simply draws nothing.
    action: str | None = None

    @property
    def ttl(self) -> float:
        # Errors linger: they usually say what to do next, and four seconds is
        # not long enough to read a sentence and act on it.
        return 8.0 if self.level == "error" else 4.0

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
    selected: str | None = None
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

    def toast(self, text: str, level: str = "info", action: str | None = None) -> None:
        self.toasts.append(Toast(text=text, level=level, action=action))

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
