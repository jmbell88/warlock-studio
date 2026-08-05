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
from dataclasses import dataclass, field
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

    Not simply job["kind"]: a text job that stops at a reference and one that
    goes on to a mesh are the same kind and different things to look for.
    """
    if job.get("kind") in ("rig", "sheet"):
        return job["kind"]
    return "reference" if job.get("stage") == "reference" else "model"


@dataclass
class Toast:
    text: str
    level: str = "info"  # info | error
    born: float = field(default_factory=time.monotonic)

    @property
    def ttl(self) -> float:
        # Errors linger: they usually say what to do next, and four seconds is
        # not long enough to read a sentence and act on it.
        return 8.0 if self.level == "error" else 4.0

    def expired(self, now: float | None = None) -> bool:
        return (time.monotonic() if now is None else now) - self.born > self.ttl


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
    # home | manual | 2d | 3d | inker | clay | settings. It defaults to the
    # Home screen, which is what makes the chooser appear on every launch
    # rather than only the first ever; only the work modes are persisted.
    mode: str = "home"
    selected: str | None = None
    comparing: str | None = None
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
    # stall wants it to survive the restart they are about to do.
    show_fps: bool = False
    wireframe: bool = False
    turntable: bool = False
    show_advanced: bool = False
    source_job: str | None = None  # the 2D asset the 3D pane starts from
    last_error: str | None = None
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
    manual: ManualState = field(default_factory=ManualState)

    # -- toasts ------------------------------------------------------------

    def toast(self, text: str, level: str = "info") -> None:
        self.toasts.append(Toast(text=text, level=level))

    def expire_toasts(self) -> None:
        now = time.monotonic()
        self.toasts = [t for t in self.toasts if not t.expired(now)]

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
