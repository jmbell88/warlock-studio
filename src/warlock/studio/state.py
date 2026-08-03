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

# What a submit is composed from. Split into the two panes deliberately: a
# control belongs to exactly one of them, and `platform` is *two* fields
# because one control cannot be owned by two panes -- g_platform is a prompt
# fragment, m_platform is the geometry resolution.
DEFAULT_FORM_2D: dict[str, Any] = {
    "prompt": "",
    "negative_prompt": "",
    "base_model": "",
    "style_lora": "",
    "lora_weight": 0.8,
    "seed": 42,
    "seed_locked": False,
    "count": 1,
    "category": "",
    "setting": "",
    "genre": "",
    "mood": "",
    "art_style": "",
    "palette": "",
    "material": "",
    "wear": "",
    "detail": "",
    "silhouette": "",
    "lighting": "",
    "composition": "",
    "platform": "",
}

DEFAULT_FORM_3D: dict[str, Any] = {
    "platform": "",
    "profile": "raw",
    "custom_triangles": 0,
    "size_m": 0.0,
    "bg_removal": "",
    "mesh_seed": 0,
    "mesh_seed_locked": False,
    "rig": False,
    "rig_template": "",
}


@dataclass
class Filters:
    """The library's filter bar. Persisted, because a workshop is filtered the
    same way every session."""

    text: str = ""
    status: str = "all"  # all | done | running | error
    kind: str = "all"  # all | reference | model | rig | sheet
    favorites_only: bool = False

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
class AppState:
    """The whole UI's mutable state."""

    mode: str = "2d"  # drives which settings pane is shown, as body.mode-2d did
    selected: str | None = None
    comparing: str | None = None
    form_2d: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_FORM_2D))
    form_3d: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_FORM_3D))
    filters: Filters = field(default_factory=Filters)
    history: list[str] = field(default_factory=list)
    checked: set[str] = field(default_factory=set)
    toasts: list[Toast] = field(default_factory=list)
    # The composed-prompt preview, refreshed off-thread as the prompt is typed.
    preview: dict[str, Any] = field(default_factory=dict)
    preview_dirty_at: float = 0.0
    wireframe: bool = False
    turntable: bool = False
    show_advanced: bool = False
    source_job: str | None = None  # the 2D asset the 3D pane starts from
    last_error: str | None = None

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
