"""Which mesh candidates are still undecided, and what the picker says about them.

The ``clay_mode``/``review_mode``/``matte_preview`` shape: no imgui here at
all, so every rule about *which* group is offered and *when* it is finished is
assertable headlessly. ``panes/candidates_panel`` draws it.

The one design decision worth stating is why nothing is remembered. A group
could have been recorded on ``AppState`` when it was submitted -- but the
library hides an undecided candidate (``Filters.matches``, the same rule sweep
units follow), so a remembered group that was forgotten by a restart, or by the
user clicking something else, would leave a handful of rows nobody could reach
and nothing could offer. Deriving the group from the rows themselves makes that
impossible: while an undecided candidate exists it is offered, and once the
group is decided there is nothing left to derive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Terminal statuses, i.e. the ones a candidate can be judged from. ``cancelled``
# counts: the user stopped that one, and waiting for it is waiting forever.
_SETTLED = ("done", "error", "cancelled")


@dataclass(frozen=True)
class Group:
    """One undecided candidate group, in the order the seeds were drawn."""

    group: str
    members: list[dict[str, Any]]

    @property
    def done_count(self) -> int:
        return sum(1 for m in self.members if m.get("status") in _SETTLED)

    @property
    def finished(self) -> bool:
        """Whether every member has reached a terminal status.

        The picker is drawn either way -- watching two candidates arrive is the
        point of asking for two -- but Keep is only offered once the group has
        settled, because keeping one dissolves the group and a member still
        queued would become an ordinary asset nobody chose.
        """
        return bool(self.members) and self.done_count == len(self.members)

    def losers(self, keep_id: str) -> list[str]:
        return [m["id"] for m in self.members if m["id"] != keep_id]


def pending(jobs: list[dict[str, Any]]) -> Group | None:
    """The newest undecided group among ``jobs``, or None.

    Newest by the group's own newest row, so a group submitted while an older
    one is still undecided is the one on screen -- the same "most recent thing
    you asked for" rule the selection follows. Only one group is ever offered:
    two pickers stacked in a sidebar is a choice about which choice to make.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for job in jobs:
        group = job.get("candidate_group")
        if group:
            groups.setdefault(str(group), []).append(job)
    if not groups:
        return None
    newest = max(
        groups.items(),
        key=lambda item: max((j.get("created_at") or 0.0, j["id"]) for j in item[1]),
    )
    members = sorted(newest[1], key=lambda j: (int(j.get("candidate_index") or 0), j["id"]))
    return Group(newest[0], members)


def label(member: dict[str, Any]) -> str:
    """What one candidate's button says. Latin-1 only, like every other UI
    string here -- imgui's default atlas has nothing above it."""
    index = int(member.get("candidate_index") or 0)
    return f"#{index + 1}"


def status_line(member: dict[str, Any]) -> str:
    status = str(member.get("status") or "")
    seed = (member.get("params") or {}).get("mesh_seed")
    if seed is None:
        return status
    return f"{status} · seed {seed}"
