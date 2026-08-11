"""Every mode switch goes through ``state.set_mode`` (H14).

``set_mode`` is the one implementation: it early-returns on the mode already
in force and maintains ``previous_mode``/``mode_observed``, which is what Esc's
"back to the mode it came from" is built on. A direct write leans on
``App._note_mode``'s per-key-event sampling instead -- the exact drift the
palette's own copy of these lines already produced once. This is
``test_no_mode_is_persisted_anywhere``'s idiom pointed at the other half of the
contract: a source scan, so the next direct write fails here rather than
surfacing as Esc going two steps back.

The regex is scoped to assignments to ``.mode`` on an object *named* ``state``
(``state.mode =``, ``ctx.state.mode =``): an editor's own ``mode`` attribute --
Clay's element modes, the viewer's ``pose_mode``, ``inker_state`` -- is a
different fact about a different object and stays out of the scan.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "src" / "warlock" / "studio"

# Direct writes still tolerated, per file: named debt for their own work
# packages, each entry to be struck off as its file is routed through
# ``set_mode``. Not a rule -- a new file may not join this list.
#
# **Empty, and the debt is paid.** The last five were routed on 2026-08-11:
# ``inker_mode``, ``plotter_mode``, ``packwright_mode``, ``manual/render`` and
# ``panes/library``. The manual's two were the ones that mattered to a reader
# -- Esc out of a chapter opened from a pane's (?) now goes back to that pane
# rather than to Home -- and library's "open" is the twin of the landing page's,
# which had already been routed, so the same act reached from two surfaces
# behaved two ways. Kept as a name rather than deleted with its entries,
# because the *rule* below is what the emptiness now means.
ALLOWED: set[str] = set()

# ``\bstate`` keeps ``inker_state.mode`` and friends out (no word boundary
# after an underscore); ``(?!=)`` keeps comparisons out.
_WRITE = re.compile(r"\bstate\.mode\s*=(?!=)")


def test_every_mode_switch_goes_through_set_mode():
    offenders = [
        f"{path.relative_to(ROOT).as_posix()}:{n}"
        for path in sorted(ROOT.rglob("*.py"))
        if path.name != "state.py"
        and path.relative_to(ROOT).as_posix() not in ALLOWED
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if _WRITE.search(line)
    ]
    assert offenders == []


def test_the_allowlist_names_only_files_that_still_need_it():
    """A struck-off file must leave the list, or the list quietly becomes a
    licence rather than a debt record."""
    for rel in ALLOWED:
        path = ROOT / rel
        assert path.is_file(), rel
        assert any(
            _WRITE.search(line)
            for line in path.read_text(encoding="utf-8").splitlines()
        ), f"{rel} no longer writes .mode directly; remove it from ALLOWED"
