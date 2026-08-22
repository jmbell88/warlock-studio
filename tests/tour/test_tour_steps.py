"""Every step of every tour names something that exists.

A tour fails silently by construction: a step that points at a control nobody
marks simply draws no ring, and a step waiting on a condition nobody evaluates
waits forever. Both look, on screen, exactly like the app being broken -- so
both are gates here rather than discoveries.

The anchor check is the ``test_help_button_call_sites_match_help_targets``
pattern one level down, and in both directions for the same reason: a step
naming a key no pane marks is a step that points at nothing, and a pane marking
a key no step names is a mark nobody removed when the step that needed it went.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from warlock.studio import modes
from warlock.studio.manual import loader, parser
from warlock.studio.tour import TOURS
from warlock.studio.tour.steps import CONDITIONS

SRC = Path(__file__).resolve().parents[2] / "src" / "warlock" / "studio"

#: ``anchors.mark("x")`` and ``anchors.mark_window("x")`` call sites.
_MARK = re.compile(r"""anchors\.mark(?:_window)?\(\s*["']([^"']+)["']""")

#: The rail marks every mode from one call site, so its keys are derived rather
#: than written out -- exactly as the rail itself derives them.
_RAIL_KEYS = frozenset(f"rail/{key}" for key in modes.KEYS)


def _steps():
    for tour in TOURS:
        for step in tour.steps:
            yield tour, step


def _marked_in_source() -> set[str]:
    found: set[str] = set()
    for path in sorted(SRC.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "anchors.mark" not in text:
            continue
        found.update(_MARK.findall(text))
    return found


def test_the_sweep_finds_some_tours():
    assert TOURS, "no tours registered"
    assert list(_steps()), "no steps in any tour"


def test_every_tour_key_is_unique():
    keys = [tour.key for tour in TOURS]
    assert len(keys) == len(set(keys))


def test_every_step_id_is_unique_within_its_tour():
    for tour in TOURS:
        ids = [step.id for step in tour.steps]
        assert len(ids) == len(set(ids)), f"{tour.key}: duplicate step ids"


def test_every_step_names_a_real_mode():
    for tour, step in _steps():
        if step.mode is None:
            continue
        assert step.mode in modes.KEYS, f"{tour.key}/{step.id}: unknown mode {step.mode}"


def test_every_step_waits_for_a_condition_the_vocabulary_carries():
    for tour, step in _steps():
        assert step.done.name in CONDITIONS, (
            f"{tour.key}/{step.id}: unknown condition {step.done.name!r}. An "
            "unknown name reads as 'never satisfied', which on a point-and-wait "
            "step is a tour that hangs."
        )


def test_every_anchor_a_step_names_is_marked_somewhere():
    marked = _marked_in_source() | _RAIL_KEYS
    for tour, step in _steps():
        if step.anchor is None:
            continue
        assert step.anchor in marked, (
            f"{tour.key}/{step.id}: nothing calls anchors.mark({step.anchor!r}), "
            "so this step would ring nothing at all"
        )


def test_every_marked_anchor_is_named_by_a_step():
    """The other direction: a mark nobody points at is dead weight.

    The rail's keys are exempt because it marks every mode from one call site --
    eleven keys from one line, of which a tour naturally uses a few. Everything
    else is marked deliberately, one call per control, and should have a reason
    to exist.
    """
    used = {step.anchor for _tour, step in _steps() if step.anchor}
    stray = sorted(_marked_in_source() - used - _RAIL_KEYS)
    assert not stray, (
        f"marked but never pointed at: {stray}. Either a step wants it, or the "
        "mark outlived the step that did."
    )


def test_every_chapter_a_step_links_to_resolves():
    """The ``test_help_targets_resolve`` pattern, for the tour's own links."""

    anchors = {
        chapter.key: {
            block.anchor
            for block in parser.parse(loader.load(chapter.key))
            if isinstance(block, parser.Heading)
        }
        for chapter in loader.chapters()
    }
    for tour, step in _steps():
        if step.chapter is None:
            continue
        key, anchor = step.chapter
        assert key in anchors, f"{tour.key}/{step.id}: unknown chapter {key}"
        if anchor is not None:
            assert anchor in anchors[key], (
                f"{tour.key}/{step.id}: no anchor {key}#{anchor}"
            )


def test_a_tour_ends_by_handing_the_reader_somewhere():
    """Every tour's last step links into the manual.

    The tours are short by design and the chapters are where the depth is, so a
    tour that simply stops is one that has taught someone the shape of a thing
    and then left them holding it.
    """
    for tour in TOURS:
        last = tour.steps[-1]
        assert last.chapter is not None, (
            f"{tour.key}: the last step names no chapter to go on to"
        )


def test_step_returns_none_past_the_end_rather_than_raising():
    """The index comes back from settings, and a tour can lose a step."""

    tour = TOURS[0]
    assert tour.step(len(tour)) is None
    assert tour.step(-1) is None
    assert tour.step(0) is tour.steps[0]


def test_the_conditions_vocabulary_has_no_duplicates():
    assert len(CONDITIONS) == len(set(CONDITIONS))


def test_no_step_body_is_empty():
    """A step with nothing to say is a ring with no reason for being there."""

    for tour, step in _steps():
        assert step.title.strip(), f"{tour.key}/{step.id}: no title"
        assert step.body.strip(), f"{tour.key}/{step.id}: no body"


def test_the_marker_regex_actually_matches_the_call_sites():
    """Guard on the guard.

    Both directions above are built from one regex over the source. If it
    stopped matching -- a rename, a keyword argument, a wrapper -- every
    assertion would pass by finding nothing, in both directions at once.
    """
    found = _marked_in_source()
    assert found, "anchors.mark call sites are no longer being found"
    sample = ast.parse(
        (SRC / "panes" / "inker_tools.py").read_text(encoding="utf-8")
    )
    assert any(isinstance(node, ast.Call) for node in ast.walk(sample))
    assert "inker/tools" in found
