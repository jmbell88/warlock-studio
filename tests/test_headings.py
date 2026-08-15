"""REDESIGN.md wave 6: headings are sentence case, and a full window says what it is.

Wave 2 rebuilt :func:`widgets.section` to draw its label as sentence-case
Medium at body size, and ``tests/test_ux_phases.py`` pins that *implementation*
-- that it does not ``.upper()``, that it breathes instead of ruling. What no
test pinned was the other half of the same decision: the sixty-odd strings
handed to it. Half the tree had been written against the older small-caps
widget, which swallowed whatever case it was given, so "tools" and "Tools" drew
identically and the literals drifted to lowercase without anybody noticing.
They stopped drawing identically in wave 2 and nothing said so until wave 6
looked at a screenshot.

Hence a rule rather than a second sweep: the sweep is a commit, the rule is
what stops the sixty-first call site from being lowercase.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from warlock.studio.panes import library_full

SRC = Path(__file__).resolve().parents[1] / "src" / "warlock" / "studio"


def _section_literals() -> list[tuple[str, int, str]]:
    """Every ``section("...")`` in the studio package. -> (file, line, label).

    An AST walk rather than a regex: ``_params_section(`` and
    ``begin_table("keys/...")`` both match a naive pattern for a call whose
    name ends in "section", and a literal spanning two lines does not match one
    at all. Only constant arguments are visible here -- the dynamic call sites
    (a command group, a date group, a guidance title) are checked at their
    sources, which is where a heading's case is actually decided.
    """
    out: list[tuple[str, int, str]] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name != "section":
                continue
            first = node.args[0]
            if isinstance(first, ast.Constant) and isinstance(first.value, str):
                out.append((path.relative_to(SRC).as_posix(), first.lineno, first.value))
    return out


def test_the_sweep_found_them_all() -> None:
    """A guard on the guard: an AST walk that matches nothing passes vacuously."""
    found = _section_literals()
    assert len(found) > 50, f"only {len(found)} section literals -- did the walk break?"


def test_every_heading_is_sentence_case() -> None:
    """First letter up; the rest is the writer's, because "Image brush" and
    "SDXL" are both correct and no test can tell the second from shouting.

    ``field_label`` keeps its ``.upper()`` and earns it -- a column of them is
    chrome around the values in a dense form. A section heading is the thing
    those are grouped under, and drawing both registers the same way flattens
    exactly the hierarchy the headings exist to create.
    """
    offenders = [
        f"{path}:{line} {label!r}"
        for path, line, label in _section_literals()
        if label and label[0].islower()
    ]
    assert not offenders, "section headings are sentence case: " + ", ".join(offenders)


def test_no_call_site_lowercases_a_heading_on_the_way_in() -> None:
    """The command palette used to, which is how "Go to" reached the screen as
    "go to" while every literal beside it was already swept."""
    offenders = [
        f"{path.relative_to(SRC)}:{i}"
        for path in SRC.rglob("*.py")
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if "section(" in line and ".lower()" in line
    ]
    assert not offenders, "let the heading keep its case: " + ", ".join(offenders)


def test_the_library_full_view_says_what_it_is() -> None:
    """The one full-window pane that was named only by the navigation chrome
    outside it (REDESIGN.md wave 6)."""
    assert 'pane_title("Library")' in inspect.getsource(library_full)
