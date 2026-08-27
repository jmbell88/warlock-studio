"""Chapter 16 and the Ctrl+/ sheet have to agree.

``tests/manual/`` gates the manual's *structure* -- the chapter list, links,
anchors, help-button parity, line length -- and nothing that reads a chapter's
claims against the app. That gap is how chapter 16 came to open by telling the
reader to press a **?** button in a top bar that chapter 1 says does not
exist, and how the in-app sheet came to list "F1 -- Switch to the Manual"
three waves after F1 stopped switching to anything.

The relationship is deliberately one-way. The chapter says so itself: the
popup "is a condensed subset -- the tables below are the full list". So the
gate is **every binding the popup lists appears in the chapter**, plus every
group the popup has. A binding the chapter carries and the popup does not is
the design, not a defect.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from warlock.studio.main import shortcut_sections

CHAPTER = Path(__file__).resolve().parents[2] / "docs" / "manual" / "38-shortcuts.md"

# The popup's group titles against the chapter headings that cover them. Two
# popup groups share one chapter section, which is why this is a table rather
# than a string match: the judging pass is a mode of Review, not a mode.
SECTIONS = {
    "Everywhere": "Everywhere",
    "Create": "Create",
    "Review": "Review",
    "Review - a judging pass": "Review — a judging pass",
    "Clay": "Clay",
    "Inker": "Inker",
    "Poser": "Poser",
    "Plotter": "Plotter",
    "Packwright": "Packwright",
    "Troupe": "Troupe",
    "Sirens": "Sirens",
}

_MODIFIED = re.compile(r"^(?:Ctrl\+|Shift\+|Alt\+)+")
# A single character, a function key, or one of the named keys. Everything
# else in a key cell is prose -- "middle drag", "a card" -- and a gate that
# treated prose as a binding would fail on wording rather than on drift,
# which is the opposite of what it is for.
_NAMED = frozenset(
    {
        "Space",
        "Enter",
        "Esc",
        "Tab",
        "Delete",
        "Backspace",
        "Up",
        "Down",
        "Left",
        "Right",
        "Arrows",
        "Home",
        "End",
        "Ctrl",
        "Shift",
        "Alt",
    }
)
_PLAIN = re.compile(r"F\d{1,2}|\+?[A-Za-z0-9\[\].,-]")


def _is_binding(part: str) -> bool:
    if _MODIFIED.match(part):
        return True
    return part in _NAMED or bool(_PLAIN.fullmatch(part))


def atoms(cell: str) -> set[str]:
    """The individual bindings in one key cell.

    Two spellings have to be understood or every row is a false positive.
    ``Ctrl+N / O / W`` is three bindings and not one plus two letters -- a
    bare token after a modified one **inherits its modifiers**, which is how
    both documents write a family of chords. Parenthesised asides
    (``Shift: hardness``) are prose about a binding rather than another one,
    and a multi-word part contributes only its leading key: the chapter
    writes "Space drag" where the popup writes "Space".
    """
    text = cell.replace("`", "").replace("—", "-").replace("–", "-")
    text = re.sub(r"\(.*?\)", " ", text)
    out: set[str] = set()
    prefix = ""
    for part in re.split(r"\s*(?:/|,|\bthen\b|\bor\b)\s*", text):
        part = part.strip()
        if not part:
            continue
        # Always the leading token: "Shift+G Gradient" is one binding and a
        # name for it, and "middle drag" is neither.
        part = part.split()[0]
        if not _is_binding(part):
            continue
        found = _MODIFIED.match(part)
        if found:
            prefix = found.group(0)
        elif prefix:
            # ``Ctrl+Shift+1 / +3 / +7`` -- the chapter's spelling of the same
            # inheritance, with the plus kept.
            part = prefix + part.lstrip("+")
        out.add(part)
    return out


def _chapter_atoms() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    heading = ""
    for line in CHAPTER.read_text(encoding="utf-8").splitlines():
        title = re.match(r"^##\s+(.*)", line)
        if title:
            heading = title.group(1).strip()
        if not heading or not line.startswith("|") or line.count("|") < 3:
            continue
        cell = line.split("|")[1].strip()
        if cell in ("---", "Keys"):
            continue
        found.setdefault(heading, set()).update(atoms(cell))
    return found


def _popup_atoms() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for title, rows in shortcut_sections():
        for keys, _what in rows:
            found.setdefault(title, set()).update(atoms(keys))
    return found


def test_the_expansion_rule_reads_both_documents_spellings():
    """The gate is only as good as this, so it is pinned separately: a bug
    here reads as agreement rather than as a failure."""
    assert atoms("Ctrl+N / O / W") == {"Ctrl+N", "Ctrl+O", "Ctrl+W"}
    assert atoms("Ctrl+Shift+1 / +3 / +7") == {
        "Ctrl+Shift+1",
        "Ctrl+Shift+3",
        "Ctrl+Shift+7",
    }
    assert atoms("[ / ]") == {"[", "]"}
    # Prose is not a binding, and a parenthesised aside is prose.
    assert atoms("Right-click a card") == set()
    assert atoms("[ / ] (Shift: hardness)") == {"[", "]"}
    # The chapter's wording and the popup's reduce to the same key.
    assert atoms("Space drag, middle drag") == {"Space"}
    assert atoms("Space / middle drag") == {"Space"}


@pytest.mark.parametrize("title", sorted(SECTIONS))
def test_every_group_in_the_sheet_is_a_section_of_the_chapter(title):
    assert SECTIONS[title] in _chapter_atoms()


def test_the_sheet_lists_exactly_the_groups_the_gate_knows_about():
    """A new group must join ``SECTIONS`` rather than slip past the gate --
    which is the failure mode a per-section test cannot catch on its own."""
    assert sorted(t for t, _ in shortcut_sections()) == sorted(SECTIONS)


def test_every_mode_the_chapter_gives_a_section_has_a_group_in_the_sheet():
    """Derived from the chapter, not from ``SECTIONS``.

    ``SECTIONS`` is a second hand-written copy, so it can only catch drift
    *within* a group it already knows about -- which is how the sheet came to
    have no Poser group at all while the chapter had one and the bindings
    worked. Reading the mode list off ``studio.modes`` and the headings off the
    chapter closes that: a mode the chapter documents and the sheet is silent
    about now fails here.
    """
    from warlock.studio import modes

    headings = set(_chapter_atoms())
    groups = {title for title, _ in shortcut_sections()}
    missing = sorted(
        label
        for _key, label, _icon in modes.MODES
        if label in headings and label not in groups
    )
    assert not missing, (
        f"docs/manual/38-shortcuts.md documents {missing}, and the Ctrl+/ "
        f"sheet has no group for them -- a user in that mode learns nothing"
    )


@pytest.mark.parametrize("title", sorted(SECTIONS))
def test_no_binding_in_the_sheet_is_missing_from_the_chapter(title):
    """One direction only -- see the module docstring."""
    chapter = _chapter_atoms()[SECTIONS[title]]
    missing = sorted(_popup_atoms().get(title, set()) - chapter)
    assert not missing, (
        f"the Ctrl+/ sheet's {title} group lists {missing}, which "
        f"docs/manual/38-shortcuts.md does not"
    )


def test_the_chapter_does_not_send_the_reader_to_a_control_that_was_deleted():
    """Finding 4, as a ratchet. The header and its ``?`` button went in the
    UI redesign; the chapter went on naming both for three waves."""
    text = CHAPTER.read_text(encoding="utf-8")
    assert "top bar" not in text.lower()
    assert "Ctrl+/" in text


def test_the_installation_chapter_puts_the_health_badge_where_it_is():
    """The same class: the badge moved to the rail's footer, and it renders
    nothing at all when every check passed -- so "green when everything
    passed" described a state the app cannot be in."""
    text = (CHAPTER.parent / "39-installation.md").read_text(encoding="utf-8")
    assert "top bar" not in text.lower()
    assert "green when everything passed" not in text
