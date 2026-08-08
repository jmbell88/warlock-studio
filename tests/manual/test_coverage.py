"""O118's coverage audit, as a standing test rather than a one-off sweep.

``tests/manual/test_docs.py`` already asserts that HELP_TARGETS and the pane
call sites agree exactly. What it cannot see is a pane that has *neither* -- no
entry and no button -- which is what the audit found: two panes a user reads
and no way from either into the chapter describing it.

So this file asserts the other direction: every pane that draws a titled section
a user interacts with has a (?), and the ones that do not are named here with
the reason. A new pane fails this until it is either wired or listed.
"""

from __future__ import annotations

from pathlib import Path

PANES = Path(__file__).resolve().parents[2] / "src/warlock/studio/panes"

# Panes with no (?), and why. Each is chrome rather than a documented section:
# there is no heading to hang the button beside, and the manual describes what
# they draw under the pane that owns them.
NO_HELP_BUTTON = {
    "__init__",  # not a pane
    "inker_canvas",  # the canvas itself; its tools are inker-tools
    "inker_textures",  # a texture cache, drawn by nobody
    "clay_menu",  # a menu bar
    "overlay",  # the viewport toolbar and placeholders
    # A search box over whatever is on screen: it has no titled section to hang
    # a (?) beside, and a help button inside a list the arrow keys walk is a
    # row the arrow keys would have to skip. Documented in chapter 12, which is
    # where a reader looks for a keyboard binding.
    "palette",
}


def test_every_pane_either_has_a_help_button_or_is_listed():
    missing = {
        path.stem
        for path in PANES.glob("*.py")
        if "help_button" not in path.read_text(encoding="utf-8")
    }
    assert missing == NO_HELP_BUTTON, (
        "a pane with no (?) and no entry here is unreachable from the manual; "
        "wire it into HELP_TARGETS or name it above with the reason"
    )


def test_the_exemptions_all_exist():
    """A stale exemption is how this test quietly stops covering something."""
    stems = {path.stem for path in PANES.glob("*.py")}
    assert stems >= NO_HELP_BUTTON
