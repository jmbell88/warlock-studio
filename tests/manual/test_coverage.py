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
    # The three pieces T7 split off the canvas on 2026-09-04. Not panes: they
    # are the canvas's own drag, slice and multi-click-gesture halves, drawn
    # inside it, and a (?) in any of them would be a second help button on the
    # one surface.
    "inker_drag",
    "inker_gestures",
    "inker_slices",
    "inker_textures",  # a texture cache, drawn by nobody
    "plotter_canvas",  # the map itself; its tools are plotter-tools
    # The pattern grid itself; its controls are sirens-transport and
    # sirens-orders. Same rule as the two canvases above: it is the surface the
    # mode is about rather than a panel with a heading to hang a (?) beside.
    "sirens_patterns",
    "plotter_textures",  # a texture cache, drawn by nobody
    "packwright_preview",  # the atlas itself; its controls are packwright-settings
    "packwright_textures",  # a texture cache, drawn by nobody
    "clay_menu",  # a menu bar
    # The axis ball in the viewport's corner and the hint line under it. Chrome
    # over and under the render rather than a panel: there is no heading to hang
    # a (?) beside, and a help button inside a six-ball orientation widget would
    # be a seventh thing to click by accident. Both are documented under the
    # Clay chapter's viewport section, which the header's own (?) opens.
    "clay_hud",
    "plotter_menu",  # a menu bar
    "inker_menu",  # a menu bar
    # The row of tool options above the canvas. Not a pane with a heading: it
    # is one line of controls belonging to the tool the toolbox's own (?)
    # documents, and a help button in the middle of a row of sliders would be
    # a fifteenth control. The options themselves are the Inker chapter's.
    "inker_context",
    # ``inker_bridge`` was here until 2026-08-29 on the grounds that "a popup
    # has no heading to hang a (?) beside". Two of its dialogs are modals with
    # title bars now -- Image size and Canvas size -- and each carries an
    # inline help button beside its "Current: W x H" line, so the exemption is
    # no longer true and the file is gated like every other pane.
    # Not a pane at all: the mtime-cache rule several panes share, kept beside
    # them because inspector already imports sheet_panel and so neither of the
    # two could host it. Nothing on screen, so nothing to document.
    "stamps",
    # A search box over whatever is on screen: it has no titled section to hang
    # a (?) beside, and a help button inside a list the arrow keys walk is a
    # row the arrow keys would have to skip. Documented in
    # the Keyboard shortcuts chapter ("The command palette"), which is where a
    # reader looks for a keyboard binding. Named by *title*: this comment has
    # now outlived three renumberings, saying "chapter 12", then
    # "14-shortcuts.md", and a filename carries the number that moves just as
    # surely as the number did.
    "palette",
    # Not a pane either: two lines and a button that the sprite and pixel-sheet
    # sections draw *inside* their own (?)-bearing headers, so a second help
    # button would sit under the first and point at the same chapter. What it
    # does is documented where the button lands -- the App settings chapter's
    # Models section.
    "model_gate",
    # A startup modal rather than a workspace pane. Its Hardware and Required
    # downloads blocks are the installation chapter made actionable, and the
    # only exits dismiss the overlay or open the documented Models pane.
    "first_run",
    # Not a pane either: the guided tour is an overlay drawn from
    # ``App._overlays``, and every one of its steps already carries its own way
    # into the manual -- a "Read more" that opens the chapter the step is
    # about, which is a (?) with the target chosen per step instead of per
    # pane. A pane-level help button would have to pick one chapter for all
    # nineteen of them.
    "tour",
    # Not a pane: one job's picture, drawn *inside* whatever card or cell asked
    # for it (the UI redesign, wave 4.3). It has no section, no heading and no
    # window of its own -- a (?) beside a thumbnail would be a help button on a
    # square of pixels. Home's grid and the library's cards each carry their
    # own, pointing at the chapters that describe them.
    "thumbs",
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
