"""The polish half of the 2026-09-05 consistency review.

Every workspace's empty screen offers the way back to recent work; every
pane's vertical spacing comes from the tokens rather than a literal that
forgot DPI scaling; and the inspector's exits sit under the same heading
the bridge panes use.
"""

from __future__ import annotations

import inspect
import re

from warlock.studio import clay_viewport
from warlock.studio.panes import (
    inker_canvas,
    inspector,
    packwright_preview,
    plotter_canvas,
    sirens_patterns,
)

EMPTY_SCREENS = (clay_viewport, inker_canvas, packwright_preview, plotter_canvas, sirens_patterns)


def test_every_empty_screen_offers_its_recents():
    for module in EMPTY_SCREENS:
        source = inspect.getsource(module)
        call = source[source.index("widgets.nothing_open(") :][:1200]
        assert "recent_paths=" in call and "on_open=" in call, module.__name__


def test_no_pane_spaces_itself_with_an_unscaled_literal():
    import pathlib

    panes = pathlib.Path(inspector.__file__).parent
    literal = re.compile(r"imgui\.dummy\(\(0, (4|6|8)\)\)")
    offenders = [p.name for p in panes.glob("*.py") if literal.search(p.read_text("utf-8"))]
    assert not offenders, offenders


def test_the_inspector_exits_share_the_bridges_heading():
    source = inspect.getsource(inspector._edit_actions)
    assert 'widgets.section("Take it somewhere")' in source
    # Heading only when there is something under it: an empty titled group is
    # a heading over nothing.
    assert "if not any(exits):\n        return" in source
