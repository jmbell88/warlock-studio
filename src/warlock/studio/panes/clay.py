"""Clay: a deliberate placeholder.

The mode exists in the switch before the sculpting tools do, on purpose --
where a feature will live is worth stating, and a menu entry that appears
later moves everything beside it. There is nothing to document yet, so there
is no help button either: HELP_TARGETS is checked bidirectionally against the
call sites, and an entry with no pane behind it would fail that check.
"""

from __future__ import annotations

from typing import Any

from .. import icons, widgets


def draw(_ctx: Any) -> None:
    widgets.empty_state(icons.EGG, "Clay", "Sculpting is coming in a future update.")
