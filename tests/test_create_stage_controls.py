"""Create's mesh-side controls: what a greyed control says, and to whom.

Pure source inspection, for ``docs/INVARIANTS.md``'s reason for every pane in
this file's neighbourhood: these are imgui panes and cannot be driven
headlessly, so the *decision* -- what argument a call is made with, what an
``if`` is conditioned on -- is what gets tested, the way
``tests/test_field_error_wiring.py`` already does for the field-error rings.
"""

from __future__ import annotations

import inspect
import re

from warlock.studio.panes import settings_3d


def test_make_3d_button_states_a_reason_when_disabled_and_shows_ctrl_enter_only_when_live():
    """The 2026-09-05 audit, finding create-08.

    ``create_brief._generate`` passes the top refusal into ``primary_button``
    as ``reason=`` -- shown only while the button is disabled -- and reserves
    the "Ctrl+Enter" tooltip for while it is live. ``settings_3d._submit`` did
    the opposite: no ``reason=`` on the button at all (the refusals are only
    ever printed in red *above* it), and the manual
    ``imgui.set_tooltip("Ctrl+Enter")`` fired on hover whether or not Make 3D
    was enabled -- advertising a shortcut that does nothing while the button
    is dead.
    """
    source = inspect.getsource(settings_3d._submit)

    call = re.search(r'widgets\.primary_button\(\s*"Make 3D"(.*?)\)\n', source, re.S)
    assert call is not None, "settings_3d._submit no longer draws a \"Make 3D\" primary_button"
    assert "reason=" in call.group(1), (
        "Make 3D never states why it is disabled on the button itself, unlike "
        "create_brief._generate's identical button"
    )

    # The manual Ctrl+Enter tooltip must be conditioned on `enabled`, not fired
    # on hover alone.
    tooltip_call = re.search(
        r"if\s+(.*?):\s*\n\s*imgui\.set_tooltip\(\"Ctrl\+Enter\"\)", source, re.S
    )
    assert tooltip_call is not None, "the Ctrl+Enter tooltip line moved or was removed"
    assert "enabled" in tooltip_call.group(1), (
        "the Ctrl+Enter tooltip is not gated on `enabled`, so a greyed Make 3D "
        "still advertises a shortcut that does nothing"
    )
