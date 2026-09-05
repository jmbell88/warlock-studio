"""Every colour in Inker can be typed, and they all say so the same way.

``no_inputs`` drew a colour as a bare square: the value could only be changed
through a picker popup nothing advertised, and could not be typed at all. It
was set at five call sites independently, which is why removing it from one of
them (the Colour panel) left the other four behind.

The flags are one constant now, and these are the tests that keep it that way.
``inker_colors.FLAGS`` carries ``display_hex`` deliberately rather than the
default RGBA quartet: one box fits where four fields do not, which is what lets
a chip beside a button and a full-width row share one spelling.
"""

from __future__ import annotations

import inspect

from imgui_bundle import imgui

from warlock.studio.panes import (
    inker_bridge,
    inker_colors,
    inker_flourish,
    inker_sheet,
    inker_tools,
)

#: Every module that draws a colour the user is meant to be able to type into.
COLOUR_PANES = (inker_colors, inker_tools, inker_sheet, inker_flourish, inker_bridge)


def test_no_colour_in_inker_is_drawn_without_its_inputs():
    """The claim, over the source rather than over one constant: a new call
    site that spells ``no_inputs`` itself would satisfy any check that only
    looked at ``FLAGS``."""
    offenders = []
    for module in COLOUR_PANES:
        for line in inspect.getsource(module).splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("#:"):
                continue
            if "ColorEditFlags_.no_inputs" in stripped:
                offenders.append(f"{module.__name__}: {stripped}")
    assert offenders == [], "a colour drawn as a bare square cannot be typed into"


def test_the_shared_set_is_a_hex_box_rather_than_four_fields():
    """``display_hex`` is what makes one set serve a full-width row *and* a
    chip beside a button: the RGBA quartet needs four times the width, which is
    what ``no_inputs`` was covering for at the narrow call sites."""
    assert inker_colors.FLAGS & imgui.ColorEditFlags_.display_hex.value
    assert not (inker_colors.FLAGS & imgui.ColorEditFlags_.no_inputs.value)


def test_every_colour_call_site_reaches_for_the_one_set():
    """Five spellings is how the Colour panel came to be fixed alone. The
    chips add presentation on top -- a label suppressed, a split alpha preview
    -- and that is the only thing a caller may vary."""
    assert inker_tools._CHIP_FLAGS & inker_colors.FLAGS == inker_colors.FLAGS
    assert inker_tools._CHIP_FLAGS & imgui.ColorEditFlags_.no_label.value
    for module in (inker_tools, inker_sheet, inker_flourish, inker_bridge):
        assert "inker_colors.FLAGS" in inspect.getsource(module), module.__name__


def test_a_chip_is_sized_for_the_hex_and_not_for_a_swatch():
    """A ``color_edit4``'s item width sizes its *input*; the swatch is drawn
    after it. Sizing the two chips as if they were the whole row overflows it
    by two squares, and leaving them at ``BUTTON_W`` leaves a box too narrow to
    read a hex in -- both were shipped on the way here."""
    body = inspect.getsource(inker_tools._colour_chips)
    assert "get_content_region_avail" in body
    assert "get_frame_height" in body
    assert "max(sp(BUTTON_W)" in body, "the swatch width is the floor, not the width"
