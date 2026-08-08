"""GIMP palette files, read and written.

The interchange half of the swatch row, so the tests are about *other*
people's files as much as our own: a palette downloaded from anywhere is the
input this has to survive.
"""

from __future__ import annotations

import pytest

from warlock.studio.inker import gpl

SAMPLE = """GIMP Palette
Name: Nord
Columns: 4
#
 46  52  64\tnord0
 59  66  82\tnord1
216 222 233\tnord4
"""


def test_a_well_formed_palette_reads_in_file_order():
    assert gpl.parse(SAMPLE) == [
        (46, 52, 64, 255),
        (59, 66, 82, 255),
        (216, 222, 233, 255),
    ]


def test_a_palette_with_no_header_still_reads():
    """Plenty in the wild start straight in on the numbers, and three integers
    on a line are unambiguous enough to read without being told."""
    assert gpl.parse("1 2 3\n4 5 6\n") == [(1, 2, 3, 255), (4, 5, 6, 255)]


def test_comments_and_key_lines_are_skipped():
    text = "GIMP Palette\nName: x\nColumns: 16\n# a note\n\n10 20 30\n"
    assert gpl.parse(text) == [(10, 20, 30, 255)]


def test_a_malformed_row_is_skipped_rather_than_failing_the_file():
    """A palette that opens missing one colour is a palette the user still
    has; the alternative is one bad line costing the whole download."""
    text = "GIMP Palette\n1 2 3\nnot a colour at all\n4 5 6\n"
    assert gpl.parse(text) == [(1, 2, 3, 255), (4, 5, 6, 255)]


def test_out_of_range_components_are_clamped_rather_than_wrapped():
    assert gpl.parse("300 -20 7\n") == [(255, 0, 7, 255)]


def test_a_file_with_no_colours_at_all_is_refused():
    """"Imported nothing" reported as success is worse than a refusal."""
    with pytest.raises(ValueError):
        gpl.parse("GIMP Palette\nName: empty\n#\n")
    with pytest.raises(ValueError):
        gpl.parse("")


def test_writing_then_reading_gives_the_colours_back():
    colours = [(0, 0, 0, 255), (255, 128, 64, 255), (12, 200, 90, 255)]
    assert gpl.parse(gpl.dumps(colours)) == colours


def test_alpha_is_dropped_because_the_format_has_no_room_for_it():
    """Stated as a loss rather than worked around: a private fourth column
    would round-trip here and be ignored by every other reader, which is the
    opposite of the reason to use this format."""
    text = gpl.dumps([(10, 20, 30, 7)])
    assert gpl.parse(text) == [(10, 20, 30, 255)]


def test_the_written_header_is_the_one_other_editors_look_for():
    text = gpl.dumps([(1, 2, 3, 255)], name="Mine")
    lines = text.splitlines()
    assert lines[0] == "GIMP Palette"
    assert lines[1] == "Name: Mine"
    # Zero means "reader's choice", which is the honest answer for a row that
    # wraps to whatever the panel is wide.
    assert lines[2] == "Columns: 0"
    assert text.endswith("\n")
