"""JASC ``.pal``: the other palette format worth reading, and only that one.

``.pal`` names three unrelated files -- JASC's text form, Microsoft's RIFF
binary, and a raw 768-byte dump -- so the magic line is checked here where
``.gpl``'s is not. Three integers on a line are unambiguous on their own; a
suffix is not, and reading a RIFF palette as text produces 256 plausible-looking
wrong colours.

The written bytes are pinned exactly, CRLF included: every file of this format
in the wild has them, and some of the readers are old enough to care.
"""

from __future__ import annotations

import pytest

from warlock.studio.inker import gpl

SAMPLE = "JASC-PAL\r\n0100\r\n3\r\n0 0 0\r\n255 255 255\r\n34 139 34\r\n"


def test_a_well_formed_pal_reads_in_file_order():
    assert gpl.parse_jasc(SAMPLE) == [
        (0, 0, 0, 255),
        (255, 255, 255, 255),
        (34, 139, 34, 255),
    ]


def test_the_written_bytes_are_pinned():
    """The exact file, because "it round-trips through our own reader" is the
    one property a broken writer also has."""
    text = gpl.dumps_jasc([(0, 0, 0, 255), (255, 128, 64, 255)])
    assert text == "JASC-PAL\r\n0100\r\n2\r\n0 0 0\r\n255 128 64\r\n"


def test_the_header_is_the_two_lines_every_reader_looks_for():
    lines = gpl.dumps_jasc([(1, 2, 3, 255)]).split("\r\n")
    assert lines[0] == "JASC-PAL"
    assert lines[1] == "0100"
    assert lines[2] == "1", "the declared count"


def test_lf_line_endings_are_tolerated_on_the_way_in():
    """Written CRLF, read either: a file that came through git, a text editor or
    a paste is still the user's palette."""
    assert gpl.parse_jasc(SAMPLE.replace("\r\n", "\n")) == gpl.parse_jasc(SAMPLE)


def test_a_file_that_is_not_jasc_is_refused_rather_than_guessed():
    with pytest.raises(ValueError):
        gpl.parse_jasc("RIFF\x00\x00PAL data\x00")
    with pytest.raises(ValueError):
        gpl.parse_jasc("GIMP Palette\n1 2 3\n")
    with pytest.raises(ValueError):
        gpl.parse_jasc("")


def test_a_declared_count_that_disagrees_with_the_rows_does_not_lose_colours():
    """The count is wrong in enough files in the wild that trusting it would
    drop real colours. The rows are the palette."""
    text = "JASC-PAL\n0100\n2\n1 1 1\n2 2 2\n3 3 3\n"
    assert len(gpl.parse_jasc(text)) == 3


def test_a_malformed_row_is_skipped_rather_than_failing_the_file():
    text = "JASC-PAL\n0100\n3\n1 1 1\nnot a colour\n3 3 3\n"
    assert gpl.parse_jasc(text) == [(1, 1, 1, 255), (3, 3, 3, 255)]


def test_a_fourth_column_is_ignored_rather_than_read_as_alpha():
    """Some writers append one; the format has no alpha and neither does the
    ``.gpl`` beside it, so exported swatches stay opaque in both."""
    assert gpl.parse_jasc("JASC-PAL\n0100\n1\n10 20 30 128\n") == [(10, 20, 30, 255)]


def test_out_of_range_components_are_clamped_rather_than_wrapped():
    assert gpl.parse_jasc("JASC-PAL\n0100\n1\n300 -20 7\n") == [(255, 0, 7, 255)]


def test_a_file_with_no_colours_at_all_is_refused():
    with pytest.raises(ValueError):
        gpl.parse_jasc("JASC-PAL\r\n0100\r\n0\r\n")


def test_writing_then_reading_gives_the_colours_back():
    colours = [(0, 0, 0, 255), (255, 128, 64, 255), (12, 200, 90, 255)]
    assert gpl.parse_jasc(gpl.dumps_jasc(colours)) == colours


def test_alpha_is_dropped_because_the_format_has_no_room_for_it():
    assert gpl.parse_jasc(gpl.dumps_jasc([(10, 20, 30, 7)])) == [(10, 20, 30, 255)]


# --- one reader for both formats ---------------------------------------------


def test_the_format_is_sniffed_on_the_first_line_and_not_on_the_suffix():
    """Which is what a download got renamed to; the magic is what the file is."""
    assert gpl.parse_any(SAMPLE)[0] == (0, 0, 0, 255)
    assert gpl.parse_any("GIMP Palette\nName: x\n10 20 30\n") == [(10, 20, 30, 255)]


def test_a_headerless_gpl_still_reads_through_the_sniffer():
    assert gpl.parse_any("1 2 3\n4 5 6\n") == [(1, 2, 3, 255), (4, 5, 6, 255)]


def test_the_writer_is_chosen_by_the_suffix():
    colours = [(1, 2, 3, 255)]
    assert gpl.dumps_for(".pal", colours).startswith("JASC-PAL")
    assert gpl.dumps_for(".gpl", colours).startswith("GIMP Palette")
    assert gpl.dumps_for(".PAL", colours).startswith("JASC-PAL")


def test_an_unrecognised_suffix_writes_a_gpl_rather_than_refusing():
    """The export dialog produces whatever the user typed, and a typed name with
    no suffix at all still has to produce a file."""
    assert gpl.dumps_for("", [(1, 2, 3, 255)]).startswith("GIMP Palette")
