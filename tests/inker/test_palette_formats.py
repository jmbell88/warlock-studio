"""The four palette formats, and the two readers of them that must not drift.

``studio/inker/gpl.py`` reads a palette for the Inker and
``pipelines/pixel.py`` reads one for the pixel pipeline, and neither may import
the other: the headless Inker package may not reach the service or pipeline
layers, and ``pipelines`` may not reach the studio. So each has its own reader
for ``.gpl``, ``.hex``, ``.pal`` and ``.txt``.

Two implementations of one format is a defect waiting to happen, and the only
thing that makes it survivable is being *pinned*: the tests below feed the same
fixture bytes to both sides and assert the same colours come back. There is one
sanctioned difference, stated in both modules -- ``gpl`` keeps the alpha byte a
Paint.NET ``.txt`` carries and ``pixel`` answers in RGB triples -- so the
comparison is on the RGB channels.

The fixtures are real files on disk rather than strings in this module, one per
format, with the same four colours in the same order in each. That is what
makes "the four formats agree" a thing this file can assert at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from warlock.pipelines import pixel
from warlock.service import palettes as svc_palettes
from warlock.studio.inker import gpl

FIXTURES = Path(__file__).parent / "fixtures" / "palettes"

#: What every fixture file says, in file order.
COLOURS = ((0, 0, 0), (255, 255, 255), (34, 139, 34), (255, 128, 64))

SUFFIXES = (".gpl", ".hex", ".pal", ".txt")


def _text(suffix: str) -> str:
    return (FIXTURES / f"sample{suffix}").read_text(encoding="utf-8")


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_every_fixture_is_a_real_file_on_disk(suffix):
    """A fixture that does not exist makes every assertion below vacuous."""
    assert (FIXTURES / f"sample{suffix}").is_file()


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_the_pipeline_reader_reads_all_four(suffix):
    assert pixel.parse_palette(_text(suffix), suffix) == COLOURS


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_the_inker_reader_reads_all_four_without_being_told_which(suffix):
    """``parse_any`` sniffs the content, because the suffix is what a download
    got renamed to and the bytes are what the file is."""
    assert [c[:3] for c in gpl.parse_any(_text(suffix))] == list(COLOURS)


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_the_two_readers_agree_byte_for_byte_on_the_same_file(suffix):
    """**The pin.** Two readers of one format is the cost the layering imposes;
    drifting apart is what would make it a defect rather than a cost."""
    text = _text(suffix)
    assert [c[:3] for c in gpl.parse_any(text)] == list(
        pixel.parse_palette(text, suffix)
    )


def test_the_two_readers_agree_on_a_malformed_row_as_well_as_a_good_one():
    """Agreeing on well-formed input is the easy half. The *tolerances* are
    where a port drifts: ``.gpl`` and ``.pal`` skip a row they cannot read and
    the hex columns refuse the file, and both sides have to make the same
    choice or the same download imports differently in two places."""
    pal = "JASC-PAL\r\n0100\r\n3\r\n0 0 0\r\nnot a colour\r\n1 2 3\r\n"
    assert [c[:3] for c in gpl.parse_jasc(pal)] == list(pixel.parse_pal(pal))
    assert pixel.parse_pal(pal) == ((0, 0, 0), (1, 2, 3))

    bad_hex = "000000\nzzzzzz\n"
    with pytest.raises(ValueError):
        gpl.parse_hex(bad_hex)
    with pytest.raises(ValueError):
        pixel.parse_hex(bad_hex)


def test_the_two_readers_agree_that_an_empty_palette_is_not_one():
    """"Imported nothing" reported as success is worse than a refusal, and it
    has to be a refusal on both sides."""
    for text, suffix in (("GIMP Palette\n#\n", ".gpl"), ("; nothing here\n", ".txt")):
        with pytest.raises(ValueError):
            gpl.parse_any(text)
        with pytest.raises(ValueError):
            pixel.parse_palette(text, suffix)


def test_the_alpha_byte_is_the_one_sanctioned_difference():
    """Paint.NET's format has a fourth channel. The Inker keeps it because a
    swatch can be translucent; the pipeline drops it because a palette there is
    what pixels are *mapped onto*. Asserted rather than left implicit, so that
    a future change to either is a change to this sentence."""
    text = "; a translucent colour\n80112233\n"
    assert gpl.parse_txt(text) == [(0x11, 0x22, 0x33, 0x80)]
    assert pixel.parse_txt(text) == ((0x11, 0x22, 0x33),)


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_every_format_round_trips_through_its_own_writer(suffix):
    written = gpl.dumps_for(suffix, [(*c, 255) for c in COLOURS], "Wave 9")
    assert [c[:3] for c in gpl.parse_any(written)] == list(COLOURS)
    # And through the *other* reader, which is the half that matters for a
    # palette this app writes and its own pixel pipeline later reads.
    assert pixel.parse_palette(written, suffix) == COLOURS


def test_the_txt_writer_round_trips_alpha_and_the_others_do_not():
    colours = [(1, 2, 3, 0x40)]
    assert gpl.parse_txt(gpl.dumps_txt(colours)) == colours
    for suffix in (".gpl", ".hex", ".pal"):
        assert gpl.parse_any(gpl.dumps_for(suffix, colours)) == [(1, 2, 3, 255)]


def test_the_hex_writer_writes_the_bare_column_every_reader_takes():
    """No header, no comment: the format *is* the column, and the one thing a
    stricter third-party reader could trip over is something we added."""
    assert gpl.dumps_hex([(0, 0, 0, 255), (255, 128, 64, 255)]) == "000000\nff8040\n"


def test_the_directory_and_the_readers_offer_the_same_four_suffixes():
    """Two-way. A suffix the directory lists with no reader behind it is a file
    the picker offers and the loader then refuses; a reader with no suffix here
    is one nothing can reach, which is exactly what ``.pal`` and ``.txt`` were
    until this wave."""
    assert set(svc_palettes.SUFFIXES) == set(pixel.PARSERS)


def test_the_export_filter_offers_exactly_what_can_be_written():
    """The Inker's save filter against the writers behind it: a filter entry
    with no writer produces a file in the wrong format under the right name."""
    from warlock.studio import inker_mode

    assert set(inker_mode.PALETTE_SUFFIXES) == set(svc_palettes.SUFFIXES)
    patterns = " ".join(inker_mode.PALETTE_FILTER)
    for suffix in svc_palettes.SUFFIXES:
        assert f"*{suffix}" in patterns


@pytest.mark.parametrize("suffix", SUFFIXES)
def test_the_service_loads_a_real_file_of_every_format(tmp_path, suffix, monkeypatch):
    """The whole door, over a directory with one real file in it: this is the
    call the Inker's folder browser makes."""
    from types import SimpleNamespace

    folder = tmp_path / "palettes"
    folder.mkdir()
    (folder / f"sample{suffix}").write_bytes(
        (FIXTURES / f"sample{suffix}").read_bytes()
    )
    config = SimpleNamespace(palette_dir=folder)
    assert svc_palettes.available(config) == ["sample"]
    name, colours, digest = svc_palettes.load(config, "sample")
    assert (name, colours) == ("sample", COLOURS)
    assert digest == pixel.palette_digest(COLOURS)
