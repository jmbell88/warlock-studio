"""Palette files as the service sees them.

A palette is a file the user dropped in a directory, so every failure here is
about a *file* and every message has to name one -- there is no registry entry
to point at and no download command to print.
"""

from __future__ import annotations

import pytest

from warlock.service import palettes
from warlock.service.errors import Invalid


@pytest.fixture
def paldir(svc, tmp_path):
    directory = tmp_path / "palettes"
    # ``exist_ok`` because this is now the same directory the ``svc`` fixture
    # pins ``WARLOCK_PALETTE_DIR`` at, and ``get_config`` creates it at startup
    # -- an empty palette folder is the only instruction a new install gets.
    directory.mkdir(exist_ok=True)
    svc.config.palette_dir = directory
    return directory


def test_a_missing_directory_is_no_palettes_rather_than_an_error(svc, tmp_path):
    # The whole feature is opt-in: a user who never made the folder gets a
    # control that offers nothing, not a failure.
    svc.config.palette_dir = tmp_path / "never-created"
    assert palettes.available(svc.config) == []


def test_listing_is_by_stem_sorted_and_covers_both_formats(svc, paldir):
    (paldir / "zzz.hex").write_text("#000000\n")
    (paldir / "aaa.gpl").write_text("GIMP Palette\n0 0 0\n")
    (paldir / "notes.txt").write_text("not a palette")
    assert palettes.available(svc.config) == ["aaa", "zzz"]


def test_load_returns_colours_and_a_content_digest(svc, paldir):
    (paldir / "duo.hex").write_text("#1a1c2c\n#f4f4f4\n")
    name, colors, digest = palettes.load(svc.config, "duo")
    assert name == "duo"
    assert colors == ((26, 28, 44), (244, 244, 244))
    assert digest

    # Editing the file in place keeps the name and must change the digest --
    # that is the whole reason staleness is keyed on content.
    (paldir / "duo.hex").write_text("#1a1c2c\n#f4f4f5\n")
    assert palettes.load(svc.config, "duo")[2] != digest


def test_an_unknown_name_says_what_is_available(svc, paldir):
    (paldir / "nord.hex").write_text("#000000\n")
    with pytest.raises(Invalid) as excinfo:
        palettes.load(svc.config, "solarized")
    assert "nord" in excinfo.value.message
    assert excinfo.value.field == "palette"


def test_a_garbage_file_is_refused_by_name(svc, paldir):
    (paldir / "broken.hex").write_text("this is not a palette\n")
    with pytest.raises(Invalid) as excinfo:
        palettes.load(svc.config, "broken")
    assert "broken.hex" in excinfo.value.message


def test_a_name_cannot_escape_the_palette_directory(svc, paldir, tmp_path):
    # The name arrives from a request; string concatenation would happily
    # resolve it upwards.
    (tmp_path / "outside.hex").write_text("#000000\n")
    with pytest.raises(Invalid):
        palettes.load(svc.config, "../outside")
