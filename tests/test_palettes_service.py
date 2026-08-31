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


def test_listing_is_by_stem_sorted_and_covers_every_format(svc, paldir):
    (paldir / "zzz.hex").write_text("#000000\n")
    (paldir / "aaa.gpl").write_text("GIMP Palette\n0 0 0\n")
    (paldir / "mmm.pal").write_text("JASC-PAL\n0100\n1\n0 0 0\n")
    (paldir / "nnn.txt").write_text("ff000000\n")
    # Not a palette suffix, and the listing is what decides: a file with one of
    # the four above is read, and a file without one never appears at all.
    (paldir / "notes.md").write_text("not a palette")
    assert palettes.available(svc.config) == ["aaa", "mmm", "nnn", "zzz"]


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


# --- and the same lookup, in the worker ----------------------------------------
#
# ``queue._palette_entries`` is the queue's own restatement of ``_path`` plus
# the parse, because the queue may not import the service. Two implementations
# of a containment check is one chance to fix only one of them, so they are
# tested beside each other on purpose.


def test_the_worker_reads_the_same_colours_the_service_does(svc, paldir):
    from warlock import queue as queue_mod

    (paldir / "duo.hex").write_text("#1a1c2c\n#f4f4f4\n")
    assert queue_mod._palette_entries(svc.config, "duo") == palettes.load(
        svc.config, "duo"
    )[1]


def test_the_worker_lookup_also_refuses_a_traversal(svc, paldir, tmp_path):
    """The security property, not a formality: ``name`` reaches the worker out
    of a params blob, which outlives the door that validated it."""
    (tmp_path / "outside.hex").write_text("#000000\n")
    from warlock import queue as queue_mod

    with pytest.raises(RuntimeError, match="no longer installed"):
        queue_mod._palette_entries(svc.config, "../outside")
    assert not (paldir / "../outside.hex").resolve().is_relative_to(paldir)


def test_no_palette_named_is_no_colours_rather_than_a_refusal(svc, paldir):
    from warlock import queue as queue_mod

    assert queue_mod._palette_entries(svc.config, "") == ()


def test_a_palette_deleted_after_the_door_is_named_in_the_failure(svc, paldir):
    from warlock import queue as queue_mod

    with pytest.raises(RuntimeError, match="palette 'gone' is no longer installed"):
        queue_mod._palette_entries(svc.config, "gone")


# --- and what the picker says the folder takes ---------------------------------
#
# The failure this closes was silent in the worst way available to a form: the
# Palette combo's helper advertised ``.pal`` and ``.txt``, ``SUFFIXES`` has never
# carried either, and both the listing and the load are keyed on it -- so a user
# who dropped a JASC .pal in the folder was told it would work and then got no
# error, no row, and nothing to look at.


def test_the_helper_names_exactly_the_suffixes_the_loader_accepts():
    """Computed from ``SUFFIXES``, so it cannot name a format that never loads.

    Both directions: every suffix appears, and no other dotted token does. The
    second half is the one that would have caught the original defect.
    """
    import re

    for suffix in palettes.SUFFIXES:
        assert suffix in palettes.SUFFIX_HELP
    assert set(re.findall(r"\.[a-z]+", palettes.SUFFIX_HELP)) == set(palettes.SUFFIXES)


def test_the_2d_form_draws_that_helper_rather_than_a_list_of_its_own(svc, paldir):
    """Through the real ``_pixel_look``, with a recording form.

    The precondition is asserted rather than assumed: the palette combo is only
    drawn when there is something to pick, so a fixture with an empty folder and
    no chosen palette would make every assertion below vacuous by simply never
    reaching the control.
    """
    from types import SimpleNamespace

    from warlock.studio.panes import settings_2d

    (paldir / "duo.hex").write_text("#1a1c2c\n#f4f4f4\n")

    class _Recorder:
        def __init__(self):
            self.calls = {}

        def combo(self, field, label, current, options, **kw):
            self.calls[field] = kw
            return (False, current)

        def switch(self, field, label, value, **kw):
            self.calls[field] = kw
            return (False, value)

        def segmented_choice(self, field, label, current, options, **kw):
            self.calls[field] = kw
            return (False, current)

    ctx = SimpleNamespace(
        svc=svc, state=SimpleNamespace(palettes=None, clear_field_error=lambda _f: None)
    )
    form_ui = _Recorder()
    form = {"palette": "duo"}
    settings_2d._pixel_look(ctx, form, form_ui, sprite=True)

    # Precondition: the control this test is about was actually drawn.
    assert "palette" in form_ui.calls
    assert form_ui.calls["palette"]["helper"] == palettes.SUFFIX_HELP
    # Derived, so the line the form draws cannot name a format the loader does
    # not read. ``.pal`` and ``.txt`` were the two it wrongly advertised until
    # 2026-08-29 and the two the loader gained on 2026-08-30; what is asserted
    # is that the two agree, not that any one suffix is in or out.
    assert ".pal" in form_ui.calls["palette"]["helper"]
    assert ".aco" not in form_ui.calls["palette"]["helper"]
