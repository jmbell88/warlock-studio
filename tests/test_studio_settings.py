import json

import pytest

from warlock.service import tilesheets as svc_tilesheets
from warlock.studio import settings as settings_mod
from warlock.studio.settings import Settings, restore_form, sanitise_form
from warlock.studio.state import default_form_2d


def _write(tmp_path, data):
    """A settings file this version will read, holding *data*."""
    (tmp_path / settings_mod.FILENAME).write_text(
        json.dumps({"version": settings_mod.VERSION, "data": data}), encoding="utf-8"
    )


def test_a_corrupt_file_is_preserved_and_reported(tmp_path):
    """UX-10: it used to reset to defaults with only a log line -- and the
    first successful save then overwrote the file the user might have wanted
    back. Every preference reverted with nothing on screen saying why."""
    from warlock.studio.settings import FILENAME, Settings

    path = tmp_path / FILENAME
    path.write_text("{not json at all", encoding="utf-8")

    settings = Settings.load(tmp_path)

    assert settings.data == {}, "defaults are still the fallback"
    notice = settings.take_notice()
    assert notice is not None and "reset to defaults" in notice
    assert settings.take_notice() is None, "a notice is reported once, not every frame"
    # The original is kept rather than destroyed, under a name that a second
    # corruption cannot overwrite.
    kept = list(tmp_path.glob("*.corrupt-*.json"))
    assert len(kept) == 1
    assert kept[0].read_text(encoding="utf-8") == "{not json at all"
    assert not path.exists()


def test_a_save_that_cannot_be_written_says_so_once(tmp_path, monkeypatch):
    """A read-only or full data directory meant preferences silently stopped
    persisting for the whole session."""
    from warlock.studio import settings as settings_mod
    from warlock.studio.settings import Settings

    settings = Settings.load(tmp_path)
    settings.set("theme", "light")

    def _boom(*_a, **_kw):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(settings_mod.atomic.os, "replace", _boom)
    assert settings.flush() is False
    notice = settings.take_notice()
    assert notice is not None and "cannot be saved" in notice

    # And it does not raise a fresh notice on every retry -- a debounced tick
    # would otherwise toast once a second forever.
    assert settings.flush() is False
    assert settings.take_notice() is None


def test_a_failed_save_leaves_no_staging_file_behind(tmp_path, monkeypatch):
    """The write went through a hand-rolled mkstemp + replace, and a replace
    that raised left the temporary where it was -- with ``_dirty`` still set,
    ``tick`` then retried once a second, one orphaned ``.settings.*.json``
    per second for as long as the directory stayed unwritable."""
    from warlock.studio import settings as settings_mod
    from warlock.studio.settings import Settings

    settings = Settings.load(tmp_path)
    settings.set("theme", "light")

    def _boom(*_a, **_kw):
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(settings_mod.atomic.os, "replace", _boom)
    for _ in range(3):
        assert settings.flush() is False
    strays = [p.name for p in tmp_path.iterdir() if p.name.startswith(".")]
    assert strays == []


# -- the tile layout a stored form predates ----------------------------------


def test_a_form_saved_before_the_layout_control_reopens_on_the_grid(tmp_path):
    """The tile arm had one layout when this file was written: the 8x8 grid.

    So the absent key is not an absent opinion. Resolving it to the door's
    default would reinterpret a saved 8x8 grid request as a materials sheet --
    a request that layout cannot even satisfy, because the stored form holds no
    list of surfaces, so the first Generate after the upgrade is refused at
    ``field="prompt_items"`` over a control the user never chose.
    """
    _write(
        tmp_path,
        {
            "theme": "light",
            "form_2d": {
                "prompt": "mossy dungeon",
                "asset_type": "tileset",
                "tile_size": "48",
                "projection": "isometric",
            },
        },
    )
    loaded = Settings.load(tmp_path)
    restored = restore_form(default_form_2d(), loaded.get("form_2d"))
    assert restored["tile_mode"] == svc_tilesheets.MODE_GRID
    # Everything else in the file is the file's, including the two values only
    # the grid draws -- a 48 px tile and a view that does not wrap.
    assert (restored["tile_size"], restored["projection"]) == ("48", "isometric")
    assert loaded.get("theme") == "light"


def test_reading_a_pre_control_form_neither_resets_nor_rewrites_the_file(tmp_path):
    """The 2026-08-28 shape, refused here.

    A migration that raises is caught by ``load`` and answered by renaming the
    whole file aside and starting from defaults -- every preference the user
    has, gone, for one key. That net is why the settings-reset defect stayed
    invisible for a release, so this pins that the layout migration does not
    use it: no notice, and nothing marked dirty either, because a launch that
    changes nothing else should not rewrite the file.
    """
    _write(tmp_path, {"form_2d": {"prompt": "mossy dungeon", "tile_size": "48"}})
    loaded = Settings.load(tmp_path)
    assert loaded.get("form_2d")["tile_mode"] == svc_tilesheets.MODE_GRID
    assert loaded.take_notice() is None
    assert loaded.flush() is False


def test_a_form_the_file_never_held_is_a_fresh_one(tmp_path):
    """No 2D block, and an empty one, both describe no request at all."""
    _write(tmp_path, {"theme": "light"})
    loaded = Settings.load(tmp_path)
    assert loaded.get("form_2d") is None
    restored = restore_form(default_form_2d(), loaded.get("form_2d"))
    assert restored["tile_mode"] == svc_tilesheets.DEFAULT_MODE

    _write(tmp_path, {"form_2d": {}})
    assert "tile_mode" not in Settings.load(tmp_path).get("form_2d")


@pytest.mark.parametrize("mode", svc_tilesheets.TILE_MODES)
def test_a_chosen_layout_survives_a_save_and_a_reopen(tmp_path, mode):
    """The failure that would make the migration worse than the bug it fixes:
    a user who picks Materials, quits, and is put back on the grid next launch.
    Once the key is present the migration is a no-op, whatever it says."""
    settings = Settings.load(tmp_path)
    form = default_form_2d()
    form["tile_mode"] = mode
    settings.set("form_2d", sanitise_form(form))
    assert settings.flush() is True

    reopened = Settings.load(tmp_path)
    assert restore_form(default_form_2d(), reopened.get("form_2d"))["tile_mode"] == mode


def test_the_layout_migration_decides_on_any_stored_form_without_raising():
    """Total on purpose. Every branch above depends on this one not needing
    ``load``'s reset net, so it is asserted against the shapes a hand-edited or
    ancient file can actually hold rather than against the shapes it writes."""
    grid = svc_tilesheets.MODE_GRID
    cases = [
        ({}, {}),
        ({"prompt": None}, {"prompt": None, "tile_mode": grid}),
        ({"tile_size": 48}, {"tile_size": 48, "tile_mode": grid}),
        ({"tile_mode": "materials"}, {"tile_mode": "materials"}),
        ({"tile_mode": None}, {"tile_mode": None}),
        ({"tile_mode": {}}, {"tile_mode": {}}),
    ]
    for stored, expected in cases:
        form = dict(stored)
        settings_mod._migrate_tile_mode(form)
        assert form == expected, stored
        # Idempotent: migrations here run under version 1, every launch.
        settings_mod._migrate_tile_mode(form)
        assert form == expected, stored
