

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

    monkeypatch.setattr(settings_mod.tempfile, "mkstemp", _boom)
    assert settings.flush() is False
    notice = settings.take_notice()
    assert notice is not None and "cannot be saved" in notice

    # And it does not raise a fresh notice on every retry -- a debounced tick
    # would otherwise toast once a second forever.
    assert settings.flush() is False
    assert settings.take_notice() is None
