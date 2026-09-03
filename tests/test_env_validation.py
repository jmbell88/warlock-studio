"""Malformed ``WARLOCK_*`` values must not crash before diagnostics exist.

``get_config()`` runs *early*: inside module import, before ``studio.main``
installs a log handler, before Doctor is reachable and before there is a window
to say anything in. Several numeric fields called bare ``int()``/``float()`` in
their ``default_factory``, so one typo in a port or a timeout raised a
``ValueError`` from there -- the user saw a traceback naming ``int()`` and
nothing naming the variable.

The other half of the same finding is the inconsistency: the optional VRAM
floats already swallowed a malformed value and returned ``None``, so an explicit
safety limit with a typo in it silently became "unset". One policy now covers
both -- fall back to the documented default, record it, and report every one of
them in a single Doctor row (RUN-03).
"""

from __future__ import annotations

import pytest

from warlock import config as config_mod
from warlock import doctor
from warlock.config import Config


@pytest.fixture(autouse=True)
def _clean_notes():
    """The note list is module state, deliberately -- it is read by Doctor long
    after the parse. Cleared around each test so one case cannot see another's."""
    config_mod.INVALID_ENV.clear()
    yield
    config_mod.INVALID_ENV.clear()


@pytest.mark.parametrize(
    "name,default_attr",
    [
        ("WARLOCK_TRELLIS_PORT", "trellis_port"),
        ("WARLOCK_TRELLIS_IDLE", "trellis_idle_timeout"),
        ("WARLOCK_RIG_TIMEOUT", "rig_timeout"),
        ("WARLOCK_MESH_HOLE_MAX", "mesh_hole_max"),
        ("WARLOCK_REFERENCE_RETRIES", "reference_retries"),
    ],
)
def test_a_typo_falls_back_to_the_default_instead_of_raising(
    name, default_attr, monkeypatch
):
    baseline = getattr(Config(), default_attr)
    monkeypatch.setenv(name, "seventeen")
    built = Config()
    assert getattr(built, default_attr) == baseline
    assert any(entry[0] == name for entry in config_mod.INVALID_ENV)


def test_a_malformed_optional_limit_is_recorded_rather_than_silently_unset(monkeypatch):
    """The inverse half. ``None`` is still the fallback -- these are the VRAM
    limits, where unset genuinely is the default -- but it is no longer silent:
    an explicit safety limit with a typo in it used to become "unset" with
    nothing said anywhere."""
    monkeypatch.setenv("WARLOCK_VRAM_TOTAL", "lots")
    built = Config()
    assert built.vram_total_gib is None
    assert any(entry[0] == "WARLOCK_VRAM_TOTAL" for entry in config_mod.INVALID_ENV)


def test_decim_zero_parses_as_zero_and_a_word_is_recorded(monkeypatch):
    """``WARLOCK_TRELLIS_DECIM=0`` is the setting that turns the engine's own
    decimation off, so 0 has to survive parsing as 0 rather than as unset."""
    monkeypatch.setenv("WARLOCK_TRELLIS_DECIM", "0")
    assert Config().trellis_decim == 0
    monkeypatch.setenv("WARLOCK_TRELLIS_DECIM", "grid")
    config_mod.INVALID_ENV.clear()
    assert Config().trellis_decim is None
    assert any(entry[0] == "WARLOCK_TRELLIS_DECIM" for entry in config_mod.INVALID_ENV)


def test_auto_is_still_a_word_and_not_a_typo(monkeypatch):
    """``auto`` is the documented way to say "omit the flag and let the exe
    decide", so it must not be reported as unparseable."""
    monkeypatch.setenv("WARLOCK_TRELLIS_BAND", "auto")
    assert Config().trellis_band is None
    assert config_mod.INVALID_ENV == []


def test_doctor_reports_every_bad_value_in_one_row(monkeypatch, tmp_path):
    """One row, not one per variable: a machine with three typos has one
    problem, and three amber lines would bury the rest of the report."""
    monkeypatch.setenv("WARLOCK_TRELLIS_PORT", "eleven")
    monkeypatch.setenv("WARLOCK_RIG_TIMEOUT", "soon")
    Config()

    rows = [c for c in doctor.volatile_checks(Config()) if c.name == "environment"]
    assert len(rows) == 1
    row = rows[0]
    assert row.ok is False
    assert row.fatal is False, "the app is running, just not as configured"
    assert "WARLOCK_TRELLIS_PORT" in row.detail
    assert "WARLOCK_RIG_TIMEOUT" in row.detail


def test_a_clean_environment_says_so(tmp_path):
    rows = [c for c in doctor.volatile_checks(Config()) if c.name == "environment"]
    assert rows and rows[0].ok is True
