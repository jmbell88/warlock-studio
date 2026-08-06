"""The native library loader: absent, disabled and stale must all mean numpy.

Nothing here needs the DLL to exist. That is the point -- ``vendor/`` is
gitignored, so "no warlockc.dll" is the state a fresh clone is in and the state
CI runs in, and every one of these paths has to end in the numpy fallback
rather than in an exception.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

from warlock import native


@pytest.fixture(autouse=True)
def fresh_probe(monkeypatch):
    """The probe is cached per process; these tests move the ground under it.

    ``WARLOCK_NATIVE`` is pinned on rather than merely deleted: the whole suite
    is meant to be runnable with ``WARLOCK_NATIVE=0`` to exercise the fallback,
    and a test about *why else* the library might be unavailable has to say so
    itself instead of inheriting an answer from the environment.
    """
    monkeypatch.setenv("WARLOCK_NATIVE", "1")
    native.reset()
    yield
    native.reset()


def test_the_env_var_forces_the_fallback(monkeypatch):
    monkeypatch.setenv("WARLOCK_NATIVE", "0")
    assert native.available() is False
    ok, detail = native.status()
    assert ok is False
    assert "WARLOCK_NATIVE=0" in detail


@pytest.mark.parametrize("value", ["0", "off", "false", "no", "OFF", " False "])
def test_every_spelling_of_off_is_off(monkeypatch, value):
    monkeypatch.setenv("WARLOCK_NATIVE", value)
    assert native.available() is False


def test_a_missing_dll_is_not_an_error(monkeypatch, tmp_path):
    monkeypatch.setenv("WARLOCK_NATIVE_DLL", str(tmp_path / "nope.dll"))
    assert native.available() is False
    ok, detail = native.status()
    assert ok is False
    assert "not built" in detail


def test_a_file_that_is_not_a_library_is_not_an_error(monkeypatch, tmp_path):
    """A truncated or half-written build artifact, which a killed build leaves
    behind. It must read as 'use numpy', not take the process down."""
    fake = tmp_path / "warlockc.dll"
    fake.write_bytes(b"not a dll")
    monkeypatch.setenv("WARLOCK_NATIVE_DLL", str(fake))
    assert native.available() is False
    assert native.status()[0] is False


def test_the_override_decides_where_the_dll_is_looked_for(monkeypatch, tmp_path):
    """A git worktree has no vendor/ at all -- the same reason
    WARLOCK_GLTFPACK and the other path overrides exist."""
    target = tmp_path / "elsewhere" / "warlockc.dll"
    monkeypatch.setenv("WARLOCK_NATIVE_DLL", str(target))
    assert native.dll_path() == target


def test_the_default_path_is_the_vendor_directory(monkeypatch):
    monkeypatch.delenv("WARLOCK_NATIVE_DLL", raising=False)
    path = native.dll_path()
    assert path.parts[-3:] == ("vendor", "warlockc", "warlockc.dll")


def test_an_abi_mismatch_is_refused_rather_than_trusted(monkeypatch):
    """The failure mode this exists for: vendor/ is gitignored, so a checkout
    routinely holds a DLL built from older sources. An absent DLL is obvious; a
    stale one silently computing last week's behaviour is not."""
    calls: list[str] = []

    class _Stub:
        def __init__(self, path: str) -> None:
            calls.append(path)
            self.warlockc_abi = _Fn(native.ABI + 1)
            self.warlockc_rasterise = _Fn(None)

    class _Fn:
        def __init__(self, value):
            self.value = value
            self.restype = None
            self.argtypes: list = []

        def __call__(self, *a):
            return self.value

    monkeypatch.setattr(ctypes, "CDLL", _Stub)
    monkeypatch.setattr(native.Path, "exists", lambda self: True, raising=False)
    assert native.available() is False
    assert calls, "the stub was actually consulted"


def test_the_probe_is_cached(monkeypatch):
    """Callers ask per job; loading is a file check plus a link."""
    loads = 0
    real = native._load

    def counted():
        nonlocal loads
        loads += 1
        return real()

    monkeypatch.setattr(native, "_load", counted)
    native.reset()
    native.available()
    native.available()
    native.available()
    assert loads == 1


# --- the built library, when there is one ------------------------------------

needs_dll = pytest.mark.skipif(
    not Path(native._DEFAULT_DLL).exists(), reason="warlockc.dll not built"
)


@needs_dll
def test_the_built_library_reports_the_abi_this_build_expects(monkeypatch):
    monkeypatch.delenv("WARLOCK_NATIVE", raising=False)
    monkeypatch.delenv("WARLOCK_NATIVE_DLL", raising=False)
    native.reset()
    assert native.available() is True
    assert int(native.lib().warlockc_abi()) == native.ABI
