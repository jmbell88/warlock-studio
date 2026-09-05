"""A torch that will not load must not take the health check down with it.

Finding F3, 2026-09-05. On the clean-machine install, ``pack_worker``'s pip was
writing torch into the running ``site-packages`` while the health poll ran, and
importing it raised from the Windows DLL loader rather than from the import
machinery:

    OSError: [WinError 126] ... Error loading "...\\torch\\lib\\caffe2_nvrtc.dll"
    PermissionError: [WinError 32] The process cannot access the file because
        it is being used by another process. ... "...\\torch\\lib\\shm.dll"

Neither is an ``ImportError``, so both escaped the handlers that exist for
exactly this question and killed the whole ``health`` task -- five tracebacks in
twenty-one seconds, one per poll, until the install finished. It cleared on the
next launch, so torch was never broken; the guard was.
"""

from __future__ import annotations

import builtins

import pytest

from warlock import doctor, vram


@pytest.fixture
def torch_raises(monkeypatch):
    """Make ``import torch`` fail the way a half-written install does."""

    def fail(exc):
        real = builtins.__import__

        def fake(name, *args, **kwargs):
            if name == "torch" or name.startswith("torch."):
                raise exc
            return real(name, *args, **kwargs)

        monkeypatch.delitem(__import__("sys").modules, "torch", raising=False)
        monkeypatch.setattr(builtins, "__import__", fake)

    return fail


DLL_FAILURES = [
    OSError(
        'The specified module could not be found. Error loading '
        '"C:/Warlock Studio/python/Lib/site-packages/torch/lib/caffe2_nvrtc.dll"'
    ),
    PermissionError(
        "The process cannot access the file because it is being used by another "
        'process. Error loading ".../torch/lib/shm.dll"'
    ),
]


@pytest.mark.parametrize("failure", DLL_FAILURES, ids=["winerror-126", "winerror-32"])
def test_probe_falls_back_to_nvml_instead_of_raising(torch_raises, monkeypatch, failure):
    """``vram.probe`` answers from NVML, exactly as it does with no torch at all."""
    torch_raises(failure)
    monkeypatch.setattr(vram, "device_memory", lambda: "from-nvml")
    assert vram.probe() == "from-nvml"


@pytest.mark.parametrize("failure", DLL_FAILURES, ids=["winerror-126", "winerror-32"])
def test_the_cuda_row_reports_rather_than_raising(torch_raises, failure):
    """One row says torch will not load; the other rows still get computed."""
    torch_raises(failure)
    check = doctor._cuda_check(probe=True)
    assert check.ok is False
    assert check.fatal is False, "a transient pack install must not be fatal"
    assert "will not load" in check.detail
    assert "dependency pack" in check.detail, "the likely cause is not named"


def test_a_missing_torch_still_says_it_is_not_installed(torch_raises):
    """The pre-existing branch must survive the new one: they are different answers."""
    torch_raises(ImportError("No module named 'torch'"))
    check = doctor._cuda_check(probe=True)
    assert check.ok is False
    assert "not installed" in check.detail
