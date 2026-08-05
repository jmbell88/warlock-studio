"""The zero-network guarantee: no downloads, no hub revalidation, ever."""

from __future__ import annotations

import importlib
import os

import pytest

import warlock
from warlock import models
from warlock.config import Config
from warlock.pipelines.text2image import Text2Image


def test_package_import_sets_hf_offline_env(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_HUB_DISABLE_TELEMETRY", raising=False)
    # The package is already imported by the time tests run; re-execute its
    # __init__ to observe the env side effect from a clean slate.
    importlib.reload(warlock)
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"


def test_package_import_does_not_override_explicit_env(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    importlib.reload(warlock)
    assert os.environ["HF_HUB_OFFLINE"] == "0"


def test_missing_weights_raise_actionable_error(tmp_path):
    # Load-bearing ordering: the existence check precedes the torch import,
    # so this passes (fast) even without the text2image extra installed.
    with pytest.raises(RuntimeError, match="hf download"):
        Text2Image(models.BASE_MODELS["turbo"], tmp_path / "nope").load()


def test_vram_exclusive_flag_parses_from_env(monkeypatch):
    monkeypatch.setenv("WARLOCK_VRAM_EXCLUSIVE", "1")
    assert Config().vram_exclusive is True
    monkeypatch.setenv("WARLOCK_VRAM_EXCLUSIVE", "off")
    assert Config().vram_exclusive is False
    # Tri-state: unset is neither, so vram.plan() can decide from the card
    # without overruling a user who explicitly chose coexist.
    monkeypatch.delenv("WARLOCK_VRAM_EXCLUSIVE")
    assert Config().vram_exclusive is None


def test_vram_logging_never_imports_torch_itself(monkeypatch):
    """vram_gib runs on the event loop, and importing torch takes seconds -- an
    image job on a machine where torch isn't loaded must not pay for it. The
    number would be zero anyway: memory_allocated() only sees this process, so
    it is meaningful exactly when the SDXL pipeline is already loaded."""
    import builtins
    import sys

    from warlock.queue import vram_gib

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    real_import = builtins.__import__

    def refusing_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("vram_gib must not import torch")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refusing_import)
    assert vram_gib() is None


def test_vram_reports_from_an_already_loaded_torch(monkeypatch):
    import sys
    import types

    from warlock.queue import vram_gib

    gib = 1024**3
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                is_available=lambda: True,
                memory_allocated=lambda: 7 * gib,
                memory_reserved=lambda: 8 * gib,
            )
        ),
    )
    assert vram_gib() == (7.0, 8.0)


@pytest.mark.parametrize(
    "module",
    [
        "warlock.pipelines.asset2d",
        "warlock.pipelines.conditioning",
        "warlock.pipelines.control",
        "warlock.pipelines.reference",
        "warlock.pipelines.rank",
        "warlock.provenance",
    ],
)
def test_the_pure_modules_stay_torch_free(module):
    """Same rule pipelines/prompt.py and pipelines/sheet.py follow: everything
    decidable is importable and testable without the text2image extra. A
    top-level torch import here would cost seconds on every mesh-only job and
    make these modules untestable on a machine without CUDA."""
    import subprocess
    import sys

    code = (
        f"import {module}, sys; "
        "bad = [m for m in ('torch', 'diffusers', 'transformers') if m in sys.modules]; "
        "print(','.join(bad))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == ""
