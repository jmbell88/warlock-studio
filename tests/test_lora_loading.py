"""Which style LoRAs reach the pipe, and which adapters get applied.

``Text2Image.__init__`` builds no pipeline, so a fake stands in for one and
both halves are reachable with no torch, no weights and no GPU. What is under
test is the pairing filter in ``_load_loras`` and the predicate that replaced
``_apply_adapters``' family early return -- neither of which any test could
see before, because the family return made both functions no-ops off SDXL.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from warlock import models
from warlock.pipelines.text2image import Text2Image


class FakePipe:
    """Records what a real diffusers pipeline would have been asked to do."""

    def __init__(self) -> None:
        self.loaded: list[str] = []
        self.applied: list[tuple[list[str], list[float]]] = []
        self.disabled = 0
        self.enabled = 0
        # The order the pipe was driven in, which is half of what is under test.
        self.calls: list[str] = []

    def load_lora_weights(self, _dir, weight_name, adapter_name, **_kw) -> None:
        self.loaded.append(adapter_name)

    def set_adapters(self, names, weights) -> None:
        self.applied.append((list(names), list(weights)))
        self.calls.append("set")

    def enable_lora(self) -> None:
        self.enabled += 1
        self.calls.append("enable")

    def disable_lora(self) -> None:
        self.disabled += 1
        self.calls.append("disable")


def _t2i(tmp_path: Path, base_key: str, on_disk: list[str]) -> tuple[Text2Image, FakePipe]:
    loras = tmp_path / "loras"
    loras.mkdir(exist_ok=True)
    for name in on_disk:
        (loras / name).write_bytes(b"")
    t2i = Text2Image(models.BASE_MODELS[base_key], tmp_path)
    pipe = FakePipe()
    t2i._pipe = pipe
    return t2i, pipe


def _all_files() -> list[str]:
    return [lora.filename for lora in models.STYLE_LORAS.values()]


def _fitting(base_key: str) -> list[str]:
    return models.loras_by_base()[base_key]


@pytest.mark.parametrize("base_key", sorted(models.BASE_MODELS))
def test_only_same_family_adapters_are_loaded(tmp_path, base_key):
    """Every style LoRA in the registry is on disk; only the fitting ones are
    handed to the pipe. Loading a foreign one would raise with the checkpoint
    already resident, which is why this is a filter and not the missing-file
    tolerance beside it."""
    base = models.BASE_MODELS[base_key]
    files = _all_files() + ([base.base_lora] if base.base_lora else [])
    t2i, pipe = _t2i(tmp_path, base_key, files)
    t2i._load_loras()
    assert sorted(t2i._adapters) == sorted(_fitting(base_key))
    foreign = {
        lora.key for lora in models.STYLE_LORAS.values() if lora.family != base.family
    }
    assert not foreign & set(pipe.loaded)


def test_a_missing_same_family_adapter_is_still_skipped_not_fatal(tmp_path):
    """The pre-existing tolerance is untouched: the LoRAs are separate optional
    downloads and a user who fetched none must still be able to generate."""
    t2i, pipe = _t2i(tmp_path, "turbo", [])
    t2i._load_loras()
    assert t2i._adapters == set()
    assert pipe.loaded == []


def test_a_pipe_with_no_adapters_is_never_disabled(tmp_path):
    """The latent case the family early return hid, and it was wrong for SDXL
    too: ``turbo`` has no base LoRA, so on a host with no loras/ directory the
    empty branch called disable_lora() on a pipe with no PEFT state."""
    t2i, pipe = _t2i(tmp_path, "turbo", [])
    t2i._load_loras()
    assert not t2i._has_adapters
    t2i._apply_adapters(pipe, None, 1.0)
    assert pipe.disabled == 0
    assert pipe.applied == []


def test_a_loaded_adapter_is_disabled_when_nothing_is_selected(tmp_path):
    t2i, pipe = _t2i(tmp_path, "turbo", _all_files())
    t2i._load_loras()
    assert t2i._has_adapters
    t2i._apply_adapters(pipe, None, 1.0)
    assert pipe.disabled == 1


def test_a_selected_adapter_is_applied_at_its_weight(tmp_path):
    t2i, pipe = _t2i(tmp_path, "turbo", _all_files())
    t2i._load_loras()
    chosen = _fitting("turbo")[0]
    t2i._apply_adapters(pipe, chosen, 0.8)
    assert pipe.applied == [([chosen], [0.8])]


def test_a_foreign_adapter_selected_anyway_is_warned_about_not_applied(tmp_path, caplog):
    """guidance.normalize refuses the pair at the door and queue.py drops it
    for a stored row, so this is the third line of defence -- and it must not
    raise, because the artifact is a job the user can no longer edit."""
    base_key, lora_key = next(
        (b.key, lo.key)
        for b in models.BASE_MODELS.values()
        for lo in models.STYLE_LORAS.values()
        if not models.lora_fits(b, lo)
    )
    base = models.BASE_MODELS[base_key]
    files = _all_files() + ([base.base_lora] if base.base_lora else [])
    t2i, pipe = _t2i(tmp_path, base_key, files)
    t2i._load_loras()
    with caplog.at_level("WARNING"):
        t2i._apply_adapters(pipe, lora_key, 1.0)
    assert lora_key not in [n for names, _ in pipe.applied for n in names]
    assert any(lora_key in r.getMessage() for r in caplog.records)


def test_a_style_survives_a_run_that_had_none_before_it(tmp_path):
    """The pipe stays resident across jobs, so PEFT state outlives the job that
    set it. ``disable_lora`` sets ``_disable_adapters`` on every layer and
    ``set_adapters`` writes only the scaling, so without an explicit
    ``enable_lora`` one no-style job silently switched every later job in the
    process to no style -- and it reads as working, because the trigger words
    are still prepended and the output does change.
    """
    t2i, pipe = _t2i(tmp_path, "turbo", _all_files())
    t2i._load_loras()
    chosen = _fitting("turbo")[0]
    t2i._apply_adapters(pipe, None, 1.0)
    assert pipe.disabled == 1
    t2i._apply_adapters(pipe, chosen, 0.9)
    assert pipe.enabled == 1, "the adapter was never re-enabled"
    assert pipe.applied == [([chosen], [0.9])]


def test_enabling_precedes_the_weights_it_is_meant_to_restore(tmp_path):
    """Order matters and only one order is right: enable_lora() takes no
    weights, so setting them first and enabling after would apply the previous
    scaling for one call."""
    t2i, pipe = _t2i(tmp_path, "turbo", _all_files())
    t2i._load_loras()
    t2i._apply_adapters(pipe, _fitting("turbo")[0], 0.5)
    assert pipe.calls.index("enable") < pipe.calls.index("set")
