"""Wave 2 of the UX review: the refusal knows how to fix itself, and the
model store stops being invisible.

Every part of this existed as *data* already -- ``ServiceError.rows`` was
written by four service modules and read by nothing outside the tests,
``downloads.needed_gib`` was written so "a pane may put a figure on a button"
and had no production caller, and every size in the Models list was the
registry's declared figure rather than the disk's.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from warlock import doctor
from warlock.service import downloads as svc_downloads
from warlock.studio import widgets
from warlock.studio.panes import model_gate
from warlock.studio.state import AppState

# --- the refusal's install offer --------------------------------------------


def _state_with(field_name: str, message: str, rows=(), gib=0.0) -> AppState:
    state = AppState()
    state.note_field_error(field_name, message, rows, gib)
    return state


def test_a_refusal_carries_its_rows_and_its_deduped_figure():
    state = _state_with("base_model", "no weights", ("base:sdxl", "lora:pixelxl"), 7.2)
    assert state.field_errors["base_model"] == "no weights"
    assert state.field_error_rows["base_model"] == ("base:sdxl", "lora:pixelxl")
    assert state.field_error_gib["base_model"] == 7.2


def test_a_refusal_with_nothing_to_install_displaces_an_earlier_one():
    """Otherwise the ring for "that seed is out of range" keeps the Install
    button the previous missing-weights refusal on the same control left."""
    state = _state_with("base_model", "no weights", ("base:sdxl",), 7.0)
    state.note_field_error("base_model", "that is not a number")
    assert "base_model" not in state.field_error_rows
    assert "base_model" not in state.field_error_gib


def test_editing_the_control_forgets_the_offer_with_the_ring():
    state = _state_with("base_model", "no weights", ("base:sdxl",), 7.0)
    state.clear_field_error("base_model")
    assert not state.field_error_rows
    assert not state.field_error_gib


def test_a_new_submit_clears_every_offer():
    state = _state_with("base_model", "no weights", ("base:sdxl",), 7.0)
    state.clear_field_errors()
    assert not state.field_error_rows
    assert not state.field_error_gib


def test_the_offer_draws_nothing_for_a_refusal_that_is_not_about_weights():
    ctx = SimpleNamespace(state=_state_with("seed", "not a number"))
    assert model_gate.install_offer(ctx, "seed") is False


def test_the_offer_draws_nothing_for_a_control_with_no_refusal():
    ctx = SimpleNamespace(state=AppState())
    assert model_gate.install_offer(ctx, "base_model") is False


# --- the copyable command ----------------------------------------------------


def test_the_command_is_lifted_out_of_a_refusal():
    from warlock.service.validation import install_remedy

    text = install_remedy("SDXL 1.0", "uvx hf download stabilityai/x --local-dir y")
    assert widgets.copyable_command(text) == (
        "uvx hf download stabilityai/x --local-dir y"
    )


def test_a_sentence_with_no_command_offers_nothing_to_copy():
    assert widgets.copyable_command("That seed is out of range.") is None
    assert widgets.copyable_command("") is None


def test_doctors_hint_is_copyable_too(tmp_path):
    """The same shape, because ``_gguf_check`` indents its command the way
    ``install_remedy`` does -- which is what lets one reader serve both."""
    from warlock.config import Config

    config = Config(
        data_dir=tmp_path / "data", trellis_models_dir=tmp_path / "models"
    )
    detail = doctor._gguf_check(config).detail
    assert widgets.copyable_command(detail) == doctor.trellis_gguf_hint(config)


# --- the resolved GGUF remedy -------------------------------------------------


def test_the_gguf_remedy_is_never_a_relative_path(tmp_path):
    """``fetch``'s module docstring already owns this rule: a literal
    ``models/...`` in a download string is the README's spelling of the
    default root, never where a fetch should actually go."""
    from warlock.config import Config

    config = Config(
        data_dir=tmp_path / "data", trellis_models_dir=tmp_path / "models"
    )
    hint = doctor.trellis_gguf_hint(config)
    assert "--local-dir models/trellis2-gguf" not in hint
    assert f'--local-dir "{config.trellis_models_dir}"' in hint
    assert doctor.TRELLIS_GGUF_REVISION in hint


# --- rate and ETA -------------------------------------------------------------


@pytest.mark.parametrize(
    ("rate", "remaining", "expected"),
    [
        (0.0, 1024**3, ""),
        (1024, 1024**3, ""),  # a stalled fetch says nothing rather than "~11d"
        (10 * 1024**2, 0.0, " at 10 MB/s"),
        (10 * 1024**2, 100 * 1024**2, " at 10 MB/s, ~10s left"),
        (10 * 1024**2, 6 * 1024**3, " at 10 MB/s, ~10m left"),
        (1024**2, 8 * 1024**3, " at 1 MB/s, ~2h 16m left"),
    ],
)
def test_the_progress_label_carries_a_pace(rate, remaining, expected):
    from warlock.pipelines.fetch_worker import _pace

    assert _pace(rate, remaining) == expected


# --- what a row now says -------------------------------------------------------


def test_a_row_names_the_repositories_it_comes_from(svc):
    by_key = {r["row_key"]: r for r in svc_downloads.rows(svc)}
    row = by_key["base:sdxl"]
    assert row.get("repos")
    assert all("/" in repo for repo in row["repos"])


def test_an_adapter_row_carries_its_trigger_words(svc):
    by_key = {r["row_key"]: r for r in svc_downloads.rows(svc)}
    row = by_key["lora:pixelxl"]
    from warlock import models

    assert row.get("trigger") == models.STYLE_LORAS["pixelxl"].trigger


def test_a_base_row_carries_the_vram_number_not_only_the_verdict(svc):
    by_key = {r["row_key"]: r for r in svc_downloads.rows(svc)}
    from warlock import models

    row = by_key["base:sdxl"]
    assert row["vram_gib"] == models.BASE_MODELS["sdxl"].vram_gib


# --- the staging sweep ----------------------------------------------------------


def test_the_sweep_removes_what_a_cancelled_fetch_stranded(svc):
    root = svc.config.t2i_model_root
    root.mkdir(parents=True, exist_ok=True)
    stranded = root / "sdxl-base-1.0.fetch.part"
    stranded.mkdir()
    (stranded / "half.safetensors").write_bytes(b"x" * 16)
    keep = root / "kept-by-the-sweep"
    keep.mkdir(exist_ok=True)

    removed = svc_downloads.sweep_staging(svc)

    assert stranded.name in removed
    assert not stranded.exists()
    assert keep.exists()


def test_the_sweep_is_silent_when_there_is_nothing_to_reclaim(svc):
    svc.config.t2i_model_root.mkdir(parents=True, exist_ok=True)
    assert svc_downloads.sweep_staging(svc) == []


def test_the_sweep_never_raises_on_a_root_that_is_not_there(svc):
    assert svc_downloads.sweep_staging(svc) == []


# --- measured disk usage ----------------------------------------------------------


def test_disk_usage_measures_the_store_rather_than_declaring_it(svc):
    """A delta rather than an absolute: the fixture's home is not guaranteed
    empty, and what this pins is that the walk *measures* -- the registry's
    declared ``size_gib`` is the number it exists to stop being the only one."""
    root = svc.config.t2i_model_root
    root.mkdir(parents=True, exist_ok=True)
    before = svc_downloads.disk_usage(svc)

    added = root / "measured-by-the-walk"
    added.mkdir()
    (added / "unet.safetensors").write_bytes(b"x" * 4096)
    (added / "config.json").write_bytes(b"{}")

    found = svc_downloads.disk_usage(svc)

    assert found["files"] - before["files"] == 2
    assert found["bytes"] - before["bytes"] == 4096 + 2
    assert found["root"] == str(root)


def test_disk_usage_of_a_store_that_is_not_there_is_zero_rather_than_an_error(svc):
    import shutil

    shutil.rmtree(svc.config.t2i_model_root, ignore_errors=True)
    found = svc_downloads.disk_usage(svc)
    assert found["files"] == 0
    assert found["bytes"] == 0
