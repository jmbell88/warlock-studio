"""``downloads.needed_rows``: what a locked feature is short of.

The question every "install what this needs" surface asks. Its two interesting
properties are that it resolves through the registry rather than a hand-written
list (so a feature naming a row that does not exist is a loud failure, not a
quiet "nothing missing"), and that it deliberately does *not* dedupe sizes --
the dedupe is a composition with ``plan_for``, and the whole point of that split
is the four SDXL recipes sharing one 7 GiB checkpoint.
"""

from __future__ import annotations

import pytest

from warlock import fetch
from warlock.service import downloads as svc_downloads
from warlock.service import sheets as svc_sheets
from warlock.service import sprites as svc_sprites
from warlock.service.errors import Invalid, NotFound


def _unlink_row(svc, row_key: str) -> None:
    """Make one registry row absent by deleting the first thing it is probed on."""
    entry = fetch.find(row_key)
    assert entry is not None
    config = svc.config
    root = config.t2i_model_root
    if entry.kind == "base":
        (fetch.base_model_dir(config, entry.spec) / "model_index.json").unlink()
    elif entry.kind == "lora":
        (root / "loras" / entry.spec.filename).unlink()
    elif entry.kind == "adapter":
        spec = entry.spec
        (root / spec.dir_name / spec.subfolder / spec.weight_name).unlink()
    elif entry.kind == "control":
        (root / entry.spec.dir_name / "config.json").unlink()
    else:  # pragma: no cover -- the fixture materialises nothing else
        raise AssertionError(entry.kind)
    assert not entry.is_present(config)


def test_a_materialised_host_needs_nothing(svc):
    assert svc_downloads.needed_rows(svc, svc_sprites.SPRITE_ROWS) == []
    assert svc_downloads.needed_rows(svc, svc_sheets.PIXEL_SHEET_ROWS) == []
    assert svc_downloads.needed_keys(svc, svc_sprites.SPRITE_ROWS) == ()
    assert svc_downloads.needed_gib(svc, svc_sprites.SPRITE_ROWS) == 0.0


def test_exactly_the_absent_rows_come_back(svc):
    _unlink_row(svc, "lora:pixelxl")
    _unlink_row(svc, "control:canny")

    rows = svc_downloads.needed_rows(svc, svc_sprites.SPRITE_ROWS)
    assert [r["row_key"] for r in rows] == ["control:canny", "lora:pixelxl"]
    # rows()'s shape, so a pane can draw one without a second code path.
    assert all(r["present"] is False and r["downloadable"] for r in rows)
    assert all(r["label"] and r["size_gib"] > 0 for r in rows)


def test_an_unknown_row_raises_rather_than_reporting_nothing_missing(svc):
    with pytest.raises(NotFound):
        svc_downloads.needed_rows(svc, ("base:no-such-model",))


def test_the_size_is_deduped_only_when_composed(svc):
    """The split the docstring promises, measured on the shared checkpoint.

    Four recipes over one ``sdxl-base-1.0`` directory: summing the per-row
    figures counts 7 GiB four times, and ``needed_gib`` -- which composes with
    ``plan_for`` -- counts it once.
    """
    shared = ("base:sdxl", "base:sdxl_cfg", "base:pixel", "base:lightning")
    for row_key in shared:
        entry = fetch.find(row_key)
        assert entry is not None
    (fetch.base_model_dir(svc.config, fetch.find("base:sdxl").spec) / "model_index.json").unlink()
    for name in ("Hyper-SDXL-4steps-lora.safetensors", "lcm-lora-sdxl.safetensors",
                 "sdxl_lightning_4step_lora.safetensors"):
        (svc.config.t2i_model_root / "loras" / name).unlink()

    rows = svc_downloads.needed_rows(svc, shared)
    assert [r["row_key"] for r in rows] == list(shared)
    naive = sum(r["size_gib"] for r in rows)
    deduped = svc_downloads.needed_gib(svc, shared)
    # 7 + 0.8 + 7 + 7 + 0.4 + 7 + 0.4 counted per row, against one 7 plus the
    # three small adapters once each.
    assert naive > 25.0
    assert deduped == pytest.approx(7.0 + 0.8 + 0.4 + 0.4)


# --- the refusals carry the rows --------------------------------------------


def test_a_sprite_refusal_names_every_row_the_feature_is_short_of(svc):
    """Not merely the one that tripped: the button installs once."""
    _unlink_row(svc, "control:canny")
    _unlink_row(svc, "adapter:plus")

    with pytest.raises(Invalid) as caught:
        svc_sprites._check_weights(svc)
    assert caught.value.rows == ("adapter:plus", "control:canny")


def test_a_sprite_refusal_about_the_base_carries_the_feature_not_the_base(svc):
    _unlink_row(svc, f"base:{svc_sprites.SPRITE_BASE_MODEL}")
    _unlink_row(svc, "lora:pixelxl")

    with pytest.raises(Invalid) as caught:
        svc_sprites._check_weights(svc)
    assert caught.value.rows == (
        f"base:{svc_sprites.SPRITE_BASE_MODEL}",
        "lora:pixelxl",
    )


def test_a_text_refusal_carries_only_its_own_checkpoint(svc):
    """The default ``rows`` for the door that stands in front of one model."""
    from warlock import models
    from warlock.service import validation

    key = models.DEFAULT_BASE_MODEL
    _unlink_row(svc, f"base:{key}")
    with pytest.raises(Invalid) as caught:
        validation.check_base_model_weights(svc, models.BASE_MODELS[key])
    assert caught.value.rows == (f"base:{key}",)


def test_an_optional_selection_refusal_carries_its_row(svc):
    from warlock.service import validation

    _unlink_row(svc, "lora:pixelxl")
    with pytest.raises(Invalid) as caught:
        validation.check_weights(svc, "text", {"style_lora": "pixelxl"})
    assert caught.value.rows == ("lora:pixelxl",)


def test_the_exported_row_tuples_are_all_real_registry_rows():
    """The drift guard: a constant renamed in ``models.py`` must not leave a
    feature quoting a row key nothing resolves."""
    for row_key in (*svc_sprites.SPRITE_ROWS, *svc_sheets.PIXEL_SHEET_ROWS):
        assert fetch.find(row_key) is not None, row_key


# --- rows() carries the fit verdict -----------------------------------------


def test_rows_are_silent_about_vram_when_no_plan_is_resolved(svc):
    """The ``svc`` fixture has ``vram_plan is None``, which is also every
    headless caller. The key is absent rather than None, so a pane's ``.get``
    reads one way."""
    assert svc.vram_plan is None
    assert all("vram" not in row for row in svc_downloads.rows(svc))


def test_rows_carry_a_fit_verdict_for_base_models_once_a_plan_exists(svc):
    from warlock import vram

    svc.vram_plan = vram.plan(exclusive=False, total_gib=24.0)
    rows = {r["row_key"]: r for r in svc_downloads.rows(svc)}
    # 22.5 GiB of budget against 7 + 16 resident: it loads, and every 3D job
    # pays a trellis restart for it.
    assert rows["base:sdxl_cfg"]["vram"] == vram.FIT_TIGHT
    # Offloaded, so nothing is ever resident beside it.
    assert rows["base:flux_klein"]["vram"] == vram.FIT_OK
    # Only base models have a device footprint.
    assert "vram" not in rows["lora:pixelxl"]
    assert "vram" not in rows["control:canny"]
