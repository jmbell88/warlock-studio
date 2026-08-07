"""The cost model, the plan, and the two guarantees they exist to enforce.

All of it is arithmetic over synthetic DeviceMemory values, so it runs on a
machine with no GPU -- which is the point: the gate that decides whether a
12 GB card may start a 16 GB reconstruction must not itself need a 12 GB card
to test.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from warlock import vram
from warlock.config import Config

SRC = Path(__file__).resolve().parents[1] / "src" / "warlock"


# -- the plan -----------------------------------------------------------------


def test_a_32_gib_card_still_coexists():
    """The no-behaviour-change assertion. This is the machine it runs on."""
    plan = vram.plan(exclusive=None, device=vram.DeviceMemory(32.0, 32.0))
    assert plan.exclusive is False
    assert plan.budget_gib == pytest.approx(32.0 - vram.HEADROOM_GIB)
    assert "coexist" in plan.reason


@pytest.mark.parametrize("total", [8.0, 12.0, 16.0, 24.0])
def test_a_card_too_small_for_both_auto_selects_exclusive(total):
    plan = vram.plan(exclusive=None, device=vram.DeviceMemory(total, total))
    assert plan.exclusive is True
    assert plan.budget_gib == pytest.approx(total - vram.HEADROOM_GIB)


def test_an_explicit_choice_is_never_overridden():
    small = vram.DeviceMemory(8.0, 8.0)
    assert vram.plan(exclusive=False, device=small).exclusive is False
    big = vram.DeviceMemory(32.0, 32.0)
    assert vram.plan(exclusive=True, device=big).exclusive is True
    assert "explicitly" in vram.plan(exclusive=True, device=big).reason


def test_no_device_means_no_enforcement():
    plan = vram.plan(exclusive=None, device=None)
    assert plan.enforced is False
    assert plan.fits(999.0)


def test_an_explicit_total_stands_in_for_a_card():
    plan = vram.plan(exclusive=None, total_gib=12.0, device=vram.DeviceMemory(32.0, 32.0))
    assert plan.total_gib == 12.0
    assert plan.exclusive is True


def test_an_explicit_budget_wins_over_the_headroom_calculation():
    plan = vram.plan(exclusive=None, budget_gib=30.0, device=vram.DeviceMemory(12.0, 12.0))
    assert plan.budget_gib == 30.0
    assert plan.exclusive is False


# -- the cost model -----------------------------------------------------------


def test_coexist_sums_the_stages_and_exclusive_takes_the_larger():
    params = {"resolution": 1024}
    assert vram.estimate("text", "model", params, exclusive=False) == pytest.approx(
        vram.SDXL_GIB + vram.TRELLIS_GIB
    )
    assert vram.estimate("text", "model", params, exclusive=True) == pytest.approx(
        vram.TRELLIS_GIB
    )


def test_conditioning_costs_what_text2image_says_it_costs():
    base = vram.estimate("text", "reference", {}, exclusive=True)
    both = vram.estimate(
        "text", "reference", {"control": "depth", "ip_adapter": "plus"}, exclusive=True
    )
    assert both - base == pytest.approx(vram.CONTROLNET_GIB + vram.IP_ENCODER_GIB)


def test_resolution_scales_the_reconstruction():
    at_1024 = vram.estimate("image", "model", {"resolution": 1024}, exclusive=True)
    at_1536 = vram.estimate("image", "model", {"resolution": 1536}, exclusive=True)
    assert at_1536 > at_1024
    # An unknown or absent resolution falls back to the 1024 baseline rather
    # than to zero -- a missing key must never make a job look free.
    assert vram.estimate("image", "model", {}, exclusive=True) == at_1024
    assert vram.estimate("image", "model", {"resolution": "junk"}, exclusive=True) == at_1024


def test_a_pixel_sheet_restyle_costs_sdxl_plus_a_controlnet():
    """It is an img2img generation, not a Blender render: the same resident
    pipe a text job wants, plus a ControlNet, and never trellis -- the mesh was
    reconstructed long before."""
    exclusive = vram.estimate("pixel_sheet", "model", {}, exclusive=True)
    coexist = vram.estimate("pixel_sheet", "model", {}, exclusive=False)
    assert exclusive == pytest.approx(vram.SDXL_GIB + vram.CONTROLNET_GIB)
    # Under coexist a warm trellis is still holding its memory, exactly as the
    # reference stage accounts for.
    assert coexist == pytest.approx(exclusive + vram.TRELLIS_GIB)


def test_an_image_model_is_charged_its_own_footprint():
    """Not every checkpoint is 7 GB any more. flux_klein is offloaded and
    records 10.0, and charging it SDXL's number would admit a job the card
    cannot hold."""
    from warlock import models

    spec = models.BASE_MODELS["flux_klein"]
    assert spec.vram_gib != vram.SDXL_GIB
    at_flux = vram.estimate("text", "reference", {"base_model": "flux_klein"}, exclusive=True)
    assert at_flux == pytest.approx(spec.vram_gib)
    at_turbo = vram.estimate("text", "reference", {"base_model": "turbo"}, exclusive=True)
    assert at_turbo == pytest.approx(vram.SDXL_GIB)


@pytest.mark.parametrize("params", [{}, {"base_model": ""}, {"base_model": "gone"}])
def test_an_unknown_base_model_falls_back_to_the_sdxl_figure(params):
    # Params outlive the registry -- the same tolerance queue._generate applies
    # -- and a missing key must never make a job look free.
    assert vram.estimate("text", "reference", params, exclusive=True) == pytest.approx(
        vram.SDXL_GIB
    )


def test_a_rig_job_costs_no_vram():
    assert vram.estimate("rig", "model", {}, exclusive=False) == 0.0


def test_estimate_job_reads_a_store_row():
    job = {"kind": "image", "stage": "model", "params": {"resolution": 1024}}
    assert vram.estimate_job(job, exclusive=True) == pytest.approx(vram.TRELLIS_GIB)


# -- the refusal --------------------------------------------------------------


def test_a_small_card_cannot_afford_a_reconstruction():
    plan = vram.plan(exclusive=None, total_gib=8.0)
    need = vram.estimate("image", "model", {}, exclusive=plan.exclusive)
    assert not plan.fits(need)
    message = vram.shortfall_message(need, plan, {})
    assert "16.0 GiB" in message and "6.5 GiB" in message


def test_the_refusal_names_something_the_user_can_change():
    plan = vram.plan(exclusive=None, total_gib=24.0)
    params = {"control": "depth", "ip_adapter": "plus", "resolution": 1536}
    need = vram.estimate("text", "model", params, exclusive=plan.exclusive)
    message = vram.shortfall_message(need, plan, params)
    assert "ControlNet" in message
    assert "IP-Adapter" in message
    assert "resolution" in message
    assert message.endswith(".")


def test_the_exclusive_remedy_is_only_offered_when_it_is_available():
    already = vram.plan(exclusive=True, total_gib=8.0)
    assert "WARLOCK_VRAM_EXCLUSIVE" not in vram.shortfall_message(20.0, already, {})
    not_yet = vram.plan(exclusive=False, total_gib=8.0)
    assert "WARLOCK_VRAM_EXCLUSIVE=1" in vram.shortfall_message(20.0, not_yet, {})


# -- config wiring ------------------------------------------------------------


def test_the_env_flag_is_tri_state(monkeypatch):
    monkeypatch.delenv("WARLOCK_VRAM_EXCLUSIVE", raising=False)
    assert Config().vram_exclusive is None
    monkeypatch.setenv("WARLOCK_VRAM_EXCLUSIVE", "off")
    assert Config().vram_exclusive is False
    monkeypatch.setenv("WARLOCK_VRAM_EXCLUSIVE", "1")
    assert Config().vram_exclusive is True


def test_a_resolved_config_stops_claiming_an_env_var_nobody_set(monkeypatch):
    """``Runtime._resolve_vram`` writes a plain bool back onto the tri-state,
    so every later ``plan()`` -- every health poll, the doctor row -- saw a set
    value and reported the auto-selected mode as "set explicitly by
    WARLOCK_VRAM_EXCLUSIVE"."""
    from warlock import doctor

    monkeypatch.delenv("WARLOCK_VRAM_EXCLUSIVE", raising=False)
    config = Config(vram_total_gib=32.0)

    first = vram.plan(
        exclusive=config.vram_exclusive,
        total_gib=config.vram_total_gib,
        explicit=config.vram_exclusive_explicit,
    )
    assert first.explicit is False
    assert "auto-selected" in first.reason

    # What the runtime writes back.
    config.vram_exclusive = first.exclusive
    config.vram_budget_gib = first.budget_gib
    config.vram_exclusive_explicit = first.explicit

    assert "WARLOCK_VRAM_EXCLUSIVE" not in doctor._vram_check(config).detail


def test_an_explicit_choice_still_says_so_after_it_is_resolved(monkeypatch):
    from warlock import doctor

    monkeypatch.setenv("WARLOCK_VRAM_EXCLUSIVE", "1")
    config = Config(vram_total_gib=32.0)
    plan = vram.plan(
        exclusive=config.vram_exclusive,
        total_gib=config.vram_total_gib,
        explicit=config.vram_exclusive_explicit,
    )
    assert plan.explicit is True
    config.vram_exclusive, config.vram_exclusive_explicit = plan.exclusive, plan.explicit

    assert "WARLOCK_VRAM_EXCLUSIVE" in doctor._vram_check(config).detail


def test_the_total_and_budget_overrides_parse(monkeypatch):
    monkeypatch.setenv("WARLOCK_VRAM_TOTAL", "12")
    monkeypatch.setenv("WARLOCK_VRAM_BUDGET", "10.5")
    config = Config()
    assert config.vram_total_gib == 12.0
    assert config.vram_budget_gib == 10.5
    # Unparseable is unset, not a crash at import time on every later run.
    monkeypatch.setenv("WARLOCK_VRAM_TOTAL", "lots")
    assert Config().vram_total_gib is None


def test_a_reading_never_raises(monkeypatch):
    class Exploding:
        class cuda:
            @staticmethod
            def is_available():
                raise RuntimeError("driver fell over")

    monkeypatch.setitem(__import__("sys").modules, "torch", Exploding)
    assert vram.device_memory() is None


# -- the invariant that stops the next orphan ---------------------------------

_SPAWN = re.compile(r"subprocess\.(Popen|run)\s*\(")


def test_every_subprocess_spawn_is_in_the_kill_on_close_job():
    """Gap A was introduced silently. This is what stops the next one.

    A child that is not in the job object outlives a hard kill of the app --
    which for `import bpy` or gltfpack means a stray process, and for
    trellis-server meant a stale listener on port 17971 that the health poll
    could not tell from the server it had just spawned.
    """
    offenders = []
    for path in SRC.rglob("*.py"):
        if path.name == "winjob.py":
            continue  # the implementation itself
        lines = path.read_text(encoding="utf-8").splitlines()
        for i, line in enumerate(lines):
            if line.lstrip().startswith("#") or not _SPAWN.search(line):
                continue
            window = "\n".join(lines[i : i + 15])
            if "winjob.assign" not in window:
                offenders.append(f"{path.relative_to(SRC)}:{i + 1}")
    assert not offenders, (
        "these spawn a child outside the kill-on-close job; use winjob.run() "
        f"or call winjob.assign(proc.pid): {offenders}"
    )


def test_winjob_run_is_shaped_like_subprocess_run():
    import sys as _sys

    from warlock import winjob

    proc = winjob.run(
        [_sys.executable, "-c", "print('hi')"], capture_output=True, text=True
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == "hi"
    with pytest.raises(subprocess.TimeoutExpired):
        winjob.run([_sys.executable, "-c", "import time; time.sleep(30)"], timeout=1.0)


def test_armed_answers_without_raising():
    from warlock import winjob

    assert isinstance(winjob.armed(), bool)


def test_a_tile_costs_what_a_reference_costs():
    # Same pipe, same size, one sample -- the circular padding changes no
    # allocation. A stage the estimate did not know would fall through to the
    # mesh branch and price a trellis run that never happens.
    assert vram.estimate("text", "tile", {}, exclusive=True) == vram.estimate(
        "text", "reference", {}, exclusive=True
    )
    assert vram.estimate("text", "tile", {}, exclusive=False) == vram.estimate(
        "text", "reference", {}, exclusive=False
    )
