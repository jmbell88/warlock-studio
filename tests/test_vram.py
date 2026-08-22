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

from warlock import models, vram
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


def test_a_tile_sheet_costs_one_txt2img_pass_and_its_grid_guide():
    """Sixty-four tiles are a *slicing* of one generation, not sixty-four
    generations, so the grid is not a term in the peak. The ControlNet always
    is -- the guide is the whole mechanism. Never trellis: there is no mesh
    anywhere near it."""
    exclusive = vram.estimate("tile_sheet", "tilesheet", {}, exclusive=True)
    assert exclusive == pytest.approx(vram.SDXL_GIB + vram.CONTROLNET_GIB)
    assert vram.estimate(
        "tile_sheet", "tilesheet", {}, exclusive=False
    ) == pytest.approx(exclusive + vram.TRELLIS_GIB)
    # The grid genuinely does not move it.
    big = {"sheet": {"tile_w": 64, "tile_h": 64, "columns": 8, "rows": 8}}
    assert vram.estimate(
        "tile_sheet", "tilesheet", big, exclusive=True
    ) == pytest.approx(exclusive)


def test_a_tile_sheets_reference_encoder_is_gated_on_the_reference():
    """Unlike a sprite synthesis, this kind's IP-Adapter is optional -- so
    charging its encoder unconditionally would refuse the common prompt-only
    request on a card that fits it."""
    plain = vram.estimate("tile_sheet", "tilesheet", {}, exclusive=True)
    conditioned = vram.estimate(
        "tile_sheet", "tilesheet", {"ip_adapter": "plus"}, exclusive=True
    )
    assert conditioned == pytest.approx(plain + vram.IP_ENCODER_GIB)


def test_a_retexture_costs_one_img2img_pass_and_never_trellis():
    """Six views are six *sequential* passes through one resident pipe, so the
    view count is not a term in the peak -- and the two Blender halves are
    out-of-process and CPU-side, so they are not either."""
    exclusive = vram.estimate("retexture", "model", {}, exclusive=True)
    assert exclusive == pytest.approx(vram.SDXL_GIB)
    assert vram.estimate("retexture", "model", {}, exclusive=False) == pytest.approx(
        exclusive + vram.TRELLIS_GIB
    )
    with_control = vram.estimate("retexture", "model", {"control": "canny"}, exclusive=True)
    assert with_control == pytest.approx(exclusive + vram.CONTROLNET_GIB)


def test_a_retexture_has_no_ip_encoder_term_because_no_door_writes_one():
    """The branch that charged ``IP_ENCODER_GIB`` here was unreachable:
    ``service._jobs_rework.retexture_job`` is the only door that creates a
    ``retexture`` job and it neither accepts nor writes ``ip_adapter``, so the
    condition was always false and the estimate was already correct.

    Asserted from the door rather than only from the arithmetic: if a re-texture
    ever *does* grow an identity knob, this is what says the estimate has to
    grow with it.
    """
    import inspect

    from warlock.service import _jobs_rework

    assert "ip_adapter" not in inspect.getsource(_jobs_rework.retexture_job)
    plain = vram.estimate("retexture", "model", {}, exclusive=True)
    assert vram.estimate(
        "retexture", "model", {"ip_adapter": "plus"}, exclusive=True
    ) == pytest.approx(plain)


def test_a_retexture_is_priced_from_the_registry_not_from_sdxl():
    """It is exactly the job somebody points at an offloaded checkpoint, so the
    spec's own figure has to be the one charged."""
    from warlock import models

    spec = models.BASE_MODELS["flux_klein"]
    assert spec.vram_gib != vram.SDXL_GIB
    assert vram.estimate(
        "retexture", "model", {"base_model": "flux_klein"}, exclusive=True
    ) == pytest.approx(spec.vram_gib)


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


def test_the_dispatch_credit_reads_the_registry_when_torch_cannot_answer():
    """The other place a resident pipe is priced. When vram_gib() cannot
    measure, the credit for the to-be-freed pipe falls back to the resident
    spec's own declared footprint -- a flat SDXL_GIB is 3 GiB short of the
    offloaded klein entry's 10.0, which under-credits the headroom and refuses
    a job the card actually holds. A key the registry no longer carries keeps
    the SDXL figure, the tolerance the estimate above already applies."""
    from warlock import models
    from warlock.queue import _resident_t2i_gib

    klein = models.BASE_MODELS["flux_klein_distilled"]
    assert klein.vram_gib != vram.SDXL_GIB
    assert _resident_t2i_gib("flux_klein_distilled") == pytest.approx(klein.vram_gib)
    assert _resident_t2i_gib("turbo") == pytest.approx(
        models.BASE_MODELS["turbo"].vram_gib
    )
    assert _resident_t2i_gib("gone") == pytest.approx(vram.SDXL_GIB)
    assert _resident_t2i_gib(None) == pytest.approx(vram.SDXL_GIB)


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


def test_an_offloaded_base_is_priced_sequentially_whatever_the_flag_says():
    """The accounting half of the mandatory offload handoff.

    ``queue._needs_handoff`` stops trellis before an OFFLOAD checkpoint loads
    whether or not WARLOCK_VRAM_EXCLUSIVE is set, so the estimate has to agree
    -- otherwise the gate charges the sum of two stages that are never resident
    together and refuses the one job that is careful about VRAM.
    """
    params = {"base_model": "flux_klein"}
    assert vram.offloaded_base(params) is True
    assert vram.offloaded_base({"base_model": "sdxl"}) is False
    assert vram.offloaded_base({}) is False

    coexist = vram.estimate("text", "model", params, exclusive=False)
    sequential = vram.estimate("text", "model", params, exclusive=True)
    assert coexist == sequential
    # And it really is the max of the two stages, not their sum.
    assert coexist == max(
        models.BASE_MODELS["flux_klein"].vram_gib, vram.TRELLIS_GIB
    )


def test_a_resident_base_is_still_priced_by_the_flag():
    # The other half: nothing about SDXL's *residency* accounting moved.
    # ``sdxl`` is the Hyper-SD recipe, so it carries a required
    # step-distillation LoRA that is loaded unconditionally inside load() and
    # was charged nowhere until 2026-08-12 (MDL-17).
    params = {"base_model": "sdxl"}
    assert models.BASE_MODELS["sdxl"].base_lora
    assert vram.estimate("text", "model", params, exclusive=False) == (
        vram.SDXL_GIB + vram.BASE_LORA_GIB + vram.TRELLIS_GIB
    )
    assert vram.estimate("text", "model", params, exclusive=True) == vram.TRELLIS_GIB


def test_the_adapters_a_job_will_actually_load_are_priced(tmp_path):
    """MDL-17: ``vram_gib`` is the checkpoint alone, and the adapters loaded
    beside it are real device memory.

    A style adapter is charged only when the job selects one, because only then
    is it attached -- optional adapters stopped loading eagerly in the same pass
    (MDL-07), so summing every fitting one would now over-charge as badly as
    excluding them under-charged.
    """
    # A base with no required LoRA and no style selected: the checkpoint alone.
    plain = {"base_model": "sdxl_cfg"}
    assert models.BASE_MODELS["sdxl_cfg"].base_lora is None
    bare = vram.estimate("text", "reference", plain, exclusive=True)
    assert bare == models.BASE_MODELS["sdxl_cfg"].vram_gib

    # Select a style that fits, and its weights are charged.
    style = next(
        lo
        for lo in models.STYLE_LORAS.values()
        if models.lora_fits(models.BASE_MODELS["sdxl_cfg"], lo)
    )
    styled = vram.estimate(
        "text", "reference", {**plain, "style_lora": style.key}, exclusive=True
    )
    assert styled == pytest.approx(bare + sum(f.size_gib for f in style.fetch))

    # One fitted to another architecture never loads, so it is never charged.
    foreign = next(
        (
            lo
            for lo in models.STYLE_LORAS.values()
            if not models.lora_fits(models.BASE_MODELS["sdxl_cfg"], lo)
        ),
        None,
    )
    if foreign is not None:
        assert (
            vram.estimate(
                "text", "reference", {**plain, "style_lora": foreign.key}, exclusive=True
            )
            == bare
        )


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


# --- sprite synthesis --------------------------------------------------------


def test_a_sprite_synthesis_is_priced_and_never_silently_zero():
    """A kind with no branch falls through to ``return 0.0`` and the admission
    gate becomes a no-op that refuses nothing -- which is the failure mode this
    guards, not the exact number."""
    exclusive = vram.estimate("sprite_synthesis", "model", {"base_model": "sdxl_cfg"},
                              exclusive=True)
    coexist = vram.estimate("sprite_synthesis", "model", {"base_model": "sdxl_cfg"},
                            exclusive=False)
    assert exclusive > 0.0
    # Both adapters ride every pass, unconditionally: the guide *is* the
    # ControlNet and the identity *is* the IP-Adapter.
    assert exclusive >= vram.CONTROLNET_GIB + vram.IP_ENCODER_GIB
    # Coexist adds a warm trellis holding its own memory, exactly as the other
    # image-model kinds account for.
    assert coexist == pytest.approx(exclusive + vram.TRELLIS_GIB)


def test_a_sprite_synthesis_is_priced_from_its_own_checkpoint():
    offloaded = next(
        key for key, spec in models.BASE_MODELS.items()
        if spec.residency == models.OFFLOAD
    )
    priced = vram.estimate("sprite_synthesis", "model", {"base_model": offloaded},
                           exclusive=True)
    plain = vram.estimate("sprite_synthesis", "model", {"base_model": "sdxl_cfg"},
                          exclusive=True)
    assert priced != plain


# -- does a checkpoint fit ----------------------------------------------------


def _base(key: str):
    return models.BASE_MODELS[key]


def test_a_big_card_holds_an_sdxl_pipe_beside_trellis():
    plan = vram.plan(exclusive=None, total_gib=32.0)
    assert not plan.exclusive
    assert vram.fits(plan, _base("sdxl_cfg")) == vram.FIT_OK


def test_a_card_that_auto_selected_exclusive_fits_everything_it_can_hold():
    """Handing off is what makes a 12 GB card usable at all, so the badge must
    not call an SDXL pipe tight there: nothing is sitting beside it."""
    plan = vram.plan(exclusive=None, total_gib=12.0)
    assert plan.exclusive
    assert vram.fits(plan, _base("sdxl_cfg")) == vram.FIT_OK


def test_a_checkpoint_larger_than_the_whole_budget_does_not_fit():
    plan = vram.plan(exclusive=None, total_gib=6.0)
    assert vram.fits(plan, _base("sdxl_cfg")) == vram.FIT_NO
    assert vram.fits(plan, _base("flux_klein")) == vram.FIT_NO


def test_coexist_on_a_24_gb_card_is_tight_rather_than_refused():
    """22.5 GiB of budget against 7 + 16: it loads, and every 3D job pays a
    trellis restart for it. Explicit, because auto would have chosen exclusive
    at this size and that is a different answer."""
    plan = vram.plan(exclusive=False, total_gib=24.0)
    assert not plan.exclusive
    assert vram.fits(plan, _base("sdxl_cfg")) == vram.FIT_TIGHT


def test_an_unknown_budget_is_not_a_shortfall():
    plan = vram.plan(exclusive=None, total_gib=None)
    assert plan.budget_gib is None
    assert vram.fits(plan, _base("sdxl_cfg")) == vram.FIT_OK


def test_an_offloaded_checkpoint_is_never_tight():
    """It hands off whatever the flag says, so nothing is ever resident beside
    it -- the same rule ``estimate`` prices it by."""
    plan = vram.plan(exclusive=False, total_gib=32.0)
    klein = _base("flux_klein")
    assert klein.residency == models.OFFLOAD
    assert vram.fits(plan, klein) == vram.FIT_OK
    # And the ordinary pipe on the same card is not, so the assertion above is
    # about residency rather than about the card being large.
    assert vram.fits(plan, _base("sdxl_cfg")) == vram.FIT_OK
    assert vram.fits(vram.plan(exclusive=False, total_gib=24.0), klein) == vram.FIT_OK


def test_the_recommendation_is_the_best_thing_that_actually_fits():
    big = vram.plan(exclusive=None, total_gib=32.0)
    assert vram.recommended_base(big) == "sdxl_cfg"


def test_the_recommendation_falls_back_rather_than_returning_nothing():
    """A card that can hold no checkpoint has a problem no picker solves; the
    caller draws a badge either way and must not be handed None."""
    tiny = vram.plan(exclusive=None, total_gib=2.0)
    assert all(
        vram.fits(tiny, models.BASE_MODELS[k]) == vram.FIT_NO
        for k in vram.RECOMMENDED_BASES
    )
    assert vram.recommended_base(tiny) == models.DEFAULT_BASE_MODEL


def test_every_recommended_key_is_a_registry_key():
    for key in vram.RECOMMENDED_BASES:
        assert key in models.BASE_MODELS, key


# -- the base-model remedy ----------------------------------------------------


def test_no_plan_means_no_base_model_remedy():
    """``dispatch_shortfall_message``'s case: it refuses on free memory, where
    the budget the fit is computed against is not what ran out."""
    params = {"base_model": "flux_klein"}
    assert "smaller base model" not in vram.remedies(params, exclusive=False)


def test_a_plan_offers_the_cheaper_checkpoint_before_the_environment_variable():
    """10 GiB of klein on a card with room for 7 and no more. The order is the
    point: a click is offered ahead of a restart."""
    plan = vram.plan(exclusive=False, total_gib=12.0)
    params = {"base_model": "flux_klein"}
    text = vram.remedies(params, exclusive=plan.exclusive, plan_=plan)
    assert "smaller base model" in text
    assert text.index("smaller base model") < text.index("WARLOCK_VRAM_EXCLUSIVE")


def test_the_cheaper_remedy_is_silent_when_the_pick_is_already_the_cheapest():
    plan = vram.plan(exclusive=False, total_gib=32.0)
    assert "smaller base model" not in vram.remedies(
        {"base_model": "sdxl_cfg"}, exclusive=False, plan_=plan
    )


def test_the_cheaper_remedy_is_silent_when_no_model_was_named():
    plan = vram.plan(exclusive=False, total_gib=12.0)
    assert "smaller base model" not in vram.remedies({}, exclusive=False, plan_=plan)


def test_the_submit_refusal_carries_the_plan_and_the_dispatch_one_does_not():
    plan = vram.plan(exclusive=False, total_gib=12.0)
    params = {"base_model": "flux_klein"}
    assert "smaller base model" in vram.shortfall_message(30.0, plan, params)
    assert "smaller base model" not in vram.dispatch_shortfall_message(
        30.0, 4.0, 3.0, params, exclusive=False
    )
