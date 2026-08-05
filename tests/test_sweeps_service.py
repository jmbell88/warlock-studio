"""Planning and launching a sweep: expansion, tier routing, grouping order,
all-or-nothing admission, and what a unit row does (and does not) carry."""

from __future__ import annotations

import pytest

from warlock.service import jobs as svc_jobs
from warlock.service import sweeps as svc_sweeps
from warlock.service.errors import Invalid, NotFound
from warlock.service.sweeps import Axis, SweepPlan


def _plan(**kwargs) -> SweepPlan:
    base = {"label": "lora", "prompt": "a wooden chest", "seeds": (1, 2)}
    base.update(kwargs)
    return SweepPlan(**base)


def test_expansion_is_baseline_plus_every_differing_axis_value_per_seed():
    plan = _plan(
        base={"lora_weight": 0.9},
        axes=(Axis("lora_weight", (0.6, 0.9, 1.2)),),
    )
    labels = [svc_sweeps.unit_label(u) for u in svc_sweeps.expand(plan)]
    # 0.9 equals the base, so it is skipped rather than planned twice.
    assert labels == [
        "baseline s1",
        "baseline s2",
        "lora_weight=0.6 s1",
        "lora_weight=0.6 s2",
        "lora_weight=1.2 s1",
        "lora_weight=1.2 s2",
    ]


def test_explicit_vectors_are_units_too():
    plan = _plan(
        seeds=(1,),
        vectors=({"label": "hi-res soft", "lora_weight": 1.2, "trellis_tex_res": 2048},),
    )
    units = svc_sweeps.expand(plan)
    assert [svc_sweeps.unit_label(u) for u in units] == ["baseline s1", "hi-res soft s1"]
    assert units[1].overrides == {"lora_weight": 1.2, "trellis_tex_res": 2048}


def test_units_are_grouped_by_server_config_with_the_base_group_first():
    """Best-effort restart minimisation. Correctness is the worker's business;
    this is only about how many times trellis-server is restarted."""
    plan = _plan(
        seeds=(1,),
        axes=(
            Axis("trellis_tex_res", (2048,)),
            Axis("lora_weight", (0.6,)),
            Axis("trellis_band", (8,)),
        ),
    )
    labels = [svc_sweeps.unit_label(u) for u in svc_sweeps.expand(plan)]
    assert labels == [
        # The base server config, in planned order.
        "baseline s1",
        "lora_weight=0.6 s1",
        # Then one group per differing server config.
        "trellis_tex_res=2048 s1",
        "trellis_band=8 s1",
    ]


def test_each_param_is_routed_to_the_tier_create_job_expects():
    plan = _plan(
        seeds=(7,),
        base={"genre": "fantasy", "lora_weight": 0.9, "trellis_band": 8},
    )
    kwargs = svc_sweeps.unit_kwargs(plan, svc_sweeps.expand(plan)[0])
    assert kwargs["guidance_fields"] == {"genre": "fantasy"}
    assert kwargs["lora_weight"] == 0.9
    assert kwargs["trellis_band"] == 8
    assert kwargs["seed"] == 7
    assert kwargs["output"] == "model"


def test_a_reference_stage_sweep_submits_references():
    plan = _plan(seeds=(1,), stage="reference")
    kwargs = svc_sweeps.unit_kwargs(plan, svc_sweeps.expand(plan)[0])
    assert kwargs["output"] == "reference"


def test_an_unknown_param_is_refused_rather_than_dropped():
    plan = _plan(seeds=(1,), base={"nonsense": 1})
    with pytest.raises(Invalid):
        svc_sweeps.unit_kwargs(plan, svc_sweeps.expand(plan)[0])


# --- admission ---------------------------------------------------------------


def test_one_bad_unit_refuses_the_whole_sweep_and_writes_nothing(svc):
    plan = _plan(
        seeds=(1,),
        axes=(Axis("trellis_band", (8, 999)),),
    )
    with pytest.raises(Invalid) as exc:
        svc_sweeps.create_sweep(svc, plan)
    # Named, because "one of your units is bad" is not actionable.
    assert "trellis_band=999 s1" in str(exc.value)
    assert svc.store.list_sweeps() == []
    assert svc.store.list(50) == []


def test_a_sweep_with_no_differing_values_is_refused(svc):
    plan = _plan(seeds=(), axes=())
    with pytest.raises(Invalid):
        svc_sweeps.create_sweep(svc, plan)


def test_a_sweep_larger_than_the_cap_is_refused(svc):
    plan = _plan(
        seeds=tuple(range(9)),
        axes=(Axis("lora_weight", tuple(0.1 * i for i in range(1, 9))),),
    )
    with pytest.raises(Invalid, match="the limit is"):
        svc_sweeps.create_sweep(svc, plan)


def test_a_launched_sweep_queues_one_job_per_unit_carrying_its_columns(svc):
    plan = _plan(seeds=(1,), axes=(Axis("trellis_band", (8,)),))
    result = svc_sweeps.create_sweep(svc, plan)

    assert result["units"] == 2
    units = svc.store.sweep_jobs(result["id"])
    assert [u["sweep_unit"] for u in units] == ["baseline s1", "trellis_band=8 s1"]
    assert all(u["sweep_id"] == result["id"] for u in units)
    assert all(u["status"] == "queued" for u in units)
    # The overlaid value reached params through create_job's ordinary door.
    assert "trellis_band" not in units[0]["params"]
    assert units[1]["params"]["trellis_band"] == 8

    listed = svc_sweeps.list_sweeps(svc)
    assert (listed[0]["units"], listed[0]["todo"]) == (2, 2)
    assert listed[0]["spec"]["axes"] == [{"param": "trellis_band", "values": [8]}]


def test_a_reroll_of_a_sweep_unit_leaves_the_sweep(svc):
    plan = _plan(seeds=(1,), axes=(Axis("lora_weight", (0.6,)),))
    result = svc_sweeps.create_sweep(svc, plan)
    unit = svc.store.sweep_jobs(result["id"])[1]
    svc.store.set_status(unit["id"], "done")

    rerun = svc.store.get(svc_jobs.rerun_job(svc, unit["id"], mode="reroll")["id"])
    assert rerun["sweep_id"] is None
    assert rerun["sweep_unit"] == ""


def test_deleting_a_sweep_removes_its_jobs_and_keeps_its_verdicts(svc):
    from warlock.service import verdicts as svc_verdicts

    plan = _plan(seeds=(1,), axes=(Axis("lora_weight", (0.6,)),))
    result = svc_sweeps.create_sweep(svc, plan)
    unit = svc.store.sweep_jobs(result["id"])[0]
    svc.store.set_status(unit["id"], "done")
    svc_verdicts.record_verdict(svc, unit["id"], verdict="accept")

    svc_sweeps.delete_sweep(svc, result["id"])

    assert svc_sweeps.list_sweeps(svc) == []
    assert svc.store.get(unit["id"]) is None
    # The whole point of snapshotting the vector onto the verdict row.
    assert len(svc.store.latest_verdicts()) == 1

    with pytest.raises(NotFound):
        svc_sweeps.delete_sweep(svc, result["id"])


def test_sweep_units_are_hidden_from_the_library(svc):
    from warlock.studio.state import Filters

    filters = Filters()
    assert filters.matches({"id": "a", "status": "done"})
    assert not filters.matches({"id": "b", "status": "done", "sweep_id": "abc"})
