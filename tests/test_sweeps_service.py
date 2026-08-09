"""Planning and launching a sweep: expansion, tier routing, grouping order,
all-or-nothing admission, and what a unit row does (and does not) carry."""

from __future__ import annotations

import pytest

from warlock.service import jobs as svc_jobs
from warlock.service import sweeps as svc_sweeps
from warlock.service.errors import Invalid, NotFound
from warlock.service.sweeps import Axis, SweepPlan


def _plan(**kwargs) -> SweepPlan:
    fields = {"label": "lora", "prompt": "a wooden chest", "seeds": (1, 2)}
    fields.update(kwargs)
    # A style LoRA in the base unless the caller set one, because most of these
    # sweep ``lora_weight`` and ``guidance.normalize`` drops a weight with no
    # LoRA to weight -- so without it every unit would normalize to the same
    # job, which admission now refuses (and which was previously N identical
    # GPU runs).
    fields["base"] = {"style_lora": "render3d", **dict(fields.get("base") or {})}
    return SweepPlan(**fields)


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


def test_admission_refuses_a_named_tier_while_gltfpack_is_absent(svc):
    # Admission validated only the profile *name*, so a sweep could finish
    # wearing profile="standard" over meshes the missing binary never touched
    # -- and the verdict corpus would credit the tier. Refused naming the unit,
    # like every other admission failure.
    plan = _plan(seeds=(1,), base={"profile": "standard"})
    with pytest.raises(Invalid, match="baseline s1.*gltfpack"):
        svc_sweeps.create_sweep(svc, plan)


def test_an_overlong_prompt_is_refused_before_the_sweep_row_exists(svc, monkeypatch):
    # create_job would refuse it anyway, but only after the sweep row was
    # minted and the rollback path ran; all-or-nothing admission means the
    # refusal happens before anything exists.
    minted: list[str] = []
    real = svc.store.create_sweep
    monkeypatch.setattr(
        svc.store,
        "create_sweep",
        lambda *args, **kwargs: minted.append("row") or real(*args, **kwargs),
    )
    with pytest.raises(Invalid):
        svc_sweeps.create_sweep(svc, _plan(seeds=(1,), prompt="x" * 1001))
    assert minted == []


def test_a_rollback_leaves_a_unit_the_worker_is_inside_alone(svc, monkeypatch):
    """create_job wakes the worker after *every* unit, so by the time a later
    one fails the first is routinely mid-trellis. The rollback used to write a
    status with store.cancel -- which waits for nothing -- and then hard-delete
    the row and rmtree the directory out from under the reconstruction, the
    exact pattern delete_sweep documents as fixed."""
    real = svc_jobs.create_job
    made: list[str] = []

    def create(*args, **kwargs):
        if len(made) >= 2:
            raise OSError("disk full")
        job = real(*args, **kwargs)
        made.append(job["id"])
        # A text job writes no input.png, so give it the directory a
        # reconstruction would be writing into.
        svc.job_dir(job["id"]).mkdir(parents=True, exist_ok=True)
        return job

    monkeypatch.setattr(svc_jobs, "create_job", create)
    # The first unit is the one the worker is inside when the third fails.
    monkeypatch.setattr(
        svc_sweeps.jobs_mod, "worker_is_inside", lambda _svc, job_id: job_id == made[0]
    )

    with pytest.raises(OSError):
        svc_sweeps.create_sweep(svc, _plan(seeds=(1, 2, 3)))

    assert svc.store.get(made[0]) is not None, "a live reconstruction was deleted"
    assert svc.job_dir(made[0]).exists()
    assert svc.store.get(made[1]) is None, "an idle unit is still rolled back"
    assert not svc.job_dir(made[1]).exists()


def test_each_param_is_routed_to_the_tier_create_job_expects():
    plan = _plan(
        seeds=(7,),
        base={"genre": "fantasy", "lora_weight": 0.9, "trellis_band": 8},
    )
    kwargs = svc_sweeps.unit_kwargs(plan, svc_sweeps.expand(plan)[0])
    assert kwargs["guidance_fields"] == {"genre": "fantasy", "style_lora": "render3d"}
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
    """A model-stage *reject* is the case the denormalization argument is true
    for: the finding is "this vector produced a bad mesh", which the row carries
    whole, so the mesh is disposable. The accept case is the opposite and is
    ``test_a_sweep_delete_keeps_the_units_it_cannot_regenerate`` below."""
    from warlock.service import verdicts as svc_verdicts

    plan = _plan(seeds=(1,), axes=(Axis("lora_weight", (0.6,)),))
    result = svc_sweeps.create_sweep(svc, plan)
    unit = svc.store.sweep_jobs(result["id"])[0]
    svc.store.set_status(unit["id"], "done")
    svc_verdicts.record_verdict(svc, unit["id"], verdict="reject")

    svc_sweeps.delete_sweep(svc, result["id"])

    assert svc_sweeps.list_sweeps(svc) == []
    assert svc.store.get(unit["id"]) is None
    # The whole point of snapshotting the vector onto the verdict row.
    assert len(svc.store.latest_verdicts()) == 1

    with pytest.raises(NotFound):
        svc_sweeps.delete_sweep(svc, result["id"])


def test_a_sweep_delete_keeps_the_units_it_cannot_regenerate(svc):
    """The 2026-08-09 regression, and it was not a crash: the database held 117
    verdicts of which 100 named job directories that no longer existed, every
    one destroyed by this button under a confirmation that truthfully promised
    the verdicts would be kept. They were; the pixels were not, and three
    blocked items needed the pixels.

    ``kept`` is counted apart from ``remaining`` because they mean opposite
    things to the reader: one is transient and invites a second press, the other
    is permanent and would make that press a lie renewing itself."""
    from warlock.service import verdicts as svc_verdicts

    plan = _plan(seeds=(1, 2), axes=(Axis("lora_weight", (0.6,)),))
    result = svc_sweeps.create_sweep(svc, plan)
    units = svc.store.sweep_jobs(result["id"])
    for unit in units:
        svc.store.set_status(unit["id"], "done")
    svc_verdicts.record_verdict(svc, units[0]["id"], verdict="accept")

    outcome = svc_sweeps.delete_sweep(svc, result["id"])

    assert outcome["kept"] == 1
    assert outcome["remaining"] == 0
    assert outcome["deleted"] == len(units) - 1
    assert svc.store.get(units[0]["id"]) is not None
    # And the sweep row survives, because ``Filters.matches`` hides a sweep unit
    # from the library -- dropping it would leave the kept unit unreachable.
    assert [s["id"] for s in svc_sweeps.list_sweeps(svc)] == [result["id"]]

    # Pressing again changes nothing. That is the point, and it is why the toast
    # for this case must not say "delete again in a moment".
    again = svc_sweeps.delete_sweep(svc, result["id"])
    assert (again["deleted"], again["kept"]) == (0, 1)
    assert svc.store.get(units[0]["id"]) is not None


def test_an_axis_that_changes_nothing_is_refused_rather_than_run_n_times(svc):
    """The regression: ``expand`` compares each unit against the *base* only,
    and ``guidance.normalize`` drops a scale with nothing to scale -- so an
    ip_scale sweep with no adapter in the base produced baseline plus N units
    with byte-identical params at the same seed. Up to MAX_UNITS runs of one
    picture, and N bogus "distinct configs" in the verdict corpus."""
    plan = _plan(seeds=(1,), axes=(Axis("ip_scale", (0.4, 0.6, 0.8)),))

    with pytest.raises(Invalid) as caught:
        svc_sweeps.create_sweep(svc, plan)

    assert "same job" in str(caught.value)
    assert svc_sweeps.list_sweeps(svc) == []


def test_the_same_axis_is_fine_once_its_adapter_is_set(svc):
    """Not a rule about ip_scale: it is a rule about units that submit the same
    job. With something for the scale to apply to, the units differ."""
    plan = _plan(
        seeds=(1,),
        base={"lora_weight": 0.6},
        axes=(Axis("lora_weight", (0.4, 0.8)),),
    )

    result = svc_sweeps.create_sweep(svc, plan)

    assert len(svc.store.sweep_jobs(result["id"])) == 3  # baseline + two values


class _FakeWorker:
    """Enough of ``Worker`` for the delete guards: which job it is inside, and
    a cancel that can be awaited."""

    def __init__(self, current_job_id: str | None) -> None:
        self.current_job_id = current_job_id
        self.cancelled: list[str] = []

    async def request_cancel(self, job_id: str) -> None:
        self.cancelled.append(job_id)


def test_a_unit_the_worker_is_inside_is_cancelled_but_left_on_disk(svc):
    """The regression: ``delete_sweep`` called the unguarded ``store.delete``
    and then rmtree'd, where ``delete_job`` and ``prune_jobs`` both go through
    ``delete_if_not_running``. A status check would not have been enough either
    -- ``cancel_job`` writes ``cancelled`` and only *asks* the worker to stop,
    so the row said the job was over while the run was still writing, and the
    directory came back as an orphan nothing owned."""
    plan = _plan(seeds=(1, 2), axes=(Axis("lora_weight", (0.6,)),))
    result = svc_sweeps.create_sweep(svc, plan)
    units = svc.store.sweep_jobs(result["id"])
    busy = units[0]["id"]
    svc.store.set_status(busy, "running")
    # Stand in for the live reconstruction's output directory.
    svc.job_dir(busy).mkdir(parents=True, exist_ok=True)
    (svc.job_dir(busy) / "source.glb").write_bytes(b"partial")
    svc.worker = _FakeWorker(busy)

    outcome = svc_sweeps.delete_sweep(svc, result["id"])

    assert outcome["remaining"] == 1
    assert outcome["deleted"] == len(units) - 1
    assert svc.store.get(busy) is not None
    assert svc.job_dir(busy).exists(), "its directory is still being written to"
    # And the sweep row survives, because it is what the second press deletes.
    assert [s["id"] for s in svc_sweeps.list_sweeps(svc)] == [result["id"]]

    # A moment later the worker has unwound, and the second press finishes it.
    svc.worker = _FakeWorker(None)
    again = svc_sweeps.delete_sweep(svc, result["id"])

    assert (again["deleted"], again["remaining"]) == (1, 0)
    assert svc_sweeps.list_sweeps(svc) == []
    assert svc.store.get(busy) is None


def test_sweep_units_are_hidden_from_the_library(svc):
    from warlock.studio.state import Filters

    filters = Filters()
    assert filters.matches({"id": "a", "status": "done"})
    assert not filters.matches({"id": "b", "status": "done", "sweep_id": "abc"})
