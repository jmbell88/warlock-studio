"""Planning and launching a parameter sweep on the live queue.

What changed, and why. The old ``bench/sweep.py`` was a CLI that stood up its
own ``Runtime`` -- which collides with a running app, because the queue is one
process holding one card -- read a spec file, and could only vary *one*
parameter at a time against a baseline. It therefore could never rate a
*combination* of settings, which is the thing anyone actually wants to know.

So the unit of work here is a whole settings vector, not a single parameter.
:func:`expand` is a convenience for building one common family of vectors (the
one-factor-at-a-time fan-out, which is still the cheapest way to find out
whether a knob matters at all), but :class:`SweepPlan` takes hand-picked
``vectors`` just as happily, and both come out the far side as the same thing:
a list of :class:`UnitPlan`, each of which becomes exactly one ordinary job.

Three rules shape the rest.

**Sweeps go through the front door.** A unit is submitted with
``service.jobs.create_job`` and nothing else -- same validation, same VRAM
admission, same directory-before-row ordering. What makes it a sweep unit is
two *columns* on the row (``sweep_id``, ``sweep_unit``), which is what keeps
membership out of ``params`` and therefore out of everything that copies
params.

**Admission is all-or-nothing.** Every unit is validated before a single row is
written, because the alternative is a sweep that queues eleven jobs and then
refuses the twelfth -- leaving the user with a partial run they did not ask for
and cannot interpret. One bad unit refuses the whole sweep, naming itself.

**Units are ordered to minimise trellis restarts.** Grouped by the server
config they need, with the config's own settings first. This is best-effort
ordering and nothing depends on it for correctness: ``Worker`` restarts the
server whenever the running one does not match the job in hand, so a
mis-ordered sweep costs restarts, never a wrong config.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from .. import guidance
from . import jobs as jobs_mod
from .core import WarlockService
from .errors import Invalid, NotFound, invalid_from
from .validation import (
    ALLOWED_RESOLUTIONS,
    check_prompt,
    check_seed,
    check_trellis_band,
    check_trellis_tex_res,
    check_vram,
    check_weights,
)

log = logging.getLogger(__name__)

# Axis params that are top-level ``create_job`` kwargs rather than guidance
# fields. Ported from the deleted ``bench/sweep.py``, plus the two the old
# module accepted at parse and then refused to run: ``trellis_band`` and
# ``trellis_tex_res`` are ordinary per-job params now, so the SERVER_AXES
# refusal has nothing left to refuse.
KWARG_AXES = (
    "lora_weight",
    "profile",
    "custom_triangles",
    "negative_prompt",
    "reference_prep",
    "resolution",
    "ip_scale",
    "control_scale",
    "control_end",
    "bg_removal",
    "size_m",
    "trellis_band",
    "trellis_tex_res",
)

# Which of the above ``guidance.normalize`` is handed, and whose output is
# therefore authoritative for them -- including when it *drops* one, which is
# what it does to a scale whose adapter was never selected. Kept beside
# KWARG_AXES so the two are read together: an axis added to one and not the
# other is a unit whose identity is computed from a value the job never keeps.
NORMALIZED_KWARGS = (
    "size_m",
    "resolution",
    "lora_weight",
    "ip_scale",
    "control_scale",
    "control_end",
    "bg_removal",
    "negative_prompt",
)

# The two that decide how trellis-server is launched, and therefore the two
# units are grouped by.
SERVER_AXES = ("trellis_tex_res", "trellis_band")

# How many jobs one submit may queue -- a guard against a runaway fan-out from
# a mis-typed axis list, and nothing more.
#
# It used to be justified by time ("a model-stage unit is roughly two minutes of
# GPU -- 64 of them is already a couple of hours"), which is not what it
# measures: it bounds one *submit*, not the queue, and two submits of 64 are
# admitted without a word. Someone reading the old comment could reasonably
# raise the number believing it protected them from a long run. It never did.
# The 2026-08-07 rogue sweep hit it at 100 units and was split into two 50-unit
# sweeps, which turned out better -- each half had a coherent question -- but
# the cap chose that, not the person planning it.
#
# Bounding total queue depth is the change that would enforce the time reading,
# and it is deliberately not made here: it would refuse a second sweep
# submitted while the first is still draining, which is a real workflow.
MAX_UNITS = 64


def axis_params() -> tuple[str, ...]:
    """Every param an axis may name, sorted -- what the form offers."""
    return tuple(sorted(set(guidance.form_fields()) | set(KWARG_AXES)))


@dataclass(frozen=True, slots=True)
class Axis:
    param: str
    values: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class UnitPlan:
    """One planned job: a label, the overrides on top of the plan's base, and
    the seed. ``overrides`` empty is the baseline unit."""

    label: str
    overrides: dict[str, Any] = field(default_factory=dict)
    seed: int = 42

    def server_group(self, base: dict[str, Any]) -> tuple[Any, ...]:
        merged = {**base, **self.overrides}
        return tuple(merged.get(p) for p in SERVER_AXES)


@dataclass(frozen=True, slots=True)
class SweepPlan:
    label: str
    prompt: str
    base: dict[str, Any] = field(default_factory=dict)
    seeds: tuple[int, ...] = (42,)
    axes: tuple[Axis, ...] = ()
    vectors: tuple[dict[str, Any], ...] = ()
    stage: str = "model"


def expand(plan: SweepPlan) -> list[UnitPlan]:
    """The plan's units, in the order they should be enqueued.

    Baseline once per seed, then one unit per (axis value that differs from the
    base, seed), then one per explicit vector per seed. A value equal to the
    base is skipped: it would be an indistinguishable second copy of a unit
    already planned.

    The result is then stably sorted into server-config groups with the base
    config's group first -- best-effort restart minimisation, see the module
    docstring.
    """
    units: list[UnitPlan] = [UnitPlan(label="baseline", seed=s) for s in plan.seeds]
    for axis in plan.axes:
        for value in axis.values:
            if plan.base.get(axis.param) == value:
                continue
            units.extend(
                UnitPlan(
                    label=f"{axis.param}={value}",
                    overrides={axis.param: value},
                    seed=seed,
                )
                for seed in plan.seeds
            )
    for i, vector in enumerate(plan.vectors, 1):
        overrides = dict(vector)
        label = overrides.pop("label", None) or f"combo {i}"
        units.extend(
            UnitPlan(label=str(label), overrides=overrides, seed=seed) for seed in plan.seeds
        )

    base_group = tuple(plan.base.get(p) for p in SERVER_AXES)
    groups: list[tuple[Any, ...]] = []
    for unit in units:
        group = unit.server_group(plan.base)
        if group not in groups:
            groups.append(group)
    # The base config first, then the rest in the order they were planned.
    # ``sorted`` is stable, so within a group the planned order survives.
    order = {group: (0 if group == base_group else 1 + groups.index(group)) for group in groups}
    return sorted(units, key=lambda u: order[u.server_group(plan.base)])


def unit_kwargs(plan: SweepPlan, unit: UnitPlan) -> dict[str, Any]:
    """Exactly the kwargs ``create_job`` takes for this unit.

    The three-tier routing the old sweep module had, minus its third tier's
    refusal: a guidance field is merged into ``guidance_fields``, anything in
    ``KWARG_AXES`` is a top-level kwarg, and an unknown param is an error here
    rather than a silently dropped setting at submit time.
    """
    merged = {**plan.base, **unit.overrides}
    known_guidance = set(guidance.form_fields())
    fields: dict[str, Any] = {}
    kwargs: dict[str, Any] = {
        "kind": "text",
        "prompt": plan.prompt,
        "output": "reference" if plan.stage == "reference" else "model",
        "seed": int(unit.seed),
    }
    for param, value in merged.items():
        if value is None or value == "":
            continue
        if param in known_guidance:
            fields[param] = value
        elif param in KWARG_AXES:
            kwargs[param] = value
        else:
            raise Invalid(f"unknown sweep param {param!r}", field="axes")
    kwargs["guidance_fields"] = fields
    return kwargs


def unit_label(unit: UnitPlan) -> str:
    return f"{unit.label} s{unit.seed}"


# --- admission ---------------------------------------------------------------


def _check_unit(svc: WarlockService, plan: SweepPlan, unit: UnitPlan) -> str:
    """Everything ``create_job`` would refuse, refused before anything is
    written. Mirrors create_job's own order so the two cannot drift into
    disagreeing about which unit is admissible.

    -> a canonical key for the job this unit would actually submit, so
    ``_validate`` can tell two units apart by what they *do* rather than by
    what they were labelled.
    """
    kwargs = unit_kwargs(plan, unit)
    check_seed("seed", kwargs["seed"])
    check_trellis_band(kwargs.get("trellis_band"))
    check_trellis_tex_res(kwargs.get("trellis_tex_res"))
    resolution = kwargs.get("resolution")
    if resolution is not None and resolution not in ALLOWED_RESOLUTIONS:
        raise Invalid(
            f"resolution must be one of {sorted(ALLOWED_RESOLUTIONS)}", field="resolution"
        )
    try:
        params = guidance.normalize(
            {
                **kwargs["guidance_fields"],
                "size_m": kwargs.get("size_m"),
                "resolution": kwargs.get("resolution"),
                "lora_weight": kwargs.get("lora_weight"),
                "ip_scale": kwargs.get("ip_scale"),
                "control_scale": kwargs.get("control_scale"),
                "control_end": kwargs.get("control_end"),
                "bg_removal": kwargs.get("bg_removal"),
                "negative_prompt": kwargs.get("negative_prompt"),
            },
            # Same gate create_job applies, for the reason this whole function
            # exists: a unit's canonical key is what it would actually submit,
            # and an ungated default here would key every unit on a matte the
            # job then ran without.
            bg_default=guidance.default_bg_removal(svc.config.trellis_models_dir),
        )
    except ValueError as exc:
        raise invalid_from(exc, "A sweep unit's settings are not usable") from exc
    # A sweep unit is submitted with no reference upload, so a conditioning
    # selection can never have an image to condition on -- create_job would
    # refuse it, but only after the sweep row was minted.
    if params.get("ip_adapter") or params.get("control"):
        raise Invalid(
            "conditioning needs a reference image, which a sweep cannot carry",
            field="reference",
        )
    if "profile" in kwargs:
        # The same check create_job runs, including the gltfpack-presence
        # gate: a sweep unit that would finish wearing a tier the missing
        # binary never applied poisons the verdict corpus.
        jobs_mod.resolve_profile(svc, {}, kwargs["profile"], kwargs.get("custom_triangles"))
    # Both halves of the admission door, in ``create_job``'s order. Only the
    # VRAM half was asked here, so a sweep whose base model (or style LoRA) was
    # not on disk was admitted, minted every row, and then refused unit by unit
    # inside ``create_job`` -- which is precisely the partial run this module's
    # all-or-nothing rule exists to prevent. A style_lora fails *silently* at
    # load, so the alternative for that axis is worse still: N finished units
    # whose rows claim a style that never ran, in a corpus keyed on it.
    check_weights(svc, "text", params)
    check_vram(svc, "text", plan.stage, params)
    # What this unit would actually submit: ``normalize``'s output for
    # everything it was handed -- so a setting it *dropped* is absent here too,
    # which is the whole point -- plus the kwargs it never saw, plus the seed.
    rest = {k: kwargs[k] for k in KWARG_AXES if k in kwargs and k not in NORMALIZED_KWARGS}
    return json.dumps(
        {**params, **rest, "seed": kwargs["seed"]}, sort_keys=True, default=str
    )


def _validate(svc: WarlockService, plan: SweepPlan, units: list[UnitPlan]) -> None:
    if not plan.prompt.strip():
        raise Invalid("a sweep needs a prompt", field="prompt")
    # create_job would refuse it too, but only after the sweep row was minted
    # and the rollback ran; all-or-nothing admission means the refusal happens
    # before anything exists.
    check_prompt(plan.prompt)
    if plan.stage not in ("reference", "model"):
        raise Invalid("stage must be 'reference' or 'model'", field="stage")
    if not plan.seeds:
        raise Invalid("a sweep needs at least one seed", field="seeds")
    if not units:
        raise Invalid("that sweep plans no units; add an axis value that differs")
    if len(units) > MAX_UNITS:
        raise Invalid(f"that sweep plans {len(units)} units; the limit is {MAX_UNITS}")
    # Two units that submit the same job are refused, not silently run twice.
    # ``expand`` compares each unit against the *base* and nothing else, while
    # guidance.normalize drops a setting with nothing to apply it to -- an
    # ip_scale or control_scale axis with no adapter selected is the case, and
    # it produced up to MAX_UNITS byte-identical jobs at the same seed. That is
    # hours of GPU redrawing one picture, and N bogus "distinct configs" in the
    # verdict corpus, from a sweep that looked like it was measuring something.
    seen: dict[str, str] = {}
    for unit in units:
        try:
            key = _check_unit(svc, plan, unit)
        except Invalid as exc:
            # Named, because "one of your twelve units is bad" is not something
            # anyone can act on. And the *field* is named too when the message
            # does not already carry it (S136): a sweep refusal is read against
            # a plan the user wrote in a script rather than against a form they
            # can see highlighted, so the address has to be in the words.
            where = ""
            if exc.field and exc.field not in str(exc):
                where = f" (from {exc.field})"
            raise Invalid(f"{unit_label(unit)}{where}: {exc}", field=exc.field) from exc
        first = seen.get(key)
        if first is not None:
            raise Invalid(
                f"{unit_label(unit)} would run exactly the same job as {first}. "
                "A scale has nothing to scale unless its adapter is set in the "
                "base, so varying it alone changes nothing.",
                field="axes",
            )
        seen[key] = unit_label(unit)


# --- the operations ----------------------------------------------------------


def create_sweep(svc: WarlockService, plan: SweepPlan) -> dict[str, Any]:
    """Validate every unit, then queue them all under one sweep row."""
    units = expand(plan)
    _validate(svc, plan, units)

    spec = {
        "base": dict(plan.base),
        "seeds": list(plan.seeds),
        "axes": [{"param": a.param, "values": list(a.values)} for a in plan.axes],
        "vectors": [dict(v) for v in plan.vectors],
        "stage": plan.stage,
    }
    sweep_id = svc.store.create_sweep(plan.label or "sweep", plan.prompt, spec)
    created: list[str] = []
    try:
        # One commit for the whole batch (A10): the directory-before-row
        # ordering inside create_job is untouched -- each unit's files still
        # land before its row is inserted -- only the per-row fsync is
        # deferred. The context manager commits in ``finally``, so the
        # rollback below always sees whatever rows were created.
        with svc.store.deferred_commits():
            for unit in units:
                job = jobs_mod.create_job(
                    svc,
                    **unit_kwargs(plan, unit),
                    sweep_id=sweep_id,
                    sweep_unit=unit_label(unit),
                )
                created.append(job["id"])
    except Exception:
        # Best-effort rollback, mirroring create_job's own cleanup: the
        # validation pass makes this unreachable for a *bad* unit, so anything
        # arriving here is a genuine failure (a full disk, a DB error) and a
        # half-queued sweep is worse than none.
        #
        # Through ``delete_sweep`` rather than a bare cancel-delete-rmtree.
        # ``create_job`` wakes the worker after *every* unit, so by the time
        # unit twelve fails unit one is routinely mid-trellis -- and
        # ``store.cancel`` writes a status without waiting for anything, so the
        # hard delete that followed it tore the directory out from under a
        # running reconstruction and left an orphan nothing owned. That is the
        # exact pattern ``delete_sweep`` documents as fixed; it survived here.
        # A unit the worker is still inside keeps its row and its sweep, which
        # the Review list offers a second press against.
        log.exception("sweep %s failed partway through; rolling back", sweep_id)
        try:
            delete_sweep(svc, sweep_id)
        except Exception:
            log.exception("could not roll back sweep %s", sweep_id)
        raise
    return {"id": sweep_id, "units": len(created), "jobs": created}


def list_sweeps(svc: WarlockService) -> list[dict[str, Any]]:
    return svc.store.list_sweeps()


def delete_sweep(svc: WarlockService, sweep_id: str) -> dict[str, Any]:
    """Cancel and remove a sweep's jobs, then the sweep row.

    The verdict rows are deliberately kept. Each carries its own snapshot of
    the config vector it was filed against, which is exactly the case this
    denormalization exists for -- the assets are disposable, what was learned
    from them is not.

    **A unit carrying evidence is left behind too, and it is not "remaining".**
    The two are counted apart because they mean opposite things to the reader:
    ``remaining`` is transient and the answer is to press again in a moment,
    while ``kept`` is permanent and pressing again will never change it. Folding
    the second into the first would produce a button that asks to be pressed
    forever. See ``jobs.retained_job_ids`` for which units qualify and why the
    per-job ``delete_job`` is deliberately still able to remove one.

    **A unit the worker is still inside is left behind, and so is the sweep.**
    This used to call the unguarded ``store.delete`` and then rmtree, where
    ``delete_job`` and ``prune_jobs`` both go through ``delete_if_not_running``
    -- and a status check would not have been enough anyway, because
    ``cancel_job`` writes ``cancelled`` and only *asks* the worker to stop, so
    the row says the job is over while the reconstruction is still writing.
    The directory then came back as an orphan nothing owned. Cancelling is
    still the first thing that happens, so the second press (a moment later,
    once the worker has unwound) finishes the job -- and the count says how
    many are left rather than reporting a deletion that did not happen.
    """
    if svc.store.get_sweep(sweep_id) is None:
        raise NotFound("no such sweep")
    return _remove_units(svc, sweep_id, jobs_mod.retained_job_ids(svc))


def cleanup_sweep(
    svc: WarlockService, sweep_id: str, *, keep_job_ids: Iterable[str] = ()
) -> dict[str, Any]:
    """Remove a fully judged sweep's assets, retention and all.

    This is the **user-chosen override** of ``retained_job_ids``, and saying so
    out loud is the point: everything ``retained_job_ids`` documents about the
    2026-08-09 incident is still true, and this is not a decision that a bulk
    delete may quietly make. It is offered only where the user has just looked
    at every unit in the sweep and filed a verdict on each one -- at which point
    the assets have served the purpose retention was protecting, and leaving a
    judged sweep's twenty meshes on disk forever is the cost of a rule with no
    exit.

    The wording that reaches the user carries the same burden ``ask_clean``
    carries (``studio/panes/library.py``): the app has told them twice that a
    bulk delete keeps their evidence, so the one place that is false has to say
    which promise it is breaking rather than only that something is being
    deleted. Here the promise is narrower and so is the breach -- one sweep, all
    of whose units the user has just judged -- which is why this is a toast
    after the fact rather than a second modal, and why the offer is made up
    front on the entry card instead.

    What is *not* broken: verdicts, observations and findings live in the
    database, so an accepted unit's row and its config vector survive this
    exactly as they survive ``delete_sweep``. What is lost is the pixels and the
    meshes behind them -- which is the whole of what ``retained_job_ids`` exists
    to defend, and the whole of what the user is choosing to give up.

    ``keep_job_ids`` is the caller's own exception list, refused if it names a
    job outside this sweep: a keep id from another sweep would silently do
    nothing, which is the shape of a bug that only shows up as missing files.
    """
    if svc.store.get_sweep(sweep_id) is None:
        raise NotFound("no such sweep")
    keep = {str(job_id) for job_id in keep_job_ids}
    if keep:
        mine = {job["id"] for job in svc.store.sweep_jobs(sweep_id)}
        stray = sorted(keep - mine)
        if stray:
            raise Invalid(
                f"{stray[0]} is not a unit of this sweep.", field="keep_job_ids"
            )
    return _remove_units(svc, sweep_id, keep)


def _remove_units(
    svc: WarlockService, sweep_id: str, keep: set[str]
) -> dict[str, Any]:
    """Cancel and remove every unit outside *keep*. -> the shared result shape.

    Extracted from ``delete_sweep`` verbatim so the two callers cannot drift:
    the guards below (cancel first, then ``worker_is_inside`` /
    ``dependent_jobs`` / ``delete_if_not_running``) are the ones a rollback and
    a stale-row incident were both traced to, and a second copy of them is a
    second place for one to be forgotten. The *only* thing the two callers
    differ in is what goes into ``keep``.
    """
    removed = 0
    remaining = 0
    kept = 0
    for job in svc.store.sweep_jobs(sweep_id):
        if job["id"] in keep:
            # Checked before the cancel: a queued unit is not evidence yet, so
            # this can only match a finished one, but reaching the cancel first
            # would still be asking the worker to stop a job we are keeping.
            kept += 1
            continue
        if job["status"] in ("queued", "running"):
            # Through the service, so a running job's worker is actually asked
            # to stop rather than only having its row rewritten.
            try:
                jobs_mod.cancel_job(svc, job["id"])
            except Exception:
                log.exception("could not cancel sweep job %s", job["id"])
        if jobs_mod.worker_is_inside(svc, job["id"]) or jobs_mod.dependent_jobs(svc, job["id"]):
            remaining += 1
            continue
        if not svc.store.delete_if_not_running(job["id"]):
            remaining += 1
            continue
        shutil.rmtree(svc.job_dir(job["id"]), ignore_errors=True)
        removed += 1
    if remaining or kept:
        # The sweep row outlives its last unit deliberately: it is what the
        # Review list offers the second press against -- and, for a kept unit,
        # the only thing that still groups it. Dropping the row while its
        # accepted units kept their ``sweep_id`` would hide them from the
        # library (``Filters.matches`` excludes sweep units) with nothing left
        # in Review to reach them by.
        return {"ok": True, "deleted": removed, "remaining": remaining, "kept": kept}
    svc.store.delete_sweep(sweep_id)
    return {"ok": True, "deleted": removed, "remaining": 0, "kept": 0}
