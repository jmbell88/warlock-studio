"""Parameter sweep specs: one axis varied at a time against a fixed baseline.

A suite asks "how good is the recipe on 40 prompts"; a sweep asks a narrower
question -- "does this one parameter matter" -- on a single fixed item,
across a range of values for exactly one axis at a time. The unit of work is
the baseline (the item run once per seed with the recipe's own settings) plus
one extra unit per (axis value, seed) that differs from the baseline. A value
equal to the baseline is skipped: it would be a second, indistinguishable
copy of a unit already planned.

Like ``suite.py`` and ``recipe.py``, a sweep is a file, never edited in
place -- ``lora-weight-v1.json`` stays what it says, and a changed sweep
means a v2 file.

Axis params fall into three tiers, matched to how ``recipe.job_kwargs``
builds a job's kwargs:

* a **guidance field** (anything ``guidance.form_fields()`` knows, including
  ``base_model``/``style_lora``) is merged into ``guidance_fields``;
* a **create_job kwarg** (``KWARG_AXES``) lands as a top-level kwarg, exactly
  as the recipe's own ``lora_weight``/``profile``/etc. would;
* a **server-config axis** (``SERVER_AXES``) changes something trellis-server
  is launched with, not something a request carries. A spec may name one --
  it loads, so the file that will eventually drive it can be written and
  reviewed now -- but ``unit_kwargs`` refuses it: running one means grouping
  units and restarting the server between groups, which is phase 2.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SWEEP_DIR = Path(__file__).resolve().parent / "sweeps"

# create_job kwargs a sweep axis may set directly. Mirrors exactly the set
# recipe.job_kwargs draws from beyond guidance_fields.
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
)

# Server-launch axes. Accepted at parse so a spec can be written and
# reviewed before phase 2 exists; unit_kwargs refuses them, since running one
# means restarting trellis-server between groups of units.
SERVER_AXES = ("trellis_band", "trellis_tex_res")

_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]")
SLUG_MAX = 40


def slug(value: Any) -> str:
    """A value made filesystem-safe -- a unit key becomes a directory name."""
    text = _SLUG_RE.sub("-", str(value))
    return text[:SLUG_MAX]


@dataclass(frozen=True, slots=True)
class Axis:
    param: str
    values: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class SweepSpec:
    key: str
    label: str
    recipe_key: str
    item: Any  # suite.Item
    axes: tuple[Axis, ...]
    seeds: tuple[int, ...]
    path: Path
    notes: str = ""


@dataclass(frozen=True, slots=True)
class SweepUnit:
    """One planned job. ``param is None`` marks a baseline unit; otherwise
    ``value`` is the raw (unslugged) value under test."""

    param: str | None
    value: Any
    seed: int

    @property
    def key(self) -> str:
        if self.param is None:
            return f"baseline--s{self.seed}"
        return f"{self.param}={slug(self.value)}--s{self.seed}"


def available() -> list[str]:
    return sorted(p.stem for p in SWEEP_DIR.glob("*.json"))


def load(key: str) -> SweepSpec:
    """Load and validate a sweep spec by key, or raise ValueError."""
    path = SWEEP_DIR / f"{key}.json"
    if not path.exists():
        raise ValueError(f"unknown sweep {key!r}; available: {available()}")
    return parse(json.loads(path.read_text("utf-8")), path)


def parse(raw: dict[str, Any], path: Path) -> SweepSpec:
    """Validate a sweep payload. The item is validated exactly as
    suite.parse validates one; axis params are checked against the three
    tiers above, and a guidance-tier axis has each of its values run through
    guidance.normalize the same way a suite item's guidance is."""
    from .. import guidance
    from . import suite as suite_mod

    known_guidance = set(guidance.form_fields())
    allowed_axes = known_guidance | set(KWARG_AXES) | set(SERVER_AXES)

    recipe_key = str(raw.get("recipe_key") or "")
    if not recipe_key:
        raise ValueError(f"{path.name}: a sweep needs a recipe_key")

    seeds = tuple(int(s) for s in raw.get("seeds") or ())
    if not seeds:
        raise ValueError(f"{path.name}: a sweep needs at least one seed")

    item = _parse_item(raw.get("item") or {}, path, known_guidance, suite_mod)

    axes: list[Axis] = []
    seen: set[str] = set()
    for entry in raw.get("axes") or ():
        param = str(entry.get("param") or "")
        if not param:
            raise ValueError(f"{path.name}: an axis has no param")
        if param in seen:
            raise ValueError(f"{path.name}: duplicate axis param {param!r}")
        seen.add(param)
        if param not in allowed_axes:
            raise ValueError(
                f"{path.name}: axis names unknown param {param!r}; "
                f"expected one of {sorted(allowed_axes)}"
            )
        values = list(entry.get("values") or ())
        if not values:
            raise ValueError(f"{path.name}: axis {param!r} has no values")
        if param in known_guidance:
            for value in values:
                try:
                    guidance.normalize({param: value})
                except ValueError as exc:
                    raise ValueError(f"{path.name}: axis {param!r}: {exc}") from exc
        axes.append(Axis(param=param, values=tuple(values)))
    if not axes:
        raise ValueError(f"{path.name}: a sweep needs at least one axis")

    return SweepSpec(
        key=str(raw.get("key") or path.stem),
        label=str(raw.get("label") or path.stem),
        recipe_key=recipe_key,
        item=item,
        axes=tuple(axes),
        seeds=seeds,
        path=path,
        notes=str(raw.get("notes") or ""),
    )


def _parse_item(entry: dict[str, Any], path: Path, known: set[str], suite_mod: Any) -> Any:
    from .. import guidance

    item_id = str(entry.get("id") or "")
    if not item_id:
        raise ValueError(f"{path.name}: the sweep item has no id")
    if not str(entry.get("prompt") or "").strip():
        raise ValueError(f"{path.name}: the sweep item has no prompt")
    category = str(entry.get("category") or "")
    if category not in suite_mod.CATEGORIES:
        raise ValueError(
            f"{path.name}: item {item_id} has category {category!r}; "
            f"expected one of {list(suite_mod.CATEGORIES)}"
        )
    fields = dict(entry.get("guidance") or {})
    unknown = sorted(set(fields) - known)
    if unknown:
        raise ValueError(f"{path.name}: item {item_id} names unknown guidance {unknown}")
    try:
        guidance.normalize(fields)
    except ValueError as exc:
        raise ValueError(f"{path.name}: item {item_id}: {exc}") from exc
    return suite_mod.Item(
        id=item_id,
        category=category,
        prompt=str(entry["prompt"]),
        guidance=fields,
        tags=tuple(str(t) for t in entry.get("tags") or ()),
        notes=str(entry.get("notes") or ""),
    )


def _baseline_value(recipe: Any, item: Any, param: str) -> Any:
    """What the baseline unit would already produce for ``param``, so
    ``plan_units`` can skip an axis value that duplicates it.

    The rule, kept deliberately simple: a recipe field that pins the param
    directly (``lora_weight``, ``profile``, ``custom_triangles``,
    ``negative_prompt``, ``reference_prep``) wins; otherwise the item+recipe
    guidance, normalized the same way job submission normalizes it, supplies
    the default -- this covers every guidance table field plus the derived
    kwarg fields ``guidance.normalize`` itself computes (``resolution``,
    ``size_m``, ``bg_removal``, ``negative_prompt``, and, when a style/
    adapter/control is chosen, ``lora_weight``/``ip_scale``/``control_scale``/
    ``control_end``). A param neither names finds no baseline and is never
    skipped.
    """
    from .. import guidance

    recipe_value = getattr(recipe, param, None)
    if recipe_value is not None:
        return recipe_value
    fields = {**recipe.guidance, **item.guidance}
    return guidance.normalize(fields).get(param)


def plan_units(spec: SweepSpec) -> list[SweepUnit]:
    """Baseline (one per seed) plus one unit per (axis value, seed) that
    differs from the baseline. Loads the spec's own recipe to resolve each
    axis's baseline value."""
    from . import recipe as recipe_mod

    recipe = recipe_mod.load(spec.recipe_key)
    units: list[SweepUnit] = [SweepUnit(param=None, value=None, seed=s) for s in spec.seeds]
    for axis in spec.axes:
        baseline = _baseline_value(recipe, spec.item, axis.param)
        for value in axis.values:
            if value == baseline:
                continue
            units.extend(
                SweepUnit(param=axis.param, value=value, seed=seed) for seed in spec.seeds
            )
    return units


def unit_kwargs(
    spec: SweepSpec, recipe: Any, unit: SweepUnit, *, stage: str = "model"
) -> dict[str, Any]:
    """``recipe.job_kwargs`` for the spec's item and the unit's seed, with the
    unit's one varied param overlaid. Raises ValueError for a server-config
    axis: phase 1 has no way to restart trellis-server between groups."""
    from .. import guidance
    from . import recipe as recipe_mod

    kwargs = recipe_mod.job_kwargs(recipe, spec.item, unit.seed, stage=stage)
    if unit.param is None:
        return kwargs
    param = unit.param
    if param in SERVER_AXES:
        raise ValueError(
            f"{param!r} is a server-config axis; running it needs restarting "
            "trellis-server between groups of units, which is phase 2, not "
            "yet implemented"
        )
    if param in guidance.form_fields():
        fields = dict(kwargs["guidance_fields"])
        fields[param] = unit.value
        guidance.normalize(fields)  # re-validate the overlaid combination
        kwargs["guidance_fields"] = fields
    else:
        kwargs[param] = unit.value
    return kwargs


# --- running -----------------------------------------------------------------
#
# Mirrors runner.run almost exactly -- same threading model, same manifest /
# resume machinery -- but the unit of work is a SweepUnit against one fixed
# item rather than an (item, seed) pair over a whole suite. SweepSpec already
# carries .key/.path/.seeds, which is exactly what manifest.build_manifest
# wants from a "suite" argument, so it stands in for one directly rather than
# needing an adapter class.

SWEEP_STAGE = "model"


def _refuse_server_axes(spec: SweepSpec) -> None:
    server = sorted({a.param for a in spec.axes if a.param in SERVER_AXES})
    if server:
        raise ValueError(
            f"sweep {spec.key!r} has server-config axis {server}; running one "
            "means restarting trellis-server between groups, which is phase "
            "2, not yet implemented"
        )


def plan_sweep(
    config: Any,
    spec: SweepSpec,
    recipe: Any,
    *,
    started: str,
    run_dir: Path | None = None,
) -> tuple[Path, list[SweepUnit], dict[str, Any]]:
    """Everything decided before a single job is submitted -- the sweep
    analogue of ``runner.plan_run``, so ``--dry-run`` is this code path minus
    execution rather than a second implementation of the planning rules."""
    from . import manifest as manifest_mod

    _refuse_server_axes(spec)
    todo = plan_units(spec)
    directory = run_dir or (Path(config.bench_dir) / "runs" / f"{started}-sweep-{spec.key}")
    doc = manifest_mod.build_manifest(
        config, spec, recipe,
        stage=SWEEP_STAGE, items=[spec.item.id], seeds=list(spec.seeds), started=started,
    )
    doc["sweep"] = {
        "key": spec.key,
        "axes": [{"param": a.param, "values": list(a.values)} for a in spec.axes],
    }
    return directory, todo, doc


def run_sweep(
    config: Any,
    *,
    spec_key: str,
    started: str,
    resume: Path | None = None,
    on_event: Callable[[str], None] | None = None,
) -> Path:
    """Execute a sweep and return its run directory."""
    from ..studio.runtime import Runtime
    from . import manifest as manifest_mod
    from . import recipe as recipe_mod
    from . import runner as runner_mod

    say = on_event or (lambda msg: log.info("%s", msg))
    spec = load(spec_key)
    _refuse_server_axes(spec)
    # Before the manifest is built, not when Runtime starts -- see runner.run.
    runner_mod._resolve_vram(config)
    recipe = recipe_mod.load(spec.recipe_key)

    already: set[str] = set()
    if resume is not None:
        try:
            stored = manifest_mod.read_manifest(resume)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"{resume}: not a bench run directory "
                f"({manifest_mod.FILENAME} is unreadable: {exc})"
            ) from None
        run_dir, todo, doc = plan_sweep(config, spec, recipe, started=started, run_dir=resume)
        # Refuse before a single job is submitted, exactly as runner.run does:
        # a run half-measured against two checkpoints -- or two sweep spec
        # files -- is worse than no run.
        manifest_mod.assert_resumable(stored, doc)
        already = runner_mod.completed(resume)
        again = runner_mod.retryable(resume)
        say(f"resuming {resume.name}: {len(already)} of {len(todo)} already done")
        if again:
            say(f"retrying {len(again)} unit(s) that did not finish")
    else:
        run_dir, todo, doc = plan_sweep(config, spec, recipe, started=started)
        manifest_mod.write_manifest(run_dir, doc)

    pending = [u for u in todo if u.key not in already]
    if not pending:
        say("nothing to do")
        return run_dir

    with Runtime(config) as svc:
        for n, unit in enumerate(pending, 1):
            say(f"[{n}/{len(pending)}] {unit.key}")
            try:
                record = _run_sweep_unit(svc, config, run_dir, spec, recipe, unit)
            except KeyboardInterrupt:
                say("interrupted; the in-flight job has been cancelled")
                raise
            runner_mod.append_item(run_dir, record)
    return run_dir


def _run_sweep_unit(
    svc: Any,
    config: Any,
    run_dir: Path,
    spec: SweepSpec,
    recipe: Any,
    unit: SweepUnit,
) -> dict[str, Any]:
    from ..service import jobs as svc_jobs
    from . import runner as runner_mod

    item_dir = run_dir / "items" / unit.key
    item_dir.mkdir(parents=True, exist_ok=True)

    kwargs = unit_kwargs(spec, recipe, unit, stage=SWEEP_STAGE)
    started = time.monotonic()
    job_id = svc_jobs.create_job(svc, **kwargs)["id"]
    try:
        job = runner_mod._await_job(svc, job_id)
    except BaseException:
        # Includes KeyboardInterrupt: the queue is serial, so leaving a job
        # running would block whatever the user does next.
        with runner_mod._suppressed():
            svc_jobs.cancel_job(svc, job_id)
        raise

    elapsed = time.monotonic() - started
    runner_mod._copy_artifacts(config.job_dir(job_id), item_dir)
    (item_dir / "job.json").write_text(json.dumps(job, indent=2, default=str), encoding="utf-8")

    record = {
        "key": unit.key,
        "param": unit.param,
        "value": unit.value,
        "job": job_id,
        "status": job.get("status"),
        "error": job.get("error"),
        "seconds": round(elapsed, 2),
    }
    # Views are always attempted for a done unit with a mesh -- unlike
    # runner.run's --render flag, a sweep exists to compare meshes, so there
    # is no "skip rendering" mode. A render failure costs this unit its
    # automatic score, not the run: non-fatal, recorded as views: false.
    if job.get("status") == "done" and (item_dir / "model.glb").exists():
        record["views"] = runner_mod._render_views(item_dir)
    return record
