"""Whether a gltfpack tier preserved what it had to. Pure, and the bar is data.

`TODO.md` §3's automated half. ``vendor/gltfpack/gltfpack.exe`` has been present
since 2026-08-07, so ``draft``/``standard``/``detailed`` are live code rather
than dormant -- which changed the shape of the constraint instead of removing
it. A named tier is no longer a button that *fails*; it is a button that
silently ships a worse mesh, and that is harder to notice. The bar therefore
stands unchanged: a tier stays out of the generate forms until it has been run
against a chest, a sword and a rock and shown to keep UVs, both PBR maps and
material assignment. This module is that sentence, made checkable.

**Pure in the ``vram.py``/``meshreport`` sense.** Stdlib plus ``glbio``, no
imports from ``service``/``queue``/``studio``, and a file it cannot read is
``None`` rather than an exception -- the harness runs nine of these unattended
and one bad output must report itself rather than ending the run.

**Why the JSON chunk rather than trimesh.** trimesh merges a scene into a single
mesh with a single material, which destroys the exact property under test:
whether a multi-material mesh still assigns each primitive its own material
after a simplify pass. ``meshreport`` answers a different question (will an
importer accept this, will it sit on the floor) and answers it about one mesh;
this one is a *comparison*, and the thing being compared is the node graph.

**Every rule is "not worse than the source", never "identical to an ideal".** A
rock with one untextured material is a legitimate input, and a tier must not be
recorded as failing for losing something that was never there. The cost of that
asymmetry is that a qualification could pass by preserving nothing, so a source
that was already missing something says so in ``notes`` -- visibly, beside the
verdict, rather than being silently counted as a pass.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import glbio

log = logging.getLogger(__name__)

__all__ = [
    "Report",
    "Row",
    "Survey",
    "Verdict",
    "compare",
    "optimize_profiles",
    "qualify",
    "survey",
]


@dataclass(frozen=True, slots=True)
class Survey:
    """What one GLB claims about itself, in the terms a tier can damage."""

    path: str
    primitives: int
    uv_primitives: int
    # Materials a primitive actually points at. An unreferenced material is not
    # preservation -- gltfpack drops those deliberately, and counting them would
    # make a real loss look like a wash.
    materials: int
    declared_materials: int
    unassigned: int
    base_color: int
    metallic_roughness: int
    normal_map: int
    images: int
    extensions_required: tuple[str, ...]
    bytes: int

    @property
    def textured(self) -> bool:
        return bool(self.base_color or self.metallic_roughness or self.normal_map)


@dataclass(frozen=True, slots=True)
class Verdict:
    ok: bool
    failures: tuple[str, ...]
    notes: tuple[str, ...]


def survey(path: Path) -> Survey | None:
    """Measure one GLB, or None if it cannot be read as one."""
    try:
        gltf, _ = glbio.read_glb(Path(path))
    except Exception:
        return None
    try:
        return _survey(Path(path), gltf)
    except (TypeError, ValueError, KeyError, AttributeError):
        # A structurally broken glTF -- a primitive that is not an object, a
        # materials value that is not a list. Same answer as an unreadable
        # container: this output is not usable, and saying so is the report.
        return None


def _survey(path: Path, gltf: dict[str, Any]) -> Survey:
    materials = list(gltf.get("materials") or ())
    primitives = [
        prim
        for mesh in (gltf.get("meshes") or ())
        for prim in (mesh.get("primitives") or ())
    ]
    used: set[int] = set()
    unassigned = 0
    uv = 0
    for prim in primitives:
        index = prim.get("material")
        if isinstance(index, int) and 0 <= index < len(materials):
            used.add(index)
        else:
            unassigned += 1
        if "TEXCOORD_0" in (prim.get("attributes") or {}):
            uv += 1
    return Survey(
        path=str(path),
        primitives=len(primitives),
        uv_primitives=uv,
        materials=len(used),
        declared_materials=len(materials),
        unassigned=unassigned,
        base_color=sum(1 for i in used if _pbr(materials[i], "baseColorTexture")),
        metallic_roughness=sum(
            1 for i in used if _pbr(materials[i], "metallicRoughnessTexture")
        ),
        normal_map=sum(1 for i in used if (materials[i] or {}).get("normalTexture")),
        images=len(gltf.get("images") or ()),
        extensions_required=tuple(gltf.get("extensionsRequired") or ()),
        bytes=path.stat().st_size if path.exists() else 0,
    )


def _pbr(material: Any, key: str) -> bool:
    return bool(((material or {}).get("pbrMetallicRoughness") or {}).get(key))


def compare(before: Survey | None, after: Survey | None) -> Verdict:
    """Did the tier keep what it had to? -> a verdict naming every loss."""
    if before is None:
        return Verdict(False, ("the source GLB could not be read",), ())
    if after is None:
        return Verdict(False, ("the optimized GLB could not be read",), ())

    failures: list[str] = []
    notes: list[str] = []

    # UVs, as coverage rather than as a count: a simplify pass legitimately
    # merges primitives, so "fewer primitives with UVs" is only a loss if some
    # primitive now has none.
    if before.uv_primitives == before.primitives and before.primitives:
        if after.uv_primitives != after.primitives:
            failures.append(
                f"UVs lost: {after.primitives - after.uv_primitives} of "
                f"{after.primitives} primitives have no TEXCOORD_0"
            )
    elif not before.uv_primitives:
        notes.append("the source has no UVs, so none could be lost")

    if before.unassigned == 0 and after.unassigned:
        failures.append(
            f"{after.unassigned} primitive(s) came back unassigned to any material"
        )
    if after.materials < before.materials:
        failures.append(
            f"material assignment lost: {before.materials} material(s) used, "
            f"{after.materials} after"
        )

    for label, got, want in (
        ("base colour", after.base_color, before.base_color),
        ("metallic/roughness", after.metallic_roughness, before.metallic_roughness),
        ("normal", after.normal_map, before.normal_map),
    ):
        if want and got < want:
            failures.append(f"{label} texture lost on {want - got} of {want} material(s)")
    if not before.textured:
        notes.append("the source carries no PBR textures, so none could be lost")

    added = tuple(e for e in after.extensions_required if e not in before.extensions_required)
    if added:
        # The one failure that no texture or UV count would catch: the viewer's
        # loader *refuses* a file requiring an extension it does not implement,
        # and KHR_mesh_quantization decodes as silently wrong geometry
        # otherwise. ``-noq`` exists to stop gltfpack writing it, and this is
        # what proves the flag is still doing its job.
        failures.append(f"the output requires extensions the source did not: {', '.join(added)}")

    return Verdict(not failures, tuple(failures), tuple(notes))


# --- the qualification run ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class Row:
    """One (subject, tier) cell of the qualification table."""

    subject: str
    tier: str
    before: Survey | None
    after: Survey | None
    verdict: Verdict
    stats: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    output: str | None = None


@dataclass(frozen=True, slots=True)
class Report:
    rows: tuple[Row, ...]
    tiers: tuple[str, ...]
    subjects: tuple[str, ...]

    @property
    def qualified(self) -> tuple[str, ...]:
        """The tiers that passed on **every** subject.

        Every, because a simplifier's failure modes are shape-dependent: a sword
        is thin, a rock is a blob, a chest has flat panels and a lid. That is
        why the bar names three subjects rather than "a mesh".
        """
        return tuple(
            tier
            for tier in self.tiers
            if all(row.verdict.ok for row in self.rows if row.tier == tier)
        )

    @property
    def ok(self) -> bool:
        return bool(self.tiers) and self.qualified == self.tiers


def optimize_profiles() -> dict[str, int | None]:
    """``pipelines.optimize.PROFILES``, imported lazily.

    Lazily so this module stays importable with no ``trimesh`` and no vendored
    binary -- the survey and comparison halves are the parts a test needs, and
    they are what keep it pure.
    """
    from .pipelines.optimize import PROFILES

    return dict(PROFILES)


def _default_optimizer(exe: Path, timeout: float) -> Callable[..., dict[str, Any]]:
    from .pipelines import optimize as optimize_mod

    def run(source: Path, dest: Path, *, target_triangles: int) -> dict[str, Any]:
        return optimize_mod.run(
            source, dest, target_triangles=target_triangles, exe=exe, timeout=timeout
        )

    return run


def qualify(
    sources: Sequence[Path],
    tiers: Sequence[str],
    workdir: Path,
    *,
    optimizer: Callable[..., dict[str, Any]] | None = None,
    exe: Path | None = None,
    timeout: float = 300.0,
) -> Report:
    """Run every tier against every subject and report the whole table.

    ``optimizer`` is injected so the table's own behaviour -- one row per cell, a
    failure that does not end the run, nothing written inside the corpus -- is
    testable without gltfpack, trimesh or a 22 MB mesh. The default is
    ``pipelines.optimize.run``, i.e. the same binary and the same flags a job
    uses; there is no second spelling of the invocation.

    **Nothing is written inside the corpus.** Every output goes under
    ``workdir``. This is not tidiness: ``service.jobs.optimize_job`` rewrites
    ``model.glb`` in place and deletes the derived artifacts (and reports the rig
    sheets) that describe the old mesh, so driving *it* would consume the very
    meshes the qualification needs -- which, per §3, are the handful of accepted
    ones, the only meshes on which preservation can be judged at all. A harness
    that destroys its inputs can be run once.
    """
    profiles = optimize_profiles()
    for tier in tiers:
        if tier not in profiles:
            raise ValueError(f"unknown tier {tier!r}; expected one of {sorted(profiles)}")
        if profiles[tier] is None:
            # ``raw`` is the absence of a tier, and comparing a file with a copy
            # of itself would report a pass that means nothing.
            raise ValueError(f"{tier!r} applies no budget; there is nothing to qualify")
    run = optimizer or _default_optimizer(
        exe or Path("vendor/gltfpack/gltfpack.exe"), timeout
    )
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    rows: list[Row] = []
    for source in sources:
        source = Path(source)
        subject = source.parent.name if source.stem in ("source", "model") else source.stem
        before = survey(source)
        for tier in tiers:
            dest = workdir / f"{subject}-{tier}.glb"
            stats: dict[str, Any] = {}
            error: str | None = None
            after: Survey | None = None
            try:
                stats = run(source, dest, target_triangles=profiles[tier]) or {}
                after = survey(dest)
            except Exception as exc:
                # Logged and carried, never raised: the point of a harness is
                # the whole table, and a tier that crashes on a sword is a
                # result about that tier rather than the end of the run.
                log.exception("tier %s failed on %s", tier, subject)
                error = str(exc)
            verdict = (
                Verdict(False, (f"gltfpack failed: {error}",), ())
                if error is not None
                else compare(before, after)
            )
            rows.append(
                Row(
                    subject=subject,
                    tier=tier,
                    before=before,
                    after=after,
                    verdict=verdict,
                    stats=stats,
                    error=error,
                    output=str(dest) if after is not None else None,
                )
            )
    return Report(
        rows=tuple(rows),
        tiers=tuple(tiers),
        subjects=tuple(dict.fromkeys(row.subject for row in rows)),
    )
