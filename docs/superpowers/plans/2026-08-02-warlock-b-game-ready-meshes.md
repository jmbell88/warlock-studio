# Plan B — Game-Ready Meshes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a Warlock GLB drop into Godot or Unity untouched — budgeted triangles, a grounded pivot, an honest readiness report, a collider, and the formats an engine actually asks for.

**Architecture:** The trellis response becomes `source.glb` and is never overwritten; `model.glb` becomes a derived artifact produced by optimize-then-transform, written through a temp file and an atomic replace. Everything derived from `model.glb` (STL, OBJ zip, FBX, collider, textures) keeps using the existing lazy-convert-under-a-per-artifact-lock idiom in `app.get_file`, so no new concurrency model is introduced.

**Tech Stack:** trimesh, a vendored `gltfpack.exe`, Blender out-of-process (FBX only), FastAPI, Pillow.

## Global Constraints

See `2026-08-02-warlock-review-index.md` § Global Constraints. Every task's
requirements implicitly include that section. In particular: `bpy` only in
`pipelines/blender_worker.py`; no runtime downloads (gltfpack is vendored by hand,
exactly like `trellis-server.exe`); `uv run pytest -q` and `uv run ruff check .`
green before every commit.

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `src/warlock/meshreport.py` | topology/material/scale readiness report | **Create** |
| `src/warlock/meshaudit.py` | silhouette hole measurement | Unchanged; consumed by the report |
| `src/warlock/pipelines/postprocess.py` | GLB transforms + format conversion | Modify: grounding, collider, textures |
| `src/warlock/pipelines/optimize.py` | gltfpack invocation and validation | **Create** |
| `src/warlock/queue.py` | worker ordering: optimize → transform → report | Modify |
| `src/warlock/app.py` | `_MEDIA`, `/optimize`, `/export` routes | Modify |
| `src/warlock/pipelines/blender_worker.py` | `op_fbx` | Modify |
| `src/warlock/config.py` | `gltfpack_exe`, `export_dir` | Modify |
| `vendor/gltfpack/` | pinned binary + LICENSE + VERSION | **Create** |

Order matters: Task 1 (report) and Task 2 (grounding) are independent of Task 3
(optimize), but Task 3 changes what the report and the transform run *on*, so run
them 1 → 2 → 3 as written.

---

### Task 1: Readiness report replacing the hole badge (review item #9)

`meshaudit` is a 4-view silhouette check and the UI calls its result "watertight",
which it does not prove. `trimesh` is already a dependency and can answer the real
questions.

**Files:**
- Create: `src/warlock/meshreport.py`
- Modify: `src/warlock/queue.py:652-705` (`_audit_mesh` → `_report_mesh`)
- Modify: `src/warlock/static/app.js:565-575, 577-627` (`qualityBadge`)
- Test: `tests/test_meshreport.py` (create)

**Interfaces:**
- Produces: `meshreport.build(glb_path: Path, *, target_size_m: float | None = None, silhouette: dict | None = None) -> dict`
  returning
  `{"status": "ready"|"review"|"invalid", "reasons": list[str], "triangles": int, "vertices": int, "components": int, "degenerate": int, "boundary_edges": int, "watertight": bool, "has_uvs": bool, "has_normals": bool, "textures": {"base_color": bool, "metallic_roughness": bool, "normal": bool}, "extents_m": [float, float, float], "achieved_size_m": float, "grounded": bool, "bytes": int, "silhouette": dict | None}`
- Consumes: `meshaudit.hole_fraction` output as the `silhouette` argument.
- Stored on the job as `params["mesh_report"]`. `params["mesh_audit"]` keeps being
  written unchanged so old rows and any existing reader keep working.

- [ ] **Step 1: Write the failing test**

Create `tests/test_meshreport.py`:

```python
from __future__ import annotations

import trimesh

from warlock import meshreport


def _write(tmp_path, mesh, name="m.glb"):
    path = tmp_path / name
    trimesh.Scene(mesh).export(path)
    return path


def test_a_clean_box_is_ready(tmp_path):
    path = _write(tmp_path, trimesh.creation.box(extents=(1.0, 1.0, 1.0)))
    report = meshreport.build(path)
    assert report["status"] in ("ready", "review")
    assert report["watertight"] is True
    assert report["triangles"] == 12
    assert report["components"] == 1
    assert report["boundary_edges"] == 0


def test_an_open_surface_is_not_watertight_and_is_flagged(tmp_path):
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    box.faces = box.faces[:-2]          # tear a hole
    box.remove_unreferenced_vertices()
    path = _write(tmp_path, box)
    report = meshreport.build(path)
    assert report["watertight"] is False
    assert report["boundary_edges"] > 0
    assert report["status"] == "review"
    assert any("watertight" in r for r in report["reasons"])


def test_size_and_grounding_are_measured(tmp_path):
    box = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    box.apply_translation((0.0, 1.0, 0.0))   # glTF is Y-up: min Y == 0
    path = _write(tmp_path, box)
    report = meshreport.build(path, target_size_m=2.0)
    assert report["grounded"] is True
    assert abs(report["achieved_size_m"] - 2.0) < 1e-6


def test_an_unparseable_file_is_invalid(tmp_path):
    path = tmp_path / "broken.glb"
    path.write_bytes(b"not a glb")
    report = meshreport.build(path)
    assert report["status"] == "invalid"
    assert report["reasons"]
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_meshreport.py -v`
Expected: FAIL — `ModuleNotFoundError: warlock.meshreport`

- [ ] **Step 3: Write the module**

Create `src/warlock/meshreport.py`:

```python
"""Is this mesh usable in a game engine, and if not, why not.

Replaces the single "watertight" badge, which was a silhouette measurement
wearing a topology word. The two questions are genuinely different: meshaudit
answers "can you see through it", which is what a player notices, and this
module answers "will an importer accept it and will it sit on the floor", which
is what an engine notices. Both are reported; only topology may use the word
watertight.

Deliberately advisory. Nothing here rejects a mesh -- a `review` model is still
downloadable, because a warning the user can act on beats a job that refuses to
hand over the thing it already spent two minutes making.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# A pivot this far off the floor, as a fraction of the model's height, is a
# grounding failure rather than float noise.
GROUND_TOLERANCE = 0.001

# Achieved longest axis may miss the requested size by this fraction before it
# is worth saying so.
SIZE_TOLERANCE = 0.01

# Above this the mesh is a source reconstruction, not a game asset. The default
# trellis output is ~290k triangles, so this fires until the optimizer runs.
TRIANGLE_BUDGET = 150_000

# Silhouette hole fraction at which the mesh stops being cosmetically fine.
HOLE_WARN = 0.02


def build(
    glb_path: Path,
    *,
    target_size_m: float | None = None,
    silhouette: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Measure a finished GLB and classify it ready / review / invalid."""
    import numpy as np
    import trimesh

    try:
        loaded = trimesh.load(glb_path, process=False)
        mesh = loaded.to_mesh() if isinstance(loaded, trimesh.Scene) else loaded
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if len(faces) == 0 or not np.isfinite(vertices).all():
            raise ValueError("mesh has no finite geometry")
    except Exception as exc:
        log.warning("mesh report could not parse %s: %s", glb_path, exc)
        return {
            "status": "invalid",
            "reasons": [f"could not read the mesh: {exc}"],
            "bytes": _size(glb_path),
            "silhouette": silhouette,
        }

    reasons: list[str] = []

    # Topology. trimesh's own predicates, not a reimplementation: they are what
    # every other consumer of this format uses to decide the same questions.
    watertight = bool(mesh.is_watertight)
    components = int(len(mesh.split(only_watertight=False)))
    # An edge with exactly one adjacent face is a boundary edge; more than two
    # is non-manifold. Both fall out of counting how many times each unique
    # edge is referenced, which is one bincount rather than two traversals.
    counts = np.bincount(mesh.edges_unique_inverse, minlength=len(mesh.edges_unique))
    boundary_edges = int((counts == 1).sum())
    nonmanifold_edges = int((counts > 2).sum())
    degenerate = int((~mesh.nondegenerate_faces()).sum())

    extents = [float(v) for v in mesh.extents]
    achieved = float(max(extents)) if extents else 0.0
    bounds_min = [float(v) for v in mesh.bounds[0]]
    height = extents[1] if len(extents) > 2 else 0.0
    # glTF is Y-up, so the floor is minimum Y.
    grounded = abs(bounds_min[1]) <= max(height * GROUND_TOLERANCE, 1e-6)

    has_normals = bool(getattr(mesh, "vertex_normals", None) is not None
                       and len(mesh.vertex_normals) == len(vertices))
    has_uvs, textures = _materials(mesh)

    triangles = int(len(faces))
    if not watertight:
        reasons.append(
            f"not watertight: {boundary_edges} boundary edge(s) in {components} component(s)"
        )
    if nonmanifold_edges:
        reasons.append(f"{nonmanifold_edges} non-manifold edge(s)")
    if degenerate:
        reasons.append(f"{degenerate} degenerate triangle(s)")
    if triangles > TRIANGLE_BUDGET:
        reasons.append(f"{triangles:,} triangles is above the {TRIANGLE_BUDGET:,} budget")
    if not has_uvs:
        reasons.append("no UV coordinates")
    if not textures["base_color"]:
        reasons.append("no base-color texture")
    if not textures["metallic_roughness"]:
        reasons.append("no metallic/roughness texture")
    if not grounded:
        reasons.append(f"pivot is {bounds_min[1]:.4f} m off the floor")
    if target_size_m and achieved > 0 and abs(achieved - target_size_m) / target_size_m > SIZE_TOLERANCE:
        reasons.append(f"longest axis is {achieved:.3f} m, asked for {target_size_m:.3f} m")
    worst = (silhouette or {}).get("worst")
    if isinstance(worst, (int, float)) and worst > HOLE_WARN:
        reasons.append(f"{worst * 100:.1f}% of the worst silhouette is see-through")

    return {
        "status": "review" if reasons else "ready",
        "reasons": reasons,
        "triangles": triangles,
        "vertices": int(len(vertices)),
        "components": components,
        "degenerate": degenerate,
        "boundary_edges": boundary_edges,
        "nonmanifold_edges": nonmanifold_edges,
        "watertight": watertight,
        "has_uvs": has_uvs,
        "has_normals": has_normals,
        "textures": textures,
        "extents_m": extents,
        "achieved_size_m": achieved,
        "grounded": grounded,
        "bytes": _size(glb_path),
        "silhouette": silhouette,
    }


def _materials(mesh: Any) -> tuple[bool, dict[str, bool]]:
    """-> (has UVs, which PBR maps are present).

    Reads the material off the merged mesh rather than the glTF JSON: trimesh
    has already resolved which image feeds which slot, and re-deriving that from
    the raw JSON would be a second, disagreeing answer to the same question.
    """
    visual = getattr(mesh, "visual", None)
    uv = getattr(visual, "uv", None)
    has_uvs = uv is not None and len(uv) > 0
    material = getattr(visual, "material", None)
    return has_uvs, {
        "base_color": getattr(material, "baseColorTexture", None) is not None,
        "metallic_roughness": getattr(material, "metallicRoughnessTexture", None) is not None,
        "normal": getattr(material, "normalTexture", None) is not None,
    }


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0
```

If `mesh.nondegenerate_faces()` is not callable in the installed trimesh version,
substitute `int((mesh.area_faces <= 0).sum())`, which measures the same thing.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_meshreport.py -v`
Expected: PASS. If a trimesh API used above does not exist in the pinned version,
fix the call — not the assertion.

- [ ] **Step 5: Write the failing worker test**

In `tests/test_queue.py`:

```python
@pytest.mark.asyncio
async def test_finished_job_carries_a_mesh_report(worker_env, monkeypatch):
    worker, store = worker_env
    monkeypatch.setattr(
        "warlock.meshreport.build",
        lambda *a, **k: {"status": "ready", "reasons": [], "triangles": 42},
    )
    job_id = store.create("text", "a barrel", {"seed": 1})
    await _run_until_done(worker, store, job_id)
    assert store.get(job_id)["params"]["mesh_report"]["status"] == "ready"
```

- [ ] **Step 6: Run it and watch it fail**

Run: `uv run pytest tests/test_queue.py -k mesh_report -v`
Expected: FAIL — `KeyError: 'mesh_report'`

- [ ] **Step 7: Store the report from the worker**

In `src/warlock/queue.py`, inside `_audit_mesh`, after `params["mesh_audit"] = {...}`
and before the final `set_params`:

```python
        # The silhouette number stays exactly as it was -- it is the only thing
        # that catches trellis's disconnected-plate crust. The report adds what
        # the silhouette cannot see: topology, materials, budget, and whether
        # the thing will sit on an engine's floor.
        try:
            from . import meshreport

            params["mesh_report"] = await asyncio.to_thread(
                functools.partial(
                    meshreport.build,
                    glb_path,
                    target_size_m=params.get("size_m"),
                    silhouette=params["mesh_audit"],
                )
            )
        except Exception:
            log.exception("mesh report failed for job %s", job_id)
```

The whole thing sits inside the existing `try`'s success path, so a report failure
can never fail a job whose mesh is already on disk — the same rule the audit
already follows.

- [ ] **Step 8: Run everything**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all green

- [ ] **Step 9: Render the report in the UI**

In `static/app.js`, replace `qualityBadge` with:

```js
// Two different measurements, deliberately not merged: meshaudit says how much
// of the silhouette you can see through (what a player notices) and meshreport
// says whether an engine will accept it (topology, materials, budget, pivot).
// Only the second may use the word watertight.
function qualityBadge(job) {
  const report = job.params?.mesh_report;
  if (report) {
    if (report.status === "invalid") return { cls: "bad", text: "invalid mesh" };
    if (report.status === "ready") return { cls: "good", text: `ready · ${Number(report.triangles).toLocaleString()} tris` };
    return { cls: "warn", text: `review · ${report.reasons.length} issue(s)`, reasons: report.reasons };
  }
  const worst = job.params?.mesh_audit?.worst;
  if (typeof worst !== "number") return null;
  const pct = (worst * 100).toFixed(worst < 0.1 ? 1 : 0);
  if (worst < 0.02) return { cls: "good", text: "no visible holes" };
  return { cls: worst < 0.08 ? "warn" : "bad", text: `${pct}% see-through` };
}
```

In `updateNode`, change the call site to `qualityBadge(job)` and add the reason
list as a tooltip:

```js
  const badge = job.status === "done" ? qualityBadge(job) : null;
  n.quality.className = `job-quality ${badge ? badge.cls : ""}`;
  setText(n.quality, badge ? badge.text : "");
  n.quality.title = badge?.reasons ? badge.reasons.join("\n") : "";
```

Note the fallback branch also renames the old label from "watertight" to
"no visible holes", which NEXT.md §5 asks for explicitly.

- [ ] **Step 10: Commit**

```bash
git add src/warlock/meshreport.py src/warlock/queue.py src/warlock/static tests/
git commit -m "Warlock v0.0.1

Add a mesh readiness report and stop calling a silhouette check watertight."
```

---

### Task 2: Ground and centre the pivot (review item #8)

`scale_glb` scales but never grounds, so every import into Godot or Unity needs a
manual origin fixup.

**Files:**
- Modify: `src/warlock/pipelines/postprocess.py:109-158`
- Modify: `src/warlock/queue.py:623-650`
- Test: `tests/test_postprocess.py`

**Interfaces:**
- Produces: `postprocess.normalize_glb(glb_path: Path, target_max_m: float | None) -> dict`
  returning `{"scale": float, "translation": [float, float, float], "achieved_size_m": float}`.
  `scale_glb` stays as a thin wrapper returning just the factor, so existing
  callers and tests are unchanged.
- Consumes: nothing.
- Consumed by: Task 1's report (`grounded` becomes True), Task 3 (runs after optimize).

- [ ] **Step 1: Write the failing test**

In `tests/test_postprocess.py`:

```python
def test_normalize_grounds_and_centres(tmp_path):
    import trimesh

    from warlock.pipelines import postprocess

    box = trimesh.creation.box(extents=(1.0, 2.0, 1.0))
    box.apply_translation((5.0, 7.0, -3.0))
    path = tmp_path / "m.glb"
    trimesh.Scene(box).export(path)

    result = postprocess.normalize_glb(path, 4.0)

    loaded = trimesh.load(path)
    mesh = loaded.to_mesh()
    lo, hi = mesh.bounds
    assert abs(float(max(mesh.extents)) - 4.0) < 1e-4     # scaled
    assert abs(float(lo[1])) < 1e-6                        # grounded: min Y == 0
    assert abs(float(lo[0] + hi[0])) < 1e-6                # centred in X
    assert abs(float(lo[2] + hi[2])) < 1e-6                # centred in Z
    assert abs(result["achieved_size_m"] - 4.0) < 1e-4


def test_normalize_without_a_target_still_grounds(tmp_path):
    import trimesh

    from warlock.pipelines import postprocess

    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    box.apply_translation((0.0, 9.0, 0.0))
    path = tmp_path / "m.glb"
    trimesh.Scene(box).export(path)

    postprocess.normalize_glb(path, None)

    assert abs(float(trimesh.load(path).to_mesh().bounds[0][1])) < 1e-6
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_postprocess.py -k normalize -v`
Expected: FAIL — `AttributeError: normalize_glb`

- [ ] **Step 3: Implement it**

In `src/warlock/pipelines/postprocess.py`, replace `scale_glb` with:

```python
def normalize_glb(glb_path: Path, target_max_m: float | None) -> dict[str, Any]:
    """Scale to a real-world size, centre in X/Z and sit the model on Y=0.

    Returns the transform that was applied. Scaling alone was never enough: a
    trellis GLB's origin is wherever the reconstruction volume's centre happened
    to be, so every import into Godot or Unity needed a manual origin fixup
    before the asset would stand on the ground.

    The measurement goes through trimesh but the write does not -- re-exporting
    would re-encode every texture. Only the JSON chunk is rewritten, inserting
    one node below the scene root that carries both the scale and the
    translation; buffers and images are copied through byte-for-byte. It goes
    *below* the root and not on it because trimesh treats a scene root as the
    graph's base frame and silently discards its transform, which would leave
    the GLB transformed and the derived STL/OBJ exports not.
    """
    scene = trimesh.load(glb_path)
    extents = scene.extents
    bounds = scene.bounds
    extent = float(max(extents))
    factor = 1.0
    if target_max_m and extent > 0 and target_max_m > 0:
        factor = target_max_m / extent

    # In the *scaled* frame: centre X and Z on the origin, put minimum Y at zero.
    # glTF is Y-up, which is why Y is the odd one out here and Z is not.
    lo, hi = bounds[0], bounds[1]
    translation = [
        -float(lo[0] + hi[0]) / 2.0 * factor,
        -float(lo[1]) * factor,
        -float(lo[2] + hi[2]) / 2.0 * factor,
    ]

    header, gltf, rest = _split_glb(glb_path.read_bytes())
    gltf_scene = gltf["scenes"][gltf.get("scene", 0)]
    nodes = gltf.setdefault("nodes", [])
    for root in gltf_scene.get("nodes", []):
        _insert_transform_below(nodes, root, factor, translation)
    tmp = glb_path.with_suffix(".glb.tmp")
    tmp.write_bytes(_rebuild_glb(header, gltf, rest))
    os.replace(tmp, glb_path)
    return {
        "scale": factor,
        "translation": translation,
        "achieved_size_m": extent * factor,
    }


def scale_glb(glb_path: Path, target_max_m: float) -> float:
    """Back-compatible wrapper: normalize and return only the scale factor."""
    return normalize_glb(glb_path, target_max_m)["scale"]


def _insert_transform_below(
    nodes: list[dict], root: int, factor: float, translation: list[float]
) -> None:
    """Push everything hanging off ``root`` into a new scaled+moved child node.

    Same reasoning as the scale-only version this replaces: the transform must
    not go on the root itself, because trimesh discards a scene root's transform
    and the derived exports would come out untransformed.
    """
    node = nodes[root]
    child: dict = {"scale": [factor] * 3, "translation": translation}
    for key in ("mesh", "children", "skin", "weights"):
        if key in node:
            child[key] = node.pop(key)
    nodes.append(child)
    node["children"] = [len(nodes) - 1]
```

Delete `_insert_scale_below` (it is replaced) and add `from typing import Any` to
the imports. A glTF node applies `translation` after `scale` (T·R·S), which is why
the translation above is pre-multiplied by `factor`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_postprocess.py -v`
Expected: PASS

- [ ] **Step 5: Call it from the worker**

In `src/warlock/queue.py:_apply_scale`, replace the guard and the call. The
grounding must happen even when no `size_m` was requested, so the early return
changes:

```python
        if self._cancel is not None and self._cancel.event.is_set():
            return
        size_m = params.get("size_m")
        self.progress.update(
            job_id,
            phase="scale",
            label="Scaling and grounding",
            inner=0.0,
            inner_next=1.0,
            nominal=2.0,
            detail=f"{size_m} m" if size_m else "grounding",
        )
        from .pipelines import postprocess

        transform = await asyncio.to_thread(
            postprocess.normalize_glb, glb_path, float(size_m) if size_m else None
        )
        params["scale_factor"] = transform["scale"]
        params["transform"] = transform
        await asyncio.to_thread(self.store.set_params, job_id, params)
```

- [ ] **Step 6: Run everything**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all green. `tests/test_queue.py` may assert `scale_factor` is absent when
no `size_m` was given — if so, update that assertion: grounding now always runs and
records `scale_factor: 1.0`, which is the intended behaviour change.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Warlock v0.0.1

Ground and centre the pivot alongside scaling."
```

---

### Task 3: Triangle budgets via a vendored gltfpack (review item #7)

Every mesh ships at ~290k triangles / 22 MB. `gltfpack` retargets it without
rerunning trellis.

**Files:**
- Create: `vendor/gltfpack/gltfpack.exe`, `vendor/gltfpack/LICENSE`, `vendor/gltfpack/VERSION`
- Create: `src/warlock/pipelines/optimize.py`
- Modify: `src/warlock/config.py`, `src/warlock/doctor.py`
- Modify: `src/warlock/queue.py:436-451`, `src/warlock/app.py`
- Test: `tests/test_optimize.py` (create), `tests/test_doctor.py`

**Interfaces:**
- Produces: `optimize.PROFILES: dict[str, int | None]` =
  `{"draft": 20_000, "standard": 50_000, "detailed": 100_000, "raw": None}`;
  `optimize.run(source: Path, dest: Path, *, target_triangles: int | None, exe: Path, timeout: float = 300.0) -> dict`
  returning `{"requested": int | None, "achieved": int, "source_triangles": int, "bytes": int}`;
  `optimize.OptimizeError(RuntimeError)`.
- Produces: `Config.gltfpack_exe: Path` (env `WARLOCK_GLTFPACK`).
- Produces: `POST /api/jobs/{id}/optimize` with form field `profile`.
- Consumes: `postprocess.normalize_glb` (Task 2) — optimize runs *before* the
  transform, per NEXT.md §4.
- New on-disk contract: `source.glb` is the trellis response, `model.glb` is derived.

- [ ] **Step 1: Vendor the binary**

Download a pinned `gltfpack` release for Windows from the meshoptimizer project by
hand — this is a one-time manual step exactly like the model weights, and nothing
in the app may download it at runtime. Place:

- `vendor/gltfpack/gltfpack.exe`
- `vendor/gltfpack/LICENSE` (meshoptimizer is MIT)
- `vendor/gltfpack/VERSION` containing the release tag and the SHA-256 of the exe,
  one per line.

If `vendor/` is gitignored for binaries, check `.gitignore` and follow whatever
`vendor/trellis/` already does — do not change that policy here.

- [ ] **Step 2: Write the failing config/doctor test**

In `tests/test_doctor.py`:

```python
def test_gltfpack_check_is_non_fatal_when_missing(tmp_path, monkeypatch):
    from warlock import doctor
    from warlock.config import Config

    monkeypatch.setenv("WARLOCK_GLTFPACK", str(tmp_path / "nope.exe"))
    monkeypatch.setenv("WARLOCK_DATA_DIR", str(tmp_path))
    check = doctor._gltfpack_check(Config())
    assert check.ok is False
    assert check.fatal is False
```

- [ ] **Step 3: Run it and watch it fail**

Run: `uv run pytest tests/test_doctor.py -k gltfpack -v`
Expected: FAIL — `AttributeError: _gltfpack_check`

- [ ] **Step 4: Add the config field and the check**

In `src/warlock/config.py`, beside `trellis_server_exe`:

```python
    # Vendored like trellis-server.exe: a pinned native binary, never downloaded
    # at runtime. Missing it costs you the triangle budgets, not the app --
    # jobs then ship the raw reconstruction, which is what they did before.
    gltfpack_exe: Path = field(
        default_factory=lambda: _env_path(
            "WARLOCK_GLTFPACK", PROJECT_ROOT / "vendor" / "gltfpack" / "gltfpack.exe"
        )
    )
    # Default triangle profile for a new job. See pipelines/optimize.PROFILES.
    mesh_profile: str = field(
        default_factory=lambda: os.environ.get("WARLOCK_MESH_PROFILE", "standard")
    )
```

In `src/warlock/doctor.py`, add to `run_checks`' list (after `_birefnet_check`) and
define:

```python
def _gltfpack_check(config: Config) -> Check:
    ok = config.gltfpack_exe.exists()
    detail = (
        str(config.gltfpack_exe)
        if ok
        else f"not found at {config.gltfpack_exe} -- meshes ship at full reconstruction density"
    )
    return Check("gltfpack (mesh optimizer)", ok, detail, fatal=False)
```

- [ ] **Step 5: Run it**

Run: `uv run pytest tests/test_doctor.py -v`
Expected: PASS

- [ ] **Step 6: Write the failing optimizer test**

Create `tests/test_optimize.py`:

```python
from __future__ import annotations

import subprocess

import pytest

from warlock.pipelines import optimize


def test_profiles_cover_the_named_tiers():
    assert optimize.PROFILES["draft"] == 20_000
    assert optimize.PROFILES["standard"] == 50_000
    assert optimize.PROFILES["detailed"] == 100_000
    assert optimize.PROFILES["raw"] is None


def test_raw_profile_copies_without_invoking_the_exe(tmp_path, monkeypatch):
    def explode(*a, **k):
        raise AssertionError("gltfpack must not run for the raw profile")

    monkeypatch.setattr(subprocess, "run", explode)
    monkeypatch.setattr(optimize, "_triangles", lambda p: 7)
    src = tmp_path / "source.glb"
    src.write_bytes(b"glb")
    result = optimize.run(src, tmp_path / "model.glb", target_triangles=None,
                          exe=tmp_path / "missing.exe")
    assert (tmp_path / "model.glb").read_bytes() == b"glb"
    assert result["requested"] is None
    assert result["achieved"] == 7


def test_command_uses_the_documented_flags(tmp_path, monkeypatch):
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        # gltfpack writes its output; stand that in.
        from pathlib import Path

        Path(argv[argv.index("-o") + 1]).write_bytes(b"optimised")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(optimize, "_triangles", lambda p: 100_000 if p.name == "source.glb" else 50_000)
    exe = tmp_path / "gltfpack.exe"
    exe.write_bytes(b"")
    src = tmp_path / "source.glb"
    src.write_bytes(b"glb")
    result = optimize.run(src, tmp_path / "model.glb", target_triangles=50_000, exe=exe)

    argv = seen["argv"]
    assert argv[0] == str(exe)
    assert "-noq" in argv and "-ke" in argv and "-km" in argv
    assert argv[argv.index("-si") + 1] == "0.5"
    assert result["requested"] == 50_000
    assert result["achieved"] == 50_000
    assert (tmp_path / "model.glb").read_bytes() == b"optimised"


def test_a_failing_exe_raises_rather_than_leaving_a_stub(tmp_path, monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **k: subprocess.CompletedProcess(argv, 1, "", "boom"),
    )
    exe = tmp_path / "gltfpack.exe"
    exe.write_bytes(b"")
    src = tmp_path / "source.glb"
    src.write_bytes(b"glb")
    with pytest.raises(optimize.OptimizeError):
        optimize.run(src, tmp_path / "model.glb", target_triangles=50_000, exe=exe)
    assert not (tmp_path / "model.glb").exists()
```

- [ ] **Step 7: Run them and watch them fail**

Run: `uv run pytest tests/test_optimize.py -v`
Expected: FAIL — `ModuleNotFoundError: warlock.pipelines.optimize`

- [ ] **Step 8: Write the module**

Create `src/warlock/pipelines/optimize.py`:

```python
"""Retarget a reconstruction to a triangle budget with a vendored gltfpack.

The trellis response is ~290k triangles and 22 MB, which is a source mesh, not a
game asset. gltfpack simplifies it without re-running the reconstruction, which
is the whole point: a re-target is a two-second subprocess, and a trellis run is
two minutes of GPU.

The flags are not negotiable and each earns its place:

* ``-si <ratio>`` -- the simplification ratio. gltfpack takes a ratio, not a
  triangle count, so the caller's budget is divided by the source count here.
* ``-noq`` -- no quantisation. Quantised attributes need KHR_mesh_quantization,
  which some importers list as required and refuse the file over.
* ``-ke`` / ``-km`` -- keep extras and materials. Without them the material
  assignment (and therefore both PBR textures) can be dropped on merge.

Like ``trellis-server.exe`` the binary is vendored and pinned; nothing here
downloads anything. Missing it is not fatal -- the ``raw`` profile is always
available and is what every job did before this existed.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Named budgets. None means "ship the reconstruction untouched".
PROFILES: dict[str, int | None] = {
    "draft": 20_000,
    "standard": 50_000,
    "detailed": 100_000,
    "raw": None,
}

CUSTOM_MIN = 5_000
CUSTOM_MAX = 200_000

DEFAULT_TIMEOUT = 300.0


class OptimizeError(RuntimeError):
    """gltfpack was missing, failed, timed out, or produced an unusable file."""


def run(
    source: Path,
    dest: Path,
    *,
    target_triangles: int | None,
    exe: Path,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict[str, Any]:
    """Write an optimized copy of ``source`` to ``dest``.

    ``dest`` is only created on success -- a half-written or rejected output must
    never end up as the model the user downloads.
    """
    source_triangles = _triangles(source)
    if target_triangles is None or source_triangles <= target_triangles:
        # Already inside the budget, or no budget asked for. Copying is honest:
        # running the simplifier to a ratio above 1.0 is a no-op that still
        # re-encodes the file.
        shutil.copyfile(source, dest)
        return {
            "requested": target_triangles,
            "achieved": _triangles(dest),
            "source_triangles": source_triangles,
            "bytes": dest.stat().st_size,
        }
    if not exe.exists():
        raise OptimizeError(
            f"gltfpack not found at {exe}; use the 'raw' profile or set WARLOCK_GLTFPACK"
        )

    ratio = max(min(target_triangles / max(source_triangles, 1), 1.0), 0.0)
    tmp = dest.with_suffix(".glb.opt.tmp")
    argv = [
        str(exe),
        "-i", str(source),
        "-o", str(tmp),
        "-si", f"{ratio:g}",
        "-noq",
        "-ke",
        "-km",
    ]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except subprocess.TimeoutExpired as exc:
        tmp.unlink(missing_ok=True)
        raise OptimizeError(f"gltfpack timed out after {timeout:.0f}s") from exc
    if proc.returncode != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise OptimizeError(
            f"gltfpack exited {proc.returncode}: {(proc.stderr or proc.stdout)[:500]}"
        )

    achieved = _triangles(tmp)
    if achieved <= 0:
        tmp.unlink(missing_ok=True)
        raise OptimizeError("gltfpack produced a mesh with no triangles")
    tmp.replace(dest)
    log.info(
        "optimized %s: %d -> %d triangles (asked %d)",
        source.name, source_triangles, achieved, target_triangles,
    )
    return {
        "requested": target_triangles,
        "achieved": achieved,
        "source_triangles": source_triangles,
        "bytes": dest.stat().st_size,
    }


def resolve(profile: str, custom: int | None = None) -> int | None:
    """A profile name (or 'custom' plus a count) -> a triangle budget."""
    if profile == "custom":
        if custom is None or not CUSTOM_MIN <= custom <= CUSTOM_MAX:
            raise ValueError(f"custom triangles must be {CUSTOM_MIN}-{CUSTOM_MAX}")
        return custom
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; expected one of {sorted(PROFILES)}")
    return PROFILES[profile]


def _triangles(path: Path) -> int:
    import trimesh

    loaded = trimesh.load(path, process=False)
    mesh = loaded.to_mesh() if isinstance(loaded, trimesh.Scene) else loaded
    return int(len(mesh.faces))
```

- [ ] **Step 9: Run the tests**

Run: `uv run pytest tests/test_optimize.py -v`
Expected: PASS

- [ ] **Step 10: Write the failing worker test**

In `tests/test_queue.py`:

```python
@pytest.mark.asyncio
async def test_trellis_output_is_kept_as_source_glb(worker_env, monkeypatch):
    worker, store = worker_env
    monkeypatch.setattr(
        "warlock.pipelines.optimize.run",
        lambda source, dest, **k: (dest.write_bytes(source.read_bytes()),
                                   {"requested": 50_000, "achieved": 50_000,
                                    "source_triangles": 90_000, "bytes": 1})[1],
    )
    job_id = store.create("text", "a barrel", {"seed": 1, "profile": "standard"})
    await _run_until_done(worker, store, job_id)
    job_dir = worker.config.job_dir(job_id)
    assert (job_dir / "source.glb").exists()
    assert (job_dir / "model.glb").exists()
    assert store.get(job_id)["params"]["optimize"]["achieved"] == 50_000
```

- [ ] **Step 11: Rewire the worker**

In `src/warlock/queue.py:_generate`, replace the trellis call and what follows:

```python
        # The reconstruction is kept verbatim as source.glb and never
        # overwritten: model.glb is derived from it, so re-targeting a triangle
        # budget later (POST /api/jobs/{id}/optimize) never has to pay for
        # another trellis run.
        source_glb = job_dir / "source.glb"
        glb_path = job_dir / "model.glb"
        _log_vram("before trellis generate")
        await self.trellis.generate(
            image_path,
            source_glb,
            seed=mesh_seed,
            resolution=resolution,
            bg_removal=str(params.get("bg_removal") or "auto"),
        )
        await self._optimize(job_id, source_glb, glb_path, params)
        await self._apply_scale(job_id, glb_path, params)
        await self._audit_mesh(job_id, glb_path, params)
```

and add the method beside `_apply_scale`:

```python
    async def _optimize(
        self, job_id: str, source: Path, dest: Path, params: dict[str, Any]
    ) -> None:
        """Retarget the reconstruction to the job's triangle budget.

        Before the transform, not after: gltfpack rewrites the node graph, and
        running it over an already-grounded model would discard the transform
        node normalize_glb inserted. Optimizing first and transforming second is
        the only ordering where both survive.

        A failure here is not fatal. The reconstruction is on disk and usable;
        losing the budget costs the user file size, and failing the job would
        cost them the mesh.
        """
        if self._cancel is not None and self._cancel.event.is_set():
            return
        from .pipelines import optimize

        try:
            budget = optimize.resolve(
                str(params.get("profile") or self.config.mesh_profile),
                params.get("custom_triangles"),
            )
        except ValueError:
            log.warning("job %s has an unusable profile; shipping raw", job_id)
            budget = None
        self.progress.update(
            job_id, phase="optimize", label="Optimizing mesh", inner=0.0,
            inner_next=1.0, nominal=4.0, detail=f"{budget:,} tris" if budget else "raw",
        )
        try:
            result = await asyncio.to_thread(
                functools.partial(
                    optimize.run,
                    source,
                    dest,
                    target_triangles=budget,
                    exe=self.config.gltfpack_exe,
                )
            )
        except Exception:
            log.exception("optimize failed for job %s; shipping the reconstruction", job_id)
            await asyncio.to_thread(shutil.copyfile, source, dest)
            return
        params["optimize"] = result
        await asyncio.to_thread(self.store.set_params, job_id, params)
```

Add `import shutil` to `queue.py`'s imports.

- [ ] **Step 12: Run the tests**

Run: `uv run pytest -q && uv run ruff check .`
Expected: all green. Some existing `test_queue.py` assertions expect trellis to
write straight to `model.glb` — update those to `source.glb`; that is the intended
contract change, and `_discard_artifacts` must now delete **both**:

```python
            paths = [
                self.config.job_dir(job["id"]) / "model.glb",
                self.config.job_dir(job["id"]) / "source.glb",
            ]
```

- [ ] **Step 13: Add the optimize route**

In `app.py`, add the form field `profile` to `create_job` (validated with
`optimize.resolve`, 400 on error, stored as `params["profile"]`), and a route:

```python
    @app.post("/api/jobs/{job_id}/optimize")
    async def optimize_job(
        job_id: str,
        profile: Annotated[str, Form()] = "standard",
        custom_triangles: Annotated[int | None, Form()] = None,
    ) -> dict[str, Any]:
        """Rebuild model.glb from source.glb at a different triangle budget.

        Inline in the request rather than on the queue, under the same
        per-artifact lock the STL/OBJ exports use: gltfpack is a two-second
        subprocess, and putting it behind the serial GPU queue would make it
        wait on a trellis run.
        """
        from .pipelines import optimize, postprocess

        _check_job_id(job_id)
        job = await asyncio.to_thread(store().get, job_id)
        if job is None:
            raise HTTPException(404, "no such job")
        job_dir = config.job_dir(job_id)
        source = job_dir / "source.glb"
        if not source.exists():
            raise HTTPException(400, "this job has no source reconstruction to re-optimize")
        try:
            budget = optimize.resolve(profile, custom_triangles)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        lock = _convert_locks.setdefault((job_id, "optimize"), asyncio.Lock())
        async with lock:
            try:
                result = await asyncio.to_thread(
                    functools.partial(
                        optimize.run,
                        source,
                        job_dir / "model.glb",
                        target_triangles=budget,
                        exe=config.gltfpack_exe,
                    )
                )
            except optimize.OptimizeError as exc:
                raise HTTPException(500, str(exc)) from exc
            # The optimizer rewrote the node graph, so the grounding transform
            # went with it and has to be reapplied.
            transform = await asyncio.to_thread(
                postprocess.normalize_glb,
                job_dir / "model.glb",
                float(job["params"]["size_m"]) if job["params"].get("size_m") else None,
            )
            # Derived artifacts describe the old mesh; drop them so the next
            # request rebuilds from the new one.
            for name in ("model.stl", "model_obj.zip", "model.fbx", "collision.glb", "textures.zip"):
                with contextlib.suppress(OSError):
                    (job_dir / name).unlink()

        params = dict(job["params"])
        params["profile"] = profile
        if custom_triangles is not None:
            params["custom_triangles"] = custom_triangles
        params["optimize"] = result
        params["transform"] = transform
        params["scale_factor"] = transform["scale"]
        await asyncio.to_thread(store().set_params, job_id, params)
        return {"ok": True, "optimize": result, "transform": transform}
```

Add `"source.glb": "model/gltf-binary",` to `_MEDIA` so the reconstruction stays
downloadable.

- [ ] **Step 14: Add a test for the route**

In `tests/test_api.py`:

```python
def test_optimize_requires_a_source(client):
    r = client.post("/api/jobs", data={"kind": "text", "prompt": "x"})
    assert client.post(f"/api/jobs/{r.json()['id']}/optimize").status_code == 400
```

- [ ] **Step 15: Qualify the tiers before exposing them**

NEXT.md §3 is explicit that a named tier stays hidden until it passes
qualification. Run three representative props through each of 20k / 50k / 100k —
a box-like chest, a thin sword, a rounded rock — and confirm each output keeps
positions, normals, UVs, base color, metallic/roughness, embedded PNG data,
material assignment, and loads with no *required* glTF extension. Record the
results in `docs/NEXT.md` under §3. Only add a tier to the UI select once its row
passes.

- [ ] **Step 16: Add the UI select**

In `static/index.html`, beside the platform row:

```html
      <div id="g-profile-row">
        <label for="g-profile">Triangle budget</label>
        <select id="g-profile">
          <option value="raw">Raw reconstruction</option>
        </select>
      </div>
```

Add each qualified tier as an `<option>` in Step 15's order. In `app.js`'s submit
handler: `fd.set("profile", document.getElementById("g-profile").value);`

- [ ] **Step 17: Commit**

```bash
git add -A
git commit -m "Warlock v0.0.1

Keep the reconstruction as source.glb and derive a budgeted model.glb."
```

---

### Task 4: Collision mesh (review item #10)

A convex hull derived with trimesh saves a per-asset Blender round-trip for every
physics object.

**Files:**
- Modify: `src/warlock/pipelines/postprocess.py`, `src/warlock/app.py:617-656`
- Test: `tests/test_postprocess.py`, `tests/test_api.py`

**Interfaces:**
- Produces: `postprocess.glb_to_collision(glb_path: Path, out_path: Path, *, max_hull_faces: int = 256) -> Path`
  writing a GLB containing one convex hull.
- Consumes: nothing beyond trimesh.

- [ ] **Step 1: Write the failing test**

In `tests/test_postprocess.py`:

```python
def test_collision_hull_is_convex_and_small(tmp_path):
    import trimesh

    from warlock.pipelines import postprocess

    # A sphere is the worst case for face count and the easiest convexity check.
    src = tmp_path / "m.glb"
    trimesh.Scene(trimesh.creation.icosphere(subdivisions=4)).export(src)

    out = postprocess.glb_to_collision(src, tmp_path / "collision.glb")

    hull = trimesh.load(out).to_mesh()
    assert hull.is_convex
    assert len(hull.faces) <= 256
    assert hull.is_watertight
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/test_postprocess.py -k collision -v`
Expected: FAIL — `AttributeError: glb_to_collision`

- [ ] **Step 3: Implement it**

In `src/warlock/pipelines/postprocess.py`:

```python
def glb_to_collision(glb_path: Path, out_path: Path, *, max_hull_faces: int = 256) -> Path:
    """A convex collision shape for the mesh, as its own GLB.

    A hull rather than a decomposition: every engine accepts a convex shape
    directly, hull generation is deterministic and takes milliseconds, and a
    proper decomposition needs a VHACD binary this project would have to vendor.
    For the props Warlock produces the hull is the shape a hand-made collider
    would have been anyway.

    Simplified to a face budget because a hull over a 290k-triangle
    reconstruction can still carry thousands of faces, and a physics engine
    pays for every one of them each frame.
    """
    mesh = _load_merged(glb_path)
    hull = mesh.convex_hull
    if len(hull.faces) > max_hull_faces:
        # Simplify then re-hull: simplification can push vertices inward and
        # make the result non-convex, and a collider that is not convex is
        # silently wrong rather than loudly broken.
        hull = hull.simplify_quadric_decimation(max_hull_faces).convex_hull
    data = trimesh.Scene(hull).export(file_type="glb")
    with _staged(out_path) as tmp:
        tmp.write_bytes(data)
    return out_path
```

If `simplify_quadric_decimation` is unavailable in the pinned trimesh, fall back to
`hull` unsimplified and lower `max_hull_faces` in the assertion — do not add a new
dependency for it.

- [ ] **Step 4: Run it**

Run: `uv run pytest tests/test_postprocess.py -k collision -v`
Expected: PASS

- [ ] **Step 5: Serve it lazily**

In `app.py`, add to `_MEDIA`:

```python
        "collision.glb": "model/gltf-binary",
```

and extend the lazy-convert branch in `get_file`:

```python
        derived = {
            "model.stl": postprocess.glb_to_stl,
            "model_obj.zip": postprocess.glb_to_obj_zip,
            "collision.glb": postprocess.glb_to_collision,
        }
        if not path.exists() and name in derived and glb.exists():
            from .pipelines import postprocess

            convert = derived[name]
            lock = _convert_locks.setdefault((job_id, name), asyncio.Lock())
            async with lock:
                if not path.exists():
                    await asyncio.to_thread(convert, glb, path)
```

Move the `from .pipelines import postprocess` import above the `derived` dict so the
functions resolve, and delete the old two-way `convert = ...` conditional.

- [ ] **Step 6: Add the API test**

In `tests/test_api.py`, mirror whatever existing test covers on-demand `model.stl`
generation, asserting `GET /api/jobs/{id}/files/collision.glb` returns 200 for a
job with a `model.glb`.

- [ ] **Step 7: Add the download link**

In `static/index.html`, beside `dl-stl`:

```html
    <a id="dl-collision" download>collision</a>
```

In `app.js:showSelected`:

```js
  document.getElementById("dl-collision").href = `/api/jobs/${job.id}/files/collision.glb`;
```

- [ ] **Step 8: Run everything and commit**

```bash
uv run pytest -q && uv run ruff check .
git add -A
git commit -m "Warlock v0.0.1

Derive a convex collision GLB on demand."
```

---

### Task 5: FBX export (review item #11)

Unity and Unreal want FBX. Blender already runs out-of-process; this is one more op.

**Files:**
- Modify: `src/warlock/pipelines/blender_worker.py` (add `op_fbx`, register in `OPS`)
- Modify: `src/warlock/rigging.py` (add `fbx_spec`)
- Modify: `src/warlock/app.py` (`_MEDIA` + lazy convert)
- Test: `tests/test_rig_worker.py` (spec shape), `tests/test_api.py`

**Interfaces:**
- Produces: `rigging.fbx_spec(source_glb: Path, out_fbx: Path, result_dir: Path) -> dict`
- Produces: `blender_worker.op_fbx(bpy, spec) -> {"ok": True, "objects": int}`
- Consumes: `rigging.run_worker` (unchanged).

- [ ] **Step 1: Write the failing spec test**

In `tests/test_rig_worker.py` (or `tests/test_rigging.py`, wherever the other
`*_spec` shape tests live):

```python
def test_fbx_spec_names_the_op_and_paths(tmp_path):
    from warlock import rigging

    spec = rigging.fbx_spec(tmp_path / "model.glb", tmp_path / "model.fbx", tmp_path)
    assert spec["op"] == "fbx"
    assert spec["source_glb"].endswith("model.glb")
    assert spec["out_fbx"].endswith("model.fbx")
    assert spec["result_path"].startswith(str(tmp_path))
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/ -k fbx_spec -v`
Expected: FAIL — `AttributeError: fbx_spec`

- [ ] **Step 3: Add the spec and the op**

In `src/warlock/rigging.py`, beside `pose_spec`:

```python
def fbx_spec(source_glb: Path, out_fbx: Path, result_dir: Path) -> dict[str, Any]:
    """The worker spec for converting a GLB to FBX.

    Blender is the converter because it is already here and already
    out-of-process; adding an FBX library to the app process would mean a second
    importer disagreeing with the one that produces every other artifact.
    """
    return {
        "op": "fbx",
        "source_glb": str(source_glb),
        "out_fbx": str(out_fbx),
        "result_path": str(result_dir / ".fbx_result.json"),
    }
```

In `src/warlock/pipelines/blender_worker.py`:

```python
def op_fbx(bpy: Any, spec: dict[str, Any]) -> dict[str, Any]:
    """Import a GLB and write it back out as FBX, skins and all."""
    source = Path(spec["source_glb"])
    if not source.exists():
        raise RuntimeError(f"nothing to convert at {source}")

    progress(0.10, "Loading model")
    _reset_scene(bpy)
    bpy.ops.import_scene.gltf(filepath=str(source), bone_heuristic="BLENDER")
    _purge_import_helpers(bpy)

    progress(0.60, "Writing FBX")
    out = Path(spec["out_fbx"])
    out.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.export_scene.fbx(
        filepath=str(out),
        use_selection=True,
        path_mode="COPY",
        embed_textures=True,
        # Unity and Unreal both read Y-up FBX; matching the GLB's axes means the
        # FBX and the GLB describe the same orientation rather than two.
        axis_forward="-Z",
        axis_up="Y",
        bake_anim=False,
    )
    progress(1.0, "FBX written")
    return {"ok": True, "objects": len(bpy.context.scene.objects)}
```

and register it: `OPS = {"rig": op_rig, "pose": op_pose, "sheet": op_sheet, "fbx": op_fbx}`.

- [ ] **Step 4: Run it**

Run: `uv run pytest tests/ -k fbx_spec -v`
Expected: PASS

- [ ] **Step 5: Serve it lazily**

In `app.py`, add `"model.fbx": "application/octet-stream",` to `_MEDIA`, and in
`get_file` add a branch before the trimesh-derived ones (it needs a subprocess, not
a trimesh call, so it does not fit the `derived` dict):

```python
        if name == "model.fbx" and not path.exists() and glb.exists():
            lock = _convert_locks.setdefault((job_id, name), asyncio.Lock())
            async with lock:
                if not path.exists():
                    spec = rigging.fbx_spec(glb, path, job_dir)
                    try:
                        await asyncio.to_thread(
                            functools.partial(
                                rigging.run_worker, spec, timeout=config.pose_timeout
                            )
                        )
                    except rigging.BlenderError as exc:
                        log.error("fbx export for %s failed: %s", job_id, exc)
                        raise HTTPException(500, "could not export FBX") from exc
```

FBX export is import-plus-export like a pose bake, so it reuses `pose_timeout`
rather than gaining a config knob of its own.

- [ ] **Step 6: Add the download link**

In `static/index.html` beside `dl-obj`: `<a id="dl-fbx" download>FBX</a>`, and in
`app.js:showSelected`:

```js
  const dlFbx = document.getElementById("dl-fbx");
  dlFbx.href = `/api/jobs/${job.id}/files/model.fbx`;
  // Blender does the conversion, so hide it when rigging isn't installed.
  dlFbx.hidden = !rig.available;
```

- [ ] **Step 7: Run everything and commit**

```bash
uv run pytest -q && uv run ruff check .
git add -A
git commit -m "Warlock v0.0.1

Export FBX via the Blender worker."
```

---

### Task 6: Texture extraction (review item #12)

Textures only ride inside the GLB or the OBJ zip. A `textures.zip` makes
engine-side material work and palette edits possible.

**Files:**
- Modify: `src/warlock/pipelines/postprocess.py`, `src/warlock/app.py`
- Test: `tests/test_postprocess.py`

**Interfaces:**
- Produces: `postprocess.glb_to_textures_zip(glb_path: Path, zip_path: Path) -> Path`,
  writing `base_color.png` and `metallic_roughness.png` (and `normal.png` when present).

- [ ] **Step 1: Write the failing test**

In `tests/test_postprocess.py`:

```python
def test_textures_zip_contains_the_pbr_maps(tmp_path):
    import zipfile

    import numpy as np
    import trimesh
    from PIL import Image

    from warlock.pipelines import postprocess

    mesh = trimesh.creation.box(extents=(1, 1, 1))
    mesh.visual = trimesh.visual.TextureVisuals(
        uv=np.zeros((len(mesh.vertices), 2)),
        material=trimesh.visual.material.PBRMaterial(
            baseColorTexture=Image.new("RGB", (8, 8), (255, 0, 0))
        ),
    )
    src = tmp_path / "m.glb"
    trimesh.Scene(mesh).export(src)

    out = postprocess.glb_to_textures_zip(src, tmp_path / "textures.zip")

    with zipfile.ZipFile(out) as zf:
        assert "base_color.png" in zf.namelist()


def test_textures_zip_on_an_untextured_mesh_raises(tmp_path):
    import pytest
    import trimesh

    from warlock.pipelines import postprocess

    src = tmp_path / "m.glb"
    trimesh.Scene(trimesh.creation.box()).export(src)
    with pytest.raises(ValueError):
        postprocess.glb_to_textures_zip(src, tmp_path / "textures.zip")
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_postprocess.py -k textures -v`
Expected: FAIL — `AttributeError: glb_to_textures_zip`

- [ ] **Step 3: Implement it**

In `src/warlock/pipelines/postprocess.py`:

```python
# glTF PBR slot -> the filename it gets in the zip. Named for what the map *is*
# rather than what the GLB called it, because a texture pulled out for editing
# ends up in an engine's material slot, not back in a glTF.
_TEXTURE_SLOTS = {
    "baseColorTexture": "base_color.png",
    "metallicRoughnessTexture": "metallic_roughness.png",
    "normalTexture": "normal.png",
    "emissiveTexture": "emissive.png",
    "occlusionTexture": "occlusion.png",
}


def glb_to_textures_zip(glb_path: Path, zip_path: Path) -> Path:
    """Every PBR map in the GLB, as loose PNGs in a zip.

    Raises ValueError when there is nothing to extract: an empty zip looks like
    a successful export of a model with no textures, which is a different and
    much more alarming thing than "this GLB has no maps".
    """
    mesh = _load_merged(glb_path)
    material = getattr(getattr(mesh, "visual", None), "material", None)
    found: list[tuple[str, Any]] = []
    for slot, filename in _TEXTURE_SLOTS.items():
        image = getattr(material, slot, None)
        if image is not None:
            found.append((filename, image))
    if not found:
        raise ValueError("this model has no textures to extract")

    import io

    with _staged(zip_path) as tmp, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, image in found:
            buf = io.BytesIO()
            image.save(buf, "PNG")
            zf.writestr(filename, buf.getvalue())
    return zip_path
```

- [ ] **Step 4: Run it**

Run: `uv run pytest tests/test_postprocess.py -k textures -v`
Expected: PASS

- [ ] **Step 5: Serve it**

Add `"textures.zip": "application/zip",` to `_MEDIA` and
`"textures.zip": postprocess.glb_to_textures_zip,` to the `derived` dict from
Task 4. A `ValueError` from an untextured mesh will surface as a 500 — wrap the
`await asyncio.to_thread(convert, glb, path)` call:

```python
                if not path.exists():
                    try:
                        await asyncio.to_thread(convert, glb, path)
                    except ValueError as exc:
                        raise HTTPException(404, str(exc)) from exc
```

- [ ] **Step 6: Add the link, run everything, commit**

`<a id="dl-textures" download>textures</a>` in `index.html`, its `href` in
`showSelected`, then:

```bash
uv run pytest -q && uv run ruff check .
git add -A
git commit -m "Warlock v0.0.1

Extract PBR textures as a zip."
```

---

### Task 7: Bulk export and save-to-project (review item #13)

One file per request today. A batch download and an optional "save into my Godot
project" action close the last manual step.

**Files:**
- Modify: `src/warlock/config.py` (`export_dir`), `src/warlock/app.py`
- Modify: `src/warlock/static/app.js`, `index.html`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: the lazy-convert artifacts from Tasks 4–6.
- Produces: `POST /api/export` with form fields `ids` (repeated) and `files`
  (repeated, defaulting to `model.glb`), streaming a zip;
  `POST /api/export/folder` with the same fields, copying into `Config.export_dir`
  and returning `{"copied": int, "dir": str}`.
- Produces: `Config.export_dir: Path | None` (env `WARLOCK_EXPORT_DIR`).

- [ ] **Step 1: Write the failing test**

In `tests/test_api.py`:

```python
def test_bulk_export_zips_the_selected_jobs(client, tmp_path):
    import io
    import zipfile

    import warlock.config as config_mod

    ids = []
    for _ in range(2):
        job_id = client.post("/api/jobs", data={"kind": "text", "prompt": "x"}).json()["id"]
        job_dir = config_mod.get_config().job_dir(job_id)
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "model.glb").write_bytes(b"glb")
        ids.append(job_id)

    r = client.post("/api/export", data=[("ids", i) for i in ids])
    assert r.status_code == 200
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = zf.namelist()
    assert len(names) == 2
    assert all(n.endswith("model.glb") for n in names)


def test_bulk_export_rejects_an_unknown_file_name(client):
    r = client.post("/api/export", data={"ids": "0" * 12, "files": "../secrets"})
    assert r.status_code == 400


def test_export_to_folder_is_404_when_unconfigured(client):
    assert client.post("/api/export/folder", data={"ids": "0" * 12}).status_code == 404
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_api.py -k export -v`
Expected: FAIL — 404 on `/api/export`

- [ ] **Step 3: Add the config field**

In `src/warlock/config.py`:

```python
    # Optional: a project folder assets can be copied straight into (e.g. a
    # Godot project's assets/). Unset means the feature is off and its routes
    # 404 -- writing outside data_dir is opt-in, never a default.
    export_dir: Path | None = field(
        default_factory=lambda: (
            _env_path("WARLOCK_EXPORT_DIR", PROJECT_ROOT)
            if os.environ.get("WARLOCK_EXPORT_DIR")
            else None
        )
    )
```

- [ ] **Step 4: Add the routes**

In `app.py`, near the other job routes:

```python
    # Anything a caller may name in a bulk export. Exactly the _MEDIA key set:
    # the point of the allowlist is that a name never becomes a path component
    # without passing through it first.
    def _export_names(files: list[str] | None) -> list[str]:
        names = [f for f in (files or []) if f] or ["model.glb"]
        unknown = [n for n in names if n not in _MEDIA]
        if unknown:
            raise HTTPException(400, f"unknown file(s): {sorted(unknown)}")
        return names

    def _collect(ids: list[str], names: list[str]) -> list[tuple[str, Path]]:
        """-> (arcname, path) for every requested file that exists.

        Silently skips what is missing rather than failing the batch: a
        selection of ten jobs where one never produced an OBJ should still
        deliver nine, and the zip's contents say which.
        """
        out: list[tuple[str, Path]] = []
        for job_id in ids:
            _check_job_id(job_id)
            for name in names:
                path = config.job_dir(job_id) / name
                if path.exists():
                    out.append((f"{job_id}/{name}", path))
        return out

    @app.post("/api/export")
    async def bulk_export(
        ids: Annotated[list[str], Form()],
        files: Annotated[list[str] | None, Form()] = None,
    ) -> FileResponse:
        """Zip the named artifacts of several jobs into one download.

        Built into a temp file rather than streamed from memory: a selection of
        twenty 22 MB GLBs is not something to hold in the event loop's heap.
        Derived artifacts are *not* generated on demand here -- a batch export
        should not be able to kick off twenty Blender subprocesses.
        """
        names = _export_names(files)
        members = await asyncio.to_thread(_collect, ids, names)
        if not members:
            raise HTTPException(404, "nothing to export")
        fd, raw = tempfile.mkstemp(dir=config.data_dir, prefix=".export.", suffix=".zip")
        os.close(fd)
        archive = Path(raw)

        def build() -> None:
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
                for arcname, path in members:
                    zf.write(path, arcname)

        await asyncio.to_thread(build)
        return FileResponse(
            archive,
            media_type="application/zip",
            filename="warlock_export.zip",
            background=BackgroundTask(archive.unlink, missing_ok=True),
        )

    @app.post("/api/export/folder")
    async def export_to_folder(
        ids: Annotated[list[str], Form()],
        files: Annotated[list[str] | None, Form()] = None,
    ) -> dict[str, Any]:
        """Copy the same selection into WARLOCK_EXPORT_DIR."""
        if config.export_dir is None:
            raise HTTPException(404, "no export folder configured (set WARLOCK_EXPORT_DIR)")
        names = _export_names(files)
        members = await asyncio.to_thread(_collect, ids, names)
        if not members:
            raise HTTPException(404, "nothing to export")

        def copy() -> int:
            copied = 0
            for arcname, path in members:
                dest = config.export_dir / arcname
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)
                copied += 1
            return copied

        return {"copied": await asyncio.to_thread(copy), "dir": str(config.export_dir)}
```

Add to `app.py`'s imports: `os`, `tempfile`, `zipfile`, and
`from starlette.background import BackgroundTask`.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_api.py -k export -v`
Expected: PASS

- [ ] **Step 6: Add the UI**

Add a checkbox to each job card in `createNode` (`const pick = document.createElement("input"); pick.type = "checkbox"; pick.className = "job-pick";`,
appended into `actions`, with `e.stopPropagation()` on its click), and a toolbar
above the list in `index.html`:

```html
  <div id="bulk-bar" hidden>
    <span id="bulk-count"></span>
    <button type="button" id="bulk-zip">download zip</button>
    <button type="button" id="bulk-folder" hidden>save to project</button>
  </div>
```

In `app.js`, collect the checked ids and POST them:

```js
function selectedIds() {
  return [...nodes].filter(([, n]) => n.pick.checked).map(([id]) => id);
}

document.getElementById("bulk-zip").addEventListener("click", () => {
  const ids = selectedIds();
  if (!ids.length) return;
  // A form POST rather than fetch: the browser then handles the download
  // itself, including the filename, instead of us buffering a 200 MB blob.
  const form = document.createElement("form");
  form.method = "POST";
  form.action = "/api/export";
  for (const id of ids) {
    const input = document.createElement("input");
    input.type = "hidden";
    input.name = "ids";
    input.value = id;
    form.append(input);
  }
  document.body.append(form);
  form.submit();
  form.remove();
});
```

Reveal `bulk-folder` only when `/api/health` reports an export dir — add
`"export_dir": str(config.export_dir) if config.export_dir else None` to the health
payload and check it in the health poll.

- [ ] **Step 7: Run everything and commit**

```bash
uv run pytest -q && uv run ruff check .
git add -A
git commit -m "Warlock v0.0.1

Bulk export as zip, and optional save-to-project-folder."
```

---

## Plan B self-review notes

- Items #7–#13 each map to a task: #9→T1, #8→T2, #7→T3, #10→T4, #11→T5, #12→T6, #13→T7.
- Ordering is load-bearing in two places: optimize runs *before* normalize (T3 Step 11),
  and the `/optimize` route reapplies the transform because gltfpack rewrites the node
  graph (T3 Step 13).
- `scale_glb` survives as a wrapper so no existing caller or test breaks.
- Task 3 changes the on-disk contract (`source.glb` + derived `model.glb`); the
  `_discard_artifacts` and `_MEDIA` updates in that task are what keep cancel and
  download correct afterwards.
