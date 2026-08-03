"""Skeleton templates, rig/pose validation, and the Blender subprocess boundary.

This module is the *host* side of rigging and never imports ``bpy``. That split
is deliberate and load-bearing:

* ``bpy`` is process-global and not thread-safe, and it hard-crashes (not
  raises) on some degenerate geometry -- which trellis output frequently is.
  Importing it into the app process would put a segfault between the user and
  every other job. So Blender work runs out-of-process, mirroring how
  ``pipelines/trellis.py`` treats ``trellis-server.exe``.
* Everything decidable without Blender -- which templates exist, whether a pose
  payload is well-formed, where a template's joints land on a given bbox -- is
  decidable here, in plain Python, under test, with no 400 MB import.

``pipelines/blender_worker.py`` is the other side: it runs as
``python -m warlock.pipelines.blender_worker`` in this same interpreter and
imports :func:`fit_template` from here, so host and worker can never disagree
about where a joint goes.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).parent / "templates"

# Same shape and generator as a job id (uuid4().hex[:12]), and validated for the
# same reason: config.job_dir() and every path built under it do no sanitisation.
RESOURCE_ID_RE = re.compile(r"^[0-9a-f]{12}$")

# Progress lines the worker prints. Anything else on its stdout is log noise.
PROGRESS_PREFIX = "[blender]"
RE_PROGRESS = re.compile(r"^\[blender\]\s+([\d.]+)\s+(.*)$")

# A bone shorter than this fraction of the mesh's largest dimension is one
# Blender silently deletes on leaving edit mode, taking its children with it.
MIN_BONE_FRACTION = 0.002

# Rigging a 300k-face mesh with automatic weights is minutes of CPU, not hours.
BLENDER_TIMEOUT = 1800.0

# Tail of the worker's output kept for the error message when it fails.
ERROR_TAIL_CHARS = 2000


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def is_valid_id(value: str) -> bool:
    return bool(RESOURCE_ID_RE.match(value))


# --- template registry ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Template:
    key: str
    label: str
    root: str
    bones: tuple[dict[str, Any], ...]
    mirror_pairs: tuple[tuple[str, str], ...]


_templates: dict[str, Template] | None = None


def _load_templates() -> dict[str, Template]:
    found: dict[str, Template] = {}
    for path in sorted(TEMPLATE_DIR.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            found[raw["key"]] = _parse_template(raw)
        except Exception:
            # A malformed template must cost you that template, not the app.
            log.exception("skipping unusable skeleton template %s", path)
    return found


def _parse_template(raw: dict[str, Any]) -> Template:
    bones = raw["bones"]
    names = {b["name"] for b in bones}
    if len(names) != len(bones):
        raise ValueError("duplicate bone names")
    for b in bones:
        if b["parent"] is not None and b["parent"] not in names:
            raise ValueError(f"bone {b['name']!r} has unknown parent {b['parent']!r}")
        for end in ("head", "tail"):
            if len(b[end]) != 3:
                raise ValueError(f"bone {b['name']!r} {end} is not a 3-vector")
    if raw["root"] not in names:
        raise ValueError(f"root {raw['root']!r} is not a bone")
    return Template(
        key=raw["key"],
        label=raw["label"],
        root=raw["root"],
        bones=tuple(
            {
                "name": b["name"],
                "parent": b["parent"],
                "head": [float(v) for v in b["head"]],
                "tail": [float(v) for v in b["tail"]],
            }
            for b in bones
        ),
        mirror_pairs=tuple((a, b) for a, b in raw.get("mirror_pairs", [])),
    )


def templates() -> dict[str, Template]:
    global _templates
    if _templates is None:
        _templates = _load_templates()
    return _templates


def get_template(key: str) -> Template:
    try:
        return templates()[key]
    except KeyError:
        raise ValueError(f"unknown skeleton template {key!r}") from None


def catalog() -> list[dict[str, str]]:
    """The template table in the same {key, label} shape the UI selects use."""
    return [{"key": t.key, "label": t.label} for t in templates().values()]


# --- shipped pose libraries -------------------------------------------------
#
# A pose is a bone-name -> local-quaternion map, and fit_template puts a given
# template's bones in the same place on every mesh. So a pose authored against
# one humanoid rig applies to every other humanoid rig -- which is what makes a
# shipped library possible at all, and why these live next to the templates
# rather than being seeded into each job's poses/ directory.

PRESET_DIR = TEMPLATE_DIR / "poses"

_presets: dict[str, list[dict[str, Any]]] | None = None


def _load_presets() -> dict[str, list[dict[str, Any]]]:
    found: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(PRESET_DIR.glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            found[path.stem] = [
                {"name": str(p["name"]), "bones": p["bones"]} for p in raw["poses"]
            ]
        except Exception:
            # A malformed library costs you that library, not the app -- the
            # same rule _load_templates follows.
            log.exception("skipping unusable pose library %s", path)
    return found


def preset_poses(template_key: str) -> list[dict[str, Any]]:
    """The shipped poses for a template. Raises ValueError on an unknown key."""
    global _presets
    get_template(template_key)  # validates the key against the registry
    if _presets is None:
        _presets = _load_presets()
    return [dict(p) for p in _presets.get(template_key, [])]


# --- fitting ----------------------------------------------------------------

Vec3 = Sequence[float]


def fit_template(
    template: Template, bounds_min: Vec3, bounds_max: Vec3
) -> list[dict[str, Any]]:
    """Scale a template's normalized landmarks onto a mesh's bounding box.

    Normalized coordinates use Blender axes: x and y span -0.5..0.5 about the
    bbox centre, z spans 0 (bbox floor) to 1 (bbox ceiling). This is
    bbox-proportional, not learned, so it is approximate on unusual
    silhouettes -- the fitted positions are written into ``rig.json`` precisely
    so a later "adjust joints" pass can correct them without re-rigging from
    scratch.
    """
    size = [float(hi) - float(lo) for lo, hi in zip(bounds_min, bounds_max, strict=True)]
    largest = max(size) if max(size) > 0 else 1.0
    # A flat axis (a billboard-thin mesh, or a subject with no measurable depth)
    # would collapse every bone onto a plane and produce zero-length bones.
    # Borrowing the largest dimension keeps the skeleton three-dimensional; the
    # rig is approximate either way and a zero-length bone is not recoverable.
    size = [s if s > largest * 1e-4 else largest for s in size]
    centre = [(float(lo) + float(hi)) / 2.0 for lo, hi in zip(bounds_min, bounds_max, strict=True)]
    floor_z = float(bounds_min[2])
    min_len = largest * MIN_BONE_FRACTION

    def place(p: Vec3) -> list[float]:
        return [
            centre[0] + p[0] * size[0],
            centre[1] + p[1] * size[1],
            floor_z + p[2] * size[2],
        ]

    fitted: list[dict[str, Any]] = []
    for bone in template.bones:
        head = place(bone["head"])
        tail = place(bone["tail"])
        if _distance(head, tail) < min_len:
            tail = [head[0], head[1], head[2] + min_len]
        fitted.append({"name": bone["name"], "parent": bone["parent"], "head": head, "tail": tail})
    return fitted


def _distance(a: Vec3, b: Vec3) -> float:
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b, strict=True)) ** 0.5


def validate_joints(payload: dict[str, Any], template: Template) -> list[dict[str, Any]]:
    """Normalize a corrected skeleton, or raise ValueError.

    The whole skeleton, not a patch: a partial correction would leave the caller
    and the worker disagreeing about which joints came from the fit and which
    from the user, and the fitted positions are already in rig.json for the
    editor to start from. Parentage comes from the template and is never
    caller-supplied -- a client cannot restructure the hierarchy, only move it.

    A zero-length bone is rejected rather than nudged: Blender silently deletes
    one on leaving edit mode and takes its children with it, so accepting it
    would produce a rig missing limbs with nothing to explain why.
    """
    raw = payload.get("bones")
    if not isinstance(raw, list) or not raw:
        raise ValueError("joints payload requires a non-empty 'bones' list")
    by_name: dict[str, dict[str, Any]] = {}
    for entry in raw:
        name = str(entry.get("name") or "")
        points: dict[str, list[float]] = {}
        for end in ("head", "tail"):
            point = entry.get(end)
            if not isinstance(point, (list, tuple)) or len(point) != 3:
                raise ValueError(f"bone {name!r} {end} must be a 3-vector")
            try:
                points[end] = [float(v) for v in point]
            except (TypeError, ValueError):
                raise ValueError(f"bone {name!r} {end} is not numeric") from None
        by_name[name] = points

    expected = [b["name"] for b in template.bones]
    missing = [n for n in expected if n not in by_name]
    if missing:
        raise ValueError(f"joints payload is missing bone(s): {missing}")
    unknown = [n for n in by_name if n not in expected]
    if unknown:
        raise ValueError(f"joints payload names unknown bone(s): {unknown}")

    # The skeleton's own extent, so the minimum bone length scales with the
    # mesh: MIN_BONE_FRACTION is a fraction of the subject, not of a metre.
    heads = [by_name[n]["head"] for n in expected]
    span = max(
        (hi - lo)
        for lo, hi in (
            (min(p[axis] for p in heads), max(p[axis] for p in heads)) for axis in range(3)
        )
    ) or 1.0
    out = []
    for bone in template.bones:
        entry = by_name[bone["name"]]
        if _distance(entry["head"], entry["tail"]) < span * MIN_BONE_FRACTION:
            raise ValueError(f"bone {bone['name']!r} would be zero-length")
        out.append(
            {
                "name": bone["name"],
                "parent": bone["parent"],
                "head": entry["head"],
                "tail": entry["tail"],
            }
        )
    return out


# --- pose payloads ----------------------------------------------------------

MAX_POSE_NAME = 64
QUAT_EPSILON = 1e-3


def validate_pose(
    payload: dict[str, Any], known_bones: Sequence[str] | None = None
) -> dict[str, Any]:
    """Normalize a client pose payload, or raise ValueError.

    A pose is a bone -> local rotation quaternion map. Quaternions are stored
    XYZW, matching three.js's ``Quaternion.toArray()``, because the browser is
    the only thing that ever produces one; the Blender worker converts to
    WXYZ on the way in.
    """
    name = str(payload.get("name") or "").strip()
    if not name:
        raise ValueError("pose requires a name")
    if len(name) > MAX_POSE_NAME:
        raise ValueError(f"pose name must be at most {MAX_POSE_NAME} characters")

    raw = payload.get("bones")
    if not isinstance(raw, dict) or not raw:
        raise ValueError("pose requires a non-empty 'bones' map")
    allowed = set(known_bones) if known_bones is not None else None

    bones: dict[str, list[float]] = {}
    for bone, quat in raw.items():
        if allowed is not None and bone not in allowed:
            raise ValueError(f"unknown bone {bone!r}")
        if not isinstance(quat, (list, tuple)) or len(quat) != 4:
            raise ValueError(f"bone {bone!r} rotation must be a 4-element quaternion")
        try:
            values = [float(v) for v in quat]
        except (TypeError, ValueError):
            raise ValueError(f"bone {bone!r} rotation is not numeric") from None
        norm = sum(v * v for v in values) ** 0.5
        if abs(norm - 1.0) > QUAT_EPSILON:
            # Renormalize rather than reject: a browser accumulating gizmo
            # drags drifts off the unit sphere by ~1e-7 per drag, and refusing
            # the save over float noise would be indefensible. Only a
            # degenerate zero quaternion is unrecoverable.
            if norm < QUAT_EPSILON:
                raise ValueError(f"bone {bone!r} rotation is degenerate")
            values = [v / norm for v in values]
        bones[bone] = values
    return {"name": name, "bones": bones}


def mirror_quaternion(q: Sequence[float]) -> list[float]:
    """Reflect a local joint rotation across the subject's YZ plane.

    Every template is symmetric about X (the mirror normal), and reflecting a
    rotation conjugates it: the components perpendicular to the normal flip
    sign and the one along it does not. So (x, y, z, w) -> (x, -y, -z, w).

    Lives here rather than only in the browser because it is the kind of sign
    convention that is wrong in a way you cannot see -- a mirrored arm that
    rotates the wrong way about one axis still looks plausible in a static
    pose. The JS copy in app.js must stay identical to this.
    """
    x, y, z, w = (float(v) for v in q)
    return [x, -y, -z, w]


def mirror_pose(
    bones: dict[str, Any], pairs: Sequence[Sequence[str]]
) -> dict[str, list[float]]:
    """Copy every posed bone onto its mirror partner, reflected.

    Bones with no partner (a spine, a tail) are left exactly as they are: they
    sit on the mirror plane, so reflecting them would rotate a centred limb off
    centre.
    """
    partner: dict[str, str] = {}
    for a, b in pairs:
        partner[a] = b
        partner[b] = a
    out = {name: [float(v) for v in q] for name, q in bones.items()}
    for name, quat in bones.items():
        other = partner.get(name)
        if other is not None:
            out[other] = mirror_quaternion(quat)
    return out


# --- the Blender subprocess -------------------------------------------------


class BlenderError(RuntimeError):
    """The worker exited non-zero, timed out, or produced no result file."""


def run_worker(
    spec: dict[str, Any],
    *,
    on_progress: Callable[[float, str], None] | None = None,
    on_start: Callable[[subprocess.Popen[str]], None] | None = None,
    timeout: float = BLENDER_TIMEOUT,
) -> dict[str, Any]:
    """Run one Blender operation out-of-process and return its result JSON.

    Synchronous and blocking -- every caller in the app dispatches it through
    ``asyncio.to_thread``, like every other multi-second call in this codebase.

    ``spec`` is handed over on stdin; the worker writes its result to
    ``spec["result_path"]`` rather than stdout, so a stray print from bpy (and
    it does print) can never corrupt the payload.

    ``on_start`` receives the live ``Popen`` so a cancel can kill it. Like
    trellis-server, there is no polite abort: bpy is inside a C weighting solve
    and checks nothing, so killing the process is the only thing that stops it.
    """
    result_path = Path(spec["result_path"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.unlink(missing_ok=True)

    proc = subprocess.Popen(
        [sys.executable, "-m", "warlock.pipelines.blender_worker"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    tail: list[str] = []
    if on_start is not None:
        on_start(proc)
    # stdout is drained on a helper thread so the *whole* run has a deadline, not
    # just the wait() after EOF: a bpy process that hangs mid-solve (or mid-render)
    # produces no further output and never closes stdout, and reading it inline
    # would block past any timeout and wedge the serial job queue forever.
    lines: queue.Queue[str | None] = queue.Queue()

    def _pump(stream: Any) -> None:
        try:
            for raw in stream:
                lines.put(raw)
        finally:
            lines.put(None)

    assert proc.stdin is not None and proc.stdout is not None
    reader = threading.Thread(target=_pump, args=(proc.stdout,), daemon=True)
    reader.start()
    deadline = time.monotonic() + timeout
    try:
        proc.stdin.write(json.dumps(spec))
        proc.stdin.close()
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(proc.args, timeout)
            try:
                raw = lines.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if raw is None:
                break
            line = raw.rstrip()
            m = RE_PROGRESS.match(line)
            if m and on_progress is not None:
                on_progress(float(m.group(1)), m.group(2))
            tail.append(line)
            if len(tail) > 200:
                del tail[:100]
        code = proc.wait(timeout=max(deadline - time.monotonic(), 0.0))
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        result_path.unlink(missing_ok=True)
        raise BlenderError(f"Blender worker timed out after {timeout:.0f}s") from None
    finally:
        if proc.poll() is None:
            proc.kill()

    output = "\n".join(tail)[-ERROR_TAIL_CHARS:]
    if code != 0:
        # A killed-late worker may still have written the handoff file; it
        # would otherwise sit in the source job's directory until the next run.
        result_path.unlink(missing_ok=True)
        raise BlenderError(f"Blender worker exited with code {code}:\n{output}")
    if not result_path.exists():
        raise BlenderError(f"Blender worker wrote no result:\n{output}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    # The handoff file has served its purpose; a rig writes it into the source
    # job's directory, where it would otherwise sit next to model.glb forever.
    result_path.unlink(missing_ok=True)
    return payload


# The worker writes to these temp names; the queue renames them onto the
# served names on success (finalize_rig). Two reasons, both load-bearing:
# Blender's GLB export writes in place over seconds, and rig.json -- the
# completion gate _attach_files and the file route key on -- is already
# satisfied by an earlier rig during a re-rig, so pointing the worker at the
# served names would let a concurrent GET read a truncated rig.glb. It also
# makes cancellation naturally non-destructive: discarding a half-written
# re-rig deletes the temps, never the previous successful rig's artifacts.
# The suffixes are load-bearing: Blender's glTF exporter appends ".glb" to a
# filepath that does not already end in it, so a ".tmp" suffix would make the
# worker write ".rig.tmp.glb.glb" and finalize_rig find nothing.
RIG_GLB_TMP = ".rig.tmp.glb"
RIG_JSON_TMP = ".rig.tmp.json"


def rig_spec(
    job_dir: Path, template_key: str, bones: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """The worker spec for rigging a finished job's mesh.

    ``bones`` overrides the bbox-proportional fit with joints the user moved.
    The fit is deliberately approximate -- rig.json records what it chose so a
    correction can be made without re-running anything else.
    """
    get_template(template_key)  # fail here, not three seconds into a subprocess
    spec = {
        "op": "rig",
        "source_glb": str(job_dir / "model.glb"),
        "out_glb": str(job_dir / RIG_GLB_TMP),
        "out_json": str(job_dir / RIG_JSON_TMP),
        "result_path": str(job_dir / ".blender_result.json"),
        "template": template_key,
    }
    if bones is not None:
        spec["bones"] = bones
    return spec


def finalize_rig(job_dir: Path) -> None:
    """Rename the worker's temp artifacts onto the served names.

    GLB first, json second: rig.json is the completion marker, so at the
    moment it lands the rig.glb beside it is already the matching one.

    The renames retry briefly: on Windows, replacing rig.glb while a
    FileResponse is mid-stream from it raises PermissionError (files open
    without FILE_SHARE_DELETE), and failing an otherwise successful re-rig
    at the last step over a transient reader is the wrong trade.
    """
    for src, dest in ((RIG_GLB_TMP, "rig.glb"), (RIG_JSON_TMP, "rig.json")):
        for attempt in range(10):
            try:
                os.replace(job_dir / src, job_dir / dest)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.5)


def discard_rig_temps(job_dir: Path) -> None:
    """Remove whatever a failed or cancelled rig run left behind. Idempotent."""
    (job_dir / RIG_GLB_TMP).unlink(missing_ok=True)
    (job_dir / RIG_JSON_TMP).unlink(missing_ok=True)


def read_rig(job_dir: Path) -> dict[str, Any] | None:
    path = job_dir / "rig.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.exception("unreadable rig.json at %s", path)
        return None


def rig_bone_names(job_dir: Path) -> list[str] | None:
    rig = read_rig(job_dir)
    if rig is None:
        return None
    return [b["name"] for b in rig.get("bones", [])]


# --- pose storage -----------------------------------------------------------
#
# One file per pose under <job_dir>/poses/, not a single poses.json. A pose is
# saved from the browser while the queue may be writing that same job's
# directory, and per-file writes need no read-modify-write and so no lock; the
# derived <pose_id>.glb sits beside its json for the same reason.

POSE_DIR_NAME = "poses"

# A ceiling so a scripted client can't turn a job directory into a million
# files. Far above any plausible hand-authored set.
MAX_POSES = 500


def pose_dir(job_dir: Path) -> Path:
    return job_dir / POSE_DIR_NAME


def pose_path(job_dir: Path, pose_id: str) -> Path:
    """The record for one pose. Raises ValueError on an id that isn't ours.

    The guard is the point: this is the only place a caller-supplied pose id is
    turned into a path, and ``job_dir / pose_id`` does no sanitising of its own.
    """
    if not is_valid_id(pose_id):
        raise ValueError(f"malformed pose id {pose_id!r}")
    return pose_dir(job_dir) / f"{pose_id}.json"


def pose_glb_path(job_dir: Path, pose_id: str) -> Path:
    return pose_path(job_dir, pose_id).with_suffix(".glb")


def save_pose(job_dir: Path, pose: dict[str, Any], pose_id: str | None = None) -> dict[str, Any]:
    """Write a validated pose payload and return the stored record.

    Passing an existing ``pose_id`` overwrites it in place -- that is how the
    editor saves an edit rather than accumulating near-duplicates. The derived
    GLB is dropped on overwrite, since it no longer depicts the pose.
    """
    pose_id = pose_id or new_id()
    path = pose_path(job_dir, pose_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "id": pose_id,
        "name": pose["name"],
        "bones": pose["bones"],
        "created": time.time(),
    }
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    pose_glb_path(job_dir, pose_id).unlink(missing_ok=True)
    return record


def read_pose(job_dir: Path, pose_id: str) -> dict[str, Any] | None:
    path = pose_path(job_dir, pose_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.exception("unreadable pose at %s", path)
        return None


def list_poses(job_dir: Path) -> list[dict[str, Any]]:
    """Every saved pose, oldest first. A corrupt file costs itself, not the list."""
    directory = pose_dir(job_dir)
    if not directory.is_dir():
        return []
    poses = []
    for path in sorted(directory.glob("*.json")):
        if not is_valid_id(path.stem):
            continue
        record = read_pose(job_dir, path.stem)
        if record is not None:
            poses.append(record)
    poses.sort(key=lambda p: p.get("created", 0.0))
    return poses


def delete_pose(job_dir: Path, pose_id: str) -> bool:
    path = pose_path(job_dir, pose_id)
    if not path.exists():
        return False
    path.unlink()
    pose_glb_path(job_dir, pose_id).unlink(missing_ok=True)
    return True


# --- sprite sheets ----------------------------------------------------------
#
# Same storage shape as poses, and beside them, for the same reasons: a sheet
# belongs to the mesh it depicts, one file per sheet needs no locking, and the
# id guard is the only thing between a caller and an arbitrary path.

SHEET_DIR_NAME = "sheets"

MAX_SHEETS = 200


def sheet_dir(job_dir: Path) -> Path:
    return job_dir / SHEET_DIR_NAME


def sheet_path(job_dir: Path, sheet_id: str) -> Path:
    if not is_valid_id(sheet_id):
        raise ValueError(f"malformed sheet id {sheet_id!r}")
    return sheet_dir(job_dir) / f"{sheet_id}.json"


def sheet_png_path(job_dir: Path, sheet_id: str) -> Path:
    return sheet_path(job_dir, sheet_id).with_suffix(".png")


def read_sheet(job_dir: Path, sheet_id: str) -> dict[str, Any] | None:
    path = sheet_path(job_dir, sheet_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        log.exception("unreadable sheet at %s", path)
        return None


def list_sheets(job_dir: Path) -> list[dict[str, Any]]:
    """Every finished sheet, oldest first.

    A sidecar with no PNG beside it is skipped: the queue writes the PNG first
    and the sidecar last, so the sidecar is the completion marker -- but a
    half-cleaned directory should not advertise an image that isn't there.
    """
    directory = sheet_dir(job_dir)
    if not directory.is_dir():
        return []
    sheets = []
    for path in sorted(directory.glob("*.json")):
        if not is_valid_id(path.stem) or not path.with_suffix(".png").exists():
            continue
        record = read_sheet(job_dir, path.stem)
        if record is not None:
            sheets.append(record)
    sheets.sort(key=lambda s: s.get("created", 0.0))
    return sheets


def delete_sheet(job_dir: Path, sheet_id: str) -> bool:
    path = sheet_path(job_dir, sheet_id)
    png = sheet_png_path(job_dir, sheet_id)
    if not path.exists() and not png.exists():
        return False
    path.unlink(missing_ok=True)
    png.unlink(missing_ok=True)
    return True


def sheet_spec(
    source_glb: Path,
    frames_dir: Path,
    cells: list[dict[str, Any]],
    *,
    frame_size: int,
    elevation: float,
    lighting: str,
) -> dict[str, Any]:
    """The worker spec for rendering one frame per sheet cell.

    ``cells`` carries each cell's pose bones inline rather than a pose id: the
    worker has no access to the job directory's pose files, and shipping the
    rotations with the cell keeps it a pure renderer.
    """
    return {
        "op": "sheet",
        "source_glb": str(source_glb),
        "frames_dir": str(frames_dir),
        "result_path": str(frames_dir / "result.json"),
        "frame_size": frame_size,
        "elevation": elevation,
        "lighting": lighting,
        "cells": cells,
    }


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


def pose_spec(job_dir: Path, pose_id: str, bones: dict[str, Any]) -> dict[str, Any]:
    """The worker spec for baking one saved pose into its own GLB."""
    return {
        "op": "pose",
        "rig_glb": str(job_dir / "rig.glb"),
        "out_glb": str(pose_glb_path(job_dir, pose_id)),
        "result_path": str(pose_dir(job_dir) / f".{pose_id}.result.json"),
        "bones": bones,
    }
