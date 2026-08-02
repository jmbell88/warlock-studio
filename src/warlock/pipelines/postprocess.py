"""Mesh conversion and scaling. GLB is the primary output; STL/OBJ are derived on demand.

Decimation and UV atlasing happen inside trellis-server (quadric simplify to
300K faces @1024), so post-processing here is format conversion plus the one
transform trellis-server cannot do itself: scaling the result to a real-world
size (see guidance.py).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import struct
import tempfile
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path

import trimesh

log = logging.getLogger(__name__)

_GLB_MAGIC = 0x46546C67  # 'glTF'
_CHUNK_JSON = 0x4E4F534A  # 'JSON'


@contextlib.contextmanager
def _staged(dest: Path) -> Iterator[Path]:
    """Yield a temp path beside ``dest``, renamed onto it on clean exit.

    Both exports are produced on demand by the file-serving route, so two
    requests for the same artifact (a double-clicked download link is enough)
    can run concurrently. Writing to ``dest`` directly lets a reader observe a
    half-written file; a per-call unique temp plus os.replace -- an atomic
    rename on POSIX and Windows alike when both share a filesystem -- means a
    reader sees either the old file or the complete new one. Same reasoning as
    scale_glb below, which had this bug first.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, raw = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".tmp")
    os.close(fd)
    tmp = Path(raw)
    try:
        yield tmp
        _replace_or_accept(tmp, dest)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


def _replace_or_accept(tmp: Path, dest: Path) -> None:
    """os.replace(tmp, dest), tolerating a concurrent writer.

    Windows fails the rename with ERROR_ACCESS_DENIED when two threads replace
    the same target at once, where POSIX would just let the last one win. These
    artifacts are pure functions of the GLB, so whoever landed first produced a
    file identical to ours: losing the race is a success, not an error. Only a
    failure that leaves no file at all is worth raising.
    """
    for attempt in range(3):
        try:
            os.replace(tmp, dest)
            return
        except OSError:
            if attempt < 2:
                time.sleep(0.05)
    if dest.exists() and dest.stat().st_size > 0:
        log.debug("%s was written by a concurrent export; keeping theirs", dest)
        return
    os.replace(tmp, dest)  # re-raise the real error with the real traceback


def glb_to_stl(glb_path: Path, stl_path: Path) -> Path:
    """Geometry-only STL (for reference / printing previews)."""
    mesh = _load_merged(glb_path)
    # Exported to bytes rather than to the temp path: trimesh infers the format
    # from the suffix, and the temp name deliberately doesn't have one.
    data = mesh.export(file_type="stl")
    with _staged(stl_path) as tmp:
        tmp.write_bytes(data)
    return stl_path


def glb_to_obj_zip(glb_path: Path, zip_path: Path) -> Path:
    """OBJ + MTL + textures bundled in a zip (OBJ is inherently multi-file)."""
    scene = trimesh.load(glb_path)
    # A private temp dir, not a fixed "obj_export" next to the zip: the fixed
    # name was both left behind forever (duplicating the model and every
    # texture beside the zip) and shared by concurrent exports, which
    # interleaved their writes into it.
    workdir = Path(tempfile.mkdtemp(dir=zip_path.parent, prefix=".obj_export."))
    try:
        scene.export(workdir / "model.obj")
        with _staged(zip_path) as tmp, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            # rglob, not iterdir: trimesh may emit textures into a subdirectory,
            # which zf.write would refuse as a directory entry.
            for f in sorted(workdir.rglob("*")):
                if f.is_file():
                    zf.write(f, f.relative_to(workdir).as_posix())
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return zip_path


def scale_glb(glb_path: Path, target_max_m: float) -> float:
    """Rescale the GLB in place so its longest bounding-box axis is target_max_m.

    Returns the factor applied (1.0 when nothing was done).

    Measuring goes through trimesh, but the write does not: re-exporting would
    re-encode every texture and material. Instead only the GLB's JSON chunk is
    rewritten, inserting a node that carries the scale. Buffers and images are
    copied through byte-for-byte.
    """
    extent = float(trimesh.load(glb_path).extents.max())
    if not extent > 0 or target_max_m <= 0:
        log.warning("skipping scale of %s: extent %s, target %s", glb_path, extent, target_max_m)
        return 1.0
    factor = target_max_m / extent
    if abs(factor - 1.0) < 1e-6:
        return 1.0

    header, gltf, rest = _split_glb(glb_path.read_bytes())
    scene = gltf["scenes"][gltf.get("scene", 0)]
    nodes = gltf.setdefault("nodes", [])
    for root in scene.get("nodes", []):
        _insert_scale_below(nodes, root, factor)
    # Write to a temp file and rename into place instead of write_bytes(),
    # which opens "wb" and truncates glb_path to 0 bytes before writing --
    # a concurrent reader (the file-serving route, or another export) could
    # observe a half-written file. os.replace is an atomic rename on both
    # POSIX and Windows as long as tmp and glb_path share a filesystem.
    tmp = glb_path.with_suffix(".glb.tmp")
    tmp.write_bytes(_rebuild_glb(header, gltf, rest))
    os.replace(tmp, glb_path)
    return factor


def _insert_scale_below(nodes: list[dict], root: int, factor: float) -> None:
    """Push everything hanging off ``root`` into a new scaled child node.

    The scale deliberately does not go on the root itself. trimesh treats a
    scene root as the graph's base frame and silently discards its transform,
    which would leave the GLB scaled but the trimesh-derived STL/OBJ exports
    (glb_to_stl, glb_to_obj_zip) at the original size.
    """
    node = nodes[root]
    scaled: dict = {"scale": [factor] * 3}
    for key in ("mesh", "children", "skin", "weights"):
        if key in node:
            scaled[key] = node.pop(key)
    nodes.append(scaled)
    node["children"] = [len(nodes) - 1]


def _split_glb(data: bytes) -> tuple[bytes, dict, bytes]:
    """-> (12-byte header, parsed JSON chunk, every following byte verbatim)."""
    magic, version, _length = struct.unpack_from("<III", data, 0)
    if magic != _GLB_MAGIC:
        raise ValueError("not a GLB file")
    if version != 2:
        raise ValueError(f"unsupported GLB version {version}")
    chunk_len, chunk_type = struct.unpack_from("<II", data, 12)
    if chunk_type != _CHUNK_JSON:
        raise ValueError("first GLB chunk is not JSON")
    start = 20
    return data[:12], json.loads(data[start : start + chunk_len]), data[start + chunk_len :]


def _rebuild_glb(header: bytes, gltf: dict, rest: bytes) -> bytes:
    # The JSON chunk pads with spaces (0x20) per the glTF spec, not zeros.
    payload = json.dumps(gltf, separators=(",", ":")).encode()
    payload += b" " * (-len(payload) % 4)
    total = len(header) + 8 + len(payload) + len(rest)
    return (
        header[:8]
        + struct.pack("<I", total)
        + struct.pack("<II", len(payload), _CHUNK_JSON)
        + payload
        + rest
    )


def _load_merged(glb_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(glb_path)
    if isinstance(loaded, trimesh.Scene):
        return loaded.to_mesh()
    return loaded
