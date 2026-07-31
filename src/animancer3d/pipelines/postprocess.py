"""Mesh format conversion. GLB is the primary output; STL/OBJ are derived on demand.

Decimation and UV atlasing happen inside trellis-server (quadric simplify to
300K faces @1024), so post-processing here is only format conversion.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import trimesh


def glb_to_stl(glb_path: Path, stl_path: Path) -> Path:
    """Geometry-only STL (for reference / printing previews)."""
    mesh = _load_merged(glb_path)
    mesh.export(stl_path)
    return stl_path


def glb_to_obj_zip(glb_path: Path, zip_path: Path) -> Path:
    """OBJ + MTL + textures bundled in a zip (OBJ is inherently multi-file)."""
    scene = trimesh.load(glb_path)
    workdir = zip_path.parent / "obj_export"
    workdir.mkdir(parents=True, exist_ok=True)
    scene.export(workdir / "model.obj")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in workdir.iterdir():
            zf.write(f, f.name)
    return zip_path


def _load_merged(glb_path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(glb_path)
    if isinstance(loaded, trimesh.Scene):
        return loaded.to_mesh()
    return loaded
