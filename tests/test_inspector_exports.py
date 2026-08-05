"""Which artifacts a job's Export tab offers.

Pure: the grid is a function of the job row, so what a reference offers and
what a mesh offers is assertable without a GL context.
"""

from __future__ import annotations

import json
import os
import types

from warlock.studio import widgets
from warlock.studio.panes import inspector


def _job(stage="reference", files=()):
    return {"id": "abc123abc123", "stage": stage, "status": "done", "files": list(files)}


def test_a_reference_offers_the_2d_exports():
    names = [n for n, _label in widgets.artifacts_for(_job())]
    assert "icon.png" in names
    assert "sprite.png" in names
    assert "manifest.json" in names


def test_a_reference_is_not_offered_mesh_exports_it_can_never_have():
    names = [n for n, _label in widgets.artifacts_for(_job())]
    assert "model.stl" not in names
    assert "model.fbx" not in names


def test_a_mesh_offers_the_mesh_exports():
    names = [n for n, _label in widgets.artifacts_for(_job(stage="model"))]
    assert "model.glb" in names
    assert "model.stl" in names


def test_a_mesh_is_not_offered_a_sprite_of_its_own_input():
    # input.png on a mesh job is what it was reconstructed from, not an asset.
    names = [n for n, _label in widgets.artifacts_for(_job(stage="model"))]
    assert "sprite.png" not in names


def test_every_job_can_still_take_away_its_source_image():
    for stage in ("reference", "model"):
        names = [n for n, _label in widgets.artifacts_for(_job(stage=stage))]
        assert "input.png" in names


def test_every_offered_name_is_servable():
    from warlock.service import files as svc_files

    for stage in ("reference", "model"):
        for name, _label in widgets.artifacts_for(_job(stage=stage)):
            assert name in svc_files.MEDIA


# -- the manifest the Export tab shows under the grid ----------------------


class _Ctx:
    """Just enough of AppCtx for the manifest read: a job dir and a state slot.

    The pane's drawing half needs imgui and a GL context; the read does not,
    and it is the half with a cache in it worth pinning.
    """

    def __init__(self, root):
        self._root = root
        self.state = types.SimpleNamespace(manifest=None)

    def job_dir(self, job_id):
        return self._root


def test_a_missing_manifest_is_not_an_error(tmp_path):
    assert inspector._manifest(_Ctx(tmp_path), "abc123abc123") is None


def test_a_mangled_manifest_is_not_an_error(tmp_path):
    (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")
    assert inspector._manifest(_Ctx(tmp_path), "abc123abc123") is None


def test_the_manifest_is_parsed_once_per_version_not_once_per_frame(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"artifacts": {"icon.png": {}}}), encoding="utf-8")
    ctx = _Ctx(tmp_path)

    # Identity, not equality: a second parse would produce an equal dict, and
    # parsing a file sixty times a second is exactly what the cache exists to
    # stop.
    first = inspector._manifest(ctx, "abc123abc123")
    for _ in range(4):
        assert inspector._manifest(ctx, "abc123abc123") is first


def test_a_rewritten_manifest_is_re_read(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"artifacts": {"icon.png": {}}}), encoding="utf-8")
    ctx = _Ctx(tmp_path)
    assert list(inspector._manifest(ctx, "abc")["artifacts"]) == ["icon.png"]
    # A derivation running on the TaskRunner rewrites it under an open tab.
    # The mtime is forced rather than trusted: two writes inside one clock tick
    # would make the test assert the filesystem's resolution, not the cache.
    later = path.stat().st_mtime_ns + 10**9
    path.write_text(json.dumps({"artifacts": {"sprite.png": {}}}), encoding="utf-8")
    os.utime(path, ns=(later, later))
    assert list(inspector._manifest(ctx, "abc")["artifacts"]) == ["sprite.png"]
