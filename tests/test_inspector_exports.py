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
    # "ground" included: its input.png is the finished atlas rather than a
    # source, which makes taking it away the *whole* point of the row.
    for stage in ("reference", "model", "ground"):
        names = [n for n, _label in widgets.artifacts_for(_job(stage=stage))]
        assert "input.png" in names, stage


def test_every_offered_name_is_servable():
    """Every stage, including any newly added one. The loop is the point: an
    offered name that ``files.MEDIA`` does not carry is a download button that
    answers NotReady for ever, and a stage left out of this list is exactly how
    one ships."""
    from warlock.service import files as svc_files

    for stage in ("reference", "tile", "model", "ground"):
        offered = widgets.artifacts_for(_job(stage=stage))
        assert offered, f"{stage} offers nothing at all"
        for name, _label in offered:
            assert name in svc_files.MEDIA, f"{stage}: {name}"


def test_a_tile_is_not_offered_the_cutout_exports():
    # A cutout is the subject lifted off its background, and a seamless texture
    # is background: an icon of one is the whole frame with a matte guessed
    # over it, which is a picture of nothing.
    names = [n for n, _label in widgets.artifacts_for(_job(stage="tile"))]
    assert "icon.png" not in names
    assert "sprite.png" not in names
    assert not [n for n in names if n.startswith("pixel_")]


def test_a_tile_offers_the_texture_itself_its_wrapped_view_and_its_material():
    names = [n for n, _label in widgets.artifacts_for(_job(stage="tile"))]
    assert names == [
        "input.png",
        "wrap_preview.png",
        "material.zip",
        "material_normal.png",
        "material_roughness.png",
        "material_height.png",
        "manifest.json",
    ]


def test_every_estimated_map_says_so_on_its_button():
    """The one thing the labels must carry.

    `pipelines/material` is written around the point that these maps describe
    the albedo's contrast rather than a measured surface, and a button called
    "Normal map" undoes that in the only place the user actually reads. The zip
    carries the same sentence in its README, for the same reason.
    """
    labels = dict(widgets.artifacts_for(_job(stage="tile")))
    for name in ("material_normal.png", "material_roughness.png", "material_height.png"):
        assert "est." in labels[name], labels[name]


def test_the_grid_offers_exactly_what_each_stage_can_derive():
    """The label tuples and ``derived_2d_for`` are one list written twice.

    The grid is literals so the *order* can be a UI decision, which means a
    name added to ``TILE_2D`` or ``REFERENCE_2D`` and not to the tuple beside
    it loses its button entirely -- not a wrong button, no button. Nothing else
    catches that: the servable test only asks about MEDIA, and the cross-check
    against ``files.ready`` never looks at the grid at all.
    """
    from warlock.service import files as svc_files

    for stage in ("reference", "tile", "ground"):
        offered = {n for n, _label in widgets.artifacts_for(_job(stage=stage))}
        # input.png is the source image every job may take away, and is served
        # rather than derived -- so it is the one name in the grid that is not
        # in the derivable set.
        assert "input.png" in offered
        assert offered - {"input.png"} == set(svc_files.derived_2d_for(stage))


def test_a_reference_is_not_offered_a_wrap_preview():
    # Nothing wraps: the ratio the preview exists to make visible is only ever
    # measured on a tile.
    names = [n for n, _label in widgets.artifacts_for(_job())]
    assert "wrap_preview.png" not in names


# -- whether a button is pressable ----------------------------------------


def _derivable(job, name):
    return inspector._derivable(job, set(job.get("files") or []), name)


def test_a_finished_reference_can_derive_its_2d_exports():
    job = _job(files=["input.png"])
    assert _derivable(job, "icon.png")
    assert _derivable(job, "manifest.json")


def test_a_reference_still_running_cannot():
    # derivable_2d answers a question about the *name*; without the status the
    # grid lit six buttons whose only outcome was an error toast.
    job = _job(files=["input.png"])
    job["status"] = "running"
    assert not _derivable(job, "icon.png")


def test_a_reference_with_no_pixels_yet_cannot():
    assert not _derivable(_job(), "icon.png")


def test_a_reference_is_never_offered_a_mesh_derivation():
    assert not _derivable(_job(files=["input.png"]), "model.stl")


def test_a_mesh_derives_from_its_glb_and_not_from_its_input():
    job = _job(stage="model", files=["input.png", "model.glb"])
    assert _derivable(job, "model.stl")
    assert not _derivable(job, "icon.png")


def test_the_pane_agrees_with_the_service_about_every_artifact(tmp_path):
    """``inspector._derivable`` and ``files.ready`` must answer identically.

    The pane restates the service's rules rather than calling it, because
    ``ready`` takes a job dir and would be a stat per artifact per frame. That
    restatement is only safe while something holds the two together: the tile
    arm of ``files.ready`` is slated to change in B8, and without this the copy
    in the pane would quietly keep the old behaviour and light the wrong
    buttons. ``files`` is built by ``attach_files``, which is itself ``ready``
    over ``LISTED`` -- so this also pins the step the restatement leans on,
    that ``"input.png" in files`` means what ``(job_dir / "input.png").exists()``
    means.
    """
    from warlock.service import files as svc_files

    # The comparison is against ``ready or derivable``, which is what the grid
    # actually composes: ``ready`` answers "may this be served", and for a file
    # already on disk -- input.png, model.glb -- that is true while "could it be
    # derived" is false. Deriving is the half the pane restates; serving it
    # reads straight off the listing.
    names = (*svc_files.DERIVED_2D, "model.glb", "model.stl", "model.fbx", "input.png")
    checked = 0
    for stage in ("reference", "tile", "model"):
        for status in ("queued", "running", "done", "error"):
            for has_input in (False, True):
                for has_glb in (False, True):
                    if has_glb and stage != "model":
                        # Not a state that occurs: only a mesh job's directory
                        # ever holds a model.glb, and inventing one would have
                        # the test assert about a job that cannot exist.
                        continue
                    job_dir = tmp_path / f"{stage}-{status}-{has_input}-{has_glb}"
                    job_dir.mkdir()
                    if has_input:
                        (job_dir / "input.png").write_bytes(b"")
                    if has_glb:
                        (job_dir / "model.glb").write_bytes(b"")
                    job = {"id": job_dir.name, "stage": stage, "status": status}
                    svc_files.attach_files(job, job_dir)
                    files = set(job["files"])
                    for name in names:
                        pane = name in files or inspector._derivable(job, files, name)
                        assert pane == svc_files.ready(job, job_dir, name), (
                            f"{stage}/{status} input={has_input} glb={has_glb} {name}"
                        )
                        checked += 1
    # Guards against a matrix that silently collapses to nothing: 8 states each
    # for reference and tile, 16 for a mesh (which alone can hold a model.glb).
    assert checked == 32 * len(names)


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


def test_a_manifest_that_is_not_an_object_is_not_an_error(tmp_path):
    (tmp_path / "manifest.json").write_text("[1, 2, 3]", encoding="utf-8")
    assert inspector._manifest(_Ctx(tmp_path), "abc123abc123") is None


def _settled(monkeypatch, path):
    """Move the clock a tick past ``path``'s mtime.

    Every test that wants a *cache hit* has to do this, and that is the
    racily-clean rule working rather than a wrinkle in it: a file written a
    moment ago is answered correctly and deliberately not remembered, because a
    rewrite landing inside its mtime's own 15.6 ms tick would otherwise match
    that stamp forever. Same helper, same reason, as ``test_inspector_rig``'s.
    """
    import warlock.studio.panes.stamps as stamps_mod
    from warlock.service.files import MTIME_RACE_NS

    settled = path.stat().st_mtime_ns + MTIME_RACE_NS * 2
    monkeypatch.setattr(stamps_mod.time, "time_ns", lambda: settled)


def test_an_unreadable_manifest_is_not_re_read_every_frame(tmp_path, monkeypatch):
    # The one state that would otherwise defeat the cache entirely: "cannot be
    # parsed" is an answer about this version of the file just as a dict is,
    # and without the sentinel a mangled manifest is a read plus a failed parse
    # sixty times a second for as long as it stays mangled.
    path = tmp_path / "manifest.json"
    path.write_text("{not json", encoding="utf-8")
    ctx = _Ctx(tmp_path)
    _settled(monkeypatch, path)
    assert inspector._manifest(ctx, "abc123abc123") is None
    assert ctx.state.manifest == (("abc123abc123", path.stat().st_mtime_ns), None)


def test_a_manifest_written_a_moment_ago_is_not_cached_at_all(tmp_path):
    """The other half of the same rule, and the one that makes it safe: inside
    the racy window the answer is re-read every frame rather than remembered."""
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"artifacts": {"icon.png": {}}}), encoding="utf-8")
    ctx = _Ctx(tmp_path)

    assert inspector._manifest(ctx, "abc123abc123") is not None
    assert ctx.state.manifest is None


def test_the_manifest_is_parsed_once_per_version_not_once_per_frame(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"artifacts": {"icon.png": {}}}), encoding="utf-8")
    ctx = _Ctx(tmp_path)
    _settled(monkeypatch, path)

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
    assert list(inspector._manifest(ctx, "abc123abc123")["artifacts"]) == ["icon.png"]
    # A derivation running on the TaskRunner rewrites it under an open tab.
    # The mtime is forced rather than trusted: two writes inside one clock tick
    # would make the test assert the filesystem's resolution, not the cache.
    later = path.stat().st_mtime_ns + 10**9
    path.write_text(json.dumps({"artifacts": {"sprite.png": {}}}), encoding="utf-8")
    os.utime(path, ns=(later, later))
    assert list(inspector._manifest(ctx, "abc123abc123")["artifacts"]) == ["sprite.png"]


# -- what the notes under the grid actually say ----------------------------


def _lines(monkeypatch, fn, *args):
    """Run a note function with ``widgets.muted`` captured.

    The notes are the only part of the pane whose *wording* is a contract --
    with B3, which records ``hand_edited`` and ``matte`` specifically so this
    task can show them -- and capturing the one widget they use is what makes
    that assertable without a GL context.
    """
    out: list[str] = []
    monkeypatch.setattr(inspector.widgets, "muted", out.append)
    fn(*args)
    return out


def test_the_summary_carries_b3s_hand_edited_caveat(monkeypatch):
    manifest = {"hand_edited": True, "artifacts": {}}
    lines = _lines(monkeypatch, inspector._manifest_summary, manifest)
    assert any("hand-edited" in line for line in lines)


def test_an_untouched_reference_says_nothing_about_hand_edits(monkeypatch):
    manifest = {"hand_edited": False, "artifacts": {}}
    assert _lines(monkeypatch, inspector._manifest_summary, manifest) == []


def test_the_matte_is_reported_per_artifact_not_from_the_current_config(monkeypatch):
    # An icon cut before the weights were installed is still a corner fill, and
    # the config-level note would stop mentioning it the moment they landed.
    manifest = {
        "artifacts": {
            "icon.png": {"matte": "flood"},
            "sprite.png": {"matte": "birefnet", "pivot": [12.0, 34.0]},
        }
    }
    lines = _lines(monkeypatch, inspector._manifest_summary, manifest)
    assert "icon.png - corner fill" in lines
    assert "sprite.png - pivot 12,34" in lines


def test_the_summary_reports_a_matte_that_came_apart(monkeypatch):
    manifest = {"artifacts": {"icon.png": {"alpha": {"islands": 3}}}}
    lines = _lines(monkeypatch, inspector._manifest_summary, manifest)
    assert "icon.png - 3 separate pieces" in lines


def test_a_mangled_entry_does_not_break_the_summary(monkeypatch):
    manifest = {"artifacts": {"icon.png": "not a dict"}}
    assert _lines(monkeypatch, inspector._manifest_summary, manifest) == []


class _MatteCtx:
    def __init__(self):
        self.svc = types.SimpleNamespace(config=None)


def test_the_config_note_appears_only_before_anything_is_derived(monkeypatch):
    from warlock.pipelines import matting

    monkeypatch.setattr(matting, "available", lambda _config: False)
    ctx = _MatteCtx()
    assert _lines(monkeypatch, inspector._matte_note, ctx, None) != []
    assert _lines(monkeypatch, inspector._matte_note, ctx, {"artifacts": {}}) != []
    # Once entries exist they answer the question per artifact and truthfully,
    # so repeating the config-level guess would be noise beside them.
    assert _lines(monkeypatch, inspector._matte_note, ctx, {"artifacts": {"icon.png": {}}}) == []


def test_no_config_note_when_the_weights_are_installed(monkeypatch):
    from warlock.pipelines import matting

    monkeypatch.setattr(matting, "available", lambda _config: True)
    assert _lines(monkeypatch, inspector._matte_note, _MatteCtx(), None) == []


# -- the pixel-art preview's pure halves -----------------------------------


def test_pixel_provenance_reads_the_palette_off_the_manifest():
    manifest = {"artifacts": {"pixel_32.png": {"palette": 16}}}
    assert inspector.pixel_provenance(manifest, "pixel_32.png") == "32 px - 16 colours"


def test_pixel_provenance_calls_an_uncapped_artifact_full_colour():
    manifest = {"artifacts": {"pixel_64.png": {"palette": None}}}
    assert inspector.pixel_provenance(manifest, "pixel_64.png") == "64 px - full colour"


def test_pixel_provenance_is_none_without_an_entry():
    assert inspector.pixel_provenance(None, "pixel_64.png") is None
    assert inspector.pixel_provenance({"artifacts": {}}, "pixel_64.png") is None
    assert inspector.pixel_provenance({"artifacts": {"pixel_64.png": "junk"}}, "pixel_64.png") is (
        None
    )


def test_pixel_scale_is_a_whole_multiple_and_at_least_one():
    # Integer scaling is what keeps NEAREST crisp: a fractional factor samples
    # some source pixels twice and others once, which reads as banding.
    assert inspector.pixel_scale((32, 32), 192) == 6
    assert inspector.pixel_scale((96, 64), 192) == 2
    assert inspector.pixel_scale((256, 256), 192) == 1


class _Prefs:
    def __init__(self, values):
        self._values = values

    def get(self, key):
        return self._values.get(key)


def test_a_preview_derivation_never_claims_the_save_key():
    """The regression: the pixel panel submitted ``derive.get_file`` under
    ``save:<job>:<name>``, and the app toasts "Saved to <result>" for every
    finished ``save:`` key. ``get_file`` returns the path *inside* the job
    directory and is never None, so pressing "Preview pixels" told the user a
    file had been saved to a path they never chose, with no dialog shown.
    """
    from warlock.studio import app_ctx

    assert app_ctx.save_key("abc123abc123", "pixel_32.png").startswith("save:")
    assert not app_ctx.derive_key("abc123abc123", "pixel_32.png").startswith("save:")


def test_an_artifact_is_busy_under_either_of_its_two_keys():
    """Separate keys, one answer: a preview and an export of one name run the
    same ``get_file`` under the same per-artifact lock, so a control watching
    only its own key would offer a button that then blocked invisibly."""
    from warlock.studio import app_ctx

    busy: set[str] = set()
    ctx = types.SimpleNamespace(busy=lambda key: key in busy)
    ctx.artifact_busy = types.MethodType(app_ctx.Ctx.artifact_busy, ctx)

    assert ctx.artifact_busy("abc123abc123", "pixel_32.png") is False
    busy.add(app_ctx.derive_key("abc123abc123", "pixel_32.png"))
    assert ctx.artifact_busy("abc123abc123", "pixel_32.png") is True
    busy.clear()
    busy.add(app_ctx.save_key("abc123abc123", "pixel_32.png"))
    assert ctx.artifact_busy("abc123abc123", "pixel_32.png") is True


def test_pixel_prefs_defaults_and_survives_a_hand_mangled_settings_file():
    # The settings JSON is user-editable, and the pane runs on the frame
    # thread: a bad value must coerce to the default, never raise.
    assert inspector.pixel_prefs(_Prefs({})) == (128, 0, None, False)
    assert inspector.pixel_prefs(_Prefs({"pixel_size": 64, "pixel_colors": 16})) == (
        64, 16, None, False,
    )
    assert inspector.pixel_prefs(_Prefs({"pixel_size": "wide", "pixel_colors": "many"})) == (
        128, 0, None, False,
    )
    assert inspector.pixel_prefs(_Prefs({"pixel_size": 48, "pixel_colors": 7})) == (
        128, 0, None, False,
    )


def test_pixel_prefs_coerces_a_mangled_palette_to_no_palette():
    # A palette name is a string the user can hand-edit, and the pane cannot
    # stat the directory per frame to check it -- so the frame thread coerces
    # and service.palettes is what refuses an unknown one, on the task thread.
    assert inspector.pixel_prefs(_Prefs({"pixel_palette": 7}))[2] is None
    assert inspector.pixel_prefs(_Prefs({"pixel_palette": "  "}))[2] is None
    assert inspector.pixel_prefs(_Prefs({"pixel_palette": "nord", "pixel_dither": 1})) == (
        128, 0, "nord", True,
    )


# -- why a button is disabled ----------------------------------------------


class _BlockCtx:
    rigging_available = True


def test_an_unfinished_job_says_so_rather_than_claiming_it_never_can(monkeypatch):
    job = _job(files=["input.png"])
    job["status"] = "running"
    assert (
        inspector._why_blocked(_BlockCtx(), job, "icon.png", False, _derivable(job, "icon.png"))
        == "not finished yet"
    )


def test_a_failed_job_is_not_promised_an_export_that_is_not_coming():
    job = _job(files=["input.png"])
    job["status"] = "error"
    assert (
        inspector._why_blocked(_BlockCtx(), job, "icon.png", False, _derivable(job, "icon.png"))
        == "not available for this asset"
    )


def test_a_finished_reference_blocks_nothing_it_can_derive():
    job = _job(files=["input.png"])
    assert (
        inspector._why_blocked(_BlockCtx(), job, "icon.png", False, _derivable(job, "icon.png"))
        is None
    )
