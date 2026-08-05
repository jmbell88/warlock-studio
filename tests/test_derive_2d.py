"""Icon, sprite and pixel art derived from a finished reference.

The same lazy-derivation contract the STL and OBJ exports have -- produced on
first request, cached beside the source, one lock per (job, artifact) -- with
the source being input.png rather than model.glb. Any reference already on
disk gains them retroactively, which is the whole reason they are derived
rather than generated.
"""

from __future__ import annotations

import json

import pytest
from PIL import Image, ImageDraw

from warlock.service import derive as svc_derive
from warlock.service import files as svc_files
from warlock.service.errors import NotFound, NotReady


def _reference(svc, *, status="done", stage="reference"):
    job_id = svc.store.create("text", "a barrel", {"seed": 1}, stage=stage, status=status)
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (128, 128), (200, 200, 200))
    ImageDraw.Draw(im).rectangle((32, 24, 96, 104), fill=(40, 40, 40))
    im.save(job_dir / "input.png")
    return job_id


def test_an_icon_is_derived_on_first_request(svc):
    job_id = _reference(svc)
    path = svc_derive.get_file(svc, job_id, "icon.png")
    assert path.exists()
    with Image.open(path) as im:
        assert im.size == (512, 512)
        assert im.mode == "RGBA"


def test_a_second_request_serves_the_cached_file(svc):
    job_id = _reference(svc)
    first = svc_derive.get_file(svc, job_id, "icon.png")
    stamp = first.stat().st_mtime_ns
    again = svc_derive.get_file(svc, job_id, "icon.png")
    assert again == first and again.stat().st_mtime_ns == stamp


def test_a_sprite_is_trimmed_to_the_subject(svc):
    job_id = _reference(svc)
    with Image.open(svc_derive.get_file(svc, job_id, "sprite.png")) as im:
        assert im.size == (65, 81)


@pytest.mark.parametrize("name,size", [("pixel_32.png", 32), ("pixel_64.png", 64)])
def test_each_pixel_size_is_its_own_artifact(svc, name, size):
    job_id = _reference(svc)
    with Image.open(svc_derive.get_file(svc, job_id, name)) as im:
        assert max(im.size) == size


def test_the_manifest_accumulates_an_entry_per_artifact(svc):
    job_id = _reference(svc)
    svc_derive.get_file(svc, job_id, "icon.png")
    svc_derive.get_file(svc, job_id, "sprite.png")

    manifest = json.loads(svc_derive.get_file(svc, job_id, "manifest.json").read_text("utf-8"))

    assert manifest["job"] == job_id
    assert set(manifest["artifacts"]) >= {"icon.png", "sprite.png"}
    assert manifest["artifacts"]["sprite.png"]["pivot"] == [32.5, 81.0]
    assert manifest["artifacts"]["icon.png"]["canvas"] == [512, 512]


def test_the_manifest_records_which_matte_produced_the_alpha(svc):
    job_id = _reference(svc)
    svc_derive.get_file(svc, job_id, "icon.png")
    manifest = json.loads((svc.job_dir(job_id) / "manifest.json").read_text("utf-8"))
    entry = manifest["artifacts"]["icon.png"]
    assert entry["matte"] in ("flood", "alpha", "birefnet")
    assert entry["alpha"]["islands"] >= 1


def test_the_manifest_records_the_source_recipe(svc):
    job_id = svc.store.create(
        "text", "a barrel",
        {"seed": 1, "recipe": {"reference": {"base_model": "turbo", "seed": 1}}},
        stage="reference", status="done",
    )
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (64, 64), (200, 200, 200))
    ImageDraw.Draw(im).rectangle((16, 16, 48, 48), fill=(0, 0, 0))
    im.save(job_dir / "input.png")

    svc_derive.get_file(svc, job_id, "icon.png")

    manifest = json.loads((job_dir / "manifest.json").read_text("utf-8"))
    assert manifest["recipe"] is not None and len(manifest["recipe"]) == 12


def test_the_manifest_can_be_the_first_thing_derived(svc):
    # The manifest's *artifact* lock and the manifest lock are one and the same
    # lock, and convert_lock hands out plain, non-reentrant Locks -- so a
    # manifest asked for before any artifact exists is the one request that can
    # deadlock against itself. It answers with the header alone.
    job_id = _reference(svc)
    manifest = json.loads(svc_derive.get_file(svc, job_id, "manifest.json").read_text("utf-8"))
    assert manifest["job"] == job_id
    assert manifest["artifacts"] == {}


def test_two_artifacts_derived_at_once_both_land_in_the_manifest(svc):
    # The lock ordering is what this asserts: each derivation takes its own
    # artifact lock and only then the manifest's, so they serialise on the
    # manifest instead of deadlocking, and neither one's whole-file write
    # drops the other's entry.
    import threading

    job_id = _reference(svc)
    start = threading.Barrier(2)
    errors: list[BaseException] = []

    def derive(name):
        start.wait(timeout=5)
        try:
            svc_derive.get_file(svc, job_id, name)
        except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
            errors.append(exc)

    threads = [threading.Thread(target=derive, args=(n,)) for n in ("icon.png", "pixel_32.png")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
        assert not t.is_alive(), "a derivation is still holding a lock"
    assert not errors

    manifest = json.loads((svc.job_dir(job_id) / "manifest.json").read_text("utf-8"))
    assert set(manifest["artifacts"]) == {"icon.png", "pixel_32.png"}


def test_a_mesh_job_cannot_derive_a_sprite(svc):
    # The 2D exports are about a reference's pixels. A mesh job's input.png is
    # the picture it was reconstructed *from*, and exporting a sprite of it
    # would quietly claim to be an export of the asset.
    job_id = _reference(svc, stage="model")
    with pytest.raises(NotReady):
        svc_derive.get_file(svc, job_id, "icon.png")


def test_an_unfinished_reference_cannot_derive_anything(svc):
    job_id = _reference(svc, status="running")
    with pytest.raises(NotReady):
        svc_derive.get_file(svc, job_id, "icon.png")


def test_an_unknown_2d_artifact_is_still_refused(svc):
    job_id = _reference(svc)
    with pytest.raises(NotFound):
        svc_derive.get_file(svc, job_id, "pixel_9999.png")


def test_derivable_2d_answers_for_the_whole_set():
    for name in svc_files.DERIVED_2D:
        assert svc_derive.derivable_2d(name)
    assert not svc_derive.derivable_2d("model.stl")
    assert not svc_derive.derivable_2d("nonsense.png")


def test_every_2d_artifact_is_in_the_media_allowlist():
    # MEDIA is what keeps a caller-supplied name off the filesystem; an
    # artifact that skipped it would be underivable and unserveable.
    for name in svc_files.DERIVED_2D:
        assert name in svc_files.MEDIA


def test_the_2d_artifacts_are_not_listed_as_files_on_the_job(svc):
    # Same rule the mesh exports follow: derived artifacts are produced on
    # request, so listing them would claim files that usually are not there.
    job_id = _reference(svc)
    svc_derive.get_file(svc, job_id, "icon.png")
    job = svc.store.get(job_id)
    svc_files.attach_files(job, svc.job_dir(job_id))
    assert "icon.png" not in job["files"]
