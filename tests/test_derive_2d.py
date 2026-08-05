"""Icon, sprite and pixel art derived from a finished reference.

The same lazy-derivation contract the STL and OBJ exports have -- produced on
first request, cached beside the source, one lock per (job, artifact) -- with
the source being input.png rather than model.glb. Any reference already on
disk gains them retroactively, which is the whole reason they are derived
rather than generated.
"""

from __future__ import annotations

import json
import os

import pytest
from PIL import Image, ImageDraw

from warlock.service import derive as svc_derive
from warlock.service import export as svc_export
from warlock.service import files as svc_files
from warlock.service.errors import NotFound, NotReady


def _draw(job_dir, box):
    im = Image.new("RGB", (128, 128), (200, 200, 200))
    ImageDraw.Draw(im).rectangle(box, fill=(40, 40, 40))
    im.save(job_dir / "input.png")


def _reference(svc, *, status="done", stage="reference"):
    job_id = svc.store.create("text", "a barrel", {"seed": 1}, stage=stage, status=status)
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    _draw(job_dir, (32, 24, 96, 104))
    return job_id


def _hand_edit(svc, job_id, box=(8, 8, 40, 40)):
    """Overwrite input.png the way save_edited_image does, newer than anything.

    The explicit utime is not a workaround for a fast test: mtime granularity
    is a property of the filesystem, and pinning the ordering is what makes
    these assertions about the staleness rule rather than about how long a PNG
    encode happened to take.
    """
    job_dir = svc.job_dir(job_id)
    _draw(job_dir, box)
    source = job_dir / "input.png"
    latest = max(p.stat().st_mtime_ns for p in job_dir.iterdir())
    os.utime(source, ns=(latest + 1_000_000_000, latest + 1_000_000_000))
    return job_dir


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


@pytest.mark.parametrize(
    "name,size", [("pixel_32.png", 32), ("pixel_64.png", 64), ("pixel_128.png", 128)]
)
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


def test_an_edited_reference_derives_its_exports_again(svc):
    # get_file caches on freshness, not existence. A hand edit or a revert
    # rewrites input.png in place, and an export older than it depicts pixels
    # that are gone -- which, cached on existence, it would depict forever.
    job_id = _reference(svc)
    with Image.open(svc_derive.get_file(svc, job_id, "sprite.png")) as im:
        assert im.size == (65, 81)

    _hand_edit(svc, job_id, box=(8, 8, 40, 40))

    with Image.open(svc_derive.get_file(svc, job_id, "sprite.png")) as im:
        assert im.size == (33, 33)


def test_an_edit_drops_the_manifest_entries_it_invalidates(svc):
    # The manifest must not outlive its entries. An entry is a claim about one
    # file's pivot, alpha and matte, so an entry for an export that was not
    # re-derived is a verdict on an image the user has replaced.
    job_id = _reference(svc)
    svc_derive.get_file(svc, job_id, "icon.png")
    svc_derive.get_file(svc, job_id, "sprite.png")

    job_dir = _hand_edit(svc, job_id)
    svc_derive.get_file(svc, job_id, "icon.png")

    manifest = json.loads((job_dir / "manifest.json").read_text("utf-8"))
    assert set(manifest["artifacts"]) == {"icon.png"}
    # And the surviving entry describes the new pixels, not the old ones.
    assert manifest["artifacts"]["icon.png"]["source"] == [128, 128]
    assert manifest["artifacts"]["icon.png"]["trim"] == [8, 8, 41, 41]


def test_a_revert_invalidates_the_exports_of_the_edit(svc):
    # Driven through the real editor calls, because revert is the one write
    # where the content changes and the timestamp could go *backwards*: the
    # backup was copied when the first edit was made and copyfile does not
    # preserve mtimes, so a restored input.png would carry that older moment
    # and leave the edit's exports looking current.
    import io
    import time

    job_id = _reference(svc)
    job_dir = svc.job_dir(job_id)

    buf = io.BytesIO()
    edited = Image.new("RGB", (128, 128), (200, 200, 200))
    ImageDraw.Draw(edited).rectangle((8, 8, 40, 40), fill=(40, 40, 40))
    edited.save(buf, "PNG")
    svc_files.save_edited_image(svc, job_id, buf.getvalue())

    with Image.open(svc_derive.get_file(svc, job_id, "sprite.png")) as im:
        assert im.size == (33, 33)

    # Age everything, so the revert's own timestamp is unambiguously newer
    # whatever the host clock's granularity is -- Windows file times advance in
    # ~15 ms steps, which two writes in one test can easily share. This is what
    # keeps the assertion about the rule instead of about the clock.
    old = time.time_ns() - 5_000_000_000
    for entry in job_dir.iterdir():
        os.utime(entry, ns=(old, old))

    svc_files.revert_reference(svc, job_id)

    with Image.open(svc_derive.get_file(svc, job_id, "sprite.png")) as im:
        assert im.size == (65, 81)


def test_a_manifest_asked_for_after_an_edit_claims_nothing_stale(svc):
    job_id = _reference(svc)
    svc_derive.get_file(svc, job_id, "icon.png")
    _hand_edit(svc, job_id)

    manifest = json.loads(svc_derive.get_file(svc, job_id, "manifest.json").read_text("utf-8"))

    assert manifest["artifacts"] == {}


def test_a_manifest_entry_goes_away_with_its_artifact(svc):
    # Nothing deletes a 2D export today, but the entry outliving the file is
    # the same failure as the entry outliving the pixels, and one rule covers
    # both: an entry survives only while its file is on disk and current.
    job_id = _reference(svc)
    svc_derive.get_file(svc, job_id, "icon.png")
    job_dir = svc.job_dir(job_id)
    (job_dir / "icon.png").unlink()

    svc_derive.get_file(svc, job_id, "sprite.png")

    manifest = json.loads((job_dir / "manifest.json").read_text("utf-8"))
    assert set(manifest["artifacts"]) == {"sprite.png"}


def test_a_bulk_export_never_zips_a_stale_2d_artifact(svc):
    # bulk_export deliberately does not derive on demand, so freshness is the
    # only thing keeping a superseded icon out of the zip.
    job_id = _reference(svc)
    svc_derive.get_file(svc, job_id, "icon.png")
    assert svc_export.collect(svc, [job_id], ["icon.png"])

    _hand_edit(svc, job_id)

    assert svc_export.collect(svc, [job_id], ["icon.png"]) == []
    # An artifact that is not a 2D export is unaffected by the same gate.
    assert svc_export.collect(svc, [job_id], ["input.png"])


def test_the_manifest_records_a_hand_edit_beside_the_recipe(svc):
    # The recipe hash claims a seed and a model produced these pixels. After an
    # edit that is no longer the whole truth, which is the reason _remeasure
    # records the flag at all -- so it has to travel with the claim.
    job_id = svc.store.create(
        "text",
        "a barrel",
        {"seed": 1, "recipe": {"reference": {"seed": 1}}, "hand_edited": True},
        stage="reference",
        status="done",
    )
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    _draw(job_dir, (32, 24, 96, 104))

    svc_derive.get_file(svc, job_id, "icon.png")

    manifest = json.loads((job_dir / "manifest.json").read_text("utf-8"))
    assert manifest["hand_edited"] is True
    assert manifest["recipe"] is not None


def test_an_untouched_reference_is_not_claimed_to_be_hand_edited(svc):
    job_id = _reference(svc)
    svc_derive.get_file(svc, job_id, "icon.png")
    manifest = json.loads((svc.job_dir(job_id) / "manifest.json").read_text("utf-8"))
    assert manifest["hand_edited"] is False


def test_a_corrupt_manifest_is_rebuilt_rather_than_failing_the_export(svc):
    # Every entry is reproducible by re-deriving the artifact it describes, so
    # a truncated manifest costs metadata and never an export -- but it must
    # not be *silently* merged into either, which would produce valid JSON
    # around whatever survived the truncation.
    job_id = _reference(svc)
    svc_derive.get_file(svc, job_id, "icon.png")
    job_dir = svc.job_dir(job_id)
    (job_dir / "manifest.json").write_text('{"artifacts": {"icon.p', encoding="utf-8")

    svc_derive.get_file(svc, job_id, "sprite.png")

    manifest = json.loads((job_dir / "manifest.json").read_text("utf-8"))
    assert manifest["job"] == job_id
    assert set(manifest["artifacts"]) == {"sprite.png"}


def test_a_manifest_that_is_not_an_object_is_replaced(svc):
    job_id = _reference(svc)
    job_dir = svc.job_dir(job_id)
    (job_dir / "manifest.json").write_text("[1, 2, 3]", encoding="utf-8")
    _hand_edit(svc, job_id, box=(32, 24, 96, 104))

    svc_derive.get_file(svc, job_id, "icon.png")

    manifest = json.loads((job_dir / "manifest.json").read_text("utf-8"))
    assert manifest["version"] == 1
    assert set(manifest["artifacts"]) == {"icon.png"}


def test_an_image_with_no_subject_refuses_rather_than_writing_a_blank(svc):
    # A fully transparent icon.png is indistinguishable from a successful
    # export until somebody opens it, so NoSubject becomes NotReady and no
    # file is written at all.
    job_id = _reference(svc)
    job_dir = svc.job_dir(job_id)
    Image.new("RGB", (64, 64), (200, 200, 200)).save(job_dir / "input.png")

    # The message is asserted because "file not ready" is the generic refusal
    # this same call makes for half a dozen other reasons; without it the test
    # would pass on any of them.
    with pytest.raises(NotReady, match="no subject"):
        svc_derive.get_file(svc, job_id, "icon.png")
    assert not (job_dir / "icon.png").exists()


def test_every_2d_artifact_has_a_derivation():
    # The table and the branches in _derive_2d have to agree: an artifact added
    # to DERIVED_2D and nowhere else would reach a request before anyone found
    # out. Caught here, at edit time, instead.
    assert set(svc_files.DERIVED_2D) == {
        svc_derive.MANIFEST,
        "icon.png",
        "sprite.png",
        "wrap_preview.png",
    } | set(svc_files.PIXEL_ARTIFACTS)


def test_a_tile_derives_a_wrapped_view_of_itself(svc):
    job_id = _reference(svc, stage="tile")
    path = svc_derive.get_file(svc, job_id, "wrap_preview.png")
    assert path.name == "wrap_preview.png"
    with Image.open(path) as im:
        assert im.size == (128, 128)
        # Rolled by half in both axes: the corner pixel of the source is now
        # the centre pixel, which is the whole point -- the wrap seam runs
        # through the middle of the frame where a discontinuity is visible.
        assert im.convert("RGB").getpixel((64, 64)) == (200, 200, 200)


def test_a_tile_cannot_derive_the_cutout_exports(svc):
    # The cutout half of the 2D set is about lifting a subject off a
    # background, and a tile is background. Refused at the service, not merely
    # hidden in the pane: the two answer the same question and the pane's copy
    # is pinned to this one.
    job_id = _reference(svc, stage="tile")
    for name in ("icon.png", "sprite.png", "pixel_64.png"):
        with pytest.raises(NotReady):
            svc_derive.get_file(svc, job_id, name)


def test_a_reference_has_nothing_to_wrap(svc):
    job_id = _reference(svc)
    with pytest.raises(NotReady):
        svc_derive.get_file(svc, job_id, "wrap_preview.png")


def test_a_wrapped_view_is_re_derived_after_a_hand_edit(svc):
    # The same staleness rule every other 2D export follows: it is a view of
    # input.png, so an edit makes the cached one a picture of pixels that are
    # gone.
    job_id = _reference(svc, stage="tile")
    first = svc_derive.get_file(svc, job_id, "wrap_preview.png")
    stamp = first.stat().st_mtime_ns
    _hand_edit(svc, job_id)
    again = svc_derive.get_file(svc, job_id, "wrap_preview.png")
    assert again.stat().st_mtime_ns > stamp


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
    for name in svc_files.REFERENCE_2D:
        assert svc_derive.derivable_2d(name, "reference")
    for name in svc_files.TILE_2D:
        assert svc_derive.derivable_2d(name, "tile")
    # Each stage is refused the other's half, which is the whole reason the
    # stage is an argument.
    assert not svc_derive.derivable_2d("icon.png", "tile")
    assert not svc_derive.derivable_2d("wrap_preview.png", "reference")
    assert not svc_derive.derivable_2d("icon.png", "model")
    assert not svc_derive.derivable_2d("model.stl", "reference")
    assert not svc_derive.derivable_2d("nonsense.png", "reference")


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
