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
from warlock.service.errors import Invalid, NotFound, NotReady


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
    } | set(svc_files.PIXEL_ARTIFACTS) | set(svc_files.MATERIAL_2D)


def _entry(svc, job_id, name):
    manifest = json.loads((svc.job_dir(job_id) / "manifest.json").read_text("utf-8"))
    return manifest["artifacts"][name]


def _manifest_palette(svc, job_id, name):
    return _entry(svc, job_id, name).get("palette")


def test_a_palette_request_re_derives_a_fresh_pixel_artifact(svc):
    # The knob is derive-time state, not a job param, so the only record of
    # which palette cut the file on disk is its manifest entry -- and a file
    # cut with the wrong one is mtime-fresh. The compound test is the rule.
    job_id = _reference(svc)
    svc_derive.get_file(svc, job_id, "pixel_32.png")
    assert _manifest_palette(svc, job_id, "pixel_32.png") is None

    path = svc_derive.get_file(svc, job_id, "pixel_32.png", pixel_colors=8)

    assert _manifest_palette(svc, job_id, "pixel_32.png") == 8
    with Image.open(path) as im:
        rgba = im.convert("RGBA")
        colours = {c[:3] for _, c in rgba.getcolors(rgba.width * rgba.height) if c[3] > 0}
    assert len(colours) <= 8


def test_a_repeat_palette_request_serves_the_cached_file(svc):
    job_id = _reference(svc)
    first = svc_derive.get_file(svc, job_id, "pixel_32.png", pixel_colors=8)
    stamp = first.stat().st_mtime_ns
    again = svc_derive.get_file(svc, job_id, "pixel_32.png", pixel_colors=8)
    assert again == first and again.stat().st_mtime_ns == stamp


def test_turning_the_palette_off_re_derives_full_colour(svc):
    job_id = _reference(svc)
    svc_derive.get_file(svc, job_id, "pixel_32.png", pixel_colors=8)
    svc_derive.get_file(svc, job_id, "pixel_32.png", pixel_colors=0)
    assert _manifest_palette(svc, job_id, "pixel_32.png") is None


def test_a_plain_derived_artifact_is_current_under_no_palette_cap(svc):
    # Every pixel artifact already on disk predates the knob and carries
    # "palette": None. Asking for no cap must serve it untouched, or shipping
    # the feature churns every existing asset directory once.
    job_id = _reference(svc)
    first = svc_derive.get_file(svc, job_id, "pixel_32.png")
    stamp = first.stat().st_mtime_ns
    again = svc_derive.get_file(svc, job_id, "pixel_32.png", pixel_colors=0)
    assert again.stat().st_mtime_ns == stamp


def test_no_palette_preference_serves_whatever_is_on_disk(svc):
    # pixel_colors=None is today's callers: bulk export and anyone who has no
    # opinion. They take the file as last derived, whatever cut it.
    job_id = _reference(svc)
    first = svc_derive.get_file(svc, job_id, "pixel_32.png", pixel_colors=8)
    stamp = first.stat().st_mtime_ns
    again = svc_derive.get_file(svc, job_id, "pixel_32.png")
    assert again.stat().st_mtime_ns == stamp
    assert _manifest_palette(svc, job_id, "pixel_32.png") == 8


def test_a_missing_manifest_costs_one_re_derive_then_caches(svc):
    # No manifest means no provenance: the artifact might have been cut with
    # any palette, so a caller with a preference re-derives once and the
    # rebuilt entry answers from then on.
    job_id = _reference(svc)
    svc_derive.get_file(svc, job_id, "pixel_32.png")
    (svc.job_dir(job_id) / "manifest.json").unlink()

    first = svc_derive.get_file(svc, job_id, "pixel_32.png", pixel_colors=0)
    assert _manifest_palette(svc, job_id, "pixel_32.png") is None
    stamp = first.stat().st_mtime_ns
    again = svc_derive.get_file(svc, job_id, "pixel_32.png", pixel_colors=0)
    assert again.stat().st_mtime_ns == stamp


def _palette_dir(svc, tmp_path, text="#000000\n#ffffff\n"):
    directory = tmp_path / "palettes"
    directory.mkdir(exist_ok=True)
    (directory / "duo.hex").write_text(text)
    svc.config.palette_dir = directory
    return directory


def test_a_palette_file_maps_the_export_and_is_recorded(svc, tmp_path):
    _palette_dir(svc, tmp_path)
    job_id = _reference(svc)
    path = svc_derive.get_file(svc, job_id, "pixel_32.png", pixel_palette="duo")
    entry = _entry(svc, job_id, "pixel_32.png")
    assert entry["palette_file"] == "duo"
    assert entry["palette_hash"]
    # The int cap and a palette file are different things; conflating them
    # would make the "rebuild" prompt in the inspector fire forever.
    assert entry["palette"] is None
    with Image.open(path) as im:
        rgba = im.convert("RGBA")
        colours = {c[:3] for _, c in rgba.getcolors(rgba.width * rgba.height) if c[3] > 0}
    assert colours <= {(0, 0, 0), (255, 255, 255)}


def test_editing_a_palette_in_place_makes_the_artifact_stale(svc, tmp_path):
    # Editing a palette is the normal way to work on one, and it keeps the
    # file's name -- so freshness has to be keyed on the content digest.
    directory = _palette_dir(svc, tmp_path)
    job_id = _reference(svc)
    first = svc_derive.get_file(svc, job_id, "pixel_32.png", pixel_palette="duo")
    stamp = first.stat().st_mtime_ns
    again = svc_derive.get_file(svc, job_id, "pixel_32.png", pixel_palette="duo")
    assert again.stat().st_mtime_ns == stamp

    (directory / "duo.hex").write_text("#000000\n#ff0000\n")
    rebuilt = svc_derive.get_file(svc, job_id, "pixel_32.png", pixel_palette="duo")
    assert rebuilt.stat().st_mtime_ns != stamp


def test_turning_dither_on_re_derives(svc, tmp_path):
    _palette_dir(svc, tmp_path)
    job_id = _reference(svc)
    first = svc_derive.get_file(svc, job_id, "pixel_32.png", pixel_palette="duo")
    stamp = first.stat().st_mtime_ns
    dithered = svc_derive.get_file(
        svc, job_id, "pixel_32.png", pixel_palette="duo", pixel_dither=True
    )
    assert dithered.stat().st_mtime_ns != stamp
    assert _entry(svc, job_id, "pixel_32.png")["dither"] is True


def test_dropping_a_palette_goes_back_to_the_plain_export(svc, tmp_path):
    _palette_dir(svc, tmp_path)
    job_id = _reference(svc)
    svc_derive.get_file(svc, job_id, "pixel_32.png", pixel_palette="duo")
    svc_derive.get_file(svc, job_id, "pixel_32.png", pixel_colors=0)
    entry = _entry(svc, job_id, "pixel_32.png")
    assert entry["palette_file"] is None and entry["palette"] is None


def test_a_pre_upgrade_manifest_reads_current_when_no_palette_is_asked_for(svc):
    # An asset directory written before palettes existed carries none of the
    # new keys. Serving it untouched is what stops the feature churning every
    # asset on disk the first time it is installed.
    job_id = _reference(svc)
    first = svc_derive.get_file(svc, job_id, "pixel_32.png", pixel_colors=0)
    manifest_path = svc.job_dir(job_id) / "manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    entry = manifest["artifacts"]["pixel_32.png"]
    for key in ("palette_file", "palette_hash", "dither", "grid", "cleanup", "qa"):
        entry.pop(key, None)
    manifest_path.write_text(json.dumps(manifest), "utf-8")
    stamp = first.stat().st_mtime_ns

    again = svc_derive.get_file(svc, job_id, "pixel_32.png", pixel_colors=0)
    assert again.stat().st_mtime_ns == stamp


def test_an_unknown_palette_name_is_refused_before_any_disk_work(svc, tmp_path):
    _palette_dir(svc, tmp_path)
    job_id = _reference(svc)
    with pytest.raises(Invalid):
        svc_derive.get_file(svc, job_id, "pixel_32.png", pixel_palette="nope")
    assert not (svc.job_dir(job_id) / "pixel_32.png").exists()


def test_an_unknown_palette_size_is_refused_before_any_disk_work(svc):
    job_id = _reference(svc)
    with pytest.raises(Invalid):
        svc_derive.get_file(svc, job_id, "pixel_32.png", pixel_colors=7)
    assert not (svc.job_dir(job_id) / "pixel_32.png").exists()


def test_a_palette_preference_does_not_touch_non_pixel_artifacts(svc):
    job_id = _reference(svc)
    first = svc_derive.get_file(svc, job_id, "icon.png")
    stamp = first.stat().st_mtime_ns
    again = svc_derive.get_file(svc, job_id, "icon.png", pixel_colors=8)
    assert again.stat().st_mtime_ns == stamp


def test_a_hand_edit_and_a_palette_change_resolve_in_one_derive(svc):
    job_id = _reference(svc)
    svc_derive.get_file(svc, job_id, "pixel_32.png")
    _hand_edit(svc, job_id, box=(8, 8, 40, 40))

    path = svc_derive.get_file(svc, job_id, "pixel_32.png", pixel_colors=8)

    assert _manifest_palette(svc, job_id, "pixel_32.png") == 8
    with Image.open(path) as im:
        # The edited subject is square, so a re-derive against the old pixels
        # would keep the old aspect.
        assert im.size[0] == im.size[1]


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


def test_a_failed_roll_leaves_no_staging_file_behind(svc, monkeypatch):
    # Nothing ever looks at a dotfile again, so one stranded by a failure sits
    # in the job directory until the job is pruned. The cutout branch stages
    # the same way and now cleans up the same way.
    from warlock.pipelines import seam

    def boom(src, dest):
        dest.write_bytes(b"half a png")
        raise OSError("disk full")

    monkeypatch.setattr(seam, "wrap_preview", boom)
    job_id = _reference(svc, stage="tile")
    with pytest.raises(OSError):
        svc_derive.get_file(svc, job_id, "wrap_preview.png")
    assert not list(svc.job_dir(job_id).glob(".*.tmp"))


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
    before = first.read_bytes()
    _hand_edit(svc, job_id)
    again = svc_derive.get_file(svc, job_id, "wrap_preview.png")

    # By content, not by mtime. The rule under test is "a stale view is
    # re-derived", and the edit genuinely changes the pixels -- whereas two
    # writes a few milliseconds apart can share an mtime, because the system
    # clock behind a file timestamp ticks far more coarsely than the timestamp
    # can express. An mtime assertion here failed once for exactly that reason
    # while the behaviour it was checking was correct.
    assert again.read_bytes() != before


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


# -- the material set --------------------------------------------------------


@pytest.mark.parametrize("name", ["material_height.png", "material_normal.png",
                                  "material_roughness.png"])
def test_a_tile_derives_each_material_map_at_its_own_size(svc, name):
    job_id = _reference(svc, stage="tile")
    path = svc_derive.get_file(svc, job_id, name)
    assert path.name == name
    with Image.open(path) as im:
        assert im.size == (128, 128)


def test_a_reference_has_no_material_to_derive(svc):
    # The wrapped view's rule, from the other side: these are whole-frame
    # transforms, and a normal map of a barrel on a background is a normal map
    # of the background too. Refused at the service rather than hidden in a
    # pane, so the two cannot drift.
    job_id = _reference(svc)
    for name in svc_files.MATERIAL_2D:
        with pytest.raises(NotReady):
            svc_derive.get_file(svc, job_id, name)


def test_the_material_zip_carries_every_map_the_albedo_and_the_gltf(svc):
    import zipfile

    from warlock.pipelines import material as material_lib

    job_id = _reference(svc, stage="tile")
    path = svc_derive.get_file(svc, job_id, "material.zip")
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        doc = json.loads(zf.read("material.json"))
        readme = zf.read("README.txt").decode("utf-8")
    assert names == {"albedo.png", "material.json", "README.txt", *material_lib.MAP_NAMES}
    assert doc["pbrMetallicRoughness"]["metallicFactor"] == 0.0
    # The estimate has to travel with the file. A folder of textures called
    # material_normal.png carries every implication of a measured one, and the
    # docstring saying otherwise is in a repository the person unzipping this
    # does not have.
    assert "ESTIMATED" in readme


def test_the_material_zip_needs_no_map_already_on_disk(svc):
    # It re-derives into its own staging area rather than asking get_file for
    # the three maps, which would nest four artifact locks where the module has
    # an ordering rule about exactly two.
    job_id = _reference(svc, stage="tile")
    svc_derive.get_file(svc, job_id, "material.zip")
    for name in ("material_height.png", "material_normal.png", "material_roughness.png"):
        assert not (svc.job_dir(job_id) / name).exists()


def test_a_material_map_is_re_derived_after_a_hand_edit(svc):
    job_id = _reference(svc, stage="tile")
    before = svc_derive.get_file(svc, job_id, "material_height.png").read_bytes()
    _hand_edit(svc, job_id)
    assert svc_derive.get_file(svc, job_id, "material_height.png").read_bytes() != before


def test_a_failed_material_derivation_leaves_no_staging_file_behind(svc, monkeypatch):
    from warlock.pipelines import material as material_lib

    def boom(src, dest, name, **kwargs):
        dest.write_bytes(b"half a png")
        raise OSError("disk full")

    monkeypatch.setattr(material_lib, "write_map", boom)
    job_id = _reference(svc, stage="tile")
    with pytest.raises(OSError):
        svc_derive.get_file(svc, job_id, "material_normal.png")
    assert not list(svc.job_dir(job_id).glob(".*.tmp"))


def test_a_failed_zip_leaves_neither_its_own_staging_file_nor_a_maps(svc, monkeypatch):
    # The zip stages the maps *beside* its own tmp, so a failure part way
    # through has two kinds of leftover to clean up rather than one.
    from warlock.pipelines import material as material_lib

    real = material_lib.write_map
    calls = {"n": 0}

    def flaky(src, dest, name, **kwargs):
        calls["n"] += 1
        if calls["n"] > 1:
            dest.write_bytes(b"half a png")
            raise OSError("disk full")
        return real(src, dest, name, **kwargs)

    monkeypatch.setattr(material_lib, "write_map", flaky)
    job_id = _reference(svc, stage="tile")
    with pytest.raises(OSError):
        svc_derive.get_file(svc, job_id, "material.zip")
    assert not list(svc.job_dir(job_id).glob(".*.tmp*"))
    assert not (svc.job_dir(job_id) / "material.zip").exists()
