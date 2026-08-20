"""What trellis actually sees. Pillow-drawn fixtures only."""

from __future__ import annotations

from PIL import Image, ImageDraw

from warlock.pipelines import reference

BG = (200, 200, 200)


def _subject(size=256, box=(64, 64, 191, 191), mode="RGB"):
    im = Image.new(mode, (size, size), BG + ((255,) if mode == "RGBA" else ()))
    fill = (40, 90, 160) + ((255,) if mode == "RGBA" else ())
    ImageDraw.Draw(im).rectangle(list(box), fill=fill)
    return im


def test_measure_finds_the_subject_against_a_plain_background():
    report = reference.measure(_subject())
    assert report.ok
    assert report.bbox == (64, 64, 192, 192)
    assert 0.2 < report.occupancy < 0.3
    assert report.components == 1
    assert report.touches == ()


def test_measure_writes_nothing(tmp_path):
    src = tmp_path / "a.png"
    _subject().save(src)
    before = src.read_bytes()
    reference.measure_file(src)
    assert src.read_bytes() == before
    assert list(tmp_path.iterdir()) == [src]


def test_a_speck_is_rejected_as_too_small():
    # ~2% of the frame: small enough to fail MIN_OCCUPANCY, big enough that
    # the mask is still a measurement rather than a leaked fill.
    report = reference.measure(_subject(box=(110, 110, 145, 145)))
    assert not report.ok
    assert any("only" in r for r in report.reasons)


def test_a_mask_that_ate_the_subject_is_not_a_rejection():
    """A flood fill that escapes through a light subject's rim leaves the same
    near-empty mask a genuine speck would, and the leak is vastly more common
    -- a false "too small" would refuse perfectly good references."""
    report = reference.measure(_subject(box=(126, 126, 133, 133)))
    assert report.ok
    assert report.as_dict()["measured"] is False
    assert not report.reasons


def test_a_subject_running_off_the_frame_is_rejected():
    report = reference.measure(_subject(box=(-1, -1, 300, 200)))
    assert not report.ok
    assert any("runs off the edge" in r for r in report.reasons)


def test_two_objects_are_rejected():
    im = _subject(box=(20, 100, 90, 170))
    ImageDraw.Draw(im).rectangle([160, 100, 230, 170], fill=(160, 40, 40))
    report = reference.measure(im)
    assert not report.ok
    assert any("more than one object" in r for r in report.reasons)


def test_an_empty_frame_is_rejected():
    report = reference.measure(Image.new("RGB", (128, 128), BG))
    assert not report.ok
    assert report.reasons == ("No subject found against the background.",)


def test_alpha_is_used_as_the_mask_when_present():
    im = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle([32, 32, 95, 95], fill=(255, 0, 0, 255))
    report = reference.measure(im)
    assert report.alpha_source
    assert report.bbox == (32, 32, 96, 96)


def test_normalise_hits_the_target_and_centres_the_subject():
    small = _subject(box=(10, 10, 40, 40))
    out, report = reference.normalise(small)
    assert report.normalised
    assert report.bbox is not None
    x0, y0, x1, y1 = report.bbox
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    assert abs(cx - out.width / 2) <= 2
    assert abs(cy - out.height / 2) <= 2
    assert report.occupancy > 0.3


def test_normalise_never_invents_or_strips_an_alpha_channel():
    # The mask drives geometry only -- background removal stays trellis's job.
    rgb, _ = reference.normalise(_subject())
    assert rgb.mode == "RGB"
    rgba, _ = reference.normalise(_subject(mode="RGBA"))
    assert rgba.mode == "RGBA"


def test_normalise_carries_the_machine_codes_with_the_reasons():
    """``vectors`` turns codes into refused_<code> rates; a rebuilt report that
    dropped them would record a prep-enabled refusal with every bucket at zero,
    which counts as a failure mode that stopped happening."""
    im = _subject(box=(20, 100, 90, 170))
    ImageDraw.Draw(im).rectangle([160, 100, 230, 170], fill=(160, 40, 40))
    _out, report = reference.normalise(im)
    assert not report.ok
    assert len(report.codes) == len(report.reasons)
    assert "multi_object" in report.codes


def test_prepare_disabled_is_a_byte_exact_copy(tmp_path):
    src = tmp_path / "ref.png"
    _subject().save(src)
    dest = tmp_path / "reference.png"

    report = reference.prepare(src, dest, enabled=False)

    assert dest.read_bytes() == src.read_bytes()
    assert not report.normalised
    assert report.ok
    assert not list(tmp_path.glob(".*tmp*"))


def test_prepare_is_idempotent(tmp_path):
    src = tmp_path / "ref.png"
    _subject(box=(20, 20, 60, 60)).save(src)
    once = tmp_path / "one.png"
    twice = tmp_path / "two.png"

    reference.prepare(src, once)
    reference.prepare(once, twice)

    assert once.read_bytes() == twice.read_bytes()


def test_every_refusal_carries_a_machine_readable_code():
    """Refusal codes. ``reasons`` are sentences written for a person -- the
    corpus needs a key it can count, and parsing English back out of a stored
    report is how a reworded sentence silently empties a bucket."""
    two = _subject(box=(20, 100, 90, 170))
    ImageDraw.Draw(two).rectangle([160, 100, 230, 170], fill=(160, 40, 40))

    assert reference.measure(two).codes == ("multi_object",)
    assert reference.measure(_subject(box=(110, 110, 145, 145))).codes == ("occupancy",)
    assert reference.measure(_subject(box=(-1, -1, 300, 200))).codes == ("edge",)
    assert reference.measure(Image.new("RGB", (128, 128), BG)).codes == ("empty",)
    assert reference.measure(_subject()).codes == ()


def test_a_code_and_a_reason_are_raised_together():
    """One refusal, one code -- a rule that grew a reason and no code would
    make a refused job look like a passing one to the corpus."""
    wide = _subject(box=(-1, -1, 300, 200))
    report = reference.measure(wide)
    assert len(report.codes) == len(report.reasons)
    assert set(report.codes) <= set(reference.REFUSAL_CODES)


def test_the_codes_survive_the_round_trip_through_params():
    report = reference.measure(Image.new("RGB", (64, 64), BG))
    assert report.as_dict()["codes"] == ["empty"]


def test_report_is_json_safe():
    import json

    json.dumps(reference.measure(_subject()).as_dict())
