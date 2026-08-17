"""The E-section of NEXT_ROADMAP Phase 2: failures that reach the user.

Every test here pins a *message* or a *state that makes one reachable*, which
is unusual and deliberate. The bugs this section fixed were not wrong answers --
each of these paths already did the right thing internally and logged it. What
they got wrong was saying so, and a message nothing asserts is a message the
next refactor quietly deletes.
"""

from __future__ import annotations

import pytest

from warlock import guidance
from warlock.service import files as svc_files
from warlock.service.errors import Failed, Invalid, invalid_from
from warlock.service.validation import not_done_message

# --- E43 / E45: the two silent cache failures -------------------------------


class _BrokenCount:
    """A store whose ``count`` raises and whose ``list`` does not."""

    def __init__(self, rows):
        self._rows = rows

    def list(self, *a, **k):
        return list(self._rows)

    def count(self):
        raise RuntimeError("database is locked")


class _Svc:
    def __init__(self, store):
        self.store = store

    def job_dir(self, job_id):  # pragma: no cover - not reached here
        raise AssertionError


def _cache(rows, limit):
    from warlock.studio.jobs_cache import JobsCache

    cache = JobsCache(_Svc(_BrokenCount(rows)), limit=limit)
    return cache


def test_a_failed_count_is_recorded_rather_than_hidden_behind_a_full_page(monkeypatch):
    """E43. The fallback ``total = len(jobs)`` is indistinguishable from "this
    *is* the whole history" on a full page, which silently retracts the only
    control that reaches anything older."""
    from warlock.studio import jobs_cache as mod

    rows = [{"id": f"j{i}", "status": "done"} for i in range(4)]
    monkeypatch.setattr(mod.svc_jobs, "list_jobs", lambda *a, **k: rows)
    cache = _cache(rows, limit=4)  # a *full* page: len(jobs) == limit

    assert cache.tick() is True
    assert cache.total == len(rows)
    assert cache.count_error == "database is locked"


def test_a_recovered_count_clears_the_warning(monkeypatch):
    from warlock.studio import jobs_cache as mod

    rows = [{"id": "j0", "status": "done"}]
    monkeypatch.setattr(mod.svc_jobs, "list_jobs", lambda *a, **k: rows)
    cache = _cache(rows, limit=4)  # a short page never runs COUNT(*) at all
    cache.tick()
    assert cache.count_error is None


def test_a_failed_storage_walk_says_so_and_keeps_the_last_good_figure(monkeypatch):
    """E45. Returning the previous dict is right -- a stale figure beats a
    blank one -- but on its own it presents last hour's number as current."""
    from warlock.studio import jobs_cache as mod

    def boom(_svc):
        raise OSError("the data directory vanished")

    monkeypatch.setattr(mod.svc_jobs, "storage_sizes", boom)
    cache = _cache([], limit=4)
    cache.storage = {"job_dirs": 3, "bytes": 99}

    assert cache.measure() == {"job_dirs": 3, "bytes": 99}
    assert "vanished" in (cache.storage_error or "")


# --- E44: a cancelled picker and a broken one are different facts -----------


def test_a_picker_that_fails_to_open_raises_rather_than_looking_like_a_cancel(monkeypatch):
    from warlock.studio import dialogs

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("no portal on this host")

    monkeypatch.setattr(dialogs.pfd, "open_file", _Boom)
    with pytest.raises(Failed):
        dialogs.open_file("Pick one")

    monkeypatch.setattr(dialogs.pfd, "save_file", _Boom)
    with pytest.raises(Failed):
        dialogs.save_file("Save", "x.png")


def test_a_cancelled_picker_is_still_plain_none(monkeypatch):
    """The other half, and the reason the raise is safe: every caller reads
    ``None`` as "the user changed their mind" and must keep doing so."""
    from warlock.studio import dialogs

    class _Cancelled:
        def __init__(self, *a, **k):
            pass

        def result(self):
            return []

    monkeypatch.setattr(dialogs.pfd, "open_file", _Cancelled)
    assert dialogs.open_file("Pick one") is None


# --- E49 / S137: library text never reaches the user unframed ---------------


def test_a_wrapped_library_error_keeps_the_detail_and_gains_a_sentence():
    exc = ValueError("bones payload is empty")
    wrapped = invalid_from(exc, "That pose cannot be saved")
    assert isinstance(wrapped, Invalid)
    assert wrapped.message == "That pose cannot be saved: bones payload is empty"


def test_a_guidance_refusal_carries_the_control_it_came_from():
    """S137. ``Invalid`` has had ``field`` since it was written; the guidance
    passthroughs could never supply one, so the app's most common refusals
    highlighted nothing."""
    with pytest.raises(guidance.GuidanceError) as caught:
        guidance.normalize({"platform": "not-a-platform"})
    assert caught.value.field == "platform"

    wrapped = invalid_from(caught.value, "Those generation settings are not usable")
    assert wrapped.field == "platform"


def test_a_guidance_error_is_still_a_value_error():
    """The whole reason the field could be added without touching any caller:
    every ``except ValueError`` in the repo keeps working unchanged."""
    assert issubclass(guidance.GuidanceError, ValueError)


@pytest.mark.parametrize(
    "raw, field",
    [
        ({"size_m": 900.0}, "size_m"),
        ({"lora_weight": 12.0}, "lora_weight"),
        ({"bg_removal": "magic"}, "bg_removal"),
        ({"negative_prompt": "x" * 5000}, "negative_prompt"),
        ({"ip_scale": 99.0}, "ip_scale"),
        ({"platform": "nonsense"}, "platform"),
    ],
)
def test_every_bounded_guidance_field_names_itself(raw, field):
    with pytest.raises(guidance.GuidanceError) as caught:
        guidance.normalize(raw)
    assert caught.value.field == field


def test_the_service_wraps_rather_than_forwards_a_guidance_message(svc):
    from warlock.service import jobs as svc_jobs

    with pytest.raises(Invalid) as caught:
        svc_jobs.create_job(
            svc, kind="text", prompt="a rock", guidance_fields={"platform": "???"}
        )
    assert caught.value.message.startswith("Those generation settings are not usable:")
    assert caught.value.field == "platform"


# --- E50: a status word is not a sentence ----------------------------------


@pytest.mark.parametrize("status", ["queued", "running", "error", "cancelled"])
def test_a_refused_reference_explains_the_status_rather_than_naming_it(status):
    message = not_done_message("That reference", status)
    assert message != f"That reference is {status}."
    assert message.endswith(".")


def test_an_unknown_status_falls_back_to_the_word_rather_than_a_wrong_sentence():
    assert not_done_message("That reference", "paused") == "That reference is paused."


# --- E51: "file not ready" was true of five different situations ------------


def test_an_unready_artifact_says_which_of_the_reasons_it_is(tmp_path):
    job = {"status": "queued", "stage": "model"}
    reason = svc_files.unready_reason(job, tmp_path, "model.glb")
    assert "queue" in reason
    assert "model.glb" in reason


def test_an_unrigged_asset_says_it_is_unrigged_rather_than_not_ready(tmp_path):
    job = {"status": "done", "stage": "model"}
    assert "rigged" in svc_files.unready_reason(job, tmp_path, "rig.glb")


def test_a_2d_export_asked_of_a_mesh_says_which_stage_it_belongs_to(tmp_path):
    job = {"status": "done", "stage": "model"}
    reason = svc_files.unready_reason(job, tmp_path, "icon.png")
    assert "reference" in reason


def test_the_explanation_never_contradicts_the_gate(tmp_path):
    """``ready`` stays the authority; this only ever runs having been refused.
    Pinned so the two cannot be swapped by mistake."""
    job = {"status": "done", "stage": "reference"}
    (tmp_path / "input.png").write_bytes(b"x")
    assert svc_files.ready(job, tmp_path, "input.png") is True


# --- E53: what would have worked -------------------------------------------


def test_an_unreadable_upload_names_the_formats_that_would_have_worked():
    with pytest.raises(Invalid) as caught:
        svc_files._check_pixels(b"not an image at all")
    message = caught.value.message
    for fmt in ("PNG", "JPEG", "WebP", "BMP"):
        assert fmt in message


# --- E52: a Failed defers to the log, and nothing else does -----------------


def test_a_failed_carries_the_log_action_and_an_invalid_does_not():
    from warlock.studio.tasks import TaskRunner

    runner = TaskRunner(workers=1)
    try:
        runner.submit("a", _raise, Failed("gltfpack returned garbage"))
        runner.submit("b", _raise, Invalid("size must be between 0.01 and 100 m"))
        done = _drain(runner, 2)
    finally:
        runner.shutdown()
    by_key = {d.key: d for d in done}
    assert by_key["a"].action == "log"
    assert by_key["b"].action is None
    # The message itself is untouched either way: the action is the addition.
    assert by_key["a"].message == "gltfpack returned garbage"


def _raise(exc):
    raise exc


def _drain(runner, count, timeout=5.0):
    import time

    out = []
    deadline = time.monotonic() + timeout
    while len(out) < count and time.monotonic() < deadline:
        out.extend(runner.poll())
        time.sleep(0.01)
    assert len(out) == count, f"only {len(out)} of {count} tasks finished"
    return out
