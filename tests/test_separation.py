"""Stem separation: the registry, the door, and the names on disk.

No card, no weights and no child process. What is asserted is everything that
decides *whether* a separation runs and *where its files land* -- the two halves
that are wrong silently. The model itself is the gpu lane's job.

The recurring theme is that four separate places name the same four stems, and
nothing but this file makes them agree: ``SeparationModel.sources`` (the model's
constructor argument), ``files.MEDIA`` (the export allowlist), ``files.LISTED``
(what a finished job reports) and the queue's own literal (which cannot import
the service).
"""

from __future__ import annotations

import pytest

from warlock import _q_music, fetch, models, vram
from warlock.service import _jobs_rework as rework
from warlock.service import files
from warlock.service.errors import Conflict, Invalid

# --- the four names ----------------------------------------------------------


def test_every_source_the_model_returns_has_a_media_entry():
    """``sources`` is three things at once -- the constructor argument, the
    filenames and the allowlist keys -- so a fifth stem added to the registry
    with no MEDIA entry would be a file the worker writes and nothing serves."""
    spec = models.SEPARATION_MODELS[models.DEFAULT_SEPARATION]
    for name in spec.sources:
        assert f"{files.STEMS_DIR}/{name}.wav" in files.MEDIA


def test_the_stem_names_are_literals_rather_than_a_pattern():
    """MEDIA is the allowlist that keeps a caller-supplied string off the
    filesystem, and a ``stems/{name}.wav`` pattern is a hole in exactly that."""
    assert set(files.STEM_FILES) <= set(files.MEDIA)
    assert not any("{" in key or "*" in key for key in files.MEDIA)


def test_no_media_key_escapes_the_job_directory():
    """``collect`` and ``derive.get_file`` both do ``job_dir / name``.

    ``stems/drums.wav`` is the first MEDIA key with a separator in it at all, so
    this is the moment that join stops being obviously safe. Asserted once here
    rather than defended at every join -- ``sirens_io._under``'s belt and
    braces, as a test.
    """
    from pathlib import PurePosixPath, PureWindowsPath

    for name in files.MEDIA:
        for flavour in (PurePosixPath, PureWindowsPath):
            path = flavour(name)
            assert not path.is_absolute(), name
            assert ".." not in path.parts, name
            assert not path.drive, name


def test_the_queues_literal_agrees_with_the_services():
    """The queue may not import the service (``test_queue`` enforces it), so
    ``stems`` is written down twice. This is what stops that being drift."""
    from pathlib import Path

    source = Path(_q_music.__file__).read_text(encoding="utf-8")
    assert f'source_dir / "{files.STEMS_DIR}"' in source


def test_a_finished_take_lists_its_stems():
    assert set(files.STEM_FILES) <= set(files.LISTED)


# --- readiness ---------------------------------------------------------------


def _music_row(status: str = "done") -> dict:
    return {"kind": "music", "stage": "music", "status": status}


def test_a_stem_is_not_ready_until_the_sidecar_lands(tmp_path):
    """``stems.json`` is the completion gate, for ``rig.json``'s reason: the
    four WAVs appear one at a time, so their existence cannot say the set is
    finished -- and a reader that took it that way would offer a take with
    three stems as a take with four."""
    stems = tmp_path / files.STEMS_DIR
    stems.mkdir()
    (stems / "drums.wav").write_bytes(b"x")
    name = f"{files.STEMS_DIR}/drums.wav"
    assert files.ready(_music_row(), tmp_path, name) is False
    (stems / "stems.json").write_text("{}")
    assert files.ready(_music_row(), tmp_path, name) is True


def test_an_unsplit_take_says_so_rather_than_naming_a_file(tmp_path):
    name = f"{files.STEMS_DIR}/vocals.wav"
    assert "stems" in files.unready_reason(_music_row(), tmp_path, name)


# --- the derived audio formats ----------------------------------------------


def test_the_derived_formats_are_derived_from_the_track_not_the_mesh(tmp_path):
    """Its own tuple rather than a member of ``DERIVED``: that one is keyed on
    ``model.glb``, so a fourth name in it would make a take's FLAC wait for a
    mesh it will never have."""
    assert not set(files.DERIVED_AUDIO) & set(files.DERIVED)
    (tmp_path / "track.wav").write_bytes(b"x")
    for name in files.DERIVED_AUDIO:
        assert files.ready(_music_row(), tmp_path, name) is True


def test_a_derived_format_of_a_take_with_no_audio_names_the_track(tmp_path):
    reason = files.unready_reason(_music_row(), tmp_path, "track.flac")
    assert "track" in reason


def test_every_derived_format_has_a_media_type_and_an_encoder():
    from warlock.pipelines import audioout

    assert set(files.DERIVED_AUDIO) == set(audioout.FORMATS)
    for name in files.DERIVED_AUDIO:
        assert name in files.MEDIA


def test_the_encoder_refuses_a_name_it_does_not_write(tmp_path):
    from warlock.pipelines import audioout

    with pytest.raises(ValueError, match="not a format"):
        audioout.convert(tmp_path / "a.wav", tmp_path / "b", "track.aiff")


def test_a_take_round_trips_through_every_format(tmp_path):
    """libsndfile encodes all four with no ffmpeg and no second binary. Worth
    an actual round trip rather than a claim, because everyone assumes
    otherwise and would reach for a converter."""
    import numpy as np
    import soundfile as sf

    from warlock.pipelines import audioout

    source = tmp_path / "track.wav"
    tone = np.sin(np.arange(4410, dtype=np.float32) / 20.0) * 0.5
    sf.write(str(source), np.stack([tone, tone], axis=1), 44100, subtype="PCM_16")
    for name in audioout.FORMATS:
        out = tmp_path / name
        audioout.convert(source, out, name)
        data, rate = sf.read(str(out), always_2d=True)
        assert rate == 44100
        assert data.shape[1] == 2


# --- the registry ------------------------------------------------------------


def test_the_separation_model_is_labelled_non_commercial():
    """Meta states the trained weights are for scientific purposes only, and
    htdemucs was trained the same way with no new grant. This app's purpose is
    making assets people sell, so the flag is what puts the red marker and the
    warning in front of the download."""
    spec = models.SEPARATION_MODELS[models.DEFAULT_SEPARATION]
    assert spec.commercial is False
    assert spec.license_note


def test_the_checkpoint_is_pinned_by_digest_rather_than_a_revision():
    """It is not on the Hub, so there is no commit to name. A digest is the
    stronger half of the same promise: a revision names an immutable commit, a
    digest *is* the artifact."""
    spec = models.SEPARATION_MODELS[models.DEFAULT_SEPARATION]
    one = spec.fetch[0]
    assert one.repo_id == ""
    assert one.url.startswith("https://")
    assert len(one.sha256) == 64
    assert one.filename in spec.probe


def test_it_has_its_own_downloadable_row():
    keys = {entry.kind for entry in fetch.entries()}
    assert "separation" in keys


def test_a_partial_directory_reads_as_absent(tmp_path, monkeypatch):
    """The presence probe names files rather than using the generic
    ``config.json`` + safetensors tail, and it has to: this download is a
    single ``.pt`` with no config beside it, so the tail would report it
    absent forever."""

    class _Config:
        t2i_model_root = tmp_path

    spec = models.SEPARATION_MODELS[models.DEFAULT_SEPARATION]
    assert fetch.present(_Config(), "separation", spec) is False
    (tmp_path / spec.dir_name).mkdir(parents=True)
    (tmp_path / spec.dir_name / spec.probe[0]).write_bytes(b"x")
    assert fetch.present(_Config(), "separation", spec) is True


def test_a_zero_length_checkpoint_is_reported_as_suspect(tmp_path):
    """One file *is* the model, so a zero-length one is the whole thing missing
    while every presence probe says it is installed (MDL-08)."""

    class _Config:
        t2i_model_root = tmp_path

    spec = models.SEPARATION_MODELS[models.DEFAULT_SEPARATION]
    (tmp_path / spec.dir_name).mkdir(parents=True)
    (tmp_path / spec.dir_name / spec.probe[0]).write_bytes(b"")
    assert fetch.suspect_files(_Config(), "separation", spec)


# --- admission ---------------------------------------------------------------


def test_a_one_shot_child_credits_no_resident_weights_back():
    """The interesting difference from the music branch.

    The second return value is the resident-checkpoint credit
    ``queue._check_resources`` gives back, and a process that dies at the end of
    the job has nothing to credit -- pricing it like the music pipe would tell
    the queue a permanent 4 GiB had been freed.
    """
    cost, credit = vram.estimate_parts("separate", "music", {}, exclusive=True)
    assert cost > 0.0
    assert credit == 0.0


def test_the_progress_phases_are_registered():
    """An unregistered kind draws ``PHASES_IMAGE``, whose phases it never
    emits -- so the bar sits at zero and then jumps."""
    from warlock import progress

    assert progress.phases_for("separate") is progress.PHASES_SEPARATE


# --- the door ----------------------------------------------------------------


def _take(svc, status: str = "done") -> str:
    job_id = svc.store.create("music", "dark ambient", {}, stage="music")
    (svc.config.job_dir(job_id)).mkdir(parents=True, exist_ok=True)
    (svc.config.job_dir(job_id) / "track.wav").write_bytes(b"x")
    svc.store.set_status(job_id, status)
    return job_id


@pytest.fixture(autouse=True)
def _admitted(monkeypatch):
    monkeypatch.setattr(rework, "check_weights", lambda svc, kind, params: None)
    monkeypatch.setattr(rework, "check_vram", lambda svc, kind, stage, params: None)


def test_a_split_is_a_queued_row_naming_its_source(svc):
    """A queued job rather than a task-thread action, and
    ``retexture_job``'s docstring is the deciding sentence: a TaskRunner thread
    racing the worker for VRAM is the OOM that only reproduces under load."""
    take = _take(svc)
    out = rework.separate_job(svc, take)
    row = svc.store.get(out["id"])
    assert row["kind"] == "separate"
    assert row["params"]["source_job"] == take
    assert row["params"]["separation_model"] == models.DEFAULT_SEPARATION


def test_only_a_track_can_be_split(svc):
    job_id = svc.store.create("image", "a goblin", {}, stage="model")
    svc.store.set_status(job_id, "done")
    with pytest.raises(Invalid) as caught:
        rework.separate_job(svc, job_id)
    assert caught.value.field == "source_job"


def test_a_take_with_no_audio_is_refused_in_the_same_words(svc):
    """``muse_mode.play``'s sentence and ``derive_music_job``'s, so all three
    surfaces say the same thing about the same missing file."""
    take = _take(svc)
    (svc.config.job_dir(take) / "track.wav").unlink()
    with pytest.raises(Invalid) as caught:
        rework.separate_job(svc, take)
    assert "no audio on disk" in str(caught.value)


def test_a_running_take_cannot_be_split_yet(svc):
    take = _take(svc, status="running")
    with pytest.raises(Conflict):
        rework.separate_job(svc, take)


def test_an_unknown_model_is_refused_rather_than_defaulted(svc):
    take = _take(svc)
    with pytest.raises(Invalid) as caught:
        rework.separate_job(svc, take, separation_model="demucs_v4")
    assert caught.value.field == "separation_model"


def test_a_split_cannot_be_rerolled(svc):
    """It is deterministic: re-running writes the identical four files over
    themselves, so a reroll is a press with no outcome."""
    from warlock.service import _jobs_resubmit as resubmit

    take = _take(svc)
    split = svc.store.create("separate", "x", {"source_job": take})
    svc.store.set_status(split, "done")
    with pytest.raises(Invalid, match="no seed to change"):
        resubmit.rerun_job(svc, split)
