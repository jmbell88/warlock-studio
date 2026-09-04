"""What only a real card and real weights can answer about Muse.

Three things, and none of them is provable on CPU: that the vendored pipeline
loads and produces a valid WAV at all; that the cancel modification actually
stops a running generation rather than merely being present in the source; and
what the model really costs, which is where ``MusicModel.vram_gib`` and
``host_peak_gib`` stop being documented estimates.

Run with: uv run pytest tests/test_music_gpu.py -m gpu -n 0

Serial is enforced for the whole lane -- N workers means N simultaneous 8.3 GiB
loads onto one card -- and everything here goes through the real
``MusicClient``, so it is also the only place the subprocess boundary is
crossed with weights on the other side of it.
"""

from __future__ import annotations

import threading
import time
import wave

import pytest

from warlock import fetch, models
from warlock.config import get_config
from warlock.pipelines.music_client import MusicCancelled, MusicClient

pytestmark = [pytest.mark.gpu, pytest.mark.timeout(1800)]

#: Short, because every test here pays for it and none of them is judging the
#: music. Still above ``_jobs_music.MIN_DURATION``, so what runs is a request
#: the door would have accepted.
DURATION = 10.0

#: Likewise: enough steps that the sampler is genuinely sampling, few enough
#: that the lane stays runnable.
STEPS = 8


@pytest.fixture(scope="module")
def client():
    config = get_config()
    spec = models.MUSIC_MODELS[models.DEFAULT_MUSIC_MODEL]
    if not fetch.present(config, "music", spec):
        pytest.skip(f"{spec.label} weights not downloaded")
    made = MusicClient(spec, config.t2i_model_root / spec.dir_name)
    yield made
    made.close()


def _wav(path):
    """-> (rate, channels, frames). Raises if it is not a readable WAV."""
    with wave.open(str(path), "rb") as fh:
        return fh.getframerate(), fh.getnchannels(), fh.getnframes()


def test_a_generate_produces_a_wav_the_rest_of_the_app_can_read(client, tmp_path):
    """The end-to-end claim, and the format two other things depend on.

    44.1 kHz is not incidental: ``sirens_audio``'s mixer is open at exactly that
    rate and *refuses* a buffer at any other rather than resampling, so a model
    that started writing 48 kHz would silently break both the audition and the
    Sirens bridge. Asserting the rate here is what makes that a test rather
    than a hope.
    """
    out = tmp_path / "track.wav"
    got = client.generate(
        "solo piano, slow, minor key",
        out,
        audio_duration=DURATION,
        infer_step=STEPS,
        seed=1234,
    )
    assert got == out and out.is_file()
    rate, channels, frames = _wav(out)
    from warlock.studio import sirens_audio

    assert rate == sirens_audio.RATE
    assert channels == sirens_audio.CHANNELS
    # Roughly the length asked for. Loose, because the model rounds to its own
    # latent grid -- what is being caught here is a file that is silent, empty
    # or a different piece of music entirely.
    assert frames > rate * DURATION * 0.5


def test_the_same_seed_gives_the_same_track(client, tmp_path):
    """The parity check for the vendored code.

    A checksum rather than an ear: what this catches is a re-vendoring that
    changed the arithmetic, which is the failure mode
    ``pipelines/acestep/ATTRIBUTION.md``'s update procedure points at.
    """
    import hashlib

    def _run(path):
        client.generate(
            "solo piano, slow, minor key",
            path,
            audio_duration=DURATION,
            infer_step=STEPS,
            seed=4242,
        )
        return hashlib.sha256(path.read_bytes()).hexdigest()

    assert _run(tmp_path / "a.wav") == _run(tmp_path / "b.wav")


def test_a_different_seed_gives_a_different_track(client, tmp_path):
    """The other half, and the one that catches a seed that is not wired.

    A seed silently ignored passes the parity test above perfectly.
    """
    import hashlib

    def _run(path, seed):
        client.generate(
            "solo piano, slow, minor key",
            path,
            audio_duration=DURATION,
            infer_step=STEPS,
            seed=seed,
        )
        return hashlib.sha256(path.read_bytes()).hexdigest()

    assert _run(tmp_path / "a.wav", 1) != _run(tmp_path / "b.wav", 2)


def test_a_cancel_stops_a_running_generation(client, tmp_path):
    """WARLOCK 1/3, proved rather than read.

    ``ACEStepPipeline.__call__`` takes no cancel hook upstream; this is the one
    vendored modification the feature cannot work without, and its presence in
    the source is not evidence that the event reaches the loop. The generation
    is asked for at a step count that guarantees it is still running when the
    flag is set.
    """
    cancel = threading.Event()
    threading.Timer(8.0, cancel.set).start()
    started = time.monotonic()
    with pytest.raises(MusicCancelled):
        client.generate(
            "orchestral, slow build",
            tmp_path / "track.wav",
            audio_duration=120.0,
            infer_step=60,
            cancel_event=cancel,
        )
    # It returned because of the cancel, not because it finished: a full
    # 120 s / 60-step generation is minutes.
    assert time.monotonic() - started < 120.0
    # And the child survives it -- a cancel must not cost the warm pipeline,
    # which is the whole reason it is an event rather than a kill.
    assert client.loaded is True


def test_unload_ends_the_process_and_gives_the_card_back(client, tmp_path):
    """The measured claim behind the whole subprocess arrangement.

    In-process, an unload returns the VRAM and leaves the allocator's arenas
    holding host commit. Here the process ends, so both come back -- and the
    device reading the child published before it died is what the assertion
    compares against.
    """
    from warlock import vram

    client.generate(
        "solo piano", tmp_path / "track.wav", audio_duration=DURATION, infer_step=STEPS
    )
    proc = client._proc
    assert proc is not None and proc.poll() is None
    before = vram.device_memory()
    client.unload()
    assert proc.poll() is not None
    assert client.loaded is False
    # Re-read after the process has gone. Not asserted to be *exactly* the
    # spec's figure -- other things on the card move -- only that a meaningful
    # amount came back.
    after = vram.device_memory()
    if before is not None and after is not None:
        assert after.free_gib >= before.free_gib


@pytest.mark.perf
def test_the_registry_figures_are_not_under_the_real_cost(client, tmp_path, capsys):
    """Where ``vram_gib``/``host_peak_gib`` stop being estimates.

    Under-pricing is the one direction the door forbids: admission that lets a
    job in on a figure below what it actually takes is a job that OOMs at load,
    which is exactly the failure the check exists to prevent. Over-pricing only
    refuses a job that might have fitted, which is the safe error.

    The measurement is *printed* as well as asserted, because the figures it
    produces belong in a ``docs/measurements/`` document -- this test is how
    that document gets its numbers.
    """
    from warlock import memlog, vram

    spec = models.MUSIC_MODELS[models.DEFAULT_MUSIC_MODEL]
    client.unload()
    idle_device = vram.device_memory()
    idle_host = memlog.system_memory()
    client.generate(
        "solo piano", tmp_path / "track.wav", audio_duration=DURATION, infer_step=STEPS
    )
    loaded_device = vram.device_memory()
    loaded_host = memlog.system_memory()

    device_cost = host_cost = None
    if idle_device is not None and loaded_device is not None:
        device_cost = idle_device.free_gib - loaded_device.free_gib
    if idle_host is not None and loaded_host is not None:
        host_cost = getattr(loaded_host, "commit_gib", 0.0) - getattr(
            idle_host, "commit_gib", 0.0
        )
    with capsys.disabled():
        print(
            f"\nace_step_v1 measured: vram {device_cost} GiB "
            f"(registry {spec.vram_gib}), host commit {host_cost} GiB "
            f"(registry {spec.host_peak_gib})"
        )
    if device_cost is not None:
        assert spec.vram_gib >= device_cost, (
            "the registry under-prices this model; admission would let a job in "
            "that OOMs at load"
        )
