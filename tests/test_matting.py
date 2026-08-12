"""The matte: a model when the weights are there, a flood fill when not.

The fallback is the interesting half -- it is what makes every 2D export work
on a fresh checkout, and it must be indistinguishable in *shape* from the model
path so nothing downstream has to know which one ran.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from warlock import models
from warlock.pipelines import matting


@pytest.fixture(autouse=True)
def _no_model_carried_between_tests():
    # The cache (and the failure sentinel beside it) is module state that
    # outlives a test by design; nothing here should inherit either.
    matting.unload()
    yield
    matting.unload()


def _config(tmp_path):
    return SimpleNamespace(t2i_model_root=tmp_path)


def _weights(tmp_path):
    spec = models.MATTING_MODELS[models.DEFAULT_MATTING]
    root = tmp_path / spec.dir_name
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    return root


def _subject():
    im = Image.new("RGB", (96, 96), (200, 200, 200))
    ImageDraw.Draw(im).rectangle((24, 24, 72, 72), fill=(30, 30, 30))
    return im


def test_the_registry_entry_carries_a_download_command():
    spec = models.MATTING_MODELS[models.DEFAULT_MATTING]
    assert spec.dir_name and "hf download" in spec.download


def test_no_weights_means_not_available(tmp_path):
    assert matting.available(_config(tmp_path)) is False


def test_weights_on_disk_mean_available(tmp_path):
    spec = models.MATTING_MODELS[models.DEFAULT_MATTING]
    root = tmp_path / spec.dir_name
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    assert matting.available(_config(tmp_path)) is True


def test_without_weights_the_flood_fill_produces_the_mask(tmp_path):
    mask, source = matting.mask(_subject(), _config(tmp_path))
    assert source == "flood"
    assert mask.dtype == bool
    assert mask.shape == (96, 96)
    assert mask[48, 48] and not mask[2, 2]


def test_an_image_with_alpha_uses_it_whatever_the_weights_say(tmp_path):
    # subject_mask already prefers a real alpha channel, and a matting model
    # asked to re-cut an existing cutout can only make it worse.
    im = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle((8, 8, 24, 24), fill=(255, 0, 0, 255))
    mask, source = matting.mask(im, _config(tmp_path))
    assert source == "alpha"
    assert mask[16, 16] and not mask[0, 0]


def test_a_failing_model_falls_back_rather_than_raising(tmp_path, monkeypatch):
    # A corrupt or half-downloaded checkpoint must cost the user edge quality,
    # not the export: the flood fill is always there.
    spec = models.MATTING_MODELS[models.DEFAULT_MATTING]
    root = tmp_path / spec.dir_name
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        matting, "_model_mask", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    mask, source = matting.mask(_subject(), _config(tmp_path))
    assert source == "flood"
    assert mask.any()


def test_the_model_path_is_used_when_it_works(tmp_path, monkeypatch):
    spec = models.MATTING_MODELS[models.DEFAULT_MATTING]
    root = tmp_path / spec.dir_name
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    fake = np.zeros((96, 96), dtype=bool)
    fake[10:20, 10:20] = True
    monkeypatch.setattr(matting, "_model_mask", lambda image, path, device: fake)

    mask, source = matting.mask(_subject(), _config(tmp_path))

    assert source == "birefnet"
    assert np.array_equal(mask, fake)


def test_a_model_mask_that_finds_nothing_falls_back(tmp_path, monkeypatch):
    # An all-false matte would make every export raise NoSubject. The flood
    # fill's answer is worse-looking and right, which beats correct and empty.
    spec = models.MATTING_MODELS[models.DEFAULT_MATTING]
    root = tmp_path / spec.dir_name
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        matting, "_model_mask", lambda image, path, device: np.zeros((96, 96), dtype=bool)
    )
    _mask, source = matting.mask(_subject(), _config(tmp_path))
    assert source == "flood"


def test_an_opaque_alpha_channel_is_not_a_cutout(tmp_path):
    # Having a channel is not using one: a great many tools write RGBA whether
    # or not anything is transparent. Believing an opaque one returns the whole
    # frame as subject, labelled "alpha" -- the manifest's word for exact.
    mask, source = matting.mask(_subject().convert("RGBA"), _config(tmp_path))
    assert source == "flood"
    assert mask[48, 48] and not mask[2, 2]
    assert not mask.all()


def test_a_checkpoint_that_will_not_load_is_only_tried_once(tmp_path, monkeypatch, caplog):
    # An export is a loop over images. A broken install must cost one load
    # attempt and one line, not one attempt and one traceback per image.
    #
    # The memo lives in the *parent* now that the model loads in a child, and
    # that is the whole point of it: the child dies with its failure, so
    # without a parent-side sentinel a broken install would cost a spawn and a
    # from_pretrained per image rather than once per session.
    _weights(tmp_path)
    calls: list[str] = []

    def boom(payload, key):
        calls.append("load")
        raise matting._ChildFailed("RuntimeError: half a download", stage="load")

    monkeypatch.setattr(matting, "_request", boom)
    with caplog.at_level(logging.WARNING, logger="warlock.pipelines.matting"):
        for _ in range(3):
            _m, source = matting.mask(_subject(), _config(tmp_path))
            assert source == "flood"

    assert len(calls) == 1
    assert sum(record.exc_info is not None for record in caplog.records) == 1
    # ... and the memory of it is droppable, so repairing the download does not
    # need a restart.
    matting.unload()
    matting.mask(_subject(), _config(tmp_path))
    assert len(calls) == 2


def test_an_image_that_fails_to_matte_is_not_remembered(tmp_path, monkeypatch):
    # The other side of the memo, and the reason the worker reports a `stage`
    # at all: a checkpoint that will not load will not load again, but one
    # image that failed to matte -- a timeout, an unreadable PNG, a child that
    # died -- says nothing about the next one. Memoizing that would silently
    # turn one bad frame into a flood-filled batch.
    _weights(tmp_path)
    calls: list[str] = []

    def boom(payload, key):
        calls.append("run")
        raise matting._ChildFailed("the matting worker did not answer in 1s")

    monkeypatch.setattr(matting, "_request", boom)
    for _ in range(3):
        _m, source = matting.mask(_subject(), _config(tmp_path))
        assert source == "flood"
    assert len(calls) == 3
    assert matting.last_error() is None


def test_a_failed_load_is_remembered_where_doctor_can_read_it(tmp_path, monkeypatch):
    # The failure sentinel in _cache is enough to stop a second attempt, but it
    # is an anonymous object: doctor could see "matting is broken" from it and
    # never "how". A green weights row above a silent fall-back to the flood
    # fill is the exact confusion this module's docstring exists to prevent, so
    # the words of the exception are kept, not just the fact of one.
    _weights(tmp_path)
    assert matting.last_error() is None

    def boom(payload, key):
        raise matting._ChildFailed(
            "RuntimeError: No module named 'einops'", stage="load"
        )

    monkeypatch.setattr(matting, "_request", boom)
    matting.mask(_subject(), _config(tmp_path))
    recorded = matting.last_error()
    assert recorded is not None
    assert "einops" in recorded
    # Named, so the reader can tell an import from a corrupt safetensors. The
    # child is what produces this string now, but the shape of it is unchanged
    # -- doctor.py reads last_error() and knows nothing about a process.
    assert "RuntimeError" in recorded
    # And forgotten with the model, or repairing the install leaves doctor
    # reporting a failure that no longer happens.
    matting.unload()
    assert matting.last_error() is None


def test_the_service_fixture_does_not_inherit_this_machines_matting_weights(svc):
    # The same rule the svc fixture already applies to WARLOCK_TRELLIS_MODELS
    # and WARLOCK_GLTFPACK, and it went unnoticed only because matting was
    # broken: with the dtype bug fixed, every 2D export in the suite that ran
    # on a machine with models/birefnet downloaded started doing a real
    # ~12 s BiRefNet inference per image -- tests/test_derive_2d.py alone
    # burned 4624 CPU-seconds. Which matte a test gets must be a property of
    # the test, not of what the developer happened to download.
    assert matting.available(svc.config) is False


def test_the_input_tensor_carries_the_loaded_models_dtype(tmp_path, monkeypatch):
    # The published BiRefNet checkpoint stores fp16 weights, and the
    # preprocessing here is hand-rolled numpy, which is float32. The two met at
    # the first conv as "Input type (float) and bias type (struct c10::Half)"
    # -- caught by mask()'s blanket except, so every export on a host with the
    # weights present fell back to the corner fill and looked like no model.
    torch = pytest.importorskip("torch")
    seen: dict[str, object] = {}

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 1, 1)

        def forward(self, x):
            seen["dtype"] = x.dtype
            return [self.conv(x)]

    # Asserted against _infer, which is where the arithmetic went when
    # _model_mask became the IPC half: this is the *child's* code path, and it
    # is testable in-process precisely because it takes a loaded model rather
    # than a path.
    stub = Stub().half()
    out = matting._infer(_subject(), stub)
    assert seen["dtype"] is torch.float16
    assert out.shape == (96, 96)


def test_a_model_kept_on_the_cpu_is_cast_to_float32(tmp_path, monkeypatch):
    # Half precision on the CPU is emulated: the real checkpoint took 73 s a
    # frame against 11.5 s at float32 on this machine. This module's whole
    # bargain is "a second or two of host compute per export instead of VRAM",
    # so the cast is what keeps that bargain true. A CUDA device is left alone
    # -- there half is both correct and faster.
    torch = pytest.importorskip("torch")
    _weights(tmp_path)

    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 1, 1)

    # ``birefnet.load``, not ``transformers``: the modelling code is vendored
    # now and nothing is built out of the checkpoint directory. What is being
    # tested is unchanged -- the cast is ``matting``'s, and it is the reason
    # the CPU path is usable at all.
    from warlock.pipelines import birefnet

    monkeypatch.setattr(birefnet, "load", lambda _path: Stub().half())
    model = matting._load(tmp_path / "birefnet", "cpu")
    assert next(model.parameters()).dtype is torch.float32


# -- the child ----------------------------------------------------------------
#
# BiRefNet loads in a subprocess because the 1475 MB it costs does not come back
# in-process (docs/measurements/2026-08-08-load-probe-memory.md). These drive
# the real spawn, through matting.CHILD_ARGV, with a scripted child in place of
# the real worker -- so the pipe, the marker filtering and the PNG round trip
# are exercised rather than patched out.

_CHILD_PREAMBLE = """
import sys, types
sys.modules["transformers"] = types.SimpleNamespace(
    AutoModelForImageSegmentation=types.SimpleNamespace(from_pretrained=None)
)
from warlock.pipelines import matting, matting_worker
"""


def _scripted_child(body: str) -> list[str]:
    return [sys.executable, "-c", _CHILD_PREAMBLE + body]


def test_a_matte_round_trips_through_the_child(tmp_path, monkeypatch):
    # The whole boundary end to end: the request goes out as one JSON line, the
    # pixels travel as files (a 4096-square matte is 16 MB and the pipe buffer
    # is 64 KB), and what comes back is a boolean array the caller cannot tell
    # from an in-process one.
    _weights(tmp_path)
    monkeypatch.setattr(
        matting,
        "CHILD_ARGV",
        _scripted_child(
            """
import numpy as np
# A "model" that is a rectangle, and the arithmetic half stubbed with it: this
# test is about the transport, not about BiRefNet.
def fake_infer(image, model):
    out = np.zeros((image.size[1], image.size[0]), dtype=bool)
    out[10:20, 10:20] = True
    return out
matting._load = lambda path, device: object()
matting._infer = fake_infer
# Library chatter on the same stream, which the marker exists to survive.
print("Loading checkpoint shards: 100%", file=sys.stderr)
raise SystemExit(matting_worker.main())
"""
        ),
    )
    mask, source = matting.mask(_subject(), _config(tmp_path))
    assert source == "birefnet"
    assert mask.shape == (96, 96)
    assert mask[15, 15] and not mask[50, 50]

    # And the child is held across requests -- that is the whole reason it is
    # persistent rather than one-shot like loadprobe: derive.get_file mattes an
    # icon, a sprite set and a pixel set for one asset.
    pid = matting._proc.pid
    matting.mask(_subject(), _config(tmp_path))
    assert matting._proc.pid == pid  # one load, three mattes

    # unload() means it now: it used to clear a dict, which returned about a
    # third of what the load cost. The process is gone, and with it every byte
    # the allocator's arenas were holding.
    proc = matting._proc
    matting.unload()
    assert matting._proc is None
    # _reset_child_locked kills and *waits*, so a returned unload is the
    # evidence -- poll() is not None rather than "we asked it to die".
    assert proc.poll() is not None


def test_a_child_that_cannot_be_spawned_falls_back_to_the_fill(tmp_path, monkeypatch):
    # A spawn failure is an exception mask() already catches, and the corner
    # fill is always there. It memoizes as a load failure, because a python
    # that will not start will not start for the next image either.
    _weights(tmp_path)
    monkeypatch.setattr(
        matting, "CHILD_ARGV", [str(tmp_path / "no-such-interpreter"), "-c", ""]
    )
    mask, source = matting.mask(_subject(), _config(tmp_path))
    assert source == "flood"
    assert mask.any()
    assert matting.last_error() is not None


def test_a_child_that_never_answers_times_out_and_falls_back(tmp_path, monkeypatch):
    # A worker wedged inside a C inference call checks nothing and closes
    # nothing, so a bare readline() would wedge the export with it. The deadline
    # kills it and the fill answers -- and it is *not* memoized, because a
    # timeout is about one image.
    _weights(tmp_path)
    monkeypatch.setattr(matting, "REQUEST_TIMEOUT", 1.0)
    monkeypatch.setattr(
        matting,
        "CHILD_ARGV",
        _scripted_child("import time\nsys.stdin.readline()\ntime.sleep(60)\n"),
    )
    mask, source = matting.mask(_subject(), _config(tmp_path))
    assert source == "flood"
    assert mask.any()
    assert matting.last_error() is None
    assert matting._proc is None


def test_a_child_that_dies_mid_request_falls_back(tmp_path, monkeypatch):
    _weights(tmp_path)
    monkeypatch.setattr(
        matting, "CHILD_ARGV", _scripted_child("sys.stdin.readline()\nraise SystemExit(3)\n")
    )
    _mask, source = matting.mask(_subject(), _config(tmp_path))
    assert source == "flood"
    assert matting._proc is None


def test_the_packages_birefnets_modelling_code_imports_are_declared(tmp_path):
    # BiRefNet's own modelling code -- which trust_remote_code runs out of the
    # checkpoint directory -- imports these, and transformers reaches for
    # torchvision to build the fast image processor DINOv2 ranking uses. None
    # of them were declared, so _load failed on a machine that had every weight
    # on disk and `uv sync` would remove torchvision from one that worked.
    import tomllib

    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    extra = data["project"]["optional-dependencies"]["text2image"]
    names = {req.split(">")[0].split("=")[0].split(";")[0].strip() for req in extra}
    assert {"einops", "kornia", "timm", "torchvision", "transformers"} <= names
