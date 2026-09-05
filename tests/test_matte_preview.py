"""The promote-time matte preview, and the alpha it hands to trellis.

Three separate claims, and they are separable on purpose:

* the preview itself is a pure function of a job's ``input.png`` -- a cutout
  composited over a checkerboard, plus the composition gate's own verdict --
  and it is cached under git's racily-clean rule, so a file written inside its
  own mtime tick is never remembered;
* a reference that carries a real (non-opaque) alpha channel promotes as an
  *approved* matte: the flag is recorded and the server is told the preserving
  mode, because ``trellis-server.exe --help`` says ``auto`` keeps a pre-matted
  image's alpha and ``birefnet`` -- today's default -- re-cuts it;
* ``matte`` is an input, so it joins the config vector and stays out of
  ``DERIVED_PARAMS``.
"""

from __future__ import annotations

import io
import time
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from warlock import vectors
from warlock.service import jobs as svc_jobs
from warlock.service import matte as svc_matte
from warlock.service.errors import Invalid
from warlock.service.validation import DERIVED_PARAMS
from warlock.studio import matte_preview


def _png(pixels: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(pixels, "RGBA" if pixels.shape[2] == 4 else "RGB").save(buf, "PNG")
    return buf.getvalue()


def _subject_rgb(size: int = 96) -> np.ndarray:
    """A dark square on a plain light background -- what the corner fill is for."""
    arr = np.full((size, size, 3), 230, dtype=np.uint8)
    arr[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = (20, 30, 40)
    return arr


def _cutout_rgba(size: int = 96) -> np.ndarray:
    """The same subject, already matted: transparent everywhere else."""
    arr = np.zeros((size, size, 4), dtype=np.uint8)
    arr[size // 4 : 3 * size // 4, size // 4 : 3 * size // 4] = (20, 30, 40, 255)
    return arr


def _reference(svc, pixels: np.ndarray) -> str:
    result = svc_jobs.import_reference(svc, _png(pixels), name="subject")
    return result["id"]


# --- the preview ------------------------------------------------------------


def test_preview_returns_a_checkerboarded_cutout_and_the_gates_verdict(svc):
    job_id = _reference(svc, _subject_rgb())

    preview = svc_matte.preview(svc, job_id)

    assert preview.job_id == job_id
    assert preview.width > 0 and preview.height > 0
    assert len(preview.rgb) == preview.width * preview.height * 3
    # No weights on this host, so the corner fill answered -- which is a
    # measurement, not a failure.
    assert preview.source in ("alpha", "birefnet", "flood")
    assert 0.0 < preview.coverage < 1.0
    # The background is the checkerboard rather than the source's light grey.
    pixels = np.frombuffer(preview.rgb, dtype=np.uint8).reshape(
        preview.height, preview.width, 3
    )
    corner = set(int(v) for v in pixels[0, :16, 0])
    assert corner <= {svc_matte.CHECKER_LIGHT, svc_matte.CHECKER_DARK}


def test_preview_reports_the_reference_gates_warnings(svc):
    # A subject that runs to every edge: the gate refuses it, and the promote
    # preview is where the user should learn that -- before the GPU minutes.
    arr = np.full((96, 96, 3), 20, dtype=np.uint8)
    job_id = _reference(svc, arr)

    preview = svc_matte.preview(svc, job_id)

    assert preview.reasons or preview.warnings


def test_preview_refuses_a_job_with_no_image(svc, store):
    job_id = svc.store.create("text", "a crate", {}, "deadbeefcafe", stage="reference")
    with pytest.raises(Invalid):
        svc_matte.preview(svc, "deadbeefcafe")
    assert job_id is not None


def test_a_stamp_is_only_remembered_once_its_mtime_is_safely_past(svc):
    """git's racily-clean rule, exactly as ``files.attach_files`` applies it.

    A directory's -- or a file's -- mtime on Windows comes off a clock that
    ticks every 15.6 ms, so a write landing after the read but inside the
    stamped mtime's own tick is invisible to the stamp *forever*. A stamp is
    therefore stored only once it is comfortably in the past.
    """
    job_id = _reference(svc, _subject_rgb())
    cache: dict = {}

    fresh = svc_matte.preview(svc, job_id)
    fresh = svc_matte.replace_stamp(fresh, time.time_ns())
    assert svc_matte.remember(cache, fresh) is False
    assert svc_matte.cached(cache, job_id, fresh.stamp) is None

    old = svc_matte.replace_stamp(fresh, time.time_ns() - 10 * svc_matte.MTIME_RACE_NS)
    assert svc_matte.remember(cache, old) is True
    assert svc_matte.cached(cache, job_id, old.stamp) is old
    # A different stamp is a different file: the cache must not answer for it.
    assert svc_matte.cached(cache, job_id, old.stamp + 1) is None


def test_the_stamp_is_the_input_pngs_own_mtime(svc):
    job_id = _reference(svc, _subject_rgb())
    src = svc.job_dir(job_id) / "input.png"

    assert svc_matte.stamp_for(src) == src.stat().st_mtime_ns
    assert svc_matte.stamp_for(src.with_name("nothing.png")) is None


def test_pump_submits_and_reports_when_input_png_is_missing_on_first_check(svc):
    """2026-09-05 audit, finding create-03.

    ``MatteState.failed_stamp`` defaults to ``None``, and ``stamp_for`` also
    returns ``None`` for a missing ``input.png`` -- so a reference whose file
    is absent the first time ``pump`` looks made the guard
    ``failed_stamp == stamp`` true before any submit was ever tried. The
    preview task was never submitted, ``on_task_failed`` never ran, and the
    modal sat on "Cutting the subject out..." with no toast, forever.
    """
    job_id = _reference(svc, _subject_rgb())
    (svc.job_dir(job_id) / "input.png").unlink()

    class Ctx:
        def __init__(self) -> None:
            self.state = SimpleNamespace(matte=None)
            self.svc = svc
            self.submitted: list = []

        def submit(self, key, fn, *args, **kwargs):
            self.submitted.append(key)
            return True

    ctx = Ctx()
    matte_preview.open_for(ctx, job_id, {})

    state = matte_preview.pump(ctx)

    assert ctx.submitted == [matte_preview.key(job_id)]
    assert state.preview is None


# --- approved alpha ---------------------------------------------------------


def test_a_matted_reference_promotes_as_an_approved_matte(svc):
    job_id = _reference(svc, _cutout_rgba())

    result = svc_jobs.promote_to_model(svc, job_id, force=True)
    job = svc.require_job(result["id"])

    assert job["params"]["matte"] == "approved"
    # `auto` is the mode that keeps a pre-matted image's alpha; `birefnet`
    # would re-cut it, and it is what the default would otherwise have been.
    assert job["params"]["bg_removal"] == "auto"


def test_an_opaque_reference_promotes_without_a_matte_claim(svc):
    job_id = _reference(svc, _subject_rgb())

    result = svc_jobs.promote_to_model(svc, job_id, force=True)
    job = svc.require_job(result["id"])

    assert "matte" not in job["params"]


def test_a_fully_opaque_alpha_channel_is_not_a_matte(svc):
    """Having the channel is not using it -- the ``matting._OPAQUE`` rule."""
    arr = np.dstack([_subject_rgb(), np.full((96, 96), 255, np.uint8)])
    job_id = _reference(svc, arr)

    result = svc_jobs.promote_to_model(svc, job_id, force=True)

    assert "matte" not in svc.require_job(result["id"])["params"]


def test_an_uploaded_cutout_records_the_same_approval(svc):
    result = svc_jobs.create_job(svc, kind="image", image=_png(_cutout_rgba()))
    job = svc.require_job(result["id"])

    assert job["params"]["matte"] == "approved"
    assert job["params"]["bg_removal"] == "auto"


# --- the vocabulary ---------------------------------------------------------


def test_matte_is_a_config_vector_param_and_not_a_derived_one():
    assert "matte" in vectors.VECTOR_PARAMS
    assert "matte" not in DERIVED_PARAMS
    assert vectors.config_vector({"params": {"matte": "approved"}})["matte"] == "approved"
