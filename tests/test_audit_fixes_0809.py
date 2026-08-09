"""Regressions for the smaller fixes of the 2026-08-09 audit.

Each of these is a promise that was stated somewhere and not kept anywhere: a
retarget that refuses the tier it cannot run, a pixel-sheet estimate that prices
the checkpoint it will actually load, an error log that survives a non-ASCII
traceback, a TCP table that is read again when it grows, and one vocabulary of
download kinds rather than three hand-copies of it.
"""

from __future__ import annotations

import sys

import pytest

from warlock import doctor, errors, fetch, vram
from warlock.service import jobs as svc_jobs
from warlock.service.errors import Invalid

# --- #2: the retarget shares _resolve_profile's refusals ---------------------


@pytest.fixture
def done_job_with_source(svc):
    job_id = svc_jobs.create_job(svc, kind="text", prompt="x")["id"]
    svc.store.set_status(job_id, "done")
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    # Never parsed: both refusals below happen before optimize.run is reached.
    (job_dir / "source.glb").write_bytes(b"glTF")
    return job_id


def test_a_retarget_refuses_a_tier_gltfpack_cannot_run(svc, done_job_with_source):
    """The whole point of _resolve_profile's gltfpack rule, on the door it was
    missing from: without it the raw copy shipped and params["profile"] named a
    tier that never ran -- and profile is in VECTOR_PARAMS, so the corpus
    learned it. The svc fixture pins gltfpack absent."""
    assert not svc.config.gltfpack_exe.exists()

    with pytest.raises(Invalid) as exc:
        svc_jobs.optimize_job(svc, done_job_with_source, profile="standard")

    assert "gltfpack" in str(exc.value)
    assert exc.value.field == "profile"
    # And nothing was written: the mesh the job already has is untouched.
    assert not (svc.job_dir(done_job_with_source) / "model.glb").exists()


def test_a_retargets_refusal_names_the_control_it_came_from(svc, done_job_with_source):
    with pytest.raises(Invalid) as exc:
        svc_jobs.optimize_job(svc, done_job_with_source, profile="nonsense")
    assert exc.value.field == "profile"


def test_raw_is_still_allowed_without_gltfpack(svc, done_job_with_source, monkeypatch):
    """The tier that needs nothing must stay reachable -- it is the default."""
    from warlock.pipelines import optimize, postprocess

    monkeypatch.setattr(
        optimize, "run", lambda *a, **k: {"requested": None, "achieved": None}
    )
    monkeypatch.setattr(postprocess, "normalize_glb", lambda *a, **k: {"scale": 1.0})
    assert svc_jobs.optimize_job(svc, done_job_with_source, profile="raw")


# --- #4: a pixel sheet is priced from the registry ---------------------------


def test_a_pixel_sheet_is_priced_from_its_own_checkpoint():
    """`queue._pixel_sheet` reads params["base_model"], so a flat SDXL_GIB was
    correct only for as long as the one caller kept writing "sdxl_cfg"."""
    from warlock import models

    offloaded = next(
        (key for key, spec in models.BASE_MODELS.items() if spec.vram_gib != vram.SDXL_GIB),
        None,
    )
    if offloaded is None:  # pragma: no cover -- every spec costs the same today
        pytest.skip("no base model with a distinct vram_gib to tell the two apart")

    priced = vram.estimate("pixel_sheet", "model", {"base_model": offloaded}, exclusive=True)
    expected = models.BASE_MODELS[offloaded].vram_gib + vram.CONTROLNET_GIB
    assert priced == pytest.approx(expected)


def test_an_unknown_checkpoint_still_falls_back_to_sdxl():
    """params outlive the registry; the tolerance queue._generate applies."""
    priced = vram.estimate("pixel_sheet", "model", {"base_model": "gone"}, exclusive=True)
    assert priced == pytest.approx(vram.SDXL_GIB + vram.CONTROLNET_GIB)


# --- #16: an error log survives a non-ASCII traceback ------------------------


def test_an_error_log_takes_a_non_ascii_message(tmp_path):
    """Written with the platform default, this raised UnicodeEncodeError on
    Windows -- which is not an OSError, so it escaped the suppress(OSError)
    around it and took down the handler recording why the job failed."""
    exc = RuntimeError("gltfpack ne s'est pas terminé — dossier « modèles » 中文")
    errors.write_error_log(tmp_path / "job", exc)
    written = (tmp_path / "job" / "error.log").read_text(encoding="utf-8")
    assert "modèles" in written


# --- #24: the TCP table is read again when it grows --------------------------


def test_the_tcp_table_is_re_read_when_it_grows(monkeypatch):
    """Table growth between the sizing call and the reading one used to
    degrade "who holds this port" to "nobody knows" -- silently."""
    if sys.platform != "win32":
        pytest.skip("listener_pid is a no-op off Windows")
    import ctypes
    import os
    import socket

    from warlock import winjob

    calls = {"n": 0}
    real = ctypes.windll.iphlpapi.GetExtendedTcpTable

    def growing(buf, size, order, af, table_class, reserved):
        calls["n"] += 1
        if buf is not None and calls["n"] == 2:
            # What the API does when a connection opened in between: it writes
            # the new size and refuses, rather than filling the buffer.
            size._obj.value = size._obj.value + 4096
            return winjob._ERROR_INSUFFICIENT_BUFFER
        return real(buf, size, order, af, table_class, reserved)

    monkeypatch.setattr(ctypes.windll.iphlpapi, "GetExtendedTcpTable", growing)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        found = winjob.listener_pid(sock.getsockname()[1])

    assert found == os.getpid()
    assert calls["n"] >= 3


# --- #6/#53: one vocabulary of download kinds --------------------------------


def test_every_download_row_has_a_group_and_a_doctor_prefix():
    kinds = {kind.key for kind in fetch.KINDS}
    assert {entry.kind for entry in fetch.entries()} == kinds
    assert set(fetch.CHECK_PREFIXES) == kinds
    assert {key for key, _label in fetch.GROUPS} == kinds


def test_the_pane_groups_every_kind_the_registry_offers():
    """The comment on _GROUPS claimed a new registry kind could not silently
    append unlabelled; the loop dropped it entirely instead. Derived now."""
    pytest.importorskip("imgui_bundle")
    from warlock.studio.panes import app_settings

    assert app_settings._GROUPS == fetch.GROUPS
    assert app_settings._UNGROUPED


def test_doctor_reports_every_downloadable_row_under_its_own_check_name(svc):
    """The prefix table is only worth having if it is the one doctor uses: a
    prefix that stopped matching would mark a downloaded model missing forever."""
    names = {check.name for check in doctor.run_checks(svc.config, probe_slow=False)}
    for entry in fetch.entries():
        assert entry.check_name in names


# --- #18: the volatile rows are spliced in by name ---------------------------


def test_the_volatile_rows_sit_after_the_install_rows(svc):
    rows = [c.name for c in doctor.run_checks(svc.config, probe_slow=False)]
    assert doctor.VOLATILE_AFTER in rows
    seam = rows.index(doctor.VOLATILE_AFTER)
    volatile = [c.name for c in doctor.volatile_checks(svc.config)]
    assert rows[seam + 1 : seam + 1 + len(volatile)] == volatile
    # And the per-model rows are still below them, not above.
    assert rows.index(fetch.entries()[0].check_name) > seam
