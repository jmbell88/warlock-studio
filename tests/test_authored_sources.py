"""The authored documents Plotter and Packwright keep beside an exported asset.

The ``paint.ora`` / ``build.wblk`` precedent, applied twice more. What matters
is not that the bytes land -- it is that they land *invisibly*: absent from the
served file list, never downloadable, and gone with the job directory for free.

The other half is the marker on the row. A reopen has no fallback (unlike *Edit
in Clay*, which imports ``model.glb`` when there is no ``.wblk``), so the
library has to know from the cached row alone whether the source is there --
which is what ``params["authored"]`` is for, and why it is an input rather than
a derived value.
"""

from __future__ import annotations

import io
import zipfile

import pytest
from PIL import Image

from warlock.service import files as svc_files
from warlock.service import jobs as svc_jobs
from warlock.service.errors import Invalid, NotFound, TooLarge
from warlock.service.validation import DERIVED_PARAMS


def _png(size=(8, 8), colour=(200, 30, 30, 255)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGBA", size, colour).save(buf, "PNG")
    return buf.getvalue()


def _zip(name: str = "map.json") -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr(name, "{}")
    return out.getvalue()


def _exported(svc, mode: str) -> str:
    return svc_jobs.import_reference(svc, _png(), name="Level 1", authored=mode)["id"]


# --- the marker ---------------------------------------------------------------


def test_an_export_names_the_mode_that_can_reopen_it(svc):
    for mode in ("plotter", "packwright"):
        job = svc.store.get(_exported(svc, mode))
        assert job["params"]["authored"] == mode
        assert job["status"] == "done" and job["stage"] == "reference"


def test_an_ordinary_import_carries_no_marker(svc):
    """Absent rather than empty: the library asks ``== "plotter"``, and an
    empty string that happened to compare equal to nothing is a different kind
    of answer from a key that is not there."""
    job = svc.store.get(svc_jobs.import_reference(svc, _png())["id"])
    assert "authored" not in job["params"]


def test_the_marker_is_an_input_and_survives_a_promotion(svc):
    """It describes where the *picture* came from, which stays true of a mesh
    reconstructed from it -- so it is not a derived value and must not be in
    the strip list."""
    assert "authored" not in DERIVED_PARAMS
    job_id = _exported(svc, "plotter")
    out = svc_jobs.promote_to_model(svc, job_id, force=True)
    assert svc.store.get(out["id"])["params"]["authored"] == "plotter"


# --- the files ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("save", "path", "name"),
    [
        (svc_files.save_plotter_source, svc_files.plotter_source_path, "map.wmap"),
        (svc_files.save_packwright_source, svc_files.packwright_source_path, "pack.wpack"),
    ],
)
def test_a_source_lands_beside_the_asset_under_its_own_name(svc, save, path, name):
    job_id = _exported(svc, "plotter")
    save(svc, job_id, _zip())
    assert path(svc, job_id).name == name
    assert path(svc, job_id).read_bytes() == _zip()


@pytest.mark.parametrize(
    "name", [svc_files.PLOTTER_SOURCE, svc_files.PACKWRIGHT_SOURCE]
)
def test_a_source_is_never_served_and_never_listed(name):
    """The whole of what makes these invisible: a name outside both tables is a
    file the API has no route to and the library has no row for."""
    assert name not in svc_files.MEDIA
    assert name not in svc_files.LISTED


def test_a_source_does_not_appear_in_a_job_s_file_list(svc):
    job_id = _exported(svc, "plotter")
    svc_files.save_plotter_source(svc, job_id, _zip())
    job = dict(svc.store.get(job_id))
    svc_files.attach_files(job, svc.job_dir(job_id))
    assert svc_files.PLOTTER_SOURCE not in job["files"]
    assert "input.png" in job["files"]


def test_a_source_is_written_through_a_temporary(svc):
    """A torn file is a document that will not open, and the reader reads it
    whole."""
    job_id = _exported(svc, "packwright")
    svc_files.save_packwright_source(svc, job_id, _zip())
    names = {p.name for p in svc.job_dir(job_id).iterdir()}
    assert not [n for n in names if n.endswith(".tmp")]


def test_a_failed_write_leaves_no_staging_file(svc, monkeypatch):
    """The replace is what makes the write atomic; the ``finally`` is what keeps
    a failed one from abandoning a fragment in the job directory, where nothing
    else would ever remove it."""
    job_id = _exported(svc, "plotter")

    def boom(*_args, **_kwargs):
        raise OSError("no room")

    monkeypatch.setattr(svc_files.os, "replace", boom)
    with pytest.raises(OSError):
        svc_files.save_plotter_source(svc, job_id, _zip())
    names = {p.name for p in svc.job_dir(job_id).iterdir()}
    assert not [n for n in names if n.endswith(".tmp")]


def test_a_source_goes_away_with_the_job(svc):
    job_id = _exported(svc, "plotter")
    svc_files.save_plotter_source(svc, job_id, _zip())
    directory = svc.job_dir(job_id)
    svc_jobs.delete_job(svc, job_id)
    assert not directory.exists()


# --- refusals -----------------------------------------------------------------


def test_something_that_is_not_an_archive_is_refused(svc):
    job_id = _exported(svc, "plotter")
    with pytest.raises(Invalid):
        svc_files.save_plotter_source(svc, job_id, b"not a zip at all")


def test_an_oversized_source_is_refused(svc):
    job_id = _exported(svc, "packwright")
    huge = _zip() + b"\0" * svc_files.MAX_PACK_SOURCE_BYTES
    with pytest.raises(TooLarge):
        svc_files.save_packwright_source(svc, job_id, huge)


def test_a_bad_job_id_never_reaches_the_filesystem(svc):
    with pytest.raises(NotFound):
        svc_files.save_plotter_source(svc, "../../etc", _zip())
    with pytest.raises(NotFound):
        svc_files.save_packwright_source(svc, "0" * 12, _zip())


def test_a_saved_source_lands_where_the_path_helper_says_it_will(svc):
    """What the ``*_source_status`` pair used to assert. They were the only
    callers of themselves -- nothing in the app ever asked -- so the question
    is put to the path helpers that the readers actually use."""
    job_id = _exported(svc, "plotter")
    assert not svc_files.plotter_source_path(svc, job_id).exists()
    svc_files.save_plotter_source(svc, job_id, _zip())
    assert svc_files.plotter_source_path(svc, job_id).exists()
    # And the two documents are separate files, not one flag between them.
    assert not svc_files.packwright_source_path(svc, job_id).exists()


def test_job_dir_file_validates_the_id_it_is_given(svc):
    """The half that can come from outside is checked; the *name* is a fixed
    string every caller chooses, so it is not."""
    job_id = _exported(svc, "plotter")
    assert svc_files.job_dir_file(svc, job_id, "input.png").exists()
    with pytest.raises(NotFound):
        svc_files.job_dir_file(svc, "../../etc", "input.png")
