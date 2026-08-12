"""Hash verification: at the door of a publish, and on demand afterwards.

MDL-08. "Installed" is a handful of ``Path.exists()`` calls, so an empty file, a
truncated download, a copied directory or a hard-killed publication all read as
a finished model -- and the failure surfaces minutes later with a checkpoint
already resident in VRAM. ``fetch.suspect_files`` is the cheap half (sizes
only); this is the complete one.

Two halves, and they compare against different things on purpose. The worker
checks the *staged* tree against the ETags the hub itself recorded per file, so
it catches a download that arrived wrong before anything is published.
``fetch.verify_manifest`` checks an *installed* tree against the digests that
download wrote down, so it catches a file that went wrong afterwards. Neither
would find what the other does.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from warlock import fetch, hashes
from warlock.config import Config

# ``fetch_worker`` is imported *inside* the tests that need it, never at module
# scope. Importing it sets ``HF_HUB_OFFLINE=0`` in the current process -- which
# is correct for the child, whose whole job is to go online, and catastrophic
# here: it would make the pytest process online-capable for every test that
# runs after this file, and the offline invariant is the one thing the suite
# cannot check by inspection. ``tests/test_fetch.py`` learned this the same way.


def _worker():
    from warlock.pipelines import fetch_worker

    return fetch_worker

# --- the two digest shapes ----------------------------------------------------


def test_a_large_file_is_matched_by_its_sha256(tmp_path: Path):
    """Every weight file in this registry is LFS, and an LFS ETag *is* the
    content's SHA-256. This is the case that matters."""
    path = tmp_path / "w.safetensors"
    path.write_bytes(b"weights")
    assert hashes.matches(path, hashlib.sha256(b"weights").hexdigest())
    assert not hashes.matches(path, hashlib.sha256(b"other").hexdigest())


def test_a_small_file_is_matched_by_its_git_blob_id(tmp_path: Path):
    """The trap. A non-LFS ETag is the **git blob SHA-1** -- sha1 of
    ``b"blob <len>\\0" + content``, not of the content -- so a verifier
    comparing a plain SHA-1 would report every config.json in the registry as
    corrupt, which is the failure mode that makes a check worse than none: it
    teaches the user to ignore it."""
    path = tmp_path / "config.json"
    path.write_bytes(b"{}")
    blob = hashlib.sha1(b"blob 2\0{}").hexdigest()  # noqa: S324
    assert hashes.matches(path, blob)
    assert blob != hashlib.sha1(b"{}").hexdigest()  # noqa: S324
    assert not hashes.matches(path, hashlib.sha1(b"{}").hexdigest())  # noqa: S324


def test_a_digest_shape_nobody_recognises_is_not_a_failure(tmp_path: Path):
    """An unrecognised record is not evidence of corruption, and reporting it
    as such would train the reader to dismiss the whole check."""
    path = tmp_path / "f"
    path.write_bytes(b"x")
    assert hashes.matches(path, "not-a-digest")
    assert hashes.matches(path, "")


def test_a_missing_file_never_matches(tmp_path: Path):
    assert not hashes.matches(tmp_path / "gone", "0" * 64)


# --- the sidecars snapshot_download leaves behind -----------------------------


def _sidecar(root: Path, rel: str, etag: str) -> None:
    path = root / hashes.METADATA_DIR / f"{rel}.metadata"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"deadbeef\n{etag}\n1700000000\n", encoding="utf-8")


def test_the_expected_digests_come_off_the_sidecars(tmp_path: Path):
    (tmp_path / "a.safetensors").write_bytes(b"a")
    _sidecar(tmp_path, "a.safetensors", "abc")
    assert hashes.expected_digests(tmp_path) == {"a.safetensors": "abc"}


def test_a_nested_path_keeps_its_forward_slashes(tmp_path: Path):
    """The manifest records forward slashes, so both halves of the check have
    to speak one spelling of a path -- on Windows especially."""
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"b")
    _sidecar(tmp_path, "sub/b.bin", "abc")
    assert list(hashes.expected_digests(tmp_path)) == ["sub/b.bin"]


def test_a_directory_with_no_sidecars_expects_nothing(tmp_path: Path):
    assert hashes.expected_digests(tmp_path) == {}


# --- the mandatory check, before anything is published ------------------------


def test_a_staged_tree_that_matches_its_sidecars_passes(tmp_path: Path):
    body = b"the weights"
    (tmp_path / "w.safetensors").write_bytes(body)
    _sidecar(tmp_path, "w.safetensors", hashlib.sha256(body).hexdigest())
    _worker()._verify_staged(tmp_path, {"repo_id": "acme/thing"})


def test_a_truncated_file_is_refused_before_it_is_published(tmp_path: Path):
    """The whole point. A file that reaches ``dest`` wrong reads as a finished
    model forever, because every presence probe is a handful of exists() calls
    -- so the check has to be at the door and it has to be mandatory."""
    (tmp_path / "w.safetensors").write_bytes(b"half")
    _sidecar(tmp_path, "w.safetensors", hashlib.sha256(b"the whole thing").hexdigest())
    with pytest.raises(ValueError, match="do not match the digests"):
        _worker()._verify_staged(tmp_path, {"repo_id": "acme/thing"})


def test_a_file_with_no_sidecar_is_skipped_rather_than_failed(tmp_path: Path):
    """Sidecars are written per download, so a resumed fetch that found a file
    already complete has no new one. "Nothing to compare against" is not
    evidence of corruption."""
    (tmp_path / "w.safetensors").write_bytes(b"x")
    (tmp_path / "extra.bin").write_bytes(b"y")
    _sidecar(tmp_path, "w.safetensors", hashlib.sha256(b"x").hexdigest())
    _worker()._verify_staged(tmp_path, {"repo_id": "acme/thing"})


def test_a_download_with_no_sidecars_at_all_is_not_refused(tmp_path: Path):
    """A hub client that stopped writing them, or a stubbed download in a
    test: neither is a corrupt file, and refusing here would make the app
    undownloadable rather than safe."""
    (tmp_path / "w.safetensors").write_bytes(b"x")
    _worker()._verify_staged(tmp_path, {"repo_id": "acme/thing"})


def test_the_verify_runs_before_the_cache_is_deleted():
    """A source check, because the ordering is the whole design and a
    behavioural test would need a real download to break it. The digests live
    in the ``.cache`` subtree that ``fetch_one`` deletes on its way to the
    publish; verifying after that would verify nothing, silently."""
    import inspect

    source = inspect.getsource(_worker().fetch_one)
    verify = source.index("_verify_staged(")
    drop_cache = source.index('rmtree(staging / ".cache"')
    publish = source.index("_move_into(staging, dest)")
    assert verify < drop_cache < publish


# --- the on-demand check, afterwards ------------------------------------------


def _install(root: Path, name: str, files: dict[str, bytes], *, digests=True) -> Path:
    dest = root / name
    dest.mkdir(parents=True, exist_ok=True)
    for rel, body in files.items():
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        (dest / rel).write_bytes(body)
    record = {
        "repo_id": "acme/thing",
        "revision": "a" * 40,
        "files": sorted(files),
    }
    if digests:
        record["digests"] = {
            rel: {"size": len(body), "sha256": hashlib.sha256(body).hexdigest()}
            for rel, body in files.items()
        }
    (dest / fetch.MANIFEST_NAME).write_text(
        json.dumps({"repos": {"acme/thing": record}}), encoding="utf-8"
    )
    return dest


def test_an_intact_install_verifies(tmp_path: Path):
    dest = _install(tmp_path, "thing", {"w.safetensors": b"weights"})
    result = fetch.verify_manifest(Config(), dest)
    assert result.status == fetch.VERIFY_OK
    assert result.checked == 1
    assert "1 file(s) match" in result.detail


def test_a_file_changed_after_the_download_is_reported(tmp_path: Path):
    dest = _install(tmp_path, "thing", {"w.safetensors": b"weights"})
    (dest / "w.safetensors").write_bytes(b"something else")
    result = fetch.verify_manifest(Config(), dest)
    assert result.status == fetch.VERIFY_BAD
    assert result.bad == ("w.safetensors",)
    assert "corrupt" in result.detail


def test_a_file_deleted_after_the_download_is_reported_separately(tmp_path: Path):
    """Missing and corrupt are different facts: one is a file somebody removed
    and the other is a file that changed under the app."""
    dest = _install(tmp_path, "thing", {"a.bin": b"a", "b.bin": b"b"})
    (dest / "a.bin").unlink()
    result = fetch.verify_manifest(Config(), dest)
    assert result.status == fetch.VERIFY_BAD
    assert result.missing == ("a.bin",)
    assert result.bad == ()


def test_a_manifest_with_no_digests_is_unknown_and_never_red(tmp_path: Path):
    """The back-compatibility rule. Every directory installed before this --
    and every one downloaded by hand with the pasted command -- has a manifest
    with no digests in it, and reporting those as a problem would light up the
    whole models directory and teach the reader that the check is noise."""
    dest = _install(tmp_path, "thing", {"w.safetensors": b"weights"}, digests=False)
    result = fetch.verify_manifest(Config(), dest)
    assert result.status == fetch.VERIFY_UNKNOWN
    assert result.bad == () and result.missing == ()
    assert "no digests recorded" in result.detail


def test_a_directory_with_no_manifest_at_all_is_unknown(tmp_path: Path):
    dest = tmp_path / "bare"
    dest.mkdir()
    assert fetch.verify_manifest(Config(), dest).status == fetch.VERIFY_UNKNOWN


def test_verify_all_walks_the_manifests_rather_than_the_registry(tmp_path, monkeypatch):
    """The question is "is what is installed intact", and a directory whose
    registry row has since been renamed is still a directory on this disk."""
    monkeypatch.setenv("WARLOCK_T2I_ROOT", str(tmp_path))
    _install(tmp_path, "thing", {"a.bin": b"a"})
    _install(tmp_path, "other", {"b.bin": b"b"})
    (tmp_path / "no-manifest").mkdir()
    config = Config()
    results = fetch.verify_all(config)
    assert {r.dest.name for r in results} == {"thing", "other"}
    assert all(r.status == fetch.VERIFY_OK for r in results)


# --- and it stays out of the hot path -----------------------------------------


def test_present_does_not_hash_anything():
    """``present`` answers "are the files here", is called from the frame loop
    and from the admission doors, and making it hash would turn a form's every
    frame into a disk read of a checkpoint. It is also what keeps the fixtures
    working: conftest writes one-byte placeholders, which are present and would
    never verify."""
    import inspect

    source = inspect.getsource(fetch.present)
    for forbidden in ("sha256", "verify_manifest", "hashes."):
        assert forbidden not in source


def test_nothing_verifies_on_startup():
    """The C29/C30 lesson, as a scan. A check that runs on every launch is a
    check that has to be fast, and hashing 16 GB of weights is neither fast nor
    something anybody asked for at the moment they opened the app."""
    import inspect

    from warlock.studio import runtime

    source = inspect.getsource(runtime)
    assert "verify_manifest" not in source
    assert "verify_all" not in source
