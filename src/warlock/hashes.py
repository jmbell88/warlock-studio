"""File digests, and the two shapes Hugging Face records them in.

Pure in the ``vram.py`` sense -- stdlib only, no imports from ``service``,
``queue`` or ``studio`` -- because three callers want it and one of them is a
subprocess that must not import the app: the fetch worker verifies a staged
download before publishing it, :mod:`warlock.fetch` verifies an installed one on
demand, and the tests exercise both.

**Two digests, because the hub uses two.** ``snapshot_download`` leaves a
``.cache/huggingface/download/<path>.metadata`` sidecar beside every file it
fetched, three lines: the commit, the ETag, and a timestamp. For an LFS file the
ETag *is* the content's SHA-256, which is the case that matters -- every weight
file in this registry is LFS. For a small non-LFS file (the ``config.json`` and
``*.txt`` this app also downloads) it is the **git blob SHA-1**, which is
``sha1(b"blob <len>\\0" + content)`` and is not a plain hash of the bytes. A
verifier that compared a plain SHA-1 against it would report every JSON file in
the registry as corrupt, which is the failure mode that makes an integrity check
worse than none: it teaches the user to ignore it.

**What the digests are for.** MDL-08. "Installed" is a handful of
``Path.exists()`` calls, so an empty file, a truncated download, a copied
directory or a hard-killed publication all read as a finished model -- and the
failure surfaces minutes later with a checkpoint already resident in VRAM.
``fetch.suspect_files`` is the cheap half (sizes only, one ``stat`` per file);
this is the complete one.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

#: How much is read at a time. Large enough that a 7 GB checkpoint is not
#: sixty thousand syscalls, small enough that nothing here holds a file in
#: memory -- these run in the fetch child, whose whole purpose is that the app
#: process never pays the RSS of a download.
CHUNK = 1 << 20

#: Where ``snapshot_download`` leaves its per-file sidecars, relative to the
#: directory it downloaded into.
METADATA_DIR = Path(".cache") / "huggingface" / "download"


def sha256(path: Path) -> str:
    """The file's SHA-256, streamed."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    """The git object id of the file's contents, streamed.

    ``sha1(b"blob <size>\\0" + content)`` -- the header is what makes this a
    git blob id rather than a hash of the bytes, and it is why a non-LFS
    sidecar's ETag cannot be compared against :func:`sha256` or against a
    plain SHA-1.
    """
    path = Path(path)
    digest = hashlib.sha1(b"blob %d\0" % path.stat().st_size)  # noqa: S324
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def read_etag(metadata: Path) -> str:
    """The ETag line of one ``.metadata`` sidecar, or "" if unreadable.

    Never raises. A sidecar that is missing, truncated or from a future format
    means "nothing to check against", which is a different answer from "this
    file is wrong" and must not be confused with it.
    """
    try:
        lines = Path(metadata).read_text(encoding="utf-8").splitlines()
    except OSError:
        return ""
    return lines[1].strip().strip('"') if len(lines) > 1 else ""


def expected_digests(root: Path) -> dict[str, str]:
    """``{relative path: expected digest}`` from the sidecars under ``root``.

    Keys use forward slashes, matching what ``_move_into`` records in the
    manifest, so the two halves of the check speak one spelling of a path.
    """
    out: dict[str, str] = {}
    base = Path(root) / METADATA_DIR
    if not base.is_dir():
        return out
    for sidecar in base.rglob("*.metadata"):
        etag = read_etag(sidecar)
        if not etag:
            continue
        rel = sidecar.relative_to(base).with_suffix("")
        out[str(rel).replace("\\", "/")] = etag
    return out


def matches(path: Path, expected: str) -> bool:
    """Whether the file's contents answer to ``expected``, whichever shape it is.

    Dispatched on the digest's *length*, not on a flag the caller passes: 64 hex
    characters is a SHA-256 (LFS), 40 is a git blob SHA-1 (non-LFS), and the
    sidecar does not say which it wrote. Anything else is a shape this function
    does not know, and it answers True rather than False -- an unrecognised
    record is not evidence of corruption, and reporting it as such would train
    the user to dismiss the check.
    """
    expected = (expected or "").strip().lower()
    path = Path(path)
    if not path.is_file():
        return False
    if len(expected) == 64:
        return sha256(path) == expected
    if len(expected) == 40:
        return git_blob_sha1(path) == expected
    return True


def digest_files(root: Path, names: list[str]) -> dict[str, dict[str, object]]:
    """``{name: {"size": int, "sha256": str}}`` for the manifest.

    SHA-256 for every file, whether or not the hub's own record for it was one:
    the manifest is *our* record and is what a later verify compares against, so
    it wants one shape rather than two, and a small non-LFS file costs nothing
    to hash. A file that has gone missing between the publish and this call is
    skipped rather than recorded as empty.
    """
    out: dict[str, dict[str, object]] = {}
    for name in names:
        path = Path(root) / name
        try:
            out[name] = {"size": path.stat().st_size, "sha256": sha256(path)}
        except OSError:
            continue
    return out
