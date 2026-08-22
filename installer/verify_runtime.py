"""Verify the installer runtime manifest without importing Warlock.

The build calls this once against the checkout and once against the staged
tree. Keeping it stdlib-only means it can run before dependencies are installed
and with the copied standalone interpreter afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST_VERSION = 1
SHA256 = re.compile(r"[0-9a-f]{64}")


class ManifestError(ValueError):
    """The manifest is malformed or the runtime differs from its pin."""


def _entry_path(root: Path, raw: object) -> Path:
    text = str(raw or "")
    relative = PurePosixPath(text)
    if not text or relative.is_absolute() or ".." in relative.parts or "\\" in text:
        raise ManifestError(f"runtime path is not a safe relative POSIX path: {text!r}")
    target = (root / Path(*relative.parts)).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        raise ManifestError(f"runtime path escapes the staged root: {text!r}") from None
    return target


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read runtime manifest {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("version") != MANIFEST_VERSION:
        raise ManifestError(f"runtime manifest must be version {MANIFEST_VERSION}")
    python = payload.get("python")
    expected_python = {
        "implementation": "CPython",
        "distribution": "python-build-standalone via uv",
        "major_minor": "3.13",
        "architecture": "x86_64",
        "torch_cuda": "12.8",
    }
    if not isinstance(python, Mapping) or any(
        python.get(key) != value for key, value in expected_python.items()
    ):
        raise ManifestError("runtime manifest must pin uv CPython 3.13 x64 and CUDA 12.8")
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ManifestError("runtime manifest names no files")
    roots = payload.get("roots")
    if not isinstance(roots, list) or not roots:
        raise ManifestError("runtime manifest names no payload roots")
    return payload


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            checksum.update(block)
    return checksum.hexdigest()


def verify_runtime(root: Path, manifest_path: Path) -> tuple[Path, ...]:
    root = root.resolve()
    payload = load_manifest(manifest_path)
    verified: list[Path] = []
    seen: set[str] = set()
    for raw in payload["files"]:
        if not isinstance(raw, Mapping):
            raise ManifestError("every runtime manifest entry must be an object")
        name = str(raw.get("path") or "")
        if name in seen:
            raise ManifestError(f"runtime manifest repeats {name!r}")
        seen.add(name)
        target = _entry_path(root, name)
        if not target.is_file():
            raise ManifestError(f"runtime file is missing: {name}")
        try:
            wanted_size = int(raw.get("size"))
        except (TypeError, ValueError):
            raise ManifestError(f"runtime file has no valid size pin: {name}") from None
        if wanted_size < 1 or target.stat().st_size != wanted_size:
            raise ManifestError(
                f"runtime file size differs: {name} "
                f"({target.stat().st_size} != {wanted_size})"
            )
        wanted_hash = str(raw.get("sha256") or "")
        if SHA256.fullmatch(wanted_hash) is None:
            raise ManifestError(f"runtime file has no valid SHA-256 pin: {name}")
        actual = digest(target)
        if actual != wanted_hash:
            raise ManifestError(f"runtime file SHA-256 differs: {name}")
        verified.append(target)
    actual_files = {
        path.relative_to(root).as_posix()
        for raw_root in payload["roots"]
        for path in _entry_path(root, raw_root).rglob("*")
        if path.is_file()
    }
    extra = sorted(actual_files - seen)
    if extra:
        raise ManifestError(f"runtime payload contains unpinned files: {extra}")
    return tuple(verified)


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=here.parents[1])
    parser.add_argument("--manifest", type=Path, default=here.with_name("runtime-manifest.json"))
    args = parser.parse_args(argv)
    try:
        files = verify_runtime(args.root, args.manifest)
    except ManifestError as exc:
        parser.exit(1, f"runtime verification failed: {exc}\n")
    print(f"verified {len(files)} pinned runtime files under {args.root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
