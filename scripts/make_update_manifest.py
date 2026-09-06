"""Write the ``update-manifest.json`` a release has to publish to be offered.

Run after ``installer\\build.ps1`` (or ``scripts/rebuild.ps1``) has produced
``dist\\WarlockSetup-v<version>.exe``:

    uv run python scripts/make_update_manifest.py

It writes ``dist\\update-manifest.json``, and that file is **uploaded to the
GitHub Release beside the installer**, by the same hand that uploads the
installer today. It is deliberately not shipped *inside* the installer, unlike
``packs.json``: it describes the release a running copy is being offered, not
the build it came from, and a copy of it inside the app could only ever
describe the version already installed.

The generated shape is the whole contract with
``warlock.pipelines.update_worker``:

    {"version": "0.0.37",
     "installer": {"filename": "WarlockSetup-v0.0.37.exe",
                   "size_bytes": 846950916,
                   "sha256": "..."}}

The digest is the point of the file. Without it a user's copy would have to
trust that whatever ``browser_download_url`` returned was what the release
published -- which is the same bar ``packs.json`` sets for every wheel, for the
same reason, and a lower one would be indefensible for an executable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "update-manifest.json"
CHUNK = 1 << 20


def project_version() -> str:
    """The version ``pyproject.toml`` declares.

    Read from there rather than from ``warlock.__version__`` because the
    installer's filename comes from the same place (``installer/warlock.iss``'s
    ``AppVersion``), and the manifest has to name the file that exists.
    """
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return str(data["project"]["version"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest(installer: Path, version: str) -> dict[str, object]:
    return {
        "version": version,
        "installer": {
            "filename": installer.name,
            "size_bytes": installer.stat().st_size,
            "sha256": sha256(installer),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--installer",
        type=Path,
        default=None,
        help="the built setup executable (default: dist/WarlockSetup-v<version>.exe)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"where to write it (default: beside the installer, as {MANIFEST_NAME})",
    )
    args = parser.parse_args(argv)

    version = project_version()
    installer = args.installer or (ROOT / "dist" / f"WarlockSetup-v{version}.exe")
    if not installer.is_file():
        # Named rather than raised through: the ordinary way to reach this is
        # running the generator before the build, and "build it first" is the
        # answer.
        print(f"{installer} does not exist -- build the installer first.", file=sys.stderr)
        return 1
    out = args.out or (installer.parent / MANIFEST_NAME)
    payload = manifest(installer, version)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"{out}")
    print(f"  version   {payload['version']}")
    installer_info = payload["installer"]
    assert isinstance(installer_info, dict)
    print(f"  installer {installer_info['filename']}")
    print(f"  sha256    {installer_info['sha256']}")
    print(f"  size      {int(installer_info['size_bytes']) / float(1024**2):.0f} MB")
    print("Upload this file to the GitHub Release beside the installer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
