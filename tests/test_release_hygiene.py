"""What must be true of the repository and of what `uv build` produces.

The 2026-08-24 release audit found that `examples/` -- 20 git-tracked files
including Nintendo-derived game art and ULPC assets under CC-BY-SA/GPL -- was
swept into every source distribution, ~26 MB of a 41 MB tarball, because
hatchling's sdist default is "everything git does not ignore" and `.gitignore`
said nothing about the directory. CI runs `uv build` on every push.

Pinned the way `tests/test_ux_todo_fixes.py` pins the deleted plan filenames:
prose calling the exclusion non-negotiable is what the project already had, and
what it did not have was anything that failed when the prose stopped being true.
"""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _tracked() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if out.returncode != 0:  # pragma: no cover - not a checkout
        pytest.skip("not a git checkout")
    return out.stdout.splitlines()


def test_no_third_party_sample_art_is_tracked():
    """Making the repository public publishes whatever git is holding."""
    offenders = [p for p in _tracked() if p.startswith("examples/")]
    assert not offenders, (
        "examples/ is tracked again: these are Nintendo-derived and "
        f"ULPC CC-BY-SA/GPL files this project may not redistribute -- {offenders}"
    )


def test_the_sdist_ships_an_allowlist_rather_than_whatever_is_lying_around():
    """An allowlist, not an exclude list.

    An exclude only stops what somebody thought of, and the failure here was a
    directory nobody thought of.
    """
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    sdist = meta["tool"]["hatch"]["build"]["targets"]["sdist"]
    assert "include" in sdist, "the sdist has no allowlist; hatchling will sweep the tree"
    assert not any(p.strip("/").startswith("examples") for p in sdist["include"])
    for required in ("/LICENSE", "/THIRD-PARTY-NOTICES.md", "/src/warlock"):
        assert required in sdist["include"], required


def test_the_licence_and_notices_exist_where_every_reference_points():
    assert (ROOT / "LICENSE").is_file()
    assert (ROOT / "THIRD-PARTY-NOTICES.md").is_file()
    # Referenced from the licence header and from the installer input, both of
    # which are read by somebody deciding whether they may use this.
    assert "THIRD-PARTY-NOTICES.md" in (ROOT / "LICENSE").read_text(encoding="utf-8")


def test_the_installer_shows_the_user_the_licence():
    """Inno Setup shows no terms at all without ``LicenseFile=``, which is the
    same posture as having no licence for everyone who installs rather than
    clones."""
    iss = (ROOT / "installer" / "warlock.iss").read_text(encoding="utf-8", errors="replace")
    assert "LicenseFile=" in iss


def test_the_installer_stages_the_notices_beside_the_binaries():
    """MIT requires the notice to travel with the binary; the NVIDIA
    redistributable EULA has its own terms. Both were copied bare."""
    build = (ROOT / "installer" / "build.ps1").read_text(encoding="utf-8", errors="replace")
    assert "THIRD-PARTY-NOTICES.md" in build
    assert "LICENSE" in build
