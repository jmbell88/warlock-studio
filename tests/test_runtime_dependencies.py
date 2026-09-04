from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_fetch_worker_dependency_is_direct() -> None:
    """Model fetching must not rely on an optional ML extra pulling Hub in."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    names = {
        entry.split("[", 1)[0].split(";", 1)[0].split(">", 1)[0].lower()
        for entry in project["dependencies"]
    }
    assert "huggingface-hub" in names


# The six places that decide what a machine actually gets. Muse shipped in the
# UI for a release cycle while ``music`` was in none of them: the extra landed
# with the ACE-Step subprocess, after these lists were written, and because
# every one of them is flat text nothing noticed. So the assertion below is
# derived from ``[project.optional-dependencies]`` rather than from a hand-kept
# copy of it -- a fifth extra fails here until it is wired or exempted, which is
# the only version of this check that would have caught the fourth.
INSTALL_SITES = (
    "installer/build.ps1",
    ".github/workflows/windows-ci.yml",
    ".github/workflows/security.yml",
    "README.md",
    "CONTRIBUTING.md",
    "CLAUDE.md",
)


def test_every_extra_is_installed_everywhere_that_installs() -> None:
    """No declared extra may be missing from a site that installs extras."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = set(project["optional-dependencies"])
    assert extras, "the check is vacuous if nothing is declared"

    missing: dict[str, set[str]] = {}
    for site in INSTALL_SITES:
        text = (ROOT / site).read_text(encoding="utf-8")
        absent = {name for name in extras if f"--extra {name}" not in text}
        if absent:
            missing[site] = absent

    assert not missing, (
        "these install sites omit a declared extra, so a machine provisioned "
        f"from them cannot run the mode that extra exists for: {missing}"
    )
