"""Chapter discovery: the packaged copy if installed, the repo docs otherwise.

The canonical files live at docs/manual/ -- readable on GitHub -- and hatchling
force-includes them into the wheel as warlock/manual, the same
single-source-two-locations bargain the skeleton templates make. Pure: no
imgui anywhere, so tests run headless.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

PARTS: tuple[tuple[str, range], ...] = (
    ("Using Warlock Studio", range(1, 9)),
    ("Setup & operations", range(9, 13)),
    ("Architecture", range(13, 17)),
)

_H1 = re.compile(r"^# +(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class Chapter:
    key: str  # filename stem: "01-overview"
    number: int
    title: str
    part: str


def manual_dir() -> Path:
    packaged = Path(str(resources.files("warlock"))) / "manual"
    if packaged.is_dir():
        return packaged
    # Dev checkout: src/warlock/studio/manual/loader.py -> repo root is
    # parents[4], and the canonical files are docs/manual there.
    return Path(__file__).resolve().parents[4] / "docs" / "manual"


def chapters(root: Path | None = None) -> list[Chapter]:
    base = root or manual_dir()
    found: list[Chapter] = []
    for path in sorted(base.glob("[0-9][0-9]-*.md")):
        number = int(path.stem[:2])
        match = _H1.search(path.read_text(encoding="utf-8"))
        title = match.group(1).strip() if match else path.stem
        part = next((label for label, rng in PARTS if number in rng), "")
        found.append(Chapter(path.stem, number, title, part))
    return found


def load(key: str, root: Path | None = None) -> str:
    base = root or manual_dir()
    if not any(c.key == key for c in chapters(root=base)):
        raise KeyError(key)
    return (base / f"{key}.md").read_text(encoding="utf-8")
