"""What changed in each release, read off the shipped ``CHANGELOG.md``.

Pure in the way :mod:`~warlock.vram` and :mod:`~warlock.memlog` are: stdlib
only, no imports from ``service``/``queue``/``studio``, and ``[]`` rather than
an exception for every way of having nothing to say -- the file is absent, it is
unreadable, it parses to nothing. Home draws this, and a screen the app opens on
may not be the thing that fails to start.

**Nothing here is derived from git.** Every commit subject in this repository is
``Warlock vN.N.N`` and carries no detail at all, so a generated changelog would
be a list of version numbers. The file is hand-written and this module only
reads it.

The parse is deliberately forgiving in one direction and one direction only: a
line that is neither a ``##`` heading nor a ``-`` bullet is *ignored* rather
than fatal, so the file can carry a preamble, a note or a table without this
becoming a Markdown implementation. What it will not do is guess -- a bullet
before any heading belongs to no release and is dropped.

Two concessions to the file being real Markdown, and both were bugs first. An
*indented* line under a bullet continues it, because the file is hard-wrapped
at 80 columns like everything else in this repository and treating a
continuation as an ignorable line silently truncated every bullet at its first
newline. And ``**emphasis**`` is stripped, because imgui draws one weight: the
markers rendered literally, which is worse than losing the emphasis.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

log = logging.getLogger(__name__)

FILENAME = "CHANGELOG.md"

# ``## 0.0.15 — 2026-08-09``, with the date optional and either dash accepted.
# The version is the first whitespace-delimited run after the hashes, which is
# what keeps "## 0.0.15" and "## v0.0.15 - unreleased" both readable.
_HEADING = re.compile(r"^##\s+v?(?P<version>[0-9][^\s]*)\s*(?:[-–—]\s*(?P<date>.+))?$")
_BULLET = re.compile(r"^[-*]\s+(?P<text>.+)$")
_EMPHASIS = re.compile(r"\*\*(.+?)\*\*|(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")


def _plain(text: str) -> str:
    """``**bold**`` and ``*italic*`` down to their words. imgui draws one
    weight, so the markers rendered literally and looked like a parse bug."""
    return _EMPHASIS.sub(lambda m: m.group(1) or m.group(2) or "", text)


@dataclass(frozen=True)
class Release:
    version: str
    date: str = ""
    bullets: tuple[str, ...] = field(default_factory=tuple)


def changelog_path() -> Path:
    """The packaged copy if there is one, the repo root otherwise.

    Mirrors :func:`.studio.manual.loader.manual_dir` exactly, and for the same
    reason: the canonical file is at the repo root where GitHub renders it, and
    hatchling force-includes it into the wheel beside the manual.
    """
    packaged = Path(str(resources.files("warlock"))) / FILENAME
    if packaged.is_file():
        return packaged
    # Dev checkout: src/warlock/changelog.py -> the repo root is parents[2].
    return Path(__file__).resolve().parents[2] / FILENAME


def parse(text: str) -> list[Release]:
    """The releases in ``text``, in the order they appear (newest first)."""
    out: list[Release] = []
    version = ""
    date = ""
    bullets: list[str] = []

    def flush() -> None:
        if version:
            out.append(Release(version=version, date=date, bullets=tuple(bullets)))

    for raw in text.splitlines():
        line = raw.strip()
        heading = _HEADING.match(line)
        if heading is not None:
            flush()
            version = heading.group("version")
            date = (heading.group("date") or "").strip()
            bullets = []
            continue
        bullet = _BULLET.match(line)
        # A bullet before any heading belongs to no release, so it is dropped
        # rather than attached to the first one that comes along.
        if bullet is not None and version:
            bullets.append(_plain(bullet.group("text").strip()))
            continue
        # An indented continuation of the bullet above it. Indentation in the
        # *raw* line, not the stripped one: an unindented paragraph after a
        # bullet is prose the file is allowed to carry, and folding that into
        # the last bullet is how a note becomes a release note.
        if bullets and line and raw[:1] in (" ", "\t"):
            bullets[-1] = f"{bullets[-1]} {_plain(line)}"
    flush()
    return out


def entries(path: Path | None = None) -> list[Release]:
    """Every release, newest first. ``[]`` when there is nothing to read."""
    target = path or changelog_path()
    try:
        text = target.read_text(encoding="utf-8")
    except OSError:
        log.debug("no changelog at %s", target)
        return []
    try:
        return parse(text)
    except Exception:
        log.exception("could not parse %s", target)
        return []


def current(version: str, path: Path | None = None) -> Release | None:
    """The release ``version`` names, else the newest, else ``None``.

    Falling back to the newest rather than to nothing is the point: a dev
    checkout routinely runs a version the file has not been written for yet,
    and "what changed recently" is still the right answer there.
    """
    releases = entries(path)
    if not releases:
        return None
    return next((r for r in releases if r.version == version), releases[0])
