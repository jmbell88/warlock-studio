"""What the app remembers between runs, replacing the browser's localStorage.

One JSON file in the data dir, written atomically and no more than once a
second. Atomically because a half-written settings file is a corrupt one that
takes the next launch down with it; debounced because the alternative is a disk
write per keystroke.

The seed is deliberately *not* persisted: it is the one field where remembering
last session's value silently reproduces last session's output, which reads as
"generate is broken" rather than as "the seed was pinned".
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

FILENAME = "studio_settings.json"
DEBOUNCE = 1.0
VERSION = 1

# Fields that must never survive a restart, whatever the form dict holds.
#
# ref_path for a different reason than the seeds: a remembered path to a file
# that has since moved or been deleted would silently condition next week's
# generation on nothing, and the failure is invisible -- the image simply
# comes out unconditioned.
VOLATILE = ("seed", "mesh_seed", "ref_path")


def _migrate(data: dict[str, Any]) -> dict[str, Any]:
    """Rewrite keys a previous version of the app wrote, in place.

    Deliberately *not* a VERSION bump: ``load`` discards the whole file on a
    version mismatch, so bumping to rename one key would wipe every setting the
    user has. Migrations here run under version 1 and must be idempotent, and
    must not mark the settings dirty -- a launch that changes nothing else
    should not rewrite the file.
    """
    # The stored "mode" is deliberately not migrated: nothing reads it any
    # more (the app opens on Home every launch), so rewriting it would only
    # move a dead key from one spelling to another. Old files keep it; it is
    # ignored either way.
    if "paint" in data and "inker" not in data:
        data["inker"] = data.pop("paint")
    return data


class Settings:
    """A small persistent dict, saved lazily."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = {}
        self._dirty = False
        self._next_save = 0.0
        # What went wrong with this file, for the app to say out loud once.
        #
        # Both halves used to be invisible. A corrupt settings file was reset
        # to defaults with only a log line, so a user whose theme, UI scale,
        # pane layout and remembered form fields had all reverted had nothing
        # on screen telling them why -- and, worse, the first successful save
        # then overwrote the file they might have wanted back. A *failed* save
        # was ignored entirely, so a read-only or full data directory meant
        # every preference silently stopped persisting for the whole session
        # (UX-10).
        self.notice: str | None = None
        self._save_failed = False

    @classmethod
    def load(cls, data_dir: Path) -> Settings:
        out = cls(Path(data_dir) / FILENAME)
        try:
            raw = json.loads(out.path.read_text("utf-8"))
        except FileNotFoundError:
            return out
        except (OSError, ValueError):
            # A corrupt file is not worth failing startup over, and the
            # defaults are all recoverable by using the app for a minute. But
            # it is worth *saying*, and worth not destroying: the original is
            # renamed aside rather than overwritten by the first save, so a
            # user who wants their layout back has something to go to.
            log.warning("ignoring an unreadable %s", out.path)
            kept = out._preserve_corrupt()
            out.notice = (
                f"Your Studio preferences could not be read and have been reset "
                f"to defaults. The old file was kept as {kept.name}."
                if kept is not None
                else "Your Studio preferences could not be read and have been "
                "reset to defaults."
            )
            return out
        if isinstance(raw, dict) and raw.get("version") == VERSION:
            # Type-checked, not just present: a "data" that is a list or null
            # passes the version gate and then takes the first get() down --
            # before run()'s try/finally exists, so nothing tears down either.
            data = raw.get("data")
            out.data = _migrate(data) if isinstance(data, dict) else {}
        return out

    # -- access ------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        if self.data.get(key) != value:
            self.data[key] = value
            self._dirty = True

    def update(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            self.set(key, value)

    # -- persistence -------------------------------------------------------

    def tick(self) -> bool:
        """Save if something changed and the debounce has elapsed."""
        if not self._dirty:
            return False
        now = time.monotonic()
        if now < self._next_save:
            return False
        self._next_save = now + DEBOUNCE
        return self.flush()

    def flush(self) -> bool:
        """Write now. Called on exit, where a debounce would lose the last edit."""
        if not self._dirty:
            return False
        payload = json.dumps({"version": VERSION, "data": self.data}, indent=2)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, raw = tempfile.mkstemp(dir=self.path.parent, prefix=".settings.", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(raw, self.path)
        except OSError as exc:
            log.exception("could not save settings")
            # Latched separately from ``notice``, which ``take_notice``
            # clears: a read-only data directory fails on every debounced tick,
            # and guarding on the notice alone would re-raise it the moment the
            # last toast was read -- one toast per second, forever.
            if not self._save_failed:
                self._save_failed = True
                self.notice = (
                    f"Studio preferences cannot be saved ({exc.strerror or exc}). "
                    f"Changes will be lost when Warlock closes."
                )
            return False
        # Recovered: a later failure is a new problem and is worth saying again.
        self._save_failed = False
        self._dirty = False
        return True

    def take_notice(self) -> str | None:
        """The pending problem with this file, cleared by reading it.

        Cleared on read so the frame loop can poll it and raise exactly one
        toast: this is a *notice*, not a status.
        """
        notice, self.notice = self.notice, None
        return notice

    def _preserve_corrupt(self) -> Path | None:
        """Rename an unreadable settings file aside. -> where it went, or None.

        Timestamped rather than a single ``.bad`` name, so a second corruption
        does not overwrite the evidence from the first.
        """
        stamp = time.strftime("%Y%m%d-%H%M%S")
        kept = self.path.with_name(f"{self.path.stem}.corrupt-{stamp}.json")
        try:
            os.replace(self.path, kept)
        except OSError:
            log.warning("could not preserve %s", self.path, exc_info=True)
            return None
        return kept


def sanitise_form(form: dict[str, Any]) -> dict[str, Any]:
    """A form dict with the fields that must not persist removed."""
    return {k: v for k, v in form.items() if k not in VOLATILE}


def restore_form(defaults: dict[str, Any], stored: Any) -> dict[str, Any]:
    """Merge a stored form over its defaults, keeping only known keys.

    Unknown keys are dropped rather than carried: they are either a field that
    was removed or a file someone edited by hand, and neither should be able to
    put a value the UI has no control for into a submit.
    """
    out = dict(defaults)
    if isinstance(stored, dict):
        for key, value in stored.items():
            if key in out and key not in VOLATILE and type(value) is type(out[key]):
                out[key] = value
    return out
