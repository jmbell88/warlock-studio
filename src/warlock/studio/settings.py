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
import math
import os
import time
from pathlib import Path
from typing import Any

from .. import generation
from . import atomic

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


def as_dict(value: Any, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """*value* if it is a mapping, else *default* (or ``{}``).

    The free function behind :meth:`Settings.get_dict`, exported because the
    same question is asked *inside* a stored blob as often as at the top of
    one: ``(raw.get("columns") or {}).items()`` reads as a type guard and is
    not one, and a stored ``"columns": "left"`` is truthy all the way to the
    ``AttributeError``. ``layouts.Arrangement`` alone asks it four times.
    """
    return value if isinstance(value, dict) else dict(default or {})


def as_list(value: Any, default: list[Any] | None = None) -> list[Any]:
    """*value* if it is a list, else *default* (or ``[]``).

    :func:`as_dict`'s twin, and the sharper of the two: ``or []`` on a stored
    string yields the string, which then *iterates* -- one element per
    character -- so the failure is not an exception but a hidden list of
    single letters that every downstream ``str()`` waves through.
    """
    return list(value) if isinstance(value, list) else list(default or [])


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
    form = data.get("form_2d")
    if isinstance(form, dict):
        # asset_type is authoritative from this release forward.  Older files
        # expressed it as three coupled switches; translate that combination
        # once, retaining every unrelated field verbatim.
        from . import create_assets

        if "asset_type" not in form:
            form["asset_type"] = create_assets.legacy_asset_type(form)
        elif form.get("asset_type") not in create_assets.ASSET_TYPES:
            # Once the new identity exists, the old switches are no longer an
            # alternate source of truth. A corrupt value falls back safely;
            # it must not resurrect a contradictory legacy selection.
            form["asset_type"] = create_assets.DEFAULT_ASSET_TYPE
        form["generation_type"] = create_assets.asset_type_from_params(form)
        if form.get("projection") == "orthogonal":
            form["projection"] = "top_down"
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
        """Read the file, or start from defaults and say why.

        Three ways this file can fail, and until this pass only the first had
        an answer a user could see.
        """
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
            out._reset("could not be read")
            return out
        if not isinstance(raw, dict) or raw.get("version") != VERSION:
            # A version mismatch used to discard every preference in silence:
            # no notice, no rename, and the first successful save overwrote the
            # file. That is the *same* loss as the unreadable case above -- the
            # theme, the UI scale, the pane layout and every remembered form
            # field all revert -- so it gets the same treatment. Renaming aside
            # matters more here, not less: the likeliest way to reach this
            # branch is running an older build against a newer file, where the
            # data is not corrupt at all and is the only copy there is.
            found = raw.get("version") if isinstance(raw, dict) else type(raw).__name__
            log.warning("ignoring %s: version %r", out.path, found)
            out._reset("were written by a different version of Warlock")
            return out
        # Type-checked, not just present: a "data" that is a list or null
        # passes the version gate and then takes the first get() down --
        # before run()'s try/finally exists, so nothing tears down either.
        data = raw.get("data")
        if not isinstance(data, dict):
            out._reset("could not be read")
            return out
        try:
            out.data = _migrate(data)
        except Exception:
            # ``_migrate`` ran *outside* this try, and it reads stored values
            # as though they were the shapes it wrote: ``settings.py``'s own
            # legacy-asset-type branch asks ``value in ASSET_TYPES``, which on
            # a stored ``asset_type`` of ``{}`` is ``TypeError: unhashable
            # type``. That is an exception out of ``Settings.load``, which runs
            # before there is a window, a GL context or an excepthook that can
            # draw anything -- a permanent boot loop from one hand-edited byte.
            log.exception("could not migrate %s", out.path)
            out._reset("could not be read")
        return out

    def _reset(self, why: str) -> None:
        """Keep the old file aside, start from defaults, and say so once."""
        kept = self._preserve_corrupt()
        self.data = {}
        self.notice = (
            f"Your Studio preferences {why} and have been reset to defaults. "
            f"The old file was kept as {kept.name}."
            if kept is not None
            else f"Your Studio preferences {why} and have been reset to defaults."
        )

    # -- access ------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """The raw stored value. **Not a type guard.**

        This returns whatever JSON held, which is the right primitive and the
        wrong default habit: ``settings.get("panes") or {}`` reads as a type
        guard and is not one, because a stored ``"panes": "left"`` is truthy
        and then raises ``AttributeError`` on the ``.get`` after it. Seven
        sites read that way, one of them inside ``widgets.section``, which most
        panes draw every frame -- so one wrong byte in the file was a crash on
        the first frame of the mode that read it, every launch.

        Wrap it in :func:`as_dict` or :func:`as_list` instead. Free functions
        rather than ``get_dict``/``get_list`` methods, deliberately: the same
        question is asked *inside* a stored blob as often as at the top of one
        (``layouts.Arrangement`` asks it four times about values that never
        came from a ``Settings`` at all), and one spelling that works on any
        value beats two that differ only in where the value came from. A scan
        test refuses the ``or {}`` form.
        """
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
            # ``atomic.staged`` rather than a hand-rolled mkstemp + replace:
            # the temporary is unlinked in its ``finally``, where the old
            # version left one behind on every failed replace -- and with
            # ``_dirty`` still set, ``tick`` retried once a second, one
            # orphaned ``.settings.*.json`` per second for as long as the
            # directory stayed unwritable.
            with atomic.staged(self.path) as tmp:
                tmp.write_text(payload, encoding="utf-8")
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
        values = dict(stored)
        if "asset_type" in out:
            from . import create_assets

            if "asset_type" not in values:
                values["asset_type"] = create_assets.legacy_asset_type(values)
            elif values.get("asset_type") not in create_assets.ASSET_TYPES:
                values["asset_type"] = create_assets.DEFAULT_ASSET_TYPE
            if values.get("projection") == "orthogonal":
                values["projection"] = "top_down"
            if "generation_type" not in values:
                values["generation_type"] = create_assets.legacy_asset_type(values)
        for key, value in values.items():
            if (
                key in out
                and key not in VOLATILE
                and type(value) is type(out[key])
                and _safe_form_value(key, value)
            ):
                out[key] = value
    # Keep the service-facing compatibility fields consistent even when a
    # hand-edited settings file supplied a contradictory combination.
    if "asset_type" in out:
        from . import create_assets

        if out.get("generation_type") not in create_assets.ASSET_TYPES:
            out["generation_type"] = create_assets.legacy_asset_type(out)
        create_assets.sync_legacy_fields(out)
    return out


def _safe_form_value(key: str, value: Any) -> bool:
    """Whether a same-typed persisted form value is safe to restore.

    This is intentionally a boundary check, not submit validation.  Settings
    are untrusted JSON and are read before any controls can clamp values; NaN,
    an obsolete enum or a billion candidates must not make the first Create
    frame or its cost summary raise.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return False
    choices: dict[str, set[str]] = {
        "output": {"reference", "tile", "sheet"},
        "sheet_type": {"tile", "sprite"},
        "projection": {"top_down", "three_quarter", "isometric", "orthogonal"},
        "sheet_layout": {"turnaround", "walk"},
        "tile_size": {"16", "32", "48", "64"},
        "cell_size": {"32", "48", "64"},
        "expand": {"off", "asset", "scene"},
        "generation_type": {"image", "model_3d", "seamless_material", "tileset", "sprite_sheet"},
        "quality": {"fast", "quality"},
        "model_mode": {"auto", "advanced"},
        "target_cell_px": {"", "8", "16", "24", "32", "48", "64", "96", "128", "256"},
    }
    if key == "asset_type":
        from .create_assets import ASSET_TYPES

        return value in ASSET_TYPES
    if key in choices:
        if key == "target_cell_px":
            if value == "":
                return True
            try:
                return generation.TARGET_CELL_MIN <= int(value) <= generation.TARGET_CELL_MAX
            except (TypeError, ValueError):
                return False
        return value in choices[key]
    if key == "base_model":
        from ..models import BASE_MODELS

        return value in BASE_MODELS
    if key == "style_lora":
        from ..models import STYLE_LORAS

        return value == "" or value in STYLE_LORAS
    if key == "count":
        from ..service.validation import MAX_REFERENCE_COUNT

        return 1 <= value <= MAX_REFERENCE_COUNT
    if key == "lora_weight":
        from ..models import LORA_WEIGHT_MAX, LORA_WEIGHT_MIN

        return LORA_WEIGHT_MIN <= value <= LORA_WEIGHT_MAX
    if key in {"ip_scale", "control_scale", "control_end"}:
        return 0.0 <= value <= 1.5
    return True
