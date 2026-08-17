"""Named 2D style profiles.

A profile is the *look* half of the 2D form -- which checkpoint, which LoRA,
its weight and the negative prompt -- saved under a name so a user who works
on two kinds of asset does not re-pick them each time they switch. What it
deliberately does not carry is the per-generation half: the prompt, the seed,
its lock and the reference count are about one submit, not about a look. The
taxonomy selects it used to carry retired with the taxonomy on 2026-08-17;
``apply`` already tolerates old saved profiles that still hold those keys.

Stored in the same ``studio_settings.json`` the forms already live in, under
``user_profiles`` -- named that rather than ``profiles`` because the app
already has another thing by that name: the 3D form's ``profile`` (a mesh
optimisation tier).
"""

from __future__ import annotations

import contextlib
import re
import secrets
from pathlib import Path
from typing import Any

from .. import models

KEY = "user_profiles"
ACTIVE_KEY = "active_profile"

# Where the anchor images live, relative to the data dir. Beside the job
# directories rather than inside one: an anchor outlives every job it was
# taken from, and prune_jobs walks that root.
ANCHOR_DIR = "profiles"

# The two keys a profile carries that are *not* form fields, so capture()
# never sees them and save_profile has to preserve them by hand.
ANCHOR_FIELDS = ("anchor", "anchor_scale")

# Which adapter an anchor is applied through. "plus" conditions on 16 patch
# tokens rather than one pooled embedding, which is the difference between
# "same kind of object" and "this look" -- exactly what an anchor is for.
ANCHOR_ADAPTER = "plus"

# Exactly what _new_anchor_name generates. A profile is a JSON blob a user can
# edit by hand, and this string becomes a path.
_ANCHOR_RE = re.compile(r"^[0-9a-f]{12}\.png$")

_FIXED = ("base_model", "style_lora", "lora_weight", "negative_prompt")


def profile_fields() -> tuple[str, ...]:
    """Every form_2d key a profile captures."""
    return _FIXED


def capture(form: dict[str, Any]) -> dict[str, Any]:
    """The profile-relevant subset of a form."""
    return {k: form[k] for k in profile_fields() if k in form}


def apply(form: dict[str, Any], fields: dict[str, Any]) -> dict[str, Any]:
    """Overlay a profile onto a form, in place.

    Only keys both the profile and the current field list hold are touched, so
    an old saved profile that still carries retired taxonomy keys applies its
    surviving fields and the rest are silently ignored.
    """
    known = set(profile_fields())
    for key, value in (fields or {}).items():
        if key in known and key in form:
            form[key] = value
    return form


# --- storage ----------------------------------------------------------------


def list_profiles(settings: Any) -> dict[str, dict[str, Any]]:
    stored = settings.get(KEY)
    if not isinstance(stored, dict):
        return {}
    return {k: v for k, v in stored.items() if isinstance(v, dict)}


def save_profile(settings: Any, name: str, fields: dict[str, Any]) -> None:
    name = (name or "").strip()
    if not name:
        return
    profiles = list_profiles(settings)
    merged = dict(fields)
    existing = profiles.get(name) or {}
    for key in ANCHOR_FIELDS:
        # Preserved rather than captured: the anchor is not a form field, so
        # capture() cannot see it and an ordinary re-save would drop it every
        # time the user changed a select. An explicit value still wins, which
        # is what makes clear_anchor a plain save.
        if key not in merged and key in existing:
            merged[key] = existing[key]
    # A fresh dict rather than a mutation in place: Settings.set compares the
    # old value to the new one to decide whether anything is dirty, and an
    # object edited under it compares equal to itself.
    profiles[name] = merged
    settings.set(KEY, profiles)


def delete_profile(settings: Any, name: str, config: Any = None) -> None:
    profiles = list_profiles(settings)
    removed = profiles.pop(name, None)
    if removed is None:
        return
    settings.set(KEY, profiles)
    if config is not None:
        # After the write, and only when nothing else points at it: the editor
        # renames by saving under the new name and deleting the old, so both
        # entries name the same file for exactly one call.
        _drop_anchor_file(settings, config, removed.get("anchor"))
    if get_active(settings) == name:
        set_active(settings, None)


def get_active(settings: Any) -> str | None:
    active = settings.get(ACTIVE_KEY)
    return active if isinstance(active, str) and active else None


def set_active(settings: Any, name: str | None) -> None:
    settings.set(ACTIVE_KEY, name or None)


def active_fields(settings: Any) -> dict[str, Any]:
    """The active profile's fields, or {} if none is set or it was deleted."""
    active = get_active(settings)
    if active is None:
        return {}
    return list_profiles(settings).get(active) or {}


# --- the style anchor -------------------------------------------------------


def anchor_dir(config: Any) -> Path:
    return Path(config.data_dir) / ANCHOR_DIR


def anchor_path(config: Any, fields: dict[str, Any] | None) -> Path | None:
    """The anchor image on disk, or None.

    None covers all three ways there is no usable anchor: the profile never
    had one, the recorded name is not one this module wrote (studio_settings
    .json is a file a user can edit, and this string becomes a path), or the
    file has since been deleted -- in which case a stale name must read as no
    anchor rather than as a missing-file crash at submit time.
    """
    name = str((fields or {}).get("anchor") or "")
    if not _ANCHOR_RE.match(name):
        return None
    path = anchor_dir(config) / name
    return path if path.exists() else None


def set_anchor(
    settings: Any, config: Any, name: str, png: bytes, scale: float | None = None
) -> None:
    """Point a profile at a new anchor image, replacing any it had."""
    profiles = list_profiles(settings)
    if name not in profiles:
        return
    previous = profiles[name].get("anchor")
    filename = f"{secrets.token_hex(6)}.png"
    directory = anchor_dir(config)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / filename).write_bytes(png)
    save_profile(
        settings,
        name,
        {
            **profiles[name],
            "anchor": filename,
            "anchor_scale": (
                models.DEFAULT_IP_SCALE if scale is None else float(scale)
            ),
        },
    )
    _drop_anchor_file(settings, config, previous)


def clear_anchor(settings: Any, config: Any, name: str) -> None:
    profiles = list_profiles(settings)
    if name not in profiles:
        return
    previous = profiles[name].get("anchor")
    save_profile(settings, name, {**profiles[name], "anchor": "", "anchor_scale": ""})
    _drop_anchor_file(settings, config, previous)


def active_anchor(settings: Any, config: Any) -> tuple[Path, float] | None:
    """-> (image, strength) for the active profile's anchor, or None."""
    fields = active_fields(settings)
    path = anchor_path(config, fields)
    if path is None:
        return None
    try:
        scale = float(fields.get("anchor_scale") or models.DEFAULT_IP_SCALE)
    except (TypeError, ValueError):
        scale = models.DEFAULT_IP_SCALE
    return (path, scale)


def _drop_anchor_file(settings: Any, config: Any, filename: Any) -> None:
    """Unlink an anchor image nothing points at any more.

    The reference count is the point. A rename saves the profile under its new
    name and deletes the old entry, so for one call two entries name the same
    file -- unlinking on sight would delete the anchor of the profile that was
    just created.
    """
    name = str(filename or "")
    if not _ANCHOR_RE.match(name):
        return
    if any(p.get("anchor") == name for p in list_profiles(settings).values()):
        return
    with contextlib.suppress(OSError):
        (anchor_dir(config) / name).unlink()
