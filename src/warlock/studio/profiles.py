"""Named 2D style profiles.

A profile is the *look* half of the 2D form -- which checkpoint, which LoRA,
and the four taxonomy selects that decide a house style (genre, art style,
setting, palette) -- saved under a name so a user who works on two kinds of
asset does not re-pick them each time they switch. What it deliberately does
not carry is the per-generation half: the prompt, the seed, its lock and the
reference count are about one submit, not about a look -- nor the taxonomy
fields that describe the *subject* rather than the style (category, material,
condition, emissive, rarity, silhouette, mood), which change per asset and
would otherwise be dragged along by every profile switch.

Stored in the same ``studio_settings.json`` the forms already live in, under
``user_profiles`` -- named that rather than ``profiles`` because the app
already has two other things by that name: ``guidance.PRESETS`` (shipped
starting points) and the 3D form's ``profile`` (a mesh optimisation tier).
"""

from __future__ import annotations

from typing import Any

KEY = "user_profiles"
ACTIVE_KEY = "active_profile"

_FIXED = ("base_model", "style_lora", "lora_weight", "negative_prompt", "platform")

# The taxonomy selects that describe a *look* rather than a subject. Named
# explicitly rather than derived from guidance.py: a new table there is far
# more likely to be another per-asset field than another style one, so it must
# be opted in here.
TAXONOMY = ("genre", "art_style", "setting", "palette")


def profile_fields() -> tuple[str, ...]:
    """Every form_2d key a profile captures."""
    return _FIXED + TAXONOMY


def capture(form: dict[str, Any]) -> dict[str, Any]:
    """The profile-relevant subset of a form."""
    return {k: form[k] for k in profile_fields() if k in form}


def apply(form: dict[str, Any], fields: dict[str, Any]) -> dict[str, Any]:
    """Overlay a profile onto a form, in place.

    Only keys the profile actually holds are touched, so a profile saved before
    a taxonomy field existed leaves the new field at whatever the form had
    rather than blanking it.
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
    profiles[name] = dict(fields)
    # A fresh dict rather than a mutation in place: Settings.set compares the
    # old value to the new one to decide whether anything is dirty, and an
    # object edited under it compares equal to itself.
    settings.set(KEY, profiles)


def delete_profile(settings: Any, name: str) -> None:
    profiles = list_profiles(settings)
    if profiles.pop(name, None) is None:
        return
    settings.set(KEY, profiles)
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
