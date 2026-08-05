"""Saved config vectors, applied to the generate forms with one click.

Modelled on ``profiles.py``, which saves the *look* half of the 2D form under a
name. The difference is where the fields come from: a profile is what the user
picked, a vector preset is what the recorded verdicts say worked -- a whole
configuration ranked by accept rate, lifted out of ``findings.json``.

Stored in the same ``studio_settings.json`` under ``vector_presets``, beside
``user_profiles`` and for the same reason: it is user data about how they like
to work, not app state.

Applying one fills the forms and leaves everything visible and editable, which
is the contract every preset in this app has. It is a starting point, not a
mode.
"""

from __future__ import annotations

from typing import Any

KEY = "vector_presets"

MAX_NAME = 60

# Which form each key belongs to. A control is owned by exactly one pane, so a
# vector is applied by splitting it the same way -- there is no key both halves
# may write. ``platform`` is the deliberate exception the whole app carries: it
# is two fields, and a vector's single value is the *geometry* one, so it goes
# to the 3D form. ``resolution`` is derived from platform and is not applied at
# all, nor is ``stage``, which is not a setting.
FORM_3D_KEYS = ("platform", "profile", "custom_triangles", "size_m", "bg_removal",
                "reference_prep")
SKIP_KEYS = ("stage", "resolution")


def describe(vector: dict[str, Any]) -> str:
    """A vector as one readable line, skipping what is not a knob."""
    parts = [
        f"{k}={v}" for k, v in sorted(vector.items()) if k not in SKIP_KEYS
    ]
    return ", ".join(parts) or "the defaults"


def apply(state: Any, vector: dict[str, Any]) -> None:
    """Overlay a vector onto the two generate forms, in place.

    Only keys a form already has are touched, and each is coerced to the type
    the form's default has -- the ``form_from_params`` rule, and for the same
    reason: a value that arrived through JSON can come back as an int where the
    form holds a float, and the persisted-settings merge would then drop it.
    """
    for form, keys in (
        (state.form_2d, None),
        (state.form_3d, FORM_3D_KEYS),
    ):
        for key, value in vector.items():
            if key in SKIP_KEYS:
                continue
            if keys is not None and key not in keys:
                continue
            if keys is None and key in FORM_3D_KEYS:
                # Owned by the 3D form; see the note on platform above.
                continue
            if key not in form:
                continue
            form[key] = _coerce(form[key], value)


def _coerce(default: Any, value: Any) -> Any:
    try:
        if isinstance(default, bool):
            return bool(value)
        if isinstance(default, float):
            return float(value)
        if isinstance(default, int):
            return int(value)
        if isinstance(default, str):
            return str(value)
    except (TypeError, ValueError):
        return default
    return value


# --- storage ----------------------------------------------------------------


def list_presets(settings: Any) -> dict[str, dict[str, Any]]:
    stored = settings.get(KEY)
    if not isinstance(stored, dict):
        return {}
    return {k: v for k, v in stored.items() if isinstance(v, dict)}


def save_preset(settings: Any, name: str, vector: dict[str, Any]) -> bool:
    name = (name or "").strip()[:MAX_NAME]
    if not name:
        return False
    presets = list_presets(settings)
    # A fresh dict rather than a mutation in place: Settings.set compares the
    # old value to the new one to decide whether anything is dirty, and an
    # object edited under it compares equal to itself.
    presets[name] = {k: v for k, v in vector.items() if k not in SKIP_KEYS}
    settings.set(KEY, presets)
    return True


def delete_preset(settings: Any, name: str) -> None:
    presets = list_presets(settings)
    if presets.pop(name, None) is None:
        return
    settings.set(KEY, presets)
