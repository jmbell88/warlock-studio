"""The "this feature needs models you haven't got" line, and its button.

Two features -- sprite synthesis and the pixel-sheet restyle -- are locked
behind weights that are not part of the core download. Before this, the only
way to find that out was to press the button and read the refusal, which named
one missing thing at a time and offered a terminal command.

The service layer owns *what* each feature needs
(``sprites.SPRITE_ROWS``/``sheets.PIXEL_SHEET_ROWS``, derived from the same
constants the refusals check). This module owns only the drawing and the
navigation, and it answers "is it here" from ``ctx.model_rows`` -- the presence
snapshot the app already refreshes when a download finishes -- rather than
touching the disk, because this runs on the frame thread sixty times a second.

The pane stays honest about being a pane: the refusal at the door is still the
authority. This is a courtesy ahead of it, and a stale ``model_rows`` costs a
wrong-looking button rather than a wrong outcome.
"""

from __future__ import annotations

from typing import Any

from .. import controls, widgets
from ..state import set_mode


def missing(ctx: Any, row_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    """Which of these rows ``ctx.model_rows`` says are absent, in its order.

    A row the snapshot has never heard of is skipped rather than reported
    missing: an empty ``model_rows`` (a headless ctx, or the first frame before
    the answers land) must read as "nothing to say", not as "everything is
    missing" -- which would lock both features on a fully-installed host.
    """
    by_key = {str(row.get("row_key")): row for row in (ctx.model_rows or [])}
    return [
        by_key[key]
        for key in row_keys
        if key in by_key and not by_key[key].get("present")
    ]


def request_install(ctx: Any, row_keys: tuple[str, ...]) -> None:
    """Tick exactly these rows in app-Settings and go there.

    A union rather than a replacement: the user may already have ticked
    something else, and silently dropping it would be a worse surprise than an
    extra row. ``set_mode`` rather than a direct ``state.mode`` write -- the
    mode-switch contract (H14), which ``tests/test_mode_writes.py`` scans this
    whole tree for.
    """
    from . import app_settings

    ctx.model_picks |= set(row_keys)
    ctx.state.preview[app_settings.CATEGORY_SLOT] = "models"
    set_mode(ctx.state, "settings")


def draw(ctx: Any, row_keys: tuple[str, ...], *, what: str = "This") -> bool:
    """Draw the gate if anything is missing. True means the feature is locked.

    The size is the *sum of the missing rows' own figures*, which over-counts
    when two of them share a checkpoint -- deliberately, because the honest
    deduped figure needs ``downloads.plan_for`` and this may not call the
    service on the frame thread. The Settings pane, which can, shows the real
    total beside its Download button before anything is fetched.
    """
    rows = missing(ctx, row_keys)
    if not rows:
        return False
    total = sum(float(row.get("size_gib") or 0.0) for row in rows)
    noun = "model" if len(rows) == 1 else "models"
    widgets.muted_wrapped(
        f"{what} needs {len(rows)} {noun} you haven't downloaded "
        f"(about {total:.1f} GB)."
    )
    if controls.small_button("Install in Settings"):
        request_install(ctx, row_keys)
    return True


def install_offer(ctx: Any, field: str) -> bool:
    """The Install button under a refusal's ring. -> whether it drew.

    ``draw`` above is the *pre-emptive* gate, shown before anything is
    submitted. This is the other half: the refusal that actually came back,
    which carries ``ServiceError.rows`` -- exact registry rows, so the button
    pre-ticks them rather than sending the user to a list of twenty-four.

    The figure here is deduped (``downloads.needed_gib``, computed once when
    the refusal arrived), so unlike ``draw``'s sum it does not count a shared
    checkpoint twice.
    """
    rows = (getattr(ctx.state, "field_error_rows", None) or {}).get(field) or ()
    if not rows:
        return False
    gib = float((getattr(ctx.state, "field_error_gib", None) or {}).get(field) or 0.0)
    noun = "model" if len(rows) == 1 else "models"
    label = f"Install {len(rows)} {noun}"
    if gib > 0.0:
        label += f" (~{gib:.1f} GB)"
    if controls.small_button(label):
        request_install(ctx, tuple(rows))
    return True
