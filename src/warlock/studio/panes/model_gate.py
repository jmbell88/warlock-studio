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
    # ``getattr`` rather than an attribute read: this is now called from the
    # rail and from Home, which draw against contexts (and test doubles) built
    # before the snapshot exists. A ctx with no ``model_rows`` at all is the
    # same "nothing to say" as an empty one -- see the docstring above --
    # rather than an AttributeError in a draw function.
    by_key = {
        str(row.get("row_key")): row for row in (getattr(ctx, "model_rows", None) or [])
    }
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


def missing_packs(ctx: Any, key: str) -> list[dict[str, Any]]:
    """The dependency packs this mode needs and does not have, in pane order.

    **The other half of a mode's door, and until F1's run it did not exist**
    (2026-09-05). ``packs.Pack.modes`` has always named which mode each pack
    gates, and it was read in exactly one place: a label inside Settings. So on
    a base install -- where the heavy extras are packs and the weights are
    downloads -- both halves were missing at once, only the weights half was
    checked, and the user was sent to download about 23 GB that still would not
    let Create generate anything, because ``torch`` was not installed.

    Read from ``ctx.pack_rows``, the snapshot the app already refreshes, for
    ``missing``'s reason: this runs on the frame thread sixty times a second
    and must not touch the disk. A ctx with no snapshot says "nothing to
    report" rather than "everything is missing", which is the same rule and the
    same failure it prevents.
    """
    return [
        row
        for row in (getattr(ctx, "pack_rows", None) or [])
        if key in (row.get("modes") or ()) and not row.get("present")
    ]


def _library_has_work(ctx: Any) -> bool:
    """Whether this machine has any finished job on it.

    The escape both gates share. Create's later stages act on jobs that already
    exist and Muse plays takes it did not have to generate, so a gate that
    fired on an empty toolchain alone would lock a user out of their own
    finished work the moment they reclaimed some disk.
    """
    cache = getattr(ctx, "cache", None)
    return cache is not None and (getattr(cache, "total", 0) or 0) > 0


def mode_block(ctx: Any, key: str) -> tuple[str, ...]:
    """Which of a mode's required rows are missing. Empty means it can open.

    The rail's grey-out and ``state.set_mode``'s refusal are both this
    function, so the picture and the behaviour cannot disagree.

    **A mode is only blocked on a machine with no work on it.** Create's later
    stages (mesh, rig, pose, export) act on jobs that already exist, and Muse
    plays and exports takes it did not have to generate -- so greying purely on
    absent weights would lock a user out of finished work after they removed a
    model to reclaim disk. ``cache.total`` is the whole library, so the gate
    fires exactly on the case it is for: a fresh install with nothing in it.
    """
    from .. import modes

    rows = modes.NEEDS_ROWS.get(key) or ()
    if not rows or not missing(ctx, rows):
        return ()
    if _library_has_work(ctx):
        return ()
    return tuple(row["row_key"] for row in missing(ctx, rows))


def mode_gate(ctx: Any, key: str) -> tuple[str, tuple[str, ...]]:
    """What stands at this mode's door: ``("packs" | "models" | "", keys)``.

    **Packs come first when both are missing**, which is the whole point of the
    ordering rather than a preference: a pack is the code and the weights are
    what the code reads, so weights installed without their pack buy the user
    nothing at all, and 23 GB is an expensive way to find that out. Sending
    them to Packs first means the smaller download is also the one that has to
    happen first.
    """
    packs = missing_packs(ctx, key)
    if packs and not _library_has_work(ctx):
        return ("packs", tuple(str(row["key"]) for row in packs))
    rows = mode_block(ctx, key)
    return ("models", rows) if rows else ("", ())


def mode_reason(ctx: Any, key: str) -> str:
    """The sentence the greyed rail item shows, or "" when it is not gated."""
    where, blocked = mode_gate(ctx, key)
    if not where:
        return ""
    if where == "packs":
        rows = missing_packs(ctx, key)
        total = sum(float(row.get("download_gib") or 0.0) for row in rows)
        named = ", ".join(str(row.get("label") or row.get("key")) for row in rows)
        size = f" (about {total:.1f} GB)" if total > 0 else ""
        # Named rather than counted, because a pack's name is what the Settings
        # row is labelled with and "1 pack" is not something to look for.
        return (
            f"Needs the {named} pack{'s' if len(rows) > 1 else ''} "
            f"installed{size}. Click to install it in Settings."
        )
    by_key = {
        str(row.get("row_key")): row for row in (getattr(ctx, "model_rows", None) or [])
    }
    total = sum(float((by_key.get(k) or {}).get("size_gib") or 0.0) for k in blocked)
    noun = "download" if len(blocked) == 1 else "downloads"
    return (
        f"Needs {len(blocked)} {noun} you haven't made yet "
        f"(about {total:.0f} GB). Click to choose them in Settings."
    )


def request_pack(ctx: Any, _keys: tuple[str, ...] = ()) -> None:
    """Open Settings at Packs. The pack half of :func:`request_install`.

    No pre-ticking to do: the Packs pane installs one row at a time by its own
    button, deliberately, because two children unpacking torch into one
    site-packages at once is the worst concurrency this app could have. So the
    navigation is the whole of it.
    """
    from . import app_settings

    ctx.state.preview[app_settings.CATEGORY_SLOT] = "packs"
    set_mode(ctx.state, "settings")


def request_for_mode(ctx: Any, key: str) -> None:
    """Send the user wherever this mode's door actually points.

    One function so the rail cannot route to Models while the tooltip names a
    pack -- the disagreement F4 was.
    """
    where, keys = mode_gate(ctx, key)
    if where == "packs":
        request_pack(ctx, keys)
    elif where == "models":
        request_install(ctx, keys)
