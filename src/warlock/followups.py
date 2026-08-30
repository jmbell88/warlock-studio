"""Durable, user-facing records for automatic follow-up admission failures.

The parent artifact is already complete when an automatic rig or sheet is
queued. Failing that second insert must not fail the parent, but logging alone
also makes the requested work appear to have vanished. This module owns the
small params schema that bridges those two requirements and is shared by the
queue writer and Studio reader.
"""

from __future__ import annotations

import time
from typing import Any

PARAM_KEY = "followup_failures"
MAX_MESSAGE = 500

LABELS = {
    "rig": "Automatic rig",
    "sprite_synthesis": "Sprite sheet",
    "charsheet": "Character sheet",
}

#: What a *finished* follow-up is called, for the toast that announces it.
#: Separate from :data:`LABELS` because these two vocabularies read in different
#: sentence frames -- "Automatic rig was not queued." against "Rig for a fire
#: guardian is ready." -- and because the failure wording is snapshotted into
#: stored records, so rewording it there would change what old rows say.
#:
#: Every kind here mints its row with its *source's* prompt, so without a noun
#: the toast said "fire guardian finished." twice for one character and named
#: neither half. Six entries, because six kinds carry ``params["source_job"]``
#: -- the same list :mod:`warlock.studio.asset_open` routes.
PRODUCTS = {
    "rig": "Rig",
    "sheet": "Sprite sheet",
    "pixel_sheet": "Pixel sheet",
    "sprite_synthesis": "Sprite sheet",
    "charsheet": "Character sheet",
    "retexture": "Re-texture",
    "remesh": "Remesh",
}

#: Which stage a follow-up can be attempted from, and therefore which rows may
#: carry a failure record about it. A record on any other stage is not evidence
#: of anything -- it is the fingerprint of a missing guard. ``_maybe_queue_rig``
#: had no stage guard until 2026-08-21, so it ran on every Troupe *reference*
#: completion and wrote "the generated mesh artifact is missing" onto rows that
#: had not been asked for a mesh yet. The guard is in place now; this keeps the
#: rows already written from showing a warning that was never true.
#:
#: Filtered on read rather than migrated: params are a JSON blob behind one
#: connection, and a schema-version step to rewrite them would be a migration
#: for a bug whose records are all in unreleased working-tree data -- while this
#: also covers a row restored from a backup taken before the guard landed.
APPLICABLE_STAGES = {
    "rig": ("model",),
    "charsheet": ("model",),
    "sprite_synthesis": ("reference",),
}


def failure_record(
    followup: str, error: BaseException | str, *, recorded_at: float | None = None
) -> dict[str, Any]:
    """Build one bounded, JSON-safe failure record."""
    if isinstance(error, BaseException):
        error_type = type(error).__name__
        message = str(error).strip() or error_type
    else:
        error_type = "Unavailable"
        message = str(error).strip() or "The follow-up could not be queued."
    return {
        "kind": followup,
        "label": LABELS.get(followup, followup.replace("_", " ").title()),
        "message": message[:MAX_MESSAGE],
        "error_type": error_type,
        "recorded_at": time.time() if recorded_at is None else float(recorded_at),
    }


def records(params: Any, stage: Any = None) -> list[dict[str, Any]]:
    """Validated follow-up records in stable label order.

    ``stage`` is the row's own stage, and is what drops a record about a
    follow-up this row could never have had -- see :data:`APPLICABLE_STAGES`.
    Optional, and unfiltered when omitted, because a caller that does not know
    the stage is better served by the whole list than by a silent empty one.
    """
    if not isinstance(params, dict):
        return []
    raw = params.get(PARAM_KEY)
    if not isinstance(raw, dict):
        return []
    out: list[dict[str, Any]] = []
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        allowed = APPLICABLE_STAGES.get(key)
        if stage is not None and allowed is not None and str(stage) not in allowed:
            continue
        message = value.get("message")
        if not isinstance(message, str) or not message.strip():
            continue
        out.append(
            {
                "kind": key,
                "label": str(value.get("label") or LABELS.get(key) or key),
                "message": message,
                "error_type": str(value.get("error_type") or "Unavailable"),
                "recorded_at": value.get("recorded_at"),
            }
        )
    return sorted(out, key=lambda item: (item["label"].casefold(), item["kind"]))


def persist(store: Any, parent_id: str, followup: str, error: BaseException | str) -> bool:
    """Upsert one failure into a parent's params. Return False if it vanished."""
    # One hold of the store's lock, not two. This used to ``get`` the row, build
    # the new ``failures`` dict from what it read, and then ``merge_params`` the
    # whole thing back -- a read-modify-write split across two separate holds,
    # which is precisely the last-write-wins shape ``merge_params`` exists to
    # prevent, one level down. Two callers recording two *different* follow-up
    # failures against the same parent would each read the same dict and the
    # second write would drop the first. Only one caller exists today and it is
    # awaited sequentially, so nothing was losing records yet; the invariant was
    # broken regardless, and the next caller would have paid for it.
    return store.merge_param_entry(
        parent_id, PARAM_KEY, followup, failure_record(followup, error)
    ) is not None
