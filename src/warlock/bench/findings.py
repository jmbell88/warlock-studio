"""The studio side of ``findings.json`` -- a pure stdlib reader with no
import of torch or imgui, so a generate pane can call it every frame.

``load`` is the whole cost model: one ``stat()`` per call, and a re-read only
when the file's mtime moves. A missing file is cached as a miss too, keyed on
the path alone (there is no mtime to key on), so a bench dir that has never
been reported costs one failed ``stat()`` per frame forever, not an
exception path. ``hint`` is a pure lookup against the loaded doc -- see
``report.py`` for the schema (``{"params": {param: {value_str: {"n",
"accepts", ...}}}}``, values keyed by ``str(value)``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# {path: (mtime_or_None, doc_or_None)} -- mtime is None for a cached miss
# (file absent or unreadable), which can never collide with a real mtime.
_CACHE: dict[Path, tuple[float | None, dict[str, Any] | None]] = {}


def load(path: Path) -> dict[str, Any] | None:
    """The parsed ``findings.json`` at ``path``, or ``None`` if it is absent
    or unreadable. Cached by mtime: unchanged since the last call returns the
    same object with no re-read."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        _CACHE[path] = (None, None)
        return None

    cached = _CACHE.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    try:
        import json

        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # Cached by *this* mtime, not None: stat() succeeds for a
        # corrupt-but-present file, so caching (None, None) here would never
        # match a real mtime and every call would re-read and re-parse the
        # same broken file, forever -- exactly the per-frame cost the mtime
        # gate exists to avoid.
        _CACHE[path] = (mtime, None)
        return None

    _CACHE[path] = (mtime, doc)
    return doc


def hint(doc: dict[str, Any] | None, param: str, value: Any, *, min_n: int = 5) -> str | None:
    """``"accept 6/8"`` for ``param``/``value`` in ``doc``, or ``None`` when
    there is no bucket, or it has fewer than ``min_n`` verdicts -- a thin
    bucket is noise, not a finding.

    ``findings.json`` keys a bucket by ``str()`` of the JSON value the sweep
    recorded (e.g. ``"0.6"``), but an imgui float32 slider hands back
    ``0.6000000238418579`` -- the float32 rounding of ``0.6``, not the
    ``float`` Python would make from the string. A straight ``str(value)``
    lookup misses every such slider, silently, forever. So a float value that
    misses the literal-string lookup gets a second pass: every bucket key
    that itself parses as a float is compared to ``value`` rounded to 6
    decimals, which absorbs float32's error (~1e-7 relative) while still
    telling "0.6" apart from "0.65"."""
    if doc is None:
        return None
    bucket = (doc.get("params") or {}).get(param) or {}
    entry = bucket.get(str(value))
    if entry is None and isinstance(value, float):
        target = round(value, 6)
        for key, candidate in bucket.items():
            try:
                key_value = float(key)
            except ValueError:
                continue
            if round(key_value, 6) == target:
                entry = candidate
                break
    if entry is None or entry.get("n", 0) < min_n:
        return None
    return f"accept {entry.get('accepts', 0)}/{entry['n']}"
