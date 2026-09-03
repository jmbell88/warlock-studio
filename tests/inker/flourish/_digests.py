"""Regenerate ``digests.json`` -- deliberately, when a primitive's arithmetic
is meant to change. ``uv run python tests/inker/flourish/_digests.py``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent


def main() -> int:
    sys.path.insert(0, str(HERE))
    from _recipes import FIREBALL, solo  # noqa: PLC0415

    # The same functions the test uses, imported from it so they cannot drift.
    from test_flourish_render import _digest, _frames  # noqa: PLC0415

    from warlock.studio.inker import flourish
    from warlock.studio.inker.flourish import prims

    got = {"fireball": _digest(_frames(FIREBALL)), "fireball@90": _digest(_frames(FIREBALL, 90.0))}
    for kind in prims.KINDS:
        got[f"solo:{kind}"] = _digest(_frames(solo(kind)))
    out = HERE / "digests.json"
    out.write_text(json.dumps(got, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(got)} digests, schema {flourish.SCHEMA_VERSION})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
