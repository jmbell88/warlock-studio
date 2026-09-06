"""Regenerate ``digests.json`` -- deliberately, when the motion is meant to
change, and never to make a red test go green.
``uv run python tests/inker/walk/_digests.py``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent


def main() -> int:
    sys.path.insert(0, str(HERE))
    # The same two functions the test uses, imported from it so they cannot drift.
    from test_walk_render import _composites, _digest  # noqa: PLC0415

    from warlock.studio.inker import walk  # noqa: PLC0415

    got = {"figure": _digest(_composites())}
    out = HERE / "digests.json"
    out.write_text(json.dumps(got, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(got)} digests, {walk.WALK_FRAMES} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
