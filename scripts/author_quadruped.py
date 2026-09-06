"""Bake the quadruped archetype's silhouette meshes, and report the numbers.

    uv run python scripts/author_quadruped.py            # measure, write nothing
    uv run python scripts/author_quadruped.py --write    # regenerate the assets

The assets under ``src/warlock/characters/quadruped/`` are **checked in**, for
``scripts/author_humanoid.py``'s two reasons and they are both about somebody
else's build rather than this one: ``manifold3d`` does not promise byte-stable
output across versions, and Blender has to import a file whatever we do. So the
generator is the record of *how* and the ``.glb`` / ``.masks.npz`` pair is the
record of *what*; ``tests/characters/test_quadruped.py`` re-runs the first and
compares it to the second.

The report at the bottom is the argument for baking per **silhouette group**
rather than per species.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ARCHETYPE = "quadruped"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the assets")
    parser.add_argument("--only", default="", help="one silhouette key")
    args = parser.parse_args(argv)

    from warlock.characters import family as familylib
    from warlock.characters.quadruped import generate

    directory = ROOT / "src" / "warlock" / "characters" / ARCHETYPE
    groups = familylib.silhouettes(ARCHETYPE)
    wanted = [args.only] if args.only else sorted(groups)

    total_glb = total_npz = 0
    for silhouette in wanted:
        species = groups[silhouette]
        if args.write:
            stats = generate.write_assets(silhouette, directory)
        else:
            import io

            import numpy as np

            baked = generate.build(silhouette)
            data, arrays = generate.bake(baked)
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **arrays)
            stats = {
                "faces": baked.face_count,
                "vertices": len(baked.positions),
                "glb_bytes": len(data),
                "npz_bytes": buffer.tell(),
            }
        total_glb += stats["glb_bytes"]
        total_npz += stats["npz_bytes"]
        print(
            f"{silhouette:<10} {stats['faces']:>6} faces  {stats['vertices']:>6} verts  "
            f"glb {stats['glb_bytes'] / 1024:7.1f} KB  npz {stats['npz_bytes'] / 1024:7.1f} KB  "
            f"-> {len(species)} species: {', '.join(species)}"
        )

    count = sum(len(v) for v in groups.values())
    per_group = (total_glb + total_npz) / max(len(wanted), 1)
    print()
    print(
        f"{len(wanted)} silhouettes serve {count} species: "
        f"{(total_glb + total_npz) / 1024:.1f} KB checked in. "
        f"One asset per species would be {count * per_group / 1024:.1f} KB "
        f"for the same {count} characters."
    )
    print("Wrote nothing (pass --write)." if not args.write else f"Written to {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
