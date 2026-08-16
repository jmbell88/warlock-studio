"""Build the overhang fixture: a plate floating in front of a wall.

The minimal mesh on which the missing occlusion test is not a judgement call:
from the front camera the plate hides the centre of the wall completely, the
wall still *faces* that camera square-on, so the un-anchored bake smears the
plate's colours over wall it cannot see -- and a depth-tested bake must leave
that region byte-still. Everything else about the fixture is chosen to keep
the experiment one-variable:

- Both quads face front (+Z in glTF, -Y in Blender once imported). The four
  side/top/bottom axis views see them edge-on (below MIN_FACING) or from
  behind (clamped to zero), so with ``--views 6`` exactly one view speaks and
  the contamination number is the front view's alone. Run the probe with
  ``--views 6`` -- the diagonals legitimately see behind the plate and would
  repaint part of the region honestly.
- The wall's UVs are the atlas's left half, the plate's the right, so neither
  can bleed into the other through dilation at the scales involved.
- The region the plate hides is a fixed rectangle of the wall's UV range:
  ``--occluded-rect 0.125,0.25,0.375,0.75``.

    uv run python scripts/make_overhang_fixture.py --out docs/measurements/data/overhang
    uv run python scripts/retexture_probe.py --out <dir> --views 6 \
        --glb docs/measurements/data/overhang/model.glb="rusted iron plating" \
        --occluded-rect 0.125,0.25,0.375,0.75 [--occlusion --control]
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build(out_dir: Path) -> Path:
    import numpy as np
    import trimesh
    from PIL import Image

    def quad(x0, y0, x1, y1, z, u0, v0, u1, v1, base):
        vertices = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
        uv = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
        faces = [(base, base + 1, base + 2), (base, base + 2, base + 3)]
        return vertices, uv, faces

    wall_v, wall_uv, wall_f = quad(-1, -1, 1, 1, 0.0, 0.0, 0.0, 0.5, 1.0, 0)
    plate_v, plate_uv, plate_f = quad(-0.5, -0.5, 0.5, 0.5, 0.6, 0.5, 0.0, 1.0, 1.0, 4)

    vertices = np.array(wall_v + plate_v, dtype=np.float64)
    uv = np.array(wall_uv + plate_uv, dtype=np.float64)
    faces = np.array(wall_f + plate_f, dtype=np.int64)
    normals = np.tile(np.array([[0.0, 0.0, 1.0]]), (len(vertices), 1))

    # Two flat greys, wall darker: "kept its old skin" is then visible on the
    # contact sheet without opening a diff tool.
    atlas = Image.new("RGB", (512, 512), (90, 90, 96))
    atlas.paste((150, 150, 156), (256, 0, 512, 512))

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=atlas, metallicFactor=0.0, roughnessFactor=0.9
    )
    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=normals,
        visual=trimesh.visual.TextureVisuals(uv=uv, material=material),
        process=False,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "model.glb"
    mesh.export(dest)
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    dest = build(args.out)
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
