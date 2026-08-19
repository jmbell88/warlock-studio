"""A grid pack as a Tiled tileset.

The payoff of the two modes being genuinely different: a grid pack *is* a
tileset -- uniform cells, a constant gutter, a constant border -- so it can be
handed to Plotter, or to Tiled, with no conversion at all. A MaxRects pack is
not one and is refused rather than approximated.

**This module writes no XML.** It builds a ``tilegrid.tileset.Tileset`` and calls
``plotter.tsx.tsx_bytes``, so there is exactly one ``.tsx`` writer in the repo
and a Packwright tileset and a Plotter one cannot become two dialects of the
format. That import is the reason grid geometry is spelled as margin-and-spacing
in :mod:`.layout` rather than as an offset table.
"""

from __future__ import annotations

import numpy as np

from ..plotter.tsx import tsx_bytes
from ..tilegrid.tileset import Tileset
from .layout import Layout


def grid_tileset(layout: Layout, atlas: np.ndarray, *, name: str) -> Tileset:
    """The layout and its atlas as a sliced tileset.

    ``margin`` and ``spacing`` are both the pack's ``padding``, which is what
    :func:`~.layout.grid_layout` builds the geometry to satisfy;
    :class:`~..tilegrid.tileset.Tileset` validates the slicing on construction,
    so a layout that had drifted from that promise fails here rather than in
    Tiled.
    """
    if not layout.is_grid:
        raise ValueError(
            "only a grid pack is a tileset -- a MaxRects atlas has no uniform "
            "cell, so switch the mode or export the JSON sidecar instead"
        )
    # Tiled derives columns/rows from the image alone -- ``image width minus
    # margin, divided by tile + spacing`` -- with no knowledge of what this
    # layout actually placed. Auto packs keep those two in permanent agreement
    # (:func:`~.layout.grid_layout` re-derives after power-of-two rounding for
    # exactly this reason); an *explicit* ``columns`` setting does not, because
    # honouring the user's exact count is the point of setting it. When
    # rounding leaves the image wide enough for Tiled to read more cells than
    # this pack placed, ``tile_rect``'s row-major index math runs against the
    # wrong column count and every frame past the first row lands on the wrong
    # tile -- so this refuses by name instead, the same taste every other
    # geometry conflict here is held to.
    step_w, step_h = layout.cell_w + layout.padding, layout.cell_h + layout.padding
    tiled_columns = max(0, (layout.width - layout.padding) // step_w)
    tiled_rows = max(0, (layout.height - layout.padding) // step_h)
    if tiled_columns != layout.columns or tiled_rows != layout.rows:
        raise ValueError(
            f"this pack is {layout.columns}x{layout.rows} cells, but its "
            f"power-of-two rounding leaves Tiled reading {tiled_columns}x{tiled_rows} "
            "from the image -- turn off power-of-two, or choose a columns count "
            "whose rounded width has no room left over, to export a .tsx"
        )
    return Tileset(
        name=name,
        pixels=atlas,
        tile_w=layout.cell_w,
        tile_h=layout.cell_h,
        spacing=layout.padding,
        margin=layout.padding,
    )


def grid_tsx(layout: Layout, atlas: np.ndarray, *, name: str, image_name: str) -> bytes:
    """The ``.tsx`` file itself."""
    return tsx_bytes(grid_tileset(layout, atlas, name=name), image_name=image_name)
