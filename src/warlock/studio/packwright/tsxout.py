"""A grid pack as a Tiled tileset.

The payoff of the two modes being genuinely different: a grid pack *is* a
tileset -- uniform cells, a constant gutter, a constant border -- so it can be
handed to Plotter, or to Tiled, with no conversion at all. A MaxRects pack is
not one and is refused rather than approximated.

**This module writes no XML.** It builds a ``plotter.tileset.Tileset`` and calls
``plotter.tsx.tsx_bytes``, so there is exactly one ``.tsx`` writer in the repo
and a Packwright tileset and a Plotter one cannot become two dialects of the
format. That import is the reason grid geometry is spelled as margin-and-spacing
in :mod:`.layout` rather than as an offset table.
"""

from __future__ import annotations

import numpy as np

from ..plotter.tileset import Tileset
from ..plotter.tsx import tsx_bytes
from .layout import Layout


def grid_tileset(layout: Layout, atlas: np.ndarray, *, name: str) -> Tileset:
    """The layout and its atlas as a sliced tileset.

    ``margin`` and ``spacing`` are both the pack's ``padding``, which is what
    :func:`~.layout.grid_layout` builds the geometry to satisfy;
    :class:`~..plotter.tileset.Tileset` validates the slicing on construction,
    so a layout that had drifted from that promise fails here rather than in
    Tiled.
    """
    if not layout.is_grid:
        raise ValueError(
            "only a grid pack is a tileset -- a MaxRects atlas has no uniform "
            "cell, so switch the mode or export the JSON sidecar instead"
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
