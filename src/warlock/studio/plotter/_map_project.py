"""Where a cell is, in pixels -- the document's half of the projection.

Placement is deferred to :mod:`.project` rather than inlined, because the canvas
and the flat renderer must agree about where a cell is or an export stops being
a picture of the screen -- the same reason they both take orientation from
:mod:`.gid`. What this mixin adds is the *document's* :class:`~.project.Lattice`,
which every one of those calls needs and none of them should have to gather.

That gathering is :meth:`ProjectionOps._lattice`, and it is the only new thing
here: six methods repeated ``self.projection, self.width, self.height,
self.tile_w, self.tile_h`` verbatim, which is five chances apiece to pass tile
height where tile width goes and get an answer that is wrong only for
non-square tiles. The document has no ``stagger_axis``/``stagger_index``/
``hex_side`` fields yet -- M5 seam, see :mod:`.project` -- so ``_lattice``
leaves those three at :class:`~.project.Lattice`'s own defaults.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from . import project
from .project import Lattice

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .tilemap import MapDoc


class ProjectionOps:
    """Cell-to-pixel placement, mixed into :class:`~.tilemap.MapDoc`."""

    def _lattice(self: MapDoc) -> Lattice:
        """The one object every :mod:`.project` call takes."""
        return Lattice(self.projection, self.width, self.height, self.tile_w, self.tile_h)

    @property
    def isometric(self: MapDoc) -> bool:
        return self.projection == project.ISOMETRIC

    @property
    def pixel_width(self: MapDoc) -> int:
        return project.map_size(self._lattice())[0]

    @property
    def pixel_height(self: MapDoc) -> int:
        return project.map_size(self._lattice())[1]

    def cell_origin(self: MapDoc, column: int, row: int) -> tuple[float, float]:
        """The top-left of a cell's image rectangle, in map pixels."""
        return project.cell_origin(self._lattice(), column, row)

    def cell_corner(self: MapDoc, column: float, row: float) -> tuple[float, float]:
        """A lattice node -- where four cells meet. What a grid line joins."""
        return project.cell_corner(self._lattice(), column, row)

    def cell_at(self: MapDoc, x: float, y: float) -> tuple[int, int]:
        """Which cell a map-pixel point lands in. Unclamped; every tool clips."""
        return project.cell_at(self._lattice(), x, y)

    def cell_bounds(
        self: MapDoc, x0: float, y0: float, x1: float, y1: float
    ) -> tuple[int, int, int, int]:
        return project.cell_bounds(self._lattice(), x0, y0, x1, y1)

    def draw_order(self: MapDoc):
        """Every cell, back to front for this projection."""
        return project.draw_order(self._lattice())
