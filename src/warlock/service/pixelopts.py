"""The pixelisation options a door validates, once, for every path that has them.

Five paths in this program end in a pixel-art atlas and each one asks the same
four or five questions: how big is a logical pixel, how many colours, which
authored palette, dither, and -- where there is a reduction or an outline pass
to configure -- which mode. ``service/troupe._check_options`` answered them
first and answered them well; this is that function with its two Troupe-shaped
constants (the size ladder and the colour ladder) lifted into parameters, so a
second and third caller is a call rather than a copy.

The wordings are Troupe's verbatim, down to the ``field=`` names, because a
refusal is a string a user reads and a field a form highlights: two paths that
refuse the same value in two different sentences is the drift this module
exists to stop.

Same layer as ``service/troupe``, which is why that module delegates here
rather than the other way round. It states the rule it is now obeying: its own
numbers "come from ``pipelines.charsheet`` and ``pipelines.pixelize`` rather
than being restated ... a second copy here would be one edit away from a form
that offers a size the renderer refuses". Four more doors restating these four
refusals is the same hazard one layer up.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from ..pipelines import pixelize
from .errors import Invalid

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .core import WarlockService


def check_pixel_options(
    svc: WarlockService,
    entries: Mapping[str, Any],
    *,
    sizes: Sequence[int],
    size_default: int,
    colors: Sequence[int],
    colors_default: int,
    outline_default: str = "none",
    allow_outline: bool = True,
    allow_reduce_mode: bool = True,
) -> dict[str, Any]:
    """Validate one request's pixelisation block and return it normalised.

    ``sizes`` and ``colors`` are the ladders this particular path offers --
    they differ per path (a Troupe sheet is laid out at ``charsheet.SIZES``, a
    tile sheet is not) and there is no defensible common set, so they are
    passed rather than picked here.

    ``allow_outline`` and ``allow_reduce_mode`` gate the two options a path may
    not *have*. False means the key is neither validated nor returned, which is
    deliberately stronger than validating it and dropping it: a caller that
    hands an outline mode to a path with no outline pass has made a mistake,
    and the params blob quietly carrying a setting nothing reads is exactly the
    dead field ``_charsheet`` found ``reduce_mode`` to be.

    The palette is **loaded and thrown away**, which is the point of doing it
    here: a palette file that has been deleted, emptied or corrupted since the
    form listed it costs the *request*, rather than a minute of GPU and a sheet
    that merely came back the wrong colours. ``palettes.load`` raises
    ``Invalid`` with the field already set.
    """
    from . import palettes

    try:
        logical = int(entries.get("logical_size") or size_default)
    except (TypeError, ValueError):
        raise Invalid("logical_size must be a whole number", field="logical_size") from None
    if logical not in tuple(sizes):
        raise Invalid(
            f"logical_size must be one of {list(sizes)}",
            field="logical_size",
        )
    try:
        count = int(entries.get("colors") or colors_default)
    except (TypeError, ValueError):
        raise Invalid("colors must be a whole number", field="colors") from None
    if count not in tuple(colors):
        raise Invalid(f"colors must be one of {list(colors)}", field="colors")

    out: dict[str, Any] = {"logical_size": logical, "colors": count}

    if allow_outline:
        outline = str(entries.get("outline") or outline_default)
        if outline not in pixelize.OUTLINE_MODES:
            raise Invalid(
                f"outline must be one of {list(pixelize.OUTLINE_MODES)}",
                field="outline",
            )
        out["outline"] = outline
    if allow_reduce_mode:
        reduce_mode = str(entries.get("reduce_mode") or pixelize.REDUCE_MODES[0])
        if reduce_mode not in pixelize.REDUCE_MODES:
            raise Invalid(
                f"reduce_mode must be one of {list(pixelize.REDUCE_MODES)}",
                field="reduce_mode",
            )
        out["reduce_mode"] = reduce_mode

    palette = str(entries.get("palette") or "").strip()
    if palette:
        palettes.load(svc.config, palette)
    out["dither"] = bool(entries.get("dither"))
    out["palette"] = palette
    return out
