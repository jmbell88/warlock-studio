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


#: What ``entries`` may hold for a number and still mean "the form said
#: nothing". ``None`` is an absent key and ``""`` is a form field nobody typed
#: in; **0 is neither**. It is a value, it is on no ladder any path publishes,
#: and answering it with the default is the failure ``service/jobs`` already
#: names on the tile size beside this one -- "make me 96px tiles" answered with
#: 32px tiles and nobody told. So 0 falls through to the ladder check and is
#: refused there by name, in the sentence that lists what it could have been.
_UNSET: tuple[Any, ...] = (None, "")


def _on_the_ladder(
    entries: Mapping[str, Any], key: str, default: int, ladder: Sequence[int]
) -> int:
    """One whole-number field, defaulted when absent and refused when wrong."""
    raw = entries.get(key)
    if raw in _UNSET:
        raw = default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise Invalid(f"{key} must be a whole number", field=key) from None
    if value not in tuple(ladder):
        raise Invalid(f"{key} must be one of {list(ladder)}", field=key)
    return value


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
    outline_refusal: str = "",
) -> dict[str, Any]:
    """Validate one request's pixelisation block and return it normalised.

    ``sizes`` and ``colors`` are the ladders this particular path offers --
    they differ per path (a Troupe sheet is laid out at ``charsheet.SIZES``, a
    tile sheet is not) and there is no defensible common set, so they are
    passed rather than picked here.

    ``allow_outline`` and ``allow_reduce_mode`` gate the two options a path may
    not *have*, and what False does depends on whether the caller **asked**:

    * A request that said nothing has the key dropped -- neither validated nor
      returned -- so the params blob never carries a setting nothing reads,
      which is the dead field ``_charsheet`` found ``reduce_mode`` to be.
    * A request that named a mode is **refused**, on ``field="outline"`` /
      ``field="reduce_mode"``. A caller that hands an outline mode to a path
      with no outline pass has made a mistake, and diagnosing a mistake and
      then swallowing it is how the caller comes to believe the setting took.
      Dropping is the *weaker* answer, not the stronger one.

    ``"none"`` is not an ask. It is the word every one of these doors spells
    "off", and refusing it would refuse a request for exactly what this path
    does anyway.

    ``outline_refusal`` is the sentence that refusal carries, because *why*
    there is no outline pass is a fact about the caller's kind rather than
    about this function -- a tile sheet's reason is ``pixelize._edge_mask``
    padding ``constant_values=False``, and that argument belongs beside the
    tiles. Absent, a plainer sentence stands in.

    The palette is **loaded and thrown away**, which is the point of doing it
    here: a palette file that has been deleted, emptied or corrupted since the
    form listed it costs the *request*, rather than a minute of GPU and a sheet
    that merely came back the wrong colours. ``palettes.load`` raises
    ``Invalid`` with the field already set.
    """
    from . import palettes

    logical = _on_the_ladder(entries, "logical_size", size_default, sizes)
    count = _on_the_ladder(entries, "colors", colors_default, colors)

    out: dict[str, Any] = {"logical_size": logical, "colors": count}

    if allow_outline:
        outline = str(entries.get("outline") or outline_default)
        if outline not in pixelize.OUTLINE_MODES:
            raise Invalid(
                f"outline must be one of {list(pixelize.OUTLINE_MODES)}",
                field="outline",
            )
        out["outline"] = outline
    elif str(entries.get("outline") or "none") != "none":
        raise Invalid(
            outline_refusal
            or "this kind has no outline pass, so there is no outline mode to set",
            field="outline",
        )
    if allow_reduce_mode:
        reduce_mode = str(entries.get("reduce_mode") or pixelize.REDUCE_MODES[0])
        if reduce_mode not in pixelize.REDUCE_MODES:
            raise Invalid(
                f"reduce_mode must be one of {list(pixelize.REDUCE_MODES)}",
                field="reduce_mode",
            )
        out["reduce_mode"] = reduce_mode
    elif entries.get("reduce_mode") not in _UNSET:
        # The same rule as the outline above, and there is no second argument
        # for it: a path with one reduction has no alternative to pick, and a
        # request that picked one has to hear that rather than watch it vanish.
        raise Invalid(
            "this kind has one reduction and no choice of it, so there is no "
            "reduce_mode to set",
            field="reduce_mode",
        )

    palette = str(entries.get("palette") or "").strip()
    if palette:
        palettes.load(svc.config, palette)
    out["dither"] = bool(entries.get("dither"))
    out["palette"] = palette
    return out
