"""What a new map is, asked before the map exists.

Plotter used to answer "New map" by making a 32x32 grid of 32px cells,
orthogonal, and saying so nowhere. Two of those four numbers cannot be taken
back by any amount of editing afterwards -- **projection** is fixed the moment
anything is painted (a set drawn for one lattice paints the wrong shape into
every cell drawn for the other), and the **tile size** is what a plain image is
sliced at when it is added, so a tileset added to a 32px map is a 32px tileset
forever. A default nobody was asked about is a poor place to put a decision that
permanent, which is what this module exists to move.

Pure, and deliberately: no imgui, no ``ctx``. It owns *what the numbers mean* --
the presets, the caps, and what Create does with the answer -- so the popup that
collects them is a body that draws fields, and the tests do not need a window.
The pane half lives in :mod:`.panes.plotter_canvas`, registered per pane for
the reason ``inker_canvas.new_canvas_popup`` documents: a popup belongs to the
window that begins it.
"""

from __future__ import annotations

from typing import Any

from .plotter import project

#: The starting points, as ``(label, width, height, tile_w, tile_h, projection)``.
#: Sizes in tiles, tile sizes in pixels. Three rather than a page of them: a
#: preset list is a way of *not* answering the question, and the useful answers
#: are "the two common cell sizes" and "the other lattice".
PRESETS: tuple[tuple[str, int, int, int, int, str], ...] = (
    ("Small, 16 px tiles", 40, 30, 16, 16, project.ORTHOGONAL),
    ("Standard, 32 px tiles", 32, 32, 32, 32, project.ORTHOGONAL),
    # 2:1 is the isometric convention, and the generator already warns when a
    # map departs from it -- so the preset that exists to be correct is 2:1.
    ("Isometric, 64 x 32", 32, 32, 64, 32, project.ISOMETRIC),
)

#: The preset Create starts on.
DEFAULT = PRESETS[1]

#: What a *typed* number may do. The fields accept free text, so one stray digit
#: turns 32 into 320 -- and unlike Inker's canvas, a map multiplies: 512 tiles
#: square at 32 px is the 268-megapixel composite ``plotter_mode.export_library``
#: already warns about. The tile cap matches, for the same arithmetic read the
#: other way. Both are limits on a slip of the keyboard, not on the engine.
MAX_TILES = 512
MAX_TILE_PX = 512

#: What to do about a tileset once the map exists. The map is unpaintable until
#: it has one, so the dialog offers the two doors rather than leaving the user
#: to find them -- these are the keys, and ``panes.plotter_canvas`` routes them.
NEXT_EMPTY = "empty"
NEXT_FILE = "file"
NEXT_GENERATE = "generate"


def blank_form() -> dict[str, Any]:
    """The dialog's state, on :attr:`DEFAULT`."""
    label, width, height, tile_w, tile_h, projection = DEFAULT
    return {
        "width": width,
        "height": height,
        "tile_w": tile_w,
        "tile_h": tile_h,
        "projection": projection,
        "next": NEXT_FILE,
        "preset": label,
    }


def apply_preset(form: dict[str, Any], label: str) -> dict[str, Any]:
    """``form`` moved onto the named preset, in place. Unknown labels are kept
    as a custom entry rather than refused: the label is only a note about where
    the numbers came from, and the numbers are the answer."""
    for name, width, height, tile_w, tile_h, projection in PRESETS:
        if name == label:
            form.update(
                width=width,
                height=height,
                tile_w=tile_w,
                tile_h=tile_h,
                projection=projection,
                preset=name,
            )
            break
    return form


def clamp(form: dict[str, Any]) -> dict[str, Any]:
    """Every number brought inside its cap, in place.

    Clamped on the way *into* the fields as well as on the way out, the rule
    ``inker_canvas``'s new-canvas popup states: a box that goes on showing a
    number the Create button will not honour is a box that lies.
    """
    form["width"] = max(1, min(int(form.get("width", 1) or 1), MAX_TILES))
    form["height"] = max(1, min(int(form.get("height", 1) or 1), MAX_TILES))
    form["tile_w"] = max(1, min(int(form.get("tile_w", 1) or 1), MAX_TILE_PX))
    form["tile_h"] = max(1, min(int(form.get("tile_h", 1) or 1), MAX_TILE_PX))
    if form.get("projection") not in project.PROJECTIONS:
        form["projection"] = project.ORTHOGONAL
    return form


def size_of(form: dict[str, Any]) -> tuple[int, int, int, int]:
    """The ``new_document`` size tuple this form describes."""
    clamp(form)
    return (int(form["width"]), int(form["height"]), int(form["tile_w"]), int(form["tile_h"]))


def summary(form: dict[str, Any]) -> str:
    """One line saying what Create will make, in both units.

    Both, because the two numbers that matter are in different ones: a map is
    authored in tiles and *exported* in pixels, and 512 square at 32 px is the
    difference between a sentence and a surprise.
    """
    width, height, tile_w, tile_h = size_of(form)
    return (
        f"{width} x {height} tiles at {tile_w} x {tile_h} px "
        f"-- {width * tile_w} x {height * tile_h} px overall"
    )


def isometric_warning(form: dict[str, Any]) -> str:
    """The 2:1 note, or "" when there is nothing to say.

    The generator's own wording and the same rule; repeated here because this
    dialog is now the first place a projection is chosen, and it would be the
    one place that let the mistake through silently.
    """
    clamp(form)
    if form["projection"] != project.ISOMETRIC:
        return ""
    if form["tile_w"] == form["tile_h"] * 2:
        return ""
    return (
        f"An isometric cell is conventionally 2:1 -- {form['tile_h'] * 2} x {form['tile_h']} "
        f"rather than {form['tile_w']} x {form['tile_h']}. It will still be created."
    )
