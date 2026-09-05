"""What each finished stage can hand the user, as data. Headless.

Lifted out of ``widgets`` on 2026-09-03: ``create_stages`` asks
:func:`artifacts_for` on every frame to decide whether the Export segment of
the rail is reached, and ``widgets`` imports imgui at module scope -- from a
module whose own docstring says it imports nothing from imgui. These are
tables and one lookup; there is nothing to draw.
"""

from __future__ import annotations

from typing import Any

# Every artifact the downloads section offers, in the order it offers them:
# what the mesh *is* first, then what it can be turned into.
ARTIFACTS = (
    ("model.glb", "GLB"),
    ("source.glb", "Source GLB"),
    ("model.stl", "STL"),
    ("model_obj.zip", "OBJ (zip)"),
    ("model.fbx", "FBX"),
    ("collision.glb", "Collision"),
    ("textures.zip", "Textures"),
    ("rig.glb", "Rigged GLB"),
    # The pixels the mesh was reconstructed from. Last because it is an input
    # rather than an output, but present because a promoted job copies it and
    # then had no way to give it back: the inspector showed it as a 96px
    # thumbnail and offered no path to the file.
    ("input.png", "Reference image"),
)

# What a finished *reference* can hand over. A separate tuple rather than a
# filtered ARTIFACTS: the two lists have nothing in common but input.png, and a
# reference offered eight greyed mesh buttons -- which is what it used to get --
# reads as a broken asset rather than as a 2D one.
ARTIFACTS_2D = (
    ("icon.png", "Icon PNG"),
    ("sprite.png", "Sprite PNG"),
    ("pixel_32.png", "Pixel 32"),
    ("pixel_64.png", "Pixel 64"),
    ("pixel_128.png", "Pixel 128"),
    ("manifest.json", "Manifest"),
    ("input.png", "Source image"),
)

# And what a finished *tile* can. Almost none of the list above: every cutout
# is the operation of lifting a subject off a background, and a seamless
# texture is background. The texture itself comes first here rather than last,
# because for a tile input.png is the asset and not the input to one.
ARTIFACTS_TILE = (
    ("input.png", "Tile PNG"),
    ("wrap_preview.png", "Wrapped view"),
    # The zip leads the material group because it is what somebody taking this
    # into an engine wants: all four images plus a glTF material fragment in
    # one file. The individual maps follow, for whoever wants one of them.
    #
    # Every one of those three says "est." and that is not modesty. They are
    # derived from the albedo's own contrast and describe nothing about a
    # surface; a button labelled "Normal map" claims a measurement, and the
    # docstring explaining otherwise is in a repository the user does not have.
    ("material.zip", "Material set (zip)"),
    ("material_normal.png", "Normal (est.)"),
    ("material_roughness.png", "Roughness (est.)"),
    ("material_height.png", "Height (est.)"),
    ("manifest.json", "Manifest"),
)

# And what a finished *tile sheet* can: its own sheet and nothing else. Every
# cutout above is the operation of lifting a subject off a background and a
# grid of tiles has sixty-four; every material map is a claim about one
# surface and this is sixty-four of them. The sheet is the asset, so it is the
# whole list -- an empty grid would read as broken and the mesh list would read
# as eight buttons that produce nothing. Cutting the sheet *into* tiles is a
# real operation and deliberately is not here: it belongs to Packwright's
# tileset import, which takes the sheet as a file and asks what size the tiles
# are.
ARTIFACTS_TILESHEET = (("input.png", "Tile sheet PNG"),)

# What a finished *take* can hand over: the three compressed/lossless
# re-encodings of track.wav, lazily derived exactly the way the mesh exports
# derive from model.glb (``service.derive.get_file``, the DERIVED_AUDIO arm).
#
# track.wav itself is deliberately not a fourth row here. Every other list
# above ends with (or leads with) the artifact its derivations came from --
# model.glb/source.glb, input.png -- but a take's WAV already has a save
# dialog of its own, Muse's own player's "Export the track"
# (studio/muse_io.py), which writes the identical bytes. A second button
# here would be a second door to the same file rather than the door the
# 2026-09-05 audit (finding muse-01) found missing.
ARTIFACTS_MUSIC = (
    ("track.flac", "FLAC"),
    ("track.mp3", "MP3"),
    ("track.ogg", "OGG"),
)


def artifacts_for(job: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """The Export tab's grid for one job.

    Keyed on the stage rather than on which files happen to exist: every entry
    in all three tuples is *derivable*, so a list built from what is on disk
    would hide exactly the exports that have not been produced yet -- which is
    all of them, the first time.

    The two image stages' lists are labels for exactly what
    ``service.files.derived_2d_for`` says each can produce, plus the source
    image every job may take away. They are literals rather than a lookup
    because the *order* is a UI decision the service has no opinion about --
    a tile leads with its own PNG, a reference ends with the image it was
    drawn from -- and the price of that is a second place to edit. What keeps
    the two from drifting is ``test_the_grid_offers_exactly_what_each_stage_
    can_derive``, which fails on a name added to one and not the other: a
    label missing here is not a wrong button, it is no button at all.

    A music job gets its own branch for the same reason a mesh job does not
    fall through to the 2D lists: ``ARTIFACTS`` and ``ARTIFACTS_MUSIC`` share
    nothing (not even a source row -- see that tuple's comment), so a music
    job answered with the mesh default used to draw eight permanently
    blocked buttons and no audio one (the 2026-09-05 audit, finding muse-01).
    """
    stage = job.get("stage")
    if stage == "tile":
        return ARTIFACTS_TILE
    if stage == "reference":
        return ARTIFACTS_2D
    if stage == "tilesheet":
        return ARTIFACTS_TILESHEET
    if stage == "music":
        return ARTIFACTS_MUSIC
    return ARTIFACTS
