"""``.wpack`` -- the Packwright document on disk, as a zip.

``pack.json`` plus one ``sources/<n>.png``, epoch-stamped throughout, so two
saves of an unchanged document are byte-identical -- the ``.wblk``/``.wmap``
rule, applied a third time and for the third time because a file that changes
every time it is written is undiffable and its content hash useless.

**The atlas is not in here, and neither is the layout.** Both are derived from
what is: storing an atlas would mean the file could disagree with the settings
beside it, and the packer is deterministic so there is nothing to gain by it.
Exporting is a separate act with its own destination.

**Source pixels are embedded rather than referenced.** A source is routinely a
*frame of an Inker document* that has since been edited, or a layer of one, and
neither has a path at all -- so a reference would be unresolvable for the
majority of them. What the document records is what was packed; re-adding the
source is how you pick up a change. This is the ``.wmap`` tileset argument with
a sharper case.
"""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any

import numpy as np

from ..plotter.pngio import png_bytes
from .document import PackDoc, Source, new_uid
from .layout import PackSettings
from .sources import Sprite

VERSION = 1
MANIFEST = "pack.json"
SOURCE_DIR = "sources"

_EPOCH = (1980, 1, 1, 0, 0, 0)

# A zip's directory declares what each member unpacks to and nothing makes that
# number honest -- a few kilobytes of archive can claim terabytes, and the read
# that discovers this is the one that has already exhausted memory. One gigabyte
# is far past any atlas document this editor produces (a few hundred sprites of
# a megapixel each is tens of megabytes) and far short of a machine's RAM, so
# the ceiling is only ever hit by a file that was not written by us. The
# ``clay/serialize.py`` constant verbatim, read from module globals at call time
# for its reason too: a test lowers it rather than building a gigabyte.
MAX_DECOMPRESSED_BYTES = 1 << 30

# And a ceiling per source image, because the byte ceiling above is about the
# archive rather than about any one member: a single 60000-square PNG deflates
# to very little and decodes to fourteen gigabytes of RGBA. The number mirrors
# ``service.validation.MAX_IMAGE_PIXELS`` rather than importing it -- this
# package may not reach for the service layer, which its import pin enforces --
# so the two are one answer written twice on purpose.
MAX_SOURCE_PIXELS = 16_000_000


def manifest_json(doc: PackDoc) -> str:
    settings = doc.settings
    payload = {
        "version": VERSION,
        "settings": {
            "mode": settings.mode,
            "padding": settings.padding,
            "extrude": settings.extrude,
            "trim": settings.trim,
            "max_size": settings.max_size,
            "power_of_two": settings.power_of_two,
        },
        "sources": [
            {
                "key": source.key,
                "name": source.sprite.name,
                "name_override": source.name_override,
                "image": f"{SOURCE_DIR}/{index}.png",
            }
            for index, source in enumerate(doc.sources)
        ],
    }
    return json.dumps(payload, sort_keys=True, indent=2)


def wpack_bytes(doc: PackDoc) -> bytes:
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(zipfile.ZipInfo(MANIFEST, _EPOCH), manifest_json(doc))
        for index, source in enumerate(doc.sources):
            # Stored, not deflated: a PNG is already compressed.
            zf.writestr(
                zipfile.ZipInfo(f"{SOURCE_DIR}/{index}.png", _EPOCH),
                png_bytes(source.sprite.pixels),
                zipfile.ZIP_STORED,
            )
    return out.getvalue()


def read_wpack(data: bytes) -> PackDoc:
    """A ``.wpack``'s bytes back into a :class:`~.document.PackDoc`.

    Restored by construction, so the document reads clean: a file that has just
    been opened is not unsaved.
    """
    from . import layout as laylib

    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ValueError("this is not a Warlock atlas document") from exc

    # Everything that can raise is inside the ``with``, the version check and
    # the manifest parse included. They used to sit in the gap between the open
    # and the ``with``, where a refusal -- the one thing they exist to do --
    # left the archive open with nothing to close it but the collector. The
    # ``clay/serialize.read_wblk`` and ``wmap.read_wmap`` shape, third instance.
    with zf:
        # Before anything is read: the directory says what every member unpacks
        # to, and the cheapest place to refuse an archive that claims more than
        # this build will hold is before the first ``read``.
        claimed = sum(int(info.file_size) for info in zf.infolist())
        ceiling = MAX_DECOMPRESSED_BYTES
        if claimed > ceiling:
            raise ValueError(
                f"this atlas document claims {claimed} bytes unpacked, "
                f"past the {ceiling} this build will read"
            )

        try:
            manifest = json.loads(zf.read(MANIFEST))
        except (
            zipfile.BadZipFile,
            KeyError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise ValueError("this is not a Warlock atlas document") from exc
        if not isinstance(manifest, dict):
            raise ValueError("this atlas document's manifest is malformed")

        try:
            version = int(manifest.get("version", 0))
        except (TypeError, ValueError) as exc:
            raise ValueError("this atlas document's manifest is malformed") from exc
        if version > VERSION:
            raise ValueError(
                f"this atlas document was written by a newer version of Warlock "
                f"(format {version}, this build reads {VERSION})"
            )

        entries = manifest.get("sources", [])
        if not isinstance(entries, list):
            raise ValueError("this atlas document's manifest is malformed")
        # The same ceiling the packer refuses at, asked before any of them is
        # decoded: a manifest listing a million sources is a million PNG decodes
        # ahead of the refusal that was always going to come.
        if len(entries) > laylib.MAX_SPRITES:
            raise ValueError(
                f"this atlas document lists {len(entries)} sources; "
                f"{laylib.MAX_SPRITES} is the most this build will open"
            )

        settings = _settings_from(manifest.get("settings"))
        sources: list[Source] = []
        seen: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("this atlas document holds a malformed source")
            key = str(entry.get("key", ""))
            if not key:
                raise ValueError("this atlas document holds a source with no key")
            if key in seen:
                # Refused rather than deduplicated: the two would pack into one
                # slot and the loser would be missing with nothing to say so.
                raise ValueError(f"this atlas document holds {key!r} twice")
            seen.add(key)
            name = str(entry.get("image", ""))
            try:
                raw = zf.read(name)
            except KeyError as exc:
                raise ValueError(
                    f"this atlas document names a source image the file does not "
                    f"carry ({name})"
                ) from exc
            sources.append(
                Source(
                    uid=new_uid(),
                    sprite=Sprite(
                        key=key,
                        name=str(entry.get("name", key)),
                        pixels=_pixels_from(raw, name),
                    ),
                    name_override=str(entry.get("name_override", "")),
                )
            )

    doc = PackDoc(sources=sources, settings=settings)
    doc.mark_saved()
    return doc


def _pixels_from(raw: bytes, name: str) -> np.ndarray:
    """One embedded member as RGBA, refused at the door if it is not one.

    ``UnidentifiedImageError`` is an ``OSError``, so a member that is not an
    image at all and one that is a truncated PNG come out of the same clause.
    The pixel ceiling is asked *before* ``convert``, because that is the call
    that allocates: a header saying 60000 squared is four bytes on disk and
    fourteen gigabytes decoded.
    """
    from PIL import Image

    try:
        image = Image.open(io.BytesIO(raw))
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError(
            f"this atlas document's {name} is not an image this build can read"
        ) from exc
    with image:
        if image.width * image.height > MAX_SOURCE_PIXELS:
            raise ValueError(
                f"this atlas document's {name} is {image.width}x{image.height}; "
                f"the limit is {MAX_SOURCE_PIXELS} pixels"
            )
        return np.asarray(image.convert("RGBA"), dtype=np.uint8)


def _settings_from(entry: Any) -> PackSettings:
    values = entry if isinstance(entry, dict) else {}
    default = PackSettings()
    try:
        kwargs = {
            "mode": str(values.get("mode", default.mode)),
            "padding": int(values.get("padding", default.padding)),
            "extrude": int(values.get("extrude", default.extrude)),
            "trim": bool(values.get("trim", default.trim)),
            "max_size": int(values.get("max_size", default.max_size)),
            "power_of_two": bool(values.get("power_of_two", default.power_of_two)),
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("this atlas document's settings are malformed") from exc
    # Constructed *outside* the coercion guard on purpose. ``PackSettings``
    # validates on construction -- including the padding-against-extrude rule --
    # so a hand-edited manifest is refused with the reason rather than producing
    # an atlas that bleeds at some zoom levels, and catching that here would
    # replace "padding must be at least twice extrude" with a generic sentence.
    return PackSettings(**kwargs)
