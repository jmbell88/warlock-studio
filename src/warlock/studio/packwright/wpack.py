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

WPACK_SUFFIX = ".wpack"


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
    from PIL import Image

    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
        manifest = json.loads(zf.read(MANIFEST))
    except (zipfile.BadZipFile, KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("this is not a Warlock atlas document") from exc
    if not isinstance(manifest, dict):
        raise ValueError("this atlas document's manifest is malformed")

    version = int(manifest.get("version", 0))
    if version > VERSION:
        raise ValueError(
            f"this atlas document was written by a newer version of Warlock "
            f"(format {version}, this build reads {VERSION})"
        )

    with zf:
        settings = _settings_from(manifest.get("settings"))
        sources: list[Source] = []
        seen: set[str] = set()
        for entry in manifest.get("sources", []):
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
            image = Image.open(io.BytesIO(raw)).convert("RGBA")
            sources.append(
                Source(
                    uid=new_uid(),
                    sprite=Sprite(
                        key=key,
                        name=str(entry.get("name", key)),
                        pixels=np.asarray(image, dtype=np.uint8),
                    ),
                    name_override=str(entry.get("name_override", "")),
                )
            )

    doc = PackDoc(sources=sources, settings=settings)
    doc.mark_saved()
    return doc


def _settings_from(entry: Any) -> PackSettings:
    values = entry if isinstance(entry, dict) else {}
    default = PackSettings()
    # ``PackSettings`` validates on construction -- including the
    # padding-against-extrude rule -- so a hand-edited manifest is refused here
    # rather than producing an atlas that bleeds at some zoom levels.
    return PackSettings(
        mode=str(values.get("mode", default.mode)),
        padding=int(values.get("padding", default.padding)),
        extrude=int(values.get("extrude", default.extrude)),
        trim=bool(values.get("trim", default.trim)),
        max_size=int(values.get("max_size", default.max_size)),
        power_of_two=bool(values.get("power_of_two", default.power_of_two)),
    )
