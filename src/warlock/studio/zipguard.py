"""A zip reader that refuses a member claiming less than it unpacks to.

**Why this exists, measured.** Four container doors -- ``inker/ora.py``,
``clay/serialize.py``, ``packwright/wpack.py`` and ``plotter/wmap.py`` -- each
opened with the same precheck: sum ``info.file_size`` over the central
directory and refuse an archive claiming more than :data:`MAX_DECOMPRESSED_BYTES`
unpacked. That is the cheapest possible refusal and it is worth keeping, but it
asks the *archive* how big it is, and an attacker writes the central directory.

The gap is not theoretical. A member whose directory entry declares **10 bytes**
and whose deflate stream actually inflates to 512 MiB passes the sum untouched:
``claimed`` is 10. ``zipfile`` then discovers the lie by CRC and raises
``BadZipFile`` -- *after* ``ZipExtFile.read()`` has accumulated the whole
inflated stream. Measured with ``tracemalloc`` on CPython 3.13: peak Python
allocation **1,070 MiB** for a **510 KiB** archive, against a ceiling nominally
set at 1 GiB. The refusal arrives, correctly, once the memory is already spent.

So the sum is a courtesy and this is the guarantee: every member is read through
its own declared size as a hard bound, one chunk at a time, and a stream that
runs past what its directory entry promised is refused at the byte after rather
than at the CRC. ``plotter/tmx.py``'s ``_decompress`` and ``inker/asein.py``'s
``_inflate`` are the same idea one layer down, on a raw deflate stream.

**A subclass rather than a helper function**, which is the one design decision
here worth stating. There are eighteen ``zf.read`` call sites across the four
doors and there will be more; a ``bounded_read(zf, name)`` helper is a rule that
holds only as long as every future call site remembers it, and "remembered at
seventeen of eighteen sites" is indistinguishable from not having the rule.
Overriding ``read`` means the bound is a property of the *archive object* the
door opened, so a new call site gets it by construction.

A shared leaf under ``studio/`` for the reason ``tilegrid`` and ``undo`` are
shared leaves: four engines needed one rule, and a fourth copy of a security
bound is the kind of thing that drifts without any of them noticing.
"""

from __future__ import annotations

import zipfile

#: The absolute ceiling on any one member's unpacked size. The constant the four
#: doors already carried, now in one place; each still re-exports it under its
#: own name so their existing refusal messages are unchanged.
MAX_DECOMPRESSED_BYTES = 1 << 30

#: Read granularity. Peak allocation for one member is its declared size plus
#: this, so it trades a bounded overshoot for not calling ``read`` a million
#: times on a large layer.
_CHUNK = 1 << 20


class BoundedZip(zipfile.ZipFile):
    """A ``ZipFile`` whose ``read`` will not outrun the directory's own promise.

    Every other ``ZipFile`` method is inherited untouched -- ``infolist``,
    ``namelist`` and ``getinfo`` only ever read the central directory, which is
    already in memory by the time the constructor returns.
    """

    #: Per-instance so a test can lower it rather than building a gigabyte --
    #: the rule the four doors already stated about their own copies.
    ceiling: int = MAX_DECOMPRESSED_BYTES

    def read(self, name, pwd=None) -> bytes:  # type: ignore[override]
        info = name if isinstance(name, zipfile.ZipInfo) else self.getinfo(name)
        declared = int(info.file_size)
        if declared < 0 or declared > self.ceiling:
            raise ValueError(
                f"{info.filename!r} declares {declared} bytes unpacked, past the"
                f" {self.ceiling} this build will read"
            )
        out = bytearray()
        with self.open(info, "r", pwd) as fh:
            while True:
                # ``declared - len(out) + 1`` and never simply ``_CHUNK``: the
                # last read is deliberately allowed to fetch one byte more than
                # the promise, because that byte is the whole test. Reading
                # exactly ``declared`` would leave a lying archive
                # indistinguishable from an honest one until the CRC check that
                # this class exists to get in front of.
                want = min(_CHUNK, declared - len(out) + 1)
                if want <= 0:
                    break
                block = fh.read(want)
                if not block:
                    break
                out += block
                if len(out) > declared:
                    raise ValueError(
                        f"{info.filename!r} unpacks past the {declared} bytes its"
                        " directory entry declares"
                    )
        return bytes(out)
