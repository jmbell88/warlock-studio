"""A ``.npy``/``.npz`` reader that asks the header what it costs before paying.

**The hole this closes.** ``.wmap`` stores each tile layer as a ``.npy`` member
and ``.wblk`` stores each mesh as an ``.npz``; both went straight to
``np.lib.format.read_array`` / ``np.load`` and checked the shape *afterwards*
(``wmap._read_layer_array`` compares against the map's own size on the line
after the read). A ``.npy`` header is about 128 bytes and it is the header that
decides the allocation: one declaring ``(2**20, 2**20)`` uint32 asks numpy for
4 TB, and the shape check that would have refused it never runs.

``zipguard.BoundedZip`` does not see this. Its guarantee is that a member does
not unpack past what its own directory entry declared, and a 128-byte member
that declares 128 bytes is honest about every byte of itself. The lie is inside
the payload, one format down -- exactly the relationship ``tmx._decompress`` and
``asein._inflate`` have with the archives above them.

``.npz`` is worse and it is worth being precise about why: ``np.load`` on one
opens a **nested zip with numpy's own plain ``zipfile``**, so the ``BoundedZip``
the outer document was opened through is not in the path at all. The archive
ceiling that the ``.wblk`` door pays for buys nothing past the outer member.
:func:`read_npz` re-opens that inner archive through ``BoundedZip`` and reads
each member through :func:`read_array`, which puts both bounds back.

``allow_pickle`` never appears here because there is nothing to pass it to:
this reader parses the header itself and refuses an object dtype by name. An
``.npz`` is a zip a user can be handed, and an object array in one is arbitrary
code.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np

from . import zipguard

#: The ceiling on one decoded array. A ``.wmap`` layer at the engine's own
#: ``MAX_DIMENSION`` squared is 4096 x 4096 uint32 -- 64 MiB -- and a Clay mesh
#: at ``MAX_TRIANGLES`` is a few tens of MiB across its five fields, so 256 MiB
#: is generous against both and four orders short of the header's reach.
#: Module-level so a test lowers it rather than writing a terabyte-shaped
#: header, and read at call time so lowering it works.
MAX_ARRAY_BYTES = 1 << 28

#: ``read_array_header_1_0`` and ``..._2_0`` are numpy's own public API; there
#: is no public reader for 3.0, and nothing in this repo writes a structured
#: dtype, so a 3.0 file is refused by name rather than read through a private
#: function that may move.
_HEADERS = {
    (1, 0): "read_array_header_1_0",
    (2, 0): "read_array_header_2_0",
}


def read_array(raw: bytes, what: str) -> np.ndarray:
    """One ``.npy``'s bytes as an array, its declared cost checked first.

    Takes ``bytes`` rather than a file object on purpose: the header has to be
    read, judged and then *re-read* by numpy's own loader, and a caller handing
    over a stream it had already consumed part of is the one way to get that
    wrong. Every caller here already holds the whole member anyway -- it came
    out of a zip.
    """
    fh = io.BytesIO(raw)
    try:
        version = np.lib.format.read_magic(fh)
    except ValueError as exc:
        raise ValueError(f"{what} is not stored as a numpy array: {exc}") from exc
    reader = _HEADERS.get(version)
    if reader is None:
        raise ValueError(
            f"{what} is stored in numpy format {version[0]}.{version[1]},"
            " which this build does not read"
        )
    try:
        shape, _fortran, dtype = getattr(np.lib.format, reader)(fh)
    except (ValueError, EOFError) as exc:
        raise ValueError(f"{what} has an unreadable array header: {exc}") from exc
    if dtype.hasobject:
        # ``allow_pickle=False``'s refusal, made here because this reader never
        # reaches numpy's. A pickled object inside a document a user was handed
        # is arbitrary code, and the header is where it announces itself.
        raise ValueError(f"{what} stores Python objects, which this build never unpickles")
    count = 1
    for axis in shape:
        count *= int(axis)
    cost = count * int(dtype.itemsize)
    if count < 0 or cost > MAX_ARRAY_BYTES:
        raise ValueError(
            f"{what} declares {shape} of {dtype}, which is {cost} bytes -- past"
            f" the {MAX_ARRAY_BYTES} this build will allocate"
        )
    fh.seek(0)
    try:
        return np.lib.format.read_array(fh, allow_pickle=False)
    except (ValueError, EOFError) as exc:
        raise ValueError(f"{what} is unreadable: {exc}") from exc


def read_npz(raw: bytes, what: str) -> dict[str, np.ndarray]:
    """One ``.npz``'s bytes as ``{name: array}``, both bounds applied.

    Every member, eagerly, rather than a lazy mapping: ``np.load`` returns an
    ``NpzFile`` that decodes on ``__getitem__`` and holds the archive open until
    it is closed, and the two callers here read every field they were given
    anyway. Eager also means the refusal arrives while the caller is still in
    its own ``try``, rather than three frames away at first use.
    """
    try:
        archive = zipguard.BoundedZip(io.BytesIO(raw))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"{what} is not a numpy archive") from exc
    out: dict[str, np.ndarray] = {}
    with archive:
        for name in archive.namelist():
            if not name.endswith(".npy"):
                # numpy writes nothing else into one, so a stray member is
                # something else's file riding along; skipping it is what
                # ``np.load`` does with it too.
                continue
            out[name[: -len(".npy")]] = read_array(archive.read(name), f"{name} in {what}")
    return out
