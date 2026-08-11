"""GLB container reading and rewriting.

Factored out of ``pipelines.postprocess`` so the viewer's loader and the
transform-insertion code share one implementation of the binary format. The
split/rebuild pair is deliberately lossless in the second half: ``rest`` is
every byte after the JSON chunk, verbatim, so rewriting the JSON never touches
the binary payload (which is the only thing large enough to be worth not
copying wrongly).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

GLB_MAGIC = 0x46546C67  # 'glTF'
CHUNK_JSON = 0x4E4F534A  # 'JSON'
CHUNK_BIN = 0x004E4942  # 'BIN\0'


def split_glb(data: bytes) -> tuple[bytes, dict, bytes]:
    """-> (12-byte header, parsed JSON chunk, every following byte verbatim)."""
    magic, version, _length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC:
        raise ValueError("not a GLB file")
    if version != 2:
        raise ValueError(f"unsupported GLB version {version}")
    chunk_len, chunk_type = struct.unpack_from("<II", data, 12)
    if chunk_type != CHUNK_JSON:
        raise ValueError("first GLB chunk is not JSON")
    if 20 + chunk_len > len(data):
        # A slice past the end is silently short in Python, so a body truncated
        # mid-JSON would reach json.loads as a partial document and fail there
        # with a decoder's error rather than the format's. Callers key on
        # ValueError either way; this one says what actually went wrong.
        raise ValueError("truncated GLB: the JSON chunk overruns the file")
    start = 20
    return data[:12], json.loads(data[start : start + chunk_len]), data[start + chunk_len :]


def rebuild_glb(header: bytes, gltf: dict, rest: bytes) -> bytes:
    # The JSON chunk pads with spaces (0x20) per the glTF spec, not zeros.
    payload = json.dumps(gltf, separators=(",", ":")).encode()
    payload += b" " * (-len(payload) % 4)
    total = len(header) + 8 + len(payload) + len(rest)
    return (
        header[:8]
        + struct.pack("<I", total)
        + struct.pack("<II", len(payload), CHUNK_JSON)
        + payload
        + rest
    )


def read_glb(path: Path | bytes) -> tuple[dict, bytes]:
    """-> (glTF JSON, the BIN chunk's payload).

    What a loader wants, as opposed to what a rewriter wants: the binary buffer
    unwrapped from its chunk header. A GLB with no BIN chunk yields ``b""``
    rather than raising -- the accessors then have to name an external buffer,
    which the loader refuses on its own terms with a better message.
    """
    data = path if isinstance(path, bytes) else Path(path).read_bytes()
    _header, gltf, rest = split_glb(data)
    offset = 0
    while offset + 8 <= len(rest):
        chunk_len, chunk_type = struct.unpack_from("<II", rest, offset)
        body = rest[offset + 8 : offset + 8 + chunk_len]
        if chunk_type == CHUNK_BIN:
            return gltf, body
        offset += 8 + chunk_len
    return gltf, b""
