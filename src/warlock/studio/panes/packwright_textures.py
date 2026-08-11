"""The atlas texture, one per open pack document.

The ``inker_textures`` / ``plotter_textures`` rules: registered with the imgui
backend as well as created, and ``forget_texture``d before release.

The upload is gated on ``PackTab.pack_generation``, which moves in exactly one
place -- ``adopt_pack``. Gating on anything derived (a hash of the pixels, the
settings, the document's rev) would either re-upload a megapixel every frame or
miss a repack that produced the same-sized atlas with different contents.
"""

from __future__ import annotations

from typing import Any

from .. import docmodes

PREFIX = "packwright_tex:"


def atlas_texture(ctx: Any, tab: Any) -> Any:
    """The packed atlas as a GL texture, or ``None`` with no pack or no GL."""
    if ctx.viewer is None or tab.atlas is None:
        return None
    key = f"{PREFIX}{tab.uid}:atlas"
    gen_key = f"{key}:gen"
    texture = ctx.state.preview.get(key)
    size = (int(tab.atlas.shape[1]), int(tab.atlas.shape[0]))
    if texture is not None and (
        texture.size != size or ctx.state.preview.get(gen_key) != tab.pack_generation
    ):
        docmodes.forget_texture(texture)
        texture = None
    if texture is None:
        texture = ctx.viewer.ctx.texture(size, 4, tab.atlas.tobytes())
        # Nearest: an atlas is inspected at whole-pixel zooms and a linear
        # filter would blur exactly the seam the padding exists to protect.
        nearest = ctx.viewer.ctx.NEAREST
        texture.filter = (nearest, nearest)
        ctx.state.preview[key] = texture
        ctx.state.preview[gen_key] = tab.pack_generation
    return texture


def release_doc(ctx: Any, uid: str) -> None:
    docmodes.release_prefix(ctx, f"{PREFIX}{uid}:")


def release_all(ctx: Any) -> None:
    docmodes.release_prefix(ctx, PREFIX)
