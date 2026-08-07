"""Pure, torch-free prompt assembly and CLIP-token chunking.

Mirrors the pipelines/sheet.py split CLAUDE.md already establishes:
decidable, testable logic lives here with no torch import; only the tensor
work (_encode_long_prompt) stays in pipelines/text2image.py. transformers is
imported only inside the functions that need it, so this module stays
importable without the text2image extra installed.

CLIP's text encoders cap out at 77 tokens (BOS + up to 75 content tokens +
EOS). chunk() splits a longer prompt into multiple <=77-token chunks on
comma boundaries -- guidance.compose_prompt and PROMPT_TEMPLATE both join
fragments with ", ", so a comma is always a safe break point that never
splits a concept mid-phrase. Packing by phrase rather than by raw token
slice also guarantees CLIP-L and CLIP-G (SDXL's two text encoders) produce
the same chunk count for the same text, which text2image._encode_long_prompt
requires: their hidden states concatenate on the feature axis per chunk, so
a mismatched chunk count between the two encoders would misalign every chunk
after the first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Bias generations toward images TRELLIS handles well: one object, clean
# silhouette. Moved here from text2image.py so prompt.build() (used by the
# /api/prompt-preview endpoint) and text2image.generate() share one copy;
# text2image.py re-exports this name so existing readers are unchanged.
#
# Two edits on 2026-08-07, both from the rogue sweep's 17 refusals. Every one
# was reference.py's multi-object rule and they were one family: concept-art
# layouts -- character sheets, turnarounds, multi-view plates -- reproducing
# across every checkpoint, so a property of the prompt rather than of a model.
#
# "game asset concept art" became "game asset render", because a character
# sheet is the canonical form of the genre this asked for: the template was
# requesting the failure. And "single object centered" became "a single subject
# centered ... no other objects", stating the constraint rather than implying
# it. Positive side deliberately, per TODO item 1: `sdxl_cfg` -- full CFG at 30
# steps, where negative adherence is *strongest* -- refused most of all, which
# points at the positive prompt driving the layout; and the negative prompt is
# a field the user can empty, so a constraint living only there is one the
# composed prompt can lose. It also keeps `negative_prompt` -- which is in
# vectors.VECTOR_PARAMS -- unchanged, so no stored vector is re-keyed and every
# unit recorded before today still pairs in findings.comparisons.
PROMPT_TEMPLATE = (
    "{prompt}, a single subject centered on a plain light gray background, "
    "no other objects, 3/4 perspective view, studio lighting, game asset render, "
    "full object in frame, no cropping, no text, no watermark"
)

# The tile template, and every clause of it is the opposite of the one above.
# PROMPT_TEMPLATE asks for a single centred object, the full object in frame
# and no cropping -- which is a description of exactly what a tileable texture
# must not be. A flat orthographic top-down framing is what makes the circular
# padding in text2image produce something that reads as a surface rather than
# as a photograph of one.
TILE_TEMPLATE = (
    "{prompt}, seamless tileable texture, flat top-down orthographic view, "
    "even diffuse lighting, no shadows, uniform scale, repeating pattern, "
    "no single focal object, no text, no watermark, no border"
)

# The sheet template. A contact sheet of orthographic views is neither a single
# centred object nor a texture, and both templates above actively fight it:
# "single object centered" asks the model to compose one subject out of eight
# cells, and the tile template asks for no focal object at all. What this one
# has to protect is the *grid* -- every cell keeping its own subject, its own
# framing and the layout it arrived with, because the cells are already exact
# renders of one mesh and the restyle is only allowed to change how they look.
SHEET_TEMPLATE = (
    "{prompt}, sprite sheet, grid of separate character poses, "
    "flat even lighting, plain background, consistent scale across cells, "
    "each cell a complete figure, no text, no watermark"
)

# The half of the taxonomy that describes a *surface*. The rest -- category,
# silhouette, rarity, mood, emissive, platform -- describes an object, and
# naming one in a tile prompt is how a "cobblestone" tile comes back as a
# picture of a cobblestone.
TILE_FIELDS = ("material", "condition", "palette", "setting", "genre", "art_style")

# Bumped whenever PROMPT_TEMPLATE, TILE_TEMPLATE, TILE_FIELDS or chunk()
# changes. Recorded by provenance.versions() so a prompt-compiler edit cannot
# silently invalidate a benchmark comparison -- no dependency version moves
# when this file does.
#
# 2: TILE_TEMPLATE and the tile field subset. The object path's output is
# unchanged, so an object recipe recorded under 1 still reproduces exactly;
# the bump is about the compiler, not about any one prompt.
# 3: SHEET_TEMPLATE. Same shape of change as 2 -- a third template, reachable
# only from the pixel-sheet restyle, with the object and tile paths untouched.
# 4: PROMPT_TEMPLATE's concept-art and single-subject clauses (see above). The
# object path's output genuinely moves, which is the case this counter exists
# for: an object recipe recorded under 1-3 no longer reproduces byte-for-byte,
# and a benchmark comparing across the bump is comparing two compilers.
PROMPT_VERSION = 4

_tokenizer_cache: dict[Path, tuple[Any, Any]] = {}


def load_tokenizers(model_dir: Path) -> tuple[Any, Any]:
    """CLIP-L and CLIP-G tokenizers for a local diffusers checkout, cached by
    directory. ~1.5 MB of vocab/merges each -- no weights, no network, no
    VRAM, so this is safe to call from a request handler.
    """
    cached = _tokenizer_cache.get(model_dir)
    if cached is not None:
        return cached
    from transformers import CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(
        str(model_dir), subfolder="tokenizer", local_files_only=True
    )
    tok2 = CLIPTokenizer.from_pretrained(
        str(model_dir), subfolder="tokenizer_2", local_files_only=True
    )
    _tokenizer_cache[model_dir] = (tok, tok2)
    return (tok, tok2)


def count(text: str, tokenizers: list[Any]) -> int:
    """Token count under the strictest of ``tokenizers``, BOS/EOS included --
    this is what the encoder actually receives and what the 77-token limit
    means in practice.

    Calls the tokenizer with ``verbose=False``: this is the one place we
    deliberately measure text longer than the model's stated max length (the
    whole point of chunking), and without it transformers logs its own
    "Token indices sequence length is longer than..." warning every time --
    a false alarm here since nothing is actually truncated, just measured.
    """
    return max(len(tok(text, verbose=False)["input_ids"]) for tok in tokenizers)


def chunk(text: str, tokenizers: list[Any], limit: int = 75) -> list[str]:
    """Split ``text`` into chunks that fit ``limit`` content tokens (+2 for
    BOS/EOS = 77, the encoder's real max) under every tokenizer in
    ``tokenizers``, without splitting a phrase across a chunk boundary.

    Splits on comma boundaries first. A single phrase that is still over the
    limit on its own (no smaller comma boundary inside it) falls back to a
    greedy whitespace split, so one abnormally long fragment still chunks
    instead of producing a chunk no tokenizer can hold.
    """

    def fits(s: str) -> bool:
        return count(s, tokenizers) <= limit + 2

    def hard_split(piece: str) -> list[str]:
        """Last resort for a single whitespace-free atom that alone exceeds
        the limit (a pasted URL, a run of garbage): binary-search the longest
        character prefix that fits and repeat. Without this the atom would be
        emitted as an over-limit chunk and silently truncated at encode time.
        """
        parts: list[str] = []
        while piece and not fits(piece):
            lo, hi, best = 1, len(piece) - 1, 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if fits(piece[:mid]):
                    best, lo = mid, mid + 1
                else:
                    hi = mid - 1
            parts.append(piece[:best])
            piece = piece[best:]
        if piece:
            parts.append(piece)
        return parts

    phrases = [p.strip() for p in text.split(",") if p.strip()]
    if not phrases:
        return [""]

    # (piece, separator to place before it when appended to a chunk).
    atoms: list[tuple[str, str]] = []
    for i, phrase in enumerate(phrases):
        sep = ", " if i > 0 else ""
        if fits(phrase):
            atoms.append((phrase, sep))
        else:
            for j, word in enumerate(phrase.split(" ")):
                word_sep = sep if j == 0 else " "
                if fits(word):
                    atoms.append((word, word_sep))
                else:
                    for k, frag in enumerate(hard_split(word)):
                        atoms.append((frag, word_sep if k == 0 else ""))

    chunks: list[str] = []
    current = ""
    for piece, sep in atoms:
        candidate = f"{current}{sep}{piece}" if current else piece
        if current and not fits(candidate):
            chunks.append(current)
            current = piece
        else:
            current = candidate
    chunks.append(current)
    return chunks


def pad_pair(a: list[str], b: list[str]) -> tuple[list[str], list[str]]:
    """Pad the shorter list with "" so both have equal chunk counts.

    Required because diffusers' own StableDiffusionXLPipeline.__call__
    concatenates positive and negative embeddings on the batch axis under
    classifier-free guidance (torch.cat([negative_prompt_embeds,
    prompt_embeds], dim=0)), which needs equal sequence lengths -- here,
    equal chunk counts.
    """
    n = max(len(a), len(b))
    return (a + [""] * (n - len(a)), b + [""] * (n - len(b)))


def build(
    user_prompt: str, params: dict[str, Any], *, trigger: str = "", tile: bool = False
) -> str:
    """The final positive prompt.

    Guidance fragments, then the LoRA trigger (if any), then the framing
    template -- the same assembly text2image.generate() does by hand, exposed
    here so the prompt preview can show it before a job runs. ``tile`` swaps
    both halves at once: the surface-only field subset and the tileable
    template, which have to travel together because either alone produces the
    wrong picture.
    """
    from .. import guidance

    composed = guidance.compose_prompt(
        user_prompt, params, fields=TILE_FIELDS if tile else None
    )
    template = TILE_TEMPLATE if tile else PROMPT_TEMPLATE
    text = template.format(prompt=composed)
    return f"{trigger}, {text}" if trigger else text
