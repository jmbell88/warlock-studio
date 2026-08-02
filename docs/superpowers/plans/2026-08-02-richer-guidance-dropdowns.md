# Richer Design Guidance (12 dropdowns + chunked prompt encoding) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add eight new optional guidance dropdowns (material, condition, setting, palette, emissive, rarity, silhouette, mood) alongside the existing four, and fix the pre-existing bug this surfaces: the composed SDXL prompt already exceeds CLIP's 77-token limit and is silently truncated today.

**Architecture:** Guidance fields stay data-driven (one `Option` table per field in `guidance.py`, consumed by both the API and the UI via `catalog()`/`form_fields()`). The truncation fix moves prompt assembly to pre-encoded embeddings: a new pure module (`pipelines/prompt.py`) splits the composed prompt into ≤77-token chunks on comma boundaries, each chunk is encoded through both SDXL text encoders and concatenated on the sequence axis (`pipelines/text2image.py`), removing the hard ceiling without changing what a short prompt (the common case) produces bit-for-bit. A new `/api/prompt-preview` endpoint and a debounced preview line let the user see the token/chunk cost of their selections before submitting.

**Tech Stack:** Python 3.12, FastAPI 0.141.1 / Starlette 1.3.1, diffusers 0.39.0, transformers 5.14.1 (`CLIPTokenizer`), plain-JS frontend (no build step), pytest.

## Global Constraints

- Fully offline: no new runtime downloads. `CLIPTokenizer.from_pretrained(..., local_files_only=True)` only.
- No frontend build step: `static/index.html` / `static/app.js` are edited directly.
- Keep every new prompt fragment to 2-4 words — chunking removes the hard 77-token ceiling but not the soft one (length still dilutes cross-attention).
- Do not trim or reword the four existing fragments (`GENRES`, `ART_STYLES`, `CATEGORIES`, `PLATFORMS`) or `PROMPT_TEMPLATE`/`DEFAULT_NEGATIVE_PROMPT` — job params are stored and recomposed at run time, so an old job must keep reproducing identically.
- `generate()`'s signature in `pipelines/text2image.py` must not change — `tests/test_fakes_match_real_signatures.py:34` pins it against `conftest.FakeText2Image.generate`.
- Every verified line number below is against commit `13b0259` (current HEAD, `master`). Re-check line numbers if the working tree has moved since.

---

## Task 1: Eight new option tables in `guidance.py`

**Files:**
- Modify: `src/warlock/guidance.py`
- Test: `tests/test_guidance.py`

**Interfaces:**
- Produces: `guidance.MATERIALS`, `CONDITIONS`, `SETTINGS`, `PALETTES`, `EMISSIVES`, `RARITIES`, `SILHOUETTES`, `MOODS` (each `dict[str, Option]`, same shape as the existing `GENRES`); `guidance.form_fields() -> tuple[str, ...]`.
- Consumes: nothing new — reuses `Option`, `_table()`, `_OPTION_TABLES`, `_TABLES`, `normalize()`, `compose_prompt()`, `catalog()`, all already in this file.

- [ ] **Step 1: Add the eight tables**

Insert after `PLATFORMS` (currently ends at line 119 in `src/warlock/guidance.py`), before `_OPTION_TABLES`:

```python
MATERIALS = _table(
    Option("wood", "Wood", "wooden construction"),
    Option("iron", "Iron", "iron construction"),
    Option("steel", "Steel", "steel construction"),
    Option("bronze", "Bronze", "bronze construction"),
    Option("stone", "Stone", "carved stone"),
    Option("leather", "Leather", "leather construction"),
    Option("bone", "Bone", "bone construction"),
    Option("crystal", "Crystal", "faceted crystal"),
    Option("glass", "Glass", "glass construction"),
    Option("ceramic", "Ceramic", "glazed ceramic"),
    Option("gold", "Gold", "gilded gold accents"),
    Option("fabric", "Fabric", "woven fabric"),
)

CONDITIONS = _table(
    Option("pristine", "Pristine", "pristine condition"),
    Option("worn", "Worn", "worn weathered surfaces"),
    Option("damaged", "Damaged", "battle-damaged surfaces"),
    Option("ancient", "Ancient", "ancient timeworn surfaces"),
    Option("rusted", "Rusted", "rusted corroded surfaces"),
    Option("overgrown", "Overgrown", "moss-covered and overgrown"),
    Option("burned", "Burned", "charred burned surfaces"),
)

SETTINGS = _table(
    Option("medieval", "Medieval", "medieval European setting"),
    Option("norse", "Norse", "Norse Viking setting"),
    Option("japanese", "Japanese", "feudal Japanese setting"),
    Option("egyptian", "Egyptian", "ancient Egyptian setting"),
    Option("greco", "Greco-Roman", "Greco-Roman setting"),
    Option("steampunk", "Steampunk", "steampunk brass setting"),
    Option("cyberpunk", "Cyberpunk", "cyberpunk neon setting"),
    Option("tribal", "Tribal", "tribal primitive setting"),
    Option("deco", "Art Deco", "art deco setting"),
    Option("military", "Military", "modern military setting"),
)

PALETTES = _table(
    Option("earth", "Earth tones", "earthy natural palette"),
    Option("steel", "Cool steel", "cool steel palette"),
    Option("muted", "Muted", "muted desaturated palette"),
    Option("vibrant", "Vibrant", "vibrant saturated palette"),
    Option("mono", "Monochrome", "monochrome palette"),
    Option("crimson", "Crimson", "crimson red palette"),
    Option("verdigris", "Verdigris", "verdigris green patina"),
    Option("ivory", "Ivory", "ivory cream palette"),
)

EMISSIVES = _table(
    Option("runes", "Glowing runes", "glowing magic runes"),
    Option("neon", "Neon", "glowing neon accents"),
    Option("molten", "Molten cracks", "glowing molten cracks"),
    Option("holo", "Holographic", "holographic light accents"),
    Option("arcane", "Arcane glow", "arcane energy glow"),
    Option("toxic", "Toxic glow", "toxic green glow"),
)

RARITIES = _table(
    Option("common", "Common", "common plain quality"),
    Option("uncommon", "Uncommon", "uncommon refined quality"),
    Option("rare", "Rare", "rare exceptional quality"),
    Option("epic", "Epic", "epic masterwork quality"),
    Option("legendary", "Legendary", "legendary mythical quality"),
)

SILHOUETTES = _table(
    Option("bulky", "Bulky", "bulky heavy silhouette"),
    Option("slender", "Slender", "slender narrow silhouette"),
    Option("compact", "Compact", "compact dense silhouette"),
    Option("angular", "Angular", "angular sharp silhouette"),
    Option("rounded", "Rounded", "rounded soft silhouette"),
    Option("elongated", "Elongated", "elongated tall silhouette"),
)

MOODS = _table(
    Option("heroic", "Heroic", "heroic noble mood"),
    Option("grim", "Grim", "grim dark mood"),
    Option("whimsical", "Whimsical", "whimsical playful mood"),
    Option("sacred", "Sacred", "sacred reverent mood"),
    Option("sinister", "Sinister", "sinister menacing mood"),
    Option("regal", "Regal", "regal majestic mood"),
)
```

Note: `PALETTES["steel"]` and `MATERIALS["steel"]` share a key name — this is fine, each `_table()` is its own namespace (`_TABLES["palette"]["steel"]` vs `_TABLES["material"]["steel"]` never collide).

- [ ] **Step 2: Register the tables and reorder `_PROMPT_FIELDS`**

Replace (current lines 121-140):

```python
_OPTION_TABLES: dict[str, dict[str, Option]] = {
    "genre": GENRES,
    "art_style": ART_STYLES,
    "category": CATEGORIES,
    "platform": PLATFORMS,
}

# The model-selection fields are validated and stored here so the API gets its
# 400-on-unknown from the same place as everything else, but they are owned by
# models.py and are deliberately absent from _PROMPT_FIELDS: a checkpoint is
# not a prompt fragment, and a LoRA's trigger words are model scaffolding that
# belongs next to PROMPT_TEMPLATE, not creative direction.
_TABLES: dict[str, dict[str, Any]] = {
    **_OPTION_TABLES,
    "base_model": models.BASE_MODELS,
    "style_lora": models.STYLE_LORAS,
}

# Order matters: this is the order fragments appear in the composed prompt.
_PROMPT_FIELDS = ("category", "genre", "art_style", "platform")
```

with:

```python
_OPTION_TABLES: dict[str, dict[str, Option]] = {
    "genre": GENRES,
    "art_style": ART_STYLES,
    "category": CATEGORIES,
    "platform": PLATFORMS,
    "material": MATERIALS,
    "condition": CONDITIONS,
    "setting": SETTINGS,
    "palette": PALETTES,
    "emissive": EMISSIVES,
    "rarity": RARITIES,
    "silhouette": SILHOUETTES,
    "mood": MOODS,
}

# The model-selection fields are validated and stored here so the API gets its
# 400-on-unknown from the same place as everything else, but they are owned by
# models.py and are deliberately absent from _PROMPT_FIELDS: a checkpoint is
# not a prompt fragment, and a LoRA's trigger words are model scaffolding that
# belongs next to PROMPT_TEMPLATE, not creative direction.
_TABLES: dict[str, dict[str, Any]] = {
    **_OPTION_TABLES,
    "base_model": models.BASE_MODELS,
    "style_lora": models.STYLE_LORAS,
}

# Order matters: this is the order fragments appear in the composed prompt, so
# it should read like a sentence. This preserves the relative order of the
# four original fields (category -> genre -> art_style -> platform), which
# tests/test_guidance.py pins.
_PROMPT_FIELDS = (
    "category", "silhouette", "material", "condition", "rarity", "emissive",
    "setting", "genre", "mood", "art_style", "palette", "platform",
)


def form_fields() -> tuple[str, ...]:
    """Every field name the API accepts as a taxonomy value."""
    return tuple(_TABLES)
```

- [ ] **Step 3: Fix the hardcoded persist loop**

Replace (current lines 229-232, inside `normalize()`):

```python
    for field in ("genre", "art_style", "category"):
        option = chosen[field]
        if option is not None:
            out[field] = option.key
    return out
```

with:

```python
    # Every optional taxonomy field except platform, which is always written
    # explicitly above. Derived from _OPTION_TABLES rather than a hand-picked
    # tuple so a new table can never be silently dropped from params again.
    for field in _OPTION_TABLES:
        if field == "platform":
            continue
        option = chosen[field]
        if option is not None:
            out[field] = option.key
    return out
```

- [ ] **Step 4: Add the module-docstring token-budget rule**

In the module docstring (top of `src/warlock/guidance.py`), after the existing paragraph about `pipelines/text2image.PROMPT_TEMPLATE`, add:

```python
Every Option.prompt fragment here is kept to 2-4 words. Chunked encoding in
pipelines/prompt.py and pipelines/text2image.py removes CLIP's hard 77-token
ceiling, but not the soft one -- a longer conditioning sequence still dilutes
cross-attention, so brevity stays a rule even though truncation no longer is.
```

- [ ] **Step 5: Enrich the four presets**

In `PRESETS` (current lines 256-307), add four new-field values to each preset's `"fields"` dict:

```python
PRESETS: tuple[dict[str, Any], ...] = (
    {
        "key": "handpainted_prop",
        "label": "Hand-painted fantasy prop",
        "prompt": "a weathered wooden crate bound with iron",
        "fields": {
            "category": "prop",
            "genre": "fantasy",
            "art_style": "handpainted",
            "platform": "desktop",
            "base_model": "sdxl",
            "style_lora": "render3d",
            "material": "wood",
            "condition": "worn",
            "setting": "medieval",
            "palette": "earth",
        },
    },
    {
        "key": "ps1_character",
        "label": "PS1 low-poly character",
        "prompt": "a hooded adventurer standing in a neutral pose",
        "fields": {
            "category": "character",
            "genre": "fantasy",
            "art_style": "lowpoly",
            "platform": "mobile",
            "base_model": "sdxl",
            "style_lora": "ps1",
            "silhouette": "slender",
            "condition": "worn",
            "setting": "medieval",
            "mood": "heroic",
        },
    },
    {
        "key": "scifi_hero_weapon",
        "label": "Sci-fi hero weapon",
        "prompt": "a compact energy rifle with panel seams and glowing vents",
        "fields": {
            "category": "weapon",
            "genre": "scifi",
            "art_style": "realistic",
            "platform": "hero",
            "base_model": "playground",
            "material": "steel",
            "condition": "pristine",
            "emissive": "neon",
            "rarity": "epic",
        },
    },
    {
        "key": "modern_pickup",
        "label": "Modern consumable pickup",
        "prompt": "a small first-aid kit",
        "fields": {
            "category": "consumable",
            "genre": "modern",
            "art_style": "stylized",
            "platform": "mobile",
            "base_model": "turbo",
            "material": "fabric",
            "condition": "pristine",
            "palette": "vibrant",
            "rarity": "common",
        },
    },
)
```

- [ ] **Step 6: Update `test_guidance.py` for the new taxonomy**

In `tests/test_guidance.py`, replace the field-count assertion at line 91:

```python
def test_catalog_covers_every_field_and_is_json_safe():
    import json

    catalog = guidance.catalog()
    assert set(catalog["fields"]) == {
        "genre", "art_style", "category", "platform", "base_model", "style_lora",
        "material", "condition", "setting", "palette", "emissive", "rarity",
        "silhouette", "mood",
    }
    assert all(o["resolution"] for o in catalog["fields"]["platform"])
    assert all(o["default_size_m"] for o in catalog["fields"]["category"])
    assert all(o["default_weight"] for o in catalog["fields"]["style_lora"])
    json.dumps(catalog)
```

Add new tests at the end of the file:

```python
def test_new_fields_validate_and_reject_unknown():
    for field in (
        "material", "condition", "setting", "palette", "emissive",
        "rarity", "silhouette", "mood",
    ):
        table = getattr(guidance, {
            "material": "MATERIALS", "condition": "CONDITIONS", "setting": "SETTINGS",
            "palette": "PALETTES", "emissive": "EMISSIVES", "rarity": "RARITIES",
            "silhouette": "SILHOUETTES", "mood": "MOODS",
        }[field])
        some_key = next(iter(table))
        assert guidance.normalize({field: some_key})[field] == some_key
        with pytest.raises(ValueError, match=field):
            guidance.normalize({field: "nonsense"})


def test_a_chosen_new_field_value_survives_into_params():
    params = guidance.normalize({"material": "iron", "mood": "grim"})
    assert params["material"] == "iron"
    assert params["mood"] == "grim"


def test_compose_prompt_emits_new_fragments_in_prompt_field_order():
    params = guidance.normalize(
        {"category": "weapon", "silhouette": "angular", "material": "steel",
         "genre": "scifi", "platform": "hero"}
    )
    composed = guidance.compose_prompt("a rifle", params)
    positions = [
        composed.index(guidance.CATEGORIES["weapon"].prompt),
        composed.index(guidance.SILHOUETTES["angular"].prompt),
        composed.index(guidance.MATERIALS["steel"].prompt),
        composed.index(guidance.GENRES["scifi"].prompt),
        composed.index(guidance.PLATFORMS["hero"].prompt),
    ]
    assert positions == sorted(positions)


def test_form_fields_covers_every_table():
    assert set(guidance.form_fields()) == {
        "genre", "art_style", "category", "platform", "base_model", "style_lora",
        "material", "condition", "setting", "palette", "emissive", "rarity",
        "silhouette", "mood",
    }
```

`tests/test_guidance.py:74` (`test_compose_prompt_orders_fragments_after_the_user_text`) and `:186` (`test_every_shipped_preset_normalizes`) must pass **unchanged** — the first proves the field reorder preserved the four original fields' relative order, the second proves the enriched presets are valid.

- [ ] **Step 7: Run the guidance tests**

Run: `uv run pytest tests/test_guidance.py -v`
Expected: all pass, including the untouched `test_compose_prompt_orders_fragments_after_the_user_text` and `test_every_shipped_preset_normalizes`.

- [ ] **Step 8: Commit**

```bash
git add src/warlock/guidance.py tests/test_guidance.py
git commit -m "feat: add eight design-guidance dropdowns to the taxonomy"
```

---

## Task 2: New pure module `pipelines/prompt.py`

**Files:**
- Create: `src/warlock/pipelines/prompt.py`
- Test: `tests/test_prompt.py`

**Interfaces:**
- Consumes: `guidance.compose_prompt(user_prompt, params) -> str` (Task 1, unchanged signature).
- Produces: `PROMPT_TEMPLATE: str`; `chunk(text: str, tokenizers: list, limit: int = 75) -> list[str]`; `pad_pair(a: list[str], b: list[str]) -> tuple[list[str], list[str]]`; `load_tokenizers(model_dir: Path) -> tuple[CLIPTokenizer, CLIPTokenizer]`; `count(text: str, tokenizers: list) -> int`; `build(user_prompt: str, params: dict, *, trigger: str = "") -> str`. These are consumed by Task 3 (`text2image.py`) and Task 5 (`app.py`'s `/api/prompt-preview`).

- [ ] **Step 1: Write the module**

```python
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
PROMPT_TEMPLATE = (
    "{prompt}, single object centered on a plain light gray background, "
    "3/4 perspective view, studio lighting, game asset concept art, "
    "full object in frame, no cropping, no text, no watermark"
)

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
    means in practice."""
    return max(len(tok(text)["input_ids"]) for tok in tokenizers)


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
                atoms.append((word, sep if j == 0 else " "))

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

    Required before text2image._encode_long_prompt concatenates positive and
    negative embeddings on the batch axis: torch.cat needs equal sequence
    lengths, which here means equal chunk counts.
    """
    n = max(len(a), len(b))
    return (a + [""] * (n - len(a)), b + [""] * (n - len(b)))


def build(user_prompt: str, params: dict[str, Any], *, trigger: str = "") -> str:
    """The final positive prompt: guidance fragments, then the LoRA trigger
    (if any), then PROMPT_TEMPLATE's TRELLIS-friendly framing -- the same
    assembly text2image.generate() has always done by hand, exposed here so
    /api/prompt-preview can show it before a job runs.
    """
    from .. import guidance

    composed = guidance.compose_prompt(user_prompt, params)
    text = PROMPT_TEMPLATE.format(prompt=composed)
    return f"{trigger}, {text}" if trigger else text
```

- [ ] **Step 2: Write `tests/test_prompt.py`**

```python
from __future__ import annotations

from pathlib import Path

import pytest

from warlock.pipelines import prompt

pytest.importorskip("transformers")

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "sdxl-turbo"
pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "tokenizer" / "vocab.json").exists(),
    reason="local sdxl-turbo tokenizer not downloaded",
)


@pytest.fixture(scope="module")
def tokenizers():
    return prompt.load_tokenizers(MODEL_DIR)


def test_short_prompt_yields_exactly_one_chunk(tokenizers):
    chunks = prompt.chunk("a wooden crate, worn condition", tokenizers)
    assert chunks == ["a wooden crate, worn condition"]


def test_a_long_prompt_yields_more_than_one_chunk(tokenizers):
    fragment = "an ornate medieval fantasy weapon with intricate engravings"
    long_text = ", ".join([fragment] * 8)
    chunks = prompt.chunk(long_text, tokenizers)
    assert len(chunks) > 1
    for c in chunks:
        assert prompt.count(c, tokenizers) <= 77


def test_chunking_never_splits_a_phrase(tokenizers):
    text = "alpha one, beta two, gamma three"
    chunks = prompt.chunk(text, tokenizers, limit=5)
    for phrase in ("alpha one", "beta two", "gamma three"):
        assert any(phrase in c for c in chunks)


def test_a_lone_overlong_phrase_falls_back_to_whitespace_split(tokenizers):
    single_phrase = " ".join(["overlong"] * 40)
    chunks = prompt.chunk(single_phrase, tokenizers, limit=10)
    assert len(chunks) > 1
    for c in chunks:
        assert prompt.count(c, tokenizers) <= 12


def test_empty_prompt_yields_one_empty_chunk(tokenizers):
    assert prompt.chunk("", tokenizers) == [""]


def test_pad_pair_equalises_chunk_counts():
    a, b = prompt.pad_pair(["x", "y", "z"], ["only"])
    assert len(a) == len(b) == 3
    assert b == ["only", "", ""]

    a2, b2 = prompt.pad_pair(["x"], ["x"])
    assert a2 == ["x"]
    assert b2 == ["x"]


def test_build_reproduces_the_trigger_and_template_order():
    text = prompt.build("a barrel", {}, trigger="3d style, 3d render")
    assert text.startswith("3d style, 3d render, a barrel, single object centered")
    assert text.endswith("no cropping, no text, no watermark")


def test_build_with_no_trigger_has_no_leading_comma():
    text = prompt.build("a barrel", {})
    assert text.startswith("a barrel, single object centered")
```

- [ ] **Step 3: Run the new tests**

Run: `uv run pytest tests/test_prompt.py -v`
Expected: all pass (skipped if the local `models/sdxl-turbo/tokenizer` weights are not present, matching how the rest of the suite treats missing weights).

- [ ] **Step 4: Commit**

```bash
git add src/warlock/pipelines/prompt.py tests/test_prompt.py
git commit -m "feat: add pure prompt chunking module for CLIP's 77-token limit"
```

---

## Task 3: Chunked encoding in `pipelines/text2image.py`

**Files:**
- Modify: `src/warlock/pipelines/text2image.py`

**Interfaces:**
- Consumes: `pipelines.prompt.{PROMPT_TEMPLATE, chunk, pad_pair}` (Task 2).
- Produces: `Text2Image.last_prompt: str` (set inside `generate()`, read by Task 4's `queue.py`); `Text2Image.model_dir -> Path` property (read by Task 5's `/api/prompt-preview`).

No automated test in this task: `Text2Image.generate()` needs a loaded SDXL pipeline (~7 GB VRAM + local weights), which nothing in this suite currently exercises directly — the existing coverage goes through `conftest.FakeText2Image` instead (see Task 4). This dev machine has both a CUDA GPU and the `sdxl-turbo` weights already downloaded, so the acceptance criteria in Task 8 (manual GPU verification) are the real test for this module.

- [ ] **Step 1: Replace the module-level `PROMPT_TEMPLATE` with a re-export**

Replace (current lines 30-35):

```python
# Bias generations toward images TRELLIS handles well: one object, clean silhouette.
PROMPT_TEMPLATE = (
    "{prompt}, single object centered on a plain light gray background, "
    "3/4 perspective view, studio lighting, game asset concept art, "
    "full object in frame, no cropping, no text, no watermark"
)
```

with:

```python
from .prompt import PROMPT_TEMPLATE, chunk, pad_pair  # noqa: F401 -- re-exported
```

Place this import alongside the existing `from .. import models` line near the top of the file (both are cheap, torch-free imports).

- [ ] **Step 2: Add `last_used`-style `last_prompt` and a `model_dir` property**

In `Text2Image.__init__` (current lines 59-73), next to `self.last_used: float = 0.0`, add:

```python
        self.last_prompt: str = ""
```

Add a property near the existing `loaded` property (current lines 75-77):

```python
    @property
    def model_dir(self) -> Path:
        """Where this base model's weights live -- same resolution Text2Image
        itself uses, exposed so a caller (the prompt-preview endpoint) can
        load matching tokenizers without reimplementing WARLOCK_T2I_ROOT."""
        return self._model_dir
```

- [ ] **Step 3: Add `_encode_long_prompt`**

Add as a module-level function, above the `Text2Image` class:

```python
def _encode_long_prompt(pipe, chunks: list[str]):
    """Encode ``chunks`` through both SDXL text encoders and concatenate on
    the sequence axis, replicating per-chunk what
    StableDiffusionXLPipeline.encode_prompt does for a single <=77-token
    prompt: padding="max_length", max_length=77, truncation=True,
    hidden_states[-2], concatenated across the two encoders on the feature
    axis. The UNet's cross-attention has no sequence-length limit, so N
    encoded 77-token chunks concatenated on the sequence axis is a valid,
    longer conditioning sequence -- this is what A1111, ComfyUI and compel
    do to work around the same CLIP limit.

    Pooled is a single vector, not a sequence, so it cannot be concatenated
    across chunks: it comes from text_encoder_2's output on the first chunk
    only, matching encode_prompt's own "only ALWAYS interested in the pooled
    output of the final text encoder."
    """
    import torch

    device = pipe._execution_device
    per_chunk_embeds = []
    pooled = None
    for i, chunk_text in enumerate(chunks):
        chunk_embeds = []
        for tokenizer, text_encoder in (
            (pipe.tokenizer, pipe.text_encoder),
            (pipe.tokenizer_2, pipe.text_encoder_2),
        ):
            inputs = tokenizer(
                chunk_text,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                output = text_encoder(
                    inputs.input_ids.to(device), output_hidden_states=True
                )
            if i == 0 and text_encoder is pipe.text_encoder_2:
                pooled = output[0]
            chunk_embeds.append(output.hidden_states[-2])
        per_chunk_embeds.append(torch.cat(chunk_embeds, dim=-1))
    embeds = torch.cat(per_chunk_embeds, dim=1)
    assert pooled is not None
    dtype = pipe.text_encoder_2.dtype
    return embeds.to(dtype=dtype, device=device), pooled.to(dtype=dtype, device=device)
```

- [ ] **Step 4: Wire it into `generate()`**

Replace (current lines 237-254):

```python
        # The LoRA's trigger words sit here rather than in guidance.py: they are
        # what the adapter was fitted on, so they belong with the rest of the
        # model-facing scaffolding.
        style = models.STYLE_LORAS.get(lora or "")
        text = PROMPT_TEMPLATE.format(prompt=prompt)
        if style is not None and style.trigger and lora in self._adapters:
            text = f"{style.trigger}, {text}"

        image = self._pipe(
            text,
            negative_prompt=negative_prompt or None,
            num_inference_steps=steps,
            guidance_scale=self.spec.guidance_scale,
            width=self.spec.image_size,
            height=self.spec.image_size,
            generator=torch.Generator("cuda").manual_seed(seed),
            callback_on_step_end=step_cb,
        ).images[0]
```

with:

```python
        # The LoRA's trigger words sit here rather than in guidance.py: they are
        # what the adapter was fitted on, so they belong with the rest of the
        # model-facing scaffolding.
        style = models.STYLE_LORAS.get(lora or "")
        text = PROMPT_TEMPLATE.format(prompt=prompt)
        if style is not None and style.trigger and lora in self._adapters:
            text = f"{style.trigger}, {text}"
        self.last_prompt = text

        tokenizers = [self._pipe.tokenizer, self._pipe.tokenizer_2]
        positive_chunks = chunk(text, tokenizers)
        # Only playground (guidance_scale > 1) runs classifier-free guidance;
        # turbo and sdxl+Hyper-SD run at guidance_scale=0.0, where diffusers
        # ignores the negative prompt outright (force_zeros_for_empty_prompt),
        # so skipping the extra encode on the default path costs nothing.
        negative_chunks: list[str] | None = None
        if self.spec.guidance_scale > 1.0:
            negative_chunks = chunk(negative_prompt or "", tokenizers)
            positive_chunks, negative_chunks = pad_pair(positive_chunks, negative_chunks)

        prompt_embeds, pooled_prompt_embeds = _encode_long_prompt(self._pipe, positive_chunks)
        negative_embeds = negative_pooled = None
        if negative_chunks is not None:
            negative_embeds, negative_pooled = _encode_long_prompt(self._pipe, negative_chunks)

        image = self._pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            negative_pooled_prompt_embeds=negative_pooled,
            num_inference_steps=steps,
            guidance_scale=self.spec.guidance_scale,
            width=self.spec.image_size,
            height=self.spec.image_size,
            generator=torch.Generator("cuda").manual_seed(seed),
            callback_on_step_end=step_cb,
        ).images[0]
```

- [ ] **Step 5: Run the full non-GPU test suite to confirm nothing broke**

Run: `uv run pytest -v`
Expected: same pass/skip counts as before this task (this module has no GPU-independent tests of its own; this step just confirms the edit didn't break an import elsewhere, e.g. `queue.py`'s `from .pipelines.text2image import Text2Image`).

- [ ] **Step 6: Commit**

```bash
git add src/warlock/pipelines/text2image.py
git commit -m "feat: chunk-encode the SDXL prompt past CLIP's 77-token limit"
```

---

## Task 4: Record the true final prompt in `queue.py`

**Files:**
- Modify: `src/warlock/queue.py`
- Modify: `tests/conftest.py`
- Modify: `tests/test_queue.py`

**Interfaces:**
- Consumes: `Text2Image.last_prompt` (Task 3).

- [ ] **Step 1: Add `last_prompt` to `FakeText2Image`**

In `tests/conftest.py`, in `FakeText2Image.__init__` (current lines 89-102), add:

```python
        self.last_prompt = ""
```

In `FakeText2Image.generate` (current lines 104-137), set it alongside the existing bookkeeping (right after `self.prompts.append(prompt)`):

```python
        self.prompts.append(prompt)
        self.last_prompt = prompt
```

This keeps `generate()`'s signature untouched (`test_fakes_match_real_signatures.py:34` is unaffected), and reproduces the real `Text2Image.generate()`'s behavior of setting `last_prompt` to the fully-assembled text it actually samples from. Note the fake never applies `PROMPT_TEMPLATE` or the LoRA trigger (it has no diffusers pipeline to format against), so in tests `last_prompt` equals the `composed` string passed in, same as `prompts[0]` today.

- [ ] **Step 2: Change what `queue.py` records**

Replace (current lines 441-442, inside `Worker._generate`):

```python
                params["composed_prompt"] = composed
                await asyncio.to_thread(self.store.set_params, job_id, params)
```

with:

```python
                params["composed_prompt"] = t2i.last_prompt or composed
                await asyncio.to_thread(self.store.set_params, job_id, params)
```

This is read *inside* the existing `try` block, after `t2i.generate(...)` has returned and before the `finally` (exclusive-mode unload) runs, so `t2i.last_prompt` is always populated by the time it's read.

- [ ] **Step 3: Update the pinned test**

In `tests/test_queue.py`, `test_guidance_is_folded_into_the_image_prompt` (current lines 113-130), the assertion at line 130 already reads `worker._text2image.prompts[0]` for `prompt`, which (per Step 1) now equals `worker._text2image.last_prompt` too — no change needed to this specific test, since the fake doesn't apply a template. Add one new test after it that pins the *real* semantics being recorded (that `composed_prompt` comes from `last_prompt`, not the pre-template `composed` string it used to be), using the fake's controllable `last_prompt`:

```python
async def test_composed_prompt_is_read_from_last_prompt_not_recomputed(worker):
    """queue.py must record t2i.last_prompt, not its own local `composed` --
    otherwise the UI's "prompt sent" row would show the pre-trigger,
    pre-PROMPT_TEMPLATE string forever, as it did before this change."""
    job_id = worker.store.create(
        "text", "a barrel", {"seed": 1, "resolution": 512, "genre": "scifi"},
    )
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert (
        worker.store.get(job_id)["params"]["composed_prompt"]
        == worker._text2image.last_prompt
    )
```

- [ ] **Step 4: Run the queue tests**

Run: `uv run pytest tests/test_queue.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/warlock/queue.py tests/conftest.py tests/test_queue.py
git commit -m "fix: record the true sampled prompt, not the pre-template string"
```

---

## Task 5: Generic guidance intake + `/api/prompt-preview` in `app.py`

**Files:**
- Modify: `src/warlock/app.py`
- Test: `tests/test_api.py`

**Interfaces:**
- Consumes: `guidance.form_fields()` (Task 1), `pipelines.prompt.{build, load_tokenizers, count, chunk}` (Task 2), `Text2Image.model_dir` (Task 3).
- Produces: `GET /api/prompt-preview` returning `{"prompt": str, "negative_prompt": str, "tokens": int | None, "chunks": int | None}`.

Verified empirically against the installed fastapi 0.141.1 / starlette 1.3.1: a route can declare `request: Request` alongside `Annotated[..., Form()]` / `Annotated[..., File()]` params, and `await request.form()` inside the handler returns the same parsed form Starlette already cached for the declared params — including blank values (`""`) surviving as `""`, not being dropped. This was confirmed with a throwaway app exercising multipart-with-file, urlencoded-without-file, and an explicit blank value.

- [ ] **Step 1: Add the shared guidance-picking helper**

Add near the top of `create_app()` in `src/warlock/app.py`, just before the `create_job` route (current line ~185, right after the `@app.get("/api/guidance")` handler):

```python
    def _pick_guidance(mapping: Any) -> dict[str, Any]:
        """Every taxonomy field present in ``mapping``, keyed by name.

        Shared between the POST form parser and the GET query-param parser
        for /api/prompt-preview, so a new guidance.py table (Task 1) is
        picked up by both without another app.py edit.
        """
        return {f: mapping.get(f) for f in guidance.form_fields() if f in mapping}

    async def _form_guidance(request: Request) -> dict[str, Any]:
        return _pick_guidance(await request.form())

    def _query_guidance(request: Request) -> dict[str, Any]:
        return _pick_guidance(request.query_params)
```

- [ ] **Step 2: Replace `create_job`'s six taxonomy params with the form helper**

Replace the `create_job` signature (current lines 186-210) — remove the six `Annotated[str | None, Form()]` taxonomy params (`genre`, `art_style`, `category`, `platform`, `base_model`, `style_lora`) and add `request: Request`:

```python
    @app.post("/api/jobs")
    async def create_job(
        request: Request,
        kind: Annotated[str, Form()],
        prompt: Annotated[str | None, Form()] = None,
        seed: Annotated[int, Form()] = 42,
        reference_seed: Annotated[int | None, Form()] = None,
        mesh_seed: Annotated[int | None, Form()] = None,
        resolution: Annotated[int | None, Form()] = None,
        size_m: Annotated[float | None, Form()] = None,
        lora_weight: Annotated[float | None, Form()] = None,
        bg_removal: Annotated[str | None, Form()] = None,
        negative_prompt: Annotated[str | None, Form()] = None,
        rig: Annotated[bool, Form()] = False,
        rig_template: Annotated[str | None, Form()] = None,
        profile: Annotated[str | None, Form()] = None,
        custom_triangles: Annotated[int | None, Form()] = None,
        image: Annotated[UploadFile | None, File()] = None,
        output: Annotated[str, Form()] = "model",
        count: Annotated[int, Form()] = 1,
    ) -> dict[str, Any]:
```

Replace the `guidance.normalize(...)` call (current lines 239-256):

```python
        # Validated up front: a rejected request must not leave an input.png behind.
        try:
            params = guidance.normalize(
                {
                    "genre": genre,
                    "art_style": art_style,
                    "category": category,
                    "platform": platform,
                    "size_m": size_m,
                    "resolution": resolution,
                    "base_model": base_model,
                    "style_lora": style_lora,
                    "lora_weight": lora_weight,
                    "bg_removal": bg_removal,
                    "negative_prompt": negative_prompt,
                }
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
```

with:

```python
        # Validated up front: a rejected request must not leave an input.png behind.
        try:
            params = guidance.normalize(
                {
                    **await _form_guidance(request),
                    "size_m": size_m,
                    "resolution": resolution,
                    "lora_weight": lora_weight,
                    "bg_removal": bg_removal,
                    "negative_prompt": negative_prompt,
                }
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
```

Every other line in `create_job` (validation, the `count` loop, the DB writes) is unchanged — it never referenced the six removed params directly, only `params` after `normalize()`.

- [ ] **Step 3: Add `models` to the app.py imports**

Change (current line 26):

```python
from . import doctor, guidance, rigging
```

to:

```python
from . import doctor, guidance, models, rigging
```

- [ ] **Step 4: Add the `/api/prompt-preview` route**

Add after the `@app.get("/api/guidance")` handler (current lines 180-183), before `_pick_guidance` from Step 1 or after it — either position works since it's a sibling route:

```python
    @app.get("/api/prompt-preview")
    async def prompt_preview(request: Request) -> dict[str, Any]:
        """The composed prompt and its token/chunk cost, before submission.

        Closes docs/NEXT.md's "Prompt preview" item: today the composed
        prompt is only visible after a run. tokens/chunks are best-effort --
        null when transformers isn't installed or the base model's weights
        aren't downloaded, the same degrade-not-fail pattern doctor.py uses.
        """
        from .pipelines import prompt as prompt_pipeline
        from .pipelines.text2image import Text2Image

        raw: dict[str, Any] = dict(_query_guidance(request))
        raw["size_m"] = request.query_params.get("size_m")
        raw["resolution"] = request.query_params.get("resolution")
        raw["lora_weight"] = request.query_params.get("lora_weight")
        raw["bg_removal"] = request.query_params.get("bg_removal")
        if "negative_prompt" in request.query_params:
            raw["negative_prompt"] = request.query_params["negative_prompt"]

        try:
            params = guidance.normalize(raw)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

        style = models.STYLE_LORAS.get(params.get("style_lora") or "")
        trigger = style.trigger if style else ""
        user_prompt = request.query_params.get("prompt") or ""
        positive = prompt_pipeline.build(user_prompt, params, trigger=trigger)

        tokens = chunks = None
        try:
            spec = models.BASE_MODELS[params["base_model"]]
            t2i = Text2Image(
                spec,
                config.t2i_model_root,
                config.t2i_turbo_dir
                if params["base_model"] == models.DEFAULT_BASE_MODEL
                else None,
            )
            tokenizers = prompt_pipeline.load_tokenizers(t2i.model_dir)
            tokens = prompt_pipeline.count(positive, tokenizers)
            chunks = len(prompt_pipeline.chunk(positive, tokenizers))
        except (ImportError, OSError):
            pass  # transformers not installed, or this base model's weights aren't downloaded

        return {
            "prompt": positive,
            "negative_prompt": params["negative_prompt"],
            "tokens": tokens,
            "chunks": chunks,
        }
```

- [ ] **Step 5: Add tests to `tests/test_api.py`**

Add after `test_presets_appear_in_the_catalog`-style tests (near the guidance tests around line 220), or at the end of the guidance-related test block:

```python
def test_new_guidance_fields_accepted_on_post(client):
    r = client.post(
        "/api/jobs",
        data={
            "kind": "text",
            "prompt": "a rifle",
            "material": "steel",
            "condition": "pristine",
            "rarity": "epic",
        },
    )
    assert r.status_code == 200
    params = client.get(f"/api/jobs/{r.json()['id']}").json()["params"]
    assert params["material"] == "steel"
    assert params["condition"] == "pristine"
    assert params["rarity"] == "epic"


def test_unknown_new_guidance_field_value_is_a_400(client):
    r = client.post(
        "/api/jobs", data={"kind": "text", "prompt": "x", "material": "unobtainium"}
    )
    assert r.status_code == 400


def test_prompt_preview_returns_the_composed_prompt(client):
    r = client.get("/api/prompt-preview", params={"prompt": "a barrel", "genre": "scifi"})
    assert r.status_code == 200
    body = r.json()
    assert body["prompt"].startswith("a barrel, ")
    assert "science fiction" in body["prompt"]
    assert body["negative_prompt"]  # falls back to the guidance default


def test_prompt_preview_rejects_unknown_guidance(client):
    r = client.get("/api/prompt-preview", params={"prompt": "x", "material": "unobtainium"})
    assert r.status_code == 400


def test_prompt_preview_degrades_to_null_tokens_when_tokenizer_unavailable(client, monkeypatch):
    from warlock.pipelines import prompt as prompt_pipeline

    def _raise(_model_dir):
        raise OSError("no tokenizer on disk")

    monkeypatch.setattr(prompt_pipeline, "load_tokenizers", _raise)
    r = client.get("/api/prompt-preview", params={"prompt": "a barrel"})
    assert r.status_code == 200
    body = r.json()
    assert body["tokens"] is None
    assert body["chunks"] is None
```

- [ ] **Step 6: Run the API tests**

Run: `uv run pytest tests/test_api.py -v`
Expected: all pass, including the pre-existing `test_guidance_catalog_is_served`, `test_guidance_is_stored_on_the_job`, `test_invalid_guidance_rejected`, and `test_rejected_guidance_leaves_no_input_png_behind`, which exercise the same `create_job` code path this task changed and must be unaffected.

- [ ] **Step 7: Commit**

```bash
git add src/warlock/app.py tests/test_api.py
git commit -m "feat: derive guidance intake from form_fields(); add /api/prompt-preview"
```

---

## Task 6: Frontend — new selects, regrouped fieldsets, live preview

**Files:**
- Modify: `src/warlock/static/index.html`
- Modify: `src/warlock/static/app.js`
- Modify: `tests/test_static_js.py`

**Interfaces:**
- Consumes: `GET /api/guidance` (already returns the 8 new tables automatically via `catalog()`, Task 1), `GET /api/prompt-preview` (Task 5).

- [ ] **Step 1: Regroup the "Design guidance" fieldset into three, add eight rows and a preview line**

Replace the single `<fieldset><legend>Design guidance</legend>...</fieldset>` block (current lines 336-386 of `src/warlock/static/index.html`) with three fieldsets. Also move `#negative-row` (currently lines 325-329, inside the Prompt fieldset) into the new Output fieldset — its id-based JS wiring (`document.getElementById("negative-row").hidden = ...`) is unaffected by where the div physically sits in the DOM.

First, remove `#negative-row` from inside `#text-input` (delete current lines 325-329):

```html
    <fieldset id="text-input">
      <legend>Prompt</legend>
      <div id="preset-row">
        <select id="preset"><option value="">preset…</option></select>
        <select id="history"><option value="">recent…</option></select>
      </div>
      <textarea id="prompt" placeholder="a wooden treasure chest with iron bands, fantasy game prop"></textarea>
      <div id="prompt-preview" class="hint"></div>
    </fieldset>
```

(the new `<div id="prompt-preview">` replaces the removed negative-prompt block in this fieldset).

Then replace the old single "Design guidance" fieldset with:

```html
    <fieldset>
      <legend>Subject</legend>
      <div id="g-category-row">
        <label for="g-category">Asset category</label>
        <select id="g-category" data-guidance="category"></select>
      </div>
      <div id="g-silhouette-row">
        <label for="g-silhouette">Silhouette</label>
        <select id="g-silhouette" data-guidance="silhouette"></select>
      </div>
      <div id="g-material-row">
        <label for="g-material">Material</label>
        <select id="g-material" data-guidance="material"></select>
      </div>
      <div id="g-condition-row">
        <label for="g-condition">Condition</label>
        <select id="g-condition" data-guidance="condition"></select>
      </div>
      <div id="g-rarity-row">
        <label for="g-rarity">Rarity</label>
        <select id="g-rarity" data-guidance="rarity"></select>
      </div>
      <div id="g-emissive-row">
        <label for="g-emissive">Emissive detail</label>
        <select id="g-emissive" data-guidance="emissive"></select>
      </div>
    </fieldset>
    <fieldset>
      <legend>Style</legend>
      <div id="g-setting-row">
        <label for="g-setting">Setting</label>
        <select id="g-setting" data-guidance="setting"></select>
      </div>
      <div id="g-genre-row">
        <label for="g-genre">Genre</label>
        <select id="g-genre" data-guidance="genre"></select>
      </div>
      <div id="g-mood-row">
        <label for="g-mood">Mood</label>
        <select id="g-mood" data-guidance="mood"></select>
      </div>
      <div id="g-art_style-row">
        <label for="g-art_style">Art style</label>
        <select id="g-art_style" data-guidance="art_style"></select>
      </div>
      <div id="g-palette-row">
        <label for="g-palette">Colour palette</label>
        <select id="g-palette" data-guidance="palette"></select>
      </div>
    </fieldset>
    <fieldset>
      <legend>Output</legend>
      <div id="g-base_model-row">
        <label for="g-base_model">Image model</label>
        <select id="g-base_model" data-guidance="base_model"></select>
        <p class="hint">Draws the reference image the 3D stage works from.</p>
      </div>
      <div id="g-style_lora-row">
        <label for="g-style_lora">Style LoRA</label>
        <select id="g-style_lora" data-guidance="style_lora"></select>
        <div class="seed-row">
          <input type="range" id="g-lora-weight" min="0" max="1.5" step="0.05">
          <output id="g-lora-weight-out"></output>
        </div>
        <p class="hint">Weakest on SDXL-Turbo; strongest on the Hyper-SD model.</p>
      </div>
      <div id="g-platform-row">
        <label for="g-platform">Target platform</label>
        <select id="g-platform" data-guidance="platform"></select>
        <p class="hint" id="platform-hint"></p>
      </div>
      <div id="g-profile-row">
        <label for="g-profile">Triangle budget</label>
        <select id="g-profile">
          <option value="raw">Raw reconstruction</option>
        </select>
        <p class="hint">
          Named budgets appear here once gltfpack is vendored and each tier has
          passed qualification (see docs/NEXT.md &sect;3).
        </p>
      </div>
      <div id="g-size-row">
        <label for="g-size">Physical size (metres)</label>
        <input type="number" id="g-size" step="0.01" min="0.01" max="100">
        <p class="hint">The longest axis of the finished model.</p>
      </div>
      <div id="g-bg_removal-row">
        <label for="g-bg_removal">Background removal</label>
        <select id="g-bg_removal" data-guidance="bg_removal"></select>
      </div>
      <div id="negative-row">
        <label for="negative-prompt">Negative prompt</label>
        <textarea id="negative-prompt" rows="2"></textarea>
        <p class="hint">Only applies to CFG models (Playground); distilled 4-step bases ignore it.</p>
      </div>
    </fieldset>
```

Note `#g-size` is now wrapped in `#g-size-row` (it previously had no row wrapper at all); nothing reads `#g-size-row`'s `.hidden`, so this is purely cosmetic grouping.

- [ ] **Step 2: Add the new fields to `GUIDANCE_FIELDS` and the hide list in `app.js`**

Replace (current lines 387-389):

```javascript
const GUIDANCE_FIELDS = [
  "category", "genre", "art_style", "base_model", "style_lora", "platform", "bg_removal",
];
```

with:

```javascript
const GUIDANCE_FIELDS = [
  "category", "genre", "art_style", "base_model", "style_lora", "platform", "bg_removal",
  "material", "condition", "setting", "palette", "emissive", "rarity", "silhouette", "mood",
];
```

Replace the hide-on-image-tab loop (current lines 547-552):

```javascript
    // Genre, art style and the image-model selects only affect the SDXL stage,
    // and image jobs never run SDXL -- hide them rather than offer controls
    // that do nothing.
    for (const field of ["genre", "art_style", "base_model", "style_lora"]) {
      document.getElementById(`g-${field}-row`).hidden = k !== "text";
    }
```

with:

```javascript
    // Genre, art style, the image-model selects and all eight new fields are
    // pure prompt fragments that only affect the SDXL stage, and image jobs
    // never run SDXL -- hide them rather than offer controls that do nothing.
    for (const field of [
      "genre", "art_style", "base_model", "style_lora",
      "material", "condition", "setting", "palette", "emissive", "rarity",
      "silhouette", "mood",
    ]) {
      document.getElementById(`g-${field}-row`).hidden = k !== "text";
    }
```

(`category` and `platform` stay out of this list, unchanged — they drive size and geometry resolution for both job kinds.)

- [ ] **Step 3: Add the debounced prompt preview**

Add near the end of the "design guidance" section of `app.js`, after `copySettingsToForm` (current lines 494-514):

```javascript
// --- prompt preview ----------------------------------------------------
// A live token/chunk count so twelve guidance selects don't feel like a
// black box. Best-effort: /api/prompt-preview returns null tokens/chunks
// when the tokenizer isn't available, and this just shows the prompt text
// alone in that case.

let previewTimer = null;

async function refreshPromptPreview() {
  const el = document.getElementById("prompt-preview");
  if (kind !== "text") { setText(el, ""); return; }
  const text = document.getElementById("prompt").value.trim();
  if (!text) { setText(el, ""); return; }

  const params = new URLSearchParams({ prompt: text });
  for (const field of GUIDANCE_FIELDS) {
    const value = guidanceSelects[field]?.value;
    if (value) params.set(field, value);
  }
  if (sizeInput.value) params.set("size_m", sizeInput.value);
  if (loraWeight.value) params.set("lora_weight", loraWeight.value);
  const negative = document.getElementById("negative-prompt")?.value;
  if (negative) params.set("negative_prompt", negative);

  let body;
  try {
    const r = await fetch(`/api/prompt-preview?${params}`);
    if (!r.ok) return;
    body = await r.json();
  } catch (e) {
    console.error("prompt preview failed", e);
    return;
  }
  const cost = body.tokens != null
    ? `${body.tokens} tokens / ${body.chunks} chunk${body.chunks === 1 ? "" : "s"}`
    : "token count unavailable";
  setText(el, `${body.prompt} — ${cost}`);
}

function schedulePreview() {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(refreshPromptPreview, 400);
}

document.getElementById("form").addEventListener("change", schedulePreview);
document.getElementById("prompt").addEventListener("input", schedulePreview);
```

- [ ] **Step 4: Update `tests/test_static_js.py`'s id check**

The existing hardcoded id list in `test_every_element_the_module_grabs_at_load_time_exists_in_the_html` (current lines 72-81) does not check any `g-<field>` id today (only `g-profile`, which is grabbed via `document.getElementById("g-profile")` inside the submit handler). It is not the right place to add the new guidance ids, since `GUIDANCE_FIELDS`'s selects are found via `document.querySelector('[data-guidance="..."]')` inside `loadGuidance()`, which already runs at module load time (`loadGuidance().catch(...)` at the bottom of the guidance section) — a missing `data-guidance` attribute or row div for a new field would throw inside that promise chain and be swallowed by `.catch()`, silently disabling every feature `loadGuidance()` sets up after the failure point (not "kill the whole page", since it's wrapped in a catch — but silently breaking initialization is still worth a static check). Add a dedicated test instead of extending the unrelated hardcoded list:

```python
def test_every_guidance_field_has_a_matching_select_and_row():
    """loadGuidance() in app.js queries `[data-guidance="<field>"]` for every
    name in GUIDANCE_FIELDS and toggles `#g-<field>-row` on tab switch. A
    field present in one but missing the matching HTML breaks initialization
    silently (loadGuidance's promise chain ends in .catch(console.error)) or
    throws from inside the tab-click handler. Parse app.js's own array so
    this test can't drift from the source of truth.
    """
    app_js = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    match = re.search(r"const GUIDANCE_FIELDS = \[(.*?)\];", app_js, re.DOTALL)
    assert match, "GUIDANCE_FIELDS array not found in app.js"
    fields = re.findall(r'"([\w]+)"', match.group(1))
    assert fields, "GUIDANCE_FIELDS parsed empty"

    for field in fields:
        assert f'data-guidance="{field}"' in html, f"no select for guidance field {field!r}"
        assert f'id="g-{field}-row"' in html, f"no row div for guidance field {field!r}"
```

Add `import re` to the top of `tests/test_static_js.py` alongside the existing `shutil`/`subprocess` imports.

- [ ] **Step 5: Run the static JS tests**

Run: `uv run pytest tests/test_static_js.py -v`
Expected: all pass (the `node --check` parse tests too — confirm `node` is on PATH first with `node --version`; if absent those tests skip, same as before).

- [ ] **Step 6: Commit**

```bash
git add src/warlock/static/index.html src/warlock/static/app.js tests/test_static_js.py
git commit -m "feat: add eight guidance selects, regroup fieldsets, live prompt preview"
```

---

## Task 7: Docs

**Files:**
- Modify: `docs/NEXT.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Remove the now-closed NEXT.md item**

In `docs/NEXT.md`, remove the "Prompt preview" bullet (current lines 42-44):

```markdown
- **Prompt preview**: a server-owned endpoint returning the composed prompt
  before submission, so the form can show what will actually be sent. Today
  the composed prompt is only visible after a run.
```

- [ ] **Step 2: Add a CLAUDE.md invariant note**

In `D:\Projects\Warlock\CLAUDE.md`, add a new paragraph under "## Hard invariants" (after the existing "Grounding is not conditional on a size" paragraph and before "Two different mesh measurements, deliberately not merged" — or any position among the other invariant paragraphs, they are not order-dependent):

```markdown
**The composed SDXL prompt is chunk-encoded, not truncated.** CLIP's text
encoders cap out at 77 tokens, and with twelve optional guidance fields the
composed prompt routinely exceeds that. `pipelines/prompt.chunk()` splits it
into multiple <=77-token chunks on comma boundaries (never mid-phrase), each
encoded separately and concatenated on the sequence axis in
`pipelines/text2image._encode_long_prompt` -- the UNet's cross-attention has
no sequence-length limit, only the text encoders do. A single-chunk prompt
(the common case) is bit-identical to the old direct-string path. Guidance
fragments in `guidance.py` stay 2-4 words each regardless: chunking removes
the hard ceiling, not the soft one where a longer sequence dilutes
cross-attention.
```

- [ ] **Step 3: Commit**

```bash
git add docs/NEXT.md CLAUDE.md
git commit -m "docs: note the chunk-encoding invariant; close the prompt-preview item"
```

---

## Task 8: Full verification pass

**Files:** none (verification only)

- [ ] **Step 1: Full unit test suite**

Run: `uv run pytest -v`
Expected: every test passes or skips for a documented reason (missing `node`, missing local weights) — no new failures relative to the pre-change baseline.

- [ ] **Step 2: Lint**

Run: `uv run ruff check .`
Expected: clean.

- [ ] **Step 3: Manual GPU verification (this machine has CUDA + local sdxl-turbo weights, so this is runnable, not just aspirational)**

1. Start the app (`uv run uvicorn warlock.app:create_app --factory --reload` or the project's usual run command). Confirm `GET /api/guidance` returns 14 fields (12 taxonomy + `base_model` + `style_lora`) and every new select populates in the browser.
2. Set all twelve guidance fields on the Text → 3D tab. Confirm the preview line under the prompt box shows the full composed prompt, a token count above 77, and a chunk count of 2.
3. Generate. Tail the server log; confirm it contains **no** `Token indices sequence length is longer than the specified maximum` warning — that line appearing at all is the bug this plan fixes, so its absence is the acceptance criterion.
4. Regression: pick a short prompt with none of the new fields set, fix the seed, generate once before this branch's changes (or compare against a job generated on `master`) and once after. The two output images must be pixel-identical — this is the "single chunk is bit-identical to today's path" guarantee from Task 3.
5. Switch the image model to Playground (guidance_scale 3.0), set all twelve fields plus a custom negative prompt, and generate. This is the only path that exercises negative-chunk padding (`pad_pair`) and the only one that can shape-mismatch if something in Task 3 is wrong.
6. Confirm the finished job's "prompt sent" row (in the job detail panel) now matches what the preview showed before submission.
7. Apply each of the four shipped presets from the preset dropdown; confirm the new selects populate with the enriched values from Task 1 Step 5.
8. Switch to the Image → 3D tab; confirm all twelve prompt-fragment rows (`genre`, `art_style`, `base_model`, `style_lora`, plus the eight new fields) hide, and that `category` and `platform` stay visible.

- [ ] **Step 4: Report results**

If step 3's manual pass finds a regression (non-identical images in 4, a warning in 3, a shape mismatch in 5), stop and fix before considering this plan complete — these are the acceptance criteria the whole truncation fix exists to satisfy, and nothing in the automated suite can catch a GPU-shape bug.
