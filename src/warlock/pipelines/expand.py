"""The local prompt expander: a short prompt -> a dense descriptive one.

The one documented mechanism behind the hosted image services' quality gap
(DALL-E 3's "Improving Image Generation with Better Captions"; Google's
recommended dual-model chain for its image API): put a language model between
the user and the pixel model, and let it supply the descriptive density a
two-word prompt lacks. This is the offline form of it -- the Fooocus GPT-2
expander (~124M parameters), which appends comma-separated aesthetic phrases
drawn from a fixed whitelist (``positive.txt``, shipped beside the weights).

CPU always, and cheap there: one short greedy-ish sampling pass of a 124M
model. It runs on the worker beside a resident SDXL pipe and a resident
trellis-server, and a helper must not take VRAM from the models making the
asset -- the same rule the DINOv2 anchor follows.

Torch-carrying, so nothing UI-adjacent may import it at module scope; the
worker and the preview both reach it through function-scope imports. Module
scope here is stdlib-only, matching the rest of ``pipelines/``.

Two deliberate behaviours, both measured elsewhere before they were rules:

* **A detailed prompt is not expanded.** Mandatory rewriting measurably hurts
  prompts that already carry their own description (arXiv 2407.14333), so
  ``should_expand`` gates on the prompt's own token count and the caller keeps
  the original text when it says no.
* **The whitelist is the safety.** Generation is constrained to the tokens of
  ``positive.txt`` (plus the comma), so the expander can only append aesthetic
  vocabulary -- it cannot rewrite the subject, contradict the negative prompt,
  or leak the model's unconstrained completions into a stored param.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Prompts already carrying this many GPT-2 tokens are left alone -- they are a
# description, and appending tag soup to a paragraph dilutes it (2407.14333).
DETAILED_PROMPT_TOKENS = 40

# How much the expander may append. 48 tokens is roughly a dozen short
# phrases -- about what PROMPT_TEMPLATE itself adds -- and keeps the whole
# composed prompt inside two CLIP chunks for typical subjects.
MAX_NEW_TOKENS = 48

# Fooocus's own sampling settings: top-k keeps the tail of the whitelist
# reachable without letting the distribution wander.
TOP_K = 100

# The whitelist file the model card ships beside the weights.
POSITIVE_WORDS = "positive.txt"

# (tokenizer, model, allowed-token-ids) per model directory. Process-local for
# fetch.store_generation's reason not to matter here: the expander directory is
# immutable once fetched, and a re-download lands under the same pin.
_cache: dict[Path, tuple[Any, Any, Any]] = {}


def clean(text: str) -> str:
    """Whatever the model emitted -> one tidy comma-joined phrase list.

    Pure and total: collapse whitespace, drop empty phrases, dedupe while
    keeping first occurrence (sampling under a whitelist repeats itself), and
    strip stray punctuation the token soup leaves at the edges.
    """
    phrases: list[str] = []
    seen: set[str] = set()
    for piece in text.replace("\n", ",").split(","):
        word = re.sub(r"\s+", " ", piece).strip(" .;:-")
        key = word.lower()
        if word and key not in seen:
            seen.add(key)
            phrases.append(word)
    return ", ".join(phrases)


def _allowed_ids(tokenizer: Any, words: list[str]) -> list[int]:
    """Vocabulary ids the sampler may emit: the whitelist plus the comma.

    GPT-2's BPE marks a leading space with ``Ġ``, so each whitelisted word is
    matched with and without it -- mid-phrase and phrase-initial forms are
    different tokens. Multi-token words contribute the tokens of their pieces,
    which is looser than exact-word matching but keeps every whitelisted word
    actually reachable.
    """
    wanted = {w.strip().lower() for w in words if w.strip()}
    allowed: set[int] = set()
    for token, idx in tokenizer.get_vocab().items():
        if token.lstrip("Ġ").lower() in wanted:
            allowed.add(idx)
    for word in wanted:
        for form in (word, " " + word):
            allowed.update(tokenizer(form, add_special_tokens=False)["input_ids"])
    allowed.update(tokenizer(",", add_special_tokens=False)["input_ids"])
    if tokenizer.eos_token_id is not None:
        allowed.add(int(tokenizer.eos_token_id))
    return sorted(allowed)


def _load(model_dir: Path) -> tuple[Any, Any, Any]:
    """Tokenizer, model (eval, CPU, fp32) and the whitelist ids, cached."""
    cached = _cache.get(model_dir)
    if cached is not None:
        return cached
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True)
    model.eval()
    words = (model_dir / POSITIVE_WORDS).read_text(encoding="utf-8").splitlines()
    loaded = (tokenizer, model, _allowed_ids(tokenizer, words))
    _cache[model_dir] = loaded
    return loaded


def should_expand(prompt: str, model_dir: Path) -> bool:
    """Whether this prompt is short enough to benefit from expansion."""
    tokenizer, _model, _allowed = _load(model_dir)
    return len(tokenizer(prompt)["input_ids"]) <= DETAILED_PROMPT_TOKENS


def expand(prompt: str, model_dir: Path, *, seed: int) -> str | None:
    """``prompt`` with the expander's phrases appended, or None to keep it.

    None -- never the original string -- when the gate says a prompt is too
    detailed to touch, so the caller can tell "expanded" from "left alone" and
    record only the former. Deterministic in ``seed``: the sampling runs under
    a forked CPU RNG, so an expansion re-run at the job's recorded seed
    reproduces (given the same weights) and nothing else's randomness moves.
    """
    import torch

    tokenizer, model, allowed = _load(model_dir)
    if len(tokenizer(prompt)["input_ids"]) > DETAILED_PROMPT_TOKENS:
        return None

    from transformers import LogitsProcessor, LogitsProcessorList

    class _Whitelist(LogitsProcessor):
        def __init__(self) -> None:
            self.mask: torch.Tensor | None = None

        def __call__(self, input_ids: Any, scores: Any) -> Any:
            if self.mask is None:
                mask = torch.full_like(scores[0], float("-inf"))
                mask[allowed] = 0.0
                self.mask = mask
            return scores + self.mask

    text = prompt.strip().rstrip(",.") + ","
    inputs = tokenizer(text, return_tensors="pt")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed & 0x7FFFFFFF)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                top_k=TOP_K,
                logits_processor=LogitsProcessorList([_Whitelist()]),
                pad_token_id=tokenizer.eos_token_id,
            )
    appended = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    extra = clean(appended)
    if not extra:
        return None
    return f"{prompt.strip().rstrip(',.')}, {extra}"
