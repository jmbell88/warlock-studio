"""A small instruct model turns words into a Flourish recipe change. One-shot child.

**Why a child, and why one-shot.** The model is a few hundred megabytes of
weights that transformers holds in arenas no ``del`` returns
(``docs/measurements/2026-08-08-load-probe-memory.md``, the rule every helper
model here follows), and a prompt is typed a few times an hour, so the right
trade is the ``loadprobe`` one: load, answer, exit. ``winjob.run`` puts the
child in the kill-on-close job, so a hard kill of the app leaves nothing
behind, and the parent waits with a timeout rather than a pending read.

**The protocol.** One JSON object on stdin -- the whole of it, read to EOF --
and one JSON line on stdout back::

    {"model_dir": ..., "recipe": <keywords.describe_for_model>, "request": "colder"}
    {"kind": "ok", "diff": {...}, "raw": "..."}  |  {"kind": "error", "error": "..."}

The model is asked for **a diff and nothing else**, in the shape
``keywords.DIFF_SCHEMA`` states, and whatever it says is only ever landed
through ``keywords.apply_diff`` on the parent side: every number clamped by
the parameter it names, every unknown name dropped and reported. That funnel,
not the prompt, is what keeps a model's mistake from becoming a stored value
-- the deleted GPT-2 expander's whitelist argument, applied to numbers.

Offline by construction: ``warlock`` sets ``HF_HUB_OFFLINE`` at import, and
``from_pretrained`` reads a directory the user fetched through
``fetch_worker``. CPU always; a helper must not take VRAM from the models
making the asset. Module scope is stdlib-only, ``pipelines``' rule.
"""

from __future__ import annotations

import json
import re
import sys
from typing import Any

MAX_NEW_TOKENS = 400

SYSTEM = (
    "You edit parameters of a 2D game visual effect. You are given the effect as JSON: "
    "its layers (each with a name, a kind, and parameters with current values and allowed "
    "ranges) and its phases. The user asks for a change in plain words. Reply with ONE JSON "
    "object and nothing else -- no prose, no code fence -- of this shape: {schema}. Name "
    "layers by their exact name. Only include what changes. Colours are hex strings like "
    '"#FF8020". Stay inside each parameter\'s range.'
)


def build_messages(recipe_view: dict[str, Any], request: str, schema: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM.format(schema=schema)},
        {
            "role": "user",
            "content": (
                "Effect:\n"
                + json.dumps(recipe_view, separators=(",", ":"))
                + "\n\nChange: "
                + request
            ),
        },
    ]


def extract_json(text: str) -> dict[str, Any] | None:
    """The first balanced ``{...}`` in ``text`` that parses, or None. A model
    that adds a sentence before or after its object is still answered."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        found = json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
                    return found if isinstance(found, dict) else None
        start = text.find("{", start + 1)
    return None


def answer(req: dict[str, Any]) -> dict[str, Any]:
    """Load, generate, extract. Torch-carrying imports inside, deliberately."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_dir = str(req["model_dir"])
    schema = str(req.get("schema") or "{}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.eval()
    messages = build_messages(req["recipe"], str(req.get("request") or ""), schema)
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = "\n\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\n\nassistant:"
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    text = tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    diff = extract_json(text)
    if diff is None:
        return {
            "kind": "error",
            "error": "the model did not answer with a JSON object",
            "raw": text,
        }
    return {"kind": "ok", "diff": diff, "raw": text}


def main(argv: list[str] | None = None) -> int:
    raw = sys.stdin.read()
    try:
        req = json.loads(raw)
        if not isinstance(req, dict):
            raise ValueError("request is not an object")
        result = answer(req)
    except Exception as exc:  # noqa: BLE001 -- the whole point is to report it
        detail = re.sub(r"\s+", " ", str(exc))
        result = {"kind": "error", "error": f"{type(exc).__name__}: {detail}"}
    sys.stdout.write(json.dumps(result) + "\n")
    sys.stdout.flush()
    return 0 if result.get("kind") == "ok" else 1


if __name__ == "__main__":  # pragma: no cover -- the child's entry
    raise SystemExit(main())
