"""A recipe: the settings under test, held apart from the prompts.

The suite says *what* to generate; the recipe says *how*. Keeping them
separate is the whole point of the exercise -- an A/B is two recipes over one
suite, and anything that varies between the two is what the comparison
measures.

``job_kwargs`` returns exactly what ``service.jobs.create_job`` takes. The
harness never assembles a params dict by hand and never bypasses create_job's
caps: a benchmark that submitted differently from the app would be measuring
a code path no user can reach.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

RECIPE_DIR = Path(__file__).resolve().parent / "recipes"


@dataclass(frozen=True, slots=True)
class Recipe:
    key: str
    label: str
    guidance: dict[str, Any] = field(default_factory=dict)
    lora_weight: float | None = None
    profile: str | None = None
    custom_triangles: int | None = None
    negative_prompt: str | None = None
    reference_prep: bool | None = None
    path: Path | None = None
    notes: str = ""


def available() -> list[str]:
    return sorted(p.stem for p in RECIPE_DIR.glob("*.json"))


def load(key: str) -> Recipe:
    path = RECIPE_DIR / f"{key}.json"
    if not path.exists():
        raise ValueError(f"unknown recipe {key!r}; available: {available()}")
    return parse(json.loads(path.read_text("utf-8")), path)


def parse(raw: dict[str, Any], path: Path | None = None) -> Recipe:
    """Validate a recipe payload against the live registries.

    Checked here rather than at submit time for the same reason a rig template
    is: a recipe naming a removed LoRA should fail this module's tests, not a
    run's 47th job.
    """
    from .. import guidance

    fields = dict(raw.get("guidance") or {})
    unknown = sorted(set(fields) - set(guidance.form_fields()))
    if unknown:
        name = path.name if path else raw.get("key")
        raise ValueError(f"{name}: recipe names unknown guidance {unknown}")
    # normalize() is the authority on whether the *values* are real, and it
    # raises with the field name and the valid set.
    guidance.normalize(fields)
    return Recipe(
        key=str(raw.get("key") or (path.stem if path else "")),
        label=str(raw.get("label") or ""),
        guidance=fields,
        lora_weight=_opt_float(raw.get("lora_weight")),
        profile=raw.get("profile") or None,
        custom_triangles=_opt_int(raw.get("custom_triangles")),
        negative_prompt=raw.get("negative_prompt"),
        reference_prep=raw.get("reference_prep"),
        path=path,
        notes=str(raw.get("notes") or ""),
    )


def _opt_float(value: Any) -> float | None:
    return None if value in (None, "") else float(value)


def _opt_int(value: Any) -> int | None:
    return None if value in (None, "") else int(value)


def job_kwargs(recipe: Recipe, item: Any, seed: int, *, stage: str = "reference") -> dict[str, Any]:
    """Exactly the kwargs ``service.jobs.create_job`` accepts.

    The item's own guidance wins over the recipe's: the recipe sets the
    backend (which checkpoint, which LoRA, which platform), the item describes
    the subject, and a suite that asks for a stone material must not have it
    overwritten by a recipe that happens to mention one.
    """
    fields = {**recipe.guidance, **item.guidance}
    kwargs: dict[str, Any] = {
        "kind": "text",
        "prompt": item.prompt,
        "output": "reference" if stage == "reference" else "model",
        "seed": int(seed),
        "guidance_fields": fields,
    }
    if recipe.lora_weight is not None and fields.get("style_lora"):
        kwargs["lora_weight"] = recipe.lora_weight
    if recipe.negative_prompt is not None:
        kwargs["negative_prompt"] = recipe.negative_prompt
    if recipe.profile is not None:
        kwargs["profile"] = recipe.profile
        if recipe.custom_triangles is not None:
            kwargs["custom_triangles"] = recipe.custom_triangles
    if recipe.reference_prep is not None:
        kwargs["reference_prep"] = recipe.reference_prep
    return kwargs
