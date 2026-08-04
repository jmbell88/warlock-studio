"""Loading and validating a benchmark suite. Pure; no torch, no GPU, no DB.

A suite is a file, not code -- the same rule ``templates/*.json`` follows.
Adding a suite means adding a file, and a suite file is *never* edited in
place: a run's manifest records its fingerprint, so changing a prompt after
the fact would silently invalidate every comparison made against it. Change
means core-v2.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUITE_DIR = Path(__file__).resolve().parent / "suites"
DEFAULT_SUITE = "core-v1"

# The categories a suite may use. Enforced at load: ``--filter`` matches on
# this string, so a typo'd category in a suite file selects zero items and
# reports nothing wrong -- a run that silently does nothing, which is the one
# failure mode worse than a crash.
CATEGORIES = ("prop", "weapon", "character", "vehicle", "environment")


@dataclass(frozen=True, slots=True)
class Item:
    id: str
    category: str
    prompt: str
    guidance: dict[str, Any]
    tags: tuple[str, ...] = ()
    notes: str = ""


@dataclass(frozen=True, slots=True)
class Suite:
    key: str
    label: str
    seeds: tuple[int, ...]
    items: tuple[Item, ...]
    path: Path
    notes: str = ""

    def by_category(self) -> dict[str, list[Item]]:
        out: dict[str, list[Item]] = {}
        for item in self.items:
            out.setdefault(item.category, []).append(item)
        return out

    def filter(self, *, categories: tuple[str, ...] = (), ids: tuple[str, ...] = ()) -> list[Item]:
        return [
            i
            for i in self.items
            if (not categories or i.category in categories) and (not ids or i.id in ids)
        ]


def available() -> list[str]:
    return sorted(p.stem for p in SUITE_DIR.glob("*.json"))


def load(key: str = DEFAULT_SUITE) -> Suite:
    """Load and validate a suite by key, or raise ValueError."""
    path = SUITE_DIR / f"{key}.json"
    if not path.exists():
        raise ValueError(f"unknown suite {key!r}; available: {available()}")
    raw = json.loads(path.read_text("utf-8"))
    return parse(raw, path)


def parse(raw: dict[str, Any], path: Path) -> Suite:
    """Validate a suite payload. Every guidance key must be one guidance.py
    actually accepts -- a typo would otherwise fail 160 submits one at a time,
    two hours into a run."""
    from .. import guidance

    known = set(guidance.form_fields())
    seeds = tuple(int(s) for s in raw.get("seeds") or ())
    if not seeds:
        raise ValueError(f"{path.name}: a suite needs at least one seed")

    items: list[Item] = []
    seen: set[str] = set()
    for entry in raw.get("items") or ():
        item_id = str(entry.get("id") or "")
        if not item_id:
            raise ValueError(f"{path.name}: an item has no id")
        if item_id in seen:
            raise ValueError(f"{path.name}: duplicate item id {item_id!r}")
        seen.add(item_id)
        if not str(entry.get("prompt") or "").strip():
            raise ValueError(f"{path.name}: item {item_id} has no prompt")
        category = str(entry.get("category") or "")
        if category not in CATEGORIES:
            raise ValueError(
                f"{path.name}: item {item_id} has category {category!r}; "
                f"expected one of {list(CATEGORIES)}"
            )
        fields = dict(entry.get("guidance") or {})
        unknown = sorted(set(fields) - known)
        if unknown:
            raise ValueError(f"{path.name}: item {item_id} names unknown guidance {unknown}")
        # And the *values*, the same way recipe.parse does. Checking only the
        # keys is what this docstring's promise could not keep: "material":
        # "stonee" names a real field, loads without complaint, and fails at
        # submit -- one item at a time, two hours into a run.
        try:
            guidance.normalize(fields)
        except ValueError as exc:
            raise ValueError(f"{path.name}: item {item_id}: {exc}") from exc
        items.append(
            Item(
                id=item_id,
                category=category,
                prompt=str(entry["prompt"]),
                guidance=fields,
                tags=tuple(str(t) for t in entry.get("tags") or ()),
                notes=str(entry.get("notes") or ""),
            )
        )
    if not items:
        raise ValueError(f"{path.name}: a suite needs at least one item")
    return Suite(
        key=str(raw.get("key") or path.stem),
        label=str(raw.get("label") or path.stem),
        seeds=seeds,
        items=tuple(items),
        path=path,
        notes=str(raw.get("notes") or ""),
    )
