"""The studio side of ``findings.json`` -- a pure stdlib reader with no
import of torch or imgui, so a generate pane can call it every frame.

``load`` is the whole cost model: one ``stat()`` per call, and a re-read only
when the file's mtime moves. A missing file is cached as a miss too, keyed on
the path alone (there is no mtime to key on), so a bench dir that has never
been reported costs one failed ``stat()`` per frame forever, not an
exception path. ``hint`` and ``comparison_lines`` are pure lookups against
the loaded doc -- ``service/findings.py`` owns the schema (``params``
marginals keyed by ``str(value)``, wilson-ranked ``vectors``, matched-pair
``comparisons``) and this reader tolerates every older shape it ever wrote.

Strings here render through imgui's Inter atlas, which is baked with the
default Basic-Latin + Latin-1 glyph range: the middle dot (U+00B7) is safe,
but anything past U+00FF (a real ``>=`` sign, a Greek delta) would come out
as the missing-glyph box -- hence "41%+" and "worst-hole" rather than the
typographically nicer spellings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# {path: (mtime_or_None, doc_or_None)} -- mtime is None for a cached miss
# (file absent or unreadable), which can never collide with a real mtime.
_CACHE: dict[Path, tuple[float | None, dict[str, Any] | None]] = {}


def load(path: Path) -> dict[str, Any] | None:
    """The parsed ``findings.json`` at ``path``, or ``None`` if it is absent
    or unreadable. Cached by mtime: unchanged since the last call returns the
    same object with no re-read."""
    try:
        mtime = path.stat().st_mtime
    except OSError:
        _CACHE[path] = (None, None)
        return None

    cached = _CACHE.get(path)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    try:
        import json

        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # Cached by *this* mtime, not None: stat() succeeds for a
        # corrupt-but-present file, so caching (None, None) here would never
        # match a real mtime and every call would re-read and re-parse the
        # same broken file, forever -- exactly the per-frame cost the mtime
        # gate exists to avoid.
        _CACHE[path] = (mtime, None)
        return None

    _CACHE[path] = (mtime, doc)
    return doc


def hint(doc: dict[str, Any] | None, param: str, value: Any, *, min_n: int = 5) -> str | None:
    """``"accept 6/8 (41%+)"`` for ``param``/``value`` in ``doc``, or the
    machine-evidence fallback, or ``None`` when neither has ``min_n`` behind
    it -- a thin bucket is noise, not a finding.

    Three tiers. Enough verdicts: the count plus the writer's Wilson lower
    bound as a floor of confidence ("41%+"); a v1 file that never computed
    the bound still hints as the bare count. Too few verdicts but enough
    *observations*: what the worker measured -- ``"holes 3% · watertight 71%
    (21 meshes)"`` -- so a value generated with often but never reviewed still
    says something true. Otherwise nothing.

    ``findings.json`` keys a bucket by ``str()`` of the JSON value the sweep
    recorded (e.g. ``"0.6"``), but an imgui float32 slider hands back
    ``0.6000000238418579`` -- the float32 rounding of ``0.6``, not the
    ``float`` Python would make from the string. A straight ``str(value)``
    lookup misses every such slider, silently, forever. So a float value that
    misses the literal-string lookup gets a second pass: every bucket key
    that itself parses as a float is compared to ``value`` rounded to 6
    decimals, which absorbs float32's error (~1e-7 relative) while still
    telling "0.6" apart from "0.65"."""
    if doc is None:
        return None
    bucket = (doc.get("params") or {}).get(param) or {}
    entry = bucket.get(str(value))
    if entry is None and isinstance(value, float):
        target = round(value, 6)
        for key, candidate in bucket.items():
            try:
                key_value = float(key)
            except ValueError:
                continue
            if round(key_value, 6) == target:
                entry = candidate
                break
    if entry is None:
        return None
    if entry.get("n", 0) >= min_n:
        base = f"accept {entry.get('accepts', 0)}/{entry['n']}"
        bound = entry.get("wilson_low")
        if isinstance(bound, (int, float)) and not isinstance(bound, bool):
            base += f" ({round(bound * 100)}%+)"
        return base
    return _metrics_hint(entry.get("metrics"), min_n)


def _reading(metrics: dict[str, Any], key: str, min_n: int) -> tuple[float, int] | None:
    """``(value, how many observations carried it)``, or None below ``min_n``.

    The count is the metric's own, not the bucket's. The audit can succeed
    while ``meshreport.build`` fails or returns ``status: "invalid"``, so a
    bucket of twenty-one observations can hold exactly one ``watertight``
    reading -- which the bucket count would have advertised as twenty-one.
    Falling back to ``n`` keeps a doc written before ``counts`` existed
    readable, where that assumption was at least the one being made.
    """
    value = metrics.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    counts = metrics.get("counts")
    count = counts.get(key) if isinstance(counts, dict) else None
    if not isinstance(count, int) or isinstance(count, bool):
        count = metrics.get("n")
    if not isinstance(count, int) or isinstance(count, bool) or count < min_n:
        return None
    return float(value), count


def _measured(parts: list[tuple[str, int]]) -> str:
    """Join measurements, saying how many meshes each rests on.

    One trailing count while they agree -- which is the ordinary case, since
    both measurements usually succeed together -- and a count per measurement
    the moment they do not, because a single number after two metrics of
    different weight is a claim about whichever one the reader assumes.

    *meshes*, not *runs*, and that word is doing work. These lines appear
    under 2D prompt controls as well as mesh ones, because an observation
    credits every param in its vector exactly as a verdict does -- so
    "art_style toon" carries the hole fraction of the meshes reconstructed
    from toon references. That is a true and useful statement, but only if the
    reader can tell it is about geometry rather than about the picture.
    """
    if len({count for _text, count in parts}) == 1:
        return f"{' · '.join(text for text, _ in parts)} ({parts[0][1]} meshes)"
    return " · ".join(f"{text} ({count} meshes)" for text, count in parts)


def _metrics_hint(metrics: Any, min_n: int) -> str | None:
    """The machine-evidence tier, defensively: the file crossed a disk.

    Each measurement earns its place separately: ``min_n`` is a threshold on
    the readings behind *that number*, so a thin one is dropped rather than
    carried along by a fat one in the same bucket.
    """
    if not isinstance(metrics, dict):
        return None
    parts: list[tuple[str, int]] = []
    hole = _reading(metrics, "hole_worst", min_n)
    if hole is not None:
        parts.append((f"holes {round(hole[0] * 100)}%", hole[1]))
    watertight = _reading(metrics, "watertight_rate", min_n)
    if watertight is not None:
        parts.append((f"watertight {round(watertight[0] * 100)}%", watertight[1]))
    if not parts:
        return None
    return _measured(parts)


def metrics_line(metrics: Any) -> str | None:
    """A recipe's machine evidence on one muted line -- ``"holes 3% worst ·
    watertight 71% · 24,120 tri (21 meshes)"`` -- or ``None`` when nothing was
    measured. Unlike the hint tier this has no threshold: it annotates a
    recipe that already earned its place with verdicts."""
    if not isinstance(metrics, dict):
        return None
    parts: list[tuple[str, int]] = []
    hole = _reading(metrics, "hole_worst", 1)
    if hole is not None:
        parts.append((f"holes {round(hole[0] * 100)}% worst", hole[1]))
    watertight = _reading(metrics, "watertight_rate", 1)
    if watertight is not None:
        parts.append((f"watertight {round(watertight[0] * 100)}%", watertight[1]))
    triangles = _reading(metrics, "triangles", 1)
    if triangles is not None:
        parts.append((f"{int(triangles[0]):,} tri", triangles[1]))
    if not parts:
        return None
    return _measured(parts)


def vector_line(entry: Any) -> str:
    """A ranked recipe's headline -- ``"80% of 20 (61%+)"``.

    The bound is *omitted* rather than defaulted to zero when the file does not
    carry one. A v1 findings.json never computed it and ranked by the raw rate,
    so substituting 0.0 labelled every recipe in an existing user's Review pane
    "(0%+)" -- a confidence floor of nothing, under a heading claiming the
    Wilson ranking. ``hint`` already drops the suffix on the same file for the
    same reason; this is that rule applied where the ranking is displayed.

    Pure, so the wording is assertable without a GL context.
    """
    if not isinstance(entry, dict):
        return ""
    n = _int_count(entry.get("n"))
    rate = entry.get("accept_rate")
    ok = isinstance(rate, (int, float)) and not isinstance(rate, bool)
    percent = round(float(rate) * 100) if ok else 0
    line = f"{percent}% of {n}"
    bound = entry.get("wilson_low")
    if isinstance(bound, (int, float)) and not isinstance(bound, bool):
        line += f" ({round(bound * 100)}%+)"
    return line


# What a delta line calls each metric, and how its mean is rendered. Holes and
# the two rates are differences of fractions (percent, one decimal, signed);
# triangles is a count.
_DELTA_LABELS = {
    "hole_worst": "worst-hole",
    "hole_mean": "mean-hole",
    "watertight": "watertight",
    "ready": "ready",
}


def comparison_lines(doc: dict[str, Any] | None, *, min_pairs: int = 5) -> list[str]:
    """The axis verdicts, as printable lines, winner first.

    One header line per contrast with at least ``min_pairs`` matched human
    pairs -- ``"lora_weight: 0.6 beat 0.9 in 7 of 8 matched pairs (2 sweeps,
    2 prompts)"`` -- followed by an indented delta line per machine metric
    with ``min_pairs`` paired runs behind it. A contrast with machine pairs
    but too few human ones still shows its deltas under a "machine evidence
    only" header: a sweep says something the moment it finishes."""
    if not isinstance(doc, dict):
        return []
    comparisons = doc.get("comparisons")
    if not isinstance(comparisons, dict):
        return []
    lines: list[str] = []
    for param in sorted(comparisons):
        entries = comparisons[param]
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            lines.extend(_entry_lines(param, entry, min_pairs))
    return lines


def _int_count(value: Any) -> int:
    """A count from a doc that crossed a disk: anything that is not an int
    (or is the bool impostor) is zero, never a raise on the frame thread."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _entry_lines(param: str, entry: dict[str, Any], min_pairs: int) -> list[str]:
    pairs = _int_count(entry.get("pairs"))
    a, b = str(entry.get("a", "?")), str(entry.get("b", "?"))
    a_wins = _int_count(entry.get("a_wins"))
    b_wins = _int_count(entry.get("b_wins"))

    raw_deltas: list[tuple[str, float, int]] = []
    deltas_src = entry.get("deltas")
    if isinstance(deltas_src, dict):
        for metric, stat in sorted(deltas_src.items()):
            if not isinstance(stat, dict):
                continue
            delta_pairs = _int_count(stat.get("pairs"))
            mean = stat.get("mean")
            if delta_pairs < min_pairs:
                continue
            if not isinstance(mean, (int, float)) or isinstance(mean, bool):
                continue
            raw_deltas.append((str(metric), float(mean), delta_pairs))

    def delta_lines(flip: bool) -> list[str]:
        # The stored mean is a-minus-b (a = lexicographically low value), but
        # these lines sit unattributed under a header that leads with the
        # *winner* -- so when b won, the sign must be re-oriented to
        # winner-minus-loser or the same outcome reads backwards depending on
        # which value happens to sort first.
        lines = []
        for metric, mean, delta_pairs in raw_deltas:
            shown_mean = -mean if flip else mean
            shown = (
                f"{shown_mean:+,.0f}" if metric == "triangles" else f"{shown_mean * 100:+.1f}%"
            )
            label = _DELTA_LABELS.get(metric, metric)
            lines.append(f"    {label} {shown} over {delta_pairs} paired runs")
        return lines

    def plural(count: int, noun: str) -> str:
        return f"{count} {noun}{'' if count == 1 else 's'}"

    breadth = (
        f"({plural(_int_count(entry.get('sweeps')), 'sweep')},"
        f" {plural(_int_count(entry.get('prompts')), 'prompt')})"
    )
    if pairs >= min_pairs:
        if a_wins == b_wins:
            header = f"{param}: {a} vs {b} - tied over {pairs} matched pairs {breadth}"
            return [header, *delta_lines(False)]
        winner, loser = (a, b) if a_wins > b_wins else (b, a)
        header = (
            f"{param}: {winner} beat {loser} in {max(a_wins, b_wins)} of {pairs}"
            f" matched pairs {breadth}"
        )
        return [header, *delta_lines(b_wins > a_wins)]
    if raw_deltas:
        # The "a vs b" header names a first, so the stored orientation stands.
        return [f"{param}: {a} vs {b} - machine evidence only", *delta_lines(False)]
    return []
