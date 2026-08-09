"""Remove Private-Use-Area codepoints from the vendored Inter faces.

Why this exists, because it is not obvious and it cost half the icon set:

`fonts.load` builds each face as Inter with `lucide.ttf` merged in, and imgui's
merge resolves a codepoint against its sources **in order** -- the base font
wins and a merged source only *fills gaps*. Inter ships 745 PUA cmap entries
(stylistic-set alternates: `six.squared`, `question.squared`, `less.circled`,
`Gdotaccent.1.ss07` ...), and Lucide 0.525.0 assigns its icons across
U+E038-U+E682. The two overlap on 478 codepoints, of which **41 of the 83
constants in `icons.py`** were live: Settings drew Inter's `divide.squared`,
Clay's ruler drew `question.squared`, Quit's power symbol drew `six.squared`,
and Save, Search, Plus, Undo, Redo, Play, Check, Pencil, Eraser, Brush and
Palette were all drawing punctuation and digit variants.

Nothing catches this below a rendered frame. The GL smoke suite runs on imgui's
default atlas, where every icon is a missing-glyph box by design and its
docstring says so; and the shadowing is *silent* -- a wrong glyph is drawn
confidently at the right size in the right place.

The fix is at the font rather than at the loader because imgui 1.92's Python
binding exposes no glyph-exclusion on `ImFontConfig`, and the two alternatives
are worse: making lucide the base source would take the line metrics from a
font with ascent 1000 / descent 0, and remapping lucide's codepoints would
break the one rule `icons.py` states -- that its numbers come from a named
release's `info.json` and are never guessed.

The entries removed are alternates reachable through OpenType features, which
imgui does not run; nothing can address them by codepoint except by accident,
which is precisely what was happening. Glyphs and GSUB are untouched -- only
the cmap loses the PUA range -- so no text in the app changes.

Inter is OFL 1.1 and declares no Reserved Font Name, so a modified copy may
keep the name.

Idempotent. Run over the vendored faces after any font bump:

    uv run --with fonttools python scripts/strip_font_pua.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

FONT_DIR = Path(__file__).resolve().parent.parent / "src/warlock/studio/resources/fonts"
FACES = ("Inter-Regular.ttf", "Inter-Medium.ttf", "Inter-SemiBold.ttf")

# The Basic Multilingual Plane's private use area. The supplementary planes'
# (U+F0000+, U+100000+) are out of imgui's 16-bit ImWchar reach anyway, and
# Inter uses none of them.
PUA = (0xE000, 0xF8FF)


def strip(path: Path) -> int:
    """Drop PUA codepoints from every cmap subtable. -> how many went."""
    from fontTools.ttLib import TTFont

    font = TTFont(str(path))
    removed: set[int] = set()
    for table in font["cmap"].tables:
        doomed = [cp for cp in table.cmap if PUA[0] <= cp <= PUA[1]]
        for cp in doomed:
            del table.cmap[cp]
        removed.update(doomed)
    if removed:
        font.save(str(path))
    return len(removed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--font-dir", type=Path, default=FONT_DIR)
    args = ap.parse_args()
    for name in FACES:
        path = args.font_dir / name
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            return 2
        print(f"{name}: removed {strip(path)} PUA codepoints")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
