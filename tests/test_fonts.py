"""What the vendored atlas is allowed to contain.

The one thing here is a rule about *collisions*, and it is written down because
nothing else in the suite can see it. `fonts.load` builds each face as an Inter
TTF with `lucide.ttf` merged in, and imgui resolves a codepoint against a
merged font's sources **in order**: the base font wins, and a merged source
only fills gaps. So any codepoint Inter carries is a codepoint Lucide cannot
reach, silently -- a confident, wrongly-shaped glyph at the right size in the
right place.

That is not hypothetical. Inter ships 745 PUA cmap entries (stylistic-set
alternates), Lucide 0.525.0 assigns its icons across U+E038-U+E682, and the two
overlapped on 478 codepoints -- 41 of the 83 constants in `icons.py`. Settings
drew `divide.squared`, Clay's ruler drew `question.squared`, Quit's power
symbol drew `six.squared`. It shipped, because the GL smoke suite runs on
imgui's default atlas where every icon is a missing-glyph box by design.

`scripts/strip_font_pua.py` is the fix and this is the assertion that keeps it:
a font bump that re-introduces the PUA fails here rather than on somebody's
screen.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from warlock.studio import fonts, icons

pytest.importorskip("fontTools", reason="fonttools is a dev-group tool")

FACES = ("Inter-Regular.ttf", "Inter-Medium.ttf", "Inter-SemiBold.ttf")


def _cmap(name: str) -> dict[int, str]:
    from fontTools.ttLib import TTFont

    return TTFont(str(fonts.FONT_DIR / name)).getBestCmap()


def _icon_codepoints() -> dict[str, int]:
    out: dict[str, int] = {}
    for attr in dir(icons):
        if not attr.isupper():
            continue
        value = getattr(icons, attr)
        if isinstance(value, str) and len(value) == 1:
            out[attr] = ord(value)
    return out


def test_the_icon_constants_are_all_in_the_vendored_lucide():
    """The other direction, and the cheaper failure: a stale constant renders
    as a missing-glyph box, which is at least visible."""
    cmap = _cmap("lucide.ttf")
    missing = {n: cp for n, cp in _icon_codepoints().items() if cp not in cmap}
    assert not missing, f"codepoints absent from lucide.ttf: {missing}"


@pytest.mark.parametrize("face", FACES)
def test_no_inter_face_shadows_an_icon(face: str):
    cmap = _cmap(face)
    shadowed = {
        name: f"U+{cp:04X} would draw {cmap[cp]}"
        for name, cp in _icon_codepoints().items()
        if cp in cmap
    }
    assert not shadowed, (
        f"{face} carries codepoints the icon font needs; re-run "
        f"scripts/strip_font_pua.py: {shadowed}"
    )


@pytest.mark.parametrize("face", FACES)
def test_no_inter_face_carries_a_private_use_codepoint_at_all(face: str):
    """Wider than the test above on purpose.

    Shadowing only *one* of today's 83 constants is what the app is broken by,
    but the icon set grows, and a collision that arrives with a new icon would
    be attributed to the icon rather than to the font. The PUA entries are
    alternates reachable through OpenType features imgui does not run, so
    nothing loses anything by their absence.
    """
    live = [cp for cp in _cmap(face) if 0xE000 <= cp <= 0xF8FF]
    assert not live, f"{face} has {len(live)} PUA codepoints; run scripts/strip_font_pua.py"


def test_every_vendored_face_is_present():
    for name in (*FACES, "lucide.ttf"):
        assert (Path(fonts.FONT_DIR) / name).is_file()
