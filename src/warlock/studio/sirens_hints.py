"""What the grid says without being asked: one line of the keyboard in hand.

``clay_hints``' hint line, for the surface that needs it more. A tracker grid is
five columns of dots and **the column the caret is in decides what a key means**
-- ``c`` is a note in the first column, the hex digit twelve in the third and
fifth, and nothing at all in the fourth. That is the whole of what confuses
somebody opening Sirens for the first time, and until this line there was
nowhere on screen it was said: a wrong-column keystroke is a silent no-op, so
the app's answer to "why did nothing happen" was nothing at all.

Pure -- strings and nothing else, no imgui and no ``service`` -- which is what
lets ``tests/test_sirens_hints.py`` ask the question a screenshot cannot: does
every key this line names exist. A hint naming a binding nothing implements is
worse than no hint, because it is read as a promise.

``panes/sirens_patterns.py`` draws it, under the grid rather than over it.
"""

from __future__ import annotations

from .sirens import synth

#: The two piano rows, spelled as the reader sees them on the keyboard rather
#: than derived from ``sirens_keys.PIANO_KEYS``: that module reaches
#: ``sirens_mode`` and importing it here would cost this one its purity. The
#: parity test asserts every letter of both runs is a key that module maps, so
#: the copy cannot drift into a lie.
LOWER_ROW = "zsxdcvgbhnjm"
UPPER_ROW = "q2w3er5t6y7u"

#: The effect letters, from the engine's own table rather than a second list of
#: them. Sorted by effect id, which is the order the manual's table is in.
EFFECT_LETTERS: tuple[str, ...] = tuple(
    letter for _fx, (letter, _what) in sorted(synth.EFFECT_NAMES.items())
)


def _effects() -> str:
    return " ".join(EFFECT_LETTERS)


#: What each column's keyboard is, in the document's column order -- the order
#: ``sirens_state.COLUMN_LABELS`` is in, and indexed the same way, because the
#: label and the line under it are two halves of one sentence.
#:
#: Built once at import: the strings never change, and this is read every frame.
_COLUMNS: tuple[str, ...] = (
    f"{LOWER_ROW} and {UPPER_ROW} are the piano . + / - octave"
    " . ` note-off, Shift+` release",
    "two hex digits, the instrument number . Ctrl+Up / Ctrl+Down pick it",
    "one hex digit . F loudest, 0 silent",
    f"one effect letter: {_effects()}",
    "two hex digits, what the effect on its left takes",
)

#: Always true and always last: the keys that mean the same thing in all five
#: columns. ``clay_hints._NAVIGATE``'s job -- the keys a newcomer asks about
#: first and the ones a manual is least likely to be open at.
_ALWAYS = (
    "Arrows move . Shift+Arrows select . Space play . Delete clear"
    " . Insert / Shift+Delete shift rows"
)

#: What a live block selection adds, and only then: four chords that do nothing
#: without one, offered in every column would be four promises the app refuses.
_SELECTION = "Ctrl+C / Ctrl+X / Ctrl+V block . Ctrl+G interpolate"


def hint(column: int, *, has_selection: bool = False) -> str:
    """One line of what the keyboard does in the column the caret is in.

    ``column`` is a document column index; anything outside the five is
    clamped rather than raising, because this is a readout drawn every frame
    and a caret briefly out of range must cost a wrong line, never the frame.
    """

    index = max(0, min(int(column), len(_COLUMNS) - 1))
    parts = [_COLUMNS[index]]
    if has_selection:
        parts.append(_SELECTION)
    parts.append(_ALWAYS)
    return " . ".join(parts)


def keys_named(text: str) -> set[str]:
    """Every key or chord the line mentions, for the parity test.

    ``clay_hints.keys_named``, and deliberately the same crude rules: a token
    is a key if it is a capital letter, a chord with a ``+``, or one of the
    named keys. **A single letter counts only when it is capital** -- the piano
    rows above are lowercase precisely because they are a run of letters to
    read rather than a list of bindings, and a parser that took them for keys
    would demand this mode implement ``a``.

    ``+`` is itself a binding here (it is the octave up), which is why the
    chord rule catches it and why the parity test expects to see it.
    """

    words = {
        token.strip(".,")
        for chunk in text.split(" . ")
        for token in chunk.split()
    }
    named = {
        "Arrows", "Space", "Delete", "Insert", "Enter", "Esc", "Tab",
        "Home", "End", "PageUp", "PageDown",
    }
    out = set()
    for word in words:
        if word in named or "+" in word or (len(word) == 1 and word.isupper()):
            out.add(word)
    return out
