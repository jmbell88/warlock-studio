"""The refusal ledger, machine-checked against the code that does the refusing.

``docs/PLOTTER_COMPAT.md`` is a promise about what Plotter does with a Tiled
file, and a promise in a markdown table drifts the moment somebody adds a
refusal without opening it. So the table is not prose: the ``refused`` rows
are checked, both ways, against every ``TiledUnsupported`` raised in the
engine.

Both ways is the part that matters. A missing row means a refusal the docs do
not admit to; a stale row means a feature the docs still claim we refuse after
somebody taught us to load it -- and that second one is exactly what happens
during a parity milestone, which is when this file earns its keep.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

MATRIX = Path(__file__).resolve().parents[2] / "docs" / "PLOTTER_COMPAT.md"
ENGINE = Path(__file__).resolve().parents[2] / "src" / "warlock" / "studio" / "plotter"

STATES = {"round-trips", "refused", "preserved-verbatim"}


def _normal_form(node: ast.expr) -> str | None:
    """A refusal's feature argument as one comparable string.

    A literal is itself; an f-string keeps its text and writes every
    interpolation as ``{}``, so ``f"a {orientation} map"`` is ``a {} map`` --
    one row for one refusal, rather than a row per value it can take.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        out = []
        for part in node.values:
            if isinstance(part, ast.Constant) and isinstance(part.value, str):
                out.append(part.value)
            else:
                out.append("{}")
        return "".join(out)
    return None


# Refusal sites whose feature argument is computed rather than written out.
# The scan cannot read those, so each is declared here with the features that
# site can actually raise, read off the code. A *new* computed site fails the
# scan until it is added here -- the exception is enumerated rather than
# waived, which is what keeps both directions of the ledger honest.
DYNAMIC_REFUSALS: dict[tuple[str, str], tuple[str, ...]] = {
    ("tmx.py", "_refuse_object_shape"): (
        "ellipse objects",
        "polygon objects",
        "polyline objects",
        "text objects",
    ),
}


def _raised_features() -> set[str]:
    found: set[str] = set()
    for path in sorted(ENGINE.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for func in ast.walk(tree):
            if not isinstance(func, ast.FunctionDef):
                continue
            for node in ast.walk(func):
                if not isinstance(node, ast.Raise) or node.exc is None:
                    continue
                call = node.exc
                if not isinstance(call, ast.Call):
                    continue
                if not (isinstance(call.func, ast.Name) and call.func.id == "TiledUnsupported"):
                    continue
                if not call.args:
                    continue
                feature = _normal_form(call.args[0])
                if feature is not None:
                    found.add(feature)
                    continue
                declared = DYNAMIC_REFUSALS.get((path.name, func.name))
                assert declared is not None, (
                    f"{path.name}:{func.name} raises TiledUnsupported with a computed "
                    "feature the scan cannot read -- declare it in DYNAMIC_REFUSALS "
                    "with the features it raises, or make the argument a literal"
                )
                found.update(declared)
    return found


# Rows under this heading describe things Tiled itself does not have, so
# there is no refusal in the source to check them against and they are not
# part of the two-way ledger. Skipped by name rather than by a fourth state,
# because "we will never do this" is a different kind of statement from the
# three states and giving it one would blur them.
_UNCHECKED_SECTION = "Permanent non-goals"


def _rows() -> list[tuple[str, str, str]]:
    """``(feature, state, note)`` for every checked row of the matrix table."""
    rows = []
    section = ""
    for line in MATRIX.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            section = line[3:].strip()
        if section == _UNCHECKED_SECTION:
            continue
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        feature = cells[0].strip("`")
        rows.append((feature, cells[1], cells[2]))
    return rows


def test_the_matrix_exists_and_parses():
    assert MATRIX.is_file()
    assert _rows(), "the matrix has no rows"


def test_every_row_is_in_exactly_one_state():
    for feature, state, _ in _rows():
        assert state in STATES, f"{feature!r} is in unknown state {state!r}"


def test_no_feature_is_listed_twice():
    features = [feature for feature, _, _ in _rows()]
    assert len(features) == len(set(features)), "a feature appears in two rows"


def test_every_refusal_in_the_source_has_a_row():
    """The direction that catches a refusal added without a doc change."""
    refused = {feature for feature, state, _ in _rows() if state == "refused"}
    assert _raised_features() - refused == set()


def test_every_refused_row_still_refuses():
    """The direction that catches a row left behind by a milestone. When a
    feature starts loading, its row moves to ``round-trips`` in the same
    commit -- that is the ritual, and this is what enforces it."""
    refused = {feature for feature, state, _ in _rows() if state == "refused"}
    assert refused - _raised_features() == set()


def test_a_row_that_claims_to_round_trip_names_a_fixture_that_exists():
    """A ``round-trips`` claim is only worth what backs it. The note column
    carries the fixture stem in backticks; the file has to be there."""
    from ._corpus import FIXTURE_DIR

    for feature, state, note in _rows():
        if state != "round-trips":
            continue
        stems = re.findall(r"`([a-z0-9-]+)`", note)
        assert stems, f"{feature!r} claims to round-trip but names no fixture"
        for stem in stems:
            assert (FIXTURE_DIR / f"{stem}.tmx").is_file(), (
                f"{feature!r} names fixture {stem!r}, which is not in the corpus"
            )
