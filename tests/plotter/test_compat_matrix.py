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

MATRIX = Path(__file__).resolve().parents[2] / "docs" / "PLOTTER_COMPAT.md"
ENGINE = Path(__file__).resolve().parents[2] / "src" / "warlock" / "studio" / "plotter"

STATES = {"round-trips", "refused", "preserved-verbatim", "silently-dropped"}


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
#
# Keyed by (filename, enclosing function name); a call at module scope keys
# under "<module>". The key names *where the call to TiledUnsupported sits*,
# not how it reaches ``raise`` -- the scan below no longer cares whether the
# call is the direct argument of a ``raise`` or is assigned to a name and
# raised later, so the key does not need to either.
DYNAMIC_REFUSALS: dict[tuple[str, str], tuple[str, ...]] = {
    ("tmx.py", "_refuse_object_shape"): (
        "ellipse objects",
        "polygon objects",
        "polyline objects",
        "text objects",
    ),
    # The writer door. Same five sentences the readers use, looked up by shape
    # kind out of ``tmx._UNWRITABLE_SHAPES`` -- one table rather than a second
    # list, so the two doors cannot drift into refusing an ellipse under two
    # different names.
    ("tmx.py", "_refuse_unwritable_objects"): (
        "ellipse objects",
        "polygon objects",
        "polyline objects",
        "text objects",
        "tile objects",
    ),
}


class _RefusalCallVisitor(ast.NodeVisitor):
    """Every call to ``TiledUnsupported`` in one module, wherever it sits.

    Deliberately not a ``raise ... `` matcher: ``raise TiledUnsupported(...)`` is
    the common spelling but not the only legal one -- ``err =
    TiledUnsupported(...); raise err`` constructs the same exception, and a scan
    that only looks at ``ast.Raise.exc`` would wave that spelling straight
    through with no assertion at all. Finding the *call* rather than the
    *raise* is what makes both spellings, and a callee written as
    ``tsx.TiledUnsupported`` or raised at module scope, visible to the ledger.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stack: list[str] = []
        self.found: set[str] = set()

    def _enter_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self._stack.append(node.name)
        self.generic_visit(node)
        self._stack.pop()

    visit_FunctionDef = _enter_function
    visit_AsyncFunctionDef = _enter_function

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        callee = func.id if isinstance(func, ast.Name) else (
            func.attr if isinstance(func, ast.Attribute) else None
        )
        if callee == "TiledUnsupported":
            self._record(node)
        self.generic_visit(node)

    def _record(self, node: ast.Call) -> None:
        feature = _normal_form(node.args[0]) if node.args else None
        if feature is not None:
            self.found.add(feature)
            return
        where = self._stack[-1] if self._stack else "<module>"
        declared = DYNAMIC_REFUSALS.get((self.path.name, where))
        assert declared is not None, (
            f"{self.path.name}:{node.lineno} ({where}) calls TiledUnsupported with a "
            "computed feature the scan cannot read -- declare it in DYNAMIC_REFUSALS "
            "with the features it raises, or make the argument a literal"
        )
        self.found.update(declared)


def _raised_features() -> set[str]:
    found: set[str] = set()
    for path in sorted(ENGINE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        visitor = _RefusalCallVisitor(path)
        visitor.visit(tree)
        found |= visitor.found
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
    carries the fixture stem behind a ``fixture:`` marker -- see the ``Feature
    names are...`` intro paragraph and the ``round-trips`` bullet in the doc's
    own header -- rather than in any backticked word, because notes already
    carry unrelated backticked prose (` ``zstd`` `, format names, and so on)
    that an unanchored pattern would mistake for a fixture stem."""
    from ._corpus import FIXTURE_DIR

    for feature, state, note in _rows():
        if state != "round-trips":
            continue
        stems = re.findall(r"fixture:\s*`([a-z0-9-]+)`", note)
        assert stems, f"{feature!r} claims to round-trip but names no fixture"
        for stem in stems:
            assert (FIXTURE_DIR / f"{stem}.tmx").is_file(), (
                f"{feature!r} names fixture {stem!r}, which is not in the corpus"
            )
