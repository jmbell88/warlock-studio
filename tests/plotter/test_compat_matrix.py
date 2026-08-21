"""The refusal ledger, machine-checked against the code that does the refusing.

``docs/COMPAT.md``'s **Plotter/Tiled** part is a promise about what Plotter does
with a Tiled file, and a promise in a markdown table drifts the moment somebody
adds a refusal without opening it. So the table is not prose: the ``refused``
rows are checked, both ways, against every ``TiledUnsupported`` raised in the
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

MATRIX = Path(__file__).resolve().parents[2] / "docs" / "COMPAT.md"
ENGINE = Path(__file__).resolve().parents[2] / "src" / "warlock" / "studio" / "plotter"

STATES = {
    "round-trips",
    # A construct this editor reads and writes that no Tiled release does --
    # an oblique orientation, a layer blend mode, an object opacity, a capsule
    # shape, a list property. A separate state rather than a ``round-trips``
    # row with a caveat in its note, because the two make different claims and
    # a note is not machine-checkable: ``round-trips`` says a Tiled file
    # survives the trip, and this says only that *our* file does. Both are
    # positive rows and both must name a fixture; see the fixture test below.
    "warlock-dialect",
    "refused",
    "preserved-verbatim",
    "silently-dropped",
}

#: The states that assert a feature is carried rather than stopped. Both owe a
#: fixture.
POSITIVE_STATES = {"round-trips", "warlock-dialect"}


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
DYNAMIC_REFUSALS: dict[tuple[str, str], tuple[str, ...]] = {}


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

# ``docs/COMPAT.md`` holds two ledgers, one per foreign format, and only the
# Tiled one is executable. The Aseprite part's tables carry rows spelled the
# same way -- ``| `Track.alpha_lock` | dropped | ...`` -- over an entirely
# different state vocabulary, so a scan that read the whole file would report
# those as Tiled rows in unknown states. **Do not widen this**: the scope is
# what lets one file hold both ledgers without one gating the other. A ``##``
# heading opens a part; a ``###`` heading opens a section within it.
_CHECKED_PART = "Plotter"


def _rows() -> list[tuple[str, str, str]]:
    """``(feature, state, note)`` for every checked row of the matrix table."""
    rows = []
    part = ""
    section = ""
    for line in MATRIX.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            part = line[3:].strip()
            section = ""
        elif line.startswith("### "):
            section = line[4:].strip()
        if not part.startswith(_CHECKED_PART):
            continue
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


def test_the_scan_stops_at_the_aseprite_part():
    """The scoping above, asserted rather than trusted.

    The two ledgers merged into one file on 2026-08-21, and the failure mode of
    that merge is silent in the wrong direction: a widened scan pulls Aseprite
    rows in, they fail ``test_every_row_is_in_exactly_one_state`` loudly, and
    somebody "fixes" it by adding ``dropped`` to ``STATES`` -- at which point
    the two vocabularies are one and neither ledger says what it meant. So the
    boundary itself is the assertion.
    """
    states = {state for _, state, _ in _rows()}
    assert states <= STATES
    assert "dropped" not in states and "warned" not in states


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


def test_a_positive_row_names_a_fixture_that_exists():
    """A positive claim is only worth what backs it. The note column carries
    the fixture stem behind a ``fixture:`` marker -- see the state bullets in
    the doc's own header -- rather than in any backticked word, because notes
    already carry unrelated backticked prose (` ``zstd`` `, format names, and
    so on) that an unanchored pattern would mistake for a fixture stem.

    Both positive states are checked. A ``warlock-dialect`` row makes a smaller
    claim than a ``round-trips`` one, but it is still a claim that the
    construct survives being written and read back, and an unbacked one is
    exactly as empty."""
    from ._corpus import FIXTURE_DIR

    for feature, state, note in _rows():
        if state not in POSITIVE_STATES:
            continue
        stems = re.findall(r"fixture:\s*`([a-z0-9-]+)`", note)
        assert stems, f"{feature!r} is a {state} row but names no fixture"
        for stem in stems:
            assert (FIXTURE_DIR / f"{stem}.tmx").is_file(), (
                f"{feature!r} names fixture {stem!r}, which is not in the corpus"
            )


def test_every_dialect_row_says_tiled_does_not_have_it():
    """The one thing a ``warlock-dialect`` row exists to say.

    The state name alone is a word in a table; what stops the row being read as
    a normal feature is the sentence naming what Tiled actually has instead.
    Pinned because this table is the file a future milestone edits in a hurry,
    and a dialect row that quietly loses its warning is indistinguishable from
    a compatibility claim."""
    for feature, state, note in _rows():
        if state != "warlock-dialect":
            continue
        assert "Tiled has no" in note or "Tiled has per-" in note, (
            f"{feature!r} is a dialect row but its note never says what Tiled "
            "has instead"
        )
