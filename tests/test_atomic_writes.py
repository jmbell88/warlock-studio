"""A destination the user picked is never truncated to write it.

``studio/atomic.py``, and the rule it exists for. There are twenty-two
``dialogs.save_file`` sites in ``studio/``; every one writes to a path the user
named, and a path the user named is one they may well have named before --
"export it again over the last one" is the ordinary case, not the odd one.
``Path.write_bytes`` and ``Image.save`` both truncate before they write a byte,
so a crash, a full disk or a yanked drive halfway through leaves the previous
file destroyed and no new one in its place.

The scan is the point. ``_write_atomic`` existed in ``inker_mode`` for a year
with four callers while nine other exports wrote straight onto the user's file,
because "remember to stage this one" is not a mechanism -- the same argument
``zipguard.BoundedZip`` was created to settle for archive bounds ("a property
of the archive object rather than a rule 18 call sites must remember").
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "warlock" / "studio"

#: Methods that write a file where they are pointed, in place.
WRITERS = {"write_bytes", "write_text", "save"}


def _studio_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _picked_names(node: ast.AST) -> set[str]:
    """Every local name that holds a path the user chose in a save dialog.

    The closure matters as much as the seed: an exporter picks one file and
    then writes a *family* beside it (``out = dest.parent / f"{name}.png"``,
    ``png_path = dest.with_suffix(".png")``), and those are the same user's
    directory and the same risk. So a name assigned from an expression that
    mentions a picked name is itself picked, to a fixed point.
    """
    picked: set[str] = set()
    assigns: list[tuple[str, ast.AST]] = []
    for child in ast.walk(node):
        if not isinstance(child, ast.Assign) or len(child.targets) != 1:
            continue
        target = child.targets[0]
        if not isinstance(target, ast.Name):
            continue
        assigns.append((target.id, child.value))
        for call in ast.walk(child.value):
            if isinstance(call, ast.Call) and _called_name(call) == "save_file":
                picked.add(target.id)
    grew = True
    while grew:
        grew = False
        for name, value in assigns:
            if name in picked:
                continue
            mentioned = {
                sub.id for sub in ast.walk(value) if isinstance(sub, ast.Name)
            }
            if mentioned & picked:
                picked.add(name)
                grew = True
    return picked


def _called_name(call: ast.Call) -> str:
    func = call.func
    if isinstance(func, ast.Attribute):
        return func.attr
    if isinstance(func, ast.Name):
        return func.id
    return ""


def _root_name(node: ast.AST) -> str:
    """The leftmost ``Name`` of an attribute/subscript/binop chain."""
    while True:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute | ast.Subscript):
            node = node.value
        elif isinstance(node, ast.BinOp):
            node = node.left
        elif isinstance(node, ast.Call):
            node = node.func
        else:
            return ""


def _offences(tree: ast.AST) -> list[tuple[int, str]]:
    # A set, because the exporters that matter are nested: ``run()`` inside
    # ``export_png()`` means ``ast.walk`` reaches every offence once per
    # enclosing function, and a doubled list reads as two bugs.
    found: set[tuple[int, str]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        picked = _picked_names(node)
        if not picked:
            continue
        for call in ast.walk(node):
            if not isinstance(call, ast.Call):
                continue
            if _called_name(call) not in WRITERS:
                continue
            func = call.func
            if not isinstance(func, ast.Attribute):
                continue
            # ``atomic.write_bytes(dest, ...)`` is the answer, not an offence:
            # the receiver is the module, and the destination is an argument.
            if _root_name(func.value) == "atomic":
                continue
            receiver = _root_name(func.value)
            if receiver in picked:
                found.add((call.lineno, f"{receiver}.{func.attr}()"))
            # ``Image.fromarray(...).save(out)`` and friends: the destination
            # is the first argument rather than the receiver.
            elif call.args and _root_name(call.args[0]) in picked:
                found.add((call.lineno, f"{func.attr}({_root_name(call.args[0])})"))
    return sorted(found)


@pytest.mark.parametrize("path", _studio_files(), ids=lambda p: p.name)
def test_no_save_dialog_writes_its_destination_in_place(path: Path) -> None:
    """Everything downstream of ``dialogs.save_file`` goes through ``atomic``.

    Reported per file so a failure names the module and the line rather than
    handing back one list of everything in ``studio/``.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offences = _offences(tree)
    assert not offences, (
        f"{path.name} writes a user-picked destination in place: "
        + ", ".join(f"line {line}: {what}" for line, what in offences)
        + " -- use studio.atomic (write_bytes/write_text/save_image/staged)"
    )


def test_the_scan_would_catch_a_regression() -> None:
    """The scan itself, against the shape it exists to refuse.

    Written because the first version of this scan passed on source it should
    have failed: it looked only at the *receiver*, and ``Image.save(out)``
    carries the destination as an argument.
    """
    source = (
        "def export(ctx):\n"
        "    def run():\n"
        "        dest = dialogs.save_file('t', 'x.png', F)\n"
        "        out = dest.parent / 'a.png'\n"
        "        dest.write_bytes(b'x')\n"
        "        img.save(out, 'PNG')\n"
    )
    offences = _offences(ast.parse(source))
    assert [what for _line, what in offences] == ["dest.write_bytes()", "save(out)"]


def test_the_scan_accepts_the_atomic_helpers() -> None:
    source = (
        "def export(ctx):\n"
        "    def run():\n"
        "        dest = dialogs.save_file('t', 'x.png', F)\n"
        "        out = dest.with_suffix('.json')\n"
        "        atomic.write_bytes(dest, b'x')\n"
        "        atomic.write_text(out, 'x')\n"
        "        with atomic.staged(dest) as tmp:\n"
        "            write_gif(tmp)\n"
    )
    assert _offences(ast.parse(source)) == []
