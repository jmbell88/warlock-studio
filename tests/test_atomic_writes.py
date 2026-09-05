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
import contextlib
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "warlock" / "studio"

#: Methods that write a file where they are pointed, in place.
WRITERS = {"write_bytes", "write_text", "save"}


def _studio_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


#: The pickers that hand back a destination the user chose. ``select_folder``
#: joined them with Sirens' export: it is the one picker in the app that returns
#: a *directory*, and every file written under one is as much "a path the user
#: named" as a file they typed the name of -- more so, since the names inside it
#: are chosen by the exporter and so are the ones that collide with a previous
#: export.
PICKERS = {"save_file", "select_folder"}


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
            if isinstance(call, ast.Call) and _called_name(call) in PICKERS:
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


# --- M03: concurrent writers to one destination must not share a temp name ---
#
# Export task keys can differ per tab (``packwright_io.py``'s save/export/
# library keys are all ``f"packwright-...:{tab.uid}"``), so two tabs exporting
# under one basename are two genuinely concurrent stagings of one destination
# -- not two nested calls with a well-defined finish order. ``staged`` and
# ``staged_set`` both used to build the temp name from *only* the
# destination's own name, so two such stagings picked the same file.


def test_overlapping_stagings_of_one_destination_get_different_temp_names(tmp_path) -> None:
    """The collision, reproduced directly rather than inferred.

    Two staging contexts are opened over one destination without nesting them
    -- two tabs racing an export, not two ``with`` blocks whose exit order is
    fixed by Python's stack discipline. Under the old fixed name
    (``f".{name}.tmp"``) ``tmp_a`` and ``tmp_b`` were the same path, so writing
    through one silently overwrote the other's bytes; finishing the second one
    first (the ordinary case -- whichever tab's encode happens to land first)
    published *its* content under the name, and then the first writer's own
    ``os.replace`` raised ``FileNotFoundError`` against a temp file the other
    side had already renamed away.
    """
    from warlock.studio import atomic

    dest = tmp_path / "shared.bin"
    cm_a = atomic.staged(dest)
    cm_b = atomic.staged(dest)
    tmp_a = cm_a.__enter__()
    tmp_b = cm_b.__enter__()
    try:
        assert tmp_a != tmp_b, "two overlapping stagings of one destination shared a temp name"
        assert tmp_a.parent == tmp_b.parent == dest.parent, "still a sibling of the destination"
        tmp_a.write_bytes(b"first writer")
        tmp_b.write_bytes(b"second writer")
        # B finishes first: its replace and cleanup must not touch A's
        # still-open staging file.
        cm_b.__exit__(None, None, None)
        assert dest.read_bytes() == b"second writer"
        assert tmp_a.exists(), "A's own staging file was consumed by B's cleanup"
        cm_a.__exit__(None, None, None)
        assert dest.read_bytes() == b"first writer"
    finally:
        # Only reached if an assertion above failed before both contexts
        # closed; a clean run has already exited both.
        with contextlib.suppress(Exception):
            cm_a.__exit__(None, None, None)
        with contextlib.suppress(Exception):
            cm_b.__exit__(None, None, None)


def test_staged_set_gives_every_call_its_own_temp_name(tmp_path, monkeypatch) -> None:
    """``staged_set``'s own construction, held to the same rule as ``staged``.

    ``packwright_io.export_files`` and ``export_library`` can each build a
    ``staged_set`` around one PNG basename from a different task key, so two
    concurrent calls over the same destination is not a hypothetical shape.
    Reproduced by stalling the first call mid-write (past its own
    ``tmp.write_bytes``, before its ``os.replace``) and running a second call
    over the same destination while it waits: under the old fixed name, the
    second call's ``write_bytes`` clobbers the first call's still-unpublished
    staging file, and the first call then publishes the *second* call's bytes
    under its own name.
    """
    from warlock.studio import atomic

    dest = tmp_path / "atlas.png"
    real_write_bytes = Path.write_bytes
    released = False

    def _stalling_write(self: Path, data: bytes):
        real_write_bytes(self, data)
        nonlocal released
        if not released and self.name.startswith(f".{dest.name}."):
            # Hand control to a concurrent second call the instant the first
            # call's own temp file has bytes in it -- the exact window a
            # shared name made unsafe.
            released = True
            atomic.staged_set({dest: b"second writer"})

    monkeypatch.setattr(Path, "write_bytes", _stalling_write)
    atomic.staged_set({dest: b"first writer"})

    # The second, nested call finished first (it has no writer of its own to
    # wait on) and published its bytes; the outer call then published its own
    # -- each into a temp file the other could not see, because the names
    # differed. Under the old shared name the outer call's own write would
    # have overwritten the inner call's temp file (or vice versa), and the
    # final content would not be traceable to either call cleanly.
    assert dest.read_bytes() == b"first writer"
