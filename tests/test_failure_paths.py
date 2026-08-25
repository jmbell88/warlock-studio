"""What the user is told when something the app did on their behalf goes wrong.

Two rules, and the second is the one that needed a scan rather than a fix.

**A task result has to be claimed by somebody.** ``main`` states the rule --
"the app claims results by prefix; a key without one is a result delivered
nowhere" -- and it was enforced by nothing, on the one path where the failure
is invisible from outside: *success*. A key that matched none of the forty-odd
branches in ``_on_task_done`` simply fell off the end, so a mode closing while
its own task was in flight and a new key whose author had never read that
function were indistinguishable from work that had been handled.

**A broad ``except Exception`` has to leave a trace.** The plan for this batch
expected a sweep of a hundred and sixty-odd of them looking for swallowed user
actions. There were none: every broad catch under ``studio/`` either surfaces
to the user, sets an error field the pane draws, logs, or degrades to a
documented default. Six of them were silent *and* unexplained, which is the
only shape that cannot be reviewed, and each now says why. What this file adds
is the ratchet -- ``test_winjob``'s idiom, applied to the other rule that was
being held by nobody.
"""

from __future__ import annotations

import ast
import io
import tokenize
from pathlib import Path
from types import SimpleNamespace

from warlock.studio import main as main_mod

STUDIO = Path(main_mod.__file__).parent


# --- a result with nowhere to go ---------------------------------------------


class _Toasts(list):
    def __call__(self, message, kind="info", action=None, **_kw):
        self.append((message, kind))


def _app(monkeypatch):
    """An ``App`` with only what ``_on_task_done``'s tail touches."""
    app = main_mod.App.__new__(main_mod.App)
    app._unclaimed = set()
    toasts = _Toasts()
    app.app_ctx = SimpleNamespace(toast=toasts, state=SimpleNamespace(preview={}))
    return app, toasts


def _done(key, result=None):
    return SimpleNamespace(key=key, result=result, ok=True, message="", action=None)


def test_a_result_nothing_claims_is_reported_once(monkeypatch, caplog):
    """The two ways to get here are a mode closing while its own task was in
    flight, and a new key whose author never read ``_on_task_done``."""
    app, _toasts = _app(monkeypatch)
    with caplog.at_level("INFO"):
        app._on_task_done(_done("nobody-claims-this:7"))
        app._on_task_done(_done("nobody-claims-this:7"))
    said = [r for r in caplog.records if "nowhere to deliver" in r.message]
    assert len(said) == 1, "once per key, not once per arrival"
    assert "nobody-claims-this:7" in said[0].getMessage()


def test_a_second_unclaimed_key_is_reported_separately(monkeypatch, caplog):
    app, _toasts = _app(monkeypatch)
    with caplog.at_level("INFO"):
        app._on_task_done(_done("one:1"))
        app._on_task_done(_done("two:2"))
    assert len([r for r in caplog.records if "nowhere to deliver" in r.message]) == 2


def test_a_deliberately_silent_key_says_nothing(monkeypatch, caplog):
    """The list is the point: each of these is silent for a reason written
    beside it, and everything *not* on it is a defect.

    One test over the tuple rather than a ``parametrize`` on it: the tuple is
    the thing under test, so reading it at collection time makes the whole
    file fail to collect when it is absent, which reports a great deal less
    than one failing assertion does.
    """
    app, _toasts = _app(monkeypatch)
    with caplog.at_level("INFO"):
        for prefix in main_mod.SILENT_TASK_KEYS:
            app._on_task_done(_done(f"{prefix}whatever"))
    assert not [r for r in caplog.records if "nowhere to deliver" in r.message]


def test_every_silent_key_is_a_key_something_actually_submits():
    """The other direction, so the allowlist cannot rot into a list of prefixes
    that used to mean something -- which is how it would come to hide a real
    unclaimed key."""
    text = "\n".join(
        path.read_text(encoding="utf-8") for path in STUDIO.rglob("*.py")
    )
    for prefix in main_mod.SILENT_TASK_KEYS:
        assert f'"{prefix}' in text, f"nothing submits {prefix}"


def test_a_screenshot_says_where_it_went(monkeypatch):
    """It was outside every branch, so a capture the user had just chosen a
    destination for finished in silence -- and the only other thing that looks
    like that is a save that did not happen."""
    app, toasts = _app(monkeypatch)
    app._on_task_done(_done("screenshot", result="C:/shots/view.png"))
    assert toasts and "C:/shots/view.png" in toasts[0][0]


def test_a_cancelled_screenshot_picker_says_nothing(monkeypatch):
    """``None`` is a dismissed dialog, which the three save keys beside it
    already rely on meaning nothing."""
    app, toasts = _app(monkeypatch)
    app._on_task_done(_done("screenshot", result=None))
    assert not toasts


# --- the ratchet --------------------------------------------------------------


def _broad_handlers(path: Path):
    """Every ``except Exception``/``BaseException`` in one file.

    -> ``(lineno, body-dump, whether a comment sits inside the handler)``. The
    comment span runs from the ``except`` line to the last line of its body, so
    a note on the ``except`` itself (``# noqa: BLE001 - ...``, the form this
    repo already uses) counts, and so does one written inside.
    """
    src = path.read_text(encoding="utf-8")
    comments = {
        tok.start[0]
        for tok in tokenize.generate_tokens(io.StringIO(src).readline)
        if tok.type == tokenize.COMMENT
    }
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.ExceptHandler):
            continue
        caught = node.type
        names = []
        if isinstance(caught, ast.Name):
            names = [caught.id]
        elif isinstance(caught, ast.Tuple):
            names = [e.id for e in caught.elts if isinstance(e, ast.Name)]
        if not ({"Exception", "BaseException"} & set(names)):
            continue
        end = max(getattr(stmt, "end_lineno", stmt.lineno) for stmt in node.body)
        yield (
            node.lineno,
            ast.dump(ast.Module(body=node.body, type_ignores=[])),
            any(node.lineno <= line <= end for line in comments),
        )


#: What counts as leaving a trace. ``error`` covers both a pane's ``.error``
#: field and a returned sentence, which are the two ways a refusal reaches the
#: screen without a toast.
_SURFACES = ("toast", "note_field_error", "Raise(", "alert", "log", "error")


def test_no_broad_except_in_the_studio_layer_swallows_in_silence():
    """The rule, held by a scan rather than by review.

    A broad catch is not a defect here -- the frame loop genuinely has to
    degrade past a lost display, a stand-in backend and a headless run -- but a
    broad catch that does *nothing and says nothing* is, because there is no
    way to tell one from a bug afterwards. So each must either surface, log, or
    carry a note saying why it is silent.

    ``studio/`` and not the whole tree deliberately: this is a rule about what
    the *user* is told, and the worker and service layers have no user in front
    of them -- ``queue.py``'s catches are reported through the job row instead.
    """
    offenders = []
    for path in sorted(STUDIO.rglob("*.py")):
        for lineno, body, commented in _broad_handlers(path):
            if any(word in body for word in _SURFACES) or commented:
                continue
            offenders.append(f"{path.relative_to(STUDIO)}:{lineno}")
    assert not offenders, (
        "these broad excepts neither surface, log, nor say why they are "
        f"silent: {offenders}"
    )


def test_the_scan_is_actually_looking_at_something():
    """The guard the assertion above cannot make: it is a ``for`` over a glob,
    and every version of it passes against an empty one."""
    seen = sum(1 for path in STUDIO.rglob("*.py") for _ in _broad_handlers(path))
    assert seen > 50, f"only {seen} broad handlers found; the scan has stopped scanning"


def test_the_scan_would_catch_a_silent_one(tmp_path):
    """And the guard *that* cannot make: a rule whose detector never fires is
    a rule that passes because it is broken."""
    sample = tmp_path / "sample.py"
    sample.write_text(
        "def f():\n    try:\n        g()\n    except Exception:\n        return None\n",
        encoding="utf-8",
    )
    found = list(_broad_handlers(sample))
    assert len(found) == 1
    _lineno, body, commented = found[0]
    assert not commented
    assert not any(word in body for word in _SURFACES)
