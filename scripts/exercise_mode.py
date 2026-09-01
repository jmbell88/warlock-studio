"""Press every control in one mode, through the app's real input path.

``tests/test_studio_smoke.py`` asserts that every pane builds; it asserts
nothing about whether a control is *wired* to anything.
``screenshot_modes.py`` photographs a mode at rest. So the failure nobody
catches is a control that draws correctly and does nothing -- clipped past its
content region, disabled with no reason, wired to a handler that was renamed,
or reaching one that raises. Every one of those passes the smoke suite and
looks right in a screenshot.

This clicks each of them and records what changed. The clicks are **posted
pygame events**, not direct handler calls: ``App.frame`` -> ``_events`` ->
``pygame.event.get`` is the path a real click takes, and a driver that called
the handler itself would prove the handler works while saying nothing about
whether anything on screen reaches it -- which is the entire question.

Two things are neutralised, and only two. ``TaskRunner.submit`` is replaced
with a recorder, so "did this button reach a handler" is answered without a
real pipeline running; and a short list of labels is refused outright -- Quit,
and anything that opens an OS file dialog, which blocks the frame loop the way
a browser modal does and would hang the run rather than fail it.

Every verdict is machine-assigned. That is what keeps a reader's job small: an
``inert`` control (no task, no toast, no state change, no pixel change) is the
prime suspect and is worth a look; a ``state-changed`` one mostly is not.

Needs a real window -- it draws on screen. Must not run while ``pytest`` is
running: several tests read module source, and ``src/`` must not move under
them.

Run against a throwaway home, all three variables, because ``WARLOCK_DATA_DIR``
alone does not move the sqlite store::

    WARLOCK_HOME=... WARLOCK_DATA_DIR=... WARLOCK_DB=... WARLOCK_UI_PROBE=1
        uv run python scripts/exercise_mode.py --mode inker --out <dir>
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from _appharness import (  # noqa: E402
    SETTLE_FRAMES,
    WARMUP,
    boot,
    close_popups,
    seed,
    seed_asset,
    seed_review,
    seed_troupe,
)

from warlock.studio import (
    create_stages,  # noqa: E402
    guard,  # noqa: E402
)
from warlock.studio import modes as _modes  # noqa: E402

MODES = _modes.KEYS

#: Controls that must not be pressed, matched against the user-visible half of
#: the label (case-folded). Deliberately short and explicit: a broad pattern
#: would quietly skip the controls this pass exists to test.
#:
#: Two reasons only appear here. Something that ends the process, and something
#: that opens a native file dialog -- the latter is a modal message pump owned
#: by the OS, so it blocks ``App.frame`` and the run hangs rather than failing,
#: which is the worst outcome a harness can have.
REFUSED = (
    "quit",
    "exit",
    "open...",
    "open…",
    "save as...",
    "save as…",
    "import...",
    "import…",
    "browse...",
    "browse…",
    "choose folder",
    "choose file",
    "reveal in explorer",
    "open log",
)

#: How many frontier rounds the walk will run before giving up. A tab, a
#: collapsed header or a menu hides controls from the first census, and opening
#: one exposes more -- so the walk repeats until a round finds nothing new.
#: Bounded rather than a bare ``while``, and what the bound dropped is reported:
#: a silent cap reads as full coverage.
MAX_ROUNDS = 6

#: Fraction of pixels that must differ for a frame to count as changed. Not
#: zero: a caret blinks, a hover wash animates, and a control that redraws its
#: own hover state has not *done* anything.
PIXEL_EPSILON = 0.0015


def refused(text: str) -> bool:
    """Whether a control with this visible label is on the do-not-press list."""

    low = text.strip().casefold()
    return any(low == name or low.startswith(name) for name in REFUSED)


# --- what a click changed ----------------------------------------------------


def digest(app) -> tuple:
    """A cheap, comparable summary of everything a click could move.

    Every component is a public read the app already offers. It is not a hash
    of the world -- it is the set of things whose movement means "that control
    did something", which is the question a verdict answers.
    """
    from warlock.studio import palette

    ctx = app.app_ctx
    state = ctx.state
    module, tab = palette._doc_mode(ctx)
    history = palette._doc_history(ctx)
    tool = ""
    docs: tuple = ()
    holder = getattr(module, "ensure", None) if module is not None else None
    if callable(holder):
        owner = holder(ctx)
        tool = str(getattr(owner, "tool", "") or "")
        # ``docs`` is what every workspace calls its open-document list; the
        # uid is the only field all four share, so the tuple is a count and an
        # identity rather than a set of names.
        docs = tuple(
            str(getattr(one, "uid", "") or getattr(one, "name", "") or id(one))
            for one in getattr(owner, "docs", ()) or ()
        )
    return (
        state.mode,
        getattr(state, "create_stage", ""),
        state.selected,
        tool,
        getattr(history, "head", 0) if history is not None else 0,
        docs,
        len(state.toasts),
        bool(getattr(tab, "dirty", False)) if tab is not None else False,
        # The overlays, which are the whole point of half the palette: the
        # Manual, the tour, the shortcut list and the palette itself are none
        # of them modes, so without this every command that raises one came
        # out ``inert`` -- the driver reporting its own blind spot as
        # fourteen dead controls.
        (
            state.manual.open,
            state.manual.chapter if state.manual.open else "",
            state.tour.key,
            state.shortcuts_requested,
            state.palette_open,
        ),
    )


#: The digest's components, in order. Named here rather than inline so the
#: delta reads as prose in the manifest instead of as tuple indices.
DIGEST_NAMES = (
    "mode",
    "create_stage",
    "selected",
    "tool",
    "undo_head",
    "documents",
    "toasts",
    "dirty",
    "overlays",
)


def describe_delta(before: tuple, after: tuple) -> list[str]:
    """The named components of the digest that moved."""

    return [
        f"{name}: {old!r} -> {new!r}"
        for name, old, new in zip(DIGEST_NAMES, before, after, strict=True)
        if old != new
    ]


def verdict(
    *,
    raised: str | None,
    enabled: bool,
    reason: str,
    toast_levels: tuple[str, ...],
    submitted: tuple[str, ...],
    state_delta: list[str],
    pixel_delta: float,
) -> str:
    """Classify one press. Pure -- which is what makes it testable.

    Order is severity, not convenience: a control that raised is a crash
    whatever else it also did, and a disabled control with no explanation is a
    defect by ``palette.Command``'s own docstring even though pressing it
    correctly does nothing.
    """
    if raised:
        return "raised"
    if "error" in toast_levels:
        return "toast-error"
    if not enabled:
        return "disabled" if reason.strip() else "disabled-no-reason"
    if submitted:
        return "submitted"
    if state_delta:
        return "state-changed"
    if pixel_delta > PIXEL_EPSILON:
        return "pixels-changed"
    return "inert"


#: Verdicts whose picture is always worth a reader looking at.
ALWAYS_LOOK = ("raised", "inert", "toast-error", "disabled-no-reason", "hard-reset")


# --- driving -----------------------------------------------------------------


@dataclass
class Recorder:
    """Stands in for ``TaskRunner.submit`` and watches ``State.toast``."""

    submitted: list[str] = field(default_factory=list)
    toasts: list[tuple[str, str]] = field(default_factory=list)

    def reset(self) -> None:
        self.submitted.clear()
        self.toasts.clear()


def install_stubs(app, rec: Recorder) -> None:
    """Replace the two doors a control's effects go through.

    ``TaskRunner.submit`` on the *instance*, so ``ctx.submit`` and every direct
    caller are covered by one patch -- the single-door invariant is what makes
    that possible, and patching ``ctx.submit`` alone would have missed the
    panes that hold the runner.

    Nothing is restored: the app is torn down at the end of the run.
    """
    runner = app.runtime.tasks
    real_toast = app.app_ctx.state.toast

    def fake_submit(key, fn, *args, tag=None, **kwargs):
        rec.submitted.append(str(key))
        return True

    def watched_toast(text, level="info", action=None, action_arg=None):
        rec.toasts.append((str(level), str(text)))
        return real_toast(text, level, action, action_arg)

    runner.submit = fake_submit  # type: ignore[method-assign]
    app.app_ctx.state.toast = watched_toast  # type: ignore[method-assign]


def recover_frame() -> None:
    """Close a frame a raising pane left open, so the next one can begin.

    An exception out of a pane unwinds past ``imgui.render``, and imgui's next
    ``new_frame`` then asserts that the previous one was never ended -- so one
    bad control would take the whole run with it and the manifest naming the
    culprit would never be written. Best effort by construction: the id stack
    may also be unbalanced, which imgui reports and recovers from itself.
    """
    from imgui_bundle import imgui

    try:
        if imgui.get_current_context() is None:
            return
        imgui.end_frame()
    except Exception:  # noqa: BLE001 -- a failed rescue must not mask the crash
        return


def pump(app, frames: int = 1) -> None:
    import pygame

    for _ in range(frames):
        app.frame(1.0 / 60.0)
        pygame.display.flip()


def settle(app) -> None:
    """Warm up, then wait out any animation -- exactly as ``capture`` does."""
    from warlock.studio import motion

    pump(app, WARMUP)
    for _ in range(SETTLE_FRAMES):
        if not motion.animating():
            break
        pump(app)


def click(app, x: float, y: float) -> None:
    """One real click at screen ``(x, y)``, posted as pygame events.

    Three separate frames rather than one batch: hover state settles on the
    motion frame, and imgui fires a button on *release while hovered*, so the
    down and the up have to be observed as two states rather than as two
    entries in one queue.
    """
    import pygame

    pos = (int(x), int(y))
    pygame.event.post(
        pygame.event.Event(pygame.MOUSEMOTION, pos=pos, rel=(0, 0), buttons=(0, 0, 0))
    )
    pump(app)
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1))
    pump(app)
    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=1))
    pump(app)


def shoot(app, path: Path):
    """Capture the framebuffer to ``path``. -> the image, for differencing."""
    import pygame
    from PIL import Image

    width, height = pygame.display.get_window_size()
    data = app.ctx.screen.read(components=3, alignment=1)
    # GL's origin is bottom-left and everybody else's is top-left.
    image = Image.frombytes("RGB", (width, height), data).transpose(
        Image.FLIP_TOP_BOTTOM
    )
    image.save(path)
    return image


def pixel_delta(before, after) -> float:
    """Fraction of pixels that differ. Cheap, and good enough to threshold on."""
    from PIL import ImageChops

    if before.size != after.size:
        return 1.0
    grey = ImageChops.difference(before.convert("RGB"), after.convert("RGB")).convert("L")
    # Above a small threshold rather than any difference at all: the viewport
    # dithers and a single-level change everywhere is not a change.
    changed = sum(count for value, count in enumerate(grey.histogram()) if value > 8)
    return changed / float(before.size[0] * before.size[1])


def seed_mode(app, mode: str) -> None:
    """Put ``mode`` in a state that has controls in it, not an empty state."""

    if mode == create_stages.MODE:
        seed_asset(app)
    elif mode == "review":
        seed_review(app)
    elif mode == "troupe":
        # Troupe's own seed *as well as* the shared one: the shared canvases
        # are what the rail's other modes need, and this mode's Sheet pane,
        # frame table and handoffs are drawn only for a selected character.
        # Without it the pass covers the empty state and reports a coverage
        # number that reads like the whole mode.
        seed(app)
        seed_troupe(app)
    else:
        seed(app)


#: The digest components a press is *entitled* to leave changed and the driver
#: can simply put back: which mode is in front, which stage, what is selected,
#: which tool is armed. Restoring these is not cheating -- they are settings, so
#: an action that changes one has done its job and nothing has to be undone.
_SETTINGS = ("mode", "create_stage", "selected", "tool")

#: The components that describe the *document*, and the ones a restore has to
#: genuinely reverse. If these are back, the app is back.
_DOCUMENT = ("undo_head", "documents", "dirty")

_UNDO_HEAD = DIGEST_NAMES.index("undo_head")


def _document_half(one: tuple) -> tuple:
    return tuple(
        value for name, value in zip(DIGEST_NAMES, one, strict=True) if name in _DOCUMENT
    )


def put_settings_back(app, baseline: tuple, mode: str, stage: str) -> None:
    """Re-arm the settings a press is allowed to have changed.

    Written back rather than undone. A tool switch pushes nothing on the undo
    stack because it is not an edit -- so a restore that only undid would find
    the digest still moved and re-seed the whole workspace, which is how the
    first run of this driver reported twenty-two ``hard-reset``s that were all
    its own.
    """
    from warlock.studio import palette

    ctx = app.app_ctx
    ctx.state.mode = mode
    if mode == create_stages.MODE:
        ctx.state.create_stage = stage
    ctx.state.select(baseline[DIGEST_NAMES.index("selected")])
    module, _tab = palette._doc_mode(ctx)
    holder = getattr(module, "ensure", None) if module is not None else None
    if callable(holder):
        owner = holder(ctx)
        if hasattr(owner, "tool"):
            owner.tool = baseline[DIGEST_NAMES.index("tool")]


def close_overlays(app) -> None:
    """Put away the Manual, the tour, the palette, the sheet and the shortcuts.

    Through each owner's own closer where there is one, because a flag written
    from outside is a second way to close something -- which is how two of them
    came to disagree in the first place.
    """
    from warlock.studio.manual import render as manual_render
    from warlock.studio.panes import palette as palette_pane
    from warlock.studio.panes import tour as tour_pane

    ctx = app.app_ctx
    manual_render.close(ctx)
    palette_pane.close(ctx)
    tour_pane.stop(ctx)
    ctx.state.shortcuts_requested = False


def restore(app, baseline: tuple, mode: str, stage: str) -> str:
    """Get back to the baseline. -> how, for the record.

    Three steps, in order of how much they cost and how much they admit. Close
    whatever transient surface the press raised; undo the document back to the
    baseline's serial; put the settings back. Only if the *document* half is
    still wrong does the workspace get re-seeded -- and that is recorded as
    ``hard-reset``, because an action that cannot be undone is itself a finding.

    Undone to the baseline's serial rather than to an empty history: the seeded
    document arrives with edits of its own, and a loop that ran while
    ``can_undo`` unmade the seed.
    """
    from warlock.studio import palette

    ctx = app.app_ctx
    close_popups(app)
    close_overlays(app)
    for _ in range(64):
        history = palette._doc_history(ctx)
        if history is None or history.head == baseline[_UNDO_HEAD]:
            break
        if not palette._can_undo(ctx):
            break
        palette._doc_undo(ctx)
    put_settings_back(app, baseline, mode, stage)
    pump(app)
    if _document_half(digest(app)) == _document_half(baseline):
        return "undo"
    seed_mode(app, mode)
    put_settings_back(app, baseline, mode, stage)
    settle(app)
    return "hard-reset"


def key_for(control, seen: dict[str, int]) -> str:
    """A stable id for a control. A duplicated label gets an index suffix."""

    base = f"{control.where or 'floating'}/{control.kind}/{control.name}"
    index = seen.get(base, 0)
    seen[base] = index + 1
    return f"{base}#{index}"


def safe_name(key: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in key)[:80]


def _fresh(census, pressed: set[str]) -> list:
    """This census's controls that have not been pressed yet, keyed stably."""

    seen: dict[str, int] = {}
    out = []
    for control in census:
        key = key_for(control, seen)
        if key not in pressed:
            out.append((key, control))
    return out


def exercise(
    app,
    mode: str,
    out: Path,
    max_rounds: int,
    rec: Recorder,
    skip: tuple[str, ...] = (),
) -> dict:
    """Click everything ``mode`` draws, round by round, until nothing is new.

    The rounds are the frontier walk: a control behind a tab, a collapsed
    header or a menu is not in the first census, and opening one puts more on
    screen. So the census is re-read after each round and anything new joins
    the queue, until a round finds nothing.
    """
    from warlock.studio import probe

    ctx = app.app_ctx
    stage = ctx.state.create_stage
    install_stubs(app, rec)

    ctx.state.mode = mode
    # The doctor banner, off. It is a property of the throwaway home this runs
    # against (no weights downloaded into it) rather than of the mode under
    # test, and it is a full-width strip across the top of every pane -- so
    # leaving it up makes the baseline image disagree with every later one by
    # its own height, and the pixel signal stops meaning anything. Its two
    # buttons are global, not this mode's; the report says they were skipped.
    ctx.state.dismiss_errors()
    settle(app)
    base_image = shoot(app, out / "00-baseline.png")
    baseline = digest(app)

    records: list[dict] = []
    pressed: set[str] = set()
    rounds = 0
    aborted = ""
    while rounds < max_rounds:
        rounds += 1
        fresh = _fresh(probe.census(), pressed)
        if not fresh:
            break
        for key, control in fresh:
            pressed.add(key)
            record = {
                "key": key,
                "label": control.name,
                "raw_label": control.label,
                "tooltip": control.tooltip,
                "kind": control.kind,
                "pane": control.pane,
                "window": control.window,
                "enabled": control.enabled,
                "reason": control.reason,
                "selected": control.selected,
                "visible": control.visible,
                # All three, because they differ: ``rect`` is the whole item,
                # ``hit`` the part a click lands on -- not the same for a field
                # imgui draws a label beside -- and ``click`` the point this
                # walk actually pressed. A control the pass calls ``inert`` is
                # diagnosed from these before anything else: a click x that
                # sits past the widget is a click that went onto the label.
                "rect": [round(v, 1) for v in control.rect],
                "hit": [round(v, 1) for v in control.hit],
                "click": [round(v, 1) for v in control.centre],
                "round": rounds,
                "shot": None,
                "submitted": [],
                "toasts": [],
                "raised": None,
                "state_delta": [],
                "pixel_delta": 0.0,
                "restored": "",
            }
            if refused(control.name):
                record["verdict"] = "refused"
                records.append(record)
                continue
            if any(one in key for one in skip):
                # Named on the command line, almost always because a previous
                # run found that this control wedges imgui: an exception out of
                # a pane leaves the frame unended and the id stack unbalanced,
                # and nothing after it can draw. Skipping it is how the rest of
                # the mode gets covered while the defect is still open.
                record["verdict"] = "skipped"
                records.append(record)
                continue
            if not control.visible or control.rect[2] <= 0 or control.rect[3] <= 0:
                # Not a failure of this pass: it is the finding. A control that
                # imgui clipped away cannot be clicked by anybody.
                record["verdict"] = "clipped"
                records.append(record)
                continue
            rec.reset()
            raised = None
            guard.HISTORY.clear()
            try:
                click(app, *control.centre)
                settle(app)
            except Exception:  # noqa: BLE001 -- catching it is the whole point
                raised = traceback.format_exc(limit=8)
                recover_frame()
            if raised is None and guard.HISTORY:
                # The pane guard caught it first. Since ``studio/guard.py``
                # landed, an exception inside a pane's draw is unwound and
                # replaced by a placeholder rather than reaching the ``except``
                # above -- which is right for a user and exactly wrong here,
                # because this script exists to find controls that crash. Left
                # unread, a control that takes its whole pane down would come
                # back "ok" with a tidy screenshot of the placeholder.
                raised = chr(10).join(f.traceback for f in guard.HISTORY)
            shot = f"{len(records):03d}-{safe_name(key)}.png"
            try:
                image = shoot(app, out / shot)
                delta = pixel_delta(base_image, image)
            except Exception:  # noqa: BLE001 -- a lost frame must not hide a crash
                shot, delta = None, 0.0
            state_delta = describe_delta(baseline, digest(app))
            record |= {
                "shot": shot,
                "submitted": list(rec.submitted),
                "toasts": [{"level": lvl, "text": txt} for lvl, txt in rec.toasts],
                "raised": raised,
                "state_delta": state_delta,
                "pixel_delta": round(delta, 5),
            }
            record["verdict"] = verdict(
                raised=raised,
                enabled=control.enabled,
                reason=control.reason,
                toast_levels=tuple(lvl for lvl, _ in rec.toasts),
                submitted=tuple(rec.submitted),
                state_delta=state_delta,
                pixel_delta=delta,
            )
            try:
                record["restored"] = restore(app, baseline, mode, stage)
            except Exception:  # noqa: BLE001
                # The app could not be got back to a usable state. Stop here
                # and write what we have: a manifest naming the control that
                # did it is the finding, and a run that died without one is
                # nothing at all.
                record["restored"] = "unrecoverable"
                record["recovery_error"] = traceback.format_exc(limit=8)
                records.append(record)
                aborted = record["key"]
                break
            if record["restored"] == "hard-reset" and record["verdict"] != "raised":
                record["verdict"] = "hard-reset"
            records.append(record)
        if aborted:
            break
    # What the bound dropped, if it was hit. Reported rather than swallowed: a
    # silent cap reads as full coverage.
    left = len(_fresh(probe.census(), pressed)) if rounds >= max_rounds or aborted else 0
    return {
        "records": records,
        "rounds": rounds,
        "frontier_left": left,
        "aborted_at": aborted,
    }


def exercise_palette(app, mode: str, rec: Recorder) -> list[dict]:
    """Record every palette command for this mode, and run the enabled ones.

    The palette is the non-visible half of the inventory: a command reachable
    only by Ctrl+K is a control no census of *drawn* items can see. A disabled
    command with an empty ``why`` is a defect by ``palette.Command``'s own
    docstring, so it is recorded whether or not it can be run.
    """
    from warlock.studio import palette

    ctx = app.app_ctx
    out: list[dict] = []
    for command in palette.commands(ctx):
        record: dict = {
            "key": command.key,
            "label": command.label,
            "group": command.group,
            "why": command.why,
            "kind": "palette",
            "enabled": False,
            "raised": None,
        }
        try:
            record["enabled"] = bool(command.enabled(ctx))
        except Exception:  # noqa: BLE001
            record |= {"verdict": "raised", "raised": traceback.format_exc(limit=8)}
            out.append(record)
            continue
        if not record["enabled"]:
            record["verdict"] = (
                "disabled" if command.why.strip() else "disabled-no-reason"
            )
            out.append(record)
            continue
        if refused(command.label):
            record["verdict"] = "refused"
            out.append(record)
            continue
        before = digest(app)
        rec.reset()
        try:
            command.run(ctx)
            pump(app)
            record["submitted"] = list(rec.submitted)
            record["toasts"] = [
                {"level": lvl, "text": txt} for lvl, txt in rec.toasts
            ]
            record["verdict"] = verdict(
                raised=None,
                enabled=True,
                reason="",
                toast_levels=tuple(lvl for lvl, _ in rec.toasts),
                submitted=tuple(rec.submitted),
                state_delta=describe_delta(before, digest(app)),
                # A palette command draws nothing of its own, so there is no
                # frame to difference: the other four signals are the whole of
                # what it can be judged on.
                pixel_delta=0.0,
            )
        except Exception:  # noqa: BLE001
            record |= {"verdict": "raised", "raised": traceback.format_exc(limit=8)}
        # Back to the mode under test: half of these commands are "Go to X".
        ctx.state.mode = mode
        pump(app)
        out.append(record)
    return out


#: Widget names that mean a control was drawn without going through
#: ``controls.py`` -- and so without passing ``_finish_item``, and so invisibly
#: to the probe.
_RAW_WIDGETS = frozenset(
    {
        "button",
        "small_button",
        "checkbox",
        "radio_button",
        "selectable",
        "collapsing_header",
        "menu_item",
        "menu_item_simple",
    }
)


def raw_imgui_controls() -> int:
    """How many controls bypass ``controls.py``, and so are invisible here.

    Counted rather than chased. The number belongs in the report: a coverage
    line that omits a known blind spot is the same lie as a silent cap.
    """
    import ast

    from warlock.studio import controls as controls_mod

    def records_itself(node) -> bool:
        return any(
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "record"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "probe"
            for call in ast.walk(node)
        )

    root = Path(controls_mod.__file__).resolve().parent
    total = 0
    for path in root.rglob("*.py"):
        if path.name == "controls.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # A raw call inside a function that records itself is reachable after
        # all -- ``widgets._button_with_note`` draws with ``imgui.button`` so
        # the primary and ghost fills it sits inside are not painted over, and
        # hands the census its own row. Counting it would overstate the gap.
        censused = {
            inner.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and records_itself(node)
            for inner in ast.walk(node)
            if hasattr(inner, "lineno")
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if getattr(node, "lineno", None) in censused:
                continue
            owner = node.func.value
            if (
                isinstance(owner, ast.Name)
                and owner.id == "imgui"
                and (
                    node.func.attr in _RAW_WIDGETS
                    or node.func.attr.startswith(("input_", "drag_", "slider_"))
                )
            ):
                total += 1
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    ap.add_argument(
        "--skip",
        action="append",
        default=[],
        metavar="SUBSTRING",
        help=(
            "do not press any control whose key contains this. Repeatable. "
            "For getting past a control a previous run found wedges imgui, so "
            "the rest of the mode is still covered."
        ),
    )
    args = ap.parse_args()

    if args.mode not in MODES:
        print(f"unknown mode {args.mode!r}; one of {', '.join(MODES)}", file=sys.stderr)
        return 2
    if os.environ.get("WARLOCK_UI_PROBE") != "1":
        print(
            "WARLOCK_UI_PROBE=1 is required: without it the census is empty.",
            file=sys.stderr,
        )
        return 2
    args.out.mkdir(parents=True, exist_ok=True)

    app = boot(args.scale)
    result: dict = {"records": [], "rounds": 0, "frontier_left": 0, "aborted_at": ""}
    try:
        seed_mode(app, args.mode)
        rec = Recorder()
        result = exercise(
            app, args.mode, args.out, args.max_rounds, rec, tuple(args.skip)
        )
        if not result["aborted_at"]:
            result["palette"] = exercise_palette(app, args.mode, rec)
    finally:
        # A wedged imgui must not eat the manifest: the record naming the
        # control that wedged it is the entire product of the run.
        with contextlib.suppress(Exception):
            app.teardown()
    result.setdefault("palette", [])

    result |= {
        "mode": args.mode,
        "skipped": list(args.skip),
        "always_look": list(ALWAYS_LOOK),
        "raw_imgui_controls": raw_imgui_controls(),
    }
    (args.out / "manifest.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    counts: dict[str, int] = {}
    for one in result["records"]:
        counts[one["verdict"]] = counts.get(one["verdict"], 0) + 1
    print(f"{len(result['records'])} controls in {result['rounds']} round(s)")
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")
    print(f"  palette commands: {len(result['palette'])}")
    print(f"  not probe-visible (raw imgui): {result['raw_imgui_controls']}")
    if result["aborted_at"]:
        print(f"  ABORTED at {result['aborted_at']} -- the app could not be recovered")
    if result["frontier_left"]:
        print(f"  NOT REACHED: {result['frontier_left']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
