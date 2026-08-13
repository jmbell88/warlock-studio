"""The document journal: one crash-recovery mechanism for six document kinds.

UX-05. Inker had a crash-safe autosave since it shipped and nothing else did —
a Clay model, a Plotter map, a Packwright atlas, a pose being authored and a
profile draft were all one power cut away from gone. The mechanism was right;
the whole of what was wrong was that it lived in one mode.

``tests/test_inker_mode.py`` still owns the Inker-shaped assertions (they are
the ones that pin the migration behaved identically). This file owns the parts
that are about the *journal*: the completion gate, the storage discipline, the
provider protocol and the crash-to-recovery span.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from warlock.studio import journal


class _Confirms:
    def __init__(self) -> None:
        self.pending: Any = None

    def ask(self, confirm: Any) -> None:
        self.pending = confirm

    def accept(self) -> None:
        pending, self.pending = self.pending, None
        if pending is not None:
            pending.on_confirm()

    def dismiss(self) -> None:
        self.pending = None


class _Ctx:
    """A Ctx with a task runner that runs inline, so a submit is a write."""

    def __init__(self, root: Path, *, accept: bool = True) -> None:
        self.svc = SimpleNamespace(config=SimpleNamespace(autosave_dir=root))
        self.state = SimpleNamespace()
        self.confirms = _Confirms()
        self.submitted: list[str] = []
        self.toasts: list[tuple[str, str]] = []
        self.accept = accept

    def submit(self, key: str, run: Any, *args: Any, **kwargs: Any) -> bool:
        self.submitted.append(key)
        if not self.accept:
            return False
        run(*args, **kwargs)
        return True

    def toast(self, text: str, level: str = "info", **_kw: Any) -> None:
        self.toasts.append((text, level))


class _Slot:
    """A minimal journallable thing: the three marks and something to encode."""

    def __init__(self, uid: str = "s1", title: str = "thing", body: bytes = b"a") -> None:
        self.uid = uid
        self.title = title
        self.body = body
        self.head = 1
        self.journal_name = ""
        self.journal_head = None
        self.journal_at = 0.0


@pytest.fixture
def kind(monkeypatch):
    """A registered test provider, removed again afterwards.

    Registered into the real table because ``pump`` walks that table, and
    restored by hand because the registry is module state that outlives a test
    -- the same reason ``matting``'s cache fixture exists.
    """
    slots: list[_Slot] = []
    provider = journal.Provider(
        kind="probe",
        ext=".probe",
        label="probe",
        slots=lambda ctx: list(slots),
        uid_of=lambda s: s.uid,
        title_of=lambda s: s.title,
        head_of=lambda s: s.head,
        encode=lambda s: s.body,
        adopt=lambda ctx, path, meta: True,
    )
    before = dict(journal._PROVIDERS)
    journal.register(provider)
    yield SimpleNamespace(provider=provider, slots=slots)
    journal._PROVIDERS.clear()
    journal._PROVIDERS.update(before)


# --- the completion gate ------------------------------------------------------


def test_a_copy_is_a_payload_and_a_sidecar(tmp_path, kind):
    ctx = _Ctx(tmp_path)
    slot = _Slot()
    kind.slots.append(slot)
    journal.pump(ctx, now=journal.JOURNAL_SECONDS + 1.0)
    payload = tmp_path / slot.journal_name
    assert payload.read_bytes() == b"a"
    meta = json.loads(journal.meta_path(payload).read_text(encoding="utf-8"))
    assert meta["kind"] == "probe" and meta["title"] == "thing"


def test_the_sidecar_is_written_last_and_is_what_offers_the_payload(tmp_path, kind):
    """The gate. tmp+replace makes each file whole or absent; writing the
    sidecar second makes the *pair* atomic, so a crash between them leaves a
    payload nothing offers -- which is right, because it is a copy that was
    interrupted mid-write."""
    ctx = _Ctx(tmp_path)
    (tmp_path / "orphan-x.probe").write_bytes(b"interrupted")
    assert journal.recoverable(ctx) == []


def test_both_files_go_through_a_temp_name(tmp_path):
    """A crash *during* the crash copy must not leave a truncated file where a
    whole one used to be, which is the one outcome worse than no copy."""
    payload = tmp_path / "x.probe"
    import warlock.studio.journal as mod

    original = mod.os.replace
    replaced: list[tuple[str, str]] = []
    mod.os.replace = lambda a, b: replaced.append((Path(a).name, Path(b).name)) or original(a, b)
    try:
        mod._write_pair(payload, b"body", {"version": journal.VERSION, "kind": "probe"})
    finally:
        mod.os.replace = original
    assert [dst for _src, dst in replaced] == ["x.probe", "x.probe.meta.json"]
    assert all(src.startswith(".") and src.endswith(".tmp") for src, _dst in replaced)


# --- the loop -----------------------------------------------------------------


def test_nothing_is_written_before_the_interval(tmp_path, kind):
    """From zero, not from the first copy: a document dirtied in the first two
    minutes of a session waits like every other one."""
    ctx = _Ctx(tmp_path)
    kind.slots.append(_Slot())
    assert journal.pump(ctx, now=0.0) == 0
    assert ctx.submitted == []


def test_an_unchanged_slot_is_not_rewritten(tmp_path, kind):
    """Gated on the head rather than on a flag, so an undo back to the copied
    position is not a new edit either."""
    ctx = _Ctx(tmp_path)
    slot = _Slot()
    kind.slots.append(slot)
    journal.pump(ctx, now=200.0)
    journal.pump(ctx, now=400.0)
    assert len(ctx.submitted) == 1
    slot.head = 2
    journal.pump(ctx, now=600.0)
    assert len(ctx.submitted) == 2


def test_a_refused_submit_is_retried_rather_than_recorded_as_done(tmp_path, kind):
    """``submit`` refuses a key already in flight, which is the backpressure --
    a slow encode skips a beat. Advancing the mark there would turn the skip
    into a copy that never happened."""
    ctx = _Ctx(tmp_path, accept=False)
    slot = _Slot()
    kind.slots.append(slot)
    journal.pump(ctx, now=200.0)
    assert slot.journal_name == "" and slot.journal_head is None
    ctx.accept = True
    journal.pump(ctx, now=400.0)
    assert slot.journal_name


def test_one_broken_provider_does_not_stop_the_others(tmp_path, kind):
    """This runs every frame in every mode. One kind taking the whole journal
    down is the failure the journal exists to prevent."""
    broken = journal.Provider(
        kind="broken",
        ext=".x",
        label="x",
        slots=lambda ctx: (_ for _ in ()).throw(RuntimeError("no")),
        uid_of=lambda s: "s",
        title_of=lambda s: "s",
        head_of=lambda s: 1,
        encode=lambda s: b"",
        adopt=lambda *a: True,
    )
    journal.register(broken)
    ctx = _Ctx(tmp_path)
    kind.slots.append(_Slot())
    assert journal.pump(ctx, now=10_000.0) == 1


def test_an_encode_that_raises_writes_nothing_and_does_not_mark(tmp_path, kind):
    ctx = _Ctx(tmp_path)
    slot = _Slot()
    slot.body = None  # encode returns non-bytes -> the write raises
    kind.slots.append(slot)

    def boom(_s):
        raise RuntimeError("cannot encode")

    journal.register(
        journal.Provider(
            **{**kind.provider.__dict__, "encode": boom},
        )
    )
    journal.pump(ctx, now=10_000.0)
    assert ctx.submitted == []
    assert slot.journal_name == ""


# --- dropping -----------------------------------------------------------------


def test_dropping_removes_both_files_and_forgets_the_mark(tmp_path, kind):
    ctx = _Ctx(tmp_path)
    slot = _Slot()
    kind.slots.append(slot)
    journal.pump(ctx, now=10_000.0)
    payload = tmp_path / slot.journal_name
    journal.drop(ctx, slot)
    assert not payload.exists()
    assert not journal.meta_path(payload).exists()
    assert slot.journal_name == ""


def test_dropping_something_that_was_never_copied_does_nothing(tmp_path, kind):
    journal.drop(_Ctx(tmp_path), _Slot())


def test_dropping_a_copy_already_gone_does_not_raise(tmp_path, kind):
    """Cleanup, not an edit."""
    ctx = _Ctx(tmp_path)
    slot = _Slot()
    kind.slots.append(slot)
    journal.pump(ctx, now=10_000.0)
    (tmp_path / slot.journal_name).unlink()
    journal.drop(ctx, slot)


# --- the offer ----------------------------------------------------------------


def test_one_question_covers_every_kind(tmp_path, kind):
    """4a: one Confirm for the lot. Per-row choosing is a real want and a
    bigger dialog; what it cannot be is the first version, because the common
    case is one crash and one or two documents."""
    ctx = _Ctx(tmp_path)
    for i, title in enumerate(("sketch", "level", "atlas")):
        payload = tmp_path / f"{title}-{i}.probe"
        payload.write_bytes(b"x")
        journal.meta_path(payload).write_text(
            json.dumps(
                {"version": journal.VERSION, "kind": "probe", "title": title, "at": i}
            ),
            encoding="utf-8",
        )
    assert journal.offer(ctx) is True
    assert ctx.confirms.pending is not None
    message = ctx.confirms.pending.message
    for title in ("sketch", "level", "atlas"):
        assert title in message


def test_the_newest_is_listed_first(tmp_path, kind):
    ctx = _Ctx(tmp_path)
    for title, at in (("older", 1.0), ("newer", 9.0)):
        payload = tmp_path / f"{title}-x.probe"
        payload.write_bytes(b"x")
        journal.meta_path(payload).write_text(
            json.dumps(
                {"version": journal.VERSION, "kind": "probe", "title": title, "at": at}
            ),
            encoding="utf-8",
        )
    assert [r.title for r in journal.recoverable(ctx)] == ["newer", "older"]


def test_declining_keeps_the_files(tmp_path, kind):
    """"Not now" is not "delete my work"."""
    ctx = _Ctx(tmp_path)
    payload = tmp_path / "sketch-x.probe"
    payload.write_bytes(b"x")
    journal.meta_path(payload).write_text(
        json.dumps({"version": journal.VERSION, "kind": "probe", "title": "s", "at": 1})
    , encoding="utf-8")
    journal.offer(ctx)
    ctx.confirms.dismiss()
    assert payload.exists() and journal.meta_path(payload).exists()


def test_an_adopter_that_declines_is_counted_honestly(tmp_path, kind):
    """A pose whose rig is not open declines and says so; the file stays, and
    ``adopt`` must not claim it took something it did not."""
    journal.register(
        journal.Provider(**{**kind.provider.__dict__, "adopt": lambda *a: False})
    )
    ctx = _Ctx(tmp_path)
    payload = tmp_path / "p-x.probe"
    payload.write_bytes(b"x")
    journal.meta_path(payload).write_text(
        json.dumps({"version": journal.VERSION, "kind": "probe", "title": "p", "at": 1})
    , encoding="utf-8")
    assert journal.adopt(ctx, journal.recoverable(ctx)) == 0
    assert payload.exists()


# --- the whole span, 1 through 5 ----------------------------------------------


def test_a_crash_between_the_copy_and_the_next_launch_gives_the_work_back(tmp_path, kind):
    """The end-to-end property, simulated by throwing the session away.

    Session one journals a slot and then stops existing -- no save, no close,
    no drop, which is exactly what a power cut leaves. Session two is a fresh
    ctx over the same directory, and it must find the work, offer it, and hand
    it to the provider.
    """
    adopted: list[Path] = []
    journal.register(
        journal.Provider(
            **{
                **kind.provider.__dict__,
                "adopt": lambda ctx, path, meta: bool(adopted.append(path)) or True,
            }
        )
    )
    first = _Ctx(tmp_path)
    slot = _Slot(title="work in progress", body=b"the pixels")
    kind.slots.append(slot)
    journal.pump(first, now=journal.JOURNAL_SECONDS + 1.0)
    del first  # the crash

    second = _Ctx(tmp_path)
    assert journal.offer(second) is True
    assert "work in progress" in second.confirms.pending.message
    second.confirms.accept()
    assert len(adopted) == 1
    assert adopted[0].read_bytes() == b"the pixels"


def test_the_status_line_is_computed_rather_than_promised(tmp_path, kind):
    """UX-06's sentence. "Your work is safe" is only worth saying when it is
    true, and the crash dialog runs in a process on its way out that may have
    nothing left to ask."""
    ctx = _Ctx(tmp_path)
    assert "No unsaved work" in journal.status_line(ctx)
    kind.slots.append(_Slot())
    journal.pump(ctx, now=10_000.0)
    line = journal.status_line(ctx)
    assert "1 unsaved document" in line


def test_the_status_line_never_raises(tmp_path):
    """It is called in a ``finally`` on the way out of a process that is
    already failing."""
    broken = SimpleNamespace(svc=SimpleNamespace(config=SimpleNamespace()))
    assert journal.status_line(broken)


# --- the discipline this module has to keep -----------------------------------


def test_the_journal_imports_nothing_heavy_at_module_scope():
    """``docmodes``' rule, and for its reason: this is called from the frame
    loop in every mode, including the ones that have loaded none of the
    editors. A top-level import of an engine would make opening the app pay for
    all four of them."""
    import subprocess
    import sys

    code = (
        "import warlock.studio.journal, sys; "
        "bad = [m for m in sys.modules if m.startswith('warlock.studio.') "
        "and m.split('.')[2] in ('inker', 'clay', 'plotter', 'packwright')]; "
        "print(','.join(sorted(bad)))"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == ""


def test_every_real_provider_is_registered_by_ensure():
    """The startup offer runs before any mode has been opened, so registration
    as a side effect of importing a mode is not enough on its own."""
    journal.ensure_providers()
    kinds = {p.kind for p in journal.providers()}
    assert kinds >= {"inker", "clay", "plotter", "packwright", "pose", "profile"}


def test_no_two_kinds_share_a_name_or_an_extension():
    """A shared extension would let two providers claim one file, and the
    sidecar's ``kind`` would be the only thing distinguishing them."""
    journal.ensure_providers()
    all_of = journal.providers()
    assert len({p.kind for p in all_of}) == len(all_of)
    assert len({p.ext for p in all_of}) == len(all_of)


@pytest.mark.parametrize("suffix", [".ora", ".wblk", ".wmap", ".wpack"])
def test_each_document_kind_writes_its_own_format(suffix: str):
    """A recovered file is openable by hand and by the mode's ordinary reader,
    which is what makes a crash copy inspectable rather than opaque."""
    journal.ensure_providers()
    assert suffix in {p.ext for p in journal.providers()}
