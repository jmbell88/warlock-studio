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
    journal.pump(ctx, now=1.0)  # first sight arms the debounce
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


def test_a_failed_write_does_not_strand_its_staging_file(tmp_path, monkeypatch):
    """The staged-write rule's second half (``service/derive.py::_staged``): a
    staging file is a dotfile and nothing ever sweeps one, so the unlink has to
    sit in a ``finally`` or a failed encode leaves it beside the journal for
    good."""
    import warlock.studio.journal as mod

    # The payload's own write failing: non-bytes data raises inside write_bytes.
    with pytest.raises(TypeError):
        mod._write_pair(tmp_path / "x.probe", None, {"version": journal.VERSION})
    assert list(tmp_path.glob(".*")) == []

    # And the sidecar's replace failing after the payload already landed.
    original = mod.os.replace

    def full_disk(src, dst):
        if str(dst).endswith(journal.META_SUFFIX):
            raise OSError("disk full")
        return original(src, dst)

    monkeypatch.setattr(mod.os, "replace", full_disk)
    with pytest.raises(OSError):
        mod._write_pair(tmp_path / "y.probe", b"body", {"version": journal.VERSION})
    monkeypatch.setattr(mod.os, "replace", original)
    assert (tmp_path / "y.probe").exists(), "the payload half still landed whole"
    assert not journal.meta_path(tmp_path / "y.probe").exists()
    assert list(tmp_path.glob(".*")) == [], "no .tmp stranded by the failure"


# --- the loop -----------------------------------------------------------------


def test_nothing_is_written_before_the_interval(tmp_path, kind):
    """From first-dirty, on the clock production actually runs on.

    ``pump``'s default stamp is ``time.monotonic()``, which is *boot*-relative
    -- hours or days, never seconds -- so a debounce that compared it against
    ``Mark``'s 0.0 default was a no-op outside a test with a synthetic session
    clock: the very first pump after an edit wrote. The first pump that sees a
    dirty slot arms the clock instead, and the copy comes JOURNAL_SECONDS
    later."""
    ctx = _Ctx(tmp_path)
    kind.slots.append(_Slot())
    boot = 3 * 86_400.0  # the shape monotonic really has: days of uptime
    assert journal.pump(ctx, now=boot) == 0, "first sight arms, never writes"
    assert journal.pump(ctx, now=boot + journal.JOURNAL_SECONDS - 1.0) == 0
    assert ctx.submitted == []
    assert journal.pump(ctx, now=boot + journal.JOURNAL_SECONDS + 1.0) == 1


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
    journal.pump(ctx, now=100.0)  # arm
    journal.pump(ctx, now=300.0)  # due; the submit is refused
    assert ctx.submitted == ["journal:probe:s1"]
    assert slot.journal_name == "" and slot.journal_head is None
    ctx.accept = True
    journal.pump(ctx, now=500.0)
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
    journal.pump(ctx, now=9_000.0)  # arm
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
    journal.pump(ctx, now=9_000.0)  # arm
    journal.pump(ctx, now=10_000.0)
    assert ctx.submitted == []
    assert slot.journal_name == ""


def test_a_failed_write_is_not_recorded_as_a_copy_that_exists(tmp_path, kind, monkeypatch):
    """The most direct data-loss path this codebase had.

    ``set_mark`` ran the instant ``ctx.submit`` was *accepted*, and the write
    landed later on a task thread. A full disk, a yanked directory or a
    permission change there left the slot recorded as journalled **at the
    current head** -- so ``_write_if_due``'s first gate saw nothing changed and
    no retry ever came, for the life of the session. The app believed a crash
    copy existed when none did, which is the one thing it cannot be wrong
    about: every other promise the journal makes is conditional on that one.

    The head is the task's to advance now. The debounce is still the frame
    thread's, because that is what stops a failing disk being re-encoded
    against at the frame rate.
    """
    import warlock.studio.journal as mod

    ctx = _Ctx(tmp_path)
    queued: list[Any] = []

    def defer(key: str, run: Any, *args: Any, **kwargs: Any) -> bool:
        ctx.submitted.append(key)
        queued.append(run)
        return True

    ctx.submit = defer
    slot = _Slot()
    kind.slots.append(slot)
    journal.pump(ctx, now=9_000.0)  # arm
    journal.pump(ctx, now=10_000.0)
    assert len(queued) == 1 and slot.journal_name, "the name is the debounce's"
    assert slot.journal_head is None, "nothing is on disk yet, so nothing is claimed"

    real_write_pair = mod._write_pair

    def full_disk(*_args: Any) -> None:
        raise OSError("no space left on device")

    monkeypatch.setattr(mod, "_write_pair", full_disk)
    with pytest.raises(OSError):
        queued[0]()
    assert slot.journal_head is None, "a write that failed is not a copy"

    # And the retry really arrives -- one interval later, not next frame.
    monkeypatch.setattr(mod, "_write_pair", real_write_pair)
    assert journal.pump(ctx, now=10_000.0 + journal.JOURNAL_SECONDS - 1.0) == 0
    assert journal.pump(ctx, now=10_000.0 + journal.JOURNAL_SECONDS + 1.0) == 1
    queued[-1]()
    assert slot.journal_head == slot.head, "the copy that landed is the one recorded"


def test_the_head_is_advanced_by_the_write_and_not_by_the_submit(tmp_path, kind):
    """The other side of the same rule, in the ordinary case: an accepted
    submit is not a copy, and the mark says so until the write returns."""
    ctx = _Ctx(tmp_path)
    queued: list[Any] = []
    ctx.submit = lambda key, run, *a, **k: (queued.append(run), True)[1]
    slot = _Slot()
    kind.slots.append(slot)
    journal.pump(ctx, now=9_000.0)
    journal.pump(ctx, now=10_000.0)
    assert slot.journal_head is None
    queued[0]()
    assert slot.journal_head == slot.head
    assert (tmp_path / slot.journal_name).read_bytes() == b"a"


def test_a_write_that_lands_inside_the_submit_keeps_its_head(tmp_path, kind):
    """The debounce mark is set *after* ``submit`` returns, and ``submit`` may
    have run the write to completion by then -- an inline runner always has, and
    on a small document (a pose, a profile) so may a real task thread. Setting
    it unconditionally lowered ``head`` back to the pre-write value, so the next
    tick saw an unwritten document and an *idle* one re-encoded itself every
    ``JOURNAL_SECONDS`` for the session."""
    ctx = _Ctx(tmp_path)  # its runner is inline: a submit is a write
    slot = _Slot()
    kind.slots.append(slot)
    journal.pump(ctx, now=9_000.0)
    journal.pump(ctx, now=10_000.0)
    assert slot.journal_head == slot.head, "the write landed and said so"

    # And nothing is written again while the document has not moved.
    journal.pump(ctx, now=20_000.0)
    journal.pump(ctx, now=30_000.0)
    assert len(ctx.submitted) == 1


def test_a_disk_that_fills_between_the_halves_publishes_neither(tmp_path, monkeypatch):
    """The sibling of the mark bug, and the reason both stagings come first.

    Written the obvious way -- stage the payload, rename it, stage the sidecar,
    rename it -- a disk that fills between the two leaves a *whole* first-ever
    copy on disk with no sidecar to offer it. ``recoverable`` is sidecar-driven
    by design, so that payload is present, permanently invisible and never
    swept: the work is there and the app will never mention it again.
    """
    import warlock.studio.journal as mod

    payload = tmp_path / "sketch-x.probe"
    real_write_text = Path.write_text

    def full_disk(self, *args, **kwargs):
        # The sidecar's *byte* write, which is the half that can run out of
        # room. Its rename cannot: by then both files are already on the disk.
        if self.name.endswith(f"{journal.META_SUFFIX}.tmp"):
            raise OSError("no space left on device")
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", full_disk)
    with pytest.raises(OSError):
        mod._write_pair(payload, b"the pixels", {"version": journal.VERSION})
    monkeypatch.undo()

    assert not payload.exists(), "an unofferable payload is not published at all"
    assert list(tmp_path.glob("*")) == [], "and no staging file is left behind"


def test_every_left_behind_document_is_offered_rather_than_the_first_eight(tmp_path, kind):
    """``recoverable`` returned ``out[:MAX_RECOVERY]`` and nothing pruned the
    rest, so a crash with twelve documents open buried four -- never listed,
    never offered, never swept. ``status_line`` is the sharpest form of it: the
    sentence the crash dialog shows counted the truncated list and told the
    user there were eight."""
    ctx = _Ctx(tmp_path)
    for i in range(journal.MAX_RECOVERY + 4):
        payload = tmp_path / f"doc{i}-x{i}.probe"
        payload.write_bytes(b"x")
        journal.meta_path(payload).write_text(
            json.dumps(
                {"version": journal.VERSION, "kind": "probe", "title": f"doc{i}", "at": i}
            ),
            encoding="utf-8",
        )
    assert len(journal.recoverable(ctx)) == journal.MAX_RECOVERY + 4
    assert f"{journal.MAX_RECOVERY + 4} unsaved documents" in journal.status_line(ctx)


# --- dropping -----------------------------------------------------------------


def test_dropping_removes_both_files_and_forgets_the_mark(tmp_path, kind):
    ctx = _Ctx(tmp_path)
    slot = _Slot()
    kind.slots.append(slot)
    journal.pump(ctx, now=9_000.0)  # arm
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
    journal.pump(ctx, now=9_000.0)  # arm
    journal.pump(ctx, now=10_000.0)
    (tmp_path / slot.journal_name).unlink()
    journal.drop(ctx, slot)


def test_a_write_still_queued_when_the_copy_is_dropped_does_not_resurrect_it(tmp_path, kind):
    """``drop`` unlinks on the frame thread, but the write it may be racing
    sits queued on a multi-worker pool: ``ctx.submit``'s key-dedupe stops two
    writes overlapping and sequences nothing against a drop. A copy submitted
    a moment before save-and-close used to be re-created after drop had
    deleted it -- and with the mark already reset, the orphaned pair was
    offered back on the next launch for a properly-saved document."""
    ctx = _Ctx(tmp_path)
    queued: list[Any] = []

    def defer(key: str, run: Any, *args: Any, **kwargs: Any) -> bool:
        ctx.submitted.append(key)
        queued.append(run)
        return True

    ctx.submit = defer
    slot = _Slot()
    kind.slots.append(slot)
    journal.pump(ctx, now=9_000.0)  # arm
    journal.pump(ctx, now=10_000.0)
    name = slot.journal_name
    assert name and len(queued) == 1
    journal.drop(ctx, slot)
    for run in queued:  # the pool only gets to the write after the drop
        run()
    payload = tmp_path / name
    assert not payload.exists()
    assert not journal.meta_path(payload).exists()


def test_a_write_in_flight_when_the_copy_is_dropped_does_not_re_mark_it(
    tmp_path, kind, monkeypatch
):
    """The half of the tombstone that moving the mark onto the write task opened.

    ``drop`` bumps the generation under this name's write lock, so a write
    already *inside* ``_write_pair`` makes it wait -- which is what lets the
    unlinks remove what that write just put down. The reset of the mark has to
    happen on the far side of that wait. Above it, the task's own ``set_mark``
    lands afterwards and records a mark for a document that has just been saved
    and closed, naming a payload the unlinks are about to remove: the slot then
    believes it holds a crash copy that is not there, which is the bug this
    whole batch is about, arriving by the other door.

    Threaded because the ordering is the property. A queued-but-unstarted write
    takes the generation branch and never reaches ``set_mark`` at all, so it
    passes either way and says nothing.
    """
    import threading

    import warlock.studio.journal as mod

    ctx = _Ctx(tmp_path)
    queued: list[Any] = []
    ctx.submit = lambda key, run, *a, **k: (queued.append(run), True)[1]
    slot = _Slot()
    kind.slots.append(slot)
    journal.pump(ctx, now=9_000.0)  # arm
    journal.pump(ctx, now=10_000.0)
    assert len(queued) == 1 and slot.journal_name

    entered, gate = threading.Event(), threading.Event()
    real = mod._write_pair

    def slow(payload, data, meta):
        entered.set()
        assert gate.wait(10.0)
        return real(payload, data, meta)

    monkeypatch.setattr(mod, "_write_pair", slow)
    writer = threading.Thread(target=queued[0], daemon=True)
    writer.start()
    assert entered.wait(10.0), "the write holds this name's lock"

    dropped = threading.Event()
    threading.Thread(
        target=lambda: (journal.drop(ctx, slot), dropped.set()), daemon=True
    ).start()
    assert not dropped.wait(0.3), "the drop waits for its own document's write"
    gate.set()
    assert dropped.wait(10.0)
    writer.join(10.0)

    assert slot.journal_name == "" and slot.journal_head is None
    assert list(tmp_path.glob("*.probe")) == [], "and the pair really went"


def test_a_fresh_copy_taken_after_a_drop_still_writes(tmp_path, kind):
    """The tombstone is a generation, not a flag: it kills exactly the writes
    submitted before the drop, and the next edit journals normally."""
    ctx = _Ctx(tmp_path)
    slot = _Slot()
    kind.slots.append(slot)
    journal.pump(ctx, now=9_000.0)  # arm
    journal.pump(ctx, now=10_000.0)
    journal.drop(ctx, slot)
    slot.head = 2
    journal.pump(ctx, now=20_000.0)  # re-arm after the drop
    journal.pump(ctx, now=20_000.0 + journal.JOURNAL_SECONDS + 1.0)
    payload = tmp_path / slot.journal_name
    assert payload.read_bytes() == b"a"
    assert journal.meta_path(payload).exists()


def test_a_drop_re_arms_the_debounce_rather_than_re_firing_it(tmp_path, kind):
    """``drop`` resets the mark to its never-seen default, and on the
    boot-relative production clock a zeroed ``at`` used to mean the *next*
    dirty pump wrote immediately -- a copy taken mid-first-stroke of the very
    document the user just saved. A dropped slot waits the full interval from
    the pump that next sees it dirty, like any other one."""
    ctx = _Ctx(tmp_path)
    slot = _Slot()
    kind.slots.append(slot)
    journal.pump(ctx, now=9_000.0)  # arm
    journal.pump(ctx, now=10_000.0)
    journal.drop(ctx, slot)
    slot.head = 2
    assert journal.pump(ctx, now=20_000.0) == 0, "first sight after the drop arms"
    assert journal.pump(ctx, now=20_000.0 + journal.JOURNAL_SECONDS - 1.0) == 0
    assert journal.pump(ctx, now=20_000.0 + journal.JOURNAL_SECONDS + 1.0) == 1


def test_dropping_one_document_does_not_wait_on_anothers_write(tmp_path, kind, monkeypatch):
    """``drop`` runs on the frame thread, and the frame loop never blocks. The
    write task holds a lock across its whole disk write so a drop of the *same*
    document removes what it just wrote -- but that lock is per payload name,
    because a module-wide one parked closing a saved sketch behind an unrelated
    map's multi-MB write."""
    import threading

    import warlock.studio.journal as mod

    ctx = _Ctx(tmp_path)
    queued: list[Any] = []
    ctx.submit = lambda key, run, *a, **k: (queued.append(run), True)[1]
    a, b = _Slot(uid="a", title="doc-a"), _Slot(uid="b", title="doc-b")
    kind.slots.extend([a, b])
    journal.pump(ctx, now=9_000.0)  # arm both
    journal.pump(ctx, now=10_000.0)
    assert len(queued) == 2 and a.journal_name and b.journal_name

    entered, gate = threading.Event(), threading.Event()
    real = mod._write_pair

    def slow(payload, data, meta):
        if b.journal_name in payload.name or payload.name == b.journal_name:
            entered.set()
            assert gate.wait(10.0)
        return real(payload, data, meta)

    monkeypatch.setattr(mod, "_write_pair", slow)
    b_writer = threading.Thread(target=queued[1], daemon=True)  # b's queued write
    b_writer.start()
    assert entered.wait(10.0), "b's write is in flight and holding its own lock"

    dropped = threading.Event()
    dropper = threading.Thread(
        target=lambda: (journal.drop(ctx, a), dropped.set()), daemon=True
    )
    dropper.start()
    assert dropped.wait(10.0), "dropping a must not wait for b's write to land"
    assert entered.is_set() and not gate.is_set(), "b's write was still in flight"

    # And the same-name half of the contract survives the narrowing: dropping b
    # waits for b's own in-flight write, so the unlinks remove what it wrote.
    b_name = b.journal_name
    b_dropped = threading.Event()
    b_dropper = threading.Thread(
        target=lambda: (journal.drop(ctx, b), b_dropped.set()), daemon=True
    )
    b_dropper.start()
    assert not b_dropped.wait(0.3), "b's drop waits on b's own write"
    gate.set()
    assert b_dropped.wait(10.0)
    b_writer.join(10.0)
    assert not (tmp_path / b_name).exists(), "the drop removed what the write landed"
    assert list(tmp_path.glob("*.probe")) == [], "both pairs are gone"


# --- the offer ----------------------------------------------------------------


def test_every_kind_is_offered_and_each_on_its_own_row(tmp_path, kind):
    """The offer covers all six document kinds and always did; what changed is
    that it is a row each rather than one Confirm naming three of them and
    counting the rest. The old shape could not express "reopen this one".
    """
    ctx = _Ctx(tmp_path)
    ctx.state.recovery = None
    for i, title in enumerate(("sketch", "level", "atlas")):
        payload = tmp_path / f"{title}-{i}.probe"
        payload.write_bytes(b"x")
        journal.meta_path(payload).write_text(
            json.dumps(
                {"version": journal.VERSION, "kind": "probe", "title": title, "at": i}
            ),
            encoding="utf-8",
        )
    assert sorted(r.title for r in journal.snapshot(ctx)) == ["atlas", "level", "sketch"]


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


def test_ignoring_the_offer_keeps_the_files(tmp_path, kind):
    """"Not now" is not "delete my work", and on a panel "not now" is simply
    not clicking anything -- so the do-nothing path is the one worth pinning.
    ``discard`` is now the only thing that deletes, and it is a button.
    """
    ctx = _Ctx(tmp_path)
    ctx.state.recovery = None
    payload = tmp_path / "sketch-x.probe"
    payload.write_bytes(b"x")
    journal.meta_path(payload).write_text(
        json.dumps({"version": journal.VERSION, "kind": "probe", "title": "s", "at": 1})
    , encoding="utf-8")
    assert len(journal.snapshot(ctx)) == 1
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
    journal.pump(first, now=1.0)  # first sight arms the debounce
    journal.pump(first, now=journal.JOURNAL_SECONDS + 1.0)
    del first  # the crash

    second = _Ctx(tmp_path)
    second.state.recovery = None
    offered = journal.snapshot(second)
    assert [r.title for r in offered] == ["work in progress"]
    assert journal.take(second, offered[0]) is True
    assert len(adopted) == 1
    assert adopted[0].read_bytes() == b"the pixels"
    assert journal.snapshot(second) == [], "the row leaves the list once taken"


def test_the_status_line_is_computed_rather_than_promised(tmp_path, kind):
    """UX-06's sentence. "Your work is safe" is only worth saying when it is
    true, and the crash dialog runs in a process on its way out that may have
    nothing left to ask."""
    ctx = _Ctx(tmp_path)
    assert "No unsaved work" in journal.status_line(ctx)
    kind.slots.append(_Slot())
    journal.pump(ctx, now=9_000.0)  # arm
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
    assert kinds >= {"inker", "clay", "plotter", "packwright", "pose"}


@pytest.mark.parametrize(
    "module,cls",
    [
        ("warlock.studio.inker_state", "InkerDoc"),
        ("warlock.studio.clay_state", "ClayTab"),
        ("warlock.studio.plotter_state", "PlotterDoc"),
        ("warlock.studio.packwright_state", "PackTab"),
    ],
)
def test_every_journalled_state_class_declares_all_three_mark_fields(module, cls):
    """``set_mark`` writes three attributes and ``mark_of`` reads them back
    through ``getattr`` defaults -- so a slot class that *drops* one of them
    keeps working, silently, with the debounce or the head reset to its default
    on every restart. That is exactly how ``PlotterDoc.journal_at`` went missing
    on 2026-08-18: it sat directly above a block of fields that were deleted
    with the ground set, and nothing anywhere went red.

    Declared fields rather than a round-trip through ``set_mark``, because the
    round-trip is what passes either way. These four are the dataclasses; the
    pose and profile slots carry the same three as properties over state that
    outlives one document, which is checked by their own modes' tests.
    """
    import dataclasses
    import importlib

    target = getattr(importlib.import_module(module), cls)
    names = {f.name for f in dataclasses.fields(target)}
    assert {"journal_name", "journal_head", "journal_at"} <= names, sorted(names)


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


def test_a_map_with_a_layer_tree_journals_and_comes_back():
    """The plotter provider encodes through ``wmap.wmap_bytes``, so what the
    journal can save is exactly what that format can store -- and until version
    3 that excluded a group, an image layer and every per-layer decoration.
    A crash while the user had a nested map open wrote *nothing at all* for
    that tab (``_journal_write`` catches the encode and skips the slot), which
    is the one failure mode the whole mechanism exists to prevent. This is the
    case that says it is over; imported inside the test because the mode is one
    of the four engines this file's import-discipline case forbids at module
    scope."""
    import numpy as np

    from warlock.studio import plotter_mode
    from warlock.studio.plotter import wmap
    from warlock.studio.plotter.tilemap import MapDoc

    doc = MapDoc(4, 4, 16, 16)
    group = doc.add_group_layer("G")
    doc.add_tile_layer("Ground", parent_uid=group.uid)
    doc.add_image_layer(
        "Sky", pixels=np.zeros((2, 2, 4), dtype=np.uint8), parent_uid=group.uid
    )
    doc.set_layer_props(group.uid, tint=(255, 0, 0, 255), offset_x=4.0, class_name="Deco")

    payload = plotter_mode._journal_encode(SimpleNamespace(doc=doc))
    back = wmap.read_wmap(payload)
    recovered = back.layers[0]
    assert (recovered.name, recovered.class_name, recovered.offset_x) == ("G", "Deco", 4.0)
    assert [child.name for child in recovered.children] == ["Ground", "Sky"]


# --- the home-screen offer ----------------------------------------------------
#
# The offer used to be a modal on the first frame: one question, all-or-nothing,
# asked before the user had seen the app. It is a section at the top of the home
# screen now, one row per document, each recovered on its own. What did *not*
# change is when the directory is read -- see ``snapshot``.


def _home_ctx(root: Path) -> _Ctx:
    ctx = _Ctx(root)
    ctx.state.recovery = None
    return ctx


def _left_behind(root: Path, title: str, kind: str = "probe") -> Path:
    payload = root / f"{title}.probe"
    payload.write_bytes(b"body")
    journal.meta_path(payload).write_text(
        json.dumps(
            {"version": journal.VERSION, "kind": kind, "title": title, "uid": title, "at": 1.0}
        ),
        encoding="utf-8",
    )
    return payload


def test_the_snapshot_is_taken_once_and_never_refreshed(tmp_path, kind):
    """The load-bearing half of moving the offer out of a modal.

    The autosave directory is *also* where this session writes its own copies,
    so a home screen that rescanned would start listing the user's open
    documents back at them as though they had been crashed out of. Reading once,
    on the first frame, is what made the modal correct and is the one thing the
    move has to preserve.
    """
    ctx = _home_ctx(tmp_path)
    _left_behind(tmp_path, "before")
    assert [r.title for r in journal.snapshot(ctx)] == ["before"]

    _left_behind(tmp_path, "written-by-this-session")
    assert [r.title for r in journal.snapshot(ctx)] == ["before"], "not rescanned"


def test_an_empty_directory_still_counts_as_scanned(tmp_path, kind):
    """Nothing found is an answer, not an absent one -- a falsy-not-None test
    here would rescan every frame, which is the bug above with extra steps."""
    ctx = _home_ctx(tmp_path)
    assert journal.snapshot(ctx) == []
    _left_behind(tmp_path, "later")
    assert journal.snapshot(ctx) == []


def test_recovering_one_row_leaves_the_others_alone(tmp_path, kind):
    """Per-row is the whole point of the move. The old modal took all of them
    or none, so a session that crashed with one document worth keeping and two
    worth abandoning had no answer that was right."""
    ctx = _home_ctx(tmp_path)
    _left_behind(tmp_path, "keep")
    _left_behind(tmp_path, "other")
    found = journal.snapshot(ctx)
    assert len(found) == 2

    wanted = next(r for r in found if r.title == "keep")
    assert journal.take(ctx, wanted) is True
    assert [r.title for r in journal.snapshot(ctx)] == ["other"]


def test_recovering_a_row_leaves_its_files_for_the_document_to_drop(tmp_path, kind):
    """``drop`` is the owner of deletion and fires when the recovered document
    is saved or closed. Deleting here would take the copy away in the window
    between reopening it and saving it -- which is exactly the window a second
    crash would land in."""
    ctx = _home_ctx(tmp_path)
    payload = _left_behind(tmp_path, "keep")
    journal.take(ctx, journal.snapshot(ctx)[0])
    assert payload.exists()
    assert journal.meta_path(payload).exists()


def test_discarding_a_row_removes_the_pair_sidecar_first(tmp_path, kind):
    """``drop``'s ordering, for ``drop``'s reason: the sidecar is the
    completion gate, so unlinking it first means a crash between the two leaves
    an unoffered payload rather than a sidecar naming a file that has gone."""
    ctx = _home_ctx(tmp_path)
    payload = _left_behind(tmp_path, "junk")
    _left_behind(tmp_path, "other")

    journal.discard(ctx, journal.snapshot(ctx)[0])
    gone = [r for r in journal.snapshot(ctx)]
    assert len(gone) == 1
    survivor = gone[0]
    assert survivor.path.exists()
    assert not (payload.exists() and journal.meta_path(payload).exists())


def test_discarding_never_raises_when_the_files_have_already_gone(tmp_path, kind):
    """It is cleanup, not an edit -- ``drop``'s rule. A row whose files another
    process removed must still leave the list."""
    ctx = _home_ctx(tmp_path)
    payload = _left_behind(tmp_path, "junk")
    found = journal.snapshot(ctx)[0]
    journal.meta_path(payload).unlink()
    payload.unlink()

    journal.discard(ctx, found)
    assert journal.snapshot(ctx) == []
