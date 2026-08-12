"""The model-store lease: the primitive the audit found missing.

The finding it answers is structural rather than a bug in any one line. Every
guard around model mutation was either a UI convention (the Settings pane
disables the other buttons while a fetch runs) or a check on the worker's event
loop -- and **event-loop serialization does not serialize the model thread**.
``Worker._generate`` spends its whole life awaiting ``asyncio.to_thread``, so
while ``from_pretrained`` is reading a checkpoint the loop is *idle*: an unload
scheduled onto it runs immediately, and a service callable can delete the very
files being read.

So the tests here are all about threads. A lease that only excluded coroutines
would pass a test written with coroutines and change nothing about the failure.
"""

from __future__ import annotations

import threading
import time

import pytest

from warlock.leases import ModelLease


def test_a_maintainer_waits_for_an_in_flight_model_operation():
    """The core guarantee: weights do not move while a thread is inside one.

    ``inside`` is the model thread's "I am in from_pretrained"; the maintainer
    must not get in until it leaves, however long that takes.
    """
    lease = ModelLease()
    order: list[str] = []
    inside = threading.Event()
    release = threading.Event()

    def user() -> None:
        with lease.use():
            order.append("use-start")
            inside.set()
            release.wait(5.0)
            order.append("use-end")

    thread = threading.Thread(target=user)
    thread.start()
    assert inside.wait(5.0)

    def maintainer() -> None:
        with lease.maintain(timeout=5.0):
            order.append("maintain")

    other = threading.Thread(target=maintainer)
    other.start()
    # Give the maintainer every chance to get in early. It must not.
    time.sleep(0.2)
    assert "maintain" not in order

    release.set()
    thread.join(5.0)
    other.join(5.0)
    assert order == ["use-start", "use-end", "maintain"]


def test_a_user_waits_for_an_active_maintainer():
    """The other direction: a job must not start reading weights that are being
    replaced. Staging makes a publish atomic per file, not per model."""
    lease = ModelLease()
    order: list[str] = []
    inside = threading.Event()
    release = threading.Event()

    def maintainer() -> None:
        with lease.maintain(timeout=5.0):
            order.append("maintain-start")
            inside.set()
            release.wait(5.0)
            order.append("maintain-end")

    thread = threading.Thread(target=maintainer)
    thread.start()
    assert inside.wait(5.0)

    def user() -> None:
        with lease.use():
            order.append("use")

    other = threading.Thread(target=user)
    other.start()
    time.sleep(0.2)
    assert "use" not in order

    release.set()
    thread.join(5.0)
    other.join(5.0)
    assert order == ["maintain-start", "maintain-end", "use"]


def test_a_waiting_maintainer_is_not_starved_by_a_stream_of_users():
    """Writer-preferring, deliberately. A store that can never be maintained
    because jobs keep arriving is the same bug approached from the other side."""
    lease = ModelLease()
    first = threading.Event()
    release_first = threading.Event()
    maintained = threading.Event()
    late_user_ran = threading.Event()

    def holder() -> None:
        with lease.use():
            first.set()
            release_first.wait(5.0)

    def maintainer() -> None:
        with lease.maintain(timeout=5.0):
            maintained.set()

    def late_user() -> None:
        with lease.use():
            late_user_ran.set()

    a = threading.Thread(target=holder)
    a.start()
    assert first.wait(5.0)
    b = threading.Thread(target=maintainer)
    b.start()
    time.sleep(0.1)  # let the maintainer register as waiting
    c = threading.Thread(target=late_user)
    c.start()
    time.sleep(0.1)
    assert not late_user_ran.is_set(), "a new user jumped the queued maintainer"

    release_first.set()
    for thread in (a, b, c):
        thread.join(5.0)
    assert maintained.is_set() and late_user_ran.is_set()


def test_use_is_reentrant_on_one_thread():
    """``generate`` takes the lease and then calls ``load``, which takes it
    again. Without re-entrancy a maintainer arriving between the two deadlocks
    the pair: the outer waits for the maintainer, the maintainer waits for the
    outer."""
    lease = ModelLease()
    with lease.use():
        with lease.use():
            assert lease.users == 1
        assert lease.users == 1
    assert lease.users == 0


def test_a_maintainer_gives_up_rather_than_hanging():
    """A wait with no end is a hang with a nicer name. The caller turns the
    timeout into its own layer's refusal -- the service raises ``Conflict``."""
    lease = ModelLease()
    inside = threading.Event()
    release = threading.Event()

    def user() -> None:
        with lease.use():
            inside.set()
            release.wait(5.0)

    thread = threading.Thread(target=user)
    thread.start()
    assert inside.wait(5.0)
    try:
        with pytest.raises(TimeoutError), lease.maintain(timeout=0.1):
            pass
        # And giving up leaves the lease usable rather than wedged.
        assert lease.users == 1
    finally:
        release.set()
        thread.join(5.0)
    with lease.maintain(timeout=1.0):
        pass


def test_a_failed_maintainer_releases_the_users_queued_behind_it():
    """The failure path has to notify too, or a user parked behind a maintainer
    that gave up waits for a condition nobody will signal again."""
    lease = ModelLease()
    inside = threading.Event()
    release = threading.Event()
    ran = threading.Event()

    def user() -> None:
        with lease.use():
            inside.set()
            release.wait(5.0)

    holder = threading.Thread(target=user)
    holder.start()
    assert inside.wait(5.0)

    def late_user() -> None:
        with lease.use():
            ran.set()

    def maintainer() -> None:
        with pytest.raises(TimeoutError), lease.maintain(timeout=0.2):
            pass

    b = threading.Thread(target=maintainer)
    b.start()
    time.sleep(0.05)
    c = threading.Thread(target=late_user)
    c.start()
    b.join(5.0)
    # The maintainer has given up; the queued user may now proceed even though
    # the original holder is still inside -- users do not exclude each other.
    assert ran.wait(5.0)
    release.set()
    holder.join(5.0)
    c.join(5.0)
