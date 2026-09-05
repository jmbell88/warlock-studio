"""What a failed download says to the person who pressed the button.

Finding F2, 2026-09-05. Her machine reported

    ConnectError: [WinError 10054] An existing connection was forcibly closed
    by the remote host

five times, in a transient toast with no log button and no remedy, and every
useful thing about the incident had to be recovered from a log file she had to
be told how to find. ``fetch_worker.describe_failure`` is the fix and this file
is its claim: a transport error is translated into something actionable, an
exception this project wrote itself is passed through untouched, and anything
unrecognised says what it was and points at the log.
"""

from __future__ import annotations

import errno

# From ``download`` and never from ``fetch_worker``: importing the worker in
# this process sets ``HF_HUB_OFFLINE=0``, which is the one thing the offline
# invariant forbids and which two tests in ``test_fetch.py`` assert against.
# The translator lives in ``download`` precisely so it can be tested without
# that side effect.
from warlock.pipelines.download import describe_failure


def _reset(code: int = 10054) -> OSError:
    """The real shape: an OSError carrying a Windows socket number."""
    exc = OSError("An existing connection was forcibly closed by the remote host")
    exc.winerror = code
    return exc


def test_a_reset_names_the_things_that_actually_cause_it():
    said = describe_failure(_reset())
    assert "WinError 10054" in said, "the number the user can search for is gone"
    for word in ("antivirus", "firewall", "VPN", "proxy"):
        assert word.lower() in said.lower(), f"{word} is not offered as a cause"
    assert "forcibly closed by the remote host" not in said, (
        "the raw winsock text survived into the sentence"
    )


def test_a_reset_says_the_download_was_kept():
    """The other half of F1: pressing Install again is worth doing now."""
    said = describe_failure(_reset())
    assert "continues from where it stopped" in said


def test_a_wrapped_reset_is_still_recognised():
    """The transports bury it.

    The exception that reached the top on her machine was an ``httpx``
    ``ConnectError`` whose own attributes held nothing -- the number was two
    levels down the ``__cause__`` chain. A translator that only looked at the
    outermost exception would have produced the generic sentence.
    """

    class ConnectError(Exception):
        pass

    outer = ConnectError("[WinError 10054] An existing connection was forcibly closed")
    outer.__cause__ = _reset()
    assert "WinError 10054" in describe_failure(outer)


def test_dns_and_timeout_get_their_own_remedies():
    """Five conditions, five remedies -- they were one undifferentiated string."""
    lookup = describe_failure(_reset(11001))
    assert "DNS" in lookup and "antivirus" not in lookup.lower()
    timed = describe_failure(_reset(10060))
    assert "timed out" in timed.lower()
    refused = describe_failure(_reset(10061))
    assert "refused" in refused.lower()


def test_an_errno_reset_is_recognised_without_a_winerror():
    exc = ConnectionResetError(errno.ECONNRESET, "reset by peer")
    said = describe_failure(exc)
    assert "antivirus" in said.lower()


def test_a_sentence_this_project_wrote_is_passed_through_untouched():
    """A digest mismatch already reads like English and must not be reworded.

    ``_verify_staged`` and ``_fetch_url`` raise ``ValueError`` with a remedy in
    it; that is the case the old blanket ``f"{type(exc).__name__}: {exc}"``
    served well, and the translation must not cost it.
    """
    authored = ValueError(
        "acme/thing downloaded 2 file(s) that do not match the digests the hub "
        "recorded for them (a.bin, b.bin). Nothing was installed; try the "
        "download again."
    )
    assert describe_failure(authored) == str(authored)
    assert "ValueError" not in describe_failure(authored)


def test_an_unrecognised_failure_names_itself_and_points_at_the_log():
    """No guessing. The one thing worse than a raw exception is a wrong remedy."""
    said = describe_failure(RuntimeError("something else entirely"))
    assert "RuntimeError" in said and "something else entirely" in said
    assert "warlock.log" in said


def test_a_network_error_with_no_code_still_gets_a_remedy():
    """``hf_xet`` raises from Rust and carries no errno at all.

    The failure that ended the best attempt of 2026-09-05 was
    ``ConnectionError: Network error: Request middleware error ...`` on a Xet
    token refresh, with no number anywhere on it. It must not fall through to
    the "unrecognised" branch.
    """

    class ConnectionError_(Exception):
        pass

    ConnectionError_.__name__ = "ConnectionError"
    said = describe_failure(ConnectionError_("Network error: Request middleware error"))
    assert "online" in said.lower() and "warlock.log" not in said
