"""How this application identifies itself when it downloads something.

One constant and one helper, and they exist because the *absence* of them was
a 403. On 2026-09-05 ``scripts/make_packs.py`` stopped being able to collect
torch, torchaudio and torchvision: the URLs ``uv.lock`` records for those three
are on ``download-r2.pytorch.org``, and Cloudflare answers Python's default
``Python-urllib/3.13`` agent there with 403 and ``error code: 1010`` -- "banned
based on your browser's signature". Every other host involved, both
``files.pythonhosted.org`` and ``download.pytorch.org``, still serves that
agent, which is why it surfaced as three files failing rather than as no
network at all.

**The build was the cheap half of that bug.** The same default agent was in
``pack_worker``, which downloads those same three wheels *on the user's
machine* when they choose Create, Poser or Muse from Settings -> Packs -- so
the shipped installer's pack flow was one Cloudflare rule away from refusing
every heavy pack, with a traceback rather than an explanation. ``fetch_worker``
downloads model weights from hosts that are not affected today, and is here
because "not affected today" is the whole reason to have one spelling.

Any non-default agent is accepted; uv's, pip's, curl's and a browser's were all
tried against the banned URL and all returned 206. This one names the
application and its version, which is what a server operator looking at their
logs would want to see.
"""

from __future__ import annotations

import errno
import urllib.request
from typing import Any

from .. import __version__

USER_AGENT = f"Warlock-Studio/{__version__} (+https://github.com/jmbell88/warlock)"


def request(url: str) -> urllib.request.Request:
    """The URL as a request that identifies itself."""
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def open_url(url: str, *, timeout: float | None = None) -> Any:
    """``urlopen`` with this application's agent attached."""
    if timeout is None:
        return urllib.request.urlopen(request(url))
    return urllib.request.urlopen(request(url), timeout=timeout)


#: The exceptions the fetch worker raises for itself, each already carrying a
#: remedy written for a person: a digest mismatch, a missing rename source, a
#: registry entry with no digest. Named here because :func:`describe_failure`
#: is what has to tell them from a transport error, and imported back by the
#: worker as its terminal set so the two cannot disagree.
AUTHORED = (ValueError, FileNotFoundError)


#: Windows socket errors this project has actually seen, mapped to what a
#: person can do about them. Keyed on ``winerror`` because that is the number
#: Windows reports and the one a user can search for; the ``errno`` fallbacks
#: below cover the same conditions where Python normalises them.
#:
#: 10054 is the one the 2026-09-05 clean-machine install produced, against
#: several hosts, on a machine with working internet -- so the remedy names the
#: things that reset a connection from the near side rather than blaming the
#: network.
_WINSOCK_REMEDIES = {
    10054: (
        "The connection was closed by the far end part-way through. This is "
        "most often antivirus, a firewall, a VPN or a workplace proxy "
        "inspecting the transfer; try pausing them, or a different network."
    ),
    10060: (
        "The connection timed out. The server may be busy, or something "
        "between here and it is dropping the transfer."
    ),
    10061: (
        "The connection was refused. If you are behind a proxy, it may not be "
        "configured for this application."
    ),
    11001: (
        "The server's address could not be looked up. Check that this machine "
        "is online and that its DNS is working."
    ),
}

#: The same conditions where they arrive as an ``errno`` instead -- a
#: ``ConnectionResetError`` raised through Python's own socket layer carries
#: ``ECONNRESET`` and no ``winerror``.
_ERRNO_REMEDIES = {
    errno.ECONNRESET: _WINSOCK_REMEDIES[10054],
    errno.ETIMEDOUT: _WINSOCK_REMEDIES[10060],
    errno.ECONNREFUSED: _WINSOCK_REMEDIES[10061],
}


def _socket_code(exc: BaseException) -> tuple[int | None, int | None]:
    """The ``winerror``/``errno`` pair on ``exc`` or anything it wraps.

    Walks ``__cause__`` and ``__context__`` because the transports bury it: the
    reset that stopped the 2026-09-05 run reached the top as an ``httpx``
    ``ConnectError``, whose own string carried the number while the attributes
    that hold it sat two levels down.
    """
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        win = getattr(exc, "winerror", None)
        num = getattr(exc, "errno", None)
        if isinstance(win, int) or isinstance(num, int):
            return (win if isinstance(win, int) else None, num if isinstance(num, int) else None)
        exc = exc.__cause__ or exc.__context__
    return (None, None)


def describe_failure(exc: BaseException) -> str:
    """One sentence a non-developer can act on, for a fetch that failed.

    **The message is the product, and for a transport error it never was**
    (finding F2, 2026-09-05). ``fetch_one`` raises two kinds of exception: the
    ones this module writes itself, which carry a real remedy and are passed
    through verbatim, and whatever the network raised, which until now was
    stringified as ``ConnectType: <repr>`` and shown to the user in a toast
    with no log button. Her machine reported ``ConnectError: [WinError 10054]
    An existing connection was forcibly closed by the remote host``, five
    times, and nothing on screen suggested what to do about it.

    Translated here rather than in the parent because this is the last place
    that holds the *exception*: ``service.downloads`` only ever sees a string,
    so classifying there would mean matching on substrings.
    """
    if isinstance(exc, AUTHORED):
        # Written by this module for a person to read -- a digest mismatch, a
        # missing rename source, a registry entry with no digest.
        return str(exc)
    win, num = _socket_code(exc)
    remedy = _WINSOCK_REMEDIES.get(win or -1) or _ERRNO_REMEDIES.get(num or -1)
    if remedy is None and _looks_like_a_network_error(exc):
        remedy = (
            "The transfer could not be completed. Check that this machine is "
            "online; antivirus, a firewall, a VPN or a workplace proxy can "
            "also break a large download."
        )
    if remedy is None:
        # Not recognisably a network failure. Say what it was and point at the
        # log, which is the honest answer and is what was missing before.
        return f"{type(exc).__name__}: {exc} See warlock.log for the details."
    code = f" (WinError {win})" if win else ""
    return (
        f"{remedy}{code} What was already downloaded has been kept, so "
        f"pressing Install again continues from where it stopped."
    )


def _looks_like_a_network_error(exc: BaseException) -> bool:
    """Whether ``exc`` came from a transport, by class name rather than type.

    ``httpx`` and ``hf_xet`` are dependencies of ``huggingface_hub`` rather
    than of this project, and neither is imported here -- importing a transport
    to name its exceptions would be a heavier coupling than the question
    deserves, and ``hf_xet``'s errors are raised from Rust. The names are
    stable and a false positive costs a slightly generic sentence.
    """
    names = {type(e).__name__ for e in _chain(exc)}
    return any(
        "Connect" in name
        or "Timeout" in name
        or "Network" in name
        or name in {"ConnectionError", "URLError", "HTTPError", "ProtocolError"}
        for name in names
    )


def _chain(exc: BaseException) -> list[BaseException]:
    out: list[BaseException] = []
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        out.append(exc)
        exc = exc.__cause__ or exc.__context__
    return out
