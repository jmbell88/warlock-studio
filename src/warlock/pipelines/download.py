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
