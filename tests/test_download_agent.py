"""Every download this project makes says who it is.

Python's ``urllib`` sends ``Python-urllib/3.13`` unless told otherwise, and on
2026-09-05 that agent stopped being served: the URLs ``uv.lock`` records for
torch, torchaudio and torchvision are on ``download-r2.pytorch.org``, and
Cloudflare answers the default agent there with 403 and ``error code: 1010`` --
"banned based on your browser's signature". Every other host involved still
served it, so it surfaced as three wheels failing rather than as no network.

That was three call sites with one spelling between them, and only one of them
was the build. ``pack_worker`` downloads those same three wheels *on the user's
machine* when Create, Poser or Muse is chosen from Settings -> Packs, so the
shipped pack flow was refusing every heavy pack with a traceback.

The claim these tests make is the general one rather than the incident: a bare
``urlopen`` in a module that downloads is the defect, whoever is banned this
year. So the source is read -- the alternative is a test that only fails when
the ban is live, which is a test that would have passed all the way up to the
morning it mattered.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from warlock.pipelines import download

# The modules that perform a download. ``fetch.py`` and ``packs.py`` plan them
# and are deliberately networkless, so they are not here.
PERFORMERS = (
    Path("src/warlock/pipelines/pack_worker.py"),
    Path("src/warlock/pipelines/fetch_worker.py"),
    Path("src/warlock/pipelines/update_worker.py"),
    Path("scripts/make_packs.py"),
)

ROOT = Path(__file__).resolve().parents[1]


def test_agent_is_not_the_default_urllib_one():
    assert "Python-urllib" not in download.USER_AGENT
    assert download.USER_AGENT.startswith("Warlock-Studio/")


def test_request_carries_the_agent():
    # ``Request`` title-cases header names, which is why this is not
    # ``headers["User-Agent"]``.
    assert download.request("https://example.com/x.whl").get_header("User-agent") == (
        download.USER_AGENT
    )


def test_open_url_accepts_a_timeout_and_builds_the_same_request(monkeypatch):
    seen: dict[str, object] = {}

    def fake(request, **kwargs):
        seen["agent"] = request.get_header("User-agent")
        seen["url"] = request.full_url
        seen["timeout"] = kwargs.get("timeout")
        return object()

    monkeypatch.setattr(download.urllib.request, "urlopen", fake)
    download.open_url("https://example.com/x.whl", timeout=60)
    assert seen == {
        "agent": download.USER_AGENT,
        "url": "https://example.com/x.whl",
        "timeout": 60,
    }


@pytest.mark.parametrize("relative", PERFORMERS, ids=lambda p: p.name)
def test_no_performer_calls_urlopen_bare(relative):
    """The regression itself: ``urlopen(<url>)`` with no agent attached.

    Fails against the unfixed tree three times over, which is the point -- all
    three call sites were written the same way and broke on the same day.
    """
    source = (ROOT / relative).read_text(encoding="utf-8")
    for number, line in enumerate(source.splitlines(), start=1):
        if "urlopen(" not in line:
            continue
        # The one permitted shape passes a Request that download.py built.
        assert "request(" in line, (
            f"{relative}:{number} calls urlopen without a Request carrying the "
            f"agent; use pipelines/download.py"
        )
