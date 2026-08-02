"""Insurance against conftest.py's GPU-pipeline fakes silently drifting from
the real classes they stand in for. If FakeTrellisServer.generate or
FakeText2Image.generate stops matching the real call contract, fake_pipelines
would quietly start testing the wrong thing -- these fail loudly instead."""

from __future__ import annotations

from inspect import Parameter, Signature, signature

from conftest import FakeText2Image, FakeTrellisServer

from warlock.pipelines.text2image import Text2Image
from warlock.pipelines.trellis import TrellisServer


def _bare_signature(func) -> Signature:
    """Parameter names/kinds/defaults only, annotations stripped. The real
    pipeline modules use `from __future__ import annotations` (string
    annotations) while the fakes here carry none at all, so a literal
    signature comparison would always fail on annotations alone even when
    the actual call contract -- the thing that matters -- matches."""
    sig = signature(func)
    return sig.replace(
        parameters=[p.replace(annotation=Parameter.empty) for p in sig.parameters.values()],
        return_annotation=Signature.empty,
    )


def test_fake_trellis_server_mirrors_real_signature():
    assert _bare_signature(FakeTrellisServer.generate) == _bare_signature(TrellisServer.generate)


def test_fake_text2image_mirrors_real_signature():
    assert _bare_signature(FakeText2Image.generate) == _bare_signature(Text2Image.generate)
