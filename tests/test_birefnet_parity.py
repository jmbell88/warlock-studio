"""The vendored BiRefNet, against the remote-code path it replaced.

MDL-03. ``matting`` used to build this model with ``trust_remote_code=True``,
which executes whatever ``birefnet.py`` the snapshot on disk holds -- in this
process, with the user's filesystem and network permissions, and with nothing
in ``uv.lock`` describing it. The modelling code lives in
``src/warlock/pipelines/birefnet/`` now; see its ``ATTRIBUTION.md``.

The risk of a vendoring is not that it fails to import. It is that it imports,
loads, runs, and returns a *slightly different* mask -- which would show up as
a gradual change in cutout quality across every 2D export and every promotion,
with nothing to attribute it to. So the mask was captured through the old path
before the switch and is compared against here.

The comparison is **bit-identical**, not approximate. Nothing in the four
modifications touches arithmetic (they delete download paths and replace
``eval`` with table lookups), so any difference at all is a difference nobody
intended.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

GOLDEN = Path(__file__).parent / "fixtures" / "birefnet_golden.npy"


def _subject():
    """The same generated subject the golden was captured from.

    Generated rather than stored: no binary in the repo, and no dependence on
    an asset that could be regenerated differently. The seed is the fixture.
    """
    from PIL import Image

    rng = np.random.default_rng(20260812)
    rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    rgb[:, :] = (240, 238, 232)
    yy, xx = np.mgrid[0:256, 0:256]
    body = ((xx - 128) ** 2 / 60**2 + (yy - 150) ** 2 / 80**2) <= 1.0
    head = ((xx - 128) ** 2 / 34**2 + (yy - 60) ** 2 / 34**2) <= 1.0
    rgb[body] = (70, 90, 140)
    rgb[head] = (200, 160, 120)
    noise = rng.integers(-12, 12, size=rgb.shape, dtype=np.int16)
    return Image.fromarray(np.clip(rgb.astype(np.int16) + noise, 0, 255).astype(np.uint8))


# --- headless: everything decidable without the weights -----------------------


def test_nothing_in_the_source_tree_trusts_remote_code():
    """The whole point, as a ratchet. One ``trust_remote_code=True`` anywhere
    puts the hole back, and it would look entirely ordinary at the call site.

    Parsed rather than grepped: the phrase appears in half a dozen comments and
    docstrings that *explain* why it is gone, and a substring scan would either
    fail on those or have to be weakened until it stopped meaning anything.
    What is forbidden is passing it, which is a keyword argument.
    """
    import ast

    src = Path(__file__).resolve().parents[1] / "src" / "warlock"
    offenders: list[str] = []
    for path in sorted(src.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and any(
                kw.arg == "trust_remote_code" for kw in node.keywords
            ):
                offenders.append(f"{path.relative_to(src).as_posix()}:{node.lineno}")
    assert offenders == []


def test_the_registry_no_longer_declares_remote_code():
    """The field stays -- it is where a future entry would have to say it, and
    doctor's warning sentence keys off it -- but nothing sets it."""
    from warlock import models

    assert not any(spec.remote_code for spec in models.MATTING_MODELS.values())


def test_the_download_no_longer_pulls_python_into_a_model_directory():
    """Not merely unused: Python left in a model directory that nothing runs is
    worse than either extreme, because a reader finding it would reasonably
    assume it *is* what runs."""
    from warlock import models

    spec = models.MATTING_MODELS["birefnet"]
    patterns = [p for one in spec.fetch for p in one.allow_patterns]
    assert "*.py" not in patterns
    assert "*.safetensors" in patterns


def test_the_vendored_module_imports_with_no_weights_and_no_network():
    """It is ordinary Python in this checkout now, so importing it is a fact
    about the repository rather than about the user's models directory."""
    from warlock.pipelines.birefnet import modeling

    assert modeling.BACKBONES, "the backbone table is populated at import"
    assert "swin_v1_l" in modeling.BACKBONES


def test_every_dispatch_that_was_an_eval_is_a_table_now():
    """A block name out of a config file reaching ``eval`` is arbitrary
    execution with extra steps, and ``dec_blk``/``lat_blk``/``squeeze_block``/
    ``refine`` are all strings a config can set."""
    from warlock.pipelines.birefnet import modeling

    assert set(modeling.BLOCKS) >= {"BasicDecBlk", "BasicLatBlk", "ResBlk"}
    assert set(modeling.REFINERS) == {"RefUNet", "Refiner", "RefinerPVTInChannels4"}
    with pytest.raises(KeyError):
        modeling.BLOCKS["HierarAttDecBlk"]


def test_an_unknown_backbone_is_refused_by_name():
    from warlock.pipelines.birefnet import modeling

    with pytest.raises(ValueError, match="unknown BiRefNet backbone"):
        modeling.build_backbone("not_a_backbone")


def test_the_torchvision_backbones_refuse_rather_than_downloading():
    """They were reachable only through ``pretrained=True``, whose weights are
    URLs -- and this app may never touch the network outside the fetch child."""
    from warlock.pipelines.birefnet import modeling

    for name in ("vgg16", "vgg16bn", "resnet50"):
        with pytest.raises(ValueError, match="deliberately cannot download"):
            modeling.build_backbone(name)


def test_no_weights_enum_survives_in_the_vendored_source():
    """A source check as well as the behavioural one above: the import itself
    is what would put ``download.pytorch.org`` back in reach."""
    path = (
        Path(__file__).resolve().parents[1]
        / "src" / "warlock" / "pipelines" / "birefnet" / "modeling.py"
    )
    # Comments excluded for ``test_nothing_..._trusts_remote_code``'s reason:
    # the modification's own note names the symbols it deleted, and it should.
    code = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("#")
    ]
    text = "\n".join(code)
    for forbidden in ("VGG16_Weights", "ResNet50_Weights", "VGG16_BN_Weights"):
        assert forbidden not in text


def test_the_attribution_names_the_commit_the_registry_pins():
    """A vendored file whose provenance note points at a different snapshot
    than the one the registry downloads is a note that misleads."""
    from warlock import models

    here = Path(__file__).resolve().parents[1]
    doc = (here / "src/warlock/pipelines/birefnet/ATTRIBUTION.md").read_text(encoding="utf-8")
    revision = models.MATTING_MODELS["birefnet"].fetch[0].revision
    assert revision in doc


@pytest.mark.parametrize("name", ["pvt_v2_b0"])
def test_a_real_backbone_is_constructed_by_the_dispatch(name: str):
    """The dispatch that replaced ``eval``, exercised on the smallest backbone
    in the table -- small enough for a unit test, real enough that a broken
    constructor signature would fail here rather than only under the full
    checkpoint.

    **Constructed, not run.** ``pvt_v2_b0``'s forward pass is broken upstream
    and is broken here identically: ``modeling.py`` concatenates several source
    files and defines ``Mlp`` twice (line ~199 for pvt_v2, line ~640 for Swin),
    so the second shadows the first and pvt_v2's blocks call Swin's ``Mlp``
    with three arguments. That is not something the vendoring introduced and
    not something it may fix -- a fifth modification that changed behaviour
    would be the one thing this exercise cannot afford. Nothing reaches it:
    the checkpoint's ``Config`` selects ``swin_v1_l``, which is what the parity
    test below actually runs.
    """
    pytest.importorskip("torch")
    from warlock.pipelines.birefnet import modeling

    bb = modeling.build_backbone(name)
    assert bb is not None
    assert type(bb).__name__ == name


def test_the_two_mlp_classes_are_still_upstreams_problem_and_not_ours():
    """Pinned so the shadowing above is a recorded fact rather than a mystery
    the next reader has to rediscover -- and so that if a future upstream fixes
    it, this fails and the note in the test above can go."""
    path = (
        Path(__file__).resolve().parents[1]
        / "src" / "warlock" / "pipelines" / "birefnet" / "modeling.py"
    )
    defs = [
        n
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if line.startswith("class Mlp(")
    ]
    assert len(defs) == 2, f"upstream defines Mlp twice; found {defs}"


def test_params_settings_is_parsed_rather_than_executed(monkeypatch):
    """Upstream formatted this into source text and ran it through ``eval``.

    The only value any code path supplies is ``in_channels=4``, and what
    matters is that the parse is *faithful*: the same keyword, with an ``int``
    rather than the string ``"4"``, reaching the same constructor. Asserted
    against a recording stand-in registered in the table, because neither real
    backbone can show it -- ``pvt_v2_b0`` swallows ``**kwargs`` without using
    them and ``swin_v1_t`` takes no arguments at all.
    """
    from warlock.pipelines.birefnet import modeling

    seen: dict = {}

    def _fake(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setitem(modeling.BACKBONES, "recorder", _fake)
    modeling.build_backbone("recorder", params_settings="in_channels=4")
    assert seen == {"in_channels": 4}
    assert isinstance(seen["in_channels"], int)

    seen.clear()
    modeling.build_backbone("recorder")
    assert seen == {}, "no settings means no keywords, as an empty eval string did"


def test_the_parse_keeps_upstreams_answer_for_the_two_real_backbones():
    """Faithfulness the other way: whatever ``eval`` did with these, the parse
    does too. ``pvt_v2_b0(in_channels=4)`` was accepted and ignored;
    ``swin_v1_t(in_channels=4)`` was a TypeError. Both still are -- which is
    also why this argument is a dead path upstream: the only caller is
    ``RefinerPVTInChannels4``, and ``Config`` disables the refiner."""
    pytest.importorskip("torch")
    from warlock.pipelines.birefnet import modeling

    assert modeling.build_backbone("pvt_v2_b0", params_settings="in_channels=4") is not None
    with pytest.raises(TypeError):
        modeling.build_backbone("swin_v1_t", params_settings="in_channels=4")


# --- the parity itself, which needs the real checkpoint -----------------------


@pytest.mark.gpu
def test_the_vendored_model_reproduces_the_mask_bit_for_bit():
    """The golden was captured through ``AutoModelForImageSegmentation`` with
    ``trust_remote_code=True``, on this machine, before the switch -- and
    through ``matting._infer``, so it is the *pipeline's* answer rather than a
    script's.

    ``gpu``-marked because it needs the real 424 MB checkpoint, not because it
    needs a card: it runs on the CPU. The marker is this repo's word for "needs
    weights this host may not have".
    """
    pytest.importorskip("torch")
    from warlock import models
    from warlock.config import Config
    from warlock.pipelines import birefnet, matting

    weights = Path(Config().t2i_model_root) / models.MATTING_MODELS["birefnet"].dir_name
    if not (weights / birefnet.WEIGHTS_NAME).exists():
        pytest.skip(f"no BiRefNet checkpoint at {weights}")

    # ``strict=True`` is inside ``load``: reaching the next line at all means
    # the vendored architecture maps exactly onto the published state dict.
    model = birefnet.load(weights).float()
    mask = matting._infer(_subject(), model)

    golden = np.load(GOLDEN)
    assert mask.shape == golden.shape
    differing = int((mask != golden).sum())
    assert differing == 0, (
        f"{differing} of {golden.size} pixels differ from the mask the "
        f"remote-code path produced. Nothing in the four vendoring "
        f"modifications touches arithmetic, so any difference is unintended."
    )


def test_the_golden_is_in_the_tree_and_is_a_mask():
    """The parity test skips without weights, so this is what notices the
    fixture going missing -- otherwise the skip and the deletion look the
    same from a test report."""
    golden = np.load(GOLDEN)
    assert golden.dtype == np.dtype(bool)
    assert golden.shape == (256, 256)
    # A mask that is all one value would make the comparison above vacuous.
    assert 0 < int(golden.sum()) < golden.size
