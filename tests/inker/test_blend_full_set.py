"""The seven modes that took the set from twelve to nineteen.

Exclusion, subtract and divide are separable and are tested the way every other
separable mode is -- a formula on a two-pixel array. The other four are not:
hue, saturation, color and luminosity read all three channels of a pixel to
decide one of them, so what is assertable about them is not an expression but a
*property* -- which quantity of the backdrop survives and which of the source
does. Those properties are the whole reason to implement the W3C pseudo-code
literally rather than something that looks about right on a screenshot.

Two of the seven are not in the SVG set at all, and the interop question they
raise is settled here rather than in the writer: ``subtract`` and ``divide``
carry Krita's op names because Krita's are the definitions we implement, and
``divide``'s zero convention is Krita's too. **That spelling still wants a
Krita-authored fixture** -- these tests pin our writer against our reader, which
is the half that cannot catch a name Krita spells differently.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest
from PIL import Image

from warlock import native
from warlock.studio import inker
from warlock.studio.inker import composite as cp

# The twelve that were there before, and the seven appended after them.
OLD_MODES = (
    "normal",
    "darken",
    "multiply",
    "color-burn",
    "lighten",
    "screen",
    "color-dodge",
    "add",
    "overlay",
    "hard-light",
    "soft-light",
    "difference",
)
NEW_MODES = ("exclusion", "subtract", "divide", "hue", "saturation", "color", "luminosity")
NON_SEPARABLE = ("hue", "saturation", "color", "luminosity")


def _rgb(*values: float) -> np.ndarray:
    """One pixel, shaped the way ``over`` hands colour to ``blend``."""
    return np.array([[list(values)]], dtype=np.float32)


def _lum(colour: np.ndarray) -> np.ndarray:
    return (colour * np.array([0.30, 0.59, 0.11], dtype=np.float32)).sum(axis=-1)


def _sat(colour: np.ndarray) -> np.ndarray:
    return colour.max(axis=-1) - colour.min(axis=-1)


def _noise(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return (
        rng.random((9, 7, 3), dtype=np.float32),
        rng.random((9, 7, 3), dtype=np.float32),
    )


# --- the table --------------------------------------------------------------


def test_the_seven_are_appended_rather_than_filed_into_their_families():
    """The menu order is a user-facing decision and this is where it is made.

    Slotting ``exclusion`` next to ``difference`` would read better on paper and
    would move every mode below it in a menu people already have muscle memory
    for. Nothing keys on the position, so the only cost of appending is
    cosmetic and the only cost of interleaving is somebody's hand.
    """
    assert cp.BLEND_MODES == OLD_MODES + NEW_MODES
    assert len(cp.BLEND_MODES) == 19


def test_every_one_of_the_seven_is_a_kernel_mode():
    """``_MODE_IDS`` is the contract with ``native/composite.c``, and a number
    here with no ``case`` there composites as normal, silently.

    The seven were numpy-only until 2026-08-16, which is why this test used to
    assert the opposite: the reading was that the four non-separable ones "could
    not be a per-channel C case", which is true and was the wrong conclusion --
    they are per-*pixel* independent, and the kernel now takes the whole pixel.
    What the gap cost is in ``docs/measurements/2026-08-16-blend-modes-and-dither.md``:
    ``_stack_native`` is all-or-nothing, so one such layer put the whole stack
    on the numpy fold and a dab invalidate went 5.4 ms to 49.

    Asserted over ``BLEND_MODES`` rather than over the seven, because the thing
    worth pinning now is that the map is *complete* -- a mode added to the menu
    and not to the kernel is the next silent cliff.
    """
    assert set(cp._MODE_IDS) == set(cp.BLEND_MODES)


@pytest.mark.skipif(not native.available(), reason="needs the built DLL")
@pytest.mark.parametrize("mode", NEW_MODES)
def test_a_new_mode_reaches_the_kernel_and_agrees_with_numpy_bit_for_bit(mode):
    """The pair that matters: the kernel accepts the mode *and* what it computes
    is the numpy body's answer exactly, not closely."""
    rng = np.random.default_rng(hash(mode) % 2**32)
    backdrop = rng.random((11, 13, 4), dtype=np.float32)
    source = rng.random((11, 13, 4), dtype=np.float32)

    fast = cp._over_native(backdrop, source, 0.75, mode)
    assert fast is not None, f"{mode} was refused by the kernel"

    cb, cs = backdrop[..., :3], source[..., :3]
    ab = backdrop[..., 3:4]
    a_s = source[..., 3:4] * 0.75
    ao = a_s + ab * (1.0 - a_s)
    mixed = cp.blend(cb, cs, mode)
    num = a_s * (1.0 - ab) * cs + a_s * ab * mixed + (1.0 - a_s) * ab * cb
    shown = ao > 0.0
    ref = np.empty_like(backdrop)
    np.divide(num, np.where(shown, ao, 1.0), out=ref[..., :3])
    ref[..., :3] = np.where(shown, ref[..., :3], 0.0)
    ref[..., 3:4] = ao

    assert np.array_equal(ref.view(np.uint32), fast.view(np.uint32))


@pytest.mark.skipif(not native.available(), reason="needs the built DLL")
def test_one_layer_in_a_new_mode_no_longer_drops_the_stack_off_the_kernel():
    """The cliff this file used to document, pinned in the direction it now
    falls. ``_stack_native`` is still all-or-nothing -- that has not changed and
    is still the right trade -- but there is no longer a mode in the menu that
    triggers it.

    Not monkeypatched, unlike the version of this test that asserted the
    opposite: a *refusal* can be tested against a claimed library because it
    never calls into one, and an acceptance cannot.
    """
    plane = np.zeros((4, 4, 4), dtype=np.uint8)
    for mode in NEW_MODES:
        entries = [(plane, 1.0, "normal"), (plane, 1.0, mode)]
        assert cp._stack_native(entries, (0, 0, 4, 4), None) is not None, mode


def test_the_numpy_fold_still_answers_for_the_seven(monkeypatch):
    """The half that has to hold with no DLL at all: the reference is the
    fallback and is never deleted."""
    monkeypatch.setattr(native, "available", lambda: False)
    plane = np.zeros((4, 4, 4), dtype=np.uint8)
    for mode in NEW_MODES:
        entries = [(plane, 1.0, "normal"), (plane, 1.0, mode)]
        assert cp.stack_region(entries, (0, 0, 4, 4)).shape == (4, 4, 4), mode


def test_divide_stays_in_float32_like_every_other_mode():
    """The one mode that did not, until 2026-08-16.

    ``np.where(backdrop > 0.0, 1.0, 0.0)`` is a bool array between two Python
    scalars -- no array operand to take a dtype from, so numpy answered float64
    and ``over`` then ran its whole ``num`` chain at double width for a divide
    layer. In a module whose first paragraph says it speaks float32, and which
    a bit-parity kernel has to agree with.
    """
    for mode in cp.BLEND_MODES:
        cb = np.full((2, 2, 3), 0.5, dtype=np.float32)
        cs = np.full((2, 2, 3), 0.25, dtype=np.float32)
        assert cp.blend(cb, cs, mode).dtype == np.float32, mode


def test_divide_by_an_empty_channel_still_follows_kritas_convention():
    """The dtype fix rewrote the expression, so the convention it encodes is
    re-pinned here rather than trusted to have survived: dividing by zero gives
    white where there is anything to divide and stays black where there is not.
    """
    cb = _rgb(0.5, 0.0, 1.0)
    cs = _rgb(0.0, 0.0, 0.0)
    assert np.array_equal(cp.blend(cb, cs, "divide"), _rgb(1.0, 0.0, 1.0))


# --- the three separable ones -----------------------------------------------


def test_exclusion_is_difference_with_the_hard_edge_taken_off():
    """Cb + Cs - 2·Cb·Cs. Same endpoints as difference, no kink at the middle:
    it leaves black alone, inverts under white, and sends two mid-greys to
    mid-grey rather than to black."""
    assert np.allclose(cp.blend(_rgb(0.2, 0.5, 0.9), _rgb(0.0, 0.0, 0.0), "exclusion"),
                       _rgb(0.2, 0.5, 0.9))
    assert np.allclose(cp.blend(_rgb(0.2, 0.5, 0.9), _rgb(1.0, 1.0, 1.0), "exclusion"),
                       _rgb(0.8, 0.5, 0.1))
    assert np.allclose(cp.blend(_rgb(0.5, 0.5, 0.5), _rgb(0.5, 0.5, 0.5), "exclusion"),
                       _rgb(0.5, 0.5, 0.5))
    # Symmetric in its operands, which difference is too and neither of the
    # other two new separable modes is.
    cb, cs = _noise(1)
    assert np.array_equal(
        cp.blend(cb, cs, "exclusion"), cp.blend(cs, cb, "exclusion")
    )


def test_subtract_floors_at_black_rather_than_wrapping():
    """max(Cb - Cs, 0). The clamp is the test: an unclamped subtract in 8-bit
    turns a slightly-too-dark region bright, which looks like a different filter
    rather than like an overshoot."""
    out = cp.blend(_rgb(0.6, 0.25, 0.0), _rgb(0.2, 0.5, 0.5), "subtract")
    assert np.allclose(out, _rgb(0.4, 0.0, 0.0))
    assert float(out.min()) >= 0.0


def test_divide_takes_krita_zero_convention_in_both_directions():
    """``cfDivide``: a zero divisor gives white where there is anything to
    divide and **black where there is not**.

    The second half is the part worth pinning. Calling every division by zero
    white makes a black backdrop under a black source flash to white -- and a
    black backdrop under a black source is most of the area a divide layer is
    usually pointed at, so the wrong convention is loud rather than subtle.
    """
    assert np.allclose(
        cp.blend(_rgb(0.5, 0.0, 1.0), _rgb(0.0, 0.0, 0.0), "divide"),
        _rgb(1.0, 0.0, 1.0),
    )
    # An ordinary divisor: the ratio, clamped at white.
    assert np.allclose(
        cp.blend(_rgb(0.25, 0.9, 0.5), _rgb(0.5, 0.5, 1.0), "divide"),
        _rgb(0.5, 1.0, 0.5),
    )
    # Dividing by white is the identity, which is what makes the mode usable as
    # a dodge whose strength is the source's own darkness.
    cb, cs = _noise(2)
    assert np.allclose(cp.blend(cb, np.ones_like(cs), "divide"), cb)


# --- the four non-separable ones --------------------------------------------


@pytest.mark.parametrize("mode", NON_SEPARABLE)
def test_a_non_separable_mode_needs_an_rgb_last_axis(mode):
    """They are not per-channel functions, so a single plane is not something
    they can be asked about -- and answering anyway would mean quietly treating
    a (H, W) array's width as its channels."""
    plane = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        cp.blend(plane, plane, mode)


@pytest.mark.parametrize("mode", NON_SEPARABLE)
def test_a_non_separable_mode_is_the_identity_on_equal_operands(mode):
    """B(C, C) = C for all four -- each takes some attributes from one side and
    the rest from the other, so when the two sides agree there is nothing left
    to choose. It is also the cheapest way to catch a swapped operand pair."""
    colour, _ = _noise(3)
    assert np.allclose(cp.blend(colour, colour, mode), colour, atol=1e-6)


@pytest.mark.parametrize(
    "mode,keeps_backdrop_luminance",
    [("hue", True), ("saturation", True), ("color", True), ("luminosity", False)],
)
def test_luminance_comes_from_exactly_one_side(mode, keeps_backdrop_luminance):
    """The property the whole family is built on.

    ``SetLum`` establishes it and ``ClipColor`` is written to preserve it while
    pulling an out-of-gamut colour back in -- which is why this holds on random
    values rather than only on colours that happen to stay inside the cube.
    Three of the four keep the backdrop's luminance and hand it the source's
    colour; luminosity is the one that does the opposite, and it is the one a
    user reaches for to relight a drawing without recolouring it.
    """
    cb, cs = _noise(4)
    out = cp.blend(cb, cs, mode)
    want = cb if keeps_backdrop_luminance else cs
    assert np.allclose(_lum(out), _lum(want), atol=1e-5)


def test_hue_takes_the_saturation_and_luminance_of_the_backdrop():
    """In-gamut on purpose: saturation is preserved only when ClipColor has
    nothing to do, because clipping trades saturation to keep luminance -- the
    priority the spec sets and the reason the two properties are asserted on
    different inputs."""
    cb = _rgb(0.5, 0.4, 0.3)
    cs = _rgb(0.2, 0.6, 0.9)
    out = cp.blend(cb, cs, "hue")
    assert np.allclose(_sat(out), _sat(cb), atol=1e-6)
    assert np.allclose(_lum(out), _lum(cb), atol=1e-6)
    # The *ordering* of the channels is the source's -- that is the hue.
    assert np.array_equal(np.argsort(out, axis=-1), np.argsort(cs, axis=-1))


def test_saturation_takes_the_source_saturation_onto_the_backdrop_hue():
    cb = _rgb(0.2, 0.6, 0.9)
    cs = _rgb(0.5, 0.4, 0.3)
    out = cp.blend(cb, cs, "saturation")
    assert np.allclose(_sat(out), _sat(cs), atol=1e-6)
    assert np.allclose(_lum(out), _lum(cb), atol=1e-6)


def test_a_grey_backdrop_stays_grey_under_saturation():
    """``SetSat`` of a colour with no span is the spec's else-branch: every
    channel goes to zero, and SetLum then puts the grey back. A grey layer is
    exactly the case a naive ``(mid - min) / span`` divides by zero on."""
    out = cp.blend(_rgb(0.4, 0.4, 0.4), _rgb(1.0, 0.0, 0.0), "saturation")
    assert np.allclose(out, _rgb(0.4, 0.4, 0.4), atol=1e-6)


def test_color_is_luminosity_with_the_operands_swapped():
    """Not a coincidence to preserve -- both are ``SetLum(C, Lum(other))``, so
    the identity is exact rather than exact-to-a-rounding. Asserting it is what
    stops either one from being "simplified" into a second implementation of
    SetLum that agrees only to within an ulp. ``hard-light``/``overlay``'s rule,
    applied to the pair that has it in this family.
    """
    cb, cs = _noise(6)
    assert np.array_equal(
        cp.blend(cb, cs, "color"), cp.blend(cs, cb, "luminosity")
    )


def _scalar_reference(cb: list[float], cs: list[float], mode: str) -> list[float]:
    """The W3C pseudo-code transcribed one pixel at a time, as literally as
    Python allows -- ``for`` loops, ``min``/``max``, no numpy.

    This is the second implementation the vectorised one is checked against, and
    it earns its keep because the two are wrong in different ways: the argsort
    form of SetSat and the ``np.where``-guarded ClipColor are exactly where a
    plausible mistake hides, and neither shape exists here.
    """

    def lum(c):
        return 0.30 * c[0] + 0.59 * c[1] + 0.11 * c[2]

    def clip(c):
        light, low, high = lum(c), min(c), max(c)
        if low < 0.0:
            c = [light + (v - light) * light / (light - low) for v in c]
        if high > 1.0:
            c = [light + (v - light) * (1.0 - light) / (high - light) for v in c]
        return c

    def set_lum(c, light):
        delta = light - lum(c)
        return clip([v + delta for v in c])

    def sat(c):
        return max(c) - min(c)

    def set_sat(c, target):
        low, mid, high = sorted(range(3), key=lambda i: c[i])
        out = [0.0, 0.0, 0.0]
        if c[high] > c[low]:
            out[mid] = (c[mid] - c[low]) * target / (c[high] - c[low])
            out[high] = target
        return out

    if mode == "hue":
        return set_lum(set_sat(cs, sat(cb)), lum(cb))
    if mode == "saturation":
        return set_lum(set_sat(cb, sat(cs)), lum(cb))
    if mode == "color":
        return set_lum(cs, lum(cb))
    return set_lum(cb, lum(cs))


@pytest.mark.parametrize("mode", NON_SEPARABLE)
def test_the_vectorised_form_agrees_with_the_pseudo_code_pixel_by_pixel(mode):
    """Five hundred random pairs, against the loop above. Not bit-identity --
    the weights are float32 here and Python floats there, so the bar is the
    float32 epsilon rather than equality; the kernels' identity rule is about a
    kernel and its reference, and both sides of this are the reference.
    """
    rng = np.random.default_rng(12)
    for _ in range(500):
        cb = rng.random(3)
        cs = rng.random(3)
        got = cp.blend(cb.reshape(1, 1, 3), cs.reshape(1, 1, 3), mode)[0, 0]
        want = _scalar_reference(list(cb), list(cs), mode)
        assert np.allclose(got, want, atol=1e-6), (mode, cb, cs)


# --- through ``over`` -------------------------------------------------------


@pytest.mark.parametrize("mode", NEW_MODES)
def test_a_transparent_source_changes_nothing_whatever_the_mode(mode):
    """Opacity 0 is the sanity check every mode has to pass through ``over``:
    the blend runs, its answer is weighted by ``as`` and ``as`` is zero, so the
    backdrop must come back untouched rather than nearly untouched."""
    rng = np.random.default_rng(7)
    backdrop = rng.random((6, 5, 4), dtype=np.float32)
    backdrop[..., 3] = 1.0
    source = rng.random((6, 5, 4), dtype=np.float32)
    out = cp.over(backdrop, source, opacity=0.0, mode=mode)
    assert np.allclose(out, backdrop, atol=1e-6)


@pytest.mark.parametrize("mode", NEW_MODES)
def test_a_new_mode_composites_a_real_stack_without_warning_or_nan(mode):
    """The integration shape: uint8 layers, the fold, the uint8 narrowing --
    which is where a NaN or a negative would actually surface as a black hole
    in somebody's drawing."""
    rng = np.random.default_rng(8)
    layers = [rng.integers(0, 256, (12, 12, 4), dtype=np.uint8) for _ in range(2)]
    with np.errstate(all="raise"):
        out = cp.stack_region([(layers[0], 1.0, "normal"), (layers[1], 0.7, mode)],
                              (0, 0, 12, 12))
    assert np.all(np.isfinite(out))
    assert cp.to_uint8(out).shape == (12, 12, 4)


# --- OpenRaster -------------------------------------------------------------


def _foreign(path: Path, layers: str, *, w: int = 4, h: int = 4) -> None:
    """A minimal ORA written by hand, standing in for another editor's."""
    pixels = np.zeros((h, w, 4), dtype=np.uint8)
    pixels[:, :] = (255, 0, 0, 255)
    buf = io.BytesIO()
    Image.fromarray(pixels, "RGBA").save(buf, "PNG")
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("mimetype", b"image/openraster")
        zf.writestr(
            "stack.xml",
            f'<image version="0.0.3" w="{w}" h="{h}"><stack>{layers}</stack></image>',
        )
        zf.writestr("data/a.png", buf.getvalue())


@pytest.mark.parametrize("mode", NEW_MODES)
def test_every_new_mode_survives_a_real_file_round_trip(mode, tmp_path: Path):
    """Per mode and through a real file, because the op string is the whole of
    what another editor reads a blend mode from -- and half of these strings are
    not derivable from our own names."""
    doc = inker.Document.blank(4, 4)
    doc.stack[0].blend = mode
    path = tmp_path / f"{mode}.ora"
    inker.write_ora(doc, path)

    with zipfile.ZipFile(path) as zf:
        root = ElementTree.fromstring(zf.read("stack.xml"))
    written = [element.get("composite-op") for element in root.iter("layer")]
    assert written == [cp.ORA_OPS[mode]]

    assert inker.Document.load(path).stack[0].blend == mode


def test_the_two_krita_spellings_are_namespaced_and_the_rest_are_svg():
    """OpenRaster's own rule for an operator the SVG set has no name for: the
    application that defines it namespaces it. Inventing ``svg:subtract`` would
    be a name no reader knows, which is worse than a name one reader knows.

    **Integration TODO**: this pins the spelling against its rationale, not
    against Krita. A Krita-authored ORA carrying all seven is what settles
    ``krita:subtract``/``krita:divide`` for certain -- the read side is a
    ``.get`` either way, so being wrong costs a mode on import and never a
    refusal.
    """
    assert cp.ORA_OPS["subtract"] == "krita:subtract"
    assert cp.ORA_OPS["divide"] == "krita:divide"
    for mode in ("exclusion", "hue", "saturation", "color", "luminosity"):
        assert cp.ORA_OPS[mode] == f"svg:{mode}"


def test_an_op_we_still_cannot_reproduce_degrades_to_normal(tmp_path: Path):
    """The control for the round-trip above: the reader is lossy on purpose, so
    the pair of assertions is "a real op comes back" *and* "an unreal one
    becomes normal rather than refusing the file"."""
    path = tmp_path / "foreign.ora"
    _foreign(
        path,
        '<layer name="known" src="data/a.png" composite-op="krita:divide"/>'
        '<layer name="unknown" src="data/a.png" composite-op="krita:dissolve"/>',
    )
    blends = {layer.name: layer.blend for layer in inker.Document.load(path).stack}
    assert blends == {"known": "divide", "unknown": "normal"}


def _rewrite_track_blend(path: Path, blend: str) -> None:
    """Put *blend* on track 0 of an animated ORA, in place."""
    with zipfile.ZipFile(path) as zf:
        members = {name: zf.read(name) for name in zf.namelist()}
    payload = json.loads(members["animation.json"])
    payload["tracks"][0]["blend"] = blend
    members["animation.json"] = json.dumps(payload).encode("utf-8")
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)


def _animated(tmp_path: Path, name: str) -> Path:
    doc = inker.Document.blank(4, 4)
    doc.stack[0].pixels[1:3, 1:3] = (0, 0, 255, 255)
    doc.add_frame(copy=True)
    path = tmp_path / name
    inker.write_ora(doc, path)
    return path


def test_a_track_blend_we_do_not_carry_costs_the_mode_and_not_the_timeline(tmp_path):
    """The degradation that was missing, and it was the expensive kind.

    ``Layer`` refuses an unknown mode, the refusal happened while the grid was
    being built, and the ``except`` around that build drops to the **flat**
    read -- so a file written by a build with one more mode than this one lost
    every frame of its animation over a string. ``stack.xml`` never did this;
    ``animation.json`` now does not either.
    """
    path = _animated(tmp_path, "future.ora")
    _rewrite_track_blend(path, "krita:something-newer")

    doc = inker.Document.load(path)
    assert doc.anim is not None, "the timeline is the thing that must survive"
    assert len(doc.anim.frames) == 2
    assert doc.anim.tracks[0].blend == "normal"


def test_a_track_blend_we_do_carry_is_kept(tmp_path: Path):
    """The control: the softening above must not be a reader that ignores the
    field."""
    path = _animated(tmp_path, "known.ora")
    _rewrite_track_blend(path, "luminosity")
    doc = inker.Document.load(path)
    assert doc.anim is not None
    assert doc.anim.tracks[0].blend == "luminosity"


# --- the byte pin -----------------------------------------------------------

# ``_pinned_doc()``'s ``stack.xml``, byte for byte, as the writer produced it on
# the tree *before* the seven modes existed.
#
# The member and not the archive. A digest of the whole ``.ora`` would also
# cover Pillow's PNG encoder, zlib's deflate streams and the member list -- so
# it would fail on a Pillow bump, and on any later track that legitimately adds
# a member (groups, tags and slices all touch this writer). None of those are
# what this pin is about, and a pin that cries wolf gets deleted. ``stack.xml``
# is the whole of what a mode table can leak into: it is where the op string is
# written, where an unconditional new attribute would appear, and where a table
# serialised in full would land.
MODELESS_STACK_XML = (
    b"<?xml version='1.0' encoding='UTF-8'?>\n"
    b'<image version="0.0.3" w="8" h="8"><stack>'
    b'<layer name="Ink" src="data/layer0.png" x="0" y="0" opacity="0.500000"'
    b' visibility="visible" composite-op="svg:multiply" />'
    b'<layer name="Background" src="data/layer1.png" x="0" y="0" opacity="1.000000"'
    b' visibility="visible" composite-op="svg:src-over" />'
    b"</stack></image>"
)


def _pinned_doc():
    doc = inker.Document.blank(8, 8)
    doc.stack.active.pixels[2:6, 2:6] = (255, 0, 0, 255)
    doc.add_layer(name="Ink")
    doc.stack.active.pixels[0:3, 0:3] = (0, 0, 255, 255)
    doc.stack[1].blend = "multiply"
    doc.stack[1].opacity = 0.5
    return doc


def test_a_document_that_uses_no_new_mode_is_written_exactly_as_it_was():
    """Widening a table must not move a file that never touches the new part.

    Twelve modes to nineteen is exactly the shape of change that quietly
    rewrites every document -- a new attribute written unconditionally, a
    default that is now spelled differently, a table serialised in full. Any of
    those would make every saved ``.ora`` in a library differ from its backup
    for no reason a user could see, and ``test_inker_ora_determinism`` would not
    catch it: that suite compares two saves *of the same build*, which is a
    different question from "does this build write what the last one did".

    The literal was taken from the pre-change tree. If it fails, the question is
    which change moved these bytes and whether that change was meant to: a
    deliberate one re-measures the literal here and says so in the commit; an
    accidental one is the bug this exists to find. A **new member** in the
    archive is not this test's business and deliberately does not fail it.
    """
    with zipfile.ZipFile(io.BytesIO(inker.ora_bytes(_pinned_doc()))) as zf:
        assert zf.read("stack.xml") == MODELESS_STACK_XML
