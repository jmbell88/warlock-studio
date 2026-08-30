"""Masked regeneration through a real sampler: does the mask actually confine it?

``tests/test_fix_not_reroll.py`` pins the arithmetic and the doors -- the crop
grows by the margin and sends a stride-aligned size, the mask survives the wire,
``apply_pixels`` lands by uid as one undo step, a mask without a start image is
refused. All of it with the sampler stubbed, which is correct for a fast lane
and leaves the one claim the feature is *sold* on untested:

**that pixels outside the selection do not change.**

That claim is not arithmetic. It is a property of the inpaint pipeline class
that ``Conditioning.uses_mask`` swaps in, and it is exactly the kind of thing
that is true of a fake by construction -- a stub returning the input image
passes any "outside is preserved" assertion trivially, and a stub returning
noise fails it for the wrong reason. Only a real sample can distinguish a mask
that confines the denoise from one the pipeline quietly ignored, and a pipeline
that ignores the mask repaints the whole crop: the user loses hand-drawn work
outside their selection, silently, with no error anywhere.

The second claim here is the negative one. ``inpaint.DEFAULT_STRENGTH`` is 0.6
and higher than the reference form's default, on the argument that "the mask
already confines the change, so the model can be allowed to invent more inside
it". That argument is only sound if the confinement is real, so it is checked
rather than assumed.

Run with: uv run pytest tests/test_inpaint_gpu.py -m gpu -n 0
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from warlock import models
from warlock.config import get_config
from warlock.pipelines import conditioning as conditioning_mod
from warlock.pipelines.text2image import Text2Image
from warlock.studio.inker import inpaint

#: One checkpoint load and one masked sample.
pytestmark = [pytest.mark.gpu, pytest.mark.timeout(1800)]

SEED = 42

#: The canvas the selection is cut out of. Square, so ``send_size`` resolves to
#: a clean 1024 and the resize is not itself a variable in the comparison.
CANVAS = 512

#: The selection: a centred square, well inside the canvas so that
#: ``crop_box``'s margin has room on every side and the untouched region below
#: is genuinely outside the crop as well as outside the mask.
#:
#: **Square is a limitation of this file, and it once hid a real bug.**
#: ``_sample_sdxl`` passed the spec's square ``width``/``height`` to every
#: class, and the inpaint and ControlNet-img2img pipelines *honour* that pair by
#: resizing the init image and the mask to it -- so a wide selection, which
#: ``inpaint.send_size`` sends at its own aspect, was stretched to 1024²,
#: denoised distorted, and squashed back by ``fit_back``. A square selection is
#: the one shape that escapes it, and it is what this box is.
#:
#: The frame choice is pinned properly in ``tests/test_conditioning.py``
#: (``_init_frame``), in the fast lane and at several aspects, which is the
#: right home for it: it is a pure decision made before any pipeline is called
#: and it needs no card. Do not read the tests below as covering aspect.
BOX = (192, 192, 320, 320)


@pytest.fixture(scope="module")
def pipe():
    config = get_config()
    spec = models.BASE_MODELS[models.DEFAULT_BASE_MODEL]
    t2i = Text2Image(spec, config.t2i_model_root)
    if not (t2i.model_dir / "model_index.json").exists():
        pytest.skip(f"{spec.label} weights not downloaded")
    yield t2i
    t2i.unload()


@pytest.fixture(scope="module")
def canvas() -> np.ndarray:
    """A flattened canvas with structure outside the selection to lose.

    Deliberately not flat: the whole question is whether *something* outside
    the mask survives, and a uniform field cannot answer it -- a pipeline that
    repainted the entire crop in the same colour would pass.
    """
    flat = np.zeros((CANVAS, CANVAS, 4), dtype=np.uint8)
    ys, xs = np.mgrid[0:CANVAS, 0:CANVAS]
    flat[..., 0] = (xs * 255 // CANVAS).astype(np.uint8)
    flat[..., 1] = (ys * 255 // CANVAS).astype(np.uint8)
    flat[..., 2] = ((xs ^ ys) % 256).astype(np.uint8)
    flat[..., 3] = 255
    return flat


@pytest.fixture(scope="module")
def mask() -> np.ndarray:
    """The selection's coverage: white inside ``BOX``, black everywhere else.

    Hard-edged on purpose. A feathered selection is the ordinary case and it
    makes "outside is unchanged" a statement about a gradient; a hard edge
    makes it a statement about equality, which is the one worth asserting.
    """
    weight = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    x0, y0, x1, y1 = BOX
    weight[y0:y1, x0:x1] = 255
    return weight


@pytest.fixture(scope="module")
def regenerated(pipe, canvas, mask, tmp_path_factory):
    """One real masked sample. -> (the returned RGBA, the crop box it covers).

    Composed through ``inpaint.prepare`` rather than by hand: the point is to
    exercise the bytes the editor actually sends, so a change to the margin,
    the stride or the resampling filter is felt here.
    """
    scratch = tmp_path_factory.mktemp("inpaint-gpu")
    crop_png, mask_png, box = inpaint.prepare(canvas, mask, BOX)
    init_path = scratch / "init.png"
    mask_path = scratch / "mask.png"
    init_path.write_bytes(crop_png)
    mask_path.write_bytes(mask_png)

    out = scratch / "out.png"
    pipe.generate(
        "a small red apple",
        out,
        seed=SEED,
        conditioning=conditioning_mod.Conditioning(
            init_image=init_path,
            mask_image=mask_path,
            strength=inpaint.DEFAULT_STRENGTH,
        ),
    )
    with Image.open(out) as image:
        return inpaint.fit_back(image, box), box


def test_the_sample_comes_back_at_the_crops_own_size(regenerated):
    """``fit_back`` resizes to the box, so the caller can blend without maths.

    First because everything below indexes into it: a shape disagreement here
    would make the preservation assertions read out of the wrong pixels and
    fail for a reason that has nothing to do with the mask.
    """
    pixels, box = regenerated
    x0, y0, x1, y1 = box
    assert pixels.shape == (y1 - y0, x1 - x0, 4)
    assert pixels.dtype == np.uint8


def test_the_selection_was_actually_repainted(regenerated, canvas):
    """The positive half: inside the mask, the picture changed.

    Without this, the preservation test below is passed by a pipeline that did
    nothing at all -- which is the most likely way for a silently-ignored
    conditioning object to present.

    Measured on a *localised* statistic and against the margin as its own
    control, and neither is a taste: a mean over the whole selection was the
    first spelling of this and it read 6.97 on a run that had plainly worked
    (2026-08-30, seed 42, `sdxl_cfg`). The prompt asks for "a small red apple"
    and the surround is a strong gradient, so SDXL correctly spends most of the
    selection *continuing the background* and draws the apple in a corner of
    it -- 93 pixels past 40 levels out of 16,384, against a maximum of 125. A
    mean asks the model to repaint the whole selection with something
    different, which is neither what the prompt says nor what an inpaint is
    for; averaged over an area, a small object is indistinguishable from no
    object.

    So: something was drawn (the tail), and the drawing is confined to the
    selection rather than a round trip everything shares (the ratio against the
    margin, which is the same population the preservation test below measures
    at 1.4 levels). A stub returning its input fails both.
    """
    pixels, box = regenerated
    x0, y0, x1, y1 = box
    sx0, sy0, sx1, sy1 = BOX
    inside_new = pixels[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0, :3]
    inside_old = canvas[sy0:sy1, sx0:sx1, :3]
    difference = np.abs(inside_new.astype(np.int16) - inside_old.astype(np.int16))
    per_pixel = difference.mean(axis=2)

    drawn = float((per_pixel > 40).mean())
    assert drawn > 0.001, (
        f"only {drawn:.4%} of the selection moved by more than 40 levels "
        f"(peak {per_pixel.max():.0f}) -- nothing was drawn inside the mask"
    )

    margin = np.ones((y1 - y0, x1 - x0), dtype=bool)
    margin[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = False
    outside = np.abs(
        pixels[..., :3].astype(np.int16) - canvas[y0:y1, x0:x1, :3].astype(np.int16)
    )[margin].mean()
    assert difference.mean() > 2.5 * outside, (
        f"inside the mask moved {difference.mean():.2f} levels against "
        f"{outside:.2f} outside it -- the whole crop drifted by the same "
        "amount, which is a VAE round trip and not a repaint"
    )


def test_outside_the_selection_is_left_alone(regenerated, canvas):
    """The claim the feature is sold on, measured on the crop's own margin.

    ``crop_box`` grows the selection by ``MARGIN`` so the model can see what it
    has to match, and every one of those pixels is *sent* to the sampler while
    being masked black. They are the pixels a pipeline ignoring the mask would
    repaint, and they are inside the returned image, so they can be compared
    directly against what went in.

    A tolerance rather than equality, and the reason is not the mask: the crop
    makes a 1024px round trip through two LANCZOS resizes and a VAE, so a few
    levels of drift are the resampling rather than the denoise. What this
    refuses is a *repaint* -- and an ignored mask does not drift a margin by
    three levels, it draws something else there.
    """
    pixels, box = regenerated
    x0, y0, x1, y1 = box
    sx0, sy0, sx1, sy1 = BOX

    margin = np.ones((y1 - y0, x1 - x0), dtype=bool)
    margin[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = False
    assert margin.any(), "the crop added no margin, so this test asks nothing"

    sent = canvas[y0:y1, x0:x1, :3].astype(np.int16)
    back = pixels[..., :3].astype(np.int16)
    drift = np.abs(back - sent)[margin]
    assert drift.mean() < 16, (
        f"the region outside the selection moved by {drift.mean():.1f} levels on "
        "average -- the mask did not confine the denoise"
    )
