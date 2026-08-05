"""One boolean mask per image, from a model when there is one.

Three sources, in a fixed order of preference, and the order is the design:

* an alpha channel the image already has *and actually uses* -- ground truth,
  and a model asked to re-cut an existing cutout can only make it worse;
* BiRefNet, if its weights are on disk;
* the corner flood fill in ``pipelines/reference.py``, which needs nothing.

The fallback is what makes every 2D export work on a fresh checkout, and it is
never skipped on failure: a corrupt checkpoint, an out-of-memory, or a matte
that comes back empty all fall through to the flood fill rather than raising.
Missing weights cost the user edge quality, never the export.

The model is loaded on first use, kept on the CPU, and dropped by ``unload()``
-- never moved to the GPU by default. That is the same trade
``text2image._conditioned`` makes for the ControlNet, in the other direction:
this runs beside a resident trellis and a resident SDXL pipe, and a matting
model holding VRAM would take room from the models that are actually producing
the asset. A second or two of host compute per export is the cheaper half.

Nothing here is imported at module level that a fresh checkout might not have:
torch, numpy and Pillow all load inside the functions that need them, so
``available()`` -- the question every caller asks first -- is answerable on a
machine with no image extra installed at all.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import models
from .reference import has_alpha, subject_mask

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL import Image as _ImageModule

    PILImage = _ImageModule.Image
else:  # pragma: no cover - runtime alias
    PILImage = Any

log = logging.getLogger(__name__)

# What counts as subject in the model's probability map. 0.5 is the threshold
# BiRefNet's own demo uses; the map is confidently bimodal on a plain
# background, so the exact value only matters on the soft rim.
THRESHOLD = 0.5

# The model's expected input. It is fully convolutional, but it was trained at
# this size and a matte taken at the source resolution is measurably worse at
# the same cost.
INPUT_SIZE = 1024

# ImageNet statistics, which is what the published preprocessing normalises
# with. Written out here rather than pulled in through torchvision: this is
# three lines of arithmetic, and a dependency the project does not otherwise
# have would be a strange price for them.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)

# Below this, an alpha channel is carrying a cutout; at or above it every pixel
# is opaque and the channel says nothing. Not 255, because a saved cutout's rim
# is antialiased and a re-encode can leave the interior a step or two short of
# full. Having a channel is not the same as using one: half the tools in the
# world write RGBA unconditionally, and trusting an opaque one would hand every
# such upload back as a full-frame "cutout" labelled ground truth.
_OPAQUE = 250


def model_dir(config: Any = None) -> Path:
    from ..config import get_config

    spec = models.MATTING_MODELS[models.DEFAULT_MATTING]
    root = (config or get_config()).t2i_model_root
    return Path(root) / spec.dir_name


def available(config: Any = None) -> bool:
    """Whether the weights are on disk. Checked before any torch import, which
    is the ordering tests/test_offline.py requires everywhere."""
    return (model_dir(config) / "config.json").exists()


def mask(image: PILImage, config: Any = None, *, device: str = "cpu") -> tuple[Any, str]:
    """-> (boolean mask, which source produced it).

    The source is returned rather than logged because it goes in the manifest:
    an exported set whose alpha came from the flood fill has visibly rougher
    edges than one that did not, and that is a fact about the files.
    """
    if _alpha_is_a_cutout(image):
        return (subject_mask(image), "alpha")
    # An alpha channel that says nothing must not be read by the fill either:
    # subject_mask keys on the *presence* of the channel, so it would take the
    # same branch and hand back the same all-true mask. Dropping it here is
    # what makes the fallback mean what its name says.
    flat = image.convert("RGB") if has_alpha(image) else image
    if available(config):
        try:
            found = _model_mask(flat, model_dir(config), device)
            if found is not None and found.any():
                return (found, models.DEFAULT_MATTING)
            log.warning("the matting model found no subject; falling back to the fill")
        except _AlreadyFailed as exc:
            # Decided once, reported once per image and without the traceback:
            # an export is a loop over images, and the same stack fifty times
            # buries every other line in the log.
            log.warning("%s; using the corner fill", exc)
        except Exception:
            # Never fatal. The flood fill is worse-looking and always right
            # enough to produce a file, which beats a failed export.
            log.exception("matting failed; falling back to the corner fill")
    return (subject_mask(flat), "flood")


def _alpha_is_a_cutout(image: PILImage) -> bool:
    """Whether the image's alpha channel is a matte somebody already made.

    Having the channel is not the same as using it. A PNG saved by almost any
    editor is RGBA whether or not anything in it is transparent, and treating a
    fully opaque one as ground truth returns the whole frame as subject --
    labelled ``"alpha"``, which is the manifest's way of saying "this boundary
    is exact". So the channel has to contain at least one non-opaque pixel
    before it is believed.
    """
    import numpy as np

    if not has_alpha(image):
        return False
    alpha = np.asarray(image.convert("RGBA"))[:, :, 3]
    return bool(alpha.min() < _OPAQUE)


def unload() -> None:
    """Drop the loaded model, and any memory of one that would not load.

    The counterpart to ``_load``: the cache holds one CPU model for as long as
    the process lives, which is right while exports keep coming and pure waste
    once the user has moved on, so the release is a call somebody can make
    rather than a decision baked into the load. It also forgets the failure
    sentinel, so a user who repairs a half-downloaded checkpoint gets another
    attempt without restarting the app.
    """
    _cache.clear()


class _AlreadyFailed(RuntimeError):
    """This checkpoint was tried once this session and could not be loaded."""


# The sentinel a failed load leaves behind, in the same dict as the model so
# that clearing one clears the other.
_FAILED = object()

_cache: dict[str, Any] = {}


def _load(path: Path, device: str):
    from transformers import AutoModelForImageSegmentation

    key = f"{path}|{device}"
    hit = _cache.get(key)
    if hit is _FAILED:
        # A checkpoint that cannot load cannot load again, and from_pretrained
        # is seconds of work per attempt. An export is a loop over images, so
        # without this the first broken install costs the whole batch.
        raise _AlreadyFailed(f"the matting model at {path} failed to load earlier this session")
    if hit is not None:
        return hit
    try:
        model = AutoModelForImageSegmentation.from_pretrained(
            str(path),
            # The repo's own modelling code, from the snapshot on disk. Nothing
            # is fetched: local_files_only is what makes that true, and
            # doctor's row states the trade rather than hiding it.
            trust_remote_code=models.MATTING_MODELS[models.DEFAULT_MATTING].remote_code,
            local_files_only=True,
        )
        model.eval()
        model = model.to(device)
    except Exception:
        _cache[key] = _FAILED
        raise
    _cache[key] = model
    return model


def _model_mask(image: PILImage, path: Path, device: str):
    """The model half, isolated so the fallback logic above can be tested
    without weights by patching exactly this."""
    import numpy as np
    import torch
    from PIL import Image

    model = _load(path, device)
    # Pillow and NumPy do what torchvision's Resize/ToTensor/Normalize do,
    # which is the whole of the published preprocessing: bilinear to the
    # trained size, scale to 0..1, subtract the ImageNet statistics.
    small = image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    array = np.asarray(small, dtype=np.float32) / 255.0
    array = (array - np.asarray(_MEAN, dtype=np.float32)) / np.asarray(_STD, dtype=np.float32)
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        # The published model returns a list of maps at descending scales; the
        # last is the full-resolution one, which is what its own demo reads.
        out = model(tensor)[-1].sigmoid().cpu()
    probability = out[0].squeeze()
    resized = Image.fromarray((probability.numpy() * 255).astype(np.uint8)).resize(
        image.size, Image.BILINEAR
    )
    return np.asarray(resized) > int(THRESHOLD * 255)
