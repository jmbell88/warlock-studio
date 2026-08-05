"""One boolean mask per image, from a model when there is one.

Three sources, in a fixed order of preference, and the order is the design:

* an alpha channel the image already has -- ground truth, and a model asked to
  re-cut an existing cutout can only make it worse;
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
from .reference import subject_mask

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
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        return (subject_mask(image), "alpha")
    if available(config):
        try:
            found = _model_mask(image, model_dir(config), device)
            if found is not None and found.any():
                return (found, models.DEFAULT_MATTING)
            log.warning("the matting model found no subject; falling back to the fill")
        except Exception:
            # Never fatal. The flood fill is worse-looking and always right
            # enough to produce a file, which beats a failed export.
            log.exception("matting failed; falling back to the corner fill")
    return (subject_mask(image), "flood")


def unload() -> None:
    """Drop the loaded model.

    The counterpart to ``_load``: the cache holds one CPU model for as long as
    the process lives, which is right while exports keep coming and pure waste
    once the user has moved on, so the release is a call somebody can make
    rather than a decision baked into the load.
    """
    _cache.clear()


_cache: dict[str, Any] = {}


def _load(path: Path, device: str):
    from transformers import AutoModelForImageSegmentation

    key = f"{path}|{device}"
    hit = _cache.get(key)
    if hit is not None:
        return hit
    model = AutoModelForImageSegmentation.from_pretrained(
        str(path),
        # The repo's own modelling code, from the snapshot on disk. Nothing is
        # fetched: local_files_only is what makes that true, and doctor's row
        # states the trade rather than hiding it.
        trust_remote_code=models.MATTING_MODELS[models.DEFAULT_MATTING].remote_code,
        local_files_only=True,
    )
    model.eval()
    model = model.to(device)
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
