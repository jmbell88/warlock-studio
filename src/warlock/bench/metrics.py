"""Two numbers per rendered view, and what each one is worth.

* ``silhouette_iou`` -- does the mesh have the shape the picture showed?
  Pure NumPy, no weights, no network. This is the metric that actually
  measures what conditioning changes.
* ``dino_cosine`` -- do the two images depict the same object? Needs
  DINOv2, a one-time manual download. It measures style as much as identity,
  so an EEVEE render against SDXL concept art scores low in absolute terms
  and is only ever read as A-against-B.

LPIPS is deliberately absent. Both ``lpips`` and ``torchmetrics`` fetch an
AlexNet/VGG backbone from download.pytorch.org at import time -- HF_HUB_OFFLINE
does nothing about that, so it would need a pre-seeded TORCH_HOME plus a
torchvision entry in the CUDA index. And on this data the score is dominated
by the render-versus-art gap, which is constant across recipes and cancels in
an A/B anyway.

torch is imported inside ``dino_cosine`` only, so this module stays importable
(and silhouette_iou stays usable) without the text2image extra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import imageprep

# Both masks are resized to this before the intersection, so the number does
# not depend on either image's original resolution.
IOU_SIZE = 256

_model_cache: dict[str, Any] = {}


def silhouette_iou(reference_path: Path, render_path: Path) -> float | None:
    """Intersection over union of the two subject masks, each cropped to its
    own bounding box first.

    Cropping before comparing is what makes this a *shape* measurement rather
    than a framing one: the reference's subject and the render's subject are
    at different scales in their frames by construction, and without the crop
    this would mostly measure the camera distance.

    None when either side has no subject.
    """
    import numpy as np
    from PIL import Image

    def mask_of(path: Path, is_render: bool):
        with Image.open(path) as im:
            im.load()
            mask = imageprep.render_mask(im) if is_render else imageprep.reference_mask(im)
        if mask is None:
            return None
        ys, xs = np.nonzero(mask)
        cropped = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        # Padded to a square, *not* stretched to one. Stretching would
        # normalise the aspect ratio away, and a tall sword would then score
        # 1.0 against a wide shield -- which is exactly the confusion this
        # metric exists to catch.
        side = max(cropped.shape)
        square = np.zeros((side, side), dtype=np.uint8)
        top = (side - cropped.shape[0]) // 2
        left = (side - cropped.shape[1]) // 2
        square[top : top + cropped.shape[0], left : left + cropped.shape[1]] = cropped
        resized = Image.fromarray(square * 255).resize((IOU_SIZE, IOU_SIZE), Image.NEAREST)
        return np.asarray(resized) > 127

    a = mask_of(reference_path, False)
    b = mask_of(render_path, True)
    if a is None or b is None:
        return None
    union = np.logical_or(a, b).sum()
    if not union:
        return None
    return float(np.logical_and(a, b).sum()) / float(union)


def dino_available(config: Any = None) -> bool:
    """Whether the weights are on disk. Checked before the torch import, the
    same ordering test_offline.py requires everywhere else."""
    return _dino_dir(config).exists()


def _dino_dir(config: Any = None) -> Path:
    from .. import models
    from ..config import get_config

    spec = models.METRIC_MODELS["dinov2"]
    root = (config or get_config()).t2i_model_root
    return root / spec.dir_name


def _dino_model(config: Any = None):
    """Load DINOv2 once per process -- a 160-item run would otherwise pay for
    it 1280 times."""
    key = str(_dino_dir(config))
    hit = _model_cache.get(key)
    if hit is not None:
        return hit

    path = _dino_dir(config)
    if not path.exists():
        from .. import models

        raise RuntimeError(
            f"DINOv2 not found at {path}. Download once with:\n"
            f"  {models.METRIC_MODELS['dinov2'].download}"
        )
    import torch
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(str(path), local_files_only=True)
    model = AutoModel.from_pretrained(str(path), local_files_only=True)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    _model_cache[key] = (processor, model)
    return (processor, model)


def _embed(image: Any, config: Any = None):
    import torch

    processor, model = _dino_model(config)
    inputs = processor(images=image, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    # The CLS token, which is DINOv2's whole-image descriptor.
    cls = out.last_hidden_state[:, 0]
    return torch.nn.functional.normalize(cls, dim=-1)


def dino_cosine(reference_path: Path, render_path: Path, config: Any = None) -> float | None:
    """Cosine similarity of the two DINOv2 CLS embeddings, higher is better.

    None when either side has no subject.
    """
    import torch

    ref, render = imageprep.prepare_pair(reference_path, render_path)
    if ref is None or render is None:
        return None
    a, b = _embed(ref, config), _embed(render, config)
    return float(torch.sum(a * b).item())


def available(config: Any = None) -> tuple[str, ...]:
    """Which metrics can actually run here. silhouette_iou always can."""
    out = ["silhouette_iou"]
    if dino_available(config):
        out.append("dino_cosine")
    return tuple(out)


def score_view(
    reference_path: Path, render_path: Path, config: Any = None
) -> dict[str, float | None]:
    """Every available metric for one (reference, render) pair."""
    out: dict[str, float | None] = {
        "silhouette_iou": silhouette_iou(reference_path, render_path)
    }
    if dino_available(config):
        out["dino_cosine"] = dino_cosine(reference_path, render_path, config)
    return out
