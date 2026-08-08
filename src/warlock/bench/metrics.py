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


def _dino_model(config: Any = None, device: str | None = None):
    """Load DINOv2 once per (path, device) -- a 160-item run would otherwise
    pay for it 1280 times.

    ``device`` is explicit rather than "cuda if available" for one caller: the
    app scores candidates on the job queue, beside a resident trellis and a
    resident SDXL pipe, and a metric has no business taking VRAM away from
    the models that are producing the asset. The benchmark passes None and
    keeps the old behaviour.
    """
    # Load-bearing ordering, the same one dino_available's docstring states and
    # test_offline.py pins for Text2Image.load(): the existence check precedes
    # the torch import, so a checkout without the text2image extra gets the
    # actionable "download it once with ..." rather than an ImportError about a
    # dependency that was never the problem.
    path = _dino_dir(config)
    if not path.exists():
        from .. import models

        raise RuntimeError(
            f"DINOv2 not found at {path}. Download once with:\n"
            f"  {models.METRIC_MODELS['dinov2'].download}"
        )
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    key = f"{path}|{device}"
    hit = _model_cache.get(key)
    if hit is not None:
        return hit

    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(str(path), local_files_only=True)
    model = AutoModel.from_pretrained(str(path), local_files_only=True)
    model.eval()
    model = model.to(device)
    _model_cache[key] = (processor, model, device)
    return _model_cache[key]


def _embed(image: Any, config: Any = None, device: str | None = None):
    import torch

    processor, model, resolved = _dino_model(config, device)
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(resolved) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    # The CLS token, which is DINOv2's whole-image descriptor.
    cls = out.last_hidden_state[:, 0]
    return torch.nn.functional.normalize(cls, dim=-1)


def cls_embedding(image_path: Path, config: Any = None, device: str | None = None):
    """One image's normalized DINOv2 CLS embedding, as a plain numpy vector.

    The public half of ``_embed``, added for ``judge.py`` -- which is stdlib plus
    numpy and may not hold a torch tensor, and which would otherwise have needed
    its own loader. A second loader would be a second answer to "which weights,
    on which device, offline or not", and this one already has all three right.

    Deliberately the *raw* image rather than ``imageprep.prepare_pair``'s cutout:
    the probes are asked whether a whole plate is a good asset or a good blank,
    and background and framing are exactly what those questions are about.
    """
    from PIL import Image

    with Image.open(image_path) as image:
        vector = _embed(image.convert("RGB"), config, device)
    return vector.detach().to("cpu").numpy().reshape(-1)


def dino_cosine(reference_path: Path, render_path: Path, config: Any = None) -> float | None:
    """Cosine similarity of the two DINOv2 CLS embeddings, higher is better.

    None when either side has no subject.
    """
    import torch

    ref, render = imageprep.prepare_pair(reference_path, render_path)
    if ref is None or render is None:
        return None
    a, b = _embed(ref, config, None), _embed(render, config, None)
    return float(torch.sum(a * b).item())


def reference_cosine(
    a_path: Path, b_path: Path, config: Any = None, device: str | None = None
) -> float | None:
    """Cosine similarity between two *references*, higher is more alike.

    The A-against-A caveat in this module's docstring is not merely satisfied
    here, it is the whole point: both sides are SDXL output on a plain
    background, so the style gap that makes ``dino_cosine`` unreadable in
    absolute terms is absent, and comparing candidates of one submit against
    one anchor is exactly the comparison the number supports.
    """
    a, b = imageprep.prepare_references(a_path, b_path)
    if a is None or b is None:
        return None
    # Below the pairing, for the same reason the existence check sits above the
    # import in _dino_model: a pair with no subject is answered without ever
    # needing torch, and paying seconds for an import to return None is a cost
    # with nothing on the other side of it.
    import torch

    return float(torch.sum(_embed(a, config, device) * _embed(b, config, device)).item())


def available(config: Any = None) -> tuple[str, ...]:
    """Which metrics can actually run here. silhouette_iou always can."""
    out = ["silhouette_iou"]
    if dino_available(config):
        out.append("dino_cosine")
    return tuple(out)


# What a *reference-stage* run is scored on. Its own tuple rather than an
# addition to available(): those are per-view metrics of a reconstruction, and
# a model-stage scores.json gaining five always-null pixel rows would be five
# rows of nothing on every existing run.
REFERENCE_METRICS = (
    "pixel_grid_residual",
    "pixel_grid_scale",
    "pixel_colors_128",
    "pixel_colors_64",
    "pixel_orphans_128",
)


def score_reference(reference_path: Path, config: Any = None) -> dict[str, Any]:
    """The pixel-art metrics for one generated reference.

    Measured on the raw generation -- ``input.png`` at full resolution -- and
    deliberately not on the exported pixel artifact. The grid residual is a
    statement about *the generator*: whether it drew on a lattice at all. The
    export's own palette mapping and orphan cleanup would launder exactly the
    failures this is here to detect, so measuring the post-processed file would
    make every arm score the same.

    The colour and orphan counts are taken *through* the export's own reduction
    (``pipelines.pixel``), because a reduction is how anyone would ever look at
    the image, and they are what "reads as pixel art" comes down to once the
    grid is there.

    Pure numpy and Pillow, no torch, so a reference-stage run scores on a
    machine that cannot load a model.
    """
    from PIL import Image

    from ..pipelines import pixel as pixelmod

    with Image.open(reference_path) as im:
        im.load()
        image = im.convert("RGBA")

    grid = pixelmod.detect_grid(image)
    out: dict[str, Any] = {
        "pixel_grid_residual": float(grid["residual"]),
        "pixel_grid_scale": grid["scale"],
    }
    for size in (128, 64):
        reduced = image.resize((size, size), Image.NEAREST)
        report = pixelmod.report(reduced)
        out[f"pixel_colors_{size}"] = float(report["colors"])
        if size == 128:
            out["pixel_orphans_128"] = float(report["orphan_fraction"])
    return out


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
