"""Two numbers per rendered view, and what each one is worth.

* ``pixel_similarity`` -- is this literally the same picture? A 64-bit
  difference hash, pure NumPy and Pillow. Survives rescaling and re-encoding,
  and cannot rank two *different* images against each other at all.
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

# The difference hash: an 8x9 grayscale downscale, compared left-to-right, so
# 8*8 = 64 bits. HASH_FLOOR is the *measured* agreement between unrelated
# images, which is chance and not zero -- see docs/measurements/. Every
# user-facing number goes through hash_similarity, which subtracts it.
HASH_SIDE = 8
HASH_BITS = HASH_SIDE * HASH_SIDE
HASH_FLOOR = 0.5

_model_cache: dict[str, Any] = {}


def unload() -> None:
    """Drop the cached DINOv2 processor/model. Safe to call at any time.

    The counterpart ``matting.unload`` and ``pose2d.unload`` already had, and
    the one of the three that can hold *VRAM*: the cache key is
    ``f"{path}|{device}"`` and the benchmark path passes ``device=None``, which
    resolves to cuda whenever it is available -- so a single scoring run left a
    DINOv2 on the card for the life of the process, beside the models actually
    producing assets. ``queue.Worker._maybe_evict_caches`` calls it past the
    idle timeout.

    A dict clear and nothing more, deliberately: this module must stay
    importable with no torch installed (``tests/test_bench_metrics.py`` pins
    that), so it may not reach for ``torch.cuda.empty_cache()`` here. Dropping
    the last reference is what returns the allocation; the caching allocator
    gives its pool back on the next pass that needs it.
    """
    _model_cache.clear()


def silhouette_iou(
    reference_path: Path, render_path: Path, *, auto_mask: bool = False
) -> float | None:
    """Intersection over union of the two subject masks, each cropped to its
    own bounding box first.

    Cropping before comparing is what makes this a *shape* measurement rather
    than a framing one: the reference's subject and the render's subject are
    at different scales in their frames by construction, and without the crop
    this would mostly measure the camera distance.

    None when either side has no subject.

    ``auto_mask`` takes the render side from its alpha only if it *has* one,
    falling back to the flood fill the reference side uses. Off by default
    because the scoring path's render is a transparent PNG by construction and
    a silent fallback there would hide a broken render; ``compare`` turns it on
    because both of its sides are arbitrary and usually opaque.
    """
    import numpy as np
    from PIL import Image

    def mask_of(path: Path, is_render: bool):
        with Image.open(path) as im:
            im.load()
            if not is_render:
                mask = imageprep.reference_mask(im)
            else:
                mask = imageprep.render_mask(im)
                if mask is None and auto_mask:
                    mask = imageprep.reference_mask(im)
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


def perceptual_hash(image_path: Path) -> int | None:
    """The image's 64-bit difference hash, or None if it cannot be read as one.

    dHash rather than aHash or a downscaled mean-absolute-difference: each bit
    compares a pixel to its right-hand neighbour, so the hash keys on *gradient
    direction* and largely survives brightness and contrast shifts. Measured at
    0.828 raw agreement across a 1.4x brightness multiply, where an absolute
    metric would have collapsed.

    The limits are as load-bearing as the property, and are the reason this is
    reported beside ``dino_cosine`` rather than instead of it:

    * Not rotation- or reflection-invariant. A mirrored image is unrelated here.
    * 64 bits over an 8x9 downscale cannot see fine detail -- two renders that
      differ only in a decal score 1.0.
    * It does not distinguish "different subject" from "unrelated noise". A
      drawn sword against a drawn shield scored 0.531 raw -- the mean over 400
      unrelated random-noise pairs is 0.520. Ranking *related but different*
      images is not something this metric can do at all.
    * A 6-pixel translation on a 256px image already costs 0.28 raw. Two
      renders from an almost-identical camera are a job for ``silhouette_iou``,
      not for this.

    It answers "is this the same image", not "is this a good image".
    """
    import numpy as np
    from PIL import Image

    try:
        with Image.open(image_path) as im:
            im.load()
            small = im.convert("L").resize((HASH_SIDE + 1, HASH_SIDE), Image.LANCZOS)
    except (OSError, ValueError):
        return None
    arr = np.asarray(small, dtype=np.int16)
    bits = (arr[:, 1:] > arr[:, :-1]).reshape(-1)
    out = 0
    for bit in bits:
        out = (out << 1) | int(bit)
    return out


def hash_similarity(a: int, b: int) -> float:
    """Two hashes' agreement, rescaled so unrelated reads as 0 and not as 0.5.

    Independent hash bits agree half the time by chance -- measured mean 0.520
    over 400 unrelated pairs, p95 0.625 -- so raw agreement is a percentage
    with a floor under it, and reporting it as-is would call a worst-case pair
    of pure-noise plates "75% similar". Subtracting the floor and rescaling is
    what ``pipelines.rank`` already does to the cosine's -1..1 range. The corpus
    is ``docs/measurements/2026-08-11-perceptual-hash-floor.md``; do not
    "simplify" this back to a bit count.
    """
    agreement = 1.0 - bin(a ^ b).count("1") / HASH_BITS
    return max(0.0, (agreement - HASH_FLOOR) / (1.0 - HASH_FLOOR))


def pixel_similarity(a_path: Path, b_path: Path) -> float | None:
    """Near-duplicate score for two images, 1.0 identical, ~0 unrelated.

    None when either file cannot be read as an image.
    """
    a, b = perceptual_hash(a_path), perceptual_hash(b_path)
    if a is None or b is None:
        return None
    return hash_similarity(a, b)


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


def pickscore_available(config: Any = None) -> bool:
    """Whether the preference model's weights are on disk. Existence before
    the torch import, the ordering this whole module keeps."""
    return (_pickscore_dir(config) / "model.safetensors").exists()


def _pickscore_dir(config: Any = None) -> Path:
    from .. import models
    from ..config import get_config

    spec = models.METRIC_MODELS["pickscore"]
    root = (config or get_config()).t2i_model_root
    return root / spec.dir_name


def _pickscore_model(config: Any = None, device: str | None = None):
    """Load PickScore once per (path, device), the ``_dino_model`` pattern
    exactly -- same cache, same existence-before-torch ordering, same explicit
    ``device`` so the app can pin it to the CPU beside a resident trellis."""
    path = _pickscore_dir(config)
    if not (path / "model.safetensors").exists():
        from .. import models

        raise RuntimeError(
            f"PickScore not found at {path}. Download once with:\n"
            f"  {models.METRIC_MODELS['pickscore'].download}"
        )
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    key = f"pickscore|{path}|{device}"
    hit = _model_cache.get(key)
    if hit is not None:
        return hit

    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(str(path), local_files_only=True)
    model = AutoModel.from_pretrained(str(path), local_files_only=True)
    model.eval()
    model = model.to(device)
    _model_cache[key] = (processor, model, device)
    return _model_cache[key]


def preference_score(
    prompt: str, image_path: Path, config: Any = None, device: str | None = None
) -> float | None:
    """PickScore's raw preference logit for this image given this prompt.

    Higher is better; the scale is the model's own (logit_scale times a
    cosine, landing around 17-24 on SDXL-class output). Deliberately returned
    raw rather than squashed here: the normalization is a ranking decision and
    lives in ``pipelines/rank.py`` where it is pure and pinned by tests.

    None when the image has no readable pixels; a missing model raises (the
    caller decides whether absence is a skip or an error, exactly as
    ``_dino_model``'s callers do).
    """
    from PIL import Image

    try:
        with Image.open(image_path) as im:
            image = im.convert("RGB")
    except OSError:
        return None
    import torch

    processor, model, resolved = _pickscore_model(config, device)
    image_inputs = processor(images=image, return_tensors="pt").to(resolved)
    text_inputs = processor(
        text=prompt or "", padding=True, truncation=True, max_length=77,
        return_tensors="pt",
    ).to(resolved)
    def _features(out: Any) -> Any:
        # transformers 4.x returns the projected tensor; 5.x returns the full
        # BaseModelOutputWithPooling with the projection written into
        # ``pooler_output``. Both are the same vector.
        return out if torch.is_tensor(out) else out.pooler_output

    with torch.no_grad():
        image_emb = _features(model.get_image_features(**image_inputs))
        image_emb = image_emb / image_emb.norm(dim=-1, keepdim=True)
        text_emb = _features(model.get_text_features(**text_inputs))
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        score = model.logit_scale.exp() * (text_emb @ image_emb.T)
    return float(score.item())


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


def score_reference(reference_path: Path) -> dict[str, Any]:
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

    Takes no ``config``, deliberately, and used to take one it never read. Its
    sibling ``score_view`` needs one because ``dino_cosine`` is gated on
    whether a checkpoint is present; every metric in ``REFERENCE_METRICS`` is
    arithmetic on pixels and there is nothing here to gate. A parameter that is
    accepted and ignored reads as "configurable, one day", which for this
    function is the opposite of the promise the paragraph above makes -- and
    ``test_bench_metrics`` asserts that promise by failing the call if it
    imports torch at all.
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


def compare(a_path: Path, b_path: Path, config: Any = None) -> dict[str, float | None]:
    """Every available metric for an arbitrary pair of images.

    Three numbers answering three different questions, deliberately reported
    side by side and never blended:

    * ``pixel_similarity`` -- the same picture, modulo rescaling and codec.
    * ``silhouette_iou`` -- the same shape, modulo framing and scale.
    * ``dino_cosine`` -- the same subject, modulo everything. Present only when
      the DINOv2 weights are on disk, and readable only as A-against-B: the
      art-versus-render style gap dominates its absolute value.

    Distinct from ``score_view``, which pairs a *reference* against a *render*
    and takes the render's subject from its alpha. Both sides here are
    arbitrary and usually opaque, so the DINOv2 half goes through
    ``prepare_references``; the alpha route would score most pairs None.

    No threshold and no boolean, on the same reasoning ``service.judge`` returns
    a probability with no cut: what counts as "similar enough" depends on what
    the caller is deciding, and a constant here would pretend otherwise. One
    would owe its own pre-registration first.
    """
    out: dict[str, float | None] = {
        "pixel_similarity": pixel_similarity(a_path, b_path),
        "silhouette_iou": silhouette_iou(a_path, b_path, auto_mask=True),
    }
    if dino_available(config):
        out["dino_cosine"] = reference_cosine(a_path, b_path, config)
    return out
