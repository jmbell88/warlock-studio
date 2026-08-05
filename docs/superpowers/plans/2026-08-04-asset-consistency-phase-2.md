# Asset-Consistency Phase 2 Implementation Plan — Standalone 2D Assets

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Warlock produce finished 2D assets, not only concept stages for trellis — icons, sprites and pixel art derived on demand from any reference job, and seamless tiles as a new output kind — each packaged with a JSON manifest.

**Architecture:** Two shapes, deliberately different. *Object* assets (icon / sprite / pixel) are **derived**: pure functions of a done reference's `input.png`, produced on first request under the existing per-artifact `_convert_locks` idiom, exactly as STL and OBJ derive from `model.glb`. Any reference already on disk retroactively gains them. *Tiles* are **generated**: a new `output="tile"` whose seamlessness comes from patching the resident pipe's `Conv2d` layers to `padding_mode="circular"` for the duration of one job, so no inpainting model and no second checkpoint is involved. All the decidable logic lives in new pure modules (`pipelines/asset2d.py`, `pipelines/seam.py`, `pipelines/matting.py`'s fallback path) under the same contract `pipelines/sheet.py` states: no imgui, no service, no queue, testable headlessly.

**Tech Stack:** Python 3.12+, uv, pytest, ruff. Pillow and NumPy for the raster work, OpenCV (already a dependency via `pipelines/reference.py`) for connected components, `transformers` for the optional BiRefNet matting weights, diffusers for the circular-padding patch.

## Global Constraints

- Run every command from `D:\Projects\Warlock`. Verify with `uv run pytest` and `uv run ruff check .` — both must pass before each commit.
- Commit subject format is exactly `Warlock v0.0.7` (project name + fixed version). **Do not bump the version.** Detail goes in the body. End every message with `Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>`.
- **Fully offline.** No runtime network call, ever. Weights load with `local_files_only=True`; a missing weight degrades the feature, reports the one-time `hf download`, and never fetches.
- **A caller-supplied name never becomes a path component.** Every new artifact goes in `service.files.MEDIA` first; that dict is the allowlist, which is why every artifact name in this plan is a fixed literal (`pixel_64.png`, not `pixel_{n}.png`).
- **New pure modules import nothing from `service`, `queue`, `studio`, `imgui`, `moderngl` or `pygame`,** and import Pillow / NumPy / cv2 / torch *inside* functions, as `pipelines/sheet.py`, `pipelines/reference.py` and `pipelines/prompt.py` all do.
- **The frame loop never blocks.** Derivation and matting run on the TaskRunner via `ctx.submit` / `ctx.save_artifact`, or in the worker via `asyncio.to_thread`.
- **Anything the worker records about a finished job's artifacts joins `service.validation.DERIVED_PARAMS`** (`src/warlock/service/validation.py:60`).
- **Lock ordering, once and only in this direction: the per-artifact lock first, `manifest.json`'s lock inside it.** Nothing may take the manifest lock first, or two concurrent derivations deadlock.
- **The bit-identity rule survives.** A job that asks for no tiling must produce the same bytes it produces today: the circular patch is applied per call and reverted in a `finally`, and the unpatched path never reaches the patch function at all.
- Docstrings here explain *why*, at length, British-inflected, with `--` for dashes. Match the surrounding prose density; this codebase does not do terse one-liners.
- **This plan assumes Phase 1 has landed** (`docs/superpowers/plans/2026-08-04-asset-consistency-phase-1.md`). Only Task 8 actually touches Phase 1 code (`settings_2d`'s note helpers); if Phase 1 has not landed, that step's edit is against the pre-Phase-1 `_advanced` instead and everything else is unaffected.

---

### Task 1: `pipelines/asset2d.py` — the pure raster half

Trim, pad, canvas, pivot, quantize. No matting model, no filesystem beyond reading one PNG and writing another, no knowledge that jobs exist. This is where every decision about what an icon *is* gets made, and it is all assertable with synthetic images.

**Files:**
- Create: `src/warlock/pipelines/asset2d.py`
- Test: create `tests/test_asset2d.py`

**Interfaces:**
- Consumes: `reference.subject_mask(image)` (`src/warlock/pipelines/reference.py:129`) as the fallback matte — a boolean NumPy array, True where the subject is.
- Produces:
  - `ICON_SIZE: int`, `PIXEL_SIZES: tuple[int, ...]`, `DEFAULT_PAD: float`, `MIN_ISLAND_PX: int`
  - `trim_box(mask) -> tuple[int, int, int, int] | None`
  - `cutout(image, mask) -> Image` — RGBA, alpha from the mask
  - `icon(image, mask, *, size=ICON_SIZE, pad=DEFAULT_PAD) -> tuple[Image, dict]`
  - `sprite(image, mask, *, pad=0.0) -> tuple[Image, dict]`
  - `pixel(image, mask, *, size, colors=0) -> tuple[Image, dict]`
  - `alpha_report(image) -> dict` with keys `islands`, `partial_fraction`
  - `recipe_hash(recipe: dict | None) -> str | None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_asset2d.py`:

```python
"""The 2D exports, decided without a model and asserted without a GPU.

Same contract pipelines/sheet.py has: everything about what an icon *is* --
where the subject is trimmed to, how much margin it keeps, where the pivot
sits -- is decided here, so the manifest, the file and the preview can never
disagree, and the whole thing is testable with a rectangle on a grey field.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from warlock.pipelines import asset2d


def _subject(size=(128, 128), box=(32, 20, 96, 100), colour=(200, 30, 30)):
    im = Image.new("RGB", size, (200, 200, 200))
    ImageDraw.Draw(im).rectangle(box, fill=colour)
    mask = np.zeros(size[::-1], dtype=bool)
    mask[box[1] : box[3] + 1, box[0] : box[2] + 1] = True
    return im, mask


def test_the_trim_box_is_the_subjects_own_bounds():
    _im, mask = _subject(box=(32, 20, 96, 100))
    assert asset2d.trim_box(mask) == (32, 20, 97, 101)


def test_an_empty_mask_has_no_trim_box():
    assert asset2d.trim_box(np.zeros((16, 16), dtype=bool)) is None


def test_a_cutout_carries_the_mask_as_alpha():
    im, mask = _subject()
    out = asset2d.cutout(im, mask)
    alpha = np.asarray(out)[:, :, 3]
    assert out.mode == "RGBA"
    assert alpha[60, 60] == 255
    assert alpha[2, 2] == 0


def test_an_icon_is_square_at_the_asked_for_size():
    im, mask = _subject()
    out, meta = asset2d.icon(im, mask, size=256)
    assert out.size == (256, 256)
    assert out.mode == "RGBA"
    assert meta["canvas"] == [256, 256]


def test_an_icon_keeps_the_subjects_aspect_ratio():
    # A tall sword must not come out as a square sword. The subject is fitted
    # inside the canvas, never stretched to it.
    im, mask = _subject(box=(50, 10, 70, 110))
    out, _meta = asset2d.icon(im, mask, size=200, pad=0.0)
    alpha = np.asarray(out)[:, :, 3] > 0
    ys, xs = np.nonzero(alpha)
    height = ys.max() - ys.min() + 1
    width = xs.max() - xs.min() + 1
    assert height > width * 2


def test_icon_padding_leaves_a_margin_on_the_long_axis():
    im, mask = _subject(box=(10, 10, 110, 110))
    tight, _ = asset2d.icon(im, mask, size=100, pad=0.0)
    padded, _ = asset2d.icon(im, mask, size=100, pad=0.2)

    def extent(image):
        alpha = np.asarray(image)[:, :, 3] > 0
        ys, _xs = np.nonzero(alpha)
        return ys.max() - ys.min() + 1

    assert extent(padded) < extent(tight)


def test_an_icon_records_where_it_trimmed_from():
    im, mask = _subject(box=(32, 20, 96, 100))
    _out, meta = asset2d.icon(im, mask)
    assert meta["trim"] == [32, 20, 97, 101]
    assert meta["source"] == [128, 128]


def test_a_sprite_is_trimmed_to_the_subject_and_nothing_more():
    im, mask = _subject(box=(32, 20, 96, 100))
    out, meta = asset2d.sprite(im, mask)
    assert out.size == (66, 82)
    assert meta["trim"] == [32, 20, 97, 101]


def test_a_sprites_pivot_is_bottom_centre_by_default():
    im, mask = _subject(box=(32, 20, 96, 100))
    _out, meta = asset2d.sprite(im, mask)
    assert meta["pivot"] == [33.0, 82.0]


def test_a_sprite_is_a_cutout_not_a_crop_of_the_background():
    im, mask = _subject()
    out, _meta = asset2d.sprite(im, mask)
    corner = np.asarray(out)[0, 0]
    assert corner[3] == 0


def test_pixel_art_comes_out_at_the_asked_for_size():
    im, mask = _subject()
    out, meta = asset2d.pixel(im, mask, size=32)
    assert max(out.size) == 32
    assert meta["size"] == 32


def test_pixel_art_uses_nearest_neighbour_so_edges_stay_hard():
    # A resample that blends would put a ramp of in-between colours along the
    # subject's edge, which is the one thing pixel art must not have.
    im, mask = _subject(box=(20, 20, 108, 108), colour=(255, 0, 0))
    out, _meta = asset2d.pixel(im, mask, size=32)
    rgb = np.asarray(out.convert("RGB")).reshape(-1, 3)
    opaque = np.asarray(out)[:, :, 3].reshape(-1) > 0
    reds = rgb[opaque][:, 0]
    assert set(np.unique(reds)) <= {255}


def test_a_palette_cap_bounds_the_colour_count():
    im = Image.new("RGB", (64, 64))
    pixels = im.load()
    for y in range(64):
        for x in range(64):
            pixels[x, y] = (x * 4 % 256, y * 4 % 256, (x + y) * 2 % 256)
    mask = np.ones((64, 64), dtype=bool)

    out, meta = asset2d.pixel(im, mask, size=32, colors=8)

    rgb = np.asarray(out.convert("RGB")).reshape(-1, 3)
    assert len({tuple(c) for c in rgb}) <= 8
    assert meta["palette"] == 8


def test_no_palette_cap_is_recorded_as_none():
    im, mask = _subject()
    _out, meta = asset2d.pixel(im, mask, size=32)
    assert meta["palette"] is None


def test_an_empty_mask_is_refused_rather_than_producing_a_blank_file():
    im = Image.new("RGB", (32, 32), (200, 200, 200))
    empty = np.zeros((32, 32), dtype=bool)
    for call in (
        lambda: asset2d.icon(im, empty),
        lambda: asset2d.sprite(im, empty),
        lambda: asset2d.pixel(im, empty, size=32),
    ):
        with pytest.raises(asset2d.NoSubject):
            call()


def test_the_alpha_report_counts_islands():
    im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    draw.rectangle((4, 4, 20, 20), fill=(255, 0, 0, 255))
    draw.rectangle((40, 40, 58, 58), fill=(0, 255, 0, 255))
    assert asset2d.alpha_report(im)["islands"] == 2


def test_the_alpha_report_ignores_specks_below_the_floor():
    im = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle((4, 4, 40, 40), fill=(255, 0, 0, 255))
    im.putpixel((60, 60), (255, 0, 0, 255))
    assert asset2d.alpha_report(im)["islands"] == 1


def test_the_alpha_report_measures_the_soft_rim():
    hard = Image.new("RGBA", (32, 32), (255, 0, 0, 255))
    soft = hard.copy()
    for x in range(32):
        soft.putpixel((x, 0), (255, 0, 0, 128))
    assert asset2d.alpha_report(soft)["partial_fraction"] > 0
    assert asset2d.alpha_report(hard)["partial_fraction"] == 0.0


def test_a_recipe_hash_is_stable_and_order_independent():
    a = asset2d.recipe_hash({"seed": 1, "base_model": "turbo"})
    b = asset2d.recipe_hash({"base_model": "turbo", "seed": 1})
    assert a == b and len(a) == 12


def test_no_recipe_hashes_to_nothing():
    assert asset2d.recipe_hash(None) is None
    assert asset2d.recipe_hash({}) is None
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_asset2d.py -v`
Expected: FAIL with `ImportError: cannot import name 'asset2d' from 'warlock.pipelines'`.

- [ ] **Step 3: Implement the module**

Create `src/warlock/pipelines/asset2d.py`:

```python
"""Finished 2D assets from a reference image: icon, sprite, pixel art.

The same split ``pipelines/sheet.py`` makes, and for the same reason.
Everything about *what an export is* -- where the subject is trimmed to, how
much margin it keeps, where the pivot sits, how many colours survive -- is
decided here against a boolean mask, so the file, the manifest and any preview
can never disagree, and the whole thing is testable with a rectangle on a grey
field. Producing the mask is somebody else's job (``pipelines/matting.py``, or
``reference.subject_mask`` when the weights are absent), because that is the
part that needs a model.

Pure and torch-free: Pillow, NumPy and cv2 are imported inside the functions,
so this module stays importable without the text2image extra.

Every function takes ``(image, mask)`` and returns ``(image, metadata)``. The
metadata is what the manifest records, and it is returned rather than written
because a pure module has no business deciding where a file goes.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL import Image as _ImageModule

    PILImage = _ImageModule.Image
else:  # pragma: no cover - runtime alias
    PILImage = Any

# The icon canvas. Fixed rather than configurable because the artifact name is
# fixed: ``icon.png`` is one entry in the MEDIA allowlist, and a size knob
# would either need a name per size or would silently mean different things in
# two job directories. 512 downsamples cleanly to every icon size an engine or
# a launcher asks for.
ICON_SIZE = 512

# What a pixel-art export may be reduced to. Each is its own artifact name
# (pixel_32.png ...), for the same allowlist reason.
PIXEL_SIZES = (32, 64, 128)

# Margin left around an icon's subject, as a fraction of the canvas. Enough
# that a silhouette does not touch the frame -- which reads as cropped at
# thumbnail size -- and no more.
DEFAULT_PAD = 0.08

# Below this many pixels an alpha island is a speck of matting noise rather
# than a part of the subject, and counting it would make every export look
# like it had come apart.
MIN_ISLAND_PX = 16

# Alpha strictly between these is "partial" -- the soft rim a matte leaves.
_ALPHA_FLOOR = 0
_ALPHA_CEIL = 255


class NoSubject(ValueError):
    """The mask found nothing to export.

    A named exception rather than a blank image: every caller here is about to
    write a file, and a fully transparent icon.png is indistinguishable from a
    successful export until someone opens it.
    """


def trim_box(mask: Any) -> tuple[int, int, int, int] | None:
    """The subject's bounds as a PIL crop box (left, top, right, bottom).

    Right/bottom are exclusive, matching ``Image.crop``, so the box can be
    handed straight to it without an off-by-one at the call site.
    """
    import numpy as np

    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def cutout(image: PILImage, mask: Any) -> PILImage:
    """The image with the mask as its alpha channel.

    This is the one place the mask becomes transparency, and it is deliberately
    *not* what ``pipelines/reference.py`` does: there the mask drives geometry
    only and is never written back as alpha, because a bad mask would punch
    holes in what trellis reconstructs. Here the export *is* a cutout, so a bad
    mask produces a visibly ragged PNG -- a failure the user can see, which is
    the whole difference.
    """
    import numpy as np
    from PIL import Image

    rgba = np.array(image.convert("RGBA"))
    rgba[:, :, 3] = np.where(mask, 255, 0).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def _trimmed(image: PILImage, mask: Any) -> tuple[PILImage, tuple[int, int, int, int]]:
    box = trim_box(mask)
    if box is None:
        raise NoSubject("the matte found no subject in this image")
    return (cutout(image, mask).crop(box), box)


def icon(
    image: PILImage, mask: Any, *, size: int = ICON_SIZE, pad: float = DEFAULT_PAD
) -> tuple[PILImage, dict[str, Any]]:
    """A square, centred, transparent icon.

    Fitted inside the canvas rather than stretched to it: stretching would
    normalise the aspect ratio away, and a tall sword would come out the same
    shape as a round shield -- which is exactly the confusion an icon set
    exists to avoid.
    """
    from PIL import Image

    cropped, box = _trimmed(image, mask)
    inner = max(1, int(round(size * (1.0 - 2 * pad))))
    scale = min(inner / cropped.width, inner / cropped.height)
    target = (
        max(1, int(round(cropped.width * scale))),
        max(1, int(round(cropped.height * scale))),
    )
    resized = cropped.resize(target, Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(
        resized, ((size - target[0]) // 2, (size - target[1]) // 2), resized
    )
    return (
        canvas,
        {
            "kind": "icon",
            "canvas": [size, size],
            "trim": list(box),
            "source": [image.width, image.height],
            "pad": float(pad),
        },
    )


def sprite(
    image: PILImage, mask: Any, *, pad: float = 0.0
) -> tuple[PILImage, dict[str, Any]]:
    """The subject alone, at its native resolution, with a pivot.

    No canvas and no resize: a sprite is placed by its pivot, so padding it to
    a square would only move the pivot away from the thing it is meant to
    anchor. Bottom-centre is the default because that is where an engine puts
    a standing character's feet, and it is recorded rather than assumed --
    an importer that guesses is an importer that is wrong for half a set.
    """
    from PIL import Image

    cropped, box = _trimmed(image, mask)
    if pad:
        margin = int(round(max(cropped.size) * pad))
        canvas = Image.new(
            "RGBA",
            (cropped.width + 2 * margin, cropped.height + 2 * margin),
            (0, 0, 0, 0),
        )
        canvas.paste(cropped, (margin, margin), cropped)
        cropped = canvas
    return (
        cropped,
        {
            "kind": "sprite",
            "canvas": [cropped.width, cropped.height],
            "trim": list(box),
            "source": [image.width, image.height],
            "pivot": [cropped.width / 2.0, float(cropped.height)],
            "pivot_rule": "bottom-centre",
        },
    )


def pixel(
    image: PILImage, mask: Any, *, size: int, colors: int = 0
) -> tuple[PILImage, dict[str, Any]]:
    """A downsampled, optionally palette-limited cutout.

    Nearest neighbour, never a filter: a resample that blends puts a ramp of
    in-between colours along every edge, and hard edges are the one property
    that makes the result read as pixel art rather than as a small photograph.

    The quantization runs on RGB with alpha carried around it, because Pillow's
    median cut treats alpha as a fourth channel to spend palette entries on --
    which on a cutout means most of the palette describing the transparent
    background.
    """
    import numpy as np
    from PIL import Image

    cropped, box = _trimmed(image, mask)
    scale = size / max(cropped.width, cropped.height)
    target = (
        max(1, int(round(cropped.width * scale))),
        max(1, int(round(cropped.height * scale))),
    )
    small = cropped.resize(target, Image.NEAREST)
    palette = None
    if colors and colors > 0:
        palette = int(colors)
        alpha = small.getchannel("A")
        flat = small.convert("RGB").quantize(
            colors=palette, method=Image.Quantize.MEDIANCUT
        )
        small = flat.convert("RGBA")
        small.putalpha(alpha)
        # Re-cut to the alpha we started with: quantize is nearest-in-palette
        # per pixel and knows nothing about the cutout, so a background pixel
        # can pick up a subject colour and reappear once alpha is restored.
        arr = np.array(small)
        arr[:, :, 3] = np.asarray(alpha)
        small = Image.fromarray(arr, "RGBA")
    return (
        small,
        {
            "kind": "pixel",
            "size": int(size),
            "canvas": [small.width, small.height],
            "trim": list(box),
            "source": [image.width, image.height],
            "palette": palette,
        },
    )


def alpha_report(image: PILImage) -> dict[str, Any]:
    """Two numbers about a finished cutout, both advisory.

    ``islands`` catches a matte that came apart -- a sword whose crossguard
    became a separate object is something the user should see before they ship
    a set. ``partial_fraction`` is the soft-rim measure: a flood-fill fallback
    matte has a hard edge and scores zero, while a model matte legitimately
    leaves a rim, so this is read as "which matte produced this", not as a
    fault.
    """
    import cv2
    import numpy as np

    alpha = np.asarray(image.convert("RGBA"))[:, :, 3]
    solid = (alpha > _ALPHA_FLOOR).astype(np.uint8)
    count, labels = cv2.connectedComponents(solid, connectivity=8)
    islands = 0
    if count > 1:
        sizes = np.bincount(labels.ravel(), minlength=count)[1:]
        islands = int((sizes >= MIN_ISLAND_PX).sum())
    opaque = alpha > _ALPHA_FLOOR
    partial = np.logical_and(opaque, alpha < _ALPHA_CEIL)
    total = int(opaque.sum())
    return {
        "islands": islands,
        "partial_fraction": (float(partial.sum()) / total) if total else 0.0,
    }


def recipe_hash(recipe: dict[str, Any] | None) -> str | None:
    """A short, stable fingerprint of what produced the source image.

    Recorded in the manifest so an exported set can be traced back to the run
    that made it. Sorted keys, so two dicts that say the same thing hash the
    same however they were assembled -- and short, because this is a label in
    a JSON file a human reads, not a cryptographic claim.
    """
    if not recipe:
        return None
    blob = json.dumps(recipe, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_asset2d.py -v`
Expected: PASS (21 tests).

- [ ] **Step 5: Check the purity contract holds**

Run:
```bash
uv run python -c "
import sys, warlock.pipelines.asset2d
bad = [m for m in sys.modules if m.split('.')[0] in ('imgui_bundle','moderngl','pygame','torch')]
bad += [m for m in sys.modules if m.startswith('warlock.service') or m.startswith('warlock.studio') or m == 'warlock.queue']
print(bad or 'clean')
"
```
Expected: `clean`.

- [ ] **Step 6: Run everything and commit**

Run: `uv run pytest && uv run ruff check .`

```bash
git add src/warlock/pipelines/asset2d.py tests/test_asset2d.py
git commit -m "Warlock v0.0.7

pipelines/asset2d: the pure half of the 2D exports -- trim, canvas, pivot,
palette and the alpha QA -- decided against a mask and testable without a GPU.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: `pipelines/matting.py` — an optional model, a mandatory fallback

`asset2d` takes a mask; something has to produce one. BiRefNet is preferred because it is what `trellis-server` uses internally for `bg_removal="birefnet"`, so a 2D export and a 3D input see the same subject boundary. It is a one-time manual download and its absence is non-fatal — the flood-fill matte in `reference.subject_mask` takes over, with visibly rougher edges.

**Note on `trust_remote_code`:** BiRefNet's HF repo ships its own modelling code, which `transformers` executes on load. That code comes from the local snapshot the user downloaded once — nothing is fetched at load time and the offline invariant is intact — but it is still third-party Python running in-process, and the doctor row says so.

**Files:**
- Modify: `src/warlock/models.py` (a `MattingModel` table beside `METRIC_MODELS`)
- Create: `src/warlock/pipelines/matting.py`
- Modify: `src/warlock/doctor.py` (one non-fatal row)
- Test: create `tests/test_matting.py`; extend `tests/test_doctor.py` and `tests/test_offline.py` if it enumerates registries

**Interfaces:**
- Consumes: `models.MATTING_MODELS`, `Config.t2i_model_root`, `reference.subject_mask`.
- Produces:
  - `models.MATTING_MODELS: dict[str, MattingModel]`, `models.DEFAULT_MATTING = "birefnet"`
  - `matting.available(config) -> bool`
  - `matting.mask(image, config=None, *, device="cpu") -> tuple[Any, str]` — `(boolean mask, source)` where source is `"birefnet"` or `"flood"`
  - `matting.unload() -> None`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_matting.py`:

```python
"""The matte: a model when the weights are there, a flood fill when not.

The fallback is the interesting half -- it is what makes every 2D export work
on a fresh checkout, and it must be indistinguishable in *shape* from the model
path so nothing downstream has to know which one ran.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from warlock import models
from warlock.pipelines import matting


def _config(tmp_path):
    return SimpleNamespace(t2i_model_root=tmp_path)


def _subject():
    im = Image.new("RGB", (96, 96), (200, 200, 200))
    ImageDraw.Draw(im).rectangle((24, 24, 72, 72), fill=(30, 30, 30))
    return im


def test_the_registry_entry_carries_a_download_command():
    spec = models.MATTING_MODELS[models.DEFAULT_MATTING]
    assert spec.dir_name and "hf download" in spec.download


def test_no_weights_means_not_available(tmp_path):
    assert matting.available(_config(tmp_path)) is False


def test_weights_on_disk_mean_available(tmp_path):
    spec = models.MATTING_MODELS[models.DEFAULT_MATTING]
    root = tmp_path / spec.dir_name
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    assert matting.available(_config(tmp_path)) is True


def test_without_weights_the_flood_fill_produces_the_mask(tmp_path):
    mask, source = matting.mask(_subject(), _config(tmp_path))
    assert source == "flood"
    assert mask.dtype == bool
    assert mask.shape == (96, 96)
    assert mask[48, 48] and not mask[2, 2]


def test_an_image_with_alpha_uses_it_whatever_the_weights_say(tmp_path):
    # subject_mask already prefers a real alpha channel, and a matting model
    # asked to re-cut an existing cutout can only make it worse.
    im = Image.new("RGBA", (32, 32), (0, 0, 0, 0))
    ImageDraw.Draw(im).rectangle((8, 8, 24, 24), fill=(255, 0, 0, 255))
    mask, source = matting.mask(im, _config(tmp_path))
    assert source == "alpha"
    assert mask[16, 16] and not mask[0, 0]


def test_a_failing_model_falls_back_rather_than_raising(tmp_path, monkeypatch):
    # A corrupt or half-downloaded checkpoint must cost the user edge quality,
    # not the export: the flood fill is always there.
    spec = models.MATTING_MODELS[models.DEFAULT_MATTING]
    root = tmp_path / spec.dir_name
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        matting, "_model_mask", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    mask, source = matting.mask(_subject(), _config(tmp_path))
    assert source == "flood"
    assert mask.any()


def test_the_model_path_is_used_when_it_works(tmp_path, monkeypatch):
    spec = models.MATTING_MODELS[models.DEFAULT_MATTING]
    root = tmp_path / spec.dir_name
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    fake = np.zeros((96, 96), dtype=bool)
    fake[10:20, 10:20] = True
    monkeypatch.setattr(matting, "_model_mask", lambda image, path, device: fake)

    mask, source = matting.mask(_subject(), _config(tmp_path))

    assert source == "birefnet"
    assert np.array_equal(mask, fake)


def test_a_model_mask_that_finds_nothing_falls_back(tmp_path, monkeypatch):
    # An all-false matte would make every export raise NoSubject. The flood
    # fill's answer is worse-looking and right, which beats correct and empty.
    spec = models.MATTING_MODELS[models.DEFAULT_MATTING]
    root = tmp_path / spec.dir_name
    root.mkdir(parents=True)
    (root / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        matting, "_model_mask", lambda image, path, device: np.zeros((96, 96), dtype=bool)
    )
    _mask, source = matting.mask(_subject(), _config(tmp_path))
    assert source == "flood"
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_matting.py -v`
Expected: FAIL with `AttributeError: module 'warlock.models' has no attribute 'MATTING_MODELS'`.

- [ ] **Step 3: Add the registry entry**

In `src/warlock/models.py`, after `METRIC_MODELS` (line 317):

```python
DEFAULT_MATTING = "birefnet"


@dataclass(frozen=True, slots=True)
class MattingModel:
    """A model that separates a subject from its background, host-side.

    Its own table rather than another MetricModel: a metric measures a finished
    asset and its absence costs a number, while this one *produces* an asset's
    alpha and its absence costs edge quality on every 2D export. Both are
    optional and neither is ever downloaded at runtime, which is all they have
    in common.

    ``remote_code`` is stated rather than implied. The published repo ships its
    own modelling code and transformers executes it on load. It comes from the
    snapshot the user downloaded once -- nothing is fetched, and the offline
    invariant holds -- but it is third-party Python running in this process,
    and doctor says so out loud rather than leaving it in a docstring.
    """

    key: str
    label: str
    dir_name: str
    remote_code: bool = False
    download: str = ""


MATTING_MODELS: dict[str, MattingModel] = _table(
    MattingModel(
        # BiRefNet and not something smaller, because trellis-server already
        # uses BiRefNet internally for bg_removal="birefnet" -- so a 2D export
        # and the 3D input derived from the same reference agree about where
        # the subject ends, which two different matting models would not.
        "birefnet",
        "BiRefNet (background removal)",
        "birefnet",
        remote_code=True,
        download=(
            "uvx hf download ZhengPeng7/BiRefNet "
            '--include "*.json" --include "*.py" --include "*.safetensors" '
            "--local-dir models/birefnet"
        ),
    ),
)
```

- [ ] **Step 4: Implement the matting module**

Create `src/warlock/pipelines/matting.py`:

```python
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

The model is loaded per call and dropped, the same reasoning
``text2image._conditioned`` gives for the ControlNet: this runs beside a
resident trellis and a resident SDXL pipe, and a matting model that stayed
loaded would take room from the models that are producing the asset. It runs
on the CPU for the same reason -- a second or two per export, against VRAM
that is genuinely scarce.
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
    """Drop the cached model. Called by nothing today -- the per-call load has
    nothing to keep -- and kept as the counterpart to ``_load`` so a future
    decision to cache it has one place to change."""
    _cache.clear()


_cache: dict[str, Any] = {}


def _load(path: Path, device: str):
    import torch
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
    torch.set_grad_enabled(False)
    _cache[key] = model
    return model


def _model_mask(image: PILImage, path: Path, device: str):
    """The model half, isolated so the fallback logic above can be tested
    without weights by patching exactly this."""
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms

    model = _load(path, device)
    prep = transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    tensor = prep(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        # The published model returns a list of maps at descending scales; the
        # last is the full-resolution one, which is what its own demo reads.
        out = model(tensor)[-1].sigmoid().cpu()
    probability = out[0].squeeze()
    resized = Image.fromarray((probability.numpy() * 255).astype(np.uint8)).resize(
        image.size, Image.BILINEAR
    )
    return np.asarray(resized) > int(THRESHOLD * 255)
```

If `torchvision` is not already a dependency, replace the `transforms` pipeline with plain Pillow resizing plus a NumPy normalise — do not add a dependency for three lines. Check first:

```bash
uv run python -c "import torchvision; print(torchvision.__version__)"
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_matting.py -v`
Expected: PASS.

- [ ] **Step 6: Add the doctor row**

In `src/warlock/doctor.py`, after `_metric_checks`:

```python
def _matting_checks(config: Config) -> list[Check]:
    """The host-side matting weights, non-fatal.

    Missing, every 2D export still works -- the corner flood fill in
    pipelines/reference.py produces the alpha instead, with visibly rougher
    edges on anything that is not on a plain background. That is a quality
    difference the user should be able to see the cause of, which is what this
    row is for.
    """
    checks: list[Check] = []
    for spec in models.MATTING_MODELS.values():
        path = config.t2i_model_root / spec.dir_name
        ok = (path / "config.json").exists()
        if ok:
            detail = str(path)
            if spec.remote_code:
                detail += " -- loads the repo's own modelling code from this directory"
        else:
            detail = (
                f"not found at {path} -- 2D exports fall back to the corner fill; "
                f"download with:\n  {spec.download}"
            )
        checks.append(Check(f"matting model: {spec.label}", ok, detail, fatal=False))
    return checks
```

and add `*_matting_checks(config),` to `run_checks`'s list, directly after `*_t2i_checks(config),`.

Check whether `_metric_checks` is actually in `run_checks` — the grep shows it defined but the list in `run_checks` does not name it. If it is genuinely unreferenced, leave it alone; that is a separate bug and not this plan's to fix. Note it in the commit body if so.

- [ ] **Step 7: Run everything and commit**

Run: `uv run pytest && uv run ruff check .`

```bash
git add src/warlock/models.py src/warlock/pipelines/matting.py src/warlock/doctor.py tests/test_matting.py
git commit -m "Warlock v0.0.7

pipelines/matting: BiRefNet when its weights are on disk, the corner flood fill
when they are not, and never a failed export either way.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Derive the 2D exports on request

`service/derive.py` already derives STL, OBJ, collision and textures from `model.glb` under a per-artifact lock. This adds the same machinery for a *reference*: five new artifacts derived from `input.png`, plus the manifest that describes them.

**Files:**
- Modify: `src/warlock/service/files.py` (`MEDIA`, `DERIVED_2D`, `ready`)
- Modify: `src/warlock/service/derive.py` (`get_file`, `derivable_2d`, `_derive_2d`)
- Test: extend `tests/test_editor_service.py` or create `tests/test_derive_2d.py`

**Interfaces:**
- Consumes: `asset2d.icon/sprite/pixel/alpha_report/recipe_hash`, `matting.mask`, `svc.convert_lock`, `files.ready`.
- Produces:
  - `files.DERIVED_2D: tuple[str, ...]` == `("icon.png", "sprite.png", "pixel_32.png", "pixel_64.png", "pixel_128.png", "manifest.json")`
  - `files.PIXEL_ARTIFACTS: dict[str, int]` mapping artifact name to pixel size
  - `derive.derivable_2d(name) -> bool`
  - each artifact on disk in the job dir, and `manifest.json` accumulating one entry per artifact under its own name

- [ ] **Step 1: Write the failing tests**

Create `tests/test_derive_2d.py`:

```python
"""Icon, sprite and pixel art derived from a finished reference.

The same lazy-derivation contract the STL and OBJ exports have -- produced on
first request, cached beside the source, one lock per (job, artifact) -- with
the source being input.png rather than model.glb. Any reference already on
disk gains them retroactively, which is the whole reason they are derived
rather than generated.
"""

from __future__ import annotations

import json

import pytest
from PIL import Image, ImageDraw

from warlock.service import derive as svc_derive
from warlock.service import files as svc_files
from warlock.service.errors import NotFound, NotReady


def _reference(svc, *, status="done", stage="reference"):
    job_id = svc.store.create("text", "a barrel", {"seed": 1}, stage=stage, status=status)
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (128, 128), (200, 200, 200))
    ImageDraw.Draw(im).rectangle((32, 24, 96, 104), fill=(40, 40, 40))
    im.save(job_dir / "input.png")
    return job_id


def test_an_icon_is_derived_on_first_request(svc):
    job_id = _reference(svc)
    path = svc_derive.get_file(svc, job_id, "icon.png")
    assert path.exists()
    with Image.open(path) as im:
        assert im.size == (512, 512)
        assert im.mode == "RGBA"


def test_a_second_request_serves_the_cached_file(svc):
    job_id = _reference(svc)
    first = svc_derive.get_file(svc, job_id, "icon.png")
    stamp = first.stat().st_mtime_ns
    again = svc_derive.get_file(svc, job_id, "icon.png")
    assert again == first and again.stat().st_mtime_ns == stamp


def test_a_sprite_is_trimmed_to_the_subject(svc):
    job_id = _reference(svc)
    with Image.open(svc_derive.get_file(svc, job_id, "sprite.png")) as im:
        assert im.size == (65, 81)


@pytest.mark.parametrize("name,size", [("pixel_32.png", 32), ("pixel_64.png", 64)])
def test_each_pixel_size_is_its_own_artifact(svc, name, size):
    job_id = _reference(svc)
    with Image.open(svc_derive.get_file(svc, job_id, name)) as im:
        assert max(im.size) == size


def test_the_manifest_accumulates_an_entry_per_artifact(svc):
    job_id = _reference(svc)
    svc_derive.get_file(svc, job_id, "icon.png")
    svc_derive.get_file(svc, job_id, "sprite.png")

    manifest = json.loads(svc_derive.get_file(svc, job_id, "manifest.json").read_text("utf-8"))

    assert manifest["job"] == job_id
    assert set(manifest["artifacts"]) >= {"icon.png", "sprite.png"}
    assert manifest["artifacts"]["sprite.png"]["pivot"] == [32.5, 81.0]
    assert manifest["artifacts"]["icon.png"]["canvas"] == [512, 512]


def test_the_manifest_records_which_matte_produced_the_alpha(svc):
    job_id = _reference(svc)
    svc_derive.get_file(svc, job_id, "icon.png")
    manifest = json.loads((svc.job_dir(job_id) / "manifest.json").read_text("utf-8"))
    entry = manifest["artifacts"]["icon.png"]
    assert entry["matte"] in ("flood", "alpha", "birefnet")
    assert entry["alpha"]["islands"] >= 1


def test_the_manifest_records_the_source_recipe(svc):
    job_id = svc.store.create(
        "text", "a barrel",
        {"seed": 1, "recipe": {"reference": {"base_model": "turbo", "seed": 1}}},
        stage="reference", status="done",
    )
    job_dir = svc.job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (64, 64), (200, 200, 200))
    ImageDraw.Draw(im).rectangle((16, 16, 48, 48), fill=(0, 0, 0))
    im.save(job_dir / "input.png")

    svc_derive.get_file(svc, job_id, "icon.png")

    manifest = json.loads((job_dir / "manifest.json").read_text("utf-8"))
    assert manifest["recipe"] is not None and len(manifest["recipe"]) == 12


def test_a_mesh_job_cannot_derive_a_sprite(svc):
    # The 2D exports are about a reference's pixels. A mesh job's input.png is
    # the picture it was reconstructed *from*, and exporting a sprite of it
    # would quietly claim to be an export of the asset.
    job_id = _reference(svc, stage="model")
    with pytest.raises(NotReady):
        svc_derive.get_file(svc, job_id, "icon.png")


def test_an_unfinished_reference_cannot_derive_anything(svc):
    job_id = _reference(svc, status="running")
    with pytest.raises(NotReady):
        svc_derive.get_file(svc, job_id, "icon.png")


def test_an_unknown_2d_artifact_is_still_refused(svc):
    job_id = _reference(svc)
    with pytest.raises(NotFound):
        svc_derive.get_file(svc, job_id, "pixel_9999.png")


def test_derivable_2d_answers_for_the_whole_set():
    for name in svc_files.DERIVED_2D:
        assert svc_derive.derivable_2d(name)
    assert not svc_derive.derivable_2d("model.stl")
    assert not svc_derive.derivable_2d("nonsense.png")


def test_every_2d_artifact_is_in_the_media_allowlist():
    # MEDIA is what keeps a caller-supplied name off the filesystem; an
    # artifact that skipped it would be underivable and unserveable.
    for name in svc_files.DERIVED_2D:
        assert name in svc_files.MEDIA


def test_the_2d_artifacts_are_not_listed_as_files_on_the_job(svc):
    # Same rule the mesh exports follow: derived artifacts are produced on
    # request, so listing them would claim files that usually are not there.
    job_id = _reference(svc)
    svc_derive.get_file(svc, job_id, "icon.png")
    job = svc.store.get(job_id)
    svc_files.attach_files(job, svc.job_dir(job_id))
    assert "icon.png" not in job["files"]
```

Before running these, check `JobStore.create`'s signature (`src/warlock/db.py:135`) — it takes `status` only because `import_reference` needed it; confirm the keyword name matches what the helper above uses.

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_derive_2d.py -v`
Expected: FAIL with `AttributeError: module 'warlock.service.files' has no attribute 'DERIVED_2D'`.

- [ ] **Step 3: Extend the artifact tables**

In `src/warlock/service/files.py`, add the five artifacts plus the manifest to `MEDIA` (after `"thumb.png"`):

```python
    # The 2D exports. Derived from input.png on a finished reference exactly
    # the way the mesh exports derive from model.glb -- so every reference
    # already on disk gains them, which is the whole reason they are derived
    # rather than produced by a second kind of job.
    #
    # Each pixel size is its own literal name because MEDIA is the allowlist
    # that keeps a caller-supplied string off the filesystem: a pixel_{n}.png
    # pattern would put the number back in the caller's hands.
    "icon.png": "image/png",
    "sprite.png": "image/png",
    "pixel_32.png": "image/png",
    "pixel_64.png": "image/png",
    "pixel_128.png": "image/png",
    "manifest.json": "application/json",
```

and after the `DERIVED` tuple (line 312):

```python
# Everything that is a pure function of a *reference's* input.png. Kept apart
# from DERIVED rather than merged into it: the two sets have different sources,
# different readiness rules and different jobs they apply to, and one tuple
# would have to be filtered at every use anyway.
DERIVED_2D = (
    "icon.png",
    "sprite.png",
    "pixel_32.png",
    "pixel_64.png",
    "pixel_128.png",
    "manifest.json",
)

# Which pixel-art size each artifact name means. The names are literals for the
# allowlist's sake; this is where they get their number back.
PIXEL_ARTIFACTS = {"pixel_32.png": 32, "pixel_64.png": 64, "pixel_128.png": 128}
```

In `ready`, before the `DERIVED` branch:

```python
    if name in DERIVED_2D:
        # A reference's pixels, and only a reference's: a mesh job's input.png
        # is the picture it was reconstructed *from*, so an icon derived from
        # it would quietly claim to be an export of the mesh.
        return (
            job.get("stage") in ("reference", "tile")
            and job.get("status") == "done"
            and (job_dir / "input.png").exists()
        )
```

(`"tile"` is forward-looking and is why Task 7 needs no second edit here — a tile is a finished 2D image and its cutout exports are meaningless but its manifest is not. If you prefer to keep it strictly to `"reference"` now and widen it in Task 7, do that instead and say so in the commit.)

- [ ] **Step 4: Implement the derivation**

In `src/warlock/service/derive.py`, add `derivable_2d` beside `derivable`:

```python
def derivable_2d(name: str) -> bool:
    """Whether ``name`` can be produced from a reference's input.png."""
    return name in files.DERIVED_2D
```

and, in `get_file`, after the mesh `derived` block and before the final `if not path.exists()`:

```python
    if not path.exists() and name in files.DERIVED_2D:
        _derive_2d(svc, job, job_id, job_dir, name)
```

then the implementation, at the end of the module:

```python
# The manifest is written under its own lock, always taken *inside* the
# artifact's. Two derivations of different artifacts genuinely race for it, and
# a consistent order is the only thing standing between that and a deadlock --
# nothing anywhere may take the manifest lock first.
MANIFEST = "manifest.json"


def _derive_2d(
    svc: WarlockService, job: dict, job_id: str, job_dir: Path, name: str
) -> None:
    """Produce one 2D export from the reference's input.png.

    Blocking, like every other derivation here: the matte is a model or a
    flood fill and the quantize is real work, so this must never be called
    from the frame thread.
    """
    from PIL import Image

    from ..pipelines import asset2d, matting

    source = job_dir / "input.png"
    if not source.exists():
        raise NotReady("this job has no reference image")
    with svc.convert_lock(job_id, name):
        # Re-checked inside the lock, for the reason the mesh exports give:
        # whoever waited here was waiting for exactly this file.
        if (job_dir / name).exists():
            return
        if name == MANIFEST:
            # Nothing to compute: the manifest is written by the artifacts
            # themselves, so asking for it before any of them exist means
            # writing the header alone.
            _write_manifest(svc, job, job_id, job_dir, None, None)
            return
        with Image.open(source) as image:
            image.load()
            mask, matte = matting.mask(image, svc.config)
            try:
                if name == "icon.png":
                    out, meta = asset2d.icon(image, mask)
                elif name == "sprite.png":
                    out, meta = asset2d.sprite(image, mask)
                else:
                    out, meta = asset2d.pixel(
                        image, mask, size=files.PIXEL_ARTIFACTS[name]
                    )
            except asset2d.NoSubject as exc:
                # A fact about the image, not a fault -- the same shape
                # glb_to_textures_zip's "this model has no textures" takes.
                raise NotReady(str(exc)) from exc
        meta["matte"] = matte
        meta["alpha"] = asset2d.alpha_report(out)
        # Staged and renamed: a concurrent reader of an artifact this job
        # derived a moment ago must never see a half-written PNG.
        tmp = job_dir / f".{name}.tmp"
        out.save(tmp, "PNG")
        os.replace(tmp, job_dir / name)
        _write_manifest(svc, job, job_id, job_dir, name, meta)


def _write_manifest(
    svc: WarlockService,
    job: dict,
    job_id: str,
    job_dir: Path,
    name: str | None,
    meta: dict | None,
) -> None:
    """Merge one artifact's metadata into the job's manifest.

    Read-modify-write under the manifest's own lock, because that is what it
    is: several artifacts derived concurrently each add their own entry, and
    a whole-file write from a stale read would drop whichever landed first.
    """
    import json

    from ..pipelines import asset2d

    path = job_dir / MANIFEST
    with svc.convert_lock(job_id, MANIFEST):
        try:
            manifest = json.loads(path.read_text("utf-8"))
        except (OSError, ValueError):
            manifest = {}
        if not isinstance(manifest, dict):
            manifest = {}
        manifest.setdefault("version", 1)
        manifest["job"] = job_id
        manifest["prompt"] = job.get("prompt") or ""
        manifest["recipe"] = asset2d.recipe_hash(
            (job.get("params") or {}).get("recipe", {}).get("reference")
        )
        artifacts = manifest.setdefault("artifacts", {})
        if name is not None and meta is not None:
            artifacts[name] = meta
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(tmp, path)
```

with `import os` added to `derive.py`'s imports.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_derive_2d.py -v`
Expected: PASS. If the sprite-size assertion is off by one, fix the *test* against `asset2d.trim_box`'s exclusive-bounds contract — the box is the source of truth and Task 1 pinned it.

- [ ] **Step 6: Run everything and commit**

Run: `uv run pytest && uv run ruff check .`

```bash
git add src/warlock/service/files.py src/warlock/service/derive.py tests/test_derive_2d.py
git commit -m "Warlock v0.0.7

Derive icon/sprite/pixel exports and their manifest from a finished
reference's input.png, under the same per-artifact locks the mesh exports use.
Every reference already on disk gains them.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: The export controls in the inspector

`_downloads` draws one grid from a single `widgets.ARTIFACTS` tuple and gates every entry on `model.glb`. A reference job therefore sees eight buttons it can never press. This makes the grid depend on what the job *is*.

**Files:**
- Modify: `src/warlock/studio/widgets.py` (`ARTIFACTS_2D`, a selector)
- Modify: `src/warlock/studio/panes/inspector.py` (`_downloads`, `_why_blocked`, a manifest summary)
- Test: create `tests/test_inspector_exports.py`

**Interfaces:**
- Consumes: `svc_derive.derivable`, `svc_derive.derivable_2d`, `ctx.save_artifact`.
- Produces: `widgets.ARTIFACTS_2D: tuple[tuple[str, str], ...]`; `widgets.artifacts_for(job) -> tuple[tuple[str, str], ...]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_inspector_exports.py`:

```python
"""Which artifacts a job's Export tab offers.

Pure: the grid is a function of the job row, so what a reference offers and
what a mesh offers is assertable without a GL context.
"""

from __future__ import annotations

from warlock.studio import widgets


def _job(stage="reference", files=()):
    return {"id": "abc123abc123", "stage": stage, "status": "done", "files": list(files)}


def test_a_reference_offers_the_2d_exports():
    names = [n for n, _label in widgets.artifacts_for(_job())]
    assert "icon.png" in names
    assert "sprite.png" in names
    assert "manifest.json" in names


def test_a_reference_is_not_offered_mesh_exports_it_can_never_have():
    names = [n for n, _label in widgets.artifacts_for(_job())]
    assert "model.stl" not in names
    assert "model.fbx" not in names


def test_a_mesh_offers_the_mesh_exports():
    names = [n for n, _label in widgets.artifacts_for(_job(stage="model"))]
    assert "model.glb" in names
    assert "model.stl" in names


def test_a_mesh_is_not_offered_a_sprite_of_its_own_input():
    # input.png on a mesh job is what it was reconstructed from, not an asset.
    names = [n for n, _label in widgets.artifacts_for(_job(stage="model"))]
    assert "sprite.png" not in names


def test_every_job_can_still_take_away_its_source_image():
    for stage in ("reference", "model"):
        names = [n for n, _label in widgets.artifacts_for(_job(stage=stage))]
        assert "input.png" in names


def test_every_offered_name_is_servable():
    from warlock.service import files as svc_files

    for stage in ("reference", "model"):
        for name, _label in widgets.artifacts_for(_job(stage=stage)):
            assert name in svc_files.MEDIA
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_inspector_exports.py -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'artifacts_for'`.

- [ ] **Step 3: Implement the selector**

In `src/warlock/studio/widgets.py`, after `ARTIFACTS` (line 36):

```python
# What a finished *reference* can hand over. A separate tuple rather than a
# filtered ARTIFACTS: the two lists have nothing in common but input.png, and a
# reference offered eight greyed mesh buttons -- which is what it used to get --
# reads as a broken asset rather than as a 2D one.
ARTIFACTS_2D = (
    ("icon.png", "Icon PNG"),
    ("sprite.png", "Sprite PNG"),
    ("pixel_32.png", "Pixel 32"),
    ("pixel_64.png", "Pixel 64"),
    ("pixel_128.png", "Pixel 128"),
    ("manifest.json", "Manifest"),
    ("input.png", "Source image"),
)


def artifacts_for(job: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """The Export tab's grid for one job.

    Keyed on the stage rather than on which files happen to exist: every entry
    in both tuples is *derivable*, so a list built from what is on disk would
    hide exactly the exports that have not been produced yet -- which is all of
    them, the first time.
    """
    if job.get("stage") in ("reference", "tile"):
        return ARTIFACTS_2D
    return ARTIFACTS
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_inspector_exports.py -v`
Expected: PASS.

- [ ] **Step 5: Use it in the pane**

In `src/warlock/studio/panes/inspector.py`, `_downloads`:

```python
    widgets.section("Downloads")
    job_id = job["id"]
    files = set(job.get("files") or [])
    has_mesh = "model.glb" in files
    two_d = job.get("stage") in ("reference", "tile")
    if not imgui.begin_table("downloads", 2):
        return
    for name, label in widgets.artifacts_for(job):
        ready = name in files
        derivable = (
            svc_derive.derivable_2d(name)
            if two_d
            else has_mesh and svc_derive.derivable(name)
        )
        blocked = _why_blocked(ctx, name, ready, derivable)
        ...unchanged from here...
```

and `_why_blocked` gains one branch, before the FBX one:

```python
    if name in ("icon.png", "sprite.png") and not _matting_ready(ctx):
        # Not a refusal -- the fill still produces a file -- so this is a
        # tooltip on an *enabled* button. Handled in _quality_note below.
        return None
```

Actually keep `_why_blocked` unchanged and put the matte note where it belongs — a muted line under the grid, in `_downloads` after `imgui.end_table()`:

```python
    if two_d:
        _matte_note(ctx)
        _manifest_summary(ctx, job)
```

with, after `_why_blocked`:

```python
def _matte_note(ctx: Any) -> None:
    """Why the cutouts look the way they do.

    The exports always work; the question a user has when the edges are ragged
    is whether that is the model or the fallback, and nothing in the UI could
    answer it. The doctor row says the same thing in a place nobody opens
    mid-export.
    """
    from ...pipelines import matting

    if matting.available(ctx.svc.config):
        return
    widgets.muted(
        "Cutouts use the corner fill -- edges are rougher than the matting "
        "model's. See the matting row under Settings for the one-time download."
    )


def _manifest_summary(ctx: Any, job: Any) -> None:
    """The pivot and the alpha QA, read off the manifest that was written.

    Read from the file rather than recomputed: the manifest is the thing an
    importer will consume, so showing anything else here would let the two
    disagree about the asset.
    """
    import json

    path = ctx.job_dir(job["id"]) / "manifest.json"
    if not path.exists():
        return
    try:
        manifest = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return
    for name, entry in sorted((manifest.get("artifacts") or {}).items()):
        bits = [name]
        if entry.get("pivot"):
            bits.append(f"pivot {entry['pivot'][0]:.0f},{entry['pivot'][1]:.0f}")
        alpha = entry.get("alpha") or {}
        if alpha.get("islands", 0) > 1:
            bits.append(f"{alpha['islands']} separate pieces")
        widgets.muted(" - ".join(bits))
```

`_manifest_summary` reads a small JSON file on the frame thread. That is the same class of read `_reference` already does with `ctx.textures`, and the file is a few hundred bytes — but if a profiler shows it, cache it in `AppState` keyed on (job id, mtime) rather than moving it off-thread.

- [ ] **Step 6: Run everything and commit**

Run: `uv run pytest && uv run ruff check .`

```bash
git add src/warlock/studio/widgets.py src/warlock/studio/panes/inspector.py tests/test_inspector_exports.py
git commit -m "Warlock v0.0.7

Give a reference its own Export grid -- icon, sprite, three pixel sizes and the
manifest -- instead of eight greyed mesh buttons it can never press.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: A tile's prompt, and the guidance fields that belong to it

`PROMPT_TEMPLATE` asks for "single object centered on a plain light gray background, full object in frame, no cropping" — every clause of which fights a tileable texture. A tile needs its own template and a narrower slice of the taxonomy: material, palette, condition and setting describe a surface; category, silhouette and rarity describe an object that is not there.

**Files:**
- Modify: `src/warlock/pipelines/prompt.py` (`TILE_TEMPLATE`, `TILE_FIELDS`, `build(..., tile=False)`, `PROMPT_VERSION`)
- Modify: `src/warlock/guidance.py` (`compose_prompt(..., fields=None)`)
- Test: extend `tests/test_prompt.py` and `tests/test_guidance.py`

**Interfaces:**
- Consumes: `guidance._PROMPT_FIELDS`, `guidance._TABLES`.
- Produces:
  - `prompt.TILE_TEMPLATE: str`
  - `prompt.TILE_FIELDS: tuple[str, ...]` == `("material", "condition", "palette", "setting", "genre", "art_style")`
  - `prompt.build(user_prompt, params, *, trigger="", tile=False) -> str`
  - `guidance.compose_prompt(user_prompt, params, fields=None) -> str`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_prompt.py`:

```python
def test_a_tile_prompt_does_not_ask_for_a_single_centred_object():
    from warlock.pipelines import prompt as prompt_mod

    out = prompt_mod.build("mossy cobblestone", {}, tile=True)
    assert "single object" not in out
    assert "seamless" in out and "tileable" in out


def test_a_tile_prompt_keeps_the_users_words_first():
    from warlock.pipelines import prompt as prompt_mod

    out = prompt_mod.build("mossy cobblestone", {}, tile=True)
    assert out.startswith("mossy cobblestone")


def test_a_tile_uses_only_the_surface_half_of_the_taxonomy():
    from warlock import guidance
    from warlock.pipelines import prompt as prompt_mod

    params = guidance.normalize(
        {"material": "stone", "category": "weapon", "silhouette": "tall"}
    )
    out = prompt_mod.build("cobblestone", params, tile=True)
    assert guidance.MATERIALS["stone"].prompt in out
    assert guidance.CATEGORIES["weapon"].prompt not in out


def test_the_object_prompt_is_unchanged_by_the_tile_addition():
    # The default path must be byte-identical: every recipe on disk was
    # recorded against it.
    from warlock.pipelines import prompt as prompt_mod

    assert prompt_mod.build("a barrel", {}) == prompt_mod.PROMPT_TEMPLATE.format(
        prompt="a barrel"
    )


def test_the_tile_field_list_is_a_real_subset_of_the_taxonomy():
    from warlock import guidance
    from warlock.pipelines import prompt as prompt_mod

    assert set(prompt_mod.TILE_FIELDS) < set(guidance.form_fields())
```

(`guidance.MATERIALS["stone"]` and `guidance.CATEGORIES["weapon"]` are both real entries — verified against `src/warlock/guidance.py:126` and `:93`.)

Append to `tests/test_guidance.py`:

```python
def test_compose_prompt_can_be_restricted_to_a_field_subset():
    params = guidance.normalize({"material": "stone", "category": "weapon"})
    both = guidance.compose_prompt("x", params)
    narrow = guidance.compose_prompt("x", params, fields=("material",))
    assert len(narrow) < len(both)
    assert guidance.MATERIALS["stone"].prompt in narrow


def test_compose_prompt_with_no_subset_is_unchanged():
    params = guidance.normalize({"material": "stone", "category": "weapon"})
    assert guidance.compose_prompt("x", params) == guidance.compose_prompt(
        "x", params, fields=None
    )
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_prompt.py tests/test_guidance.py -v`
Expected: FAIL — `build() got an unexpected keyword argument 'tile'`.

- [ ] **Step 3: Implement**

In `src/warlock/guidance.py`, `compose_prompt` gains an optional subset:

```python
def compose_prompt(
    user_prompt: str, params: dict[str, Any], fields: tuple[str, ...] | None = None
) -> str:
    """Fold the guidance fragments into the subject clause of the SDXL prompt.

    ``fields`` restricts which tables contribute, defaulting to all of them in
    their canonical order. The one caller that narrows it is the tile path: a
    tile has no subject, so category, silhouette and rarity describe an object
    that is not in the picture, and a prompt that names one gets an object.

    Unknown or absent values are skipped rather than raising: params may come
    from a job row created before a taxonomy entry was renamed or removed, and
    a slightly less specific prompt beats failing an otherwise valid job.
    """
    parts = [user_prompt.strip()]
    for field in _PROMPT_FIELDS:
        if fields is not None and field not in fields:
            continue
        option = _TABLES[field].get(str(params.get(field, "")))
        if option is not None:
            parts.append(option.prompt)
    return ", ".join(p for p in parts if p)
```

In `src/warlock/pipelines/prompt.py`, after `PROMPT_TEMPLATE`:

```python
# The tile template, and every clause of it is the opposite of the one above.
# PROMPT_TEMPLATE asks for a single centred object, the full object in frame
# and no cropping -- which is a description of exactly what a tileable texture
# must not be. A flat orthographic top-down framing is what makes the circular
# padding in text2image produce something that reads as a surface rather than
# as a photograph of one.
TILE_TEMPLATE = (
    "{prompt}, seamless tileable texture, flat top-down orthographic view, "
    "even diffuse lighting, no shadows, uniform scale, repeating pattern, "
    "no single focal object, no text, no watermark, no border"
)

# The half of the taxonomy that describes a *surface*. The rest -- category,
# silhouette, rarity, mood, emissive, platform -- describes an object, and
# naming one in a tile prompt is how a "cobblestone" tile comes back as a
# picture of a cobblestone.
TILE_FIELDS = ("material", "condition", "palette", "setting", "genre", "art_style")
```

and bump `PROMPT_VERSION` to `2`, with the reason:

```python
# Bumped whenever PROMPT_TEMPLATE, TILE_TEMPLATE, TILE_FIELDS or chunk()
# changes. Recorded by provenance.versions() so a prompt-compiler edit cannot
# silently invalidate a benchmark comparison -- no dependency version moves
# when this file does.
#
# 2: TILE_TEMPLATE and the tile field subset. The object path's output is
# unchanged, so an object recipe recorded under 1 still reproduces exactly;
# the bump is about the compiler, not about any one prompt.
PROMPT_VERSION = 2
```

and `build`:

```python
def build(
    user_prompt: str, params: dict[str, Any], *, trigger: str = "", tile: bool = False
) -> str:
    """The final positive prompt.

    Guidance fragments, then the LoRA trigger (if any), then the framing
    template -- the same assembly text2image.generate() does by hand, exposed
    here so the prompt preview can show it before a job runs. ``tile`` swaps
    both halves at once: the surface-only field subset and the tileable
    template, which have to travel together because either alone produces the
    wrong picture.
    """
    from .. import guidance

    composed = guidance.compose_prompt(
        user_prompt, params, fields=TILE_FIELDS if tile else None
    )
    template = TILE_TEMPLATE if tile else PROMPT_TEMPLATE
    text = template.format(prompt=composed)
    return f"{trigger}, {text}" if trigger else text
```

Check whether `provenance.versions()` reads `PROMPT_VERSION` and whether any test pins it to 1 — `grep -rn PROMPT_VERSION src tests` — and update that test's expectation if so.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_prompt.py tests/test_guidance.py tests/test_provenance.py -v`
Expected: PASS.

- [ ] **Step 5: Run everything and commit**

Run: `uv run pytest && uv run ruff check .`

```bash
git add src/warlock/pipelines/prompt.py src/warlock/guidance.py tests/test_prompt.py tests/test_guidance.py
git commit -m "Warlock v0.0.7

A tile prompt: the tileable-surface template and the surface-only half of the
taxonomy, which have to travel together. The object path is byte-identical.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Circular padding, applied per call and reverted

The whole seamlessness mechanism. Every `Conv2d` in the UNet and the VAE pads its input; switching that padding from zeros to circular for the duration of one sample makes the model's receptive field wrap, and SDXL produces a natively tiling image with no inpainting pass. The pipe is shared with ordinary jobs, so the restore is not optional.

**Files:**
- Modify: `src/warlock/pipelines/text2image.py` (`_circular_padding` context manager, a `tile` argument to `generate`)
- Test: create `tests/test_tiling.py`

**Interfaces:**
- Consumes: `torch.nn.Conv2d.padding_mode`.
- Produces: `text2image.circular_padding(*modules)` — a context manager; `Text2Image.generate(..., tile: bool = False)`; `Text2Image.last_recipe` gains `"tile": bool`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tiling.py`:

```python
"""Circular padding: on for one sample, off again afterwards.

The restore is the part worth testing. The pipe is resident and shared, so a
patch that leaked would make every later job tile -- silently, and only
visibly at the edges of an image nobody looks at the edges of.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from warlock.pipelines import text2image  # noqa: E402


class _Net(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = torch.nn.Conv2d(3, 3, 3, padding=1)
        self.inner = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, padding=1))
        self.linear = torch.nn.Linear(4, 4)


def _modes(net):
    return [m.padding_mode for m in net.modules() if isinstance(m, torch.nn.Conv2d)]


def test_every_conv_is_switched_inside_the_block():
    net = _Net()
    with text2image.circular_padding(net):
        assert _modes(net) == ["circular", "circular"]


def test_every_conv_is_restored_afterwards():
    net = _Net()
    before = _modes(net)
    with text2image.circular_padding(net):
        pass
    assert _modes(net) == before


def test_the_restore_survives_an_exception():
    net = _Net()
    before = _modes(net)
    with pytest.raises(RuntimeError), text2image.circular_padding(net):
        raise RuntimeError("boom")
    assert _modes(net) == before


def test_a_conv_that_was_already_circular_stays_circular():
    # Restored to what each module *was*, not to a blanket "zeros": a model
    # that ships a circular conv must come back the way it arrived.
    net = _Net()
    net.a.padding_mode = "circular"
    with text2image.circular_padding(net):
        pass
    assert net.a.padding_mode == "circular"


def test_nothing_but_convs_is_touched():
    net = _Net()
    with text2image.circular_padding(net):
        assert not hasattr(net.linear, "padding_mode")


def test_none_modules_are_skipped_rather_than_crashing():
    # A pipeline component can legitimately be absent (a VAE-less pipe in a
    # test, a future spec without one), and the context manager is the wrong
    # place to discover that.
    with text2image.circular_padding(None, _Net()):
        pass


def test_the_patch_actually_wraps_the_receptive_field():
    # The claim the whole feature rests on, checked against torch rather than
    # assumed: with circular padding, a conv sees the far edge as adjacent.
    conv = torch.nn.Conv2d(1, 1, 3, padding=1, bias=False)
    torch.nn.init.constant_(conv.weight, 0.0)
    with torch.no_grad():
        conv.weight[0, 0, 1, 0] = 1.0  # take the pixel to the left
    x = torch.zeros(1, 1, 1, 4)
    x[0, 0, 0, 3] = 1.0

    zeros = conv(x)
    with text2image.circular_padding(conv):
        circular = conv(x)

    assert zeros[0, 0, 0, 0].item() == 0.0
    assert circular[0, 0, 0, 0].item() == pytest.approx(1.0)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_tiling.py -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'circular_padding'`.

- [ ] **Step 3: Implement the context manager**

In `src/warlock/pipelines/text2image.py`, after `_noop`:

```python
@contextlib.contextmanager
def circular_padding(*modules: Any):
    """Make every Conv2d in ``modules`` pad circularly, then put it back.

    This is the whole seamlessness mechanism, and it is three lines because
    torch already does the work: ``_ConvNd`` computes
    ``_reversed_padding_repeated_twice`` in ``__init__`` regardless of mode and
    ``_conv_forward`` branches on ``padding_mode != 'zeros'`` at call time, so
    flipping the attribute is enough -- no rebuild, no reload, no second
    checkpoint. With it, the model's receptive field wraps, and SDXL produces
    an image whose left edge continues into its right.

    Reverted in a finally, and to each module's *own* previous mode rather than
    to a blanket "zeros". The pipe is resident and shared with ordinary jobs:
    a patch that leaked would make every later reference tile, which is
    invisible until someone looks at the edges of an image nobody looks at the
    edges of. A ``None`` module is skipped -- a pipeline component can
    legitimately be absent, and this is the wrong place to find out.
    """
    import torch

    previous: list[tuple[Any, str]] = []
    try:
        for module in modules:
            if module is None:
                continue
            for child in module.modules():
                if isinstance(child, torch.nn.Conv2d):
                    previous.append((child, child.padding_mode))
                    child.padding_mode = "circular"
        yield
    finally:
        for child, mode in previous:
            child.padding_mode = mode
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_tiling.py -v`
Expected: PASS. If `test_the_patch_actually_wraps_the_receptive_field` fails, torch's internals have changed and the whole approach needs rechecking — stop and report rather than deleting the test.

- [ ] **Step 5: Thread it through `generate`**

In `Text2Image.generate`, add `tile: bool = False` to the signature and document it:

```python
        ``tile`` switches every Conv2d in the UNet and the VAE to circular
        padding for the duration of this call, which makes the output natively
        seamless. It is a property of one job, never of the pipe: the same
        resident pipeline serves ordinary references, so the patch is applied
        here and reverted before this method returns.
```

Wrap the sampling call. The cleanest place is around the existing `try:` body — replace

```python
        try:
            # After from_pipe, never before: ...
```

with

```python
        stack = contextlib.ExitStack()
        try:
            if tile:
                # The VAE decoder as well as the UNet: a seamless latent
                # decoded through zero-padded convolutions grows a visible
                # border, which is the failure that makes people reach for an
                # inpainting pass they do not need.
                stack.enter_context(
                    circular_padding(self._pipe.unet, getattr(self._pipe, "vae", None))
                )
            # After from_pipe, never before: ...
```

and change the existing `finally:` to

```python
        finally:
            stack.close()
            teardown()
```

Then thread `tile` into the template choice — `generate` currently does `text = PROMPT_TEMPLATE.format(prompt=prompt)`. Replace with:

```python
            from .prompt import TILE_TEMPLATE

            template = TILE_TEMPLATE if tile else PROMPT_TEMPLATE
            text = template.format(prompt=prompt)
```

(the guidance-field subset is applied by the caller, which composes `prompt`; only the framing template belongs here, exactly as it does today.)

Finally, record it: in `_recipe`, add `"tile": bool(tile)` to the dict, and pass `tile` through from `generate`'s call to `self._recipe(...)`.

- [ ] **Step 6: Keep the fake in step**

`tests/conftest.py`'s `FakeText2Image.generate` must accept `tile=False` and record it — `tests/test_fakes_match_real_signatures.py` exists precisely to catch this drift, so run it:

Run: `uv run pytest tests/test_fakes_match_real_signatures.py -v`

Then add to `FakeText2Image.__init__`: `self.tiles: list[bool] = []`, to its `generate` signature: `tile=False`, and in the body: `self.tiles.append(tile)`.

- [ ] **Step 7: Run everything and commit**

Run: `uv run pytest && uv run ruff check .`

```bash
git add src/warlock/pipelines/text2image.py tests/test_tiling.py tests/conftest.py
git commit -m "Warlock v0.0.7

Circular padding on the UNet and VAE for one sample, reverted in a finally.
That is the whole seamless-tile mechanism -- no inpainting model, no second
checkpoint, and the shared pipe comes back exactly as it was.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: `output="tile"` end to end, and the seam report

A tile is a third stage. It stops before trellis like a reference does, but it skips reference normalization and every composition gate — there is no subject to be off-centre — and it is not promotable to 3D.

**Files:**
- Create: `src/warlock/pipelines/seam.py`
- Modify: `src/warlock/service/jobs.py` (`create_job`, `promote_to_model`)
- Modify: `src/warlock/service/validation.py` (`DERIVED_PARAMS`)
- Modify: `src/warlock/vram.py` (`estimate`)
- Modify: `src/warlock/queue.py` (`_generate` text branch)
- Modify: `src/warlock/studio/state.py` (`_kind_of`, `primary_action`)
- Modify: `src/warlock/studio/panes/library.py` (the kind filter)
- Test: create `tests/test_seam.py`; extend `tests/test_queue.py`, `tests/test_service.py`, `tests/test_studio_state.py`, `tests/test_vram.py`

**Interfaces:**
- Consumes: `asset2d` (nothing), `prompt.TILE_FIELDS` (via `text2image`'s caller — the worker composes with `guidance.compose_prompt(..., fields=prompt.TILE_FIELDS)`).
- Produces:
  - `seam.SEAM_MAX: float`, `seam.report(path) -> dict` with `horizontal`, `vertical`, `worst`, `seamless`
  - `seam.wrap_preview(src, dest) -> Path`
  - stage `"tile"` accepted by `create_job(output="tile")`
  - `params["seam_report"]` on a finished tile

- [ ] **Step 1: Write the failing seam tests**

Create `tests/test_seam.py`:

```python
"""Does this image actually tile?

A ratio, not an absolute difference: a busy texture legitimately differs a lot
between any two adjacent columns, so the only meaningful question is whether
the wrap seam differs *more* than the interior does. That normalisation is
what makes one threshold work for cobblestone and for flat plaster alike.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from warlock.pipelines import seam


def _gradient(width=64, height=64, wrap=False):
    """A left-to-right ramp. Wrapped, it is a triangle wave and tiles."""
    x = np.arange(width)
    row = np.abs((x * 2.0 / width) - 1.0) if wrap else x / width
    arr = np.tile((row * 255).astype(np.uint8), (height, 1))
    return Image.fromarray(np.stack([arr] * 3, axis=-1), "RGB")


def test_a_seamless_image_scores_near_the_interior(tmp_path):
    path = tmp_path / "tile.png"
    _gradient(wrap=True).save(path)
    out = seam.report(path)
    assert out["horizontal"] < seam.SEAM_MAX
    assert out["seamless"] is True


def test_a_hard_seam_is_caught(tmp_path):
    path = tmp_path / "tile.png"
    _gradient(wrap=False).save(path)
    out = seam.report(path)
    assert out["horizontal"] > seam.SEAM_MAX
    assert out["seamless"] is False


def test_both_axes_are_measured(tmp_path):
    path = tmp_path / "tile.png"
    _gradient(wrap=False).transpose(Image.ROTATE_90).save(path)
    out = seam.report(path)
    assert out["vertical"] > seam.SEAM_MAX
    assert out["horizontal"] < seam.SEAM_MAX


def test_the_worst_axis_decides_the_verdict(tmp_path):
    path = tmp_path / "tile.png"
    _gradient(wrap=False).save(path)
    out = seam.report(path)
    assert out["worst"] == max(out["horizontal"], out["vertical"])


def test_a_flat_image_is_seamless_and_says_so_without_dividing_by_zero(tmp_path):
    path = tmp_path / "tile.png"
    Image.new("RGB", (32, 32), (128, 128, 128)).save(path)
    out = seam.report(path)
    assert out["seamless"] is True
    assert out["horizontal"] == 0.0


def test_the_wrap_preview_rolls_by_half(tmp_path):
    src, dest = tmp_path / "tile.png", tmp_path / "preview.png"
    image = _gradient(wrap=False)
    image.save(src)

    seam.wrap_preview(src, dest)

    with Image.open(dest) as out:
        rolled = np.asarray(out.convert("RGB"))
    expected = np.roll(np.asarray(image), (32, 32), axis=(0, 1))
    assert np.array_equal(rolled, expected)


def test_a_tiny_image_is_refused_rather_than_measured(tmp_path):
    path = tmp_path / "tile.png"
    Image.new("RGB", (2, 2)).save(path)
    with pytest.raises(ValueError):
        seam.report(path)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_seam.py -v`
Expected: FAIL with `ImportError: cannot import name 'seam'`.

- [ ] **Step 3: Implement `pipelines/seam.py`**

```python
"""Does this image tile, and by how much does it fail to?

The measurement is a ratio and that is the whole design. A hard number -- the
mean absolute difference between the first and last column -- says nothing on
its own, because a photograph of gravel legitimately differs by a lot between
*any* two adjacent columns while flat plaster differs by almost nothing. So the
wrap seam is divided by the mean interior difference: a value near 1 means the
seam is no more of a discontinuity than the texture already contains, and a
value of 8 means it is eight times worse than the picture's own grain.

Advisory, like every other measurement in this codebase. Nothing here fails a
job whose PNG is already on disk; the number goes on the row and the user
decides.

Pure: Pillow and NumPy inside the functions, no torch, no service imports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Above this the seam is a visible edge rather than part of the texture.
# Chosen as "twice the picture's own grain", which is comfortably past the
# noise on a real SDXL tile and comfortably short of an unpatched generation --
# a plain image scores several times this. Re-measure before moving it.
SEAM_MAX = 2.0

# Below this many pixels on a side there is no interior to compare against.
MIN_SIDE = 8


def _ratios(arr: Any) -> tuple[float, float]:
    import numpy as np

    def axis_ratio(a: Any) -> float:
        # a is (rows, columns, channels); the wrap seam is the first column
        # against the last, the interior is every adjacent pair.
        edge = float(np.abs(a[:, 0].astype(float) - a[:, -1].astype(float)).mean())
        interior = float(
            np.abs(np.diff(a.astype(float), axis=1)).mean()
        )
        if interior <= 0.0:
            # A flat image has no grain to normalise against, and a seam in it
            # would be a difference from nothing -- so zero is the honest
            # answer rather than an infinity.
            return 0.0 if edge <= 0.0 else float("inf")
        return edge / interior

    horizontal = axis_ratio(arr)
    vertical = axis_ratio(arr.transpose(1, 0, 2))
    return (horizontal, vertical)


def report(path: Path) -> dict[str, Any]:
    """The seam verdict for one image.

    ``horizontal`` compares the left edge against the right, ``vertical`` the
    top against the bottom, and the worse of the two decides -- a tile that
    wraps one way and not the other is not a tile.
    """
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        im.load()
        if min(im.size) < MIN_SIDE:
            raise ValueError(f"{path.name} is too small to measure a seam in")
        arr = np.asarray(im.convert("RGB"))
    horizontal, vertical = _ratios(arr)
    worst = max(horizontal, vertical)
    return {
        "horizontal": horizontal,
        "vertical": vertical,
        "worst": worst,
        "seamless": bool(worst <= SEAM_MAX),
        "threshold": SEAM_MAX,
    }


def wrap_preview(src: Path, dest: Path) -> Path:
    """The image rolled by half its size in both axes.

    What was the wrap seam is now a cross through the middle of the frame,
    which is the only way to *see* the failure the ratio above measures -- an
    edge is invisible at the edge of a picture and obvious in the centre of
    one.
    """
    import numpy as np
    from PIL import Image

    with Image.open(src) as im:
        im.load()
        arr = np.asarray(im.convert("RGBA" if "A" in im.getbands() else "RGB"))
    rolled = np.roll(arr, (arr.shape[0] // 2, arr.shape[1] // 2), axis=(0, 1))
    out = Image.fromarray(rolled)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.save(dest, "PNG")
    return dest
```

- [ ] **Step 4: Run the seam tests**

Run: `uv run pytest tests/test_seam.py -v`
Expected: PASS.

- [ ] **Step 5: Write the failing service and worker tests**

Append to `tests/test_service.py`:

```python
def test_a_tile_job_is_accepted_and_lands_at_the_tile_stage(svc):
    out = svc_jobs.create_job(svc, kind="text", prompt="cobblestone", output="tile")
    assert svc.store.get(out["id"])["stage"] == "tile"


def test_only_text_jobs_can_be_tiles(svc):
    with pytest.raises(Invalid):
        svc_jobs.create_job(svc, kind="image", image=PNG_BYTES, output="tile")


def test_tiles_can_be_batched_like_references(svc):
    out = svc_jobs.create_job(svc, kind="text", prompt="cobblestone", output="tile", count=3)
    assert len(out["ids"]) == 3


def test_a_tile_cannot_be_promoted_to_a_mesh(svc):
    # There is no subject to reconstruct. Refusing at the door beats two
    # minutes of trellis turning a texture into a lumpy plane.
    out = svc_jobs.create_job(svc, kind="text", prompt="cobblestone", output="tile")
    svc.store.finish(out["id"], "done", None)
    with pytest.raises(Invalid):
        svc_jobs.promote_to_model(svc, out["id"])


def test_a_seam_report_never_survives_into_a_new_job():
    from warlock.service.validation import DERIVED_PARAMS

    assert "seam_report" in DERIVED_PARAMS
```

Reuse whatever `PNG_BYTES`/`Invalid` fixtures `tests/test_service.py` already has rather than adding new ones — read its top before writing this.

Append to `tests/test_queue.py`:

```python
async def test_a_tile_job_never_reaches_trellis(worker):
    job_id = worker.store.create("text", "cobblestone", {"seed": 1}, stage="tile")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert worker.trellis.generate_calls == []
    assert (worker.config.job_dir(job_id) / "input.png").exists()


async def test_a_tile_job_asks_the_pipeline_to_tile(worker):
    job_id = worker.store.create("text", "cobblestone", {"seed": 1}, stage="tile")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert worker._text2image.tiles == [True]


async def test_an_ordinary_reference_does_not_tile(worker):
    job_id = worker.store.create("text", "a barrel", {"seed": 1}, stage="reference")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert worker._text2image.tiles == [False]


async def test_a_tile_is_measured_for_seams_not_for_composition(worker, monkeypatch):
    from warlock.pipelines import seam as seam_mod

    monkeypatch.setattr(
        seam_mod, "report",
        lambda path: {"horizontal": 1.1, "vertical": 1.2, "worst": 1.2,
                      "seamless": True, "threshold": 2.0},
    )
    job_id = worker.store.create("text", "cobblestone", {"seed": 1}, stage="tile")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    params = worker.store.get(job_id)["params"]
    assert params["seam_report"]["seamless"] is True
    # A tile has no subject, so the composition report would be a verdict about
    # something that is deliberately not in the picture.
    assert "reference_report" not in params


async def test_a_failing_seam_measurement_does_not_fail_the_job(worker, monkeypatch):
    from warlock.pipelines import seam as seam_mod

    monkeypatch.setattr(
        seam_mod, "report", lambda path: (_ for _ in ()).throw(ValueError("too small"))
    )
    job_id = worker.store.create("text", "cobblestone", {"seed": 1}, stage="tile")
    worker.start()
    await _wait_until(lambda: worker.store.get(job_id)["status"] == "done")
    await worker.shutdown()

    assert worker.store.get(job_id)["status"] == "done"
    assert "seam_report" not in worker.store.get(job_id)["params"]
```

Append to `tests/test_studio_state.py`:

```python
def test_the_kind_filter_knows_a_tile():
    assert statelib._kind_of({"kind": "text", "stage": "tile"}) == "tile"


def test_a_finished_tile_offers_no_mesh():
    job = {"status": "done", "stage": "tile", "kind": "text", "files": ["input.png"]}
    assert statelib.primary_action(job) != "promote"
```

Append to `tests/test_vram.py`:

```python
def test_a_tile_costs_what_a_reference_costs():
    # Same pipe, same size, one sample -- the circular padding changes no
    # allocation. A stage the estimate did not know would fall through to the
    # mesh branch and price a trellis run that never happens.
    assert vram.estimate("text", "tile", {}, exclusive=True) == vram.estimate(
        "text", "reference", {}, exclusive=True
    )
```

- [ ] **Step 6: Run them and watch them fail**

Run: `uv run pytest tests/test_service.py tests/test_queue.py tests/test_studio_state.py tests/test_vram.py -k "tile or seam" -v`
Expected: FAIL — `Invalid: output must be 'reference' or 'model'`.

- [ ] **Step 7: Accept the new output kind**

In `src/warlock/service/jobs.py`, `create_job`:

```python
    if output not in ("reference", "model", "tile"):
        raise Invalid("output must be 'reference', 'model' or 'tile'", field="output")
    if output in ("reference", "tile") and kind != "text":
        # An image job's reference is the upload; there is nothing to approve.
        # And a tile is generated by definition -- an uploaded one is just an
        # image, and nothing here would do anything to it.
        raise Invalid(f"only text jobs can produce a {output}", field="output")
    if not 1 <= count <= MAX_REFERENCE_COUNT:
        raise Invalid(f"count must be between 1 and {MAX_REFERENCE_COUNT}", field="count")
    if count > 1 and output == "model":
        # N meshes per submit is minutes of GPU each; only the cheap 4-step
        # image stages are worth batching.
        raise Invalid("count > 1 requires output=reference or output=tile", field="count")
```

and the `check_vram` call:

```python
    check_vram(svc, kind, "model" if output == "model" else output, params)
```

In `promote_to_model`, tighten the existing stage check:

```python
    if source["stage"] != "reference":
        raise Invalid(
            "a tile has no subject to reconstruct" if source["stage"] == "tile"
            else "this job is not a reference"
        )
```

In `src/warlock/service/validation.py`, add to `DERIVED_PARAMS`:

```python
    # Advisory, and about this run's pixels -- a reroll inheriting it would
    # claim a seam verdict about an image it is about to replace.
    "seam_report",
```

In `src/warlock/vram.py`, `estimate`:

```python
    if stage in ("reference", "tile"):
        # No reconstruction at all -- a tile never reaches trellis, and the
        # circular-padding patch allocates nothing. Under coexist a warm
        # trellis is still resident beside the pipe and its memory is not
        # given back.
        return sdxl if exclusive else sdxl + TRELLIS_GIB
```

- [ ] **Step 8: Teach the worker**

In `src/warlock/queue.py`'s `_generate` text branch, the tile stage differs in three places.

The composed prompt (line 693):

```python
            from .pipelines import prompt as prompt_lib

            is_tile = job.get("stage") == "tile"
            # A tile's guidance is the surface half of the taxonomy only:
            # category, silhouette and rarity describe an object that is
            # deliberately not in the picture, and naming one gets an object.
            composed = guidance.compose_prompt(
                job["prompt"] or "",
                params,
                fields=prompt_lib.TILE_FIELDS if is_tile else None,
            )
```

The generate call gains `tile=is_tile` (inside the `functools.partial`, alongside `seed=`).

The measurement, replacing the reference-stage `if` block's body for a tile:

```python
                    if is_tile:
                        # A seam verdict, never a composition one: the
                        # rejection rules are all about where a *subject* sits,
                        # and a tile has none. Advisory and swallowed on
                        # failure, the same rule _audit_mesh follows -- the PNG
                        # is on disk and fine.
                        try:
                            from .pipelines import seam

                            params["seam_report"] = await asyncio.to_thread(
                                seam.report, image_path
                            )
                        except Exception:
                            log.exception("seam measurement failed for job %s", job_id)
                    elif is_reference:
                        params["reference_report"] = ...unchanged...
```

and the early return (line 734) widens:

```python
            if job.get("stage") in ("reference", "tile"):
                # The whole point of the split: the user judges the image
                # before anything pays for a trellis run. A tile never has a
                # mesh stage at all.
                _log_mem("after image-only job")
                return
```

The second early return at line 743 (the non-text path) widens the same way, and its comment gains: a tile can only be a text job, so this branch is unreachable for one — but the stage check must still not fall through to trellis.

If Phase 1's retry loop is in place, `is_reference` is already a local there; add `is_tile` beside it and make the "should we measure at all" condition `is_reference or is_tile or retries`.

- [ ] **Step 9: Teach the UI what a tile is**

In `src/warlock/studio/state.py`, `_kind_of`:

```python
def _kind_of(job: dict[str, Any]) -> str:
    """What the filter calls this row.

    Not simply job["kind"]: a text job that stops at a reference, one that
    stops at a tile and one that goes on to a mesh are all the same kind and
    three different things to look for.
    """
    if job.get("kind") in ("rig", "sheet"):
        return job["kind"]
    stage = job.get("stage")
    if stage in ("reference", "tile"):
        return stage
    return "model"
```

and `primary_action`:

```python
    if job.get("stage") == "tile":
        # No mesh, no rig: a tile's next step is to be exported, which the
        # inspector's Export tab is. "Open" selects it and shows that tab.
        return "open" if "input.png" in files else None
    if job.get("stage") == "reference":
        ...unchanged...
```

Check `run_action`'s `"open"` branch (`library.py:321`): it sets `ctx.state.mode = "3d"`, which is wrong for a tile. Make it:

```python
    elif action == "open":
        select(ctx, job_id)
        ctx.state.mode = "2d" if job.get("stage") in ("reference", "tile") else "3d"
```

and add the filter entry in `_filters`:

```python
            ("tile", "tiles"),
```

directly after `("reference", "references")`.

Also check `src/warlock/studio/panes/landing.py:165`, which picks a mode from the stage — a tile should open in `2d` there too, and the existing expression already does that by falling through to `"3d"` only for a non-reference stage, so widen it:

```python
    ctx.state.mode = "2d" if job.get("stage") in ("reference", "tile") else "3d"
```

- [ ] **Step 10: Run everything**

Run: `uv run pytest && uv run ruff check .`
Expected: PASS. Pay particular attention to `tests/test_rerun_regressions.py` — a reroll of a tile must stay a tile, which `rerun_job`'s `stage = source["stage"]` already gives for free; add a test there asserting it if one does not exist.

- [ ] **Step 11: Commit**

```bash
git add src/warlock/pipelines/seam.py src/warlock/service/jobs.py src/warlock/service/validation.py src/warlock/vram.py src/warlock/queue.py src/warlock/studio/state.py src/warlock/studio/panes/library.py src/warlock/studio/panes/landing.py tests/test_seam.py tests/test_service.py tests/test_queue.py tests/test_studio_state.py tests/test_vram.py
git commit -m "Warlock v0.0.7

output=tile: a third stage that stops before trellis, skips every
subject-composition gate it has no subject for, and carries a seam ratio
measured against the texture's own grain.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Tiles in the 2D pane, and the seam report in the inspector

The last piece: a control that submits one. The 2D pane owns every prompt control by invariant, so tile lives there — as an output switch that changes which guidance groups are drawn, because half of them are about a subject.

**Files:**
- Modify: `src/warlock/studio/state.py` (`default_form_2d` gains `output`)
- Modify: `src/warlock/studio/panes/settings_2d.py` (`_output`, `_guidance`, `submit_kwargs`, `validate`, `_submit`)
- Modify: `src/warlock/studio/panes/inspector.py` (`_seam`)
- Test: extend `tests/test_settings_2d_notes.py` or create `tests/test_tile_form.py`

**Interfaces:**
- Consumes: `prompt.TILE_FIELDS`, `settings_2d.submit_kwargs`.
- Produces: `form_2d["output"]` in `("reference", "tile")`; `settings_2d.guidance_groups(form) -> tuple`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_tile_form.py`:

```python
"""The 2D pane's tile switch.

Everything asserted here is a pure function of the form dict, because the pane
draws what these return -- the same split the composed-prompt helpers use.
"""

from __future__ import annotations

from warlock.pipelines import prompt as prompt_lib
from warlock.studio.panes import settings_2d
from warlock.studio.state import default_form_2d


def test_a_new_form_makes_references():
    assert default_form_2d()["output"] == "reference"


def test_the_default_form_submits_a_reference():
    form = default_form_2d()
    form["prompt"] = "a barrel"
    assert settings_2d.submit_kwargs(form)["output"] == "reference"


def test_switching_to_tile_changes_what_is_submitted():
    form = default_form_2d()
    form["prompt"] = "cobblestone"
    form["output"] = "tile"
    assert settings_2d.submit_kwargs(form)["output"] == "tile"


def test_a_tile_form_only_draws_the_surface_guidance_groups():
    form = default_form_2d()
    form["output"] = "tile"
    shown = {f for _title, fields in settings_2d.guidance_groups(form) for f in fields}
    assert shown <= set(prompt_lib.TILE_FIELDS)
    assert "category" not in shown
    assert "material" in shown


def test_an_object_form_draws_every_group():
    form = default_form_2d()
    shown = {f for _title, fields in settings_2d.guidance_groups(form) for f in fields}
    assert "category" in shown and "material" in shown


def test_a_tile_does_not_carry_subject_guidance_it_cannot_use():
    # The fields stay in the form -- switching back must not lose them -- but
    # they must not reach a submit that will ignore them, or the job row claims
    # a taxonomy that did not touch the prompt.
    form = default_form_2d()
    form["prompt"] = "cobblestone"
    form["category"] = "weapon"
    form["material"] = "stone"
    form["output"] = "tile"

    fields = settings_2d.submit_kwargs(form)["guidance_fields"]

    assert "category" not in fields
    assert fields["material"] == "stone"


def test_switching_back_keeps_what_was_typed():
    form = default_form_2d()
    form["category"] = "weapon"
    form["output"] = "tile"
    form["output"] = "reference"
    assert settings_2d.submit_kwargs(form)["guidance_fields"]["category"] == "weapon"


def test_a_tile_still_needs_a_prompt():
    form = default_form_2d()
    form["output"] = "tile"
    assert settings_2d.validate(form)
```

(Both keys are real; see the note in Task 5.)

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_tile_form.py -v`
Expected: FAIL with `KeyError: 'output'`.

- [ ] **Step 3: Add the field and the group selector**

In `src/warlock/studio/state.py`, `default_form_2d`'s dict gains:

```python
        # reference | tile. What this pane submits. A tile is the same pipeline
        # with circular padding and a different framing template, so it belongs
        # to the pane that owns the prompt rather than to a mode of its own --
        # and it is persisted, unlike the seed, because someone making a
        # texture set is making several.
        "output": "reference",
```

In `src/warlock/studio/panes/settings_2d.py`, replace the module-level `GUIDANCE_GROUPS` use with a selector. Keep the tuple and add:

```python
def guidance_groups(form: dict[str, Any]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """The taxonomy groups this form should draw.

    A tile has no subject, so category, silhouette, rarity, mood and emissive
    describe something that is deliberately not in the picture -- drawing them
    would offer controls the prompt compiler then throws away. The fields are
    not *cleared*, only hidden and unsubmitted, so switching back brings back
    what was typed.
    """
    if form.get("output") != "tile":
        return GUIDANCE_GROUPS
    allowed = set(prompt_lib.TILE_FIELDS)
    out = []
    for title, fields in GUIDANCE_GROUPS:
        kept = tuple(f for f in fields if f in allowed)
        if kept:
            out.append((title, kept))
    return tuple(out)
```

with `from ...pipelines import prompt as prompt_lib` in the imports, and `_guidance` iterating `guidance_groups(form)` instead of `GUIDANCE_GROUPS`.

`_guidance` also draws the `platform detail` combo unconditionally — wrap that in `if form.get("output") != "tile":`, since `platform` is not in `TILE_FIELDS`.

- [ ] **Step 4: Submit the right thing**

In `submit_kwargs`:

```python
def submit_kwargs(form: dict[str, Any]) -> dict[str, Any]:
    """The 2D form as create_job takes it.

    ``output`` is this pane's own switch and never "model": this pane is the
    first stage of a two-stage pipeline made visible, and going straight to a
    mesh from here would spend two minutes of GPU on an image nobody has
    approved. A tile has no second stage at all.
    """
    tile = form.get("output") == "tile"
    known = set(guidancelib.form_fields())
    if tile:
        # The hidden groups must not reach the submit either: a row claiming a
        # category the prompt compiler discarded is a lie about what produced
        # the image.
        known &= set(prompt_lib.TILE_FIELDS) | {"base_model", "style_lora"}
    fields = {k: v for k, v in form.items() if k in known and v not in ("", None)}
    return {
        "kind": "text",
        "prompt": form["prompt"].strip(),
        "output": "tile" if tile else "reference",
        ...unchanged...
    }
```

Note the `{"base_model", "style_lora"}` re-add: they are model identity, not subject taxonomy, and a tile still needs a checkpoint.

- [ ] **Step 5: Draw the switch**

In `draw`, between `_profiles` and the Prompt section:

```python
    _output(ctx, form)
```

and the function, near `_presets`:

```python
def _output(ctx: Any, form: dict[str, Any]) -> None:
    """Object or tile -- the one thing that changes what this pane submits.

    A segmented control rather than a combo: there are exactly two, and the
    choice changes which guidance groups are on screen, so it has to read as a
    mode and not as one more select in a column of selects.
    """
    before = form.get("output", "reference")
    form["output"] = widgets.segmented_control(
        "output",
        [("reference", "Object"), ("tile", "Seamless tile")],
        before,
    )
    if form["output"] != before:
        ctx.state.preview_dirty_at = time.monotonic()
    if form["output"] == "tile":
        widgets.muted(
            "A tile is drawn with wrapping convolutions, so its edges match "
            "when repeated. It has no subject, so the object taxonomy is "
            "hidden and it cannot be made into a mesh."
        )
```

and in `_submit`, the cost line:

```python
    count = int(form["count"])
    noun = "tile" if form.get("output") == "tile" else "reference"
    widgets.muted(
        f"{count} {noun}s - a few seconds each"
        if count > 1
        else f"One {noun} - a few seconds"
    )
```

The prompt preview (`_preview`) calls `svc_system.prompt_preview`, which builds through `prompt.build` without a `tile` flag — so a tile's preview would show the object template. Thread it: `svc_system.prompt_preview(svc, raw, prompt, tile=False)` gains the flag, passes it to `prompt_pipeline.build(..., tile=tile)`, and `_preview` sends `state.form_2d.get("output") == "tile"`. Add a test in `tests/test_service.py` asserting the preview of a tile form contains "seamless".

- [ ] **Step 6: Show the seam verdict**

In `src/warlock/studio/panes/inspector.py`, add to `_details_tab`, after `_reference(ctx, job)`:

```python
    _seam(ctx, job)
```

and the function:

```python
def _seam(ctx: Any, job: Any) -> None:
    """Whether the tile actually tiles, and what it looks like wrapped.

    The ratio alone is a number nobody can calibrate against by eye, which is
    what the preview is for: rolled by half, what was the wrap seam runs
    through the middle of the frame, where a discontinuity is obvious.
    """
    report = (job.get("params") or {}).get("seam_report")
    if not isinstance(report, dict):
        return
    if not widgets.header("Seam"):
        return
    worst = float(report.get("worst") or 0.0)
    if report.get("seamless"):
        widgets.text_colored(theme.OK, f"seamless (edge/grain {worst:.2f})")
    else:
        widgets.text_colored(
            theme.WARN,
            f"visible seam (edge/grain {worst:.2f}, over "
            f"{float(report.get('threshold') or 0.0):.2f})",
        )
    widgets.muted(
        f"left/right {float(report.get('horizontal') or 0.0):.2f} - "
        f"top/bottom {float(report.get('vertical') or 0.0):.2f}"
    )
    if ctx.textures is None:
        return
    preview = ctx.job_dir(job["id"]) / "tile_preview.png"
    if not preview.exists():
        if imgui.small_button("Show it wrapped"):
            ctx.submit(
                f"wrap:{job['id']}",
                _make_wrap_preview,
                ctx.job_dir(job["id"]) / "input.png",
                preview,
            )
        return
    texture = ctx.textures.get(f"{job['id']}:wrap", preview)
    if texture is not None:
        imgui.image(widgets.texture_ref(texture), (THUMB_SIZE * 2, THUMB_SIZE * 2))


def _make_wrap_preview(src: Any, dest: Any) -> Any:
    """Off the frame thread: it reads and rewrites a full-size PNG."""
    from ...pipelines import seam

    return seam.wrap_preview(src, dest)
```

`tile_preview.png` is written into the job dir but is deliberately **not** in `MEDIA` or `LISTED` — it is a view, not an artifact, exactly as `input.orig.png` is internal working state. Say so in a comment beside `_make_wrap_preview`.

- [ ] **Step 7: Run everything**

Run: `uv run pytest && uv run ruff check .`
Expected: PASS. `tests/test_studio_state.py` has a persisted-form test — check that adding `output` to `default_form_2d` does not break `restore_form`'s `type(value) is type(default)` rule (it is a `str`, so it restores fine).

- [ ] **Step 8: Commit**

```bash
git add src/warlock/studio/state.py src/warlock/studio/panes/settings_2d.py src/warlock/studio/panes/inspector.py src/warlock/service/system.py tests/test_tile_form.py tests/test_service.py
git commit -m "Warlock v0.0.7

Tile as an output in the 2D pane: it hides the object taxonomy it cannot use,
submits only what reaches the prompt, and the inspector shows the seam ratio
with a half-rolled preview.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>"
```

---

## Final verification

- [ ] `uv run pytest` — full suite green.
- [ ] `uv run ruff check .` — clean.
- [ ] Purity holds for all three new pure modules:
```bash
uv run python -c "
import sys
import warlock.pipelines.asset2d, warlock.pipelines.seam
banned = ('imgui_bundle', 'moderngl', 'pygame', 'torch')
bad = [m for m in sys.modules if m.split('.')[0] in banned]
bad += [m for m in sys.modules if m.startswith('warlock.service') or m.startswith('warlock.studio')]
print(bad or 'clean')
"
```
- [ ] Every new artifact name is in `service.files.MEDIA`: `icon.png`, `sprite.png`, `pixel_32.png`, `pixel_64.png`, `pixel_128.png`, `manifest.json`. `tile_preview.png` is deliberately **not**.
- [ ] Every new `params` key is in `DERIVED_PARAMS`: `seam_report`.
- [ ] Launch the app (`uv run warlock-studio` — confirm the entry point in `pyproject.toml`) and check by hand:
  - Select an **old** reference generated before this work. Its Export tab offers icon, sprite, three pixel sizes and the manifest. Save the icon; open it and confirm a transparent background and a centred subject.
  - Open the saved `manifest.json` and confirm the pivot, trim, canvas, matte source, alpha QA and recipe hash are all there.
  - Without the BiRefNet weights: confirm the muted "corner fill" note under the Export grid, and that the exports still work. Then download the weights per the doctor row, restart, and confirm the note disappears and the cutout edge visibly improves on a subject that is not on a plain background.
  - Switch the 2D pane to **Seamless tile**, confirm the object taxonomy groups vanish and the material/palette ones stay, and generate one. Confirm the seam report reads "seamless", press "Show it wrapped", and confirm no cross-shaped seam through the middle of the preview.
  - Generate an ordinary reference immediately afterwards and confirm it is *not* tiled — this is the circular-padding restore, and it is the one regression that would be invisible without looking for it.
  - Confirm the tile's card offers no "Make 3D", and that `promote_to_model` on it refuses with a message about having no subject.
- [ ] Tile seamlessness by eye: take a generated tile, build a 2×2 montage, and confirm no visible edge:
```bash
uv run python -c "
from pathlib import Path
from PIL import Image
src = Path('assets/<tile job id>/input.png')
with Image.open(src) as im:
    w, h = im.size
    out = Image.new('RGB', (w*2, h*2))
    for x in (0, w):
        for y in (0, h):
            out.paste(im, (x, y))
    out.save('tile_2x2.png')
print('wrote tile_2x2.png')
"
```
