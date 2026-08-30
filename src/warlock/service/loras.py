"""Style LoRAs the user brings: importing a file, training one, removing one.

The door for ``generation.import_lora`` -- which was fully built and called
from no pane -- and for the trainer child. Everything here refuses with a
``field`` so the settings form can ring the control; the pane never
validates a second time.

A trained adapter is imported through the same call a downloaded file is, so
there is one registry path: a LoRA is a ``STYLE_LORAS`` entry with a file
under ``t2i_model_root/loras``, however it got there.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .. import generation, models
from ..pipelines import lora_train
from .core import WarlockService
from .errors import Invalid
from .validation import check_base_model_weights, check_vram

log = logging.getLogger(__name__)

#: What a training image may be. Read through Pillow at the door, so a file
#: the trainer cannot open costs the request, not a queue slot and a load.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


def catalog(svc: WarlockService) -> list[dict[str, Any]]:
    """Every style LoRA the picker offers, imported ones marked by ``source``."""
    return generation.lora_catalog(svc.config)


def imported(svc: WarlockService) -> list[dict[str, Any]]:
    """Only the adapters the user added, newest last."""
    return [asdict(m) for m in generation.load_lora_manifests(svc.config)]


def import_lora(
    svc: WarlockService,
    source: Path | str,
    *,
    label: str,
    family: str = models.FAMILY_SDXL,
    trigger_text: str = "",
    tuned_weight: float = models.DEFAULT_LORA_WEIGHT,
    commercial: bool = False,
    license: str = "",
) -> dict[str, Any]:
    """Copy a ``.safetensors`` adapter into managed storage and register it."""
    text = (label or "").strip()
    if not text:
        raise Invalid("give the style a name", field="label")
    if len(text) > lora_train.MAX_LABEL:
        raise Invalid(f"a name is at most {lora_train.MAX_LABEL} characters", field="label")
    if family not in models.FAMILIES:
        raise Invalid(f"family must be one of {list(models.FAMILIES)}", field="family")
    trigger = (trigger_text or "").strip()
    if len(trigger) > lora_train.MAX_TRIGGER:
        raise Invalid(
            f"trigger words are at most {lora_train.MAX_TRIGGER} characters", field="trigger_text"
        )
    try:
        weight = float(tuned_weight)
    except (TypeError, ValueError) as exc:
        raise Invalid("weight must be a number", field="tuned_weight") from exc
    if not 0.0 < weight <= models.LORA_WEIGHT_MAX:
        raise Invalid(
            f"weight must be between 0 and {models.LORA_WEIGHT_MAX}", field="tuned_weight"
        )
    path = Path(source)
    if path.suffix.lower() != ".safetensors":
        raise Invalid("a LoRA is a .safetensors file", field="source")
    if not path.is_file():
        raise Invalid(f"{path.name} is not a file", field="source")
    manifest = generation.import_lora(
        svc.config,
        path,
        label=text,
        family=family,
        trigger_text=trigger,
        tuned_weight=weight,
        license=license,
        commercial=bool(commercial),
        source_url="local file",
    )
    log.info("imported style LoRA %s as %s", path.name, manifest.key)
    return asdict(manifest)


def remove_lora(svc: WarlockService, key: str) -> dict[str, Any]:
    """Delete an imported adapter's file and manifest. Built-ins are refused."""
    manifest = generation.imported_lora(svc.config, key)
    if manifest is None:
        raise Invalid("that style was not imported here, so it cannot be removed", field="key")
    generation.remove_imported_lora(svc.config, key)
    return {"ok": True, "key": key}


def train_lora(
    svc: WarlockService,
    images: Sequence[Path | str],
    *,
    label: str,
    trigger: str,
    steps: int = lora_train.DEFAULT_STEPS,
    base_model: str | None = None,
) -> dict[str, Any]:
    """Queue a training run. The images are copied into the job's directory
    at the door, so a folder the user edits later cannot change a row.

    The card is priced as exclusive (``vram.LORA_TRAIN_GIB``): the worker
    stops trellis and evicts the image pipe before it spawns the trainer.
    """
    text = (label or "").strip()
    if not text:
        raise Invalid("give the style a name", field="label")
    if len(text) > lora_train.MAX_LABEL:
        raise Invalid(f"a name is at most {lora_train.MAX_LABEL} characters", field="label")
    words = (trigger or "").strip()
    if not words:
        raise Invalid(
            "give the style trigger words -- the phrase that summons it in a prompt",
            field="trigger",
        )
    if len(words) > lora_train.MAX_TRIGGER:
        raise Invalid(
            f"trigger words are at most {lora_train.MAX_TRIGGER} characters", field="trigger"
        )
    try:
        count = lora_train.check_steps(steps)
    except ValueError as exc:
        raise Invalid(str(exc), field="steps") from exc
    paths = [Path(p) for p in images]
    if len(paths) < lora_train.MIN_IMAGES:
        raise Invalid(
            f"a style needs at least {lora_train.MIN_IMAGES} images", field="images"
        )
    if len(paths) > lora_train.MAX_IMAGES:
        raise Invalid(
            f"at most {lora_train.MAX_IMAGES} images per style", field="images"
        )
    from PIL import Image

    for path in paths:
        if path.suffix.lower() not in IMAGE_SUFFIXES or not path.is_file():
            raise Invalid(f"{path.name} is not an image file", field="images")
        try:
            with Image.open(path) as im:
                im.verify()
        except Exception as exc:
            raise Invalid(f"{path.name} could not be read: {exc}", field="images") from exc

    base_key = str(base_model or models.DEFAULT_BASE_MODEL)
    spec = models.BASE_MODELS.get(base_key)
    if spec is None:
        raise Invalid(f"unknown base model {base_key!r}", field="base_model")
    if spec.family != models.FAMILY_SDXL or base_key not in models.cfg_bases():
        # An SDXL checkpoint at full CFG: a distilled base (turbo, Hyper-SD)
        # shares the architecture but not the schedule the loss assumes, and
        # FLUX.2 is a different network altogether. ``cfg_bases`` is the one
        # spelling of "undistilled" the registry already has.
        usable = sorted(
            k for k in models.cfg_bases() if models.BASE_MODELS[k].family == models.FAMILY_SDXL
        )
        raise Invalid(
            f"a style is trained on an undistilled SDXL checkpoint; pick one of {usable}",
            field="base_model",
        )
    check_base_model_weights(svc, spec)

    params: dict[str, Any] = {
        "base_model": base_key,
        "label": text,
        "trigger": words,
        "steps": count,
        "images": len(paths),
    }
    check_vram(svc, "lora_train", "reference", params)

    new_id = uuid.uuid4().hex[:12]
    job_dir = svc.job_dir(new_id)
    train_dir = job_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(paths):
        with Image.open(path) as im:
            im.convert("RGB").save(train_dir / f"{index:03d}.png")
    svc.store.create("lora_train", text, params, new_id)
    svc.wake_worker()
    return {"id": new_id, "images": len(paths)}
