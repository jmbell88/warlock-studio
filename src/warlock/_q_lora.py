"""The LoRA training stage of :class:`~.queue.Worker`.

One kind, ``lora_train``: stop everything else on the card, run the trainer
child, register what it wrote. The registration goes through
``generation.import_lora`` so the trained adapter is indistinguishable from a
downloaded one to every reader downstream.
"""

from __future__ import annotations

import asyncio
import functools
import logging
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .queue import Worker

log = logging.getLogger(__name__)


class LoraOps:
    """Mixed into :class:`~.queue.Worker`."""

    async def _lora_train(self: Worker, job: dict[str, Any]) -> None:
        from . import fetch, generation, models, rigging
        from .pipelines import lora_train

        job_id = job["id"]
        params = job["params"]
        job_dir = self.config.job_dir(job_id)
        images = sorted((job_dir / "train").glob("*.png"))
        if not images:
            raise RuntimeError("this job has no training images")
        base_key = str(params.get("base_model") or models.DEFAULT_BASE_MODEL)
        spec = models.BASE_MODELS[base_key]
        base_dir = fetch.base_model_dir(self.config, spec)
        trigger = str(params.get("trigger") or "")
        steps = int(params.get("steps") or lora_train.DEFAULT_STEPS)

        self.progress.update(
            job_id, phase="prepare", label="Freeing the card", inner=0.0,
            inner_next=1.0, nominal=20.0, detail=f"{len(images)} images",
        )
        # The card, alone: a training run is priced exclusive, so the warm
        # trellis and the resident image pipe both go before the child spawns.
        await asyncio.to_thread(self.trellis.stop)
        await self._evict_t2i()

        def on_progress(frac: float, label: str) -> None:
            self.progress.update(
                job_id, phase="train", label=label, inner=frac,
                inner_next=min(frac + 0.02, 1.0), nominal=float(steps) / 1.5, detail="",
            )

        out_dir = job_dir / "lora"
        result = await asyncio.to_thread(
            functools.partial(
                rigging.run_worker,
                lora_train.train_spec(
                    base_dir,
                    images,
                    out_dir,
                    job_dir,
                    trigger=trigger,
                    steps=steps,
                    seed=int(params.get("seed", 0)),
                ),
                on_progress=on_progress,
                on_start=self._note_blender,
                timeout=lora_train.TIMEOUT,
                module="warlock.pipelines.lora_train_worker",
                marker=lora_train.MARKER,
                name="LoRA trainer",
            )
        )
        if self._cancel is not None and self._cancel.event.is_set():
            return
        weights = Path(result.get("weights") or (out_dir / lora_train.WEIGHTS_NAME))
        if not weights.exists():
            raise RuntimeError("the trainer reported success but wrote no adapter")

        self.progress.update(
            job_id, phase="publish", label="Registering the style", inner=0.95,
            inner_next=1.0, nominal=2.0, detail="",
        )
        manifest = await asyncio.to_thread(
            functools.partial(
                generation.import_lora,
                self.config,
                weights,
                label=str(params.get("label") or "Trained style"),
                family=models.FAMILY_SDXL,
                trigger_text=trigger,
                tuned_weight=lora_train.TRAINED_WEIGHT,
                license="trained locally from the user's images",
                commercial=True,
                source_url=f"trained:{job_id}",
            )
        )
        if self._cancel is not None:
            self._cancel.commit()
        report = {k: result.get(k) for k in ("steps", "images", "rank", "loss")}
        report["lora"] = asdict(manifest)
        params["lora_result"] = report
        await asyncio.to_thread(self.store.set_params, job_id, params)
        log.info("trained style LoRA %s (%s) in job %s", manifest.key, manifest.label, job_id)
