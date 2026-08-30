"""The LoRA trainer child. ``python -m warlock.pipelines.lora_train_worker``.

Runs under the same contract as ``blender_worker``: a JSON spec on stdin,
``[train] <frac> <label>`` progress lines on stdout, a result JSON staged and
renamed at ``spec["result_path"]``, and an exit code the host turns into a
sentence. It is a child for the reason the image pipeline is one
(``text2image_worker``): a training run charges the host ~20 GiB of commit
that an in-process ``del`` never gives back.

The loop is DreamBooth-LoRA on SDXL, reduced to what a *style* needs: one
trigger phrase for every image, latents and text embeddings precomputed once
(so the VAE and both text encoders leave the card before the UNet trains),
rank-r LoRA on the attention projections, fp16 autocast with fp32 adapter
weights, gradient checkpointing, AdamW. No prior-preservation set: this is
"draw like these", not "draw this object", and a regulariser set would be a
second generation pass a user did not ask for.

The card is assumed free: the queue stops trellis and evicts the image pipe
before spawning this, and admission priced it (``vram.LORA_TRAIN_GIB``).
"""

from __future__ import annotations

import contextlib
import json
import random
import sys
from pathlib import Path
from typing import Any

MARKER = "train"


def progress(frac: float, label: str) -> None:
    print(f"[{MARKER}] {frac:.3f} {label}", flush=True)


def _load_images(paths: list[Path], resolution: int) -> list[Any]:
    """Square RGB crops at ``resolution``, as float tensors in [-1, 1]."""
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    out = []
    for path in paths:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            im = ImageOps.fit(im, (resolution, resolution), Image.Resampling.LANCZOS)
            arr = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
        out.append(torch.from_numpy(arr).permute(2, 0, 1))
    return out


def train(spec: dict[str, Any]) -> dict[str, Any]:
    import torch
    from diffusers import DDPMScheduler, StableDiffusionXLPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    base_dir = Path(spec["base_dir"])
    images = [Path(p) for p in spec["images"]]
    out_dir = Path(spec["out_dir"])
    trigger = str(spec["trigger"]).strip()
    steps = int(spec["steps"])
    rank = int(spec["rank"])
    lr = float(spec["learning_rate"])
    resolution = int(spec["resolution"])
    seed = int(spec.get("seed", 0))
    if not images:
        raise RuntimeError("no training images")
    if not torch.cuda.is_available():
        raise RuntimeError("training needs a CUDA card")
    device = torch.device("cuda")
    torch.manual_seed(seed)
    rng = random.Random(seed)

    progress(0.01, "Loading the base model")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        str(base_dir), torch_dtype=torch.float16, local_files_only=True, variant="fp16"
    )
    unet = pipe.unet
    vae = pipe.vae
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # --- precompute: latents in fp32 (the SDXL VAE overflows in fp16), then
    # the text embedding for the one prompt every image shares.
    progress(0.03, f"Encoding {len(images)} images")
    vae.to(device, dtype=torch.float32)
    latents = []
    with torch.no_grad():
        for tensor in _load_images(images, resolution):
            pixel = tensor.unsqueeze(0).to(device, dtype=torch.float32)
            latent = vae.encode(pixel).latent_dist.sample() * vae.config.scaling_factor
            latents.append(latent.to(torch.float16).cpu())
    vae.to("cpu")
    del vae
    pipe.vae = None

    progress(0.05, "Encoding the trigger")
    pipe.text_encoder.to(device)
    pipe.text_encoder_2.to(device)
    with torch.no_grad():
        prompt_embeds, _neg, pooled, _neg_pooled = pipe.encode_prompt(
            trigger, device=device, num_images_per_prompt=1, do_classifier_free_guidance=False
        )
    prompt_embeds = prompt_embeds.detach().to(torch.float16)
    pooled = pooled.detach().to(torch.float16)
    pipe.text_encoder.to("cpu")
    pipe.text_encoder_2.to("cpu")
    pipe.text_encoder = None
    pipe.text_encoder_2 = None
    torch.cuda.empty_cache()

    # --- the adapter
    unet.requires_grad_(False)
    unet.to(device, dtype=torch.float16)
    unet.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
    )
    with contextlib.suppress(Exception):
        unet.enable_gradient_checkpointing()
    trainable = [p for p in unet.parameters() if p.requires_grad]
    for p in trainable:
        p.data = p.data.to(torch.float32)
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-2)
    add_time_ids = torch.tensor(
        [[resolution, resolution, 0, 0, resolution, resolution]],
        device=device, dtype=torch.float16,
    )
    scaler = torch.amp.GradScaler("cuda")

    unet.train()
    loss_value = 0.0
    progress(0.06, "Training")
    for step in range(steps):
        latent = latents[rng.randrange(len(latents))].to(device)
        noise = torch.randn_like(latent)
        timestep = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (1,), device=device
        ).long()
        noisy = noise_scheduler.add_noise(latent, noise, timestep)
        with torch.autocast("cuda", dtype=torch.float16):
            pred = unet(
                noisy,
                timestep,
                encoder_hidden_states=prompt_embeds,
                added_cond_kwargs={"text_embeds": pooled, "time_ids": add_time_ids},
            ).sample
            loss = torch.nn.functional.mse_loss(pred.float(), noise.float())
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        scaler.step(optimizer)
        scaler.update()
        loss_value = float(loss.detach())
        if step % 10 == 0 or step == steps - 1:
            progress(
                0.06 + 0.9 * (step + 1) / steps,
                f"Step {step + 1}/{steps}, loss {loss_value:.3f}",
            )

    progress(0.97, "Saving the adapter")
    unet.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionXLPipeline.save_lora_weights(
        save_directory=str(out_dir),
        unet_lora_layers=state,
        weight_name="pytorch_lora_weights.safetensors",
        safe_serialization=True,
    )
    progress(1.0, "Trained")
    return {
        "ok": True,
        "steps": steps,
        "images": len(images),
        "rank": rank,
        "loss": loss_value,
        "weights": str(out_dir / "pytorch_lora_weights.safetensors"),
    }


def main() -> int:
    """``blender_worker.main``'s shape: a malformed spec is a sentence and an
    exit code, the result is staged and renamed."""
    try:
        spec = json.loads(sys.stdin.read())
        result_path = Path(spec["result_path"])
    except (ValueError, TypeError, KeyError) as exc:
        print(f"the trainer spec on stdin is not usable: {exc}", file=sys.stderr)
        return 2
    try:
        import diffusers  # noqa: F401
        import peft  # noqa: F401
        import torch  # noqa: F401
    except ImportError as exc:
        print(f"the text2image extra is not installed: {exc}", file=sys.stderr)
        return 3
    result = train(spec)
    tmp = result_path.with_name(result_path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(result), encoding="utf-8")
        tmp.replace(result_path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
