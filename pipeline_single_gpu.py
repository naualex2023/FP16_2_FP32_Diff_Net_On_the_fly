"""
pipeline_single_gpu.py — Full FP32 pipeline on a SINGLE GPU (no split).

The pipeline-parallel modules (:mod:`pipeline_parallel_sdxl`,
:mod:`pipeline_parallel_sd15`) split the UNet across two GPUs because FP32
SDXL (~25 GB) does not fit on one Tesla P40 (24 GB).  That split is only
necessary for the very largest models; most checkpoints (SD 1.5 FP32 ~6.8 GB,
SDXL-Turbo, or any FP16 model) sit comfortably on a single GPU.

"Quadro" mode uses this module to run **four** independent single-GPU
pipelines in parallel — one per GPU — each producing an image of the same
prompt with its own seed.  No activation ever crosses a PCIe boundary, so it
is simpler and (for models that fit) faster than the split path.

Usage
-----
    from pipeline_single_gpu import generate_single_gpu
    img = generate_single_gpu("a cat", device="cuda:0", seed=42)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class _NullLock:
    """Context manager that does nothing (used when caching is disabled)."""

    def __enter__(self) -> "_NullLock":
        return self

    def __exit__(self, *exc) -> None:
        pass


def create_single_gpu_pipeline(
    model_path: str = "./models/sdxl-base-fp16",
    device: str = "cuda:0",
    arch: str = "sdxl",
    use_fp32: bool = True,
    lora_path: Optional[str] = None,
    lora_scale: float = 1.0,
    scheduler: str = "default",
):
    """Build a full diffusers pipeline on a SINGLE device (no UNet split).

    Parameters
    ----------
    model_path : str
        Local path or HuggingFace repo ID (resolved via :mod:`model_resolver`).
    device : str
        Single CUDA device, e.g. ``"cuda:0"``.
    arch : str
        ``"sdxl"`` or ``"sd15"``.
    use_fp32 : bool
        If True, load weights as FP32 (recommended for quality on the P40;
        only feasible for models that fit in 24 GB).  If False, FP16.
    lora_path, lora_scale : optional LoRA adapter (loaded + fused before moving
        to the device, same as the pipeline-parallel path).
    scheduler : str
        ``"default"``, ``"ddim"``, ``"euler"``, or ``"dpmpp_2m"``.
    """
    from diffusers import (
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        EulerDiscreteScheduler,
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
    )
    from model_resolver import resolve_model_path

    model_path = resolve_model_path(model_path)
    dtype = torch.float32 if use_fp32 else torch.float16
    logger.info(
        "Loading %s from %s (%s) on %s",
        arch.upper(), model_path, "FP32" if use_fp32 else "FP16", device,
    )

    pipe_cls = StableDiffusionXLPipeline if arch == "sdxl" else StableDiffusionPipeline
    pipe = pipe_cls.from_pretrained(model_path, torch_dtype=dtype)

    # Optional LoRA — load + fuse before placing on the device, mirroring the
    # pipeline-parallel path so adapter weights are baked into the base layers.
    if lora_path:
        logger.info("Loading LoRA weights from %s (scale=%.2f)", lora_path, lora_scale)
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora(lora_scale=lora_scale)
        pipe.unload_lora_weights()

    pipe.safety_checker = None  # save VRAM

    # Scheduler
    if scheduler == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif scheduler == "euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler == "dpmpp_2m":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Place the whole pipeline on the single device.  Diffusers manages the
    # rest internally — no need to disable the offload hooks here because we
    # never call them.
    pipe = pipe.to(device)
    return pipe


def generate_single_gpu(
    prompt: str,
    negative_prompt: str = "",
    model_path: str = "./models/sdxl-base-fp16",
    device: str = "cuda:0",
    arch: str = "sdxl",
    steps: int = 25,
    width: int = 1024,
    height: int = 1024,
    seed: int = 42,
    guidance_scale: float = 7.5,
    use_fp32: bool = True,
    output_path: str = "output_single.png",
    scheduler: str = "default",
    lora_path: Optional[str] = None,
    lora_scale: float = 1.0,
    force_unload: bool = False,
    callback: Optional[callable] = None,
    callback_kwargs: Optional[dict] = None,
):
    """Generate a single image on one GPU (no split).

    Like the pipeline-parallel generators, the loaded pipeline is kept resident
    across calls via :mod:`pipeline_cache` and reused on the next generation.
    Each ``(arch, model, device, fp32, lora, scheduler)`` combination gets its
    own cache entry, so Quadro mode keeps up to four pipelines resident (one
    per GPU).
    """
    from pipeline_cache import cached_single_gpu_pipeline, get_cache

    cache = get_cache()
    pipe, entry = cached_single_gpu_pipeline(
        model_path=model_path,
        device=device,
        arch=arch,
        use_fp32=use_fp32,
        lora_path=lora_path,
        lora_scale=lora_scale,
        scheduler=scheduler,
        cache=cache,
    )

    is_turbo = guidance_scale == 0.0 or steps <= 4
    call_kwargs = dict(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    if not is_turbo:
        call_kwargs["negative_prompt"] = negative_prompt

    if callback is not None:
        from callback_utils import make_progress_callback

        call_kwargs["callback_on_step_end"] = make_progress_callback(callback)
        call_kwargs["callback_on_step_end_inputs"] = callback_kwargs or []

    logger.info(
        "Generating %s %dx%d, %d steps, CFG %.1f on %s",
        arch.upper(), width, height, steps, guidance_scale, device,
    )
    # Hold the per-entry lock so concurrent prompts on the same GPU serialize.
    entry_lock = entry.lock if entry is not None else _NullLock()
    with entry_lock:
        t0 = time.perf_counter()
        image = pipe(**call_kwargs).images[0]
        torch.cuda.synchronize(device)
        dt = time.perf_counter() - t0

    image.save(output_path)
    logger.info("Saved %s in %.1fs (%.2fs/step)", output_path, dt, dt / steps)

    if force_unload:
        from model_resolver import resolve_model_path

        resolved = resolve_model_path(model_path)
        key = (
            "single",
            str(arch),
            os.path.abspath(resolved),
            str(device),
            bool(use_fp32),
            os.path.abspath(lora_path) if lora_path else None,
            float(lora_scale),
            str(scheduler),
        )
        if cache.release(key):
            logger.info("force_unload: pipeline freed")
    return image


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    generate_single_gpu(
        prompt=(
            "A serene Japanese garden with a koi pond, cherry blossoms falling, "
            "golden hour light, photorealistic, 8k, detailed"
        ),
        negative_prompt="blurry, low quality, distorted, ugly",
        arch="sd15",
        steps=20,
        seed=42,
    )