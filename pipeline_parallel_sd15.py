"""
pipeline_parallel_sd15.py — SD 1.5 pipeline-parallel FP32 on 2× Tesla P40.

Educational example: SD 1.5 FP32 (~6.8 GB UNet) fits comfortably on a single
P40 (24 GB), so pipeline parallelism is NOT needed in practice.  This module
exists to demonstrate the technique on a small, fast-to-iterate model and to
provide a correctness oracle (compare 1-GPU FP32 vs 2-GPU pipeline output).

Usage
-----
    from pipeline_parallel_sd15 import generate_sd15_pipeline_parallel
    img = generate_sd15_pipeline_parallel("a cat", steps=20)
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import torch

from pp_unet import PipelineParallelUNet

logger = logging.getLogger(__name__)


class _NullLock:
    """Context manager that does nothing (used when caching is disabled)."""

    def __enter__(self) -> "_NullLock":
        return self

    def __exit__(self, *exc) -> None:
        pass


def create_sd15_pipeline_parallel(
    model_path: str = "./models/sd15-fp16",
    device_down: str = "cuda:0",
    device_up: str = "cuda:1",
) -> "StableDiffusionPipeline":
    """Build a 2-GPU pipeline-parallel SD 1.5 pipeline (FP32)."""
    from diffusers import StableDiffusionPipeline

    logger.info("Loading SD 1.5 FP32 from %s", model_path)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.float32
    )

    # Text encoder → stage 0; VAE → stage 0 too (it's tiny, ~0.3 GB)
    pipe.text_encoder = pipe.text_encoder.to(device_down)
    pipe.vae = pipe.vae.to(device_down)
    pipe.safety_checker = None  # save VRAM
    pipe.enable_attention_slicing() if hasattr(pipe, "enable_attention_slicing") else None

    logger.info("Wrapping UNet with PipelineParallelUNet (%s + %s)", device_down, device_up)
    pp_unet = PipelineParallelUNet(
        unet=pipe.unet, device_down=device_down, device_up=device_up
    )
    pipe.unet = pp_unet

    # Prevent the pipeline from trying its own device management.
    pipe.enable_model_cpu_offload = lambda: None  # type: ignore[method-assign]
    return pipe


def generate_sd15_pipeline_parallel(
    prompt: str,
    model_path: str = "./models/sd15-fp16",
    device_down: str = "cuda:0",
    device_up: str = "cuda:1",
    steps: int = 20,
    seed: int = 42,
    guidance_scale: float = 7.5,
    width: int = 512,
    height: int = 512,
    negative_prompt: str = "",
    output_path: str = "output_sd15_pp.png",
    force_unload: bool = False,
    callback: Optional[callable] = None,
    callback_kwargs: Optional[dict] = None,
) -> "PIL.Image.Image":
    """Generate a single SD 1.5 image across 2 GPUs (FP32).

    Pipeline caching
    ----------------
    The loaded pipeline is kept resident across calls and reused on the next
    generation.  It is automatically freed after an idle timeout (see
    :mod:`pipeline_cache`, configurable via ``SD_IDLE_TIMEOUT`` /
    ``SD_KEEP_ALIVE``).

    Pass ``force_unload=True`` to free VRAM immediately after this call.
    """
    from pipeline_cache import cached_sd15_pipeline, get_cache

    cache = get_cache()
    pipe, entry = cached_sd15_pipeline(
        model_path=model_path,
        device_down=device_down,
        device_up=device_up,
        cache=cache,
    )

    generator = torch.Generator(device="cpu").manual_seed(seed)
    logger.info("Generating '%s' (%d steps, %dx%d)", prompt[:60], steps, width, height)

    call_kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    )

    # Optional progress callback.  diffusers >= 0.27 requires the callback to
    # RETURN the ``callback_kwargs`` dict (it assigns our return value back to
    # its own callback_kwargs and then calls ``.pop()`` on it).  Returning
    # None raises "'NoneType' object has no attribute 'pop'" — the same
    # Twin-mode failure seen on the SDXL path.  ``make_progress_callback``
    # handles this contract for us.
    if callback is not None:
        from callback_utils import make_progress_callback

        call_kwargs["callback_on_step_end"] = make_progress_callback(callback)
        call_kwargs["callback_on_step_end_inputs"] = callback_kwargs or []

    # Hold the per-entry lock so concurrent prompts on this GPU pair serialize.
    entry_lock = entry.lock if entry is not None else _NullLock()
    with entry_lock:
        t0 = time.perf_counter()
        image = pipe(**call_kwargs).images[0]
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    image.save(output_path)
    logger.info("Saved %s in %.1fs (%.2fs/step)", output_path, dt, dt / steps)

    if force_unload:
        key = (
            "sd15",
            os.path.abspath(model_path),
            str(device_down),
            str(device_up),
        )
        if cache.release(key):
            logger.info("force_unload: pipeline freed")
    return image


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    generate_sd15_pipeline_parallel(
        prompt="a photo of a cat sitting on a windowsill, golden hour, detailed fur",
        steps=20,
        seed=42,
    )