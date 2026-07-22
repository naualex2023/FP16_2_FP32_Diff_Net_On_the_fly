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
import time

import torch

from pp_unet import PipelineParallelUNet

logger = logging.getLogger(__name__)


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
) -> "PIL.Image.Image":
    """Generate a single SD 1.5 image across 2 GPUs (FP32)."""
    pipe = create_sd15_pipeline_parallel(
        model_path=model_path, device_down=device_down, device_up=device_up
    )

    generator = torch.Generator(device="cpu").manual_seed(seed)
    logger.info("Generating '%s' (%d steps, %dx%d)", prompt[:60], steps, width, height)

    t0 = time.perf_counter()
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    ).images[0]
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    image.save(output_path)
    logger.info("Saved %s in %.1fs (%.2fs/step)", output_path, dt, dt / steps)
    return image


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    generate_sd15_pipeline_parallel(
        prompt="a photo of a cat sitting on a windowsill, golden hour, detailed fur",
        steps=20,
        seed=42,
    )