"""
pipeline_parallel_sdxl.py — SDXL FP32 pipeline-parallel on 2× Tesla P40.

This is the main deliverable.  SDXL FP32 (~25.4 GB) exceeds a single P40's
24 GB, so the UNet is split: stage 0 (down) on device_down, stage 1 (mid+up)
on device_up.  Only ~55 MB of activations cross the PCIe boundary per step.

Usage
-----
    from pipeline_parallel_sdxl import generate_sdxl
    generate_sdxl("a cat", steps=25)
"""

from __future__ import annotations

import gc
import logging
import time
from typing import Optional

import torch

from pp_unet import PipelineParallelUNet

logger = logging.getLogger(__name__)


def create_sdxl_pipeline_parallel(
    model_path: str = "./models/sdxl-base-fp16",
    device_down: str = "cuda:0",
    device_up: str = "cuda:1",
    use_fp32: bool = True,
    compile: bool = False,
    lora_path: Optional[str] = None,
    lora_scale: float = 1.0,
    scheduler: str = "default",
):
    """Build a 2-GPU pipeline-parallel SDXL pipeline.

    Parameters
    ----------
    model_path : str
        Local path to SDXL diffusers model (FP16 weights are upcast to FP32
        when ``use_fp32=True``).
    device_down, device_up : str
        CUDA devices for stage 0 / stage 1.
    use_fp32 : bool
        If True, upcast all weights to FP32 (recommended on P40).
    compile : bool
        If True, apply ``torch.compile`` to each block (first run is slow).
    lora_path : str, optional
        Path to a LoRA ``.safetensors`` file to load before splitting.
    lora_scale : float
        Scale for the LoRA adapter.
    scheduler : str
        ``"default"``, ``"ddim"``, ``"euler"``, or ``"dpmpp_2m"``.
    """
    from diffusers import (
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        EulerDiscreteScheduler,
        StableDiffusionXLPipeline,
    )

    dtype = torch.float32 if use_fp32 else torch.float16
    logger.info("Loading SDXL from %s (%s)", model_path, "FP32" if use_fp32 else "FP16")

    pipe = StableDiffusionXLPipeline.from_pretrained(model_path, torch_dtype=dtype)

    # Optional LoRA — load BEFORE splitting so adapter weights are placed
    # alongside their base layers automatically.
    if lora_path:
        logger.info("Loading LoRA weights from %s (scale=%.2f)", lora_path, lora_scale)
        pipe.load_lora_weights(lora_path)
        pipe.fuse_lora(lora_scale=lora_scale)  # bake into base weights
        pipe.unload_lora_weights()

    # Place text encoders on stage 0 (device_down)
    pipe.text_encoder = pipe.text_encoder.to(device_down)
    pipe.text_encoder_2 = pipe.text_encoder_2.to(device_down)
    # VAE must be on device_down: the scheduler loop and final latents live
    # there, and diffusers calls ``self.vae.decode(latents)`` without moving
    # latents to the VAE's device.  (VAE is only ~0.6 GB; device_down still
    # fits in 24 GB: TE ~5 GB + down_blocks ~6.5 GB + VAE ~0.6 GB ≈ 12 GB.)
    pipe.vae = pipe.vae.to(device_down)
    pipe.safety_checker = None

    # Scheduler
    if scheduler == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif scheduler == "euler":
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler == "dpmpp_2m":
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # Wrap UNet
    logger.info(
        "Wrapping SDXL UNet with PipelineParallelUNet (%s + %s)",
        device_down,
        device_up,
    )
    pp_unet = PipelineParallelUNet(
        unet=pipe.unet, device_down=device_down, device_up=device_up
    )
    pipe.unet = pp_unet

    # Optional torch.compile
    if compile:
        logger.warning(
            "torch.compile enabled — first run will be slow (graph tracing)"
        )
        pp_unet.conv_in = torch.compile(pp_unet.conv_in)  # type: ignore[assignment]
        pp_unet.down_blocks = torch.nn.ModuleList(
            [torch.compile(b) for b in pp_unet.down_blocks]  # type: ignore[call-overload]
        )
        pp_unet.mid_block = torch.compile(pp_unet.mid_block)  # type: ignore[assignment]
        pp_unet.up_blocks = torch.nn.ModuleList(
            [torch.compile(b) for b in pp_unet.up_blocks]  # type: ignore[call-overload]
        )
        pp_unet.conv_out = torch.compile(pp_unet.conv_out)  # type: ignore[assignment]

    # The pipeline must not attempt its own device management.
    pipe.enable_model_cpu_offload = lambda: None  # type: ignore[method-assign]
    pipe.enable_sequential_cpu_offload = lambda: None  # type: ignore[method-assign]
    return pipe


def generate_sdxl(
    prompt: str,
    negative_prompt: str = "",
    model_path: str = "./models/sdxl-base-fp16",
    device_down: str = "cuda:0",
    device_up: str = "cuda:1",
    steps: int = 25,
    width: int = 1024,
    height: int = 1024,
    seed: int = 42,
    guidance_scale: float = 7.5,
    use_fp32: bool = True,
    output_path: str = "output_sdxl_pp.png",
    scheduler: str = "default",
    lora_path: Optional[str] = None,
    lora_scale: float = 1.0,
):
    """Generate a single SDXL image across 2 GPUs (FP32)."""
    pipe = create_sdxl_pipeline_parallel(
        model_path=model_path,
        device_down=device_down,
        device_up=device_up,
        use_fp32=use_fp32,
        lora_path=lora_path,
        lora_scale=lora_scale,
        scheduler=scheduler,
    )

    # SDXL-Turbo uses guidance_scale = 0.0 and needs no negative prompt.
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

    logger.info(
        "Generating SDXL %dx%d, %d steps, CFG %.1f",
        width, height, steps, guidance_scale,
    )
    t0 = time.perf_counter()
    image = pipe(**call_kwargs).images[0]
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    image.save(output_path)
    logger.info("Saved %s in %.1fs (%.2fs/step)", output_path, dt, dt / steps)

    # Cleanup
    del pipe
    gc.collect()
    torch.cuda.empty_cache()
    return image


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    generate_sdxl(
        prompt=(
            "A serene Japanese garden with a koi pond, cherry blossoms falling, "
            "golden hour light, photorealistic, 8k, detailed"
        ),
        negative_prompt="blurry, low quality, distorted, ugly",
        steps=25,
        seed=42,
    )