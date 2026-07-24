"""
pipeline_parallel_dit.py — DiT FP32 pipeline-parallel on 2× Tesla P40.

DiT-based diffusion models (PixArt-Alpha/Sigma, Sana, Stable Diffusion 3, Flux,
Lumina, …) replace the classic UNet with a Transformer backbone.  In FP32 these
easily reach **30–50 GB**, far beyond a single 24 GB P40.

This module is the DiT analogue of :mod:`pipeline_parallel_sdxl`: it splits the
transformer's block sequence in half across two GPUs via
:class:`pp_dit.PipelineParallelDiT`.  Text encoders / VAE stay on
``device_down``; only ~one hidden-state tensor crosses the PCIe boundary per
step.

Usage
-----
    from pipeline_parallel_dit import generate_dit
    generate_dit("a cat", model_path="PixArt-alpha/PixArt-XL-2-1024-MS")
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import torch

from pp_dit import PipelineParallelDiT

logger = logging.getLogger(__name__)


# Diffusers pipeline classes that use a transformer backbone (DiT).  The first
# match wins.  Add more here as diffusers grows new DiT families.
_DIT_PIPELINE_CLASSES: tuple[str, ...] = (
    "PixArtAlphaPipeline",
    "PixArtSigmaPipeline",
    "SanaPipeline",
    "StableDiffusion3Pipeline",
    "FluxPipeline",
    "LuminaPipeline",
    "AuraFlowPipeline",
)


def _load_dit_pipeline_class(model_path: str):
    """Import the right diffusers pipeline class for *model_path*.

    Reads ``model_index.json`` → ``_class_name`` and falls back to
    ``PixArtAlphaPipeline``.
    """
    import json
    import diffusers

    idx_file = os.path.join(model_path, "model_index.json")
    cls_name = "PixArtAlphaPipeline"
    if os.path.isfile(idx_file):
        try:
            with open(idx_file, "r", encoding="utf-8") as f:
                cls_name = json.load(f).get("_class_name", cls_name)
        except Exception:
            pass

    if not hasattr(diffusers, cls_name):
        raise RuntimeError(
            f"Installed diffusers has no class {cls_name!r} (needed for "
            f"{model_path}).  Upgrade diffusers."
        )
    return getattr(diffusers, cls_name)


def create_dit_pipeline_parallel(
    model_path: str = "PixArt-alpha/PixArt-XL-2-1024-MS",
    device_down: str = "cuda:0",
    device_up: str = "cuda:1",
    use_fp32: bool = True,
    scheduler: str = "default",
):
    """Build a 2-GPU pipeline-parallel DiT pipeline.

    Parameters
    ----------
    model_path : str
        Local path or HuggingFace repo ID of a DiT diffusers model.
    device_down, device_up : str
        CUDA devices for stage 0 / stage 1.
    use_fp32 : bool
        If True (recommended on P40), upcast weights to FP32.
    scheduler : str
        ``"default"``, ``"ddim"``, ``"euler"``, or ``"dpmpp_2m"``.
    """
    from model_resolver import resolve_model_path

    model_path = resolve_model_path(model_path)
    dtype = torch.float32 if use_fp32 else torch.float16
    logger.info("Loading DiT from %s (%s)", model_path, "FP32" if use_fp32 else "FP16")

    pipe_cls = _load_dit_pipeline_class(model_path)
    pipe = pipe_cls.from_pretrained(model_path, torch_dtype=dtype)

    # Place text encoders + VAE on stage 0 (device_down).  These are small
    # relative to a 30–50 GB transformer.
    for attr in ("text_encoder", "text_encoder_2", "text_encoder_3", "vae"):
        sub = getattr(pipe, attr, None)
        if sub is not None:
            setattr(pipe, attr, sub.to(device_down))
    if getattr(pipe, "safety_checker", None) is not None:
        pipe.safety_checker = None

    # Scheduler (best-effort — not all schedulers exist for every pipeline).
    if scheduler != "default":
        from diffusers import (
            DDIMScheduler,
            DPMSolverMultistepScheduler,
            EulerDiscreteScheduler,
        )
        try:
            if scheduler == "ddim":
                pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            elif scheduler == "euler":
                pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            elif scheduler == "dpmpp_2m":
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config
                )
        except Exception:
            logger.warning("Could not switch scheduler to %s; keeping default", scheduler)

    # Wrap the transformer.
    logger.info(
        "Wrapping %s with PipelineParallelDiT (%s + %s)",
        type(pipe.transformer).__name__, device_down, device_up,
    )
    pipe.transformer = PipelineParallelDiT(
        pipe.transformer, device_down=device_down, device_up=device_up
    )

    # The pipeline must not attempt its own device management.
    pipe.enable_model_cpu_offload = lambda: None  # type: ignore[method-assign]
    pipe.enable_sequential_cpu_offload = lambda: None  # type: ignore[method-assign]
    return pipe


def generate_dit(
    prompt: str,
    negative_prompt: str = "",
    model_path: str = "PixArt-alpha/PixArt-XL-2-1024-MS",
    device_down: str = "cuda:0",
    device_up: str = "cuda:1",
    steps: int = 20,
    width: int = 1024,
    height: int = 1024,
    seed: int = 42,
    guidance_scale: float = 4.5,
    use_fp32: bool = True,
    output_path: str = "output_dit_pp.png",
    scheduler: str = "default",
    force_unload: bool = False,
    callback: Optional[callable] = None,
    callback_kwargs: Optional[dict] = None,
):
    """Generate a single image with a DiT model across 2 GPUs (FP32).

    Uses the same keep-alive cache as the SDXL path (see
    :mod:`pipeline_cache`).  Pass ``force_unload=True`` to free VRAM at once.
    """
    from pipeline_cache import cached_dit_pipeline, get_cache

    cache = get_cache()
    pipe, entry = cached_dit_pipeline(
        model_path=model_path,
        device_down=device_down,
        device_up=device_up,
        use_fp32=use_fp32,
        scheduler=scheduler,
        cache=cache,
    )

    call_kwargs = dict(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    if negative_prompt:
        call_kwargs["negative_prompt"] = negative_prompt

    if callback is not None:
        from callback_utils import make_progress_callback
        call_kwargs["callback_on_step_end"] = make_progress_callback(callback)
        call_kwargs["callback_on_step_end_inputs"] = callback_kwargs or []

    logger.info(
        "Generating DiT %dx%d, %d steps, CFG %.1f", width, height, steps, guidance_scale
    )
    entry_lock = entry.lock if entry is not None else _NullLock()
    with entry_lock:
        t0 = time.perf_counter()
        image = pipe(**call_kwargs).images[0]
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    image.save(output_path)
    logger.info("Saved %s in %.1fs (%.2fs/step)", output_path, dt, dt / steps)

    if force_unload:
        from model_resolver import resolve_model_path
        resolved = resolve_model_path(model_path)
        key = (
            "dit",
            os.path.abspath(resolved),
            str(device_down),
            str(device_up),
            bool(use_fp32),
            str(scheduler),
        )
        if cache.release(key):
            logger.info("force_unload: pipeline freed")
    return image


class _NullLock:
    """Context manager that does nothing (used when caching is disabled)."""

    def __enter__(self) -> "_NullLock":
        return self

    def __exit__(self, *exc) -> None:
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    generate_dit(
        prompt=(
            "A serene Japanese garden with a koi pond, cherry blossoms falling, "
            "golden hour light, photorealistic, 8k, detailed"
        ),
        negative_prompt="blurry, low quality, distorted, ugly",
        steps=20,
        seed=42,
    )