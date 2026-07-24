"""
pipeline_gguf.py — GGUF-quantized diffusion pipeline on a single Tesla P40.

GGUF (Generalized Gaussian Unit Format) models are heavily quantized (Q4/Q8),
making 30-50 GB models fit on a **single** 24 GB P40.  For example:

    SD3.5-large Q4_K_M:  transformer ~5 GB + T5 ~3 GB + CLIP ~1 GB = ~9 GB
    → fits on one P40 with room for activations.

This module loads a pre-downloaded GGUF model (in ``./models/``) using
diffusers' built-in GGUF support, places everything on one GPU, and provides
``generate_gguf()`` compatible with the project's cache + callback system.

GGUF files must be downloaded in advance (they are NOT regular diffusers repos).
Place them in ``./models/`` alongside a minimal diffusers directory structure
containing the non-transformer components (text encoders, VAE, scheduler config).

Usage
-----
    from pipeline_gguf import generate_gguf
    generate_gguf("a cat", model_path="./models/sd35-large-Q4_K_M.gguf")
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# Mapping: pipeline class → transformer model class (for from_single_file).
_PIPELINE_TO_TRANSFORMER = {
    "StableDiffusion3Pipeline": "SD3Transformer2DModel",
    "FluxPipeline": "FluxTransformer2DModel",
    "PixArtAlphaPipeline": "Transformer2DModel",
    "PixArtSigmaPipeline": "Transformer2DModel",
    "SanaPipeline": "SanaTransformer2DModel",
}


def _detect_gguf_pipeline_type(model_path: str) -> str:
    """Detect which diffusers pipeline to use for this GGUF file.

    Strategy:
      1. Look for a sibling directory (same name without .gguf) with a
         ``model_index.json``.
      2. Fall back to filename heuristics (sd3 → SD3Pipeline, flux → Flux, ...).
    """
    import json

    # Check for a sibling diffusers directory.
    base = model_path.rsplit("/", 1)[-1]
    if base.endswith(".gguf"):
        base = base[:-5]
    sibling = os.path.join(os.path.dirname(model_path), base)
    idx_file = os.path.join(sibling, "model_index.json")
    if os.path.isfile(idx_file):
        try:
            with open(idx_file, "r", encoding="utf-8") as f:
                cls_name = json.load(f).get("_class_name", "")
                if cls_name in _PIPELINE_TO_TRANSFORMER:
                    return cls_name
        except Exception:
            pass

    # Filename heuristics.
    lower = model_path.lower()
    if "sd3" in lower or "stable-diffusion-3" in lower or "sd35" in lower:
        return "StableDiffusion3Pipeline"
    if "flux" in lower:
        return "FluxPipeline"
    if "sana" in lower:
        return "SanaPipeline"
    if "pixart" in lower:
        return "PixArtAlphaPipeline" if "sigma" not in lower else "PixArtSigmaPipeline"

    logger.warning("Could not detect pipeline type for %s; defaulting to SD3", model_path)
    return "StableDiffusion3Pipeline"


def create_gguf_pipeline(
    model_path: str,
    device: str = "cuda:0",
    use_fp16_compute: bool = True,
    scheduler: str = "default",
):
    """Build a single-GPU GGUF-quantized pipeline.

    Parameters
    ----------
    model_path : str
        Path to a ``.gguf`` file OR a directory containing a GGUF transformer
        + diffusers components (text encoders, VAE, scheduler).
    device : str
        Target CUDA device.
    use_fp16_compute : bool
        If True (recommended), compute dtype is FP16 (GGUF dequantizes to FP16
        on-the-fly). P40 FP16 compute is slow but GGUF models are small enough
        that the reduced memory access often compensates.
    scheduler : str
        Scheduler preset.
    """
    import diffusers

    pipe_cls_name = _detect_gguf_pipeline_type(model_path)
    transformer_cls_name = _PIPELINE_TO_TRANSFORMER.get(pipe_cls_name, "Transformer2DModel")

    pipe_cls = getattr(diffusers, pipe_cls_name)
    transformer_cls = getattr(diffusers, transformer_cls_name)

    logger.info("Loading GGUF pipeline: %s (%s transformer)", pipe_cls_name, transformer_cls_name)

    # ---- Case A: model_path is a .gguf file ---------------------------
    if model_path.endswith(".gguf"):
        gguf_file = model_path
        # Sibling directory for non-transformer components.
        base = os.path.basename(gguf_file)[:-5]
        base_dir = os.path.join(os.path.dirname(gguf_file), base)

        if not os.path.isdir(base_dir):
            raise FileNotFoundError(
                f"GGUF file needs a sibling diffusers directory for non-transformer "
                f"components. Expected: {base_dir}/"
            )

        # Load quantized transformer from GGUF.
        compute_dtype = torch.float16 if use_fp16_compute else torch.float32

        # diffusers >= 0.30 supports GGUF via GGUFQuantizationConfig.
        try:
            from diffusers import GGUFQuantizationConfig
            quant_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)
        except ImportError:
            # Older diffusers: pass compute_dtype directly.
            quant_config = compute_dtype

        logger.info("Loading quantized transformer from %s", gguf_file)
        transformer = transformer_cls.from_single_file(
            gguf_file,
            quantization_config=quant_config,
            torch_dtype=compute_dtype,
        )

        # Load the rest of the pipeline from the sibling directory.
        pipe = pipe_cls.from_pretrained(
            base_dir,
            transformer=transformer,
            torch_dtype=compute_dtype,
        )

    # ---- Case B: model_path is a directory ----------------------------
    else:
        # Look for a .gguf file inside the directory.
        gguf_files = [
            f for f in os.listdir(model_path)
            if f.endswith(".gguf")
        ]
        if not gguf_files:
            raise FileNotFoundError(f"No .gguf file found in {model_path}")
        gguf_file = os.path.join(model_path, gguf_files[0])

        compute_dtype = torch.float16 if use_fp16_compute else torch.float32
        try:
            from diffusers import GGUFQuantizationConfig
            quant_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)
        except ImportError:
            quant_config = compute_dtype

        logger.info("Loading quantized transformer from %s", gguf_file)
        transformer = transformer_cls.from_single_file(
            gguf_file,
            quantization_config=quant_config,
            torch_dtype=compute_dtype,
        )

        pipe = pipe_cls.from_pretrained(
            model_path,
            transformer=transformer,
            torch_dtype=compute_dtype,
        )

    # ---- place everything on one GPU -----------------------------------
    pipe = pipe.to(device)

    if getattr(pipe, "safety_checker", None) is not None:
        pipe.safety_checker = None

    # Scheduler.
    if scheduler != "default":
        from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler
        try:
            if scheduler == "ddim":
                pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            elif scheduler == "euler":
                pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            elif scheduler == "dpmpp_2m":
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        except Exception:
            logger.warning("Could not switch scheduler to %s; keeping default", scheduler)

    pipe.enable_model_cpu_offload = lambda: None
    pipe.enable_sequential_cpu_offload = lambda: None

    logger.info("GGUF pipeline ready on %s", device)
    return pipe


def generate_gguf(
    prompt: str,
    negative_prompt: str = "",
    model_path: str = "",
    device: str = "cuda:0",
    steps: int = 20,
    width: int = 1024,
    height: int = 1024,
    seed: int = 42,
    guidance_scale: float = 4.5,
    use_fp16_compute: bool = True,
    output_path: str = "output_gguf.png",
    scheduler: str = "default",
    force_unload: bool = False,
    callback: Optional[callable] = None,
    callback_kwargs: Optional[dict] = None,
):
    """Generate a single image with a GGUF-quantized model on one GPU."""
    from pipeline_cache import cached_gguf_pipeline, get_cache

    cache = get_cache()
    pipe, entry = cached_gguf_pipeline(
        model_path=model_path,
        device=device,
        use_fp16_compute=use_fp16_compute,
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

    logger.info("Generating GGUF %dx%d, %d steps, CFG %.1f", width, height, steps, guidance_scale)
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
            "gguf", os.path.abspath(model_path), str(device),
            bool(use_fp16_compute), str(scheduler),
        )
        if cache.release(key):
            logger.info("force_unload: pipeline freed")
    return image


class _NullLock:
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    import sys
    if len(sys.argv) > 1:
        generate_gguf(
            prompt="A serene Japanese garden, koi pond, golden hour, 8k",
            model_path=sys.argv[1],
            steps=20, seed=42,
        )
    else:
        print("Usage: python pipeline_gguf.py <path/to/model.gguf or model_dir>")
