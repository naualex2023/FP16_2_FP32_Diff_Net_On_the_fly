#!/usr/bin/env python3
"""
benchmark.py — Compare SDXL FP32 pipeline-parallel vs alternatives on P40.

Runs, with warmup, three configurations (skipping any that OOM):
  1. FP16 on 1 GPU            (baseline — broken/slow on Pascal)
  2. FP32 pipeline-parallel   (the recommended P40 configuration)
  3. FP32 with model CPU offload (fallback if only 1 GPU is available)

Usage:
    python benchmark.py --model ./models/sdxl-base-fp16 --steps 25
"""

from __future__ import annotations

import argparse
import gc
import logging
import time

import torch

logger = logging.getLogger(__name__)


def benchmark(
    name: str,
    pipe,
    prompt: str,
    steps: int,
    warmup: int = 1,
    runs: int = 3,
    width: int = 1024,
    height: int = 1024,
) -> float:
    """Benchmark a pipeline; return mean seconds per image."""
    def gen(seed: int):
        return pipe(
            prompt=prompt,
            num_inference_steps=steps,
            width=width,
            height=height,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).images[0]

    # Warmup
    for i in range(warmup):
        gen(i)
        torch.cuda.synchronize()

    times = []
    for i in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gen(warmup + i)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        logger.info("  [%s] run %d: %.2fs", name, i + 1, dt)

    avg = sum(times) / len(times)
    logger.info("  [%s] MEAN: %.2fs (%.2fs/step)", name, avg, avg / steps)
    return avg


def run_all(model_path: str, steps: int, width: int, height: int):
    from diffusers import StableDiffusionXLPipeline
    from pipeline_parallel_sdxl import create_sdxl_pipeline_parallel

    prompt = (
        "a photograph of a cat sitting on a windowsill, golden hour light, "
        "detailed fur texture, 8k, professional photography"
    )
    results: dict[str, float] = {}

    # --- 1. FP16 on 1 GPU --------------------------------------------------
    logger.info("=" * 60)
    logger.info("1. SDXL FP16, 1 GPU — known-broken FP16 on Pascal")
    logger.info("=" * 60)
    try:
        pipe_fp16 = StableDiffusionXLPipeline.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).to("cuda:0")
        pipe_fp16.safety_checker = None
        results["FP16 1GPU"] = benchmark(
            "FP16 1GPU", pipe_fp16, prompt, steps, width=width, height=height
        )
        del pipe_fp16
        gc.collect(); torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        logger.warning("FP16 1GPU skipped: %s", exc)

    # --- 2. FP32 pipeline-parallel, 2 GPU ----------------------------------
    logger.info("=" * 60)
    logger.info("2. SDXL FP32, Pipeline Parallel, 2 GPU (cuda:0 + cuda:1)")
    logger.info("=" * 60)
    try:
        pipe_pp = create_sdxl_pipeline_parallel(
            model_path, "cuda:0", "cuda:1", use_fp32=True
        )
        results["FP32 PP 2GPU"] = benchmark(
            "FP32 PP 2GPU", pipe_pp, prompt, steps, width=width, height=height
        )
        del pipe_pp
        gc.collect(); torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        logger.warning("FP32 PP 2GPU skipped: %s", exc)

    # --- 3. FP32 with CPU offload, 1 GPU -----------------------------------
    logger.info("=" * 60)
    logger.info("3. SDXL FP32, CPU Offload, 1 GPU (cuda:0)")
    logger.info("=" * 60)
    try:
        pipe_off = StableDiffusionXLPipeline.from_pretrained(
            model_path, torch_dtype=torch.float32
        )
        pipe_off.enable_model_cpu_offload()
        pipe_off.safety_checker = None
        results["FP32 Offload 1GPU"] = benchmark(
            "FP32 Offload 1GPU", pipe_off, prompt, steps, width=width, height=height
        )
        del pipe_off
        gc.collect(); torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001
        logger.warning("FP32 Offload 1GPU skipped: %s", exc)

    # --- Summary ------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for name, secs in results.items():
        logger.info("  %-22s %.2fs/image  (%.1f img/min)",
                    name, secs, 60.0 / secs)
    logger.info("Pipeline Parallel FP32 is the recommended config on P40.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./models/sdxl-base-fp16")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    args = ap.parse_args()
    run_all(args.model, args.steps, args.width, args.height)