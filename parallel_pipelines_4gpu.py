"""
parallel_pipelines_4gpu.py — Run 2 pipeline-parallel SDXL instances on 4 GPUs.

Each pipeline occupies a pair of GPUs:
    Pipeline A: cuda:0 (down) + cuda:1 (up)
    Pipeline B: cuda:2 (down) + cuda:3 (up)

Two back-ends are provided:

* :func:`generate_batch_parallel_threads` — single process, two Python threads.
  PyTorch releases the GIL during CUDA kernels, so this works, but it shares
  a CUDA context (slightly higher VRAM overhead) and a single Python process.

* :func:`generate_batch_parallel_processes` — spawns one child process per
  GPU pair via :mod:`multiprocessing` with the ``spawn`` start method.  This
  is the recommended, most robust approach on Linux + CUDA.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import time
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Threaded back-end
# ---------------------------------------------------------------------------

def generate_batch_parallel_threads(
    prompts: list[str],
    model_path: str = "./models/sdxl-base-fp16",
    steps: int = 25,
    width: int = 1024,
    height: int = 1024,
    seeds: Optional[list[int]] = None,
    output_dir: str = "batch_output",
):
    """Generate a batch using two in-process threads (one per GPU pair).

    Note: builds the pipeline once per job (no caching).  For repeated
    generation prefer :func:`generate_batch_parallel_processes`.
    """
    import threading
    import torch  # noqa: F401  (ensures CUDA init in main process)

    from pipeline_parallel_sdxl import generate_sdxl

    gpu_pairs = [("cuda:0", "cuda:1"), ("cuda:2", "cuda:3")]
    os.makedirs(output_dir, exist_ok=True)
    if seeds is None:
        seeds = [42 + i for i in range(len(prompts))]

    results: dict[int, str] = {}
    lock = threading.Lock()

    def worker(idx: int, prompt: str, seed: int):
        pair = gpu_pairs[idx % len(gpu_pairs)]
        out = os.path.join(output_dir, f"img_{idx:03d}.png")
        try:
            generate_sdxl(
                prompt=prompt,
                model_path=model_path,
                device_down=pair[0],
                device_up=pair[1],
                steps=steps,
                width=width,
                height=height,
                seed=seed,
                output_path=out,
            )
            with lock:
                results[idx] = out
            logger.info("[%s+%s] saved %s", pair[0], pair[1], out)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Worker %d failed: %s", idx, exc)

    # Run at most len(gpu_pairs) jobs concurrently.
    threads: list[threading.Thread] = []
    queue = list(range(len(prompts)))
    next_idx = 0

    t0 = time.perf_counter()
    while next_idx < len(queue) or threads:
        # Top up active threads
        while next_idx < len(queue) and len(threads) < len(gpu_pairs):
            i = queue[next_idx]
            next_idx += 1
            th = threading.Thread(
                target=worker, args=(i, prompts[i], seeds[i]), name=f"job-{i}"
            )
            th.start()
            threads.append(th)
        # Wait for at least one to finish
        if threads:
            threads[0].join()
            threads = [t for t in threads if t.is_alive()]
            time.sleep(0.01)

    dt = time.perf_counter() - t0
    logger.info(
        "Batch of %d images in %.1fs (%.1f img/min)",
        len(prompts), dt, len(prompts) / dt * 60,
    )
    return results


# ---------------------------------------------------------------------------
# Multiprocess back-end (recommended)
# ---------------------------------------------------------------------------

def _mp_worker(
    prompt: str,
    seed: int,
    device_down: str,
    device_up: str,
    output_path: str,
    model_path: str,
    steps: int,
    width: int,
    height: int,
    guidance_scale: float,
    negative_prompt: str,
):
    """Process entry-point: isolate CUDA, build pipeline, generate one image."""
    import torch
    from pipeline_parallel_sdxl import generate_sdxl

    # Pin the process to the right GPU pair (best-effort; optional).
    generate_sdxl(
        prompt=prompt,
        negative_prompt=negative_prompt,
        model_path=model_path,
        device_down=device_down,
        device_up=device_up,
        steps=steps,
        width=width,
        height=height,
        seed=seed,
        guidance_scale=guidance_scale,
        output_path=output_path,
    )


def generate_batch_parallel_processes(
    prompts: list[str],
    model_path: str = "./models/sdxl-base-fp16",
    steps: int = 25,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 7.5,
    negative_prompt: str = "blurry, low quality, distorted, ugly",
    seeds: Optional[list[int]] = None,
    output_dir: str = "batch_output",
    gpu_pairs: Optional[list[tuple[str, str]]] = None,
):
    """Generate a batch with one child process per GPU pair (recommended).

    Each prompt is dispatched round-robin to the available GPU pairs.  This
    avoids the GIL and gives full CUDA-context isolation.
    """
    os.makedirs(output_dir, exist_ok=True)
    if gpu_pairs is None:
        gpu_pairs = [("cuda:0", "cuda:1"), ("cuda:2", "cuda:3")]
    if seeds is None:
        seeds = [42 + i for i in range(len(prompts))]

    ctx = mp.get_context("spawn")  # CUDA-safe
    procs: list[ctx.Process] = []  # type: ignore[name-defined]
    next_idx = 0

    t0 = time.perf_counter()
    while next_idx < len(prompts) or procs:
        # Top up
        while next_idx < len(prompts) and len(procs) < len(gpu_pairs):
            i = next_idx
            next_idx += 1
            dev_down, dev_up = gpu_pairs[i % len(gpu_pairs)]
            out = os.path.join(output_dir, f"img_{i:03d}.png")
            p = ctx.Process(
                target=_mp_worker,
                args=(
                    prompts[i],
                    seeds[i],
                    dev_down,
                    dev_up,
                    out,
                    model_path,
                    steps,
                    width,
                    height,
                    guidance_scale,
                    negative_prompt,
                ),
                name=f"job-{i}",
            )
            p.start()
            procs.append(p)
        # Reap at least one finished process before topping up again.
        if procs:
            procs[0].join()
            if procs[0].exitcode != 0:
                logger.error(
                    "Process %s exited with code %s",
                    procs[0].name,
                    procs[0].exitcode,
                )
            # Drop the joined process; keep the rest.
            procs = procs[1:]
            time.sleep(0.05)

    dt = time.perf_counter() - t0
    logger.info(
        "Batch of %d images in %.1fs (%.1f img/min)",
        len(prompts), dt, len(prompts) / dt * 60,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    demo_prompts = [
        "A cyberpunk cityscape at night, neon lights reflecting in wet streets, 8k",
        "A peaceful mountain lake at dawn, mist rising, photorealistic",
        "An astronaut floating above Earth, stars visible, detailed spacesuit",
        "A cozy bookshop interior, warm lighting, shelves full of colorful books",
    ]
    # Choose one of the two back-ends:
    generate_batch_parallel_processes(demo_prompts, steps=25)