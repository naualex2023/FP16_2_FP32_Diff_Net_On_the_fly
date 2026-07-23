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

    The loaded pipeline is cached and reused across jobs via
    :mod:`pipeline_cache` (the first job on each GPU pair loads the model;
    subsequent jobs reuse it).  Two prompts routed to the same GPU pair
    serialize on the cache's per-entry lock, so only one generates at a time
    per pair — matching the hardware constraint.
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

def _mp_worker_persistent(
    in_q: "mp.Queue",
    out_q: "mp.Queue",
    device_down: str,
    device_up: str,
    model_path: str,
    steps: int,
    width: int,
    height: int,
    guidance_scale: float,
    negative_prompt: str,
):
    """Persistent worker: loads model once, then serves many jobs from a queue.

    Each item on ``in_q`` is ``(idx, prompt, seed, output_path)`` or ``None``
    to signal shutdown.  Each result on ``out_q`` is ``(idx, ok, output_path
    or error)``.  The loaded pipeline is cached per process via
    :mod:`pipeline_cache`, so only the first job pays the load cost.
    """
    import torch  # noqa: F401  (CUDA init)
    from pipeline_parallel_sdxl import generate_sdxl

    while True:
        job = in_q.get()
        if job is None:  # shutdown sentinel
            break
        idx, prompt, seed, output_path = job
        try:
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
            out_q.put((idx, True, output_path))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Worker %s+%s failed job %d", device_down, device_up, idx)
            out_q.put((idx, False, str(exc)))

    # Free VRAM before the process exits.
    try:
        from pipeline_cache import get_cache
        get_cache().unload_all()
    except Exception:  # noqa: BLE001
        pass


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
    """Generate a batch with one persistent child process per GPU pair.

    Each GPU pair runs one long-lived worker that loads the model once and
    reuses it for every prompt routed to that pair (cache-backed).  Prompts
    are dispatched round-robin across the pairs; results are collected as
    they finish.  This gives full CUDA-context isolation *and* amortizes the
    model load across the batch.
    """
    os.makedirs(output_dir, exist_ok=True)
    if gpu_pairs is None:
        gpu_pairs = [("cuda:0", "cuda:1"), ("cuda:2", "cuda:3")]
    if seeds is None:
        seeds = [42 + i for i in range(len(prompts))]

    ctx = mp.get_context("spawn")  # CUDA-safe
    in_q: "mp.Queue" = ctx.Queue()
    out_q: "mp.Queue" = ctx.Queue()

    # Enqueue all jobs tagged with their GPU-pair index.
    for i, (prompt, seed) in enumerate(zip(prompts, seeds)):
        out = os.path.join(output_dir, f"img_{i:03d}.png")
        in_q.put((i, prompt, seed, out))

    # One persistent worker per GPU pair.
    workers = []
    for pair in gpu_pairs:
        in_q.put(None)  # one shutdown sentinel per worker
        p = ctx.Process(
            target=_mp_worker_persistent,
            args=(
                in_q, out_q, pair[0], pair[1], model_path,
                steps, width, height, guidance_scale, negative_prompt,
            ),
            name=f"worker-{pair[0]}+{pair[1]}",
        )
        p.start()
        workers.append(p)

    t0 = time.perf_counter()
    done = 0
    total = len(prompts)
    while done < total:
        idx, ok, payload = out_q.get()
        done += 1
        if ok:
            logger.info("[%d/%d] saved %s", done, total, payload)
        else:
            logger.error("[%d/%d] job %d failed: %s", done, total, idx, payload)

    # Wait for workers to drain their shutdown sentinels and exit.
    for p in workers:
        p.join()

    dt = time.perf_counter() - t0
    logger.info(
        "Batch of %d images in %.1fs (%.1f img/min)",
        total, dt, total / dt * 60 if dt > 0 else 0.0,
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