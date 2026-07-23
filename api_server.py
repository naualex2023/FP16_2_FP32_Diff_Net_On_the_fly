"""
api_server.py — FastAPI web layer for the FP32 pipeline-parallel diffusion backend.

This module exposes the existing generation functions (single image on 2 GPUs,
batch on 4 GPUs, and the new "twin" mode — 2 images of the SAME prompt on 4
GPUs) over HTTP + Server-Sent Events so a React frontend can drive the 4× P40
server comfortably.

The server runs in ONE process so the PipelineCache singleton is shared across
all requests — the model stays resident (cache HIT) between back-to-back
generations.

Run:
    uvicorn api_server:app --host 0.0.0.0 --port 8000
    # (or)  python api_server.py
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

logger = logging.getLogger("api_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.environ.get("SD_MODELS_DIR", os.path.join(BASE_DIR, "models"))
LORAS_DIR = os.environ.get("SD_LORAS_DIR", os.path.join(BASE_DIR, "loras"))
OUTPUT_DIR = os.environ.get("SD_OUTPUT_DIR", os.path.join(BASE_DIR, "gallery"))
WEB_DIR = os.path.join(BASE_DIR, "web")
HISTORY_FILE = os.path.join(OUTPUT_DIR, "history.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LORAS_DIR, exist_ok=True)

# Two generation threads max: GPU pair A (0+1) and pair B (2+3) can run in
# parallel, so we allow 2 concurrent generation jobs. A third "twin" job uses
# both pairs at once.
EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="gen")

# ---------------------------------------------------------------------------
# In-memory job registry + event bus (SSE)
# ---------------------------------------------------------------------------

class JobManager:
    """Tracks generation jobs and broadcasts progress to SSE subscribers."""

    def __init__(self) -> None:
        self._jobs: Dict[str, dict] = {}
        self._lock = threading.Lock()
        # Per-job list of asyncio.Queue objects for SSE subscribers.
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def create(self, job_id: str, job: dict) -> None:
        with self._lock:
            self._jobs[job_id] = job
            self._subscribers[job_id] = []

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> List[dict]:
        with self._lock:
            return list(self._jobs.values())

    def update(self, job_id: str, **patch) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job.update(patch)
            event = dict(patch)
            event["job_id"] = job_id
            subs = list(self._subscribers.get(job_id, []))
        # Push to SSE subscribers from the asyncio loop thread.
        if self._loop and subs:
            for q in subs:
                self._loop.call_soon_threadsafe(q.put_nowait, event)
        return job

    def subscribe(self, job_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._subscribers.setdefault(job_id, []).append(q)
        return q

    def unsubscribe(self, job_id: str, q: asyncio.Queue) -> None:
        with self._lock:
            subs = self._subscribers.get(job_id, [])
            if q in subs:
                subs.remove(q)


JOBS = JobManager()

# ---------------------------------------------------------------------------
# Pydantic models (request bodies)
# ---------------------------------------------------------------------------

ARCH_CHOICES = ["sdxl", "sd15"]
SCHEDULER_CHOICES = ["default", "ddim", "euler", "dpmpp_2m"]
ASPECT_PRESETS = ["1:1", "3:4", "4:3", "16:9", "9:16", "2:3", "3:2"]


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Text prompt")
    negative_prompt: str = "blurry, low quality, distorted, ugly"
    model_path: str = "./models/sdxl-base-fp16"
    arch: str = Field("sdxl", description="Model architecture: sdxl or sd15")
    steps: int = Field(25, ge=1, le=100)
    width: int = Field(1024, ge=64, le=2048, multiple_of=8)
    height: int = Field(1024, ge=64, le=2048, multiple_of=8)
    seed: int = Field(-1, description="-1 = random")
    guidance_scale: float = Field(7.5, ge=0.0, le=30.0)
    scheduler: str = Field("default")
    lora_path: Optional[str] = None
    lora_scale: float = Field(1.0, ge=0.0, le=2.0)
    gpu_pair: str = Field("0+1", description="GPU pair: '0+1' or '2+3'")
    use_fp32: bool = True


class TwinRequest(BaseModel):
    """Generate 2 images of the SAME prompt on 4 GPUs simultaneously."""

    prompt: str = Field(...)
    negative_prompt: str = "blurry, low quality, distorted, ugly"
    model_path: str = "./models/sdxl-base-fp16"
    arch: str = "sdxl"
    steps: int = Field(25, ge=1, le=100)
    width: int = Field(1024, ge=64, le=2048, multiple_of=8)
    height: int = Field(1024, ge=64, le=2048, multiple_of=8)
    seed_a: int = Field(-1, description="Seed for pair A; -1 = random")
    seed_b: int = Field(-1, description="Seed for pair B; -1 = random")
    guidance_scale: float = Field(7.5, ge=0.0, le=30.0)
    scheduler: str = Field("default")
    lora_path: Optional[str] = None
    lora_scale: float = Field(1.0, ge=0.0, le=2.0)
    use_fp32: bool = True


class BatchRequest(BaseModel):
    prompts: List[str] = Field(..., description="List of prompts")
    negative_prompt: str = "blurry, low quality, distorted, ugly"
    model_path: str = "./models/sdxl-base-fp16"
    arch: str = "sdxl"
    steps: int = Field(25, ge=1, le=100)
    width: int = Field(1024, ge=64, le=2048, multiple_of=8)
    height: int = Field(1024, ge=64, le=2048, multiple_of=8)
    base_seed: int = Field(42, description="Seeds increment from this value")
    guidance_scale: float = Field(7.5, ge=0.0, le=30.0)
    scheduler: str = Field("default")
    lora_path: Optional[str] = None
    lora_scale: float = Field(1.0, ge=0.0, le=2.0)


class CacheControlRequest(BaseModel):
    action: str = Field(..., description="unload_all | unload_one")
    key: Optional[str] = None


# ---------------------------------------------------------------------------
# History (JSON sidecar)
# ---------------------------------------------------------------------------

_history_lock = threading.Lock()


def _load_history() -> List[dict]:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        logger.exception("Failed to load history")
        return []


def _save_history(history: List[dict]) -> None:
    with _history_lock:
        tmp = HISTORY_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        os.replace(tmp, HISTORY_FILE)


def _add_history_entry(entry: dict) -> None:
    history = _load_history()
    history.insert(0, entry)
    # Cap at 500 entries to keep the file manageable.
    if len(history) > 500:
        history = history[:500]
    _save_history(history)


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def _detect_gpus() -> List[dict]:
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_vram_gb": round(props.total_memory / 1e9, 1),
                }
            )
        return gpus
    except Exception:
        return []


def _gpu_vram() -> List[dict]:
    """Live VRAM usage per GPU (best-effort)."""
    try:
        import torch

        out = []
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            props = torch.cuda.get_device_properties(i)
            out.append(
                {
                    "index": i,
                    "name": props.name,
                    "total_vram_gb": round(props.total_memory / 1e9, 1),
                    "allocated_gb": round(allocated, 2),
                    "reserved_gb": round(reserved, 2),
                }
            )
        return out
    except Exception:
        return []


def _pair_to_devices(gpu_pair: str) -> tuple[str, str]:
    if gpu_pair == "2+3":
        return "cuda:2", "cuda:3"
    return "cuda:0", "cuda:1"


# ---------------------------------------------------------------------------
# Generation orchestration
# ---------------------------------------------------------------------------

def _random_seed() -> int:
    import random

    return random.randint(0, 2**31 - 1)


def _resolve_seed(seed: int) -> int:
    return _random_seed() if seed < 0 else seed


def _run_single_generation(
    job_id: str,
    req: GenerateRequest,
) -> None:
    """Worker that runs in the thread pool; emits progress via JobManager."""
    seed = _resolve_seed(req.seed)
    JOBS.update(
        job_id,
        status="running",
        seed=seed,
        progress=0,
        stage="loading",
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    dev_down, dev_up = _pair_to_devices(req.gpu_pair)

    # Progress callback (called from the diffusers loop on THIS thread).
    def cb(step: int, timestep: float, kwargs: Any) -> None:
        pct = int(step / req.steps * 100) if req.steps > 0 else 0
        JOBS.update(
            job_id,
            progress=pct,
            step=step,
            total_steps=req.steps,
            timestep=timestep,
            stage="generating",
        )

    try:
        if req.arch == "sd15":
            from pipeline_parallel_sd15 import generate_sd15_pipeline_parallel

            out_path = os.path.join(OUTPUT_DIR, f"{job_id}.png")
            generate_sd15_pipeline_parallel(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                model_path=req.model_path,
                device_down=dev_down,
                device_up=dev_up,
                steps=req.steps,
                seed=seed,
                guidance_scale=req.guidance_scale,
                width=req.width,
                height=req.height,
                output_path=out_path,
                callback=cb,
            )
        else:
            from pipeline_parallel_sdxl import generate_sdxl

            out_path = os.path.join(OUTPUT_DIR, f"{job_id}.png")
            generate_sdxl(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt,
                model_path=req.model_path,
                device_down=dev_down,
                device_up=dev_up,
                steps=req.steps,
                width=req.width,
                height=req.height,
                seed=seed,
                guidance_scale=req.guidance_scale,
                use_fp32=req.use_fp32,
                output_path=out_path,
                scheduler=req.scheduler,
                lora_path=req.lora_path,
                lora_scale=req.lora_scale,
                callback=cb,
            )

        JOBS.update(
            job_id,
            status="done",
            progress=100,
            stage="complete",
            output_path=out_path,
            image_url=f"/gallery/{job_id}.png",
            finished_at=datetime.now(timezone.utc).isoformat(),
        )

        _add_history_entry(
            {
                "job_id": job_id,
                "prompt": req.prompt,
                "negative_prompt": req.negative_prompt,
                "arch": req.arch,
                "model_path": req.model_path,
                "steps": req.steps,
                "width": req.width,
                "height": req.height,
                "seed": seed,
                "guidance_scale": req.guidance_scale,
                "scheduler": req.scheduler,
                "lora_path": req.lora_path,
                "lora_scale": req.lora_scale,
                "gpu_pair": req.gpu_pair,
                "image_url": f"/gallery/{job_id}.png",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )
    except Exception as exc:
        logger.exception("Job %s failed", job_id)
        JOBS.update(
            job_id,
            status="failed",
            stage="error",
            error=str(exc),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )


def _run_twin_generation(job_id: str, req: TwinRequest) -> None:
    """Run 2 images of the same prompt on 4 GPUs simultaneously."""
    seed_a = _resolve_seed(req.seed_a)
    seed_b = _resolve_seed(req.seed_b)
    JOBS.update(
        job_id,
        status="running",
        stage="loading",
        progress=0,
        sub_jobs={"a": {"progress": 0}, "b": {"progress": 0}},
        seeds={"a": seed_a, "b": seed_b},
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    progress = {"a": 0, "b": 0}

    def make_cb(label: str):
        def cb(step: int, timestep: float, kwargs: Any) -> None:
            pct = int(step / req.steps * 100) if req.steps > 0 else 0
            progress[label] = pct
            overall = (progress["a"] + progress["b"]) // 2
            JOBS.update(
                job_id,
                progress=overall,
                sub_jobs={
                    "a": {"progress": progress["a"], "step": None},
                    "b": {"progress": progress["b"], "step": None},
                },
                stage="generating",
            )

        return cb

    errors: List[str] = []

    def work_a():
        try:
            from pipeline_parallel_sdxl import generate_sdxl

            out_a = os.path.join(OUTPUT_DIR, f"{job_id}_a.png")
            if req.arch == "sd15":
                from pipeline_parallel_sd15 import generate_sd15_pipeline_parallel

                generate_sd15_pipeline_parallel(
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt,
                    model_path=req.model_path,
                    device_down="cuda:0",
                    device_up="cuda:1",
                    steps=req.steps,
                    seed=seed_a,
                    guidance_scale=req.guidance_scale,
                    width=req.width,
                    height=req.height,
                    output_path=out_a,
                    callback=make_cb("a"),
                )
            else:
                generate_sdxl(
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt,
                    model_path=req.model_path,
                    device_down="cuda:0",
                    device_up="cuda:1",
                    steps=req.steps,
                    width=req.width,
                    height=req.height,
                    seed=seed_a,
                    guidance_scale=req.guidance_scale,
                    use_fp32=req.use_fp32,
                    output_path=out_a,
                    scheduler=req.scheduler,
                    lora_path=req.lora_path,
                    lora_scale=req.lora_scale,
                    callback=make_cb("a"),
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Pair A: {exc}")

    def work_b():
        try:
            out_b = os.path.join(OUTPUT_DIR, f"{job_id}_b.png")
            if req.arch == "sd15":
                from pipeline_parallel_sd15 import generate_sd15_pipeline_parallel

                generate_sd15_pipeline_parallel(
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt,
                    model_path=req.model_path,
                    device_down="cuda:2",
                    device_up="cuda:3",
                    steps=req.steps,
                    seed=seed_b,
                    guidance_scale=req.guidance_scale,
                    width=req.width,
                    height=req.height,
                    output_path=out_b,
                    callback=make_cb("b"),
                )
            else:
                from pipeline_parallel_sdxl import generate_sdxl

                generate_sdxl(
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt,
                    model_path=req.model_path,
                    device_down="cuda:2",
                    device_up="cuda:3",
                    steps=req.steps,
                    width=req.width,
                    height=req.height,
                    seed=seed_b,
                    guidance_scale=req.guidance_scale,
                    use_fp32=req.use_fp32,
                    output_path=out_b,
                    scheduler=req.scheduler,
                    lora_path=req.lora_path,
                    lora_scale=req.lora_scale,
                    callback=make_cb("b"),
                )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Pair B: {exc}")

    # Run both pairs in parallel threads (they occupy different GPUs).
    ta = threading.Thread(target=work_a, name=f"{job_id}-pairA")
    tb = threading.Thread(target=work_b, name=f"{job_id}-pairB")
    ta.start()
    tb.start()
    ta.join()
    tb.join()

    if errors:
        JOBS.update(
            job_id,
            status="failed",
            stage="error",
            error="; ".join(errors),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )
        return

    JOBS.update(
        job_id,
        status="done",
        progress=100,
        stage="complete",
        image_url_a=f"/gallery/{job_id}_a.png",
        image_url_b=f"/gallery/{job_id}_b.png",
        finished_at=datetime.now(timezone.utc).isoformat(),
    )

    for label, seed in (("a", seed_a), ("b", seed_b)):
        _add_history_entry(
            {
                "job_id": f"{job_id}_{label}",
                "prompt": req.prompt,
                "negative_prompt": req.negative_prompt,
                "arch": req.arch,
                "model_path": req.model_path,
                "steps": req.steps,
                "width": req.width,
                "height": req.height,
                "seed": seed,
                "guidance_scale": req.guidance_scale,
                "scheduler": req.scheduler,
                "lora_path": req.lora_path,
                "lora_scale": req.lora_scale,
                "gpu_pair": "0+1" if label == "a" else "2+3",
                "image_url": f"/gallery/{job_id}_{label}.png",
                "twin_group": job_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        )


def _run_batch_generation(job_id: str, req: BatchRequest) -> None:
    """Batch via the multiprocess 4-GPU backend."""
    JOBS.update(
        job_id,
        status="running",
        stage="loading",
        progress=0,
        total=len(req.prompts),
        completed=0,
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    try:
        import parallel_pipelines_4gpu as pp4

        out_dir = os.path.join(OUTPUT_DIR, job_id)
        os.makedirs(out_dir, exist_ok=True)

        # We cannot stream progress from the multiprocess backend easily, so
        # we run it in a thread and poll the output directory for finished
        # images to report approximate progress.
        done_holder: Dict[str, int] = {"n": 0}

        def runner():
            seeds = [req.base_seed + i for i in range(len(req.prompts))]
            pp4.generate_batch_parallel_processes(
                prompts=req.prompts,
                model_path=req.model_path,
                steps=req.steps,
                width=req.width,
                height=req.height,
                guidance_scale=req.guidance_scale,
                negative_prompt=req.negative_prompt,
                seeds=seeds,
                output_dir=out_dir,
            )

        t = threading.Thread(target=runner, name=f"{job_id}-batch")
        t.start()

        # Poll for completion by counting files in out_dir.
        total = len(req.prompts)
        while t.is_alive():
            try:
                files = [f for f in os.listdir(out_dir) if f.endswith(".png")]
                done_holder["n"] = len(files)
                pct = int(len(files) / total * 100) if total else 0
                JOBS.update(
                    job_id,
                    progress=pct,
                    completed=len(files),
                    stage="generating",
                )
            except Exception:
                pass
            t.join(timeout=2.0)

        files = sorted(f for f in os.listdir(out_dir) if f.endswith(".png"))
        image_urls = [f"/gallery/{job_id}/{f}" for f in files]

        JOBS.update(
            job_id,
            status="done",
            progress=100,
            stage="complete",
            completed=len(files),
            image_urls=image_urls,
            output_dir=out_dir,
            finished_at=datetime.now(timezone.utc).isoformat(),
        )

        for i, (prompt, fname) in enumerate(zip(req.prompts, files)):
            _add_history_entry(
                {
                    "job_id": f"{job_id}_{i:03d}",
                    "batch_id": job_id,
                    "prompt": prompt,
                    "negative_prompt": req.negative_prompt,
                    "arch": req.arch,
                    "model_path": req.model_path,
                    "steps": req.steps,
                    "width": req.width,
                    "height": req.height,
                    "seed": req.base_seed + i,
                    "guidance_scale": req.guidance_scale,
                    "scheduler": req.scheduler,
                    "image_url": f"/gallery/{job_id}/{fname}",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            )
    except Exception as exc:
        logger.exception("Batch job %s failed", job_id)
        JOBS.update(
            job_id,
            status="failed",
            stage="error",
            error=str(exc),
            finished_at=datetime.now(timezone.utc).isoformat(),
        )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="FP32 Diffusion Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup():
    JOBS.set_loop(asyncio.get_running_loop())
    logger.info("API server started. Models dir: %s", MODELS_DIR)


# --- Discovery -------------------------------------------------------------

@app.get("/api/models")
async def list_models():
    models = []
    if os.path.isdir(MODELS_DIR):
        for name in sorted(os.listdir(MODELS_DIR)):
            full = os.path.join(MODELS_DIR, name)
            if os.path.isdir(full):
                models.append({"name": name, "path": full})
    return {"models": models}


@app.get("/api/lora")
async def list_lora():
    loras = []
    if os.path.isdir(LORAS_DIR):
        for name in sorted(os.listdir(LORAS_DIR)):
            if name.lower().endswith((".safetensors", ".ckpt", ".bin")):
                loras.append(
                    {"name": name, "path": os.path.join(LORAS_DIR, name)}
                )
    return {"loras": loras}


@app.get("/api/gpus")
async def get_gpus():
    return {"gpus": _detect_gpus(), "live_vram": _gpu_vram()}


@app.get("/api/config")
async def get_config():
    return {
        "arch_choices": ARCH_CHOICES,
        "scheduler_choices": SCHEDULER_CHOICES,
        "aspect_presets": ASPECT_PRESETS,
        "default_model": "./models/sdxl-base-fp16",
        "default_model_sdxl": "./models/sdxl-base-fp16",
        "default_model_sd15": "./models/sd15-fp16",
        "default_model_turbo": "./models/sdxl-turbo",
        "gpu_pairs": ["0+1", "2+3"],
    }


# --- Generation ------------------------------------------------------------

@app.post("/api/generate")
async def api_generate(req: GenerateRequest):
    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id,
        "type": "single",
        "status": "queued",
        "prompt": req.prompt,
        "arch": req.arch,
        "steps": req.steps,
        "width": req.width,
        "height": req.height,
        "gpu_pair": req.gpu_pair,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "progress": 0,
    }
    JOBS.create(job_id, job)
    EXECUTOR.submit(_run_single_generation, job_id, req)
    return {"job_id": job_id}


@app.post("/api/twin")
async def api_twin(req: TwinRequest):
    """2 images of the same prompt on 4 GPUs simultaneously."""
    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id,
        "type": "twin",
        "status": "queued",
        "prompt": req.prompt,
        "arch": req.arch,
        "steps": req.steps,
        "width": req.width,
        "height": req.height,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "progress": 0,
    }
    JOBS.create(job_id, job)
    # Twin uses BOTH pairs, so run it directly (it manages its own threads).
    threading.Thread(target=_run_twin_generation, args=(job_id, req), name=f"{job_id}-twin").start()
    return {"job_id": job_id}


@app.post("/api/batch")
async def api_batch(req: BatchRequest):
    job_id = uuid.uuid4().hex[:12]
    job = {
        "job_id": job_id,
        "type": "batch",
        "status": "queued",
        "prompts": req.prompts,
        "total": len(req.prompts),
        "completed": 0,
        "arch": req.arch,
        "steps": req.steps,
        "width": req.width,
        "height": req.height,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "progress": 0,
    }
    JOBS.create(job_id, job)
    threading.Thread(target=_run_batch_generation, args=(job_id, req), name=f"{job_id}-batch").start()
    return {"job_id": job_id}


# --- Job status & SSE ------------------------------------------------------

@app.get("/api/jobs")
async def api_list_jobs():
    return {"jobs": JOBS.list_jobs()}


@app.get("/api/jobs/{job_id}")
async def api_get_job(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


@app.get("/api/jobs/{job_id}/events")
async def api_job_events(job_id: str, request: Request):
    """Server-Sent Events stream for a single job's progress."""
    if JOBS.get(job_id) is None:
        raise HTTPException(404, "job not found")

    queue = JOBS.subscribe(job_id)

    async def event_stream():
        try:
            # Send the current state immediately.
            job = JOBS.get(job_id)
            if job:
                yield f"data: {json.dumps(job)}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("status") in ("done", "failed"):
                        break
                except asyncio.TimeoutError:
                    # heartbeat keeps the connection alive.
                    yield ": heartbeat\n\n"
        finally:
            JOBS.unsubscribe(job_id, queue)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# --- Gallery / history -----------------------------------------------------

@app.get("/api/history")
async def api_history():
    return {"history": _load_history()}


@app.delete("/api/history/{job_id}")
async def api_delete_history(job_id: str):
    history = _load_history()
    remaining = [h for h in history if not h.get("job_id", "").startswith(job_id)]
    _save_history(remaining)
    # Best-effort file deletion.
    for h in history:
        jid = h.get("job_id", "")
        if jid.startswith(job_id) and h.get("image_url"):
            fpath = os.path.join(BASE_DIR, h["image_url"].lstrip("/"))
            try:
                os.remove(fpath)
            except OSError:
                pass
    return {"deleted": len(history) - len(remaining)}


# --- Cache & GPU management ------------------------------------------------

@app.get("/api/cache/stats")
async def api_cache_stats():
    try:
        from pipeline_cache import get_cache

        return get_cache().stats()
    except Exception as exc:
        return {"error": str(exc)}


@app.post("/api/cache/control")
async def api_cache_control(req: CacheControlRequest):
    try:
        from pipeline_cache import get_cache

        cache = get_cache()
        if req.action == "unload_all":
            n = cache.unload_all()
            return {"unloaded": n}
        return {"error": "unknown action"}
    except Exception as exc:
        raise HTTPException(500, str(exc))


# --- Static file serving ---------------------------------------------------

# Gallery images
app.mount("/gallery", StaticFiles(directory=OUTPUT_DIR), name="gallery")

# React build (production) — serves web/dist if present.
_WEB_DIST = os.path.join(WEB_DIR, "dist")
if os.path.isdir(_WEB_DIST):
    app.mount("/", StaticFiles(directory=_WEB_DIST, html=True), name="web")
else:
    # Dev fallback: serve a placeholder so the root route doesn't 404.
    @app.get("/")
    async def root():
        return JSONResponse(
            {
                "message": "FP32 Diffusion API is running.",
                "frontend": "Build the React app in web/ (npm run build) or run it in dev mode on port 5173.",
                "docs": "/docs",
            }
        )


# ---------------------------------------------------------------------------
# Direct-run entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("SD_API_PORT", "8765"))
    host = os.environ.get("SD_API_HOST", "0.0.0.0")
    uvicorn.run("api_server:app", host=host, port=port, reload=False)