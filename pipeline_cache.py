"""
pipeline_cache.py — Keep-alive cache for pipeline-parallel SD/SDXL pipelines.

Without this cache, every call to :func:`generate_sdxl` /
:func:`generate_sd15_pipeline_parallel` reloads the model from disk, re-splits
the UNet across GPUs, and (for SDXL) re-upcasts ~25 GB to FP32.  That makes
back-to-back generations painfully slow.

This module keeps a loaded pipeline resident in VRAM across generations and
**automatically frees it after a configurable idle timeout** (the "regulated
timeout" requested in the task).  Each successful ``get``/generation resets the
idle timer ("touch on use"), so the model stays loaded while you are actively
generating and is reaped only when the pipeline is truly idle.

Key features
------------
* **Process-local singleton** — reuse within a single Python process / server.
* **Idle reaper** — a daemon thread evicts entries that have been untouched for
  longer than ``idle_timeout`` seconds, releasing VRAM via
  ``del pipe; gc.collect(); torch.cuda.empty_cache()``.
* **Thread-safe** — one :class:`threading.RLock` guards the registry; each
  cached entry additionally has its own lock so two prompts routed to the same
  GPU pair serialize instead of corrupting the shared pipeline.
* **Configurable** — via constructor args or the environment variables
  ``SD_IDLE_TIMEOUT`` (seconds, default ``300``) and ``SD_KEEP_ALIVE``
  (``"1"``/``"0"``, default ``"1"``).  Set ``SD_IDLE_TIMEOUT=0`` to disable the
  reaper (model stays loaded until the process exits or :meth:`unload_all`).

Public API
----------
* :data:`get_cache` — process-wide :class:`PipelineCache` instance.
* :func:`cached_sdxl_pipeline` / :func:`cached_sd15_pipeline` — convenience
  helpers that build the right pipeline and return it from the cache.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r, using default %s", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# Defaults driven by env vars so operators can tune without code changes.
DEFAULT_IDLE_TIMEOUT = _env_float("SD_IDLE_TIMEOUT", 300.0)
DEFAULT_KEEP_ALIVE = _env_bool("SD_KEEP_ALIVE", True)
DEFAULT_REAPER_INTERVAL = _env_float("SD_REAPER_INTERVAL", 30.0)


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------

class _CacheEntry:
    """One resident pipeline + bookkeeping."""

    __slots__ = ("pipe", "factory_key", "last_used", "lock", "build_seconds")

    def __init__(self, pipe: Any, factory_key: Tuple[Any, ...], build_seconds: float):
        self.pipe = pipe
        self.factory_key = factory_key
        self.last_used = time.monotonic()
        self.lock = threading.RLock()
        self.build_seconds = build_seconds

    def touch(self) -> None:
        self.last_used = time.monotonic()


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class PipelineCache:
    """Keep-alive cache of loaded diffusers pipelines with an idle reaper.

    Parameters
    ----------
    idle_timeout : float
        Seconds of inactivity after which an entry is eligible for eviction.
        ``0`` (or negative) disables the reaper — entries persist until
        :meth:`unload` / :meth:`unload_all` / process exit.
    keep_alive : bool
        Master switch.  When ``False``, :meth:`get_or_load` always rebuilds and
        never stores, reproducing the original per-call behavior.  Useful for
        tests / one-shot scripts.
    reaper_interval : float
        How often the background thread scans for idle entries.
    """

    def __init__(
        self,
        idle_timeout: float = DEFAULT_IDLE_TIMEOUT,
        keep_alive: bool = DEFAULT_KEEP_ALIVE,
        reaper_interval: float = DEFAULT_REAPER_INTERVAL,
    ) -> None:
        self.idle_timeout = float(idle_timeout)
        self.keep_alive = bool(keep_alive)
        self.reaper_interval = float(reaper_interval)

        self._entries: Dict[Tuple[Any, ...], _CacheEntry] = {}
        self._registry_lock = threading.RLock()
        self._reaper_stop = threading.Event()
        self._reaper: Optional[threading.Thread] = None

        if self.keep_alive and self.idle_timeout > 0:
            self._start_reaper()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_load(
        self,
        factory_key: Tuple[Any, ...],
        factory: Callable[[], Any],
    ) -> Tuple[Any, Optional[_CacheEntry]]:
        """Return the cached pipeline for *factory_key*, loading it if absent.

        Parameters
        ----------
        factory_key : tuple
            Hashable identity of the pipeline (model path, devices, dtype,
            LoRA, scheduler...).  Two different keys get two entries.
        factory : callable
            Called (under the registry lock, only on miss) to build the
            pipeline.  Must return a diffusers pipeline object.

        Returns
        -------
        (pipe, entry)
            The pipeline and its :class:`_CacheEntry` (or ``None`` when
            ``keep_alive`` is disabled — callers should not hold the entry lock
            in that case).
        """
        if not self.keep_alive:
            logger.debug("keep_alive disabled — building pipeline without caching")
            return factory(), None

        with self._registry_lock:
            entry = self._entries.get(factory_key)
            if entry is None:
                logger.info("Cache MISS — building pipeline for %s", self._safe_key(factory_key))
                t0 = time.perf_counter()
                pipe = factory()
                dt = time.perf_counter() - t0
                entry = _CacheEntry(pipe, factory_key, dt)
                self._entries[factory_key] = entry
                logger.info(
                    "Pipeline cached (%.1fs to build); %d resident",
                    dt, len(self._entries),
                )
            else:
                logger.info(
                    "Cache HIT — reusing resident pipeline (%d resident)",
                    len(self._entries),
                )
            entry.touch()
            return entry.pipe, entry

    def release(self, factory_key: Tuple[Any, ...]) -> bool:
        """Evict and free one entry by key.  Returns ``True`` if something was freed."""
        with self._registry_lock:
            entry = self._entries.pop(factory_key, None)
        if entry is None:
            return False
        self._destroy_entry(entry)
        return True

    def unload_all(self) -> int:
        """Evict and free every cached entry.  Returns the number freed."""
        with self._registry_lock:
            entries = list(self._entries.values())
            self._entries.clear()
        for entry in entries:
            self._destroy_entry(entry)
        return len(entries)

    def stats(self) -> Dict[str, Any]:
        """Return a snapshot of the cache state (for logging / debugging)."""
        with self._registry_lock:
            now = time.monotonic()
            return {
                "keep_alive": self.keep_alive,
                "idle_timeout": self.idle_timeout,
                "resident": len(self._entries),
                "entries": [
                    {
                        "key": self._safe_key(e.factory_key),
                        "idle_seconds": round(now - e.last_used, 1),
                        "build_seconds": round(e.build_seconds, 2),
                    }
                    for e in self._entries.values()
                ],
            }

    def shutdown(self) -> None:
        """Stop the reaper and free everything.  Safe to call multiple times."""
        self._reaper_stop.set()
        if self._reaper is not None and self._reaper.is_alive():
            self._reaper.join(timeout=self.reaper_interval + 5)
            self._reaper = None
        self.unload_all()

    # Allow ``with PipelineCache(...) as c:`` for deterministic teardown.
    def __enter__(self) -> "PipelineCache":
        return self

    def __exit__(self, *exc) -> None:
        self.shutdown()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_key(key: Tuple[Any, ...]) -> str:
        try:
            return str(key)
        except Exception:  # pragma: no cover - defensive
            return "<unprintable key>"

    def _start_reaper(self) -> None:
        self._reaper = threading.Thread(
            target=self._reaper_loop,
            name="PipelineCacheReaper",
            daemon=True,
        )
        self._reaper.start()
        logger.info(
            "PipelineCache idle reaper started (timeout=%.0fs, interval=%.0fs)",
            self.idle_timeout, self.reaper_interval,
        )

    def _reaper_loop(self) -> None:
        while not self._reaper_stop.wait(timeout=self.reaper_interval):
            try:
                self._reap_once()
            except Exception:  # pragma: no cover - never let the reaper die
                logger.exception("Reaper iteration failed (continuing)")

    def _reap_once(self) -> None:
        now = time.monotonic()
        with self._registry_lock:
            expired = [
                (key, entry)
                for key, entry in self._entries.items()
                if now - entry.last_used > self.idle_timeout
            ]
            for key, _ in expired:
                self._entries.pop(key, None)
        for key, entry in expired:
            idle = now - entry.last_used
            logger.info(
                "Reaping idle pipeline (idle %.0fs > timeout %.0fs): %s",
                idle, self.idle_timeout, self._safe_key(key),
            )
            self._destroy_entry(entry)

    def _destroy_entry(self, entry: _CacheEntry) -> None:
        """Free one entry's VRAM.  Best-effort; logs but never raises."""
        # Take the per-entry lock so we don't tear down mid-generation.  The
        # RLock means an in-flight forward pass that holds it will block us
        # here until it finishes — which is the correct behavior.
        with entry.lock:
            try:
                del entry.pipe
            except Exception:  # pragma: no cover
                logger.exception("Error deleting pipeline for %s", self._safe_key(entry.factory_key))
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # pragma: no cover
                logger.exception("torch.cuda.empty_cache() failed during teardown")


# ---------------------------------------------------------------------------
# Process-wide singleton
# ---------------------------------------------------------------------------

_CACHE: Optional[PipelineCache] = None
_CACHE_LOCK = threading.Lock()


def get_cache() -> PipelineCache:
    """Return the process-wide :class:`PipelineCache` (created lazily)."""
    global _CACHE
    if _CACHE is None:
        with _CACHE_LOCK:
            if _CACHE is None:
                _CACHE = PipelineCache(
                    idle_timeout=DEFAULT_IDLE_TIMEOUT,
                    keep_alive=DEFAULT_KEEP_ALIVE,
                    reaper_interval=DEFAULT_REAPER_INTERVAL,
                )
    return _CACHE


# ---------------------------------------------------------------------------
# Convenience builders that pair this cache with the project's pipelines.
# ---------------------------------------------------------------------------

def cached_sdxl_pipeline(
    model_path: str,
    device_down: str,
    device_up: str,
    use_fp32: bool = True,
    lora_path: Optional[str] = None,
    lora_scale: float = 1.0,
    scheduler: str = "default",
    compile: bool = False,
    cache: Optional[PipelineCache] = None,
) -> Tuple[Any, Optional[_CacheEntry]]:
    """Return a (possibly cached) SDXL pipeline-parallel object.

    The cache key includes every parameter that changes what gets loaded, so
    different models / GPUs / LoRAs / schedulers get distinct entries.  The
    ``model_path`` is normalised via :mod:`model_resolver` so that a repo ID
    and its downloaded local copy share one cache entry.
    """
    from model_resolver import resolve_model_path

    cache = cache or get_cache()
    # Resolve early so the cache key is identical whether the caller passed a
    # local path or a HuggingFace repo ID.
    resolved_path = resolve_model_path(model_path)
    key = (
        "sdxl",
        os.path.abspath(resolved_path),
        str(device_down),
        str(device_up),
        bool(use_fp32),
        os.path.abspath(lora_path) if lora_path else None,
        float(lora_scale),
        str(scheduler),
        bool(compile),
    )

    def _factory():
        from pipeline_parallel_sdxl import create_sdxl_pipeline_parallel
        return create_sdxl_pipeline_parallel(
            model_path=resolved_path,
            device_down=device_down,
            device_up=device_up,
            use_fp32=use_fp32,
            compile=compile,
            lora_path=lora_path,
            lora_scale=lora_scale,
            scheduler=scheduler,
        )

    return cache.get_or_load(key, _factory)


def cached_sd15_pipeline(
    model_path: str,
    device_down: str,
    device_up: str,
    cache: Optional[PipelineCache] = None,
) -> Tuple[Any, Optional[_CacheEntry]]:
    """Return a (possibly cached) SD 1.5 pipeline-parallel object.

    The ``model_path`` is normalised via :mod:`model_resolver` so that a repo
    ID and its downloaded local copy share one cache entry.
    """
    from model_resolver import resolve_model_path

    cache = cache or get_cache()
    resolved_path = resolve_model_path(model_path)
    key = (
        "sd15",
        os.path.abspath(resolved_path),
        str(device_down),
        str(device_up),
    )

    def _factory():
        from pipeline_parallel_sd15 import create_sd15_pipeline_parallel
        return create_sd15_pipeline_parallel(
            model_path=resolved_path,
            device_down=device_down,
            device_up=device_up,
        )

    return cache.get_or_load(key, _factory)


def cached_single_gpu_pipeline(
    model_path: str,
    device: str,
    arch: str = "sdxl",
    use_fp32: bool = True,
    lora_path: Optional[str] = None,
    lora_scale: float = 1.0,
    scheduler: str = "default",
    cache: Optional[PipelineCache] = None,
) -> Tuple[Any, Optional[_CacheEntry]]:
    """Return a (possibly cached) single-GPU pipeline object (no UNet split).

    Used by Quadro mode, which runs four of these concurrently — one per GPU.
    The cache key is distinct from the pipeline-parallel keys (prefix
    ``"single"`` plus the ``arch``), so a model used in both split and
    single-GPU modes keeps separate resident entries.  The ``model_path`` is
    normalised via :mod:`model_resolver` so a repo ID and its downloaded local
    copy share one cache entry.
    """
    from model_resolver import resolve_model_path

    cache = cache or get_cache()
    resolved_path = resolve_model_path(model_path)
    key = (
        "single",
        str(arch),
        os.path.abspath(resolved_path),
        str(device),
        bool(use_fp32),
        os.path.abspath(lora_path) if lora_path else None,
        float(lora_scale),
        str(scheduler),
    )

    def _factory():
        from pipeline_single_gpu import create_single_gpu_pipeline
        return create_single_gpu_pipeline(
            model_path=resolved_path,
            device=device,
            arch=arch,
            use_fp32=use_fp32,
            lora_path=lora_path,
            lora_scale=lora_scale,
            scheduler=scheduler,
        )

    return cache.get_or_load(key, _factory)


def cached_dit_pipeline(
    model_path: str,
    device_down: str,
    device_up: str,
    use_fp32: bool = True,
    scheduler: str = "default",
    num_gpus: Optional[int] = None,
    cache: Optional[PipelineCache] = None,
) -> Tuple[Any, Optional[_CacheEntry]]:
    """Return a (possibly cached) DiT pipeline-parallel object.

    DiT-based models (PixArt, Sana, SD3, Flux, …) use a Transformer backbone
    instead of a UNet; in FP32 they reach 30–50 GB and are split across two
    GPUs by :class:`pp_dit.PipelineParallelDiT`.  The cache key is distinct
    from the SDXL/SD15/single keys (prefix ``"dit"``), so a model used in both
    split and single-GPU modes keeps separate resident entries.  The
    ``model_path`` is normalised via :mod:`model_resolver` so a repo ID and its
    downloaded local copy share one cache entry.
    """
    from model_resolver import resolve_model_path

    cache = cache or get_cache()
    resolved_path = resolve_model_path(model_path)
    key = (
        "dit",
        os.path.abspath(resolved_path),
        str(device_down),
        str(device_up),
        bool(use_fp32),
        str(scheduler),
        str(num_gpus),
    )

    def _factory():
        from pipeline_parallel_dit import create_dit_pipeline_parallel
        return create_dit_pipeline_parallel(
            model_path=resolved_path,
            device_down=device_down,
            device_up=device_up,
            use_fp32=use_fp32,
            scheduler=scheduler,
            num_gpus=num_gpus,
        )

    return cache.get_or_load(key, _factory)


def cached_gguf_pipeline(
    model_path: str,
    device: str = "cuda:0",
    use_fp16_compute: bool = True,
    scheduler: str = "default",
    cache: Optional[PipelineCache] = None,
) -> Tuple[Any, Optional[_CacheEntry]]:
    """Return a (possibly cached) GGUF-quantized pipeline object (1 GPU).

    GGUF models (Q4/Q8 quantized) are small enough to fit on a single P40.
    Used by Generate / Quadro / Batch modes when the user selects the GGUF
    architecture.  The cache key is prefixed ``"gguf"`` so a quantized model
    and its full-precision counterpart keep separate resident entries.
    """
    cache = cache or get_cache()
    key = (
        "gguf",
        os.path.abspath(model_path),
        str(device),
        bool(use_fp16_compute),
        str(scheduler),
    )

    def _factory():
        from pipeline_gguf import create_gguf_pipeline
        return create_gguf_pipeline(
            model_path=model_path,
            device=device,
            use_fp16_compute=use_fp16_compute,
            scheduler=scheduler,
        )

    return cache.get_or_load(key, _factory)
