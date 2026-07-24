"""
dit_size_estimator.py — Measure DiT model component sizes on disk.

Pure-Python (no torch dependency) so it can be unit-tested anywhere and used
by :mod:`pipeline_parallel_dit` for GPU auto-detection.

The estimator has two strategies:

1. **Primary — measure on-disk checkpoint files (exact):**
   Sum ``.safetensors`` / ``.bin`` / ``.pt`` / ``.pth`` file sizes per
   component directory, then scale from the storage dtype (read from each
   ``config.json``'s ``torch_dtype``) to the requested compute dtype.

2. **Fallback — shape heuristic (approximate):**
   If no weight files are found anywhere, estimate parameter counts from
   ``config.json`` fields.  Less accurate (under-counts MMDiT and T5-XXL)
   but better than nothing for unusual layouts.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Storage bytes per parameter by dtype.
BYTES_PER_PARAM: dict[str, int] = {"float16": 2, "bfloat16": 2, "float32": 4}

# The diffusers component subdirectories we know how to size.
COMPONENTS: tuple[str, ...] = (
    "transformer", "text_encoder", "text_encoder_2", "text_encoder_3", "vae",
)


def sum_weight_bytes(directory: str) -> int:
    """Sum sizes of all checkpoint files in *directory*.

    Recognises ``.safetensors``, ``.bin``, ``.pt`` and ``.pth``.  Returns
    the total number of bytes on disk (before any dtype scaling).
    """
    total = 0
    for pattern in ("*.safetensors", "*.bin", "*.pt", "*.pth"):
        for f in glob.glob(os.path.join(directory, pattern)):
            try:
                total += os.path.getsize(f)
            except OSError:
                pass
    return total


def on_disk_dtype(config: dict) -> str:
    """Determine a component's storage dtype from its ``config.json``.

    Falls back to ``"float16"`` (this project downloads in FP16 for
    compactness; weights are upcast to FP32 at load time).
    """
    raw = str(config.get("torch_dtype", "")).lower().strip()
    # Order matters: "float16" contains "float", so check the more specific
    # float16/bfloat16 patterns BEFORE the float32 / bare-float matchers.
    if "bfloat16" in raw or "bf16" in raw:
        return "bfloat16"
    if "float16" in raw or "fp16" in raw or "half" in raw:
        return "float16"
    if "float32" in raw or "fp32" in raw or raw == "float":
        return "float32"
    return "float16"  # project default


def estimate_params_heuristic(config: dict) -> float:
    """Rough parameter-count estimate from config fields (in billions).

    Used only as a *fallback* when no weight files are found on disk.
    The formula ``12 * hidden^2 * layers`` is a coarse approximation for
    decoder-style transformers; it under-counts MMDiT and T5 architectures.
    """
    hidden = config.get("hidden_size") or config.get("num_attention_heads", 0) * (
        config.get("attention_head_dim", 0) or config.get("dim", 0)
    )
    layers = config.get("num_layers") or config.get("num_hidden_layers", 0)
    vocab = config.get("vocab_size", 0)

    if hidden and layers:
        layer_params = 12 * hidden * hidden
        total = layer_params * layers
        if vocab:
            total += vocab * hidden * 2
        return total / 1e9
    return 0.0


def estimate_model_size_gb(
    model_path: str,
    target_dtype: str = "float32",
) -> Optional[dict]:
    """Estimate each component's size (GB) scaled to *target_dtype*.

    Returns a dict like ``{"transformer": 32.0, "text_encoder_3": 18.8}``
    or ``None`` if nothing can be determined.
    """
    sizes: dict[str, float] = {}
    found_files = False

    target_bpp = BYTES_PER_PARAM.get(target_dtype, 4)

    for comp in COMPONENTS:
        comp_dir = os.path.join(model_path, comp)
        if not os.path.isdir(comp_dir):
            continue

        raw_bytes = sum_weight_bytes(comp_dir)
        if raw_bytes > 0:
            found_files = True
            cfg: dict = {}
            cfg_path = os.path.join(comp_dir, "config.json")
            if os.path.isfile(cfg_path):
                try:
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        cfg = json.load(f)
                except Exception:
                    pass
            disk_bpp = BYTES_PER_PARAM.get(on_disk_dtype(cfg), 2)
            scale = target_bpp / disk_bpp
            sizes[comp] = (raw_bytes * scale) / 1e9

    if found_files:
        return sizes

    # ---- fallback: shape heuristic -----------------------------------------
    logger.debug("No checkpoint files found; falling back to shape heuristic")
    for comp in COMPONENTS:
        cfg_path = os.path.join(model_path, comp, "config.json")
        if not os.path.isfile(cfg_path):
            continue
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            continue
        params_bn = estimate_params_heuristic(cfg)
        if params_bn > 0:
            sizes[comp] = params_bn * target_bpp  # → GB in target dtype

    return sizes if sizes else None