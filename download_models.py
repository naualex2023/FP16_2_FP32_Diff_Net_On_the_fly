#!/usr/bin/env python3
"""
download_models.py — Fetch diffusion models into ./models/.

Two modes are supported:

1. **Preset models** (the three originals):
    python download_models.py --model sd15
    python download_models.py --model sdxl
    python download_models.py --model sdxl-turbo
    python download_models.py --model all

2. **Any HuggingFace repo** (new — removes the hardcoding):
    python download_models.py --repo stabilityai/sdxl-turbo
    python download_models.py --repo runwayml/stable-diffusion-v1-5 --class StableDiffusionPipeline
    python download_models.py --repo <org>/<name> --dtype float32

Models are stored in their native dtype (FP16 by default) for compactness;
weights are upcast to FP32 at load time by the pipeline-parallel code.
"""

from __future__ import annotations

import argparse
import logging
import os

logger = logging.getLogger(__name__)

MODELS = {
    "sd15": {
        "class": "StableDiffusionPipeline",
        "repo": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "dst": "./models/sd15-fp16",
        "dtype": "float16",
    },
    "sdxl": {
        "class": "StableDiffusionXLPipeline",
        "repo": "stabilityai/stable-diffusion-xl-base-1.0",
        "dst": "./models/sdxl-base-fp16",
        "dtype": "float16",
    },
    "sdxl-turbo": {
        "class": "StableDiffusionXLPipeline",
        "repo": "stabilityai/sdxl-turbo",
        "dst": "./models/sdxl-turbo",
        "dtype": "float16",
    },
}


def download(name: str, force: bool = False) -> None:
    """Download one of the preset models by short name."""
    import torch  # noqa: F401
    import diffusers
    info = MODELS[name]
    dst = info["dst"]

    if os.path.isdir(dst) and os.listdir(dst) and not force:
        logger.info("[%s] already present at %s (use --force to re-download)", name, dst)
        return

    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    logger.info("[%s] downloading %s -> %s", name, info["repo"], dst)

    cls = getattr(diffusers, info["class"])
    dtype = getattr(torch, info["dtype"])
    pipe = cls.from_pretrained(info["repo"], torch_dtype=dtype)
    pipe.save_pretrained(dst)
    logger.info("[%s] done.", name)

    # Удалить модель из кэша HuggingFace, чтобы не дублировать на диске
    # (модель уже сохранена в ./models/ через save_pretrained).
    try:
        from clear_cache import purge_hf_cache_entry
        purge_hf_cache_entry(info["repo"])
    except Exception as exc:
        logger.warning("[%s] could not purge HF cache: %s", name, exc)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(
        description="Download preset or arbitrary HuggingFace diffusion models."
    )
    ap.add_argument(
        "--model",
        choices=list(MODELS) + ["all"],
        default=None,
        help="Preset model short name (sd15 / sdxl / sdxl-turbo / all)",
    )
    ap.add_argument(
        "--repo",
        default=None,
        help="Arbitrary HuggingFace repo ID, e.g. stabilityai/sdxl-turbo",
    )
    ap.add_argument(
        "--class",
        dest="pipeline_class",
        default=None,
        help="Diffusers pipeline class (auto-detected if omitted)",
    )
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.repo:
        from model_resolver import download_hf_model, infer_arch

        path = download_hf_model(
            repo_id=args.repo,
            dtype=args.dtype,
            pipeline_class=args.pipeline_class,
            force=args.force,
        )
        arch = infer_arch(path)
        logger.info("OK: %s -> %s (arch=%s)", args.repo, path, arch)
        return

    if not args.model:
        ap.error("one of --model or --repo is required")

    targets = list(MODELS) if args.model == "all" else [args.model]
    for t in targets:
        download(t, force=args.force)


if __name__ == "__main__":
    main()
