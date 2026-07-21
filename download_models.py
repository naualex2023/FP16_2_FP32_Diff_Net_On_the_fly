#!/usr/bin/env python3
"""
download_models.py — Fetch SD 1.5 / SDXL / SDXL-Turbo into ./models/.

Models are stored in their native dtype (FP16) for compactness; weights are
upcast to FP32 at load time by the pipeline-parallel code.

Usage:
    python download_models.py --model sd15
    python download_models.py --model sdxl
    python download_models.py --model sdxl-turbo
    python download_models.py --model all
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


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        choices=list(MODELS) + ["all"],
        required=True,
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    targets = list(MODELS) if args.model == "all" else [args.model]
    for t in targets:
        download(t, force=args.force)


if __name__ == "__main__":
    main()
