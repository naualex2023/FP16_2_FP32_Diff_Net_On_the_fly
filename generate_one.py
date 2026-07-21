#!/usr/bin/env python3
"""
generate_one.py — CLI for single image generation with pipeline parallelism.

Examples
--------
# SDXL FP32 on GPU 0 + GPU 1:
python generate_one.py --model ./models/sdxl-base-fp16 \
    --gpu-down 0 --gpu-up 1 \
    --prompt "A cyberpunk cityscape at night" \
    --output out_0.png

# SDXL Turbo (4 steps, CFG=0):
python generate_one.py --model ./models/sdxl-turbo \
    --gpu-down 0 --gpu-up 1 \
    --prompt "a cinematic photo of a cat" \
    --steps 4 --cfg 0.0 --output turbo.png

# SD 1.5 (educational, pipeline parallel on small model):
python generate_one.py --model ./models/sd15-fp16 --arch sd15 \
    --gpu-down 0 --gpu-up 1 --prompt "a cat" --output sd15_pp.png
"""

from __future__ import annotations

import argparse
import logging
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate one image with pipeline-parallel FP32 on Tesla P40",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True, help="Path to diffusers model directory")
    p.add_argument(
        "--arch",
        choices=["sdxl", "sd15"],
        default="sdxl",
        help="Model architecture",
    )
    p.add_argument("--gpu-down", type=int, default=0, help="Stage-0 GPU index")
    p.add_argument("--gpu-up", type=int, default=1, help="Stage-1 GPU index")
    p.add_argument("--prompt", required=True, help="Text prompt")
    p.add_argument("--negative", default="blurry, low quality, distorted, ugly")
    p.add_argument("--output", default="output.png")
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--cfg", type=float, default=7.5, help="Guidance scale (CFG)")
    p.add_argument(
        "--scheduler",
        choices=["default", "ddim", "euler", "dpmpp_2m"],
        default="default",
    )
    p.add_argument("--lora", default=None, help="Path to LoRA .safetensors")
    p.add_argument("--lora-scale", type=float, default=1.0)
    p.add_argument("--fp16", action="store_true", help="Use FP16 (NOT recommended on P40)")
    p.add_argument(
        "--async-transfer",
        action="store_true",
        help="Use CUDA-stream overlapped transfers (experimental)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    dev_down = f"cuda:{args.gpu_down}"
    dev_up = f"cuda:{args.gpu_up}"

    if args.arch == "sd15":
        from pipeline_parallel_sd15 import generate_sd15_pipeline_parallel

        generate_sd15_pipeline_parallel(
            prompt=args.prompt,
            negative_prompt=args.negative,
            model_path=args.model,
            device_down=dev_down,
            device_up=dev_up,
            steps=args.steps,
            seed=args.seed,
            width=args.width,
            height=args.height,
            guidance_scale=args.cfg,
            output_path=args.output,
        )
    else:  # sdxl
        from pipeline_parallel_sdxl import generate_sdxl

        generate_sdxl(
            prompt=args.prompt,
            negative_prompt=args.negative,
            model_path=args.model,
            device_down=dev_down,
            device_up=dev_up,
            steps=args.steps,
            width=args.width,
            height=args.height,
            seed=args.seed,
            guidance_scale=args.cfg,
            use_fp32=not args.fp16,
            output_path=args.output,
            scheduler=args.scheduler,
            lora_path=args.lora,
            lora_scale=args.lora_scale,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
