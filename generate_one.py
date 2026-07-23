#!/usr/bin/env python3
"""
generate_one.py — CLI for single image generation with pipeline parallelism.

By default each invocation is a fresh Python process, so the loaded model is
released when the process exits.  For back-to-back generations use
``--serve``: the process stays alive, reads prompts from stdin, and *reuses*
the already-loaded model for every image (no reload).  The model is kept in
VRAM for an idle timeout (env ``SD_IDLE_TIMEOUT``, default 300s) between
generations and freed automatically when idle.

Examples
--------
# SDXL FP32 on GPU 0 + GPU 1 (one-shot):
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

# Persistent server mode (reuse the loaded model across many prompts):
python generate_one.py --model ./models/sdxl-base-fp16 --serve
#   then type prompts, one per line:
#       a cat on the moon|out_cat.png|seed=123
#       a mountain lake|out_lake.png
#       quit
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
    p.add_argument("--prompt", default=None, help="Text prompt (omit with --serve)")
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
    p.add_argument(
        "--serve",
        action="store_true",
        help=(
            "Persistent mode: keep the process alive, read prompts from stdin, "
            "and reuse the loaded model across generations.  Lines look like: "
            "'prompt|output.png|seed=123|steps=20|cfg=7|w=1024|h=1024|neg=...'."
        ),
    )
    p.add_argument(
        "--no-keep-alive",
        action="store_true",
        help="Disable the idle-timeout cache (unload immediately after each job).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _generate(arch, **kw):
    """Dispatch to the right generator (keeps imports lazy)."""
    if arch == "sd15":
        from pipeline_parallel_sd15 import generate_sd15_pipeline_parallel

        generate_sd15_pipeline_parallel(**kw)
    else:  # sdxl
        from pipeline_parallel_sdxl import generate_sdxl

        generate_sdxl(**kw)


def _run_one(args) -> int:
    """Run a single one-shot generation."""
    if not args.prompt:
        logging.error("--prompt is required unless --serve is used")
        return 2

    dev_down = f"cuda:{args.gpu_down}"
    dev_up = f"cuda:{args.gpu_up}"

    if args.arch == "sd15":
        _generate(
            args.arch,
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
            force_unload=args.no_keep_alive,
        )
    else:  # sdxl
        _generate(
            args.arch,
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
            force_unload=args.no_keep_alive,
        )
    return 0


def _parse_serve_line(line: str, args) -> dict:
    """Parse a 'prompt|out.png|key=val...' serve-mode line into generate kwargs."""
    parts = [p.strip() for p in line.split("|")]
    prompt = parts[0]
    if not prompt:
        raise ValueError("empty prompt")

    kw = dict(
        prompt=prompt,
        negative_prompt=args.negative,
        model_path=args.model,
        device_down=f"cuda:{args.gpu_down}",
        device_up=f"cuda:{args.gpu_up}",
        steps=args.steps,
        seed=args.seed,
        width=args.width,
        height=args.height,
        guidance_scale=args.cfg,
        output_path="output.png",
        force_unload=False,
    )
    if args.arch == "sdxl":
        kw.update(
            use_fp32=not args.fp16,
            scheduler=args.scheduler,
            lora_path=args.lora,
            lora_scale=args.lora_scale,
        )

    # Second positional field, if present and not key=val, is the output path.
    idx = 1
    if len(parts) > 1 and "=" not in parts[1]:
        kw["output_path"] = parts[1]
        idx = 2

    short = {"seed": int, "steps": int, "cfg": float, "w": int, "h": int,
             "width": int, "height": int, "neg": str,
             "negative": str, "negative_prompt": str}
    for field in parts[idx:]:
        if "=" not in field:
            continue
        k, v = field.split("=", 1)
        k, v = k.strip(), v.strip()
        if k == "out" or k == "output":
            kw["output_path"] = v
        elif k in short:
            key = {"w": "width", "h": "height", "neg": "negative_prompt",
                   "negative": "negative_prompt"}.get(k, k)
            kw[key] = short[k](v)
        elif k == "cfg":
            kw["guidance_scale"] = float(v)
        else:
            logging.warning("Ignoring unknown serve field: %s", field)
    return kw


def _run_serve(args) -> int:
    """Persistent mode: read prompts from stdin, reuse the cached model."""
    logging.info(
        "Serve mode: model will be kept resident and reused. "
        "Type one prompt per line, 'quit' to exit. Idle timeout via SD_IDLE_TIMEOUT."
    )
    print("READY — enter prompts as 'prompt|output.png|seed=123|steps=20|cfg=7' (or 'quit'):", flush=True)

    count = 0
    for raw in sys.stdin:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower() in ("quit", "exit", "q"):
            logging.info("Serve: received quit, shutting down")
            break
        try:
            kw = _parse_serve_line(line, args)
            logging.info("Serve: generating -> %s", kw["output_path"])
            _generate(args.arch, **kw)
            count += 1
            print(f"OK {count} -> {kw['output_path']}", flush=True)
        except Exception as exc:  # noqa: BLE001
            logging.exception("Serve: generation failed: %s", exc)
            print(f"ERROR {exc}", flush=True)

    # Release VRAM on shutdown.
    try:
        from pipeline_cache import get_cache

        get_cache().unload_all()
    except Exception:  # noqa: BLE001
        logging.exception("Serve: cache teardown failed")
    logging.info("Serve: done (%d images generated)", count)
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.serve:
        return _run_serve(args)
    return _run_one(args)


if __name__ == "__main__":
    sys.exit(main())
