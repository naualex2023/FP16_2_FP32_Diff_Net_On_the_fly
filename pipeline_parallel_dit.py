"""
pipeline_parallel_dit.py — DiT FP32 pipeline-parallel on N× Tesla P40.

DiT-based diffusion models (PixArt, Sana, Stable Diffusion 3, Flux, Lumina, …)
replace the classic UNet with a Transformer backbone.  In FP32 these easily
reach 30-60 GB, far beyond a single 24 GB P40.

This module **auto-detects** how many GPUs are needed based on the model's
parameter count and available VRAM, then splits the transformer blocks across
that many GPUs via :class:`pp_dit.PipelineParallelDiT`.

Usage
-----
    from pipeline_parallel_dit import generate_dit
    generate_dit("a cat", model_path="PixArt-alpha/PixArt-XL-2-1024-MS")
"""

from __future__ import annotations

import logging
import os
import time
from typing import List, Optional

import torch

from pp_dit import PipelineParallelDiT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model size estimation
# ---------------------------------------------------------------------------

def _estimate_model_size_gb(model_path: str) -> Optional[dict]:
    """Estimate component sizes (GB) from config.json files on disk.

    Returns a dict like ``{"transformer": 32.0, "text_encoder_3": 18.8, ...}``
    in **FP32** bytes, or ``None`` if it cannot be determined.
    """
    import json

    def _read_config(subdir: str) -> dict:
        cfg_path = os.path.join(model_path, subdir, "config.json")
        if not os.path.isfile(cfg_path):
            return {}
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _estimate_params(config: dict) -> float:
        """Rough parameter count estimate from common config fields (in billions)."""
        # Transformer models: estimate from hidden_size, num_layers, etc.
        hidden = config.get("hidden_size") or config.get("num_attention_heads", 0) * (
            config.get("attention_head_dim", 0) or config.get("dim", 0)
        )
        layers = config.get("num_layers") or config.get("num_hidden_layers", 0)
        vocab = config.get("vocab_size", 0)
        intermediate = config.get("intermediate_size") or hidden * 4

        if hidden and layers:
            # Very rough: each transformer layer ≈ 12 * hidden^2 params
            layer_params = 12 * hidden * hidden
            total = layer_params * layers
            if vocab:
                total += vocab * hidden * 2  # embeddings + LM head
            return total / 1e9
        return 0.0

    sizes = {}
    for comp in ("transformer", "text_encoder", "text_encoder_2", "text_encoder_3", "vae"):
        cfg = _read_config(comp)
        if cfg:
            params_bn = _estimate_params(cfg)
            if params_bn > 0:
                sizes[comp] = params_bn * 4.0  # FP32 = 4 bytes/param → GB

    return sizes if sizes else None


def _auto_detect_num_gpus(model_path: str, use_fp32: bool = True) -> tuple[int, float]:
    """Determine how many GPUs are needed for this model.

    Returns (num_gpus, total_model_gb).
    """
    if not torch.cuda.is_available():
        return 1, 0.0

    n_available = torch.cuda.device_count()
    vram_per_gpu = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB

    sizes = _estimate_model_size_gb(model_path)
    if not sizes:
        logger.warning("Could not estimate model size; defaulting to 2 GPUs")
        return min(2, n_available), 0.0

    byte_factor = 1.0 if use_fp32 else 0.5  # FP16 halves memory
    total_gb = sum(sizes.values()) * byte_factor

    # Reserve ~20% for activations.
    usable_per_gpu = vram_per_gpu * 0.75

    needed = max(1, int(-(-total_gb // usable_per_gpu)))  # ceil division
    needed = min(needed, n_available)

    logger.info(
        "Auto-detect: model components (FP%d) ≈ %.1f GB total, VRAM/GPU=%.1f GB "
        "→ need %d GPU(s)",
        32 if use_fp32 else 16, total_gb, vram_per_gpu, needed,
    )
    return needed, total_gb


# ---------------------------------------------------------------------------
# Helpers for component placement and device relays
# ---------------------------------------------------------------------------

def _install_device_relay(module, target_device, return_device):
    """Make *module* callable from *return_device* while living on *target_device*."""
    def _move_any(obj, device):
        if obj is None:
            return None
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, dict):
            return {k: _move_any(v, device) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            moved = [_move_any(x, device) for x in obj]
            return type(obj)(moved)
        if hasattr(obj, "__dict__"):
            for k, v in list(vars(obj).items()):
                if isinstance(v, torch.Tensor):
                    setattr(obj, k, v.to(device))
            return obj
        return obj

    def _pre(_m, args, kwargs):
        new_args = tuple(_move_any(a, target_device) for a in args)
        new_kwargs = {k: _move_any(v, target_device) for k, v in kwargs.items()}
        return new_args, new_kwargs

    def _post(_m, _args, output):
        return _move_any(output, return_device)

    module.register_forward_pre_hook(_pre, with_kwargs=True)
    module.register_forward_hook(_post)


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

# DiT pipeline classes for structure-based detection.
_DIT_PIPELINE_CLASSES: tuple[str, ...] = (
    "PixArtAlphaPipeline", "PixArtSigmaPipeline", "SanaPipeline",
    "StableDiffusion3Pipeline", "FluxPipeline", "LuminaPipeline", "AuraFlowPipeline",
)


def _detect_dit_class_by_structure(model_path: str) -> Optional[str]:
    """Detect DiT pipeline class from model directory structure."""
    def _has(name):
        sub = os.path.join(model_path, name)
        return os.path.isdir(sub) or os.path.isfile(sub + ".json")

    if not _has("transformer"):
        return None

    n = sum(_has(f"text_encoder{i}") for i in ("", "_2", "_3"))
    if n >= 3:
        return "StableDiffusion3Pipeline"
    if n == 2:
        return "FluxPipeline"
    try:
        import json
        tc = os.path.join(model_path, "transformer", "config.json")
        if os.path.isfile(tc):
            with open(tc, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            arch = (cfg.get("arch") or "").lower()
            if "sana" in arch:
                return "SanaPipeline"
            if "lumina" in arch:
                return "LuminaPipeline"
            if "auraflow" in arch or "aura_flow" in arch:
                return "AuraFlowPipeline"
    except Exception:
        pass
    return "PixArtAlphaPipeline"


def _load_dit_pipeline_class(model_path: str):
    """Import the right diffusers pipeline class for *model_path*."""
    import json
    import diffusers

    detected = _detect_dit_class_by_structure(model_path)
    if detected:
        cls_name = detected
    else:
        idx_file = os.path.join(model_path, "model_index.json")
        cls_name = "PixArtAlphaPipeline"
        if os.path.isfile(idx_file):
            try:
                with open(idx_file, "r", encoding="utf-8") as f:
                    cls_name = json.load(f).get("_class_name", cls_name)
            except Exception:
                pass
        if cls_name not in _DIT_PIPELINE_CLASSES:
            cls_name = "PixArtAlphaPipeline"

    if not hasattr(diffusers, cls_name):
        raise RuntimeError(
            f"Installed diffusers has no class {cls_name!r}. Upgrade diffusers."
        )
    return getattr(diffusers, cls_name)


def create_dit_pipeline_parallel(
    model_path: str = "PixArt-alpha/PixArt-XL-2-1024-MS",
    device_down: str = "cuda:0",
    device_up: str = "cuda:1",
    use_fp32: bool = True,
    scheduler: str = "default",
    token: Optional[str] = None,
    num_gpus: Optional[int] = None,
):
    """Build an N-GPU pipeline-parallel DiT pipeline.

    Parameters
    ----------
    model_path : str
        Local path or HuggingFace repo ID of a DiT model.
    device_down : str
        First GPU (stage 0). Additional GPUs are auto-selected sequentially.
    device_up : str
        Legacy: second GPU. Ignored when ``num_gpus > 2``.
    use_fp32 : bool
        If True (recommended on P40), upcast weights to FP32.
    scheduler : str
        Scheduler preset.
    token : str, optional
        HuggingFace access token for gated repos.
    num_gpus : int, optional
        Number of GPUs to use. If ``None``, auto-detected from model size.
    """
    from model_resolver import resolve_model_path

    model_path = resolve_model_path(model_path, token=token)
    dtype = torch.float32 if use_fp32 else torch.float16
    logger.info("Loading DiT from %s (%s)", model_path, "FP32" if use_fp32 else "FP16")

    # ---- auto-detect number of GPUs ------------------------------------
    if num_gpus is None:
        num_gpus, model_gb = _auto_detect_num_gpus(model_path, use_fp32)
    else:
        _, model_gb = _auto_detect_num_gpus(model_path, use_fp32)

    # ---- build device list ---------------------------------------------
    start_idx = int(device_down.split(":")[-1]) if ":" in device_down else 0
    devices = [f"cuda:{start_idx + i}" for i in range(num_gpus)]

    logger.info(
        "Using %d GPU(s): %s (model ≈ %.1f GB in %s)",
        num_gpus, devices, model_gb, "FP32" if use_fp32 else "FP16",
    )

    pipe_cls = _load_dit_pipeline_class(model_path)
    pipe = pipe_cls.from_pretrained(model_path, torch_dtype=dtype, token=token)

    # ---- place components across GPUs ----------------------------------
    # Strategy: text encoders on device[0], but T5-XXL (if present) on the
    # LAST device (it has the fewest transformer blocks → more room).
    # VAE on device[0] (needed at the end for decode).
    home_dev = torch.device(devices[0])
    last_dev = torch.device(devices[-1])

    for attr in ("text_encoder", "text_encoder_2", "vae"):
        sub = getattr(pipe, attr, None)
        if sub is not None:
            sub.to(home_dev)

    # T5-XXL on the last GPU with relay hooks.
    te3 = getattr(pipe, "text_encoder_3", None)
    if te3 is not None:
        if num_gpus > 1:
            te3.to(last_dev)
            _install_device_relay(te3, last_dev, home_dev)
            logger.info("text_encoder_3 on %s with relay → %s", last_dev, home_dev)
        else:
            te3.to(home_dev)

    if getattr(pipe, "safety_checker", None) is not None:
        pipe.safety_checker = None

    # Scheduler.
    if scheduler != "default":
        from diffusers import DDIMScheduler, DPMSolverMultistepScheduler, EulerDiscreteScheduler
        try:
            if scheduler == "ddim":
                pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            elif scheduler == "euler":
                pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            elif scheduler == "dpmpp_2m":
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        except Exception:
            logger.warning("Could not switch scheduler to %s; keeping default", scheduler)

    # ---- wrap transformer across N GPUs --------------------------------
    logger.info(
        "Wrapping %s with PipelineParallelDiT (%d GPUs: %s)",
        type(pipe.transformer).__name__, num_gpus, devices,
    )
    pipe.transformer = PipelineParallelDiT(
        pipe.transformer, devices=devices,
    )

    pipe.enable_model_cpu_offload = lambda: None
    pipe.enable_sequential_cpu_offload = lambda: None
    return pipe


def generate_dit(
    prompt: str,
    negative_prompt: str = "",
    model_path: str = "PixArt-alpha/PixArt-XL-2-1024-MS",
    device_down: str = "cuda:0",
    device_up: str = "cuda:1",
    steps: int = 20,
    width: int = 1024,
    height: int = 1024,
    seed: int = 42,
    guidance_scale: float = 4.5,
    use_fp32: bool = True,
    output_path: str = "output_dit_pp.png",
    scheduler: str = "default",
    force_unload: bool = False,
    callback: Optional[callable] = None,
    callback_kwargs: Optional[dict] = None,
    num_gpus: Optional[int] = None,
):
    """Generate a single image with a DiT model across N GPUs."""
    from pipeline_cache import cached_dit_pipeline, get_cache

    cache = get_cache()
    pipe, entry = cached_dit_pipeline(
        model_path=model_path,
        device_down=device_down,
        device_up=device_up,
        use_fp32=use_fp32,
        scheduler=scheduler,
        num_gpus=num_gpus,
        cache=cache,
    )

    call_kwargs = dict(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )
    if negative_prompt:
        call_kwargs["negative_prompt"] = negative_prompt

    if callback is not None:
        from callback_utils import make_progress_callback
        call_kwargs["callback_on_step_end"] = make_progress_callback(callback)
        call_kwargs["callback_on_step_end_inputs"] = callback_kwargs or []

    logger.info("Generating DiT %dx%d, %d steps, CFG %.1f", width, height, steps, guidance_scale)
    entry_lock = entry.lock if entry is not None else _NullLock()
    with entry_lock:
        t0 = time.perf_counter()
        image = pipe(**call_kwargs).images[0]
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    image.save(output_path)
    logger.info("Saved %s in %.1fs (%.2fs/step)", output_path, dt, dt / steps)

    if force_unload:
        from model_resolver import resolve_model_path
        resolved = resolve_model_path(model_path)
        key = (
            "dit", os.path.abspath(resolved), str(device_down), str(device_up),
            bool(use_fp32), str(scheduler), str(num_gpus),
        )
        if cache.release(key):
            logger.info("force_unload: pipeline freed")
    return image


class _NullLock:
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    generate_dit(
        prompt="A serene Japanese garden with a koi pond, golden hour, 8k",
        negative_prompt="blurry, low quality, distorted, ugly",
        steps=20, seed=42,
    )
