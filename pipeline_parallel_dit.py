"""
pipeline_parallel_dit.py — DiT FP32 pipeline-parallel on 2× Tesla P40.

DiT-based diffusion models (PixArt-Alpha/Sigma, Sana, Stable Diffusion 3, Flux,
Lumina, …) replace the classic UNet with a Transformer backbone.  In FP32 these
easily reach **30–50 GB**, far beyond a single 24 GB P40.

This module is the DiT analogue of :mod:`pipeline_parallel_sdxl`: it splits the
transformer's block sequence in half across two GPUs via
:class:`pp_dit.PipelineParallelDiT`.  Text encoders / VAE stay on
``device_down``; only ~one hidden-state tensor crosses the PCIe boundary per
step.

Usage
-----
    from pipeline_parallel_dit import generate_dit
    generate_dit("a cat", model_path="PixArt-alpha/PixArt-XL-2-1024-MS")
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import torch

from pp_dit import PipelineParallelDiT

logger = logging.getLogger(__name__)


# Diffusers pipeline classes that use a transformer backbone (DiT).  The first
# match wins.  Add more here as diffusers grows new DiT families.
def _install_device_relay(module, target_device, return_device):
    """Make *module* (living on *target_device*) callable from *return_device*.

    Registers hooks that move the module's positional args / kwargs to
    *target_device* before the forward pass, and move its output back to
    *return_device* afterwards.  This lets a large module (e.g. T5-XXL) live on
    a different GPU than the rest of the pipeline without the caller noticing.
    """
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


_DIT_PIPELINE_CLASSES: tuple[str, ...] = (
    "PixArtAlphaPipeline",
    "PixArtSigmaPipeline",
    "SanaPipeline",
    "StableDiffusion3Pipeline",
    "FluxPipeline",
    "LuminaPipeline",
    "AuraFlowPipeline",
)


def _detect_dit_class_by_structure(model_path: str) -> Optional[str]:
    """Detect the DiT pipeline class by inspecting the model directory structure.

    When a DiT model was downloaded through a non-DiT pipeline class (e.g. the
    gated-repo fallback to ``StableDiffusionXLPipeline``), its
    ``model_index.json`` carries the WRONG ``_class_name``.  We recover the
    correct class by looking at which sub-directories exist.

    Heuristics (based on the components each diffusers DiT family ships):
        - 3 text encoders + transformer      → StableDiffusion3Pipeline
        - 2 text encoders + transformer      → FluxPipeline
        - 1 text encoder + transformer       → PixArtAlphaPipeline
          (Sana/Lumina/AuraFlow also have 1 TE; PixArt-α is the safe default)
    """
    import os as _os

    def _has(name: str) -> bool:
        sub = _os.path.join(model_path, name)
        return _os.path.isdir(sub) or _os.path.isfile(sub + ".json")

    if not _has("transformer"):
        return None  # not a DiT layout at all

    n_encoders = sum(_has(f"text_encoder{i}") for i in ("", "_2", "_3"))
    if n_encoders >= 3:
        return "StableDiffusion3Pipeline"
    if n_encoders == 2:
        return "FluxPipeline"
    # 1 text encoder — disambiguate Sana/Lumina/AuraFlow vs PixArt by config.
    try:
        import json as _json
        tc = _os.path.join(model_path, "transformer", "config.json")
        if _os.path.isfile(tc):
            with open(tc, "r", encoding="utf-8") as f:
                cfg = _json.load(f)
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
    """Import the right diffusers pipeline class for *model_path*.

    Strategy (in order):
      1. If the model dir looks like a DiT layout (has ``transformer/``), detect
         the class from the directory structure.  This is robust against a wrong
         ``_class_name`` in ``model_index.json`` (which happens when a gated DiT
         repo was downloaded via the SDXL fallback class).
      2. Otherwise trust ``model_index.json`` → ``_class_name``.
      3. Fallback to ``PixArtAlphaPipeline``.
    """
    import json
    import diffusers

    # (1) Structure-based detection takes priority for DiT layouts.
    detected = _detect_dit_class_by_structure(model_path)
    if detected:
        cls_name = detected
    else:
        # (2) Trust model_index.json.
        idx_file = os.path.join(model_path, "model_index.json")
        cls_name = "PixArtAlphaPipeline"
        if os.path.isfile(idx_file):
            try:
                with open(idx_file, "r", encoding="utf-8") as f:
                    cls_name = json.load(f).get("_class_name", cls_name)
            except Exception:
                pass
        # If model_index.json names a non-DiT class but we got here, default.
        if cls_name not in _DIT_PIPELINE_CLASSES:
            cls_name = "PixArtAlphaPipeline"

    if not hasattr(diffusers, cls_name):
        raise RuntimeError(
            f"Installed diffusers has no class {cls_name!r} (needed for "
            f"{model_path}).  Upgrade diffusers."
        )
    return getattr(diffusers, cls_name)


def _compute_split_ratio(pipe, device_down: str, device_up: str) -> float:
    """Compute the optimal transformer-block split ratio from live VRAM.

    Measures:
      - How much VRAM is already used on each GPU (by text encoders / VAE).
      - The transformer's actual parameter memory footprint.
    Then splits blocks so that both GPUs stay under ~90 % VRAM.

    Returns a float in (0, 1): fraction of blocks for *device_down*.
    """
    import torch

    dev_down = torch.device(device_down)
    dev_up = torch.device(device_up)

    def _param_bytes(module) -> int:
        return sum(p.numel() * p.element_size() for p in module.parameters())

    def _free_vram_gb(device: torch.device) -> float:
        try:
            total = torch.cuda.get_device_properties(device).total_memory
            allocated = torch.cuda.memory_allocated(device)
            return max((total - allocated) / 1e9, 0.5)
        except Exception:
            return 20.0  # fallback assumption

    # Transformer size in GB.
    transformer_gb = _param_bytes(pipe.transformer) / 1e9

    # Free VRAM on each GPU after text encoders / VAE / T5 are placed.
    free_down = _free_vram_gb(dev_down)
    free_up = _free_vram_gb(dev_up)

    # Reserve 15 % headroom on each GPU for activations during forward pass.
    usable_down = free_down * 0.85
    usable_up = free_up * 0.85

    # How many GB of transformer blocks can each GPU hold?
    total_usable = usable_down + usable_up
    if total_usable <= 0 or transformer_gb <= 0:
        logger.warning("Could not compute split ratio; defaulting to 0.5")
        return 0.5

    ratio = usable_down / total_usable
    # Clamp to reasonable bounds.
    ratio = max(0.3, min(0.8, ratio))

    logger.info(
        "Dynamic split: transformer=%.1f GB, free_down=%.1f GB, free_up=%.1f GB, "
        "ratio=%.2f (%.0f%% down / %.0f%% up)",
        transformer_gb, free_down, free_up, ratio, ratio * 100, (1 - ratio) * 100,
    )
    return ratio


def create_dit_pipeline_parallel(
    model_path: str = "PixArt-alpha/PixArt-XL-2-1024-MS",
    device_down: str = "cuda:0",
    device_up: str = "cuda:1",
    use_fp32: bool = True,
    scheduler: str = "default",
    token: Optional[str] = None,
):
    """Build a 2-GPU pipeline-parallel DiT pipeline.

    Parameters
    ----------
    model_path : str
        Local path or HuggingFace repo ID of a DiT diffusers model.
    device_down, device_up : str
        CUDA devices for stage 0 / stage 1.
    use_fp32 : bool
        If True (recommended on P40), upcast weights to FP32.
    scheduler : str
        ``"default"``, ``"ddim"``, ``"euler"``, or ``"dpmpp_2m"``.
    """
    from model_resolver import resolve_model_path

    model_path = resolve_model_path(model_path, token=token)
    dtype = torch.float32 if use_fp32 else torch.float16
    logger.info("Loading DiT from %s (%s)", model_path, "FP32" if use_fp32 else "FP16")

    pipe_cls = _load_dit_pipeline_class(model_path)
    pipe = pipe_cls.from_pretrained(model_path, torch_dtype=dtype, token=token)

    # Place text encoders + VAE on stage 0 (device_down).  These are small
    # relative to a 30–50 GB transformer.
    # Balance components between the two GPUs to avoid OOM on device_down.
    # T5-XXL (text_encoder_3) is the heaviest non-transformer component and
    # would OOM device_down.  Keep it on device_up with a transparent device
    # relay (forward hooks): inputs move to device_up before the forward, the
    # output moves back to device_down afterwards.
    for attr in ("text_encoder", "text_encoder_2", "vae"):
        sub = getattr(pipe, attr, None)
        if sub is not None:
            sub.to(device_down)

    te3 = getattr(pipe, "text_encoder_3", None)
    if te3 is not None:
        te3.to(device_up)
        _install_device_relay(te3, torch.device(device_up), torch.device(device_down))
        logger.info(
            "text_encoder_3 placed on %s with device relay back to %s",
            device_up, device_down,
        )

    if getattr(pipe, "safety_checker", None) is not None:
        pipe.safety_checker = None

    # Scheduler (best-effort — not all schedulers exist for every pipeline).
    if scheduler != "default":
        from diffusers import (
            DDIMScheduler,
            DPMSolverMultistepScheduler,
            EulerDiscreteScheduler,
        )
        try:
            if scheduler == "ddim":
                pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            elif scheduler == "euler":
                pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            elif scheduler == "dpmpp_2m":
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe.scheduler.config
                )
        except Exception:
            logger.warning("Could not switch scheduler to %s; keeping default", scheduler)

    # Wrap the transformer.
    logger.info(
        "Wrapping %s with PipelineParallelDiT (%s + %s)",
        type(pipe.transformer).__name__, device_down, device_up,
    )
    # Dynamically compute split_ratio based on actual model size and free VRAM.
    # SD3.5-large transformer is ~32 GB in FP32 — a naive 50/50 split OOMs both
    # GPUs because device_up also carries T5-XXL (~10 GB) and device_down
    # carries CLIP-L + CLIP-G + VAE (~2 GB).  We measure the transformer's real
    # footprint and the free VRAM on each GPU, then split proportionally.
    split_ratio = _compute_split_ratio(pipe, device_down, device_up)

    pipe.transformer = PipelineParallelDiT(
        pipe.transformer, device_down=device_down, device_up=device_up,
        split_ratio=split_ratio,
    )

    # The pipeline must not attempt its own device management.
    pipe.enable_model_cpu_offload = lambda: None  # type: ignore[method-assign]
    pipe.enable_sequential_cpu_offload = lambda: None  # type: ignore[method-assign]
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
):
    """Generate a single image with a DiT model across 2 GPUs (FP32).

    Uses the same keep-alive cache as the SDXL path (see
    :mod:`pipeline_cache`).  Pass ``force_unload=True`` to free VRAM at once.
    """
    from pipeline_cache import cached_dit_pipeline, get_cache

    cache = get_cache()
    pipe, entry = cached_dit_pipeline(
        model_path=model_path,
        device_down=device_down,
        device_up=device_up,
        use_fp32=use_fp32,
        scheduler=scheduler,
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

    logger.info(
        "Generating DiT %dx%d, %d steps, CFG %.1f", width, height, steps, guidance_scale
    )
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
            "dit",
            os.path.abspath(resolved),
            str(device_down),
            str(device_up),
            bool(use_fp32),
            str(scheduler),
        )
        if cache.release(key):
            logger.info("force_unload: pipeline freed")
    return image


class _NullLock:
    """Context manager that does nothing (used when caching is disabled)."""

    def __enter__(self) -> "_NullLock":
        return self

    def __exit__(self, *exc) -> None:
        pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    generate_dit(
        prompt=(
            "A serene Japanese garden with a koi pond, cherry blossoms falling, "
            "golden hour light, photorealistic, 8k, detailed"
        ),
        negative_prompt="blurry, low quality, distorted, ugly",
        steps=20,
        seed=42,
    )