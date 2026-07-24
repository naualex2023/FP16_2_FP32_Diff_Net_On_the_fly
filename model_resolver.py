#!/usr/bin/env python3
"""
model_resolver.py — Resolve model specs (local path OR HuggingFace repo ID).

This module removes the hardcoding that previously limited the system to three
preset models (sd15, sdxl, sdxl-turbo).  Any ``model_path`` argument in the
pipeline / API / CLI layer is now passed through :func:`resolve_model_path`,
which transparently handles two cases:

1. **Local directory** — ``./models/sdxl-base-fp16`` or ``/abs/path/to/model``
   → returned as-is (after an existence check).

2. **HuggingFace repo ID** — ``stabilityai/stable-diffusion-xl-base-1.0`` →
   downloaded once into ``./models/<sanitised-name>`` (or a custom
   ``SD_MODELS_DIR``) and the local path is returned.  Subsequent references
   hit the local copy without re-downloading.

Architecture (``sdxl`` / ``sd15`` / ``dit``) auto-detection is also provided
via :func:`infer_arch`, which reads ``model_index.json`` from the resolved
path.

Public API
----------
* :func:`is_hf_repo_id` — heuristic: does this string look like a repo ID?
* :func:`resolve_model_path` — local path or repo ID → local directory path.
* :func:`infer_arch` — detect ``sdxl`` / ``sd15`` / ``dit`` from a model dir.
* :func:`download_hf_model` — explicit download of any HF repo.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.environ.get("SD_MODELS_DIR", os.path.join(BASE_DIR, "models"))

# A repo ID is ``org/name`` (or just ``name``) with no path separators beyond
# the single slash.  We reject anything that looks like a filesystem path
# (starts with ``.``, ``/``, ``~``, or contains backslashes on Windows).
_HF_REPO_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]*/[A-Za-z0-9][A-Za-z0-9._\-]*$")

# Mapping from diffusers pipeline class name → our architecture tag.
_CLASS_TO_ARCH = {
    "StableDiffusionXLPipeline": "sdxl",
    "StableDiffusionXLImg2ImgPipeline": "sdxl",
    "StableDiffusionXLInpaintPipeline": "sdxl",
    "StableDiffusionPipeline": "sd15",
    "StableDiffusionImg2ImgPipeline": "sd15",
    "StableDiffusionInpaintPipeline": "sd15",
    # DiT (Transformer backbone) pipelines — 30–50 GB in FP32, split across two
    # GPUs via pp_dit.PipelineParallelDiT instead of the UNet wrapper.
    "PixArtAlphaPipeline": "dit",
    "PixArtSigmaPipeline": "dit",
    "SanaPipeline": "dit",
    "StableDiffusion3Pipeline": "dit",
    "StableDiffusion3Img2ImgPipeline": "dit",
    "FluxPipeline": "dit",
    "FluxImg2ImgPipeline": "dit",
    "LuminaPipeline": "dit",
    "AuraFlowPipeline": "dit",
}

# Known repos for quick architecture lookup (avoids reading model_index.json
# before a download when the repo is well-known).
_KNOWN_ARCH_HINTS = {
    "sdxl": ("sdxl", "stable-diffusion-xl", "sd-xl", "sdxl-turbo", "juggernaut-xl"),
    "sd15": ("stable-diffusion-v1-5", "sd-v1-5", "sd15", "dreamshaper", "deliberate"),
    "dit": (
        "pixart", "sana", "stable-diffusion-3", "sd3", "flux", "lumina",
        "auraflow",
    ),
}


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def is_hf_repo_id(s: str) -> bool:
    """Return ``True`` if *s* looks like a HuggingFace repo identifier.

    A repo ID has the form ``org/name`` or ``name`` and does **not** look like
    a filesystem path.  If *s* is an existing local path it is never treated
    as a repo ID.
    """
    if not s or not isinstance(s, str):
        return False
    s = s.strip()
    # Existing local path → definitely not a repo ID.
    if os.path.exists(s):
        return False
    # Filesystem-path indicators.
    if s.startswith(("./", "/", "~", "../", ".\\")):
        return False
    # ``org/name`` pattern (exactly one slash, valid characters).
    if "/" in s:
        parts = s.split("/")
        if len(parts) == 2 and all(parts):
            return bool(_HF_REPO_PATTERN.match(s))
    return False


def is_local_path(s: str) -> bool:
    """Return ``True`` if *s* is an existing local model directory."""
    return bool(s) and os.path.isdir(s)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _sanitise_repo_id(repo_id: str) -> str:
    """Convert a repo ID into a safe directory name.

    ``stabilityai/stable-diffusion-xl-base-1.0`` →
    ``stabilityai--stable-diffusion-xl-base-1.0``
    """
    return repo_id.replace("/", "--")


def resolve_model_path(
    model_spec: str,
    models_dir: Optional[str] = None,
    dtype: str = "float16",
    pipeline_class: Optional[str] = None,
    force_download: bool = False,
) -> str:
    """Resolve *model_spec* to a local model directory.

    Parameters
    ----------
    model_spec : str
        Either a local directory path or a HuggingFace repo ID
        (``org/name``).
    models_dir : str, optional
        Base directory for downloaded models.  Defaults to ``SD_MODELS_DIR``
        env var or ``./models``.
    dtype : str
        Torch dtype for the download (``"float16"`` keeps weights compact;
        they are upcast to FP32 at load time by the pipeline-parallel code).
    pipeline_class : str, optional
        Diffusers pipeline class to use (e.g. ``StableDiffusionXLPipeline``).
        If omitted, a generic auto-detection is attempted.
    force_download : bool
        Re-download even if a local copy exists.

    Returns
    -------
    str
        Absolute path to a local model directory.
    """
    if not model_spec:
        raise ValueError("model_spec is empty")

    # Already a local path?
    if os.path.isdir(model_spec) and not force_download:
        return os.path.abspath(model_spec)

    # HF repo ID → download (or reuse local copy).
    if is_hf_repo_id(model_spec):
        return download_hf_model(
            repo_id=model_spec,
            models_dir=models_dir,
            dtype=dtype,
            pipeline_class=pipeline_class,
            force=force_download,
        )

    # Not a directory and not a recognised repo ID — let diffusers handle it
    # (it may be a repo ID without a slash, or the path just doesn't exist
    # yet and the caller wants the error to surface from from_pretrained).
    logger.warning(
        "model_spec %r is neither an existing path nor a recognised HF repo ID; "
        "passing through to diffusers as-is",
        model_spec,
    )
    return model_spec


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_hf_model(
    repo_id: str,
    models_dir: Optional[str] = None,
    dtype: str = "float16",
    pipeline_class: Optional[str] = None,
    force: bool = False,
) -> str:
    """Download a HuggingFace diffusers repo into a local directory.

    Parameters
    ----------
    repo_id : str
        HuggingFace repo identifier, e.g. ``stabilityai/sdxl-turbo``.
    models_dir : str, optional
        Destination base directory (default ``SD_MODELS_DIR`` / ``./models``).
    dtype : str
        Storage dtype for the on-disk weights: ``"float16"`` (default,
        compact) or ``"float32"``.  **The pipeline-parallel code upcasts
        FP16 weights to FP32 in memory at load time** via
        ``from_pretrained(..., torch_dtype=torch.float32)``, so FP16
        storage is recommended to save disk space.
    pipeline_class : str, optional
        Diffusers pipeline class.  If ``None``, auto-detect via
        :func:`_detect_pipeline_class`.
    force : bool
        Re-download even if the local copy exists.

    Returns
    -------
    str
        Absolute path to the downloaded model directory.
    """
    import torch
    import diffusers

    models_dir = models_dir or MODELS_DIR
    dst_name = _sanitise_repo_id(repo_id)
    dst = os.path.join(models_dir, dst_name)

    if os.path.isdir(dst) and os.listdir(dst) and not force:
        logger.info("Model %s already present at %s (use force=True to re-download)", repo_id, dst)
        return os.path.abspath(dst)

    os.makedirs(models_dir, exist_ok=True)
    logger.info("Downloading %s → %s (%s)", repo_id, dst, dtype)

    cls_name = pipeline_class or _detect_pipeline_class(repo_id)
    cls = getattr(diffusers, cls_name)
    torch_dtype = getattr(torch, dtype)
    pipe = cls.from_pretrained(repo_id, torch_dtype=torch_dtype)
    pipe.save_pretrained(dst)
    logger.info("Downloaded %s → %s", repo_id, dst)
    return os.path.abspath(dst)


def _detect_pipeline_class(repo_id: str) -> str:
    """Best-effort detection of the diffusers pipeline class for *repo_id*.

    Checks the repo's ``model_index.json`` for a ``_class_name`` field.
    Falls back to keyword heuristics on the repo ID, and ultimately to
    ``StableDiffusionXLPipeline``.
    """
    # Try reading model_index.json from the HF API (without full download).
    try:
        from huggingface_hub import hf_hub_download

        idx_path = hf_hub_download(repo_id, "model_index.json")
        with open(idx_path, "r", encoding="utf-8") as f:
            idx = json.load(f)
        cls_name = idx.get("_class_name", "")
        if cls_name in _CLASS_TO_ARCH:
            return cls_name
        # Diffusers may store the class name directly.
        if cls_name.startswith("StableDiffusion"):
            return cls_name
        # DiT pipeline classes are returned verbatim if diffusers knows them.
        if cls_name in {
            "PixArtAlphaPipeline", "PixArtSigmaPipeline", "SanaPipeline",
            "StableDiffusion3Pipeline", "FluxPipeline", "LuminaPipeline",
            "AuraFlowPipeline",
        }:
            return cls_name
    except Exception:
        pass  # not available / offline — fall through to heuristics

    # Keyword heuristics on the repo name (case-insensitive).
    lower = repo_id.lower()
    if any(h in lower for h in _KNOWN_ARCH_HINTS["sdxl"]):
        return "StableDiffusionXLPipeline"
    if any(h in lower for h in _KNOWN_ARCH_HINTS["sd15"]):
        return "StableDiffusionPipeline"
    # DiT families — pick the matching pipeline class by keyword.
    if "pixart" in lower and "sigma" in lower:
        return "PixArtSigmaPipeline"
    if "pixart" in lower:
        return "PixArtAlphaPipeline"
    if "sana" in lower:
        return "SanaPipeline"
    if "stable-diffusion-3" in lower or "sd3" in lower:
        return "StableDiffusion3Pipeline"
    if "flux" in lower:
        return "FluxPipeline"
    if "lumina" in lower:
        return "LuminaPipeline"
    if "auraflow" in lower:
        return "AuraFlowPipeline"

    logger.info("Could not auto-detect pipeline class for %s; defaulting to SDXL", repo_id)
    return "StableDiffusionXLPipeline"


# ---------------------------------------------------------------------------
# Architecture inference
# ---------------------------------------------------------------------------

def infer_arch(model_path: str) -> str:
    """Detect the architecture tag (``sdxl`` / ``sd15`` / ``dit``) from a dir.

    Reads ``model_index.json`` and maps the pipeline class.  Falls back to
    ``"sdxl"`` if detection fails.
    """
    idx_file = os.path.join(model_path, "model_index.json")
    if os.path.isfile(idx_file):
        try:
            with open(idx_file, "r", encoding="utf-8") as f:
                idx = json.load(f)
            cls_name = idx.get("_class_name", "")
            arch = _CLASS_TO_ARCH.get(cls_name)
            if arch:
                return arch
            if "XL" in cls_name:
                return "sdxl"
            if "StableDiffusionPipeline" in cls_name:
                return "sd15"
        except Exception:
            pass

    # Heuristic on directory name.
    lower = (model_path or "").lower()
    if any(h in lower for h in _KNOWN_ARCH_HINTS["sd15"]):
        return "sd15"
    if any(h in lower for h in _KNOWN_ARCH_HINTS["dit"]):
        return "dit"
    return "sdxl"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Download any HuggingFace diffusers model.")
    ap.add_argument("--repo", required=True, help="HuggingFace repo ID, e.g. stabilityai/sdxl-turbo")
    ap.add_argument("--class", dest="pipeline_class", default=None, help="Pipeline class (auto-detect if omitted)")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    ap.add_argument("--models-dir", default=None, help="Destination base dir (default: ./models)")
    ap.add_argument("--force", action="store_true", help="Re-download even if present")
    args = ap.parse_args()

    path = download_hf_model(
        repo_id=args.repo,
        models_dir=args.models_dir,
        dtype=args.dtype,
        pipeline_class=args.pipeline_class,
        force=args.force,
    )
    arch = infer_arch(path)
    print(f"OK: {args.repo} → {path}  (arch={arch})")