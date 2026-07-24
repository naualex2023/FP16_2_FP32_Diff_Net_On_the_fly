# FP16 -> FP32 Diffusion Net (On-the-fly) — Pipeline-Parallel FP32 on 4x Tesla P40

Run **SDXL / SD 1.5 / DiT / GGUF** models in pure FP32 across 4 NVIDIA Tesla P40
GPUs (24 GB VRAM, Pascal, compute 6.1) using **pipeline parallelism** — splitting
the backbone (UNet *or* Transformer) across GPUs and transferring only
activations, not weights.

> **Why FP32 on P40?** The Tesla P40 is Pascal (2016): FP16 compute is broken
> (0.18 TFLOPS, 1/64 of FP32). SDXL in FP32 needs ~25 GB, and a large DiT
> (PixArt/Sana/SD3/Flux) in FP32 can reach **30-60 GB** — both exceed one card's
> 24 GB. Pipeline parallelism across 2-4 GPUs solves this.

---

## Project layout

| File | Purpose |
|------|---------|
| `pp_unet.py` | **Core (UNet)** — `PipelineParallelUNet`: splits `UNet2DConditionModel` across two GPUs (down blocks <-> mid+up blocks). |
| `pp_dit.py` | **Core (DiT)** — `PipelineParallelDiT`: splits a Transformer backbone across **N GPUs** (1-4) by cutting the `transformer_blocks` sequence into chunks. Supports custom `chunk_sizes` for budget-aware placement. Hook-based; numerically identical to single-GPU. |
| `pp_t5.py` | **Core (T5)** — `PipelineParallelT5`: shards the T5-XXL encoder (`text_encoder_3`) across N GPUs when it exceeds one GPU's budget (e.g. SD3.5's 23 GB T5). Same hook-based technique as `pp_dit`. |
| `dit_size_estimator.py` | **Size estimator** (pure-Python, no torch) — measures on-disk checkpoint files per component, scales FP16->FP32. Replaces the old inaccurate shape heuristic that under-counted SD3.5 by 4x. |
| `dit_placement_planner.py` | **Placement planner** (pure-Python, no torch) — computes a budget-aware plan: small components on GPU 0, T5-XXL sharded if too large, transformer blocks distributed proportionally to remaining VRAM. |
| `pipeline_parallel_sdxl.py` | **UNet use case** — SDXL FP32 on 2 GPUs. LoRA, schedulers, Turbo. Routes through keep-alive cache. |
| `pipeline_parallel_dit.py` | **DiT use case** — PixArt/Sana/SD3/Flux FP32 on **N GPUs** (auto-detected). Measures model size on disk, computes a budget-aware placement plan (shards T5 + transformer), logs the allocation table. Routes through keep-alive cache. |
| `pipeline_gguf.py` | **GGUF quantized** — loads pre-downloaded GGUF models (Q4/Q8) on a **single GPU** via `GGUFQuantizationConfig`. SD3.5-large Q4 ~9 GB on one P40. Works in Generate/Quadro/Batch. |
| `pipeline_parallel_sd15.py` | SD 1.5 on 2 GPUs (educational; fits on 1 P40). |
| `pipeline_single_gpu.py` | Single-GPU pipeline (no split). Used by Quadro mode (4 images x 4 GPUs). |
| `pipeline_cache.py` | **Keep-alive cache** — keeps loaded pipelines resident across generations; frees after idle timeout. Supports all arch types (sdxl/sd15/dit/gguf). |
| `api_server.py` | **FastAPI server** — exposes Generate/Twin/Quadro/Batch over HTTP + SSE. Auto-routes by `arch`: sdxl/sd15 (UNet split), dit (N-GPU auto), gguf (1 GPU). |
| `model_resolver.py` | Generic model loader — resolves local path *or* HF repo ID, auto-detects architecture (sdxl/sd15/dit), supports gated repos via `--hf-token`. |
| `download_models.py` | Download preset or arbitrary HF models into `./models/`. Purges HF cache after save (no duplicates). |
| `clear_cache.py` | Interactive HuggingFace cache cleaner with confirmation. Lists models with sizes, selective or bulk deletion. |
| `fix_dit_model_index.py` | Fixes `_class_name` in `model_index.json` for DiT models that were downloaded with the wrong pipeline class. Detects correct class by directory structure (transformer/ + text_encoder count). |
| `generate_one.py` | CLI for single-image generation. `--serve` keeps model resident. |
| `test_pp_unet.py` | Tests: PP UNet output == reference (numerically identical). |
| `test_pp_dit.py` | Tests: PP DiT output == reference, split count, stage placement. Run with `py_test test_pp_dit.py -v`. |
| `test_dit_size_estimator.py` | Tests: size estimator (file measurement, dtype scaling, SD3.5 regression). Pure-Python, no torch. |
| `test_dit_placement_planner.py` | Tests: placement planner (T5 sharding, proportional transformer split, error handling). Pure-Python, no torch. |
| `requirements.txt` | Python dependencies (includes `gguf>=0.10.0`). |

---

## Architecture modes

### 1. UNet (SDXL / SD 1.5) — 2-GPU pipeline parallel
Splits the UNet: down blocks on GPU 0, mid+up on GPU 1. ~55 MB activations cross
per step. Twin mode runs two such pipelines (4 GPUs) for 2x throughput.

### 2. DiT (PixArt/Sana/SD3/Flux) — N-GPU pipeline parallel (auto)
DiT models use a Transformer backbone (30-60 GB in FP32). The system
**auto-detects** how many GPUs are needed by measuring the actual checkpoint
files on disk, then computes a **budget-aware placement plan**:

```
Model size (float32): text_encoder=0.5, text_encoder_2=2.8, text_encoder_3=23.1,
                      transformer=32.6, vae=0.3 — total 59.3 GB
Auto-detect: model ~ 59.3 GB total (FP32), VRAM/GPU=24.0 GB -> need 4 GPU(s)
Placement plan (24 GB/GPU, 3 GB margin):
  cuda:0: text_encoder, text_encoder_2, vae, transformer[N blocks] = ~15 GB
  cuda:1: transformer[N blocks]                                     = ~11 GB
  cuda:2: t5[12 blocks], transformer[N blocks]                     = ~14 GB
  cuda:3: t5[12 blocks]                                             = ~12 GB
```

**Key placement rules:**
- Small components (CLIP text encoders, VAE) go on GPU 0 (home).
- **T5-XXL is sharded across multiple GPUs** when it exceeds one GPU's budget
  (e.g. SD3.5's 23 GB T5 splits 12+12 blocks across 2 GPUs via `PipelineParallelT5`).
- Transformer blocks are distributed **proportionally to remaining VRAM** on
  each GPU, so no single GPU overflows.

Available in **Generate** mode.

### 3. GGUF (quantized) — single GPU
Pre-quantized GGUF models (Q4/Q8) fit on a single P40. For example,
SD3.5-large Q4_K_M: transformer ~5 GB + T5 ~3 GB + CLIP ~1 GB = ~9 GB total.
Available in **Generate**, **Quadro** (4 images), and **Batch** modes.

---

## Quick start

```bash
# 1. Environment
python3 -m venv ~/diffusion-env
source ~/diffusion-env/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. Download models
python download_models.py --model sdxl
python download_models.py --repo stabilityai/stable-diffusion-3.5-large --hf-token hf_xxx

# 3. Run tests
python -m pytest test_pp_unet.py test_pp_dit.py test_pipeline_cache.py -v
python -m pytest test_dit_size_estimator.py test_dit_placement_planner.py -v  # pure-Python, no torch

# 4. Start the web server
python api_server.py
# Open http://localhost:8765
```

---

## API usage

### Generate (single image)

```python
# SDXL (UNet, 2 GPUs)
from pipeline_parallel_sdxl import generate_sdxl
generate_sdxl("a cat", model_path="./models/sdxl-base-fp16")

# DiT (auto N GPUs, T5 + transformer sharded)
from pipeline_parallel_dit import generate_dit
generate_dit("a cat", model_path="./models/stabilityai--stable-diffusion-3.5-large")

# GGUF (1 GPU)
from pipeline_gguf import generate_gguf
generate_gguf("a cat", model_path="./models/sd35-large-Q4_K_M.gguf")
```

Via web API: `POST /api/generate` with `arch` field:
- `"sdxl"` / `"sd15"` — UNet pipeline parallel (2 GPUs)
- `"dit"` — DiT auto N-GPU pipeline parallel (T5 + transformer sharded)
- `"gguf"` — quantized single-GPU

### Quadro (4 images, 4 GPUs)
Each GPU runs a complete pipeline independently. Works with sdxl/sd15/gguf.
```python
POST /api/quadro  {"arch":"gguf", "prompt":"...", "model_path":"./models/sd35-large-Q4_K_M.gguf"}
```

### Batch (multiple prompts, 4 GPUs)
Round-robin distribution across 4 GPUs. Works with sdxl/sd15/gguf.

---

## GGUF model setup

GGUF files must be pre-downloaded and placed in `./models/` with a sibling
diffusers directory for non-transformer components:

```
./models/
  sd35-large-Q4_K_M.gguf           # quantized transformer
  sd35-large-Q4_K_M/               # sibling: text encoders, VAE, scheduler
    model_index.json
    text_encoder/
    text_encoder_2/
    text_encoder_3/
    vae/
    scheduler/
```

```python
from pipeline_gguf import generate_gguf
generate_gguf(
    prompt="a cat",
    model_path="./models/sd35-large-Q4_K_M.gguf",
    device="cuda:0",
    steps=20,
)
```

---

## Gated repos (SD3.5, FLUX)

Gated repos require a HuggingFace access token:

```bash
# Option 1: environment variable (recommended)
export HF_TOKEN=hf_xxxxxxxxxxxx
python download_models.py --repo stabilityai/stable-diffusion-3.5-large

# Option 2: --hf-token flag
python download_models.py --repo stabilityai/stable-diffusion-3.5-large --hf-token hf_xxx

# Option 3: API
POST /api/models/download  {"repo_id":"...", "hf_token":"hf_xxx..."}

# Option 4: huggingface-cli login
huggingface-cli login
```

---

## Utilities

### clear_cache.py — clean HuggingFace cache
```bash
python clear_cache.py --list              # show cache with sizes
python clear_cache.py                     # interactive (select by number)
python clear_cache.py --all -y            # delete everything
python clear_cache.py --model "sd-legacy/stable-diffusion-v1-5" -y
```

### fix_dit_model_index.py — repair model_index.json
If a DiT model was downloaded with the wrong pipeline class (e.g. gated-repo
fallback to SDXL), this fixes `_class_name` by inspecting directory structure:
```bash
python fix_dit_model_index.py --all       # scan and fix all ./models/*
python fix_dit_model_index.py ./models/sd35-large --dry-run
```

---

## Keep-alive model cache

Loaded pipelines stay resident in VRAM across generations and are freed after an
idle timeout (`SD_IDLE_TIMEOUT`, default 300s). Cache keys include arch type,
model path, devices, dtype, scheduler, and GPU count — so different configurations
get distinct entries.

```bash
SD_IDLE_TIMEOUT=600 python api_server.py   # keep for 10 min
SD_KEEP_ALIVE=0 python api_server.py       # disable (reload every call)
```

---

## Command-line interface

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | Path to model directory or `.gguf` file |
| `--arch` | `sdxl` | `sdxl`, `sd15`, `dit`, or `gguf` |
| `--gpu-down` | `0` | Stage-0 GPU index |
| `--gpu-up` | `1` | Stage-1 GPU index (legacy; N-GPU auto for dit) |
| `--hf-token` | *None* | HuggingFace token for gated repos |
| `--steps` | `25` | Inference steps |
| `--seed` | `42` | RNG seed |
| `--cfg` | `7.5` | Guidance scale |
| `--scheduler` | `default` | `default`/`ddim`/`euler`/`dpmpp_2m` |
| `--fp16` | off | Use FP16 (not recommended on P40) |
| `--serve` | off | Persistent mode (reuse loaded model) |

---

## Testing

```bash
pip install pytest

# Torch-based tests (run on the P40 server):
python -m pytest test_pp_unet.py test_pp_dit.py test_pipeline_cache.py -v

# Pure-Python tests (run anywhere, no torch required):
python -m pytest test_dit_size_estimator.py test_dit_placement_planner.py -v
```

---

## Expected performance on 4x P40

| Config | Time/image | Notes |
|--------|-----------|-------|
| SDXL FP32, 2 GPU | ~40-50 s | Recommended (UNet) |
| DiT FP32, 3-4 GPU (auto) | model-dependent | For 30-60 GB models; T5 + transformer sharded |
| **GGUF Q4, 1 GPU** | model-dependent | Small memory, 1 P40 |
| 2x parallel pipelines, 4 GPU | ~25 s/image avg | 2 images concurrently |
| SDXL-Turbo FP32, 2 GPU | ~8-12 s | 4 steps, CFG=0 |

---

## License

Project code is provided as-is for the 4x Tesla P40 deployment. Model weights
retain their respective licenses.
