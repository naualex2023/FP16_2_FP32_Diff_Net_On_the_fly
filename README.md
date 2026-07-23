# FP16 → FP32 Diffusion Net (On-the-fly) — Pipeline-Parallel FP32 on 4× Tesla P40

Run **SDXL / SD 1.5 in pure FP32** across 4 NVIDIA Tesla P40 GPUs (24 GB VRAM,
Pascal, compute 6.1) using **pipeline parallelism** — splitting the UNet across
GPUs and transferring only activations (~55 MB/step), not weights.

> **Why FP32 on P40?** The Tesla P40 is Pascal (2016): FP16 compute is broken
> (0.18 TFLOPS, 1/64 of FP32). SDXL in FP32 needs ~25 GB, which exceeds one
> card's 24 GB. Pipeline parallelism across 2 GPUs solves this; the other 2
> GPUs run a second pipeline for 2× throughput.

This implementation follows `DEV_GUIDE_pipeline_parallel_fp32_p40.md` and
**fixes two critical bugs** in the guide's reference code (see
*Corrections to the dev guide* below).

---

## Project layout

| File | Purpose |
|------|---------|
| `pp_unet.py` | **Core** — `PipelineParallelUNet`: splits a diffusers `UNet2DConditionModel` across two GPUs. Correct SDXL `text_time` aug-embedding, `conv_norm_out`/`conv_act`, ControlNet/adapter residuals. |
| `pipeline_parallel_sdxl.py` | **Primary use case** — SDXL FP32 on 2 GPUs (down on GPU0, mid+up on GPU1). LoRA, schedulers, Turbo support. Routes through the keep-alive cache. |
| `pipeline_parallel_sd15.py` | Educational SD 1.5 example on 2 GPUs (SD 1.5 FP32 fits on 1 P40, so PP isn't needed — good for testing). Routes through the keep-alive cache. |
| `pipeline_single_gpu.py` | **Single-GPU pipeline (no split)** — loads one complete SDXL/SD 1.5 pipeline on a single device in FP32/FP16. Used by **Quadro mode** (4 images × 4 GPUs). LoRA, schedulers, Turbo support. Routes through the keep-alive cache. |
| `pipeline_cache.py` | **Keep-alive cache** — `PipelineCache` keeps loaded pipelines resident across generations and frees them after an idle timeout (`SD_IDLE_TIMEOUT`). Fixes the "reload on every generation" problem. |
| `async_transfer.py` | Optional `AsyncPipelineParallelUNet` — overlapped compute + PCIe transfer via CUDA streams. |
| `parallel_pipelines_4gpu.py` | Run **two** pipeline-parallel SDXL instances on **four** GPUs (threaded + multiprocess back-ends). Reuses the cache per GPU pair / per worker process. |
| `generate_one.py` | CLI for single-image generation (SDXL / SD 1.5 / Turbo, LoRA, schedulers). `--serve` keeps the model resident across many prompts. |
| `benchmark.py` | Compare FP16-1GPU vs FP32-pipeline-2GPU vs FP32-CPU-offload-1GPU. |
| `gpu_diagnostics.py` | Verify hardware: per-GPU VRAM/compute, FP32 GEMM GFLOPS, P2P PCIe bandwidth, `nvidia-smi topo -m`. |
| `download_models.py` | Fetch preset models (SD 1.5 / SDXL / Turbo) **or any HuggingFace repo** (`--repo org/name`) into `./models/`. |
| `model_resolver.py` | **Generic model loader** — resolves a local path *or* HF repo ID, downloads repo IDs on first use, auto-detects architecture (sdxl/sd15). Used everywhere `model_path` is accepted. |
| `run_parallel.sh` | Launch two 2-GPU pipeline-parallel jobs in parallel (4 GPUs total). |
| `test_pp_unet.py` | Unit tests — **verifies PP UNet output == reference UNet output** (numerically identical). |
| `test_pipeline_cache.py` | Unit tests for the keep-alive cache (hit/miss, idle eviction, thread safety). No CUDA required. |
| `requirements.txt` | Python dependencies. |
| `DEV_GUIDE_pipeline_parallel_fp32_p40.md` | Original development guide (Russian). |
| `USER_GUIDE_diffusers_p40.md` | User guide for diffusers on P40 (Russian). |

---

## Quick start (on the 4× P40 server)

```bash
# 1. Environment (one-time)
python3 -m venv ~/diffusion-env
source ~/diffusion-env/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 2. Verify hardware
python gpu_diagnostics.py
# Expect: 4× Tesla P40, ~2000-3000 GFLOPS each, 10-13 GB/s P2P

# 3. Download models
python download_models.py --model sdxl

# 4. Run tests (validates the pipeline split is numerically correct)
pip install pytest
python -m pytest test_pp_unet.py -v

# 5. Generate one SDXL image on GPU 0 + GPU 1
python generate_one.py \
    --model ./models/sdxl-base-fp16 \
    --gpu-down 0 --gpu-up 1 \
    --prompt "A serene Japanese garden, koi pond, cherry blossoms, golden hour, 8k" \
    --steps 25 --output garden.png

# 6. Generate two images in parallel across all 4 GPUs
./run_parallel.sh ./models/sdxl-base-fp16 \
    "A cyberpunk cityscape at night, neon lights" \
    "A peaceful mountain lake at dawn"
```
Short sequence:
source ~/diffusion-env/bin/activate
pip install -r requirements.txt
python gpu_diagnostics.py
python download_models.py --model sdxl
python -m pytest test_pp_unet.py -v
python generate_one.py --model ./models/sdxl-base-fp16 --gpu-down 0 --gpu-up 1 \
    --prompt "A serene Japanese garden, koi pond, golden hour, 8k" --output garden.png
./run_parallel.sh ./models/sdxl-base-fp16 "cyberpunk city" "mountain lake"

---

## Why pipeline parallel? (Architecture)

```
GPU 0 (Stage 0)                    GPU 1 (Stage 1)
┌──────────────────────────┐       ┌──────────────────────────┐
│  Text Encoders (CLIP×2)  │       │                          │
│  conv_in                 │       │  Mid Block               │
│  Down Block 0            │       │  Up Block 0 ← skip_2     │
│  Down Block 1   skip_0 ──┼───►   │  Up Block 1 ← skip_1     │
│  Down Block 2   skip_1 ──┼───►   │  Up Block 2 ← skip_0     │
│                hidden ───┼───►   │  conv_norm_out/conv_out  │
│                          │  ◄─── │  output noise pred       │
└──────────────────────────┘       └──────────────────────────┘
   ~11-13 GB VRAM                     ~8-11 GB VRAM
```

- **Transfer per step:** ~55 MB activations (hidden states + skip connections
  + text embeddings) ≈ **5 ms** over PCIe 3.0 ×16 (~12 GB/s).
- **Compute per step:** ~2000 ms (P40 FP32).
- **Transfer overhead:** ~0.2 % — negligible.

For the **4-GPU** configuration, two such pipelines run in parallel:

```
Pipeline A: GPU 0 (down) + GPU 1 (up)
Pipeline B: GPU 2 (down) + GPU 3 (up)
```

giving 2× the throughput (two images at once).

---

## API usage

### Single SDXL image (2 GPUs)

```python
from pipeline_parallel_sdxl import generate_sdxl

generate_sdxl(
    prompt="a cinematic photo of a cat, detailed fur, 8k",
    negative_prompt="blurry, low quality",
    model_path="./models/sdxl-base-fp16",
    device_down="cuda:0",
    device_up="cuda:1",
    steps=25,
    width=1024, height=1024,
    seed=42,
    guidance_scale=7.5,
    output_path="cat.png",
)
```

### Batch on 4 GPUs (single-GPU per prompt, no split)

Batch mode distributes different prompts across all 4 GPUs, each running a
complete (un-split) model in FP32/FP16 — same as Quadro mode but with different
prompts instead of the same prompt. Up to 4 prompts generate in parallel.

```python
from parallel_pipelines_4gpu import generate_batch_parallel_processes

generate_batch_parallel_processes(
    prompts=["a cyberpunk city", "a mountain lake", "an astronaut", "a bookshop"],
    model_path="./models/sdxl-base-fp16",
    steps=25,
)
```

Via the web API (`POST /api/batch`) or the **📦 Batch** tab.


### Quadro mode — 4 images on 4 GPUs (no UNet split)

Twin/Generate split the UNet across two GPUs because FP32 SDXL (~25 GB) doesn't
fit on one P40.  **Most other models do fit** (SD 1.5 FP32 ~6.8 GB, SDXL-Turbo,
any FP16 model).  Quadro mode exploits this: each of the 4 GPUs runs **one
complete (un-split) FP32 pipeline** on the *same prompt* with its own seed,
producing **four variations in the time of one**.  No activation ever crosses a
PCIe boundary, so it's simpler and (for models that fit) faster than the split
path.

```python
from pipeline_single_gpu import generate_single_gpu

# One image on one GPU (the building block Quadro uses 4× in parallel):
generate_single_gpu(
    prompt="a serene Japanese garden, koi pond, golden hour",
    arch="sd15",
    device="cuda:0",
    seed=42,
    output_path="garden.png",
)
```

Via the web API (`POST /api/quadro`) or the **🔮 Quadro** tab — pass the same
prompt plus four seeds (`seed_a`/`seed_b`/`seed_c`/`seed_d`, one per GPU).
> **Note:** full SDXL FP32 (~25 GB) does *not* fit on a single 24 GB P40, so
> for SDXL use FP16 or Turbo in Quadro mode, or use Twin/Generate (which split
> the model).  SD 1.5 FP32 works at full size.

### SDXL-Turbo (4 steps, CFG=0)

```python
from pipeline_parallel_sdxl import generate_sdxl

generate_sdxl(
    prompt="a cat warrior, dramatic lighting",
    model_path="./models/sdxl-turbo",
    steps=4,
    guidance_scale=0.0,   # important for Turbo
    output_path="turbo.png",
)
```

### With LoRA

```python
generate_sdxl(
    prompt="a cat in MY_STYLE",
    model_path="./models/sdxl-base-fp16",
    lora_path="./loras/my_style.safetensors",
    lora_scale=0.8,
    output_path="cat_lora.png",
)
```

---

## Keep-alive model cache (no reload between generations)

By default a one-shot CLI/script **reloads the model from disk on every call**
(for SDXL that's ~25 GB + FP32 upcast + UNet split — slow).  The
**keep-alive cache** fixes this: a loaded pipeline stays resident in VRAM and
is reused by the next generation.  After a period of inactivity it is freed
automatically (the "regulated timeout").

### How it works

`pipeline_cache.PipelineCache` is a process-wide singleton.  Every
`generate_sdxl(...)` / `generate_sd15_pipeline_parallel(...)` call goes through
it:

* **First call** ("cache MISS") — loads + splits the model, stores it.
* **Subsequent calls** with the same model/GPU/dtype/LoRA/scheduler ("cache
  HIT") — returns the resident pipeline instantly; only the generation runs.
* **Idle reaper** — a background thread frees any pipeline that hasn't been
  used for `idle_timeout` seconds, calling `del pipe; gc.collect();
  torch.cuda.empty_cache()`.  Each generation **resets the timer**, so the
  model stays loaded while you're actively working and is reaped only when
  genuinely idle.

### Configuration (environment variables)

| Variable | Default | Meaning |
|----------|---------|---------|
| `SD_KEEP_ALIVE` | `1` | `1` enable the cache (recommended); `0` disables it (old per-call reload behavior). |
| `SD_IDLE_TIMEOUT` | `300` | Seconds of inactivity before an idle pipeline is freed. `0` = never auto-free (stays loaded until the process exits). |
| `SD_REAPER_INTERVAL` | `30` | How often (s) the background thread scans for idle entries. |

Examples:

```bash
# Keep the model for 10 minutes between generations:
SD_IDLE_TIMEOUT=600 python generate_one.py --model ./models/sdxl-base-fp16 ...

# Never auto-unload (model stays resident until the process exits):
SD_IDLE_TIMEOUT=0 python generate_one.py --model ./models/sdxl-base-fp16 --serve

# Disable caching entirely (reload every call, like before):
SD_KEEP_ALIVE=0 python generate_one.py --model ./models/sdxl-base-fp16 ...
```

### Two ways to benefit

**1. Same Python process (library/interactive use).** Any second call reuses
the model automatically:

```python
from pipeline_parallel_sdxl import generate_sdxl

# First call: loads + caches the pipeline (~30 s load + ~45 s generate)
generate_sdxl("a cat", output_path="cat.png")
# Second call: NO reload — only ~45 s generate (cache HIT)
generate_sdxl("a dog", output_path="dog.png")
# After 5 min idle: the reaper frees VRAM automatically.
```

**2. Persistent CLI server (`--serve`).** Because each `python generate_one.py`
is a *fresh process*, back-to-back invocations can't share a cache.  Use
`--serve` to keep one process alive and feed it prompts over stdin (the model
loads once and is reused for every image):

```bash
python generate_one.py --model ./models/sdxl-base-fp16 --gpu-down 0 --gpu-up 1 --serve
# READY — type prompts one per line, then 'quit':
a cat on the moon|out_cat.png|seed=123|steps=20
a mountain lake|out_lake.png|cfg=5
quit
```

Each serve line is `prompt|output.png|key=value...` where keys can be
`seed`, `steps`, `cfg`, `w`/`width`, `h`/`height`, `neg`/`negative`.
The second field may be a bare output filename.

### Programmatic control

```python
from pipeline_cache import get_cache

cache = get_cache()
# ... generate ...
print(cache.stats())           # see resident entries + idle times
cache.unload_all()             # free everything now
```

Pass `force_unload=True` to `generate_sdxl`/`generate_sd15_pipeline_parallel`
to free that specific pipeline immediately after one generation.

---

## Command-line interface

```bash
python generate_one.py --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | *(required)* | Path to diffusers model directory |
| `--arch` | `sdxl` | `sdxl` or `sd15` |
| `--gpu-down` | `0` | Stage-0 GPU index |
| `--gpu-up` | `1` | Stage-1 GPU index |
| `--prompt` | *(required)* | Text prompt |
| `--negative` | `blurry, low quality...` | Negative prompt |
| `--output` | `output.png` | Output file |
| `--steps` | `25` | Inference steps |
| `--seed` | `42` | RNG seed |
| `--width/--height` | `1024` | Image size |
| `--cfg` | `7.5` | Guidance scale (0.0 for Turbo) |
| `--scheduler` | `default` | `default`/`ddim`/`euler`/`dpmpp_2m` |
| `--lora` | *None* | Path to LoRA `.safetensors` |
| `--lora-scale` | `1.0` | LoRA strength |
| `--fp16` | off | Use FP16 (**not recommended on P40**) |
| `--serve` | off | **Persistent mode:** keep the process alive, read prompts from stdin, reuse the loaded model for each image (no reload). Type `quit` to exit. |
| `--no-keep-alive` | off | Disable the idle-timeout cache — unload immediately after each job (old behavior). |

---

## Corrections to the dev guide

The reference code in `DEV_GUIDE_pipeline_parallel_fp32_p40.md` had two
critical correctness bugs that this project fixes:

1. **Wrong `temb` passed to UNet blocks.** The guide passed the raw `timestep`
   tensor as `temb` to every down/mid/up block. The real diffusers forward
   computes `emb` from the timestep via `time_proj` + `time_embedding`, adds
   the SDXL `add_embedding` (text+time aug embedding), applies
   `time_embed_act`, and it is **`emb`** that must be passed to blocks.
   The guide's version produced garbage output. → Fixed in `pp_unet.py`.

2. **Incorrect `conv_norm_out` handling for SDXL.** The guide applied a
   nonexistent `scale`/`shift` to `conv_norm_out` (a `GroupNorm`). The real
   post-processing is: `conv_norm_out(x)` → `conv_act(x)` (SiLU) →
   `conv_out(x)`. → Fixed in `pp_unet.py`.

Additional robustness improvements over the guide:
- Full `added_cond_kwargs`/`attention_mask`/`encoder_attention_mask` device
  migration via the recursive `_move()` helper.
- Correct up-block skip-connection slicing (`len(up_block.resnets)`, not `[-1:]`).
- Proper `forward_upsample_size` handling for non-standard resolutions.
- LoRA loading **before** the split (so adapter weights land on the right GPU).
- `enable_model_cpu_offload`/`enable_sequential_cpu_offload` neutralized so
  diffusers doesn't fight our manual placement.
- Multiprocess back-end for 4-GPU batch (avoids GIL and shared CUDA context).

---

## Testing

```bash
pip install pytest
python -m pytest test_pp_unet.py test_pipeline_cache.py -v
```

- `test_pp_unet.py::test_pp_unet_matches_reference` builds a miniature
  SDXL-config UNet, runs it both as a plain module and wrapped in
  `PipelineParallelUNet` (both stages on the same device), and asserts the
  outputs match to within `1e-4`. This proves the split + transfer logic is
  faithful to the original `UNet2DConditionModel.forward`. *(requires CUDA)*
- `test_pipeline_cache.py` covers the keep-alive cache (hit/miss, idle
  eviction, thread safety). *(no CUDA / no real model required — fast)*

---

## Expected performance on 4× P40

| Config | Time/image | Notes |
|--------|-----------|-------|
| SDXL FP16, 1 GPU | ~50–60 s | Broken FP16 on Pascal (slow) |
| **SDXL FP32, pipeline-parallel, 2 GPU** | **~40–50 s** | **Recommended** |
| SDXL FP32, CPU offload, 1 GPU | ~120–300 s | PCIe + RAM bottleneck |
| SDXL-Turbo FP32, 2 GPU | ~8–12 s | 4 steps, CFG=0 |
| 2× parallel pipelines, 4 GPU | ~25 s/image avg | 2 images concurrently |

Throughput on 4 GPUs: ~2.4–4.8 img/min (SDXL FP32, 1024², 25 steps).

---

## Troubleshooting

See `USER_GUIDE_diffusers_p40.md` §12 and `DEV_GUIDE...` §10. Highlights:

- **CUDA OOM:** reduce `width/height` to 832×832 (−40 % VRAM); ensure text
  encoders are only on `device_down`; `gc.collect(); torch.cuda.empty_cache()`.
- **"Expected tensor ... to be on the same device":** every tensor crossing
  the stage boundary is moved by `_move()`; if you add a new kwarg, move it too.
- **Slow:** confirm `next(pipe.unet.parameters()).dtype == torch.float32`;
  check `nvidia-smi topo -m` (PXB links are slower than SYS).

---

## License

Project code is provided as-is for the 4× Tesla P40 deployment described in
the development guide. Model weights retain their respective licenses
(SD 1.5: CreativeML OpenRAIL-M; SDXL: OpenRAIL++).
