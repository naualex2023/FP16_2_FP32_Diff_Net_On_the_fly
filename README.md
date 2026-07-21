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
| `pipeline_parallel_sdxl.py` | **Primary use case** — SDXL FP32 on 2 GPUs (down on GPU0, mid+up on GPU1). LoRA, schedulers, Turbo support. |
| `pipeline_parallel_sd15.py` | Educational SD 1.5 example on 2 GPUs (SD 1.5 FP32 fits on 1 P40, so PP isn't needed — good for testing). |
| `async_transfer.py` | Optional `AsyncPipelineParallelUNet` — overlapped compute + PCIe transfer via CUDA streams. |
| `parallel_pipelines_4gpu.py` | Run **two** pipeline-parallel SDXL instances on **four** GPUs (threaded + multiprocess back-ends). |
| `generate_one.py` | CLI for single-image generation (SDXL / SD 1.5 / Turbo, LoRA, schedulers). |
| `benchmark.py` | Compare FP16-1GPU vs FP32-pipeline-2GPU vs FP32-CPU-offload-1GPU. |
| `gpu_diagnostics.py` | Verify hardware: per-GPU VRAM/compute, FP32 GEMM GFLOPS, P2P PCIe bandwidth, `nvidia-smi topo -m`. |
| `download_models.py` | Fetch SD 1.5 / SDXL / SDXL-Turbo into `./models/`. |
| `run_parallel.sh` | Launch two 2-GPU pipeline-parallel jobs in parallel (4 GPUs total). |
| `test_pp_unet.py` | Unit tests — **verifies PP UNet output == reference UNet output** (numerically identical). |
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

### Batch on 4 GPUs (multiprocess, recommended)

```python
from parallel_pipelines_4gpu import generate_batch_parallel_processes

generate_batch_parallel_processes(
    prompts=["a cyberpunk city", "a mountain lake", "an astronaut", "a bookshop"],
    model_path="./models/sdxl-base-fp16",
    steps=25,
)
```

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
python -m pytest test_pp_unet.py -v
```

The key test, `test_pp_unet_matches_reference`, builds a miniature
SDXL-config UNet, runs it both as a plain module and wrapped in
`PipelineParallelUNet` (both stages on the same device), and asserts the
outputs match to within `1e-4`. This proves the split + transfer logic is
faithful to the original `UNet2DConditionModel.forward`.

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
