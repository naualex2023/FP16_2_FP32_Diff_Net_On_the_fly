# Pipeline Parallel FP32 Diffusion Inference on Multi-Tesla P40

## Development Guide

> Целевое железо: 4x NVIDIA Tesla P40 (24 GB VRAM, Pascal, compute 6.1), 64 GB RAM, 2x Xeon E5-2680, Ubuntu 22.04.
> Цель: запустить SDXL / SD 1.5 в чистом FP32 (12 TFLOPS) с pipeline parallelism между GPU, обойдя 24 GB лимит одной карты.

---

## Содержание

1. [Архитектура и обоснование](#1-архитектура-и-обоснование)
2. [Подготовка окружения](#2-подготовка-окружения)
3. [Архитектура UNet: где резать](#3-архитектура-unet-где-резать)
4. [Базовая реализация: SD 1.5 на 2 GPU](#4-базовая-реализация-sd-15-на-2-gpu)
5. [SDXL Pipeline Parallel](#5-sdxl-pipeline-parallel)
6. [4 GPU: параллельные пайплайны](#6-4-gpu-параллельные-пайплайны)
7. [Оптимизация: асинхронный transfer](#7-оптимизация-асинхронный-transfer)
8. [Интеграция с LoRA и ControlNet](#8-интеграция-с-lora-и-controlnet)
9. [Бенчмаркинг и тестирование](#9-бенчмаркинг-и-тестирование)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Архитектура и обоснование

### 1.1 Проблема

Tesla P40 — Pascal (2016). FP16 работает на 0.18 TFLOPS (1/64 от FP32). Для диффузионных моделей это означает, что FP16 inference **медленнее** FP32. FP32 — единственный рабочий формат на этой карте (12 TFLOPS).

Но FP32 модель в 2 раза тяжелее:
- SD 1.5: FP16 = 3.4 GB, FP32 = 6.8 GB — влезает в 24 GB
- SDXL: FP16 = 12.7 GB, FP32 = 25.4 GB — **не влезает** в 24 GB

### 1.2 Решение: Pipeline Parallelism

Разрезать UNet на две части, разместить на разных GPU. Между частями передавать **активации** (тензор фича-карт), а не веса.

```
Почему активации, а не веса:

Веса блока Down:      ~200-800 MB (зависит от блока)
Активации (skip conn):  ~5-30 MB каждый

PCIe 3.0 x16:          ~12 GB/s реальная полоса
Пересылка весов:       200 MB / 12 GB/s = 17 ms на блок
Пересылка активаций:   30 MB / 12 GB/s = 2.5 ms всего

Время compute шага:    ~1000-2000 ms

Overhead активаций:    2.5 ms / 1500 ms = 0.17%  — незаметно
Overhead весов:        17 ms × N блоков = значимо   — боль
```

### 1.3 Data Flow

```
GPU 0 (Stage 0)                    GPU 1 (Stage 1)
┌──────────────────────────┐       ┌──────────────────────────┐
│  Text Encoder (CLIP)     │       │                          │
│  ↓                       │       │                          │
│  Down Block 0            │       │                          │
│  Down Block 1            │       │                          │
│  Down Block 2            │       │                          │
│  ↓                       │       │                          │
│  hidden_states (~5 MB)   ├───►   │  Mid Block               │
│  skip_0 (~20 MB)         ├───►   │  ↓                       │
│  skip_1 (~10 MB)         ├───►   │  Up Block 0  ← skip_2    │
│  skip_2 (~5 MB)          ├───►   │  Up Block 1  ← skip_1    │
│                          │       │  Up Block 2  ← skip_0    │
│                          │  ◄─── │  output_noise_pred (~2MB)│
└──────────────────────────┘       └──────────────────────────┘
     ~12-13 GB VRAM                     ~12-13 GB VRAM
```

---

## 2. Подготовка окружения

### 2.1 CUDA Toolkit

Pascal (compute capability 6.1) поддерживается CUDA 11.x–12.x. Рекомендуем CUDA 12.1 (последняя стабильная с полной поддержкой Pascal).

```bash
# Проверить текущую версию
nvidia-smi
nvcc --version

# Если CUDA не установлена или старая:
# Скачай с https://developer.nvidia.com/cuda-12-1-0-download-archive
# Выбери: Linux > x86_64 > Ubuntu > 22.04 > runfile (local)

wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run --toolkit --silent --override

# Добавить в ~/.bashrc:
export PATH=/usr/local/cuda-12.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
```

### 2.2 Python и PyTorch

```bash
# Создать venv
python3 -m venv ~/diffusion-env
source ~/diffusion-env/bin/activate

# PyTorch с CUDA 12.1 (Pascal не поддерживает cuDNN fp16 efficiently,
# но PyTorch нужен для CUDA runtime)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Проверить:
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {props.name}, {props.total_mem / 1e9:.1f} GB, compute {props.major}.{props.minor}')
    # Проверить, что FP32 работает:
    a = torch.randn(4096, 4096, device=f'cuda:{i}', dtype=torch.float32)
    import time
    torch.cuda.synchronize(i)
    t0 = time.perf_counter()
    for _ in range(10):
        c = torch.mm(a, a)
    torch.cuda.synchronize(i)
    t1 = time.perf_counter()
    gflops = 2 * 4096**3 * 10 / (t1 - t0) / 1e9
    print(f'  FP32 GEMM performance: ~{gflops:.0f} GFLOPS (expect ~2000-3000 for single matmul)')
"
```

Ожидаемый вывод:
```
PyTorch: 2.x.x
CUDA available: True
GPU count: 4
  GPU 0: Tesla P40, 24.0 GB, compute 6.1
  FP32 GEMM performance: ~2500 GFLOPS
  GPU 1: Tesla P40, 24.0 GB, compute 6.1
  ...
```

### 2.3 Diffusers и зависимости

```bash
pip install diffusers transformers accelerate safetensors
pip install compel  # для продвинутых промптов
pip install peft    # для LoRA
pip install controlnet-aux  # для ControlNet препроцессоров
```

### 2.4 Скачивание моделей

```bash
# SD 1.5 (FP32 чекпоинт, ~6.8 GB)
# Большинство чекпоинтов на HuggingFace в FP16.
# Для FP32 нужно сконвертировать:

python3 -c "
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_single_file(
    'https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned-emaonly.safetensors',
    torch_dtype=torch.float32,  # Конвертация в FP32 при загрузке
)
pipe.save_pretrained('./models/sd15-fp32')
print('SD 1.5 FP32 сохранён в ./models/sd15-fp32')
"

# SDXL (нужен только FP16, конвертим в runtime)
# Скачаем через diffusers:
python3 -c "
from diffusers import StableDiffusionXLPipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0',
    torch_dtype=torch.float16,
)
pipe.save_pretrained('./models/sdxl-base-fp16')
print('SDXL FP16 сохранён в ./models/sdxl-base-fp16')
"
```

### 2.5 Проверка PCIe топологии

```bash
# Узнать, как GPU подключены (важно для скорости transfer):
nvidia-smi topo -m

# Идеально: все GPU на одном NUMA node через PLX switch
# Плохо: GPU разбросаны по разным CPU (QPI bottleneck)

python3 -c "
import torch
# Быстрый тест PCIe bandwidth между GPU 0 и GPU 1:
a = torch.randn(100_000_000, device='cuda:0')  # ~400 MB
import time
torch.cuda.synchronize()
t0 = time.perf_counter()
b = a.to('cuda:1')
torch.cuda.synchronize()
t1 = time.perf_counter()
print(f'GPU 0 → GPU 1: 400 MB за {(t1-t0)*1000:.1f} ms = {400/(t1-t0)/1e9:.1f} GB/s')
# Ожидание: 10-13 GB/s (PCIe 3.0 x16)
# Если < 5 GB/s — GPU на разных CPU, будет медленнее
"
```

---

## 3. Архитектура UNet: где резать

### 3.1 SD 1.5 UNet

```
UNet2DConditionModel (860M параметров)

Вход: latent [B, 4, 64, 64] + timesteps + text_embeddings [B, 77, 768]

Down Blocks (encoder):
  down_blocks.0:  ResNet(320→320) + Attn(320)   → skip_0: [B, 320, 64, 64]
  down_blocks.1:  ResNet(320→640) + Attn(640)   → skip_1: [B, 640, 32, 32]
  down_blocks.2:  ResNet(640→1280) + Attn(1280) → skip_2: [B, 1280, 16, 16]
  down_blocks.3:  ResNet(1280→1280)              → skip_3: [B, 1280, 8, 8]

Mid Block:
  mid_block: ResNet(1280) + Attn(1280) + ResNet(1280)

Up Blocks (decoder):
  up_blocks.0:  ResNet(2560→1280) + Attn(1280)  ← skip_3
  up_blocks.1:  ResNet(2560→1280) + Attn(1280)  ← skip_2
  up_blocks.2:  ResNet(1920→640)  + Attn(640)   ← skip_1
  up_blocks.3:  ResNet(960→320)   + Attn(320)   ← skip_0

Выход: noise_pred [B, 4, 64, 64]

VRAM в FP32:
  Down blocks (0-3):  ~2.5 GB
  Mid block:          ~0.8 GB
  Up blocks (0-3):    ~2.5 GB
  ---------------------------
  Итого UNet:        ~5.8 GB  (влезает в 24 GB даже с TE и активациями!)
```

**Вывод: SD 1.5 в FP32 влезает на одну P40 (6.8 GB UNet + TE). Pipeline parallelism для SD 1.5 не нужен — используй обычный FP32.**

### 3.2 SDXL UNet

```
UNet2DConditionModel (2.56B параметров)

Вход: latent [B, 4, 128, 128] + timesteps + text_embeddings (два: CLIP-L + CLIP-G)

Down Blocks:
  down_blocks.0:  InCh(320→320) + 2×(ResNet+Attn(320))     → skip_0: [B, 320, 128, 128]
  down_blocks.1:  DownCh(320→640) + 2×(ResNet+Attn(640))    → skip_1: [B, 640, 64, 64]
  down_blocks.2:  DownCh(640→1280) + 2×(ResNet+Attn(1280))  → skip_2: [B, 1280, 32, 32]

Mid Block:
  mid_block: ResNet(1280) + Attn(1280) + ResNet(1280)

Up Blocks:
  up_blocks.0:  UpCh(1280→1280) + 3×(ResNet+Attn(1280))   ← skip_2
  up_blocks.1:  UpCh(2560→640) + 3×(ResNet+Attn(640))     ← skip_1
  up_blocks.2:  UpCh(1280→320) + 3×(ResNet+Attn(320))     ← skip_0
  up_blocks.3:  OutCh(640→320)  + ResNet(320)

VRAM в FP32:
  Down blocks (0-2):  ~6.5 GB
  Mid block:          ~1.5 GB
  Up blocks (0-3):    ~7.5 GB
  ---------------------------
  Итого UNet:        ~15.5 GB

  + Text Encoders:    ~5.0 GB (CLIP-L 1.2B + OpenCLIP-G 0.4B в FP32)
  + VAE:              ~0.6 GB
  + Latents + act:    ~2.0 GB
  ---------------------------
  ВСЕГО:             ~23.1 GB  (впритык к 24 GB, без запаса)
```

**Точка разреза для pipeline parallel:**

```
STAGE 0 (GPU 0):             STAGE 1 (GPU 1):
  Text Encoders (5.0 GB)       Mid Block (1.5 GB)
  Down Block 0 (2.0 GB)       Up Block 0 (2.5 GB)
  Down Block 1 (2.0 GB)       Up Block 1 (2.5 GB)
  Down Block 2 (2.5 GB)       Up Block 2 (1.5 GB)
  ----------------             Up Block 3 (0.5 GB)
  ~11.5 GB                    ----------------
                               ~8.5 GB

Активации для передачи:
  hidden_states: [B, 1280, 32, 32] × 4 bytes = ~20 MB
  skip_0: [B, 320, 128, 128] × 4 bytes = ~20 MB
  skip_1: [B, 640, 64, 64] × 4 bytes = ~10 MB
  skip_2: [B, 1280, 32, 32] × 4 bytes = ~5 MB
  text_embeds: [B, 77+77, 2048] × 4 bytes = ~0.5 MB
  timesteps: negligible
  -------------------------------
  Итого: ~55 MB per step
  Transfer time: 55 MB / 12 GB/s ≈ 5 ms
```

---

## 4. Базовая реализация: SD 1.5 на 2 GPU

> **Примечание:** как показано выше, SD 1.5 в FP32 влезает на одну P40 (6.8 GB UNet). Этот раздел — учебный пример pipeline parallelism на простой модели. Для реальной работы используй обычный FP32 без pipeline.

### 4.1 Минимальный PipelineParallelUNet

```python
"""
pipeline_parallel_sd15.py
SD 1.5 с pipeline parallelism на 2 GPU в FP32.
УЧЕБНЫЙ ПРИМЕР — для SD 1.5 проще использовать обычный FP32 на 1 GPU.
"""

import torch
import torch.nn as nn
from diffusers import StableDiffusionPipeline, UNet2DConditionModel


class PipelineParallelUNetSD15(nn.Module):
    """
    Разрезает SD 1.5 UNet на 2 стадии:
      Stage 0 (device_down): Text Encoders + Down blocks
      Stage 1 (device_up):   Mid block + Up blocks + VAE decoder
    """

    def __init__(
        self,
        unet: UNet2DConditionModel,
        device_down: str = "cuda:0",
        device_up: str = "cuda:1",
    ):
        super().__init__()
        self.device_down = device_down
        self.device_up = device_up

        # === Разрезаем UNet ===
        # Stage 0: все down_blocks
        self.down_blocks = nn.ModuleList(unet.down_blocks).to(device_down)
        # Stage 1: mid + up blocks
        self.mid_block = unet.mid_block.to(device_up)
        self.up_blocks = nn.ModuleList(unet.up_blocks).to(device_up)

        # Config'и для resnet/attention (нужны для forward)
        self.config = unet.config
        # Прокидываем conv_in на stage 0
        self.conv_in = unet.conv_in.to(device_down)
        # Прокидываем conv_out на stage 1
        self.conv_out = unet.conv_out.to(device_up)

    def forward(
        self,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Аналог unet.forward(), но с transfer между GPU.
        """
        # Убеждаемся, что входы на device_down
        latent_model_input = latent_model_input.to(self.device_down)
        timestep = timestep.to(self.device_down)
        encoder_hidden_states = encoder_hidden_states.to(self.device_down)

        # === STAGE 0: conv_in + down_blocks (GPU 0) ===
        sample = self.conv_in(latent_model_input)

        down_block_res_samples = ()
        for down_block in self.down_blocks:
            if hasattr(down_block, "has_cross_attention") and down_block.has_cross_attention:
                sample, res_samples = down_block(
                    hidden_states=sample,
                    temb=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample, res_samples = down_block(
                    hidden_states=sample,
                    temb=timestep,
                )
            down_block_res_samples += res_samples

        # === TRANSFER: GPU 0 → GPU 1 ===
        # Пересылаем активации (НЕ веса!)
        sample = sample.to(self.device_up)
        down_block_res_samples = tuple(
            s.to(self.device_up) for s in down_block_res_samples
        )
        timestep = timestep.to(self.device_up)
        encoder_hidden_states = encoder_hidden_states.to(self.device_up)

        # === STAGE 1: mid_block + up_blocks (GPU 1) ===
        sample = self.mid_block(
            sample,
            timestep,
            encoder_hidden_states,
        )

        for up_block in self.up_blocks:
            res_samples = down_block_res_samples[-1:]
            down_block_res_samples = down_block_res_samples[:-1]
            if hasattr(up_block, "has_cross_attention") and up_block.has_cross_attention:
                sample = up_block(
                    hidden_states=sample,
                    temb=timestep,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                )
            else:
                sample = up_block(
                    hidden_states=sample,
                    temb=timestep,
                    res_hidden_states_tuple=res_samples,
                )

        # conv_out
        sample = self.conv_out(sample)

        return sample.to(self.device_down)  # Вернуть на GPU 0 для scheduler


def generate_sd15_pipeline_parallel(
    prompt: str,
    model_path: str = "./models/sd15-fp32",
    device_down: str = "cuda:0",
    device_up: str = "cuda:1",
    steps: int = 20,
    seed: int = 42,
):
    """Полный цикл генерации SD 1.5 с pipeline parallelism."""
    import torch
    from diffusers import StableDiffusionPipeline, DDIMScheduler

    print(f"Загрузка модели из {model_path} (FP32)...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
    )

    # Заменяем UNet на pipeline-parallel версию
    print(f"Создание pipeline parallel UNet: {device_down} + {device_up}")
    pp_unet = PipelineParallelUNetSD15(
        unet=pipe.unet,
        device_down=device_down,
        device_up=device_up,
    )
    pipe.unet = pp_unet

    # Text encoder оставляем на device_down (он там уже)
    # VAE decoder — на device_up или device_down (маленький, ~0.3 GB)

    # Генерация
    generator = torch.Generator(device="cpu").manual_seed(seed)

    print(f"Генерация: '{prompt}' ({steps} шагов)...")
    import time
    t0 = time.perf_counter()

    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        generator=generator,
    ).images[0]

    t1 = time.perf_counter()
    print(f"Готово за {t1-t0:.1f} сек")

    image.save("output_sd15_pp.png")
    print("Сохранено: output_sd15_pp.png")
    return image


if __name__ == "__main__":
    generate_sd15_pipeline_parallel(
        prompt="a photo of a cat sitting on a windowsill, golden hour, detailed fur",
        steps=20,
    )
```

### 4.2 Запуск и проверка

```bash
# Запуск:
python pipeline_parallel_sd15.py

# Ожидаемый вывод:
# Загрузка модели... (несколько секунд)
# Создание pipeline parallel UNet: cuda:0 + cuda:1
# Генерация... (~7-10 секунд для 20 шагов)
# Готово за 8.3 сек
```

### 4.3 Бенчмарк: pipeline vs обычный FP32

```bash
python3 -c "
import torch, time
from pipeline_parallel_sd15 import PipelineParallelUNetSD15, generate_sd15_pipeline_parallel

# 1. Обычный FP32 на 1 GPU
pipe = StableDiffusionPipeline.from_pretrained('./models/sd15-fp32', torch_dtype=torch.float32).to('cuda:0')
gen = torch.Generator(device='cpu').manual_seed(42)
t0 = time.perf_counter()
for _ in range(5):
    pipe('a cat', num_inference_steps=20, generator=torch.Generator(device='cpu').manual_seed(_))
    torch.cuda.synchronize()
t1 = time.perf_counter()
print(f'Обычный FP32 1 GPU: {(t1-t0)/5:.2f} сек/изображение')

# 2. Pipeline parallel 2 GPU
t0 = time.perf_counter()
for _ in range(5):
    generate_sd15_pipeline_parallel('a cat', steps=20, seed=_)
t1 = time.perf_counter()
print(f'Pipeline FP32 2 GPU: {(t1-t0)/5:.2f} сек/изображение')
# Ожидание: pipeline на ~5-15% медленнее из-за PCIe overhead
# (Для SD 1.5 pipeline не нужен — это учебный пример)
"
```

---

## 5. SDXL Pipeline Parallel

> **Вот где pipeline parallelism реально нужен.** SDXL FP32 = 25.4 GB > 24 GB лимит.

### 5.1 Полная реализация

```python
"""
pipeline_parallel_sdxl.py
SDXL в FP32 с pipeline parallelism на 2 GPU.
Это основное решение для Tesla P40.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
from diffusers import StableDiffusionXLPipeline


class PipelineParallelUNetSDXL(nn.Module):
    """
    SDXL UNet, разрезанный на 2 GPU в FP32.

    Stage 0 (device_down): conv_in + down_blocks.0-2 + Text Encoders
    Stage 1 (device_up):   mid_block + up_blocks.0-3 + conv_out

    Активации между стадиями: ~55 MB (transfer ~5 ms via PCIe 3.0)
    """

    def __init__(
        self,
        unet,
        device_down: str = "cuda:0",
        device_up: str = "cuda:1",
    ):
        super().__init__()
        self.device_down = device_down
        self.device_up = device_up

        # === Stage 0: входная часть ===
        self.conv_in = unet.conv_in.to(device_down)
        self.down_blocks = nn.ModuleList(unet.down_blocks).to(device_down)

        # === Stage 1: выходная часть ===
        self.mid_block = unet.mid_block.to(device_up)
        self.up_blocks = nn.ModuleList(unet.up_blocks).to(device_up)
        self.conv_out = unet.conv_out.to(device_up)
        self.conv_norm_out = unet.conv_norm_out if hasattr(unet, 'conv_norm_out') else None
        if self.conv_norm_out is not None:
            self.conv_norm_out = self.conv_norm_out.to(device_up)

        self.config = unet.config

        # Для add_embedding (added_cond_kwargs в SDXL)
        if hasattr(unet, 'add_embedding'):
            self.add_embedding = unet.add_embedding.to(device_up)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass с автоматическим transfer между GPU.
        """
        # === Входы на device_down ===
        sample = sample.to(self.device_down)
        timestep = timestep.to(self.device_down)
        encoder_hidden_states = encoder_hidden_states.to(self.device_down)

        # === STAGE 0: conv_in + down_blocks (GPU 0) ===
        sample = self.conv_in(sample)

        down_block_res_samples = ()
        for down_block in self.down_blocks:
            sample, res_samples = down_block(
                hidden_states=sample,
                temb=timestep,
                encoder_hidden_states=encoder_hidden_states,
                **kwargs,
            )
            down_block_res_samples += res_samples

        # === TRANSFER: GPU 0 → GPU 1 (~5 ms) ===
        sample = sample.to(self.device_up)
        down_block_res_samples = tuple(
            s.to(self.device_up) for s in down_block_res_samples
        )
        timestep = timestep.to(self.device_up)
        encoder_hidden_states = encoder_hidden_states.to(self.device_up)

        # Подготовка added_cond_kwargs для SDXL
        if added_cond_kwargs is not None:
            added_cond_kwargs = {
                k: v.to(self.device_up) if isinstance(v, torch.Tensor) else v
                for k, v in added_cond_kwargs.items()
            }

        # === STAGE 1: mid + up blocks (GPU 1) ===
        # SDXL mid_block может принимать added_cond_kwargs
        if added_cond_kwargs is not None:
            sample = self.mid_block(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
            )
        else:
            sample = self.mid_block(
                sample,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
            )

        for up_block in self.up_blocks:
            res_samples = down_block_res_samples[-1:]
            down_block_res_samples = down_block_res_samples[:-1]
            sample = up_block(
                hidden_states=sample,
                temb=timestep,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=encoder_hidden_states,
                **kwargs,
            )

        # conv_out (+ optional norm)
        if self.conv_norm_out is not None:
            sample = self.conv_norm_out(sample)
            sample = sample * (1 + self.conv_norm_out.scale) + self.conv_norm_out.shift
            sample = self.conv_out(sample)
        else:
            sample = self.conv_out(sample)

        return sample.to(self.device_down)


def create_sdxl_pipeline_parallel(
    model_path: str = "./models/sdxl-base-fp16",
    device_down: str = "cuda:0",
    device_up: str = "cuda:1",
    use_fp32: bool = True,
    compile: bool = False,
):
    """
    Создаёт SDXL pipeline с pipeline parallelism.

    Args:
        model_path: путь к локальной модели SDXL
        device_down: GPU для down blocks + text encoder
        device_up: GPU для mid + up blocks
        use_fp32: конвертировать в FP32 (рекомендуется для P40)
        compile: применить torch.compile() к каждой стадии
    """
    dtype = torch.float32 if use_fp32 else torch.float16

    print(f"Загрузка SDXL из {model_path} ({'FP32' if use_fp32 else 'FP16'})...")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
    )

    # Text encoders на device_down
    pipe.text_encoder = pipe.text_encoder.to(device_down)
    pipe.text_encoder_2 = pipe.text_encoder_2.to(device_down)
    pipe.tokenizer = pipe.tokenizer  # CPU
    pipe.tokenizer_2 = pipe.tokenizer_2  # CPU

    # VAE — можно оставить на device_up (небольшой)
    pipe.vae = pipe.vae.to(device_up)
    # Или на device_down — не критично, VAE вызывается 1 раз после всех шагов

    # Scheduler остаётся на CPU

    # Создаём pipeline-parallel UNet
    print(f"Создание pipeline parallel UNet: {device_down} (down) + {device_up} (up)")
    pp_unet = PipelineParallelUNetSDXL(
        unet=pipe.unet,
        device_down=device_down,
        device_up=device_up,
    )
    pipe.unet = pp_unet

    # Опционально: torch.compile для каждой стадии
    if compile:
        print("Компиляция stage 0 (down)...")
        pp_unet.conv_in = torch.compile(pp_unet.conv_in)
        pp_unet.down_blocks = nn.ModuleList([
            torch.compile(b) for b in pp_unet.down_blocks
        ])
        print("Компиляция stage 1 (up)...")
        pp_unet.mid_block = torch.compile(pp_unet.mid_block)
        pp_unet.up_blocks = nn.ModuleList([
            torch.compile(b) for b in pp_unet.up_blocks
        ])
        pp_unet.conv_out = torch.compile(pp_unet.conv_out)
        print("Компиляция завершена (первый запуск будет медленнее — трассировка)")

    # Отключаем неиспользуемое
    pipe.enable_model_cpu_offload = lambda: None  # Не нужен, мы сами управляем
    pipe.safety_checker = None  # Экономим VRAM

    return pipe


def generate_sdxl(
    prompt: str,
    negative_prompt: str = "",
    model_path: str = "./models/sdxl-base-fp16",
    device_down: str = "cuda:0",
    device_up: str = "cuda:1",
    steps: int = 25,
    width: int = 1024,
    height: int = 1024,
    seed: int = 42,
    guidance_scale: float = 7.5,
    use_fp32: bool = True,
    output_path: str = "output_sdxl_pp.png",
):
    """Генерация SDXL с pipeline parallelism."""

    pipe = create_sdxl_pipeline_parallel(
        model_path=model_path,
        device_down=device_down,
        device_up=device_up,
        use_fp32=use_fp32,
    )

    generator = torch.Generator(device="cpu").manual_seed(seed)

    print(f"\n{'='*60}")
    print(f"Генерация SDXL ({'FP32' if use_fp32 else 'FP16'})")
    print(f"  Prompt: {prompt[:80]}...")
    print(f"  Размер: {width}x{height}, шагов: {steps}")
    print(f"  GPU: {device_down} (down) + {device_up} (up)")
    print(f"{'='*60}\n")

    import time
    t0 = time.perf_counter()

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        generator=generator,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
    ).images[0]

    t1 = time.perf_counter()
    total = t1 - t0
    print(f"\nГотово за {total:.1f} сек ({total/steps:.2f} сек/шаг)")
    image.save(output_path)
    print(f"Сохранено: {output_path}")
    return image


if __name__ == "__main__":
    generate_sdxl(
        prompt=(
            "A serene Japanese garden with a koi pond, cherry blossoms falling, "
            "golden hour light, photorealistic, 8k, detailed"
        ),
        negative_prompt="blurry, low quality, distorted, ugly",
        steps=25,
        seed=42,
    )
```

### 5.2 Запуск

```bash
# Базовый запуск (FP32, 2 GPU):
python pipeline_parallel_sdxl.py

# Ожидание:
# Загрузка... (~30 сек при первом запуске, потом из кэша)
# Создание pipeline parallel UNet: cuda:0 (down) + cuda:1 (up)
# Генерация SDXL (FP32)
#   Размер: 1024x1024, шагов: 25
# Готово за 50-70 сек (~2-3 сек/шаг)
```

### 5.3 VRAM мониторинг

```bash
# В другом терминале, во время генерации:
watch -n 1 'nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv'

# Ожидание:
# GPU 0: ~12-14 GB (TE + down blocks + latents)
# GPU 1: ~9-11 GB (mid + up blocks + activations)
# GPU 2: 0 MB (свободна)
# GPU 3: 0 MB (свободна)
```

---

## 6. 4 GPU: параллельные пайплайны

```python
"""
parallel_pipelines_4gpu.py
Запуск 2 параллельных SDXL FP32 пайплайнов на 4 GPU.
"""

import torch
import threading
import time
from pipeline_parallel_sdxl import create_sdxl_pipeline_parallel


def generate_on_gpu_pair(
    prompt: str,
    seed: int,
    device_down: str,
    device_up: str,
    output_path: str,
    model_path: str = "./models/sdxl-base-fp16",
    steps: int = 25,
):
    """Генерация на одной паре GPU."""
    pipe = create_sdxl_pipeline_parallel(
        model_path=model_path,
        device_down=device_down,
        device_up=device_up,
        use_fp32=True,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)

    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        generator=generator,
    ).images[0]

    image.save(output_path)
    print(f"[{device_down}+{device_up}] Сохранено: {output_path}")


def generate_batch_parallel(
    prompts: list[str],
    model_path: str = "./models/sdxl-base-fp16",
    steps: int = 25,
):
    """
    Генерация батча изображений на 4 GPU (2 параллельных пайплайна).

    Args:
        prompts: список промптов
        model_path: путь к модели
        steps: количество шагов денойзинга

    Каждый пайплайн использует 2 GPU:
      Pipeline 0: cuda:0 (down) + cuda:1 (up)
      Pipeline 1: cuda:2 (down) + cuda:3 (up)
    """
    gpu_pairs = [("cuda:0", "cuda:1"), ("cuda:2", "cuda:3")]

    results = []
    lock = threading.Lock()

    def worker(prompt_idx):
        pair_idx = prompt_idx % len(gpu_pairs)
        dev_down, dev_up = gpu_pairs[pair_idx]

        generate_on_gpu_pair(
            prompt=prompts[prompt_idx],
            seed=42 + prompt_idx,
            device_down=dev_down,
            device_up=dev_up,
            output_path=f"output_batch_{prompt_idx:03d}.png",
            model_path=model_path,
            steps=steps,
        )
        with lock:
            results.append(prompt_idx)

    # Запускаем до 2 потоков параллельно (по числу GPU-пар)
    max_parallel = len(gpu_pairs)
    threads = []

    t0 = time.perf_counter()
    for i, prompt in enumerate(prompts):
        t = threading.Thread(target=worker, args=(i,))
        threads.append(t)
        t.start()

        # Не запускаем больше, чем GPU-пар одновременно
        active = sum(1 for t in threads if t.is_alive())
        if active >= max_parallel:
            threads[i].join()  # Ждём освобождение
            # Перезапускаем потоки по мере освобождения

    for t in threads:
        t.join()

    t1 = time.perf_counter()
    print(f"\nБатч: {len(prompts)} изображений за {t1-t0:.1f} сек")
    print(f"Throughput: {len(prompts)/(t1-t0)*60:.1f} img/min")


if __name__ == "__main__":
    prompts = [
        "A cyberpunk cityscape at night, neon lights reflecting in wet streets, 8k",
        "A peaceful mountain lake at dawn, mist rising, photorealistic",
        "An astronaut floating above Earth, stars visible, detailed spacesuit",
        "A cozy bookshop interior, warm lighting, shelves full of colorful books",
    ]

    generate_batch_parallel(prompts, steps=25)
```

### 6.1 Альтернатива: мультипроцессы (более надёжно)

Потоки в Python имеют GIL, но PyTorch освобождает GIL во время CUDA-операций. Тем не менее, для максимальной изоляции лучше использовать **мультипроцессы**:

```bash
# run_parallel.sh — простой скрипт для 4 параллельных генераций

#!/bin/bash
MODEL="./models/sdxl-base-fp16"

# Pipeline 0: GPU 0 + GPU 1
python generate_one.py \
  --model "$MODEL" --gpu-down 0 --gpu-up 1 \
  --prompt "A cyberpunk cityscape" \
  --output "out_0.png" &

# Pipeline 1: GPU 2 + GPU 3
python generate_one.py \
  --model "$MODEL" --gpu-down 2 --gpu-up 3 \
  --prompt "A mountain lake at dawn" \
  --output "out_1.png" &

wait
echo "Оба изображения готовы"
```

```python
# generate_one.py —单个生成器 с CLI
import argparse
from pipeline_parallel_sdxl import generate_sdxl

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--gpu-down", type=int, default=0)
parser.add_argument("--gpu-up", type=int, default=1)
parser.add_argument("--prompt", required=True)
parser.add_argument("--output", default="output.png")
parser.add_argument("--steps", type=int, default=25)
args = parser.parse_args()

generate_sdxl(
    prompt=args.prompt,
    model_path=args.model,
    device_down=f"cuda:{args.gpu_down}",
    device_up=f"cuda:{args.gpu_up}",
    steps=args.steps,
    output_path=args.output,
)
```

---

## 7. Оптимизация: асинхронный transfer

PyTorch поддерживает **CUDA streams** — можно overlapping вычисления и пересылку данных:

```python
"""
async_transfer.py
Оптимизация: overlapped compute + PCIe transfer.
"""

import torch

class AsyncPipelineParallelUNet(PipelineParallelUNetSDXL):
    """
    Расширяет базовый pipeline parallel с CUDA streams
    для overlapped transfer и compute.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Создаём отдельные CUDA streams для transfer
        self.transfer_stream = torch.cuda.Stream()

    def forward(self, sample, timestep, encoder_hidden_states,
                added_cond_kwargs=None, **kwargs):
        # === Stage 0: Down blocks на default stream (GPU 0) ===
        sample = sample.to(self.device_down)
        timestep = timestep.to(self.device_down)
        encoder_hidden_states = encoder_hidden_states.to(self.device_down)

        sample = self.conv_in(sample)

        down_block_res_samples = ()
        for down_block in self.down_blocks:
            sample, res_samples = down_block(
                hidden_states=sample,
                temb=timestep,
                encoder_hidden_states=encoder_hidden_states,
                **kwargs,
            )
            down_block_res_samples += res_samples

        # === ASYNC TRANSFER с помощью CUDA stream ===
        # Запускаем transfer в фоне
        with torch.cuda.stream(self.transfer_stream):
            sample_up = sample.to(self.device_up, non_blocking=True)
            res_up = tuple(
                s.to(self.device_up, non_blocking=True)
                for s in down_block_res_samples
            )
            ts_up = timestep.to(self.device_up, non_blocking=True)
            ehs_up = encoder_hidden_states.to(self.device_up, non_blocking=True)

            if added_cond_kwargs is not None:
                ack_up = {
                    k: v.to(self.device_up, non_blocking=True)
                    if isinstance(v, torch.Tensor) else v
                    for k, v in added_cond_kwargs.items()
                }
            else:
                ack_up = None

        # Синхронизируемся перед Stage 1
        torch.cuda.current_stream(self.device_up).wait_stream(self.transfer_stream)

        # === Stage 1: Mid + Up blocks (GPU 1) ===
        if ack_up is not None:
            sample_up = self.mid_block(
                sample_up,
                ts_up,
                encoder_hidden_states=ehs_up,
                added_cond_kwargs=ack_up,
            )
        else:
            sample_up = self.mid_block(
                sample_up, ts_up, encoder_hidden_states=ehs_up,
            )

        for up_block in self.up_blocks:
            res_samples = res_up[-1:]
            res_up = res_up[:-1]
            sample_up = up_block(
                hidden_states=sample_up,
                temb=ts_up,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=ehs_up,
                **kwargs,
            )

        if self.conv_norm_out is not None:
            sample_up = self.conv_norm_out(sample_up)
            sample_up = sample_up * (1 + self.conv_norm_out.scale) + self.conv_norm_out.shift
        sample_up = self.conv_out(sample_up)

        return sample_up.to(self.device_down)
```

> **Ожидаемый выигрыш от async transfer:** минимальный (0.3-0.5%), потому что transfer и так занимает ~5 ms при compute ~2000 ms. Но при частых маленьких transfer'ах (много skip connections) может дать 1-3%.

---

## 8. Интеграция с LoRA и ControlNet

### 8.1 LoRA

LoRA веса нужно разместить на **обоих GPU**, потому что они применяются к слоям на обеих стадиях:

```python
from peft import PeftModel

def load_lora_pipeline_parallel(
    base_model_path: str,
    lora_path: str,
    device_down: str = "cuda:0",
    device_up: str = "cuda:1",
):
    """Загрузка SDXL + LoRA с pipeline parallelism."""
    from diffusers import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path, torch_dtype=torch.float32,
    )

    # Применяем LoRA ДО разделения на стадии
    pipe.load_lora_weights(lora_path)

    # Теперь создаём pipeline parallel — LoRA веса уже включены в слои
    # Но нужно убедиться, что все параметры на нужных устройствах
    pp_unet = PipelineParallelUNetSDXL(
        unet=pipe.unet,
        device_down=device_down,
        device_up=device_up,
    )
    pipe.unet = pp_unet

    return pipe
```

**Важно:** при загрузке LoRA через `peft`, адаптерные веса (A, B матрицы) автоматически перемещаются вместе с базовыми слоями через `.to(device)` в конструкторе `PipelineParallelUNetSDXL`.

### 8.2 ControlNet

ControlNet — отдельная модель, которая параллельно UNet. Для pipeline parallel:

```python
from diffusers import ControlNetModel

def setup_controlnet_pp(
    pipe,
    controlnet_path: str,
    device_down: str = "cuda:0",
    device_up: str = "cuda:1",
):
    """
    ControlNet нужно продублировать или тоже разделить.

    Простой подход: запустить ControlNet на device_down,
    передать hint_embeddings на device_up.
    """
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float32)
    controlnet.to(device_down)  # ControlNet целиком на GPU 0

    pipe.controlnet = controlnet

    # При inference ControlNet forward идёт на device_down,
    # результаты (controlnet_residuals) пересылаются в Stage 1
    return pipe
```

> **Примечание:** полная интеграция ControlNet с pipeline parallel требует модификации forward pass — ControlNet residual'ы нужно добавлять к каждому down block output перед transfer на GPU 1. Это ~50 строк дополнительного кода.

---

## 9. Бенчмаркинг и тестирование

### 9.1 Скрипт для бенчмарков

```python
"""
benchmark.py
Сравнение скоростей: обычный vs pipeline parallel vs RAM offload.
"""

import torch
import time
from pipeline_parallel_sdxl import create_sdxl_pipeline_parallel


def benchmark(
    name: str,
    pipe,
    prompt: str = "a photo of a cat, detailed, 8k",
    steps: int = 25,
    warmup: int = 2,
    runs: int = 5,
):
    """Замер скорости генерации."""
    gen_fn = lambda seed: pipe(
        prompt=prompt,
        num_inference_steps=steps,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    )

    # Warmup
    for i in range(warmup):
        gen_fn(i)
        torch.cuda.synchronize()

    # Benchmark
    times = []
    for i in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gen_fn(warmup + i)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        print(f"  Run {i+1}: {t1-t0:.2f} сек")

    avg = sum(times) / len(times)
    per_step = avg / steps
    print(f"\n  [{name}] Среднее: {avg:.2f} сек | {per_step:.2f} сек/шаг")
    return avg


def run_all_benchmarks():
    from diffusers import StableDiffusionXLPipeline

    prompt = "a photograph of a cat sitting on a windowsill, golden hour light, detailed fur texture, 8k, professional photography"
    model_path = "./models/sdxl-base-fp16"
    steps = 25

    # 1. FP16 на 1 GPU (базовый, сломанный FP16 на P40)
    print("=" * 60)
    print("1. SDXL FP16, 1 GPU (cuda:0) — сломанный FP16 на Pascal")
    print("=" * 60)
    pipe_fp16 = StableDiffusionXLPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16,
    ).to("cuda:0")
    pipe_fp16.safety_checker = None
    benchmark("FP16 1GPU", pipe_fp16, prompt, steps)

    del pipe_fp16
    torch.cuda.empty_cache()

    # 2. FP32 pipeline parallel, 2 GPU
    print("\n" + "=" * 60)
    print("2. SDXL FP32, Pipeline Parallel, 2 GPU (cuda:0 + cuda:1)")
    print("=" * 60)
    pipe_pp = create_sdxl_pipeline_parallel(
        model_path, "cuda:0", "cuda:1", use_fp32=True,
    )
    benchmark("FP32 PP 2GPU", pipe_pp, prompt, steps)

    del pipe_pp
    torch.cuda.empty_cache()

    # 3. FP32 с model CPU offload (для сравнения)
    print("\n" + "=" * 60)
    print("3. SDXL FP32, CPU Offload, 1 GPU (cuda:0)")
    print("=" * 60)
    pipe_offload = StableDiffusionXLPipeline.from_pretrained(
        model_path, torch_dtype=torch.float32,
    )
    pipe_offload.enable_model_cpu_offload()
    pipe_offload.safety_checker = None
    benchmark("FP32 Offload 1GPU", pipe_offload, prompt, steps)


if __name__ == "__main__":
    run_all_benchmarks()
```

### 9.2 Ожидаемые результаты

```
1. FP16 1GPU:        ~50-60 сек (сломанный FP16, медленно)
2. FP32 PP 2GPU:     ~40-50 сек (полноценный FP32 compute)
3. FP32 Offload 1GPU: ~120-300 сек (PCIe + RAM bottleneck)

Вывод: Pipeline Parallel FP32 — самый быстрый вариант на P40 для SDXL.
```

---

## 10. Troubleshooting

### CUDA Out of Memory

```
Ошибка: torch.cuda.OutOfMemoryError: CUDA out of memory

Решения:
1. Уменьшить размер: width=832, height=832 вместо 1024 (экономит ~40% VRAM)
2. Проверить, что TE тоже на device_down (не дублируется):
   pipe.text_encoder.to(device_down)
3. Добавить garbage collection между генерациями:
   import gc; gc.collect(); torch.cuda.empty_cache()
4. Проверить, что нет лишних .to() — каждый transfer создаёт копию
```

### Ошибка: «Expected tensor for argument #1 'indices' to be on the same device»

```
Причина: какой-то тензор остался на неправильном GPU.

Решение: в forward() PipelineParallelUNetSDXL убедись, что ВСЕ тензоры
перемещены через .to(self.device_up) перед Stage 1.
Особенно часто забывают:
  - added_cond_kwargs ('text_embeds', 'time_ids')
  - attention_mask
  - cross_attention_kwargs
```

### Очень медленно (дольше ожидаемого)

```
Проверки:
1. Действительно ли FP32?
   print(next(pipe.unet.parameters()).dtype)
   Должно быть torch.float32

2. Не включился ли случайно autocast?
   # Убедись, что нигде нет:
   with torch.cuda.amp.autocast():
   # Или отключи явно в pipe:
   pipe.unet = pipe.unet  # без обёрток

3. PCIe топология:
   nvidia-smi topo -m
   Если GPU на разных CPU (PXB), transfer медленнее (~6 GB/s вместо 12).

4. torch.compile помощь:
   pp_unet.down_blocks = nn.ModuleList([torch.compile(b) for b in pp_unet.down_blocks])
   # Первый запуск медленный (трассировка), последующие на 10-30% быстрее
```

### Модель не загружается (ошибка скачивания)

```
# Если HuggingFace недоступен, используй зеркало:
export HF_ENDPOINT=https://hf-mirror.com

# Или скачай вручную:
from huggingface_hub import snapshot_download
snapshot_download(
    "stabilityai/stable-diffusion-xl-base-1.0",
    local_dir="./models/sdxl-base-fp16",
)
```

### Quality check: FP32 vs FP16

```python
# Генерируй одно и то же изображение в FP32 и FP16, сравни:
from PIL import Image
import numpy as np

img_fp32 = Image.open("output_fp32.png")
img_fp16 = Image.open("output_fp16.png")

arr32 = np.array(img_fp32)
arr16 = np.array(img_fp16)

diff = np.abs(arr32.astype(float) - arr16.astype(float))
print(f"Max pixel diff: {diff.max()}")
print(f"Mean pixel diff: {diff.mean():.2f}")
# Ожидание: max diff < 5, mean < 1 — визуально неразличимо
```