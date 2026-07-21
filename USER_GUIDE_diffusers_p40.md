# User Guide: Генерация изображений через Diffusers на Tesla P40

> Гайд написан для Ubuntu 22.04, 4x Tesla P40, Python 3.10+. Предполагается, что NVIDIA драйверы уже установлены (`nvidia-smi` работает). Никакого предварительного опыта с diffusers не требуется.

---

## Содержание

1. [Что такое Diffusers и почему он](#1-что-такое-diffusers-и-почему-он)
2. [Установка с нуля](#2-установка-с-нуля)
3. [Первое изображение: SD 1.5](#3-первое-изображение-sd-15)
4. [Разбор параметров генерации](#4-разбор-параметров-генерации)
5. [SDXL: более качественные изображения](#5-sdxl-более-качественные-изображения)
6. [FP32 на P40: как и зачем](#6-fp32-на-p40-как-и-зачем)
7. [Использование LoRA](#7-использование-lora)
8. [Image-to-Image: вариации и редактирование](#8-image-to-image-вариации-и-редактирование)
9. [ControlNet: контроль композиции](#9-controlnet-контроль-композиции)
10. [Pipeline Parallel FP32: запуск готовой реализации](#10-pipeline-parallel-fp32-запуск-готовой-реализации)
11. [Пакетная генерация](#11-пакетная-генерация)
12. [Частые проблемы и решения](#12-частые-проблемы-и-решения)

---

## 1. Что такое Diffusers и почему он

### 1.1 Кратко

**Diffusers** — это библиотека от HuggingFace для работы с диффузионными моделями. Она предоставляет готовые Python классы (`Pipeline`) для генерации изображений, видео и аудио. Это «бэкенд» — программный интерфейс, а не GUI.

### 1.2 Diffusers vs ComfyUI vs WebUI (Automatic1111)

| Критерий | Diffusers (Python) | ComfyUI (Node GUI) | WebUI (Automatic1111) |
|---|---|---|---|
| **Интерфейс** | Код / скрипты | Визуальный node-редактор | Браузерный GUI |
| **Гибкость** | Максимум (полный контроль) | Высокая (ноды) | Средняя |
| **Мульти-GPU** | Полный контроль | Ограниченный | Минимальный |
| **FP32 на P40** | Легко настраивается | Сложно | Почти невозможно |
| **Pipeline Parallel** | Можно реализовать | Нет | Нет |
| **Автоматизация** | Отлично (скрипты, API) | Через API (ComfyUI API) | Через API |
| **LoRA** | Из коробки | Из коробки | Из коробки |
| **Кривая обучения** | Средняя (нужен Python) | Низкая (визуальный) | Низкая (GUI) |
| **Для чего** | Продакшен, автоматизация, эксперименты | Интерактивный монтаж | Быстрые эксперименты |

**Выбирай Diffusers, если:**
- Хочешь максимальный контроль над GPU и памятью
- Нужна автоматизация / пакетная обработка
- Хочешь custom решения (pipeline parallel, FP32 на Pascal)
- Планируешь интегрировать генерацию в свои проекты / API

**Выбирай ComfyUI, если:**
- Хочешь визуально собирать пайплайны
- Не хочешь писать код
- Нужны сложные workflows (inpainting, upscaling, ControlNet одновременно)

### 1.3 Как устроен Pipeline

```
Простая аналогия: фотограф в студии.

Text Encoder (CLIP):  «Переводчик» — переводит текстовый промпт
                      в числовое представление, которое понимает модель.
                      Текст "a red cat" → вектор [0.23, -1.45, 0.67, ...]

UNet (Диффузионная модель):  «Художник» — постепенно превращает шум
                      в изображение, используя подсказки от Text Encoder.
                      Шаг 1: чистый шум → Шаг 2: что-то похожее → ... → Шаг 25: кот

Scheduler:  «Режиссёр» — определяет, КАК именно убирать шум на каждом шаге.
            Разные scheduler'ы дают разный баланс скорость/качество.
            DDIM, Euler, DPM++ — это имена разных scheduler'ов.

VAE (Декодер):  «Принтер» — преобразует латентное представление
                (компрессированное изображение) в полноценный PNG/JPEG.
                UNet работает в «сжатом» пространстве (4 канала × 64×64),
                VAE разжимает это в 3 канала × 512×512 (или 1024×1024 для SDXL).

Порядок работы:
  Промпт → Text Encoder → (вместе со случайным шумом) →
  UNet × N шагов → VAE → PNG файл
```

---

## 2. Установка с нуля

### 2.1 Создание окружения

```bash
# Проверь, что GPU видны:
nvidia-smi
# Должно показать 4x Tesla P40

# Создай изолированное Python-окружение:
python3 -m venv ~/diffusion-env
source ~/diffusion-env/bin/activate

# Обнови pip:
pip install --upgrade pip
```

### 2.2 Установка PyTorch (с поддержкой CUDA)

```bash
# Проверь версию CUDA (должна быть 11.8+):
nvidia-smi | grep "CUDA Version"
# Показывает версию драйвера, а не toolkit. Для Pascal достаточно CUDA 12.1.

# Установи PyTorch с CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Проверь установку:
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'  GPU {i}: {p.name}, {p.total_mem/1e9:.1f} GB')
"
```

Ожидаемый вывод:
```
PyTorch 2.x.x
CUDA: True
GPU count: 4
  GPU 0: Tesla P40, 24.0 GB
  GPU 1: Tesla P40, 24.0 GB
  GPU 2: Tesla P40, 24.0 GB
  GPU 3: Tesla P40, 24.0 GB
```

### 2.3 Установка Diffusers

```bash
pip install diffusers transformers accelerate safetensors

# Опциональные зависимости:
pip install peft            # для LoRA
pip install controlnet-aux  # для ControlNet препроцессоров
pip install compel          # для продвинутых промптов
pip install image           # Pillow для сохранения изображений (обычно уже установлен)
```

### 2.4 Скачивание первой модели

```bash
# Создай папку для моделей:
mkdir -p ~/models

# Скачай SD 1.5 (базовая модель, ~1.7 GB в FP16):
python3 -c "
from diffusers import StableDiffusionPipeline
print('Скачивание SD 1.5 (первый раз может занять несколько минут)...')
pipe = StableDiffusionPipeline.from_pretrained(
    'stable-diffusion-v1-5/stable-diffusion-v1-5',
    torch_dtype=torch.float32,  # FP32 для P40!
)
pipe.save_pretrained('./models/sd15-fp32')
print('Готово! Модель сохранена в ./models/sd15-fp32')
"
```

Если HuggingFace недоступен из-за блокировок:
```bash
# Вариант 1: зеркало
export HF_ENDPOINT=https://hf-mirror.com

# Вариант 2: скачай через huggingface-cli
pip install huggingface-hub
huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --local-dir ./models/sd15-fp16

# Вариант 3: скачай safetensors файл вручную с зеркала и загрузи:
# Скачал файл v1-5-pruned-emaonly.safetensors (~1.7 GB)
python3 -c "
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_single_file(
    './v1-5-pruned-emaonly.safetensors',
    torch_dtype=torch.float32,
)
pipe.save_pretrained('./models/sd15-fp32')
"
```

---

## 3. Первое изображение: SD 1.5

### 3.1 Минимальный скрипт

Создай файл `generate.py`:

```python
"""
generate.py — самая простая генерация изображения SD 1.5 на P40.
"""

import torch
from diffusers import StableDiffusionPipeline

# 1. Загрузка модели
# from_pretrained — загружает из локальной папки
# torch_dtype=torch.float32 — КАЖДОЕ слово здесь важно для P40
pipe = StableDiffusionPipeline.from_pretrained(
    "./models/sd15-fp32",
    torch_dtype=torch.float32,   # FP32 — родной формат для Pascal
)

# 2. Размещение на GPU
pipe = pipe.to("cuda:0")  # Используем первую P40

# 3. Отключаем safety checker (экономим ~700 MB VRAM)
pipe.safety_checker = None

# 4. Генерация
prompt = "a photograph of a cat sitting on a windowsill, golden hour light, detailed fur"

image = pipe(
    prompt=prompt,
    num_inference_steps=20,  # Количество шагов денойзинга (default: 50)
    guidance_scale=7.5,       # CFG scale (default: 7.5)
    generator=torch.Generator(device="cpu").manual_seed(42),  # Seed для повторяемости
).images[0]

# 5. Сохранение
image.save("my_first_image.png")
print("Сохранено: my_first_image.png")
```

Запуск:
```bash
source ~/diffusion-env/bin/activate
python generate.py
```

Ожидаемый вывод:
```
100%|████████████████████████████| 20/20 [00:08<00:00,  2.50it/s]
Сохранено: my_first_image.png
```

Первый запуск может быть медленнее (PyTorch компилирует CUDA kernels). Последующие — быстрее.

### 3.2 Использование scheduler'а DDIM (рекомендуется)

DDIM — детерминированный scheduler, обычно даёт лучший результат за меньше шагов:

```python
from diffusers import DDIMScheduler

pipe = StableDiffusionPipeline.from_pretrained("./models/sd15-fp32", torch_dtype=torch.float32)

# Заменяем стандартный scheduler на DDIM
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Теперь можно использовать меньше шагов (15-20 вместо 50):
image = pipe(
    prompt="a cat on a windowsill, golden hour",
    num_inference_steps=15,  # DDIM хорошо работает с 15 шагами
    guidance_scale=7.5,
).images[0]
```

### 3.3 Использование нескольких GPU для параллельной генерации

Для SD 1.5 pipeline parallelism не нужен (модель влезает в 24 GB). Но можно генерировать на 4 GPU параллельно:

```python
"""generate_parallel_sd15.py — 4 изображения одновременно на 4 GPU."""
import torch
from diffusers import StableDiffusionPipeline
import threading
import time

def generate_on_gpu(gpu_id, prompt, seed, output_path):
    pipe = StableDiffusionPipeline.from_pretrained(
        "./models/sd15-fp32", torch_dtype=torch.float32,
    ).to(f"cuda:{gpu_id}")
    pipe.safety_checker = None

    image = pipe(
        prompt=prompt,
        num_inference_steps=20,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).images[0]
    image.save(output_path)
    print(f"  [GPU {gpu_id}] {output_path} готов")

prompts = [
    ("a cat on a windowsill", 42, "cat.png"),
    ("a dog in a park", 43, "dog.png"),
    ("a bird on a branch", 44, "bird.png"),
    ("a fish in a coral reef", 45, "fish.png"),
]

print(f"Запуск {len(prompts)} генераций на 4 GPU параллельно...")
t0 = time.perf_counter()

threads = []
for gpu_id, (prompt, seed, path) in enumerate(prompts):
    t = threading.Thread(target=generate_on_gpu, args=(gpu_id, prompt, seed, path))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

t1 = time.perf_counter()
print(f"\nВсе готово за {t1-t0:.1f} сек ({len(prompts)/(t1-t0)*60:.0f} img/min)")
```

---

## 4. Разбор параметров генерации

### 4.1 Основные параметры

```python
image = pipe(
    prompt="a beautiful landscape with mountains, 8k, detailed",
    negative_prompt="blurry, low quality, distorted, ugly, watermark, text",
    num_inference_steps=20,
    guidance_scale=7.5,
    width=512,
    height=512,
    generator=torch.Generator(device="cpu").manual_seed(42),
)
```

| Параметр | Что делает | Рекомендуемые значения (SD 1.5) | Влияние на скорость |
|---|---|---|---|
| `prompt` | Текстовое описание | Английский, детальный | Нет |
| `negative_prompt` | Что НЕ должно быть | Стандартные негативы | Нет |
| `num_inference_steps` | Количество шагов денойзинга | 15-30 (SD 1.5), 20-30 (SDXL) | **Линейное** (20 шагов = 2× быстрее 40) |
| `guidance_scale` (CFG) | Насколько сильно промпт влияет | 7-12 (стандарт), 1-3 (более креативно), 15+ (строго по промпту) | Нет (но высокий CFG может потребовать больше шагов) |
| `width` / `height` | Размер изображения | Кратен 64: 512, 640, 768, 1024 | **Квадратичное** (1024² = 4× медленнее 512²) |
| `num_images_per_prompt` | Сколько картинок за один вызов | 1 (стандарт) | Линейное |

### 4.2 Guidance Scale (CFG) — подробнее

```
CFG = 1.0:   Модель игнорирует промпт, генерирует «что хочет»
CFG = 3.0:   Слабое следование промпту, более креативно
CFG = 7.5:   Баланс (дефолт)
CFG = 12.0:  Строго следует промпту, но может быть перенасыщенность
CFG = 20.0+:  Часто артефакты, «жареный» вид

Совет: для P40 используй CFG 7-10. Высокий CFG не замедляет,
но при малом числе шагов может давать худшее качество.
```

### 4.3 Seed и повторяемость

```python
# Одинаковый seed = одинаковое изображение (при тех же параметрах)
gen = torch.Generator(device="cpu").manual_seed(42)
image1 = pipe("a cat", generator=gen).images[0]

gen = torch.Generator(device="cpu").manual_seed(42)
image2 = pipe("a cat", generator=gen).images[0]
# image1 == image2 (пиксельно идентичны)

# Разный seed = разное изображение
gen = torch.Generator(device="cpu").manual_seed(43)
image3 = pipe("a cat", generator=gen).images[0]
# image3 != image1
```

---

## 5. SDXL: более качественные изображения

### 5.1 Скачивание

```bash
python3 -c "
from diffusers import StableDiffusionXLPipeline
print('Скачивание SDXL Base 1.0 (~6.5 GB в FP16)...')
pipe = StableDiffusionXLPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0',
    torch_dtype=torch.float16,  # Скачиваем в FP16 (компактнее)
)
pipe.save_pretrained('./models/sdxl-base-fp16')
print('Готово!')

# Также скачай SDXL Refiner (опционально, для двухэтапной генерации):
from diffusers import StableDiffusionXLImg2ImgPipeline
print('Скачивание SDXL Refiner 1.0...')
pipe_ref = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-refiner-1.0',
    torch_dtype=torch.float16,
)
pipe_ref.save_pretrained('./models/sdxl-refiner-fp16')
print('Готово!')
"
```

### 5.2 Базовая генерация SDXL

```python
"""generate_sdxl.py — SDXL генерация на P40."""
import torch
from diffusers import StableDiffusionXLPipeline, DDIMScheduler

# Загрузка (из FP16, конвертируем в FP32 для compute)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "./models/sdxl-base-fp16",
    torch_dtype=torch.float32,  # FP32 для P40!
)

# Scheduler
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# Размещение
pipe = pipe.to("cuda:0")
pipe.safety_checker = None
pipe.force_zeros_for_empty_prompt = False

# SDXL любит длинные, детальные промпты
prompt = (
    "A serene Japanese garden with a koi pond, cherry blossoms falling gently, "
    "golden hour sunlight filtering through maple trees, moss-covered stones, "
    "wooden arched bridge, photorealistic, 8k uhd, high detail, soft bokeh background"
)
negative = "blurry, low quality, distorted, ugly, watermark, text, deformed"

image = pipe(
    prompt=prompt,
    negative_prompt=negative,
    num_inference_steps=25,
    guidance_scale=7.5,
    width=1024,
    height=1024,
    generator=torch.Generator(device="cpu").manual_seed(42),
).images[0]

image.save("sdxl_garden.png")
print("Сохранено: sdxl_garden.png")
```

### 5.3 Двухэтапная генерация (Base + Refiner)

SDXL изначально проектировался как двухэтапный: Base генерирует, Refiner улучшает детали. На P40 это занимает ~2× времени, но качество выше.

```python
"""sdxl_two_stage.py — Base + Refiner."""
import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DDIMScheduler,
)

# Base (генерация)
base = StableDiffusionXLPipeline.from_pretrained(
    "./models/sdxl-base-fp16", torch_dtype=torch.float32,
).to("cuda:0")
base.scheduler = DDIMScheduler.from_config(base.scheduler.config)
base.safety_checker = None

# Refiner (улучшение) — на cuda:1
refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "./models/sdxl-refiner-fp16", torch_dtype=torch.float32,
).to("cuda:1")
refiner.safety_checker = None

prompt = "a photo of an astronaut floating above Earth, detailed spacesuit, stars, nebula"
negative = "blurry, low quality"

# Этап 1: Base генерирует латент
gen = torch.Generator(device="cpu").manual_seed(42)
latents = base(
    prompt=prompt,
    negative_prompt=negative,
    num_inference_steps=25,
    guidance_scale=7.5,
    width=1024,
    height=1024,
    generator=gen,
    output_type="latent",  # Не декодируем в PNG — передаём латент в Refiner
).images[0]

# Переносим латент на cuda:1
latents = latents.to("cuda:1")

# Этап 2: Refiner улучшает
image = refiner(
    prompt=prompt,
    negative_prompt=negative,
    image=latents,
    num_inference_steps=25,
    guidance_scale=7.5,
    generator=gen,
).images[0]

image.save("sdxl_astronaut_refined.png")
print("Сохранено: sdxl_astronaut_refined.png")
```

### 5.4 SDXL Turbo (быстрая генерация)

SDXL Turbo генерирует за 1-4 шага. Идеально для P40:

```python
"""sdxl_turbo.py — SDXL Turbo (1-4 шага, очень быстро)."""
import torch
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float32,
)
pipe = pipe.to("cuda:0")
pipe.safety_checker = None

# Turbo использует独特的 scheduler — не меняй его!
# И guidance_scale должен быть ~0.0 (или не указывай)
image = pipe(
    prompt="a cinematic photo of a cat warrior in ancient Egypt",
    num_inference_steps=4,      # 4 шага! (Turbo обучен так)
    guidance_scale=0.0,           # Важный параметр для Turbo
    generator=torch.Generator(device="cpu").manual_seed(42),
).images[0]

image.save("sdxl_turbo_cat.png")
print("Сохранено: sdxl_turbo_cat.png")
# Ожидаемое время: ~15-20 сек на P40 (vs ~50-60 сек для обычного SDXL)
```

---

## 6. FP32 на P40: как и зачем

### 6.1 Почему FP32

На Tesla P40 (Pascal, 2016):
- **FP32 compute:** 12 TFLOPS — нормальная, рабочая скорость
- **FP16 compute:** 0.18 TFLOPS — сломан (в 64 раза медленнее FP32!)
- Нет Tensor Cores

Когда ты загружаешь модель в FP16 (`torch_dtype=torch.float16`), PyTorch использует сломанный FP16 путь для вычислений. Результат: медленнее, чем FP32.

**Всегда используй `torch_dtype=torch.float32` на P40.**

### 6.2 Проверка, какой формат используется

```python
import torch
from diffusers import StableDiffusionPipeline

# Правильный способ для P40:
pipe = StableDiffusionPipeline.from_pretrained(
    "./models/sd15-fp32",
    torch_dtype=torch.float32,
)

# Проверь:
print(f"UNet dtype: {pipe.unet.dtype}")
# Должно быть: torch.float32

# Если torch.float16 — модель работает медленнее, чем должна!
```

### 6.3 Конвертация FP16 модели в FP32

Если ты скачал модель в FP16 (большинство моделей на HuggingFace — FP16):

```python
from diffusers import StableDiffusionPipeline

# Способ 1: конвертация при загрузке (сохраняет в FP32)
pipe = StableDiffusionPipeline.from_pretrained(
    "./models/sd15-fp16",  # FP16 модель
    torch_dtype=torch.float32,  # Конвертация в FP32
)
pipe.save_pretrained("./models/sd15-fp32")  # Сохраним для следующего раза

# Способ 2: явная конвертация
pipe = StableDiffusionPipeline.from_pretrained("./models/sd15-fp16")
pipe = pipe.to(torch.float32)  # Конвертация всех параметров
```

### 6.4 VRAM: что влезает в 24 GB

| Модель | FP16 | FP32 | Влезает на 1 P40? |
|---|---|---|---|
| SD 1.5 (U-Net + TE + VAE) | ~3.4 GB | ~6.8 GB | Оба формата ✓ |
| SDXL (U-Net + 2 TE + VAE) | ~12.7 GB | ~25.4 GB | FP16 ✓, FP32 ✗ |
| SDXL + активации (1024²) | ~15-16 GB | ~23-25 GB | FP16 ✓, FP32 на грани |
| SDXL Turbo | ~12.7 GB | ~25.4 GB | FP16 ✓, FP32 ✗ (нужен pipeline parallel) |

**Для SDXL в FP32 — используй Pipeline Parallel (см. раздел 10).**

---

## 7. Использование LoRA

### 7.1 Что такое LoRA

LoRA — это маленький файл (~50-300 MB), который модифицирует стиль или добавляет концепт в базовую модель. Его можно «накладывать» поверх любого чекпоинта.

### 7.2 Где найти LoRA

- **CivitAI** (https://civitai.com) — крупнейший репозиторий
  - Скачивай `.safetensors` файлы
  - Выбирай SD 1.5 или SDXL версии
- **HuggingFace** — поиск по "lora sdxl"

### 7.3 Загрузка и использование

```python
"""generate_with_lora.py — SD 1.5 + LoRA."""
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "./models/sd15-fp32", torch_dtype=torch.float32,
).to("cuda:0")
pipe.safety_checker = None

# Загрузка LoRA (путь к скачанному .safetensors файлу)
pipe.load_lora_weights("./loras/my_style.safetensors")

# Использование — промпт может содержать trigger word
image = pipe(
    prompt="a cat, in the style of MY_STYLE_TRIGGER",
    num_inference_steps=20,
    generator=torch.Generator(device="cpu").manual_seed(42),
).images[0]

image.save("cat_with_lora.png")
print("Сохранено: cat_with_lora.png")
```

### 7.4 Управление силой LoRA

```python
# По умолчанию LoRA применяется со scale=1.0
# Можно изменить:

# Слабое влияние:
pipe.load_lora_weights("./loras/my_style.safetensors", adapter_name="style")
pipe.set_adapters("style", weight=0.5)

# Сильное влияние:
pipe.set_adapters("style", weight=1.5)

# Несколько LoRA одновременно:
pipe.load_lora_weights("./loras/style_a.safetensors", adapter_name="style_a")
pipe.load_lora_weights("./loras/style_b.safetensors", adapter_name="style_b")

# Применить оба с разными весами:
pipe.set_adapters(["style_a", "style_b"], weights=[0.8, 0.5])

# Вернуть к базовой модели:
pipe.disable_lora()
```

### 7.5 SDXL + LoRA

```python
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "./models/sdxl-base-fp16", torch_dtype=torch.float32,
).to("cuda:0")
pipe.safety_checker = None

# SDXL LoRA загружается так же:
pipe.load_lora_weights("./loras/sdxl_my_style.safetensors")

image = pipe(
    prompt="a cyberpunk cityscape, neon lights, MY_TRIGGER_WORD",
    num_inference_steps=25,
    width=1024,
    height=1024,
).images[0]
```

---

## 8. Image-to-Image: вариации и редактирование

### 8.1 Базовое img2img

```python
"""img2img.py — генерация вариаций существующего изображения."""
import torch
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "./models/sd15-fp32", torch_dtype=torch.float32,
).to("cuda:0")
pipe.safety_checker = None

# Загрузи исходное изображение
init_image = Image.open("my_photo.jpg").resize((512, 512))

# strength: насколько сильно изменить (0.0 = не менять, 1.0 = полностью новое)
image = pipe(
    prompt="a watercolor painting of this scene, soft colors, artistic",
    image=init_image,
    strength=0.7,  # 70% изменения
    num_inference_steps=20,
    generator=torch.Generator(device="cpu").manual_seed(42),
).images[0]

image.save("watercolor_version.png")
```

### 8.2 Inpainting: замена части изображения

```python
"""inpainting.py — замена части изображения."""
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float32,
).to("cuda:0")
pipe.safety_checker = None

# Загрузи изображение и маску
image = Image.open("photo.jpg").resize((512, 512))
mask = Image.open("mask.png").resize((512, 512))
# Маска: белые пиксели = область для перерисовки, чёрные = оставить

result = pipe(
    prompt="a red hat",
    image=image,
    mask_image=mask,
    num_inference_steps=20,
).images[0]

result.save("inpainted.png")
```

### 8.3 Создание маски программно

```python
import numpy as np
from PIL import Image, ImageDraw

# Создать маску: закрасить область для перерисовки
mask = Image.new("RGB", (512, 512), (0, 0, 0))  # Чёрный фон
draw = ImageDraw.Draw(mask)
draw.ellipse([200, 150, 312, 262], fill=(255, 255, 255))  # Белый эллипс
# Всё внутри эллипса будет перерисовано

mask.save("mask.png")
```

---

## 9. ControlNet: контроль композиции

### 9.1 Что такое ControlNet

ControlNet — это дополнительная модель, которая принимает **структурную карту** (границы, глубину, скелет позы) и заставляет генерацию следовать этой структуре. Позволяет контролировать позу персонажа, композицию, расположение объектов.

### 9.2 Установка и загрузка

```bash
pip install controlnet-aux
```

### 9.3 Canny Edge: контроль по контурам

```python
"""controlnet_canny.py — генерация с контролем по контурам."""
import torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import CannyDetector

# Загрузка
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float32,
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "./models/sd15-fp32",
    controlnet=controlnet,
    torch_dtype=torch.float32,
).to("cuda:0")
pipe.safety_checker = None

# Исходное изображение → контуры (Canny edge detection)
source_image = Image.open("reference.jpg").resize((512, 512))
canny_detector = CannyDetector()
canny_map = canny_detector(source_image)
# canny_map — чёрно-белое изображение с контурами

# Генерация с контролем
image = pipe(
    prompt="a futuristic city at sunset, cyberpunk style, detailed",
    image=canny_map,  # Контурная карта управляет композицией
    num_inference_steps=20,
    guidance_scale=7.5,
    controlnet_conditioning_scale=1.0,  # Сила контроля (0-2)
    generator=torch.Generator(device="cpu").manual_seed(42),
).images[0]

image.save("controlnet_result.png")
```

### 9.4 Depth: контроль по глубине

```python
from controlnet_aux import MidasDetector

depth_detector = MidasDetector.from_pretrained("lllyasviel/Annotators")
depth_map = depth_detector(source_image)

image = pipe(
    prompt="a painting of a mountain landscape, oil on canvas",
    image=depth_map,
    num_inference_steps=20,
).images[0]
```

### 9.5 OpenPose: контроль позы персонажа

```python
from controlnet_aux import OpenposeDetector

pose_detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
pose_map = pose_detector(source_image)

image = pipe(
    prompt="a superhero in a dynamic action pose, detailed armor",
    image=pose_map,
    num_inference_steps=20,
).images[0]
```

---

## 10. Pipeline Parallel FP32: запуск готовой реализации

> Для SDXL в FP32 на P40 нужна реализация из Development Guide. Вот как её запустить.

### 10.1 Подготовка файлов

```bash
# У тебя должны быть:
ls -la
# pipeline_parallel_sdxl.py  (из Development Guide)
# models/sdxl-base-fp16/    (скачанная модель)

# Если файла pipeline_parallel_sdxl.py нет, создай его из Development Guide
```

### 10.2 Одиночная генерация на 2 GPU

```bash
# Базовый запуск:
python pipeline_parallel_sdxl.py

# С кастомными параметрами:
python -c "
from pipeline_parallel_sdxl import generate_sdxl

generate_sdxl(
    prompt='A dramatic mountain landscape at sunset, photorealistic, 8k',
    negative_prompt='blurry, low quality, distorted',
    model_path='./models/sdxl-base-fp16',
    device_down='cuda:0',
    device_up='cuda:1',
    steps=25,
    width=1024,
    height=1024,
    seed=42,
    use_fp32=True,
    output_path='my_sdxl_pp.png',
)
"
```

### 10.3 Батч на 4 GPU

```bash
# Создай файл generate_one.py (CLI-обёртка):
cat > generate_one.py << 'EOF'
import argparse
from pipeline_parallel_sdxl import generate_sdxl

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--gpu-down", type=int, default=0)
parser.add_argument("--gpu-up", type=int, default=1)
parser.add_argument("--prompt", required=True)
parser.add_argument("--negative", default="blurry, low quality")
parser.add_argument("--output", default="output.png")
parser.add_argument("--steps", type=int, default=25)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--width", type=int, default=1024)
parser.add_argument("--height", type=int, default=1024)
args = parser.parse_args()

generate_sdxl(
    prompt=args.prompt,
    negative_prompt=args.negative,
    model_path=args.model,
    device_down=f"cuda:{args.gpu_down}",
    device_up=f"cuda:{args.gpu_up}",
    steps=args.steps,
    width=args.width,
    height=args.height,
    seed=args.seed,
    use_fp32=True,
    output_path=args.output,
)
EOF

# Запуск 4 параллельных генераций:
python generate_one.py \
  --model ./models/sdxl-base-fp16 \
  --gpu-down 0 --gpu-up 1 \
  --prompt "A cyberpunk cityscape at night" \
  --output out_0.png &

python generate_one.py \
  --model ./models/sdxl-base-fp16 \
  --gpu-down 2 --gpu-up 3 \
  --prompt "A peaceful mountain lake at dawn" \
  --output out_1.png &

wait
echo "Готово: out_0.png, out_1.png"
```

### 10.4 SDXL Turbo через Pipeline Parallel

```python
"""sdxl_turbo_pp.py — SDXL Turbo через pipeline parallel."""
from pipeline_parallel_sdxl import generate_sdxl

# SDXL Turbo — 4 шага, guidance=0.0
generate_sdxl(
    prompt="a cinematic photo of a cat warrior in ancient Egypt, dramatic lighting",
    model_path="./models/sdxl-turbo",  # Скачай отдельно
    device_down="cuda:0",
    device_up="cuda:1",
    steps=4,
    guidance_scale=0.0,  # Важно для Turbo!
    use_fp32=True,
    output_path="turbo_cat_pp.png",
)
```

---

## 11. Пакетная генерация

### 11.1 Генерация из списка промптов

```python
"""batch_generate.py — пакетная генерация с прогрессом."""
import torch
import time
import os
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "./models/sd15-fp32", torch_dtype=torch.float32,
).to("cuda:0")
pipe.safety_checker = None

prompts = [
    "a cat sitting on a windowsill, golden hour, 8k, detailed fur",
    "a dog playing in autumn leaves, warm sunlight, photorealistic",
    "a bird singing on a cherry blossom branch, spring, soft focus",
    "a horse galloping through a meadow, dramatic sky, motion blur",
    "a fish swimming in a coral reef, crystal clear water, vibrant colors",
]

os.makedirs("batch_output", exist_ok=True)

for i, prompt in enumerate(prompts):
    t0 = time.perf_counter()
    image = pipe(
        prompt=prompt,
        num_inference_steps=20,
        generator=torch.Generator(device="cpu").manual_seed(i),
    ).images[0]
    t1 = time.perf_counter()

    path = f"batch_output/{i:03d}.png"
    image.save(path)
    print(f"[{i+1}/{len(prompts)}] {path} ({t1-t0:.1f} сек)")

print(f"\nВсего {len(prompts)} изображений в ./batch_output/")
```

### 11.2 Генерация с CSV файлом промптов

```python
"""batch_from_csv.py — чтение промптов из CSV."""
import csv
import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
    "./models/sd15-fp32", torch_dtype=torch.float32,
).to("cuda:0")
pipe.safety_checker = None

with open("prompts.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        prompt = row["prompt"]
        seed = int(row.get("seed", i))
        filename = row.get("filename", f"{i:03d}.png")

        image = pipe(
            prompt=prompt,
            num_inference_steps=20,
            generator=torch.Generator(device="cpu").manual_seed(seed),
        ).images[0]
        image.save(f"batch_output/{filename}")
        print(f"[{i}] {filename}")
```

Формат `prompts.csv`:
```csv
prompt,seed,filename
a cat sitting on a windowsill,42,cat_windowsill.png
a dog in a park with autumn leaves,43,dog_autumn.png
a mountain landscape at sunset,44,mountain_sunset.png
```

---

## 12. Частые проблемы и решения

### 12.1 «CUDA out of memory»

```
Ошибка:
  torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate X GiB

Решения (в порядке приоритета):
  1. Уменьши размер: width=512, height=512 вместо 1024
  2. Уменьши batch: num_images_per_prompt=1
  3. Отключи safety_checker: pipe.safety_checker = None
  4. Для SDXL — используй Pipeline Parallel (раздел 10)
  5. Очисти кэш между генерациями:
     import gc; gc.collect(); torch.cuda.empty_cache()
  6. Проверь, что FP32, а не FP64:
     print(pipe.unet.dtype)  # должно быть torch.float32
```

### 12.2 Генерация очень медленная

```
Проверки:
  1. Убедись, что FP32:
     print(pipe.unet.dtype)  # torch.float32, НЕ torch.float16

  2. Не включён ли autocast?
     # Если где-то есть with torch.cuda.amp.autocast(): — убери

  3. Какой scheduler?
     pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
     # DDIM быстрее PNDM по умолчанию

  4. Сколько шагов?
     # SD 1.5: 15-20 достаточно
     # SDXL: 20-25
     # SDXL Turbo: 4

  5. Какой размер?
     # 512×512 в 4× быстрее, чем 1024×1024
```

### 12.3 Качество плохое / артефакты

```
Возможные причины:
  1. Слишком мало шагов → увеличь до 20-25
  2. Слишком высокий CFG (>15) → снизь до 7-10
  3. Слишком низкий CFG (<3) → увеличь до 7-10
  4. Плохой промпт → добавь деталей, укажи стиль, освещение
  5. Модель не подходит для задачи → попробуй другой чекпоинт
```

### 12.4 «module 'diffusers' has no attribute ...»

```
Версия diffusers устарела:
  pip install --upgrade diffusers transformers accelerate
```

### 12.5 Модель не скачивается (HuggingFace недоступен)

```bash
# Вариант 1: зеркало
export HF_ENDPOINT=https://hf-mirror.com
python generate.py

# Вариант 2: ручная загрузка
pip install huggingface-hub
huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5 \
  --local-dir ./models/sd15-fp16

# Вариант 3: через API-токен (если есть)
huggingface-cli login
# Вставь токен с https://huggingface.co/settings/tokens
```

### 12.6 Полезные команды для мониторинга

```bash
# GPU utilisation в реальном времени:
watch -n 1 nvidia-smi

# Более детально:
nvidia-smi dmon -s pucm

# VRAM для каждого процесса:
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv

# Проверить PCIe топологию:
nvidia-smi topo -m
```

### 12.7 Чеклист перед первой генерацией

```
[ ] nvidia-smi показывает 4x Tesla P40
[ ] Python 3.10+ установлен
[ ] venv активирован: source ~/diffusion-env/bin/activate
[ ] PyTorch видит CUDA: python -c "import torch; print(torch.cuda.is_available())"
[ ] diffusers установлен: pip show diffusers
[ ] Модель скачана: ls ./models/sd15-fp32/
[ ] torch_dtype=torch.float32 указан
[ ] .to("cuda:0") указан
[ ] safety_checker отключён для экономии VRAM
```