#!/usr/bin/env python3
"""
fix_dit_model_index.py — Починить ``_class_name`` в ``model_index.json`` DiT-модели.

Проблема: если gated DiT-модель (SD3.5, FLUX, ...) скачалась через fallback-класс
``StableDiffusionXLPipeline`` (из-за gated-repo ошибки детекции), то
``model_index.json`` содержит НЕВЕРНЫЙ ``_class_name``, и последующая загрузка
падает с "expected ['unet', ...] but only {'transformer', ...} were passed".

Эта утилита определяет правильный DiT-класс по структуре папок и переписывает
``_class_name``.  Веса не трогаются — операция мгновенная.

Использование:
    python fix_dit_model_index.py ./models/stabilityai--stable-diffusion-3.5-large
    python fix_dit_model_index.py --all            # просканировать все ./models/*
    python fix_dit_model_index.py --dry-run ./models/...
"""

from __future__ import annotations

import argparse
import json
import os
import sys


# Порядок важен: более специфичные классы раньше.
DIT_CLASSES = {
    3: "StableDiffusion3Pipeline",  # 3 text encoder'а (CLIP×2 + T5)
    2: "FluxPipeline",              # 2 text encoder'а (CLIP + T5)
    1: "PixArtAlphaPipeline",       # 1 text encoder (PixArt-α/Sigma, Sana, ...)
}


def has_component(model_dir: str, name: str) -> bool:
    sub = os.path.join(model_dir, name)
    return os.path.isdir(sub) or os.path.isfile(sub + ".json")


def count_text_encoders(model_dir: str) -> int:
    return sum(has_component(model_dir, f"text_encoder{i}") for i in ("", "_2", "_3"))


def detect_dit_class(model_dir: str) -> str | None:
    """Определить DiT-класс по структуре папок.  None если это не DiT."""
    if not has_component(model_dir, "transformer"):
        return None

    n = count_text_encoders(model_dir)

    # Уточнение для 1 TE: Sana/Lumina/AuraFlow vs PixArt.
    if n == 1:
        try:
            tc = os.path.join(model_dir, "transformer", "config.json")
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

    return DIT_CLASSES.get(n)


def read_class_name(model_dir: str) -> str | None:
    idx = os.path.join(model_dir, "model_index.json")
    if not os.path.isfile(idx):
        return None
    try:
        with open(idx, "r", encoding="utf-8") as f:
            return json.load(f).get("_class_name")
    except Exception:
        return None


def fix_model_index(model_dir: str, dry_run: bool = False) -> bool:
    """Починить model_index.json.  Возвращает True если что-то изменено."""
    idx_path = os.path.join(model_dir, "model_index.json")
    if not os.path.isfile(idx_path):
        print(f"  [SKIP] {model_dir}: нет model_index.json")
        return False

    detected = detect_dit_class(model_dir)
    if detected is None:
        print(f"  [SKIP] {model_dir}: нет transformer/ — не DiT модель")
        return False

    current = read_class_name(model_dir)
    if current == detected:
        print(f"  [OK]   {model_dir}: уже {detected}")
        return False

    print(f"  [FIX]  {model_dir}: {current!r} → {detected!r}")
    if dry_run:
        print("         (dry-run — не записываем)")
        return True

    with open(idx_path, "r", encoding="utf-8") as f:
        idx = json.load(f)
    idx["_class_name"] = detected
    tmp = idx_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2, ensure_ascii=False)
    os.replace(tmp, idx_path)
    print(f"         записано: {idx_path}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Починить _class_name в model_index.json DiT-модели (случай A).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры:\n"
            "  python fix_dit_model_index.py ./models/stabilityai--stable-diffusion-3.5-large\n"
            "  python fix_dit_model_index.py --all              # все ./models/*\n"
            "  python fix_dit_model_index.py --all --dry-run    # показать, не менять\n"
        ),
    )
    ap.add_argument("model_dir", nargs="?", help="Путь к папке модели")
    ap.add_argument("--all", action="store_true", help="Просканировать все подпапки ./models/")
    ap.add_argument("--models-dir", default="./models", help="Базовая папка моделей (default: ./models)")
    ap.add_argument("--dry-run", action="store_true", help="Только показать, не записывать")
    args = ap.parse_args()

    if not args.all and not args.model_dir:
        ap.error("укажите model_dir или --all")

    if args.all:
        base = args.models_dir
        if not os.path.isdir(base):
            print(f"Папка {base} не существует.")
            return 1
        targets = [
            os.path.join(base, d)
            for d in sorted(os.listdir(base))
            if os.path.isdir(os.path.join(base, d))
        ]
    else:
        targets = [args.model_dir]

    fixed = 0
    for t in targets:
        if fix_model_index(t, dry_run=args.dry_run):
            fixed += 1

    print(f"\nГотово: починено {fixed} из {len(targets)} моделей.")
    return 0


if __name__ == "__main__":
    sys.exit(main())