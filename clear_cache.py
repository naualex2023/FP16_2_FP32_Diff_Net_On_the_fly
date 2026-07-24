#!/usr/bin/env python3
"""
clear_cache.py — Аккуратная очистка кэша HuggingFace с подтверждением.

Показывает список всех моделей в кэше HF с размерами, спрашивает подтверждение
и удаляет выбранные (или все). Безопасен по умолчанию: ничего не удаляет без
явного подтверждения пользователя.

Использование:
    python clear_cache.py                 # интерактивный режим (показать + спросить)
    python clear_cache.py --list          # только показать, ничего не удалять
    python clear_cache.py --all -y        # удалить весь кэш без подтверждения
    python clear_cache.py --model "sd-legacy/stable-diffusion-v1-5" -y
    python clear_cache.py --prefix "stable-diffusion" -y

Кэш HF по умолчанию: ~/.cache/huggingface/hub/
Меняется через переменную окружения HF_HOME.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Расположение кэша HF: учитывает HF_HOME и XDG_CACHE_HOME.
def _hf_hub_dir() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "huggingface" / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


HUB_DIR = _hf_hub_dir()


def _dir_size(path: Path) -> int:
    """Размер директории в байтах."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _fmt_size(n: int) -> str:
    """Человекочитаемый размер."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _repo_id_from_dirname(name: str) -> str:
    """models--org--name → org/name."""
    if name.startswith("models--"):
        return name[len("models--"):].replace("--", "/")
    return name


def purge_hf_cache_entry(repo_id: str) -> bool:
    """Удалить одну модель из кэша HuggingFace по repo_id (org/name).

    Используется download_models.py / model_resolver.py после save_pretrained(),
    чтобы не держать дубликат в кэше HF.  Возвращает True если что-то удалено.
    Безопасно: если модели в кэше нет — просто вернёт False.
    """
    from pathlib import Path
    dirname = "models--" + repo_id.replace("/", "--")
    target = HUB_DIR / dirname
    if target.is_dir():
        shutil.rmtree(target)
        print(f"  [clear_cache] removed HF cache entry: {repo_id} ({dirname})")
        return True
    return False


def list_models() -> list[tuple[str, Path, int]]:
    """Возвращает [(repo_id, path, size_bytes), ...] отсортировано по размеру."""
    if not HUB_DIR.is_dir():
        return []
    entries = []
    for entry in sorted(HUB_DIR.iterdir()):
        if entry.is_dir() and entry.name.startswith("models--"):
            repo_id = _repo_id_from_dirname(entry.name)
            size = _dir_size(entry)
            entries.append((repo_id, entry, size))
    entries.sort(key=lambda x: x[2], reverse=True)
    return entries


def print_models(entries: list[tuple[str, Path, int]]) -> None:
    """Красивая таблица моделей."""
    if not entries:
        print(f"Кэш HF пуст: {HUB_DIR}")
        return
    print(f"\nКэш HuggingFace: {HUB_DIR}")
    print(f"{'#':>3}  {'Размер':>10}  {'repo_id':<50}")
    print("-" * 68)
    total = 0
    for i, (repo_id, _path, size) in enumerate(entries):
        print(f"{i + 1:>3}  {_fmt_size(size):>10}  {repo_id:<50}")
        total += size
    print("-" * 68)
    print(f"     {_fmt_size(total):>10}  ВСЕГО ({len(entries)} моделей)\n")


def delete_entry(path: Path, repo_id: str) -> bool:
    """Удаляет директорию модели. Возвращает True при успехе."""
    try:
        shutil.rmtree(path)
        print(f"  ✓ Удалено: {repo_id}")
        return True
    except Exception as e:
        print(f"  ✗ Ошибка при удалении {repo_id}: {e}", file=sys.stderr)
        return False


def confirm(prompt: str, default: bool = False) -> bool:
    """Да/нет с подтверждением."""
    suffix = " [Y/n] " if default else " [y/N] "
    answer = input(prompt + suffix).strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes", "да")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Аккуратная очистка кэша HuggingFace с подтверждением.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Примеры:\n"
            "  python clear_cache.py --list\n"
            "  python clear_cache.py                          # интерактивно\n"
            "  python clear_cache.py --model 'sd-legacy/stable-diffusion-v1-5' -y\n"
            "  python clear_cache.py --all -y                 # удалить всё\n"
        ),
    )
    ap.add_argument("--list", action="store_true", help="Только показать кэш, не удалять")
    ap.add_argument("--all", action="store_true", help="Удалить ВЕСЬ кэш HF")
    ap.add_argument("--model", metavar="REPO_ID", help="Удалить конкретную модель (repo_id, напр. org/name)")
    ap.add_argument("--prefix", metavar="SUBSTR", help="Удалить все модели, в чьём repo_id есть подстрока")
    ap.add_argument("-y", "--yes", action="store_true", help="Не спрашивать подтверждение")
    args = ap.parse_args()

    entries = list_models()

    # --- режим --list ---

    # --- режим --list ---
    if args.list:
        print_models(entries)
        return 0

    # Нет моделей — нечего делать.
    if not entries:
        print(f"Кэш HF пуст: {HUB_DIR}")
        return 0

    # --- выбор целей для удаления ---
    targets: list[tuple[str, Path, int]] = []

    if args.all:
        targets = entries
    elif args.model:
        wanted = args.model.strip().lower()
        targets = [e for e in entries if e[0].lower() == wanted]
        if not targets:
            print(f"Модель '{args.model}' не найдена в кэше. Доступные:")
            print_models(entries)
            return 1
    elif args.prefix:
        sub = args.prefix.strip().lower()
        targets = [e for e in entries if sub in e[0].lower()]
        if not targets:
            print(f"Нет моделей, содержащих '{args.prefix}'. Доступные:")
            print_models(entries)
            return 1
    else:
        # Интерактивный режим: показать и спросить номера.
        print_models(entries)
        print("Введите номера моделей для удаления через запятую (напр. 1,3,5),")
        print("или 'all' для удаления всех, или Enter для отмены:")
        choice = input("> ").strip().lower()
        if not choice:
            print("Отменено — ничего не удалено.")
            return 0
        if choice == "all":
            targets = entries
        else:
            try:
                indices = [int(x.strip()) - 1 for x in choice.split(",")]
            except ValueError:
                print(f"Некорректный ввод: {choice!r}")
                return 1
            for idx in indices:
                if 0 <= idx < len(entries):
                    targets.append(entries[idx])
                else:
                    print(f"Предупреждение: номер {idx + 1} вне диапазона, пропущен")

    if not targets:
        print("Ничего не выбрано для удаления.")
        return 0

    # --- подтверждение ---
    total_size = sum(t[2] for t in targets)
    print("\nБудут удалены:")
    for repo_id, _path, size in targets:
        print(f"  {_fmt_size(size):>10}  {repo_id}")
    print(f"  {'ИТОГО':>10}: {_fmt_size(total_size)} ({len(targets)} моделей)")
    print(f"Путь кэша: {HUB_DIR}\n")

    if not args.yes:
        if not confirm("Удалить эти модели из кэша HF?", default=False):
            print("Отменено пользователем.")
            return 0

    # --- удаление ---
    deleted = 0
    for repo_id, path, _size in targets:
        if delete_entry(path, repo_id):
            deleted += 1

    print(f"\nГотово: удалено {deleted} из {len(targets)} моделей, освобождено {_fmt_size(total_size)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())