"""
test_dit_size_estimator.py — Unit tests for the DiT size estimator.

Pure-Python (no torch required), so these run on the dev machine too.

Run:
    pytest test_dit_size_estimator.py -v
"""

from __future__ import annotations

import json
import os

import pytest

from dit_size_estimator import (
    BYTES_PER_PARAM,
    estimate_model_size_gb,
    on_disk_dtype,
    sum_weight_bytes,
)


# ---------------------------------------------------------------------------
# Fixtures: build a fake diffusers model directory on disk
# ---------------------------------------------------------------------------

@pytest.fixture
def fake_model_dir(tmp_path):
    """Create a minimal model directory tree with dummy checkpoint files.

    Layout (sizes chosen so arithmetic is easy to verify):
        transformer/diffusion_pytorch_model.safetensors   10 GB (FP16 on disk)
        transformer/config.json                            {"torch_dtype": "float16"}
        text_encoder_3/model.safetensors                   10 GB (FP16 on disk)
        text_encoder_3/config.json                         {"torch_dtype": "float16"}
        vae/diffusion_pytorch_model.bin                    1 GB  (FP16 on disk)
        vae/config.json                                    {"torch_dtype": "float16"}
    """
    def _mk(component, files, config):
        d = tmp_path / component
        d.mkdir()
        for name, gb in files.items():
            # Create a sparse file of the requested size (instant, no disk use).
            p = d / name
            p.touch()
            os.truncate(str(p), int(gb * 1e9))
        (d / "config.json").write_text(json.dumps(config))
        return d

    _mk("transformer", {"diffusion_pytorch_model.safetensors": 10.0},
        {"torch_dtype": "float16"})
    _mk("text_encoder_3", {"model.safetensors": 10.0},
        {"torch_dtype": "float16"})
    _mk("vae", {"diffusion_pytorch_model.bin": 1.0},
        {"torch_dtype": "float16"})
    return tmp_path


# ---------------------------------------------------------------------------
# sum_weight_bytes
# ---------------------------------------------------------------------------

def test_sum_weight_bytes_counts_all_extensions(fake_model_dir):
    """Should count .safetensors and .bin in a component dir."""
    t_bytes = sum_weight_bytes(str(fake_model_dir / "transformer"))
    assert t_bytes == int(10.0 * 1e9)

    v_bytes = sum_weight_bytes(str(fake_model_dir / "vae"))
    assert v_bytes == int(1.0 * 1e9)


def test_sum_weight_bytes_empty_dir(tmp_path):
    assert sum_weight_bytes(str(tmp_path)) == 0


def test_sum_weight_bytes_ignores_non_checkpoints(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "README.md").write_text("hello")
    assert sum_weight_bytes(str(tmp_path)) == 0


# ---------------------------------------------------------------------------
# on_disk_dtype
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("float16", "float16"),
    ("torch.float16", "float16"),
    ("fp16", "float16"),
    ("half", "float16"),
    ("float32", "float32"),
    ("torch.float32", "float32"),
    ("fp32", "float32"),
    ("bfloat16", "bfloat16"),
    ("torch.bfloat16", "bfloat16"),
    ("bf16", "bfloat16"),
    ("", "float16"),        # default
    ("unknown", "float16"),
    (None, "float16"),
])
def test_on_disk_dtype(raw, expected):
    assert on_disk_dtype({"torch_dtype": raw}) == expected


# ---------------------------------------------------------------------------
# estimate_model_size_gb — primary (files on disk) path
# ---------------------------------------------------------------------------

def test_estimate_fp32_scales_fp16_on_disk(fake_model_dir):
    """FP16 files on disk → ×2 scaling when target_dtype='float32'."""
    sizes = estimate_model_size_gb(str(fake_model_dir), target_dtype="float32")
    # 10 GB fp16 → 20 GB fp32 for transformer and text_encoder_3; 1 GB → 2 GB for vae.
    assert sizes is not None
    assert sizes["transformer"] == pytest.approx(20.0, abs=0.01)
    assert sizes["text_encoder_3"] == pytest.approx(20.0, abs=0.01)
    assert sizes["vae"] == pytest.approx(2.0, abs=0.01)
    assert pytest.approx(sum(sizes.values()), abs=0.05) == 42.0


def test_estimate_fp16_target_no_scaling(fake_model_dir):
    """target_dtype='float16' matches on-disk dtype → no scaling."""
    sizes = estimate_model_size_gb(str(fake_model_dir), target_dtype="float16")
    assert sizes is not None
    assert sizes["transformer"] == pytest.approx(10.0, abs=0.01)
    assert sizes["text_encoder_3"] == pytest.approx(10.0, abs=0.01)
    assert sizes["vae"] == pytest.approx(1.0, abs=0.01)


def test_estimate_missing_component_omitted(fake_model_dir):
    """Components with no dir should simply be absent from the result."""
    sizes = estimate_model_size_gb(str(fake_model_dir), target_dtype="float32")
    assert "text_encoder" not in sizes
    assert "text_encoder_2" not in sizes


def test_estimate_returns_none_for_empty_dir(tmp_path):
    assert estimate_model_size_gb(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# estimate_model_size_gb — fallback (no weight files) path
# ---------------------------------------------------------------------------

@pytest.fixture
def heuristic_model_dir(tmp_path):
    """Model dir with only config.json files (no checkpoints) → heuristic path."""
    tdir = tmp_path / "transformer"
    tdir.mkdir()
    # hidden=2048, layers=24 → 12*2048^2*24 / 1e9 ≈ 1.21 B params
    (tdir / "config.json").write_text(json.dumps({
        "hidden_size": 2048,
        "num_layers": 24,
        "vocab_size": 0,
    }))
    return tmp_path


def test_estimate_falls_back_to_heuristic(heuristic_model_dir):
    sizes = estimate_model_size_gb(str(heuristic_model_dir), target_dtype="float32")
    assert sizes is not None
    assert "transformer" in sizes
    expected_bn = 12 * 2048 * 2048 * 24 / 1e9
    assert sizes["transformer"] == pytest.approx(expected_bn * 4, abs=0.01)


# ---------------------------------------------------------------------------
# Regression: the SD3.5 Large scenario that motivated this rewrite
# ---------------------------------------------------------------------------

def test_sd35_scenario_needs_multiple_gpus(tmp_path):
    """Reproduce SD3.5-Large-like sizing and assert it exceeds one 24 GB GPU.

    Previously the heuristic reported ~14.5 GB (→ 1 GPU → OOM).  Measuring
    real files must report ~62 GB FP32, requiring 4 P40s.
    """
    def _mk(comp, fp16_gb):
        d = tmp_path / comp
        d.mkdir()
        f = d / "model.safetensors"
        f.touch()
        os.truncate(str(f), int(fp16_gb * 1e9))
        (d / "config.json").write_text(json.dumps({"torch_dtype": "float16"}))

    # Approximate real FP16 on-disk sizes for SD3.5 Large.
    _mk("transformer", 16.0)       # ~32 GB FP32
    _mk("text_encoder", 0.4)       # CLIP-L
    _mk("text_encoder_2", 2.5)     # CLIP-G
    _mk("text_encoder_3", 9.4)     # T5-XXL → ~18.8 GB FP32 (was 0 before!)
    _mk("vae", 0.5)

    sizes = estimate_model_size_gb(str(tmp_path), target_dtype="float32")
    assert sizes is not None
    total = sum(sizes.values())
    # Must exceed a single 24 GB P40's usable VRAM (~18 GB).
    assert total > 18.0, f"Expected >18 GB FP32, got {total:.1f} GB"
    # T5-XXL must be present (the original bug dropped it entirely).
    assert "text_encoder_3" in sizes
    assert sizes["text_encoder_3"] > 15.0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))