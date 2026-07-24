"""
test_dit_placement_planner.py - Unit tests for the DiT placement planner.

Pure-Python (no torch required), so these run on the dev machine too.

Run:
    pytest test_dit_placement_planner.py -v
"""

from __future__ import annotations

import pytest

from dit_placement_planner import plan_placement


def test_small_model_equal_split():
    """PixArt-like model: 10 GB transformer, no T5, 2 GPUs -> 5+5 blocks."""
    plan = plan_placement(
        component_sizes={"transformer": 10.0},
        n_transformer_blocks=10,
        devices=["cuda:0", "cuda:1"],
        vram_gb=24.0,
        margin_gb=3.0,
    )
    assert plan.transformer_devices == [0, 1]
    assert plan.transformer_chunk_sizes == [5, 5]
    assert plan.t5_devices == []


def test_sd35_t5_is_sharded():
    """SD3.5 Large: T5 (23 GB) exceeds one GPU's 21 GB usable -> must shard."""
    plan = plan_placement(
        component_sizes={
            "text_encoder": 0.5,
            "text_encoder_2": 2.8,
            "text_encoder_3": 23.1,
            "transformer": 32.6,
            "vae": 0.3,
        },
        n_transformer_blocks=24,
        devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        vram_gb=24.0,
        margin_gb=3.0,
        n_t5_blocks=24,
    )
    assert len(plan.t5_devices) >= 2, "T5-XXL must be sharded across >=2 GPUs"
    assert sum(plan.t5_chunk_sizes) == 24
    for gpu_idx, load in plan.per_gpu_load.items():
        assert load <= 21.0 + 0.01, f"GPU {gpu_idx} overloaded: {load:.1f} GB"


def test_sd35_transformer_fits_after_t5():
    """After T5 sharding, transformer must still fit across remaining GPUs."""
    plan = plan_placement(
        component_sizes={
            "text_encoder": 0.5,
            "text_encoder_2": 2.8,
            "text_encoder_3": 23.1,
            "transformer": 32.6,
            "vae": 0.3,
        },
        n_transformer_blocks=24,
        devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        vram_gb=24.0,
        margin_gb=3.0,
        n_t5_blocks=24,
    )
    assert sum(plan.transformer_chunk_sizes) == 24
    for chunk in plan.transformer_chunk_sizes:
        assert chunk >= 1


def test_sd35_describe_output():
    """The describe() table should mention all 4 GPUs."""
    plan = plan_placement(
        component_sizes={
            "text_encoder": 0.5,
            "text_encoder_2": 2.8,
            "text_encoder_3": 23.1,
            "transformer": 32.6,
            "vae": 0.3,
        },
        n_transformer_blocks=24,
        devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        vram_gb=24.0,
        margin_gb=3.0,
        n_t5_blocks=24,
    )
    desc = plan.describe(["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    assert "cuda:0" in desc
    assert "cuda:3" in desc
    assert "transformer" in desc


def test_small_t5_single_gpu():
    """When T5 fits one GPU's budget, it should NOT shard."""
    plan = plan_placement(
        component_sizes={
            "text_encoder": 0.5,
            "text_encoder_3": 5.0,
            "transformer": 10.0,
        },
        n_transformer_blocks=10,
        devices=["cuda:0", "cuda:1"],
        vram_gb=24.0,
        margin_gb=3.0,
        n_t5_blocks=10,
    )
    assert len(plan.t5_devices) == 1
    assert plan.t5_chunk_sizes == [10]


def test_single_gpu_no_sharding():
    """With 1 GPU, everything goes there (if it fits)."""
    plan = plan_placement(
        component_sizes={"transformer": 10.0},
        n_transformer_blocks=10,
        devices=["cuda:0"],
        vram_gb=24.0,
        margin_gb=3.0,
    )
    assert plan.transformer_devices == [0]
    assert plan.transformer_chunk_sizes == [10]


def test_impossibly_large_model_raises():
    """If the model can't fit across all GPUs, the planner should raise."""
    with pytest.raises(RuntimeError):
        plan_placement(
            component_sizes={
                "transformer": 100.0,
                "text_encoder_3": 100.0,
            },
            n_transformer_blocks=100,
            devices=["cuda:0", "cuda:1"],
            vram_gb=24.0,
            margin_gb=3.0,
            n_t5_blocks=100,
        )


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
