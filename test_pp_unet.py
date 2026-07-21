"""
test_pp_unet.py — Offline unit tests for the pipeline-parallel UNet.

These tests verify:
  1. The PipelineParallelUNet produces output numerically identical to the
     original single-GPU UNet (on a tiny synthetic SDXL-shaped model).
  2. Stage placement is correct (stage-0 modules on device_down, stage-1 on
     device_up).
  3. The `_move` helper handles tensors/tuples/dicts/None.

They require a GPU but only a tiny one (the test builds a small UNet, not
SDXL).  Run with:  pytest test_pp_unet.py -v
"""

from __future__ import annotations

import pytest
import torch

from pp_unet import PipelineParallelUNet, _move


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for placement tests"
)


# ---------------------------------------------------------------------------
# Helper: build a tiny SDXL-shaped UNet
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_sdxl_unet():
    """A miniature SDXL-config UNet (tiny channels / small image)."""
    from diffusers import UNet2DConditionModel

    unet = UNet2DConditionModel(
        sample_size=16,
        in_channels=4,
        out_channels=4,
        block_out_channels=(32, 64, 128),       # 3 down/up blocks (SDXL-shaped)
        layers_per_block=1,
        transformer_layers_per_block=1,
        down_block_types=(
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
        ),
        up_block_types=(
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "UpBlock2D",
        ),
        cross_attention_dim=32,
        attention_head_dim=2,
        use_linear_projection=True,             # SDXL uses True
        addition_embed_type="text_time",
        addition_time_embed_dim=8,
        projection_class_embeddings_input_dim=64,  # 32 text + 32 time
        time_embedding_dim=128,
        norm_num_groups=1,
    )
    return unet.eval()


def _dummy_inputs(unet, batch=1, device="cpu", dtype=torch.float32):
    """Build sample/timestep/encoder_hidden_states/added_cond_kwargs."""
    sample = torch.randn(batch, 4, unet.config.sample_size, unet.config.sample_size, dtype=dtype)
    timestep = torch.tensor([500], dtype=dtype)
    encoder_hidden_states = torch.randn(batch, 5, unet.config.cross_attention_dim, dtype=dtype)
    added_cond_kwargs = {
        "text_embeds": torch.randn(batch, 32, dtype=dtype),
        "time_ids": torch.randn(batch, 4, dtype=dtype),
    }
    return sample, timestep, encoder_hidden_states, added_cond_kwargs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_move_helper():
    t = torch.zeros(2)
    assert _move(None, "cpu") is None
    assert _move(t, "cpu").device.type == "cpu"
    tpl = _move((t, t), "cpu")
    assert isinstance(tpl, tuple) and len(tpl) == 2
    d = _move({"a": t, "b": "str"}, "cpu")
    assert d["a"].device.type == "cpu" and d["b"] == "str"


def test_pp_unet_matches_reference(tiny_sdxl_unet):
    """PP UNet (both stages on same device) == reference UNet output."""
    dev = "cuda:0"
    unet = tiny_sdxl_unet.to(dev)
    sample, ts, ehs, ack = _dummy_inputs(unet, device=dev)
    sample, ts, ehs = sample.to(dev), ts.to(dev), ehs.to(dev)
    ack = {k: v.to(dev) for k, v in ack.items()}

    with torch.no_grad():
        ref = unet(sample, ts, ehs, added_cond_kwargs=ack).sample

        pp = PipelineParallelUNet(unet, device_down=dev, device_up=dev).to(dev)
        out = pp(sample, ts, ehs, added_cond_kwargs=ack).sample

    assert out.shape == ref.shape
    max_diff = (out - ref).abs().max().item()
    assert max_diff < 1e-4, f"PP UNet diverges from reference: max diff {max_diff}"


def test_stage_placement(tiny_sdxl_unet):
    """Stage-0 modules on device_down, stage-1 modules on device_up."""
    pp = PipelineParallelUNet(tiny_sdxl_unet, device_down="cuda:0", device_up="cuda:0")
    assert pp.conv_in.parameters().__next__().device.type == "cuda"
    assert pp.mid_block.parameters().__next__().device.type == "cuda"
    assert pp.conv_out.parameters().__next__().device.type == "cuda"


def test_dtype_property(tiny_sdxl_unet):
    pp = PipelineParallelUNet(tiny_sdxl_unet, device_down="cuda:0", device_up="cuda:0")
    assert pp.dtype == tiny_sdxl_unet.dtype


def test_pp_unet_scalar_timestep(tiny_sdxl_unet):
    """Regression: diffusers pipeline passes a scalar/0-d timestep.

    Previously, calling ``time_proj`` directly on a scalar raised
    'Timesteps should be a 1d-array'. This ensures scalar inputs work.
    """
    dev = "cuda:0"
    unet = tiny_sdxl_unet.to(dev)
    sample = torch.randn(1, 4, 16, 16, device=dev)
    ehs = torch.randn(1, 5, 32, device=dev)
    ack = {
        "text_embeds": torch.randn(1, 32, device=dev),
        "time_ids": torch.randn(1, 4, device=dev),
    }

    with torch.no_grad():
        pp = PipelineParallelUNet(unet, device_down=dev, device_up=dev).to(dev)
        # scalar int (what the SDXL scheduler actually passes at runtime)
        out_scalar = pp(sample, 500, ehs, added_cond_kwargs=ack).sample
        # 0-d tensor
        out_0d = pp(sample, torch.tensor(500, device=dev), ehs, added_cond_kwargs=ack).sample

    assert out_scalar.shape == sample.shape
    assert out_0d.shape == sample.shape
    assert torch.allclose(out_scalar, out_0d, atol=1e-5)


if __name__ == "__main__":
    # Allow running without pytest
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
