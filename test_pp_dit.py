"""
test_pp_dit.py — Offline unit tests for the pipeline-parallel DiT wrapper.

These tests verify:
  1. ``PipelineParallelDiT`` produces output numerically identical to the
     original single-GPU DiT (on a tiny synthetic ``Transformer2DModel``).
  2. Stage placement is correct: first-half blocks on ``device_down``,
     second-half blocks on ``device_up``.
  3. The split count is correct (blocks split in half).
  4. Scalar / 0-d / 1-d timesteps all work.

They require a GPU but only a tiny one (the test builds a small DiT, not a
30–50 GB model).  Run on the P40 server with:

    py_test test_pp_dit.py -v
"""

from __future__ import annotations

import pytest
import torch

from pp_dit import PipelineParallelDiT


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for placement tests"
)


# ---------------------------------------------------------------------------
# Helper: build a tiny DiT-shaped Transformer2DModel
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_dit():
    """A miniature diffusers ``Transformer2DModel`` with 4 transformer blocks."""
    from diffusers import Transformer2DModel

    model = Transformer2DModel(
        num_attention_heads=2,
        attention_head_dim=8,                 # inner_dim = 2 * 8 = 16
        in_channels=4,
        out_channels=4,
        num_layers=4,                         # → split 2 + 2
        cross_attention_dim=32,
        sample_size=16,
        patch_size=1,                         # no patchify (simplest path)
        activation_fn="geglu",
        num_embeds_ada_norm=1000,             # AdaLayerNorm → accepts timestep
        norm_type="ada_norm_single",          # timestep-conditioned norm
    )
    return model.eval()


def _dummy_inputs(batch=1, device="cpu", dtype=torch.float32):
    """Build hidden_states / timestep / encoder_hidden_states."""
    sample = torch.randn(batch, 4, 16, 16, dtype=dtype)
    timestep = torch.tensor([500], dtype=dtype)
    encoder_hidden_states = torch.randn(batch, 5, 32, dtype=dtype)
    return sample, timestep, encoder_hidden_states


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pp_dit_matches_reference(tiny_dit):
    """PP DiT (both stages on same device) == reference DiT output."""
    dev = "cuda:0"
    dit = tiny_dit.to(dev)
    sample, ts, ehs = _dummy_inputs(device=dev)
    sample, ts, ehs = sample.to(dev), ts.to(dev), ehs.to(dev)

    with torch.no_grad():
        ref = dit(
            sample, timestep=ts, encoder_hidden_states=ehs
        ).sample

        pp = PipelineParallelDiT(dit, device_down=dev, device_up=dev)
        out = pp(
            sample, timestep=ts, encoder_hidden_states=ehs
        ).sample

    assert out.shape == ref.shape
    max_diff = (out - ref).abs().max().item()
    assert max_diff < 1e-4, f"PP DiT diverges from reference: max diff {max_diff}"


def test_split_count(tiny_dit):
    """4 blocks → 2 + 2."""
    pp = PipelineParallelDiT(
        tiny_dit, device_down="cuda:0", device_up="cuda:0"
    )
    assert len(pp.first_half_layers) == 2
    assert len(pp.second_half_layers) == 2


def test_stage_placement(tiny_dit):
    """First-half blocks on device_down, second-half on device_up."""
    pp = PipelineParallelDiT(
        tiny_dit, device_down="cuda:0", device_up="cuda:0"
    )
    assert (
        next(pp.first_half_layers.parameters()).device.type == "cuda"
    )
    assert (
        next(pp.second_half_layers.parameters()).device.type == "cuda"
    )


def test_dtype_property(tiny_dit):
    pp = PipelineParallelDiT(
        tiny_dit, device_down="cuda:0", device_up="cuda:0"
    )
    assert pp.dtype == tiny_dit.dtype


def test_pp_dit_scalar_timestep(tiny_dit):
    """Regression: pipelines pass scalar / 0-d timesteps.

    Ensures ``int``, ``0-d tensor`` and ``1-d tensor`` timesteps all produce
    identical output.
    """
    dev = "cuda:0"
    dit = tiny_dit.to(dev)
    sample = torch.randn(1, 4, 16, 16, device=dev)
    ehs = torch.randn(1, 5, 32, device=dev)

    with torch.no_grad():
        pp = PipelineParallelDiT(dit, device_down=dev, device_up=dev)
        out_scalar = pp(sample, timestep=500, encoder_hidden_states=ehs).sample
        out_0d = pp(
            sample, timestep=torch.tensor(500, device=dev),
            encoder_hidden_states=ehs,
        ).sample
        out_1d = pp(
            sample, timestep=torch.tensor([500], device=dev),
            encoder_hidden_states=ehs,
        ).sample

    assert out_scalar.shape == sample.shape
    assert out_0d.shape == sample.shape
    assert torch.allclose(out_scalar, out_0d, atol=1e-5)
    assert torch.allclose(out_scalar, out_1d, atol=1e-5)


def test_output_returns_to_device_down(tiny_dit):
    """Result tensor must end up on device_down (where the scheduler loop lives)."""
    dev = "cuda:0"
    dit = tiny_dit.to(dev)
    sample = torch.randn(1, 4, 16, 16, device=dev)
    ehs = torch.randn(1, 5, 32, device=dev)

    with torch.no_grad():
        pp = PipelineParallelDiT(dit, device_down=dev, device_up=dev)
        out = pp(sample, timestep=500, encoder_hidden_states=ehs).sample

    assert out.device.type == "cuda"
    assert out.device == torch.device(dev)


if __name__ == "__main__":
    # Allow running without pytest:  py_test test_pp_dit.py -v
    import sys

    sys.exit(pytest.main([__file__, "-v"]))