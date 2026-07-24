"""
pp_dit.py — Pipeline-parallel DiT (Diffusion Transformer) for Tesla P40 (FP32).

This is the DiT analogue of :mod:`pp_unet`.  Models based on a Transformer
backbone (PixArt-Alpha/Sigma, Sana, Stable Diffusion 3, Flux, Lumina, …) rather
than a UNet can reach **30–50 GB in FP32** — far beyond a single 24 GB P40.
``PipelineParallelDiT`` splits the heavy sequence of *transformer blocks* in two
halves across two GPUs so such models run on a pair of P40s.

Stage 0 (device_down):
    Input projection / patch-embed, timestep embedding and the **first half**
    of the transformer blocks.

Stage 1 (device_up):
    The **second half** of the transformer blocks.

After stage 1 the activations are brought back to *device_down* so the (cheap)
final norm / output projection, unpatchify and any residual connection run on
the same device as the latents/scheduler loop — exactly like the UNet variant.

Design choice — why hooks instead of re-implementing ``forward``
----------------------------------------------------------------
The UNet wrapper (:class:`pp_unet.PipelineParallelUNet`) re-implements the whole
``UNet2DConditionModel.forward`` by hand.  That is tractable because the UNet
forward is stable.  DiT forwards are **not**: every diffusers family
(``Transformer2DModel``, ``SD3Transformer2DModel``, ``FluxTransformer2DModel``,
``SanaTransformer2DModel`` …) differs in its timestep embedding, unpatchify,
joint/single block layout and output norm.  Hand-copying each would be fragile.

Instead we keep the original transformer object and split only its
``transformer_blocks`` sequence across devices.  A ``forward_pre_hook`` on every
second-half block moves that block's inputs to ``device_up``; a ``forward_hook``
on the last second-half block moves its output back to ``device_down``.  The
model's own ``forward`` then does **all** the math unchanged, so the result is
bit-identical to a single-GPU run (the tests assert this).

Usage
-----
    from pp_dit import PipelineParallelDiT
    pipe.transformer = PipelineParallelDiT(pipe.transformer,
                                           device_down="cuda:0",
                                           device_up="cuda:1")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

# Re-use the recursive ``.to()`` helper from the UNet wrapper.
from pp_unet import _move

logger = logging.getLogger(__name__)


# Candidate attribute names that hold the block sequence on different diffusers
# transformer classes.  ``Transformer2DModel`` / ``SD3Transformer2DModel`` /
# ``SanaTransformer2DModel`` all expose ``transformer_blocks``.
_BLOCK_ATTRS: Tuple[str, ...] = ("transformer_blocks", "blocks")

# Input-side modules — kept on device_down (stage 0).
_INPUT_MODULES: Tuple[str, ...] = (
    "pos_embed",
    "proj_in",
    "time_proj",
    "time_pos_embed",
    "time_embed",
    "add_time_proj",
    "add_embedding",
    "encoder_hid_proj",
)

# Output-side modules — also kept on device_down (the post-hook brings the
# activations back to device_down before they run).  They are light; the heavy
# memory is in the blocks, which is what we split.
_OUTPUT_MODULES: Tuple[str, ...] = ("norm_out", "proj_out")


class PipelineParallelDiT(nn.Module):
    """Split a diffusers DiT (``Transformer2DModel`` and friends) across two GPUs.

    The transformer's block sequence (``transformer_blocks``) is cut in half:

        first_half_layers  → device_down
        second_half_layers → device_up

    All input-side modules (patch embed / ``proj_in``, timestep embedders) and
    output-side modules (``norm_out``, ``proj_out``) stay on *device_down*.
    Device transfers happen only at the block boundary.

    Parameters
    ----------
    dit : nn.Module
        The original DiT backbone (e.g. ``diffusers.Transformer2DModel``).  It is
        dissected in place — its blocks are moved to the target devices.
    device_down : str
        CUDA device for stage 0 (input embed + first half of blocks + output).
    device_up : str
        CUDA device for stage 1 (second half of blocks).
    """

    def __init__(
        self,
        dit: nn.Module,
        device_down: str = "cuda:0",
        device_up: str = "cuda:1",
        split_ratio: float = 0.5,
    ) -> None:
        super().__init__()

        self.device_down = torch.device(device_down)
        self.device_up = torch.device(device_up)

        # Keep a reference to the *whole* original transformer so we can delegate
        # ``forward`` to it.  Store it WITHOUT registering it as a child module
        # (nn.Module.__setattr__ would otherwise register it and double-count
        # parameters / move it on ``.to()``).
        object.__setattr__(self, "_dit", dit)

        # ---- locate the block sequence -------------------------------------
        blocks: Optional[list] = None
        for attr in _BLOCK_ATTRS:
            obj = getattr(dit, attr, None)
            if isinstance(obj, nn.ModuleList) and len(obj) > 0:
                blocks = list(obj)
                self._blocks_attr = attr
                break
        if not blocks:
            raise ValueError(
                f"{type(dit).__name__} has no non-empty 'transformer_blocks'/"
                f"'blocks' ModuleList to split."
            )

        split_idx = int(len(blocks) * split_ratio)  # configurable split point
        self.first_half_layers = nn.ModuleList(
            [blk.to(self.device_down) for blk in blocks[:split_idx]]
        )
        self.second_half_layers = nn.ModuleList(
            [blk.to(self.device_up) for blk in blocks[split_idx:]]
        )

        # ---- place input/output modules (light) on device_down -------------
        for name in _INPUT_MODULES:
            mod = getattr(dit, name, None)
            if isinstance(mod, nn.Module):
                mod.to(self.device_down)
                # Also register as a child so ``.eval()`` / ``.train()`` reach it.
                setattr(self, name, mod)
        for name in _OUTPUT_MODULES:
            mod = getattr(dit, name, None)
            if isinstance(mod, nn.Module):
                mod.to(self.device_down)
                setattr(self, name, mod)

        # ---- bookkeeping ---------------------------------------------------
        self.config = dit.config
        self._orig_cls = type(dit).__name__
        # ``lora_scale`` is read by the pipeline's LoRA scale decorator.
        self.lora_scale: float = 1.0

        # ---- device-boundary hooks ----------------------------------------
        self._install_hooks()

        logger.info(
            "PipelineParallelDiT created: orig=%s, blocks=%d (split %d + %d, ratio=%.2f), "
            "stage0=%s, stage1=%s",
            self._orig_cls,
            len(blocks),
            split_idx,
            len(blocks) - split_idx,
            split_ratio,
            self.device_down,
            self.device_up,
        )

    # ------------------------------------------------------------------
    # Hook installation
    # ------------------------------------------------------------------

    def _install_hooks(self) -> None:
        """Register device-transfer hooks on the second-half blocks.

        * A ``forward_pre_hook(with_kwargs=True)`` on every second-half block
          moves that block's tensor args/kwargs to ``device_up`` before it runs.
        * A ``forward_hook`` on the **last** second-half block moves its output
          back to ``device_down`` so the subsequent norm/proj_out (and any
          residual add) run on stage 0's device.
        """
        if len(self.second_half_layers) == 0:
            return

        device_up = self.device_up
        device_down = self.device_down

        def _pre_hook(_module, args, kwargs):
            new_args = tuple(_move(a, device_up) for a in args)
            new_kwargs = {
                k: _move(v, device_up) for k, v in kwargs.items()
            }
            return new_args, new_kwargs

        def _post_last(_module, _args, output):
            return _move(output, device_down)

        for i, blk in enumerate(self.second_half_layers):
            blk.register_forward_pre_hook(_pre_hook, with_kwargs=True)
            if i == len(self.second_half_layers) - 1:
                blk.register_forward_hook(_post_last)

    # ------------------------------------------------------------------
    # Properties required by diffusers pipelines
    # ------------------------------------------------------------------

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the first stage-0 parameter."""
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32

    @property
    def device(self) -> torch.device:
        """Return stage 0's device (where latents are managed)."""
        return self.device_down

    # ------------------------------------------------------------------
    # LoRA helpers (delegate to submodules)
    # ------------------------------------------------------------------

    def scale_lora_weights(self, scale: float) -> None:
        self.lora_scale = scale
        for module in self.modules():
            if hasattr(module, "scale_lora_weights"):
                module.scale_lora_weights(scale)

    def unscale_lora_weights(self) -> None:
        self.lora_scale = 1.0
        for module in self.modules():
            if hasattr(module, "unscale_lora_weights"):
                module.unscale_lora_weights()

    # ------------------------------------------------------------------
    # No-op stubs the pipeline might call
    # ------------------------------------------------------------------

    def enable_xformers_memory_efficient_attention(self) -> None:
        logger.warning(
            "enable_xformers_memory_efficient_attention() is a no-op on P40 FP32"
        )

    def set_attention_slice(self, slice_size="auto") -> None:
        for module in self.modules():
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size)

    def enable_gradient_checkpointing(self) -> None:
        for module in self.modules():
            if hasattr(module, "enable_gradient_checkpointing"):
                module.enable_gradient_checkpointing()

    # ------------------------------------------------------------------
    # Forward — delegate to the original transformer, with device hooks.
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.Tensor | int | float] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
        class_labels: Optional[torch.Tensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Pipeline-parallel forward.

        All inputs are moved to *device_down*; the actual computation is the
        original transformer's ``forward`` (numerically identical), while the
        installed hooks cross the device boundary at the block split point.  The
        returned sample ends up on *device_down*.
        """
        # ---- move inputs to stage 0 ---------------------------------------
        hidden_states = hidden_states.to(self.device_down)
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.to(self.device_down)
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.device_down)
        if class_labels is not None:
            class_labels = class_labels.to(self.device_down)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device_down)
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.to(self.device_down)
        if added_cond_kwargs is not None:
            added_cond_kwargs = _move(added_cond_kwargs, self.device_down)
        if cross_attention_kwargs is not None:
            cross_attention_kwargs = _move(cross_attention_kwargs, self.device_down)

        # ---- delegate to the original transformer (hooks do the split) -----
        out = self._dit(  # type: ignore[attr-defined]
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=return_dict,
            **kwargs,
        )

        # ---- defensive: ensure the result is back on device_down ----------
        if isinstance(out, torch.Tensor):
            return out.to(self.device_down)
        if hasattr(out, "sample"):
            if out.sample.device != self.device_down:
                out.sample = out.sample.to(self.device_down)
            return out
        # tuple (return_dict=False)
        return tuple(
            _move(t, self.device_down) if isinstance(t, torch.Tensor) else t
            for t in out
        )