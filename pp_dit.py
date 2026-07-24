"""
pp_dit.py — Pipeline-parallel DiT (Diffusion Transformer) for Tesla P40 (FP32).

Splits a diffusers Transformer backbone (PixArt/Sana/SD3/Flux/...) across **N
GPUs** (1-4) so that large FP32 models (30-60 GB) can run on 4× Tesla P40.

Stage 0 (devices[0]):
    Input projection / patch-embed, timestep embedding and the first chunk of
    transformer blocks.

Stage i (devices[i], i > 0):
    Subsequent chunks of transformer blocks.

Stage N-1 returns activations to devices[0] for final norm / output projection.

Design: hook-based (see pp_unet for the 2-GPU UNet analogue). We keep the
original transformer object and register forward_pre_hooks / forward_hooks on
the block chunks to move activations across device boundaries. The model's own
forward does all the math unchanged → numerically identical to single-GPU.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from pp_unet import _move

logger = logging.getLogger(__name__)

_BLOCK_ATTRS: Tuple[str, ...] = ("transformer_blocks", "blocks")
_INPUT_MODULES: Tuple[str, ...] = (
    "pos_embed", "proj_in", "time_proj", "time_pos_embed", "time_embed",
    "add_time_proj", "add_embedding", "encoder_hid_proj",
)
_OUTPUT_MODULES: Tuple[str, ...] = ("norm_out", "proj_out")


class PipelineParallelDiT(nn.Module):
    """Split a diffusers DiT across **N** GPUs.

    Parameters
    ----------
    dit : nn.Module
        The original DiT backbone.
    device_down, device_up : str
        **Legacy 2-GPU API** (kept for backward compat). If ``devices`` is
        given, these are ignored.
    devices : list[str], optional
        List of CUDA devices, one per pipeline stage. E.g.
        ``["cuda:0", "cuda:1", "cuda:2"]`` → 3-way split. If omitted, falls
        back to ``[device_down, device_up]`` (2-GPU).
    split_ratio : float
        Only used for the 2-GPU legacy path. Ignored when ``devices`` has >2
        entries (chunks are as equal as possible).
    """

    def __init__(
        self,
        dit: nn.Module,
        device_down: str = "cuda:0",
        device_up: str = "cuda:1",
        devices: Optional[List[str]] = None,
        split_ratio: float = 0.5,
        chunk_sizes: Optional[List[int]] = None,
    ) -> None:
        super().__init__()

        # ---- resolve devices ------------------------------------------------
        if devices:
            self.devices = [torch.device(d) for d in devices]
        else:
            self.devices = [torch.device(device_down), torch.device(device_up)]
        self.n_stages = len(self.devices)

        # Keep reference to original transformer without registering as child.
        object.__setattr__(self, "_dit", dit)

        # ---- locate block sequence -----------------------------------------
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

        n_blocks = len(blocks)

        # ---- compute chunk sizes -------------------------------------------
        if chunk_sizes is not None:
            # Explicit caller-supplied distribution (budget-aware placement).
            if len(chunk_sizes) != self.n_stages:
                raise ValueError(
                    f"chunk_sizes (len {len(chunk_sizes)}) must match number "
                    f"of devices ({self.n_stages})."
                )
            if sum(chunk_sizes) != n_blocks:
                raise ValueError(
                    f"chunk_sizes sum to {sum(chunk_sizes)} but transformer "
                    f"has {n_blocks} blocks."
                )
        elif self.n_stages == 2 and not devices and split_ratio != 0.5:
            # Legacy 2-GPU with custom ratio.
            split_idx = int(n_blocks * split_ratio)
            chunk_sizes = [split_idx, n_blocks - split_idx]
        else:
            # Equal-ish split across N stages.
            base = n_blocks // self.n_stages
            extra = n_blocks % self.n_stages
            chunk_sizes = [base + (1 if i < extra else 0) for i in range(self.n_stages)]

        # ---- place blocks on their stages ----------------------------------
        self.stage_layers = nn.ModuleList()
        self._stage_boundaries: List[int] = []  # block index where each stage starts
        idx = 0
        for stage, size in enumerate(chunk_sizes):
            self._stage_boundaries.append(idx)
            dev = self.devices[stage]
            chunk = nn.ModuleList([blk.to(dev) for blk in blocks[idx:idx + size]])
            self.stage_layers.append(chunk)
            idx += size

        # Backward-compat aliases.
        self.first_half_layers = self.stage_layers[0] if self.n_stages >= 1 else nn.ModuleList()
        self.second_half_layers = self.stage_layers[1] if self.n_stages >= 2 else nn.ModuleList()

        # ---- place input/output modules (light) on devices[0] --------------
        for name in _INPUT_MODULES + _OUTPUT_MODULES:
            mod = getattr(dit, name, None)
            if isinstance(mod, nn.Module):
                mod.to(self.devices[0])
                setattr(self, name, mod)

        # ---- bookkeeping ---------------------------------------------------
        self.config = dit.config
        self._orig_cls = type(dit).__name__
        self.lora_scale: float = 1.0

        # ---- install device-transfer hooks ---------------------------------
        self._install_hooks()

        chunk_desc = " + ".join(str(s) for s in chunk_sizes)
        logger.info(
            "PipelineParallelDiT: orig=%s, blocks=%d, stages=%d (%s), devices=%s",
            self._orig_cls, n_blocks, self.n_stages, chunk_desc,
            [str(d) for d in self.devices],
        )

    def _install_hooks(self) -> None:
        """Register device-transfer hooks between stages.

        For stage i > 0: a forward_pre_hook moves inputs to devices[i].
        For the last stage: a forward_hook moves output back to devices[0].
        """
        if self.n_stages <= 1:
            return

        devices = self.devices
        last_device = devices[-1]
        home_device = devices[0]

        for stage_idx in range(1, self.n_stages):
            target = devices[stage_idx]
            stage_blocks = self.stage_layers[stage_idx]

            def _make_pre(target_dev):
                def _pre(_module, args, kwargs):
                    new_args = tuple(_move(a, target_dev) for a in args)
                    new_kwargs = {k: _move(v, target_dev) for k, v in kwargs.items()}
                    return new_args, new_kwargs
                return _pre

            # Register pre-hook on first block of this stage.
            first_blk = stage_blocks[0]
            first_blk.register_forward_pre_hook(_make_pre(target), with_kwargs=True)

            # If this is the last stage, register post-hook on its last block
            # to bring output back home.
            if stage_idx == self.n_stages - 1:
                last_blk = stage_blocks[-1]

                def _make_post(home_dev):
                    def _post(_module, _args, output):
                        return _move(output, home_dev)
                    return _post

                last_blk.register_forward_hook(_make_post(home_device))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32

    @property
    def device(self) -> torch.device:
        return self.devices[0]

    # Backward-compat
    @property
    def device_down(self) -> torch.device:
        return self.devices[0]

    @property
    def device_up(self) -> torch.device:
        return self.devices[-1]

    # ------------------------------------------------------------------
    # LoRA helpers
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

    def enable_xformers_memory_efficient_attention(self) -> None:
        logger.warning("enable_xformers_memory_efficient_attention() is a no-op on P40 FP32")

    def set_attention_slice(self, slice_size="auto") -> None:
        for module in self.modules():
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size)

    def enable_gradient_checkpointing(self) -> None:
        for module in self.modules():
            if hasattr(module, "enable_gradient_checkpointing"):
                module.enable_gradient_checkpointing()

    # ------------------------------------------------------------------
    # Forward — delegate to original transformer (hooks do the split).
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
        home = self.devices[0]
        hidden_states = hidden_states.to(home)
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.to(home)
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(home)
        if class_labels is not None:
            class_labels = class_labels.to(home)
        if attention_mask is not None:
            attention_mask = attention_mask.to(home)
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.to(home)
        if added_cond_kwargs is not None:
            added_cond_kwargs = _move(added_cond_kwargs, home)
        if cross_attention_kwargs is not None:
            cross_attention_kwargs = _move(cross_attention_kwargs, home)

        out = self._dit(
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

        if isinstance(out, torch.Tensor):
            return out.to(home)
        if hasattr(out, "sample"):
            if out.sample.device != home:
                out.sample = out.sample.to(home)
            return out
        return tuple(_move(t, home) if isinstance(t, torch.Tensor) else t for t in out)
