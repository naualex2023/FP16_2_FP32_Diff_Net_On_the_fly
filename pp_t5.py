"""
pp_t5.py - Pipeline-parallel T5-XXL encoder for Tesla P40 (FP32).

T5-XXL (``text_encoder_3`` in SD3/SD3.5) is ~18-23 GB in FP32 - too large
for a single 24 GB P40 once any margin for activations is reserved.  This
module shards the T5 encoder across N GPUs using the same hook-based
technique as :mod:`pp_dit`: the original encoder's ``encoder.block``
ModuleList is split into chunks, each chunk lives on one GPU, and
forward hooks move the hidden state across device boundaries.

Usage
-----
    from pp_t5 import PipelineParallelT5
    pipe.text_encoder_3 = PipelineParallelT5(
        pipe.text_encoder_3,
        devices=["cuda:2", "cuda:3"],
        chunk_sizes=[12, 12],
    )
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn

from pp_unet import _move

logger = logging.getLogger(__name__)


class PipelineParallelT5(nn.Module):
    """Shard a HuggingFace T5 encoder across N GPUs."""

    def __init__(
        self,
        encoder: nn.Module,
        devices: List[str],
        chunk_sizes: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.devices = [torch.device(d) for d in devices]
        self.n_stages = len(self.devices)

        object.__setattr__(self, "_encoder", encoder)

        block_list = getattr(encoder, "block", None)
        if not isinstance(block_list, nn.ModuleList) or len(block_list) == 0:
            raise ValueError(
                f"{type(encoder).__name__} has no non-empty 'block' "
                f"ModuleList to split."
            )
        blocks = list(block_list)
        n_blocks = len(blocks)

        if chunk_sizes is None:
            base = n_blocks // self.n_stages
            extra = n_blocks % self.n_stages
            chunk_sizes = [base + (1 if i < extra else 0) for i in range(self.n_stages)]
        if len(chunk_sizes) != self.n_stages:
            raise ValueError(
                f"chunk_sizes (len {len(chunk_sizes)}) must match devices "
                f"(len {self.n_stages})."
            )
        if sum(chunk_sizes) != n_blocks:
            raise ValueError(
                f"chunk_sizes sum to {sum(chunk_sizes)} but encoder has "
                f"{n_blocks} blocks."
            )

        self.stage_layers = nn.ModuleList()
        idx = 0
        for stage, size in enumerate(chunk_sizes):
            dev = self.devices[stage]
            chunk = nn.ModuleList([blk.to(dev) for blk in blocks[idx:idx + size]])
            self.stage_layers.append(chunk)
            idx += size

        home = self.devices[0]
        for name in ("shared", "embed_tokens"):
            mod = getattr(encoder, name, None)
            if isinstance(mod, nn.Module):
                mod.to(home)
                setattr(self, name, mod)
        final = getattr(encoder, "final_layer_norm", None)
        if isinstance(final, nn.Module):
            final.to(home)
            self.final_layer_norm = final

        self.config = getattr(encoder, "config", None)
        self._orig_cls = type(encoder).__name__

        self._install_hooks()

        chunk_desc = " + ".join(str(s) for s in chunk_sizes)
        logger.info(
            "PipelineParallelT5: orig=%s, blocks=%d, stages=%d (%s), devices=%s",
            self._orig_cls, n_blocks, self.n_stages, chunk_desc,
            [str(d) for d in self.devices],
        )

    def _install_hooks(self) -> None:
        if self.n_stages <= 1:
            return

        home = self.devices[0]
        for stage_idx in range(1, self.n_stages):
            target = self.devices[stage_idx]
            stage_blocks = self.stage_layers[stage_idx]
            first_blk = stage_blocks[0]

            def _make_pre(target_dev):
                def _pre(_module, args, kwargs):
                    new_args = tuple(_move(a, target_dev) for a in args)
                    new_kwargs = {k: _move(v, target_dev) for k, v in kwargs.items()}
                    return new_args, new_kwargs
                return _pre

            first_blk.register_forward_pre_hook(_make_pre(target), with_kwargs=True)

            if stage_idx == self.n_stages - 1:
                last_blk = stage_blocks[-1]

                def _make_post(home_dev):
                    def _post(_module, _args, output):
                        return _move(output, home_dev)
                    return _post

                last_blk.register_forward_hook(_make_post(home))

    @property
    def dtype(self) -> torch.dtype:
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32

    @property
    def device(self) -> torch.device:
        return self.devices[0]

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        home = self.devices[0]
        if input_ids is not None:
            input_ids = input_ids.to(home)
        if attention_mask is not None:
            attention_mask = attention_mask.to(home)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(home)

        out = self._encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

        if hasattr(out, "last_hidden_state"):
            if out.last_hidden_state.device != home:
                out.last_hidden_state = out.last_hidden_state.to(home)
            return out
        if isinstance(out, torch.Tensor):
            return out.to(home)
        return _move(out, home)
