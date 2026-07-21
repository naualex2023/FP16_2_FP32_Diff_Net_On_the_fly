"""
async_transfer.py — Overlapped compute + PCIe transfer via CUDA streams.

Extends :class:`pp_unet.PipelineParallelUNet` so that the device_down → device_up
activation transfer happens on a dedicated CUDA stream.  This allows the stage-0
compute tail to overlap with the transfer (and vice-versa).

Expected gain is small (~0.3–3%) because transfer is already ~5 ms vs ~2000 ms
of compute, but it is essentially free and can help with many small skip tensors.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from pp_unet import PipelineParallelUNet, _move

logger = logging.getLogger(__name__)


class AsyncPipelineParallelUNet(PipelineParallelUNet):
    """Pipeline-parallel UNet with stream-based async transfers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # One transfer stream per stage device (Pascal supports multi-stream)
        self.transfer_stream_down = torch.cuda.Stream(device=self.device_down)
        self.transfer_stream_up = torch.cuda.Stream(device=self.device_up)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        encoder_hidden_states: torch.Tensor,
        class_labels: torch.Tensor | None = None,
        timestep_cond: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        cross_attention_kwargs: dict[str, Any] | None = None,
        added_cond_kwargs: dict[str, torch.Tensor] | None = None,
        down_block_additional_residuals: tuple[torch.Tensor] | None = None,
        mid_block_additional_residual: torch.Tensor | None = None,
        down_intrablock_additional_residuals: tuple[torch.Tensor] | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
    ):
        # ===== STAGE 0 (default stream on device_down) =================
        sample = sample.to(self.device_down)
        encoder_hidden_states = encoder_hidden_states.to(self.device_down)
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.device_down)
        if added_cond_kwargs is not None:
            added_cond_kwargs = _move(added_cond_kwargs, self.device_down)

        # Time + class + aug embeddings → emb
        t_emb = self.time_proj(timestep)
        emb = self.time_embedding(t_emb, timestep_cond)
        if self.class_embedding is not None and class_labels is not None:
            class_emb = self.class_embedding(class_labels.to(self.device_down))
            if getattr(self.config, "class_embeddings_concat", False):
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb
        if self.config.addition_embed_type == "text_time":
            text_embeds = added_cond_kwargs["text_embeds"]
            time_ids = added_cond_kwargs["time_ids"]
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
            add_embeds = torch.cat([text_embeds, time_embeds], dim=-1).to(emb.dtype)
            emb = emb + self.add_embedding(add_embeds)
        elif self.config.addition_embed_type == "text":
            text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
            emb = emb + self.add_embedding(text_embs)
        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        # conv_in
        sample = self.conv_in(sample)

        # Down blocks
        down_block_res_samples = (sample,)
        for down_block in self.down_blocks:
            has_cross_attn = getattr(down_block, "has_cross_attention", False)
            if has_cross_attn:
                sample, res_samples = down_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample, res_samples = down_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        # ===== ASYNC TRANSFER on a dedicated stream ====================
        with torch.cuda.stream(self.transfer_stream_down):
            sample_up = sample.to(self.device_up, non_blocking=True)
            res_up = tuple(
                s.to(self.device_up, non_blocking=True)
                for s in down_block_res_samples
            )
            emb_up = emb.to(self.device_up, non_blocking=True)
            ehs_up = encoder_hidden_states.to(self.device_up, non_blocking=True)
            ack_up = (
                _move(added_cond_kwargs, self.device_up, non_blocking=True)
                if added_cond_kwargs is not None
                else None
            )

        # Make stage-1 (device_up default stream) wait for the transfer.
        torch.cuda.current_stream(self.device_up).wait_stream(self.transfer_stream_down)

        # ===== STAGE 1 (default stream on device_up) ===================
        sample = sample_up
        emb = emb_up
        encoder_hidden_states = ehs_up
        added_cond_kwargs = ack_up
        down_block_res_samples = res_up

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device_up)
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.to(self.device_up)

        # Mid block
        if self.mid_block is not None:
            has_cross_attn = getattr(self.mid_block, "has_cross_attention", False)
            if has_cross_attn:
                sample = self.mid_block(
                    sample,
                    emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = self.mid_block(sample, emb)

        # Up blocks
        for i, up_block in enumerate(self.up_blocks):
            n_resnets = len(up_block.resnets)
            res_samples = down_block_res_samples[-n_resnets:]
            down_block_res_samples = down_block_res_samples[:-n_resnets]

            upsample_size = None
            if i != len(self.up_blocks) - 1:
                forward_upsample_size = any(
                    dim % (2 ** self.num_upsamplers) != 0
                    for dim in sample.shape[-2:]
                )
                if forward_upsample_size:
                    upsample_size = down_block_res_samples[-1].shape[2:]

            has_cross_attn = getattr(up_block, "has_cross_attention", False)
            if has_cross_attn:
                sample = up_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    cross_attention_kwargs=cross_attention_kwargs,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                    encoder_attention_mask=encoder_attention_mask,
                )
            else:
                sample = up_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

        # Post-process
        if self.conv_norm_out is not None:
            sample = self.conv_norm_out(sample)
        if self.conv_act is not None:
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # ===== Return to device_down (scheduler loop lives here) =======
        sample = sample.to(self.device_down)

        if not return_dict:
            return (sample,)
        from types import SimpleNamespace
        return SimpleNamespace(sample=sample)