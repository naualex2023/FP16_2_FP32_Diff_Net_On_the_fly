"""
pp_unet.py — Core pipeline-parallel UNet for Tesla P40 (FP32).

Splits a diffusers ``UNet2DConditionModel`` across two GPUs so that FP32
models exceeding a single 24 GB P40 (notably SDXL at ~25 GB) can run.

Stage 0 (device_down):
    Time/aug embeddings, conv_in, down_blocks

Stage 1 (device_up):
    mid_block, up_blocks, conv_norm_out, conv_act, conv_out

Transfer between stages (~55 MB for SDXL 1024²):
    emb [B,C], sample [B,C,H,W], down_block_res_samples (skip connections),
    encoder_hidden_states [B,seq,dim]

Corrected from the development guide: the guide passed the raw *timestep*
tensor as ``temb`` to UNet blocks, which is incorrect.  The real diffusers
forward computes ``emb`` from the timestep via ``time_proj`` + ``time_embedding``
(+ ``add_embedding`` for SDXL), and it is ``emb`` that must be forwarded to
every block.  This module replicates that logic faithfully.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _move(
    obj: torch.Tensor | tuple | dict | None,
    device: str | torch.device,
    non_blocking: bool = False,
) -> torch.Tensor | tuple | dict | None:
    """Recursively move tensors to *device*.

    Handles tensors, tuples/lists of tensors, and dicts whose values are
    tensors (e.g. ``added_cond_kwargs``).
    """
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, dict):
        return {
            k: v.to(device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v
            for k, v in obj.items()
        }
    if isinstance(obj, (list, tuple)):
        moved = tuple(
            t.to(device, non_blocking=non_blocking) if isinstance(t, torch.Tensor) else t
            for t in obj
        )
        return type(obj)(moved) if isinstance(obj, list) else moved  # keep tuple
    return obj


# ---------------------------------------------------------------------------
# Pipeline-parallel UNet
# ---------------------------------------------------------------------------

class PipelineParallelUNet(nn.Module):
    """Split a ``UNet2DConditionModel`` across two GPUs.

    Works for SD 1.5 and SDXL (any model sharing the ``UNet2DConditionModel``
    architecture).  The split point is between ``down_blocks`` (stage 0) and
    ``mid_block`` (stage 1).

    Parameters
    ----------
    unet : UNet2DConditionModel
        The original, loaded UNet (will be dissected).
    device_down : str
        CUDA device for stage 0 (time embed + conv_in + down blocks).
    device_up : str
        CUDA device for stage 1 (mid + up blocks + conv_out).
    """

    def __init__(
        self,
        unet: nn.Module,
        device_down: str = "cuda:0",
        device_up: str = "cuda:1",
    ) -> None:
        super().__init__()

        self.device_down = torch.device(device_down)
        self.device_up = torch.device(device_up)

        # ------------------------------------------------------------------
        # Stage 0  —  embedding modules  (all tiny, stay on device_down)
        # ------------------------------------------------------------------
        self.time_proj = unet.time_proj.to(self.device_down)
        self.time_embedding = unet.time_embedding.to(self.device_down)
        # ``time_embed_act`` may be ``nn.Identity`` or ``nn.SiLU``
        self.time_embed_act = (
            unet.time_embed_act.to(self.device_down)
            if getattr(unet, "time_embed_act", None) is not None
            else None
        )

        # Class embedding (usually None for SD 1.5 / SDXL)
        self.class_embedding = (
            unet.class_embedding.to(self.device_down)
            if getattr(unet, "class_embedding", None) is not None
            else None
        )

        # Augmented embedding — SDXL: ``addition_embed_type == "text_time"``
        self.add_time_proj = (
            unet.add_time_proj.to(self.device_down)
            if hasattr(unet, "add_time_proj")
            else None
        )
        self.add_embedding = (
            unet.add_embedding.to(self.device_down)
            if getattr(unet, "add_embedding", None) is not None
            else None
        )

        # Encoder hidden projection (usually None for SD / SDXL)
        self.encoder_hid_proj = (
            unet.encoder_hid_proj.to(self.device_down)
            if getattr(unet, "encoder_hid_proj", None) is not None
            else None
        )

        # GLIGEN position net (rarely used, but move it just in case)
        self.position_net = (
            unet.position_net.to(self.device_down)
            if getattr(unet, "position_net", None) is not None
            else None
        )

        # ------------------------------------------------------------------
        # Stage 0  —  conv_in + down_blocks
        # ------------------------------------------------------------------
        self.conv_in = unet.conv_in.to(self.device_down)
        self.down_blocks = nn.ModuleList(
            [blk.to(self.device_down) for blk in unet.down_blocks]
        )

        # ------------------------------------------------------------------
        # Stage 1  —  mid_block + up_blocks
        # ------------------------------------------------------------------
        self.mid_block = (
            unet.mid_block.to(self.device_up)
            if getattr(unet, "mid_block", None) is not None
            else None
        )
        self.up_blocks = nn.ModuleList(
            [blk.to(self.device_up) for blk in unet.up_blocks]
        )

        # ------------------------------------------------------------------
        # Stage 1  —  output post-processing
        # ------------------------------------------------------------------
        self.conv_norm_out = (
            unet.conv_norm_out.to(self.device_up)
            if getattr(unet, "conv_norm_out", None) is not None
            else None
        )
        self.conv_act = (
            unet.conv_act.to(self.device_up)
            if getattr(unet, "conv_act", None) is not None
            else None
        )
        self.conv_out = unet.conv_out.to(self.device_up)

        # ------------------------------------------------------------------
        # Bookkeeping
        # ------------------------------------------------------------------
        self.config = unet.config
        self._orig_unet_cls = type(unet).__name__
        # ``lora_scale`` is read by the pipeline's LoRA scale decorator
        self.lora_scale: float = 1.0
        # Number of upsampling layers — needed for size-checking logic
        self.num_upsamplers: int = sum(
            1 for blk in self.up_blocks if hasattr(blk, "upsamplers") and blk.upsamplers
        )

        logger.info(
            "PipelineParallelUNet created: stage0=%s, stage1=%s, orig=%s",
            self.device_down,
            self.device_up,
            self._orig_unet_cls,
        )

    # ------------------------------------------------------------------
    # Properties required by diffusers pipelines
    # ------------------------------------------------------------------

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the first (stage-0) parameter."""
        try:
            return next(self.conv_in.parameters()).dtype
        except StopIteration:
            return torch.float32

    @property
    def device(self) -> torch.device:
        """Return the device of stage 0 (where latents are managed)."""
        return self.device_down

    # ------------------------------------------------------------------
    # LoRA helpers (delegate to submodules)
    # ------------------------------------------------------------------

    def scale_lora_weights(self, scale: float) -> None:
        """Scale all LoRA weights in both stages."""
        self.lora_scale = scale
        for module in self.modules():
            if hasattr(module, "scale_lora_weights"):
                module.scale_lora_weights(scale)

    def unscale_lora_weights(self) -> None:
        """Unscale all LoRA weights in both stages."""
        self.lora_scale = 1.0
        for module in self.modules():
            if hasattr(module, "unscale_lora_weights"):
                module.unscale_lora_weights()

    # ------------------------------------------------------------------
    # No-op stubs for methods the pipeline might call
    # ------------------------------------------------------------------

    def enable_xformers_memory_efficient_attention(self) -> None:
        """No-op — xformers is irrelevant for FP32 on Pascal."""
        logger.warning("enable_xformers_memory_efficient_attention() is a no-op on P40 FP32")

    def set_attention_slice(self, slice_size="auto") -> None:
        """Delegate to child modules that support attention slicing."""
        for module in self.modules():
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size)

    def enable_freeu(self, s1: float, s2: float, b1: float, b2: float) -> None:
        for blk in self.up_blocks:
            setattr(blk, "s1", s1)
            setattr(blk, "s2", s2)
            setattr(blk, "b1", b1)
            setattr(blk, "b2", b2)

    def disable_freeu(self) -> None:
        for blk in self.up_blocks:
            for k in ("s1", "s2", "b1", "b2"):
                setattr(blk, k, None)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

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
    ) -> tuple | Any:
        """Pipeline-parallel forward pass.

        Mirrors ``UNet2DConditionModel.forward`` exactly, but with a device
        boundary between down blocks (stage 0) and mid block (stage 1).
        """
        # ==================================================================
        # 0. LoRA scale (if requested via cross_attention_kwargs)
        # ==================================================================
        lora_scale = (
            cross_attention_kwargs.pop("scale", self.lora_scale)
            if cross_attention_kwargs is not None
            else self.lora_scale
        )
        if lora_scale != 1.0:
            self.scale_lora_weights(lora_scale)

        # ==================================================================
        # STAGE 0 — everything on device_down
        # ==================================================================
        sample = sample.to(self.device_down)
        encoder_hidden_states = encoder_hidden_states.to(self.device_down)
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to(self.device_down)
        if timestep_cond is not None:
            timestep_cond = timestep_cond.to(self.device_down)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device_down)
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.to(self.device_down)
        if class_labels is not None:
            class_labels = class_labels.to(self.device_down)
        if added_cond_kwargs is not None:
            added_cond_kwargs = _move(added_cond_kwargs, self.device_down)
        if down_intrablock_additional_residuals is not None:
            down_intrablock_additional_residuals = _move(
                down_intrablock_additional_residuals, self.device_down
            )
        if down_block_additional_residuals is not None:
            down_block_additional_residuals = _move(
                down_block_additional_residuals, self.device_down
            )
        if mid_block_additional_residual is not None:
            mid_block_additional_residual = mid_block_additional_residual.to(self.device_up)

        # ---- 1. Time embedding -----------------------------------------
        # NOTE: ``timestep`` may be a Python int/float or a 0-d tensor.  The
        # diffusers ``UNet2DConditionModel.get_time_embed`` normalises it to a
        # 1-d tensor on the sample's device before calling ``time_proj``.  We
        # replicate that logic here (calling ``time_proj`` on a scalar raises
        # "Timesteps should be a 1d-array").
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            t_dtype = (
                torch.float64 if isinstance(timestep, float) else torch.int64
            )
            timesteps = torch.tensor(
                [timesteps], dtype=t_dtype, device=self.device_down
            )
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(self.device_down)
        # broadcast to batch dimension (ONNX/CoreML compatible)
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        # ``Timesteps`` returns float32; cast to the model dtype.
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        # ---- Class embedding (usually None) ----------------------------
        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError(
                    "class_labels must be provided when class_embedding is set"
                )
            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)
                class_labels = class_labels.to(dtype=sample.dtype)
            class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)
            if getattr(self.config, "class_embeddings_concat", False):
                emb = torch.cat([emb, class_emb], dim=-1)
            else:
                emb = emb + class_emb

        # ---- Aug embedding (SDXL: text_time) ---------------------------
        aug_emb = None
        if self.config.addition_embed_type == "text":
            text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)
            aug_emb = self.add_embedding(text_embs)
        elif self.config.addition_embed_type == "text_time":
            text_embeds = added_cond_kwargs["text_embeds"]
            time_ids = added_cond_kwargs["time_ids"]
            time_embeds = self.add_time_proj(time_ids.flatten())
            time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
            add_embeds = torch.cat([text_embeds, time_embeds], dim=-1)
            add_embeds = add_embeds.to(emb.dtype)
            aug_emb = self.add_embedding(add_embeds)
        elif self.config.addition_embed_type == "image":
            image_embs = added_cond_kwargs.get("image_embeds")
            aug_emb = self.add_embedding(image_embs)
        if aug_emb is not None:
            emb = emb + aug_emb

        if self.time_embed_act is not None:
            emb = self.time_embed_act(emb)

        # ---- Encoder hidden states projection (usually None) -----------
        if self.encoder_hid_proj is not None:
            hid_type = getattr(self.config, "encoder_hid_dim_type", None)
            if hid_type == "text_proj":
                encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)

        # ---- Attention masks → bias ------------------------------------
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)
        if encoder_attention_mask is not None:
            encoder_attention_mask = (
                1 - encoder_attention_mask.to(sample.dtype)
            ) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # ---- 2. conv_in ------------------------------------------------
        sample = self.conv_in(sample)

        # ---- 2.5 GLIGEN ------------------------------------------------
        if (
            cross_attention_kwargs is not None
            and cross_attention_kwargs.get("gligen", None) is not None
            and self.position_net is not None
        ):
            cross_attention_kwargs = cross_attention_kwargs.copy()
            gligen_args = cross_attention_kwargs.pop("gligen")
            cross_attention_kwargs["gligen"] = {
                "objs": self.position_net(**gligen_args)
            }

        # ---- 3. Down blocks --------------------------------------------
        is_controlnet = (
            mid_block_additional_residual is not None
            and down_block_additional_residuals is not None
        )
        is_adapter = down_intrablock_additional_residuals is not None
        # backward-compat: T2I-Adapter via deprecated arg name
        if (
            not is_adapter
            and mid_block_additional_residual is None
            and down_block_additional_residuals is not None
        ):
            down_intrablock_additional_residuals = down_block_additional_residuals
            down_block_additional_residuals = None
            is_adapter = True

        down_block_res_samples = (sample,)
        for down_block in self.down_blocks:
            has_cross_attn = getattr(down_block, "has_cross_attention", False)
            if has_cross_attn:
                additional_residuals: dict = {}
                if is_adapter and down_intrablock_additional_residuals:
                    additional_residuals["additional_residuals"] = (
                        down_intrablock_additional_residuals.pop(0)
                    )
                sample, res_samples = down_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                    encoder_attention_mask=encoder_attention_mask,
                    **additional_residuals,
                )
            else:
                sample, res_samples = down_block(hidden_states=sample, temb=emb)
                if is_adapter and down_intrablock_additional_residuals:
                    sample = sample + down_intrablock_additional_residuals.pop(0)
            down_block_res_samples += res_samples

        # ---- ControlNet: add down-block residuals ----------------------
        if is_controlnet and down_block_additional_residuals is not None:
            new_samples = ()
            for res, add_res in zip(
                down_block_res_samples, down_block_additional_residuals
            ):
                new_samples += (res + add_res,)
            down_block_res_samples = new_samples

        # ==================================================================
        # TRANSFER  device_down → device_up  (~55 MB for SDXL 1024²)
        # ==================================================================
        sample = sample.to(self.device_up, non_blocking=True)
        emb = emb.to(self.device_up, non_blocking=True)
        encoder_hidden_states = encoder_hidden_states.to(
            self.device_up, non_blocking=True
        )
        down_block_res_samples = tuple(
            s.to(self.device_up, non_blocking=True) for s in down_block_res_samples
        )
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device_up)
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.to(self.device_up)

        # ==================================================================
        # STAGE 1 — everything on device_up
        # ==================================================================

        # ---- 4. Mid block ----------------------------------------------
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

        if is_controlnet and mid_block_additional_residual is not None:
            sample = sample + mid_block_additional_residual

        # ---- 5. Up blocks ----------------------------------------------
        for i, up_block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1

            n_resnets = len(up_block.resnets)
            res_samples = down_block_res_samples[-n_resnets:]
            down_block_res_samples = down_block_res_samples[:-n_resnets]

            upsample_size = None
            if not is_final_block:
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

        # ---- 6. Post-process -------------------------------------------
        if self.conv_norm_out is not None:
            sample = self.conv_norm_out(sample)
        if self.conv_act is not None:
            sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # ==================================================================
        # Return to device_down so the scheduler loop stays on one device
        # ==================================================================
        sample = sample.to(self.device_down)

        if lora_scale != 1.0:
            self.unscale_lora_weights()

        if not return_dict:
            return (sample,)

        # Return a lightweight namespace mimicking UNet2DConditionOutput
        from types import SimpleNamespace
        return SimpleNamespace(sample=sample)