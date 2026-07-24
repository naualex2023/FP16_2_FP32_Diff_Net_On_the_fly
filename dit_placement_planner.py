"""
dit_placement_planner.py — Budget-aware component placement for DiT models.

Pure-Python (no torch dependency) so it can be unit-tested anywhere.

Given the estimated per-component sizes (from :mod:`dit_size_estimator`) and
the number of transformer / T5 blocks, this planner computes a placement
that respects each GPU's VRAM budget:

1. **Small components** (CLIP text encoders, VAE) go on the home GPU
   (``devices[0]``), where the scheduler loop and VAE decode live.
2. **T5-XXL** (``text_encoder_3``), if present and large, is **sharded**
   across the fewest GPUs that can collectively hold it.  When it fits one
   GPU's free budget it stays on a single GPU (backward-compatible).
3. **Transformer** blocks are distributed across all GPUs proportionally to
   each GPU's *remaining* budget.  GPUs whose remaining budget can't hold a
   whole block are excluded from the transformer split.

The planner returns concrete device assignments and per-stage block counts
that :mod:`pipeline_parallel_dit` consumes directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PlacementPlan:
    """Concrete placement decisions for a DiT model across N GPUs."""

    # GPU index (into the ``devices`` list) for each small component.
    component_device: Dict[str, int] = field(default_factory=dict)
    # T5-XXL sharding: list of GPU indices, parallel to ``t5_chunk_sizes``.
    t5_devices: List[int] = field(default_factory=list)
    t5_chunk_sizes: List[int] = field(default_factory=list)
    # Transformer sharding: GPU indices + block counts per GPU.
    transformer_devices: List[int] = field(default_factory=list)
    transformer_chunk_sizes: List[int] = field(default_factory=list)
    # Expected load (GB) on each GPU index — for logging / validation.
    per_gpu_load: Dict[int, float] = field(default_factory=dict)

    def describe(self, devices: List[str]) -> str:
        """Human-readable allocation table for logging."""
        lines = []
        for gpu_idx in sorted(self.per_gpu_load):
            parts = []
            for comp, gi in self.component_device.items():
                if gi == gpu_idx:
                    parts.append(comp)
            if gpu_idx in self.t5_devices:
                ti = self.t5_devices.index(gpu_idx)
                parts.append(f"text_encoder_3[{self.t5_chunk_sizes[ti]} blocks]")
            if gpu_idx in self.transformer_devices:
                xi = self.transformer_devices.index(gpu_idx)
                parts.append(f"transformer[{self.transformer_chunk_sizes[xi]} blocks]")
            load = self.per_gpu_load[gpu_idx]
            lines.append(f"  {devices[gpu_idx]}: {', '.join(parts)} = {load:.1f} GB")
        return "\n".join(lines)


def plan_placement(
    component_sizes: Dict[str, float],
    n_transformer_blocks: int,
    devices: List[str],
    vram_gb: float,
    margin_gb: float = 3.0,
    n_t5_blocks: Optional[int] = None,
    transformer_block_gb: Optional[float] = None,
    t5_block_gb: Optional[float] = None,
) -> PlacementPlan:
    """Compute a budget-aware placement plan.

    Parameters
    ----------
    component_sizes : dict
        ``{component_name: size_gb}`` in the target compute dtype, e.g.
        ``{"transformer": 32.6, "text_encoder_3": 23.1, ...}``.
    n_transformer_blocks : int
        Number of transformer blocks (``len(transformer.transformer_blocks)``).
    devices : list[str]
        Available CUDA devices, e.g. ``["cuda:0", "cuda:1", ...]``.
    vram_gb : float
        Total VRAM per GPU in GB.
    margin_gb : float
        Reserve this many GB per GPU for activations / temporaries.
    n_t5_blocks : int, optional
        Number of T5 encoder blocks.  ``None`` or 0 ⇒ no T5.
    transformer_block_gb, t5_block_gb : float, optional
        Per-block size overrides.  When omitted, derived from the total
        component size divided by block count.

    Returns
    -------
    PlacementPlan
    """
    n_gpus = len(devices)
    usable = max(1.0, vram_gb - margin_gb)
    # Remaining budget per GPU index.
    free = {i: usable for i in range(n_gpus)}
    load = {i: 0.0 for i in range(n_gpus)}
    plan = PlacementPlan(per_gpu_load=load)

    # Per-block sizes (derived if not supplied).
    if transformer_block_gb is None:
        total_tx = component_sizes.get("transformer", 0.0)
        transformer_block_gb = total_tx / n_transformer_blocks if n_transformer_blocks else 0.0
    has_t5 = bool(n_t5_blocks and n_t5_blocks > 0)
    if has_t5 and t5_block_gb is None:
        total_t5 = component_sizes.get("text_encoder_3", 0.0)
        t5_block_gb = total_t5 / n_t5_blocks if n_t5_blocks else 0.0

    # ---- 1. Small components on GPU 0 (home) -------------------------------
    for comp in ("text_encoder", "text_encoder_2", "vae"):
        size = component_sizes.get(comp, 0.0)
        if size <= 0:
            continue
        plan.component_device[comp] = 0
        free[0] -= size
        load[0] += size

    # ---- 2. T5-XXL: single GPU if it fits, else shard across multiple ------
    if has_t5:
        t5_total = component_sizes.get("text_encoder_3", 0.0)
        # Try to find a single GPU with enough free budget.
        single_gpu: Optional[int] = None
        for gi in range(n_gpus):
            if free[gi] >= t5_total:
                single_gpu = gi
                break
        if single_gpu is not None:
            plan.t5_devices = [single_gpu]
            plan.t5_chunk_sizes = [n_t5_blocks]
            free[single_gpu] -= t5_total
            load[single_gpu] += t5_total
        else:
            # Shard across the fewest GPUs that can collectively hold T5.
            # Greedily fill GPUs in order until the budget is met.
            t5_devices: List[int] = []
            t5_chunks: List[int] = []
            remaining = t5_total
            remaining_blocks = n_t5_blocks
            for gi in range(n_gpus):
                if remaining <= 1e-6:
                    break
                if free[gi] < t5_block_gb:
                    continue  # can't fit even one block
                # How many whole blocks fit on this GPU's free budget.
                fit_blocks = min(remaining_blocks, int(free[gi] / t5_block_gb))
                if fit_blocks <= 0:
                    continue
                t5_devices.append(gi)
                t5_chunks.append(fit_blocks)
                chunk_gb = fit_blocks * t5_block_gb
                free[gi] -= chunk_gb
                load[gi] += chunk_gb
                remaining -= chunk_gb
                remaining_blocks -= fit_blocks
            if remaining > 1e-6:
                raise RuntimeError(
                    f"T5-XXL ({t5_total:.1f} GB) cannot be placed across "
                    f"{n_gpus} GPUs ({vram_gb:.1f} GB each, {margin_gb:.1f} GB "
                    f"margin).  {remaining:.1f} GB unfilled."
                )
            plan.t5_devices = t5_devices
            plan.t5_chunk_sizes = t5_chunks

    # ---- 3. Transformer: distribute proportionally to remaining budget -----
    tx_total = component_sizes.get("transformer", 0.0)
    if n_transformer_blocks and tx_total > 0:
        tx_devices: List[int] = []
        tx_chunks: List[int] = []
        # GPUs that can hold at least one block.
        eligible = [gi for gi in range(n_gpus) if free[gi] >= transformer_block_gb]
        if not eligible:
            raise RuntimeError(
                f"Transformer ({tx_total:.1f} GB, "
                f"{transformer_block_gb:.2f} GB/block) has no GPU with room "
                f"(free budgets: {free})."
            )
        # Split blocks proportional to free budget on eligible GPUs.
        total_free = sum(free[gi] for gi in eligible)
        assigned = 0
        for gi in eligible:
            share = free[gi] / total_free
            blocks = max(1, round(share * n_transformer_blocks)) if len(eligible) > 1 else n_transformer_blocks
            blocks = min(blocks, n_transformer_blocks - assigned)
            # Don't exceed what fits in free budget.
            max_fit = int(free[gi] / transformer_block_gb)
            blocks = min(blocks, max_fit)
            if blocks <= 0:
                continue
            tx_devices.append(gi)
            tx_chunks.append(blocks)
            chunk_gb = blocks * transformer_block_gb
            free[gi] -= chunk_gb
            load[gi] += chunk_gb
            assigned += blocks
        # Handle rounding remainder: distribute leftover blocks to the
        # GPU with the most free budget.
        while assigned < n_transformer_blocks:
            leftover = n_transformer_blocks - assigned
            best_gi = max(tx_devices, key=lambda gi: free.get(gi, 0))
            bi = tx_devices.index(best_gi)
            tx_chunks[bi] += leftover
            free[best_gi] -= leftover * transformer_block_gb
            load[best_gi] += leftover * transformer_block_gb
            assigned += leftover
        plan.transformer_devices = tx_devices
        plan.transformer_chunk_sizes = tx_chunks

    return plan