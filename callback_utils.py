"""
callback_utils.py — Diffusers ``callback_on_step_end`` adapter.

Why this exists
---------------
In diffusers >= 0.27, the ``callback_on_step_end`` contract requires the
callback to **return** the ``callback_kwargs`` dict.  Internally the pipeline
does roughly::

    callback_kwargs = {}
    for k in callback_on_step_end_inputs:
        callback_kwargs[k] = getattr(self, k)
    callback_kwargs = callback_on_step_end(self, i, t, callback_kwargs)
    # ... then later:
    res = {k: callback_kwargs.pop(k) for k in callback_on_step_end_inputs}

If the callback returns ``None`` (as a side-effect-only lambda naturally
does), the subsequent ``callback_kwargs.pop(...)`` raises::

    'NoneType' object has no attribute 'pop'

This manifested as the Twin-mode failure where **both** GPU pairs crashed
identically on every step.

This module provides :func:`make_progress_callback`, which wraps a simple
side-effect progress callback ``(step, timestep, kw) -> None`` and returns a
diffusers-compatible wrapper that always returns the ``kw`` dict.

Keeping it in a separate module makes the contract unit-testable without a
GPU or a real diffusers pipeline.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

# A user-supplied progress callback: ``(step, timestep, kw) -> None``.
# It is invoked for its side effects only (e.g. updating a job's progress).
ProgressCallback = Callable[[int, float, Optional[Dict[str, Any]]], None]


def make_progress_callback(callback: ProgressCallback) -> Callable[..., Dict[str, Any]]:
    """Wrap a side-effect progress callback for diffusers' step-end contract.

    Parameters
    ----------
    callback
        A callable with signature ``(step: int, timestep: float, kw: dict)
        -> None``.  Called once per denoising step for its side effects
        (e.g. reporting progress).

    Returns
    -------
    Callable[..., dict]
        A wrapper suitable for ``pipe(callback_on_step_end=...)``.  It accepts
        the diffusers signature ``(pipe, step, timestep, callback_kwargs)``,
        invokes *callback*, and **returns ``callback_kwargs``** so the
        pipeline's subsequent ``.pop()`` calls work.

    Notes
    -----
    The wrapper tolerates tensor timesteps by converting them to ``float``
    when possible, matching the previous inline-lambda behavior.
    """

    def _step_cb(  # noqa: ANN001 - matches diffusers' expected signature
        pipe: Any,
        step: int,
        timestep: Any,
        callback_kwargs: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        # Normalize tensor/0-d timesteps to a Python float for the user
        # callback (keeps api_server.py's progress reporting simple).  Use
        # ``.item()`` (the canonical torch way) rather than ``float(tensor)``,
        # which fails for objects that expose ``.item()`` but not
        # ``__float__``.
        ts = timestep.item() if hasattr(timestep, "item") else timestep
        callback(step, ts, callback_kwargs)
        # CRITICAL: diffusers >= 0.27 assigns our return value to its own
        # callback_kwargs and then calls ``.pop()`` on it.  Returning None
        # here triggers "'NoneType' object has no attribute 'pop'".
        return callback_kwargs

    return _step_cb