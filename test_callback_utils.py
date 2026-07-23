"""
test_callback_utils.py — Regression tests for the diffusers callback adapter.

These tests pin the fix for the Twin-mode failure:

    ✗ Failed
    Pair A: 'NoneType' object has no attribute 'pop'
    Pair B: 'NoneType' object has no attribute 'pop'

Root cause: diffusers >= 0.27 requires ``callback_on_step_end`` to RETURN the
``callback_kwargs`` dict; the pipeline then does
``callback_kwargs.pop(k)``. Our old inline lambda returned ``None`` (the
side-effect callback returns nothing), so the very first ``.pop()`` raised
``'NoneType' object has no attribute 'pop'`` on BOTH GPU pairs.

These tests are fully offline (no torch / diffusers / GPU required) so they
run in the dev environment.  Run with::

    pytest test_callback_utils.py -v
"""

from __future__ import annotations

import pytest

from callback_utils import make_progress_callback


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeTensor:
    """Mimics a 0-d torch tensor: has ``.item()`` but is not a real tensor."""

    def __init__(self, value: float) -> None:
        self._v = value

    def item(self) -> float:
        return self._v


def _simulate_diffusers_loop(cb, steps=5, inputs=("latents",)):
    """Mimic the relevant slice of diffusers' denoising loop.

    diffusers does (per step), roughly::

        callback_kwargs = {k: getattr(self, k) for k in callback_on_step_end_inputs}
        callback_kwargs = callback_on_step_end(self, i, t, callback_kwargs)
        callback_kwargs = {k: callback_kwargs.pop(k) for k in inputs}

    If ``callback_on_step_end`` returns ``None``, the dict comprehension's
    ``callback_kwargs.pop(k)`` raises ``'NoneType' object has no attribute
    'pop'``.  This helper reproduces that exact behavior.
    """
    for i in range(steps):
        t = 1000 - i * 100  # descending timesteps, like a real scheduler
        callback_kwargs = {k: f"value-{i}" for k in inputs}
        # THIS is the line that used to crash:
        callback_kwargs = cb(pipe=object(), step=i, timestep=t, callback_kwargs=callback_kwargs)
        # If cb returned None, the next line raises AttributeError.
        for k in inputs:
            callback_kwargs.pop(k)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_callback_returns_kwargs_dict():
    """The wrapper MUST return the callback_kwargs dict (not None)."""
    seen = []
    cb = make_progress_callback(lambda step, ts, kw: seen.append((step, ts, kw)))
    kw_in = {"latents": "abc"}
    kw_out = cb(pipe=object(), step=1, timestep=900, callback_kwargs=kw_in)
    assert kw_out is kw_in, "callback must return the same kwargs dict it received"
    assert seen == [(1, 900, kw_in)]


def test_simulated_pipeline_loop_does_not_crash():
    """End-to-end-ish: simulate diffusers' denoising loop with the wrapper.

    Before the fix this raised ``'NoneType' object has no attribute 'pop'``
    on the first step.  After the fix it completes all steps.
    """
    progress = []
    user_cb = lambda step, ts, kw: progress.append((step, ts))
    cb = make_progress_callback(user_cb)
    _simulate_diffusers_loop(cb, steps=10)
    assert len(progress) == 10
    assert progress[0] == (0, 1000)
    assert progress[-1] == (9, 100)


def test_tensor_timestep_is_normalized_to_float():
    """The wrapper converts tensor-like timesteps to float for the user cb."""
    seen_ts = []
    user_cb = lambda step, ts, kw: seen_ts.append(ts)
    cb = make_progress_callback(user_cb)
    cb(pipe=object(), step=0, timestep=_FakeTensor(950.0), callback_kwargs={})
    assert seen_ts == [950.0]
    assert isinstance(seen_ts[0], float)


def test_scalar_timestep_passes_through():
    """A plain int/float timestep is forwarded unchanged."""
    seen_ts = []
    user_cb = lambda step, ts, kw: seen_ts.append(ts)
    cb = make_progress_callback(user_cb)
    cb(pipe=object(), step=0, timestep=42, callback_kwargs={})
    assert seen_ts == [42]


def test_none_callback_kwargs_round_trips():
    """If diffusers ever passes None, the wrapper should still not crash.

    It returns whatever it received so the contract (return the dict) is
    honored as closely as possible without inventing data.
    """
    cb = make_progress_callback(lambda step, ts, kw: None)
    out = cb(pipe=object(), step=0, timestep=500, callback_kwargs=None)
    assert out is None  # forwarded, not dropped


def test_wrapper_is_thread_safe_for_concurrent_pairs():
    """Twin mode runs Pair A and Pair B on separate threads concurrently.

    Each pair builds its own wrapper (closure) around a shared user callback;
    this ensures the two wrappers don't interfere when driven from two
    threads at once (mirrors api_server._run_twin_generation).
    """
    import threading

    log_a, log_b = [], []
    lock = threading.Lock()

    def user_cb_factory(log):
        def cb(step, ts, kw):
            with lock:
                log.append(step)
        return cb

    cb_a = make_progress_callback(user_cb_factory(log_a))
    cb_b = make_progress_callback(user_cb_factory(log_b))

    errors = []

    def work(cb, log):
        try:
            _simulate_diffusers_loop(cb, steps=8)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

    ta = threading.Thread(target=work, args=(cb_a, log_a))
    tb = threading.Thread(target=work, args=(cb_b, log_b))
    ta.start(); tb.start()
    ta.join(); tb.join()

    assert errors == [], f"threads raised: {errors}"
    assert sorted(log_a) == list(range(8))
    assert sorted(log_b) == list(range(8))


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))