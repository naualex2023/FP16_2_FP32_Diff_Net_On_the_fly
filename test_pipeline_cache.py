"""
test_pipeline_cache.py — Unit tests for the keep-alive pipeline cache.

These tests do NOT require CUDA or a real diffusion model: the cache is
exercised with a fake "pipeline" factory that counts how many times the
(expensive) build is performed.  This verifies the core promise — that the
second call within the idle window reuses the cached object instead of
reloading.

Run with:  pytest test_pipeline_cache.py -v
"""

from __future__ import annotations

import sys
import threading
import time

import pytest

# Import after defining the module path (same directory).
from pipeline_cache import PipelineCache, _CacheEntry


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakePipe:
    """A trivial stand-in for a diffusers pipeline."""

    def __init__(self, tag: str):
        self.tag = tag
        self.calls = 0

    def __call__(self, **_):
        self.calls += 1
        return self


def make_factory(counter: dict, tag: str = "pipe"):
    """Return a factory that increments ``counter['builds']`` on each build."""

    def _factory():
        counter["builds"] += 1
        return FakePipe(tag)

    return _factory


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cache_hit_reuses_pipeline():
    """Second get_or_load with the same key must NOT rebuild."""
    c = PipelineCache(idle_timeout=60, keep_alive=True, reaper_interval=10)
    counter = {"builds": 0}
    key = ("sdxl", "/m", "cuda:0", "cuda:1")

    p1, e1 = c.get_or_load(key, make_factory(counter))
    p2, e2 = c.get_or_load(key, make_factory(counter))

    assert counter["builds"] == 1, "second call should be a cache hit"
    assert p1 is p2, "same object must be returned"
    assert e1 is e2, "same cache entry must be returned"
    c.shutdown()


def test_distinct_keys_get_distinct_entries():
    c = PipelineCache(idle_timeout=60, keep_alive=True, reaper_interval=10)
    counter = {"builds": 0}
    k1 = ("sdxl", "/a", "cuda:0", "cuda:1")
    k2 = ("sdxl", "/b", "cuda:0", "cuda:1")

    pa, _ = c.get_or_load(k1, make_factory(counter, tag="a"))
    pb, _ = c.get_or_load(k2, make_factory(counter, tag="b"))

    assert counter["builds"] == 2
    assert pa is not pb
    assert pa.tag == "a" and pb.tag == "b"
    c.shutdown()


def test_keep_alive_disabled_rebuilds_every_time():
    """With keep_alive=False, every call rebuilds (original behavior)."""
    c = PipelineCache(idle_timeout=60, keep_alive=False, reaper_interval=10)
    counter = {"builds": 0}
    key = ("sdxl", "/m", "cuda:0", "cuda:1")

    p1, e1 = c.get_or_load(key, make_factory(counter))
    p2, e2 = c.get_or_load(key, make_factory(counter))

    assert counter["builds"] == 2
    assert p1 is not p2
    assert e1 is None and e2 is None
    c.shutdown()


def test_idle_timeout_evicts_entry():
    """An entry untouched for > idle_timeout is reaped on the next sweep."""
    # Very short timeout + interval so the test is fast.
    c = PipelineCache(idle_timeout=0.4, keep_alive=True, reaper_interval=0.1)
    counter = {"builds": 0}
    key = ("sdxl", "/m", "cuda:0", "cuda:1")

    p1, _ = c.get_or_load(key, make_factory(counter))
    assert c.stats()["resident"] == 1

    # Wait long enough for the reaper to observe the idle entry.
    time.sleep(0.9)

    assert c.stats()["resident"] == 0, "idle entry should have been reaped"

    # Next get rebuilds (cache miss after eviction).
    p2, _ = c.get_or_load(key, make_factory(counter))
    assert counter["builds"] == 2
    assert p1 is not p2
    c.shutdown()


def test_touch_resets_idle_timer():
    """A get during the idle window resets last_used and prevents eviction."""
    c = PipelineCache(idle_timeout=0.6, keep_alive=True, reaper_interval=0.1)
    counter = {"builds": 0}
    key = ("sdxl", "/m", "cuda:0", "cuda:1")

    c.get_or_load(key, make_factory(counter))
    for _ in range(4):
        time.sleep(0.2)  # 0.8s total, but we touch each iteration
        c.get_or_load(key, make_factory(counter))

    # Despite 0.8s elapsing, each touch reset the timer, so no eviction.
    assert c.stats()["resident"] == 1
    assert counter["builds"] == 1
    c.shutdown()


def test_release_unload_all():
    c = PipelineCache(idle_timeout=60, keep_alive=True, reaper_interval=10)
    counter = {"builds": 0}
    k1 = ("sdxl", "/a", "cuda:0", "cuda:1")
    k2 = ("sdxl", "/b", "cuda:0", "cuda:1")

    c.get_or_load(k1, make_factory(counter))
    c.get_or_load(k2, make_factory(counter))
    assert c.stats()["resident"] == 2

    assert c.release(k1) is True
    assert c.stats()["resident"] == 1
    assert c.release(("sdxl", "/missing")) is False  # nothing to free

    assert c.unload_all() == 1
    assert c.stats()["resident"] == 0
    c.shutdown()


def test_concurrent_gets_same_key_serialize():
    """Two threads asking the same key get the SAME entry (no double build)."""
    c = PipelineCache(idle_timeout=60, keep_alive=True, reaper_interval=10)
    builds = {"n": 0}
    build_lock = threading.Lock()
    start = threading.Event()

    def slow_factory():
        start.set()  # let the test know we entered the factory once
        time.sleep(0.2)
        with build_lock:
            builds["n"] += 1
        return FakePipe("slow")

    key = ("sdxl", "/m", "cuda:0", "cuda:1")
    results = [None, None]

    def worker(i):
        results[i], _ = c.get_or_load(key, slow_factory)

    t0 = threading.Thread(target=worker, args=(0,))
    t1 = threading.Thread(target=worker, args=(1,))
    t0.start()
    # Ensure thread 0 is the one building before thread 1 races for the lock.
    start.wait(timeout=1.0)
    t1.start()
    t0.join()
    t1.join()

    assert builds["n"] == 1, "factory should run exactly once"
    assert results[0] is results[1]
    c.shutdown()


def test_per_entry_lock_serializes_callers():
    """The returned entry lock can be used to serialize callers manually."""
    c = PipelineCache(idle_timeout=60, keep_alive=True, reaper_interval=10)
    _, entry = c.get_or_load(("k",), lambda: FakePipe("x"))
    assert isinstance(entry, _CacheEntry)

    held = threading.Event()
    proceed = threading.Event()

    def holder():
        with entry.lock:
            held.set()
            proceed.wait(timeout=2.0)

    th = threading.Thread(target=holder)
    th.start()
    assert held.wait(timeout=1.0), "holder should acquire the lock"

    # A second acquire should block until 'proceed' is set.
    got_late = threading.Event()

    def waiter():
        with entry.lock:
            got_late.set()

    w = threading.Thread(target=waiter)
    w.start()
    assert not got_late.wait(timeout=0.2), "second acquire should block"
    proceed.set()
    assert got_late.wait(timeout=1.0), "second acquire should proceed after release"

    th.join()
    w.join()
    c.shutdown()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))