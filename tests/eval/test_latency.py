"""Latency tracker + percentile tests."""

from __future__ import annotations

import time

import pytest

from lcqa.eval.latency import LatencyTracker, latency_summary


def test_empty_summary_returns_zeros() -> None:
    s = latency_summary([])
    assert s["n"] == 0
    assert s["p50"] == 0.0
    assert s["p95"] == 0.0


def test_single_value_summary() -> None:
    s = latency_summary([100.0])
    assert s["n"] == 1
    assert s["p50"] == 100.0
    assert s["p95"] == 100.0
    assert s["mean"] == 100.0


def test_percentiles_match_numpy_default() -> None:
    # numpy.percentile default is linear interpolation.
    values = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
    s = latency_summary(values)
    assert s["p50"] == pytest.approx(550.0)  # (500+600)/2
    assert s["p95"] == pytest.approx(955.0)  # 0.95 * 9 = 8.55 → between idx 8 and 9
    assert s["min"] == 100.0
    assert s["max"] == 1000.0
    assert s["mean"] == pytest.approx(550.0)


def test_unsorted_input_is_handled() -> None:
    s = latency_summary([500.0, 100.0, 300.0])
    assert s["min"] == 100.0
    assert s["max"] == 500.0
    assert s["p50"] == 300.0


def test_tracker_add_and_summary() -> None:
    t = LatencyTracker()
    for ms in (100, 200, 300, 400, 500):
        t.add(ms)
    assert len(t) == 5
    s = t.summary()
    assert s["p50"] == pytest.approx(300.0)


def test_tracker_negative_latency_raises() -> None:
    t = LatencyTracker()
    with pytest.raises(ValueError, match=">= 0"):
        t.add(-1.0)


def test_tracker_filtered_by_tag() -> None:
    t = LatencyTracker()
    t.add(100, tag="rag")
    t.add(200, tag="rag")
    t.add(500, tag="rerank")
    rag = t.summary(tag="rag")
    rerank = t.summary(tag="rerank")
    assert rag["n"] == 2
    assert rerank["n"] == 1
    assert rerank["p50"] == 500.0


def test_tracker_context_manager_records_elapsed() -> None:
    t = LatencyTracker()
    with t.measure(tag="sleep"):
        time.sleep(0.01)
    s = t.summary(tag="sleep")
    assert s["n"] == 1
    assert s["p50"] >= 5.0  # at least ~5ms (we slept 10ms)
