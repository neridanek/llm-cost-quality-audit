"""Latency tracking: p50/p95/p99 percentiles + context manager API."""

from __future__ import annotations

import time
from collections.abc import Iterable
from contextlib import contextmanager
from statistics import median
from typing import Any


class LatencyTracker:
    """Records per-request latency in milliseconds.

    Usage:
        tracker = LatencyTracker()
        with tracker.measure(tag="rag-answer"):
            run_rag_query(...)
        summary = tracker.summary()
    """

    def __init__(self) -> None:
        self._records: list[tuple[float, str | None]] = []

    def __len__(self) -> int:
        return len(self._records)

    def add(self, latency_ms: float, tag: str | None = None) -> None:
        if latency_ms < 0:
            raise ValueError(f"latency_ms must be >= 0, got {latency_ms}")
        self._records.append((latency_ms, tag))

    @contextmanager
    def measure(self, tag: str | None = None) -> Any:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.add(elapsed_ms, tag=tag)

    def latencies(self, tag: str | None = None) -> list[float]:
        if tag is None:
            return [ms for ms, _ in self._records]
        return [ms for ms, t in self._records if t == tag]

    def summary(self, tag: str | None = None) -> dict[str, float | int]:
        return latency_summary(self.latencies(tag=tag))


def latency_summary(latencies_ms: Iterable[float]) -> dict[str, float | int]:
    """Compute p50/p95/p99/mean/min/max from a sequence of latencies (ms).

    Returns zeros for an empty input rather than raising — keeps regression
    pipelines resilient when an optimization removes a metric.
    """
    values = sorted(latencies_ms)
    n = len(values)
    if n == 0:
        return {"n": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "n": n,
        "p50": float(median(values)),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "mean": sum(values) / n,
        "min": values[0],
        "max": values[-1],
    }


def _percentile(sorted_values: list[float], p: float) -> float:
    """Linear-interpolation percentile (numpy default), p in [0,1]. Input must be sorted."""
    if not sorted_values:
        return 0.0
    if not 0 <= p <= 1:
        raise ValueError(f"p must be in [0,1], got {p}")
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    rank = p * (n - 1)
    lower = int(rank)
    upper = min(lower + 1, n - 1)
    frac = rank - lower
    return sorted_values[lower] + frac * (sorted_values[upper] - sorted_values[lower])
