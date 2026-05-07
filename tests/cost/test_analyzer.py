"""Cost analyzer aggregation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lcqa.cost.analyzer import CostAnalyzer, RequestRecord


def _make_record(
    *,
    provider: str = "openai",
    model: str = "gpt-4o",
    input_tokens: int = 1000,
    output_tokens: int = 500,
    cached_input_tokens: int = 0,
    tag: str | None = None,
    latency_ms: float | None = 1500.0,
) -> RequestRecord:
    return RequestRecord(
        provider=provider,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_input_tokens=cached_input_tokens,
        tag=tag,
        latency_ms=latency_ms,
    )


def test_record_validates_cache_le_input() -> None:
    with pytest.raises(ValueError, match="cached_input_tokens"):
        RequestRecord(
            provider="openai",
            model="gpt-4o",
            input_tokens=100,
            output_tokens=50,
            cached_input_tokens=200,
        )


def test_record_cost_matches_pricing() -> None:
    r = _make_record(input_tokens=1_000_000, output_tokens=500_000)
    # gpt-4o: $2.50 input + $10 output per 1M
    assert r.cost() == pytest.approx(2.50 + 5.00)


def test_empty_analyzer_frame_has_columns() -> None:
    a = CostAnalyzer()
    df = a.to_frame()
    assert df.empty
    assert "cost_usd" in df.columns
    assert "tag" in df.columns


def test_analyzer_total_cost() -> None:
    a = CostAnalyzer()
    a.add(_make_record(input_tokens=1_000_000, output_tokens=500_000))
    a.add(_make_record(input_tokens=1_000_000, output_tokens=500_000))
    assert a.total_cost() == pytest.approx(2 * (2.50 + 5.00))


def test_aggregate_by_tag_sums_correctly() -> None:
    a = CostAnalyzer()
    a.add(_make_record(tag="rag-answer", input_tokens=1_000_000, output_tokens=500_000))
    a.add(_make_record(tag="rag-answer", input_tokens=2_000_000, output_tokens=1_000_000))
    a.add(_make_record(tag="rerank", input_tokens=500_000, output_tokens=100_000))
    grouped = a.aggregate_by("tag")
    rag = grouped[grouped["tag"] == "rag-answer"].iloc[0]
    assert rag["requests"] == 2
    # 3M input * $2.50 + 1.5M output * $10 = $7.50 + $15 = $22.50
    assert rag["total_cost_usd"] == pytest.approx(7.50 + 15.0)


def test_aggregate_by_two_columns() -> None:
    a = CostAnalyzer()
    a.add(_make_record(model="gpt-4o", tag="answer"))
    a.add(_make_record(model="gpt-4o-mini", tag="rerank"))
    grouped = a.aggregate_by("model", "tag")
    assert len(grouped) == 2


def test_aggregate_by_no_columns_raises() -> None:
    a = CostAnalyzer()
    a.add(_make_record())
    with pytest.raises(ValueError, match="at least one column"):
        a.aggregate_by()


def test_to_json_writes_summary_and_records(tmp_path: Path) -> None:
    a = CostAnalyzer()
    a.add(_make_record(tag="answer", input_tokens=1000, output_tokens=500))
    a.add(_make_record(tag="rerank", input_tokens=2000, output_tokens=200))
    out = a.to_json(tmp_path / "out.json")
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["summary"]["total_requests"] == 2
    assert data["summary"]["total_cost_usd"] > 0
    assert len(data["records"]) == 2
    assert {r["tag"] for r in data["records"]} == {"answer", "rerank"}


def test_aggregate_includes_latency_percentiles() -> None:
    a = CostAnalyzer()
    for ms in (100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000):
        a.add(_make_record(tag="answer", latency_ms=ms))
    grouped = a.aggregate_by("tag")
    row = grouped[grouped["tag"] == "answer"].iloc[0]
    assert row["latency_ms_p50"] == pytest.approx(750, rel=0.1)
    assert row["latency_ms_p95"] >= 4000


def test_cached_input_reduces_cost() -> None:
    no_cache = _make_record(input_tokens=10_000, output_tokens=1_000, cached_input_tokens=0)
    with_cache = _make_record(input_tokens=10_000, output_tokens=1_000, cached_input_tokens=8_000)
    assert with_cache.cost() < no_cache.cost()
