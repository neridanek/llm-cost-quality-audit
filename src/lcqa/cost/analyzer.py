"""Cost breakdown analyzer.

Collect `RequestRecord` instances from your RAG pipeline (one per LLM call).
The analyzer computes per-request cost and aggregates by model, tag, and time
window. See ADR-003 for the per-request attribution methodology.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from lcqa.cost.pricing import ModelPricing, get_pricing


class RequestRecord(BaseModel):
    """One LLM call. `tag` lets you attribute cost to a use case (e.g., 'rag-answer')."""

    provider: str
    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    latency_ms: float | None = Field(default=None, ge=0)
    tag: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @model_validator(mode="after")
    def _validate_cache(self) -> Self:
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError(
                f"cached_input_tokens ({self.cached_input_tokens}) > "
                f"input_tokens ({self.input_tokens})"
            )
        return self

    def cost(self, pricing: ModelPricing | None = None) -> float:
        """USD cost for this request."""
        if pricing is None:
            pricing = get_pricing(self.provider, self.model)
        return pricing.cost(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cached_input_tokens=self.cached_input_tokens,
        )


class CostAnalyzer:
    """Collect records, compute costs, aggregate."""

    def __init__(self) -> None:
        self._records: list[RequestRecord] = []

    def __len__(self) -> int:
        return len(self._records)

    def add(self, record: RequestRecord) -> None:
        self._records.append(record)

    def extend(self, records: Iterable[RequestRecord]) -> None:
        self._records.extend(records)

    def to_frame(self) -> pd.DataFrame:
        """Per-request DataFrame with computed `cost_usd` column."""
        if not self._records:
            return pd.DataFrame(
                columns=[
                    "provider",
                    "model",
                    "input_tokens",
                    "output_tokens",
                    "cached_input_tokens",
                    "latency_ms",
                    "tag",
                    "timestamp",
                    "cost_usd",
                ]
            )
        rows = []
        for r in self._records:
            rows.append(
                {
                    "provider": r.provider,
                    "model": r.model,
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "cached_input_tokens": r.cached_input_tokens,
                    "latency_ms": r.latency_ms,
                    "tag": r.tag,
                    "timestamp": r.timestamp,
                    "cost_usd": r.cost(),
                }
            )
        return pd.DataFrame(rows)

    def total_cost(self) -> float:
        return sum(r.cost() for r in self._records)

    def aggregate_by(self, *columns: str) -> pd.DataFrame:
        """Aggregate cost + token usage by one or more columns (e.g., 'tag', 'model')."""
        if not columns:
            raise ValueError("aggregate_by requires at least one column name")
        df = self.to_frame()
        if df.empty:
            return df
        return (
            df.groupby(list(columns), dropna=False)
            .agg(
                requests=("cost_usd", "size"),
                total_cost_usd=("cost_usd", "sum"),
                input_tokens=("input_tokens", "sum"),
                output_tokens=("output_tokens", "sum"),
                cached_input_tokens=("cached_input_tokens", "sum"),
                latency_ms_p50=("latency_ms", lambda s: s.dropna().quantile(0.5)),
                latency_ms_p95=("latency_ms", lambda s: s.dropna().quantile(0.95)),
            )
            .reset_index()
            .sort_values("total_cost_usd", ascending=False)
        )

    def to_json(self, path: str | Path) -> Path:
        """Write per-request records + summary to a JSON file."""
        df = self.to_frame()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": {
                "total_requests": len(self),
                "total_cost_usd": self.total_cost(),
                "by_model": (
                    self.aggregate_by("provider", "model").to_dict(orient="records")
                    if len(self)
                    else []
                ),
                "by_tag": (
                    self.aggregate_by("tag").to_dict(orient="records")
                    if len(self)
                    else []
                ),
            },
            "records": json.loads(df.to_json(orient="records", date_format="iso")),
        }
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path
