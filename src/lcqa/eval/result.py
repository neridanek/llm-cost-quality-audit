"""Shared eval data models."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    """One question/answer item with optional ground truth and retrieved context.

    `contexts` is required for faithfulness scoring; `expected_answer` for
    accuracy scoring. A single case can support both metrics if both are set.
    """

    question: str
    answer: str
    contexts: list[str] | None = None
    expected_answer: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvalResult(BaseModel):
    """Output of one metric over a list of `EvalCase`."""

    metric: str
    score: float = Field(ge=0.0, le=1.0)
    per_item_scores: list[float]
    n_items: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def to_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
        return path

    @classmethod
    def from_json(cls, path: str | Path) -> EvalResult:
        return cls.model_validate(json.loads(Path(path).read_text()))
