"""RAG pipeline interface + mock implementation.

`Pipeline` is the contract any RAG implementation needs to satisfy to plug into
the demo runner. `MockPipeline` is a deterministic, no-API fallback used when
running the demo without API keys (smoke-tests the plumbing).

The real OpenAI + Cohere pipeline lands in `pipeline_openai.py` Day 5-6.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from lcqa.cost.analyzer import RequestRecord
from lcqa.eval.result import EvalCase


@dataclass(frozen=True)
class PipelineOutput:
    """One RAG run's answer + the cost record produced for it."""

    answer: str
    retrieved_contexts: list[str]
    cost_record: RequestRecord


class Pipeline(Protocol):
    """A RAG pipeline that takes a question and returns an answer + cost trace."""

    name: str

    def run(self, case: EvalCase) -> PipelineOutput: ...


class MockPipeline:
    """Deterministic mock for offline smoke-testing the demo runner.

    Answer = the first sentence of the first matching context (case-insensitive
    keyword overlap with the question). Cost = synthetic but realistic (token
    counts derived from input/output length).
    """

    def __init__(
        self,
        *,
        name: str = "mock",
        provider: str = "openai",
        model: str = "gpt-4o",
        seed: int = 42,
        latency_ms_baseline: tuple[float, float] = (1500.0, 800.0),  # (mean, stddev)
    ) -> None:
        self.name = name
        self._provider = provider
        self._model = model
        self._rng = random.Random(seed)
        self._latency_mean, self._latency_std = latency_ms_baseline

    def run(self, case: EvalCase) -> PipelineOutput:
        contexts = case.contexts or []
        answer = self._fake_answer(case.question, contexts)

        # Token counts: input = sum(len(ctx) // 4) + len(question)//4; output ≈ len(answer)//4
        input_tokens = sum(max(len(c) // 4, 1) for c in contexts) + max(len(case.question) // 4, 1)
        output_tokens = max(len(answer) // 4, 1)

        # Latency: log-normal-ish, deterministic per case via case-id hash
        case_id = case.metadata.get("_id") or case.question
        seed_offset = int(hashlib.md5(str(case_id).encode()).hexdigest(), 16) % 1000
        rng = random.Random(self._rng.randint(0, 10_000) + seed_offset)
        latency_ms = max(50.0, rng.gauss(self._latency_mean, self._latency_std))

        record = RequestRecord(
            provider=self._provider,
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            tag=f"{self.name}:rag-answer",
        )
        return PipelineOutput(
            answer=answer,
            retrieved_contexts=contexts,
            cost_record=record,
        )

    @staticmethod
    def _fake_answer(question: str, contexts: Iterable[str]) -> str:
        """Heuristic: pick the first sentence containing a question keyword."""
        q_words = {w.lower().strip(".,?!") for w in question.split() if len(w) > 3}
        for ctx in contexts:
            for sentence in ctx.split(". "):
                if any(w in sentence.lower() for w in q_words):
                    return sentence.strip()
        # No keyword match — return first sentence of first context, or a stub.
        for ctx in contexts:
            return ctx.split(". ")[0].strip()
        return "I don't know."


class OptimizedMockPipeline(MockPipeline):
    """Mock variant that simulates the optimized stack: cheaper model, faster, prompt cache.

    Behaves like `MockPipeline` but routes through gpt-4o-mini, applies a fake
    cache discount, and reports lower latency. Useful for smoke-testing the
    baseline-vs-optimized comparison without API keys.
    """

    def __init__(self, *, seed: int = 42) -> None:
        super().__init__(
            name="optimized-mock",
            provider="openai",
            model="gpt-4o-mini",
            seed=seed,
            latency_ms_baseline=(700.0, 300.0),
        )

    def run(self, case: EvalCase) -> PipelineOutput:
        out = super().run(case)
        # Simulate: 60% of input tokens served from prompt cache, fewer total input tokens
        # because of context-size reduction (rerank to top 3 of 5).
        rec = out.cost_record
        reduced_input = int(rec.input_tokens * 0.55)
        cached = int(reduced_input * 0.60)
        new_record = RequestRecord(
            provider=rec.provider,
            model=rec.model,
            input_tokens=reduced_input,
            output_tokens=rec.output_tokens,
            cached_input_tokens=cached,
            latency_ms=rec.latency_ms,
            tag=f"{self.name}:rag-answer",
        )
        return PipelineOutput(
            answer=out.answer,
            retrieved_contexts=out.retrieved_contexts[:3],  # top-3 after rerank
            cost_record=new_record,
        )
