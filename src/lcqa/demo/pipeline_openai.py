"""Real OpenAI + Cohere RAG pipelines for the HotpotQA demo.

Two pipelines implementing the `Pipeline` protocol from `pipeline.py`:

- `OpenAIBaseline` — naive RAG: GPT-4o, top-5 contexts, no cache, no rerank,
  no model routing. Establishes the cost + quality baseline.

- `OpenAIOptimized` — 3-layer optimized stack:
    1. Prompt cache: system prompt tokens flagged as cached (reflects real
       Anthropic/OpenAI prompt-cache hit after first warm request).
    2. Context reduction: retrieve top-10, rerank to top-3 via Cohere
       `rerank-english-v3.0`. Falls back to first-3 if Cohere unavailable.
    3. Model routing: short / FAQ-style questions → `gpt-4o-mini`;
       complex multi-hop → `gpt-4o`.

Both classes accept injectable `client` / `cohere_client` for unit tests.
"""

from __future__ import annotations

import time
from typing import Any

from lcqa.cost.analyzer import RequestRecord
from lcqa.demo.pipeline import PipelineOutput
from lcqa.eval.result import EvalCase

# System prompt used by both pipelines (kept identical so cache comparisons are fair).
_SYSTEM_PROMPT = (
    "You are a factual question-answering assistant. "
    "Answer questions using ONLY the provided context passages. "
    "If the answer is not in the context, respond with 'I don't know.' "
    "Be concise — one to two sentences maximum."
)
# Approximate token count for _SYSTEM_PROMPT (used for cache simulation).
_SYSTEM_PROMPT_TOKENS: int = 55

# FAQ question prefixes that signal a short, single-hop question → route to mini.
_FAQ_PREFIXES = (
    "what is ", "what was ", "what are ",
    "who is ", "who was ", "who are ",
    "when did ", "when was ", "when is ",
    "where is ", "where was ",
    "how many ", "how much ",
)


class OpenAIBaseline:
    """Naive RAG pipeline: GPT-4o, top-5 contexts, no optimizations."""

    name: str = "openai-baseline"

    def __init__(
        self,
        *,
        model: str = "gpt-4o",
        top_k: int = 5,
        client: Any | None = None,
    ) -> None:
        self._model = model
        self._top_k = top_k
        self._client = client or _default_openai_client()

    def run(self, case: EvalCase) -> PipelineOutput:
        contexts = (case.contexts or [])[: self._top_k]
        user_message = _build_user_message(case.question, contexts)

        t0 = time.monotonic()
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            seed=42,
        )
        latency_ms = (time.monotonic() - t0) * 1000

        answer = (response.choices[0].message.content or "I don't know.").strip()
        usage = response.usage

        record = RequestRecord(
            provider="openai",
            model=self._model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            latency_ms=latency_ms,
            tag="baseline:rag-answer",
        )
        return PipelineOutput(
            answer=answer,
            retrieved_contexts=contexts,
            cost_record=record,
        )


class OpenAIOptimized:
    """Optimized RAG pipeline: prompt cache + Cohere rerank + model routing."""

    name: str = "openai-optimized"

    def __init__(
        self,
        *,
        model_complex: str = "gpt-4o",
        model_simple: str = "gpt-4o-mini",
        top_k_retrieve: int = 10,
        top_k_rerank: int = 3,
        client: Any | None = None,
        cohere_client: Any | None = None,
    ) -> None:
        self._model_complex = model_complex
        self._model_simple = model_simple
        self._top_k_retrieve = top_k_retrieve
        self._top_k_rerank = top_k_rerank
        self._client = client or _default_openai_client()
        self._cohere = cohere_client or _try_default_cohere_client()

    # ------------------------------------------------------------------
    # Pipeline protocol
    # ------------------------------------------------------------------

    def run(self, case: EvalCase) -> PipelineOutput:
        contexts_broad = (case.contexts or [])[: self._top_k_retrieve]
        contexts = self._rerank(case.question, contexts_broad)
        model = self._route_model(case.question)
        user_message = _build_user_message(case.question, contexts)

        t0 = time.monotonic()
        response = self._client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            seed=42,
        )
        latency_ms = (time.monotonic() - t0) * 1000

        answer = (response.choices[0].message.content or "I don't know.").strip()
        usage = response.usage

        # Simulate prompt-cache hit for system prompt tokens (always warm after
        # first request in a real deployment; demo treats every call as warm).
        cached = min(_SYSTEM_PROMPT_TOKENS, usage.prompt_tokens)

        record = RequestRecord(
            provider="openai",
            model=model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cached_input_tokens=cached,
            latency_ms=latency_ms,
            tag="optimized:rag-answer",
        )
        return PipelineOutput(
            answer=answer,
            retrieved_contexts=contexts,
            cost_record=record,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rerank(self, query: str, contexts: list[str]) -> list[str]:
        """Return top-k contexts ranked by relevance. Falls back to slice if Cohere down."""
        if self._cohere is None or not contexts:
            return contexts[: self._top_k_rerank]
        try:
            resp = self._cohere.rerank(
                query=query,
                documents=contexts,
                top_n=self._top_k_rerank,
                model="rerank-english-v3.0",
            )
            return [contexts[r.index] for r in resp.results]
        except Exception:
            return contexts[: self._top_k_rerank]

    def _route_model(self, question: str) -> str:
        """Route short / FAQ-style questions to the cheaper model."""
        q = question.lower().strip()
        if len(question) < 60 or any(q.startswith(p) for p in _FAQ_PREFIXES):
            return self._model_simple
        return self._model_complex


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------


def _build_user_message(question: str, contexts: list[str]) -> str:
    context_str = "\n\n".join(f"[{i + 1}] {ctx}" for i, ctx in enumerate(contexts))
    return f"Context:\n{context_str}\n\nQuestion: {question}"


def _default_openai_client() -> Any:
    try:
        import openai
    except ImportError as e:
        raise ImportError(
            "openai package not installed. Run `pip install lcqa[real]` "
            "or `pip install openai`."
        ) from e
    # max_retries=3 handles transient rate-limit errors automatically.
    return openai.OpenAI(max_retries=3)


def _try_default_cohere_client() -> Any | None:
    """Return a Cohere client if the SDK is installed and COHERE_API_KEY is set."""
    try:
        import cohere
        return cohere.Client()
    except (ImportError, Exception):
        return None
