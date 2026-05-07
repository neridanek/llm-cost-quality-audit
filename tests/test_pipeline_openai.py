"""Tests for OpenAI real-mode pipelines (all API calls mocked)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from lcqa.demo.pipeline import PipelineOutput
from lcqa.demo.pipeline_openai import (
    OpenAIBaseline,
    OpenAIOptimized,
    _build_user_message,
)
from lcqa.eval.result import EvalCase

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_response(content: str, prompt_tokens: int = 120, completion_tokens: int = 20) -> MagicMock:
    """Build a minimal openai.ChatCompletion-shaped mock."""
    msg = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=msg)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(choices=[choice], usage=usage)


def _case(question: str = "Who founded Wikipedia?", contexts: list[str] | None = None) -> EvalCase:
    return EvalCase(
        question=question,
        answer="",
        contexts=contexts or ["Wikipedia was founded by Jimmy Wales and Larry Sanger in 2001."],
        expected_answer="Jimmy Wales and Larry Sanger",
    )


# ---------------------------------------------------------------------------
# OpenAIBaseline
# ---------------------------------------------------------------------------


class TestOpenAIBaseline:
    def test_run_returns_pipeline_output(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_response("Jimmy Wales and Larry Sanger.")

        pipeline = OpenAIBaseline(client=mock_client)
        result = pipeline.run(_case())

        assert isinstance(result, PipelineOutput)
        assert "Jimmy Wales" in result.answer
        assert result.cost_record.provider == "openai"
        assert result.cost_record.model == "gpt-4o"
        assert result.cost_record.tag == "baseline:rag-answer"

    def test_uses_top_k_contexts(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_response("answer")

        contexts = [f"context {i}" for i in range(10)]
        pipeline = OpenAIBaseline(client=mock_client, top_k=3)
        result = pipeline.run(_case(contexts=contexts))

        assert len(result.retrieved_contexts) == 3

    def test_no_cached_tokens(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_response("answer")

        pipeline = OpenAIBaseline(client=mock_client)
        result = pipeline.run(_case())

        assert result.cost_record.cached_input_tokens == 0

    def test_records_token_counts(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_response(
            "answer", prompt_tokens=200, completion_tokens=30
        )
        pipeline = OpenAIBaseline(client=mock_client)
        result = pipeline.run(_case())

        assert result.cost_record.input_tokens == 200
        assert result.cost_record.output_tokens == 30

    def test_latency_recorded(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_response("answer")

        pipeline = OpenAIBaseline(client=mock_client)
        result = pipeline.run(_case())

        assert result.cost_record.latency_ms is not None
        assert result.cost_record.latency_ms >= 0

    def test_empty_contexts_runs(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_response("I don't know.")

        pipeline = OpenAIBaseline(client=mock_client)
        result = pipeline.run(EvalCase(question="test?", answer="", contexts=[]))

        assert result.answer == "I don't know."


# ---------------------------------------------------------------------------
# OpenAIOptimized
# ---------------------------------------------------------------------------


class TestOpenAIOptimized:
    def test_run_returns_pipeline_output(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_response("Jimmy Wales.")

        pipeline = OpenAIOptimized(client=mock_client, cohere_client=None)
        result = pipeline.run(_case())

        assert isinstance(result, PipelineOutput)
        assert result.cost_record.tag == "optimized:rag-answer"

    def test_cached_tokens_set(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_response(
            "answer", prompt_tokens=200, completion_tokens=15
        )
        pipeline = OpenAIOptimized(client=mock_client, cohere_client=None)
        result = pipeline.run(_case())

        # cached_input_tokens should be > 0 (system prompt simulated as cached)
        assert result.cost_record.cached_input_tokens > 0
        assert result.cost_record.cached_input_tokens <= result.cost_record.input_tokens

    def test_reduces_contexts_without_cohere(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_response("answer")

        contexts = [f"context {i}" for i in range(10)]
        pipeline = OpenAIOptimized(client=mock_client, cohere_client=None, top_k_rerank=3)
        result = pipeline.run(_case(contexts=contexts))

        assert len(result.retrieved_contexts) == 3

    def test_cohere_rerank_used_when_available(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_response("answer")

        # Cohere mock: reverses the ranking (returns last 3 first)
        contexts = ["ctx-A", "ctx-B", "ctx-C", "ctx-D"]
        cohere_result = SimpleNamespace(
            results=[SimpleNamespace(index=3), SimpleNamespace(index=2), SimpleNamespace(index=1)]
        )
        mock_cohere = MagicMock()
        mock_cohere.rerank.return_value = cohere_result

        pipeline = OpenAIOptimized(client=mock_client, cohere_client=mock_cohere, top_k_rerank=3)
        result = pipeline.run(_case(contexts=contexts))

        assert result.retrieved_contexts == ["ctx-D", "ctx-C", "ctx-B"]
        mock_cohere.rerank.assert_called_once()

    def test_cohere_failure_falls_back_to_slice(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_response("answer")

        mock_cohere = MagicMock()
        mock_cohere.rerank.side_effect = RuntimeError("API down")

        contexts = [f"ctx-{i}" for i in range(5)]
        pipeline = OpenAIOptimized(client=mock_client, cohere_client=mock_cohere, top_k_rerank=2)
        result = pipeline.run(_case(contexts=contexts))

        # Should fall back to first-2 without raising
        assert len(result.retrieved_contexts) == 2
        assert result.retrieved_contexts == contexts[:2]

    def test_model_routing_simple_question(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_response("answer")

        pipeline = OpenAIOptimized(
            client=mock_client,
            cohere_client=None,
            model_simple="gpt-4o-mini",
            model_complex="gpt-4o",
        )
        pipeline.run(_case(question="What is the capital of France?"))

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"

    def test_model_routing_complex_question(self) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_response("answer")

        pipeline = OpenAIOptimized(
            client=mock_client,
            cohere_client=None,
            model_simple="gpt-4o-mini",
            model_complex="gpt-4o",
        )
        long_question = (
            "What were the key geopolitical factors that contributed to "
            "the dissolution of the Soviet Union in 1991?"
        )
        pipeline.run(_case(question=long_question))

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestBuildUserMessage:
    def test_includes_question(self) -> None:
        msg = _build_user_message("Who is Ada Lovelace?", ["She was a mathematician."])
        assert "Who is Ada Lovelace?" in msg

    def test_numbers_contexts(self) -> None:
        msg = _build_user_message("Q?", ["ctx-1", "ctx-2"])
        assert "[1] ctx-1" in msg
        assert "[2] ctx-2" in msg

    def test_empty_contexts(self) -> None:
        msg = _build_user_message("Q?", [])
        assert "Question: Q?" in msg


class TestRouteModel:
    """Tests for the FAQ / length-based routing logic via the pipeline."""

    @pytest.mark.parametrize(
        "question,expected_model",
        [
            ("What is Python?", "mini"),
            ("Who was Einstein?", "mini"),
            ("When did WW2 end?", "mini"),
            ("How many legs does a spider have?", "mini"),
            (
                "Explain the long-term economic consequences of the 2008 "
                "financial crisis on emerging markets and their debt structures.",
                "complex",
            ),
        ],
    )
    def test_routing(self, question: str, expected_model: str) -> None:
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = _fake_response("answer")

        pipeline = OpenAIOptimized(
            client=mock_client,
            cohere_client=None,
            model_simple="mini",
            model_complex="complex",
        )
        pipeline.run(_case(question=question))

        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == expected_model
