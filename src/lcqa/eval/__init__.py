"""Eval harness: faithfulness, answer accuracy, latency tracking, regression checks."""

from lcqa.eval.accuracy import AccuracyConfig, score_accuracy
from lcqa.eval.faithfulness import FaithfulnessConfig, score_faithfulness
from lcqa.eval.latency import LatencyTracker, latency_summary
from lcqa.eval.regression import RegressionCheck, RegressionConfig
from lcqa.eval.result import EvalCase, EvalResult

__all__ = [
    "AccuracyConfig",
    "EvalCase",
    "EvalResult",
    "FaithfulnessConfig",
    "LatencyTracker",
    "RegressionCheck",
    "RegressionConfig",
    "latency_summary",
    "score_accuracy",
    "score_faithfulness",
]
