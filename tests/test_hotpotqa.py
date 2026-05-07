"""HotpotQA loader tests — fixture-based, no network."""

from __future__ import annotations

from pathlib import Path

import pytest

from lcqa.demo.hotpotqa import load_hotpotqa, parse_hotpotqa

FIXTURE = Path(__file__).parent / "fixtures" / "hotpotqa_sample.json"


def test_load_from_explicit_path() -> None:
    raw = load_hotpotqa(path=FIXTURE)
    assert isinstance(raw, list)
    assert len(raw) == 3


def test_load_missing_path_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_hotpotqa(path="/nonexistent/file.json")


def test_load_no_download_no_cache_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not in cache"):
        load_hotpotqa(cache_dir=tmp_path, download=False)


def test_parse_basic_shape() -> None:
    raw = load_hotpotqa(path=FIXTURE)
    cases = parse_hotpotqa(raw)
    assert len(cases) == 3
    first = cases[0]
    assert "Scott Derrickson" in first.question
    assert first.expected_answer == "yes"
    assert first.contexts is not None
    assert len(first.contexts) == 3
    assert first.metadata["level"] == "hard"
    assert first.metadata["type"] == "comparison"


def test_parse_contexts_concatenated_per_paragraph() -> None:
    raw = load_hotpotqa(path=FIXTURE)
    cases = parse_hotpotqa(raw)
    derrickson_ctx = cases[0].contexts[0]
    assert "Scott Derrickson" in derrickson_ctx
    assert "Los Angeles" in derrickson_ctx
    assert " " in derrickson_ctx  # sentences joined


def test_parse_limit() -> None:
    raw = load_hotpotqa(path=FIXTURE)
    cases = parse_hotpotqa(raw, limit=2)
    assert len(cases) == 2


def test_parse_level_filter() -> None:
    raw = load_hotpotqa(path=FIXTURE)
    easy = parse_hotpotqa(raw, levels=("easy",))
    hard = parse_hotpotqa(raw, levels=("hard",))
    assert len(easy) == 1
    assert easy[0].metadata["level"] == "easy"
    assert len(hard) == 1
    assert hard[0].metadata["level"] == "hard"


def test_parse_keeps_id_in_metadata() -> None:
    raw = load_hotpotqa(path=FIXTURE)
    cases = parse_hotpotqa(raw)
    ids = [c.metadata.get("_id") for c in cases]
    assert all(id_ for id_ in ids)
    assert len(set(ids)) == 3
