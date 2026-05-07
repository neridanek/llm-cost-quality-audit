"""HotpotQA loader — distractor-setting dev split.

Public dataset under CC BY-SA 4.0 (https://hotpotqa.github.io/). We cache the
download under `~/.cache/lcqa/hotpotqa/` so re-runs are offline.

Each HotpotQA item maps to one `EvalCase`:
- `question` → `EvalCase.question`
- `answer` → `EvalCase.expected_answer`
- `context` → flattened to `EvalCase.contexts` (one string per source paragraph,
  joining the sentences in order)
- HotpotQA `type` and `level` ride along in `EvalCase.metadata` for analysis

The `answer` field is also pre-filled (for cases where you want to score
against gold without running a model). Override it before scoring your
RAG system's actual answer.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from lcqa.eval.result import EvalCase

HOTPOTQA_DEV_URL = "https://hotpotqa-data.s3.amazonaws.com/hotpot_dev_distractor_v1.json"
DEFAULT_CACHE_DIR = Path(
    os.environ.get("LCQA_CACHE_DIR", str(Path.home() / ".cache" / "lcqa"))
) / "hotpotqa"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(url, dest)
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(
            f"Failed to download HotpotQA from {url}. "
            "If you're offline, pre-populate the cache or pass `path=` to load_hotpotqa."
        ) from e


def load_hotpotqa(
    path: str | Path | None = None,
    *,
    cache_dir: Path | None = None,
    download: bool = True,
) -> list[dict[str, Any]]:
    """Load raw HotpotQA dev split records.

    `path` — explicit JSON file; bypasses cache + download.
    `cache_dir` — defaults to `~/.cache/lcqa/hotpotqa/` (or `$LCQA_CACHE_DIR`).
    `download` — if False and cache miss, raises FileNotFoundError instead of fetching.
    """
    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        return _load_json(path)

    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    cached = cache_dir / "hotpot_dev_distractor_v1.json"
    if not cached.exists():
        if not download:
            raise FileNotFoundError(
                f"{cached} not in cache; pass download=True to fetch."
            )
        _download(HOTPOTQA_DEV_URL, cached)
    return _load_json(cached)


def _load_json(path: Path) -> list[dict[str, Any]]:
    with path.open() as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON list at {path}, got {type(data).__name__}")
    return data


def parse_hotpotqa(
    raw: Iterable[dict[str, Any]],
    *,
    limit: int | None = None,
    levels: tuple[str, ...] | None = None,
) -> list[EvalCase]:
    """Convert raw HotpotQA records into `EvalCase` instances.

    `limit` — cap the number of cases returned.
    `levels` — filter by HotpotQA difficulty: ('easy', 'medium', 'hard').
    """
    cases: list[EvalCase] = []
    for record in raw:
        if levels is not None and record.get("level") not in levels:
            continue
        cases.append(_record_to_case(record))
        if limit is not None and len(cases) >= limit:
            break
    return cases


def _record_to_case(record: dict[str, Any]) -> EvalCase:
    contexts: list[str] = []
    for entry in record.get("context", []):
        # entry shape: [title, [sentence, sentence, ...]]
        if not isinstance(entry, list) or len(entry) != 2:
            continue
        _title, sentences = entry
        if isinstance(sentences, list) and sentences:
            contexts.append(" ".join(s.strip() for s in sentences))
    return EvalCase(
        question=record.get("question", ""),
        answer=record.get("answer", ""),
        contexts=contexts,
        expected_answer=record.get("answer"),
        metadata={
            "_id": record.get("_id"),
            "type": record.get("type"),
            "level": record.get("level"),
        },
    )
