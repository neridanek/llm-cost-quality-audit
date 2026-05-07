.PHONY: install install-dev test lint format demo demo-full demo-real clean

PYTHON ?= python3
VENV ?= .venv
ACTIVATE = . $(VENV)/bin/activate

install:
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) && pip install --upgrade pip && pip install -e .

install-dev:
	$(PYTHON) -m venv $(VENV)
	$(ACTIVATE) && pip install --upgrade pip && pip install -e ".[all]"

test:
	$(ACTIVATE) && pytest tests/ -v

lint:
	$(ACTIVATE) && ruff check src/ tests/

format:
	$(ACTIVATE) && ruff format src/ tests/ && ruff check --fix src/ tests/

demo:        ## Mock pipeline + bundled fixture (3 cases, no API keys, no download)
	$(ACTIVATE) && python -m lcqa.demo.run_hotpotqa --mode mock \
		--data tests/fixtures/hotpotqa_sample.json \
		--regression-threshold 1.0

demo-full:   ## Mock pipeline + downloaded HotpotQA dev split (no API keys, ~50MB download)
	$(ACTIVATE) && python -m lcqa.demo.run_hotpotqa --mode mock --limit 200

demo-real:   ## Real OpenAI pipeline + HotpotQA (needs OPENAI_API_KEY, lands Day 5-6)
	$(ACTIVATE) && python -m lcqa.demo.run_hotpotqa --mode real --limit 100

clean:
	rm -rf build/ dist/ *.egg-info src/*.egg-info
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
