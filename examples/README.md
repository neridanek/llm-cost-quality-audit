# LCQA Examples

Three self-contained scripts. None require API keys — they use mock scorers and synthetic data so you can verify the toolkit works before wiring real LLM calls.

| File | What it demonstrates |
|---|---|
| `01_cost_analyzer.py` | Per-request cost tracking, tag-based attribution, JSON dump |
| `02_eval_harness.py` | Faithfulness (mock scorer) + accuracy (exact match) + latency tracker |
| `03_regression_ci.py` | CI regression check (exit 0/1 based on quality drop vs baseline) |

## Run them

```bash
make install-dev
python examples/01_cost_analyzer.py
python examples/02_eval_harness.py
python examples/03_regression_ci.py
```

Outputs land in `eval_results/`.

## Real Ragas / DeepEval scoring

Once you set `OPENAI_API_KEY`, install the eval extra and swap the mock scorer for the default Ragas-backed one:

```bash
pip install -e ".[eval]"
export OPENAI_API_KEY=...
```

```python
from lcqa.eval import score_faithfulness

# Default scorer = Ragas Faithfulness with gpt-4o-mini judge
result = score_faithfulness(cases)
```

See `src/lcqa/eval/faithfulness.py` for `FaithfulnessConfig` options (judge model, seed, custom scorer override).
