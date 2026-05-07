# Architectural Decision Records

Short, focused notes on design decisions that aren't obvious from the code. Format: **Context → Decision → Alternatives → Consequences**.

| ADR | Title | Status |
|---|---|---|
| [0001](0001-ragas-over-trulens.md) | Ragas over TruLens for faithfulness scoring | accepted |
| [0002](0002-batch-vs-streaming-eval.md) | Batch eval over streaming for the demo benchmark | accepted |
| [0003](0003-cost-breakdown-methodology.md) | Cost breakdown methodology: per-request attribution | accepted |
| [0004](0004-hotpotqa-dataset-choice.md) | HotpotQA over MS MARCO for the public demo | proposed |
| [0005](0005-pricing-table-maintenance.md) | Provider pricing table maintenance + version policy | accepted |

## Adding a new ADR

1. Copy the latest ADR file as `NNNN-short-title.md` (incrementing the number).
2. Set status `proposed` until validated; `accepted` once shipped; `superseded by ADR-NNNN` if replaced.
3. Update this index.
4. Open a PR — ADRs land via review like any other change.
