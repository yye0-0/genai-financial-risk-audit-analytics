# Bounded investigation Agent evaluation

This repository evaluates a deliberately constrained investigation Agent, not an autonomous transaction-decision system. The Agent may choose only five read-only, typed evidence tools. It has no SQL, Cypher, file, network, disposition, account-freeze, or SAR-submission tool.

## Strategies compared

`fixed_workflow` invokes every tool in a prescribed sequence. `bounded_agent` uses a deterministic offline planner for reproducibility: it starts with case context, then selects account history, counterparty evidence, network evidence, and policy requirements only when the case signals warrant them. Both strategies use a hard five-tool budget.

The deterministic planner is intentionally a test double. `LLMToolPlanner` is also included for a live experiment: it accepts only a JSON decision, checks the tool name against the same allow-list, rejects repeats, and retains the hard budget. It must not receive raw database access. Configure an approved OpenAI-compatible provider first; do not use the default mock for this path because the mock produces narrative JSON rather than a tool decision.

## Synthetic benchmark

`data/investigation_cases.json` holds three hand-authored, fictional cases. The `investigation_context` field is explicitly synthetic enrichment. The benchmark is for workflow and measurement validation; it is not a claim about production AML detection, false-positive reduction, or regulatory suitability.

Each case declares the evidence tools required to reach a reviewable conclusion. The evaluator calculates:

- required-tool recall;
- task success rate, meaning all required tools were collected;
- unsafe-close rate, a hard safety signal for high/critical cases;
- tool calls and elapsed time.

Run it with:

```bash
python -m src.agent_evaluation
```

The resulting CSV files are stored locally in `artifacts/` and ignored by Git. Add a result to the README only after it has been generated from a frozen case set and reviewed.
