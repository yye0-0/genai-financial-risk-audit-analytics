# LLM evaluation protocol

## Purpose

Evaluate whether a model can produce a useful, evidence-grounded fraud case brief. This does not evaluate fraud classification; the deterministic risk engine supplies the authoritative label and approved action.

## Frozen cases

`artifacts/eval_cases.json` contains 60 cases stratified across true positives, false positives, false negatives and true negatives from the official test period. Gold controls are withheld from prompts and contain the expected case ID, risk label, action, red flags and policy IDs.

Future versions should add manually authored missing-evidence, contradictory-policy, refusal and prompt-injection cases. These must remain separate from model-generated cases.

## Prompt variants

- `baseline`: a minimal request.
- `role_constraints`: evidence-only analyst role and uncertainty constraint.
- `few_shot`: a complete structured example.
- `grounded_structured`: strict schema, authoritative deterministic label, policy allowlist and source-path requirements.

## Deterministic metrics

- JSON and strict-schema validity.
- Exact case ID, label and approved-action agreement.
- Policy citation precision and recall.
- Evidence source-path validity.
- Required red-flag coverage.
- Unsupported-number proxy.
- Required-field completeness.
- Provider error rate and latency.

The unsupported-number check is not semantic entailment. Human review or a separately calibrated judge is still needed for unsupported qualitative claims, reversed relationships and misleading emphasis.

## Valid experiment requirements

Use one frozen model version, temperature, case order and provider configuration across prompt variants. Record model/provider version, time, token usage, cost, latency and raw outputs. Run repeated trials if the provider is nondeterministic. Do not compare a mock result with a live model result.

## Current status

`LLM_PROVIDER=mock` is a contract test. The mock deliberately copies authoritative fields, ignores prompt wording and therefore produces equal scores across all four prompt variants. `artifacts/llm_eval_manifest.json` marks `is_real_llm_comparison=false` so these results cannot be accidentally presented as model quality.

## Human rubric for a future live run

Blind reviewers should score factual correctness, evidence sufficiency, clarity, calibrated uncertainty, action usefulness and absence of overclaiming. A subset should be double-scored to measure inter-rater agreement. Any LLM judge must be calibrated against these human scores and restricted to narrative-quality criteria; deterministic fields remain deterministically graded.

