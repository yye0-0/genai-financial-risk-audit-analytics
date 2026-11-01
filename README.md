# GenAI-Enhanced Financial Risk & Audit Analytics

An interview-ready, human-in-the-loop fraud alert prioritisation and evidence-grounded investigation system built and evaluated on the official Kaggle PaySim file. Conventional models rank transactions; deterministic controls attach traceable red flags; a bounded Agent collects only allow-listed evidence; an optional LLM converts supplied evidence into a schema-validated analyst brief. The LLM never owns the fraud decision.

## Verified status

- Official PaySim: **6,362,620 transactions**, **8,213 fraud labels**, 0.1291% overall prevalence.
- Dataset SHA-256: `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b`.
- Complete time-step split: train steps 1–281, validation 282–355, test 356–743.
- Untouched test set: 1,248,736 transactions and 4,250 fraud labels.
- Models: Logistic Regression, HistGradientBoosting, balance-signature ablation HGB, Isolation Forest, and the PaySim legacy flag.
- Probability calibration: Platt scaling on the natural-prevalence validation period.
- Operating point: fixed on validation at a target 0.5% review capacity.
- Automated tests: **11 passed**.
- Dashboard: tested through Streamlit's application test runner with no exceptions.
- LLM status: the strict 60-case harness is complete; current results use a deterministic mock and are explicitly **not a real-LLM comparison**.
- Investigation status: a synthetic, three-case benchmark compares a fixed workflow against a bounded Agent on required evidence collection, safety gates, and tool cost. It is a workflow test, **not** a production AML claim.

## System design

```text
Official PaySim CSV
  ├─ chunked profile, schema checks, hash and data-quality report
  ├─ complete-step train / validation / test split
  ├─ leakage-controlled point-in-time features
  │
  ├─ Logistic Regression ─┐
  ├─ HistGradientBoosting ├─ Platt calibration ─ capacity threshold
  ├─ signature ablation ──┘
  ├─ Isolation Forest novelty baseline
  └─ PaySim legacy rule baseline
               │
       non-additive risk controls
               │
       policy grounding / TF-IDF retrieval
               │
       strict structured LLM narrative
       + bounded evidence investigation Agent
               │
       Streamlit investigation workspace
       + SQLite disposition audit trail
```

The design takes inspiration from the [AI Financial Compliance & Audit System](https://github.com/Kavya24golla/AI-Financial-Compliance-Audit-System). The evaluation philosophy follows [FinanceBench](https://arxiv.org/abs/2311.11944) on evidence-grounded finance evaluation and [TieOutBench](https://github.com/DimaMerc/TieOutBench) on using deterministic checks wherever possible.

## Data and leakage controls

The source is [Synthetic Financial Datasets for Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1), licensed CC BY-SA 4.0. Post-transaction balances, `isFraud`, and `isFlaggedFraud` are excluded from model features. The model uses transaction type, log-transformed pre-transaction amounts/balances, cyclic hour, day, capped amount-to-balance ratios, remaining-balance ratio, zero-balance indicators, merchant destination, transfer/cash-out indicators, and a high-value indicator.

The training period contains 3.82M rows. To fit locally, training retains all 3,193 fraud rows and a reproducible sample of 299,819 legitimate rows. Validation and test remain at natural prevalence. Metrics are never calculated on the sampled training distribution.

## Official test results

The validation operating threshold was selected for a 0.5% review capacity. Temporal distribution shift increased the champion's test alert rate to 0.7788%.

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 | Test alerts |
|---|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting | 0.9998 | 1.0000 | 0.4369 | 0.9998 | 0.6081 | 9,725 |
| Logistic Regression | 0.7407 | 0.9976 | 0.4886 | 0.8007 | 0.6069 | 6,965 |
| HGB without origin-balance signature | 0.5061 | 0.9753 | 0.2789 | 0.6313 | 0.3869 | 9,619 |
| Isolation Forest | 0.3350 | 0.9034 | 0.2361 | 0.3711 | 0.2886 | 6,679 |
| PaySim legacy flag | 0.0065 | 0.5015 | 1.0000 | 0.0031 | 0.0061 | 13 |

At the capacity-neutral top 0.5% of test transactions, HGB captured 4,249 of 4,250 fraud labels with 68.06% precision. This is a PaySim result, not an expected real-bank result.

### The result that matters most

Global permutation importance shows `amount_to_origin_balance` dominates the full HGB. Removing origin-balance signature features reduces PR-AUC from 0.9998 to 0.5061. PaySim injects fraud patterns that strongly resemble account depletion, so the near-perfect full-model result is a simulator upper bound. The ablation is included specifically to prevent a misleading performance claim.

See [the full experiment report](docs/EXPERIMENT_REPORT.md) and generated [model metrics](artifacts/model_metrics.json).

## LLM and prompt evaluation

Four prompt strategies are implemented:

1. baseline;
2. role plus constraints;
3. few-shot structured example;
4. grounded structured output.

The strict output contract requires an exact case ID, an enumerated risk label, approved workflow actions, evidence claims with dot-path citations, valid policy IDs, and explicit uncertainties. The 60-case frozen set is stratified across TP, FP, FN, and TN outcomes.

Automated checks include JSON validity, strict-schema validity, case/label/action agreement, citation precision and recall, evidence-path validity, red-flag coverage, unsupported-number checks, completeness, errors, and latency.

The current mock scores 1.0 because it copies authoritative case fields. This is a **contract test only**. It does not establish that any prompt is superior. A valid prompt comparison requires an approved live or local model and is blocked until such a provider is configured.

The six-query TF-IDF policy retrieval smoke test achieves Hit@1, Hit@3, and MRR of 1.0. The set is deliberately tiny and must not be presented as a broad RAG benchmark. Deterministic rule-to-policy ID lookup remains the safer path when a rule already knows its controlling policy.

## Bounded investigation Agent

The Agent is deliberately constrained. It can call only `get_case_context`, `get_account_history`, `get_counterparty_profile`, `get_network_snapshot`, and `get_policy_requirements`; all are read-only, return structured evidence, and have a hard five-tool budget. It has no unrestricted SQL/Cypher, external-network, disposition, account-freeze, or SAR-submission capability.

The fixed-workflow baseline calls every tool. The bounded Agent selects tools from the allow-list based on evidence needs, records the full trace, and stops at the budget. The included offline planner makes the demo reproducible; a live model planner can be introduced only behind the same typed decision and guardrails. See [the Agent evaluation guide](docs/AGENT_EVALUATION.md).

An Agent does not require a locally deployed open-weight LLM. Its core is the bounded state machine, typed tools, evidence validation, and human approval gate. The default provider is deterministic and credential-free. For a real comparison, the existing OpenAI-compatible adapter can point to an approved hosted endpoint, including DeepSeek-compatible endpoints, through `OPENAI_BASE_URL` and `LLM_MODEL`. This repository does not download or self-host a full DeepSeek model; a future local experiment should begin with a small quantized distill model and be reported separately.

## Risk scoring and explainability

Risk controls no longer add correlated signals together. The model, policy rules, and anomaly model independently set a priority floor; the maximum determines the triage score. This avoids counting an origin-balance pattern once inside the classifier and again as additive rule points.

The dashboard provides:

- global permutation importance;
- single-case one-feature-at-a-time sensitivity;
- rule evidence and policy IDs;
- calibrated model score and operating threshold;
- anomaly result;
- explicit uncertainty and human-decision warnings.

Local sensitivity is descriptive rather than causal: one engineered feature is replaced by its training reference while others remain fixed.

## Investigation and audit workflow

The Streamlit workspace includes:

- masked account identifiers;
- labels hidden outside Evaluation mode;
- model alerts, false negatives, and retained-case queues;
- policy evidence and strict LLM briefs;
- analyst status and notes;
- append-only SQLite disposition events;
- model version and dataset hash in exported case reports.

The system recommends an authorised temporary hold; it never directly freezes a customer or declares fraud.

## Run

Clone the repository and create a local environment file if a real LLM provider
will be used:

```bash
copy .env.example .env
```

The default `mock` provider needs no credentials. Never commit `.env`.

Install the verified environment:

```bash
python -m pip install -r requirements-lock.txt
```

Download PaySim if the verified CSV is not already present:

```bash
python -m pip install kaggle
python scripts/download_paysim.py
```

Run the full official experiment:

```bash
python -m src.modeling --data data/raw/PS_20174392719_1491204439457_log.csv
python -m src.pipeline --cases 60
python scripts/evaluate_retrieval.py
python -m src.evaluation --provider mock
python -m src.agent_evaluation
python -m pytest -q
streamlit run app.py
```

For a real LLM evaluation, install the optional dependency, set an approved endpoint and API key, and use `--provider openai`:

```bash
python -m pip install -e ".[llm]"
python -m src.evaluation --provider openai
```

Do not send real customer identifiers to an external model. The included OpenAI-compatible adapter is for approved test data/endpoints only.

## Main artifacts

- `artifacts/data_profile.json`: source hash, quality checks, splits and prevalence.
- `artifacts/model_metrics.json`: complete validation/test metrics and capacity curves.
- `artifacts/model_comparison.csv`: flat model comparison.
- `artifacts/feature_importance.csv`: validation permutation importance.
- `artifacts/scored_transactions.csv`: retained test alert/evaluation queue.
- `artifacts/eval_cases.json`: frozen stratified LLM cases and gold controls.
- `artifacts/llm_eval_manifest.json`: states whether a run is a real LLM comparison.
- `artifacts/retrieval_metrics.json`: small policy retrieval smoke test.
- `artifacts/run_manifest.json`: dataset/code hashes, environment versions and commands.
- `artifacts/case_audit.sqlite3`: local disposition log, created on first save.

## Repository data policy

The repository includes source code, tests, the small synthetic demo CSV, and
lightweight verified evaluation artifacts. The official PaySim CSV/ZIP,
generated model bundle, scored transaction queue, and local SQLite audit store
are intentionally excluded. Recreate them with the commands above after
downloading PaySim under its original licence.

