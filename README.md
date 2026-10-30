# GenAI-Enhanced Financial Risk & Audit Analytics

An interview-ready, human-in-the-loop fraud alert prioritisation system built and evaluated on the official Kaggle PaySim file. Conventional models rank transactions; deterministic controls attach traceable red flags; an optional LLM converts supplied evidence into a schema-validated analyst brief. The LLM never owns the fraud decision.

## Product walkthrough and documentation

The workspace now opens directly from a clone using the 60 committed frozen
cases when the full scored queue is unavailable. This is an outcome-stratified
walkthrough, not a natural-prevalence production queue. Saved full-test metrics
are labelled separately. The default mock needs no credentials or model bundle.

```bash
python -m pip install -r requirements-lock.txt
python -m streamlit run app.py
```

Analysts can retain a generated brief across interactions, edit a separate
reviewed summary, record feedback and a disposition rationale, and export the
original draft plus review history. Schema checks are followed by deterministic
case/label/action/source/policy checks; natural-language truth still needs human
review. Local concurrent edits are detected before saving.

- [中文 PRD: users, priorities, workflow and acceptance](docs/PRD.md)
- [Quickstart and three-minute demo](docs/QUICKSTART.md)
- [User-validation plan and blank research templates](docs/USER_VALIDATION.md)
- [Product decisions and iteration log](docs/PRODUCT_CHANGELOG.md)
- [Validation evidence and remaining work](docs/PRODUCT_VALIDATION.md)

This is a personal public-data PoC. User interviews, enterprise rollout and a
live-model comparison are not claimed as completed work.

## Verified status

- Official PaySim: **6,362,620 transactions**, **8,213 fraud labels**, 0.1291% overall prevalence.
- Dataset SHA-256: `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b`.
- Complete time-step split: train steps 1–281, validation 282–355, test 356–743.
- Untouched test set: 1,248,736 transactions and 4,250 fraud labels.
- Models: Logistic Regression, HistGradientBoosting, balance-signature ablation HGB, Isolation Forest, and the PaySim legacy flag.
- Probability calibration: Platt scaling on the natural-prevalence validation period.
- Operating point: fixed on validation at a target 0.5% review capacity.
- Original experiment tests: **8 passed**; the expanded product suite is documented in [product validation](docs/PRODUCT_VALIDATION.md).
- Dashboard: tested through Streamlit's application test runner with no exceptions.
- LLM status: the strict 60-case harness is complete; current results use a deterministic mock and are explicitly **not a real-LLM comparison**.

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
- original AI draft, editable reviewed summary, feedback category and previous status;
- separate demonstration audit store and optimistic concurrent-edit checks;
- model version and dataset hash in exported case reports.

The system recommends an authorised temporary hold; it never directly freezes a customer or declares fraud.

## Run

Clone the repository and create a local environment file if a real LLM provider
will be used:

```bash
cp .env.example .env
```

The default `mock` provider needs no credentials. Never commit `.env`.
The application does not auto-load `.env`: export its values in your shell or
load them through your IDE/process manager before starting a real provider.
On Windows use `copy .env.example .env` for the copy step.

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
python -m pytest -q
streamlit run app.py
```

For a real LLM evaluation, install the optional dependency, set an approved endpoint and API key, and use `--provider openai`:

```bash
python -m pip install -e ".[llm]"
python -m src.evaluation --provider openai
```

Do not send real customer identifiers to an external model. The included OpenAI-compatible adapter is for approved test data/endpoints only.

New evaluation runs default to `artifacts/evaluations/<provider>` so they do not
overwrite published experiment results. Start with `--limit 5 --prompt-version
grounded_structured`, inspect failures, then run a full comparison. Usage and
model versions are captured when the endpoint returns them. Optional
`--input-price-per-million` and `--output-price-per-million` prices produce a USD
estimate; missing usage or prices remain unknown. A configured real provider is
not itself evidence of a successful comparison. See [the evaluation protocol](docs/LLM_EVALUATION.md).

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

