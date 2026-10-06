# Interview guide

## 60-second version

“I built a human-in-the-loop fraud alert system on the full 6.36-million-row PaySim dataset. I split by complete time periods into train, validation and test, excluded post-transaction balances and legacy flags from learned features, retained all fraud examples while downsampling only training negatives, and calibrated supervised scores on natural-prevalence validation data. I compared Logistic Regression, gradient boosting, an origin-balance ablation, Isolation Forest and the legacy rule. The full gradient-boosting model ranked 4,249 of 4,250 test fraud labels in the top 0.5%, but permutation importance and ablation showed that the near-perfect result relies heavily on PaySim's simulated balance-depletion pattern—the ablated PR-AUC fell to 0.506. I therefore treat the result as a pipeline benchmark, not expected bank performance. A deterministic risk layer attaches policy-linked evidence; an optional LLM only writes a strict, source-cited analyst brief. The Streamlit workflow masks identifiers, hides labels, records human dispositions and versions every case report.”

## Why this architecture?

Structured fraud ranking needs reproducible models and deterministic policy controls. LLMs are poorly suited to owning a transaction decision but useful for combining verified evidence into a concise brief. The architecture separates those responsibilities so a narrative failure cannot change the underlying alert.

## Why PR-AUC instead of accuracy?

Only 0.1291% of the complete dataset is labelled fraud. A model predicting every transaction as legitimate would have excellent accuracy. PR-AUC, Precision@K and Recall@K directly measure alert quality under imbalance and limited analyst capacity.

## Why a chronological split?

Random splitting leaks future transaction distributions into training. Complete-step splits approximate training on history and deploying later. They also exposed a real evaluation issue: fraud prevalence increases from 0.0595% in validation to 0.3403% in test, causing a threshold designed for 0.5% validation capacity to alert 0.7788% of test rows.

## Why downsample only training negatives?

The full training period contains 3.82M rows. Keeping all 3,193 fraud examples and about 300K legitimate examples makes local fitting practical. Validation and test remain untouched so calibration, thresholds and reported metrics reflect natural prevalence rather than the sampled distribution.

## Why calibration?

Balanced class weights and negative downsampling distort raw probabilities. Platt scaling on the untouched validation distribution maps model scores back to a more interpretable range while preserving ranking. Calibration is still provisional because test prevalence shifts and no real bank data exists.

## How to explain the near-perfect HGB result

Do not present it as proof of production capability. Fraud in PaySim strongly resembles draining an origin account. The model's dominant feature is `amount_to_origin_balance`; when origin-balance signature features are removed, test PR-AUC falls from 0.9998 to 0.5061. This ablation is one of the project's most important findings because it detects benchmark shortcut learning.

## Why keep Isolation Forest?

It offers a separate novelty signal and test PR-AUC of 0.3350, but it catches only 37.11% of test fraud at its frozen operating point. It should support—not replace—the supervised classifier. Because it is fitted on known legitimate samples, describe it as novelty detection rather than fully unsupervised learning.

## How is the risk score different from the model score?

The calibrated model score estimates risk from learned patterns. The priority score controls workflow. Model, policy rules and anomaly detection each propose a priority floor, and the maximum wins. Signals are not added because the model and balance rules may rely on the same underlying evidence.

## What does the LLM contribute?

It converts verified signals, policy excerpts and missing-information warnings into a strict case brief. Every evidence claim must cite a dot-path in the case object. It cannot alter the risk label, invent an unapproved action or declare fraud proven.

## What is actually measured for the LLM?

The harness has 60 stratified cases and checks JSON/schema validity, exact case/label/action agreement, policy citation precision/recall, source-path validity, red-flag coverage, numeric grounding, completeness, latency and provider errors. Current mock results validate plumbing only. No real LLM comparison has been run because no approved endpoint or key is configured.

## What would production require?

- Real transaction, device, authentication, KYC and beneficiary-history data.
- Point-in-time feature infrastructure and graph/velocity features.
- Cost-based thresholds tied to analyst capacity and expected loss.
- Out-of-time and cross-market validation, drift and calibration monitoring.
- Data minimisation, tokenisation, RBAC, retention controls and approved private LLM hosting.
- Model governance, challenger approvals, human override review and customer-harm controls.
- Analyst studies measuring investigation time, decision quality and trust.

## Questions to ask the interviewer

- Is the goal authorization-time blocking, post-transaction monitoring or audit sampling?
- What is the daily investigation capacity and cost of a false negative versus a false positive?
- When do labels become available, and how noisy are they?
- Which identity, device, beneficiary and network features are available at decision time?
- Must model explanations satisfy a specific regulatory or internal validation standard?
- Can an LLM receive customer-level evidence, and which deployment boundary is approved?

