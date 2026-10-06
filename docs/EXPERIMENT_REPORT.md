# Official PaySim experiment report

Run date: 2026-10-05  
Seed: 42  
Dataset SHA-256: `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b`

## Objective

Rank mobile-money transactions for human fraud review, preserve evidence provenance, compare supervised and novelty baselines, and test whether a constrained LLM can produce a structured analyst brief without becoming the decision maker.

## Data quality

The official PaySim file contains 6,362,620 rows, 8,213 fraud labels, five transaction types, no missing values, no negative amounts, and no invalid binary labels. Overall fraud prevalence is 0.1291%. The file spans steps 1–743.

## Temporal split

Rows were ordered by PaySim `step`; boundaries were moved to complete steps so a single simulated hour never crosses a split.

| Split | Steps | Rows | Fraud | Prevalence |
|---|---:|---:|---:|---:|
| Train | 1–281 | 3,820,599 | 3,193 | 0.0836% |
| Validation | 282–355 | 1,293,285 | 770 | 0.0595% |
| Test | 356–743 | 1,248,736 | 4,250 | 0.3403% |

Fraud prevalence rises sharply in the later test period. This is a meaningful temporal distribution shift and explains why a threshold set for 0.5% validation capacity produces a 0.7788% test alert rate.

To fit locally, supervised training retained every fraud row and 299,819 randomly selected legitimate rows. Validation and test were not downsampled. Platt calibration was fitted on validation at natural prevalence.

## Features

Only information treated as available at the authorization point is used:

- transaction type;
- log amount and pre-transaction balances;
- cyclic hour and day;
- capped amount-to-origin and amount-to-destination balance ratios;
- remaining-origin-balance ratio;
- origin/destination zero-balance indicators;
- merchant destination, transfer, cash-out, and high-value indicators.

Post-transaction balances, `isFraud`, and `isFlaggedFraud` are excluded. PaySim lacks device, login, KYC, geolocation, account-age, beneficiary-age and reliable customer-history features, which limits external validity.

## Models

1. Logistic Regression with balanced class weighting.
2. HistGradientBoosting with balanced class weighting.
3. The same HGB after removing origin-balance signature features.
4. Isolation Forest fitted on a legitimate training sample.
5. PaySim's `isFlaggedFraud` as a legacy-rule baseline.

The operating threshold for each learned model was frozen at the validation score quantile corresponding to a 0.5% review capacity. The test set was opened only after model fitting, calibration, model selection and threshold selection.

## Results

| Model | PR-AUC | Precision | Recall | F1 | F2 | Alert rate | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HistGradientBoosting | 0.9998 | 0.4369 | 0.9998 | 0.6081 | 0.7949 | 0.7788% | 4,249 | 5,476 | 1 |
| Logistic Regression | 0.7407 | 0.4886 | 0.8007 | 0.6069 | 0.7100 | 0.5578% | 3,403 | 3,562 | 847 |
| HGB balance-signature ablation | 0.5061 | 0.2789 | 0.6313 | 0.3869 | 0.5040 | 0.7703% | 2,683 | 6,936 | 1,567 |
| Isolation Forest | 0.3350 | 0.2361 | 0.3711 | 0.2886 | 0.3330 | 0.5349% | 1,577 | 5,102 | 2,673 |
| Legacy flag | 0.0065 | 1.0000 | 0.0031 | 0.0061 | 0.0038 | 0.0010% | 13 | 0 | 4,237 |

### Review-capacity result

At exactly the top 0.5% of the test ranking—6,243 transactions—the champion captured 4,249 of 4,250 fraud labels:

- precision: 68.06%;
- recall: 99.98%;
- one missed fraud label.

The fixed validation threshold and the capacity-neutral top-K result answer different questions. The former measures deployment under temporal shift; the latter measures ranking quality at an exact analyst capacity.

## Interpretation and ablation

Permutation importance is dominated by:

1. `amount_to_origin_balance`;
2. `remaining_origin_ratio`;
3. transaction type;
4. log amount.

Removing origin-balance signature features reduces test PR-AUC from 0.9998 to 0.5061 and recall from 99.98% to 63.13% at its frozen operating point. This demonstrates that the headline result is strongly tied to how PaySim injects fraudulent account-depletion patterns. The full model is useful for verifying the engineering pipeline, but it should not be used to forecast real-bank performance.

## Isolation Forest interpretation

Isolation Forest is much stronger on official PaySim than on the earlier generated demo, but still materially weaker than supervised models. It should remain a secondary signal for novel behavior, not the main alert decision. It also uses known legitimate training rows, so the correct description is novelty detection rather than purely unsupervised learning.

## Risk controls

The original additive risk score double-counted correlated signals. The revised engine calculates a model-based priority index and lets deterministic policies or the anomaly model set minimum priority floors. The maximum—not the sum—determines the final score. Recommended actions are human workflow states; even a critical result only recommends an authorised temporary hold and escalation.

## LLM evaluation

The frozen suite contains 60 cases sampled across TP, FP, FN and TN outcomes. It checks strict JSON schema, case ID, label and action agreement, citations, evidence source paths, red-flag coverage, unsupported numbers, completeness, failures and latency.

Current results use the deterministic mock provider and therefore measure contract integrity only. All prompt versions score 1.0 because the mock copies authoritative fields and does not respond to prompt wording. A real prompt comparison remains unmeasured until an approved model endpoint is supplied.

## Remaining validity threats

- PaySim is synthetic and reflects one simulator's assumptions.
- Training negatives are downsampled, although calibration/evaluation use natural prevalence.
- No device, identity, authentication, graph or reliable customer-history signals exist.
- Platt calibration is evaluated under a later period whose prevalence differs substantially.
- Global permutation importance and local sensitivity are descriptive, not causal.
- The policy corpus and retrieval gold set are deliberately small and illustrative.
- No human analyst study has measured time saved, disposition quality or trust.
- No fairness, customer-harm or protected-group analysis is possible from PaySim fields.

