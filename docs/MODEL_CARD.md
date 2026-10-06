# Model card

## Intended use

Educational fraud-alert ranking and analyst decision support on PaySim-format mobile-money transactions. The model is not approved for autonomous blocking, account freezing, adverse customer decisions, or production banking use.

## Data

Official Kaggle PaySim: 6,362,620 rows and 8,213 fraud labels. The recorded SHA-256 is `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b`. Complete time steps define train, validation and test periods. Training retains all fraud rows and samples legitimate rows; validation and test keep natural prevalence.

## Models

- Champion: calibrated HistGradientBoosting classifier.
- Supervised baseline: calibrated Logistic Regression.
- Bias probe: HGB without origin-balance signature features.
- Novelty baseline: Isolation Forest fitted on known legitimate training rows.
- Rule baseline: PaySim `isFlaggedFraud`.

## Features and leakage controls

Transaction type, log amount/pre-transaction balances, cyclic hour, day, capped balance ratios, remaining-balance ratio, zero-balance indicators, merchant destination, transfer/cash-out indicators and high-value indicator. Post-transaction balances, labels and the legacy flag are excluded from learned models.

## Evaluation

PR-AUC is the primary ranking metric. ROC-AUC, Brier score, precision, recall, F1, F2, confusion counts, alert rate, calibration tables and Precision/Recall@K are retained. Thresholds are fixed using validation review capacity, not test labels.

## Explainability

Global permutation importance and per-case one-feature-at-a-time sensitivity are provided alongside deterministic red flags and policy IDs. Local sensitivity is not a causal attribution and may create internally inconsistent feature combinations.

## Known risks

- The champion relies heavily on PaySim's account-depletion signature; removing related features reduces PR-AUC from 0.9998 to 0.5061.
- PaySim is synthetic and lacks identity, device, authentication, graph and customer-history context.
- Fraud prevalence shifts from 0.0595% in validation to 0.3403% in test.
- Calibrated scores are not validated on real bank traffic.
- Training uses legitimate-row downsampling for local feasibility.
- No fairness, privacy, adversarial robustness or human-factors study has been completed.

## Human oversight

Outputs prioritise review only. An authorised analyst owns disposition and must document evidence and any override. Temporary holds must follow institutional policy and approval controls.

