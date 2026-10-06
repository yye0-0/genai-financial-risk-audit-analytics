from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import PATHS
from .rag import retrieve_for_flags
from .risk_engine import assess_transaction


def _stratified_cases(scored: pd.DataFrame, n_cases: int, seed: int) -> pd.DataFrame:
    work = scored.copy()
    work["outcome"] = "TN"
    work.loc[(work["predicted_alert"] == 1) & (work["isFraud"] == 1), "outcome"] = "TP"
    work.loc[(work["predicted_alert"] == 1) & (work["isFraud"] == 0), "outcome"] = "FP"
    work.loc[(work["predicted_alert"] == 0) & (work["isFraud"] == 1), "outcome"] = "FN"
    per_group = max(1, n_cases // 4)
    parts = [group.sample(min(per_group, len(group)), random_state=seed)
             for _, group in work.groupby("outcome") if len(group)]
    selected = pd.concat(parts).drop_duplicates()
    if len(selected) < n_cases:
        remaining = work.drop(selected.index, errors="ignore")
        selected = pd.concat([selected, remaining.sample(
            min(n_cases - len(selected), len(remaining)), random_state=seed)])
    return selected.head(n_cases)


def build_cases(scored: pd.DataFrame, anomaly_threshold: float,
                model_threshold: float, n_cases: int = 60, seed: int = 42) -> list[dict]:
    ranked = _stratified_cases(scored, n_cases, seed)
    cases = []
    for idx, row in ranked.iterrows():
        assessment = assess_transaction(row.to_dict(), float(row["fraud_probability"]),
                                        float(row["anomaly_score"]), anomaly_threshold,
                                        model_threshold)
        transaction = {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()
                       if k not in {"isFraud", "isFlaggedFraud", "outcome", "predicted_alert",
                                    "fraud_probability"}
                       and not k.endswith("_score")}
        policy_ids = [f.policy_id for f in assessment.red_flags]
        case_id = f"CASE-{int(idx):06d}"
        cases.append({
            "case_id": case_id,
            "transaction": transaction,
            "assessment": assessment.to_dict(),
            "retrieved_policies": retrieve_for_flags(policy_ids),
            "gold": {
                "case_id": case_id, "risk_label": assessment.risk_label,
                "recommended_action": assessment.recommended_action,
                "red_flag_codes": [f.code for f in assessment.red_flags],
                "policy_ids": sorted(set(policy_ids)),
            },
            "metadata": {"evaluation_outcome": row.get("outcome", "unknown"),
                         "dataset_label": int(row["isFraud"])},
        })
    return cases


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--scored", type=Path, default=PATHS.artifacts / "scored_transactions.csv")
    p.add_argument("--metrics", type=Path, default=PATHS.artifacts / "model_metrics.json")
    p.add_argument("--output", type=Path, default=PATHS.artifacts / "eval_cases.json")
    p.add_argument("--cases", type=int, default=60)
    args = p.parse_args()
    scored = pd.read_csv(args.scored)
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    anomaly_threshold = metrics["validation"]["isolation_forest"]["threshold"]
    model_threshold = metrics["validation"][metrics["champion"]]["threshold"]
    scored["fraud_probability"] = scored["champion_score"]
    cases = build_cases(scored, anomaly_threshold, model_threshold, args.cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"Wrote {len(cases)} grounded evaluation cases to {args.output}")


if __name__ == "__main__":
    main()

