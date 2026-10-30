"""Small, testable helpers for the analyst workspace; no model training needed."""
from __future__ import annotations

from copy import deepcopy

import pandas as pd

from .evaluation import evaluate_response
from .llm import RiskNarrative


def evidence_case(case: dict) -> dict:
    """Exclude offline labels and gold controls at the provider boundary."""
    return deepcopy({key: case[key] for key in
                     ("case_id", "transaction", "assessment", "retrieved_policies")})


def frozen_case_queue(cases: list[dict]) -> pd.DataFrame:
    rows = []
    for case in cases:
        signals = case["assessment"]["model_signals"]
        rows.append({**case["transaction"],
                     "_case_index": int(case["case_id"].split("-")[-1]),
                     "champion_score": signals["calibrated_model_score"],
                     "anomaly_score": signals["anomaly_score"],
                     "predicted_alert": int(signals["calibrated_model_score"] >=
                                            signals["model_operating_threshold"]),
                     "isFraud": case["metadata"]["dataset_label"]})
    return pd.DataFrame(rows).set_index("_case_index")


def narrative_control_errors(parsed: RiskNarrative, case: dict) -> list[str]:
    """Validate deterministic controls, not semantic truth or causal correctness."""
    assessment = case["assessment"]
    scored_case = {**evidence_case(case), "gold": {
        "case_id": case["case_id"], "risk_label": assessment["risk_label"],
        "recommended_action": assessment["recommended_action"],
        "red_flag_codes": [f["code"] for f in assessment["red_flags"]],
        "policy_ids": [p["id"] for p in case["retrieved_policies"]],
    }}
    scores = evaluate_response(parsed.model_dump_json(), scored_case)
    checks = {"case_id_agreement": "Case ID does not match",
              "risk_label_agreement": "Risk label does not match the engine",
              "action_agreement": "Required action is missing",
              "source_path_validity": "An evidence source path does not exist",
              "citation_precision": "A policy citation was not supplied"}
    errors = [message for metric, message in checks.items() if scores[metric] != 1]
    if set(parsed.recommended_actions) != {assessment["recommended_action"]}:
        errors.append("Actions differ from the deterministic recommendation")
    # Evaluation-only keys must never become valid evidence citations.
    allowed_roots = {"transaction", "assessment", "retrieved_policies"}
    if any(path.split(".")[0] not in allowed_roots for item in parsed.key_evidence
           for path in item.source_paths):
        errors.append("Evidence must cite transaction, assessment or policy data")
    return errors
