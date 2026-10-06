from __future__ import annotations

import json

PROMPT_VERSIONS = ("baseline", "role_constraints", "few_shot", "grounded_structured")

SCHEMA = {
    "case_id": "string", "risk_label": "LOW|MEDIUM|HIGH|CRITICAL",
    "summary": "string",
    "key_evidence": [{"claim": "string", "source_paths": ["dot.path.to.case.evidence"]}],
    "recommended_actions": ["approved action string"], "policy_citations": ["POL-###"],
    "uncertainties": ["string"],
}


def build_prompt(version: str, case: dict) -> str:
    if version not in PROMPT_VERSIONS:
        raise ValueError(f"Unknown prompt version: {version}")
    prompt_case = {key: case[key] for key in
                   ("case_id", "transaction", "assessment", "retrieved_policies")}
    evidence = json.dumps(prompt_case, indent=2, sort_keys=True)
    base = f"Explain the transaction risk.\nCASE:\n{evidence}"
    if version == "baseline":
        return f"[PROMPT_VERSION=baseline]\n{base}"
    role = (
        "[PROMPT_VERSION=role_constraints]\nYou are a bank fraud-triage analyst. "
        "Use only CASE evidence. Do not invent facts, policies, or certainty. "
        "Return a concise assessment and actions.\n" + base
    )
    if version == "role_constraints":
        return role
    example = {
        "case_id": "EXAMPLE-1", "risk_label": "HIGH",
        "summary": "High-value transfer was flagged by the model.",
        "key_evidence": [
            {"claim": "The amount is 250000.", "source_paths": ["transaction.amount"]},
            {"claim": "The calibrated model score is 0.91.",
             "source_paths": ["assessment.model_signals.calibrated_model_score"]},
        ],
        "recommended_actions": ["priority_investigation"],
        "policy_citations": ["POL-001", "POL-004"],
        "uncertainties": ["No device or identity evidence was supplied."],
    }
    if version == "few_shot":
        return ("[PROMPT_VERSION=few_shot]\n" + role.split("\n", 1)[1] +
                "\nEXAMPLE OUTPUT:\n" + json.dumps(example) +
                "\nReturn the current case in the same JSON shape.")
    return (
        "[PROMPT_VERSION=grounded_structured]\nYou are a bank fraud-triage analyst. "
        "The deterministic risk engine is authoritative for the label and action. "
        "Use only CASE and RETRIEVED_POLICIES. Every policy citation must match a supplied ID. "
        "Never claim fraud is proven. If evidence is absent, state the uncertainty. "
        "Each evidence claim must cite one or more exact dot-paths from CASE. "
        "Return JSON only, exactly matching this schema:\n" + json.dumps(SCHEMA) +
        "\nCASE:\n" + evidence +
        "\nBefore answering, silently verify every number and citation against the evidence."
    )

