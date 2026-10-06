from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RedFlag:
    code: str
    description: str
    score_floor: int
    evidence: dict
    policy_id: str


@dataclass(frozen=True)
class RiskAssessment:
    risk_score: int
    risk_label: str
    recommended_action: str
    red_flags: list[RedFlag]
    model_signals: dict

    def to_dict(self) -> dict:
        out = asdict(self)
        out["red_flags"] = [asdict(x) for x in self.red_flags]
        return out


def _label(score: int) -> str:
    if score >= 75:
        return "CRITICAL"
    if score >= 50:
        return "HIGH"
    if score >= 25:
        return "MEDIUM"
    return "LOW"


def assess_transaction(row: dict, fraud_probability: float,
                       anomaly_score: float, anomaly_threshold: float,
                       model_threshold: float = .5) -> RiskAssessment:
    flags: list[RedFlag] = []
    amount = float(row["amount"])
    tx_type = str(row["type"])
    old_org = float(row.get("oldbalanceOrg", 0))
    hour = (int(row.get("step", 1)) - 1) % 24
    if amount >= 200_000:
        flags.append(RedFlag("RF_HIGH_VALUE", "Transaction meets high-value review threshold", 35,
                             {"amount": amount, "threshold": 200_000}, "POL-001"))
    if tx_type in {"TRANSFER", "CASH_OUT"} and amount >= .75 * max(old_org, 1):
        flags.append(RedFlag("RF_BALANCE_DRAIN", "Transaction consumes at least 75% of origin balance", 45,
                             {"amount": amount, "oldbalanceOrg": old_org}, "POL-002"))
    if tx_type in {"TRANSFER", "CASH_OUT"} and hour in {0, 1, 2, 3, 4, 5}:
        flags.append(RedFlag("RF_OFF_HOURS", "Risk-prone transfer type occurred off-hours", 20,
                             {"hour": hour, "type": tx_type}, "POL-003"))
    if fraud_probability >= model_threshold:
        flags.append(RedFlag("RF_MODEL_ALERT", "Calibrated model score exceeds the review operating point", 70,
                             {"model_score": round(fraud_probability, 8),
                              "operating_threshold": round(model_threshold, 8)}, "POL-004"))
    if anomaly_score >= anomaly_threshold:
        flags.append(RedFlag("RF_ANOMALY", "Isolation Forest score exceeds review threshold", 50,
                             {"anomaly_score": round(anomaly_score, 6),
                              "threshold": round(anomaly_threshold, 6)}, "POL-005"))
    # The score is a prioritisation index, not a probability. Rules set floors
    # instead of adding to the model, avoiding repeated counting of correlated evidence.
    relative = fraud_probability / max(model_threshold, 1e-12)
    if relative >= 1:
        model_component = min(100, round(70 + 10 * math.log2(relative)))
    else:
        model_component = round(70 * math.sqrt(max(0.0, relative)))
    score = max([model_component, *[flag.score_floor for flag in flags]])
    label = _label(score)
    action = {"LOW": "allow_with_monitoring", "MEDIUM": "queue_for_review",
              "HIGH": "priority_investigation",
              "CRITICAL": "recommend_temporary_hold_and_escalation"}[label]
    return RiskAssessment(score, label, action, flags, {
        "calibrated_model_score": round(fraud_probability, 8),
        "model_operating_threshold": round(model_threshold, 8),
        "relative_to_threshold": round(relative, 4),
        "anomaly_score": round(anomaly_score, 6),
        "anomaly_threshold": round(anomaly_threshold, 6),
    })

