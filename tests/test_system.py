import json
from pathlib import Path

import pandas as pd
import pytest

from src.audit_store import get_case, list_events, save_disposition
from src.data import clean_transactions, generate_demo_data
from src.evaluation import evaluate_response
from src.agent_evaluation import evaluate_strategies
from src.investigation import attach_investigation, run_investigation
from src.tool_planner import LLMToolPlanner
from src.features import MODEL_FEATURES, build_features
from src.llm import MockLLM, RiskNarrative
from src.prompts import build_prompt
from src.rag import retrieve_for_flags
from src.risk_engine import assess_transaction


def _case():
    row = {"step": 3, "type": "TRANSFER", "amount": 250000,
           "oldbalanceOrg": 260000, "nameDest": "C123"}
    assessment = assess_transaction(row, .91, .2, .1, .2)
    policies = retrieve_for_flags([flag.policy_id for flag in assessment.red_flags])
    return {
        "case_id": "CASE-1", "transaction": row,
        "assessment": assessment.to_dict(), "retrieved_policies": policies,
        "gold": {"case_id": "CASE-1", "risk_label": assessment.risk_label,
                 "recommended_action": assessment.recommended_action,
                 "red_flag_codes": [flag.code for flag in assessment.red_flags],
                 "policy_ids": sorted({flag.policy_id for flag in assessment.red_flags})},
    }


def test_demo_schema_and_cleaning():
    df = generate_demo_data(500, seed=1)
    assert len(clean_transactions(df)) == 500
    assert set(df["isFraud"].unique()) <= {0, 1}


def test_invalid_labels_are_rejected():
    df = generate_demo_data(100, seed=1)
    df.loc[0, "isFraud"] = 2
    with pytest.raises(ValueError, match="outside 0/1"):
        clean_transactions(df)


def test_features_exclude_post_transaction_fields():
    features = build_features(generate_demo_data(50, seed=2))
    assert list(features.columns) == MODEL_FEATURES
    assert "newbalanceOrig" not in features and "isFraud" not in features


def test_risk_engine_uses_floors_not_additive_double_counting():
    assessment = _case()["assessment"]
    floors = [flag["score_floor"] for flag in assessment["red_flags"]]
    assert assessment["risk_score"] >= max(floors)
    assert assessment["risk_score"] <= 100
    assert assessment["recommended_action"] == "recommend_temporary_hold_and_escalation"


def test_mock_llm_round_trip_scores():
    case = _case()
    raw = MockLLM().generate(build_prompt("grounded_structured", case), case)
    RiskNarrative.model_validate_json(raw)
    score = evaluate_response(raw, case)
    assert score["schema_valid"] == 1
    assert score["risk_label_agreement"] == 1
    assert score["citation_precision"] == 1
    assert score["source_path_validity"] == 1


def test_hallucinated_number_is_penalized():
    case = _case()
    obj = json.loads(MockLLM().generate("", case))
    obj["summary"] += " Unsupported loss was 987654321."
    score = evaluate_response(json.dumps(obj), case)
    assert score["factual_consistency"] < 1


def test_schema_rejects_unapproved_action():
    case = _case()
    obj = json.loads(MockLLM().generate("", case))
    obj["recommended_actions"] = ["freeze_customer_forever"]
    assert evaluate_response(json.dumps(obj), case)["schema_valid"] == 0


def test_audit_store_appends_events(tmp_path):
    db = tmp_path / "audit.sqlite3"
    save_disposition("CASE-1", "REVIEWING", "Checked transaction", "tester",
                     "abc", "model-v1", db)
    assert get_case("CASE-1", db)["status"] == "REVIEWING"
    events = list_events("CASE-1", db)
    assert len(events) == 1
    assert events[0]["payload"]["notes"] == "Checked transaction"


def test_bounded_agent_collects_required_evidence_with_fewer_tools_for_low_case():
    cases = json.loads((Path(__file__).parents[1] / "data" / "investigation_cases.json").read_text())
    critical, low, high = cases
    critical_run = run_investigation(critical, "bounded_agent")
    low_fixed = run_investigation(low, "fixed_workflow")
    low_agent = run_investigation(low, "bounded_agent")
    assert set(critical["investigation_gold"]["required_tools"]) == {
        item.tool_name for item in critical_run.evidence if item.status == "available"
    }
    assert len(low_agent.tool_calls) < len(low_fixed.tool_calls)
    assert low_agent.tool_calls == ["get_case_context", "get_account_history", "get_policy_requirements"]
    assert set(high["investigation_gold"]["required_tools"]) == {
        item.tool_name for item in run_investigation(high, "bounded_agent").evidence if item.status == "available"
    }


def test_agent_evaluation_and_narrative_use_only_collected_evidence():
    cases = json.loads((Path(__file__).parents[1] / "data" / "investigation_cases.json").read_text())
    details, summary = evaluate_strategies(cases)
    assert (details["unsafe_close"] == 0).all()
    assert (summary["task_success_rate"] == 1).all()
    case = attach_investigation(cases[0], run_investigation(cases[0]))
    raw = MockLLM().generate(build_prompt("grounded_structured", case), case)
    assert evaluate_response(raw, case)["source_path_validity"] == 1


def test_live_planner_rejects_an_unapproved_tool_before_execution():
    class BadPlanner:
        def generate(self, prompt, case):
            return '{"next_tool":"run_sql","reason":"query everything"}'

    cases = json.loads((Path(__file__).parents[1] / "data" / "investigation_cases.json").read_text())
    with pytest.raises(ValueError, match="non-allow-listed"):
        run_investigation(cases[0], planner=LLMToolPlanner(BadPlanner()))

