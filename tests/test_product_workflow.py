import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.audit_store import ConcurrentReviewError, get_case, list_events, save_disposition
from src.evaluation import estimate_cost, run_evaluation
from src.llm import MockLLM, RiskNarrative, generate_narrative
from src.workspace import evidence_case, frozen_case_queue, narrative_control_errors
from test_system import _case


def test_review_preserves_original_and_human_edit(tmp_path):
    db = tmp_path / "review.sqlite3"
    draft = {"provider": "mock", "parsed": {"summary": "Original AI draft"}}
    save_disposition("CASE-1", "REVIEWING", "Checked source", "reviewer", "hash", "v1", db,
                     generation=draft, reviewed_summary="Human edited summary", feedback="omission",
                     expected_updated_at="")
    current = get_case("CASE-1", db)
    save_disposition("CASE-1", "ESCALATED", "Needs further evidence", "reviewer", "hash", "v1", db,
                     reviewed_summary=current["reviewed_summary"],
                     expected_updated_at=current["updated_at"])
    events = list_events("CASE-1", db)
    assert len(events) == 2
    assert events[-1]["payload"]["generation"] == draft
    assert events[-1]["payload"]["reviewed_summary"] == "Human edited summary"
    assert events[-1]["payload"]["feedback"] == "omission"
    assert events[0]["payload"]["previous_status"] == "REVIEWING"


def test_stale_review_is_rejected_without_appending(tmp_path):
    db = tmp_path / "audit.sqlite3"
    save_disposition("CASE-1", "REVIEWING", "First", "A", "hash", "v1", db,
                     expected_updated_at="")
    with pytest.raises(ConcurrentReviewError):
        save_disposition("CASE-1", "REVIEWING", "Stale", "B", "hash", "v1", db,
                         expected_updated_at="")
    assert len(list_events("CASE-1", db)) == 1
    assert get_case("CASE-1", db)["analyst_notes"] == "First"


def test_legacy_database_migrates_without_losing_notes(tmp_path):
    db = tmp_path / "old.sqlite3"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE cases(case_id TEXT PRIMARY KEY, status TEXT, "
                    "analyst_notes TEXT, updated_at TEXT)")
        con.execute("INSERT INTO cases VALUES ('CASE-1','REVIEWING','Legacy notes','old-time')")
    assert get_case("CASE-1", db)["analyst_notes"] == "Legacy notes"
    assert get_case("CASE-1", db)["reviewed_summary"] == ""


@pytest.mark.parametrize("status", ["ESCALATED", "CLOSED_NO_FRAUD", "CLOSED_CONFIRMED"])
def test_final_disposition_requires_rationale(tmp_path, status):
    with pytest.raises(ValueError, match="rationale"):
        save_disposition("CASE-1", status, " ", "A", "hash", "v1", tmp_path / "audit.db")


def test_frozen_queue_preserves_case_identity_and_score():
    cases = json.loads(Path("artifacts/eval_cases.json").read_text())
    queue = frozen_case_queue(cases)
    assert len(queue) == 60
    first = cases[0]
    index = int(first["case_id"].split("-")[-1])
    assert queue.loc[index, "champion_score"] == first["assessment"]["model_signals"]["calibrated_model_score"]
    assert "gold" not in queue and "evaluation_outcome" not in queue


def test_provider_never_receives_gold_or_metadata():
    case = {**_case(), "metadata": {"dataset_label": 1}}

    class InspectingProvider(MockLLM):
        def generate(self, prompt, supplied):
            assert "gold" not in supplied and "metadata" not in supplied
            assert '"dataset_label"' not in prompt
            return super().generate(prompt, supplied)

    _, parsed = generate_narrative(case, provider=InspectingProvider())
    assert parsed is not None
    assert set(evidence_case(case)) == {"case_id", "transaction", "assessment", "retrieved_policies"}


def test_workspace_rejects_valid_schema_with_wrong_controls():
    case = _case()
    parsed = RiskNarrative.model_validate_json(MockLLM().generate("", case))
    assert narrative_control_errors(parsed, case) == []
    parsed.case_id = "OTHER-CASE"
    parsed.key_evidence[0].source_paths = ["gold.risk_label"]
    parsed.policy_citations.append("MADE-UP")
    errors = narrative_control_errors(parsed, case)
    assert any("Case ID" in x for x in errors)
    assert any("policy" in x for x in errors)
    assert any("Evidence must" in x for x in errors)


def test_evaluation_logs_usage_and_errors_without_gold(monkeypatch):
    class Provider(MockLLM):
        def generate(self, prompt, supplied):
            assert "gold" not in supplied
            self.last_usage = {"model": "fake-model", "input_tokens": 100, "output_tokens": 20}
            return super().generate(prompt, supplied)

    monkeypatch.setattr("src.evaluation.get_provider", lambda _: Provider())
    scores, outputs = run_evaluation([_case()], "test", prompt_versions=("grounded_structured",),
                                     input_price=1, output_price=2)
    assert scores.iloc[0]["estimated_cost_usd"] == pytest.approx(.00014)
    assert outputs[0]["usage"]["model"] == "fake-model"

    class FailingProvider(Provider):
        def generate(self, prompt, supplied):
            raise RuntimeError("Sensitive endpoint details")

    monkeypatch.setattr("src.evaluation.get_provider", lambda _: FailingProvider())
    failed, _ = run_evaluation([_case()], "test", prompt_versions=("grounded_structured",))
    assert not failed.iloc[0]["request_succeeded"]
    assert failed.iloc[0]["error"] == "RuntimeError"
    assert failed.iloc[0]["schema_valid"] == 0
    assert estimate_cost({}, 1, 2) is None
    with pytest.raises(ValueError):
        estimate_cost({}, float("nan"), 1)


def test_real_adapter_records_usage_with_fake_client(monkeypatch):
    import sys
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-not-a-real-key")
    response = SimpleNamespace(model="test-version", usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
                               choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))])
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response)))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **_: fake))
    from src.llm import OpenAICompatibleLLM
    provider = OpenAICompatibleLLM()
    assert provider.generate("prompt", {}) == "{}"
    assert provider.last_usage == {"model": "test-version", "input_tokens": 10, "output_tokens": 5}


def test_failed_live_attempt_is_not_a_completed_comparison(tmp_path, monkeypatch):
    import sys
    from src.evaluation import main

    class FailingProvider(MockLLM):
        def generate(self, prompt, supplied):
            raise ConnectionError("Unreachable test endpoint")

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([_case()]))
    output = tmp_path / "run"
    monkeypatch.setattr("src.evaluation.get_provider", lambda _: FailingProvider())
    monkeypatch.setattr(sys, "argv", ["evaluation", "--provider", "openai", "--cases", str(cases),
                                     "--output-dir", str(output)])
    main()
    manifest = json.loads((output / "llm_eval_manifest.json").read_text())
    assert manifest["is_real_provider_run"] is True
    assert manifest["is_real_llm_comparison"] is False
    assert manifest["requests_attempted"] == 4 and manifest["requests_succeeded"] == 0
    assert manifest["estimated_total_cost_usd"] is None


def test_streamlit_walkthrough_generate_edit_save_and_reload(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    db = tmp_path / "workspace.sqlite3"
    monkeypatch.setenv("WORKSPACE_AUDIT_DB", str(db))
    app = AppTest.from_file("app.py", default_timeout=20).run()
    assert not app.exception
    assert "False negatives" not in next(x for x in app.radio if x.label == "Queue").options
    next(x for x in app.button if x.label == "Generate analyst brief").click().run()
    assert not app.exception
    next(x for x in app.text_area if x.label == "Reviewed summary").set_value("Analyst edited draft").run()
    next(x for x in app.text_area if x.label == "Analyst notes").set_value("Reviewed the supplied evidence").run()
    next(x for x in app.selectbox if x.label == "Draft feedback").select("omission").run()
    next(x for x in app.button if x.label == "Save disposition").click().run()
    assert not app.exception
    with sqlite3.connect(db) as con:
        row = con.execute("SELECT case_id, reviewed_summary FROM cases").fetchone()
    assert row[1] == "Analyst edited draft"
    event = list_events(row[0], db)[0]["payload"]
    assert event["generation"]["is_mock"] is True
    assert event["feedback"] == "omission"
    next(x for x in app.button if x.label == "Reload saved review").click().run()
    assert not app.exception
    assert next(x for x in app.text_area if x.label == "Reviewed summary").value == "Analyst edited draft"
