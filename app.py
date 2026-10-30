from __future__ import annotations

import json
import os
import time
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.audit_store import (ALLOWED_STATUSES, FEEDBACK_CATEGORIES, get_case,
                            list_events, save_disposition)
from src.config import PATHS
from src.explainability import local_sensitivity
from src.llm import generate_narrative, get_provider
from src.rag import retrieve_for_flags
from src.risk_engine import assess_transaction
from src.workspace import frozen_case_queue, narrative_control_errors


def mask_account(value: object) -> str:
    text = str(value)
    return f"{text[:1]}***{text[-4:]}" if len(text) > 5 else "***"


st.set_page_config(page_title="Financial Risk & Audit Analytics", layout="wide")
st.markdown("""
<style>
div[data-testid="stMetric"] {background:#f7f9fc;border:1px solid #e4e8ef;padding:14px;border-radius:10px}
.case-kicker {color:#667085;font-size:.85rem;text-transform:uppercase;letter-spacing:.08em}
.case-title {font-size:1.45rem;font-weight:650;margin:.1rem 0 1rem 0}
</style>
""", unsafe_allow_html=True)
st.title("GenAI-Enhanced Financial Risk & Audit Analytics")
st.caption("Human-in-the-loop fraud alert prioritisation on official PaySim. Scores support review; they do not prove fraud.")

required = [PATHS.artifacts / name for name in
            ("model_metrics.json", "model_comparison.csv")]
if not all(path.exists() for path in required):
    st.error("Saved experiment metrics are missing. Restore repository artifacts or run the modeling pipeline.")
    st.stop()

metrics = json.loads((PATHS.artifacts / "model_metrics.json").read_text(encoding="utf-8"))
official_queue = PATHS.artifacts / "scored_transactions.csv"
sources = ["Frozen-case walkthrough"]
if official_queue.exists():
    sources.insert(0, "Official scored queue")
source = st.sidebar.radio("Case source", sources)
demo_mode = source == "Frozen-case walkthrough"
if demo_mode:
    frozen_path = PATHS.artifacts / "eval_cases.json"
    if not frozen_path.exists():
        st.error("Frozen cases are missing. Restore artifacts/eval_cases.json.")
        st.stop()
    df = frozen_case_queue(json.loads(frozen_path.read_text(encoding="utf-8")))
    st.warning("Frozen-case walkthrough: outcome-stratified cases for product demonstration. "
               "This queue does not represent natural prevalence or the full test ranking.")
else:
    df = pd.read_csv(official_queue)
workspace_db = Path(os.environ.get("WORKSPACE_AUDIT_DB", str(
    PATHS.artifacts / ("workspace_demo.sqlite3" if demo_mode else "case_audit.sqlite3"))))
comparison = pd.read_csv(PATHS.artifacts / "model_comparison.csv")
importance_path = PATHS.artifacts / "feature_importance.csv"
importance = pd.read_csv(importance_path) if importance_path.exists() else pd.DataFrame()
champion = metrics["champion"]
champion_metrics = metrics["test"][champion]
model_threshold = metrics["validation"][champion]["threshold"]
anomaly_threshold = metrics["validation"]["isolation_forest"]["threshold"]
dataset_sha = metrics["dataset"]["sha256"]
model_version = f"{champion}:{dataset_sha[:12]}"


@st.cache_resource
def load_model_bundle():
    return joblib.load(PATHS.artifacts / "model_bundle.joblib")

st.success(
    f"Saved official experiment: {metrics['dataset']['rows']:,} transactions, "
    f"{metrics['dataset']['fraud']:,} fraud labels, SHA-256 {dataset_sha[:12]}…"
)

overview_tab, case_tab, governance_tab = st.tabs(
    ["Portfolio overview", "Case investigation", "Model & LLM governance"])

with overview_tab:
    st.caption("These are saved full-test experiment results, not metrics computed from the selected walkthrough queue.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test transactions", f"{champion_metrics['support']:,}")
    c2.metric("Test fraud prevalence", f"{champion_metrics['fraud_count']/champion_metrics['support']:.3%}")
    c3.metric("Champion PR-AUC", f"{champion_metrics['average_precision']:.4f}")
    c4.metric("Recall at operating point", f"{champion_metrics['recall']:.2%}")
    st.caption(
        f"The operating point was fixed on validation for a {metrics['target_alert_rate']:.2%} review capacity. "
        f"Temporal shift produced a {champion_metrics['alert_rate']:.2%} test alert rate."
    )

    st.subheader("Model comparison on the untouched test period")
    test_comparison = comparison[comparison["split"] == "test"].copy()
    display_cols = ["model", "average_precision", "roc_auc", "precision", "recall",
                    "f1", "f2", "alert_rate", "tp", "fp", "fn"]
    st.dataframe(test_comparison[display_cols], use_container_width=True, hide_index=True,
                 column_config={key: st.column_config.NumberColumn(format="%.4f")
                                for key in ["average_precision", "roc_auc", "precision",
                                            "recall", "f1", "f2", "alert_rate"]})

    st.subheader("Review-capacity curve")
    top_k = pd.DataFrame([
        {"Review share": key.replace("top_", ""), **value}
        for key, value in champion_metrics["top_k"].items()
    ])
    st.dataframe(top_k, use_container_width=True, hide_index=True)
    st.warning(
        "PaySim contains a simulator-specific balance-depletion signature. The full HGB result is therefore "
        "an upper-bound benchmark, not evidence of expected performance on real bank traffic. See the ablation model below."
    )

with case_tab:
    st.subheader("Alert queue")
    evaluation_mode = st.toggle("Evaluation mode (reveals offline labels)", value=False)
    queues = ["Model alerts", "All retained test cases"]
    if evaluation_mode:
        queues.insert(1, "False negatives")
    queue_name = st.radio("Queue", queues,
                          horizontal=True)
    if queue_name == "Model alerts":
        visible = df[df["predicted_alert"] == 1]
    elif queue_name == "False negatives":
        visible = df[(df["predicted_alert"] == 0) & (df["isFraud"] == 1)]
    else:
        visible = df
    visible = visible.sort_values("champion_score", ascending=False, kind="stable")
    table = visible[["step", "type", "amount", "nameOrig", "nameDest",
                     "champion_score", "anomaly_score", "predicted_alert"]].head(500).copy()
    table["nameOrig"] = table["nameOrig"].map(mask_account)
    table["nameDest"] = table["nameDest"].map(mask_account)
    if evaluation_mode:
        table["offline_label"] = visible.loc[table.index, "isFraud"]
    st.dataframe(table, use_container_width=True)
    st.caption("Account identifiers are masked. Offline labels remain hidden unless Evaluation mode is enabled.")
    if visible.empty:
        st.info("This queue has no retained cases.")
        st.stop()

    selected = st.selectbox(
        "Alert to investigate", visible.index.tolist(),
        format_func=lambda i: f"#{i} {visible.loc[i, 'type']} — {visible.loc[i, 'amount']:,.2f}")
    row = visible.loc[selected]
    assessment = assess_transaction(
        row.to_dict(), float(row["champion_score"]), float(row["anomaly_score"]),
        anomaly_threshold, model_threshold)
    case_id = f"CASE-{int(selected):06d}"
    st.divider()
    st.markdown(
        f'<div class="case-kicker">{case_id}</div>'
        f'<div class="case-title">{row["type"]} from {mask_account(row["nameOrig"])} '
        f'to {mask_account(row["nameDest"])}</div>', unsafe_allow_html=True)
    if assessment.risk_label == "CRITICAL":
        st.error("Critical priority — recommend an authorised temporary hold and immediate human escalation.")
    elif assessment.risk_label == "HIGH":
        st.warning("High priority — send to priority investigation.")
    elif assessment.risk_label == "MEDIUM":
        st.warning("Medium priority — queue for standard review.")
    else:
        st.success("Low priority — allow with monitoring.")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Risk level", assessment.risk_label)
    r2.metric("Priority score", f"{assessment.risk_score} / 100")
    r3.metric("Calibrated model score", f"{row['champion_score']:.4%}")
    r4.metric("Anomaly signal", "Above threshold" if row["anomaly_score"] >= anomaly_threshold else "Below threshold")

    st.markdown("#### Transaction facts")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Amount", f"{row['amount']:,.2f}")
    t2.metric("Type", row["type"])
    t3.metric("Origin balance", f"{row['oldbalanceOrg']:,.2f}")
    t4.metric("Simulation hour", f"{int(row['step'])}")

    st.markdown("#### Why the case was prioritised")
    flags = pd.DataFrame([{
        "Control": flag.code, "Reason": flag.description,
        "Evidence": json.dumps(flag.evidence), "Priority floor": flag.score_floor,
        "Policy": flag.policy_id,
    } for flag in assessment.red_flags])
    if len(flags):
        st.dataframe(flags, use_container_width=True, hide_index=True)
    else:
        st.info("No rule flags.")

    with st.expander("Model sensitivity explanation"):
        st.caption(
            "Each feature is replaced with its training reference value while other engineered features stay fixed. "
            "This is a local what-if sensitivity check, not a causal explanation."
        )
        try:
            sensitivity = local_sensitivity(load_model_bundle(), row.to_dict()).head(8)
            st.dataframe(sensitivity, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.warning(f"Sensitivity explanation unavailable: {exc}")

    case = {
        "case_id": case_id,
        "transaction": {key: (value.item() if hasattr(value, "item") else value)
                        for key, value in row.items()
                        if key not in {"isFraud", "isFlaggedFraud", "predicted_alert"}
                        and not key.endswith("_score")},
        "assessment": assessment.to_dict(),
        "retrieved_policies": retrieve_for_flags([flag.policy_id for flag in assessment.red_flags]),
    }
    with st.expander("Applicable policy evidence"):
        for policy in case["retrieved_policies"]:
            st.markdown(f"**{policy['id']} — {policy['title']}**")
            st.write(policy["text"])

    st.markdown("#### AI-assisted analyst brief")
    st.caption("The LLM can summarise supplied evidence but cannot change the deterministic label or disposition the case.")
    with st.expander("Generation settings"):
        version = st.selectbox("Prompt strategy", ["grounded_structured", "few_shot",
                                                    "role_constraints", "baseline"])
        provider_name = st.selectbox("Provider", ["mock", "openai"],
                                     format_func=lambda x: "Offline contract mock" if x == "mock" else "OpenAI-compatible API")
    if st.button("Generate analyst brief", type="primary"):
        try:
            provider = get_provider(provider_name)
            started = time.perf_counter()
            raw, parsed = generate_narrative(case, version, provider)
            errors = narrative_control_errors(parsed, case) if parsed else ["Strict schema validation failed"]
            generation = {"provider": provider_name, "prompt_version": version,
                          "is_mock": provider_name == "mock", "raw": raw,
                          "parsed": parsed.model_dump() if parsed else None,
                          "control_errors": errors,
                          "latency_ms": (time.perf_counter() - started) * 1000,
                          "usage": getattr(provider, "last_usage", None)}
            st.session_state[f"generation:{source}:{case_id}"] = generation
            if not errors:
                st.session_state[f"summary:{source}:{case_id}"] = parsed.summary
        except Exception as exc:
            # Invalidate any earlier draft after a failed new request.
            st.session_state.pop(f"generation:{source}:{case_id}", None)
            message = str(exc) if isinstance(exc, RuntimeError) else type(exc).__name__
            st.error(f"Generation failed: {message}. Check provider configuration and try again.")

    generation = st.session_state.get(f"generation:{source}:{case_id}")
    if generation:
        st.caption(f"{'Offline mock' if generation['is_mock'] else 'Real provider'} · "
                   f"{generation['prompt_version']} · {generation['latency_ms']:.0f} ms")
        if generation["control_errors"]:
            st.error("Draft rejected: " + "; ".join(generation["control_errors"]))
        else:
            parsed_brief = generation["parsed"]
            st.success("Schema and deterministic controls passed. Check every claim before use.")
            st.write(parsed_brief["summary"])
            st.markdown("**Evidence**")
            for item in parsed_brief["key_evidence"]:
                st.write(item["claim"])
                st.caption("Sources: " + ", ".join(item["source_paths"]))
            st.markdown("**Uncertainties**")
            for item in parsed_brief["uncertainties"]:
                st.write(item)
        with st.expander("Original generated output and run details"):
            st.json(generation)

    st.markdown("#### Human disposition and audit trail")
    current = get_case(case_id, path=workspace_db)
    statuses = sorted(ALLOWED_STATUSES)
    status = st.selectbox("Case status", statuses,
                          index=statuses.index(current["status"]) if current["status"] in statuses else 0,
                          key=f"status:{source}:{case_id}")
    reviewer = st.text_input("Reviewer name", value="demo_analyst", key=f"reviewer:{source}:{case_id}")
    summary_key = f"summary:{source}:{case_id}"
    if summary_key not in st.session_state:
        st.session_state[summary_key] = current.get("reviewed_summary", "")
    reviewed_summary = st.text_area("Reviewed summary", key=summary_key,
                                   placeholder="Edit the AI draft or write an evidence-based summary manually.")
    notes = st.text_area("Analyst notes", value=current.get("analyst_notes", ""),
                         key=f"notes:{source}:{case_id}",
                         placeholder="Record evidence reviewed and the reason for the disposition.")
    feedback = st.selectbox("Draft feedback", sorted(FEEDBACK_CATEGORIES),
                            index=sorted(FEEDBACK_CATEGORIES).index("none"),
                            key=f"feedback:{source}:{case_id}")
    revision_key = f"revision:{source}:{case_id}"
    if revision_key not in st.session_state:
        st.session_state[revision_key] = current["updated_at"]
    if st.button("Save disposition"):
        try:
            save_disposition(case_id, status, notes, reviewer, dataset_sha, model_version,
                             path=workspace_db, reviewed_summary=reviewed_summary,
                             generation=generation, feedback=feedback,
                             expected_updated_at=st.session_state[revision_key])
            st.session_state[revision_key] = get_case(case_id, path=workspace_db)["updated_at"]
            st.success("Review, draft provenance and feedback appended to the audit trail.")
        except ValueError as exc:
            st.error(str(exc))
    if st.button("Reload saved review"):
        for prefix in ("revision", "status", "notes", "summary", "feedback", "generation"):
            st.session_state.pop(f"{prefix}:{source}:{case_id}", None)
        st.rerun()
    events = list_events(case_id, path=workspace_db)
    if events:
        st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)
    report = {**case, "audit_events": events, "model_version": model_version,
              "dataset_sha256": dataset_sha, "case_source": source}
    st.download_button("Download case report", json.dumps(report, indent=2),
                       file_name=f"{case_id}.json", mime="application/json")
    with st.expander("Technical and evaluation-only details"):
        st.json(assessment.to_dict())
        if evaluation_mode:
            st.write(f"Offline dataset label: {int(row['isFraud'])}")

with governance_tab:
    st.subheader("Dataset and split governance")
    st.json({"dataset": metrics["dataset"], "splits": metrics["splits"],
             "methodology": metrics["methodology"]}, expanded=False)
    st.subheader("Global permutation importance")
    if not importance.empty:
        st.bar_chart(importance.set_index("feature")["importance_mean"])
        st.dataframe(importance, use_container_width=True, hide_index=True)
    st.info(
        "The balance-signature ablation removes origin-balance features associated with PaySim's injected fraud pattern. "
        "Its lower performance is a deliberate realism check, not a failed experiment."
    )
    st.subheader("LLM evaluation status")
    manifest_path = PATHS.artifacts / "llm_eval_manifest.json"
    summary_path = PATHS.artifacts / "llm_eval_summary.csv"
    if manifest_path.exists() and summary_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        st.json(manifest)
        st.dataframe(pd.read_csv(summary_path), use_container_width=True, hide_index=True)
    else:
        st.warning("LLM evaluation has not been run for the current cases.")

