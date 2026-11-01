from __future__ import annotations

import json

import joblib
import pandas as pd
import streamlit as st

from src.audit_store import ALLOWED_STATUSES, get_case, list_events, save_disposition
from src.config import PATHS
from src.explainability import local_sensitivity
from src.investigation import run_investigation
from src.llm import generate_narrative, get_provider
from src.rag import retrieve_for_flags
from src.risk_engine import assess_transaction


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
            ("model_metrics.json", "scored_transactions.csv", "model_comparison.csv")]
if not all(path.exists() for path in required):
    st.error("Official experiment artifacts are missing. Run the modeling pipeline first.")
    st.stop()

metrics = json.loads((PATHS.artifacts / "model_metrics.json").read_text(encoding="utf-8"))
df = pd.read_csv(PATHS.artifacts / "scored_transactions.csv")
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
    f"Official PaySim verified: {metrics['dataset']['rows']:,} transactions, "
    f"{metrics['dataset']['fraud']:,} fraud labels, SHA-256 {dataset_sha[:12]}…"
)

overview_tab, case_tab, governance_tab = st.tabs(
    ["Portfolio overview", "Case investigation", "Model & LLM governance"])

with overview_tab:
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
    queue_name = st.radio("Queue", ["Model alerts", "False negatives", "All retained test cases"],
                          horizontal=True)
    if queue_name == "Model alerts":
        visible = df[df["predicted_alert"] == 1]
    elif queue_name == "False negatives":
        visible = df[(df["predicted_alert"] == 0) & (df["isFraud"] == 1)]
    else:
        visible = df
    visible = visible.sort_values("champion_score", ascending=False)
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

    with st.expander("Bounded evidence collection"):
        st.caption(
            "The investigation workflow has a five-tool, read-only budget. PaySim does not contain "
            "customer history or network context, so unavailable evidence is shown rather than inferred."
        )
        agent_run = run_investigation(case, "bounded_agent")
        st.dataframe(pd.DataFrame([item.to_dict() for item in agent_run.evidence]),
                     use_container_width=True, hide_index=True)
        st.caption(
            f"Called {len(agent_run.tool_calls)} allow-listed tools; stopped because "
            f"{agent_run.stopped_reason}."
        )

    st.markdown("#### AI-assisted analyst brief")
    st.caption("The LLM can summarise supplied evidence but cannot change the deterministic label or disposition the case.")
    with st.expander("Generation settings"):
        version = st.selectbox("Prompt strategy", ["grounded_structured", "few_shot",
                                                    "role_constraints", "baseline"])
        provider_name = st.selectbox("Provider", ["mock", "openai"],
                                     format_func=lambda x: "Offline contract mock" if x == "mock" else "OpenAI-compatible API")
    if st.button("Generate analyst brief", type="primary"):
        try:
            raw, parsed = generate_narrative(case, version, get_provider(provider_name))
            if parsed is None:
                st.error("Output rejected by the strict schema.")
            else:
                st.success("Output passed strict schema validation.")
                st.write(parsed.summary)
                st.markdown("**Evidence**")
                for item in parsed.key_evidence:
                    st.markdown(f"- {item.claim}  \n  Sources: `{', '.join(item.source_paths)}`")
                st.markdown("**Uncertainties**")
                for item in parsed.uncertainties:
                    st.markdown(f"- {item}")
            with st.expander("Raw generated output"):
                st.code(raw, language="json")
        except Exception as exc:
            st.error(str(exc))

    st.markdown("#### Human disposition and audit trail")
    current = get_case(case_id)
    statuses = sorted(ALLOWED_STATUSES)
    status = st.selectbox("Case status", statuses,
                          index=statuses.index(current["status"]) if current["status"] in statuses else 0)
    notes = st.text_area("Analyst notes", value=current.get("analyst_notes", ""),
                         placeholder="Record evidence reviewed and the reason for the disposition.")
    if st.button("Save disposition"):
        save_disposition(case_id, status, notes, "demo_analyst", dataset_sha, model_version)
        st.success("Disposition appended to the audit trail.")
    events = list_events(case_id)
    if events:
        st.dataframe(pd.DataFrame(events), use_container_width=True, hide_index=True)
    report = {**case, "audit_events": events, "model_version": model_version,
              "dataset_sha256": dataset_sha}
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
    st.subheader("Investigation strategy evaluation")
    agent_summary_path = PATHS.artifacts / "agent_eval_summary.csv"
    if agent_summary_path.exists():
        st.caption("Synthetic workflow benchmark only; it is not a production AML performance result.")
        st.dataframe(pd.read_csv(agent_summary_path), use_container_width=True, hide_index=True)
    else:
        st.info("Run `python -m src.agent_evaluation` to generate the bounded-Agent comparison.")

