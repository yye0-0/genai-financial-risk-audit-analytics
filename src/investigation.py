"""Bounded, evidence-first investigation workflows.

The module intentionally separates control flow from model prose.  Tools are
read-only projections of case data; an LLM may choose from their names through
an adapter in a later deployment, but it never receives database credentials or
arbitrary SQL/Cypher execution rights.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable, Literal

ToolName = Literal[
    "get_case_context", "get_account_history", "get_counterparty_profile",
    "get_network_snapshot", "get_policy_requirements",
]

TOOL_NAMES: tuple[ToolName, ...] = (
    "get_case_context", "get_account_history", "get_counterparty_profile",
    "get_network_snapshot", "get_policy_requirements",
)


@dataclass(frozen=True)
class ToolEvidence:
    evidence_id: str
    tool_name: ToolName
    status: Literal["available", "unavailable"]
    source_paths: list[str]
    payload: dict

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class InvestigationRun:
    strategy: Literal["fixed_workflow", "bounded_agent"]
    evidence: list[ToolEvidence]
    tool_calls: list[ToolName]
    stopped_reason: str
    latency_ms: float

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "evidence": [item.to_dict() for item in self.evidence],
            "tool_calls": self.tool_calls,
            "stopped_reason": self.stopped_reason,
            "latency_ms": self.latency_ms,
        }


def _context(case: dict) -> dict:
    """Optional normalized enrichment supplied by an approved data adapter."""
    return case.get("investigation_context", {})


def _available(tool_name: ToolName, case: dict, source_key: str, source_paths: list[str]) -> ToolEvidence:
    data = _context(case).get(source_key)
    if data is None:
        return ToolEvidence(
            evidence_id=f"{case['case_id']}:{tool_name}:unavailable",
            tool_name=tool_name,
            status="unavailable",
            source_paths=[],
            payload={"reason": f"{source_key} was not supplied for this case"},
        )
    return ToolEvidence(
        evidence_id=f"{case['case_id']}:{tool_name}", tool_name=tool_name,
        status="available", source_paths=source_paths, payload=data,
    )


def get_case_context(case: dict) -> ToolEvidence:
    return ToolEvidence(
        evidence_id=f"{case['case_id']}:get_case_context", tool_name="get_case_context",
        status="available",
        source_paths=["transaction", "assessment", "retrieved_policies"],
        payload={
            "transaction": case["transaction"],
            "assessment": case["assessment"],
            "policy_ids": [item["id"] for item in case.get("retrieved_policies", [])],
        },
    )


def get_account_history(case: dict) -> ToolEvidence:
    return _available("get_account_history", case, "account_history", ["investigation_context.account_history"])


def get_counterparty_profile(case: dict) -> ToolEvidence:
    return _available("get_counterparty_profile", case, "counterparty_profile", ["investigation_context.counterparty_profile"])


def get_network_snapshot(case: dict) -> ToolEvidence:
    return _available("get_network_snapshot", case, "network_snapshot", ["investigation_context.network_snapshot"])


def get_policy_requirements(case: dict) -> ToolEvidence:
    return ToolEvidence(
        evidence_id=f"{case['case_id']}:get_policy_requirements",
        tool_name="get_policy_requirements", status="available",
        source_paths=["retrieved_policies"],
        payload={"policies": case.get("retrieved_policies", [])},
    )


TOOLS: dict[ToolName, Callable[[dict], ToolEvidence]] = {
    "get_case_context": get_case_context,
    "get_account_history": get_account_history,
    "get_counterparty_profile": get_counterparty_profile,
    "get_network_snapshot": get_network_snapshot,
    "get_policy_requirements": get_policy_requirements,
}

# The callable boundary lets a live LLM planner be injected without giving the
# core workflow access to model credentials or removing its safety controls.
Planner = Callable[[dict, list[ToolName]], ToolName | None]


def _has_flag(case: dict, code: str) -> bool:
    return any(item.get("code") == code for item in case["assessment"].get("red_flags", []))


def heuristic_next_tool(case: dict, called: list[ToolName]) -> ToolName | None:
    """Deterministic offline planner used for tests and reproducible demonstrations.

    A production planner can replace this function only with a typed allow-listed
    decision.  The workflow guard still enforces the call limit and tool set.
    """
    if not called:
        return "get_case_context"
    # Investigate behavioural history before relationship data.
    if "get_account_history" not in called:
        return "get_account_history"
    transfer_like = case["transaction"].get("type") in {"TRANSFER", "CASH_OUT"}
    if transfer_like and "get_counterparty_profile" not in called:
        return "get_counterparty_profile"
    high_priority = case["assessment"].get("risk_label") in {"HIGH", "CRITICAL"}
    if high_priority and "get_network_snapshot" not in called and (
        _has_flag(case, "RF_BALANCE_DRAIN") or _has_flag(case, "RF_MODEL_ALERT")
    ):
        return "get_network_snapshot"
    if "get_policy_requirements" not in called:
        return "get_policy_requirements"
    return None


def run_investigation(case: dict, strategy: Literal["fixed_workflow", "bounded_agent"] = "bounded_agent",
                      max_steps: int = 5, planner: Planner | None = None) -> InvestigationRun:
    """Run an auditable read-only investigation with a hard tool-call budget."""
    if max_steps < 1:
        raise ValueError("max_steps must be at least one")
    started = perf_counter()
    calls: list[ToolName] = []
    evidence: list[ToolEvidence] = []
    if strategy == "fixed_workflow":
        planned: list[ToolName] = list(TOOL_NAMES)
    elif strategy == "bounded_agent":
        planned = []
        while len(planned) < max_steps:
            next_tool = (planner or heuristic_next_tool)(case, planned)
            if next_tool is None:
                break
            if next_tool not in TOOLS:
                raise ValueError(f"Planner returned a non-allow-listed tool: {next_tool}")
            if next_tool in planned:
                raise ValueError(f"Planner attempted a repeated tool call: {next_tool}")
            planned.append(next_tool)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    for tool_name in planned[:max_steps]:
        if tool_name not in TOOLS:  # defensive guard for future external planners
            raise ValueError(f"Tool is not allow-listed: {tool_name}")
        calls.append(tool_name)
        evidence.append(TOOLS[tool_name](case))

    stopped = "tool_budget_reached" if len(calls) >= max_steps else "evidence_plan_complete"
    return InvestigationRun(strategy, evidence, calls, stopped, (perf_counter() - started) * 1000)


def attach_investigation(case: dict, run: InvestigationRun) -> dict:
    """Return a copy that can safely be supplied to a narrative model."""
    return {**case, "investigation_evidence": [item.to_dict() for item in run.evidence],
            "investigation_trace": {"strategy": run.strategy, "tool_calls": run.tool_calls,
                                    "stopped_reason": run.stopped_reason}}
