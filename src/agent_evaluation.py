"""Objective evaluation of bounded investigation strategies.

Unlike a language-quality score, this checks whether the expected evidence was
collected, whether a hard safety constraint was respected, and how many tools
were consumed.  Cases use synthetic enrichment and must never be presented as
bank-production performance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import PATHS
from .investigation import attach_investigation, run_investigation


def score_run(case: dict, strategy: str) -> dict:
    run = run_investigation(case, strategy=strategy)  # type: ignore[arg-type]
    gold = case.get("investigation_gold", {})
    available = {item.tool_name for item in run.evidence if item.status == "available"}
    required = set(gold.get("required_tools", []))
    missing = sorted(required - available)
    recommended = case["assessment"]["recommended_action"]
    unsafe_close = int(
        case["assessment"]["risk_label"] in {"HIGH", "CRITICAL"}
        and recommended == "allow_with_monitoring"
    )
    return {
        "case_id": case["case_id"], "strategy": strategy,
        "required_tool_recall": len(required & available) / max(1, len(required)),
        "all_required_tools_collected": int(not missing),
        "missing_required_tools": ";".join(missing),
        "unsafe_close": unsafe_close,
        "tool_calls": len(run.tool_calls), "available_evidence": len(available),
        "latency_ms": run.latency_ms,
    }


def evaluate_strategies(cases: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = [score_run(case, strategy) for strategy in ("fixed_workflow", "bounded_agent") for case in cases]
    details = pd.DataFrame(rows)
    summary = details.groupby("strategy", as_index=False).agg(
        cases=("case_id", "count"),
        required_tool_recall=("required_tool_recall", "mean"),
        task_success_rate=("all_required_tools_collected", "mean"),
        unsafe_close_rate=("unsafe_close", "mean"),
        mean_tool_calls=("tool_calls", "mean"),
        mean_latency_ms=("latency_ms", "mean"),
    )
    return details, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=Path("data/investigation_cases.json"))
    parser.add_argument("--output-dir", type=Path, default=PATHS.artifacts)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    details, summary = evaluate_strategies(cases)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    details.to_csv(args.output_dir / "agent_eval_details.csv", index=False)
    summary.to_csv(args.output_dir / "agent_eval_summary.csv", index=False)
    print(summary.to_string(index=False))


__all__ = ["attach_investigation", "evaluate_strategies", "score_run"]


if __name__ == "__main__":
    main()
