"""Run one bounded investigation case from the command line.

The default planner is deterministic and needs no API key.  The optional LLM
planner accepts only allow-listed JSON tool decisions; it is never given direct
database access.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Support the documented `python scripts/run_investigation.py` command from a
# fresh clone without requiring package installation first.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.investigation import attach_investigation, run_investigation
from src.llm import get_provider
from src.tool_planner import LLMToolPlanner


def load_case(path: Path, case_id: str | None) -> dict:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if not cases:
        raise ValueError("Case file is empty")
    if case_id is None:
        return cases[0]
    for case in cases:
        if case["case_id"] == case_id:
            return case
    available = ", ".join(item["case_id"] for item in cases)
    raise ValueError(f"Unknown case_id={case_id}. Available: {available}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded evidence investigation")
    parser.add_argument("--cases", type=Path, default=Path("data/investigation_cases.json"))
    parser.add_argument("--case", dest="case_id", help="Case ID; defaults to the first synthetic case")
    parser.add_argument("--planner", choices=("heuristic", "llm"), default="heuristic")
    parser.add_argument("--provider", default="openai",
                        help="Provider for --planner llm; only 'openai' is currently supported")
    parser.add_argument("--max-steps", type=int, default=5)
    args = parser.parse_args()

    case = load_case(args.cases, args.case_id)
    planner = None
    if args.planner == "llm":
        if args.provider == "mock":
            raise ValueError("The mock provider emits narratives, not tool decisions. Use --planner heuristic.")
        planner = LLMToolPlanner(get_provider(args.provider))
    run = run_investigation(case, "bounded_agent", max_steps=args.max_steps, planner=planner)
    print(json.dumps({
        "case_id": case["case_id"],
        "trace": run.to_dict(),
        "narrative_input": attach_investigation(case, run),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
