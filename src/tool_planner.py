"""Optional typed LLM planner for the bounded investigation workflow."""
from __future__ import annotations

import json
import re

from pydantic import BaseModel, ConfigDict, Field

from .investigation import TOOL_NAMES, ToolName
from .llm import LLMProvider


class ToolDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    next_tool: str | None
    reason: str = Field(min_length=3, max_length=240)


def build_tool_plan_prompt(case: dict, called: list[ToolName]) -> str:
    visible = {
        "case_id": case["case_id"], "transaction": case["transaction"],
        "assessment": case["assessment"], "called_tools": called,
        "remaining_tools": [name for name in TOOL_NAMES if name not in called],
    }
    return (
        "Choose the single next READ-ONLY evidence tool for a financial-risk investigation. "
        "You must choose from remaining_tools or null if evidence collection is complete. "
        "Do not recommend an action or infer missing facts. Return JSON only: "
        '{"next_tool":"tool name or null","reason":"short evidence need"}.\nCASE:\n'
        + json.dumps(visible, sort_keys=True)
    )


class LLMToolPlanner:
    """Validates model output before the workflow executes a tool."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def __call__(self, case: dict, called: list[ToolName]) -> ToolName | None:
        raw = self.provider.generate(build_tool_plan_prompt(case, called), case)
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        decision = ToolDecision.model_validate_json(cleaned)
        if decision.next_tool is None:
            return None
        if decision.next_tool not in TOOL_NAMES:
            raise ValueError(f"Planner returned a non-allow-listed tool: {decision.next_tool}")
        if decision.next_tool in called:
            raise ValueError(f"Planner attempted a repeated tool call: {decision.next_tool}")
        return decision.next_tool  # type: ignore[return-value]
