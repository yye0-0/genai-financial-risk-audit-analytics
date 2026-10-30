from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .prompts import build_prompt


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim: str = Field(min_length=3)
    source_paths: list[str] = Field(min_length=1)


class RiskNarrative(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(min_length=3)
    risk_label: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    summary: str = Field(min_length=10)
    key_evidence: list[EvidenceItem] = Field(min_length=1)
    recommended_actions: list[Literal[
        "allow_with_monitoring", "queue_for_review", "priority_investigation",
        "recommend_temporary_hold_and_escalation",
    ]] = Field(min_length=1)
    policy_citations: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(min_length=1)


class LLMProvider(ABC):
    last_usage: dict | None = None
    @abstractmethod
    def generate(self, prompt: str, case: dict) -> str: ...


class MockLLM(LLMProvider):
    """Offline deterministic adapter for plumbing tests; not an LLM benchmark."""

    def generate(self, prompt: str, case: dict) -> str:
        assessment = case["assessment"]
        flags = assessment["red_flags"]
        out = {
            "case_id": case["case_id"],
            "risk_label": assessment["risk_label"],
            "summary": (f"Risk engine classified this transaction as {assessment['risk_label']} "
                        f"with score {assessment['risk_score']}; this is a triage signal, not proof of fraud."),
            "key_evidence": [
                {"claim": f"{flag['code']}: {json.dumps(flag['evidence'], sort_keys=True)}",
                 "source_paths": [f"assessment.red_flags.{index}.evidence"]}
                for index, flag in enumerate(flags)
            ] or [{"claim": f"Risk score is {assessment['risk_score']}.",
                   "source_paths": ["assessment.risk_score"]}],
            "recommended_actions": [assessment["recommended_action"]],
            "policy_citations": sorted({f["policy_id"] for f in flags}),
            "uncertainties": ["No device, identity, counterparty history, or analyst disposition was supplied."],
        }
        return json.dumps(out)


class OpenAICompatibleLLM(LLMProvider):
    def __init__(self, model: str | None = None):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY for the real provider; mock remains available offline")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the optional dependency: pip install -e .[llm]") from exc
        kwargs = {"api_key": os.environ.get("OPENAI_API_KEY"), "timeout": 60.0, "max_retries": 1}
        if os.environ.get("OPENAI_BASE_URL"):
            kwargs["base_url"] = os.environ["OPENAI_BASE_URL"]
        self.client = OpenAI(**kwargs)
        self.model = model or os.environ.get("LLM_MODEL", "gpt-4.1-mini")

    def generate(self, prompt: str, case: dict) -> str:
        self.last_usage = None
        response = self.client.chat.completions.create(
            model=self.model, temperature=0, messages=[{"role": "user", "content": prompt}])
        usage = response.usage
        self.last_usage = {"model": response.model,
                           "input_tokens": usage.prompt_tokens if usage else None,
                           "output_tokens": usage.completion_tokens if usage else None}
        return response.choices[0].message.content or ""


def get_provider(name: str | None = None) -> LLMProvider:
    provider = (name or os.environ.get("LLM_PROVIDER", "mock")).lower()
    if provider == "mock":
        return MockLLM()
    if provider in {"openai", "openai_compatible"}:
        return OpenAICompatibleLLM()
    raise ValueError(f"Unsupported LLM_PROVIDER={provider}")


def generate_narrative(case: dict, version: str = "grounded_structured",
                       provider: LLMProvider | None = None) -> tuple[str, RiskNarrative | None]:
    supplied_case = {key: case[key] for key in
                     ("case_id", "transaction", "assessment", "retrieved_policies")}
    raw = (provider or get_provider()).generate(build_prompt(version, supplied_case), supplied_case)
    try:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
        parsed = RiskNarrative.model_validate_json(cleaned)
    except Exception:
        parsed = None
    return raw, parsed

