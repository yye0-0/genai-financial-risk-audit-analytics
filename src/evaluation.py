from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd

from .config import PATHS
from .llm import RiskNarrative, get_provider
from .prompts import PROMPT_VERSIONS, build_prompt


def _clean_json(raw: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())


def _number_set(value: object) -> set[float]:
    values = re.findall(
        r"(?<![A-Za-z0-9_-])[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?",
        json.dumps(value),
    )
    output = set()
    for item in values:
        try:
            output.add(round(float(item.replace(",", "")), 6))
        except ValueError:
            pass
    return output


def _resolve_path(case: dict, path: str):
    current: object = case
    for piece in path.split("."):
        if isinstance(current, list):
            current = current[int(piece)]
        elif isinstance(current, dict) and piece in current:
            current = current[piece]
        else:
            raise KeyError(path)
    return current


def evaluate_response(raw: str, case: dict) -> dict:
    try:
        obj = json.loads(_clean_json(raw))
        json_valid = 1
    except Exception:
        return {key: 0.0 for key in (
            "json_valid", "schema_valid", "case_id_agreement", "risk_label_agreement",
            "action_agreement", "citation_precision", "citation_recall",
            "source_path_validity", "red_flag_coverage", "factual_consistency",
            "completeness", "overall")}
    try:
        parsed = RiskNarrative.model_validate(obj)
        schema_valid = 1
    except Exception:
        return {"json_valid": json_valid, **{key: 0.0 for key in (
            "schema_valid", "case_id_agreement", "risk_label_agreement",
            "action_agreement", "citation_precision", "citation_recall",
            "source_path_validity", "red_flag_coverage", "factual_consistency",
            "completeness", "overall")}}

    gold = case["gold"]
    case_id = int(parsed.case_id == gold["case_id"])
    label = int(parsed.risk_label == gold["risk_label"])
    action = int(gold["recommended_action"] in parsed.recommended_actions)

    predicted_citations = set(parsed.policy_citations)
    gold_citations = set(gold["policy_ids"])
    citation_precision = (1.0 if not predicted_citations and not gold_citations else
                          len(predicted_citations & gold_citations) /
                          max(1, len(predicted_citations)))
    citation_recall = (1.0 if not gold_citations and not predicted_citations else
                       len(predicted_citations & gold_citations) / max(1, len(gold_citations)))

    paths = [path for item in parsed.key_evidence for path in item.source_paths]
    valid_paths = 0
    for path in paths:
        try:
            _resolve_path(case, path)
            valid_paths += 1
        except (KeyError, IndexError, ValueError, TypeError):
            pass
    source_validity = valid_paths / max(1, len(paths))

    evidence_text = " ".join(item.claim for item in parsed.key_evidence)
    required_flags = set(gold["red_flag_codes"])
    covered_flags = {flag for flag in required_flags if flag in evidence_text}
    flag_coverage = 1.0 if not required_flags else len(covered_flags) / len(required_flags)

    allowed_numbers = _number_set({key: case[key] for key in
                                   ("case_id", "transaction", "assessment", "retrieved_policies",
                                    "investigation_evidence") if key in case})
    output_numbers = _number_set({"summary": parsed.summary,
                                  "claims": [item.claim for item in parsed.key_evidence]})
    factual = len(output_numbers & allowed_numbers) / max(1, len(output_numbers))
    completeness = sum([
        bool(parsed.summary), bool(parsed.key_evidence), bool(parsed.recommended_actions),
        bool(parsed.uncertainties), flag_coverage == 1,
    ]) / 5
    components = [json_valid, schema_valid, case_id, label, action,
                  citation_precision, citation_recall, source_validity,
                  flag_coverage, factual, completeness]
    return {
        "json_valid": json_valid, "schema_valid": schema_valid,
        "case_id_agreement": case_id, "risk_label_agreement": label,
        "action_agreement": action, "citation_precision": citation_precision,
        "citation_recall": citation_recall, "source_path_validity": source_validity,
        "red_flag_coverage": flag_coverage, "factual_consistency": factual,
        "completeness": completeness, "overall": sum(components) / len(components),
    }


def run_evaluation(cases: list[dict], provider_name: str = "mock") -> tuple[pd.DataFrame, list[dict]]:
    provider = get_provider(provider_name)
    rows, outputs = [], []
    for version in PROMPT_VERSIONS:
        for case in cases:
            started = time.perf_counter()
            error = None
            try:
                raw = provider.generate(build_prompt(version, case), case)
            except Exception as exc:
                raw, error = "", f"{type(exc).__name__}: {exc}"
            latency_ms = (time.perf_counter() - started) * 1000
            score = evaluate_response(raw, case)
            rows.append({"prompt_version": version, "case_id": case["case_id"],
                         "provider": provider_name, "latency_ms": latency_ms,
                         "error": error, **score})
            outputs.append({"prompt_version": version, "case_id": case["case_id"],
                            "provider": provider_name, "raw": raw, "error": error})
    return pd.DataFrame(rows), outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=PATHS.artifacts / "eval_cases.json")
    parser.add_argument("--provider", default="mock")
    parser.add_argument("--output-dir", type=Path, default=PATHS.artifacts)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    scores, outputs = run_evaluation(cases, args.provider)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output_dir / "llm_eval_details.csv", index=False)
    summary = scores.groupby(["provider", "prompt_version"]).mean(numeric_only=True).reset_index()
    summary.to_csv(args.output_dir / "llm_eval_summary.csv", index=False)
    (args.output_dir / "llm_outputs.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    manifest = {
        "provider": args.provider, "cases": len(cases),
        "prompt_versions": list(PROMPT_VERSIONS),
        "is_real_llm_comparison": args.provider != "mock",
        "interpretation": ("Real provider prompt comparison" if args.provider != "mock" else
                           "Offline contract test only; mock ignores prompt wording"),
    }
    (args.output_dir / "llm_eval_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
