from __future__ import annotations

import json
from dataclasses import replace

from src.config import PATHS
from src.data import generate_demo_data
from src.evaluation import run_evaluation
from src.modeling import (load_config, save_training_outputs,
                          train_official_experiment)
from src.pipeline import build_cases


def main() -> None:
    demo_dir = PATHS.artifacts / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    data_path = PATHS.raw / "demo_paysim.csv"
    generate_demo_data(n_rows=20_000, seed=42).to_csv(data_path, index=False)
    config = replace(load_config(), training_negative_sample=10_000,
                     anomaly_normal_sample=5_000, dashboard_top_alerts=500,
                     dashboard_normal_sample=100)
    bundle, metrics, scored, comparison = train_official_experiment(data_path, config, demo_dir)
    save_training_outputs(bundle, metrics, scored, comparison, demo_dir)
    scored["fraud_probability"] = scored["champion_score"]
    cases = build_cases(scored, bundle["anomaly_threshold"],
                        bundle["operating_point"]["champion_threshold"], 20)
    (demo_dir / "eval_cases.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    details, outputs = run_evaluation(cases, "mock")
    details.to_csv(demo_dir / "llm_eval_details.csv", index=False)
    details.groupby(["provider", "prompt_version"]).mean(numeric_only=True).reset_index().to_csv(
        demo_dir / "llm_eval_summary.csv", index=False)
    (demo_dir / "llm_outputs.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print("DEMO DATA ONLY — official artifacts in artifacts/ were not overwritten")
    print(json.dumps({"champion": metrics["champion"], "test": metrics["test"]}, indent=2))


if __name__ == "__main__":
    main()
