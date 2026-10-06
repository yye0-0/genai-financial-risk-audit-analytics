from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = sorted((ROOT / "src").glob("*.py")) + [ROOT / "app.py", ROOT / "config" / "modeling.yaml"]


def digest_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    metrics = json.loads((ROOT / "artifacts" / "model_metrics.json").read_text(encoding="utf-8"))
    packages = ["numpy", "pandas", "scikit-learn", "scipy", "streamlit", "pydantic", "joblib"]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "dataset_sha256": metrics["dataset"]["sha256"],
        "code_sha256": digest_files(CODE_FILES),
        "champion": metrics["champion"],
        "config": str((ROOT / "config" / "modeling.yaml").relative_to(ROOT)),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "commands": [
            "python -m src.modeling --data data/raw/PS_20174392719_1491204439457_log.csv",
            "python -m src.pipeline --cases 60",
            "python scripts/evaluate_retrieval.py",
            "python -m src.evaluation --provider mock",
        ],
    }
    output = ROOT / "artifacts" / "run_manifest.json"
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

