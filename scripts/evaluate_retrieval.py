from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PATHS
from src.rag import evaluate_retrieval


def main() -> None:
    cases = json.loads((PATHS.root / "knowledge_base" / "retrieval_eval.json").read_text(encoding="utf-8"))
    result = evaluate_retrieval(cases)
    output = PATHS.artifacts / "retrieval_metrics.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
