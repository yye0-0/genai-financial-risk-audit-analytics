from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    root: Path = ROOT
    raw: Path = ROOT / "data" / "raw"
    processed: Path = ROOT / "data" / "processed"
    artifacts: Path = ROOT / "artifacts"
    policies: Path = ROOT / "knowledge_base" / "policies.json"


PATHS = Paths()
PAYSim_COLUMNS = [
    "step", "type", "amount", "nameOrig", "oldbalanceOrg",
    "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest",
    "isFraud", "isFlaggedFraud",
]

