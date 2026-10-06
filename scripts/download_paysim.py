"""Download PaySim through the official Kaggle CLI (credentials required)."""
from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "kaggle", "datasets", "download",
               "ealaxi/paysim1", "-f", "PS_20174392719_1491204439457_log.csv",
               "-p", str(OUT), "--force"]
    subprocess.run(command, check=True)
    for archive in OUT.glob("PS_*.csv.zip"):
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(OUT)
    csvs = sorted(OUT.glob("*.csv"))
    if not csvs:
        raise SystemExit("Download completed but no CSV was found in data/raw.")
    print(f"PaySim CSV ready: {csvs[0]}")


if __name__ == "__main__":
    main()

