from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PATHS, PAYSim_COLUMNS

DTYPES = {
    "step": "int16", "type": "string", "amount": "float32",
    "nameOrig": "string", "oldbalanceOrg": "float32", "newbalanceOrig": "float32",
    "nameDest": "string", "oldbalanceDest": "float32", "newbalanceDest": "float32",
    "isFraud": "int8", "isFlaggedFraud": "int8",
}


def validate_schema(df: pd.DataFrame, require_target: bool = True) -> None:
    required = set(PAYSim_COLUMNS if require_target else PAYSim_COLUMNS[:-2])
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing PaySim columns: {missing}")


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    validate_schema(df)
    out = df.copy()
    numeric = ["step", "amount", "oldbalanceOrg", "newbalanceOrig",
               "oldbalanceDest", "newbalanceDest", "isFraud", "isFlaggedFraud"]
    for col in numeric:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=numeric + ["type", "nameOrig", "nameDest"])
    out = out[(out["amount"] >= 0) & (out["step"] >= 0)]
    for label in ["isFraud", "isFlaggedFraud"]:
        invalid = ~out[label].isin([0, 1])
        if invalid.any():
            raise ValueError(f"{label} contains {int(invalid.sum())} values outside 0/1")
        out[label] = out[label].astype("int8")
    out = out.drop_duplicates().reset_index(drop=True)
    return out


def generate_demo_data(n_rows: int = 20_000, fraud_rate: float = 0.02,
                       seed: int = 42) -> pd.DataFrame:
    """Create runnable PaySim-shaped demo data. It is not the PaySim dataset."""
    rng = np.random.default_rng(seed)
    n_fraud = max(20, int(n_rows * fraud_rate))
    fraud_idx = set(rng.choice(n_rows, n_fraud, replace=False).tolist())
    types = rng.choice(["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
                       n_rows, p=[.22, .34, .07, .30, .07]).astype(object)
    amount = rng.lognormal(9.0, 1.15, n_rows).clip(10, 900_000)
    old_org = rng.lognormal(10.0, 1.1, n_rows).clip(0, 2_000_000)
    old_dest = rng.lognormal(10.3, 1.3, n_rows).clip(0, 4_000_000)
    is_fraud = np.zeros(n_rows, dtype=int)
    for i in fraud_idx:
        is_fraud[i] = 1
        types[i] = rng.choice(["TRANSFER", "CASH_OUT"])
        amount[i] = min(old_org[i] * rng.uniform(.75, 1.0), 1_500_000)
    new_org = np.maximum(0, old_org - amount)
    new_dest = old_dest + np.where(types == "CASH_IN", -amount, amount)
    new_dest = np.maximum(0, new_dest)
    # Imperfect rule flag, intentionally separate from the target.
    flagged = ((types == "TRANSFER") & (amount > 200_000)).astype(int)
    df = pd.DataFrame({
        "step": rng.integers(1, 745, n_rows), "type": types, "amount": amount,
        "nameOrig": [f"C{x:09d}" for x in rng.integers(0, n_rows * 5, n_rows)],
        "oldbalanceOrg": old_org, "newbalanceOrig": new_org,
        "nameDest": [f"{rng.choice(['C','M'])}{x:09d}" for x in rng.integers(0, n_rows * 5, n_rows)],
        "oldbalanceDest": old_dest, "newbalanceDest": new_dest,
        "isFraud": is_fraud, "isFlaggedFraud": flagged,
    })
    return df.sort_values("step").reset_index(drop=True)


def load_transactions(path: str | Path) -> pd.DataFrame:
    return clean_transactions(pd.read_csv(path, dtype=DTYPES))


def iter_transactions(path: str | Path, chunksize: int = 500_000):
    for chunk in pd.read_csv(path, dtype=DTYPES, chunksize=chunksize):
        validate_schema(chunk)
        yield chunk


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def profile_dataset(path: str | Path, chunksize: int = 500_000,
                    train_fraction: float = .60, validation_fraction: float = .80) -> dict:
    path = Path(path)
    step_parts, type_counts = [], {}
    missing = {col: 0 for col in PAYSim_COLUMNS}
    total = fraud = flagged = negative_amount = invalid_labels = 0
    for chunk in iter_transactions(path, chunksize):
        total += len(chunk)
        fraud += int(chunk["isFraud"].sum())
        flagged += int(chunk["isFlaggedFraud"].sum())
        negative_amount += int((chunk["amount"] < 0).sum())
        invalid_labels += int((~chunk["isFraud"].isin([0, 1])).sum())
        for col, count in chunk.isna().sum().items():
            missing[col] += int(count)
        for key, value in chunk["type"].value_counts().items():
            type_counts[str(key)] = type_counts.get(str(key), 0) + int(value)
        step_parts.append(chunk.groupby("step", observed=True)["isFraud"].agg(["size", "sum"]))
    by_step = pd.concat(step_parts).groupby(level=0).sum().sort_index()
    cumulative = by_step["size"].cumsum() / total
    train_end = int(cumulative.index[cumulative.ge(train_fraction)][0])
    validation_end = int(cumulative.index[cumulative.ge(validation_fraction)][0])

    def split_stats(start: int, end: int) -> dict:
        part = by_step.loc[start:end]
        rows, frauds = int(part["size"].sum()), int(part["sum"].sum())
        return {"step_start": start, "step_end": end, "rows": rows,
                "fraud": frauds, "fraud_rate": frauds / rows}

    profile = {
        "source": "official_paySim" if total > 1_000_000 else "local_or_demo",
        "path": str(path.resolve()), "bytes": path.stat().st_size,
        "sha256": sha256_file(path), "rows": total, "fraud": fraud,
        "fraud_rate": fraud / total, "legacy_flagged": flagged,
        "step_min": int(by_step.index.min()), "step_max": int(by_step.index.max()),
        "transaction_types": type_counts, "missing_values": missing,
        "negative_amount_rows": negative_amount, "invalid_label_rows": invalid_labels,
        "splits": {
            "train": split_stats(int(by_step.index.min()), train_end),
            "validation": split_stats(train_end + 1, validation_end),
            "test": split_stats(validation_end + 1, int(by_step.index.max())),
        },
    }
    return profile


def load_time_range(path: str | Path, step_start: int, step_end: int,
                    chunksize: int = 500_000) -> pd.DataFrame:
    parts = []
    for chunk in iter_transactions(path, chunksize):
        selected = chunk[(chunk["step"] >= step_start) & (chunk["step"] <= step_end)]
        if not selected.empty:
            parts.append(selected)
    if not parts:
        raise ValueError(f"No rows found for step range {step_start}-{step_end}")
    return pd.concat(parts, ignore_index=True)


def load_training_sample(path: str | Path, step_end: int, target_negatives: int = 300_000,
                         chunksize: int = 500_000, seed: int = 42,
                         known_negative_rows: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if known_negative_rows is None:
        known_negative_rows = sum(
            int(((c["step"] <= step_end) & (c["isFraud"] == 0)).sum())
            for c in iter_transactions(path, chunksize)
        )
    fraction = min(1.0, target_negatives / max(1, known_negative_rows))
    fraud_parts, normal_parts = [], []
    for chunk in iter_transactions(path, chunksize):
        chunk = chunk[chunk["step"] <= step_end]
        if chunk.empty:
            continue
        fraud_part = chunk[chunk["isFraud"] == 1]
        normal = chunk[chunk["isFraud"] == 0]
        keep = rng.random(len(normal)) < fraction
        fraud_parts.append(fraud_part)
        normal_parts.append(normal.loc[keep])
    fraud = pd.concat(fraud_parts, ignore_index=True)
    normal = pd.concat(normal_parts, ignore_index=True)
    if len(normal) > target_negatives:
        normal = normal.sample(target_negatives, random_state=seed)
    return pd.concat([fraud, normal], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def save_profile(profile: dict, output: Path = PATHS.artifacts / "data_profile.json") -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PATHS.raw / "demo_paysim.csv")
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_demo_data(args.rows, seed=args.seed).to_csv(args.output, index=False)
    print(f"Wrote {args.rows:,} DEMO rows to {args.output}")


if __name__ == "__main__":
    main()

