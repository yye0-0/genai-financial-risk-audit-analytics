from __future__ import annotations

import numpy as np
import pandas as pd

# Only fields known at authorization time. Post-transaction balances and legacy
# flags are excluded from the model to reduce target leakage.
MODEL_FEATURES = [
    "type", "amount_log", "oldbalance_org_log", "oldbalance_dest_log",
    "hour_sin", "hour_cos", "day", "amount_to_origin_balance",
    "amount_to_dest_balance", "remaining_origin_ratio", "origin_zero_balance",
    "dest_zero_balance", "dest_is_merchant", "is_transfer", "is_cash_out",
    "is_high_value",
]
CATEGORICAL_FEATURES = ["type"]
NUMERIC_FEATURES = [x for x in MODEL_FEATURES if x not in CATEGORICAL_FEATURES]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    x["type"] = df["type"].astype(str)
    amount = df["amount"].astype(float).clip(lower=0)
    old_org = df["oldbalanceOrg"].astype(float).clip(lower=0)
    old_dest = df["oldbalanceDest"].astype(float).clip(lower=0)
    hour = (df["step"].astype(int) - 1) % 24
    x["amount_log"] = np.log1p(amount)
    x["oldbalance_org_log"] = np.log1p(old_org)
    x["oldbalance_dest_log"] = np.log1p(old_dest)
    x["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    x["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    x["day"] = (df["step"].astype(int) - 1) // 24
    x["amount_to_origin_balance"] = (amount / (old_org + 1.0)).clip(0, 10)
    x["amount_to_dest_balance"] = (amount / (old_dest + 1.0)).clip(0, 10)
    x["remaining_origin_ratio"] = ((old_org - amount).clip(lower=0) / (old_org + 1.0)).clip(0, 1)
    x["origin_zero_balance"] = (old_org <= 0).astype(int)
    x["dest_zero_balance"] = (old_dest <= 0).astype(int)
    x["dest_is_merchant"] = df["nameDest"].astype(str).str.startswith("M").astype(int)
    x["is_transfer"] = (df["type"].astype(str) == "TRANSFER").astype(int)
    x["is_cash_out"] = (df["type"].astype(str) == "CASH_OUT").astype(int)
    x["is_high_value"] = (amount >= 200_000).astype(int)
    return x[MODEL_FEATURES].replace([np.inf, -np.inf], 0).fillna(0)

