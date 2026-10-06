from __future__ import annotations

import pandas as pd

from .features import MODEL_FEATURES, build_features
from .modeling import score_supervised


def local_sensitivity(bundle: dict, transaction: dict) -> pd.DataFrame:
    """One-feature-at-a-time sensitivity; descriptive, not a causal attribution."""
    package = bundle["models"][bundle["champion"]]
    features = build_features(pd.DataFrame([transaction]))
    current = float(score_supervised(package, features)[0])
    rows = []
    for feature in MODEL_FEATURES:
        modified = features.copy()
        modified.loc[modified.index[0], feature] = bundle["feature_reference"][feature]
        counterfactual = float(score_supervised(package, modified)[0])
        rows.append({
            "feature": feature,
            "current_value": str(features.iloc[0][feature]),
            "reference_value": str(bundle["feature_reference"][feature]),
            "score_change_if_replaced": counterfactual - current,
            "absolute_sensitivity": abs(counterfactual - current),
        })
    return pd.DataFrame(rows).sort_values("absolute_sensitivity", ascending=False)

