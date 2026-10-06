from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, fbeta_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .calibration import PlattCalibrator
from .config import PATHS
from .data import (load_time_range, load_training_sample, profile_dataset,
                   save_profile)
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_features


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    chunksize: int = 500_000
    train_end_fraction: float = .60
    validation_end_fraction: float = .80
    training_negative_sample: int = 300_000
    anomaly_normal_sample: int = 100_000
    target_alert_rate: float = .005
    dashboard_top_alerts: int = 5_000
    dashboard_normal_sample: int = 1_000


def load_config(path: Path | None = None) -> ExperimentConfig:
    path = path or PATHS.root / "config" / "modeling.yaml"
    return ExperimentConfig(**yaml.safe_load(path.read_text(encoding="utf-8")))


def _preprocessor(scale: bool) -> ColumnTransformer:
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
    ])


def _supervised_estimators(seed: int) -> dict[str, Pipeline]:
    signature_features = {
        "oldbalance_org_log", "amount_to_origin_balance",
        "remaining_origin_ratio", "origin_zero_balance",
    }
    reduced_numeric = [feature for feature in NUMERIC_FEATURES
                       if feature not in signature_features]
    return {
        "logistic_regression": Pipeline([
            ("features", _preprocessor(scale=True)),
            ("model", LogisticRegression(max_iter=400, class_weight="balanced",
                                          solver="lbfgs", random_state=seed)),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("features", _preprocessor(scale=False)),
            ("model", HistGradientBoostingClassifier(
                max_iter=160, max_leaf_nodes=15, learning_rate=.08,
                min_samples_leaf=30, l2_regularization=.1,
                class_weight="balanced", random_state=seed)),
        ]),
        "hgb_balance_signature_ablation": Pipeline([
            ("features", ColumnTransformer([
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                 CATEGORICAL_FEATURES),
                ("num", Pipeline([("impute", SimpleImputer(strategy="median"))]),
                 reduced_numeric),
            ])),
            ("model", HistGradientBoostingClassifier(
                max_iter=160, max_leaf_nodes=15, learning_rate=.08,
                min_samples_leaf=30, l2_regularization=.1,
                class_weight="balanced", random_state=seed)),
        ]),
    }


def _calibration_table(y: np.ndarray, score: np.ndarray, bins: int = 10) -> list[dict]:
    frame = pd.DataFrame({"y": y, "score": score})
    try:
        frame["bin"] = pd.qcut(frame["score"], q=bins, duplicates="drop")
    except ValueError:
        frame["bin"] = "all"
    return [
        {"bin": str(label), "rows": len(part),
         "mean_score": float(part["score"].mean()),
         "observed_fraud_rate": float(part["y"].mean())}
        for label, part in frame.groupby("bin", observed=True)
    ]


def _top_k(y: np.ndarray, score: np.ndarray) -> dict:
    order = np.argsort(-score)
    total_fraud = max(1, int(y.sum()))
    output = {}
    for fraction in (.001, .005, .01, .02, .05):
        k = max(1, int(len(y) * fraction))
        selected = y[order[:k]]
        output[f"top_{fraction:.3f}"] = {
            "rows": k, "precision": float(selected.mean()),
            "recall": float(selected.sum() / total_fraud),
            "fraud_captured": int(selected.sum()),
        }
    return output


def _metrics(y: np.ndarray, scores: np.ndarray, threshold: float,
             calibrated: bool = True) -> dict:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else None,
        "average_precision": float(average_precision_score(y, scores)),
        "brier_score": float(brier_score_loss(y, scores)) if calibrated else None,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "threshold": float(threshold), "alert_rate": float(pred.mean()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "support": int(len(y)), "fraud_count": int(y.sum()),
        "top_k": _top_k(y, scores),
        "calibration": _calibration_table(y, scores) if calibrated else [],
    }


def _capacity_threshold(scores: np.ndarray, alert_rate: float) -> float:
    return float(np.quantile(scores, 1 - alert_rate))


def _f2_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y, scores)
    if not len(thresholds):
        return .5
    f2 = 5 * precision[:-1] * recall[:-1] / np.maximum(4 * precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(f2))])


def score_supervised(package: dict, features: pd.DataFrame) -> np.ndarray:
    raw = package["estimator"].predict_proba(features)[:, 1]
    return np.asarray(package["calibrator"].predict(raw), dtype=float)


def _importance(estimator: Pipeline, raw_validation: pd.DataFrame, seed: int) -> pd.DataFrame:
    fraud = raw_validation[raw_validation["isFraud"] == 1]
    normal = raw_validation[raw_validation["isFraud"] == 0].sample(
        min(20_000, int((raw_validation["isFraud"] == 0).sum())), random_state=seed)
    sample = pd.concat([fraud, normal], ignore_index=True).sample(frac=1, random_state=seed)
    x, y = build_features(sample), sample["isFraud"].to_numpy()
    result = permutation_importance(estimator, x, y, scoring="average_precision",
                                    n_repeats=3, random_state=seed, n_jobs=-1)
    return pd.DataFrame({"feature": x.columns, "importance_mean": result.importances_mean,
                         "importance_std": result.importances_std}).sort_values(
                             "importance_mean", ascending=False)


def train_official_experiment(data_path: Path, config: ExperimentConfig,
                              output_dir: Path = PATHS.artifacts) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = profile_dataset(data_path, config.chunksize, config.train_end_fraction,
                              config.validation_end_fraction)
    save_profile(profile, output_dir / "data_profile.json")
    train_info, val_info, test_info = (profile["splits"][key]
                                       for key in ("train", "validation", "test"))

    train = load_training_sample(
        data_path, train_info["step_end"], config.training_negative_sample,
        config.chunksize, config.seed, train_info["rows"] - train_info["fraud"])
    validation = load_time_range(data_path, val_info["step_start"], val_info["step_end"],
                                 config.chunksize)
    x_train, y_train = build_features(train), train["isFraud"].to_numpy()
    x_val, y_val = build_features(validation), validation["isFraud"].to_numpy()

    packages, validation_metrics = {}, {}
    for name, estimator in _supervised_estimators(config.seed).items():
        estimator.fit(x_train, y_train)
        raw_val = estimator.predict_proba(x_val)[:, 1]
        calibrator = PlattCalibrator().fit(raw_val, y_val)
        score = np.asarray(calibrator.predict(raw_val), dtype=float)
        threshold = _capacity_threshold(score, config.target_alert_rate)
        packages[name] = {"estimator": estimator, "calibrator": calibrator,
                          "threshold": threshold,
                          "f2_threshold": _f2_threshold(y_val, score)}
        validation_metrics[name] = _metrics(y_val, score, threshold)

    champion = max(validation_metrics, key=lambda key: validation_metrics[key]["average_precision"])
    anomaly_train = train[train["isFraud"] == 0].sample(
        min(config.anomaly_normal_sample, int((train["isFraud"] == 0).sum())),
        random_state=config.seed)
    anomaly = Pipeline([
        ("features", _preprocessor(scale=True)),
        ("model", IsolationForest(n_estimators=160, contamination="auto",
                                  random_state=config.seed, n_jobs=-1)),
    ])
    anomaly.fit(build_features(anomaly_train))
    anomaly_val = -anomaly.decision_function(x_val)
    anomaly_threshold = _capacity_threshold(anomaly_val, config.target_alert_rate)
    validation_metrics["isolation_forest"] = _metrics(
        y_val, anomaly_val, anomaly_threshold, calibrated=False)
    validation_metrics["legacy_flag_rule"] = _metrics(
        y_val, validation["isFlaggedFraud"].to_numpy(float), .5, calibrated=False)

    _importance(packages[champion]["estimator"], validation, config.seed).to_csv(
        output_dir / "feature_importance.csv", index=False)
    training_sample_info = {"rows": len(y_train), "fraud": int(y_train.sum()),
                            "negative_sample": int(len(y_train) - y_train.sum())}
    feature_reference = {
        **{column: float(x_train[column].median()) for column in NUMERIC_FEATURES},
        "type": str(x_train["type"].mode().iloc[0]),
    }
    del x_train, x_val, train, validation

    test = load_time_range(data_path, test_info["step_start"], test_info["step_end"],
                           config.chunksize)
    x_test, y_test = build_features(test), test["isFraud"].to_numpy()
    test_metrics, test_scores = {}, {}
    for name, package in packages.items():
        score = score_supervised(package, x_test)
        test_scores[name] = score
        test_metrics[name] = _metrics(y_test, score, package["threshold"])
    anomaly_test = -anomaly.decision_function(x_test)
    test_metrics["isolation_forest"] = _metrics(
        y_test, anomaly_test, anomaly_threshold, calibrated=False)
    test_metrics["legacy_flag_rule"] = _metrics(
        y_test, test["isFlaggedFraud"].to_numpy(float), .5, calibrated=False)

    champion_score = test_scores[champion]
    champion_threshold = packages[champion]["threshold"]
    test = test.copy()
    for name, score in test_scores.items():
        test[f"{name}_score"] = score
    test["anomaly_score"] = anomaly_test
    test["champion_score"] = champion_score
    test["predicted_alert"] = (champion_score >= champion_threshold).astype("int8")
    alert_index = test.index[test["predicted_alert"] == 1]
    top_index = test.nlargest(config.dashboard_top_alerts, "champion_score").index
    fraud_index = test.index[test["isFraud"] == 1]
    normal_pool = test[(test["isFraud"] == 0) & (test["predicted_alert"] == 0)]
    normal_index = normal_pool.sample(min(config.dashboard_normal_sample, len(normal_pool)),
                                      random_state=config.seed).index
    selected_index = alert_index.union(top_index).union(fraud_index).union(normal_index)
    scored = test.loc[selected_index].sort_values("champion_score", ascending=False).reset_index(drop=True)

    bundle = {
        "models": packages, "champion": champion, "anomaly": anomaly,
        "anomaly_threshold": anomaly_threshold,
        "operating_point": {"target_alert_rate": config.target_alert_rate,
                            "champion_threshold": champion_threshold},
        "dataset_sha256": profile["sha256"], "config": asdict(config),
        "feature_reference": feature_reference,
    }
    metrics = {
        "run_type": profile["source"], "dataset": {
            "rows": profile["rows"], "fraud": profile["fraud"],
            "fraud_rate": profile["fraud_rate"], "sha256": profile["sha256"]},
        "splits": profile["splits"], "training_sample": training_sample_info,
        "champion": champion, "target_alert_rate": config.target_alert_rate,
        "validation": validation_metrics, "test": test_metrics,
        "methodology": {
            "split": "complete time-step 60/20/20 by cumulative row count",
            "calibration": "Platt scaling on natural-prevalence validation split",
            "threshold": "validation score quantile constrained by target alert rate",
            "leakage_controls": "post-transaction balances, labels, and legacy flag excluded from model features",
        },
    }
    comparison_rows = []
    for split_name, split_metrics in (("validation", validation_metrics), ("test", test_metrics)):
        for model_name, values in split_metrics.items():
            comparison_rows.append({"split": split_name, "model": model_name,
                                    **{k: v for k, v in values.items()
                                       if k not in {"top_k", "calibration"}}})
    return bundle, metrics, scored, pd.DataFrame(comparison_rows)


def save_training_outputs(bundle: dict, metrics: dict, scored: pd.DataFrame,
                          comparison: pd.DataFrame,
                          output_dir: Path = PATHS.artifacts) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_dir / "model_bundle.joblib")
    scored.to_csv(output_dir / "scored_transactions.csv", index=False)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    (output_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=PATHS.root / "config" / "modeling.yaml")
    parser.add_argument("--output-dir", type=Path, default=PATHS.artifacts)
    args = parser.parse_args()
    config = load_config(args.config)
    bundle, metrics, scored, comparison = train_official_experiment(args.data, config, args.output_dir)
    save_training_outputs(bundle, metrics, scored, comparison, args.output_dir)
    print(json.dumps({"champion": metrics["champion"], "test": metrics["test"]}, indent=2))


if __name__ == "__main__":
    main()
