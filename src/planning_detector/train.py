from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from planning_detector.config import REPO_ROOT, PipelineConfig, default_data_path
from planning_detector.features import prepare_features
from planning_detector.labels import build_labels
from planning_detector.load import load_planning_data
from planning_detector.metrics import EvaluationResult, evaluate_classifier

MODELS_DIR = REPO_ROOT / "models"
ARTIFACT_NAME = "planning_detector.joblib"
METRICS_NAME = "metrics.json"


def create_estimator(model_name: str) -> Pipeline:
    if model_name == "logistic":
        clf = LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == "random_forest":
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_name == "xgboost":
        clf = XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            n_estimators=100,
            max_depth=4,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


def align_xy(
    df: pd.DataFrame, config: PipelineConfig
) -> tuple[pd.DataFrame, pd.Series, list[str], dict]:
    y = build_labels(df, config)
    X, feature_names, encoders = prepare_features(df, config, fit_encoders=True)
    y = y.loc[X.index]
    return X, y, feature_names, encoders


def train_pipeline(
    data_path: Path | None = None,
    config: PipelineConfig | None = None,
    output_dir: Path | None = None,
) -> tuple[Pipeline, EvaluationResult, dict]:
    config = config or PipelineConfig.load()
    data_path = data_path or default_data_path()
    output_dir = output_dir or MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_planning_data(data_path, config)
    X, y, feature_names, encoders = align_xy(df, config)

    if len(X) < 20:
        raise ValueError(f"Not enough rows after preprocessing ({len(X)}). Check data path.")

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify,
    )

    pipeline = create_estimator(config.model)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = None
    if hasattr(pipeline.named_steps["clf"], "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = evaluate_classifier(y_test, y_pred, y_proba)

    artifact = {
        "pipeline": pipeline,
        "config": asdict(config),
        "feature_names": feature_names,
        "encoders": encoders,
        "label_mode": config.label_mode,
        "feature_set": config.feature_set,
    }

    joblib.dump(artifact, output_dir / ARTIFACT_NAME)
    meta = {
        "metrics": metrics.to_dict(),
        "feature_names": feature_names,
        "label_mode": config.label_mode,
        "feature_set": config.feature_set,
        "model": config.model,
        "data_path": str(data_path),
        "n_rows": len(X),
    }
    with open(output_dir / METRICS_NAME, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return pipeline, metrics, meta
