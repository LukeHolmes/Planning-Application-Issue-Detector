from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from planning_detector.config import PipelineConfig
from planning_detector.features import prepare_features
from planning_detector.labels import TARGET_COL
from planning_detector.load import load_planning_data
from planning_detector.train import ARTIFACT_NAME, MODELS_DIR

PROB_COL = "problematic_probability"
PRED_COL = "predicted_problematic"


def load_artifact(path: Path | None = None) -> dict:
    path = path or (MODELS_DIR / ARTIFACT_NAME)
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model at {path}. Run: planning-train"
        )
    return joblib.load(path)


def score_dataframe(
    df: pd.DataFrame,
    artifact: dict | None = None,
    config: PipelineConfig | None = None,
) -> pd.DataFrame:
    artifact = artifact or load_artifact()
    config = config or PipelineConfig.load()

    encoders = artifact["encoders"]
    feature_names = artifact["feature_names"]

    X, _, _ = prepare_features(
        df,
        config,
        encoders=encoders,
        fit_encoders=False,
    )
    # Align to training feature columns
    for col in feature_names:
        if col not in X.columns:
            raise ValueError(f"Missing feature column {col} after preprocessing")

    X = X[feature_names]
    pipeline = artifact["pipeline"]
    out = df.copy()
    out[PRED_COL] = pipeline.predict(X)
    if hasattr(pipeline.named_steps["clf"], "predict_proba"):
        out[PROB_COL] = pipeline.predict_proba(X)[:, 1]
    else:
        out[PROB_COL] = out[PRED_COL].astype(float)

    if config.columns.decision in df.columns:
        from planning_detector.labels import build_labels

        out[f"actual_{TARGET_COL}"] = build_labels(df, config).values

    return out


def score_file(path: Path, artifact_path: Path | None = None) -> pd.DataFrame:
    config = PipelineConfig.load()
    df = load_planning_data(path, config)
    return score_dataframe(df, load_artifact(artifact_path), config)
