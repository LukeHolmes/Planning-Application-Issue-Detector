from __future__ import annotations

import pandas as pd

from planning_detector.config import PipelineConfig

AUTHORITY_COL = "planning_authority"
APP_TYPE_COL = "application_type"
PROCESSING_TIME_COL = "processing_time_days"
RECEIVED_COL = "received_date"
DECISION_COL = "decision_date"
UNKNOWN = "Unknown"


def add_processing_time(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    out = df.copy()
    rcv = config.columns.received_date
    dec = config.columns.decision_date
    out[RECEIVED_COL] = pd.to_datetime(out[rcv], errors="coerce")
    out[DECISION_COL] = pd.to_datetime(out[dec], errors="coerce")
    out[PROCESSING_TIME_COL] = (out[DECISION_COL] - out[RECEIVED_COL]).dt.days
    return out


def _encode_categorical(
    series: pd.Series,
    mapping: dict[str, int] | None,
    fit: bool,
) -> tuple[pd.Series, dict[str, int]]:
    cleaned = series.fillna(UNKNOWN).astype(str)
    if fit or mapping is None:
        categories = sorted(cleaned.unique())
        if UNKNOWN not in categories:
            categories.append(UNKNOWN)
        mapping = {cat: idx for idx, cat in enumerate(categories)}
    fallback = mapping.get(UNKNOWN, len(mapping))
    encoded = cleaned.map(lambda v: mapping.get(v, fallback))
    return encoded.astype(float), mapping


def prepare_features(
    df: pd.DataFrame,
    config: PipelineConfig,
    encoders: dict[str, dict[str, int]] | None = None,
    fit_encoders: bool = True,
) -> tuple[pd.DataFrame, list[str], dict[str, dict[str, int]]]:
    """Build feature matrix and return (X, feature_names, encoders)."""
    c = config.columns
    work = add_processing_time(df, config)

    encoders = encoders or {}
    feature_names: list[str] = []

    for name, col in [(AUTHORITY_COL, c.planning_authority), (APP_TYPE_COL, c.application_type)]:
        encoded, mapping = _encode_categorical(
            work[col],
            encoders.get(name),
            fit=fit_encoders,
        )
        work[f"{name}_encoded"] = encoded
        encoders[name] = mapping
        feature_names.append(f"{name}_encoded")

    if config.feature_set == "full":
        work = work.dropna(subset=[PROCESSING_TIME_COL])
        feature_names.append(PROCESSING_TIME_COL)
    elif config.feature_set != "early":
        raise ValueError(f"Unknown feature_set: {config.feature_set}")

    X = work[feature_names].astype(float)
    return X, feature_names, encoders
