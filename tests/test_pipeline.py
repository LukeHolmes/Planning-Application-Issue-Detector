from pathlib import Path

import joblib
import pandas as pd
import pytest

from planning_detector.config import REPO_ROOT, PipelineConfig
from planning_detector.features import prepare_features
from planning_detector.labels import build_labels
from planning_detector.load import load_planning_data
from planning_detector.predict import score_dataframe
from planning_detector.train import train_pipeline

SAMPLE = REPO_ROOT / "data" / "sample" / "planning_applications_sample.csv"


@pytest.fixture
def config() -> PipelineConfig:
    return PipelineConfig.load()


@pytest.fixture
def sample_df(config: PipelineConfig) -> pd.DataFrame:
    return load_planning_data(SAMPLE, config)


def test_sample_loads(sample_df: pd.DataFrame) -> None:
    assert len(sample_df) >= 50
    assert "ReceivedDate" in sample_df.columns


def test_labels_in_range(sample_df: pd.DataFrame, config: PipelineConfig) -> None:
    y = build_labels(sample_df, config)
    assert set(y.unique()).issubset({0, 1})


def test_early_features_no_processing_time(sample_df: pd.DataFrame, config: PipelineConfig) -> None:
    config.feature_set = "early"
    X, names, _ = prepare_features(sample_df, config)
    assert "processing_time_days" not in names
    assert len(X) == len(sample_df)


def test_train_and_score(
    tmp_path: Path, config: PipelineConfig, sample_df: pd.DataFrame
) -> None:
    out = tmp_path / "models"
    _, metrics, _ = train_pipeline(SAMPLE, config, out)
    assert 0 <= metrics.accuracy <= 1
    artifact = joblib.load(out / "planning_detector.joblib")
    scored = score_dataframe(sample_df, artifact, config)
    assert "predicted_problematic" in scored.columns
    assert "problematic_probability" in scored.columns
