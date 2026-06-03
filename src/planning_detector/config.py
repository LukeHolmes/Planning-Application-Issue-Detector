from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"


@dataclass
class ColumnConfig:
    received_date: str = "ReceivedDate"
    decision_date: str = "DecisionDate"
    planning_authority: str = "PlanningAuthority"
    application_type: str = "ApplicationType"
    decision: str = "Decision"
    development_description: str = "DevelopmentDescription"


@dataclass
class PipelineConfig:
    label_mode: str = "delay"
    threshold_days: int = 180
    feature_set: str = "early"
    model: str = "xgboost"
    test_size: float = 0.2
    random_state: int = 42
    columns: ColumnConfig = field(default_factory=ColumnConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineConfig:
        cols = data.pop("columns", {})
        columns = ColumnConfig(**cols) if cols else ColumnConfig()
        return cls(columns=columns, **data)

    @classmethod
    def load(cls, path: Path | None = None) -> PipelineConfig:
        path = path or Path(os.environ.get("PLANNING_CONFIG", DEFAULT_CONFIG_PATH))
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)


def default_data_path() -> Path:
    env = os.environ.get("PLANNING_DATA_PATH")
    if env:
        return Path(env)
    root_file = REPO_ROOT / "Tableau_Ready_Planning_Applications_With_Street_Town_Cleaned.xlsx"
    if root_file.exists():
        return root_file
    sample = REPO_ROOT / "data" / "sample" / "planning_applications_sample.csv"
    return sample
