from __future__ import annotations

from pathlib import Path

import pandas as pd

from planning_detector.config import PipelineConfig


def load_planning_data(path: Path, config: PipelineConfig | None = None) -> pd.DataFrame:
    """Load planning applications from Excel or CSV."""
    config = config or PipelineConfig.load()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, engine="openpyxl")
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")

    return normalize_columns(df, config)


def normalize_columns(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Map filtered notebook column names to canonical names."""
    c = config.columns
    rename_map = {
        "ReceivedDate_filtered": c.received_date,
        "DecisionDate_filtered": c.decision_date,
        "PlanningAuthority_filtered": c.planning_authority,
        "ApplicationType_filtered": c.application_type,
        "Decision_filtered": c.decision,
    }
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    if existing:
        df = df.rename(columns=existing)
    return df
