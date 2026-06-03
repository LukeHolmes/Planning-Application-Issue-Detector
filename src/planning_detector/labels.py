from __future__ import annotations

import pandas as pd

from planning_detector.config import PipelineConfig
from planning_detector.features import PROCESSING_TIME_COL, add_processing_time

TARGET_COL = "problematic"


def build_labels(df: pd.DataFrame, config: PipelineConfig) -> pd.Series:
    """Create binary problematic label according to config.label_mode."""
    work = add_processing_time(df, config)
    decision_col = config.columns.decision

    delay = work[PROCESSING_TIME_COL] > config.threshold_days
    refusal = pd.Series(False, index=work.index)
    if decision_col in work.columns:
        refusal = work[decision_col].astype(str).str.strip().str.lower() == "refused"

    mode = config.label_mode
    if mode == "delay":
        y = delay
    elif mode == "refusal":
        y = refusal
    elif mode == "either":
        y = delay | refusal
    else:
        raise ValueError(f"Unknown label_mode: {mode}")

    return y.astype(int).rename(TARGET_COL)
