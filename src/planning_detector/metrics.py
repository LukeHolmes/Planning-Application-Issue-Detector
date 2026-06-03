from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    positive_rate: float
    n_test: int

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_classifier(y_true, y_pred, y_proba=None) -> EvaluationResult:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    roc = None
    if y_proba is not None and len(np.unique(y_true)) > 1:
        roc = float(roc_auc_score(y_true, y_proba))

    return EvaluationResult(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc,
        positive_rate=float(y_true.mean()),
        n_test=len(y_true),
    )
