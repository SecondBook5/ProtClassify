"""Lightweight baseline models suitable for CAFA6 experiments."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.pipeline import Pipeline

from protclassify.paths import MODEL_TRACKER, MODELS_DIR


@dataclass
class TrainingResult:
    """Container for model, label encoder, and validation metrics."""

    model: Pipeline
    label_binarizer: MultiLabelBinarizer
    metrics: dict[str, float]
    model_path: Path


@dataclass
class ModelMetadata:
    name: str
    featureset: str
    description: str
    version: str = "v1"

    def to_dict(self, path: Path, metrics: dict[str, float]) -> dict[str, str]:
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "file_name": path.name,
            "name": self.name,
            "featureset": self.featureset,
            "version": self.version,
            "description": self.description,
            **{f"metric_{k}": v for k, v in metrics.items()},
        }


def _ensure_model_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def train_logreg_baseline(
    X: pd.DataFrame | np.ndarray,
    labels: Iterable[Iterable[str]],
    *,
    test_size: float = 0.1,
    random_state: int = 42,
    C: float = 4.0,
    max_iter: int = 200,
    n_jobs: int = -1,
    metadata: ModelMetadata | None = None,
) -> TrainingResult:
    """Train a simple logistic regression one-vs-rest baseline.

    This intentionally avoids heavy dependencies while producing a strong
    baseline for CAFA-style multilabel classification.
    """

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(labels)

    label_sums = Y.sum(axis=1)
    stratify_target = label_sums if len(np.unique(label_sums)) > 1 else None

    X_train, X_val, y_train, y_val = train_test_split(
        X, Y, test_size=test_size, random_state=random_state, stratify=stratify_target
    )

    estimator = OneVsRestClassifier(
        LogisticRegression(
            C=C,
            max_iter=max_iter,
            n_jobs=n_jobs,
            solver="lbfgs",
            class_weight="balanced",
        )
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", estimator),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    f1_micro = float(f1_score(y_val, y_pred, average="micro"))
    f1_macro = float(f1_score(y_val, y_pred, average="macro"))

    model_path = MODELS_DIR / f"{metadata.name if metadata else 'logreg_baseline'}_{metadata.version if metadata else 'v1'}.joblib"
    _ensure_model_dir(model_path)
    payload = {"model": pipeline, "label_binarizer": mlb, "metrics": {"f1_micro": f1_micro, "f1_macro": f1_macro}}
    joblib.dump(payload, model_path)

    metrics = {"f1_micro": f1_micro, "f1_macro": f1_macro}
    if metadata:
        _update_model_tracker(model_path, metadata, metrics)

    return TrainingResult(
        model=pipeline,
        label_binarizer=mlb,
        metrics=metrics,
        model_path=model_path,
    )


def predict_proba(model: Pipeline, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Return probability estimates for each GO term."""

    clf = model.named_steps["clf"]
    return clf.predict_proba(model.named_steps["scaler"].transform(X))


def _update_model_tracker(path: Path, metadata: ModelMetadata, metrics: dict[str, float]) -> None:
    tracker_df = (
        pd.read_csv(MODEL_TRACKER)
        if MODEL_TRACKER.exists()
        else pd.DataFrame(columns=["timestamp", "file_name", "name", "featureset", "version", "description"])
    )
    new_row = metadata.to_dict(path, metrics)
    tracker_df = pd.concat([tracker_df, pd.DataFrame([new_row])], ignore_index=True)
    MODEL_TRACKER.parent.mkdir(parents=True, exist_ok=True)
    tracker_df.to_csv(MODEL_TRACKER, index=False)


__all__ = [
    "ModelMetadata",
    "TrainingResult",
    "predict_proba",
    "train_logreg_baseline",
]
