"""Model registry utilities with manifest tracking.

This module centralizes how trained models are persisted and recalled.
Every save operation logs contextual metadata (feature set, optimizer,
metric, version) to a CSV tracker for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd

from protclassify.paths import MODEL_TRACKER, MODELS_DIR


@dataclass
class ModelMetadata:
    model_type: str
    featureset: str
    optimizer: str
    scoremetric: str
    version: str
    accuracy: float

    def to_dict(self, model_file: str) -> dict:
        return {
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_file": model_file,
            "model_type": self.model_type,
            "featureset": self.featureset,
            "optimizer": self.optimizer,
            "scoremetric": self.scoremetric,
            "version": self.version,
            "accuracy": round(self.accuracy, 4),
        }


class ModelRegistry:
    """Save and recall trained models with manifest logging."""

    def __init__(self, output_dir: Path = MODELS_DIR, tracker_path: Path = MODEL_TRACKER) -> None:
        self.output_dir = output_dir
        self.tracker_path = tracker_path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tracker_path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, model: object, metadata: ModelMetadata) -> Path:
        filename = f"{metadata.model_type}_{metadata.featureset}_{metadata.optimizer}_{metadata.scoremetric}_{metadata.version}.joblib"
        save_path = self.output_dir / filename
        joblib.dump(model, save_path)

        tracker_df = self._load_tracker()
        tracker_df = pd.concat(
            [tracker_df, pd.DataFrame([metadata.to_dict(filename)])],
            ignore_index=True,
        )
        tracker_df.to_csv(self.tracker_path, index=False)
        return save_path

    def load(self, model_path: Path) -> object:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return joblib.load(model_path)

    def load_best(
        self,
        metric: str = "accuracy",
        maximize: bool = True,
    ) -> object:
        tracker_df = self._load_tracker()
        if tracker_df.empty:
            raise FileNotFoundError("Model tracker is empty—no models to load.")

        idx = tracker_df[metric].idxmax() if maximize else tracker_df[metric].idxmin()
        best_row = tracker_df.loc[idx]
        model_path = self.output_dir / best_row["model_file"]
        return self.load(model_path)

    def _load_tracker(self) -> pd.DataFrame:
        if self.tracker_path.exists():
            return pd.read_csv(self.tracker_path)
        return pd.DataFrame(
            columns=[
                "timestamp",
                "model_file",
                "model_type",
                "featureset",
                "optimizer",
                "scoremetric",
                "version",
                "accuracy",
            ]
        )


def load_model(model_path: str | Path) -> object:
    registry = ModelRegistry()
    return registry.load(Path(model_path))


def load_best_model_from_tracker(
    tracker_file: str | Path = MODEL_TRACKER,
    metric: str = "accuracy",
    maximize: bool = True,
) -> object:
    registry = ModelRegistry(tracker_path=Path(tracker_file))
    return registry.load_best(metric=metric, maximize=maximize)


def save_model(
    model: object,
    model_type: str,
    featureset: str,
    optimizer: str,
    scoremetric: str,
    version: str,
    accuracy: float,
    output_dir: str | Path = MODELS_DIR,
    tracker_file: Optional[str | Path] = None,
) -> Path:
    registry = ModelRegistry(
        output_dir=Path(output_dir),
        tracker_path=Path(tracker_file) if tracker_file else MODEL_TRACKER,
    )
    metadata = ModelMetadata(
        model_type=model_type,
        featureset=featureset,
        optimizer=optimizer,
        scoremetric=scoremetric,
        version=version,
        accuracy=accuracy,
    )
    return registry.save(model, metadata)


__all__ = [
    "ModelMetadata",
    "ModelRegistry",
    "load_model",
    "load_best_model_from_tracker",
    "save_model",
]
