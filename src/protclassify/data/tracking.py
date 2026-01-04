"""Data tracking utilities for reproducible preprocessing artifacts.

This module standardizes how intermediate arrays are stored and
retrieved. Each saved artifact is recorded in a manifest CSV so we can
trace which feature set, split, and version produced a particular file.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from protclassify.paths import DATA_TRACKER, PROCESSED_DATA_DIR


@dataclass
class ArrayMetadata:
    """Metadata describing a stored NumPy artifact."""

    name: str
    featureset: str
    split: str
    version: str
    description: str

    def to_dict(self, file_name: str, array: np.ndarray) -> dict:
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_name": file_name,
            "name": self.name,
            "featureset": self.featureset,
            "split": self.split,
            "version": self.version,
            "description": self.description,
            "shape": str(array.shape),
        }


class DataTracker:
    """Manage saving/loading intermediate arrays with a manifest."""

    def __init__(
        self,
        output_dir: Path = PROCESSED_DATA_DIR,
        tracker_path: Path = DATA_TRACKER,
    ) -> None:
        self.output_dir = output_dir
        self.tracker_path = tracker_path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tracker_path.parent.mkdir(parents=True, exist_ok=True)

    def save_array(self, array: np.ndarray, metadata: ArrayMetadata) -> Path:
        """Persist an array and log it to the tracker."""
        filename = f"{metadata.name}_{metadata.featureset}_{metadata.split}_{metadata.version}.npy"
        save_path = self.output_dir / filename
        np.save(save_path, array)

        tracker_df = self._load_tracker()
        tracker_df = pd.concat(
            [tracker_df, pd.DataFrame([metadata.to_dict(filename, array)])],
            ignore_index=True,
        )
        tracker_df.to_csv(self.tracker_path, index=False)
        return save_path

    def load_array(self, file_path: Path) -> np.ndarray:
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        return np.load(file_path)

    def find_array(
        self,
        name: str,
        featureset: str,
        split: str,
        version: str,
    ) -> np.ndarray:
        tracker_df = self._load_tracker()
        match = tracker_df[
            (tracker_df["name"] == name)
            & (tracker_df["featureset"] == featureset)
            & (tracker_df["split"] == split)
            & (tracker_df["version"] == version)
        ]
        if match.empty:
            raise ValueError(
                f"No matching array found for name={name}, featureset={featureset}, split={split}, version={version}."
            )
        file_path = self.output_dir / match.iloc[0]["file_name"]
        return self.load_array(file_path)

    def _load_tracker(self) -> pd.DataFrame:
        if self.tracker_path.exists():
            return pd.read_csv(self.tracker_path)
        return pd.DataFrame(
            columns=[
                "timestamp",
                "file_name",
                "name",
                "featureset",
                "split",
                "version",
                "description",
                "shape",
            ]
        )


def load_npy(file_path: str | Path) -> np.ndarray:
    """Backwards-compatible convenience wrapper."""
    return np.load(Path(file_path))


def load_npy_from_tracker(
    name: str,
    featureset: str,
    split: str,
    version: str,
    tracker_file: Optional[str | Path] = None,
) -> np.ndarray:
    tracker = DataTracker(
        tracker_path=Path(tracker_file) if tracker_file else DATA_TRACKER
    )
    return tracker.find_array(name=name, featureset=featureset, split=split, version=version)


def save_npy(
    array: np.ndarray,
    name: str,
    description: str,
    featureset: str,
    split: str,
    version: str,
    output_dir: str | Path = PROCESSED_DATA_DIR,
    tracker_file: Optional[str | Path] = None,
) -> Path:
    tracker = DataTracker(
        output_dir=Path(output_dir),
        tracker_path=Path(tracker_file) if tracker_file else DATA_TRACKER,
    )
    metadata = ArrayMetadata(
        name=name,
        featureset=featureset,
        split=split,
        version=version,
        description=description,
    )
    return tracker.save_array(array, metadata)


__all__ = [
    "ArrayMetadata",
    "DataTracker",
    "load_npy",
    "load_npy_from_tracker",
    "save_npy",
]
