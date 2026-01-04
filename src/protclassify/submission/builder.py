"""Submission helpers for CAFA6-style prediction exports.

The competition expects `protein_id`/`GO term`/`score` triples. This
module keeps submission generation consistent and tracked via a
lightweight manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from protclassify.paths import SUBMISSION_DIR, SUBMISSION_TRACKER


@dataclass
class SubmissionMetadata:
    attempt_number: int
    description: str
    author: str = "Book_AJ"

    def to_dict(self, filename: str) -> dict:
        return {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "author": self.author,
            "attempt_number": self.attempt_number,
            "filename": filename,
            "description": self.description,
        }


class SubmissionBuilder:
    def __init__(
        self,
        output_dir: Path = SUBMISSION_DIR,
        tracker_path: Path = SUBMISSION_TRACKER,
    ) -> None:
        self.output_dir = output_dir
        self.tracker_path = tracker_path
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tracker_path.parent.mkdir(parents=True, exist_ok=True)

    def from_predictions(
        self,
        y_pred: Iterable,
        entry_df: pd.DataFrame,
        metadata: SubmissionMetadata,
    ) -> Path:
        if "Entry" not in entry_df.columns:
            raise ValueError("Entry column missing in entry_df.")

        submission_df = pd.DataFrame({"Entry": entry_df["Entry"], "ProteinClass": list(y_pred)})
        filename = f"{metadata.author}_attempt_{metadata.attempt_number}.csv"
        save_path = self.output_dir / filename
        submission_df.to_csv(save_path, index=False)

        tracker_df = self._load_tracker()
        tracker_df = pd.concat(
            [tracker_df, pd.DataFrame([metadata.to_dict(filename)])],
            ignore_index=True,
        )
        tracker_df.to_csv(self.tracker_path, index=False)
        return save_path

    def from_model(
        self,
        model: object,
        X_eval: np.ndarray,
        label_encoder: object,
        entry_df: pd.DataFrame,
        metadata: SubmissionMetadata,
    ) -> Path:
        y_pred_encoded = model.predict(X_eval)
        y_pred_decoded = label_encoder.inverse_transform(y_pred_encoded)
        return self.from_predictions(y_pred=y_pred_decoded, entry_df=entry_df, metadata=metadata)

    def _load_tracker(self) -> pd.DataFrame:
        if self.tracker_path.exists():
            return pd.read_csv(self.tracker_path)
        return pd.DataFrame(columns=["timestamp", "author", "attempt_number", "filename", "description"])


def create_submission_from_predictions(
    y_pred: np.ndarray,
    entry_df: pd.DataFrame,
    attempt_number: int,
    description: str,
    output_dir: str | Path = SUBMISSION_DIR,
    tracker_file: Optional[str | Path] = None,
) -> Path:
    builder = SubmissionBuilder(
        output_dir=Path(output_dir),
        tracker_path=Path(tracker_file) if tracker_file else SUBMISSION_TRACKER,
    )
    metadata = SubmissionMetadata(attempt_number=attempt_number, description=description)
    return builder.from_predictions(y_pred=y_pred, entry_df=entry_df, metadata=metadata)


def create_submission_from_model(
    model: object,
    X_eval: np.ndarray,
    label_encoder: object,
    entry_df: pd.DataFrame,
    attempt_number: int,
    description: str,
    output_dir: str | Path = SUBMISSION_DIR,
    tracker_file: Optional[str | Path] = None,
) -> Path:
    builder = SubmissionBuilder(
        output_dir=Path(output_dir),
        tracker_path=Path(tracker_file) if tracker_file else SUBMISSION_TRACKER,
    )
    metadata = SubmissionMetadata(attempt_number=attempt_number, description=description)
    return builder.from_model(
        model=model,
        X_eval=X_eval,
        label_encoder=label_encoder,
        entry_df=entry_df,
        metadata=metadata,
    )


__all__ = [
    "SubmissionBuilder",
    "SubmissionMetadata",
    "create_submission_from_model",
    "create_submission_from_predictions",
]
