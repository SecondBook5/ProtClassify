#!/usr/bin/env python3
# File: submission_utils.py
# Description: Utilities for generating submission CSV files for competition outputs.

from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from protclassify import paths

RUNNER_NAME = "Book_AJ"
PathLike = Union[str, Path]


def _resolve(path: PathLike) -> Path:
    """Resolve a path relative to the project root."""
    path_obj = Path(path)
    return paths.resolve_path(path_obj)


def create_submission_from_predictions(
    y_pred: np.ndarray,
    entry_df: pd.DataFrame,
    attempt_number: int,
    description: str,
    output_dir: PathLike = paths.SUBMISSIONS_DIR,
    tracker_file: PathLike = paths.SUBMISSION_TRACKER,
    id_column: str = "Entry",
    label_column: str = "ProteinClass",
) -> Path:
    """
    Create a submission CSV from predictions and Entry metadata,
    and update a manifest/tracker file.

    Args:
        y_pred: Predicted labels (decoded into string classes).
        entry_df: DataFrame containing an ID column.
        attempt_number: Attempt number (e.g., 1, 2, 3).
        description: Short description of what this attempt contains.
        output_dir: Directory to save the submission files.
        tracker_file: Tracker CSV manifest to update.
        id_column: Name of the identifier column.
        label_column: Name of the prediction column.

    Returns:
        Path to the written submission CSV.

    Raises:
        Exception: If saving or tracking fails.
    """
    try:
        output_dir_path = _resolve(output_dir)
        tracker_path = _resolve(tracker_file)

        paths.ensure_dir(output_dir_path)
        paths.ensure_dir(tracker_path.parent)

        if id_column not in entry_df.columns:
            raise ValueError(f"{id_column} column missing in entry_df.")

        submission_df = pd.DataFrame(
            {
                id_column: entry_df[id_column],
                label_column: y_pred,
            }
        )

        submission_name = f"{RUNNER_NAME}_attempt_{attempt_number}.csv"
        output_path = output_dir_path / submission_name
        submission_df.to_csv(output_path, index=False)

        tracker_columns = [
            "timestamp",
            "runner",
            "attempt_number",
            "filename",
            "description",
            "id_column",
            "label_column",
        ]

        if tracker_path.exists():
            tracker_df = pd.read_csv(tracker_path)
        else:
            tracker_df = pd.DataFrame(columns=tracker_columns)

        new_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "runner": RUNNER_NAME,
            "attempt_number": attempt_number,
            "filename": submission_name,
            "description": description,
            "id_column": id_column,
            "label_column": label_column,
        }

        if not tracker_df.empty:
            tracker_df = pd.concat(
                [tracker_df, pd.DataFrame([new_entry])], ignore_index=True
            )
        else:
            tracker_df = pd.DataFrame([new_entry])

        tracker_df.to_csv(tracker_path, index=False)

        print(f"Submission saved to {output_path}")
        print(f"Submission tracker updated at {tracker_path}")

        return output_path

    except Exception as e:
        raise Exception(f"Failed to create submission: {str(e)}")


def create_submission_from_model(
    model: object,
    X_eval: np.ndarray,
    label_encoder: object,
    entry_df: pd.DataFrame,
    attempt_number: int,
    description: str,
    output_dir: PathLike = paths.SUBMISSIONS_DIR,
    tracker_file: PathLike = paths.SUBMISSION_TRACKER,
    id_column: str = "Entry",
    label_column: str = "ProteinClass",
) -> Path:
    """
    Create a submission CSV directly from a trained model and evaluation features.

    Args:
        model: Trained model with a `.predict()` method.
        X_eval: Evaluation feature matrix (already scaled if needed).
        label_encoder: Fitted label encoder for decoding predictions.
        entry_df: DataFrame containing the identifier column.
        attempt_number: Attempt number (e.g., 1, 2, 3).
        description: Short description of what this attempt contains.
        output_dir: Directory to save the submission files.
        tracker_file: Tracker CSV manifest to update.
        id_column: Name of the identifier column.
        label_column: Name of the prediction column.

    Returns:
        Path to the written submission CSV.

    Raises:
        Exception: If saving or tracking fails.
    """
    try:
        y_pred_encoded = model.predict(X_eval)
        y_pred_decoded = label_encoder.inverse_transform(y_pred_encoded)

        return create_submission_from_predictions(
            y_pred=y_pred_decoded,
            entry_df=entry_df,
            attempt_number=attempt_number,
            description=description,
            output_dir=output_dir,
            tracker_file=tracker_file,
            id_column=id_column,
            label_column=label_column,
        )

    except Exception as e:
        raise Exception(f"Failed during submission creation: {str(e)}")
