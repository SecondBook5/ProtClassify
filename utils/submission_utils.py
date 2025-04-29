#!/usr/bin/env python3
# File: utils/submission_utils.py
# Description: Utilities for generating submission CSV files for competition.

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Always resolve relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define student name (only modify here if needed)
STUDENT_NAME = "Book_AJ"

def create_submission_from_predictions(
    y_pred: np.ndarray,
    entry_df: pd.DataFrame,
    attempt_number: int,
    description: str,
    output_dir: str = "submission",
    tracker_file: str = "submission/submission_tracker.csv"
) -> None:
    """
    Create a submission CSV from predictions and Entry metadata,
    and update a manifest/tracker file.

    Args:
        y_pred (np.ndarray): Predicted labels (decoded into string classes).
        entry_df (pd.DataFrame): DataFrame containing the 'Entry' column.
        attempt_number (int): Attempt number (e.g., 1, 2, 3).
        description (str): Short description of what this attempt contains.
        output_dir (str): Directory to save the submission files.
        tracker_file (str): Tracker CSV manifest to update.

    Raises:
        Exception: If saving or tracking fails.
    """
    try:
        output_dir_path = PROJECT_ROOT / output_dir
        tracker_path = PROJECT_ROOT / tracker_file

        os.makedirs(output_dir_path, exist_ok=True)

        if "Entry" not in entry_df.columns:
            raise ValueError("Entry column missing in entry_df.")

        submission_df = pd.DataFrame({
            "Entry": entry_df["Entry"],
            "ProteinClass": y_pred
        })

        submission_name = f"{STUDENT_NAME}_attempt_{attempt_number}.csv"
        output_path = output_dir_path / submission_name
        submission_df.to_csv(output_path, index=False)

        tracker_columns = ["timestamp", "student_name", "attempt_number", "filename", "description"]

        if tracker_path.exists():
            tracker_df = pd.read_csv(tracker_path)
        else:
            tracker_df = pd.DataFrame(columns=tracker_columns)

        new_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "student_name": STUDENT_NAME,
            "attempt_number": attempt_number,
            "filename": submission_name,
            "description": description
        }

        if not tracker_df.empty:
            tracker_df = pd.concat([tracker_df, pd.DataFrame([new_entry])], ignore_index=True)
        else:
            tracker_df = pd.DataFrame([new_entry])

        tracker_df.to_csv(tracker_path, index=False)

        print(f"Submission saved to {output_path}")
        print(f"Submission tracker updated at {tracker_path}")

    except Exception as e:
        raise Exception(f"Failed to create submission: {str(e)}")

def create_submission_from_model(
    model: object,
    X_eval: np.ndarray,
    label_encoder: object,
    entry_df: pd.DataFrame,
    attempt_number: int,
    description: str,
    output_dir: str = "submission",
    tracker_file: str = "submission/submission_tracker.csv"
) -> None:
    """
    Create a submission CSV directly from a trained model and evaluation features.

    Args:
        model (object): Trained model with a `.predict()` method.
        X_eval (np.ndarray): Evaluation feature matrix (already scaled if needed).
        label_encoder (LabelEncoder): Fitted label encoder for decoding predictions.
        entry_df (pd.DataFrame): DataFrame containing the 'Entry' column.
        attempt_number (int): Attempt number (e.g., 1, 2, 3).
        description (str): Short description of what this attempt contains.
        output_dir (str): Directory to save the submission files.
        tracker_file (str): Tracker CSV manifest to update.

    Raises:
        Exception: If saving or tracking fails.
    """
    try:
        # Predict and decode
        y_pred_encoded = model.predict(X_eval)
        y_pred_decoded = label_encoder.inverse_transform(y_pred_encoded)

        # Call the other function
        create_submission_from_predictions(
            y_pred=y_pred_decoded,
            entry_df=entry_df,
            attempt_number=attempt_number,
            description=description,
            output_dir=output_dir,
            tracker_file=tracker_file
        )

    except Exception as e:
        raise Exception(f"Failed during submission creation: {str(e)}")
