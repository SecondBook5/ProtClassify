#!/usr/bin/env python3
# File: data_utils.py
# Description: Utility functions for saving, loading, and tracking NumPy dataset files.

import os
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Always resolve relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_npy(file_path: str) -> np.ndarray:
    """
    Load a saved NumPy .npy array from disk.

    Args:
        file_path (str): Relative path to the .npy file.

    Returns:
        np.ndarray: Loaded array.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_path = PROJECT_ROOT / file_path
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    array = np.load(file_path)
    print(f"Array successfully loaded from {file_path}")
    return array

def load_npy_from_tracker(
    name: str,
    featureset: str,
    split: str,
    version: str,
    tracker_file: str = "processed_data/data_tracker.csv"
) -> np.ndarray:
    """
    Load a NumPy array based on tracker metadata.

    Args:
        name (str): Array name ('X_train', 'X_test', etc.).
        featureset (str): Featureset used.
        split (str): Data split ('train', 'test', etc.).
        version (str): Version identifier.
        tracker_file (str): Path to the tracker CSV.

    Returns:
        np.ndarray: Loaded array.

    Raises:
        ValueError: If no matching record is found.
    """
    tracker_path = PROJECT_ROOT / tracker_file
    if not tracker_path.exists():
        raise FileNotFoundError(f"Tracker file not found: {tracker_path}")

    tracker_df = pd.read_csv(tracker_path)

    match = tracker_df[
        (tracker_df["name"] == name) &
        (tracker_df["featureset"] == featureset) &
        (tracker_df["split"] == split) &
        (tracker_df["version"] == version)
    ]

    if match.empty:
        raise ValueError(f"No matching array found for {name}, {featureset}, {split}, {version}.")

    file_path = tracker_path.parent / match.iloc[0]["file_name"]
    return load_npy(str(file_path))

def save_npy(
    array: np.ndarray,
    name: str,
    description: str,
    featureset: str,
    split: str,
    version: str,
    output_dir: str = "processed_data",
    tracker_file: str = "processed_data/data_tracker.csv"
) -> None:
    """
    Save a NumPy array with standardized naming, and update a tracker CSV.

    Args:
        array (np.ndarray): The NumPy array to save.
        name (str): Descriptive name (e.g., 'X_train', 'X_test', 'X_eval').
        description (str): Brief description of what the array represents.
        featureset (str): Featureset used ('full', 'pca', 'lasso', etc.).
        split (str): Data split ('train', 'test', 'eval', 'all').
        version (str): Version identifier (e.g., 'v1', 'v2').
        output_dir (str): Directory where the .npy file will be saved.
        tracker_file (str): Path to the data tracker CSV file.

    Raises:
        Exception: If saving or tracking fails.
    """
    try:
        output_dir_path = PROJECT_ROOT / output_dir
        tracker_path = PROJECT_ROOT / tracker_file

        # Check if output directory exists — do NOT create it
        if not output_dir_path.exists():
            raise FileNotFoundError(f"Output directory does not exist: {output_dir_path}")

        filename = f"{name}_{featureset}_{split}_{version}.npy"
        save_path = output_dir_path / filename

        np.save(save_path, array)

        tracker_columns = ["timestamp", "file_name", "name", "featureset", "split", "version", "description", "shape"]

        if tracker_path.exists():
            tracker_df = pd.read_csv(tracker_path)
        else:
            tracker_df = pd.DataFrame(columns=tracker_columns)

        new_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "file_name": filename,
            "name": name,
            "featureset": featureset,
            "split": split,
            "version": version,
            "description": description,
            "shape": str(array.shape)
        }

        if not tracker_df.empty:
            tracker_df = pd.concat([tracker_df, pd.DataFrame([new_entry])], ignore_index=True)
        else:
            tracker_df = pd.DataFrame([new_entry])

        tracker_df.to_csv(tracker_path, index=False)

        print(f"Array successfully saved to {save_path}")
        print(f"Data tracker updated at {tracker_path}")

    except Exception as e:
        raise Exception(f"Error occurred while saving array: {str(e)}")
