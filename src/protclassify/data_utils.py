#!/usr/bin/env python3
# File: data_utils.py
# Description: Utility functions for saving, loading, and tracking NumPy dataset files.

from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from protclassify import paths

PathLike = Union[str, Path]


def _resolve(path: PathLike) -> Path:
    """Resolve a path relative to the project root."""
    path_obj = Path(path)
    return paths.resolve_path(path_obj)


def load_npy(file_path: PathLike) -> np.ndarray:
    """
    Load a saved NumPy .npy array from disk.

    Args:
        file_path: Relative or absolute path to the .npy file.

    Returns:
        np.ndarray: Loaded array.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_path = _resolve(file_path)
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
    tracker_file: PathLike = paths.DATA_TRACKER,
) -> np.ndarray:
    """
    Load a NumPy array based on tracker metadata.

    Args:
        name: Array name ('X_train', 'X_test', etc.).
        featureset: Featureset used.
        split: Data split ('train', 'test', etc.).
        version: Version identifier.
        tracker_file: Path to the tracker CSV.

    Returns:
        np.ndarray: Loaded array.

    Raises:
        ValueError: If no matching record is found.
    """
    tracker_path = _resolve(tracker_file)
    if not tracker_path.exists():
        raise FileNotFoundError(f"Tracker file not found: {tracker_path}")

    tracker_df = pd.read_csv(tracker_path)

    match = tracker_df[
        (tracker_df["name"] == name)
        & (tracker_df["featureset"] == featureset)
        & (tracker_df["split"] == split)
        & (tracker_df["version"] == version)
    ]

    if match.empty:
        raise ValueError(
            f"No matching array found for {name}, {featureset}, {split}, {version}."
        )

    file_path = tracker_path.parent / match.iloc[0]["file_name"]
    return load_npy(file_path)


def save_npy(
    array: np.ndarray,
    name: str,
    description: str,
    featureset: str,
    split: str,
    version: str,
    output_dir: PathLike = paths.PROCESSED_DATA_DIR,
    tracker_file: PathLike = paths.DATA_TRACKER,
) -> None:
    """
    Save a NumPy array with standardized naming, and update a tracker CSV.

    Args:
        array: The NumPy array to save.
        name: Descriptive name (e.g., 'X_train', 'X_test', 'X_eval').
        description: Brief description of what the array represents.
        featureset: Featureset used ('full', 'pca', 'lasso', etc.).
        split: Data split ('train', 'test', 'eval', 'all').
        version: Version identifier (e.g., 'v1', 'v2').
        output_dir: Directory where the .npy file will be saved.
        tracker_file: Path to the data tracker CSV file.

    Raises:
        Exception: If saving or tracking fails.
    """
    try:
        output_dir_path = _resolve(output_dir)
        tracker_path = _resolve(tracker_file)

        paths.ensure_dir(output_dir_path)
        paths.ensure_dir(tracker_path.parent)

        filename = f"{name}_{featureset}_{split}_{version}.npy"
        save_path = output_dir_path / filename

        np.save(save_path, array)

        tracker_columns = [
            "timestamp",
            "file_name",
            "name",
            "featureset",
            "split",
            "version",
            "description",
            "shape",
        ]

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
            "shape": str(array.shape),
        }

        if not tracker_df.empty:
            tracker_df = pd.concat(
                [tracker_df, pd.DataFrame([new_entry])], ignore_index=True
            )
        else:
            tracker_df = pd.DataFrame([new_entry])

        tracker_df.to_csv(tracker_path, index=False)

        print(f"Array successfully saved to {save_path}")
        print(f"Data tracker updated at {tracker_path}")

    except Exception as e:
        raise Exception(f"Error occurred while saving array: {str(e)}")
