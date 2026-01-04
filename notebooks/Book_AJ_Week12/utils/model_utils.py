#!/usr/bin/env python3
# File: model_utils.py
# Description: Utility functions for saving, loading, and tracking machine learning models.

import os
import joblib
import pandas as pd
from pathlib import Path

# Always resolve relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_model(model_path: str) -> object:
    """
    Load a saved model from disk.

    Args:
        model_path (str): Relative path to the saved model `.joblib` file.

    Returns:
        object: Loaded model object.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    model_path = PROJECT_ROOT / model_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    print(f"Model successfully loaded from {model_path}")
    return model

def load_best_model_from_tracker(
    tracker_file: str = "models/model_tracker.csv",
    metric: str = "accuracy",
    maximize: bool = True
) -> object:
    """
    Load the best-performing model from the model tracker based on a selected metric.

    Args:
        tracker_file (str): Path to the model tracker CSV file.
        metric (str): Metric used for model selection (default is "accuracy").
        maximize (bool): Whether to maximize (True) or minimize (False) the metric.

    Returns:
        object: Best model loaded from disk.

    Raises:
        Exception: If the tracker file does not exist or contains no models.
    """
    tracker_path = PROJECT_ROOT / tracker_file
    if not tracker_path.exists():
        raise FileNotFoundError(f"Tracker file not found: {tracker_path}")

    tracker_df = pd.read_csv(tracker_path)

    if tracker_df.empty:
        raise Exception("Tracker file is empty. No models available for loading.")

    best_idx = tracker_df[metric].idxmax() if maximize else tracker_df[metric].idxmin()
    best_model_row = tracker_df.loc[best_idx]

    model_path = tracker_path.parent / best_model_row["model_file"]
    print(f"Loading best model: {best_model_row['model_file']} with {metric}={best_model_row[metric]}")
    return load_model(str(model_path))

def save_model(
    model: object,
    model_type: str,
    featureset: str,
    optimizer: str,
    scoremetric: str,
    version: str,
    accuracy: float,
    output_dir: str = "models",
    tracker_file: str = "models/model_tracker.csv"
) -> None:
    """
    Save a trained model with standardized naming conventions, and update a model tracker CSV.

    Args:
        model (object): Trained model object to be saved.
        model_type (str): Short name of the model (e.g., 'rf', 'xgb', 'mlp', 'ensemble').
        featureset (str): Feature subset used for model training (e.g., 'full', 'pca', 'lasso').
        optimizer (str): Optimization method applied (e.g., 'gridsearch', 'randomsearch', 'optuna').
        scoremetric (str): Primary metric optimized (e.g., 'acc', 'f1', 'rocauc').
        version (str): Version identifier for tracking iterations (e.g., 'v1', 'v2').
        accuracy (float): Accuracy (or other relevant score) achieved by the model.
        output_dir (str): Directory where the model file will be saved.
        tracker_file (str): Path for the model tracker CSV file.

    Raises:
        Exception: If saving or tracking fails.
    """
    try:
        output_dir_path = PROJECT_ROOT / output_dir
        tracker_path = PROJECT_ROOT / tracker_file

        # Check if output_dir_path exists — do NOT create it
        if not output_dir_path.exists():
            raise FileNotFoundError(f"Output directory does not exist: {output_dir_path}")

        filename = f"{model_type}_{featureset}_{optimizer}_{scoremetric}_{version}.joblib"
        save_path = output_dir_path / filename

        joblib.dump(model, save_path)

        tracker_columns = ["timestamp", "model_file", "model_type", "featureset", "optimizer", "scoremetric", "version", "accuracy"]

        if tracker_path.exists():
            tracker_df = pd.read_csv(tracker_path)
        else:
            tracker_df = pd.DataFrame(columns=tracker_columns)

        new_entry = {
            "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_file": filename,
            "model_type": model_type,
            "featureset": featureset,
            "optimizer": optimizer,
            "scoremetric": scoremetric,
            "version": version,
            "accuracy": round(accuracy, 4)
        }

        if not tracker_df.empty:
            tracker_df = pd.concat([tracker_df, pd.DataFrame([new_entry])], ignore_index=True)
        else:
            tracker_df = pd.DataFrame([new_entry])

        tracker_df.to_csv(tracker_path, index=False)

        print(f"Model successfully saved to {save_path}")
        print(f"Model tracker updated at {tracker_path}")

    except Exception as e:
        raise Exception(f"Error occurred while saving the model: {str(e)}")
