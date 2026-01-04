"""Centralized project paths for reproducible IO.

This module exposes a set of `pathlib.Path` objects that can be imported
throughout the codebase to avoid hard-coded relative paths. All paths are
resolved relative to the repository root, inferred from the location of this
file inside `src/protclassify`.
"""

from __future__ import annotations

from pathlib import Path

# Repository root is two levels above this file: <root>/src/protclassify/paths.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Common directories used across the project
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSION_DIR = PROJECT_ROOT / "submission"
DOCS_DIR = PROJECT_ROOT / "docs"

# Manifest-style tracking files
DATA_TRACKER = PROCESSED_DATA_DIR / "data_tracker.csv"
MODEL_TRACKER = MODELS_DIR / "model_tracker.csv"
SUBMISSION_TRACKER = SUBMISSION_DIR / "submission_tracker.csv"

__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "PROCESSED_DATA_DIR",
    "MODELS_DIR",
    "SUBMISSION_DIR",
    "DOCS_DIR",
    "DATA_TRACKER",
    "MODEL_TRACKER",
    "SUBMISSION_TRACKER",
]
