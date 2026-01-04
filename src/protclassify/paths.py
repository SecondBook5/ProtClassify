"""
Path utilities centralizing the project directory layout.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data layout
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATA_TRACKER = PROCESSED_DATA_DIR / "data_tracker.csv"

# Artifacts
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"
DIM_REDUCTION_DIR = ARTIFACTS_DIR / "dim_reduction"
FEATURE_SELECTION_DIR = ARTIFACTS_DIR / "feature_selection"
ENSEMBLES_DIR = ARTIFACTS_DIR / "ensembles"
PREDICTIONS_DIR = ARTIFACTS_DIR / "predictions"
SUBMISSIONS_DIR = ARTIFACTS_DIR / "submissions"
SUBMISSION_TRACKER = SUBMISSIONS_DIR / "submission_tracker.csv"

# Models
MODELS_DIR = ARTIFACTS_DIR / "models"
FINAL_MODELS_DIR = MODELS_DIR / "final"
CHECKPOINT_MODELS_DIR = MODELS_DIR / "checkpoints"
MODEL_STUDIES_DIR = MODELS_DIR / "studies"
MODEL_TRACKER = FINAL_MODELS_DIR / "model_tracker.csv"

# Reports / envs
REPORTS_DIR = PROJECT_ROOT / "reports"
ENV_DIR = PROJECT_ROOT / "envs"


def resolve_path(path: Path) -> Path:
    """Resolve a path relative to the project root."""
    return path if path.is_absolute() else PROJECT_ROOT / path


def ensure_dir(path: Path) -> Path:
    """Create a directory path if it is missing and return it."""
    path.mkdir(parents=True, exist_ok=True)
    return path
