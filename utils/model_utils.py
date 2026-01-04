"""
Legacy shim that forwards to `protclassify.model_utils`.
"""

from protclassify.model_utils import (
    load_best_model_from_tracker,
    load_model,
    save_model,
)

__all__ = ["load_model", "load_best_model_from_tracker", "save_model"]
