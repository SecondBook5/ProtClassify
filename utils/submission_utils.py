"""
Legacy shim that forwards to `protclassify.submission_utils`.
"""

from protclassify.submission_utils import (
    create_submission_from_model,
    create_submission_from_predictions,
)

__all__ = ["create_submission_from_predictions", "create_submission_from_model"]
