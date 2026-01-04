#!/usr/bin/env python3
"""Submission helper wrappers for backward compatibility.

Core logic now lives in `protclassify.submission.builder`. This module
re-exports the modern API under the historic names to avoid breaking
notebooks while refactoring for CAFA6.
"""

from __future__ import annotations

from protclassify.submission.builder import (
    SubmissionBuilder,
    SubmissionMetadata,
    create_submission_from_model,
    create_submission_from_predictions,
)

__all__ = [
    "SubmissionBuilder",
    "SubmissionMetadata",
    "create_submission_from_model",
    "create_submission_from_predictions",
]
