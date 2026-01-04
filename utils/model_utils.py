#!/usr/bin/env python3
"""Compatibility layer for the refactored model registry.

All logic now lives in `protclassify.models.registry`. This wrapper keeps
legacy imports working while we transition notebooks and scripts.
"""

from __future__ import annotations

from protclassify.models.registry import (
    ModelMetadata,
    ModelRegistry,
    load_best_model_from_tracker,
    load_model,
    save_model,
)

__all__ = [
    "ModelMetadata",
    "ModelRegistry",
    "load_best_model_from_tracker",
    "load_model",
    "save_model",
]
